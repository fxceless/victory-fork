use super::*;
use std::ptr::null;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM;
use windows::core::{IUnknown, Interface};

fn render_target_array() -> IUnknown {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_DEFAULT,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 16,
        Height: 16,
        DepthOrArraySize: 2,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut texture = null();
    // The resource entry point takes the C layout of these D3D12 descriptions.
    assert_eq!(
        crate::resource::create_committed(
            null(),
            (&raw const heap).cast(),
            0,
            (&raw const desc).cast(),
            D3D12_RESOURCE_STATE_RENDER_TARGET.0 as u32,
            null(),
            std::ptr::from_ref(&ID3D12Resource::IID).cast(),
            &mut texture,
        ),
        crate::object::S_OK
    );
    // SAFETY: creation handed us an owned reference to the layer's COM object.
    unsafe { IUnknown::from_raw(texture.cast_mut().cast()) }
}

#[test]
fn prelude_picture_time_is_nested_only_when_a_picture_is_taken() {
    // The host and frame-shot clock are process-wide. WARP and the active shot
    // must be ours even when other GPU tests run in parallel.
    const CHILD: &str = "VICTORY_TEST_PRELUDE_SHOT";
    if std::env::var_os(CHILD).is_none() {
        let out = tempfile::tempdir().unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                std::thread::current().name().unwrap(),
                "--nocapture",
            ])
            .env(CHILD, "1")
            .env("VICTORY_D3D12_WARP", "1")
            .env("VICTORY_SHOT", out.path())
            .env("VICTORY_SHOT_TIMES", "1")
            .env("VICTORY_SHOT_MB", "64")
            .status()
            .unwrap();
        assert!(status.success());
        return;
    }

    let Some(host) = crate::host::host() else {
        eprintln!("frame shot test skipped: no D3D12 WARP device");
        return;
    };
    for (budget, pending) in [(1 << 20, false), (1 << 20, true), (0, false)] {
        arm(1, Carry::default(), folder().unwrap());
        SHOT.lock().unwrap().as_mut().unwrap().budget = budget;
        let textures = [render_target_array(), render_target_array()];
        let list = host.new_list().unwrap();
        list.note_op(|| "Read a render target".into());
        // Its first use in this list is a read. Submission records the move
        // out of RENDER_TARGET, and any pictures, in the prelude.
        for texture in &textures {
            assert!(
                list.transition(
                    texture.as_raw().cast(),
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE.0 as u32,
                )
                .is_none()
            );
        }
        list.close();
        if pending {
            // The title can present before this marked list is submitted.
            for frame in 1..=3 {
                present(frame, 0);
            }
            assert_eq!(PHASE.load(Relaxed), PENDING);
        }
        host.execute(&[&list]);

        let shot = SHOT.lock().unwrap().take().unwrap();
        assert_eq!(shot.places, 4);
        assert_eq!(shot.snaps.len(), if budget == 0 { 0 } else { 4 });
        let pairs = |label: &str| {
            shot.events
                .iter()
                .filter(|e| e.text.starts_with(label))
                .map(|e| {
                    assert_eq!(e.list, std::ptr::from_ref(&list) as usize);
                    assert!(shot.resolved.iter().any(|&(first, count)| {
                        first <= e.stamp && e.stamp + 1 < first + count
                    }));
                    let clock = CLOCK.get().unwrap().as_ref().unwrap();
                    // SAFETY: execute waited for this pair to resolve in the mapped buffer.
                    let ticks = (clock.mapped as *const u64).wrapping_add(e.stamp as usize);
                    unsafe { (ticks.read_volatile(), ticks.add(1).read_volatile()) }
                })
                .collect::<Vec<_>>()
        };
        // Existing consumers add all PRELUDE events to get the inclusive total.
        let prelude = pairs("PRELUDE");
        assert_eq!(prelude.len(), 1);
        let readbacks = pairs("SHOT READBACK");
        if budget == 0 {
            assert!(readbacks.is_empty());
        } else {
            assert_eq!(readbacks.len(), 1);
            let (begin, end) = readbacks[0];
            assert!(
                prelude[0].0 > 0 && prelude[0].0 <= begin && begin <= end && end <= prelude[0].1
            );
        }
    }
}
