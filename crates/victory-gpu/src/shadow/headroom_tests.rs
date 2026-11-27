use super::{BACK_CLASS, CARD_BYTES, find, host, register, unregister};
use std::{process::Command, sync::atomic::Ordering};
use windows::Win32::Graphics::Direct3D12::{
    D3D12_COMPUTE_PIPELINE_STATE_DESC, D3D12_SHADER_BYTECODE, ID3D12PipelineState,
    ID3D12RootSignature,
};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

/// Replay small allocations followed by one large shader-written buffer. Both
/// placements must round-trip the same shader result. Only the fallback to
/// shared memory snapshots and scans the declared write range on the CPU.
#[test]
fn late_large_buffer_keeps_its_headroom_and_shader_writes() {
    const CHILD: &str = "VICTORY_TEST_LARGE_HEADROOM";
    let Ok(value) = std::env::var(CHILD) else {
        // Host selection, budgets and counters are process-wide.
        let name = concat!(
            module_path!(),
            "::late_large_buffer_keeps_its_headroom_and_shader_writes"
        );
        let (_, name) = name.split_once("::").unwrap();
        for reserve in ["0", "2"] {
            let out = Command::new(std::env::current_exe().unwrap())
                .args(["--exact", name, "--nocapture"])
                .env(CHILD, reserve)
                .env("VICTORY_CARD_BUFFERS", "1")
                .env("VICTORY_CARD_BUFFERS_MIN_KB", "64")
                .env("VICTORY_CARD_BUFFERS_MB", "8")
                .env("VICTORY_CARD_BUFFERS_HEADROOM_MB", reserve)
                .env("VICTORY_D3D12_WARP", "1")
                .env("VICTORY_RELAY", "0")
                .env("VICTORY_INPLACE", "0")
                .output()
                .unwrap();
            let log = format!(
                "{}{}",
                String::from_utf8_lossy(&out.stdout),
                String::from_utf8_lossy(&out.stderr)
            );
            eprintln!("{log}");
            assert!(
                out.status.success() && log.contains("1 passed"),
                "headroom {reserve}"
            );
        }
        return;
    };
    let reserve = value == "2";
    const MIB: usize = 1 << 20;
    const SMALL: usize = 64 << 10;
    // 96 small buffers (6 MiB), one byte under 1 MiB, another 32 small
    // buffers (2 MiB), then the 2 MiB shader-written buffer.
    const NEAR: usize = 6 * MIB;
    const MORE: usize = 7 * MIB;
    const LARGE: usize = 9 * MIB;
    const LEN: usize = 2 * MIB;
    const MEM_WRITE_WATCH: u32 = 0x0020_0000;
    let Some(host) = host() else {
        eprintln!("no D3D12 WARP device");
        return;
    };
    assert!(
        host.card_ready(),
        "the WARP host must be able to hash its card copies"
    );
    // SAFETY: one private allocation for all simulated title buffers. All ranges
    // below are disjoint and inside it. It survives until after the fence.
    let title = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            LARGE + LEN,
            MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
            PAGE_READWRITE,
        )
    } as usize;
    assert_ne!(title, 0);
    let mut starts = Vec::new();
    let mut bind = |offset: usize, size: usize| {
        let start = title + offset;
        register(start, start as u64, size as u64);
        starts.push(start);
        find(start as u64, size as u64).expect("a registered title buffer")
    };
    let mut small = 0;
    for n in 0..96 {
        small += usize::from(bind(n * SMALL, SMALL).carded());
    }
    let near = bind(NEAR, MIB - 1).carded();
    for n in 0..32 {
        small += usize::from(bind(MORE + n * SMALL, SMALL).carded());
    }
    let large = bind(LARGE, LEN);
    let start = title + LARGE;
    const SHADER: &str = r#"
        RWByteAddressBuffer memory : register(u0);
        [RootSignature("UAV(u0)")]
        [numthreads(1,1,1)]
        void main() { memory.Store(4, memory.Load(0) + 1); }
    "#;
    let code = crate::host::compile(
        SHADER,
        windows::core::s!("main"),
        windows::core::s!("cs_5_1"),
    )
    .unwrap();
    let list = host.new_list().unwrap();
    large.refresh(&list, start as u64, LEN as u64);
    large.gpu_writes(&list, start as u64, LEN as u64);
    // SAFETY: an open list, the shader's own signature, the live buffer as UAV,
    // and CPU access to the private allocation before and after synchronous execution.
    unsafe {
        (start as *mut u32).write(41);
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
        let pso: ID3D12PipelineState = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .unwrap();
        list.list.SetComputeRootSignature(&root);
        list.list.SetPipelineState(&pso);
        list.list
            .SetComputeRootUnorderedAccessView(0, large.gpu(start as u64));
        list.list.Dispatch(1, 1, 1);
        list.close();
        host.execute(&[&list]);
        assert_eq!((start as *const u32).add(1).read_volatile(), 42);
    }
    let snapshot = BACK_CLASS[1].load(Ordering::Relaxed);
    let compared = BACK_CLASS[3].load(Ordering::Relaxed);
    let bytes = CARD_BYTES[0].load(Ordering::Relaxed);
    eprintln!(
        "headroom {value} MiB: {small} small card copies, near {near}, large {}, {bytes} card bytes, {snapshot} bytes snapshotted and {compared} compared, shader returned 42",
        large.carded()
    );
    assert_eq!(large.carded(), reserve);
    assert_eq!(near, !reserve);
    assert_eq!(small, if reserve { 96 } else { 112 });
    assert_eq!(bytes, (8 * MIB) as u64);
    assert_eq!(snapshot, if reserve { 0 } else { LEN as u64 });
    assert_eq!(compared, snapshot);
    drop(list);
    drop(large);
    for start in starts {
        unregister(start, start as u64);
    }
    assert_eq!(CARD_BYTES[0].load(Ordering::Relaxed), 0);
    // SAFETY: the private allocation returned by VirtualAlloc above, now unregistered.
    unsafe { VirtualFree(title as _, 0, MEM_RELEASE) };
}
