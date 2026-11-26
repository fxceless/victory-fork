use super::*;
use std::ptr::null;
use std::sync::atomic::Ordering::Relaxed;
use std::sync::mpsc;
use std::time::Duration;
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC};
use windows::core::{IUnknown, Interface};
use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE, UnmapViewOfFile,
};

#[test]
fn a_remap_publishes_only_its_completed_texture() {
    // The mapping lock, write guards and no-wait switch are process-wide.
    const CHILD: &str = "VICTORY_TEST_TEXTURE_REMAP";
    if std::env::var_os(CHILD).is_none() {
        for light in ["0", "1"] {
            let status = std::process::Command::new(std::env::current_exe().unwrap())
                .args([
                    "--exact",
                    std::thread::current().name().unwrap(),
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("VICTORY_TEXTURE_NO_WAIT", "1")
                .env("VICTORY_TEXTURE_GUARD", "1")
                .env("VICTORY_LIGHT_REFRESH", light)
                .env("VICTORY_D3D12_WARP", "1")
                .status()
                .unwrap();
            assert!(status.success(), "VICTORY_LIGHT_REFRESH={light}");
        }
        return;
    }
    if host().is_none() {
        eprintln!("texture remap test skipped: no D3D12 device");
        return;
    }

    const PAGE: usize = 64 << 10;
    const SIZE: usize = 2 * PAGE;
    // SAFETY: two writable views of a fresh section. The third page lets
    // us choose a texture outside its once-every-eight-frames rehash turn.
    let (title, writer) = unsafe {
        let section = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            null(),
            PAGE_READWRITE,
            0,
            (3 * PAGE) as u32,
            null(),
        );
        assert!(!section.is_null());
        let title = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 3 * PAGE);
        let writer = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 3 * PAGE);
        CloseHandle(section);
        assert!(!title.Value.is_null() && !writer.Value.is_null());
        (title, writer)
    };
    let offset = if (title.Value as usize / PAGE).is_multiple_of(8) {
        PAGE
    } else {
        0
    };
    let source = title.Value.cast::<u8>().wrapping_add(offset);
    let alias = writer.Value.cast::<u8>().wrapping_add(offset);
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 256,
        Height: 128,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut resource = null();
    assert_eq!(
        crate::resource::create_at_address(
            null(),
            source as u64,
            (&raw const desc).cast(),
            0,
            null(),
            std::ptr::from_ref(&ID3D12Resource::IID).cast(),
            &mut resource,
        ),
        crate::object::S_OK
    );
    // SAFETY: creation returned one owning COM reference.
    let owner = unsafe { IUnknown::from_raw(resource.cast_mut().cast()) };
    let staging = staging(resource, crate::resource::host_resource(resource).unwrap()).unwrap();
    assert_eq!(staging.layout.size, SIZE as u64);
    assert_eq!(staging.bytes, SIZE as u64);
    let staged = || {
        // SAFETY: the mapped upload buffer is SIZE bytes. Every refresh has
        // finished before this closure reads those bytes.
        blake3::hash(unsafe { std::slice::from_raw_parts(staging.data, SIZE) })
    };
    let old = blake3::hash(&vec![0x11; SIZE]);
    let new = blake3::hash(&vec![0x77; SIZE]);

    // On first use there are no previous contents to keep. The worker
    // must wait until the complete initial image can be staged.
    let first_waited = std::thread::scope(|scope| {
        let (refresh, next) = mpsc::sync_channel(0);
        let (done, result) = mpsc::channel();
        let s = &staging;
        scope.spawn(move || {
            for () in next {
                if done.send(s.refresh_timed()).is_err() {
                    break;
                }
            }
        });
        let changing = crate::watch::title_memory_remapping();
        // SAFETY: the alias maps SIZE bytes, and no refresh is running.
        unsafe { alias.write_bytes(0x11, SIZE) };
        refresh.send(()).unwrap();
        let waiting = result.recv_timeout(Duration::from_millis(100));
        drop(changing);
        let first_waited = waiting == Err(mpsc::RecvTimeoutError::Timeout);
        assert_eq!(waiting.unwrap_or_else(|_| result.recv().unwrap()), SIZE);
        assert_eq!(staged(), old);
        assert_eq!(staging.last.lock().unwrap().hash, Some(old));
        assert_ne!(staging.last.lock().unwrap().asked, 0);
        staging.dirty.store(false, Relaxed);

        // The whole range remains readable while only the first page has
        // changed. A refresh with previous contents can return immediately.
        let changing = crate::watch::title_memory_remapping();
        // SAFETY: this is our writable alias; no refresh is running.
        unsafe { alias.write_bytes(0x77, PAGE) };
        crate::watch::title_memory_remapped(source as usize, PAGE);
        refresh.send(()).unwrap();
        let deferred = result.recv_timeout(Duration::from_secs(5));
        if deferred.is_ok() {
            // SAFETY: the refresh finished. Writing through this alias
            // doesn't fault on the title view, so there is no new notice.
            unsafe { alias.add(PAGE).write_bytes(0x77, PAGE) };
        }
        drop(changing);
        let read = deferred.expect("the staged texture waited for the remap");
        // SAFETY: the worker returned, and both offsets lie in the staging buffer.
        let pages = unsafe { [*staging.data, *staging.data.add(PAGE)] };
        assert_eq!(
            pages,
            [0x11, 0x11],
            "refresh read {read} bytes during the remap"
        );
        assert_eq!(staged(), old);
        assert_eq!(read, 0);
        assert!(!staging.dirty.load(Relaxed));
        let last = staging.last.lock().unwrap();
        assert_eq!(last.hash, Some(old));
        assert_eq!(last.asked, 0);
        first_waited
    });
    assert!(first_waited, "a first upload must wait for a stable image");

    assert_eq!(staging.refresh_timed(), SIZE);
    assert!(staging.dirty.load(Relaxed));
    assert_eq!(staging.last.lock().unwrap().hash, Some(new));
    assert_eq!(staged(), new);
    // Once the retry has consumed the remap, an ordinary check can skip
    // the unchanged pages again in either write-tracking mode.
    staging.dirty.store(false, Relaxed);
    assert_eq!(staging.refresh_timed(), 0);
    assert!(!staging.dirty.load(Relaxed));
    drop(staging);
    drop(owner);
    crate::watch::title_memory_remapped(source as usize, SIZE);
    // SAFETY: nothing refers to the contents of either view now.
    unsafe {
        UnmapViewOfFile(writer);
        UnmapViewOfFile(title);
    }
}
