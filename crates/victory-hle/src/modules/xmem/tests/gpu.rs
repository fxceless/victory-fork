//! Exercise XMem through the console queue and the real host texture upload
//! and read-back path. No title executable or assets are needed.

use std::ptr::{null, null_mut};
use victory_gpu::object::{HRESULT, Iid, Object, S_OK, guid};
use windows_sys::Win32::Foundation::{WAIT_FAILED, WAIT_OBJECT_0};
use windows_sys::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};

use super::*;

type This = *const Object;
const FENCE: Iid = guid(
    0x0a753dcf,
    0xc4d8,
    0x4b91,
    [0xad, 0xf6, 0xbe, 0x5a, 0x60, 0xd9, 0x5a, 0x76],
);
const RESOURCE: Iid = guid(
    0x696442be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 4, 0xf, 0xad],
);
const QUEUE: Iid = guid(
    0x0ec870a6,
    0x5d7e,
    0x4c22,
    [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
);
const ALLOCATOR: Iid = guid(
    0x6102dee4,
    0xaf59,
    0x4b09,
    [0xb9, 0x99, 0xb4, 0x4d, 0x73, 0xf0, 0x9b, 0x24],
);
const LIST: Iid = guid(
    0x5b160d0f,
    0xac1b,
    0x4185,
    [0x8b, 0xa8, 0xb3, 0xae, 0x42, 0xa5, 0xa4, 0x55],
);

/// An owned reference to a VIctory D3D12.X object, using the base (2.4) layout.
struct Com(This);

impl Com {
    fn new(create: impl FnOnce(*mut This) -> HRESULT) -> Self {
        let mut object = null();
        assert_eq!(create(&mut object), S_OK);
        assert!(!object.is_null());
        Self(object)
    }

    /// SAFETY: `F` and `offset` must match this object's D3D12.X method.
    unsafe fn slot<F: Copy>(&self, offset: usize) -> F {
        assert_eq!(size_of::<F>(), size_of::<*const ()>());
        // SAFETY: VIctory objects begin with their function table pointer.
        let address = unsafe { *(*self.0.cast::<*const *const ()>()).add(offset / 8) };
        unsafe { std::mem::transmute_copy(&address) }
    }

    /// SAFETY: this object is a VIctory fence. The event must stay alive
    /// until it signals, even if the client's fence reference is released.
    unsafe fn arm(&self, value: u64, event: &mut Event) {
        assert!(!event.armed);
        let arm = unsafe { self.slot::<extern "system" fn(This, u64, HANDLE) -> HRESULT>(0x48) };
        let result = arm(self.0, value, event.handle);
        if result != S_OK {
            eprintln!("arm queue fence {value} failed: {result:#x}");
            std::process::abort();
        }
        event.armed = true;
    }

    /// SAFETY: this object is a VIctory fence.
    unsafe fn wait_for(&self, value: u64) {
        let mut event = Event::new();
        unsafe { self.arm(value, &mut event) };
        event.wait(&format!("queue fence {value}"));
    }
}

/// An event may be registered on a fence even after the client releases it.
struct Event {
    handle: HANDLE,
    armed: bool,
}

impl Event {
    fn new() -> Self {
        // SAFETY: anonymous, initially reset event with the default security.
        let handle = unsafe { CreateEventW(null(), 0, 0, null()) };
        if handle.is_null() {
            eprintln!("CreateEventW failed: {}", std::io::Error::last_os_error());
            std::process::abort();
        }
        Self {
            handle,
            armed: false,
        }
    }

    fn wait(&mut self, label: &str) {
        assert!(self.armed);
        tracing::debug!(label, "waiting for fixture fence");
        // SAFETY: this event stays alive until the fence has signalled it.
        let result = unsafe { WaitForSingleObject(self.handle, INFINITE) };
        if result != WAIT_OBJECT_0 {
            if result == WAIT_FAILED {
                eprintln!(
                    "{label}: WaitForSingleObject failed: {}",
                    std::io::Error::last_os_error()
                );
            } else {
                eprintln!("{label}: unexpected wait result {result:#x}");
            }
            // A fence or worker may still own the event and fixture memory.
            std::process::abort();
        }
        self.armed = false;
        tracing::debug!(label, "fixture fence reached");
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.armed {
            // Unwinding must not close an event a pending fence may signal.
            eprintln!("dropping an event still registered on a queue fence");
            std::process::abort();
        }
        // SAFETY: this fixture owns the event and its registration completed.
        if unsafe { CloseHandle(self.handle) } == 0 {
            eprintln!("CloseHandle failed: {}", std::io::Error::last_os_error());
            std::process::abort();
        }
    }
}

impl Drop for Com {
    fn drop(&mut self) {
        // SAFETY: all these objects use IUnknown's Release in slot 2.
        unsafe { (self.slot::<extern "system" fn(This) -> u32>(0x10))(self.0) };
    }
}

#[repr(C)]
struct CopyLocation {
    resource: This,
    kind: u32,
    footprint: Footprint,
}

#[repr(C)]
struct Footprint {
    offset: u64,
    format: u32,
    width: u32,
    height: u32,
    depth: u32,
    row_pitch: u32,
}

#[repr(C)]
struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

#[repr(C)]
struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

const WIDTH: u32 = 256;
const HEIGHT: u32 = 192;
const PITCH: u32 = WIDTH * 4;
const BYTES: usize = (PITCH * HEIGHT) as usize;
const _: () = assert!(BYTES == 3 * PAGE);

#[test]
#[ignore = "requires a real D3D12 device; run separately with diagnostics and the debug layer"]
fn regrouped_pages_refresh_a_placed_texture_on_the_gpu() {
    let log = tracing_subscriber::fmt()
        .with_test_writer()
        .with_ansi(false)
        .with_max_level(tracing::Level::DEBUG)
        .finish();
    // Async queue and host workers run on other threads. A thread-local
    // subscriber would hide their diagnostics from this dedicated process.
    if let Err(error) = tracing::subscriber::set_global_default(log) {
        eprintln!("GPU fixture could not install its tracing subscriber: {error}");
    }
    let original = FixturePages(allocate(3));
    let replacement = FixturePages(allocate(4));
    let placed = AliasWindow::new(4);
    // At frame zero, choose an address off the guard's one-in-eight hash
    // turn, so a later write actually has to invalidate the guard.
    let start = usize::from((placed.base / PAGE).is_multiple_of(8));
    let destination = placed.at(start);
    let pool = AliasWindow::new(4);
    let alias = AliasWindow::new(3);
    for (pages, start) in [(&original.0, 0x10), (&replacement.0, 0x20)] {
        for (i, id) in pages.iter().enumerate() {
            assert!(write_physical(
                std::slice::from_ref(id),
                0,
                &vec![start + i as u8; PAGE]
            ));
        }
    }
    for (i, id) in original.0.iter().enumerate() {
        let at = placed.at(start + i);
        assert_eq!(XMemMapPhysicalPages(at, 1, id), at);
    }
    assert_eq!(
        XMemMapPhysicalPages(pool.at(0), 4, replacement.0.as_ptr()),
        pool.at(0)
    );
    victory_gpu::resource::use_page_mapper(map_alias);

    let dev = Com::new(|out| {
        victory_gpu::device::create(
            null(),
            &victory_gpu::device::CHECKED_VERSION,
            &victory_gpu::device::IID,
            out,
        )
    });
    // SAFETY: the signatures and offsets below are for the VIctory base
    // D3D12.X device/queue/list/resource vtables. All descriptions and output
    // pointers are live for the calls; Com owns each returned reference.
    unsafe {
        let make_queue = dev
            .slot::<extern "system" fn(This, *const u32, *const Iid, *mut This) -> HRESULT>(0x40);
        let queue = Com::new(|out| make_queue(dev.0, [0u32; 4].as_ptr(), &QUEUE, out));
        let make_allocator =
            dev.slot::<extern "system" fn(This, u32, *const Iid, *mut This) -> HRESULT>(0x48);
        let make_list = dev.slot::<extern "system" fn(
            This,
            u32,
            u32,
            This,
            This,
            *const Iid,
            *mut This,
        ) -> HRESULT>(0x60);
        let place = dev.slot::<extern "system" fn(
            This,
            u64,
            *const u64,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT>(0x428);
        let commit = dev.slot::<extern "system" fn(
            This,
            *const u32,
            u32,
            *const u64,
            u32,
            *const c_void,
            *const Iid,
            *mut This,
        ) -> HRESULT>(0xd8);
        // Seven u64s give the console ResourceDesc its 8-byte alignment.
        // 2D, RGBA8, linear, one mip/layer/sample; buffer with a linear row layout.
        let desc = [
            3,
            0,
            u64::from(WIDTH),
            u64::from(HEIGHT) | (1 << 32) | (1 << 48),
            28 | (1 << 32),
            1 << 32,
            0,
        ];
        let buffer = [
            1,
            0,
            BYTES as u64,
            1 | (1 << 32) | (1 << 48),
            1 << 32,
            1 << 32,
            0,
        ];
        let image = Com::new(|out| {
            place(
                dev.0,
                destination as u64,
                desc.as_ptr(),
                0,
                null(),
                &RESOURCE,
                out,
            )
        });
        let readback = Com::new(|out| {
            commit(
                dev.0,
                [1, 0, 0, 1, 1].as_ptr(),
                0,
                buffer.as_ptr(),
                0,
                null(),
                &RESOURCE,
                out,
            )
        });
        let map = readback
            .slot::<extern "system" fn(This, u32, *const c_void, *mut *mut u8) -> HRESULT>(0x40);
        let mut data = null_mut();
        assert_eq!(map(readback.0, 0, null(), &mut data), S_OK);
        assert!(!data.is_null());
        let to = CopyLocation {
            resource: readback.0,
            kind: 1,
            footprint: Footprint {
                offset: 0,
                format: 28,
                width: WIDTH,
                height: HEIGHT,
                depth: 1,
                row_pitch: PITCH,
            },
        };
        let from = CopyLocation {
            resource: image.0,
            kind: 0,
            footprint: Footprint {
                offset: 0,
                ..to.footprint
            },
        };
        let make_fence =
            dev.slot::<extern "system" fn(This, u64, u32, *const Iid, *mut This) -> HRESULT>(0x120);
        let execute = queue.slot::<extern "system" fn(This, u32, *const This)>(0x50);
        let signal = queue.slot::<extern "system" fn(This, This, u64) -> HRESULT>(0x70);
        let check = |label: &str, values: [u8; 3], pending: Option<[u8; 3]>| {
            let allocator = Com::new(|out| make_allocator(dev.0, 0, &ALLOCATOR, out));
            let list = Com::new(|out| make_list(dev.0, 0, 0, allocator.0, null(), &LIST, out));
            let fence = Com::new(|out| make_fence(dev.0, 0, 0, &FENCE, out));
            let copy = list.slot::<extern "system" fn(
                This,
                *const CopyLocation,
                u32,
                u32,
                u32,
                *const CopyLocation,
                *const c_void,
            )>(0x80);
            let close = list.slot::<extern "system" fn(This) -> HRESULT>(0x48);
            data.write_bytes(0xd3, BYTES); // fail if no GPU copy reached the buffer
            copy(list.0, &to, 0, 0, 0, &from, null());
            assert_eq!(close(list.0), S_OK);
            // The default queue may submit asynchronously. Wait for its
            // signal, including the deferred readback, before checking bytes.
            execute(queue.0, 1, &list.0);
            assert_eq!(signal(queue.0, fence.0, 1), S_OK, "{label}: signal queued");
            fence.wait_for(1);
            let bytes = std::slice::from_raw_parts(data, BYTES);
            if pending.is_some_and(|prior| {
                bytes
                    .chunks_exact(PAGE)
                    .zip(prior)
                    .all(|(page, value)| page.iter().all(|&b| b == value))
            }) {
                eprintln!("{label}: staging change will upload on the next list");
                return;
            }
            for (i, (page, value)) in bytes.chunks_exact(PAGE).zip(values).enumerate() {
                let wrong = page.iter().position(|&byte| byte != value);
                assert_eq!(
                    wrong,
                    None,
                    "{label}, page {i}; first wrong value {:?}, expected {value:#x}",
                    wrong.map(|at| page[at])
                );
            }
        };
        check("initial", [0x10, 0x11, 0x12], None);
        assert_eq!(
            XMemMapPhysicalPages(destination, 3, replacement.0.as_ptr()),
            destination
        );
        check("regrouped", [0x20, 0x21, 0x22], None);
        let copy_mappings =
            queue
                .slot::<extern "system" fn(This, u32, *const PageBatch, u64, u32, u32) -> HRESULT>(
                    0xe8,
                );
        let range = PageRange {
            kind: 0,
            start: 0,
            count: 3,
        };
        let batch = PageBatch {
            destination: alias.base as u64,
            count: 1,
            ranges: &range,
        };
        assert_eq!(
            copy_mappings(queue.0, 1, &batch, pool.base as u64, 4, 0),
            S_OK
        );
        alias.assert_bytes(1, 0x21);
        assert_eq!(
            XMemMapPhysicalPages(pool.at(1), 1, &replacement.0[3]),
            pool.at(1)
        );
        pool.assert_bytes(1, 0x23);
        check(
            "pool remapped, copy remains on the old physical page",
            [0x20, 0x21, 0x22],
            None,
        );
        assert_eq!(
            query(alias.at(1)).host.Protect,
            windows_sys::Win32::System::Memory::PAGE_READONLY
        );
        alias.at(1).cast::<u8>().write_bytes(0x42, PAGE);
        // When a clean texture skips the upload at record time, the
        // submission still refreshes staging. The next list must upload it.
        check(
            "detect an alias write",
            [0x20, 0x42, 0x22],
            Some([0x20, 0x21, 0x22]),
        );
        check("write to the copied alias", [0x20, 0x42, 0x22], None);
        // DirectStorage writes through its own view, then announces the
        // completed write. Before the announcement the guard cannot see it.
        let pages = &replacement.0[1..2];
        let mut window = page_window(pages, 0, PAGE).expect("physical view");
        window.bytes().fill(0x65);
        drop(window);
        placed.assert_bytes(start + 1, 0x65);
        check("unannounced physical write", [0x20, 0x42, 0x22], None);
        pages_written(pages, 0, PAGE);
        check(
            "detect a physical write",
            [0x20, 0x65, 0x22],
            Some([0x20, 0x42, 0x22]),
        );
        check("physical write after remap", [0x20, 0x65, 0x22], None);
        pool.assert_bytes(1, 0x23);

        // The queue must not change the backing of a pending GPU readback's
        // destination. Observe both the live address and an alias of the old
        // pages so that even a remap that leaves equal data cannot hide it.
        let targets = FixturePages(allocate(3));
        let target = AliasWindow::new(3);
        let old_target = AliasWindow::new(3);
        let pattern = [0x20, 0x65, 0x22];
        for held in [true, false] {
            for (i, id) in targets.0.iter().enumerate() {
                assert!(write_physical(
                    std::slice::from_ref(id),
                    0,
                    &vec![0x70 + i as u8; PAGE]
                ));
            }
            for window in [&target, &old_target] {
                assert_eq!(
                    XMemMapPhysicalPages(window.at(0), 3, targets.0.as_ptr()),
                    window.at(0)
                );
            }
            let copy_target = Com::new(|out| {
                place(
                    dev.0,
                    target.base as u64,
                    buffer.as_ptr(),
                    0,
                    null(),
                    &RESOURCE,
                    out,
                )
            });
            let next_image = Com::new(|out| {
                place(
                    dev.0,
                    target.base as u64,
                    desc.as_ptr(),
                    0,
                    null(),
                    &RESOURCE,
                    out,
                )
            });
            let into = CopyLocation {
                resource: copy_target.0,
                kind: 1,
                footprint: Footprint {
                    offset: 0,
                    ..to.footprint
                },
            };
            let allocator = Com::new(|out| make_allocator(dev.0, 0, &ALLOCATOR, out));
            let list = Com::new(|out| make_list(dev.0, 0, 0, allocator.0, null(), &LIST, out));
            let next = Com::new(|out| make_list(dev.0, 0, 0, allocator.0, null(), &LIST, out));
            let gate = Com::new(|out| make_fence(dev.0, 0, 0, &FENCE, out));
            let done = Com::new(|out| make_fence(dev.0, 0, 0, &FENCE, out));
            let producer = Com::new(|out| make_queue(dev.0, [0u32; 4].as_ptr(), &QUEUE, out));
            let copy = list.slot::<extern "system" fn(
                This,
                *const CopyLocation,
                u32,
                u32,
                u32,
                *const CopyLocation,
                *const c_void,
            )>(0x80);
            let close = list.slot::<extern "system" fn(This) -> HRESULT>(0x48);
            copy(list.0, &into, 0, 0, 0, &from, null());
            assert_eq!(close(list.0), S_OK);
            if held {
                let wait = queue.slot::<extern "system" fn(This, This, u64) -> HRESULT>(0x78);
                assert_eq!(wait(queue.0, gate.0, 1), S_OK);
            }
            execute(queue.0, 1, &list.0);
            // Borrowed descriptors are deliberately overwritten before the
            // fence opens. The queue must own them if it delays the remap.
            let mut range = PageRange {
                kind: 0,
                start: 0,
                count: 3,
            };
            let mut batch = PageBatch {
                destination: target.base as u64,
                count: 1,
                ranges: &raw const range,
            };
            let queued = copy_mappings(queue.0, 1, &raw const batch, pool.base as u64, 4, 0);
            std::ptr::write_volatile(&raw mut range.kind, 3);
            std::ptr::write_volatile(&raw mut batch.count, 0);
            let source = CopyLocation {
                resource: next_image.0,
                kind: 0,
                footprint: Footprint {
                    offset: 0,
                    ..from.footprint
                },
            };
            data.write_bytes(0xd3, BYTES);
            copy(next.0, &to, 0, 0, 0, &source, null());
            assert_eq!(close(next.0), S_OK);
            execute(queue.0, 1, &next.0);
            if held {
                // A zero-plane Present exercises the legacy wait-forcing
                // path without opening a window or presenting an image.
                let present = queue.slot::<extern "system" fn(
                    This,
                    u32,
                    *const c_void,
                    *const c_void,
                ) -> HRESULT>(0xf0);
                assert_eq!(present(queue.0, 0, null(), null()), S_OK);
            }
            let before = held.then(|| target.at(1).cast::<u8>().read());
            assert_eq!(signal(queue.0, done.0, 1), S_OK);
            if held {
                assert_eq!(signal(producer.0, gate.0, 1), S_OK);
            }
            done.wait_for(1);
            assert_eq!(queued, S_OK);
            let old = old_target.at(1).cast::<u8>().read();
            let new = target.at(1).cast::<u8>().read();
            assert_eq!(
                (before, old, new),
                (held.then_some(0x71), 0x65, 0x23),
                "remap overtook earlier GPU readback (held={held}): before, old backing, new backing"
            );
            for (page, value) in pattern.into_iter().enumerate() {
                old_target.assert_bytes(page, value);
            }
            for (page, value) in [0x20, 0x23, 0x22].into_iter().enumerate() {
                target.assert_bytes(page, value);
                let bytes = std::slice::from_raw_parts(data.add(page * PAGE), PAGE);
                assert!(
                    bytes.iter().all(|&actual| actual == value),
                    "later GPU read used old backing, page {page} (held={held})"
                );
            }
        }

        // Register the retired queue first, then a mapping queue behind the
        // same closed gate. With joining enabled, the latter's remap is a
        // boundary: the former's list and signal go through Joined, and its
        // now-empty queue mark is cleaned up before the remap can run.
        let retired = Com::new(|out| make_queue(dev.0, [0u32; 4].as_ptr(), &QUEUE, out));
        let mapped = Com::new(|out| make_queue(dev.0, [0u32; 4].as_ptr(), &QUEUE, out));
        let producer = Com::new(|out| make_queue(dev.0, [0u32; 4].as_ptr(), &QUEUE, out));
        let gate = Com::new(|out| make_fence(dev.0, 0, 0, &FENCE, out));
        let copied = Com::new(|out| make_fence(dev.0, 0, 0, &FENCE, out));
        let done = Com::new(|out| make_fence(dev.0, 0, 0, &FENCE, out));
        let mut copied_event = Event::new();
        copied.arm(1, &mut copied_event);
        let retired_target = AliasWindow::new(1);
        assert_eq!(
            XMemMapPhysicalPages(retired_target.at(0), 1, &targets.0[1]),
            retired_target.at(0)
        );
        retired_target.assert_bytes(0, 0x65);
        let allocator = Com::new(|out| make_allocator(dev.0, 0, &ALLOCATOR, out));
        let list = Com::new(|out| make_list(dev.0, 0, 0, allocator.0, null(), &LIST, out));
        let copy = list.slot::<extern "system" fn(
            This,
            *const CopyLocation,
            u32,
            u32,
            u32,
            *const CopyLocation,
            *const c_void,
        )>(0x80);
        let close = list.slot::<extern "system" fn(This) -> HRESULT>(0x48);
        data.write_bytes(0xd3, BYTES);
        copy(list.0, &to, 0, 0, 0, &from, null());
        assert_eq!(close(list.0), S_OK);
        let range = PageRange {
            kind: 0,
            start: 1,
            count: 1,
        };
        let batch = PageBatch {
            destination: retired_target.base as u64,
            count: 1,
            ranges: &range,
        };
        let wait = retired.slot::<extern "system" fn(This, This, u64) -> HRESULT>(0x78);
        assert_eq!(wait(retired.0, gate.0, 1), S_OK);
        execute(retired.0, 1, &list.0);
        assert_eq!(signal(retired.0, copied.0, 1), S_OK);
        assert_eq!(wait(mapped.0, gate.0, 1), S_OK);
        assert_eq!(
            copy_mappings(mapped.0, 1, &batch, pool.base as u64, 4, 0),
            S_OK
        );
        assert_eq!(signal(mapped.0, done.0, 1), S_OK);
        let completed = done.slot::<extern "system" fn(This) -> u64>(0x40);
        assert_eq!(completed(copied.0), 0, "GPU signal passed its closed gate");
        assert_eq!(completed(done.0), 0, "mapping queue passed its closed gate");
        retired_target.assert_bytes(0, 0x65);
        // No client owns either queue or the copied fence when the gate
        // opens. Keep the event, source, destination and lists alive instead.
        drop(copied);
        drop(retired);
        drop(mapped);
        assert_eq!(signal(producer.0, gate.0, 1), S_OK);
        drop(gate);
        copied_event.wait("retired queue GPU copy");
        done.wait_for(1);
        for (page, value) in pattern.into_iter().enumerate() {
            let bytes = std::slice::from_raw_parts(data.add(page * PAGE), PAGE);
            assert!(
                bytes.iter().all(|&actual| actual == value),
                "retired queue GPU copy, page {page}, expected {value:#x}"
            );
        }
        old_target.assert_bytes(1, 0x65);
        retired_target.assert_bytes(0, 0x23);
    }
}
