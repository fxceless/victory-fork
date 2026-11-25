use crate::object::This;
use std::ptr::{null, null_mut};
use windows::Win32::Graphics::Direct3D12::{D3D12_PLACED_SUBRESOURCE_FOOTPRINT, ID3D12Resource};
use windows_sys::Win32::System::Memory::*;

const PLANE_SIZE: usize = 1 << 16;
const MEM_WRITE_WATCH: u32 = 0x20_0000;
const RESOURCE_IID: crate::object::Iid = crate::object::guid(
    0x696442be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);

struct Plane {
    resource: This,
}

impl Plane {
    fn new(memory: *mut u8) -> Self {
        // The fixture uses a small, one-mip R8 movie plane. Using a uniform
        // byte value makes its expected readback independent of row padding.
        let words: [u32; 14] = [3, 0, 0, 0, 16, 0, 8, 65537, 61, 1, 0, 256, 0, 0];
        // SAFETY: the console descriptor is fourteen u32 words.
        let desc: crate::resource::ResourceDesc = unsafe { std::mem::transmute(words) };
        let layout = crate::resource::texture_layout(&desc).expect("R8 plane layout");
        assert!(layout.planes[0].layout.size <= PLANE_SIZE as u64);
        let mut plane = Self { resource: null() };
        let hr = crate::resource::create_at_address(
            null(),
            memory as u64,
            &desc,
            0,
            null(),
            &RESOURCE_IID,
            &mut plane.resource,
        );
        assert_eq!(hr, 0, "create R8 plane");
        plane
    }

    fn record(
        &self,
        setup: impl FnOnce(&crate::host::HostList),
    ) -> (crate::host::HostList, Readback) {
        let host = crate::host::host().expect("physical D3D12 device");
        let texture = crate::resource::host_resource(self.resource).expect("host R8 texture");
        let list = host.new_list().expect("command list");
        list.keep(texture);
        setup(&list);
        super::upload(&list, self.resource, texture);
        crate::list::ensure_state(&list, self.resource, 0x800);
        let (readback, footprint) = crate::capture::read_back(&host.device, &list.list, texture)
            .expect("record R8 readback");
        list.close();
        (list, Readback(readback, footprint))
    }

    fn sample(&self, before_submit: impl FnOnce()) -> u8 {
        let (list, readback) = self.record(|_| {});
        before_submit();
        crate::host::host().unwrap().execute(&[&list]);
        readback.sample()
    }
}

struct Readback(ID3D12Resource, D3D12_PLACED_SUBRESOURCE_FOOTPRINT);

impl Readback {
    fn sample(&self) -> u8 {
        let mut data = null_mut();
        // SAFETY: execute waited for the copy, and the footprint names its
        // completed readback. Verify every visible texel, ignoring padding.
        unsafe {
            self.0.Map(0, None, Some(&mut data)).expect("read R8 plane");
            let f = self.1.Footprint;
            let first = data.cast::<u8>().add(self.1.Offset as usize);
            let value = *first;
            for y in 0..f.Height as usize {
                let row = std::slice::from_raw_parts(
                    first.add(y * f.RowPitch as usize),
                    f.Width as usize,
                );
                assert!(
                    row.iter().all(|&v| v == value),
                    "mixed generations within a plane"
                );
            }
            self.0.Unmap(0, None);
            value
        }
    }
}

#[test]
fn each_list_uploads_its_own_generation_in_one_submission() {
    let memory = Watched::new(41);
    let next = Watched::new(129);
    let plane = Plane::new(memory.0);
    assert_eq!(plane.sample(|| {}), 41);
    let (first, read_first) = plane.record(|_| {});
    let (second, read_second) = plane.record(|list| list.defer_copy(memory.0, next.0, PLANE_SIZE));
    memory.write(72);
    crate::host::host().unwrap().execute(&[&first, &second]);
    assert_eq!([read_first.sample(), read_second.sample()], [72, 129]);
}

impl Drop for Plane {
    fn drop(&mut self) {
        // SAFETY: the fixture owns the COM reference. Every
        // sample's list finishes before the resource can be released.
        unsafe {
            if !self.resource.is_null() {
                let table = *self.resource.cast::<*const *const ()>();
                let release: extern "system" fn(This) -> u32 = std::mem::transmute(*table.add(2));
                release(self.resource);
            }
        }
    }
}

struct Watched(*mut u8);

impl Watched {
    fn new(initial: u8) -> Self {
        // SAFETY: this reservation belongs to the fixture. Textures borrow
        // it until their COM references are released.
        let memory = unsafe {
            VirtualAlloc(
                null(),
                PLANE_SIZE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        }
        .cast::<u8>();
        assert!(!memory.is_null());
        let watched = Self(memory);
        watched.write(initial);
        watched
    }

    fn write(&self, value: u8) {
        // SAFETY: committed fixture memory borrowed by no reader here.
        unsafe { self.0.write_bytes(value, PLANE_SIZE) };
    }
}

impl Drop for Watched {
    fn drop(&mut self) {
        // SAFETY: the fixture owns this reservation.
        unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
    }
}

#[test]
fn rotating_movie_planes_show_updates_decoded_after_recording() {
    let originals = [17, 34, 51];
    let updated = [113, 130, 147];
    let memory = originals.map(Watched::new);
    let planes = memory.each_ref().map(|memory| Plane::new(memory.0));
    // Warm each host copy until recording considers it already uploaded.
    for _ in 0..3 {
        for (plane, value) in planes.iter().zip(originals) {
            assert_eq!(plane.sample(|| {}), value, "warm movie plane");
        }
    }
    let observed = std::array::from_fn(|i| planes[i].sample(|| memory[i].write(updated[i])));
    let next_use = planes.each_ref().map(|plane| plane.sample(|| {}));
    eprintln!("decoded {updated:?}, first GPU sample {observed:?}, next use {next_use:?}");
    assert_eq!(
        observed, updated,
        "the recorded consumer must get the frame decoded before submission"
    );
}

#[test]
fn movie_plane_sees_writes_through_another_view() {
    use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};

    // Pick a range guaranteed to miss the periodic guard hash at the
    // current frame. Both views expose the same physical section.
    const SIZE: usize = 2 * PLANE_SIZE;
    struct View(MEMORY_MAPPED_VIEW_ADDRESS);
    impl Drop for View {
        fn drop(&mut self) {
            crate::watch::title_memory_remapped(self.0.Value as usize, SIZE);
            // SAFETY: the fixture owns this mapping.
            unsafe { UnmapViewOfFile(self.0) };
        }
    }
    // SAFETY: both mappings hold a reference to the pagefile section.
    let (read, write) = unsafe {
        let section = CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            null(),
            PAGE_READWRITE,
            0,
            SIZE as u32,
            null(),
        );
        assert!(!section.is_null());
        let read = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
        let write = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
        CloseHandle(section);
        assert!(!read.Value.is_null() && !write.Value.is_null());
        (View(read), View(write))
    };
    let slot = crate::frame::current().wrapping_add((read.0.Value as usize >> 16) as u64);
    let offset = if slot.is_multiple_of(8) {
        PLANE_SIZE
    } else {
        0
    };
    // SAFETY: both pointers are within the corresponding mapping.
    let (read, write) = unsafe {
        (
            read.0.Value.cast::<u8>().add(offset),
            write.0.Value.cast::<u8>().add(offset),
        )
    };
    // SAFETY: both writes are through our mapping, with no concurrent reads.
    unsafe { write.write_bytes(41, PLANE_SIZE) };
    let plane = Plane::new(read);
    assert_eq!(plane.sample(|| {}), 41, "initial section contents");
    let sampled = plane.sample(|| unsafe { write.write_bytes(129, PLANE_SIZE) });
    eprintln!("decoded through alias 129, first GPU sample {sampled}");
    assert_eq!(
        sampled, 129,
        "the first GPU consumer must see the write through the alias"
    );
}
