//! Title memory with a hole, as xmem leaves it: views of a section in a
//! placeholder, with pages between them that are placeholders again (never
//! mapped, or mapped and unmapped). A texture's memory is asked whether it
//! is mapped by reading a byte of each page (`texture::committed`), and
//! title memory is copied through the kernel (`host::read_into`,
//! `host::copy_mapped`): each takes a read that succeeds as "there". If a
//! placeholder read as zeros instead of failing (Wine's kernel copies and
//! its handling of placeholders are its own), a texture streaming in would
//! be read as whole with zero blocks where its pages are missing, and never
//! read again for them. These say a hole is seen as one.

use std::ffi::c_void;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_COMMIT, MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER,
    MEM_RESERVE, MEM_RESERVE_PLACEHOLDER, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3, PAGE_NOACCESS,
    PAGE_READWRITE, SEC_COMMIT, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const PAGE: usize = 64 << 10;
const PAGES: usize = 3;

/// A placeholder of [`PAGES`] pages split into one per page, and a section
/// of as many pages whose page `i` is filled with `0x40 + i`.
struct Pool {
    base: usize,
    section: HANDLE,
}

impl Pool {
    fn new() -> Pool {
        // SAFETY: a fresh placeholder and section, released on drop.
        unsafe {
            let base = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                PAGES * PAGE,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(base, 0, "a placeholder");
            for page in 0..PAGES - 1 {
                VirtualFree((base + page * PAGE) as *mut c_void, PAGE, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER);
            }
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_COMMIT,
                0,
                (PAGES * PAGE) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null(), "a section");
            let pool = Pool { base, section };
            for page in 0..PAGES {
                assert!(pool.map(page));
                std::ptr::write_bytes(pool.at(page) as *mut u8, 0x40 + page as u8, PAGE);
                assert!(pool.unmap(page));
            }
            pool
        }
    }

    fn at(&self, page: usize) -> usize {
        self.base + page * PAGE
    }

    fn map(&self, page: usize) -> bool {
        // SAFETY: a placeholder page of ours.
        let view = unsafe {
            MapViewOfFile3(
                self.section,
                GetCurrentProcess(),
                self.at(page) as *const c_void,
                (page * PAGE) as u64,
                PAGE,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        view.Value as usize == self.at(page)
    }

    fn unmap(&self, page: usize) -> bool {
        // SAFETY: a view of ours, back to a placeholder.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: self.at(page) as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            ) != 0
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        for page in 0..PAGES {
            let _ = self.unmap(page);
        }
        // SAFETY: our placeholders and section.
        unsafe {
            for page in 0..PAGES {
                VirtualFree(self.at(page) as *mut c_void, 0, MEM_RELEASE);
            }
            CloseHandle(self.section);
        }
    }
}

/// What a refresh learns of `[at, at + len)`: whether the read through
/// the kernel's copy takes it whole, whether it counts as all mapped, and
/// the bytes the copy that stands holes reads.
fn looks(at: usize, len: usize) -> (bool, bool, Vec<u8>) {
    let mut whole = vec![0xEEu8; len];
    let read = crate::host::read_into(at as *const u8, &mut whole);
    let committed = crate::texture::committed(at as *const u8, len);
    let mut stood = vec![0xEEu8; len];
    crate::host::copy_mapped(stood.as_mut_ptr(), at as *const u8, len, false);
    (read, committed, stood)
}

#[test]
fn a_placeholder_between_views_is_a_hole() {
    let pool = Pool::new();
    assert!(pool.map(0) && pool.map(2));
    let (read, committed, stood) = looks(pool.base, PAGES * PAGE);
    assert!(!read, "the kernel's copy read a placeholder page");
    assert!(!committed, "a placeholder page counted as mapped");
    assert!(stood[..PAGE].iter().all(|&b| b == 0x40));
    assert!(stood[PAGE..2 * PAGE].iter().all(|&b| b == 0), "the hole reads as zeros");
    assert!(stood[2 * PAGE..].iter().all(|&b| b == 0x42));
    // The views alone are there.
    assert!(crate::texture::committed(pool.at(0) as *const u8, PAGE));
    assert!(crate::texture::committed(pool.at(2) as *const u8, PAGE));
    assert!(!crate::texture::committed(pool.at(1) as *const u8, 1));
}

#[test]
fn a_page_unmapped_again_is_a_hole() {
    let pool = Pool::new();
    assert!(pool.map(0) && pool.map(1) && pool.map(2));
    let (read, committed, stood) = looks(pool.base, PAGES * PAGE);
    assert!(read && committed);
    assert!(stood[PAGE..2 * PAGE].iter().all(|&b| b == 0x41));
    assert!(pool.unmap(1));
    let (read, committed, stood) = looks(pool.base, PAGES * PAGE);
    assert!(!read, "the kernel's copy read a page unmapped again");
    assert!(!committed, "a page unmapped again counted as mapped");
    assert!(stood[PAGE..2 * PAGE].iter().all(|&b| b == 0));
    // A hole at the first page, as a texture whose first mip is missing.
    assert!(pool.unmap(0));
    assert!(!looks(pool.base, PAGES * PAGE).0);
    assert!(!crate::texture::committed(pool.base as *const u8, PAGES * PAGE));
}

#[test]
fn reserved_and_inaccessible_pages_are_holes() {
    // SAFETY: a fresh reservation of ours, released below.
    unsafe {
        let base = VirtualAlloc(std::ptr::null(), PAGES * PAGE, MEM_RESERVE, PAGE_NOACCESS) as usize;
        assert_ne!(base, 0);
        for page in [0, 2] {
            let at = VirtualAlloc((base + page * PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_READWRITE);
            assert_eq!(at as usize, base + page * PAGE);
            std::ptr::write_bytes(at as *mut u8, 0x50 + page as u8, PAGE);
        }
        let (read, committed, stood) = looks(base, PAGES * PAGE);
        assert!(!read, "the kernel's copy read reserved memory");
        assert!(!committed, "reserved memory counted as mapped");
        assert!(stood[PAGE..2 * PAGE].iter().all(|&b| b == 0));
        assert!(stood[2 * PAGE..].iter().all(|&b| b == 0x52));
        // Committed, but not to be read.
        let middle = VirtualAlloc((base + PAGE) as *const c_void, PAGE, MEM_COMMIT, PAGE_NOACCESS);
        assert_eq!(middle as usize, base + PAGE);
        let (read, committed, _) = looks(base, PAGES * PAGE);
        assert!(!read, "the kernel's copy read a no-access page");
        assert!(!committed, "a no-access page counted as mapped");
        VirtualFree(base as *mut c_void, 0, MEM_RELEASE);
    }
}
