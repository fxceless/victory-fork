//! Static TLS of an executable mapped as a DLL.
//!
//! An exe always gets TLS index 0, so title code reaches its thread-local
//! block as `mov r64, gs:[0x58]` (the thread's TLS array) followed by
//! `[r64]`, without reading `_tls_index`. Mapped as a DLL, the image gets
//! another index. Every such load is turned into a call to a thunk that
//! yields `&array[index]` instead, and the image's `_tls_index` (which CRT
//! code does read, and adds to the array pointer) is set to 0.

use std::ffi::c_void;

use anyhow::{Result, bail};
use victory_pe::ImageInfo;
use windows_sys::Win32::System::Diagnostics::Debug::FlushInstructionCache;
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_EXECUTE_READWRITE, PAGE_READWRITE,
    VirtualAlloc, VirtualProtect,
};
use windows_sys::Win32::System::Threading::GetCurrentProcess;

const SCN_MEM_EXECUTE: u32 = 0x2000_0000;
const TLS64_ADDRESS_OF_INDEX: usize = 16;
/// `mov r64, gs:[0x58]` is `65 REX 8B ModRM 25 58 00 00 00`.
const LOAD_LEN: usize = 9;
const THUNK_SIZE: usize = 32;

/// Redirects the image's TLS-array loads if the OS gave it an index other
/// than 0. Returns the number of loads patched.
///
/// # Safety
/// `base` is the image `info` describes, mapped and not yet running.
pub(crate) unsafe fn redirect(base: usize, info: &ImageInfo) -> Result<usize> {
    let Some(tls) = &info.tls else { return Ok(0) };
    // SAFETY: the TLS directory and the index it points to are in the image;
    // the OS relocated the pointer and wrote the index.
    let index_at = unsafe {
        ((base + tls.directory_rva as usize + TLS64_ADDRESS_OF_INDEX) as *const usize).read()
    } as *mut u32;
    if index_at.is_null() {
        bail!("TLS AddressOfIndex is null");
    }
    // SAFETY: as above.
    let index = unsafe { index_at.read() };
    if index == 0 {
        return Ok(0);
    }
    let thunks = thunks(near(base, info.size_of_image as usize)?, index)?;
    let mut patched = 0;
    for section in info
        .sections
        .iter()
        .filter(|s| s.characteristics & SCN_MEM_EXECUTE != 0)
    {
        let len = section.virtual_size as usize;
        if len == 0 {
            continue;
        }
        let end = (section.rva as usize)
            .checked_add(len)
            .filter(|&e| e <= info.size_of_image as usize)
            .ok_or_else(|| anyhow::anyhow!("executable section runs past image bounds"))?;
        let _ = end;
        let start = base + section.rva as usize;
        let mut old = 0;
        // SAFETY: the section is part of the mapped image.
        if unsafe { VirtualProtect(start as _, len, PAGE_EXECUTE_READWRITE, &mut old) } == 0 {
            bail!(
                "VirtualProtect(code) failed: {}",
                std::io::Error::last_os_error()
            );
        }
        // SAFETY: writable now, and no title code runs yet.
        let code = unsafe { std::slice::from_raw_parts_mut(start as *mut u8, len) };
        let result = patch(code, start, thunks);
        // SAFETY: restoring the protection we changed; flushing our own code.
        unsafe {
            VirtualProtect(start as _, len, old, &mut old);
            FlushInstructionCache(GetCurrentProcess(), start as _, len);
        }
        patched += result?;
    }
    // SAFETY: see `index_at`; `.data` is writable.
    unsafe { index_at.write(0) };
    tracing::info!(index, patched, "static TLS redirected");
    Ok(patched)
}

/// Rewrites each `mov r64, gs:[0x58]` in `code` (at address `at`) as
/// `call thunks[r]; nop4`. Fails without modifying the instruction if any
/// thunk target falls outside signed 32-bit relative reach.
fn patch(code: &mut [u8], at: usize, thunks: usize) -> Result<usize> {
    let mut count = 0;
    let mut i = 0;
    while i + LOAD_LEN <= code.len() {
        let c = &code[i..i + LOAD_LEN];
        let rex = c[1];
        let reg = usize::from((c[3] >> 3) & 7) | if rex & 4 != 0 { 8 } else { 0 };
        // shortcut: a byte match, not a decode. In the title all
        // 53,246 hits are instruction starts and none is missed (checked
        // against a full disassembly); decode .text if a title shows one.
        if c[0] == 0x65
            && (rex == 0x48 || rex == 0x4C)
            && c[2] == 0x8B
            && c[3] & 0xC7 == 0x04
            && c[4..] == [0x25, 0x58, 0, 0, 0]
            && reg != 4
        {
            let target = thunks
                .checked_add(reg * THUNK_SIZE)
                .ok_or_else(|| anyhow::anyhow!("TLS thunk address overflow"))?;
            let next_rip = at
                .checked_add(i + 5)
                .ok_or_else(|| anyhow::anyhow!("TLS call site address overflow"))?;
            let rel = i32::try_from(target as i128 - next_rip as i128).map_err(|_| {
                anyhow::anyhow!(
                    "TLS thunk {target:#x} out of rel32 reach from {next_rip:#x}"
                )
            })?;
            code[i] = 0xE8;
            code[i + 1..i + 5].copy_from_slice(&rel.to_le_bytes());
            code[i + 5..i + 9].copy_from_slice(&[0x0F, 0x1F, 0x40, 0x00]);
            count += 1;
            i += LOAD_LEN;
        } else {
            i += 1;
        }
    }
    Ok(count)
}

/// Fills `block` with one thunk per register: `mov r, gs:[0x58];
/// lea r, [r + index * 8]; ret` (flags untouched, like the load it replaces).
fn thunks(block: *mut u8, index: u32) -> Result<usize> {
    let disp = (index * 8).to_le_bytes();
    for reg in 0..16u8 {
        let (r, low) = (reg >> 3, reg & 7);
        let mut code = vec![0x65, 0x48 | r << 2, 0x8B, low << 3 | 4, 0x25, 0x58, 0, 0, 0];
        code.extend([0x48 | r << 2 | r, 0x8D, 0x80 | low << 3 | low]);
        if low == 4 {
            code.push(0x24); // SIB: [r12 + disp32]
        }
        code.extend(disp);
        code.push(0xC3);
        // SAFETY: `block` holds 16 thunks and is still writable.
        unsafe {
            std::ptr::copy_nonoverlapping(
                code.as_ptr(),
                block.add(usize::from(reg) * THUNK_SIZE),
                code.len(),
            )
        };
    }
    let mut old = 0;
    // SAFETY: `block` is our allocation of one page.
    if unsafe { VirtualProtect(block as _, 4096, PAGE_EXECUTE_READ, &mut old) } == 0 {
        bail!(
            "VirtualProtect(TLS thunks) failed: {}",
            std::io::Error::last_os_error()
        );
    }
    // SAFETY: flushing our own newly written thunk page.
    unsafe { FlushInstructionCache(GetCurrentProcess(), block as _, 4096) };
    Ok(block as usize)
}

/// Candidate non-null 64 KiB-aligned allocation hints below and above `[base, base + size)`.
fn near_hints(base: usize, size: usize, k: usize) -> [Option<usize>; 2] {
    const GRANULE: usize = 0x1_0000;
    let below = (k + 1)
        .checked_mul(GRANULE)
        .and_then(|d| base.checked_sub(d))
        // Never pass 0 (`NULL`) to `VirtualAlloc`, which lets Windows pick any
        // 48-bit address outside rel32 reach.
        .filter(|&addr| addr >= GRANULE);
    let above = base
        .checked_add(size)
        .and_then(|end| end.checked_next_multiple_of(GRANULE))
        .and_then(|start| k.checked_mul(GRANULE).and_then(|d| start.checked_add(d)))
        .filter(|&addr| addr >= GRANULE);
    [below, above]
}

/// True if every thunk in `[page, page + 16 * THUNK_SIZE)` lies within signed
/// 32-bit relative call reach of every byte in `[base, base + size)`.
fn in_rel32_reach(base: usize, size: usize, page: usize) -> bool {
    let Some(img_end) = base.checked_add(size) else {
        return false;
    };
    let Some(thunk_last) = page.checked_add(15 * THUNK_SIZE) else {
        return false;
    };
    let min_rel = page as i128 - img_end as i128;
    let max_rel = thunk_last as i128 - base as i128;
    i32::try_from(min_rel).is_ok() && i32::try_from(max_rel).is_ok()
}

/// One writable page within `rel32` reach of `[base, base + size)`. Leaked:
/// title code calls into it for the life of the process.
fn near(base: usize, size: usize) -> Result<*mut u8> {
    // The search spans 1 GiB each side, so every call site stays in reach.
    if size >= 1 << 30 {
        bail!("image too large for TLS thunks within rel32 reach");
    }
    for k in 0..(1usize << 14) {
        for hint in near_hints(base, size, k).into_iter().flatten() {
            if !in_rel32_reach(base, size, hint) {
                continue;
            }
            // SAFETY: a fresh allocation at a non-null free address, or null.
            let p = unsafe {
                VirtualAlloc(
                    hint as *const c_void,
                    4096,
                    MEM_COMMIT | MEM_RESERVE,
                    PAGE_READWRITE,
                )
            };
            if !p.is_null() {
                return Ok(p.cast());
            }
        }
    }
    bail!("no free page near the image for TLS thunks")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reads the thread's TLS array pointer the way title code does.
    fn tls_array() -> usize {
        let array: usize;
        // SAFETY: reads TEB.ThreadLocalStoragePointer.
        unsafe { std::arch::asm!("mov {}, gs:[0x58]", out(reg) array) };
        array
    }

    #[test]
    fn loads_become_calls_that_yield_the_title_slot() {
        let page = near(tls_array as *const () as usize, 0).unwrap();
        let thunks = thunks(near(page as usize, 4096).unwrap(), 3).unwrap();
        // (before the load, REX, ModRM, after the load): each case returns
        // the loaded register in rax; r12 is saved, being callee-saved.
        let cases: [(&[u8], u8, u8, &[u8]); 3] = [
            (&[], 0x48, 0x04, &[0xC3]),                   // rax
            (&[], 0x4C, 0x04, &[0x4C, 0x89, 0xC0, 0xC3]), // r8
            (
                &[0x41, 0x54],
                0x4C,
                0x24,
                &[0x4C, 0x89, 0xE0, 0x41, 0x5C, 0xC3],
            ), // r12
        ];
        for (before, rex, modrm, after) in cases {
            let mut code = before.to_vec();
            code.extend([0x65, rex, 0x8B, modrm, 0x25, 0x58, 0, 0, 0]);
            code.extend(after);
            // SAFETY: the page is ours and writable.
            let bytes = unsafe { std::slice::from_raw_parts_mut(page, code.len()) };
            bytes.copy_from_slice(&code);
            assert_eq!(patch(bytes, page as usize, thunks).unwrap(), 1);
            let mut old = 0;
            // SAFETY: our page, made executable to run the patched code.
            unsafe { VirtualProtect(page as _, 4096, PAGE_EXECUTE_READ, &mut old) };
            // SAFETY: the page holds the patched load and a return.
            let f: extern "system" fn() -> usize = unsafe { std::mem::transmute(page) };
            assert_eq!(f(), tls_array() + 3 * 8, "REX {rex:#x} ModRM {modrm:#x}");
            // SAFETY: our page again.
            unsafe { VirtualProtect(page as _, 4096, PAGE_READWRITE, &mut old) };
        }
    }

    #[test]
    fn near_hints_never_emit_null_and_patch_rejects_out_of_rel32_range() {
        // When `base == (k + 1) * 0x10000`, `below` must be None (never 0/NULL).
        assert_eq!(near_hints(0x1_0000, 0x1000, 0), [None, Some(0x2_0000)]);
        assert_eq!(near_hints(0x2_0000, 0x1000, 1), [None, Some(0x4_0000)]);
        assert!(!in_rel32_reach(0x1_0000_0000, 0x1000, 0x3_0000_0000));
        assert!(in_rel32_reach(0x1_0000_0000, 0x1000, 0x1_0001_0000));

        // `patch()` must return an error and leave code untouched if thunks
        // are more than 2 GiB away instead of silently truncating `rel as i32`.
        let orig = [0x65, 0x48, 0x8B, 0x04, 0x25, 0x58, 0, 0, 0];
        let mut buf = orig;
        assert!(patch(&mut buf, 0x1_0000_0000, 0x3_0000_0000).is_err());
        assert_eq!(buf, orig);
    }
}
