//! `xmem.dll`: console memory, served from host memory. The GPU side works
//! from CPU addresses (unified memory), so memory types and bandwidth
//! classes change nothing here; what constrains addresses (alignment,
//! 2 MiB pages, address-bit limits) is honoured.
#![allow(non_snake_case)]
// Title code calls these through its import table, passing pointers the real
// API would trust.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::alloc::{Layout, alloc, dealloc};
use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::atomic::{AtomicU8, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, OnceLock};

use windows_sys::Win32::Foundation::{
    CloseHandle, ERROR_COMMITMENT_LIMIT, HANDLE, INVALID_HANDLE_VALUE,
};
use windows_sys::Win32::System::Memory::{
    CreateFileMappingW, MEM_ADDRESS_REQUIREMENTS, MEM_COMMIT, MEM_EXTENDED_PARAMETER,
    MEM_PRESERVE_PLACEHOLDER, MEM_RELEASE, MEM_REPLACE_PLACEHOLDER, MEM_RESERVE,
    MEM_RESERVE_PLACEHOLDER, MEMORY_BASIC_INFORMATION, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile3,
    MemExtendedParameterAddressRequirements, PAGE_NOACCESS, PAGE_READWRITE, SEC_COMMIT,
    SEC_RESERVE, UnmapViewOfFile2, VirtualAlloc, VirtualAlloc2, VirtualFree, VirtualQueryEx,
};
use windows_sys::Win32::System::ProcessStatus::{
    K32GetPerformanceInfo, K32GetProcessMemoryInfo, PERFORMANCE_INFORMATION,
    PROCESS_MEMORY_COUNTERS,
};
use windows_sys::Win32::System::SystemInformation::{GetSystemInfo, SYSTEM_INFO};
use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetCurrentThread};
use windows_sys::core::BOOL;

use crate::{HleModule, hle_export};

const MIB: usize = 1 << 20;

/// Whether we run under Wine (its ntdll exports `wine_get_version`).
pub(crate) fn on_wine() -> bool {
    static WINE: LazyLock<bool> = LazyLock::new(|| {
        use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleA, GetProcAddress};
        // SAFETY: plain lookups in ntdll, always loaded.
        unsafe {
            let ntdll = GetModuleHandleA(c"ntdll.dll".as_ptr().cast());
            !ntdll.is_null() && GetProcAddress(ntdll, c"wine_get_version".as_ptr().cast()).is_some()
        }
    });
    *WINE
}

/// Standard memory the title has handed to the system.
static TRANSFERRED: AtomicUsize = AtomicUsize::new(0);

// Console additions to the host's flags.
/// 2 MiB pages; the host reads this bit as large pages, which need a
/// privilege.
const PAGES_2M: u32 = 0x2000_0000;
/// With `PAGES_2M`: 64 KiB pages; the host reads it as AWE memory.
const PAGES_64K_BIT: u32 = 0x40_0000;
const PAGES_64K: u32 = PAGES_2M | PAGES_64K_BIT;
/// The host's protection constants and `PAGE_GUARD`. Above them the console
/// adds GPU access bits (0x800-0x40000). Write-combining and uncached are
/// dropped too: our GPU side reads this memory through the CPU.
const HOST_PROTECTION: u32 = 0x1ff;
/// XMem flags of memory both processors reach: all of ours.
const GRAPHICS: u64 = 3;
/// XMem flag bit of 32-bit graphics memory.
const ADDRESS_32BIT: u64 = 4;
/// The host's write watching (not in windows-sys' memory module).
const MEM_WRITE_WATCH: u32 = 0x20_0000;

/// The addresses XMem `flags` allow, first and last: the low 4 GiB for
/// 32-bit memory; else from 4 GiB (kept for 32-bit requests, which the
/// title's big pools would otherwise crowd out) up to 46 down to 40 bits
/// from the field at bit 12, or 40 bits without one: the console keeps
/// title memory there, and its ray-tracing structures must be (
/// the title's BVH builder refuses one above; the title asks for its GPU heap
/// with no field). Decision 0001.
fn address_range(flags: u64) -> (usize, usize) {
    let bits = match (flags & ADDRESS_32BIT, flags >> 12 & 7) {
        (0, 0) => 40,
        (0, field) => 47 - field,
        _ => return (0, u32::MAX as usize),
    };
    (1 << 32, (1 << bits) - 1)
}

pub extern "system" fn XMemVirtualAlloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
) -> *mut c_void {
    virtual_alloc(
        base,
        size,
        kind,
        flags,
        protect,
        victory_gpu::title_memory_committed,
    )
}

fn virtual_alloc(
    base: *mut c_void,
    size: usize,
    kind: u32,
    flags: u64,
    protect: u32,
    on_commit: impl FnOnce(usize, usize),
) -> *mut c_void {
    let (lowest, highest) = address_range(flags);
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: lowest as *mut c_void,
        HighestEndingAddress: highest as *mut c_void,
        Alignment: if kind & (PAGES_2M | PAGES_64K_BIT) == PAGES_2M {
            2 * MIB
        } else {
            0
        },
    };
    let mut protect = match protect & HOST_PROTECTION {
        0 => PAGE_NOACCESS,
        p => p,
    };
    // A reservation of 64 KiB pages takes physical pages later
    // (`XMemMapPhysicalPages`); on the host only a placeholder can.
    // shortcut: committing part of such a reservation later would fail
    // (the title's 23 calls never do); replace that part of the
    // placeholder then.
    let placeholder = kind & PAGES_64K == PAGES_64K && kind & MEM_COMMIT == 0;
    if placeholder {
        if kind & MEM_REPLACE_PLACEHOLDER != 0 {
            // Part of a placeholder the title split: it stays one.
            return base;
        }
        protect = PAGE_NOACCESS;
    }
    let mut kind = kind & !(PAGES_2M | PAGES_64K_BIT);
    if placeholder {
        kind |= MEM_RESERVE | MEM_RESERVE_PLACEHOLDER;
    }
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // Address requirements apply to reservations at an address of our choice.
    let constrained = base.is_null();
    if constrained {
        kind |= MEM_RESERVE;
        // The GPU side copies only the pages the title wrote since its last
        // copy (`victory_gpu`, decision 0007); placeholders cannot be
        // watched (their physical-page views are copied whole).
        if kind & MEM_RESERVE_PLACEHOLDER == 0 {
            kind |= MEM_WRITE_WATCH;
        }
    }
    // SAFETY: the title's own arguments, cleaned of console bits; the
    // parameter points at `requirements`, which outlives the call.
    let at = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            base,
            size,
            kind,
            protect,
            if constrained {
                &mut parameter
            } else {
                std::ptr::null_mut()
            },
            constrained.into(),
        )
    };
    if base.is_null() {
        tracing::debug!(
            size,
            kind = format_args!("{kind:#x}"),
            flags = format_args!("{flags:#x}"),
            at = format_args!("{at:p}"),
            "XMemVirtualAlloc"
        );
    }
    // Memory committed for the title: the GPU side's notes that blocks there
    // are uncommitted go (`VICTORY_ABSENT_ONCE`).
    if !at.is_null() && kind & MEM_COMMIT != 0 {
        // Windows rounds a fixed commit's address down to a page. The
        // caller's last byte can be in the next 64 KiB watch block.
        let start = if base.is_null() { at } else { base };
        on_commit(start as usize, size);
    }
    at
}

/// `MEMORY_BASIC_INFORMATION` (the console's first 0x30 bytes are the
/// same), then the XMem flags and tag.
#[repr(C)]
pub struct BasicInformation {
    host: MEMORY_BASIC_INFORMATION,
    flags: u64,
    tag_code: u32,
    tag: [u8; 4],
}

// shortcut: every committed page reports graphics memory and no tag, since
// all of it is GPU-reachable here (the title's BVH builder, the
// only caller, wants graphics memory); remember the flags of
// `XMemVirtualAlloc` regions if a title tells CPU-only memory apart.
pub extern "system" fn XMemVirtualQuery(
    process: HANDLE,
    address: *const c_void,
    out: *mut BasicInformation,
) -> BOOL {
    let size = size_of::<MEMORY_BASIC_INFORMATION>();
    // SAFETY: the title passes room for the whole structure; the host
    // fails on a null one.
    unsafe {
        if VirtualQueryEx(process, address, out.cast(), size) == 0 {
            return 0;
        }
        let info = &mut *out;
        info.flags = if info.host.State == MEM_COMMIT {
            GRAPHICS
        } else {
            0
        };
        info.tag_code = 0;
        info.tag = [0; 4];
    }
    1
}

// `XMemAlloc` attributes: alignment as log2 (5 bits at 24, 4 bytes and up
// from 2), then the memory type, then bit 33: a 32-bit address.
const ALIGNMENT_SHIFT: u64 = 24;
const ALLOC_32BIT: u64 = 1 << 33;
/// Our block header: the block's size and alignment, just below the
/// pointer handed out.
const HEADER: usize = 2 * size_of::<usize>();

fn layout(size: usize, attributes: u64) -> Option<Layout> {
    // An alignment of at least 16 leaves room for the header.
    let align = 1usize << (attributes >> ALIGNMENT_SHIFT & 0x1f).max(4);
    Layout::from_size_align(size.checked_add(align)?, align).ok()
}

type AllocRoutine = extern "system" fn(usize, u64) -> *mut c_void;
type FreeRoutine = extern "system" fn(*mut c_void, u64);

/// The allocation and free routines the title's exe exports as
/// `__XMemAllocationHooks__` (the title does): `XMemAlloc` and
/// `XMemFree` call them, for the title's own calls and its middleware's;
/// the `…Default` exports do not.
fn hooks() -> [usize; 2] {
    static AT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let at = match AT.get() {
        Some(&at) => at,
        // Looked up once the title is mapped, which is before it runs.
        None => match crate::runtime::main_image() {
            Some(base) => *AT.get_or_init(|| {
                use windows_sys::Win32::System::LibraryLoader::GetProcAddress;
                // SAFETY: a mapped image and a NUL-terminated name.
                let at = unsafe {
                    GetProcAddress(base as _, c"__XMemAllocationHooks__".as_ptr().cast())
                }
                .map_or(0, |f| f as usize);
                tracing::info!(found = at != 0, "xmem allocation hooks");
                at
            }),
            None => 0,
        },
    };
    if at == 0 {
        return [0; 2];
    }
    // SAFETY: the export is two routine pointers.
    unsafe { (at as *const [usize; 2]).read() }
}

/// The title's general allocator, for middleware too: its hook if it has
/// one, else [`XMemAllocDefault`].
pub extern "system" fn XMemAlloc(size: usize, attributes: u64) -> *mut c_void {
    match hooks()[0] {
        0 => XMemAllocDefault(size, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let alloc: AllocRoutine = unsafe { std::mem::transmute(hook) };
            alloc(size, attributes)
        }
    }
}

/// See [`XMemAlloc`].
pub extern "system" fn XMemFree(address: *mut c_void, attributes: u64) {
    match hooks()[1] {
        0 => XMemFreeDefault(address, attributes),
        hook => {
            // SAFETY: the title's own routine of this type.
            let free: FreeRoutine = unsafe { std::mem::transmute(hook) };
            free(address, attributes)
        }
    }
}

/// The system allocator behind [`XMemAlloc`].
pub extern "system" fn XMemAllocDefault(size: usize, attributes: u64) -> *mut c_void {
    if attributes & ALLOC_32BIT != 0 {
        return XMemVirtualAlloc(
            std::ptr::null_mut(),
            size,
            MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | ADDRESS_32BIT,
            PAGE_READWRITE,
        );
    }
    let Some(layout) = layout(size, attributes) else {
        return std::ptr::null_mut();
    };
    // shortcut: the host heap's addresses lie above 2^40, unlike the
    // console's; nothing seen builds ray-tracing structures in this memory
    // (the title uses its GPU heap). Sub-allocate from a
    // reservation below 2^40 if one does.
    // SAFETY: the layout has a non-zero size.
    let block = unsafe { alloc(layout) };
    if block.is_null() {
        return block.cast();
    }
    // SAFETY: the block holds the alignment (at least the header) and then
    // `size` bytes.
    unsafe {
        let user = block.add(layout.align());
        user.sub(HEADER)
            .cast::<[usize; 2]>()
            .write([layout.size(), layout.align()]);
        user.cast()
    }
}

/// Frees what [`XMemAllocDefault`] returned, given the same attributes.
pub extern "system" fn XMemFreeDefault(address: *mut c_void, attributes: u64) {
    if address.is_null() {
        return;
    }
    // SAFETY: the address came from `XMemAlloc` with these attributes: a
    // whole reservation, or a header-prefixed block.
    unsafe {
        if attributes & ALLOC_32BIT != 0 {
            VirtualFree(address, 0, MEM_RELEASE);
            return;
        }
        let user = address.cast::<u8>();
        let [size, align] = user.sub(HEADER).cast::<[usize; 2]>().read();
        dealloc(
            user.sub(align),
            Layout::from_size_align_unchecked(size, align),
        );
    }
}

/// A retail console has no tools memory.
pub extern "system" fn XMemIsToolsMemAvailable() -> BOOL {
    0
}

#[link(name = "kernel32", kind = "raw-dylib")]
unsafe extern "system" {
    /// The processor cycles a thread has used so far, in user and kernel
    /// mode.
    fn QueryThreadCycleTime(thread: HANDLE, cycles: *mut u64) -> i32;
}

/// Whether the remapping meter also reads the calling thread's processor
/// cycles around each of the system's unmapping and mapping calls
/// (`VICTORY_REMAP_METER=0`: not; the counts and the lengths are kept
/// either way). It costs two short system calls a metered call.
fn meter_cycles() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REMAP_METER").is_none_or(|v| v != "0"))
}

/// One of the system's calls that change what an address range shows,
/// metered for the frame log (`victory_gpu::framelog`, "remapping,
/// metered"): how long it took, and how many processor cycles the calling
/// thread used meanwhile against the processor's clock over the same
/// span. A thread that stood blocked in the kernel used few: run 307's
/// profile found the title's thread on a processor 10 to 15 % of its time
/// in these calls, on the one thread it followed.
struct Meter {
    started: std::time::Instant,
    cycles: u64,
    tsc: u64,
}

impl Meter {
    fn begin() -> Meter {
        let mut cycles = 0;
        if meter_cycles() {
            // SAFETY: the calling thread's own pseudo-handle and a valid
            // out pointer.
            unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) };
        }
        Meter {
            started: std::time::Instant::now(),
            cycles,
            // SAFETY: reads the processor's counter.
            tsc: unsafe { std::arch::x86_64::_rdtsc() },
        }
    }

    /// Ends the span and counts it: `mapping` false for an unmapping call,
    /// true for a mapping call.
    fn end(self, mapping: bool) {
        use victory_gpu::framelog::{T, add};
        // SAFETY: reads the processor's counter.
        let tsc = unsafe { std::arch::x86_64::_rdtsc() }.wrapping_sub(self.tsc);
        let us = self.started.elapsed().as_micros() as u64;
        let (calls, cycles_of, tsc_of, count, time) = if mapping {
            (
                T::RemapMapCalls,
                T::RemapMapCycles,
                T::RemapMapTsc,
                [T::RemapMapC0, T::RemapMapC1, T::RemapMapC2, T::RemapMapC3],
                [T::RemapMapU0, T::RemapMapU1, T::RemapMapU2, T::RemapMapU3],
            )
        } else {
            (
                T::RemapUnmapCalls,
                T::RemapUnmapCycles,
                T::RemapUnmapTsc,
                [T::RemapUnmapC0, T::RemapUnmapC1, T::RemapUnmapC2, T::RemapUnmapC3],
                [T::RemapUnmapU0, T::RemapUnmapU1, T::RemapUnmapU2, T::RemapUnmapU3],
            )
        };
        add(calls, 1);
        let class = match us {
            0..50 => 0,
            50..200 => 1,
            200..1000 => 2,
            _ => 3,
        };
        add(count[class], 1);
        add(time[class], us);
        if meter_cycles() {
            let mut cycles = 0;
            // SAFETY: as in `begin`.
            unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut cycles) };
            add(cycles_of, cycles.saturating_sub(self.cycles));
            add(tsc_of, tsc);
        }
    }
}

/// Whether a mapping is tried before the placeholder is split, and the
/// split made only if the mapping failed (`VICTORY_REMAP_SPLIT=late`;
/// `alt`: twenty seconds so, twenty as before, each change said in the
/// log; unset: the split is tried first every time, as before).
///
/// The split fails harmlessly when the range is a whole placeholder, which
/// it is after a page's own view was unmapped: then it is one call into
/// the system for nothing, and the title's thread takes its turn in the
/// kernel for it (run 307's profile: 1.05 ms a frame beside 8.75 ms of
/// mapping on the title's frame thread, on a processor 16 % of that
/// time). Tried late, a mapping into a range that is part of a larger
/// placeholder fails first (the range must match the placeholder exactly),
/// is split and tried again: two mapping calls and a split where there
/// was one of each. The frame log counts both ("splits tried, needed",
/// "tried again after a split").
/// A mapping that fails is taken to leave the placeholder as it was.
fn split_late() -> bool {
    use std::sync::atomic::{AtomicU8, Ordering::Relaxed};
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_REMAP_SPLIT").as_deref() {
        Ok("late") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: AtomicU8 = AtomicU8::new(2);
            if LAST.swap(u8::from(on), Relaxed) != u8::from(on) {
                tracing::info!(
                    "a placeholder is split only after a mapping into it failed from now (VICTORY_REMAP_SPLIT=alt): {}",
                    if on { "ON" } else { "off, split first as usual" }
                );
            }
            on
        }
    }
}

// Physical pages: on the console, 64 KiB frames a title maps into its own
// reservations, at more than one address if it likes. Here each allocation
// is a pagefile-backed section, its pages are 64 KiB slices of it, and a
// mapping is a view of consecutive slices in place of a host placeholder.

const PAGE: usize = 64 << 10;

/// The top of the console's own title memory: 40 bits (the title's BVH
/// builder refuses a structure above it, decision 0001). A persistent view
/// is placed at or above this address, so it can never be mistaken for
/// title memory and can never take an address the title is about to ask
/// for. The ceiling comes from `GetSystemInfo`, never from a constant:
/// a host whose application addresses stop below this maps nothing
/// persistent and every write keeps its temporary view.
const TITLE_MEMORY_CEILING: usize = 1 << 40;

/// A page's commit state, one `AtomicU8` each. A thread that finds
/// [`COMMITTING`] waits instead of reading or writing a page whose commit is
/// still in flight, which is the whole point: `VirtualAlloc(MEM_COMMIT)` is
/// not atomic with respect to another thread's store into the page.
const FREE: u8 = 0;
const COMMITTING: u8 = 1;
const COMMITTED: u8 = 2;

/// How many times a thread waits for another thread's commit before it gives
/// up on the persistent path. Giving up is safe (the caller falls back to a
/// temporary view, which is always correct); only a thread that died
/// mid-commit can exhaust this, and it must not hang the title.
const COMMIT_SPINS: usize = 1 << 14;

struct Section {
    handle: HANDLE,
    /// The section's whole range, mapped the first time one of its pages is
    /// written and kept until the section goes; `None` once the host has
    /// refused that mapping. See [`Section::persistent_ptr`].
    persistent: OnceLock<Option<MappedBase>>,
    /// The section's size in bytes.
    size: usize,
    /// One three-state `AtomicU8` per 64 KiB page ([`FREE`], [`COMMITTING`],
    /// [`COMMITTED`]). Not a saved call: the DirectStorage workers write
    /// this concurrently, and a page's state leaves `COMMITTING` only after
    /// `VirtualAlloc` returned, so nobody stores into a page that is not
    /// committed yet. It records this view's own commits only; the
    /// temporary path's are not recorded, by design.
    commit_state: Box<[AtomicU8]>,
    /// One generation counter per 64 KiB page, bumped after the bytes of that
    /// page are written (`Section::bump_range`). A consumer that sees a
    /// bumped counter has, by the release/acquire pair, seen the bytes; that
    /// is what lets the GPU side skip hashing texture memory it already
    /// knows (decision 0007's `VICTORY_PAGE_WRITES` counts, phase F's
    /// digest compares).
    generation: Box<[AtomicU32]>,
    /// Whether a warn has been said for a write that fell back to a
    /// temporary view, so a run with the switch on can tell which sections
    /// are really persistent.
    fallback_noted: OnceLock<()>,
    /// Bytes `[from, from + len)` of this section, from its start, that live
    /// persistent views hold. The section has one address, so two writers may
    /// share it only when their bytes do not touch; a claim that does is
    /// refused and the caller takes a temporary view instead.
    holding: Mutex<Vec<(usize, usize)>>,
}

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Section {
    fn new(handle: HANDLE, size: usize, pages: usize) -> Section {
        Section {
            handle,
            persistent: OnceLock::new(),
            size,
            commit_state: (0..pages).map(|_| AtomicU8::new(FREE)).collect(),
            generation: (0..pages).map(|_| AtomicU32::new(0)).collect(),
            fallback_noted: OnceLock::new(),
            holding: Mutex::new(Vec::new()),
        }
    }

    /// Claims bytes `[from, from + len)` of the section for one persistent
    /// view. False when a live one already holds bytes that touch: the
    /// section's mapping is a single address, so sharing it is sharing a
    /// borrow, and only writers whose bytes are disjoint may do that.
    fn hold(&self, from: usize, len: usize) -> bool {
        if len == 0 {
            return true;
        }
        let mut holding = self.holding.lock().unwrap_or_else(|e| e.into_inner());
        let end = from.saturating_add(len);
        if holding
            .iter()
            .any(|&(at, bytes)| from < at.saturating_add(bytes) && at < end)
        {
            return false;
        }
        holding.push((from, len));
        true
    }

    /// Gives back what [`Section::hold`] claimed.
    fn let_go(&self, from: usize, len: usize) {
        let mut holding = self.holding.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(held) = holding.iter().position(|&held| held == (from, len)) {
            holding.swap_remove(held);
        }
    }

    /// Says once per section that a write fell back to a temporary view.
    /// Without it a run with the switch on cannot tell a section that
    /// persisted from one that did not, which is the evidence the switch
    /// becoming the default is supposed to rest on.
    fn note_fallback(&self) {
        if self.fallback_noted.set(()).is_ok() {
            tracing::warn!(
                size = self.size,
                "a commit through the persistent view failed; this section's pages keep a temporary view per write"
            );
        }
    }

    /// The section's whole range, mapped read-write once and kept. A write
    /// through it is a store; a view per write is a map, a commit and an
    /// unmap, and every unmap flushes every processor's TLB.
    ///
    /// `None` when the host refuses the mapping, which the title's largest
    /// allocation can provoke: it asks for 6.1 GiB at once, and this maps
    /// all of it. Callers then keep the temporary view per write.
    fn persistent_ptr(&self) -> Option<*mut u8> {
        let mapped = self
            .persistent
            .get_or_init(|| map_above_title_memory(self.handle, self.size).map(MappedBase));
        mapped.as_ref().map(|base| base.0)
    }

    /// Whether all of pages `[start, start + count)` are committed in the
    /// persistent view.
    fn committed_all(&self, start: usize, count: usize) -> bool {
        let Some(end) = start.checked_add(count) else {
            return false;
        };
        end <= self.commit_state.len()
            && (start..end).all(|page| self.commit_state[page].load(Ordering::Acquire) == COMMITTED)
    }

    /// Records pages as committed. Release: the commit's own work happens
    /// before anyone who reads this state.
    fn mark_committed(&self, start: usize, count: usize) {
        for page in start..start.saturating_add(count).min(self.commit_state.len()) {
            self.commit_state[page].store(COMMITTED, Ordering::Release);
        }
    }

    /// Gives claimed pages back after a commit that failed, so the next caller
    /// retries instead of waiting for this one forever.
    fn mark_free(&self, start: usize, count: usize) {
        for page in start..start.saturating_add(count).min(self.commit_state.len()) {
            let _ = self.commit_state[page].compare_exchange(
                COMMITTING,
                FREE,
                Ordering::Release,
                Ordering::Relaxed,
            );
        }
    }

    /// Bumps the write generation of pages `[first, first + count)`. Call it
    /// **after** the bytes are written: release ordering puts the bytes before
    /// the counter, and a consumer that sees the new counter (acquire) sees
    /// the bytes. Bumping before writing would let a consumer read the new
    /// generation with the old bytes still in place.
    fn bump_range(&self, first: usize, count: usize) {
        let Some(end) = first.checked_add(count) else {
            return;
        };
        for page in first..end.min(self.generation.len()) {
            self.generation[page].fetch_add(1, Ordering::Release);
        }
    }

    /// One number standing for the write generation of pages
    /// `[start, start + count)`, `None` outside the section.
// Phase F (the GPU side's texture-hash skip) is the caller; until it
    // lands, the only caller is the digest test below, so the library build
    // would otherwise call these unused.
        #[allow(dead_code)]
    fn generation_of(&self, start: usize, count: usize) -> Option<u64> {
        let end = start.checked_add(count)?;
        if end > self.generation.len() {
            return None;
        }
        let mut digest = 0x9e37_79b9_7f4a_7c15u64;
        for page in start..end {
            digest = mix64(digest ^ (page as u64).wrapping_mul(0x2545_f491_4f6c_dd1d));
            digest = mix64(digest ^ u64::from(self.generation[page].load(Ordering::Acquire)));
        }
        Some(digest)
    }

    /// Commits pages `[start_page, start_page + count)` of the persistent
    /// view at `base`. A page already committed costs no call at all, which is
    /// the point. False when the range leaves the section or the host refuses.
    ///
    /// The bounds check is a real guard, not a saved call: it is what keeps
    /// the page arithmetic inside `commit_state`, and a zero-length range past
    /// the end is refused by it alone (pinned by
    /// `ensure_committed_refuses_ranges_outside_the_section`).
    ///
    /// Between the claim (a page set to `COMMITTING`) and the release of it
    /// (`COMMITTED`, after `VirtualAlloc` returned) no thread may touch the
    /// page's bytes: a loser of the claim spins here until the state leaves
    /// `COMMITTING`. That is the difference between this and a bitmap of
    /// "committed": `VirtualAlloc(MEM_COMMIT)` is not atomic against another
    /// thread's store, and the DirectStorage workers write concurrently.
    ///
    /// The `committed_all` early return is the surviving saved call: a probe
    /// (2026-11-25) with the branch forced off still passes every test, since
    /// `commit` over a committed page succeeds and changes nothing; its effect
    /// is measured by a run (`VICTORY_XMEM_PERSISTENT_VIEW=1`), not asserted.
    fn ensure_committed(&self, base: *mut u8, start_page: usize, count: usize) -> bool {
        let Some(end_page) = start_page.checked_add(count) else {
            return false;
        };
        if !end_page
            .checked_mul(PAGE)
            .is_some_and(|end| end <= self.size)
        {
            return false;
        }
        if self.committed_all(start_page, count) {
            return true;
        }
        for _ in 0..COMMIT_SPINS {
            if self.committed_all(start_page, count) {
                return true;
            }
            // Claim what nobody else is committing. Losing every claim means
            // another thread owns this range right now: wait for it.
            let mut claimed = false;
            for page in start_page..end_page {
                if self.commit_state[page]
                    .compare_exchange(FREE, COMMITTING, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    claimed = true;
                }
            }
            if !claimed {
                std::hint::spin_loop();
                continue;
            }
            // One commit for the whole range: pages committed before are left
            // as they are. A host at its commit limit is reported here and, if
            // the caller then retries on the temporary path, reported again by
            // that path's own `commit` -- two crash-log lines for one failure,
            // which is asked for rather than avoided: the retry is what gives
            // the caller the same answer it would have had before.
            // SAFETY: `[start_page, end_page)` lies inside the section's view.
            let at = unsafe { base.add(start_page * PAGE) };
            if !commit(at as usize, count * PAGE) {
                self.mark_free(start_page, count);
                return false;
            }
            self.mark_committed(start_page, count);
            return true;
        }
        // Another thread claimed the range and has not finished after
        // `COMMIT_SPINS`: give up rather than hang the title. The caller's
        // temporary view is correct, so this costs speed, not answers.
        false
    }
}

/// Mixes a 64-bit word (splitmix64's finaliser): one flipped bit of the input
/// changes half the output's bits, so two neighbouring pages' generations
/// cannot cancel out in a fold.
// Phase F (the GPU side's texture-hash skip) is the caller; until it
// lands, the only caller is the digest test below, so the library build
// would otherwise call these unused.
#[allow(dead_code)]
fn mix64(value: u64) -> u64 {
    let mut x = value;
    x ^= x >> 30;
    x = x.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 31)
}

/// Whether `at` is a place the persistent view may live: at or above the
/// title's own memory. Split out from the mapping so the rule has a name a
/// test can hold, since a host whose allocator is already top-down returns
/// such an address with or without the requirement.
fn above_title_memory(at: usize) -> bool {
    at >= TITLE_MEMORY_CEILING
}

/// The highest address the host's application may use, less one page: the
/// upper bound of an address requirement. Asked, never assumed.
fn highest_ending_address() -> usize {
    let mut info = SYSTEM_INFO::default();
    // SAFETY: `GetSystemInfo` writes only into the struct it is given.
    unsafe { GetSystemInfo(&mut info) };
    (info.lpMaximumApplicationAddress as usize).saturating_sub(PAGE)
}

/// Maps `handle`'s whole `size` read-write at or above
/// [`TITLE_MEMORY_CEILING`], and returns that address; `None` when the host
/// refuses, which keeps the caller on a temporary view per write.
///
/// The address matters: title memory lives below 40 bits, and a view placed
/// inside that range either takes an address the title is about to ask for
/// or looks like a title address to the title's own arithmetic. Reserving the
/// placeholder with `VirtualAlloc2`'s address requirement is how the choice is
/// made; `MEM_REPLACE_PLACEHOLDER` then maps the section into the reservation
/// rather than at an address of the host's choosing.
fn map_above_title_memory(handle: HANDLE, size: usize) -> Option<*mut u8> {
    let rounded = size.max(PAGE).checked_next_multiple_of(PAGE)?;
    let mut requirements = MEM_ADDRESS_REQUIREMENTS {
        LowestStartingAddress: TITLE_MEMORY_CEILING as *mut c_void,
        HighestEndingAddress: highest_ending_address() as *mut c_void,
        Alignment: PAGE,
    };
    let mut parameter = MEM_EXTENDED_PARAMETER::default();
    parameter.Anonymous1._bitfield = MemExtendedParameterAddressRequirements as u64;
    parameter.Anonymous2.Pointer = (&raw mut requirements).cast();
    // SAFETY: the parameter points at `requirements`, which outlives the call.
    let placeholder = unsafe {
        VirtualAlloc2(
            GetCurrentProcess(),
            std::ptr::null(),
            rounded,
            MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
            PAGE_NOACCESS,
            &mut parameter,
            1,
        )
    };
    if placeholder.is_null() {
        tracing::warn!(
            size,
            error = %std::io::Error::last_os_error(),
            "reserving a view above the title's own addresses failed; its pages keep a temporary view per write"
        );
        return None;
    }
    // SAFETY: our reservation, replaced in place by a view of our section.
    let view = unsafe {
        MapViewOfFile3(
            handle,
            GetCurrentProcess(),
            placeholder,
            0,
            rounded,
            MEM_REPLACE_PLACEHOLDER,
            PAGE_READWRITE,
            std::ptr::null_mut(),
            0,
        )
    };
    let at = view.Value as usize;
    if at != placeholder as usize {
        // SAFETY: the reservation was not replaced (the mapping failed), so it
        // is still ours to release.
        unsafe { VirtualFree(placeholder, 0, MEM_RELEASE) };
        tracing::warn!(
            size,
            error = %std::io::Error::last_os_error(),
            "mapping a physical-page section whole above the title's addresses failed; its pages keep a temporary view per write"
        );
        return None;
    }
    if !above_title_memory(at) {
        // The host ignored the requirement. Using this address is exactly what
        // the ceiling exists to prevent, so refuse it and take the slow path.
        // SAFETY: the view just mapped; releasing it also releases the
        // placeholder, which was not preserved.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, 0) };
        tracing::warn!(
            at = format_args!("{at:#x}"),
            "the host placed a persistent view inside the title's own addresses; its pages keep a temporary view per write"
        );
        return None;
    }
    Some(at as *mut u8)
}

/// The section's whole range, mapped once and unmapped when the section
/// goes rather than after every write.
struct MappedBase(*mut u8);

// SAFETY: a mapping may be written from any thread; its writers write bytes
// of their own, the rule a title's own views of the same pages follow.
unsafe impl Send for MappedBase {}
// SAFETY: as above.
unsafe impl Sync for MappedBase {}

impl Drop for MappedBase {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        // SAFETY: our view. Closing the section's handle first is safe: a
        // view keeps the section object alive on its own.
        unsafe {
            UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                Value: self.0.cast(),
            });
        }
    }
}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.handle) };
    }
}

/// Where a page lives, or what a view starts with: a section and a page
/// index in it.
#[derive(Clone)]
struct Slice {
    section: Arc<Section>,
    index: usize,
}

impl Slice {
    /// The key in `Physical::shown` of the page `k` pages after this one.
    fn key(&self, k: usize) -> (usize, usize) {
        (Arc::as_ptr(&self.section) as usize, self.index + k)
    }
}

#[derive(Default)]
struct Physical {
    next_page: usize,
    pages: HashMap<usize, Slice>,
    /// Mapped views by address: their length and what they show.
    views: BTreeMap<usize, (usize, Slice)>,
    /// Where each page shows: by its section and its index there, the
    /// address of that page in every view that shows it now ([`shown_at`];
    /// as upstream's c75fdba keeps it). The title maps pages where the
    /// card reads them and has them filled afterwards by DirectStorage,
    /// which writes through views of our own.
    shown: HashMap<(usize, usize), Vec<usize>>,
}

static PHYSICAL: LazyLock<Mutex<Physical>> = LazyLock::new(|| {
    Mutex::new(Physical {
        // Page numbers are opaque to titles; 0 stays unused.
        next_page: 0x10_0000,
        ..Default::default()
    })
});

fn physical() -> MutexGuard<'static, Physical> {
    PHYSICAL.lock().unwrap_or_else(|e| e.into_inner())
}

/// `count` is in and out (pages wanted, pages given); all or nothing here.
pub extern "system" fn XMemAllocatePhysicalPages(
    flags: u64,
    count: *mut usize,
    pages: *mut usize,
) -> BOOL {
    if count.is_null() || pages.is_null() {
        return 0;
    }
    // SAFETY: checked non-null.
    let n = unsafe { *count };
    let Some(size) = n.checked_mul(PAGE).filter(|&s| s > 0) else {
        return 0;
    };
    // A pagefile-backed section of `size` bytes, its pages committed when
    // first mapped or written (`commit`): the title asks for 6.1
    // GiB at once, which a busy host often cannot commit (error 1450).
    // SAFETY: plain section creation.
    let handle = unsafe {
        CreateFileMappingW(
            INVALID_HANDLE_VALUE,
            std::ptr::null(),
            // Under Wine a committed section is a sparse file, free until
            // touched, and committing a reserved one page by page costs a
            // wineserver request each that scans every range committed so
            // far: commit it whole there.
            PAGE_READWRITE | if on_wine() { SEC_COMMIT } else { SEC_RESERVE },
            (size as u64 >> 32) as u32,
            size as u32,
            std::ptr::null(),
        )
    };
    if handle.is_null() {
        let error = std::io::Error::last_os_error();
        tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
        // SAFETY: checked non-null.
        unsafe { count.write(0) };
        return 0;
    }
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
    // For the GPU side's in-place census: only a committed section can be
    // opened as a host heap.
    victory_gpu::title_memory_section(handle as usize, size as u64, on_wine());
    let section = Arc::new(Section::new(handle, size, n));
    let mut physical = physical();
    let first = physical.next_page;
    physical.next_page += n * FRAMES;
    for index in 0..n {
        let slice = Slice {
            section: section.clone(),
            index,
        };
        let number = first + index * FRAMES;
        physical.pages.insert(number, slice);
        // SAFETY: the title passes room for `n` page numbers.
        unsafe { pages.add(index).write(number) };
    }
    1
}

/// Page numbers count 4 KiB frames, so a 64 KiB page's is a multiple of
/// 16: the title keeps them as 21-bit `number >> 4` and rebuilds
/// them with `<< 4` (128 GiB of pages at most).
const FRAMES: usize = PAGE >> 12;

/// Pages still mapped stay readable through their views until unmapped.
pub extern "system" fn XMemFreePhysicalPages(n: usize, pages: *const usize) -> BOOL {
    if pages.is_null() {
        return 0;
    }
    let mut physical = physical();
    for i in 0..n {
        // SAFETY: the title passes `n` page numbers.
        physical.pages.remove(&unsafe { *pages.add(i) });
    }
    1
}

/// Parses `VICTORY_XMEM_PERSISTENT_VIEW`: off unless the value says yes, so
/// a run that sets nothing keeps the view per write. Opt-in while the
/// persistent path settles, because it maps each section whole and keeps
/// the mapping: 6.1 GiB of address space for the title's largest allocation.
pub(crate) fn parse_persistent_views(value: Option<&std::ffi::OsStr>) -> bool {
    let Some(value) = value else {
        return false;
    };
    let Some(text) = value.to_str() else {
        return false;
    };
    let trimmed = text.trim();
    trimmed == "1"
        || trimmed.eq_ignore_ascii_case("true")
        || trimmed.eq_ignore_ascii_case("yes")
        || trimmed.eq_ignore_ascii_case("on")
}

/// Whether pages are written through one persistent view per section
/// (`VICTORY_XMEM_PERSISTENT_VIEW=1`) instead of a view mapped, committed
/// and unmapped for every write.
pub(crate) fn persistent_views() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        parse_persistent_views(std::env::var_os("VICTORY_XMEM_PERSISTENT_VIEW").as_deref())
    })
}

/// How a write reaches its pages: the process default ([`persistent_views`]),
/// or one forced. Tests force either, since the environment is read once per
/// process and cannot be set from a test.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mapping {
    Default,
    Persistent,
    Temporary,
}

impl Mapping {
    /// The view `n` slices of `slice` on are written through.
    fn view(self, slice: &Slice, n: usize) -> Option<View> {
        let mapping = match self {
            Mapping::Default if persistent_views() => Mapping::Persistent,
            Mapping::Default => Mapping::Temporary,
            forced => forced,
        };
        match mapping {
            // The temporary path is the one every run before the persistent
            // one used, so a host refusing either half of the persistent
            // path lands there rather than failing a write.
            Mapping::Persistent => View::persistent(slice, n).or_else(|| View::temporary(slice, n)),
            // `Default` was resolved above: both are the old path.
            Mapping::Temporary | Mapping::Default => View::temporary(slice, n),
        }
    }
}

/// A view of our own on consecutive slices of one section, committed and
/// writable.
enum View {
    /// Mapped for this write alone, unmapped when dropped.
    Temporary {
        at: MEMORY_MAPPED_VIEW_ADDRESS,
        len: usize,
    },
    /// The section's persistent view, at `slice`'s pages. Nothing is unmapped
    /// when this drops: the view belongs to the section and goes with it.
    Persistent {
        /// Keeps the section, and so the view, alive while the bytes are
        /// used, and gives the held bytes back when this drops.
        section: Arc<Section>,
        ptr: *mut u8,
        /// Bytes `[from, from + len)` of the section this view holds.
        from: usize,
        len: usize,
    },
}

impl View {
    /// Maps `n` slices from `slice` on for this write alone.
    fn temporary(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // For the frame log: these views take their turn in the kernel
        // beside the title's own mapping calls.
        let started = std::time::Instant::now();
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.handle,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View::Temporary { at, len };
        let committed = commit(at.Value as usize, len);
        victory_gpu::framelog::add(victory_gpu::framelog::T::DiskViews, 1);
        victory_gpu::framelog::add(victory_gpu::framelog::T::DiskViewUs, started.elapsed().as_micros() as u64);
        committed.then_some(view)
    }

    /// The section's persistent view over `n` slices from `slice` on, those
    /// pages committed; `None` when the host refuses either half, which
    /// leaves the caller on [`View::temporary`].
    fn persistent(slice: &Slice, n: usize) -> Option<View> {
        let base = slice.section.persistent_ptr()?;
        let len = n.checked_mul(PAGE)?;
        let from = slice.index * PAGE;
        // The section's one mapping is one address, so two writers may share
        // it only when their bytes do not touch. A claim that touches a live
        // one is refused, and the caller takes a temporary view: that has an
        // address of its own, so writers whose bytes really do overlap go
        // through two addresses and never through two slices at one.
        if !slice.section.hold(from, len) {
            return None;
        }
        if !slice.section.ensure_committed(base, slice.index, n) {
            slice.section.let_go(from, len);
            slice.section.note_fallback();
            return None;
        }
        // SAFETY: `ensure_committed` proved `[slice.index, slice.index + n)`
        // lies inside the section, so this is inside the section's view.
        let ptr = unsafe { base.add(from) };
        // Said once: whether the switch is really in effect is otherwise
        // invisible in a run, and the switch becoming the default rests on a
        // run that showed it working.
        static IN_USE: OnceLock<()> = OnceLock::new();
        if IN_USE.set(()).is_ok() {
            tracing::info!(
                "physical pages are written through one persistent view per section (VICTORY_XMEM_PERSISTENT_VIEW=1)"
            );
        }
        Some(View::Persistent {
            section: slice.section.clone(),
            ptr,
            from,
            len,
        })
    }

    /// Bytes `[from, from + len)` of this view, and nothing besides.
    ///
    /// The slice covers exactly what the caller writes. That is what lets two
    /// writers through the section's one persistent mapping coexist: their
    /// byte ranges do not overlap, so their slices are disjoint even when
    /// they lie in the same page, which a slice over the whole run would not
    /// give. Writes that really did overlap would still need an exclusion
    /// rule of their own; this only stops ranges that never touch from
    /// being made to share a borrow.
    fn bytes_at(&mut self, from: usize, len: usize) -> &mut [u8] {
        let (ptr, whole) = match self {
            View::Temporary { at, len } => (at.Value.cast::<u8>(), *len),
            View::Persistent { ptr, len, .. } => (*ptr, *len),
        };
        assert!(
            from.checked_add(len).is_some_and(|end| end <= whole),
            "view bytes {from}+{len} past a view of {whole}"
        );
        // SAFETY: the view holds `whole` committed read-write bytes, and
        // `[from, from + len)` lies inside them (`whole` is `n * PAGE`, a
        // multiple of the page, so the pointer stays in the view). For a
        // temporary view those bytes have an address of their own, so no
        // other borrow of them exists. For the persistent variant the
        // section's one mapping is shared, and two live `&mut` at one base
        // are kept apart twice over: each covers only its own writer's
        // bytes, and `Section::hold` refused the view outright if those
        // bytes touched another live one.
        unsafe { std::slice::from_raw_parts_mut(ptr.add(from), len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        match self {
            // Only a temporary view is ours to take away.
            View::Temporary { at, .. } => {
                use windows_sys::Win32::System::Memory::UnmapViewOfFile;
                let started = std::time::Instant::now();
                // SAFETY: our view.
                unsafe { UnmapViewOfFile(*at) };
                victory_gpu::framelog::add(victory_gpu::framelog::T::DiskViewUs, started.elapsed().as_micros() as u64);
            }
            // The section's mapping stays; the bytes this view held do not.
            View::Persistent { section, from, len, .. } => section.let_go(*from, *len),
        }
    }
}

/// The pages behind bytes `[offset, offset + len)` of `pages` (64 KiB
/// each), as runs of consecutive slices of one section: each run's first
/// slice and its length in pages. One look at the page table, which the
/// title's own mapping calls share with every DirectStorage worker. `None`
/// if a page is unknown or `pages` ends early.
fn runs(pages: &[usize], offset: usize, len: usize) -> Option<Vec<(Slice, usize)>> {
    let end = offset.checked_add(len)?;
    let pages = pages.get(offset / PAGE..end.div_ceil(PAGE))?;
    let physical = physical();
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for page in pages {
        let slice = physical.pages.get(page)?;
        match runs.last_mut() {
            Some((first, n))
                if Arc::ptr_eq(&first.section, &slice.section)
                    && slice.index == first.index + *n =>
            {
                *n += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    Some(runs)
}

/// Whether every page under `[offset, offset + len)` is known (a write that
/// fails with them known is a host failure).
pub(crate) fn pages_known(pages: &[usize], offset: usize, len: usize) -> bool {
    runs(pages, offset, len).is_some()
}

/// One number standing for the write generation of bytes `[offset, offset +
/// len)` of `pages`, or `None` when a page is unknown or `pages` ends early:
/// equal digests mean no byte of that range was written since the last look,
/// which is what lets the GPU side skip hashing texture memory it already
/// knows (`VICTORY_TEXTURE_DIRTY_GEN`, phase F). The release the writer did
/// before the bump and the acquire here are what make an unchanged digest a
/// promise about the bytes, not only about the counter.
// Phase F (the GPU side's texture-hash skip) is the caller; until it
// lands, the only caller is the digest test below, so the library build
// would otherwise call these unused.
#[allow(dead_code)]
pub(crate) fn physical_generation_digest(
    pages: &[usize],
    offset: usize,
    len: usize,
) -> Option<u64> {
    let runs = runs(pages, offset, len)?;
    let mut within = offset % PAGE;
    let mut left = len;
    let mut digest = 0x517c_c1b7_2722_0a95u64;
    for (slice, n) in &runs {
        let bytes = (n * PAGE - within).min(left);
        if bytes == 0 {
            break;
        }
        let folded = slice
            .section
            .generation_of(slice.index + within / PAGE, (within + bytes).div_ceil(PAGE))?;
        digest = mix64(digest ^ folded);
        left -= bytes;
        within = 0;
    }
    (left == 0).then_some(digest)
}

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page; with
/// `VICTORY_XMEM_PERSISTENT_VIEW` a run costs no view at all, and without
/// it each view is a map, a commit and an unmap (which flushes every core's
/// TLB). False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    write_physical_with(Mapping::Default, pages, offset, data)
}

/// [`write_physical`] through the views `mapping` names, so tests reach
/// either path without the environment.
fn write_physical_with(mapping: Mapping, pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = mapping.view(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes_at(within, k).copy_from_slice(&data[done..done + k]);
        // After the bytes, never before: the bump's release ordering is what
        // lets a consumer that sees the new generation trust the bytes it
        // reads ([`physical_generation_digest`]).
        slice
            .section
            .bump_range(slice.index + within / PAGE, (within + k).div_ceil(PAGE));
        (within, done) = (0, done + k);
    }
    true
}

/// Bytes `[offset, offset + len)` of physical pages, writable in place
/// until dropped: DirectStorage decodes into it instead of a buffer it
/// would then copy.
pub(crate) struct PageWindow {
    view: View,
    within: usize,
    len: usize,
    /// The section the window's bytes live in, and its first page, so the
    /// window can bump their write generation when it goes: the title writes
    /// in place, and the drop is the last moment this layer knows of.
    section: Arc<Section>,
    first_page: usize,
}

impl Drop for PageWindow {
    fn drop(&mut self) {
        self.section
            .bump_range(self.first_page, (self.within + self.len).div_ceil(PAGE));
    }
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        self.view.bytes_at(within, len)
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    page_window_with(Mapping::Default, pages, offset, len)
}

/// [`page_window`] through the views `mapping` names, so tests reach either
/// path without the environment.
fn page_window_with(
    mapping: Mapping,
    pages: &[usize],
    offset: usize,
    len: usize,
) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: mapping.view(slice, *n)?,
        within: offset % PAGE,
        len,
        section: slice.section.clone(),
        first_page: slice.index + (offset % PAGE) / PAGE,
    })
}

/// The title addresses that show bytes `[offset, offset + len)` of `pages`
/// now, byte for byte, as runs (address, bytes) in address order: a write
/// to the pages through a view of our own ([`page_window`],
/// [`write_physical`]) lands there too, and nothing that watches those
/// addresses sees it (the GPU side's guards fault only on a write at the
/// guarded address itself). Empty if no view shows them. Byte for byte,
/// not by whole pages: the title packs several small textures into one
/// page and fills each with a request of its own.
pub(crate) fn shown_at(pages: &[usize], offset: usize, len: usize) -> Vec<(usize, usize)> {
    let Some(end) = offset.checked_add(len) else {
        return Vec::new();
    };
    let first = offset / PAGE;
    let Some(touched) = pages.get(first..end.div_ceil(PAGE)) else {
        return Vec::new();
    };
    let mut pieces: Vec<(usize, usize)> = Vec::new();
    {
        let physical = physical();
        for (i, page) in touched.iter().enumerate() {
            let Some(addresses) = physical.pages.get(page).and_then(|slice| physical.shown.get(&slice.key(0))) else {
                continue;
            };
            // The bytes of this page that the write covers.
            let lo = if i == 0 { offset % PAGE } else { 0 };
            let hi = (end - (first + i) * PAGE).min(PAGE);
            pieces.extend(addresses.iter().map(|&at| (at + lo, hi - lo)));
        }
    }
    pieces.sort_unstable();
    let mut runs: Vec<(usize, usize)> = Vec::new();
    for (at, bytes) in pieces {
        match runs.last_mut() {
            Some((start, n)) if *start + *n == at => *n += bytes,
            _ => runs.push((at, bytes)),
        }
    }
    runs
}

/// Bytes `[offset, offset + len)` of `pages` have been written through a
/// view of our own: the GPU side is told at every title address that
/// shows them (`victory_gpu::title_memory_written`; what it does with it
/// is `VICTORY_PAGE_WRITES`' to say, and it counts either way). Call when
/// the bytes are there, and without the page table's lock ([`shown_at`]
/// takes and leaves it).
pub(crate) fn pages_written(pages: &[usize], offset: usize, len: usize) {
    for (at, bytes) in shown_at(pages, offset, len) {
        victory_gpu::title_memory_written(at, bytes, true);
    }
}

/// Maps `n` pages at `address`, in place of what is mapped there; null
/// `pages` unmaps. Returns `address`, or null if a page or the host fails.
pub extern "system" fn XMemMapPhysicalPages(
    address: *mut c_void,
    n: usize,
    pages: *const usize,
) -> *mut c_void {
    use victory_gpu::framelog::{T, add};
    let entered = std::time::Instant::now();
    let mapped = map_pages(address, n, pages);
    add(T::RemapDirectCalls, 1);
    add(T::RemapDirectPages, n as u64);
    add(T::RemapDirectUs, entered.elapsed().as_micros() as u64);
    mapped
}

/// [`XMemMapPhysicalPages`] itself; the caller counts it for the frame log.
fn map_pages(address: *mut c_void, n: usize, pages: *const usize) -> *mut c_void {
    let base = address as usize;
    let mut physical = physical();
    // Until the pages are mapped again: `unmap` takes whole views away and
    // maps what stays of them again, and a mapped view is a reservation
    // until it is committed. A read of title memory in place must not fall
    // into that gap (`victory_gpu::title_memory_remapping`).
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, base, n * PAGE);
    if pages.is_null() {
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return std::ptr::null_mut();
        };
        // One view per run of consecutive slices of one section.
        let run = 1 + pages[i + 1..]
            .iter()
            .zip(1..)
            .take_while(|&(page, k)| {
                physical.pages.get(page).is_some_and(|s| {
                    Arc::ptr_eq(&s.section, &first.section) && s.index == first.index + k
                })
            })
            .count();
        if !map(&mut physical, base + i * PAGE, run * PAGE, first) {
            return std::ptr::null_mut();
        }
        i += run;
    }
    address
}

/// One page at each address; null `pages`, or a page number 0, unmaps
/// (the title passes 0 for addresses to clear).
pub extern "system" fn XMemMapPhysicalPagesScatter(
    addresses: *const *mut c_void,
    n: usize,
    pages: *const usize,
) -> BOOL {
    if addresses.is_null() {
        return 0;
    }
    for i in 0..n {
        // SAFETY: the title passes `n` addresses and, unless null, `n` pages.
        let (address, page) = unsafe {
            (
                *addresses.add(i),
                if pages.is_null() || *pages.add(i) == 0 {
                    std::ptr::null()
                } else {
                    pages.add(i)
                },
            )
        };
        if XMemMapPhysicalPages(address, 1, page).is_null() {
            return 0;
        }
    }
    1
}

/// Maps the physical pages shown at `[from, from + len)` at `to` as well,
/// in place of what `to` shows; `from` 0 unmaps. The GPU's page-mapping
/// copies (`victory_gpu`, queue `CopyPageMappingsBatchX`):
/// the title maps pages into a pool view, then has the GPU map them into its tiled
/// heap. Page by page; false if a source page is not mapped physical memory.
pub fn map_alias(to: u64, from: u64, len: u64) -> bool {
    use victory_gpu::framelog::{T, add};
    // From entry to return, the wait for the page table's lock included.
    let entered = std::time::Instant::now();
    let mapped = alias(to as usize, from as usize, len as usize);
    add(T::RemapAliasCalls, 1);
    add(T::RemapAliasUs, entered.elapsed().as_micros() as u64);
    mapped
}

thread_local! {
    /// The page this thread mapped last through [`map_alias`]: the address
    /// after it, its section and its index there. For the frame log's
    /// count of pages that one view could have shown together.
    static ALIASED: std::cell::Cell<(usize, usize, usize)> = const { std::cell::Cell::new((0, 0, 0)) };
}

/// [`map_alias`] itself; the caller counts it for the frame log.
fn alias(to: usize, from: usize, len: usize) -> bool {
    use victory_gpu::framelog::{T, add};
    let mut physical = physical();
    // As in `XMemMapPhysicalPages`: no read in place meets the gap.
    let _changing = victory_gpu::title_memory_remapping();
    unmap(&mut physical, to, len);
    for k in (0..len).step_by(PAGE) {
        if from == 0 {
            break;
        }
        let at = from + k;
        let Some(slice) = physical
            .views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
        else {
            return false;
        };
        // Counted only: whether this page follows the one mapped just
        // before, in address and in its section (one view could show
        // both), or is that very page again (no view can).
        let (section, index) = (Arc::as_ptr(&slice.section) as usize, slice.index);
        let (after, last_section, last_index) = ALIASED.get();
        add(T::RemapAliasPages, 1);
        if after == to + k && last_section == section {
            if index == last_index + 1 {
                add(T::RemapAliasRunPages, 1);
            } else if index == last_index {
                add(T::RemapAliasSamePages, 1);
            }
        }
        ALIASED.set((to + k + PAGE, section, index));
        if !map(&mut physical, to + k, PAGE, slice) {
            return false;
        }
    }
    true
}

/// Maps `slice` and the slices after it over `[base, base + len)`, which
/// must lie in one placeholder.
fn map(physical: &mut Physical, base: usize, len: usize, slice: Slice) -> bool {
    use victory_gpu::framelog::{T, add};
    let started = std::time::Instant::now();
    // Splits the placeholder around the range (failing harmlessly when the
    // range is the whole placeholder).
    let split = || {
        let began = std::time::Instant::now();
        // SAFETY: acts only on a placeholder.
        let done = unsafe {
            VirtualFree(
                base as *mut c_void,
                len,
                MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
            )
        } != 0;
        add(T::RemapSplits, 1);
        add(T::RemapSplitsDone, u64::from(done));
        add(T::RemapSplitUs, began.elapsed().as_micros() as u64);
    };
    // Maps the section over the range, which must be a whole placeholder.
    let map_view = || {
        let meter = Meter::begin();
        // SAFETY: acts only on a placeholder.
        let view = unsafe {
            MapViewOfFile3(
                slice.section.handle,
                GetCurrentProcess(),
                base as *const c_void,
                (slice.index * PAGE) as u64,
                len,
                MEM_REPLACE_PLACEHOLDER,
                PAGE_READWRITE,
                std::ptr::null_mut(),
                0,
            )
        };
        meter.end(true);
        view
    };
    let view = if split_late() {
        // The range is most often a whole placeholder already (a page
        // whose own view was just unmapped): mapped at once; split and
        // mapped again only if that failed ([`split_late`]).
        let first = map_view();
        if first.Value as usize == base {
            first
        } else {
            add(T::RemapMapRetries, 1);
            split();
            map_view()
        }
    } else {
        split();
        map_view()
    };
    if view.Value as usize != base {
        tracing::warn!(
            address = format_args!("{base:#x}"),
            len,
            error = %std::io::Error::last_os_error(),
            "mapping physical pages failed (not a placeholder?)"
        );
        return false;
    }
    let mapped = std::time::Instant::now();
    add(T::RemapMapUs, (mapped - started).as_micros() as u64);
    if !commit(base, len) {
        // SAFETY: the view just mapped, back to a placeholder.
        unsafe { UnmapViewOfFile2(GetCurrentProcess(), view, MEM_PRESERVE_PLACEHOLDER) };
        return false;
    }
    let committed = std::time::Instant::now();
    add(T::RemapCommitUs, (committed - mapped).as_micros() as u64);
    // New contents, read-write: the GPU side's copies no longer guard it.
    victory_gpu::title_memory_remapped(base, len);
    // What the range shows now, for the GPU side's in-place census.
    victory_gpu::title_memory_view(base, len, slice.section.handle as usize, (slice.index * PAGE) as u64);
    add(T::RemapNotesUs, committed.elapsed().as_micros() as u64);
    for k in 0..len.div_ceil(PAGE) {
        physical.shown.entry(slice.key(k)).or_default().push(base + k * PAGE);
    }
    physical.views.insert(base, (len, slice));
    true
}

/// Commits the section pages a view shows (once per page, whichever view).
fn commit(base: usize, len: usize) -> bool {
    if on_wine() {
        return true; // sections are committed whole there
    }
    // SAFETY: `[base, base + len)` is a view of a reserved section.
    if !unsafe { VirtualAlloc(base as *const c_void, len, MEM_COMMIT, PAGE_READWRITE) }.is_null() {
        return true;
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(ERROR_COMMITMENT_LIMIT as i32) {
        let mut info = PERFORMANCE_INFORMATION {
            cb: size_of::<PERFORMANCE_INFORMATION>() as u32,
            ..Default::default()
        };
        // SAFETY: the buffer and size describe a writable PERFORMANCE_INFORMATION.
        let queried = unsafe { K32GetPerformanceInfo(&mut info, info.cb) } != 0;
        victory_loader::log_crash(&commit_limit_message(len, &error, queried.then_some(&info)));
    } else {
        tracing::warn!(len, %error, "committing physical pages failed");
    }
    false
}

fn commit_limit_message(
    len: usize,
    error: &std::io::Error,
    info: Option<&PERFORMANCE_INFORMATION>,
) -> String {
    let mut message = format!(
        "VIctory: Windows commit limit reached while committing {len} bytes of physical pages: {error}.\n"
    );
    if let Some(info) = info {
        use std::fmt::Write;

        let committed = info.CommitTotal * info.PageSize / MIB;
        let limit = info.CommitLimit * info.PageSize / MIB;
        let _ = writeln!(
            message,
            "System commit: {committed} MiB of {limit} MiB in use."
        );
    }
    message.push_str(
        "Close other applications or increase the Windows paging file size, then retry.\n",
    );
    message
}

/// Returns `[base, base + len)` to placeholders; views reaching past it
/// are mapped again outside it.
fn unmap(physical: &mut Physical, base: usize, len: usize) {
    // No bytes: no view to return. The search below would still find the
    // view holding `base` and cut it in two there, and its range of views
    // starting inside would run backwards (which `BTreeMap` panics on).
    if len == 0 {
        return;
    }
    let end = base + len;
    // Views do not overlap: the one holding `base`, then those starting
    // inside (the GPU's page mappings make one view per page).
    let first = physical
        .views
        .range(..=base)
        .next_back()
        .filter(|(at, (size, _))| *at + size > base)
        .map(|(at, _)| *at);
    let hit: Vec<usize> = first
        .into_iter()
        .chain(physical.views.range(base + 1..end).map(|(at, _)| *at))
        .collect();
    for at in hit {
        let Some((size, slice)) = physical.views.remove(&at) else {
            continue;
        };
        for k in 0..size.div_ceil(PAGE) {
            let key = slice.key(k);
            if let Some(addresses) = physical.shown.get_mut(&key) {
                addresses.retain(|&a| a != at + k * PAGE);
                if addresses.is_empty() {
                    physical.shown.remove(&key);
                }
            }
        }
        use victory_gpu::framelog::{T, add};
        let started = std::time::Instant::now();
        victory_gpu::title_memory_remapped(at, size);
        let noted = std::time::Instant::now();
        add(T::RemapNotesUs, (noted - started).as_micros() as u64);
        let meter = Meter::begin();
        // SAFETY: a view we mapped; it becomes a placeholder again.
        unsafe {
            UnmapViewOfFile2(
                GetCurrentProcess(),
                MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: at as *mut c_void,
                },
                MEM_PRESERVE_PLACEHOLDER,
            )
        };
        meter.end(false);
        add(T::RemapUnmapUs, noted.elapsed().as_micros() as u64);
        // What of the view lies outside the range asked for is mapped
        // again below: the whole view went for it.
        let outside = base.saturating_sub(at) + (at + size).saturating_sub(end);
        if outside != 0 {
            add(T::RemapCuts, 1);
            add(T::RemapCutBytes, outside as u64);
        }
        if at < base {
            map(physical, at, base - at, slice.clone());
        }
        if at + size > end {
            let index = slice.index + (end - at) / PAGE;
            map(physical, end, at + size - end, Slice { index, ..slice });
        }
    }
}

/// The oldest layout we know; newer titles pass other flag values (Minecraft
/// 0x20000001) and read the same leading fields.
#[repr(C)]
#[derive(Debug, Default)]
pub struct WorkingSet {
    pub title_limit: usize,
    pub title_used: usize,
    pub tools_limit: usize,
    pub tools_used: usize,
    pub stack_used: u32,
    pub physical_pages: u32,
    pub gpu_optimal_limit: usize,
    pub gpu_optimal_used: usize,
    pub tools_gpu_optimal_limit: usize,
    pub tools_gpu_optimal_used: usize,
    pub kernel_pool_used: u32,
    pub kernel_pool_limit: u32,
}

/// The title's budget and our process's commit as what it uses. No tools
/// memory: a retail console.
// shortcut: all fields are filled whatever the flags ask for, and use is
// not split by bandwidth class (GPU-optimal use reads 0); track
// allocations per class if a title budgets from that field.
pub extern "system" fn XMemGetWorkingSetStatistics(_which: u32, out: *mut WorkingSet) -> BOOL {
    if out.is_null() {
        return 0;
    }
    let mut counters = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut counters, counters.cb) };
    let (gpu_optimal, standard) = victory_gpu::console::current().title_memory();
    let stats = WorkingSet {
        title_limit: gpu_optimal + standard - TRANSFERRED.load(Ordering::Relaxed),
        title_used: counters.PagefileUsage,
        gpu_optimal_limit: gpu_optimal,
        ..Default::default()
    };
    // What the title is told, logged at the first asking and at every
    // 1,024th: "used" is all the process has committed, the layer's own
    // memory and the card's included (27 GB in run 60's gameplay against a
    // limit of 13.5 GB). A title that budgets its streaming from these two
    // figures believes it is far over its limit.
    // shortcut: the figure is not yet the title's own memory alone (its
    // heaps and the pages it mapped); count those if this line shows the
    // title asking.
    static ASKED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let asked = ASKED.fetch_add(1, Ordering::Relaxed);
    if asked.is_multiple_of(1024) {
        tracing::info!(
            asked = asked + 1,
            which = _which,
            limit_mb = stats.title_limit >> 20,
            used_mb = stats.title_used >> 20,
            "the title asks how much memory it uses"
        );
    }
    // SAFETY: checked non-null; the title passes room for the structure.
    unsafe { out.write(stats) };
    1
}

/// Hands title memory (standard, in 2 MiB pages) to the system. The other
/// transfers involve tools memory, which a retail console has none of.
pub extern "system" fn XMemTransferMemory(kind: u32, size: usize) -> BOOL {
    const TITLE_TO_SYSTEM: u32 = 0;
    tracing::info!(kind, size, "XMemTransferMemory");
    if size == 0 {
        return 1;
    }
    let Some(size) = size.checked_next_multiple_of(2 * MIB) else {
        return 0;
    };
    let standard = victory_gpu::console::current().title_memory().1;
    let moved = kind == TITLE_TO_SYSTEM
        && TRANSFERRED
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |t| {
                t.checked_add(size).filter(|&t| t <= standard)
            })
            .is_ok();
    moved.into()
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "xmem",
        dlls: &["xmem.dll"],
        exports: vec![
            hle_export!(XMemAlloc),
            hle_export!(XMemFree),
            hle_export!(XMemAllocDefault),
            hle_export!(XMemFreeDefault),
            hle_export!(XMemIsToolsMemAvailable),
            hle_export!(XMemAllocatePhysicalPages),
            hle_export!(XMemFreePhysicalPages),
            hle_export!(XMemMapPhysicalPages),
            hle_export!(XMemMapPhysicalPagesScatter),
            hle_export!(XMemVirtualAlloc),
            hle_export!(XMemVirtualQuery),
            hle_export!(XMemGetWorkingSetStatistics),
            hle_export!(XMemTransferMemory),
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn commit_notifications_cover_both_blocks_of_an_unaligned_request() {
        let reservation = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE,
            0,
            PAGE_NOACCESS,
        );
        assert!(!reservation.is_null());
        let first = reservation as usize;
        let boundary = first + PAGE;
        let mut notified = None;
        let at = virtual_alloc(
            (boundary - 1) as _,
            2,
            MEM_COMMIT,
            0,
            PAGE_READWRITE,
            |start, size| {
                notified = Some(start..start + size);
            },
        );
        assert!(!at.is_null());
        assert!((at as usize) < boundary - 1);
        assert_eq!(query((boundary - 1) as _).host.State, MEM_COMMIT);
        assert_eq!(query(boundary as _).host.State, MEM_COMMIT);
        // SAFETY: this test owns the entire reservation.
        assert_ne!(unsafe { VirtualFree(reservation, 0, MEM_RELEASE) }, 0);
        let range = notified.expect("a successful commit notifies the GPU watch");
        assert_eq!(
            range.start / PAGE..range.end.div_ceil(PAGE),
            first / PAGE..first / PAGE + 2
        );
    }

    #[test]
    fn commit_notification_uses_the_chosen_address_for_a_null_request() {
        let mut notified = None;
        let at = virtual_alloc(
            std::ptr::null_mut(),
            PAGE,
            MEM_COMMIT,
            0,
            PAGE_READWRITE,
            |start, size| {
                notified = Some((start, size));
            },
        );
        assert!(!at.is_null());
        // SAFETY: this test owns the entire reservation.
        assert_ne!(unsafe { VirtualFree(at, 0, MEM_RELEASE) }, 0);
        assert_eq!(notified, Some((at as usize, PAGE)));
    }

    fn query(address: *const c_void) -> BasicInformation {
        // SAFETY: plain data; zero is a valid value.
        let mut info: BasicInformation = unsafe { std::mem::zeroed() };
        // SAFETY: a pseudo handle needs no closing.
        let process = unsafe { GetCurrentProcess() };
        assert_eq!(XMemVirtualQuery(process, address, &mut info), 1);
        info
    }

    #[test]
    fn alloc_honours_alignment_and_frees() {
        for (field, align) in [(0, 16), (4, 16), (0xc, 4096), (0x10, 64 * 1024)] {
            let attributes = field << ALIGNMENT_SHIFT | 1 << 29; // a graphics type
            let p = XMemAlloc(100, attributes);
            assert!(!p.is_null());
            assert_eq!(p as usize % align, 0, "field {field:#x}");
            // SAFETY: 100 bytes were allocated.
            unsafe { p.cast::<u8>().write_bytes(0xab, 100) };
            XMemFree(p, attributes);
        }
        let p = XMemAlloc(4096, ALLOC_32BIT);
        assert!(!p.is_null() && (p as usize) < 1 << 32);
        XMemFree(p, ALLOC_32BIT);
    }

    #[test]
    fn virtual_alloc_meets_console_constraints() {
        const GPU_READ_WRITE: u32 = 0x2000;
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            4 * MIB,
            PAGES_2M | MEM_RESERVE | MEM_COMMIT,
            GRAPHICS | 7 << 12, // 40-bit addresses
            PAGE_READWRITE | GPU_READ_WRITE,
        );
        assert!(!p.is_null());
        assert_eq!(p as usize % (2 * MIB), 0);
        assert!((p as usize) < 1 << 40);
        let info = query(p);
        assert_eq!(info.host.State, MEM_COMMIT);
        assert_eq!(info.host.Protect, PAGE_READWRITE);
        assert_eq!(info.flags & GRAPHICS, GRAPHICS);
        // Write-watched, for the GPU side's copies.
        let (mut found, mut count, mut granularity) = ([0usize; 4], 4, 0);
        // SAFETY: room for `count` addresses.
        let watched = unsafe {
            windows_sys::Win32::System::Memory::GetWriteWatch(
                0,
                p,
                4 * MIB,
                found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        };
        assert_eq!(watched, 0);
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
        assert_eq!(query(p).flags, 0);
        // No address field: above the 32-bit range and below 2^40, where
        // the console keeps it.
        let p = XMemVirtualAlloc(
            std::ptr::null_mut(),
            16 << 30,
            MEM_RESERVE,
            GRAPHICS,
            PAGE_NOACCESS,
        );
        assert!(!p.is_null() && (p as usize) >= 1 << 32, "{p:p}");
        assert!((p as usize) + (16 << 30) <= 1 << 40, "{p:p}");
        // SAFETY: our own reservation.
        assert_ne!(unsafe { VirtualFree(p, 0, MEM_RELEASE) }, 0);
    }

    #[test]
    fn transfers_shrink_the_title_budget() {
        let mut before = WorkingSet::default();
        assert_eq!(XMemGetWorkingSetStatistics(0x2000_0001, &mut before), 1);
        assert!(before.title_used > 0);
        assert_eq!(XMemTransferMemory(0, 128 * MIB - 1), 1);
        assert_eq!(XMemTransferMemory(2, MIB), 0, "no tools memory");
        let standard = victory_gpu::console::current().title_memory().1;
        assert_eq!(XMemTransferMemory(0, standard), 0, "over budget");
        let mut after = WorkingSet::default();
        XMemGetWorkingSetStatistics(0x2000_0001, &mut after);
        assert_eq!(before.title_limit - after.title_limit, 128 * MIB);
    }

    #[test]
    fn statistics_have_the_console_layout() {
        assert_eq!(size_of::<WorkingSet>(), 0x50);
        assert_eq!(size_of::<BasicInformation>(), 0x40);
        assert_eq!(std::mem::offset_of!(WorkingSet, gpu_optimal_limit), 0x28);
    }

    #[test]
    fn commit_limit_report_includes_request_and_system_commit() {
        let error = std::io::Error::from_raw_os_error(ERROR_COMMITMENT_LIMIT as i32);
        let info = PERFORMANCE_INFORMATION {
            CommitTotal: 512,
            CommitLimit: 768,
            PageSize: 4096,
            ..Default::default()
        };
        let report = commit_limit_message(4 * MIB, &error, Some(&info));
        assert!(report.contains("4194304 bytes"));
        assert!(report.contains("2 MiB of 3 MiB in use"));
        let fallback = commit_limit_message(4 * MIB, &error, None);
        for report in [&report, &fallback] {
            assert!(report.contains("os error 1455"));
            assert!(report.contains("Windows paging file size"));
        }
        assert!(!fallback.contains("System commit:"));
    }

    #[test]
    fn physical_pages_map_alias_and_unmap() {
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        assert_eq!(n, 3);
        // Packed into 21 bits as `number >> 4`, as the title keeps them.
        for p in pages {
            assert_eq!((p >> 4 & 0x1f_ffff) << 4, p);
        }
        // Reservations of 64 KiB pages, as titles make them for mapping.
        let reserve = || {
            XMemVirtualAlloc(
                std::ptr::null_mut(),
                3 * PAGE,
                MEM_RESERVE | PAGES_64K,
                0,
                PAGE_READWRITE,
            )
        };
        let (a, b) = (reserve(), reserve());
        assert!(!a.is_null() && !b.is_null());
        let at = |p: *mut c_void, page: usize| p.wrapping_byte_add(page * PAGE);
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);

        // The same page at a second address shows the same bytes.
        assert_eq!(
            XMemMapPhysicalPages(at(b, 1), 1, pages[2..].as_ptr()),
            at(b, 1)
        );
        // SAFETY: both pages are mapped read-write.
        unsafe {
            at(a, 2).cast::<u8>().write(7);
            assert_eq!(at(b, 1).cast::<u8>().read(), 7);
            a.cast::<u8>().write(1);
        }
        // Writes into pages (DirectStorage's) show through their views.
        assert!(write_physical(&pages[2..], 1, &[9]));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { at(b, 1).cast::<u8>().add(1).read() }, 9);

        // Unmapping the middle page keeps its neighbours.
        assert_eq!(
            XMemMapPhysicalPages(at(a, 1), 1, std::ptr::null()),
            at(a, 1)
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(at(a, 2)).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { at(a, 2).cast::<u8>().read() }, 7);

        // A scattered page lands in the hole.
        let addresses = [at(a, 1)];
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, pages.as_ptr()),
            1
        );
        // SAFETY: page 0 is now mapped at both `a` and `a + PAGE`.
        assert_eq!(unsafe { at(a, 1).cast::<u8>().read() }, 1);
        // Page number 0 unmaps.
        assert_eq!(
            XMemMapPhysicalPagesScatter(addresses.as_ptr(), 1, [0usize].as_ptr()),
            1
        );
        assert_eq!(query(at(a, 1)).host.State, MEM_RESERVE);
        // The GPU's page-mapping copy: what `a + 2 pages` shows, at `b`.
        assert!(map_alias(b as u64, at(a, 2) as u64, PAGE as u64));
        // SAFETY: mapped read-write.
        assert_eq!(unsafe { b.cast::<u8>().read() }, 7);
        assert!(map_alias(b as u64, 0, PAGE as u64));
        assert_eq!(query(b).host.State, MEM_RESERVE);
        assert!(!map_alias(b as u64, at(a, 1) as u64, PAGE as u64));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
        assert_eq!(XMemIsToolsMemAvailable(), 0);
    }

    /// A thread that reads a mapped page in place, as the GPU side reads a
    /// texture, never meets the moment in which the runtime has taken the
    /// page's view away to map it again: mapping the same pages at the same
    /// address unmaps the view and maps a new one, and without the lock the
    /// reader faults on a page "reserved, never committed" (the crashes of
    /// runs 71 and 74).
    #[test]
    fn a_read_in_place_never_meets_the_gap_of_a_remapping() {
        use std::sync::atomic::{AtomicBool, Ordering};
        let mut pages = [0usize; 3];
        let mut n = 3;
        assert_eq!(XMemAllocatePhysicalPages(0, &mut n, pages.as_mut_ptr()), 1);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            3 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        // SAFETY: mapped read-write.
        unsafe { a.cast::<u8>().write(1) };
        let first = a as usize;
        let stop = std::sync::Arc::new(AtomicBool::new(false));
        let reader = {
            let stop = stop.clone();
            std::thread::spawn(move || {
                let mut reads = 0u64;
                while !stop.load(Ordering::Relaxed) {
                    let _reading = victory_gpu::title_memory_reading();
                    // SAFETY: the page is mapped whenever no change of
                    // mapping is under way, and none is while `_reading`
                    // is held.
                    assert_eq!(unsafe { (first as *const u8).read_volatile() }, 1);
                    reads += 1;
                }
                reads
            })
        };
        for _ in 0..2000 {
            // The same pages at the same address: the view goes and comes
            // back, with the same bytes.
            assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        }
        stop.store(true, Ordering::Relaxed);
        assert!(reader.join().expect("the reader never faulted") > 0);
        assert_eq!(XMemMapPhysicalPages(a, 3, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn mapping_no_pages_changes_nothing() {
        let pages = allocate(2);
        let a = XMemVirtualAlloc(
            std::ptr::null_mut(),
            2 * PAGE,
            MEM_RESERVE | PAGES_64K,
            0,
            PAGE_READWRITE,
        );
        assert!(!a.is_null());
        assert_eq!(XMemMapPhysicalPages(a, 2, pages.as_ptr()), a);
        // SAFETY: both pages are mapped read-write.
        unsafe { a.cast::<u8>().write(5) };
        let second = a.wrapping_byte_add(PAGE);

        // A count of 0 at the start of a view, inside it and past it: with
        // pages, without (an unmap), and as the GPU's page-mapping copy.
        for at in [a, second, a.wrapping_byte_add(2 * PAGE)] {
            assert_eq!(XMemMapPhysicalPages(at, 0, pages.as_ptr()), at);
            assert_eq!(XMemMapPhysicalPages(at, 0, std::ptr::null()), at);
            assert!(map_alias(at as u64, a as u64, 0));
            assert!(map_alias(at as u64, 0, 0));
        }

        // The view is still the one view it was.
        let view = physical().views.get(&(a as usize)).map(|v| v.0);
        assert_eq!(view, Some(2 * PAGE));
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(query(second).host.State, MEM_COMMIT);
        // SAFETY: still mapped.
        assert_eq!(unsafe { a.cast::<u8>().read() }, 5);

        // A count of 1 still unmaps that page alone.
        assert_eq!(XMemMapPhysicalPages(second, 1, std::ptr::null()), second);
        assert_eq!(query(second).host.State, MEM_RESERVE);
        assert_eq!(query(a).host.State, MEM_COMMIT);
        assert_eq!(XMemMapPhysicalPages(a, 1, std::ptr::null()), a);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// `n` pages of one new allocation.
    fn allocate(n: usize) -> Vec<usize> {
        let (mut pages, mut given) = (vec![0; n], n);
        assert_eq!(
            XMemAllocatePhysicalPages(0, &mut given, pages.as_mut_ptr()),
            1
        );
        assert_eq!(given, n);
        pages
    }

    /// What a page number is: its section and index in it.
    fn slice_of(page: usize) -> Slice {
        physical().pages.get(&page).cloned().expect("known page")
    }

    /// The bytes of one page, through a view of our own. The temporary path
    /// explicitly, so the tests that existed before the persistent one are
    /// its negative control whatever the environment says.
    fn page_bytes(page: usize) -> Vec<u8> {
        Mapping::Temporary
            .view(&slice_of(page), 1)
            .expect("mapped")
            .bytes_at(0, PAGE)
            .to_vec()
    }

    #[test]
    fn runs_group_consecutive_pages_of_one_allocation() {
        let (a, b) = (allocate(3), allocate(1));
        let lengths = |pages: &[usize], offset, len| {
            runs(pages, offset, len).map(|r| r.iter().map(|&(_, n)| n).collect::<Vec<_>>())
        };
        assert_eq!(lengths(&a, 0, 3 * PAGE), Some(vec![3]));
        // Only the pages the bytes reach.
        assert_eq!(lengths(&a, PAGE + 5, 10), Some(vec![1]));
        assert_eq!(lengths(&a, PAGE - 1, 2), Some(vec![2]));
        assert_eq!(lengths(&a, 0, 0), Some(vec![]));
        // Out of order, or across allocations: one run each.
        assert_eq!(
            lengths(&[a[0], a[2], a[1]], 0, 3 * PAGE),
            Some(vec![1, 1, 1])
        );
        assert_eq!(lengths(&[a[1], a[2], b[0]], 0, 3 * PAGE), Some(vec![2, 1]));
        // A slice that follows on in another section starts a run.
        assert_eq!(lengths(&[b[0], a[1]], 0, 2 * PAGE), Some(vec![1, 1]));
        // Too few pages for the bytes, or an unknown one.
        assert_eq!(lengths(&a, 0, 3 * PAGE + 1), None);
        assert_eq!(lengths(&[a[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(lengths(&a, usize::MAX, 2), None);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    #[test]
    fn writes_cross_runs_and_allocations() {
        let (a, b) = (allocate(2), allocate(1));
        let pages = [a[0], a[1], b[0]];
        let data: Vec<u8> = (0..2 * PAGE + 64).map(|i| (i * 7 % 251) as u8).collect();
        let offset = PAGE - 32;
        // Past the third page: the second run is too short.
        assert!(!write_physical(&pages, offset, &data));
        assert!(write_physical(&pages, offset, &data[..2 * PAGE]));
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the offset");
        assert_eq!(seen[offset..offset + 2 * PAGE], data[..2 * PAGE]);
        assert!(seen[offset + 2 * PAGE..].iter().all(|&b| b == 0), "after");
        assert!(
            !write_physical(&[a[0], 0x10], 0, &data[..PAGE + 1]),
            "unknown page"
        );
        assert_eq!(XMemFreePhysicalPages(2, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// The section's persistent view is placed above the title's own memory
    /// (phase E). Below that ceiling the host would either hand out an address
    /// the title is about to ask for, or make our view look like a title
    /// address to the title's own 40-bit arithmetic (decision 0001).
    #[test]
    fn persistent_view_maps_above_the_title_memory_ceiling() {
        let pages = allocate(2);
        let section = slice_of(pages[0]).section.clone();
        let base = section
            .persistent_ptr()
            .expect("the host refused a view above 40 bits");
        assert!(
            above_title_memory(base as usize),
            "the persistent view is at {base:p}, inside the title's own addresses"
        );
        // The rule itself, which is the part a host cannot be asked to prove:
        // one page below the ceiling is refused, the ceiling itself is not.
        // (Removing `LowestStartingAddress` from the reservation does NOT fail
        // this test on a host whose allocator is already top-down, which this
        // one is: the requirement is a belt, and this predicate is the pin.)
        assert!(!above_title_memory(TITLE_MEMORY_CEILING - PAGE));
        assert!(above_title_memory(TITLE_MEMORY_CEILING));
        // The view is the section: a write through it lands in the pages a
        // temporary view of ours reads back.
        let data: Vec<u8> = (0..PAGE).map(|i| (i * 13 % 251) as u8).collect();
        assert!(write_physical_with(Mapping::Persistent, &pages[..1], 0, &data));
        assert_eq!(page_bytes(pages[0]), data);
        assert_eq!(section.persistent_ptr(), Some(base), "mapped once");
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// Eight threads commit the same uncommitted pages at once. Every one of
    /// them must succeed, and none may store into a page whose commit is still
    /// in flight: that is the half a "committed" bitmap could not promise,
    /// because a loser would write while `VirtualAlloc` was still working.
    #[test]
    fn concurrent_commit_state_never_exposes_an_uncommitted_page() {
        let pages = allocate(4);
        let section = slice_of(pages[0]).section.clone();
        let base = section
            .persistent_ptr()
            .expect("the host refused a view above 40 bits");
        assert!(!section.committed_all(0, 4), "nothing committed yet");
        let at = base as usize;
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..8)
                .map(|worker| {
                    let section = section.clone();
                    scope.spawn(move || {
                        assert!(
                            section.ensure_committed(at as *mut u8, 0, 4),
                            "a thread lost the commit race and gave up"
                        );
                        // SAFETY: the call above returned true, so the four
                        // pages are committed; each thread writes a byte of
                        // its own, so no two touch the same one.
                        unsafe { (at as *mut u8).add(worker).write_volatile(worker as u8) };
                    })
                })
                .collect();
            for worker in workers {
                worker.join().expect("no thread panicked");
            }
        });
        assert!(section.committed_all(0, 4), "all four pages committed");
        // Every thread's byte is there: none was lost or written before its
        // page was committed.
        for worker in 0..8u8 {
            // SAFETY: as above; the byte was written by a joined thread.
            assert_eq!(unsafe { base.add(usize::from(worker)).read_volatile() }, worker);
        }
        assert_eq!(XMemFreePhysicalPages(4, pages.as_ptr()), 1);
    }

    /// The generation digest moves when bytes are written and stays put when
    /// they are not (phase F's skip rests on the second half). Writes through
    /// a physical write and through a page window both count.
    #[test]
    fn dirty_generation_digest_changes_on_write_and_stays_stable_when_clean() {
        let pages = allocate(2);
        let whole = |pages: &[usize]| physical_generation_digest(pages, 0, 2 * PAGE);
        let clean = whole(&pages).expect("two pages");
        assert_eq!(whole(&pages), Some(clean), "nothing has been written");
        let second = physical_generation_digest(&pages, PAGE, PAGE).expect("page two");

        assert!(write_physical(&pages, 0, &[0xA5; 64]));
        let first_write = whole(&pages).expect("two pages");
        assert_ne!(Some(first_write), Some(clean), "a write must show");
        assert_eq!(
            physical_generation_digest(&pages, PAGE, PAGE),
            Some(second),
            "an untouched page's digest moved"
        );
        // Writes are counted, not compared: the same bytes written again move
        // the digest too, which is what makes the counter cheap.
        assert!(write_physical(&pages, 0, &[0xA5; 64]));
        assert_ne!(whole(&pages), Some(first_write), "a second write counts");

        // Bytes written in place through a window count when the window goes.
        let before_window = whole(&pages).expect("two pages");
        {
            let mut window = page_window(&pages, 0, 2 * PAGE).expect("one run");
            window.bytes()[0] = 1;
        }
        assert_ne!(
            whole(&pages),
            Some(before_window),
            "a page window's bytes must show"
        );

        // Unknown pages and ranges past the end are refused.
        assert_eq!(physical_generation_digest(&[pages[0], 0x10], 0, 2 * PAGE), None);
        assert_eq!(physical_generation_digest(&pages, 0, 2 * PAGE + 1), None);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// Pages say where the title has them mapped, byte for byte (after
    /// fork PR #571's test; its runs are whole pages).
    #[test]
    fn pages_say_where_they_are_mapped() {
        let pages = allocate(3);
        let reserve = |n: usize| {
            let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, 0, PAGE_READWRITE);
            assert!(!at.is_null());
            at
        };
        // `b` keeps a page unmapped at either end, so that no run at `b`
        // ever touches one at `a`, wherever the host puts the two.
        let (a, b) = (reserve(3), reserve(4));
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty(), "mapped nowhere");
        // All three at `a`; the last two at `b`, from its second page on.
        assert_eq!(XMemMapPhysicalPages(a, 3, pages.as_ptr()), a);
        let second = b.wrapping_byte_add(PAGE);
        assert_eq!(XMemMapPhysicalPages(second, 2, pages[1..].as_ptr()), second);
        let (a, second) = (a as usize, second as usize);
        let sorted = |mut runs: Vec<(usize, usize)>| {
            runs.sort_unstable();
            runs
        };
        assert_eq!(shown_at(&pages, 0, PAGE), vec![(a, PAGE)]);
        // A request for part of a page names those bytes alone.
        assert_eq!(shown_at(&pages, 0x4000, 0x3000), vec![(a + 0x4000, 0x3000)]);
        // Across two pages, shown at two addresses each: one run at each.
        assert_eq!(
            shown_at(&pages, PAGE + 1, PAGE),
            sorted(vec![(a + PAGE + 1, PAGE), (second + 1, PAGE)])
        );
        // A page unmapped in the middle of a view: the rest still shows.
        let middle = (a + PAGE) as *mut c_void;
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(
            shown_at(&pages, 0, 3 * PAGE),
            sorted(vec![(a, PAGE), (a + 2 * PAGE, PAGE), (second, 2 * PAGE)])
        );
        // Too few pages for the bytes: nothing (the write fails too).
        assert!(shown_at(&pages, 0, 3 * PAGE + 1).is_empty());
        for (at, n) in [(a, 3), (second, 2)] {
            let at = at as *mut c_void;
            assert_eq!(XMemMapPhysicalPages(at, n, std::ptr::null()), at);
        }
        assert!(shown_at(&pages, 0, 3 * PAGE).is_empty());
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    #[test]
    fn page_windows_cover_one_run_in_place() {
        let (a, b) = (allocate(3), allocate(1));
        let mut window = page_window(&a, PAGE + 3, PAGE).expect("one run");
        assert_eq!(window.bytes().len(), PAGE);
        window.bytes().fill(0x5a);
        drop(window);
        let (second, third) = (page_bytes(a[1]), page_bytes(a[2]));
        assert!(second[..3].iter().all(|&b| b == 0));
        assert!(second[3..].iter().all(|&b| b == 0x5a));
        assert!(third[..3].iter().all(|&b| b == 0x5a));
        assert!(third[3..].iter().all(|&b| b == 0));
        // Several runs, or an unknown page: no window.
        assert!(page_window(&[a[0], b[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[a[1], a[0]], 0, PAGE + 1).is_none());
        assert!(page_window(&[0x10], 0, 1).is_none());
        // One page of several is still one run.
        assert!(page_window(&[a[0], b[0]], 0, PAGE).is_some());
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, b.as_ptr()), 1);
    }

    /// The switch is off unless the environment says yes, so a run that sets
    /// nothing keeps the view per write — which makes every test above the
    /// negative control for the persistent path.
    ///
    /// Mutation run 2026-11-25: returning `true` for `None` or for `"0"`
    /// fails this test and nothing else.
    #[test]
    fn parse_persistent_views_accepts_only_an_explicit_yes() {
        use std::ffi::{OsStr, OsString};
        use std::os::windows::ffi::OsStringExt;
        for yes in ["1", "true", "TRUE", "on", "On", " yes "] {
            assert!(parse_persistent_views(Some(OsStr::new(yes))), "{yes:?}");
        }
        for no in ["", " ", "0", "false", "no", "off", "2", "yes!", "on!"] {
            assert!(!parse_persistent_views(Some(OsStr::new(no))), "{no:?}");
        }
        // Unset is off: the rollout default.
        assert!(!parse_persistent_views(None));
        // Not text at all (a lone surrogate is not UTF-16): off, never a
        // guess, which is the direction that keeps a run on the proven path.
        let lone_surrogate = OsString::from_wide(&[0xd800]);
        assert!(!parse_persistent_views(Some(lone_surrogate.as_os_str())));
    }

    /// A write through the section's one persistent view lands in its pages,
    /// commits each page once, and unmaps nothing when it finishes.
    ///
    /// Mutation runs 2026-11-25, both restored: (a) `mark_committed` made a
    /// no-op fails the `committed_all` assertions below, and nothing else;
    /// (b) the `commit` call removed makes the write fault, which is what
    /// proves the commit is load-bearing rather than decorative.
    #[test]
    fn persistent_views_write_and_commit_once() {
        let pages = allocate(3);
        let section = slice_of(pages[0]).section.clone();
        // Nothing is mapped whole, and no page committed through it, until a
        // write on the persistent path asks.
        assert!(section.persistent.get().is_none());
        assert!(!section.committed_all(0, 3));

        // Two pages from the second half of the first.
        let data: Vec<u8> = (0..PAGE + 8).map(|i| (i * 11 % 251) as u8).collect();
        let offset = PAGE - 8;
        assert!(write_physical_with(Mapping::Persistent, &pages, offset, &data));

        // The persistent path ran, not its fallback: only it maps the
        // section whole and commits through that mapping.
        assert!(
            section.persistent.get().is_some_and(Option::is_some),
            "the section was not mapped whole, so the write fell back to a temporary view"
        );
        let base = section.persistent_ptr().expect("mapped whole");
        assert!(section.committed_all(0, 2), "both written pages committed");
        assert!(!section.committed_all(2, 1), "the unwritten page did not");

        // The bytes are in the section's pages, read through a view of our
        // own: the same bytes a temporary view would have written.
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert_eq!(seen[offset..offset + data.len()], data[..]);
        assert!(seen[..offset].iter().all(|&b| b == 0), "before the write");
        assert!(
            seen[offset + data.len()..].iter().all(|&b| b == 0),
            "after the write"
        );

        // A second write to the same pages commits nothing again and leaves
        // the section's mapping the one it mapped first; a view that
        // unmapped it here would fault on the write above instead.
        assert!(write_physical_with(Mapping::Persistent, &pages, offset, &data));
        assert_eq!(section.persistent_ptr(), Some(base), "the first mapping");
        assert!(section.committed_all(0, 2));

        // Unmapped pages and a bad offset are still refused on this path.
        assert!(!write_physical_with(Mapping::Persistent, &[pages[0], 0x10], 0, &data[..PAGE + 1]));
        assert!(!write_physical_with(Mapping::Persistent, &pages, usize::MAX, &data[..2]));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// A window on the persistent view decodes in place as a temporary one
    /// does, and leaves the section's mapping in place when it drops: the
    /// next window is the same memory, still holding the last one's bytes.
    #[test]
    fn persistent_page_windows_share_one_mapping() {
        let pages = allocate(2);
        let section = slice_of(pages[0]).section.clone();
        {
            // Bytes [3, PAGE + 3): the third of the second page on.
            let mut window =
                page_window_with(Mapping::Persistent, &pages, 3, PAGE).expect("one run");
            assert_eq!(window.bytes().len(), PAGE);
            window.bytes().fill(0xa5);
        }
        let base = section.persistent_ptr().expect("mapped whole");
        // The window really went through the section's mapping: only that
        // path commits pages, and a fallback to a temporary view would leave
        // both bits clear while every byte assertion below still held.
        assert!(
            section.committed_all(0, 2),
            "the window fell back to a temporary view"
        );
        // The window dropped, and nothing unmapped: the next window is the
        // same map, and it shows what the first wrote.
        let mut window = page_window_with(Mapping::Persistent, &pages, 0, 4).expect("one run");
        assert_eq!(&window.bytes()[..4], &[0, 0, 0, 0xa5], "the first window's bytes");
        window.bytes()[0] = 0x11;
        drop(window);
        assert_eq!(section.persistent_ptr(), Some(base), "the first mapping");
        assert_eq!(page_bytes(pages[0])[0], 0x11, "the byte just written");
        let first = page_bytes(pages[0]);
        assert!(first[1..3].iter().all(|&b| b == 0));
        assert!(first[3..].iter().all(|&b| b == 0xa5));
        let second = page_bytes(pages[1]);
        assert_eq!(&second[..3], &[0xa5, 0xa5, 0xa5]);
        assert!(second[3..].iter().all(|&b| b == 0));
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// A view's slice covers the bytes asked for and no more, on either path.
    /// That extent is what lets two writers through the section's one mapping
    /// hold slices at once: their ranges do not overlap, so their slices are
    /// disjoint even inside a single page, where a slice over the whole run
    /// would have made ranges that never touch share one borrow.
    ///
    /// Mutation run 2026-11-25, restored: returning the whole view from
    /// `bytes_at` fails the "exactly" assertions, and removing its end check
    /// makes the past-the-end row read on instead of refusing.
    #[test]
    fn a_view_slice_covers_only_the_bytes_asked_for() {
        let pages = allocate(2);
        let slice = slice_of(pages[0]);
        for (name, mapping) in [
            ("temporary", Mapping::Temporary),
            ("persistent", Mapping::Persistent),
        ] {
            let mut view = mapping.view(&slice, 1).expect("one page mapped");
            // Exactly three bytes, not the rest of the page.
            assert_eq!(view.bytes_at(5, 3).len(), 3, "{name}: not the whole run");
            view.bytes_at(5, 3).copy_from_slice(&[1, 2, 3]);
            let page = page_bytes(pages[0]);
            assert_eq!(&page[5..8], &[1, 2, 3], "{name}");
            assert_eq!((page[4], page[8]), (0, 0), "{name}: only those bytes");
            // Past the end it refuses rather than reading on.
            let past = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                view.bytes_at(PAGE - 1, 2).len()
            }));
            assert!(past.is_err(), "{name}: read past the end of the view");
        }
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// The section's mapping is one address, so two writers may share it only
    /// when their bytes do not touch. A claim that touches a live one is
    /// refused and the caller takes a temporary view, which has an address of
    /// its own -- the exclusion the previous commit only assumed.
    ///
    /// Mutation run 2026-11-25: with the claim skipped, the second view below
    /// is `Persistent` and two writers hold slices at one base.
    #[test]
    fn a_second_overlapping_persistent_view_falls_back_to_a_temporary_one() {
        let pages = allocate(3);
        let first = View::persistent(&slice_of(pages[0]), 1).expect("one page");
        assert!(
            matches!(first, View::Persistent { .. }),
            "the first claim is kept"
        );
        // The same bytes again: refused as a persistent view...
        assert!(View::persistent(&slice_of(pages[0]), 1).is_none());
        // ...and the caller lands on a view of its own address.
        let second = Mapping::Persistent
            .view(&slice_of(pages[0]), 1)
            .expect("a temporary view");
        assert!(
            matches!(second, View::Temporary { .. }),
            "overlapping bytes shared one address"
        );
        // A neighbour's bytes do not touch, so they may share the mapping.
        let neighbour = View::persistent(&slice_of(pages[1]), 1).expect("next page");
        assert!(matches!(neighbour, View::Persistent { .. }));
        // Once the first is gone its bytes may be held again.
        drop(first);
        let again = View::persistent(&slice_of(pages[0]), 1).expect("free again");
        assert!(matches!(again, View::Persistent { .. }));
        drop((second, neighbour, again));
        assert_eq!(XMemFreePhysicalPages(3, pages.as_ptr()), 1);
    }

    /// The temporary path is what every run before this one used, and it
    /// stays exactly that: writes land, and the section is never mapped
    /// whole and never committed through that mapping.
    #[test]
    fn temporary_views_never_map_the_section_whole() {
        let pages = allocate(2);
        let section = slice_of(pages[0]).section.clone();
        let data: Vec<u8> = (0..2 * PAGE).map(|i| (i % 253) as u8).collect();
        assert!(write_physical_with(Mapping::Temporary, &pages, 0, &data));
        assert!(
            section.persistent.get().is_none(),
            "the temporary path mapped the section whole"
        );
        assert!(!section.committed_all(0, 2), "it committed through that map");
        let seen: Vec<u8> = pages.iter().flat_map(|&p| page_bytes(p)).collect();
        assert_eq!(seen, data);
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }

    /// The commit guard refuses a range that leaves the section and counts
    /// that overflow, which `runs` never produces but arithmetic can.
    #[test]
    fn ensure_committed_refuses_ranges_outside_the_section() {
        let pages = allocate(2);
        let section = slice_of(pages[0]).section.clone();
        let base = section.persistent_ptr().expect("mapped whole");
        assert!(section.ensure_committed(base, 0, 2));
        assert!(section.committed_all(0, 2));
        // All committed: the second call makes none, and says so.
        assert!(section.ensure_committed(base, 0, 2));
        // Past the section, and the two ways adding can overflow.
        assert!(!section.ensure_committed(base, 2, 1));
        assert!(!section.ensure_committed(base, 0, 3));
        assert!(!section.ensure_committed(base, usize::MAX, 1));
        assert!(!section.ensure_committed(base, 0, usize::MAX));
        assert!(!section.ensure_committed(base, usize::MAX / 2, usize::MAX / 2));
        // A range of no pages past the end: refused by the size comparison
        // alone. Delete that comparison and `committed_all` is vacuously
        // true here and the call succeeds, so this pins the clause the host
        // would otherwise supply the answer for.
        assert!(!section.ensure_committed(base, 3, 0));
        assert_eq!(XMemFreePhysicalPages(2, pages.as_ptr()), 1);
    }
}
