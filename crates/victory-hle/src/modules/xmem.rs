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
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};

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

struct Section(HANDLE);

// SAFETY: a section handle may be used from any thread.
unsafe impl Send for Section {}
// SAFETY: as above.
unsafe impl Sync for Section {}

impl Drop for Section {
    fn drop(&mut self) {
        // SAFETY: our handle; views keep the section alive on their own.
        unsafe { CloseHandle(self.0) };
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
    /// Adjacent mapping calls noted and joined (`VICTORY_XMEM_WHOLE_MB`,
    /// [`whole_bytes`]); empty while it is off.
    joins: Joins,
}

impl Physical {
    /// What the view at `at` shows there, if a view of ours is there.
    fn slice_at(&self, at: usize) -> Option<Slice> {
        self.views
            .range(..=at)
            .next_back()
            .filter(|(base, (size, _))| at < *base + size)
            .map(|(base, (_, s))| Slice {
                index: s.index + (at - base) / PAGE,
                section: s.section.clone(),
            })
    }
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
    let section = match new_section(size) {
        Ok(section) => section,
        Err(error) => {
            tracing::warn!(n, %error, "XMemAllocatePhysicalPages failed");
            // SAFETY: checked non-null.
            unsafe { count.write(0) };
            return 0;
        }
    };
    tracing::debug!(
        n,
        flags = format_args!("{flags:#x}"),
        "XMemAllocatePhysicalPages"
    );
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

/// A pagefile-backed section of `size` bytes for physical pages, its pages
/// committed when first mapped or written (`commit`): the title asks for
/// 6.1 GiB at once, which a busy host often cannot commit (error 1450).
/// Told to the GPU side.
fn new_section(size: usize) -> Result<Arc<Section>, std::io::Error> {
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
        return Err(std::io::Error::last_os_error());
    }
    // For the GPU side's in-place census: only a committed section can be
    // opened as a host heap.
    victory_gpu::title_memory_section(handle as usize, size as u64, on_wine());
    Ok(Arc::new(Section(handle)))
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

/// A view of our own on consecutive slices of one section, committed and
/// writable, unmapped when dropped.
struct View {
    at: MEMORY_MAPPED_VIEW_ADDRESS,
    len: usize,
}

impl View {
    /// Maps `n` slices from `slice` on.
    fn map(slice: &Slice, n: usize) -> Option<View> {
        use windows_sys::Win32::System::Memory::{FILE_MAP_WRITE, MapViewOfFile};
        let (offset, len) = ((slice.index * PAGE) as u64, n * PAGE);
        // For the frame log: these views take their turn in the kernel
        // beside the title's own mapping calls.
        let started = std::time::Instant::now();
        // SAFETY: slices of a section we own; the view is ours until dropped.
        let at = unsafe {
            MapViewOfFile(
                slice.section.0,
                FILE_MAP_WRITE,
                (offset >> 32) as u32,
                offset as u32,
                len,
            )
        };
        if at.Value.is_null() {
            return None;
        }
        let view = View { at, len };
        let committed = commit(at.Value as usize, len);
        victory_gpu::framelog::add(victory_gpu::framelog::T::DiskViews, 1);
        victory_gpu::framelog::add(victory_gpu::framelog::T::DiskViewUs, started.elapsed().as_micros() as u64);
        committed.then_some(view)
    }

    fn bytes(&mut self) -> &mut [u8] {
        // SAFETY: the view holds `len` committed read-write bytes, ours
        // alone while `self` lives (other views of the slices only alias
        // them, as the console's do).
        unsafe { std::slice::from_raw_parts_mut(self.at.Value.cast(), self.len) }
    }
}

impl Drop for View {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::UnmapViewOfFile;
        let started = std::time::Instant::now();
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(self.at) };
        victory_gpu::framelog::add(victory_gpu::framelog::T::DiskViewUs, started.elapsed().as_micros() as u64);
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

/// Copies `data` into physical pages, from `offset` bytes into the first,
/// whether or not they are mapped anywhere (DirectStorage destinations).
/// One view per run of consecutive pages, not one per page: each view
/// costs a map, a commit and an unmap (which flushes every core's TLB).
/// False if a page is unknown or the host fails.
pub(crate) fn write_physical(pages: &[usize], offset: usize, data: &[u8]) -> bool {
    let Some(runs) = runs(pages, offset, data.len()) else {
        return false;
    };
    let (mut within, mut done) = (offset % PAGE, 0);
    for (slice, n) in &runs {
        let Some(mut view) = View::map(slice, *n) else {
            return false;
        };
        let k = (n * PAGE - within).min(data.len() - done);
        view.bytes()[within..within + k].copy_from_slice(&data[done..done + k]);
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
}

impl PageWindow {
    pub(crate) fn bytes(&mut self) -> &mut [u8] {
        let (within, len) = (self.within, self.len);
        &mut self.view.bytes()[within..within + len]
    }
}

/// A window on bytes `[offset, offset + len)` of `pages` when they are
/// consecutive pages of one allocation (how titles usually hand them out);
/// `None` otherwise (pages of several allocations, out of order, or
/// unknown, or a host failure), which `write_physical` then handles or
/// reports.
pub(crate) fn page_window(pages: &[usize], offset: usize, len: usize) -> Option<PageWindow> {
    let runs = runs(pages, offset, len)?;
    let [(slice, n)] = runs.as_slice() else {
        return None;
    };
    Some(PageWindow {
        view: View::map(slice, *n)?,
        within: offset % PAGE,
        len,
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
        if whole_bytes().is_some() {
            physical.joins.forget(base, n * PAGE);
        }
        return address;
    }
    // SAFETY: the title passes `n` page numbers.
    let pages = unsafe { std::slice::from_raw_parts(pages, n) };
    if let Some(min) = whole_bytes() {
        // Beside earlier calls: one span with them, perhaps a section of
        // its own ([`join`], [`whole`]).
        join(&mut physical, base, pages, min);
        let slices: Vec<Option<Slice>> = pages.iter().map(|p| physical.pages.get(p).cloned()).collect();
        match whole(&mut physical, base, pages, &slices, min) {
            Whole::Not => {}
            Whole::Mapped => return address,
            Whole::Failed => return std::ptr::null_mut(),
        }
    }
    if !map_runs(&mut physical, base, pages) {
        return std::ptr::null_mut();
    }
    address
}

/// Maps the pages `pages` name at `base` (unmapped already), a view per
/// run of consecutive slices of one section; false if a page is unknown
/// or the host refuses.
fn map_runs(physical: &mut Physical, base: usize, pages: &[usize]) -> bool {
    let n = pages.len();
    let mut i = 0;
    while i < n {
        let Some(first) = physical.pages.get(&pages[i]).cloned() else {
            tracing::warn!(page = pages[i], "XMemMapPhysicalPages: unknown page");
            return false;
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
        // `VICTORY_XMEM_WHOLE_MB`: the range may be several placeholders
        // that views of a joined section left ([`map_pieces`]).
        let mapped = if whole_bytes().is_some() {
            map_pieces(physical, base + i * PAGE, run * PAGE, &first)
        } else {
            map(physical, base + i * PAGE, run * PAGE, first)
        };
        if !mapped {
            return false;
        }
        i += run;
    }
    true
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
    // `VICTORY_XMEM_WHOLE_MB`: pages onto consecutive addresses, enough to
    // be joined, are one mapping call.
    if let Some(min) = whole_bytes()
        && !pages.is_null()
        && n.saturating_mul(PAGE) >= WHOLE_PIECE.min(min / 4)
    {
        // SAFETY: the title passes `n` addresses and `n` pages.
        let (first, numbers) = unsafe { (*addresses as usize, std::slice::from_raw_parts(pages, n)) };
        // SAFETY: as above.
        let consecutive = (0..n).all(|i| unsafe { *addresses.add(i) as usize } == first.wrapping_add(i * PAGE));
        if first.is_multiple_of(PAGE) && consecutive && numbers.iter().all(|&p| p != 0) {
            return (!XMemMapPhysicalPages(first as *mut c_void, n, numbers.as_ptr()).is_null()).into();
        }
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
        let Some(slice) = physical.slice_at(at) else {
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
                slice.section.0,
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
    victory_gpu::title_memory_view(base, len, slice.section.0 as usize, (slice.index * PAGE) as u64);
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

// `VICTORY_XMEM_WHOLE_MB`: adjacent mapping calls joined into one run of a
// section of their own.

/// `VICTORY_XMEM_WHOLE_MB=<n>` (unset, 0 or not a number: off, as before):
/// adjacent mapping calls below 4 GiB that together reach `n` MB (up to
/// 1 GiB) are joined ([`join`]): their pages become consecutive slices of
/// one section made for the span, each call at its place in it, whatever
/// the order of the calls ([`whole`]). A single call of `n` MB or more is
/// taken the same way. The span is then one run of one section, which the
/// GPU side can open as a host heap where the title has it
/// (`VICTORY_INPLACE_CONSTANTS`, `OpenExistingHeapFromAddress1` on a view
/// of that run).
///
/// Why: the title reserves its 174 MB constants heap and maps it in three
/// adjacent calls, out of address order (about 47 MB, then 100 MB above
/// it, then 27 MB below them), from pages that are not one run of one
/// section. Mapped by runs, that is views of several sections.
///
/// What it costs: each joined call's pages are copied once into the new
/// section, under the mapping lock; a span that grows is made anew and its
/// pages copied again (147 MB, then 174 MB for the heap). The old pages'
/// committed bytes stay committed until their section goes. Calls above
/// 4 GiB are left alone.
fn whole_bytes() -> Option<usize> {
    static BYTES: LazyLock<Option<usize>> = LazyLock::new(|| {
        let mb = std::env::var("VICTORY_XMEM_WHOLE_MB")
            .ok()?
            .trim()
            .parse::<usize>()
            .ok()
            .filter(|&mb| mb > 0)?;
        tracing::info!(
            "xmem whole: adjacent mapping calls below 4 GiB that reach {mb} MB together (up to 1 GiB) become one run of a section \
             of their own (VICTORY_XMEM_WHOLE_MB)"
        );
        Some(mb.saturating_mul(MIB))
    });
    *BYTES
}

/// The largest span [`whole`] takes.
const WHOLE_MOST: usize = 1024 * MIB;

/// The smallest call [`join`] notes (with the switch's size under 4 MB: a
/// quarter of it). Smaller ones, page-by-page streaming, map as before.
const WHOLE_PIECE: usize = MIB;

/// [`join`] and [`whole`] take memory below this only: what the card reads
/// in place must lie there, and the title's big pools above it are shown at
/// other addresses as well.
const WHOLE_BELOW: usize = 1 << 32;

/// At most this many runs of other addresses that show a call's pages too
/// (mapped again onto the new slices) before [`whole`] leaves the call.
const WHOLE_ALIASES: usize = 4096;

/// At most this many spans are kept (the oldest not joined goes first).
const SPANS_MOST: usize = 256;

/// What [`join`] and [`whole`] keep (in [`Physical`]).
#[derive(Default)]
struct Joins {
    /// Mapping calls of [`WHOLE_PIECE`] and more below 4 GiB, as spans of
    /// adjacent calls.
    spans: Vec<Span>,
    /// Spans joined, by address: each has a section of its own.
    joined: BTreeMap<usize, Joined>,
}

/// Adjacent mapping calls.
struct Span {
    start: usize,
    end: usize,
    /// Joined: a section of its own ([`Joins::joined`] at `start`).
    joined: bool,
    /// The calls: address and page numbers.
    pieces: Vec<(usize, Vec<usize>)>,
}

/// A joined span's section.
struct Joined {
    size: usize,
    /// Made at the first call into it that [`whole`] takes.
    section: Option<Arc<Section>>,
    /// The section could not be made: left alone.
    refused: bool,
    /// The page number last given each slice of the section (0: none); a
    /// slice is another page's only while that number still names it.
    owners: Vec<usize>,
}

impl Joins {
    /// The title unmaps `[base, base + len)`: the calls noted there are no
    /// more (a span not joined with none left goes).
    fn forget(&mut self, base: usize, len: usize) {
        let end = base + len;
        for span in &mut self.spans {
            span.pieces.retain(|(at, numbers)| *at + numbers.len() * PAGE <= base || *at >= end);
        }
        self.spans.retain(|s| s.joined || !s.pieces.is_empty());
    }

    /// Keeps `span`, making room for it.
    fn keep(&mut self, span: Span) {
        if self.spans.len() >= SPANS_MOST
            && let Some(i) = self.spans.iter().position(|s| !s.joined)
        {
            self.spans.remove(i);
        }
        self.spans.push(span);
    }

    /// `[at, at + size)` is joined, its section made when first needed;
    /// joined spans it overlaps go (their sections live on in the slices
    /// their pages are).
    fn track(&mut self, at: usize, size: usize) {
        let pages = size.div_ceil(PAGE);
        let end = at + pages * PAGE;
        let over: Vec<usize> =
            self.joined.range(..end).filter(|(s, r)| **s + r.size > at).map(|(s, _)| *s).collect();
        for s in over {
            self.joined.remove(&s);
        }
        let joined = Joined { size: pages * PAGE, section: None, refused: false, owners: vec![0; pages] };
        self.joined.insert(at, joined);
    }
}

/// Before [`whole`] takes the call of `numbers` at `base` (unmapped
/// already): a call exactly beside earlier ones (a [`Span`] of them) that
/// together with them reaches `min` bytes makes the whole span a joined
/// one, with a section of its own, and the earlier calls' pages are moved
/// into it at once; [`whole`] then takes the call as one into the span. A
/// joined span that grows is made anew, its pages moved once more. Calls
/// inside a joined span take the place of what they cover in it.
fn join(physical: &mut Physical, base: usize, numbers: &[usize], min: usize) {
    let (bytes, end) = (numbers.len() * PAGE, base + numbers.len() * PAGE);
    if bytes < WHOLE_PIECE.min(min / 4) || end > WHOLE_BELOW || !base.is_multiple_of(PAGE) {
        return;
    }
    let joins = &mut physical.joins;
    if let Some(span) = joins.spans.iter_mut().find(|s| s.joined && s.start <= base && end <= s.end) {
        span.pieces.retain(|(at, ns)| *at + ns.len() * PAGE <= base || *at >= end);
        span.pieces.push((base, numbers.to_vec()));
        return;
    }
    // Spans the call overlaps otherwise are forgotten.
    let mut dropped: Vec<usize> = Vec::new();
    joins.spans.retain(|s| {
        let keep = s.end <= base || s.start >= end;
        if !keep && s.joined {
            dropped.push(s.start);
        }
        keep
    });
    for at in dropped {
        joins.joined.remove(&at);
    }
    let left = joins.spans.iter().position(|s| s.end == base);
    let right = joins.spans.iter().position(|s| s.start == end);
    let start = left.map_or(base, |i| joins.spans[i].start);
    let stop = right.map_or(end, |i| joins.spans[i].end);
    let merged = stop - start;
    let mut beside: Vec<usize> = [left, right].into_iter().flatten().collect();
    beside.sort_unstable_by(|a, b| b.cmp(a));
    let taken: Vec<Span> = beside.into_iter().map(|i| joins.spans.remove(i)).collect();
    let call = (base, numbers.to_vec());
    if taken.is_empty() {
        joins.keep(Span { start: base, end, joined: false, pieces: vec![call] });
        return;
    }
    let mut earlier: Vec<(usize, Vec<usize>)> = taken.iter().flat_map(|s| s.pieces.iter().cloned()).collect();
    earlier.sort_unstable_by_key(|p| p.0);
    let shared = {
        let mine: std::collections::HashSet<usize> = numbers.iter().copied().collect();
        earlier.iter().any(|(_, ns)| ns.iter().any(|p| mine.contains(p)))
    };
    if merged < min || merged > WHOLE_MOST || shared {
        if merged >= min {
            let why = if shared { "a page of the call is an earlier call's too" } else { "over 1024 MB together" };
            tracing::info!(
                address = format_args!("{base:#x}"),
                span = format_args!("{start:#x}+{merged:#x}"),
                why,
                "xmem whole: adjacent calls not joined (VICTORY_XMEM_WHOLE_MB)"
            );
        }
        if taken.iter().any(|s| s.joined) {
            // Those stay as they were; the call is a span of its own.
            for span in taken {
                joins.keep(span);
            }
            joins.keep(Span { start: base, end, joined: false, pieces: vec![call] });
        } else {
            let mut pieces = earlier;
            pieces.push(call);
            joins.keep(Span { start, end: stop, joined: false, pieces });
        }
        return;
    }
    let grown: Vec<String> =
        taken.iter().filter(|s| s.joined).map(|s| format!("{:#x}+{:#x}", s.start, s.end - s.start)).collect();
    joins.track(start, merged);
    tracing::info!(
        address = format_args!("{base:#x}"),
        bytes,
        span = format_args!("{start:#x}+{merged:#x}"),
        mb = merged >> 20,
        earlier = earlier.len(),
        grown_from = %grown.join(", "),
        "xmem whole: adjacent calls joined: one section for the span, the earlier calls' pages moved into it (VICTORY_XMEM_WHOLE_MB)"
    );
    let mut pieces = Vec::with_capacity(earlier.len() + 1);
    for (at, ns) in earlier {
        let len = ns.len() * PAGE;
        // Still what the title mapped there?
        let still = ns.iter().enumerate().all(|(k, p)| {
            physical.pages.get(p).is_some_and(|s| {
                physical
                    .slice_at(at + k * PAGE)
                    .is_some_and(|there| Arc::ptr_eq(&there.section, &s.section) && there.index == s.index)
            })
        });
        if !still {
            tracing::info!(
                address = format_args!("{at:#x}"),
                bytes = len,
                "xmem whole: an earlier call left where it is: the title mapped other pages there since"
            );
            continue;
        }
        unmap(physical, at, len);
        let slices: Vec<Option<Slice>> = ns.iter().map(|p| physical.pages.get(p).cloned()).collect();
        let done = match whole(physical, at, &ns, &slices, min) {
            Whole::Mapped => true,
            Whole::Failed => false,
            Whole::Not => map_runs(physical, at, &ns),
        };
        if !done {
            tracing::warn!(
                address = format_args!("{at:#x}"),
                bytes = len,
                "xmem whole: an earlier call's pages not mapped again after the move (the host refused)"
            );
        }
        pieces.push((at, ns));
    }
    pieces.push(call);
    physical.joins.keep(Span { start, end: stop, joined: true, pieces });
}

/// What [`whole`] did with a mapping call.
enum Whole {
    /// Nothing: map it as before.
    Not,
    /// Mapped, as consecutive slices of a section of its own.
    Mapped,
    /// The host failed after the pages were moved: the call fails.
    Failed,
}

/// Maps `numbers` at `base` (unmapped already, [`unmap`]) as consecutive
/// slices of one section: the section of the joined span the call lies in,
/// at the call's place in it, or else (a call of `min` bytes or more, below
/// 4 GiB) a new section of exactly the call's size. Each page that is not
/// that slice yet has its bytes copied into it, and its page number names
/// it from then on (DirectStorage writes, frees and later mappings go
/// there); any other address that shows such a page is mapped again onto
/// the new slice (the old one back if the host refuses, said in the log).
/// The old slices are left to their sections, unnamed. Pages the title maps
/// or unmaps there later go page by page as ever ([`unmap`] cuts the view).
///
/// `slices`: what each number names now. With [`PHYSICAL`] and the change
/// of mapping held. Says on one line what it did, or why not when the call
/// was one to take.
fn whole(physical: &mut Physical, base: usize, numbers: &[usize], slices: &[Option<Slice>], min: usize) -> Whole {
    let (n, bytes) = (numbers.len(), numbers.len() * PAGE);
    let end = base + bytes;
    if n == 0 || !base.is_multiple_of(PAGE) || end > WHOLE_BELOW {
        return Whole::Not;
    }
    let not = |why: String| {
        tracing::info!(
            address = format_args!("{base:#x}"),
            bytes,
            mb = bytes >> 20,
            why = %why,
            "xmem whole: a mapping call mapped as before, by runs of its pages' sections (VICTORY_XMEM_WHOLE_MB)"
        );
        Whole::Not
    };
    let within = physical
        .joins
        .joined
        .range(..=base)
        .next_back()
        .filter(|(at, r)| end <= **at + r.size)
        .map(|(at, _)| *at);
    // The section and the slice of it the call's first page becomes.
    let (section, offset, span) = match within {
        Some(at) => {
            let Some(r) = physical.joins.joined.get_mut(&at) else {
                return Whole::Not;
            };
            if r.refused || bytes < WHOLE_PIECE.min(min / 4) {
                return Whole::Not;
            }
            if r.section.is_none() {
                match new_section(r.size) {
                    Ok(section) => r.section = Some(section),
                    Err(error) => {
                        r.refused = true;
                        return not(format!("no section of {} MB for its span {at:#x}: {error}", r.size >> 20));
                    }
                }
            }
            (r.section.clone(), (base - at) / PAGE, Some((at, r.size)))
        }
        // Made below, once the call is known to be taken.
        None if (min..=WHOLE_MOST).contains(&bytes) => (None, 0, None),
        None => return Whole::Not,
    };
    let Some(old) = slices.iter().cloned().collect::<Option<Vec<Slice>>>() else {
        return not("a page the title does not own".into());
    };
    let mut seen = std::collections::HashSet::with_capacity(n);
    if !numbers.iter().all(|&p| seen.insert(p)) {
        return not("a page twice in the call".into());
    }
    // Which pages are their slice already (in a joined span's section).
    let is_slice = |s: &Slice, index: usize| {
        section.as_ref().is_some_and(|section| Arc::ptr_eq(&s.section, section) && s.index == index)
    };
    let placed: Vec<bool> = old.iter().enumerate().map(|(k, s)| is_slice(s, offset + k)).collect();
    if let Some(section) = &section
        && placed.iter().all(|&p| p)
    {
        // As they are: mapped again by the placeholders the views left
        // (one view over them all would fail).
        let again = map_pieces(physical, base, bytes, &Slice { section: section.clone(), index: offset });
        return if again { Whole::Mapped } else { Whole::Failed };
    }
    let mut runs: Vec<(Slice, usize)> = Vec::new();
    for slice in &old {
        match runs.last_mut() {
            Some((first, k)) if Arc::ptr_eq(&first.section, &slice.section) && slice.index == first.index + *k => {
                *k += 1;
            }
            _ => runs.push((slice.clone(), 1)),
        }
    }
    if span.is_none() && runs.len() == 1 {
        // One run of one section already.
        return Whole::Not;
    }
    if let Some((at, _)) = span
        && let Some(r) = physical.joins.joined.get(&at)
    {
        // A slice another page still is: that page is the title's, mapped
        // wherever; not taken from it.
        for k in (0..n).filter(|&k| !placed[k]) {
            let owner = r.owners[offset + k];
            let taken = owner != 0
                && owner != numbers[k]
                && physical.pages.get(&owner).is_some_and(|s| is_slice(s, offset + k));
            if taken {
                return not(format!("slice {:#x} of its span's section is page {owner:#x}'s still", offset + k));
            }
        }
    }
    let mut sections: Vec<usize> = runs.iter().map(|r| Arc::as_ptr(&r.0.section) as usize).collect();
    sections.sort_unstable();
    sections.dedup();
    // Other addresses that show the pages that move (the call's own range
    // is unmapped already), as runs: address, first page of the call, pages.
    let mut elsewhere: Vec<(usize, usize)> = old
        .iter()
        .enumerate()
        .filter(|&(k, _)| !placed[k])
        .flat_map(|(k, s)| {
            physical.shown.get(&s.key(0)).into_iter().flatten().map(move |&at| (at, k))
        })
        .collect();
    elsewhere.sort_unstable();
    let mut aliases: Vec<(usize, usize, usize)> = Vec::new();
    for (at, k) in elsewhere {
        match aliases.last_mut() {
            Some((from, first, len)) if *from + *len * PAGE == at && *first + *len == k => *len += 1,
            _ => aliases.push((at, k, 1)),
        }
    }
    if aliases.len() > WHOLE_ALIASES {
        return not(format!("its pages are shown elsewhere too in {} runs", aliases.len()));
    }
    let section = match section {
        Some(section) => section,
        None => match new_section(bytes) {
            Ok(section) => section,
            Err(error) => return not(format!("no section of {} MB: {error}", bytes >> 20)),
        },
    };
    // The bytes the moving pages hold (pages never committed hold zeros,
    // as the new ones do), by runs of the call that are runs of a section
    // too.
    let took = std::time::Instant::now();
    let mut copied = 0usize;
    let mut k = 0;
    while k < n {
        if placed[k] {
            k += 1;
            continue;
        }
        let first = &old[k];
        let len = 1 + (k + 1..n)
            .take_while(|&j| {
                !placed[j] && Arc::ptr_eq(&old[j].section, &first.section) && old[j].index == first.index + (j - k)
            })
            .count();
        match copy_pages(first, len, &Slice { section: section.clone(), index: offset + k }) {
            Some(bytes) => copied += bytes,
            None => return not(format!("the bytes of its pages not copied: {}", std::io::Error::last_os_error())),
        }
        k += len;
    }
    let copy_ms = took.elapsed().as_secs_f64() * 1000.0;
    for (k, number) in numbers.iter().enumerate() {
        // (A page freed meanwhile stays freed: numbers are never given out
        // again.)
        if let Some(slice) = physical.pages.get_mut(number) {
            *slice = Slice { section: section.clone(), index: offset + k };
        }
    }
    if let Some((at, _)) = span
        && let Some(r) = physical.joins.joined.get_mut(&at)
    {
        r.owners[offset..offset + n].copy_from_slice(numbers);
    }
    let mut restored = 0;
    for &(at, first, len) in &aliases {
        unmap(physical, at, len * PAGE);
        if map_pieces(physical, at, len * PAGE, &Slice { section: section.clone(), index: offset + first }) {
            continue;
        }
        // Not left showing nothing: what it showed before (its bytes as
        // they were now; the page's writes go to the new slice from now).
        unmap(physical, at, len * PAGE);
        let back = (0..len).all(|p| map_pieces(physical, at + p * PAGE, PAGE, &old[first + p]));
        tracing::warn!(
            alias = format_args!("{at:#x}"),
            bytes = len * PAGE,
            call = format_args!("{base:#x}"),
            back,
            "xmem whole: another address that showed the call's pages not mapped onto the new section; its old pages mapped \
             back there (back=false: it shows nothing)"
        );
        restored += 1;
    }
    let handle = section.0 as usize;
    let mapped = map_pieces(physical, base, bytes, &Slice { section, index: offset });
    let span = match span {
        Some((at, size)) => format!("{at:#x}+{size:#x}"),
        None => "none (a section of the call's own)".into(),
    };
    tracing::info!(
        address = format_args!("{base:#x}"),
        bytes,
        mb = bytes >> 20,
        section = format_args!("{handle:#x}"),
        span = %span,
        slice = format_args!("{:#x}", offset * PAGE),
        moved = placed.iter().filter(|&&p| !p).count(),
        was_views = runs.len(),
        was_sections = sections.len(),
        elsewhere = aliases.len(),
        elsewhere_not_moved = restored,
        copied_mb = copied >> 20,
        copy_ms = format_args!("{copy_ms:.1}"),
        mapped,
        "xmem whole: a mapping call made one run of a section of its own (VICTORY_XMEM_WHOLE_MB)"
    );
    if mapped { Whole::Mapped } else { Whole::Failed }
}

/// [`map`] of `[base, base + len)`, which [`unmap`] may have left as
/// several placeholders (one per view it took away): a view per
/// placeholder, the slices following on from `slice`; page by page where
/// the host refuses a piece.
fn map_pieces(physical: &mut Physical, base: usize, len: usize, slice: &Slice) -> bool {
    let end = base + len;
    let at_slice = |at: usize| Slice { section: slice.section.clone(), index: slice.index + (at - base) / PAGE };
    let mut at = base;
    while at < end {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query of our own address space; the structure is ours.
        let got = unsafe {
            VirtualQueryEx(GetCurrentProcess(), at as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>())
        };
        let region_end = if got == 0 { end } else { (info.BaseAddress as usize + info.RegionSize).min(end) };
        let piece = (region_end.max(at + PAGE) - at).next_multiple_of(PAGE).min(end - at);
        if !map(physical, at, piece, at_slice(at)) {
            if piece == PAGE {
                return false;
            }
            for page in (at..at + piece).step_by(PAGE) {
                if !map(physical, page, PAGE, at_slice(page)) {
                    return false;
                }
            }
        }
        at += piece;
    }
    true
}

/// Copies the committed bytes of `n` slices from `from` on into as many
/// from `to` on, committing those: the bytes copied. `None` if the host
/// gave no view or commit, or the two overlap in one section.
fn copy_pages(from: &Slice, n: usize, to: &Slice) -> Option<usize> {
    use windows_sys::Win32::System::Memory::{FILE_MAP_READ, MapViewOfFile, UnmapViewOfFile};
    if Arc::ptr_eq(&from.section, &to.section) && from.index < to.index + n && to.index < from.index + n {
        return None;
    }
    let len = n * PAGE;
    let mut target = View::map(to, n)?;
    let offset = (from.index * PAGE) as u64;
    // A view of the source that commits nothing: its pages never committed
    // are left as zeros.
    // SAFETY: slices of a section we own; the view is unmapped below.
    let source = unsafe { MapViewOfFile(from.section.0, FILE_MAP_READ, (offset >> 32) as u32, offset as u32, len) };
    if source.Value.is_null() {
        return None;
    }
    let read = source.Value as usize;
    let (mut at, mut copied) = (0, 0);
    while at < len {
        let mut info = MEMORY_BASIC_INFORMATION::default();
        // SAFETY: a query of our own view; the structure is ours.
        let got = unsafe {
            VirtualQueryEx(GetCurrentProcess(), (read + at) as *const c_void, &mut info, size_of::<MEMORY_BASIC_INFORMATION>())
        };
        // (Regions end on 4 KiB pages; our views commit whole 64 KiB ones.)
        let next = if got == 0 {
            len
        } else {
            (info.BaseAddress as usize + info.RegionSize).saturating_sub(read).max(at + 4096).min(len)
        };
        if got != 0 && info.State == MEM_COMMIT {
            // SAFETY: committed bytes of our view of the source, into our
            // view of the target (other pages of other sections).
            let bytes = unsafe { std::slice::from_raw_parts((read + at) as *const u8, next - at) };
            target.bytes()[at..next].copy_from_slice(bytes);
            copied += next - at;
        }
        at = next;
    }
    // SAFETY: our own view.
    unsafe { UnmapViewOfFile(source) };
    Some(copied)
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

    /// The bytes of one page, through a view of our own.
    fn page_bytes(page: usize) -> Vec<u8> {
        let slice = physical().pages.get(&page).cloned().expect("known page");
        View::map(&slice, 1).expect("mapped").bytes().to_vec()
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

    /// A reservation of `n` pages below 4 GiB, where `whole` takes memory.
    fn reserve_low(n: usize) -> *mut c_void {
        let at = XMemVirtualAlloc(std::ptr::null_mut(), n * PAGE, MEM_RESERVE | PAGES_64K, ADDRESS_32BIT, PAGE_READWRITE);
        assert!(!at.is_null() && (at as usize) + n * PAGE <= WHOLE_BELOW);
        at
    }

    /// A mapping call as `XMemMapPhysicalPages` makes it with
    /// `VICTORY_XMEM_WHOLE_MB` of `min` bytes (the switch itself is off in
    /// tests): [`join`] and [`whole`], else as before.
    fn call_whole(at: *mut c_void, from: usize, numbers: &[usize], min: usize) -> bool {
        let base = at as usize + from * PAGE;
        let done = {
            let mut physical = physical();
            let _changing = victory_gpu::title_memory_remapping();
            unmap(&mut physical, base, numbers.len() * PAGE);
            join(&mut physical, base, numbers, min);
            let slices: Vec<Option<Slice>> = numbers.iter().map(|p| physical.pages.get(p).cloned()).collect();
            whole(&mut physical, base, numbers, &slices, min)
        };
        match done {
            Whole::Mapped => true,
            Whole::Not => XMemMapPhysicalPages(base as *mut c_void, numbers.len(), numbers.as_ptr()) == base as *mut c_void,
            Whole::Failed => false,
        }
    }

    /// The `n` pages at `at` are consecutive slices of one section, from its
    /// first on, each with the byte `from + k + 1` first (as the GPU side's
    /// in-place census needs them: one run of one section); the section.
    fn one_run(at: *mut c_void, n: usize, from: usize, what: &str) -> Arc<Section> {
        let section = {
            let physical = physical();
            let first = physical.slice_at(at as usize).expect("mapped");
            for k in 0..n {
                let slice = physical.slice_at(at as usize + k * PAGE).expect("mapped");
                assert!(
                    Arc::ptr_eq(&slice.section, &first.section) && slice.index == k,
                    "{what}: page {k} is slice {k}"
                );
            }
            first.section.clone()
        };
        for k in 0..n {
            // SAFETY: mapped read-write.
            assert_eq!(unsafe { at.cast::<u8>().add(k * PAGE).read() }, (from + k) as u8 + 1, "{what}: page {k} kept its bytes");
        }
        section
    }

    /// One test that joins calls at a time: spans of two tests could lie
    /// side by side (their reservations may), and they would be joined.
    fn joining_alone() -> MutexGuard<'static, ()> {
        static ONE: Mutex<()> = Mutex::new(());
        ONE.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Forgets what the tests above noted at `[at, at + n pages)`.
    fn forget_joins(at: *mut c_void, n: usize) {
        let (start, end) = (at as usize, at as usize + n * PAGE);
        let mut physical = physical();
        physical.joins.spans.retain(|s| s.end <= start || s.start >= end);
        physical.joins.joined.retain(|&s, r| s + r.size <= start || s >= end);
    }

    /// `VICTORY_XMEM_WHOLE_MB`: one call of pages of two allocations, out of
    /// order, one of them shown at another address too, becomes one view of
    /// a section of its own, with their bytes; the other address, writes by
    /// page number and later page-by-page mappings follow.
    #[test]
    fn a_whole_call_is_one_run_of_a_section_of_its_own() {
        let (a, b) = (allocate(3), allocate(2));
        let numbers = [b[0], a[2], a[0], b[1], a[1]];
        for (k, &p) in numbers.iter().enumerate() {
            assert!(write_physical(&[p], 0, &[k as u8 + 1; 16]));
        }
        // Two of the pages shown at another address too, as two views.
        let (at, other) = (reserve_low(5), reserve_low(2));
        assert_eq!(XMemMapPhysicalPages(other, 2, [a[2], a[0]].as_ptr()), other);
        {
            let mut physical = physical();
            let _changing = victory_gpu::title_memory_remapping();
            let slices: Vec<Option<Slice>> = numbers.iter().map(|p| physical.pages.get(p).cloned()).collect();
            assert!(matches!(whole(&mut physical, at as usize, &numbers, &slices, PAGE), Whole::Mapped));
            let (size, slice) = physical.views.get(&(at as usize)).expect("one view");
            assert_eq!((*size, slice.index), (5 * PAGE, 0));
            assert!(slices.iter().flatten().all(|s| !Arc::ptr_eq(&s.section, &slice.section)));
        }
        let section = one_run(at, 5, 0, "one call");
        let byte = |p: *mut c_void, off: usize| {
            // SAFETY: mapped read-write.
            unsafe { p.cast::<u8>().add(off).read() }
        };
        // The other address shows the page's new slice.
        // SAFETY: mapped read-write.
        unsafe { at.cast::<u8>().add(PAGE).write(0x77) };
        assert_eq!(byte(other, 0), 0x77);
        // Writes by page number land in the run.
        assert!(write_physical(&[a[0]], 1, &[0x55]));
        assert_eq!(byte(at, 2 * PAGE + 1), 0x55);
        assert_eq!(byte(other, PAGE + 1), 0x55);
        assert!(page_window(&numbers, 0, 5 * PAGE).is_some(), "one run now");
        // A page unmapped and mapped again in the middle: page by page.
        let middle = at.wrapping_byte_add(2 * PAGE);
        assert_eq!(XMemMapPhysicalPages(middle, 1, std::ptr::null()), middle);
        assert_eq!(byte(at, 3 * PAGE), 4);
        assert_eq!(XMemMapPhysicalPages(middle, 1, [a[0]].as_ptr()), middle);
        assert_eq!(byte(middle, 1), 0x55);
        {
            let physical = physical();
            let (_, slice) = physical.views.get(&(middle as usize)).expect("a view of the page");
            assert!(Arc::ptr_eq(&slice.section, &section) && slice.index == 2);
        }
        drop(section);
        for (p, n) in [(at, 5), (other, 2)] {
            assert_eq!(XMemMapPhysicalPages(p, n, std::ptr::null()), p);
        }
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(2, b.as_ptr()), 1);
    }

    /// `VICTORY_XMEM_WHOLE_MB`: three adjacent calls of one reservation, out
    /// of address order (the top, the bottom, then the middle that joins
    /// them), each of pages out of order across allocations, end as
    /// consecutive slices of the joined span's one section. Mapping the same
    /// pages there again changes nothing, and a page the title still has
    /// keeps its slice of the section.
    #[test]
    fn adjacent_calls_out_of_order_are_one_run_of_a_section() {
        let _one = joining_alone();
        let (a, b) = (allocate(3), allocate(3));
        let numbers = [b[0], a[2], a[0], b[2], a[1], b[1]];
        for (k, &p) in numbers.iter().enumerate() {
            assert!(write_physical(&[p], 0, &[k as u8 + 1; 16]));
        }
        let at = reserve_low(6);
        let min = 6 * PAGE;
        assert!(call_whole(at, 4, &numbers[4..6], min), "the top: as before");
        assert!(call_whole(at, 0, &numbers[0..2], min), "the bottom: as before");
        assert!(call_whole(at, 2, &numbers[2..4], min), "the middle joins them");
        let section = one_run(at, 6, 0, "three calls");
        assert!(page_window(&numbers, 0, 6 * PAGE).is_some(), "one run of pages");
        // The same pages again, in one call: as they are.
        assert!(call_whole(at, 0, &numbers, min));
        assert!(Arc::ptr_eq(&one_run(at, 6, 0, "again"), &section));
        // Page numbers[3] is still the title's at slice 3: another page for
        // that place is not given the slice.
        let c = allocate(1);
        {
            let mut physical = physical();
            let _changing = victory_gpu::title_memory_remapping();
            let slices = vec![physical.pages.get(&c[0]).cloned()];
            assert!(matches!(whole(&mut physical, at as usize + 3 * PAGE, &c, &slices, min), Whole::Not));
        }
        drop(section);
        assert_eq!(XMemMapPhysicalPages(at, 6, std::ptr::null()), at);
        forget_joins(at, 6);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(3, b.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(1, c.as_ptr()), 1);
    }

    /// `VICTORY_XMEM_WHOLE_MB`: three adjacent calls in the title's order for
    /// its constants heap (the middle one, under the size alone, then the
    /// one above, then the one below) are joined into one span when the
    /// second comes, the span grown at the third, and end as consecutive
    /// slices of one section, with their bytes.
    #[test]
    fn adjacent_calls_in_the_titles_order_are_joined_and_grown() {
        let _one = joining_alone();
        let (a, b, c) = (allocate(3), allocate(3), allocate(2));
        let (bottom, middle, top) = ([c[1], c[0]], [a[2], a[0], a[1]], [b[1], b[2], b[0]]);
        let all: Vec<usize> = bottom.iter().chain(&middle).chain(&top).copied().collect();
        for (k, &p) in all.iter().enumerate() {
            assert!(write_physical(&[p], 0, &[k as u8 + 1; 16]));
        }
        let at = reserve_low(8);
        let min = 4 * PAGE;
        assert!(call_whole(at, 2, &middle, min), "under the size alone: as before");
        assert!(call_whole(at, 5, &top, min), "joined with the middle");
        let first = one_run(at.wrapping_byte_add(2 * PAGE), 6, 2, "two calls");
        assert!(call_whole(at, 0, &bottom, min), "the span grown");
        let section = one_run(at, 8, 0, "three calls");
        assert!(!Arc::ptr_eq(&first, &section), "a span that grows is made anew");
        assert!(page_window(&all, 0, 8 * PAGE).is_some(), "one run of pages");
        drop((first, section));
        assert_eq!(XMemMapPhysicalPages(at, 8, std::ptr::null()), at);
        forget_joins(at, 8);
        assert_eq!(XMemFreePhysicalPages(3, a.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(3, b.as_ptr()), 1);
        assert_eq!(XMemFreePhysicalPages(2, c.as_ptr()), 1);
    }
}
