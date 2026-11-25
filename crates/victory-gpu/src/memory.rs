//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_DIRTY_DWORDS, MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer).
const COPIES: u32 = 8191;
/// A directory page.
const PAGE: u64 = 1 << 16;
/// A copy counts as one shaders wrote lately for this many frames after a
/// dirty-page bitmask hit (`take_dirty`), covering lists recorded while
/// earlier submissions were still running.
const MARK_FRAMES: u64 = 8;

/// Maximum bytes of newly allocated non-buffer loose windows kept for
/// raw-address shaders (`VICTORY_MEMORY_MB`, 1536 MiB by default).
fn max_window_bytes() -> u64 {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_MEMORY_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(1536)
            << 20
    })
}

struct Memory {
    /// Buffer, its mapping, its GPU address.
    directory: (ID3D12Resource, *mut u32, u64),
    faults: (ID3D12Resource, *mut u32, u64),
    /// Separate fault buffer for read-only / quarantined shaders (`!writes`) so
    /// they never pollute `faults`'s dirty-page bitmask tail.
    quarantine_faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`; `None`: numbered again ([`Memory::retarget`]).
    copies: Vec<Option<Arc<Shadow>>>,
    /// `crate::resident::generation` at the last [`Memory::retarget`].
    generation: u64,
    /// Per dirty 64 KiB page (harvested by `take_dirty`): `(copy_index, last_dirty_frame)`.
    dirty_pages: BTreeMap<u32, (usize, u64)>,
    /// Total bytes of loose (non-buffer) windows added to `copies`.
    window_bytes: u64,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
    /// Per-frame cached VirtualQuery extents `(base, end, committed)`.
    extents: Vec<(u64, u64, bool)>,
}

// SAFETY: the mappings are of buffers the struct owns; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// The directory's offset for title address `at` (in `shadow`): its byte
/// offset from the copy's host buffer's first byte. For title memory in
/// place (`crate::inplace`) the buffer begins on the 64 KiB line, or at an
/// arena's base, before the title memory: `Shadow::offset` adds that bias
/// (design §11, bug 1: the offset from the title memory's first byte was
/// off by the bias).
fn entry_offset(shadow: &Shadow, at: u64) -> u32 {
    shadow.offset(at) as u32
}

/// The bytes of a copy's raw view, from the host buffer's first byte: the
/// title memory's end at its offset (with a bias, `len` alone cut the view
/// `bias` bytes short of the title memory's end).
fn view_bytes(shadow: &Shadow) -> u64 {
    let (start, len) = shadow.span();
    shadow.offset(start) + len
}

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

enum MapOutcome {
    Mapped((u64, u64), usize),
    AlreadyMapped((u64, u64), usize),
    RefusedBudget,
    RefusedUncommitted,
    RefusedNonGpu,
    RefusedFull,
    RefusedOther,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        let make = |dwords: u64, what: &str| {
            let (buffer, data) = host.shared_buffer(dwords * 4)?;
            crate::host::name(&buffer, || format!("title memory by address: {what}"));
            // SAFETY: a fresh mapping of at least that many bytes.
            unsafe { std::ptr::write_bytes(data, 0, (dwords * 4) as usize) };
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            Some((buffer, data.cast::<u32>(), gpu))
        };
        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: make(u64::from(MEMORY_PAGES) * 2, "directory")?,
            faults: make(
                u64::from(MEMORY_FAULTS) + u64::from(MEMORY_DIRTY_DWORDS),
                "faults",
            )?,
            quarantine_faults: make(
                u64::from(MEMORY_FAULTS) + u64::from(MEMORY_DIRTY_DWORDS),
                "quarantine faults",
            )?,
            copies: Vec::new(),
            generation: 0,
            dirty_pages: BTreeMap::new(),
            window_bytes: 0,
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
            extents: Vec::new(),
        })
    }

    /// Committed allocation extent `[start, end)` around `address`.
    fn extent(&mut self, address: u64) -> Option<(u64, u64)> {
        if let Some((start, len)) = crate::shadow::extent(address) {
            return Some((start, start.saturating_add(len)));
        }
        if let Some((start, len)) = crate::inplace::view_span(address as usize) {
            return Some((start, start.saturating_add(len)));
        }
        if let Some(&(_, end, committed)) = self
            .extents
            .iter()
            .find(|&&(base, end, _)| address >= base && address < end)
        {
            return committed.then(|| {
                let &(base, _, _) = self
                    .extents
                    .iter()
                    .find(|&&(b, e, _)| address >= b && address < e)
                    .expect("found above");
                (base, end)
            });
        }
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: valid out pointer of the right size.
        if unsafe { VirtualQuery(address as *const _, &mut info, size_of_val(&info)) } == 0 {
            return None;
        }
        let base = info.BaseAddress as u64;
        let end = base.saturating_add(info.RegionSize as u64);
        let committed = info.State == MEM_COMMIT;
        self.extents.push((base, end, committed));
        committed.then_some((base, end))
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's. Returns `(span, copy_index)`.
    fn map(&mut self, address: u64) -> Option<((u64, u64), usize)> {
        match self.map_detail(address) {
            MapOutcome::Mapped(span, idx) | MapOutcome::AlreadyMapped(span, idx) => {
                Some((span, idx))
            }
            _ => None,
        }
    }

    fn map_detail(&mut self, address: u64) -> MapOutcome {
        let page = address / PAGE;
        // Reject low null/offset guards (< 1 MiB) and entries past the directory.
        if page < 16 || page >= u64::from(MEMORY_PAGES) - 1 {
            return MapOutcome::RefusedOther;
        }
        // SAFETY: inside the directory's mapping.
        let entry = unsafe { self.directory.1.add(page as usize * 2) };
        // SAFETY: as above.
        let existing = unsafe { entry.read() };
        if existing != 0 {
            let idx = (existing - 1) as usize;
            return self
                .copies
                .get(idx)
                .and_then(|s| s.as_ref())
                .map(|s| MapOutcome::AlreadyMapped(s.span(), idx))
                .unwrap_or(MapOutcome::RefusedOther);
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return MapOutcome::RefusedOther;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return MapOutcome::RefusedUncommitted;
        }
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let existing_shadow = crate::shadow::find_existing_writable(start, PAGE).filter(whole);
        let is_buffer = crate::shadow::extent(start).is_some();
        let is_title_gpu = is_buffer
            || crate::inplace::view_span(start as usize).is_some()
            || crate::watch::is_write_watched(start as usize, PAGE as usize);
        if !is_title_gpu {
            self.refused.insert(page);
            return MapOutcome::RefusedNonGpu;
        }
        let reused_existing = existing_shadow.is_some();
        if !is_buffer && !reused_existing && self.window_bytes >= max_window_bytes() {
            static BUDGET_ONCE: std::sync::Once = std::sync::Once::new();
            BUDGET_ONCE.call_once(|| {
                tracing::warn!(
                    mb = self.window_bytes >> 20,
                    "title memory by address: loose window budget reached (VICTORY_MEMORY_MB)"
                );
            });
            self.refused.insert(page);
            return MapOutcome::RefusedBudget;
        }
        // Registered buffer or existing writable copy covering `[start, start + PAGE)`
        // first, or a writable 4 MiB loose window matching `shadow::find` /
        // `shadow::find_writable` so `SLOT_MEMORY` and root UAVs/descriptors share
        // the same host buffer.
        let Some(shadow) = existing_shadow
            .or_else(|| crate::shadow::find_writable(start, PAGE).filter(whole))
        else {
            self.refused.insert(page);
            return MapOutcome::RefusedOther;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return MapOutcome::RefusedOther;
        }
        let span = shadow.span();
        let n = match self.copies.iter().position(|c| c.as_ref().is_some_and(|c| Arc::ptr_eq(c, &shadow))) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                if !is_buffer && !reused_existing {
                    self.window_bytes = self.window_bytes.saturating_add(span.1);
                }
                self.copies.push(Some(shadow.clone()));
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = span;
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    if crate::texture::committed(at as *const u8, PAGE as usize) {
                        // SAFETY: inside the directory's mapping.
                        let other = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                        // SAFETY: as above; the offset first, as below.
                        unsafe {
                            if other.read() == 0 {
                                other.add(1).write_volatile(entry_offset(&shadow, at));
                                other.write_volatile(n as u32 + 1);
                            }
                        }
                    }
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return MapOutcome::RefusedFull;
            }
        };
        let offset = entry_offset(&shadow, start);
        // The offset first: a shader that sees the number sees it too.
        // SAFETY: inside the directory's mapping.
        unsafe {
            entry.add(1).write_volatile(offset);
            entry.write_volatile(n as u32 + 1);
        }
        MapOutcome::Mapped(span, n)
    }

    /// After buffers were retargeted (`crate::resident::retargeted`): each
    /// numbered copy that is a buffer no table hands out any more (in a
    /// resident replaced or gone, or a window in place a resident took
    /// over) gets the copy handed out now under a new number, and its pages
    /// in the directory name that one (the offset first, then the number).
    /// The old number is not used again: lists recorded with it may still
    /// run, and the old copy is held until they have (Stage 2 latency
    /// plan, B1: a numbered copy kept its resident for the whole run).
    fn retarget(&mut self) {
        let now = crate::resident::generation();
        if self.generation == now {
            return;
        }
        self.generation = now;
        for n in 0..self.copies.len() {
            let Some(old) = self.copies[n].clone() else { continue };
            let stale = old.in_resident().map_or_else(|| crate::resident::was_replaced(&old.buffer), |r| !crate::resident::registered(r));
            if !stale {
                continue;
            }
            let (first, len) = old.span();
            let Some(new) = crate::shadow::find(first, len.max(1)).filter(|s| !Arc::ptr_eq(s, &old) && s.offset_in(first).is_some() && s.left(first) >= len) else {
                continue;
            };
            let m = match self.copies.iter().position(|c| c.as_ref().is_some_and(|c| Arc::ptr_eq(c, &new))) {
                Some(m) => m,
                None if self.copies.len() < COPIES as usize => {
                    self.copies.push(Some(new.clone()));
                    self.copies.len() - 1
                }
                None => continue,
            };
            let mut at = first & !(PAGE - 1);
            while at < first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                // SAFETY: inside the directory's mapping.
                let entry = unsafe { self.directory.1.add((at / PAGE) as usize * 2) };
                // SAFETY: as above; the offset first.
                unsafe {
                    if entry.read() == n as u32 + 1 && new.offset_in(at).is_some() {
                        entry.add(1).write_volatile(entry_offset(&new, at));
                        entry.write_volatile(m as u32 + 1);
                    }
                }
                at += PAGE;
            }
            for (idx, _) in self.dirty_pages.values_mut() {
                if *idx == n {
                    *idx = m;
                }
            }
            self.copies[n] = None;
            crate::host::retire(old);
            crate::resident::copy_renumbered();
        }
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for buf in [self.faults.1, self.quarantine_faults.1] {
            for i in 1..MEMORY_FAULTS as usize {
                // SAFETY: inside the faults' mapping; the GPU may be writing,
                // a number missed now is written again next frame.
                let at = unsafe { buf.add(i) };
                // SAFETY: as above.
                let page = unsafe { at.read_volatile() };
                if page != 0 {
                    pages.push(page);
                    // SAFETY: as above.
                    unsafe { at.write_volatile(0) };
                }
            }
        }
        pages.sort_unstable();
        pages.dedup();
        let (
            mut mapped,
            mut already,
            mut refused_budget,
            mut refused_uncommitted,
            mut refused_non_gpu,
            mut refused_full,
        ) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        for &page in &pages {
            match self.map_detail(u64::from(page) * PAGE) {
                MapOutcome::Mapped(_, _) => mapped += 1,
                MapOutcome::AlreadyMapped(..) => already += 1,
                MapOutcome::RefusedBudget => refused_budget += 1,
                MapOutcome::RefusedUncommitted => refused_uncommitted += 1,
                MapOutcome::RefusedNonGpu => refused_non_gpu += 1,
                MapOutcome::RefusedFull => refused_full += 1,
                MapOutcome::RefusedOther => {}
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    mapped,
                    already,
                    refused_budget,
                    refused_uncommitted,
                    refused_non_gpu,
                    refused_full,
                    copies = self.copies.len(),
                    window_mb = self.window_bytes >> 20,
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
    }

    /// Root view / root descriptor address: maps its buffer or up to one
    /// adjacent 4 MiB loose window within the committed allocation (<= 8 MiB).
    fn name_root(&mut self, address: u64, mut named: Option<&mut Vec<usize>>) {
        let Some(((first, len), idx)) = self.map(address) else {
            return;
        };
        if let Some(out) = named.as_deref_mut() {
            out.push(idx);
        }
        if crate::shadow::extent(address).is_some() {
            return;
        }
        let next = first.saturating_add(len);
        if next.saturating_sub(first) <= (4 << 20)
            && let Some((_, alloc_end)) = self.extent(address)
            && next < alloc_end
            && let Some((_, next_idx)) = self.map(next)
            && let Some(out) = named.as_deref_mut()
        {
            out.push(next_idx);
        }
    }

    /// Heuristic constant-pair address: maps only the addressed buffer or
    /// single window, never walking adjacent windows.
    fn name_constant(&mut self, address: u64, named: &mut Vec<usize>) {
        if let Some((_, idx)) = self.map(address) {
            named.push(idx);
        }
    }
}

/// Merges sorted `(page, copy_idx)` entries into contiguous `(copy_idx, byte_start, byte_len)`
/// runs clamped to each copy's `[span_start, span_start + span_len)` bounds.
fn coalesce_dirty_runs(
    pages: impl IntoIterator<Item = (u32, usize)>,
    spans: &[(u64, u64)],
) -> Vec<(usize, u64, u64)> {
    let mut runs: Vec<(usize, u64, u64)> = Vec::new();
    for (page, copy_idx) in pages {
        let Some(&(copy_start, copy_len)) = spans.get(copy_idx) else {
            continue;
        };
        let copy_end = copy_start.saturating_add(copy_len);
        let p_start = (u64::from(page) * PAGE).max(copy_start);
        let p_end = u64::from(page + 1).saturating_mul(PAGE).min(copy_end);
        if p_start >= p_end {
            continue;
        }
        match runs.last_mut() {
            Some((last_idx, _, last_end)) if *last_idx == copy_idx && p_start <= *last_end => {
                *last_end = (*last_end).max(p_end);
            }
            _ => runs.push((copy_idx, p_start, p_end)),
        }
    }
    for (_, start, end) in &mut runs {
        *end -= *start;
    }
    runs
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &HostHeap,
    read_addresses: &[u64],
    write_addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.extents.clear();
        m.dirty_pages
            .retain(|_, (_, f)| frame.wrapping_sub(*f) <= MARK_FRAMES);
        m.take_faults();
    }
    if crate::resident::on() {
        m.retarget();
    }
    let mut named = Vec::new();
    for &address in read_addresses {
        m.name_root(address, None);
    }
    for &address in write_addresses {
        m.name_root(address, Some(&mut named));
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            let addr = u64::from(c) | u64::from(high) << 32;
            if crate::shadow::extent(addr).is_some() {
                m.name_constant(addr, &mut named);
            }
        }
        if c >= 0x0100_0000 {
            let addr = u64::from(c) << 8;
            if crate::shadow::extent(addr).is_some() {
                m.name_constant(addr, &mut named);
            }
        }
    }
    named.sort_unstable();
    named.dedup();
    let key = std::ptr::from_ref(heap) as usize;
    let (base, made) = match m.heaps.get(&key) {
        Some(&run) => run,
        None => {
            // Held for the heap's life: not given back with a list's slots.
            let base = heap.dynamic(COPIES + 1)?;
            heap.null_raw_uav_at(base);
            (base, 0)
        }
    };
    for (n, shadow) in m.copies.iter().enumerate().skip(made as usize) {
        match shadow {
            Some(shadow) => heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, view_bytes(shadow)),
            // Numbered again before this heap had it: no page names it.
            None => heap.null_raw_uav_at(base + 1 + n as u32),
        }
    }
    m.heaps.insert(key, (base, m.copies.len() as u32));
    // Each copy once per list, whichever list numbered it: the builder alone
    // dispatches hundreds of times a frame. The title records its lists on
    // several threads, so a copy can be numbered by another list (or by the
    // frame's fault notes) after this list's first dispatch: this list's
    // shaders reach it through the directory all the same, and a copy it
    // has not brought up to date, or not declared written, would be read as
    // it stands and its stores never looked for. Copies are only ever
    // added, so the ones a list has not taken yet are the last ones.
    const READ: usize = usize::MAX - 0x6d;
    const READ_COPY: usize = usize::MAX - 0x1_0000;
    const WRITE_COPY: usize = usize::MAX - 0x2_0000;
    const WRITE_PAGE: usize = usize::MAX - 0x400_0000;
    let first_read = list.first_use(READ);
    for (n, shadow) in m.copies.iter().enumerate().rev() {
        if !list.first_use(READ_COPY - n) {
            break;
        }
        let Some(shadow) = shadow else { continue };
        let (start, len) = shadow.span();
        shadow.refresh(list, start, len);
    }
    if writes {
        for &n in &named {
            if let Some(shadow) = m.copies.get(n).and_then(|s| s.as_ref())
                && list.first_use(WRITE_COPY - n)
            {
                let (start, len) = shadow.span();
                shadow.gpu_writes(list, start, len);
            }
        }
        let spans: Vec<(u64, u64)> = m.copies.iter().map(|s| s.as_ref().map_or((0, 0), |s| s.span())).collect();
        let dirty: Vec<(u32, usize)> = m
            .dirty_pages
            .iter()
            .filter_map(|(&page, &(idx, _))| {
                if named.binary_search(&idx).is_ok() {
                    return None;
                }
                list.first_use(WRITE_PAGE - page as usize)
                    .then_some((page, idx))
            })
            .collect();
        for (idx, run_start, run_len) in coalesce_dirty_runs(dirty, &spans) {
            if let Some(shadow) = &m.copies[idx] {
                shadow.gpu_writes(list, run_start, run_len);
            }
        }
        list.mark_uav_dirty();
        list.mark_memory_dirty();
    }
    if first_read {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
        list.keep(&m.quarantine_faults.0);
    }
    Some(Ready {
        directory: m.directory.2,
        faults: if writes { m.faults.2 } else { m.quarantine_faults.2 },
        base,
    })
}

/// Scans `bitmask` (`MEMORY_DIRTY_DWORDS` dwords), atomically clears non-zero
/// words, and appends every dirty `page` index to `out`.
///
/// # Safety
/// `bitmask` must point to at least `dwords` valid, 4-byte-aligned `u32` words.
unsafe fn drain_dirty_bits(bitmask: *mut u32, dwords: usize, out: &mut Vec<u32>) {
    use std::sync::atomic::{AtomicU32, Ordering};
    for i in 0..dwords {
        // SAFETY: caller guarantees `bitmask + i` is valid and 4-byte aligned.
        let ptr = unsafe { bitmask.add(i) };
        // SAFETY: volatile read avoids unneeded atomic RMW on zero words.
        if unsafe { ptr.read_volatile() } == 0 {
            continue;
        }
        // SAFETY: `ptr` is valid and 4-byte aligned.
        let mut word = unsafe { AtomicU32::from_ptr(ptr) }.swap(0, Ordering::AcqRel);
        let base = (i as u32) * 32;
        while word != 0 {
            let bit = word.trailing_zeros();
            word &= word - 1;
            out.push(base + bit);
        }
    }
}

/// Harvests dirty-page bits from `SLOT_FAULTS`'s bitmask tail and records each
/// dirty 64 KiB page so subsequent command lists declare only those dirty page
/// runs via `Shadow::gpu_writes`.
pub(crate) fn take_dirty() {
    let frame = crate::frame::current();
    let mut dirty_pages = Vec::new();
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_mut() else {
        return;
    };
    // SAFETY: `m.faults.1` has `MEMORY_FAULTS + MEMORY_DIRTY_DWORDS` dwords.
    unsafe {
        drain_dirty_bits(
            m.faults.1.add(MEMORY_FAULTS as usize),
            MEMORY_DIRTY_DWORDS as usize,
            &mut dirty_pages,
        );
    }
    for page in dirty_pages {
        if page >= MEMORY_PAGES - 1 {
            continue;
        }
        // SAFETY: inside `m.directory.1` (`MEMORY_PAGES * 2` dwords).
        let copy_num = unsafe { m.directory.1.add(page as usize * 2).read() };
        if copy_num != 0 {
            let idx = copy_num as usize - 1;
            if idx < m.copies.len() {
                m.dirty_pages.insert(page, (idx, frame));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{PAGE, coalesce_dirty_runs, drain_dirty_bits};

    #[test]
    fn dirty_bitmask_extracts_set_pages_and_resets_words() {
        let mut words = vec![0u32; 8];
        words[0] = (1 << 0) | (1 << 31);
        words[5] = 1 << 7;
        let mut pages = Vec::new();
        // SAFETY: `words` has 8 valid u32 elements.
        unsafe { drain_dirty_bits(words.as_mut_ptr(), words.len(), &mut pages) };
        assert_eq!(pages, vec![0, 31, 5 * 32 + 7]);
        assert!(words.iter().all(|&w| w == 0));
    }

    #[test]
    fn coalesce_dirty_runs_merges_adjacent_and_clamps_to_copy_spans() {
        let spans = [
            (0x1000_0000 + 0x4000, 3 * PAGE), // unaligned start/end inside 4 pages
            (0x2000_0000, 4 << 20),
        ];
        let b0 = (spans[0].0 / PAGE) as u32;
        let b1 = (spans[1].0 / PAGE) as u32;
        let runs = coalesce_dirty_runs(
            [
                (b0, 0),
                (b0 + 1, 0), // contiguous with b0, copy 0
                (b0 + 3, 0), // gap after b0+1, clamped end at spans[0] end
                (b1, 1),     // different copy index, not merged with copy 0
                (b1 + 2, 1), // non-adjacent in copy 1
                (b1 + 3, 1), // contiguous with b1+2 in copy 1
                (999, 99),   // out of bounds copy index ignored
            ],
            &spans,
        );
        assert_eq!(
            runs,
            vec![
                (0, spans[0].0, 2 * PAGE - 0x4000),
                (0, u64::from(b0 + 3) * PAGE, 0x4000),
                (1, u64::from(b1) * PAGE, PAGE),
                (1, u64::from(b1 + 2) * PAGE, 2 * PAGE),
            ]
        );
    }
}

#[cfg(test)]
mod inplace_tests {
    use super::{PAGE, entry_offset, view_bytes};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::core::Interface;
    use windows_sys::Win32::System::Memory as mem;

    /// Design §11, bug 1: a raw address reaches the title's byte in a copy
    /// in place, whose buffer begins `bias` bytes before the title memory;
    /// and the copy's view reaches its last byte. The software adapter
    /// reads what the directory's offset names.
    #[test]
    fn warp_raw_addresses_into_memory_in_place_take_the_bias() {
        let Some(device) = crate::inplace::tests::warp() else { return };
        if device.cast::<ID3D12Device13>().is_err() {
            return;
        }
        // SAFETY: an allocation of the test's own.
        unsafe {
            let base = mem::VirtualAlloc(std::ptr::null(), 8 << 16, mem::MEM_RESERVE | mem::MEM_COMMIT, mem::PAGE_READWRITE).cast::<u8>();
            let (start, len) = (base as u64 + 0x1230, 3 * PAGE + 0x40);
            for i in 0..len as usize {
                base.add(0x1230 + i).write((i as u8).wrapping_mul(13).wrapping_add(i as u8 >> 3));
            }
            let placed = crate::inplace::place_part(&device, start, len).unwrap();
            let bias = placed.bias;
            assert_eq!(bias, 0x1230);
            let shadow = crate::shadow::Shadow::test_in_place(placed, start, len);
            // The first whole directory page in the copy.
            let page = start.next_multiple_of(PAGE);
            let offset = entry_offset(&shadow, page);
            assert_eq!(u64::from(offset), page - start + bias, "the offset from the buffer's first byte");
            assert_eq!(view_bytes(&shadow), len + bias, "the view reaches the title memory's end");
            let mut rig = crate::inplace::Rig::new(&device).unwrap();
            let got = rig.fetch(&shadow.buffer, u64::from(offset), 4096).unwrap();
            assert_eq!(got, std::slice::from_raw_parts(page as *const u8, 4096), "the page's own bytes");
            let last = rig.fetch(&shadow.buffer, view_bytes(&shadow) - 16, 16).unwrap();
            assert_eq!(last, std::slice::from_raw_parts((start + len - 16) as *const u8, 16), "the last bytes are in the view");
            drop(shadow);
            mem::VirtualFree(base.cast(), 0, mem::MEM_RELEASE);
        }
    }
}