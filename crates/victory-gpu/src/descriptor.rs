//! Descriptor heaps: zeroed memory, with handles that are plain addresses
//! (CPU and GPU alike, as on the console).
//!
//! Descriptor sizes: views 32 bytes and samplers 16, the size of RDNA 2's
//! image and sampler descriptors, so a title that copies descriptors itself
//! moves whole ones. Render target and depth views are ours alone; 128 bytes
//! leaves room for whatever stride a title assumes.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use crate::object::{self, DESCRIPTOR_HEAP, E_INVALIDARG, HRESULT, Iid, Object, Table, This, guid};
use crate::resource::{Memory, PAGEABLE_IIDS};

#[cfg(windows)]
pub(crate) mod host;

const HEAP_IIDS: [Iid; 4] = [
    guid(
        0x8efb471d,
        0x616c,
        0x4f49,
        [0x90, 0xf7, 0x12, 0x7b, 0xb7, 0x63, 0xfa, 0x51],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct HeapDesc {
    kind: u32,
    count: u32,
    flags: u32,
    nodes: u32,
}

/// Newer titles (the title) pass the description followed by
/// memory of their own for the descriptors: an address and its size in
/// bytes, or a null address to have the heap allocate (inferred from the
/// title's call sites; older titles pass the plain description).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Placement {
    address: u64,
    size: u64,
}

#[repr(C)]
pub(crate) struct Heap {
    object: Object,
    desc: HeapDesc,
    /// The first descriptor: in `memory`, or in the title's own memory.
    pub(crate) start: u64,
    memory: Option<Memory>,
    /// Shader-visible view and sampler heaps get a host heap with the same
    /// descriptor order when a draw first uses them (decision 0007, step
    /// 3); slots written before wait in `pending`.
    #[cfg(windows)]
    host: std::sync::OnceLock<Option<host::HostHeap>>,
    #[cfg(windows)]
    pending: std::sync::Mutex<Vec<u32>>,
    /// Changed pages of a placed shader-visible heap. Some titles copy
    /// descriptor records into it as bytes, without a descriptor API call.
    #[cfg(windows)]
    written: Option<crate::watch::Tracked>,
}

pub(crate) fn heap<'a>(this: This) -> &'a Heap {
    // SAFETY: called from descriptor heap methods only.
    unsafe { &*this.cast::<Heap>() }
}

impl Heap {
    /// Whether this is a sampler heap.
    pub(crate) fn samplers(&self) -> bool {
        self.desc.kind == SAMPLERS
    }

    /// The host heap, made on first use from what the title wrote so far.
    /// Made late because a host view heap the size of the title's
    /// costs over 100 MB of commit, which its boot needs for its page pool.
    #[cfg(windows)]
    pub(crate) fn host(&self) -> Option<&host::HostHeap> {
        if let Some(h) = self.host.get() {
            return h.as_ref();
        }
        if self.desc.flags & SHADER_VISIBLE == 0 || !matches!(self.desc.kind, VIEWS_KIND | SAMPLERS)
        {
            return None;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // A recorder can take the fast path as soon as the OnceLock is
        // initialized. Publish only after replaying the queued writes.
        self.host
            .get_or_init(|| {
                let step = u64::from(increment(self.desc.kind)?);
                let h = host::HostHeap::new(self.samplers(), self.desc.count)?;
                for i in pending.drain(..) {
                    h.mirror(i, self.start + u64::from(i) * step, false);
                }
                Some(h)
            })
            .as_ref()
    }

    /// The handle of descriptor `index` of this heap.
    #[cfg(windows)]
    pub(crate) fn handle(&self, index: u32) -> Option<u64> {
        let step = u64::from(increment(self.desc.kind)?);
        (index < self.desc.count).then(|| self.start + u64::from(index) * step)
    }

    /// The descriptor index of `handle` in this heap, if it lies in it.
    pub(crate) fn index(&self, handle: u64) -> Option<u32> {
        let step = u64::from(increment(self.desc.kind)?);
        let at = handle.checked_sub(self.start)?;
        let index = at / step;
        (index < u64::from(self.desc.count)).then_some(index as u32)
    }
}

const VIEWS_KIND: u32 = 0;
const SAMPLERS: u32 = 1;
const SHADER_VISIBLE: u32 = 1;

/// Shader-visible heaps by address range, for mirroring descriptor writes:
/// start, end, heap.
///
/// Read by every lookup and by [`sync_heaps`], written only when a heap is
/// made or dropped. It was a mutex, and [`sync_heaps`] holds it for its
/// whole pass (15 ms a frame on the submitting thread): the profile of run
/// 225 has the title's six recording threads waiting 48.5 ms a frame in
/// [`shader_slot`] for it, each lookup itself a few nanoseconds. Readers no
/// longer wait for each other. Nothing a holder calls takes it again.
#[cfg(windows)]
static SHADER_HEAPS: std::sync::RwLock<Vec<(u64, u64, usize)>> = std::sync::RwLock::new(Vec::new());

/// One [`sync_heaps`] at a time, as when the list above was a mutex.
#[cfg(windows)]
static SYNCING: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// The shader-visible heap holding `handle`, and the descriptor's index.
#[cfg(windows)]
fn shader_slot<'a>(handle: u64) -> Option<(&'a Heap, u32)> {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    let &(_, _, at) = heaps
        .iter()
        .find(|&&(start, end, _)| (start..end).contains(&handle))?;
    // SAFETY: registered heaps are live (removed when dropped).
    let heap = unsafe { &*(at as *const Heap) };
    Some((heap, heap.index(handle)?))
}

/// Makes the host descriptor for what the title wrote at `handle`, if it
/// is in a shader-visible heap.
#[cfg(windows)]
fn mirror(handle: u64) {
    let Some((heap, index)) = shader_slot(handle) else {
        return;
    };
    let mut pending = heap.pending.lock().unwrap_or_else(|e| e.into_inner());
    match heap.host.get() {
        None => pending.push(index),
        Some(host) => {
            drop(pending);
            if let Some(host) = host {
                host.mirror(index, handle, false);
            }
        }
    }
}

/// Mirrors records copied directly into placed shader-visible heaps. Calls
/// to `Create*View` are already mirrored by [`mirror`]; raw CPU copies leave
/// only write-watch pages to reveal which slots changed.
///
/// Upstream VIctory 86a5ed6 does the same and counts 1,300 to 14,400 such
/// slots a frame in the world of the title this fork works on, against a
/// few hundred views made by calls; without it those host slots stayed
/// null, or went on showing what they showed before.
/// `VICTORY_HEAP_SYNC=0`: not done, as before. [`SYNCED`] counts.
#[cfg(windows)]
pub(crate) fn sync_heaps() {
    sync_heaps_avoiding(None, true);
}

/// [`sync_heaps`] before a title-read wait (`crate::host`'s
/// `VICTORY_SYNC_IN_WAIT`): pages of the heaps that touch `avoid` (title
/// memory flights in the air may write, in order, joined) are not looked
/// at now but kept in [`DEFERRED`] for the next sync (the one after the
/// wait), which looks at them first. Every other page is synced now.
#[cfg(windows)]
pub(crate) fn sync_heaps_early(avoid: &[(usize, usize)]) {
    sync_heaps_avoiding(Some(avoid), false);
}

/// [`sync_heaps_early`] for the sync that runs all along beside the
/// submissions (`crate::host`'s `VICTORY_SYNC_OVERLAP=2`): the pages under
/// flights' writes are left for the next sync as there; the records left
/// to tables' draws that are due are mirrored too, in the heaps no flight's
/// declared writes touch (the sync before a submission mirrors them with
/// flights in the air as well; those in other heaps wait for it).
#[cfg(windows)]
pub(crate) fn sync_heaps_beside(avoid: &[(usize, usize)]) {
    sync_heaps_avoiding(Some(avoid), true);
}

/// Pages (heap, from, to) a sync left for the next one ([`sync_heaps_early`]).
#[cfg(windows)]
static DEFERRED: std::sync::Mutex<Vec<(usize, usize, usize)>> = std::sync::Mutex::new(Vec::new());

/// Since the last frame line: pages synced by [`sync_heaps_early`], and
/// pages it left for the sync after the wait.
#[cfg(windows)]
pub(crate) static EARLY: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

#[cfg(windows)]
fn sync_heaps_avoiding(avoid: Option<&[(usize, usize)]>, due_left: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC").is_none_or(|v| v != "0")) {
        return;
    }
    // Whether `from..to` lies under flights' writes (`avoid`).
    let under = |from: usize, to: usize| {
        avoid.is_some_and(|ranges| {
            let i = ranges.partition_point(|r| r.1 <= from);
            ranges.get(i).is_some_and(|r| r.0 < to)
        })
    };
    let (mut early, mut deferred) = (0u64, 0u64);
    let started = std::time::Instant::now();
    host::SYNC_SEQ.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    let (mut pages, mut looked, mut mirrored) = (0u64, 0u64, 0u64);
    let mut page_bytes: Vec<u8> = Vec::new();
    // With `sync_wide`: the pages read, each looked at by one of several
    // threads afterwards.
    let wide = sync_wide() > 1;
    let mut work: Vec<SyncPage> = Vec::new();
    let _one = SYNCING.lock().unwrap_or_else(|e| e.into_inner());
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    // The pages an earlier sync left (those of heaps still registered).
    let left: Vec<(usize, usize, usize)> = std::mem::take(&mut *DEFERRED.lock().unwrap_or_else(|e| e.into_inner()));
    let mut keep: Vec<(usize, usize, usize)> = Vec::new();
    // The pre-read ([`sync_heaps_preread`], `VICTORY_SYNC_OVERLAP=3`): the
    // bytes and prints of pages it read, by (heap, page). The sync takes
    // and reads its pages as before; a page whose bytes are the ones the
    // pre-read read takes its prints from there. Used ones are dropped.
    let mut pre: std::collections::HashMap<(usize, usize), PreRead> = {
        let mut stash = PREREAD.lock().unwrap_or_else(|e| e.into_inner());
        if stash.is_empty() {
            std::collections::HashMap::new()
        } else {
            std::mem::take(&mut *stash).into_iter().map(|r| ((r.at, r.from / crate::watch::PAGE), r)).collect()
        }
    };
    let (mut pre_used, mut pre_stale) = (0u64, 0u64);
    for &(start, end, at) in heaps.iter() {
        // SAFETY: registered heaps are live (removed when dropped).
        let heap = unsafe { &*(at as *const Heap) };
        let (Some(Some(host)), Some(written), Some(step)) =
            (heap.host.get(), &heap.written, increment(heap.desc.kind))
        else {
            continue;
        };
        let step = u64::from(step);
        // Capped host heaps have no counterparts for the title slots reserved
        // for per-draw tables; raw writes there must not be mirrored.
        let mirrored_count = heap.desc.count.min(host.count());
        // Pages left before first (each lies in one page), then the
        // pages written since.
        let mut pieces: Vec<(usize, usize)> = left.iter().filter(|l| l.0 == at).map(|l| (l.1, l.2)).collect();
        for (range_from, range_end) in written.take(start as usize, end as usize) {
            let mut page = range_from & !(crate::watch::PAGE - 1);
            while page < range_end {
                pieces.push((page.max(range_from), (page + crate::watch::PAGE).min(range_end)));
                page += crate::watch::PAGE;
            }
        }
        for (from, to) in pieces {
            {
                if under(from, to) {
                    keep.push((at, from, to));
                    deferred += 1;
                    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
                    if !SAID.swap(true, Relaxed) {
                        tracing::info!(
                            heap = format_args!("{start:#x}..{end:#x}"),
                            page = format_args!("{from:#x}..{to:#x}"),
                            "a page of a shader-visible heap lies under flights' declared shader writes: its sync waits for the \
                             one after the title-read wait (VICTORY_SYNC_IN_WAIT=1; said once)"
                        );
                    }
                    continue;
                }
                if avoid.is_some() {
                    early += 1;
                }
                // The page's records are read through the kernel, into a
                // copy: a page the title unmaps meanwhile fails the read
                // where 2,048 reads in place would fault.
                if wide {
                    let mut bytes = vec![0u8; to - from];
                    if crate::host::read_into(from as *const u8, &mut bytes) {
                        pages += 1;
                        work.push(SyncPage {
                            host: std::ptr::from_ref(host) as usize,
                            start,
                            step,
                            mirrored_count,
                            from,
                            to,
                            bytes,
                        });
                    }
                    continue;
                }
                page_bytes.resize(to - from, 0);
                let reading = std::time::Instant::now();
                let read = crate::host::read_into(from as *const u8, &mut page_bytes);
                host::SYNC_STATS[host::S_READ_US].fetch_add(reading.elapsed().as_micros() as u64, Relaxed);
                host::SYNC_STATS[host::S_READ_BYTES].fetch_add((to - from) as u64, Relaxed);
                if read {
                    pages += 1;
                    // The pre-read's prints, if it read these very bytes.
                    let prints = pre.remove(&(at, from / crate::watch::PAGE)).and_then(|r| {
                        let same = r.from == from && r.to == to && r.bytes == page_bytes;
                        if same {
                            pre_used += 1;
                        } else {
                            pre_stale += 1;
                        }
                        same.then_some(r.prints)
                    });
                    let (l, m) = sync_page(host, start, step, mirrored_count, from, to, &page_bytes, prints.as_deref());
                    looked += l;
                    mirrored += m;
                }
            }
        }
        // Records left to tables' draws that none made (not before a
        // title-read wait: flights in the air may write the heap).
        if avoid.is_none() || (due_left && !under(start as usize, end as usize)) {
            mirror_left(host, start, step);
        }
    }
    // What was left and is still under flights' writes waits for the next
    // sync; a page of a heap that went is dropped with it.
    if !keep.is_empty() {
        DEFERRED.lock().unwrap_or_else(|e| e.into_inner()).extend(keep);
    }
    // Pre-read pages this sync did not take: after a sync before a title-
    // read wait (which leaves pages under flights' writes) they wait for
    // the next; after the one before a submission they are dropped (a page
    // still open is read again by the next pass), so none waits for ever.
    // Those of heaps that went are dropped. A newer read by a pass stands.
    if avoid.is_some() && !pre.is_empty() {
        let mut stash = PREREAD.lock().unwrap_or_else(|e| e.into_inner());
        for (key, read) in pre {
            if heaps.iter().any(|h| h.2 == read.at) && !stash.iter().any(|r| (r.at, r.from / crate::watch::PAGE) == key) {
                stash.push(read);
            }
        }
    }
    let s = &host::SYNC_STATS;
    s[host::S_PRE_USED].fetch_add(pre_used, Relaxed);
    s[host::S_PRE_STALE].fetch_add(pre_stale, Relaxed);
    EARLY[0].fetch_add(early, Relaxed);
    EARLY[1].fetch_add(deferred, Relaxed);
    if !work.is_empty() {
        let (l, m) = mirror_wide(work);
        looked += l;
        mirrored += m;
    }
    drop(heaps);
    for (count, n) in SYNCED.iter().zip([pages, looked, mirrored, started.elapsed().as_micros() as u64]) {
        count.fetch_add(n, Relaxed);
    }
}

/// The records of `host`'s heap (first handle `start`, `step` bytes a
/// record) that [`sync_heaps`] left to tables' draws and are due, that no
/// draw made meanwhile (`HostHeap::take_left`): each read again, through
/// the kernel, and mirrored if its slot still holds that record (one that
/// changed since is on a written page again, for the next sync).
#[cfg(windows)]
fn mirror_left(host: &host::HostHeap, start: u64, step: u64) {
    use std::sync::atomic::Ordering::Relaxed;
    let due = host.take_left(crate::frame::current());
    if due.is_empty() {
        return;
    }
    let began = std::time::Instant::now();
    let (mut mirrored, mut gone) = (0u64, 0u64);
    let mut bytes: Vec<u8> = Vec::new();
    let mut n = 0;
    while n < due.len() {
        // A run of records within 4 KiB of the first, read at once.
        let first = start + u64::from(due[n].0) * step;
        let mut end = n + 1;
        while end < due.len() && start + u64::from(due[end].0) * step + step - first <= 4096 {
            end += 1;
        }
        let last = start + u64::from(due[end - 1].0) * step + step;
        bytes.resize((last - first) as usize, 0);
        let read = crate::host::read_into(first as *const u8, &mut bytes);
        for &(index, key) in &due[n..end] {
            let handle = start + u64::from(index) * step;
            let at = (handle - first) as usize;
            if read && host::HostHeap::record_key(&bytes[at..at + step as usize]) == key {
                host.mirror_left(index, handle);
                mirrored += 1;
            } else {
                gone += 1;
            }
        }
        n = end;
    }
    let s = &host::SYNC_STATS;
    s[host::S_LEFT_MIRRORED].fetch_add(mirrored, Relaxed);
    s[host::S_LEFT_GONE].fetch_add(gone, Relaxed);
    s[host::S_LEFT_US].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
}

/// How many jobs [`sync_heaps`] splits the records it looks at into, run on
/// the host's kept threads (`crate::shadow::wide_run`) and the calling one
/// (`VICTORY_SYNC_WIDE=N`, N of 2 to 16; `1` is 8; off by default: 0).
///
/// The sync runs before every submission on the thread that submits, and
/// with the title's heaps at 250,000-280,000 records looked at and 37,000-
/// 42,000 mirrored a frame it was 29 ms a frame there (run n15), with the
/// card standing idle behind it after a title-read wait. The records are
/// gathered under the heaps' lock as before; each job mirrors a run of
/// them (`HostHeap::mirror_copied_many`; its `HostHeap::mirror` title
/// threads already call beside the sync: host slots are written
/// free-threaded, each slot by one job here); the lock is held until every job is done, so no heap goes
/// meanwhile. What a slot ends up with is what the sync gave it before:
/// each slot is looked at once per sync, in whichever order.
#[cfg(windows)]
pub(crate) fn sync_wide() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| match std::env::var("VICTORY_SYNC_WIDE").ok().and_then(|v| v.parse::<usize>().ok()) {
        Some(1) => 8,
        Some(n) => n.min(16),
        None => 0,
    })
}

/// A page of a shader-visible heap read for [`sync_heaps`] with
/// [`sync_wide`]: the host heap (a pointer: the caller holds
/// [`SHADER_HEAPS`] until the pages are done), the heap's first handle,
/// its record size and the slots it mirrors, the page's part of a written
/// range, and its bytes.
#[cfg(windows)]
struct SyncPage {
    host: usize,
    start: u64,
    step: u64,
    mirrored_count: u32,
    from: usize,
    to: usize,
    bytes: Vec<u8>,
}

/// The records of one page of a shader-visible heap, `bytes` being title
/// memory `from..to` read: each slot whose record starts there is mirrored
/// if it is not the one last seen there (`HostHeap::mirror_copied_many`).
/// Records looked at and mirrored.
/// `prints`: the records' fingerprints taken by the pre-read from these
/// same bytes ([`sync_heaps_preread`]), one per record in order.
#[cfg(windows)]
#[allow(clippy::too_many_arguments)]
fn sync_page(
    host: &host::HostHeap,
    start: u64,
    step: u64,
    mirrored_count: u32,
    from: usize,
    to: usize,
    bytes: &[u8],
    prints: Option<&[u64]>,
) -> (u64, u64) {
    let records = page_records(start, step, mirrored_count, from, to, bytes);
    // Under one take of the heap's lock of last-seen records, not one per
    // record (with `sync_wide` the jobs met there).
    // Upstream's `VICTORY_HEAP_SYNC_PAGE=0`: one record at a time.
    if !page_at_once() {
        let made = records
            .iter()
            .enumerate()
            .map(|(n, r)| host.mirror_copied_many(std::slice::from_ref(r), prints.and_then(|p| p.get(n..n + 1))))
            .sum();
        return (records.len() as u64, made);
    }
    (records.len() as u64, host.mirror_copied_many(&records, prints))
}

/// The records (title slot, handle, bytes) of the slots whose record starts
/// in `from..to` of a heap (first handle `start`, `step` bytes a record,
/// `mirrored_count` slots mirrored), `bytes` being that range as read.
#[cfg(windows)]
fn page_records(start: u64, step: u64, mirrored_count: u32, from: usize, to: usize, bytes: &[u8]) -> Vec<(u32, u64, &[u8])> {
    let mut records: Vec<(u32, u64, &[u8])> = Vec::new();
    for index in changed_slots(from, to, start as usize, step, mirrored_count) {
        let handle = start + index * step;
        // A record that straddles the range's ends is left to the range
        // that holds its first byte.
        let Some(at) = (handle as usize).checked_sub(from) else {
            continue;
        };
        let Some(record) = bytes.get(at..at + step as usize) else {
            continue;
        };
        records.push((index as u32, handle, record));
    }
    records
}

/// A page of a shader-visible heap read by the pre-read
/// ([`sync_heaps_preread`]) for the sync that takes it next: the heap (its
/// [`Heap`]'s address), the page's part of the heap, its bytes, and the
/// fingerprints of the records starting there, in order ([`page_records`]).
#[cfg(windows)]
struct PreRead {
    at: usize,
    from: usize,
    to: usize,
    bytes: Vec<u8>,
    prints: Vec<u64>,
}

/// The pages the pre-read read that no sync took since (one entry per
/// page).
#[cfg(windows)]
static PREREAD: std::sync::Mutex<Vec<PreRead>> = std::sync::Mutex::new(Vec::new());

/// The heap sync's reading on a thread of its own (`crate::host`'s
/// `VICTORY_SYNC_OVERLAP=3`; [`sync_heaps`] is otherwise unchanged): reads
/// the heap pages written since they were last taken, as far as that can
/// be told without taking them (`crate::watch::peek_written`), and
/// fingerprints their records, once per page until a sync takes it
/// ([`PREREAD`]). The sync before a submission takes and reads its pages
/// as ever; where a page's bytes are the ones read here it takes the
/// prints from here instead of fingerprinting its records again. True if
/// it read pages.
///
/// It takes no page: run n52 took them here every 2 ms, and a take guards
/// the heap's written blocks again (the title's heap is view memory the
/// guards hold), so the title's next write to each faulted again, its
/// handler waiting on the guards' lock the take held: the title's own
/// time went from 20-24 to 104 ms a frame, and the passes took 246 pages
/// a frame in 37.6 ms where the sync alone reads 130-160 in about 1.
/// It makes no host slot either and takes none of the locks title threads
/// take for views, nor the one-sync lock: what it leaves is checked
/// byte for byte by the sync.
///
/// Why it only reads: `VICTORY_SYNC_OVERLAP=2` mirrored on its own thread
/// whenever it ran, and the picture garbled (run n45, a band across the
/// bottom of the picture over two minutes); `=1`, which runs the very
/// sync of a submission beside its preparing, garbled too (run n42,
/// blocks at the foot of the window), where the sync before the preparing
/// never did. Making host slots and the sets beside them (texture queue,
/// placed spans that cut textures' own bytes, buffer slots, strides,
/// waiting slots) while a submission's steps run, between one step and the
/// next, is what both did and the sync before the preparing never does.
#[cfg(windows)]
pub(crate) fn sync_heaps_preread() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC").is_none_or(|v| v != "0")) {
        return false;
    }
    // At most this many pages wait for a sync (some 80 KiB each with their
    // prints; a frame of the world writes 130-280).
    const MOST: usize = 512;
    let began = std::time::Instant::now();
    // The pages read already (each once until a sync takes it).
    let known: std::collections::HashSet<(usize, usize)> = {
        let stash = PREREAD.lock().unwrap_or_else(|e| e.into_inner());
        if stash.len() >= MOST {
            return false;
        }
        stash.iter().map(|r| (r.at, r.from / crate::watch::PAGE)).collect()
    };
    let mut found: Vec<PreRead> = Vec::new();
    let mut n = 0u64;
    {
        let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
        for &(start, end, at) in heaps.iter() {
            // SAFETY: registered heaps are live (removed when dropped).
            let heap = unsafe { &*(at as *const Heap) };
            let (Some(Some(host)), Some(_), Some(step)) = (heap.host.get(), &heap.written, increment(heap.desc.kind))
            else {
                continue;
            };
            let step = u64::from(step);
            let mirrored_count = heap.desc.count.min(host.count());
            for page in crate::watch::peek_written(start as usize, end as usize) {
                if known.len() + found.len() >= MOST {
                    break;
                }
                if known.contains(&(at, page)) {
                    continue;
                }
                let from = (page * crate::watch::PAGE).max(start as usize);
                let to = ((page + 1) * crate::watch::PAGE).min(end as usize);
                if from >= to {
                    continue;
                }
                let mut bytes = vec![0u8; to - from];
                if !crate::host::read_into(from as *const u8, &mut bytes) {
                    continue;
                }
                let prints = host::record_prints(&page_records(start, step, mirrored_count, from, to, &bytes));
                found.push(PreRead { at, from, to, bytes, prints });
            }
        }
        // Under the heaps' list still: a heap that goes takes its pages
        // with it ([`Heap`]'s drop), after these are in.
        if !found.is_empty() {
            let mut stash = PREREAD.lock().unwrap_or_else(|e| e.into_inner());
            for read in found.drain(..) {
                let key = (read.at, read.from / crate::watch::PAGE);
                if !stash.iter().any(|r| (r.at, r.from / crate::watch::PAGE) == key) {
                    stash.push(read);
                    n += 1;
                }
            }
        }
    }
    let s = &host::SYNC_STATS;
    s[host::S_PRE_PAGES].fetch_add(n, Relaxed);
    s[host::S_PRE_US].fetch_add(began.elapsed().as_micros() as u64, Relaxed);
    n > 0
}

/// [`sync_wide`]: the pages read, looked at in [`sync_wide`] jobs (each a
/// run of pages; a slot lies in one page). Records looked at and
/// mirrored. The caller holds [`SHADER_HEAPS`] until this returns.
#[cfg(windows)]
fn mirror_wide(work: Vec<SyncPage>) -> (u64, u64) {
    use std::sync::atomic::Ordering::Relaxed;
    // Host heaps are used from several threads at once already (title
    // threads mirror the views they make, draws the slots they read).
    fn shared<T: Sync>() {}
    shared::<host::HostHeap>();
    let look = |page: &SyncPage| {
        // SAFETY: a host heap of a registered (live) heap, under the
        // heaps' lock the caller holds until every job is done.
        let host = unsafe { &*(page.host as *const host::HostHeap) };
        sync_page(host, page.start, page.step, page.mirrored_count, page.from, page.to, &page.bytes, None)
    };
    let jobs_n = sync_wide().max(1);
    // A few pages are not worth the threads.
    if work.len() < 4 || jobs_n < 2 {
        return work.iter().map(look).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
    }
    let counts = std::sync::Arc::new([std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)]);
    let each = work.len().div_ceil(jobs_n);
    let mut pages = work.into_iter();
    let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::with_capacity(jobs_n);
    loop {
        let chunk: Vec<SyncPage> = pages.by_ref().take(each).collect();
        if chunk.is_empty() {
            break;
        }
        let counts = counts.clone();
        jobs.push(Box::new(move || {
            let (looked, mirrored) = chunk.iter().map(look).fold((0, 0), |a, b| (a.0 + b.0, a.1 + b.1));
            counts[0].fetch_add(looked, Relaxed);
            counts[1].fetch_add(mirrored, Relaxed);
        }));
    }
    crate::shadow::wide_run(jobs);
    (counts[0].load(Relaxed), counts[1].load(Relaxed))
}

/// Since the last frame line, of [`sync_heaps`]: changed pages of placed
/// shader-visible heaps looked at, slots on them, slots whose record was
/// not the one last seen (mirrored), and the microseconds it took.
#[cfg(windows)]
pub(crate) static SYNCED: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// With the frame line: what [`sync_heaps`] did per frame over `frames`.
/// Nothing while no heap page changed.
#[cfg(windows)]
pub(crate) fn sync_line(frames: f64) {
    let [pages, looked, mirrored, micros] = [0, 1, 2, 3].map(|i| SYNCED[i].swap(0, std::sync::atomic::Ordering::Relaxed));
    if pages == 0 {
        return;
    }
    tracing::info!(
        "descriptor records the title copied into its heap as bytes (VICTORY_HEAP_SYNC=0: not looked for): per frame \
         {:.1} changed heap pages, {:.0} slots looked at, {:.1} whose record was new and was mirrored, {:.0} left to the \n         draws that read them through tables (VICTORY_HEAP_SYNC_TABLES=1 mirrors those too), {:.2} ms",
        pages as f64 / frames,
        looked as f64 / frames,
        mirrored as f64 / frames,
        host::COPIED_TABLED.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames,
        micros as f64 / 1000.0 / frames
    );
    let s: Vec<f64> = host::SYNC_STATS
        .iter()
        .map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames)
        .collect();
    let ms = |i: usize| s[i] / 1000.0;
    let (c, o) = (host::S_CLASS_N, host::S_OUTCOME);
    let u = host::S_CLASS_US;
    tracing::info!(
        "the heap sync's parts per frame: reading {:.1} pages' bytes ({:.2} MB) {:.2} ms; fingerprinting and sorting {:.2} ms: {:.0} records unchanged, {:.0} changed but their slot made from them already (passed over: VICTORY_SYNC_MADE_SKIP {}), {:.0} not ours in a slot made from none (passed over: VICTORY_SYNC_NONE_SKIP {}), {:.0} left to tables' draws (VICTORY_HEAP_SYNC_TABLES={}); of the records it would leave, {:.0} a draw read through a fresh table since the sync before (VICTORY_SYNC_DRAWN_BEFORE={}: 0 leaves them, 1 mirrors them now, 2 passes over the constant buffer views of slots made from nothing, 3 placed views too: {:.0} passed over); mirrored: {:.0} table entries {:.2} ms, {:.0} constant buffer views {:.2} ms, {:.0} placed views {:.2} ms, {:.0} not ours {:.2} ms; the mirror found {:.0} the same, made {:.0}, made {:.0} null for a draw or a pass, kept {:.0} older views, {:.0} no view; of those left to draws and due, {:.0} made by a draw, {:.0} mirrored after all, {:.0} changed again, {:.2} ms; root-indexed table entries outside fresh tables {:.0} (VICTORY_TABLE_INDEXED=1 makes them at their draw: {})",
        s[host::S_READ_BYTES] / crate::watch::PAGE as f64,
        s[host::S_READ_BYTES] / 1e6,
        ms(host::S_READ_US),
        ms(host::S_PRINT_US),
        s[host::S_UNCHANGED],
        s[host::S_SAME_MADE],
        if host::made_skip_on() { "on" } else { "off" },
        s[host::S_NONE_SKIPPED],
        if host::none_skip() { "on" } else { "off" },
        s[host::S_DEFERRED],
        host::sync_tables(),
        s[host::S_DRAWN_BEFORE],
        host::drawn_before(),
        s[host::S_DRAWN_PASSED],
        s[c],
        ms(u),
        s[c + 1],
        ms(u + 1),
        s[c + 2],
        ms(u + 2),
        s[c + 3],
        ms(u + 3),
        s[o],
        s[o + 1],
        s[o + 2],
        s[o + 3],
        s[o + 4],
        s[host::S_LEFT_DRAWN],
        s[host::S_LEFT_MIRRORED],
        s[host::S_LEFT_GONE],
        ms(host::S_LEFT_US),
        crate::draw::TABLE_INDEXED.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames,
        if crate::draw::table_indexed() { "on" } else { "off" },
    );
    tracing::info!(
        "the heap sync's null and pre-read parts per frame: constant buffer views whose mirror would only write the null their host slot holds, passed over (VICTORY_SYNC_NULL_SKIP {}): {:.0} changed, {:.0} left to draws and due; pre-read (VICTORY_SYNC_OVERLAP=3, takes nothing) {:.1} pages read beside in {:.2} ms; the sync took the prints of {:.1} pages from it (bytes the same), {:.1} pre-read pages had changed",
        if host::null_skip() { "on" } else { "off" },
        s[host::S_NULL_SKIPPED],
        s[host::S_LEFT_NULL_SKIPPED],
        s[host::S_PRE_PAGES],
        ms(host::S_PRE_US),
        s[host::S_PRE_USED],
        s[host::S_PRE_STALE],
    );
}

/// Whether [`sync_heaps`] hands a changed page's records to the host heap
/// in one go (`host::HostHeap::mirror_copied_many`, this fork's form of
/// upstream's `mirror_copied_page`). On by default: the
/// same records are mirrored either way. `VICTORY_HEAP_SYNC_PAGE=0`: one
/// by one, as before.
#[cfg(windows)]
fn page_at_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SYNC_PAGE").is_none_or(|v| v != "0"))
}

/// Title descriptor slots in a changed byte range that have host counterparts.
/// `range_end` is exclusive.
#[cfg(windows)]
fn changed_slots(
    range_from: usize,
    range_end: usize,
    heap_start: usize,
    step: u64,
    mirrored_count: u32,
) -> std::ops::Range<u64> {
    let from = (range_from - heap_start) as u64;
    let to = (range_end - heap_start) as u64;
    let first = from / step;
    let last = to.div_ceil(step).min(u64::from(mirrored_count));
    first.min(last)..last
}

/// Gives back dynamic slots of the host heap behind the title heap at
/// `heap`, if that heap still exists.
#[cfg(windows)]
pub(crate) fn release_dynamic(heap: usize, first: u32, n: u32) {
    let heaps = SHADER_HEAPS.read().unwrap_or_else(|e| e.into_inner());
    if heaps.iter().any(|&(_, _, h)| h == heap)
        // SAFETY: registered heaps are live (removed when dropped).
        && let Some(Some(host)) = unsafe { &*(heap as *const Heap) }.host.get()
    {
        host.release(first, n);
    }
}

#[cfg(windows)]
impl Drop for Heap {
    fn drop(&mut self) {
        let at = std::ptr::from_ref::<Heap>(self) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.retain(|&(_, _, h)| h != at);
        // Its pre-read pages go with it (a heap made later at the same
        // address must not take them).
        PREREAD.lock().unwrap_or_else(|e| e.into_inner()).retain(|r| r.at != at);
    }
}

/// Bytes per descriptor, by heap type (views, samplers, render targets,
/// depth).
fn increment(kind: u32) -> Option<u32> {
    [32, 16, 128, 128].get(kind as usize).copied()
}

/// Device method.
pub(crate) extern "system" fn increment_size(_: This, kind: u32) -> u32 {
    increment(kind).unwrap_or(0)
}

/// What our render target views hold: the resource and the view
/// description, turned into a host view when a command uses them.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct RtvRecord {
    tag: u32,
    described: u32,
    pub(crate) resource: This,
    desc: [u32; 6],
}
const RTV_TAG: u32 = u32::from_le_bytes(*b"vRTV");
const _: () = assert!(size_of::<RtvRecord>() <= 112);

impl RtvRecord {
    /// The view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_RENDER_TARGET_VIEW_DESC> {
        const TEXTURE_2D: u32 = 4;
        self.host_desc(TEXTURE_2D)
    }

    /// The depth view description in the host's type (the same layout).
    #[cfg(windows)]
    pub(crate) fn depth_desc(
        &self,
    ) -> Option<windows::Win32::Graphics::Direct3D12::D3D12_DEPTH_STENCIL_VIEW_DESC> {
        const TEXTURE_2D: u32 = 3;
        self.host_desc(TEXTURE_2D)
    }

    /// The description words as a host type, the format as the host has it.
    /// Without one, a typeless host copy gets a 2D view (`dimension`) in
    /// the title's format, which the host needs spelled out.
    #[cfg(windows)]
    fn host_desc<T>(&self, dimension: u32) -> Option<T> {
        assert!(size_of::<T>() <= size_of::<[u32; 6]>());
        let mut words = if self.described != 0 {
            self.desc
        } else {
            [
                crate::resource::view_format(self.resource)?,
                dimension,
                0,
                0,
                0,
                0,
            ]
        };
        // A target or depth view: of a target, in its stand-in format.
        words[0] = crate::resource::target_format(crate::resource::pc_format(words[0]));
        // SAFETY: both host view descriptions are 24 bytes in this layout.
        Some(unsafe { (&raw const words).cast::<T>().read_unaligned() })
    }
}

/// Device method: records the view at `handle` (descriptor memory).
pub(crate) extern "system" fn create_rtv(
    _: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    if handle == 0 {
        return;
    }
    // SAFETY: the title passes a description or null.
    let described = unsafe { desc.as_ref() };
    let record = RtvRecord {
        tag: RTV_TAG,
        described: u32::from(described.is_some()),
        resource,
        desc: described.copied().unwrap_or_default(),
    };
    // SAFETY: the handle is a descriptor slot, at least 112 bytes.
    unsafe { (handle as *mut RtvRecord).write_unaligned(record) };
}

/// Device method: a depth view, recorded as render target views are.
pub(crate) extern "system" fn create_dsv(
    this: This,
    resource: This,
    desc: *const [u32; 6],
    handle: u64,
) {
    create_rtv(this, resource, desc, handle);
    if handle != 0 {
        // SAFETY: just written by `create_rtv`.
        unsafe { (handle as *mut u32).write_unaligned(DSV_TAG) };
    }
}
const DSV_TAG: u32 = u32::from_le_bytes(*b"vDSV");

/// A shader view or sampler as the title described it (descriptions in the
/// host's layout, kept as words), for the host to build its own from.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum View {
    /// Constant buffer: GPU address and size.
    Constant(u64, u32),
    /// Shader resource: resource, description (none: the resource's own).
    Shader(usize, Option<[u32; 10]>),
    /// Unordered access: resource, counter resource, description.
    Unordered(usize, usize, Option<[u32; 10]>),
    Sampler([u32; 13]),
    /// The console's own sampler description: flags, filters per axis,
    /// addressing, LOD and anisotropy settings, border color, then
    /// console-only tuning words.
    ConsoleSampler([u32; 27]),
    /// A view of memory without a resource object: whether it is unordered
    /// access, the resource description, then the view's (the plain one
    /// followed by the memory address and console fields).
    Placed(bool, [u32; 14], [u32; 16]),
}

impl View {
    /// Whether the view reads buffer memory (constant buffers, buffer views
    /// of resources or of memory) or memory the title placed a texture in:
    /// such views get host slots only for draws.
    #[cfg(windows)]
    fn is_buffer(&self) -> bool {
        match self {
            View::Constant(..) | View::Placed(..) => true,
            View::Shader(r, _) | View::Unordered(r, _, _) => {
                crate::resource::buffer_bytes(*r as This).is_some()
            }
            View::Sampler(_) | View::ConsoleSampler(_) => false,
        }
    }
}

/// Every distinct view in use; a descriptor holds a tag, an index into it
/// and the entry's generation, which survive titles copying descriptors as
/// bytes. A view of a resource lives as long as the resource: when the
/// title releases it, its entries are freed for reuse under a new
/// generation, so descriptors (and host slots made from them) naming the
/// old one go stale instead of naming whatever is made at the same address
/// next.
///
/// Views without a resource object have nothing that releases them, and a
/// title copies descriptors as bytes, so nothing says when the last
/// descriptor naming an entry is gone: such entries were kept for the run.
/// The title writes some 14,000 constant buffer views a frame,
/// each at the address its constants have that frame: 2,450,139 entries
/// after four minutes of play, 1,047 MB with the index, and 11,000 more
/// every second (run 65). So the two kinds that carry an address are not
/// entries any more: their records name them whole ([`constant_words`],
/// [`placed_words`]), and the table holds only what is bounded: views of
/// resource objects (freed with the resource), samplers (as many as there
/// are sampler states), and the shapes of placed views (their descriptions
/// without the address: as many as the title has kinds of textures and
/// buffers, however often it moves them).
/// `VICTORY_VIEW_INLINE=0`: every view is an entry, as before.
struct Views {
    /// Each entry's view and generation.
    list: Vec<(View, u32)>,
    index: std::collections::BTreeMap<View, u32>,
    /// Entries naming each resource.
    by_resource: std::collections::BTreeMap<usize, Vec<u32>>,
    free: Vec<u32>,
    /// The shapes of views of placed memory, and where each is.
    shapes: Vec<Shape>,
    shape_index: std::collections::BTreeMap<Shape, u32>,
}

/// Read far more often than written: every slot a draw reads through a
/// table is looked up here, on several recording threads at once in the
/// world. Readers share it; only a view not seen before (a new table entry
/// or shape) and a released resource take it alone.
static VIEWS: std::sync::RwLock<Views> = std::sync::RwLock::new(Views {
    list: Vec::new(),
    index: std::collections::BTreeMap::new(),
    by_resource: std::collections::BTreeMap::new(),
    free: Vec::new(),
    shapes: Vec::new(),
    shape_index: std::collections::BTreeMap::new(),
});
const VIEW_TAG: u32 = u32::from_le_bytes(*b"vVIE");

/// A record is the tag and two words, `i` and `generation` (the host keeps
/// them as one 64-bit key per slot). The top two bits of `generation` say
/// what the other 62 bits are: 0, entry `i` of the table at that
/// generation; [`CONSTANT`], a constant buffer view, whole; [`PLACED`], a
/// view of placed memory: its address and the number of its shape.
const KIND_SHIFT: u32 = 30;
/// Table generations stay below the kind bits (and above 0).
const GENERATIONS: u32 = (1 << KIND_SHIFT) - 1;
const CONSTANT: u32 = 1;
const PLACED: u32 = 2;
/// Bits of a constant buffer view's address in its record: any address a
/// program can have. The 15 bits above them are its size.
const CONSTANT_ADDRESS_BITS: u32 = 47;
/// Bits of a placed view's address in its record: placed memory is title
/// memory, which lies below 1 TiB. The 22 bits above them number its shape.
const PLACED_ADDRESS_BITS: u32 = 40;
const SHAPES: usize = 1 << (62 - PLACED_ADDRESS_BITS);

/// A view of placed memory without its address: whether it is unordered
/// access, the resource description, and the view's words but the two that
/// hold the address (10 and 11).
type Shape = (bool, [u32; 14], [u32; 14]);

/// Whether constant buffer views and views of placed memory are named whole
/// in their records ([`Views`]). On by default; `VICTORY_VIEW_INLINE=0`
/// makes every view a table entry, as before.
fn inline() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_INLINE").is_none_or(|v| v != "0"))
}

/// Views named whole in their records since the start: constant buffer
/// views, views of placed memory (for the memory line).
static NAMED_WHOLE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The two record words of the 62 bits `payload` of kind `kind`.
fn kind_words(kind: u32, payload: u64) -> [u32; 2] {
    [payload as u32, kind << KIND_SHIFT | (payload >> 32) as u32]
}

/// The 62 bits two record words carry beside their kind.
fn payload(i: u32, generation: u32) -> u64 {
    u64::from(generation & GENERATIONS) << 32 | u64::from(i)
}

/// The record words that name the constant buffer view of `size` bytes at
/// `address` whole; `None` if they cannot: an address past 47 bits, or a
/// size of 16 KiB or more that is no multiple of 256 or reaches 4 MiB (the
/// size has 15 bits: bytes below 16 KiB, else units of 256 bytes). Such a
/// view is a table entry.
fn constant_words(address: u64, size: u32) -> Option<[u32; 2]> {
    if address >> CONSTANT_ADDRESS_BITS != 0 {
        return None;
    }
    let code = if size < 1 << 14 {
        u64::from(size)
    } else if size.is_multiple_of(256) && size >> 8 < 1 << 14 {
        1 << 14 | u64::from(size >> 8)
    } else {
        return None;
    };
    Some(kind_words(CONSTANT, code << CONSTANT_ADDRESS_BITS | address))
}

/// The constant buffer view [`constant_words`] named.
fn constant_of(i: u32, generation: u32) -> View {
    let payload = payload(i, generation);
    let code = (payload >> CONSTANT_ADDRESS_BITS) as u32;
    let size = if code >> 14 == 0 { code } else { (code & 0x3fff) << 8 };
    View::Constant(payload & ((1 << CONSTANT_ADDRESS_BITS) - 1), size)
}

/// A placed view's shape and its address.
fn shape_of(unordered: bool, resource: &[u32; 14], desc: &[u32; 16]) -> (Shape, u64) {
    let mut rest = [0u32; 14];
    rest[..10].copy_from_slice(&desc[..10]);
    rest[10..].copy_from_slice(&desc[12..]);
    (
        (unordered, *resource, rest),
        u64::from(desc[10]) | u64::from(desc[11]) << 32,
    )
}

/// The record words that name the view of placed memory at `address` with
/// shape number `shape`.
fn placed_words(shape: u32, address: u64) -> [u32; 2] {
    kind_words(PLACED, u64::from(shape) << PLACED_ADDRESS_BITS | address)
}

impl Views {
    /// The number of `shape`, a new one if it is not there yet; `None` when
    /// the shapes are full (the view is then a table entry).
    fn shape(&mut self, shape: Shape) -> Option<u32> {
        if let Some(&at) = self.shape_index.get(&shape) {
            return Some(at);
        }
        if self.shapes.len() >= SHAPES {
            return None;
        }
        let at = self.shapes.len() as u32;
        self.shapes.push(shape);
        self.shape_index.insert(shape, at);
        Some(at)
    }

    /// The view of placed memory [`placed_words`] named.
    fn placed_of(&self, i: u32, generation: u32) -> Option<View> {
        let payload = payload(i, generation);
        let address = payload & ((1 << PLACED_ADDRESS_BITS) - 1);
        let &(unordered, resource, rest) = self.shapes.get((payload >> PLACED_ADDRESS_BITS) as usize)?;
        let mut desc = [0u32; 16];
        desc[..10].copy_from_slice(&rest[..10]);
        (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
        desc[12..].copy_from_slice(&rest[10..]);
        Some(View::Placed(unordered, resource, desc))
    }

    /// The record words of `view` as a table entry, if it has one.
    fn entry_of(&self, view: &View) -> Option<[u32; 2]> {
        let &i = self.index.get(view)?;
        Some([i, self.list[i as usize].1])
    }

    /// The record words of `view` as a table entry: its entry (a new or a
    /// freed one if the view is not there yet) and that entry's generation.
    fn entry(&mut self, view: View) -> [u32; 2] {
        let i = match self.index.get(&view) {
            Some(&i) => i,
            None => {
                let i = match self.free.pop() {
                    Some(i) => {
                        self.list[i as usize].0 = view.clone();
                        i
                    }
                    None => {
                        self.list.push((view.clone(), 1));
                        self.list.len() as u32 - 1
                    }
                };
                for r in view.resources().into_iter().filter(|&r| r != 0) {
                    self.by_resource.entry(r).or_default().push(i);
                }
                self.index.insert(view, i);
                i
            }
        };
        [i, self.list[i as usize].1]
    }
}

/// The two words that name `view` in a record: the view itself where its
/// kind allows ([`Views`]), else its table entry.
fn words_for(view: View) -> [u32; 2] {
    use std::sync::atomic::Ordering::Relaxed;
    if inline() {
        match &view {
            View::Constant(address, size) => {
                if let Some(words) = constant_words(*address, *size) {
                    NAMED_WHOLE[0].fetch_add(1, Relaxed);
                    return words;
                }
            }
            View::Placed(unordered, resource, desc) => {
                let (shape, address) = shape_of(*unordered, resource, desc);
                if address >> PLACED_ADDRESS_BITS == 0 {
                    // Nearly every placed view has a shape seen before.
                    let known = views_read().shape_index.get(&shape).copied();
                    let shape = known.or_else(|| views_write().shape(shape));
                    if let Some(shape) = shape {
                        NAMED_WHOLE[1].fetch_add(1, Relaxed);
                        return placed_words(shape, address);
                    }
                }
            }
            _ => {}
        }
    }
    // A view written before keeps its entry and generation.
    if let Some(words) = views_read().entry_of(&view) {
        return words;
    }
    views_write().entry(view)
}

fn views_read() -> std::sync::RwLockReadGuard<'static, Views> {
    VIEWS.read().unwrap_or_else(|e| e.into_inner())
}

fn views_write() -> std::sync::RwLockWriteGuard<'static, Views> {
    VIEWS.write().unwrap_or_else(|e| e.into_inner())
}

/// For the memory line: the distinct views kept ([`Views`]), those of them
/// free for reuse, and the bytes the table and its index hold for them (an
/// estimate: an entry of the table, and one of the index with its share of
/// a tree node; the shapes likewise); then the entries that are constant
/// buffer views and views of placed memory (those that did not fit a
/// record, or all of them with `VICTORY_VIEW_INLINE=0`), the shapes of
/// placed views kept, and the views named whole in their records since the
/// start: constant buffer views, views of placed memory.
#[cfg(all(windows, feature = "diagnostics"))]
pub(crate) fn views_kept() -> [u64; 8] {
    use std::sync::atomic::Ordering::Relaxed;
    let views = views_read();
    let each = size_of::<(View, u32)>() as u64;
    let shape = size_of::<Shape>() as u64;
    let (mut constants, mut placed) = (0, 0);
    for (view, _) in &views.list {
        match view {
            View::Constant(..) => constants += 1,
            View::Placed(..) => placed += 1,
            _ => {}
        }
    }
    [
        views.list.len() as u64,
        views.free.len() as u64,
        views.list.capacity() as u64 * each
            + views.index.len() as u64 * each * 3 / 2
            + views.shapes.capacity() as u64 * shape
            + views.shape_index.len() as u64 * shape * 3 / 2,
        constants,
        placed,
        views.shapes.len() as u64,
        NAMED_WHOLE[0].load(Relaxed),
        NAMED_WHOLE[1].load(Relaxed),
    ]
}

impl View {
    /// The resource objects the view names.
    fn resources(&self) -> [usize; 2] {
        match *self {
            View::Shader(r, _) => [r, 0],
            View::Unordered(r, counter, _) => [r, counter],
            _ => [0, 0],
        }
    }

    /// The texture object the view is of, if that texture has no host copy
    /// yet (`crate::resource::host_waits`): a slot written with such a
    /// view is left null until the texture is first used.
    #[cfg(windows)]
    fn waiting_texture(&self) -> Option<usize> {
        match *self {
            View::Shader(r, _) | View::Unordered(r, _, _)
                if crate::resource::host_waits(r as This) =>
            {
                Some(r)
            }
            _ => None,
        }
    }
}

/// Title slots (their handles) written with a view of a texture object
/// that had no host copy then, by texture. A title texture gets its host
/// copy on first use (`resource.rs` `Hosted`), and the title
/// writes each texture's views into its view heap when it creates the
/// texture: making those host slots would make every host copy at once.
/// They are made when the texture is: translated shaders index the heap
/// directly, so a slot is not always made again by the draw that reads it.
#[cfg(windows)]
static WAITING: std::sync::Mutex<std::collections::BTreeMap<usize, Vec<u64>>> =
    std::sync::Mutex::new(std::collections::BTreeMap::new());

/// Textures given their host copy since [`make_waiting`] last ran.
#[cfg(windows)]
static MADE_LATE: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// Title texture `resource` was given its host copy (called under that
/// texture's own lock: nothing else is locked here).
#[cfg(windows)]
pub(crate) fn texture_made(resource: usize) {
    MADE_LATE
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push(resource);
}

/// The slot at `handle` was written with a view of texture `resource`,
/// which has no host copy: it waits for one.
#[cfg(windows)]
fn wait_for(resource: usize, handle: u64) {
    /// A texture with more slots waiting than this is in use for all we
    /// know (its views are being copied about): its host copy is made.
    const MOST: usize = 64;
    let many = {
        let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
        let slots = waiting.entry(resource).or_default();
        if !slots.contains(&handle) {
            slots.push(handle);
        }
        slots.len() > MOST
    };
    if many {
        // Makes it, and notes it made.
        let _ = crate::resource::host_copy(resource as This);
    } else if !crate::resource::host_waits(resource as This) {
        // Made by another thread since the slot was looked at: that making
        // may have been taken by `make_waiting` before this slot was noted.
        texture_made(resource);
    }
}

/// Makes the host slots that waited for a texture's host copy, for the
/// textures made since the last call. Called before a submission runs, by
/// the thread that submits: a slot is read when a list runs.
#[cfg(windows)]
pub(crate) fn make_waiting() {
    let made = std::mem::take(&mut *MADE_LATE.lock().unwrap_or_else(|e| e.into_inner()));
    for resource in made {
        let handles = WAITING
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&resource);
        for handle in handles.into_iter().flatten() {
            // Only if the slot still holds a view of that texture: the
            // title may have written it since (and a view of another
            // texture without a host copy must go on waiting).
            // (Looked up first: the handle is read only while its heap is
            // there.)
            let Some((heap, index)) = shader_slot(handle) else {
                continue;
            };
            let same = match view(handle) {
                Some(View::Shader(r, _) | View::Unordered(r, _, _)) => r == resource,
                _ => false,
            };
            if same && let Some(Some(host)) = heap.host.get() {
                host.mirror(index, handle, true);
            }
        }
    }
}

fn record(handle: u64, view: View) {
    if handle == 0 {
        return;
    }
    let [i, generation] = words_for(view);
    // SAFETY: the handle is a descriptor slot, at least 16 bytes.
    unsafe { (handle as *mut [u32; 3]).write_unaligned([VIEW_TAG, i, generation]) };
    #[cfg(windows)]
    mirror(handle);
}

/// The title released `resource`: the views naming it go stale.
pub(crate) fn forget(resource: usize) {
    #[cfg(windows)]
    WAITING
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&resource);
    let mut views = views_write();
    let Some(entries) = views.by_resource.remove(&resource) else {
        return;
    };
    for i in entries {
        let (view, generation) = views.list[i as usize].clone();
        // Entries freed through a view's other resource may hold another
        // view by now.
        if views.index.get(&view) != Some(&i) || !view.resources().contains(&resource) {
            continue;
        }
        views.index.remove(&view);
        // (Below the bits that say a record is no table entry.)
        views.list[i as usize].1 = ((generation + 1) & GENERATIONS).max(1);
        // The view's other resource (a counter) keeps a stale entry, which
        // `forget` of that resource skips.
        views.free.push(i);
    }
}

/// The view the record words `i` and `generation` name: the one they hold
/// whole, or the view of entry `i` if `generation` is still its own.
fn view_at(i: u32, generation: u32) -> Option<View> {
    match generation >> KIND_SHIFT {
        0 => {
            let views = views_read();
            let (view, current) = views.list.get(i as usize)?;
            (*current == generation).then(|| view.clone())
        }
        CONSTANT => Some(constant_of(i, generation)),
        PLACED => views_read().placed_of(i, generation),
        _ => None,
    }
}

/// The record words (`i`, `generation`) at `handle`, if it holds a record
/// of ours; whether they still name a view is [`records_live`]'s.
pub(crate) fn record_words(handle: u64) -> Option<[u32; 2]> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    (tag == VIEW_TAG).then_some([i, generation])
}

/// For each of `records` (record words), whether [`view_at`] would find a
/// view for it, under one lock: records that name a view whole always do,
/// a table entry's only while it has that generation (its resources were
/// not released since).
pub(crate) fn records_live(records: impl Iterator<Item = [u32; 2]>) -> Vec<bool> {
    let views = views_read();
    records
        .map(|[i, generation]| match generation >> KIND_SHIFT {
            0 => views.list.get(i as usize).is_some_and(|(_, current)| *current == generation),
            CONSTANT => true,
            PLACED => views.shapes.get((payload(i, generation) >> PLACED_ADDRESS_BITS) as usize).is_some(),
            _ => false,
        })
        .collect()
}

/// The view recorded at `handle`, if one is (and its resource is alive).
pub(crate) fn view(handle: u64) -> Option<View> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let [tag, i, generation] = unsafe { (handle as *const [u32; 3]).read_unaligned() };
    if tag != VIEW_TAG {
        return None;
    }
    view_at(i, generation)
}

/// A description of `N` words, if the title passes one.
fn words<const N: usize>(desc: *const [u32; N]) -> Option<[u32; N]> {
    // SAFETY: the title passes a description or null.
    (!desc.is_null()).then(|| unsafe { desc.read_unaligned() })
}

/// Device method.
pub(crate) extern "system" fn create_cbv(_: This, desc: *const [u32; 4], handle: u64) {
    if let Some([lo, hi, size, _]) = words(desc) {
        let address = u64::from(lo) | u64::from(hi) << 32;
        record(handle, View::Constant(address, size));
    }
}

/// Device method.
pub(crate) extern "system" fn create_srv(
    _: This,
    resource: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    record(handle, View::Shader(resource as usize, words(desc)));
}

/// Device method.
pub(crate) extern "system" fn create_uav(
    _: This,
    resource: This,
    counter: This,
    desc: *const [u32; 10],
    handle: u64,
) {
    let view = View::Unordered(resource as usize, counter as usize, words(desc));
    record(handle, view);
}

/// Device methods 0x230 and 0x238: shader resource and unordered access
/// views of memory the title placed itself, `(resource desc, view desc,
/// handle)`.
pub(crate) extern "system" fn create_placed_srv(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(false, resource, desc));
    }
}

pub(crate) extern "system" fn create_placed_uav(
    _: This,
    resource: *const [u32; 14],
    desc: *const [u32; 16],
    handle: u64,
) {
    if let (Some(resource), Some(desc)) = (words(resource), words(desc)) {
        record(handle, View::Placed(true, resource, desc));
    }
}

/// Device method.
pub(crate) extern "system" fn create_sampler(_: This, desc: *const [u32; 13], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::Sampler(desc));
    }
}

/// Device method 0x1e8.
pub(crate) extern "system" fn create_sampler_x(_: This, desc: *const [u32; 27], handle: u64) {
    if let Some(desc) = words(desc) {
        record(handle, View::ConsoleSampler(desc));
    }
}

/// Device method: `count` descriptors of `kind` from `src` to `dst`.
pub(crate) extern "system" fn copy_simple(_: This, count: u32, dst: u64, src: u64, kind: u32) {
    let step = u64::from(increment(kind).unwrap_or(0));
    let len = u64::from(count) * step;
    if dst != 0 && src != 0 {
        // SAFETY: both ranges are descriptor slots the title owns.
        unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, len as usize) };
        #[cfg(windows)]
        for i in 0..u64::from(count) {
            mirror(dst + i * step);
        }
    }
}

/// Device method: ranges of descriptors (start and count each) gathered
/// from the sources and scattered to the destinations, in order.
#[allow(clippy::too_many_arguments)] // the method's own arguments
pub(crate) extern "system" fn copy(
    _: This,
    dst_ranges: u32,
    dst_starts: *const u64,
    dst_sizes: *const u32,
    src_ranges: u32,
    src_starts: *const u64,
    src_sizes: *const u32,
    kind: u32,
) {
    let Some(step) = increment(kind) else {
        return;
    };
    // Each range as (start, count); a null size array means one descriptor
    // per range.
    let ranges = |n: u32, starts: *const u64, sizes: *const u32| -> Vec<(u64, u32)> {
        (0..n as usize)
            .map(|i| {
                // SAFETY: the title passes `n` starts and sizes (or null sizes).
                unsafe {
                    (
                        *starts.add(i),
                        if sizes.is_null() { 1 } else { *sizes.add(i) },
                    )
                }
            })
            .collect()
    };
    if dst_starts.is_null() || src_starts.is_null() {
        return;
    }
    let src = ranges(src_ranges, src_starts, src_sizes);
    let mut dst = ranges(dst_ranges, dst_starts, dst_sizes).into_iter();
    let (mut to, mut room) = dst.next().unwrap_or((0, 0));
    for (mut from, mut left) in src {
        while left > 0 {
            while room == 0 {
                let Some(next) = dst.next() else { return };
                (to, room) = next;
            }
            let n = left.min(room);
            copy_simple(std::ptr::null(), n, to, from, kind);
            let bytes = u64::from(n) * u64::from(step);
            (to, from, room, left) = (to + bytes, from + bytes, room - n, left - n);
        }
    }
}

/// The view recorded at `handle`, if one is.
pub(crate) fn rtv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, RTV_TAG)
}

/// The depth view recorded at `handle`, if one is.
pub(crate) fn dsv_record(handle: u64) -> Option<RtvRecord> {
    tagged_record(handle, DSV_TAG)
}

fn tagged_record(handle: u64, tag: u32) -> Option<RtvRecord> {
    if handle == 0 {
        return None;
    }
    // SAFETY: the handle is a descriptor slot the title passes back.
    let record = unsafe { (handle as *const RtvRecord).read_unaligned() };
    (record.tag == tag).then_some(record)
}

/// Device method.
pub(crate) extern "system" fn create_heap(
    _: This,
    desc_ptr: *const HeapDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc_ptr.as_ref() }) else {
        return E_INVALIDARG;
    };
    let Some(size) = increment(desc.kind).map(|i| u64::from(i) * u64::from(desc.count.max(1)))
    else {
        return E_INVALIDARG;
    };
    // SAFETY: read only as a hint: a title that passes the plain
    // description has its own stack or data after it.
    let placement = unsafe { desc_ptr.add(1).cast::<Placement>().read_unaligned() };
    // shortcut: a placement is recognised by its size matching the count,
    // since the plain and extended descriptions share the slot and the
    // interface version; find the flag that tells them apart if a title
    // trips over this.
    let placed = placement.address != 0 && placement.size == size;
    tracing::debug!(?desc, placed, "descriptor heap");
    let (start, memory) = if placed {
        (placement.address, None)
    } else {
        match Memory::zeroed(size, 64) {
            Ok(memory) => (memory.address(), Some(memory)),
            Err(hr) => return hr,
        }
    };
    let visible = desc.flags & SHADER_VISIBLE != 0 && matches!(desc.kind, VIEWS_KIND | SAMPLERS);
    let heap = Box::new(Heap {
        object: Object::new(&FUNCTIONS),
        desc,
        start,
        memory,
        #[cfg(windows)]
        host: std::sync::OnceLock::new(),
        #[cfg(windows)]
        pending: Default::default(),
        #[cfg(windows)]
        written: (placed && visible)
            .then(|| crate::watch::track(start as usize, size as usize, true))
            .flatten(),
    });
    #[cfg(windows)]
    if visible {
        let at = std::ptr::from_ref::<Heap>(&heap) as usize;
        let mut heaps = SHADER_HEAPS.write().unwrap_or_else(|e| e.into_inner());
        heaps.push((start, start + size, at));
    }
    #[cfg(not(windows))]
    let _ = visible;
    object::hand_out(DESCRIPTOR_HEAP, heap, iid, &HEAP_IIDS, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    let found = object::is(iid, &HEAP_IIDS).then_some(this);
    object::answer(DESCRIPTOR_HEAP, iid, found, out)
}
extern "system" fn release(this: This) -> u32 {
    object::release_boxed::<Heap>(this)
}
extern "system" fn desc(this: This, out: *mut HeapDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(heap(this).desc) };
    }
}
/// The same address for the CPU and the GPU start.
extern "system" fn start(this: This) -> u64 {
    heap(this).start
}

/// IUnknown 3, object 4, device child 1, heap 3.
static FUNCTIONS: Table<24> = {
    let mut f = object::d3d12_object(object::missing_table::<DESCRIPTOR_HEAP, 24>());
    f[0] = query as *const ();
    f[2] = release as *const ();
    f[7] = object::parent_device as *const (); // 0x38
    f[8] = desc as *const (); // 0x40
    f[9] = start as *const (); // 0x48 CPU
    f[10] = start as *const (); // 0x50 GPU
    Table(f)
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::object::{S_OK, slot};

    /// What newer titles pass; tests use it throughout, so that reading the
    /// placement stays inside the value.
    #[repr(C)]
    struct Extended(HeapDesc, Placement);

    fn create(desc: HeapDesc, placement: Placement, out: &mut This) -> HRESULT {
        let extended = Extended(desc, placement);
        create_heap(std::ptr::null(), &extended.0, &HEAP_IIDS[0], out)
    }

    #[cfg(windows)]
    #[test]
    fn raw_heap_sync_stops_before_dynamic_host_slots() {
        let start = 0x1000usize;
        let step = 32;
        let capped = 5;
        assert_eq!(
            changed_slots(
                start + 3 * step + 4,
                start + 4 * step + 8,
                start,
                step as u64,
                capped
            ),
            3..5
        );
        assert_eq!(
            changed_slots(
                start + 6 * step,
                start + 9 * step,
                start,
                step as u64,
                capped
            ),
            5..5
        );
    }

    #[test]
    fn heaps_hold_count_descriptors_from_their_start() {
        let null = std::ptr::null();
        let desc = HeapDesc {
            kind: 0,
            count: 1000,
            flags: 1,
            nodes: 0,
        };
        let none = Placement {
            address: 0,
            size: 0,
        };
        let mut h = null;
        assert_eq!(create(desc, none, &mut h), S_OK);
        let cpu = slot::<extern "system" fn(This) -> u64>(h, 0x48)(h);
        assert_eq!(cpu % 64, 0);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), cpu);
        assert_eq!(increment_size(null, 0), 32);
        // SAFETY: the heap holds 1000 descriptors of 32 bytes.
        unsafe { (cpu as *mut u8).add(999 * 32 + 31).write(1) };
        assert_eq!(increment_size(null, 9), 0);
        let bad = HeapDesc { kind: 9, ..desc };
        assert_eq!(create(bad, none, &mut h), E_INVALIDARG);
    }

    /// A descriptor copied into a placed heap as bytes is mirrored before
    /// the next submission, even though no descriptor method saw the copy.
    #[cfg(windows)]
    #[test]
    fn records_copied_as_bytes_are_made_at_the_next_sync() {
        use windows_sys::Win32::System::Memory::{
            MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        };
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        let memory = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                crate::watch::PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        };
        assert!(!memory.is_null());
        let desc = HeapDesc {
            kind: SAMPLERS,
            count: 64,
            flags: SHADER_VISIBLE,
            nodes: 1,
        };
        let placement = Placement {
            address: memory as u64,
            size: 64 * 16,
        };
        let mut h: This = std::ptr::null();
        assert_eq!(create(desc, placement, &mut h), S_OK);
        if let Some(host) = heap(h).host() {
            let mut elsewhere = [0u64; 2];
            let source = elsewhere.as_mut_ptr() as u64;
            let sampler = [21, 3, 3, 3, 0, 1, 0, 0, 0, 0, 0, 0x7f7f_ffff, 0];
            create_sampler(std::ptr::null(), &sampler, source);
            // SAFETY: the heap slot and source both hold 16 descriptor bytes.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source as *const u8,
                    memory.cast::<u8>().add(5 * 16),
                    16,
                )
            };
            assert!(!host.made(5));
            sync_heaps();
            assert!(host.made(5));
            // `VICTORY_SYNC_OVERLAP`: the sync on its own thread makes the
            // same slots, and they are made once it is joined.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    source as *const u8,
                    memory.cast::<u8>().add(9 * 16),
                    16,
                )
            };
            assert!(!host.made(9));
            let mut on = crate::host::sync_beside_start();
            assert!(on);
            crate::host::sync_beside_join(&mut on);
            assert!(!on);
            assert!(host.made(9));
            // `VICTORY_SYNC_OVERLAP=2`: a pass leaves a page under flights'
            // writes to the sync before the submission, and makes the rest.
            let page = memory as usize;
            unsafe {
                std::ptr::copy_nonoverlapping(source as *const u8, memory.cast::<u8>().add(12 * 16), 16);
            }
            sync_heaps_beside(&[(page, page + 64 * 16)]);
            assert!(!host.made(12));
            sync_heaps();
            assert!(host.made(12));
            unsafe {
                std::ptr::copy_nonoverlapping(source as *const u8, memory.cast::<u8>().add(13 * 16), 16);
            }
            sync_heaps_beside(&[]);
            assert!(host.made(13));
            // `VICTORY_SYNC_OVERLAP=3`: the pre-read makes nothing; the
            // sync takes the page and makes the record...
            let write = |slot: usize| unsafe {
                std::ptr::copy_nonoverlapping(source as *const u8, memory.cast::<u8>().add(slot * 16), 16);
            };
            write(20);
            assert!(sync_heaps_preread());
            assert!(!host.made(20));
            sync_heaps();
            assert!(host.made(20));
            // ...and where the page changed after the pre-read, the record
            // written since is made too.
            write(21);
            assert!(sync_heaps_preread());
            write(22);
            sync_heaps();
            assert!(host.made(21) && host.made(22));
            // It takes nothing: a sync before a title-read wait takes the
            // page as ever, and makes the record.
            write(23);
            assert!(sync_heaps_preread());
            sync_heaps_early(&[]);
            assert!(host.made(23));
        }
        slot::<extern "system" fn(This) -> u32>(h, 0x10)(h);
        // SAFETY: this allocation belongs to the test.
        unsafe { VirtualFree(memory, 0, MEM_RELEASE) };
    }

    #[test]
    fn views_survive_copies() {
        let null = std::ptr::null();
        let mut heap = vec![0u64; 4 * 8]; // four 32-byte slots
        let at = |i: usize| heap.as_ptr() as u64 + 32 * i as u64;
        let (s0, s1, s2, s3) = (at(0), at(1), at(2), at(3));
        create_cbv(null, &[0x1000, 1, 256, 0], s0);
        assert_eq!(view(s0), Some(View::Constant(0x1_0000_1000, 256)));
        create_srv(null, null, std::ptr::null(), s1);
        assert_eq!(view(s1), Some(View::Shader(0, None)));
        // The same view twice is one entry.
        create_srv(null, null, std::ptr::null(), s2);
        // SAFETY: both slots are in `heap`.
        assert_eq!(unsafe { *(s1 as *const [u32; 3]) }, unsafe {
            *(s2 as *const [u32; 3])
        });
        // Two ranges of one into one range of two.
        let (dst, dst_n, src, src_n) = ([s2], [2u32], [s0, s1], [1u32, 1]);
        copy(
            null,
            1,
            dst.as_ptr(),
            dst_n.as_ptr(),
            2,
            src.as_ptr(),
            src_n.as_ptr(),
            0,
        );
        assert_eq!(view(s2), view(s0));
        assert_eq!(view(s3), view(s1));
        assert_eq!(view(at(0) + 8), None);
        heap[0] = 0;
    }

    /// Constant buffer views are named whole in their records: any address
    /// a program can have, any size below 16 KiB, multiples of 256 bytes
    /// up to 4 MiB; what does not fit says so.
    #[test]
    fn constant_views_fit_their_records() {
        let top = (1u64 << CONSTANT_ADDRESS_BITS) - 1;
        for address in [0u64, 0x100, 0x2602_0000, 0xd_6b60_0000, 0x7ffd_a34f_0000, top] {
            for size in [0u32, 1, 255, 256, 4000, (1 << 14) - 1, 1 << 14, 65536, (1 << 22) - 256] {
                let [i, generation] = constant_words(address, size).expect("fits");
                assert_eq!(generation >> KIND_SHIFT, CONSTANT);
                assert_eq!(view_at(i, generation), Some(View::Constant(address, size)), "{address:#x} {size}");
            }
        }
        assert_eq!(constant_words(top + 1, 256), None, "an address past 47 bits");
        assert_eq!(constant_words(0x1000, (1 << 14) + 1), None, "16 KiB or more, no multiple of 256");
        assert_eq!(constant_words(0x1000, 1 << 22), None, "4 MiB");
        // Different views never share their words.
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1000, 512));
        assert_ne!(constant_words(0x1000, 256), constant_words(0x1100, 256));
        assert_ne!(constant_words(0x1000, 1 << 14), constant_words(0x1000, 64));
    }

    /// The leak of run 65: a title that writes constant buffer views and
    /// views of placed memory at ever new addresses adds nothing to the
    /// table; one shape stands for all the placed views alike.
    #[test]
    fn views_at_new_addresses_add_no_entries() {
        if !inline() {
            return;
        }
        let null = std::ptr::null();
        let slots = [0u64; 4];
        let at = slots.as_ptr() as u64;
        // A resource description and a view no other test uses.
        let resource: [u32; 14] = std::array::from_fn(|k| 0x5ee0_0000 + k as u32);
        let mut desc: [u32; 16] = std::array::from_fn(|k| 0x7a70_0000 + k as u32);
        let mine = |v: &View| match v {
            View::Constant(address, _) => (0x6_0000_0000..0x6_1000_0000).contains(address),
            View::Placed(_, r, _) => *r == resource,
            _ => false,
        };
        let entries = || VIEWS.read().unwrap().list.iter().filter(|(v, _)| mine(v)).count();
        let shapes = || VIEWS.read().unwrap().shapes.iter().filter(|s| s.1 == resource).count();
        for n in 0..20_000u64 {
            let address = 0x6_0000_0000 + n * 0x100;
            create_cbv(null, &[address as u32, (address >> 32) as u32, 512, 0], at);
            assert_eq!(view(at), Some(View::Constant(address, 512)));
            let placed = 0xd_4940_0000 + n * 0x1_0000;
            (desc[10], desc[11]) = (placed as u32, (placed >> 32) as u32);
            create_placed_srv(null, &resource, &desc, at);
            assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        }
        assert_eq!(entries(), 0, "no table entries");
        assert_eq!(shapes(), 1, "one shape for 20,000 placed views");
        // The same shape, unordered: another shape.
        create_placed_uav(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(true, resource, desc)));
        assert_eq!(shapes(), 2);
        // Placed memory past 1 TiB (not title memory) is a table entry.
        (desc[10], desc[11]) = (0, 0x100);
        create_placed_srv(null, &resource, &desc, at);
        assert_eq!(view(at), Some(View::Placed(false, resource, desc)));
        assert_eq!(entries(), 1);
        // A record's words are the same for the same view (the host knows
        // a slot's view by them), and differ with the address.
        // SAFETY: both handles are slots of 32 bytes above.
        let words = |handle: u64| unsafe { *(handle as *const [u32; 3]) };
        let other = [0u64; 4];
        create_cbv(null, &[0x40, 6, 512, 0], at);
        create_cbv(null, &[0x40, 6, 512, 0], other.as_ptr() as u64);
        assert_eq!(words(at), words(other.as_ptr() as u64));
        create_cbv(null, &[0x140, 6, 512, 0], other.as_ptr() as u64);
        assert_ne!(words(at), words(other.as_ptr() as u64));
    }

    /// A released resource's views go stale; the same view of a new
    /// resource at the same address is a new record.
    #[test]
    fn views_die_with_their_resource() {
        let null = std::ptr::null();
        let slots = [0u64; 2 * 4];
        let (a, b) = (slots.as_ptr() as u64, slots.as_ptr() as u64 + 32);
        let resource = 0x7777_0000usize as This;
        create_srv(null, resource, std::ptr::null(), a);
        assert_eq!(view(a), Some(View::Shader(0x7777_0000, None)));
        let words = record_words(a).expect("a record of ours");
        assert_eq!(records_live([words].into_iter()), [true]);
        forget(0x7777_0000);
        assert_eq!(view(a), None, "stale after release");
        assert_eq!(records_live([words].into_iter()), [false], "no longer live after release");
        assert_eq!(record_words(0), None);
        create_srv(null, resource, std::ptr::null(), b);
        assert_eq!(view(b), Some(View::Shader(0x7777_0000, None)));
        // SAFETY: both slots are in `slots`.
        let (old, new) = unsafe { (*(a as *const [u32; 3]), *(b as *const [u32; 3])) };
        assert_ne!(old, new, "a new generation");
    }

    /// The title places its shader-visible heap in its own memory
    /// and checks that the heap starts there.
    #[test]
    fn placed_heaps_start_at_the_title_memory() {
        let mut memory = vec![0u8; 64 * 32];
        let address = memory.as_mut_ptr() as u64;
        let desc = HeapDesc {
            kind: 0,
            count: 64,
            flags: 1,
            nodes: 1,
        };
        let mut h = std::ptr::null();
        assert_eq!(
            create(
                desc,
                Placement {
                    address,
                    size: 64 * 32
                },
                &mut h
            ),
            S_OK
        );
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
        assert_eq!(slot::<extern "system" fn(This) -> u64>(h, 0x50)(h), address);
        // A size that does not match the count is not a placement.
        assert_eq!(create(desc, Placement { address, size: 7 }, &mut h), S_OK);
        assert_ne!(slot::<extern "system" fn(This) -> u64>(h, 0x48)(h), address);
    }
}
