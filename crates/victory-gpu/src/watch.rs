//! Which title memory the title wrote since a host copy took it (decision
//! 0007, "Copies of title memory"). The runtime reserves title GPU memory
//! with write watching (`victory-hle` xmem), so the kernel keeps a dirty bit
//! per 4 KiB page; asking for a range's written pages also clears them. Host
//! copies of the same memory can overlap (nested buffers, textures in
//! buffers), so what one asks is kept here for the others: a journal of
//! each page's last write, numbered by the asking.
//!
//! Physical-page views (xmem's sections) cannot be write-watched. There a
//! copy leaves each 64 KiB block it took read-only, and our vectored handler
//! takes the first write to it: it notes the block as written and gives the
//! write access back. The runtime tells us when it maps or unmaps views
//! ([`title_memory_remapped`]), which resets their protection. Blocks found
//! wholly uncommitted are skipped while they remain so; one query can
//! confirm a whole region, including ordinary reservations between views.
//!
//! Writes into physical pages through another view are seen too: the
//! runtime tells us the views of the pages it wrote itself ([`writing`],
//! xmem's `write_physical` and DirectStorage's page destinations), and a
//! write by the title through any view of a guarded page faults like one
//! at the guarded address (the guard makes every view of the page
//! read-only, [`open`] gives them all back; not a page of more than
//! [`MANY_VIEWS`] views). A
//! system call writing into a guarded block fails (none is known to).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// The unit tracked and copied: 64 KiB, the console's page (the kernel's
/// 4 KiB pages fold into it). Finer units made the checks of the world's
/// big buffers cost more than the copies they saved.
pub(crate) const PAGE: usize = 64 << 10;
/// The kernel's page, the unit of its answers.
const KERNEL_PAGE: usize = 4096;
/// With [`fine`]: the part of a [`PAGE`] the journal also keeps, the
/// kernel's page.
pub(crate) const SUB: usize = KERNEL_PAGE;
/// Parts of a [`PAGE`] ([`SUB`]), one bit each in a mask.
pub(crate) const SUBS: usize = PAGE / SUB;
/// Every part of a page.
pub(crate) const ALL_SUBS: u16 = u16::MAX;

/// Whether the journal also keeps, per 4 KiB part of a page, the asking
/// that last found it written, and host copies copy only the parts written
/// since they last took the page (`VICTORY_FINE_PAGES=1`; `2` also checks
/// every part passed over against title memory, see
/// `crate::shadow::fine_line`; off by default. Read once).
///
/// The kernel answers which 4 KiB pages were written; the journal kept
/// 64 KiB pages, and every refresh copied, and every copy in the card's
/// memory sent up, the whole 64 KiB page for a write of a few bytes in it:
/// 89-190 MB copied and 72-175 MB sent up a frame in the opening cutscene.
/// With the switch the 64 KiB journal stays as it is (what every asker
/// reads) and a second map keeps the parts ([`Written::subs`]), asked only
/// for the pages a copy finds stale ([`subs_after`]).
///
/// What it rests on: every write into write-watched memory is reported in
/// the 4 KiB page it lands in (the kernel's dirty bit, also for the
/// kernel's own writes, ours through `NtWriteVirtualMemory` among them);
/// writes we know of without the kernel ([`wrote`]) mark the parts they
/// touch; the guards of view memory work in 64 KiB blocks, so their pages
/// are marked whole. A write the kernel does not report in the page it
/// lands in (none is known) was copied before when another part of the
/// same 64 KiB page was written; with the switch it is not.
pub(crate) fn fine() -> bool {
    fine_level() >= 1
}

/// `VICTORY_FINE_PAGES`: 0 (off), 1, or 2 (with the check). Tests keep the
/// parts whatever is set, so that the copies' tests can use them.
pub(crate) fn fine_level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VICTORY_FINE_PAGES").as_deref() {
        Ok("1") => 1,
        Ok("2") => 2,
        _ => u8::from(cfg!(test)),
    })
}

/// Whether the guards of view memory open a written block a 4 KiB part at
/// a time (`VICTORY_FINE_GUARDS=1`, with [`fine`]; off by default; on in
/// tests). Read once.
///
/// View memory has no write watch: a copy leaves each 64 KiB block it took
/// read-only, the title's first write faults, and the block is opened and
/// counts as written whole. With the switch the first write opens only its
/// 4 KiB part, and the block counts as written in that part; a write into
/// another part of it (a second fault) opens the rest, and it counts whole,
/// as before: at most two faults a block between guardings, against one.
/// Our own writes that name their bytes exactly (the merges of what the
/// card wrote, [`writing_exact`]) open just the parts they write, with no
/// fault. Writes the runtime announces for the kernel ([`writing`]) open
/// whole blocks, as before (the kernel's write must not meet a read-only
/// part, and the title's length may not be exact).
pub(crate) fn fine_guards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| fine() && (cfg!(test) || std::env::var_os("VICTORY_FINE_GUARDS").is_some_and(|v| v == "1")))
}

/// The journal's record of what was written: per 64 KiB page the asking
/// that last found it written, and with [`fine`] the same per 4 KiB part.
/// Every note goes through [`Written::mark`], so the two cannot disagree:
/// a page's asking is the latest of its parts'.
struct Written {
    /// Page number → the asking that last found it written.
    pages: BTreeMap<usize, u64>,
    /// With [`fine`]: page number → per part, the asking that last found it
    /// written (0: never since tracking began).
    subs: BTreeMap<usize, [u64; SUBS]>,
}

impl Written {
    const fn new() -> Self {
        Written { pages: BTreeMap::new(), subs: BTreeMap::new() }
    }

    /// Page `page` was found written at asking `now`, in the parts of
    /// `parts` (one bit each).
    ///
    /// Without [`fine`] the page takes `now`, as it always did. With it a
    /// page and a part keep the higher of what they hold and `now` (an
    /// asking noted late under an older number must not make a part look
    /// copied when it is not).
    fn mark(&mut self, page: usize, parts: u16, now: u64) {
        if !fine() {
            self.pages.insert(page, now);
            return;
        }
        let at = self.pages.entry(page).or_insert(now);
        *at = (*at).max(now);
        let subs = self.subs.entry(page).or_insert([0; SUBS]);
        for (i, sub) in subs.iter_mut().enumerate() {
            if parts >> i & 1 != 0 {
                *sub = (*sub).max(now);
            }
        }
    }

    /// The pages of `[first, last)` written since tracking began, with
    /// the asking that last found each written, in page order.
    fn range(&self, first: usize, last: usize) -> Vec<(usize, u64)> {
        self.pages.range(first..last).map(|(&p, &w)| (p, w)).collect()
    }

    /// Of page `page`, the parts last found written after asking `since`
    /// (all of them if `since` is 0, or if the parts are not kept).
    fn parts_after(&self, page: usize, since: u64) -> u16 {
        if since == 0 {
            return ALL_SUBS;
        }
        let Some(subs) = self.subs.get(&page) else {
            return ALL_SUBS;
        };
        // The page's own asking is the latest of its parts' (`mark`): if
        // it is not, something noted the page without its parts, and all
        // of them count.
        let page_at = self.pages.get(&page).copied().unwrap_or(0);
        if subs.iter().copied().max().unwrap_or(0) < page_at {
            return ALL_SUBS;
        }
        subs.iter().enumerate().filter(|&(_, &w)| w > since).fold(0u16, |m, (i, _)| m | 1 << i)
    }
}

/// The parts of `[at, at + len)` that lie in page `page`, as a mask.
pub(crate) fn parts_in(page: usize, at: usize, len: usize) -> u16 {
    let (lo, hi) = (page * PAGE, (page + 1) * PAGE);
    let (a, b) = (at.max(lo), at.saturating_add(len).min(hi));
    if a >= b {
        return 0;
    }
    let (first, last) = ((a - lo) / SUB, (b - lo).div_ceil(SUB));
    (first..last).fold(0u16, |m, i| m | 1 << i)
}

/// With [`fine`]: for each of `pages` (a 64 KiB page and the asking after
/// which a copy last took it, 0 for never), the parts written since then,
/// one bit each, in the same order; one look at the journal for them all.
/// Without it every part.
pub(crate) fn subs_after(pages: &[(usize, u64)], out: &mut Vec<u16>) {
    out.clear();
    if !fine() {
        out.extend(pages.iter().map(|_| ALL_SUBS));
        return;
    }
    let j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    out.extend(pages.iter().map(|&(page, since)| j.written.parts_after(page, since)));
}

/// Since the last frame line: bytes of title writes the askings noted, as
/// the kernel's 4 KiB pages, as the 64 KiB pages they lie in, as the
/// guards' 64 KiB blocks (view memory), and as the parts of those blocks
/// written ([`fine_guards`]; the blocks' bytes without it).
pub(crate) static NOTED: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// A title-memory range whose changed pages another subsystem needs to read.
pub(crate) struct Tracked {
    start: usize,
    end: usize,
    seen: Mutex<BTreeMap<usize, u64>>,
}

/// Tracks a runtime GPU-memory range, including physical-page views when
/// `views` is true. Other process memory is not tracked.
pub(crate) fn track(start: usize, len: usize, views: bool) -> Option<Tracked> {
    let end = start.checked_add(len)?;
    if len == 0 || writes(start, end, views).is_none() {
        return None;
    }
    Some(Tracked {
        start,
        end,
        seen: Mutex::default(),
    })
}

struct Journal {
    /// Numbers the askings; 0 is "never".
    asked: u64,
    /// What was written, and when.
    written: Written,
    /// Room for the kernel's answer.
    found: Vec<usize>,
    /// Page ranges asked ahead for a submission's refreshes ([`ask_ahead`])
    /// and the asking that did: `writes` answers for a range inside one
    /// from the journal, without another kernel call.
    ahead: Ahead,
    ahead_asked: u64,
    /// The ahead asking stands for a whole submission ([`ask_once`]): no
    /// other ahead asking is made, and `ahead` stays, until
    /// [`ask_once_end`].
    kept: bool,
}

static JOURNAL: Mutex<Journal> = Mutex::new(Journal {
    asked: 0,
    written: Written::new(),
    found: Vec::new(),
    ahead: Ahead::new(),
    ahead_asked: 0,
    kept: false,
});

/// The page ranges of the standing ahead asking, each asked of the kernel
/// (write-watched memory) or of the guards (view memory, with
/// [`views_once`]), and an index that answers "inside one of them?" without
/// looking at each: a submission asked ahead about dozens to hundreds of
/// ranges, and each of its thousands of refreshes looked through them all.
struct Ahead {
    /// `(first, last, view)`, in the order asked; with [`ahead_index`]
    /// sorted by `first` instead (kept so as each is pushed).
    runs: Vec<(usize, usize, bool)>,
    /// Sorted by `first`: `(first, the largest last of this entry and the
    /// ones before it, the same of write-watched ones only)`; rebuilt when
    /// `runs` changed since (with [`ahead_index`] kept current at each
    /// push).
    index: Vec<(usize, usize, usize)>,
    stale: bool,
}

/// Whether the index of the standing ahead asking is kept current as each
/// range is pushed (`VICTORY_AHEAD_INDEX=1`; off by default).
///
/// The index was rebuilt (every range cloned and sorted) at the first look
/// after any push, and a submission's one asking pushes a range and then
/// looks, range after range: some sixty rebuilds of up to a hundred ranges
/// a submission. With the switch a push puts the range in its place and
/// renews the index from there on, which answers every look the same; and
/// [`ask_ahead`] (each flush of a submission's refreshes) builds and sorts
/// nothing while the submission's one asking stands.
pub(crate) fn ahead_index() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_AHEAD_INDEX").is_some_and(|v| v == "1"))
}

impl Ahead {
    const fn new() -> Self {
        Ahead { runs: Vec::new(), index: Vec::new(), stale: false }
    }

    fn clear(&mut self) {
        self.runs.clear();
        self.index.clear();
        self.stale = false;
    }

    fn is_empty(&self) -> bool {
        self.runs.is_empty()
    }

    fn push(&mut self, first: usize, last: usize, view: bool) {
        self.push_as(first, last, view, ahead_index());
    }

    /// [`Ahead::push`]; `sorted`: as [`ahead_index`] says.
    fn push_as(&mut self, first: usize, last: usize, view: bool, sorted: bool) {
        if !sorted || self.stale {
            self.runs.push((first, last, view));
            self.stale = true;
            return;
        }
        // In its place, after the equal firsts; the entries from there on
        // take the largest lasts anew.
        let at = self.runs.partition_point(|r| r.0 <= first);
        self.runs.insert(at, (first, last, view));
        self.index.truncate(at);
        let (mut all, mut watched) = self.index.last().map_or((0, 0), |&(_, a, w)| (a, w));
        for &(a, b, view) in &self.runs[at..] {
            all = all.max(b);
            if !view {
                watched = watched.max(b);
            }
            self.index.push((a, all, watched));
        }
    }

    /// Whether pages `[first, last)` lie inside one range asked (with
    /// `views` false, one asked of the kernel): what `runs.iter().any(..)`
    /// says.
    fn covers(&mut self, first: usize, last: usize, views: bool) -> bool {
        if self.stale {
            let mut sorted: Vec<(usize, usize, bool)> = self.runs.clone();
            sorted.sort_unstable_by_key(|r| r.0);
            self.index.clear();
            let (mut all, mut watched) = (0, 0);
            for &(a, b, view) in &sorted {
                all = all.max(b);
                if !view {
                    watched = watched.max(b);
                }
                self.index.push((a, all, watched));
            }
            // With the switch the runs stay sorted from here on.
            if ahead_index() {
                self.runs = sorted;
            }
            self.stale = false;
        }
        let n = self.index.partition_point(|r| r.0 <= first);
        n > 0 && {
            let (_, all, watched) = self.index[n - 1];
            last <= if views { all } else { watched }
        }
    }
}

/// Since the last frame line, where asking which pages were written goes,
/// on all threads ([`split_line`]): counts and nanoseconds.
#[derive(Clone, Copy)]
enum S {
    /// [`writes`] and [`asked`] answered from the journal (a range inside
    /// the standing ahead asking).
    JournalN,
    JournalNs,
    /// [`writes`] asking the kernel, and finding the memory watched.
    KernelN,
    KernelNs,
    /// [`writes`] asking the kernel about memory that is not watched (view
    /// memory, or a range across reservations): the call fails.
    ProbeN,
    ProbeNs,
    /// [`writes`] asking the guards of view memory; of those, the ranges
    /// found all clean at once, and the blocks of the others walked.
    GuardN,
    GuardNs,
    GuardClean,
    GuardBlocks,
    /// Guarding again (`VirtualProtect`), and asking Windows what memory
    /// is (`VirtualQuery`: is it a view, how far is it uncommitted).
    ProtectN,
    ProtectNs,
    QueryN,
    QueryNs,
    /// Waiting for the journal's lock in [`writes`].
    LockNs,
    /// Of `ProbeN`, those not asked: known unwatched ([`unwatched_cache`]).
    ProbeSkippedN,
    /// Failed askings looked into ([`learn_unwatched`]).
    LearnN,
    /// Protections that failed (a run across views, or not all mapped).
    ProtectFailedN,
    /// Questions about unmapped blocks not asked ([`trust_placeholders`]).
    TrustedN,
    /// Title writes taken by the guards' handler.
    Faults,
    /// The ahead askings ([`ask_ahead`], [`ask_once`]): of the kernel, the
    /// ones that failed (not watched, or across reservations), of the
    /// guards ([`views_once`]), and noting what they found in the journal.
    AheadN,
    AheadNs,
    AheadFailedN,
    AheadFailedNs,
    /// Of `AheadFailedN`, those not asked: known unwatched.
    AheadSkippedN,
    AheadGuardN,
    AheadGuardNs,
    NoteNs,
    /// The guards of a page's other views ([`view_guard`]), when a block is
    /// guarded again: all of it (looking the views up, and guarding), and
    /// the protects it made.
    AliasNs,
    AliasProtectN,
    AliasProtectNs,
}
const SPLITS: usize = S::AliasProtectNs as usize + 1;
static SPLIT: [AtomicU64; SPLITS] = [const { AtomicU64::new(0) }; SPLITS];

fn tally(s: S, n: u64) {
    SPLIT[s as usize].fetch_add(n, Relaxed);
}

/// Adds the nanoseconds since `since` to `s`.
fn tally_since(s: S, since: std::time::Instant) {
    tally(s, since.elapsed().as_nanos() as u64);
}

/// The frame line's account of the askings ([`S`]), per frame, taken.
pub(crate) fn split_line(frames: f64) -> String {
    let v: Vec<f64> = SPLIT.iter().map(|a| a.swap(0, Relaxed) as f64 / frames.max(1.0)).collect();
    let n = |s: S| v[s as usize];
    let ms = |s: S| v[s as usize] / 1e6;
    format!(
        "per range, all threads: {:.0} from the journal {:.1} ms, {:.0} kernel {:.1} ms, {:.0} not watched {:.1} ms \
         ({:.0} known, not asked), {:.0} guards {:.1} ms ({:.0} all clean, {:.0} blocks walked, {:.0} protects {:.1} ms \
         ({:.0} failed), {:.0} queries {:.1} ms, {:.0} not asked in placeholders), lock waits {:.1} ms, {:.0} guard faults, \
         {:.0} failures looked into; ahead: {:.0} kernel {:.1} ms, {:.0} failed {:.1} ms ({:.0} known, not asked), \
         {:.0} view guards {:.1} ms, noting {:.1} ms; other views of guarded pages {:.1} ms ({:.0} protects {:.1} ms, \
         VICTORY_ALIAS_RUNS {})",
        n(S::JournalN),
        ms(S::JournalNs),
        n(S::KernelN),
        ms(S::KernelNs),
        n(S::ProbeN),
        ms(S::ProbeNs),
        n(S::ProbeSkippedN),
        n(S::GuardN),
        ms(S::GuardNs),
        n(S::GuardClean),
        n(S::GuardBlocks),
        n(S::ProtectN),
        ms(S::ProtectNs),
        n(S::ProtectFailedN),
        n(S::QueryN),
        ms(S::QueryNs),
        n(S::TrustedN),
        ms(S::LockNs),
        n(S::Faults),
        n(S::LearnN),
        n(S::AheadN),
        ms(S::AheadNs),
        n(S::AheadFailedN),
        ms(S::AheadFailedNs),
        n(S::AheadSkippedN),
        n(S::AheadGuardN),
        ms(S::AheadGuardNs),
        ms(S::NoteNs),
        ms(S::AliasNs),
        n(S::AliasProtectN),
        ms(S::AliasProtectNs),
        if alias_runs() { "on" } else { "off" },
    )
}

/// Notes `found` (the kernel's answer: addresses of written pages of
/// `granularity` bytes, 4 KiB, in order) in the journal as written at
/// asking `now`: once per 64 KiB page, not once per 4 KiB page of it
/// (sixteen map inserts where one does; the answer is in address order, so
/// a page's entries are next to each other), with the parts written
/// ([`fine`]).
fn note(written: &mut Written, found: &[usize], granularity: usize, now: u64) {
    // The kernel says 4 KiB; less (or nothing) is taken as 4 KiB, which
    // marks more, not less.
    let granularity = granularity.max(KERNEL_PAGE);
    let mut last = usize::MAX;
    let mut parts = 0u16;
    let mut pages = 0u64;
    for &at in found {
        // A kernel page of 4 KiB lies in one 64 KiB page; a bigger one
        // (none is known) marks every part it holds.
        for page in at / PAGE..at.saturating_add(granularity).div_ceil(PAGE) {
            if page != last {
                if last != usize::MAX {
                    written.mark(last, parts, now);
                }
                last = page;
                parts = 0;
                pages += 1;
            }
            parts |= parts_in(page, at, granularity);
        }
    }
    if last != usize::MAX {
        written.mark(last, parts, now);
    }
    NOTED[0].fetch_add((found.len() * granularity) as u64, Relaxed);
    NOTED[1].fetch_add(pages * PAGE as u64, Relaxed);
}

/// Notes `pages` written at asking `now` as the guards report them: each
/// page with the parts written (all of them but with [`fine_guards`]).
fn note_guarded(written: &mut Written, pages: impl IntoIterator<Item = (usize, u16)>, now: u64) {
    let (mut n, mut parts) = (0u64, 0u64);
    for (page, mask) in pages {
        written.mark(page, mask, now);
        n += 1;
        parts += u64::from(mask.count_ones());
    }
    NOTED[2].fetch_add(n * PAGE as u64, Relaxed);
    NOTED[3].fetch_add(parts * SUB as u64, Relaxed);
}

/// Ranges of one submission closer than this are asked about together:
/// walking the pages between them costs less than another kernel call (a
/// call that resets written pages costs 30-70 microseconds whatever the
/// range).
const AHEAD_GAP: usize = (8 << 20) / PAGE;
/// The most one asking covers (its answer needs 8 bytes per 4 KiB page).
const AHEAD_MAX: usize = (1 << 30) / PAGE;

/// Asks the kernel about many ranges `(start, end)` at once, before the
/// [`writes`] of each: ranges close together are one asking, and `writes`
/// then answers from the journal until [`ask_ahead_done`]. The world's
/// submissions refresh thousands of host copies (6,664 askings a frame,
/// 456 ms, in run 14). Ranges that are not write-watched, or lie alone, are
/// left to `writes`.
pub(crate) fn ask_ahead(ranges: &[(usize, usize)]) {
    // The submission's one asking stands: nothing is asked (found below,
    // after the pieces were sorted for nothing, without the switch).
    if ahead_index() && kept() {
        return;
    }
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    if pieces.len() < 2 {
        return;
    }
    pieces.sort_unstable();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // The submission's one asking stands ([`ask_once`]): what it does not
    // hold is left to `writes`.
    if j.kept {
        return;
    }
    // View memory the guards hold is left to `writes`, which answers it
    // from the guards ([`known_view`]): the kernel's write watch can only
    // fail there. Asked here with the others, a run holding such a range
    // failed, was asked again by halves down to single ranges, failing
    // each time, and the range then went to `writes` all the same (after
    // triippywhale d0581a55). `VICTORY_VIEW_ASK=1` asks as before.
    #[cfg(feature = "diagnostics")]
    let all = pieces.len();
    pieces.retain(|&(first, _)| !known_view(first * PAGE));
    #[cfg(feature = "diagnostics")]
    AHEAD_VIEWS.fetch_add((all - pieces.len()) as u64, std::sync::atomic::Ordering::Relaxed);
    j.asked += 1;
    j.ahead_asked = j.asked;
    j.ahead.clear();
    let mut i = 0;
    while i < pieces.len() {
        let first = pieces[i].0;
        let mut last = pieces[i].1;
        let mut k = i + 1;
        while k < pieces.len()
            && pieces[k].0 <= last + AHEAD_GAP
            && last.max(pieces[k].1) - first <= AHEAD_MAX
        {
            last = last.max(pieces[k].1);
            k += 1;
        }
        ask_run(j, &pieces[i..k]);
        i = k;
    }
}

/// Since the last frame line: the kernel askings that took a millisecond
/// or more, their microseconds, and the longest asking's. An asking walks
/// page tables, some ten microseconds for most ranges; in run 75 the 2,644
/// askings of a frame took 219 ms in the lines where the pass over the
/// heap's views ran long, 25-50 ms in the others. These numbers say
/// whether a few askings wait for something or all of them are slow.
#[cfg(feature = "diagnostics")]
static SLOW: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

#[cfg(feature = "diagnostics")]
fn slow(us: u64) {
    if us >= 1000 {
        SLOW[0].fetch_add(1, Relaxed);
        SLOW[1].fetch_add(us, Relaxed);
    }
    SLOW[2].fetch_max(us, Relaxed);
}

/// Since the last frame line: ranges of view memory the guards hold that a
/// submission's askings ahead ([`ask_ahead`], [`ask_once`]) left out. Each
/// was a kernel asking that failed, or several, before.
#[cfg(feature = "diagnostics")]
pub(crate) static AHEAD_VIEWS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// [`SLOW`], taken.
#[cfg(feature = "diagnostics")]
pub(crate) fn slow_asks() -> [u64; 3] {
    [0, 1, 2].map(|i| SLOW[i].swap(0, std::sync::atomic::Ordering::Relaxed))
}

/// Whether the submission's one asking stands ([`ask_once`]): an
/// [`ask_ahead`] now asks nothing.
pub(crate) fn kept() -> bool {
    JOURNAL.lock().unwrap_or_else(|e| e.into_inner()).kept
}

/// The submission's refreshes are done: `writes` asks the kernel again.
pub(crate) fn ask_ahead_done() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept {
        j.ahead.clear();
    }
}

/// Whether a submission asks the kernel once, before its first refresh,
/// about every range its lists refresh (`VICTORY_ASK_ONCE=1`; off by
/// default).
///
/// A submission's refreshes are done in groups, between the title's own
/// buffer copies (each copy flushes the refreshes recorded before it), and
/// every group asked the kernel again: 2,644 askings a frame, 70 a
/// submission. With the switch the journal answers all of them from one
/// asking made when the submission starts.
///
/// What changes: memory the title's other threads write while the
/// submission is being prepared is seen by the next submission's asking,
/// not by this one's later groups. What this submission's lists read was
/// written before they were submitted, so before the asking. Our own
/// writes into title memory while preparing are told to the journal
/// ([`wrote`]) or end the asking ([`ask_once_end`]).
pub(crate) fn once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default since run 81 (with card buffers: 375-450 ms a frame
    // against 573-647, the picture unchanged over four minutes);
    // `VICTORY_ASK_ONCE=0` asks per range, as before.
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE").is_none_or(|v| v != "0"))
}

/// With [`once`]: asks about `ranges` (everything a submission's lists
/// refresh) as [`ask_ahead`] does, and lets that asking stand until
/// [`ask_once_end`].
pub(crate) fn ask_once(ranges: &[(usize, usize)]) {
    if !once() {
        return;
    }
    // A submission's asking ([`hot_blocks`] counts them).
    ROUND.fetch_add(1, Relaxed);
    ask_once_end();
    ask_ahead(ranges);
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    if j.ahead.is_empty() {
        // `ask_ahead` asked nothing (fewer than two ranges).
        j.asked += 1;
        j.ahead_asked = j.asked;
    }
    // `ask_ahead` leaves a range that lies alone to `writes`, which asks
    // once per flush: with one asking for the submission that was every
    // such range asked again at each of its 70 flushes (run 76: 2,300
    // askings a frame with the switch on, as without it). They are asked
    // here, once; what is not write-watched stays with `writes`.
    for &(start, end) in ranges {
        if start >= end {
            continue;
        }
        // View memory the guards hold is not asked about: the asking can
        // only fail there, one failed kernel call for each such copy at
        // every submission, and `writes` answers it from the guards
        // (after triippywhale cc368c7c).
        if known_view(start) {
            #[cfg(feature = "diagnostics")]
            AHEAD_VIEWS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            continue;
        }
        let (first, last) = (start / PAGE, end.div_ceil(PAGE));
        if !j.ahead.covers(first, last, true) && ask_kernel(j, first, last) {
            j.ahead.push(first, last, false);
        }
    }
    j.kept = !j.ahead.is_empty();
}

/// Whether the submission's one asking ([`once`]) takes in the view memory
/// its lists refresh as well (`VICTORY_ASK_ONCE_VIEWS=1`; off by default).
///
/// View memory (xmem's physical-page views) has no write watch: each of
/// its refreshes failed to ask the kernel and asked the guards, once per
/// flush, some 70 flushes a submission, ~18 ms a frame in the guards in the
/// opening cutscene. With the switch each such range is guarded once when
/// the submission starts ([`ask_once_views`]) and the journal answers its
/// refreshes, as it does for write-watched memory: what the title's other
/// threads write while the submission is prepared is seen by the next
/// submission's asking (the write faults and opens the block, as it sets
/// the kernel's bit in watched memory), our own writes are told to the
/// journal or end the asking, as for watched memory ([`once`]).
pub(crate) fn views_once() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASK_ONCE_VIEWS").is_some_and(|v| v == "1"))
}

/// With [`views_once`], right after [`ask_once`]: the ranges the
/// submission refreshes (each as one refresh asks, cut to its copy) that
/// the asking did not take in are asked now as [`writes`] would ask them:
/// the kernel, else the guards of view memory; what either says goes into
/// the journal under the standing asking, and the refreshes inside them
/// answer from it. What neither takes is left to `writes`.
pub(crate) fn ask_once_views(ranges: &[(usize, usize)]) {
    if once() && views_once() && !ranges.is_empty() {
        ask_views(ranges);
    }
}

/// [`ask_once_views`], whatever the switches say ([`views_trim`] as set).
fn ask_views(ranges: &[(usize, usize)]) {
    ask_views_as(ranges, views_trim());
}

/// [`ask_views`], with [`views_trim`] or without (`trim`).
fn ask_views_as(ranges: &[(usize, usize)], trim: bool) {
    let mut pieces: Vec<(usize, usize)> = ranges
        .iter()
        .filter(|r| r.0 < r.1)
        .map(|&(start, end)| (start / PAGE, end.div_ceil(PAGE)))
        .collect();
    // In order of start, the longest first: a range inside one asked
    // before it is not asked again.
    pieces.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(b.1.cmp(&a.1)));
    pieces.dedup();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    // `ask_once` let its asking go if it held nothing: this one stands
    // under a number of its own.
    if !j.kept {
        j.asked += 1;
        j.ahead_asked = j.asked;
        j.ahead.clear();
    }
    let mut reach = 0;
    // Ranges the guards know to be views (their first block is guarded or
    // absent): asked below, joined where they overlap or touch.
    let mut known = Vec::new();
    // With [`views_trim`]: the farthest end of those, and the pieces passed
    // over.
    let (mut known_reach, mut passed) = (0, 0u64);
    for (first, last) in pieces {
        if last <= reach {
            continue;
        }
        // Inside a range the guards know (the one reaching farthest starts
        // at or before it): the union below takes it in.
        if trim && last <= known_reach {
            passed += 1;
            continue;
        }
        if j.ahead.covers(first, last, true) {
            continue;
        }
        // A range whose first block the guards hold cannot be write-watched
        // (the kernel's asking fails there): no asking first.
        if trim && guards_know(first) {
            known.push((first, last));
            known_reach = known_reach.max(last);
            continue;
        }
        if ask_kernel(j, first, last) {
            j.ahead.push(first, last, false);
            reach = reach.max(last);
            continue;
        }
        if !trim && guards_know(first) {
            known.push((first, last));
            continue;
        }
        // The guards decide whether it is a view, as for `writes`.
        if guard_ahead(j, first, last) {
            reach = reach.max(last);
        }
    }
    // A submission refreshes many overlapping ranges of the same views
    // (play1: ~1,000 view guards a frame, each walking the holes of its
    // range): their union is walked once, each block once. The guards say
    // of the union what they would say of its ranges one after another,
    // all of whose first blocks they know.
    let mut i = 0;
    while i < known.len() {
        let (first, mut last) = known[i];
        let mut k = i + 1;
        while k < known.len() && known[k].0 <= last {
            last = last.max(known[k].1);
            k += 1;
        }
        if !j.ahead.covers(first, last, true) {
            guard_ahead(j, first, last);
        }
        i = k;
    }
    j.kept = !j.ahead.is_empty();
    if trim {
        TRIM[0].fetch_add(passed, Relaxed);
        TRIM[1].fetch_add(known.len() as u64, Relaxed);
    }
}

/// Whether the submission's asking of view memory ([`ask_once_views`])
/// passes over a range inside one the guards already know to be a view,
/// and asks the guards before the kernel about a range whose first block
/// they hold (`VICTORY_VIEWS_TRIM=1`; off by default).
///
/// The asking sorts the submission's ~800 refreshed ranges and looked at
/// each that no earlier asking took in: the standing asking's index, the
/// kernel (its "known unwatched" cache, under its lock) and then the
/// guards (under theirs), some 6,700 ranges a frame (n28: "6677 failed
/// ... known, not asked") for some 1,660 unions guarded. A range inside one
/// the guards know is inside the union walked for it, and the kernel's
/// asking about a range that starts in a block the guards hold can only
/// fail; what is guarded and noted is the same.
fn views_trim() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEWS_TRIM").is_some_and(|v| v == "1"))
}

/// Since the last frame line, with [`views_trim`]: ranges passed over, and
/// ranges the guards knew (asked in unions).
static TRIM: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The frame line's account of [`views_trim`] and [`hot_blocks`], per
/// frame, taken (the "further:" part).
pub(crate) fn asking_cost_line(frames: f64) -> String {
    let f = frames.max(1.0);
    let take = |a: &AtomicU64| a.swap(0, Relaxed) as f64 / f;
    let (passed, known) = (take(&TRIM[0]), take(&TRIM[1]));
    let (reported, made, again) = (take(&HOT[0]), take(&HOT[1]), take(&HOT[2]));
    let open = guards().hot.open.len();
    format!(
        "views (VICTORY_VIEWS_TRIM {}): {passed:.0} ranges passed over, {known:.0} known; hot blocks (VICTORY_HOT_BLOCKS={}): \
         {reported:.0} reported with no guard, {made:.0} left open, {again:.0} guarded again, {open} open now",
        if views_trim() { "on" } else { "off" },
        hot_blocks(),
    )
}

/// [`ask_views`]: the guards asked about blocks `[first, last)`, what they
/// report noted under the standing asking; whether the range is a view
/// (and now asked ahead).
fn guard_ahead(j: &mut Journal, first: usize, last: usize) -> bool {
    let guarding = std::time::Instant::now();
    let written = guard_parts(first * PAGE, last * PAGE);
    tally(S::AheadGuardN, 1);
    tally_since(S::AheadGuardNs, guarding);
    let Some(written) = written else {
        return false;
    };
    // With the parts each block's guard opened ([`fine_guards`]).
    let now = j.ahead_asked;
    note_guarded(&mut j.written, written, now);
    j.ahead.push(first, last, true);
    true
}

/// The submission's one asking stands no longer: its refreshes are done,
/// or title memory was written in a way the journal was not told.
pub(crate) fn ask_once_end() {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if j.kept {
        j.kept = false;
        j.ahead.clear();
    }
}

/// We wrote title memory `[at, at + len)` while a submission's one asking
/// stands (the title's buffer copies, done at submission): its pages count
/// as written after that asking, so the refreshes that follow copy them.
pub(crate) fn wrote(at: usize, len: usize) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    if !j.kept || len == 0 {
        return;
    }
    j.asked += 1;
    let now = j.asked;
    // The asking that stands moves with it: `asked` and `writes` give it as
    // "now", and a page noted written after "now" stays stale for every
    // copy that takes it (copied at `now`, written at `now + 1`): copied
    // again at every refresh, and a reason to settle flights
    // (fxceless/victory-fork#410).
    j.ahead_asked = now;
    for page in at / PAGE..at.saturating_add(len).div_ceil(PAGE) {
        j.written.mark(page, parts_in(page, at, len), now);
    }
}

/// We wrote the 64 KiB page numbered `page` of title memory through a
/// view of our own (`crate::inplace::write_own`): no guard was opened and
/// no kernel write-watch bit set, so the journal is told here. The page,
/// and the same physical page at its other title addresses (`others`,
/// page numbers), count as written at a new asking, which every host copy
/// that holds them takes at its next refresh. Returns the asking that had
/// last found `page` written (0: never) and the new one, for the copy
/// that wrote to note itself current.
pub(crate) fn wrote_own(page: usize, others: &[usize]) -> (u64, u64) {
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    j.asked += 1;
    let now = j.asked;
    // As in [`wrote`]: an asking that stands for a submission, or for a
    // flush, moves with it, or a copy taking the page at the standing
    // asking would find it written "later" at every refresh.
    if j.kept || !j.ahead.is_empty() {
        j.ahead_asked = now;
    }
    // The page's asking before (its parts' latest), then all of it and
    // the other pages marked written now ([`Written::mark`]: whole, as the
    // write's own parts are not known here).
    let before = j.written.pages.get(&page).copied().unwrap_or(0);
    j.written.mark(page, ALL_SUBS, now);
    for &other in others {
        if other != page {
            j.written.mark(other, ALL_SUBS, now);
        }
    }
    (before, now)
}

/// One kernel asking for `pieces` (page ranges in order); by halves if it
/// fails, as a run across two reservations or into memory that is not
/// write-watched does. A piece alone is left to [`writes`].
fn ask_run(j: &mut Journal, pieces: &[(usize, usize)]) {
    if pieces.len() < 2 {
        return;
    }
    let first = pieces[0].0;
    let last = pieces.iter().map(|p| p.1).max().unwrap_or(first);
    if ask_kernel(j, first, last) {
        j.ahead.push(first, last, false);
        return;
    }
    let (a, b) = pieces.split_at(pieces.len() / 2);
    ask_run(j, a);
    ask_run(j, b);
}

/// Asks the kernel for the written pages of pages `[first, last)` and notes
/// them in the journal under the ahead asking; false (and nothing changed)
/// if the range is not write-watched memory of one reservation.
fn ask_kernel(j: &mut Journal, first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let started = std::time::Instant::now();
    if known_unwatched(first, last) {
        tally(S::AheadFailedN, 1);
        tally(S::AheadSkippedN, 1);
        tally_since(S::AheadFailedNs, started);
        return false;
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    // SAFETY: the answer buffer holds `count` entries.
    let failed = unsafe {
        GetWriteWatch(
            WRITE_WATCH_FLAG_RESET,
            (first * PAGE) as *const _,
            (last - first) * PAGE,
            j.found.as_mut_ptr().cast(),
            &mut count,
            &mut granularity,
        )
    } != 0;
    if failed {
        learn_unwatched(first, last);
        tally(S::AheadFailedN, 1);
        tally_since(S::AheadFailedNs, started);
        return false;
    }
    tally(S::AheadN, 1);
    tally_since(S::AheadNs, started);
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    let noting = std::time::Instant::now();
    note(&mut j.written, &j.found[..count], granularity as usize, j.ahead_asked);
    tally_since(S::NoteNs, noting);
    true
}

/// What [`writes`] would say of `[start, end)` from the journal alone, if
/// the range lies in one asked ahead ([`ask_ahead`], [`ask_once`]): the
/// pages written since tracking began and the asking that last found each
/// written, with the number of the asking that stands (as [`writes`] gives
/// it). `None` if the kernel would have to be asked.
pub(crate) fn asked(start: usize, end: usize) -> Option<(u64, Vec<(usize, u64)>)> {
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let started = std::time::Instant::now();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    let j = &mut *j;
    let answer = j
        .ahead
        .covers(first, last, true)
        .then(|| (j.ahead_asked, j.written.range(first, last)));
    if answer.is_some() {
        tally(S::JournalN, 1);
        tally_since(S::JournalNs, started);
    }
    answer
}

/// The pages of `[start, end)` (page numbers, in order) written since they
/// were last taken, as far as can be told without taking them: in view
/// memory the guards hold, the blocks a write opened since they were last
/// guarded (and those never guarded but not noted absent); in
/// write-watched memory, the kernel's bits, not reset. Nothing is guarded
/// again and no bit is reset, so the title's writes fault no more than
/// without it. A hint only (for `crate::descriptor`'s pre-read): what the
/// journal holds of earlier askings is not looked at, and memory neither
/// guarded nor watched gives nothing.
pub(crate) fn peek_written(start: usize, end: usize) -> Vec<usize> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    if first >= last {
        return Vec::new();
    }
    {
        let g = guards();
        let clean = g.clean.within(first, last);
        if !clean.is_empty() || g.parts.range(first..last).next().is_some() {
            let mut open = Vec::new();
            let mut at = first;
            for (a, b) in clean.into_iter().chain(std::iter::once((last, last))) {
                open.extend((at..a).filter(|&block| !g.absent.contains(block)));
                at = b;
            }
            return open;
        }
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    let mut found = vec![0usize; count];
    let mut granularity = 0;
    // SAFETY: the answer buffer holds `count` entries; flags 0: no reset.
    let failed = unsafe {
        GetWriteWatch(0, (first * PAGE) as *const _, (last - first) * PAGE, found.as_mut_ptr().cast(), &mut count, &mut granularity)
    } != 0;
    if failed {
        return Vec::new();
    }
    let mut pages: Vec<usize> = found[..count].iter().map(|&at| at / PAGE).collect();
    pages.dedup();
    pages
}

/// The pages of `[start, end)` (whole pages) as last written, for pages
/// written since tracking began: `(page number, asking)` in page order, and
/// this asking's number, which is higher than any in the list but those the
/// kernel just reported. `None` if the memory is not write-watched (nor,
/// with `guarded`, view memory, which is then guarded).
pub(crate) fn writes(start: usize, end: usize, guarded: bool) -> Option<(u64, Vec<(usize, u64)>)> {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    const WRITE_WATCH_FLAG_RESET: u32 = 1;
    let (first, last) = (start / PAGE, end.div_ceil(PAGE));
    let locking = std::time::Instant::now();
    let mut j = JOURNAL.lock().unwrap_or_else(|e| e.into_inner());
    tally_since(S::LockNs, locking);
    let j = &mut *j;
    // Asked ahead with the submission's other ranges: the journal has it.
    // View memory asked ahead ([`views_once`]) answers only an asker that
    // would ask the guards.
    let started = std::time::Instant::now();
    if j.ahead.covers(first, last, guarded) {
        let answer = Some((j.ahead_asked, j.written.range(first, last)));
        tally(S::JournalN, 1);
        tally_since(S::JournalNs, started);
        return answer;
    }
    let mut count = (last - first) * PAGE / KERNEL_PAGE;
    j.found.resize(count.max(j.found.len()), 0);
    let mut granularity = 0;
    // View memory the guards already hold cannot be write-watched: the
    // kernel asking would only fail. Copies of it are flushed many times a
    // frame, each time paying that failed call before the guards answer.
    // (`known_unwatched` knows it too once learned; this needs no learning.)
    let known_view = guarded && known_view(first * PAGE);
    let known = known_view || known_unwatched(first, last);
    // SAFETY: the answer buffer holds `count` entries.
    let failed = known
        || unsafe {
            GetWriteWatch(
                WRITE_WATCH_FLAG_RESET,
                (first * PAGE) as *const _,
                (last - first) * PAGE,
                j.found.as_mut_ptr().cast(),
                &mut count,
                &mut granularity,
            )
        } != 0;
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(
        if known_view {
            crate::copystat::C::WatchKnownView
        } else {
            crate::copystat::C::WatchFailed
        },
        failed as u64,
    );
    #[cfg(feature = "diagnostics")]
    if failed && !known_view && guarded {
        failed_asking(first * PAGE, last * PAGE);
    }
    let guarded = if failed {
        count = 0;
        if known {
            tally(S::ProbeSkippedN, 1);
        } else {
            learn_unwatched(first, last);
        }
        tally(S::ProbeN, 1);
        tally_since(S::ProbeNs, started);
        if !guarded {
            return None;
        }
        let guarding = std::time::Instant::now();
        let written = guard_parts(first * PAGE, last * PAGE);
        tally(S::GuardN, 1);
        tally_since(S::GuardNs, guarding);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::GuardUs, guarding.elapsed().as_micros() as u64);
        written?
    } else {
        tally(S::KernelN, 1);
        tally_since(S::KernelNs, started);
        Vec::new()
    };
    #[cfg(feature = "diagnostics")]
    {
        use crate::copystat::{C, add};
        add(C::Watches, 1);
        add(C::WatchUs, started.elapsed().as_micros() as u64);
        slow(started.elapsed().as_micros() as u64);
    }
    j.asked += 1;
    let now = j.asked;
    let noting = std::time::Instant::now();
    note(&mut j.written, &j.found[..count], granularity as usize, now);
    note_guarded(&mut j.written, guarded, now);
    tally_since(S::NoteNs, noting);
    Some((now, j.written.range(first, last)))
}

/// Diagnostics: one in 2,048 of the askings the kernel's write watch
/// refused for memory the guards did not know (some 200 a frame in the
/// first rooms, each followed by a question to Windows about the memory):
/// what the memory is and what the guards hold of its first block.
#[cfg(feature = "diagnostics")]
fn failed_asking(start: usize, end: usize) {
    use windows_sys::Win32::System::Memory::{MEMORY_BASIC_INFORMATION, VirtualQuery};
    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 2048 != 0 {
        return;
    }
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(start as *const _, &mut info, size_of_val(&info)) };
    let (held, clean, absent, viewed, aliased) = {
        let g = guards();
        let b = start / BLOCK;
        (g.held.contains(&b), g.clean.contains(b), g.absent.contains(b), g.viewed.contains(&b), g.aliased.contains(&b))
    };
    tracing::info!(
        "a failed write-watch asking (one in 2048): {start:#x}+{:#x}; Windows says (answered {n}): base {:#x} region {:#x} state {:#x} type {:#x} protect {:#x} allocation protect {:#x}; \
         the guards of its first block: held {held} clean {clean} absent {absent} viewed {viewed} aliased {aliased}; is a view: {}",
        end - start,
        info.BaseAddress as usize,
        info.RegionSize,
        info.State,
        info.Type,
        info.Protect,
        info.AllocationProtect,
        is_view(start, end)
    );
}

impl Tracked {
    /// Takes changed pages of `[from, to)`, cut to the tracked range. A
    /// partial page stays changed so a later take can read its other bytes.
    pub(crate) fn take(&self, from: usize, to: usize) -> Vec<(usize, usize)> {
        let (from, to) = (from.max(self.start), to.min(self.end));
        if from >= to {
            return Vec::new();
        }
        let Some((_, pages)) = writes(from, to, true) else {
            return Vec::new();
        };
        let mut seen = self.seen.lock().unwrap_or_else(|e| e.into_inner());
        let mut ranges: Vec<(usize, usize)> = Vec::new();
        for (page, write) in pages {
            if seen.get(&page).is_some_and(|&last| last >= write) {
                continue;
            }
            let (start, end) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
            if start >= end {
                continue;
            }
            // Keep partial pages dirty: the caller has not taken the rest.
            if (page * PAGE).max(self.start) >= from
                && ((page + 1) * PAGE).min(self.end) <= to
            {
                seen.insert(page, write);
            }
            match ranges.last_mut() {
                Some((_, last)) if *last == start => *last = end,
                _ => ranges.push((start, end)),
            }
        }
        ranges
    }
}

/// Guarded blocks of view memory.
const BLOCK: usize = PAGE;

/// A set of block numbers kept as runs, `first → last` (exclusive), apart
/// and in order. The guards kept sets of single blocks, and a frame asks
/// about some 150,000 blocks of view memory (run 54): every asking read its
/// range's blocks out of the set one by one to see whether all were clean,
/// and walked the blocks of a range that was not, looking each up in the
/// set of absent blocks. With runs, "all clean?" is one lookup and a
/// range's members come a run at a time; what the guards decide is the
/// same.
struct Runs(BTreeMap<usize, usize>);

impl Runs {
    const fn new() -> Self {
        Runs(BTreeMap::new())
    }

    fn contains(&self, block: usize) -> bool {
        self.0.range(..=block).next_back().is_some_and(|(_, &last)| block < last)
    }

    /// Whether any block of `[first, last)` is in the set.
    fn meets(&self, first: usize, last: usize) -> bool {
        first < last && self.0.range(..last).next_back().is_some_and(|(_, &end)| end > first)
    }

    /// The end of the run holding `block`, if it is in the set.
    fn run_end(&self, block: usize) -> Option<usize> {
        self.0.range(..=block).next_back().filter(|&(_, &end)| block < end).map(|(_, &end)| end)
    }

    /// Whether every block of `[first, last)` (not empty) is in the set.
    fn covers(&self, first: usize, last: usize) -> bool {
        self.0.range(..=first).next_back().is_some_and(|(_, &end)| last <= end)
    }

    /// The members of `[first, last)`, as runs cut to it, in order.
    fn within(&self, first: usize, last: usize) -> Vec<(usize, usize)> {
        let mut out = Vec::new();
        if first >= last {
            return out;
        }
        if let Some((_, &end)) = self.0.range(..first).next_back()
            && end > first
        {
            out.push((first, end.min(last)));
        }
        out.extend(self.0.range(first..last).map(|(&a, &b)| (a, b.min(last))));
        out
    }

    fn insert(&mut self, mut first: usize, mut last: usize) {
        if first >= last {
            return;
        }
        // A run that reaches `first` (or touches it) joins.
        if let Some((&a, &b)) = self.0.range(..=first).next_back()
            && b >= first
        {
            first = a;
            last = last.max(b);
        }
        // As do the runs that start inside or at `last`.
        let joined: Vec<(usize, usize)> = self.0.range(first..=last).map(|(&a, &b)| (a, b)).collect();
        for (a, b) in joined {
            self.0.remove(&a);
            last = last.max(b);
        }
        self.0.insert(first, last);
    }

    fn remove(&mut self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        if let Some((&a, &b)) = self.0.range(..first).next_back()
            && b > first
        {
            self.0.insert(a, first);
            if b > last {
                self.0.insert(last, b);
            }
        }
        let cut: Vec<(usize, usize)> = self.0.range(first..last).map(|(&a, &b)| (a, b)).collect();
        for (a, b) in cut {
            self.0.remove(&a);
            if b > last {
                self.0.insert(last, b);
            }
        }
    }
}

/// What is known of title memory's kind, for asking the kernel only where
/// it can answer ([`unwatched_cache`]).
struct Kinds {
    /// Blocks known not to be write-watched: the runtime's placeholder
    /// reservations (where the views are), and blocks a failed asking was
    /// found to lie in.
    unwatched: Runs,
    /// Block ranges found not to lie in one write-watched reservation
    /// though their ends do (across two reservations), `first → last`;
    /// none lies inside another. A range holding one fails as well.
    across: BTreeMap<usize, usize>,
    /// The runtime's placeholder reservations ([`title_memory_reserved`]):
    /// a block in one is committed only when the runtime maps it.
    placeholders: Runs,
}

static KINDS: Mutex<Kinds> = Mutex::new(Kinds {
    unwatched: Runs::new(),
    across: BTreeMap::new(),
    placeholders: Runs::new(),
});

/// Taken last: no other lock is taken while it is held.
fn kinds() -> std::sync::MutexGuard<'static, Kinds> {
    KINDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the kernel is asked only about memory not known to be other
/// than write-watched (`VICTORY_WATCH_UNWATCHED=0` asks every time, as
/// before; on by default).
///
/// A submission's asking tried the kernel for every range it refreshes,
/// view memory too, where the call fails (views and placeholders cannot be
/// write-watched): 2,600-4,500 failed calls a frame, 5-10 ms, in play1.
/// Write watching is chosen when memory is reserved, and only the runtime
/// reserves it (xmem, which tells [`title_memory_reserved`]), so memory
/// found unwatched stays so until the runtime reserves memory there anew.
/// Were it wrong, the asker would take the memory for unwatched: a copy
/// then copies its range whole at every refresh (or guards it, if it is a
/// view), which costs time and misses nothing.
fn unwatched_cache() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WATCH_UNWATCHED").is_none_or(|v| v != "0"))
}

/// Whether asking the kernel about blocks `[first, last)` is known to fail.
fn known_unwatched(first: usize, last: usize) -> bool {
    if !unwatched_cache() || first >= last {
        return false;
    }
    let k = kinds();
    k.unwatched.meets(first, last)
        || k.placeholders.meets(first, last)
        || k.across.range(first..last).any(|(_, &end)| end <= last)
}

/// Asking the kernel about blocks `[first, last)` failed: notes why, as
/// far as two small questions can tell (is its first block, or its last,
/// unwatched; else it lies across reservations).
fn learn_unwatched(first: usize, last: usize) {
    if !unwatched_cache() || first >= last {
        return;
    }
    tally(S::LearnN, 1);
    for block in [first, last - 1] {
        if !watched_block(block) {
            kinds().unwatched.insert(block, block + 1);
            return;
        }
    }
    let mut k = kinds();
    // What a range inside it says already, or a cache grown too big.
    if k.across.range(first..last).any(|(_, &end)| end <= last) {
        return;
    }
    if k.across.len() >= 4096 {
        k.across.clear();
    }
    let outer: Vec<usize> = k.across.range(..=first).filter(|&(_, &end)| end >= last).map(|(&at, _)| at).collect();
    for at in outer {
        k.across.remove(&at);
    }
    k.across.insert(first, last);
}

/// Whether block `block` is write-watched memory (asked without resetting
/// what the kernel holds).
fn watched_block(block: usize) -> bool {
    use windows_sys::Win32::System::Memory::GetWriteWatch;
    let mut found = [0usize; BLOCK / KERNEL_PAGE];
    let mut count = found.len();
    let mut granularity = 0;
    // SAFETY: the answer buffer holds `count` entries; flags 0 reset nothing.
    unsafe {
        GetWriteWatch(0, (block * BLOCK) as *const _, BLOCK, found.as_mut_ptr().cast(), &mut count, &mut granularity) == 0
    }
}

/// The runtime reserved title memory `[at, at + len)` anew: a placeholder
/// (`placeholder`, where it maps physical pages), else write-watched
/// memory. Nothing known of what was there before stands.
pub fn title_memory_reserved(at: usize, len: usize, placeholder: bool) {
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    {
        let mut g = guards();
        g.clean.remove(first, last);
        g.absent.remove(first, last);
        // New memory counts whole ([`fine_guards`]).
        let partly: Vec<usize> = g.parts.range(first..last).map(|(&block, _)| block).collect();
        for block in partly {
            g.parts.remove(&block);
        }
        let gone: Vec<usize> = g.held.range(first..last).chain(g.aliased.range(first..last)).copied().collect();
        for block in gone {
            g.held.remove(&block);
            g.aliased.remove(&block);
        }        g.hot.forget(first, last);
    }
    let mut k = kinds();
    k.unwatched.remove(first, last);
    k.placeholders.remove(first, last);
    let gone: Vec<usize> =
        k.across.range(..last).filter(|&(&a, &b)| a < last && b > first).map(|(&a, _)| a).collect();
    for a in gone {
        k.across.remove(&a);
    }
    if placeholder {
        k.placeholders.insert(first, last);
    }
}

/// Whether blocks not mapped in a runtime placeholder reservation are taken
/// to stay unmapped until the runtime maps them (`VICTORY_GUARD_PLACEHOLDERS=1`;
/// off by default).
///
/// A guarded range with unmapped blocks (absent) is never all clean, and
/// each asking asked Windows again, hole by hole, whether they were still
/// not committed, since an ordinary commit is not announced: 5,000-11,500
/// questions a frame, 3-9 ms, in play1. A placeholder cannot be committed:
/// only replaced by a mapping, which the runtime makes and announces
/// ([`title_memory_remapped`]). What this does not see: the title freeing
/// the reservation itself and other memory being placed there without the
/// runtime (it reserves title memory, [`title_memory_reserved`]).
fn trust_placeholders() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GUARD_PLACEHOLDERS").is_some_and(|v| v == "1"))
}

/// [`uncommitted_end`] for the guards' walk: with [`trust_placeholders`],
/// an absent block in a placeholder reservation is taken for unmapped to
/// the reservation's end, with no question.
fn absent_end(block: usize) -> usize {
    if trust_placeholders()
        && let Some(end) = kinds().placeholders.run_end(block)
    {
        tally(S::TrustedN, 1);
        return end;
    }
    uncommitted_end(block)
}

/// Whether the block holding `start` is view memory the guards already
/// hold (read-only by our hand, in whole or in part, or noted absent):
/// asking the kernel's write watch about it can only fail.
/// `VICTORY_VIEW_ASK=1` (or `VICTORY_GUARD_ASK=1`) asks the kernel anyway,
/// as before.
fn known_view(start: usize) -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let ask = *ASK.get_or_init(|| {
        ["VICTORY_VIEW_ASK", "VICTORY_GUARD_ASK"]
            .iter()
            .any(|n| std::env::var_os(n).is_some_and(|v| v == "1"))
    });
    if ask {
        return false;
    }
    let g = guards();
    let first = start / BLOCK;
    g.holds(first) || (view_known() && g.viewed.contains(&first))
}

struct Guards {
    /// Blocks we made read-only and that were not written since: a write
    /// (ours announced, or the title's through the handler) gives a block
    /// its write access back and takes it out. (There was a set of the
    /// blocks made read-only and one of those not written since; every
    /// change made both the same, so they were one.)
    clean: Runs,
    /// Blocks already copied as zeros. While they remain wholly uncommitted,
    /// the copy can skip hundreds of MB of failing kernel reads per world frame.
    absent: Runs,
    /// With [`fine_guards`]: blocks guarded and since opened in part, and
    /// the parts opened (written, or about to be); the other parts are
    /// still read-only. Not in `clean`. A block opened whole has no entry.
    parts: BTreeMap<usize, u16>,
    /// Blocks whose physical page was guarded together with its other views
    /// (those views' blocks are in it too): a write to one opens them all.
    /// A write range can be huge (`fread`), so only these are looked up.
    aliased: BTreeSet<usize>,
    /// Blocks made read-only as another view of pages a copy took through
    /// its own view ([`protect_alias`]) and not noted clean: reported
    /// written at their next check (they are not in `clean`), and opened by
    /// a write as any guarded block is.
    held: BTreeSet<usize>,
    /// With [`hot_blocks`]: blocks written before every asking, left open.
    hot: Hot,
    /// With [`view_known`]: blocks made read-only at least once since the
    /// runtime last mapped or unmapped them. They are view memory whether
    /// their guard stands now or a write has opened it.
    viewed: BTreeSet<usize>,
}

/// Whether a block once guarded stays known as view memory while its
/// guard is open (`VICTORY_VIEW_KNOWN=1`; off by default).
///
/// A range whose first block the title wrote since the last check has
/// that block neither read-only nor noted absent, so nothing says it is a
/// view: the kernel's write watch is asked and fails, and then Windows is
/// asked what the memory is (`is_view`), a walk of the page tables to the
/// end of the pages that share the first one's state. In runs 222 and 225
/// that was 338-450 failed askings a frame and 6.5-7.8 ms a frame of the
/// submitting thread in `VirtualQuery`. With the switch the block is
/// remembered from the time it was first made read-only until the runtime
/// maps or unmaps it ([`title_memory_remapped`]).
///
/// `VICTORY_VIEW_KNOWN=alt`: twenty seconds with, twenty without, in turn,
/// each change said in the log, for measuring it in one scene. (Blocks
/// noted in one phase are forgotten at a remapping in either.)
fn view_known() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_VIEW_KNOWN").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "view memory stays known while its guard is open from now (VICTORY_VIEW_KNOWN=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

static GUARDS: Mutex<Guards> = Mutex::new(Guards {
    clean: Runs::new(),
    absent: Runs::new(),
    parts: BTreeMap::new(),
    aliased: BTreeSet::new(),
    held: BTreeSet::new(),
    hot: Hot::new(),
    viewed: BTreeSet::new(),
});

/// Whether blocks of view memory found written at the askings of this many
/// submissions in a row are left open (`VICTORY_HOT_BLOCKS=N`, N of 2 or
/// more; off by default; not with the guards of other views,
/// `VICTORY_VIEW_GUARD`, nor [`fine_guards`]).
///
/// A guarded block found written is guarded again (`VirtualProtect`, about
/// 5 microseconds a run of view memory, 985 runs and 4.8 ms a frame in
/// n28), and the title's next write to it faults (420 faults a frame) or
/// our next merge opens it. A block written before every submission pays
/// that every submission, and is copied every submission anyway. With the
/// switch such a block is left read-write and reported written at every
/// asking that covers it, with no guard and no fault: what memory that is
/// not write-watched gets, a copy of it whole at every refresh. Every
/// [`hot_recheck`] submissions it is guarded again, and stays guarded until
/// it is found written at N askings in a row once more.
///
/// What it rests on: a block reported written at an asking is copied by
/// every refresh after it, so a hot block's copies hold what the title
/// wrote before each asking (the bytes a fault would have reported). Not
/// with the other views' guards: those leave a hot block's other views
/// unguarded and would not see a write through them.
///
/// What it changes beyond the guards (found after run n34, which showed
/// garbage with it): no title write is lost, but a hot block is reported
/// written at every asking, also at those after a submission's one asking
/// ended (work of ours that writes title memory), so it is copied again
/// within the same submission and, for a copy in the card's memory, sent
/// up again each time; bytes shaders wrote there outside a declared range
/// (views picked from the heap, `VICTORY_PICKED_GUARD`) are overwritten
/// each time. The blocks found written at every asking are mostly those
/// our copy-back writes each submission, i.e. the ones shaders write. A hot
/// block under flights' writes is also stale at every stale check (a merge
/// or a settle each submission), and a texture in one is read again at
/// every check.
pub(crate) fn hot_blocks() -> u32 {
    static N: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("VICTORY_HOT_BLOCKS").ok().and_then(|v| v.trim().parse::<u32>().ok()).unwrap_or(0);
        if n >= 2 && !view_guard() && !fine_guards() { n } else { 0 }
    })
}

/// With [`hot_blocks`]: submissions after which a hot block is guarded
/// again (`VICTORY_HOT_BLOCKS_RECHECK`, 64 unless set; at least 2).
fn hot_recheck() -> u64 {
    static N: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_HOT_BLOCKS_RECHECK").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(64).max(2)
    })
}

/// The submissions' askings counted ([`ask_once`]): what "in a row" counts
/// for [`hot_blocks`].
static ROUND: AtomicU64 = AtomicU64::new(0);

/// Tests that ask through the submission's standing asking (one journal for
/// the process) take turns.
#[cfg(test)]
fn asking_test() -> std::sync::MutexGuard<'static, ()> {
    static TURN: Mutex<()> = Mutex::new(());
    TURN.lock().unwrap_or_else(|e| e.into_inner())
}

/// [`hot_blocks`]' state, in the guards.
struct Hot {
    /// Askings in a row (0: off), and after how many a hot block is guarded
    /// again: set from the switches at each guarding, by the tests directly.
    n: u32,
    recheck: u64,
    /// The asking ([`ROUND`]) now.
    round: u64,
    /// Blocks reported written lately: the asking that last did, and at how
    /// many askings in a row.
    streak: BTreeMap<usize, (u64, u32)>,
    /// Blocks left open, and the asking that left them so.
    open: BTreeMap<usize, u64>,
}

/// Since the last frame line, of [`hot_blocks`]: blocks reported written
/// with no guard made, blocks left open, blocks guarded again.
static HOT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

impl Hot {
    const fn new() -> Self {
        Hot { n: 0, recheck: 64, round: 0, streak: BTreeMap::new(), open: BTreeMap::new() }
    }

    /// Whether a block reported written now stays (or is left) open
    /// instead of being guarded again; counts it as reported.
    fn left_open(&mut self, block: usize) -> bool {
        if self.n == 0 {
            return false;
        }
        let round = self.round;
        if let Some(&since) = self.open.get(&block) {
            if round >= since + self.recheck {
                self.open.remove(&block);
                self.streak.remove(&block);
                HOT[2].fetch_add(1, Relaxed);
                return false;
            }
            HOT[0].fetch_add(1, Relaxed);
            return true;
        }
        let times = match self.streak.get_mut(&block) {
            Some(s) if s.0 == round => s.1,
            Some(s) if s.0 + 1 == round => {
                *s = (round, s.1 + 1);
                s.1
            }
            Some(s) => {
                *s = (round, 1);
                1
            }
            None => {
                // Blocks reported once and never again: let go of now and then.
                if self.streak.len() >= 1 << 16 {
                    self.streak.retain(|_, s| s.0 + 1 >= round);
                }
                self.streak.insert(block, (round, 1));
                1
            }
        };
        if times < self.n {
            return false;
        }
        self.streak.remove(&block);
        self.open.insert(block, round);
        HOT[1].fetch_add(1, Relaxed);
        HOT[0].fetch_add(1, Relaxed);
        true
    }

    fn is_open(&self, block: usize) -> bool {
        !self.open.is_empty() && self.open.contains_key(&block)
    }

    /// Blocks `[first, last)` are new memory (mapped, unmapped, reserved):
    /// nothing learned of them stands.
    fn forget(&mut self, first: usize, last: usize) {
        if first >= last {
            return;
        }
        if !self.open.is_empty() {
            let gone: Vec<usize> = self.open.range(first..last).map(|(&b, _)| b).collect();
            for b in gone {
                self.open.remove(&b);
            }
        }
        if !self.streak.is_empty() {
            let gone: Vec<usize> = self.streak.range(first..last).map(|(&b, _)| b).collect();
            for b in gone {
                self.streak.remove(&b);
            }
        }
    }
}

fn guards() -> std::sync::MutexGuard<'static, Guards> {
    GUARDS.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether the block `page` is read-only by our hand and unwritten since a
/// copy took it (diagnostics of `VICTORY_BACK_OWN_CHECK`).
pub(crate) fn clean(page: usize) -> bool {
    guards().clean.contains(page)
}

impl Guards {
    /// Whether block `block` is one the guards hold: read-only by our hand
    /// (clean, opened in part, or held as another view) or noted absent.
    fn holds(&self, block: usize) -> bool {
        self.clean.contains(block)
            || self.absent.contains(block)
            || self.parts.contains_key(&block)
            || self.held.contains(&block)
            || self.hot.is_open(block)
    }

    /// How many blocks of `[first, last)` are read-only by our hand, in
    /// whole or in part.
    fn guarded_in(&self, first: usize, last: usize) -> u64 {
        if first >= last {
            return 0;
        }
        let clean: usize = self.clean.within(first, last).iter().map(|&(a, b)| b - a).sum();
        (clean + self.parts.range(first..last).count() + self.held.range(first..last).count()) as u64
    }
}

/// [`guard_parts`], the pages alone.
#[cfg(test)]
fn guard(start: usize, end: usize) -> Option<Vec<usize>> {
    guard_parts(start, end).map(|pages| pages.into_iter().map(|(page, _)| page).collect())
}

/// For view memory: the pages of `[start, end)`'s blocks written since
/// they were last guarded, each with the parts written (all of them unless
/// the block was opened in part, [`fine_guards`]), guarding them again
/// before the caller copies them. `None` if the memory is not a view.
fn guard_parts(start: usize, end: usize) -> Option<Vec<(usize, u16)>> {
    // A range whose first block the guards hold already (read-only by our
    // hand, or noted absent) is a view: no question to Windows. The
    // question walks the page tables to the end of the pages that share the
    // first one's state, and a guarded view is one long run of read-only
    // pages: 0.6 ms for the title's 174 MB heap, 16 times a frame, and most
    // of the 68 ms a frame of "guards" (run 58).
    // `VICTORY_GUARD_ASK=1` asks every time, as before.
    if !guards_know(start / BLOCK) && !is_view(start, end) {
        return None;
    }
    install();
    let mut g = guards();
    let g = &mut *g;
    g.hot.n = hot_blocks();
    if g.hot.n != 0 {
        g.hot.recheck = hot_recheck();
        g.hot.round = ROUND.load(Relaxed);
    }
    let mut written = Vec::new();
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    if first >= last || g.clean.covers(first, last) {
        // Nothing written, mapped or new since the range was last taken.
        tally(S::GuardClean, 1);
        return Some(Vec::new());
    }
    let walked = walk(g, first, last, &mut written, &mut absent_end, &mut protect);
    tally(S::GuardBlocks, walked as u64);
    // Every block reported was guarded again (or tried): what was open of
    // it is noted with it and forgotten. A block opened whole, or never
    // guarded, counts whole. Under the same lock as the walk, so no write
    // falls in between.
    Some(
        written
            .into_iter()
            .map(|page| (page, if g.parts.is_empty() { ALL_SUBS } else { g.parts.remove(&page).unwrap_or(ALL_SUBS) }))
            .collect(),
    )
}

/// Whether [`guard`] takes a range that begins at block `first` for a view
/// without asking Windows: the guards hold the block (read-only by our
/// hand, or noted absent). `VICTORY_GUARD_ASK=1` asks every time, as
/// before run 58.
fn guards_know(first: usize) -> bool {
    static ASK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    !*ASK.get_or_init(|| std::env::var_os("VICTORY_GUARD_ASK").is_some_and(|v| v == "1")) && {
        // A block opened in part ([`fine_guards`]) is guarded still, as is
        // one held as another view; with upstream's `VICTORY_VIEW_KNOWN=1`
        // also one once guarded and opened since.
        let g = guards();
        g.holds(first) || (view_known() && g.viewed.contains(&first))
    }
}

/// [`guard`] of blocks `[first, last)`, not all clean, with the questions
/// to Windows passed in: the blocks that are not clean, a run at a time
/// (written since they were guarded, never guarded, or absent). Each that
/// is there now is reported written (its pages appended to `written`) and
/// guarded again, in runs of neighbours. What the walk changes in the sets
/// lies behind where it stands. Returns the blocks walked.
fn walk(
    g: &mut Guards,
    first: usize,
    last: usize,
    written: &mut Vec<usize>,
    uncommitted_end: &mut dyn FnMut(usize) -> usize,
    protect: &mut dyn FnMut(&mut Guards, usize, usize),
) -> usize {
    let mut gaps = Vec::new();
    let mut at = first;
    for (a, b) in g.clean.within(first, last) {
        if a > at {
            gaps.push((at, a));
        }
        at = b;
    }
    if at < last {
        gaps.push((at, last));
    }
    let mut run: Option<(usize, usize)> = None;
    let mut take = |g: &mut Guards, a: usize, b: usize| {
        written.extend(a * BLOCK / PAGE..b * BLOCK / PAGE);
        // With [`hot_blocks`]: blocks left open are reported, not guarded.
        let open: Vec<usize> = if g.hot.n == 0 { Vec::new() } else { (a..b).filter(|&k| g.hot.left_open(k)).collect() };
        let mut at = a;
        for next in open.into_iter().chain(std::iter::once(b)) {
            if at < next {
                match run {
                    Some((_, ref mut to)) if *to == at => *to = next,
                    _ => {
                        if let Some((from, to)) = run.take() {
                            protect(g, from, to);
                        }
                        run = Some((at, next));
                    }
                }
            }
            at = next + 1;
        }
    };
    let mut uncommitted_until = first;
    let mut walked = 0;
    for (from, to) in gaps {
        walked += to - from;
        let mut at = from;
        for (a, b) in g.absent.within(from, to) {
            if a > at {
                take(g, at, a);
            }
            // Absent blocks stay so while they are not committed: ordinary
            // commits need no runtime mapping notification. One question
            // answers for the rest of its region.
            let mut block = a;
            while block < b {
                if block >= uncommitted_until {
                    uncommitted_until = uncommitted_end(block);
                }
                if block < uncommitted_until {
                    block = uncommitted_until.min(b);
                    continue;
                }
                g.absent.remove(block, block + 1);
                take(g, block, block + 1);
                block += 1;
            }
            at = b;
        }
        if at < to {
            take(g, at, to);
        }
    }
    drop(take);
    if let Some((from, to)) = run {
        protect(g, from, to);
    }
    walked
}

/// Whether `[start, end)` is view memory. The title's views lie in
/// placeholders, mapped and unmapped block by block as it streams, so a
/// range can begin in a block that is not mapped right now: the first
/// pages of the range that are there decide. A range with nothing there
/// counts as a view (its blocks are noted absent, and looked at again when
/// the runtime maps them).
///
/// Looking at the first address only took such a range for "not a view":
/// its copy was marked not watched for good and copied whole at every
/// refresh, around its unmapped pages (run 54: 300 such ranges, 570 MB
/// copied a frame for them, 255 copies a frame reading around holes).
/// `VICTORY_GUARD_HOLES=0` looks at the first address only, as before.
fn is_view(start: usize, end: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_MAPPED, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    static HOLES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let holes = *HOLES.get_or_init(|| std::env::var_os("VICTORY_GUARD_HOLES").is_none_or(|v| v != "0"));
    let mut at = start;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let asking = std::time::Instant::now();
        // SAFETY: a valid out pointer of the right size.
        let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
        tally(S::QueryN, 1);
        tally_since(S::QueryNs, asking);
        if n == 0 {
            return false;
        }
        if info.Type == MEM_MAPPED {
            return true;
        }
        // Committed memory that is no view (the title's heaps, ours), or
        // the answer as it was before.
        if info.State == MEM_COMMIT || !holes {
            return false;
        }
        // A hole (a placeholder, or free): on to what follows it.
        let next = (info.BaseAddress as usize).saturating_add(info.RegionSize);
        if next <= at || next >= end {
            return true;
        }
        at = next;
    }
}

/// Makes blocks `[from, to)` read-only and notes them clean; block by
/// block if they span views, leaving out any that fail, and noting those
/// not committed (unmapped) absent. The same physical pages at every other
/// view of them are made read-only too: a write through one of those is
/// this memory written, and only a fault can say so.
/// `VICTORY_VIEW_GUARD=0` leaves the other views as they are, as before.
fn protect(g: &mut Guards, from: usize, to: usize) {
    protect_here(g, from, to);
    if !view_guard() {
        return;
    }
    let started = std::time::Instant::now();
    if alias_runs() {
        protect_aliases_in_runs(g, from, to);
    } else {
        protect_aliases(g, from, to);
    }
    tally_since(S::AliasNs, started);
}

/// [`protect`]'s other views, block by block.
fn protect_aliases(g: &mut Guards, from: usize, to: usize) {
    for (base, bytes) in crate::inplace::views_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK)) {
        let first = (base / BLOCK).max(from);
        let last = (base.saturating_add(bytes).div_ceil(BLOCK)).min(to);
        for block in first..last {
            let shown = crate::inplace::pages_of(block * BLOCK);
            // One view (nothing else to guard), or the page the title shows
            // wherever it has nothing (thousands of views): each view of
            // that one is guarded by itself.
            if shown.len() < 2 || shown.len() > MANY_VIEWS {
                continue;
            }
            #[cfg(feature = "diagnostics")]
            ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            for alias in shown {
                let alias = alias / BLOCK;
                g.aliased.insert(alias);
                if alias != block {
                    protect_alias(g, alias);
                }
            }
        }
    }
}

/// Whether [`protect`] guards the other views of the pages it guarded in
/// runs (`VICTORY_ALIAS_RUNS=1`; off by default).
///
/// Each block guarded again looked its page's views up on its own (the
/// views' lock, two map lookups and a list made, per block) and made each
/// other view read-only with a call of its own: one `VirtualProtect` per
/// block and view. The world's views show the same physical pages in the
/// same order, so the other views of a run of blocks are runs too. With
/// the switch the views of all the run's blocks are looked up under one
/// lock, and the other views' blocks not held already are made read-only
/// a run of neighbours at a time (block by block where a run cannot be,
/// as across two views). What is guarded and noted is the same.
fn alias_runs() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ALIAS_RUNS").is_some_and(|v| v == "1"))
}

/// [`protect`]'s other views with [`alias_runs`].
fn protect_aliases_in_runs(g: &mut Guards, from: usize, to: usize) {
    let shown = crate::inplace::aliases_between(from.saturating_mul(BLOCK), to.saturating_mul(BLOCK), MANY_VIEWS);
    if shown.is_empty() {
        return;
    }
    let mut others: Vec<usize> = Vec::new();
    for (page, views) in shown {
        let block = page / BLOCK;
        #[cfg(feature = "diagnostics")]
        ALIASES[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        for alias in views {
            let alias = alias / BLOCK;
            g.aliased.insert(alias);
            if alias != block {
                others.push(alias);
            }
        }
    }
    others.sort_unstable();
    others.dedup();
    // As `protect_alias`: guarded already, in whole or in part, or held.
    others.retain(|&block| !g.holds(block));
    let mut i = 0;
    while i < others.len() {
        let mut k = i + 1;
        while k < others.len() && others[k] == others[k - 1] + 1 {
            k += 1;
        }
        let (first, last) = (others[i], others[k - 1] + 1);
        if protect_read_only(first, last) {
            g.held.extend(first..last);
        } else if last - first > 1 {
            for block in first..last {
                if protect_read_only(block, block + 1) {
                    g.held.insert(block);
                }
            }
        }
        i = k;
    }
}

/// Makes blocks `[first, last)` read-only; whether it was done (all or
/// nothing: `VirtualProtect` takes no range across two views).
fn protect_read_only(first: usize, last: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let mut old = 0;
    let started = std::time::Instant::now();
    // SAFETY: title view memory; only its protection changes.
    let done = unsafe { VirtualProtect((first * BLOCK) as *const _, (last - first) * BLOCK, PAGE_READONLY, &mut old) } != 0;
    tally(S::AliasProtectN, 1);
    tally_since(S::AliasProtectNs, started);
    done
}

/// Views of one page past which its views are not guarded together.
pub(crate) const MANY_VIEWS: usize = 64;

/// Whether all the views of a guarded page are guarded together
/// (`VICTORY_VIEW_GUARD`).
fn view_guard() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_VIEW_GUARD").is_none_or(|v| v != "0"))
}

/// Diagnostics: blocks guarded together with their other views, blocks
/// opened because another view of theirs was written, and blocks opened
/// because a view of theirs was mapped; since the start.
#[cfg(feature = "diagnostics")]
pub(crate) static ALIASES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Makes `block` read-only: another view of pages a copy has just taken
/// through its own view, so that a write through this one faults too. It is
/// not noted clean. The copies of title memory at this view have not taken
/// the pages: noted clean, the block would not be reported written at their
/// next check and they would keep the bytes from before the write (`guard`
/// reports a block that is read-only and not clean as written, then notes it
/// clean). A block that is clean stays so: had the pages been written since
/// its copies took them, `open` would have opened it.
fn protect_alias(g: &mut Guards, block: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    // Guarded already, in whole or in part (a part open was written and
    // is reported so), or held as another view.
    if g.holds(block) {
        return;
    }
    let mut old = 0;
    let started = std::time::Instant::now();
    // SAFETY: title view memory; only its protection changes.
    let done = unsafe { VirtualProtect((block * BLOCK) as *const _, BLOCK, PAGE_READONLY, &mut old) } != 0;
    tally(S::AliasProtectN, 1);
    tally_since(S::AliasProtectNs, started);
    if done {
        g.held.insert(block);
        if view_known() {
            g.viewed.insert(block);
        }
    }
}

/// [`protect`] for one range, without the other views of its pages.
fn protect_here(g: &mut Guards, from: usize, to: usize) {
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, VirtualProtect};
    let set = |from: usize, to: usize| {
        let mut old = 0;
        let started = std::time::Instant::now();
        // SAFETY: title view memory; only its protection changes.
        let done = unsafe {
            VirtualProtect(
                (from * BLOCK) as *const _,
                (to - from) * BLOCK,
                PAGE_READONLY,
                &mut old,
            ) != 0
        };
        tally(S::ProtectN, 1);
        tally_since(S::ProtectNs, started);
        if !done {
            tally(S::ProtectFailedN, 1);
        }
        done
    };
    if set(from, to) {
        g.clean.insert(from, to);
        if view_known() {
            g.viewed.extend(from..to);
        }
        if !g.held.is_empty() {
            let held: Vec<usize> = g.held.range(from..to).copied().collect();
            for block in held {
                g.held.remove(&block);
            }
        }
        return;
    }
    for block in from..to {
        if set(block, block + 1) {
            g.clean.insert(block, block + 1);
            g.held.remove(&block);
            if view_known() {
                g.viewed.insert(block);
            }
        } else {
            #[cfg(feature = "diagnostics")]
            let error = std::io::Error::last_os_error();
            if block < uncommitted_end(block) {
                g.absent.insert(block, block + 1);
            } else {
                // Diagnostics: a committed block that will not be made
                // read-only is reported written at every asking and never
                // known to the guards (one in 1,024 said).
                #[cfg(feature = "diagnostics")]
                {
                    static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                    if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 1024 == 0 {
                        tracing::info!("a block of title memory could not be made read-only (one in 1024): {:#x}: {error}", block * BLOCK);
                    }
                }
            }
        }
    }
}

/// The end (exclusive) of the whole uncommitted blocks starting at `block`.
fn uncommitted_end(block: usize) -> usize {
    use windows_sys::Win32::System::Memory::{
        MEM_FREE, MEM_RESERVE, MEMORY_BASIC_INFORMATION, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    let asking = std::time::Instant::now();
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery((block * BLOCK) as *const _, &mut info, size_of_val(&info)) };
    tally(S::QueryN, 1);
    tally_since(S::QueryNs, asking);
    if n != 0 && matches!(info.State, MEM_RESERVE | MEM_FREE) {
        (info.BaseAddress as usize).saturating_add(info.RegionSize) / BLOCK
    } else {
        block
    }
}

/// Gives write access back to the guarded blocks of `[start, end)` and
/// notes them written; true if any was guarded. Blocks opened in part
/// ([`fine_guards`]) open whole and count whole. The same physical pages at
/// every other view of them are opened too: the write that faulted (or is
/// ours, `crate::host::copy_mapped`) goes through one view, and every
/// guard over the pages must copy them again.
fn open(g: &mut Guards, start: usize, end: usize) -> bool {
    let any = open_here(g, start, end);
    open_aliases(g, start, end);
    any
}

/// The other views of the guarded pages of `[start, end)`, opened as
/// [`open`] opens them.
fn open_aliases(g: &mut Guards, start: usize, end: usize) {
    if start >= end || g.aliased.is_empty() {
        return;
    }
    let aliased: Vec<usize> = g.aliased.range(start / BLOCK..end.div_ceil(BLOCK)).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
        for alias in crate::inplace::pages_of(block * BLOCK) {
            if alias / BLOCK != block && open_here(g, alias, alias.saturating_add(BLOCK)) {
                #[cfg(feature = "diagnostics")]
                ALIASES[1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            g.aliased.remove(&(alias / BLOCK));
        }
    }
}

/// [`open`] for one range, without the other views of its pages.
fn open_here(g: &mut Guards, start: usize, end: usize) -> bool {
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let runs = g.clean.within(first, last);
    g.clean.remove(first, last);
    for &(a, b) in &runs {
        for block in a..b {
            read_write(block * BLOCK, BLOCK);
        }
    }
    let partly: Vec<usize> = g.parts.range(first..last).map(|(&block, _)| block).collect();
    for &block in &partly {
        g.parts.remove(&block);
        read_write(block * BLOCK, BLOCK);
    }
    let held: Vec<usize> = if g.held.is_empty() { Vec::new() } else { g.held.range(first..last).copied().collect() };
    for &block in &held {
        g.held.remove(&block);
        read_write(block * BLOCK, BLOCK);
    }
    !runs.is_empty() || !partly.is_empty() || !held.is_empty()
}

/// Gives write access back to `[at, at + len)` (whole pages); whether it
/// was given.
fn read_write(at: usize, len: usize) -> bool {
    use windows_sys::Win32::System::Memory::{PAGE_READWRITE, VirtualProtect};
    let mut old = 0;
    // SAFETY: memory we made read-only; only its protection changes.
    unsafe { VirtualProtect(at as *const _, len, PAGE_READWRITE, &mut old) != 0 }
}

/// Gives write access back to the parts `mask` of block `block`, a run of
/// neighbours at a time; whether all of them were given.
fn read_write_parts(block: usize, mask: u16) -> bool {
    let mut all = true;
    let mut i = 0;
    while i < SUBS {
        if mask >> i & 1 == 0 {
            i += 1;
            continue;
        }
        let mut k = i + 1;
        while k < SUBS && mask >> k & 1 != 0 {
            k += 1;
        }
        all &= read_write(block * BLOCK + i * SUB, (k - i) * SUB);
        i = k;
    }
    all
}

/// The handler's [`open`] for a write that faulted at `at`. With
/// [`fine_guards`] a guarded block opens only in the part written, and a
/// block opened in part opens whole at a write into another part. True if
/// the write may go on.
fn open_fault(g: &mut Guards, at: usize) -> bool {
    if !fine_guards() {
        return open(g, at, at + 1);
    }
    let go_on = open_fault_here(g, at);
    // The other views of the page see the write too (`open`).
    open_aliases(g, at, at + 1);
    go_on
}

/// [`open_fault`] at the view written alone.
fn open_fault_here(g: &mut Guards, at: usize) -> bool {
    let block = at / BLOCK;
    // Held as another view of a page ([`protect_alias`]): opens whole.
    if g.held.remove(&block) {
        read_write(block * BLOCK, BLOCK);
        return true;
    }
    let part = (at % BLOCK) / SUB;
    if g.clean.contains(block) {
        g.clean.remove(block, block + 1);
        if read_write_parts(block, 1 << part) {
            g.parts.insert(block, 1 << part);
        } else {
            read_write(block * BLOCK, BLOCK);
        }
        return true;
    }
    match g.parts.get(&block) {
        // Another thread opened this part since the write faulted.
        Some(&mask) if mask >> part & 1 != 0 => true,
        Some(_) => {
            g.parts.remove(&block);
            read_write(block * BLOCK, BLOCK);
            true
        }
        None => false,
    }
}

/// [`open`] for our own write of exactly `[start, end)`: with
/// [`fine_guards`] only the parts written open, and the blocks count as
/// written in them.
fn open_exact(g: &mut Guards, start: usize, end: usize) -> bool {
    if !fine_guards() {
        return open(g, start, end);
    }
    let (first, last) = (start / BLOCK, end.div_ceil(BLOCK));
    let mut any = false;
    for (a, b) in g.clean.within(first, last) {
        g.clean.remove(a, b);
        for block in a..b {
            any = true;
            let mask = parts_in(block, start, end - start);
            if mask != ALL_SUBS && read_write_parts(block, mask) {
                g.parts.insert(block, mask);
            } else {
                read_write(block * BLOCK, BLOCK);
            }
        }
    }
    let partly: Vec<(usize, u16)> = g.parts.range(first..last).map(|(&block, &mask)| (block, mask)).collect();
    for (block, mask) in partly {
        let need = parts_in(block, start, end - start) & !mask;
        if need == 0 {
            continue;
        }
        any = true;
        if mask | need != ALL_SUBS && read_write_parts(block, need) {
            g.parts.insert(block, mask | need);
        } else {
            g.parts.remove(&block);
            read_write(block * BLOCK, BLOCK);
        }
    }
    // Held as another view of a page ([`protect_alias`]): opens whole.
    let held: Vec<usize> = if g.held.is_empty() { Vec::new() } else { g.held.range(first..last).copied().collect() };
    for block in held {
        any = true;
        g.held.remove(&block);
        read_write(block * BLOCK, BLOCK);
    }
    // The other views of the pages see the write too (`open`).
    open_aliases(g, start, end);
    any
}

/// Before our own write of exactly `[at, at + len)` into title memory
/// through the kernel (what shaders wrote, merged): as [`writing`], but
/// with [`fine_guards`] only the 4 KiB parts written open.
pub(crate) fn writing_exact(at: usize, len: usize) {
    if len == 0 {
        return;
    }
    open_exact(&mut guards(), at, at.saturating_add(len));
    crate::descriptor::host::remapped(at, len);
}

/// Before our own write into title memory `[at, at + len)` through the
/// kernel (which does not fault into our handler): opens its guarded blocks.
/// The runtime calls it (as `kernel_will_write`) before the title's kernel
/// reads into its own memory.
pub fn writing(at: usize, len: usize) {
    // The length is the title's (`fread`'s size times count, saturated).
    open(&mut guards(), at, at.saturating_add(len));
    // A placed texture there changes with no descriptor written.
    crate::descriptor::host::remapped(at, len);
}

/// What `VICTORY_PAGE_WRITES` asks to be done with a write the guards do
/// not see once its bytes are there ([`title_memory_written`]), as bits:
/// 1, the guarded blocks under it are opened and count as written (host
/// copies of buffers and textures read through tables take them again);
/// 2, the placed textures picked by index that lie under it are due again;
/// 4, both also for reads to plain addresses when they have finished.
/// Unset or `1`: bits 1 and 2 (bit 2 also for finished reads to plain
/// addresses). `guards`: bit 1 alone (the default until the texture
/// variance runs). `0`: nothing, counted only (the default until run 199).
/// `slots`: bit 2 alone. `all`: everything.
/// (Bit 1: in every run 10,000 to 50,000 guarded blocks of host copies lay
/// under such writes and kept the bytes from before; runs 197 and 198
/// opened them, 9,766 and 10,796 blocks, with nothing worse for it.
/// Bit 2 by default since runs n15-n23: 5,500 to 7,300 placed textures a
/// run had been uploaded before such a write landed in their memory (a
/// quarter of all the slots under such writes), and kept the bytes from
/// before it, what the title's pages held for another texture, for the
/// rest of the run. Which ones depended on timing (the texture workers,
/// the frame's length), so one cutscene shot looked different from run to
/// run: the character's top orange or pink-red, the window mesh drawn or not.
/// Only those slots are made due, each once a write: some 25 a second
/// while the world streams in; their host copies exist, no room needed.)
pub(crate) fn page_writes() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_PAGE_WRITES").as_deref().map(str::trim) {
        Ok("0") => 0,
        Ok("guards") => 1,
        Ok("slots") => 2,
        Ok("all") => 7,
        _ => 3,
    })
}

/// Since the start: writes into physical pages through the runtime's own
/// views that landed at addresses the title has mapped (one for each run
/// of addresses); their bytes; guarded blocks that lay under them (host
/// copies that would otherwise keep the bytes from before); reads to
/// plain addresses that finished.
pub(crate) static WRITTEN: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The bytes of a write the guards do not see are there: title memory
/// `[at, at + len)`. `pages`: DirectStorage wrote them into physical pages
/// through a view of the runtime's own, and this is an address where the
/// title has those pages mapped (no title address faulted, nothing was
/// mapped or unmapped; the shortcut this file's header names). Otherwise a
/// read to a plain address has finished, of which [`writing`] was told
/// before it began.
///
/// Upstream's c75fdba does the same for its trackers (its `Writing` marks
/// the pages once more when dropped; page destinations are reported at
/// each address showing them). Here it is counted always and acted on as
/// [`page_writes`] says. Called without the runtime's page-table lock;
/// takes the guards' lock, then, alone, the lock of the ranges kept for
/// the texture pass.
pub fn title_memory_written(at: usize, len: usize, pages: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    if len == 0 {
        return;
    }
    let mode = page_writes();
    let end = at.saturating_add(len);
    if pages {
        WRITTEN[0].fetch_add(1, Relaxed);
        WRITTEN[1].fetch_add(len as u64, Relaxed);
    } else {
        WRITTEN[3].fetch_add(1, Relaxed);
        // A finished read to a plain address: the guards only with bit 4;
        // the texture slots under it with bit 2 as well (`writing` made
        // them due when the read began, a frame before its bytes may be
        // there: one uploaded meanwhile took what was there before).
        if mode & 4 == 0 {
            if mode & 2 != 0 {
                crate::descriptor::host::page_written(at, len);
            }
            return;
        }
    }
    {
        let mut g = guards();
        let guarded = g.guarded_in(at / BLOCK, end.div_ceil(BLOCK));
        if pages {
            WRITTEN[2].fetch_add(guarded, Relaxed);
        }
        if mode & 1 != 0 && guarded != 0 {
            open(&mut g, at, end);
        }
    }
    // A placed texture there changed with no descriptor written.
    crate::descriptor::host::page_written(at, len);
}

/// The runtime mapped a view of the physical bytes `[offset, offset + len)`
/// of `section` (`crate::inplace::title_memory_view`): the view is
/// writable, so the same bytes guarded at another view must be checked
/// again — and are, the pages noted written, which also has the next check
/// guard the new view (`protect`).
pub(crate) fn views_mapped(section: usize, offset: u64, len: usize) {
    if !view_guard() {
        return;
    }
    let ranges = crate::inplace::views_of_section(section, offset, len);
    if ranges.is_empty() {
        return;
    }
    let mut g = guards();
    for &(at, n) in &ranges {
        if open_here(&mut g, at, at.saturating_add(n)) {
            #[cfg(feature = "diagnostics")]
            ALIASES[2].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
}

/// Held shared by a read of title memory in place, and alone by the runtime
/// while it changes what title memory shows.
///
/// The runtime does not change a mapping in one step. To unmap one page of
/// a view it unmaps the whole view and maps what stays again
/// (`xmem.rs` `unmap`), and a view it maps is a reservation until the next
/// call commits it (`map`, then `commit`). For that moment pages the title
/// never gave up are "reserved, never committed", and a thread reading
/// them in place takes an access violation: run 71 in `Layout::detile`,
/// run 74 in the hash before it, both under `Staging::refresh_now`, which
/// had asked whether the texture was all mapped and been told yes. Asking
/// cannot help; not changing the mapping while such a read is under way
/// does.
static MAPPING: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Whether [`MAPPING`]'s fair stand-in is used (`VICTORY_FAIR_MAPPING=0`
/// takes the standard lock again; on by default).
///
/// The standard lock lets a writer that comes back take it again before the
/// readers it woke run. The runtime changes mappings one after another
/// (`XMemMapPhysicalPagesScatter` takes the lock once per page, a world
/// streaming in calls `XMemMapPhysicalPages` back to back), so a texture
/// refresh that waited for one change waited for all of them: the wait for
/// this lock was 110 of the frame line's 113.8 ms of "textures" (run copy0)
/// and 123.8 of 133.1 (copy2), for 9-34 MB read. With the fair lock a
/// change that ends hands the lock to the reads that were waiting for it,
/// and the next change waits for them: a read waits for at most the change
/// under way. What the lock keeps apart is the same: no read in place
/// while a mapping changes.
fn fair() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FAIR_MAPPING").is_none_or(|v| v != "0"))
}

/// The fair stand-in for [`MAPPING`] ([`fair`]).
struct Fair {
    state: Mutex<FairState>,
    turn: std::sync::Condvar,
}

struct FairState {
    /// Reads under way.
    readers: usize,
    /// A change under way.
    writer: bool,
    /// Changes waiting to begin.
    writers_waiting: usize,
    /// Reads waiting for a change to end.
    readers_waiting: usize,
    /// Reads that waited for a change that ended and have not begun yet:
    /// no change begins before they have.
    granted: usize,
    /// Changes ended so far: a waiting read that sees it move on is one of
    /// `granted`.
    ended: u64,
}

static FAIR: Fair = Fair {
    state: Mutex::new(FairState {
        readers: 0,
        writer: false,
        writers_waiting: 0,
        readers_waiting: 0,
        granted: 0,
        ended: 0,
    }),
    turn: std::sync::Condvar::new(),
};

impl Fair {
    fn state(&self) -> std::sync::MutexGuard<'_, FairState> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Whether a read may begin now: no change under way, and none waiting
    /// unless reads that waited before it are still let in first.
    fn readable(s: &FairState) -> bool {
        !s.writer && (s.writers_waiting == 0 || s.granted > 0)
    }

    fn read(&self) {
        let mut s = self.state();
        if Self::readable(&s) {
            s.readers += 1;
            return;
        }
        s.readers_waiting += 1;
        let ended = s.ended;
        while s.ended == ended {
            s = self.turn.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        // Counted in `granted` by the change that ended, and no change has
        // begun since (none does while `granted` is above 0).
        s.granted -= 1;
        s.readers += 1;
    }

    fn try_read(&self) -> bool {
        let mut s = self.state();
        let ok = Self::readable(&s);
        if ok {
            s.readers += 1;
        }
        ok
    }

    fn read_end(&self) {
        let mut s = self.state();
        s.readers -= 1;
        if s.readers == 0 && s.writers_waiting > 0 {
            self.turn.notify_all();
        }
    }

    fn write(&self) {
        let mut s = self.state();
        s.writers_waiting += 1;
        while s.writer || s.readers > 0 || s.granted > 0 {
            s = self.turn.wait(s).unwrap_or_else(|e| e.into_inner());
        }
        s.writers_waiting -= 1;
        s.writer = true;
    }

    fn write_end(&self) {
        let mut s = self.state();
        s.writer = false;
        s.ended += 1;
        s.granted += s.readers_waiting;
        s.readers_waiting = 0;
        self.turn.notify_all();
    }
}

/// Since the frame line last took them: changes of mapping, the
/// microseconds they held title memory (no read in place could begin), the
/// longest of them, and the microseconds the runtime waited to begin them
/// (for reads in place under way and, with [`fair`], for those that waited
/// before it).
pub(crate) static REMAPS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// What the runtime holds while it changes what title memory shows
/// ([`title_memory_remapping`]).
pub struct Remapping {
    standard: Option<std::sync::RwLockWriteGuard<'static, ()>>,
    since: std::time::Instant,
    /// When the lock was asked for (upstream's frame log).
    asked: std::time::Instant,
}

impl Drop for Remapping {
    fn drop(&mut self) {
        use std::sync::atomic::Ordering::Relaxed;
        let held = self.since.elapsed().as_micros() as u64;
        if self.standard.take().is_none() {
            FAIR.write_end();
        }
        REMAPS[0].fetch_add(1, Relaxed);
        REMAPS[1].fetch_add(held, Relaxed);
        REMAPS[2].fetch_max(held, Relaxed);
        // From asking for the lock until it is let go (the frame log).
        crate::framelog::add(crate::framelog::T::RemapUs, self.asked.elapsed().as_micros() as u64);
    }
}

/// The runtime is about to map, unmap or move physical pages of title
/// memory: until what it returns is dropped, no read of title memory in
/// place is under way and none begins ([`title_memory_reading`]). Taken
/// around the whole change, the unmapping and the mapping again.
pub fn title_memory_remapping() -> Remapping {
    let asked = std::time::Instant::now();
    let standard = if fair() {
        FAIR.write();
        None
    } else {
        Some(MAPPING.write().unwrap_or_else(|e| e.into_inner()))
    };
    let waited = asked.elapsed().as_micros() as u64;
    REMAPS[3].fetch_add(waited, std::sync::atomic::Ordering::Relaxed);
    crate::framelog::add(crate::framelog::T::Remaps, 1);
    crate::framelog::add(crate::framelog::T::RemapWaitUs, waited);
    Remapping {
        standard,
        since: std::time::Instant::now(),
        asked,
    }
}

/// What a read of title memory in place holds ([`title_memory_reading`]).
pub struct Reading {
    standard: Option<std::sync::RwLockReadGuard<'static, ()>>,
}

impl Drop for Reading {
    fn drop(&mut self) {
        if self.standard.take().is_none() {
            FAIR.read_end();
        }
    }
}

/// For reading title memory in place: until what it returns is dropped,
/// the runtime changes no mapping, so memory found mapped stays mapped.
/// Not to be taken twice by one thread, nor held across anything that
/// maps title memory.
pub fn title_memory_reading() -> Reading {
    if fair() {
        FAIR.read();
        return Reading { standard: None };
    }
    Reading {
        standard: Some(MAPPING.read().unwrap_or_else(|e| e.into_inner())),
    }
}

/// [`title_memory_reading`] if it can be had at once: `None` while the
/// runtime changes a mapping (or waits to), for a caller that has another
/// way to read and would rather not wait.
pub fn title_memory_reading_now() -> Option<Reading> {
    if fair() {
        // `then`, not `then_some`: a guard made and dropped on failure would
        // end a read that never began.
        return FAIR.try_read().then(|| Reading { standard: None });
    }
    match MAPPING.try_read() {
        Ok(guard) => Some(Reading { standard: Some(guard) }),
        Err(std::sync::TryLockError::Poisoned(e)) => Some(Reading {
            standard: Some(e.into_inner()),
        }),
        Err(std::sync::TryLockError::WouldBlock) => None,
    }
}

/// The runtime mapped or unmapped title memory `[at, at + len)`: what it
/// shows is new and read-write, so it is no longer guarded (nor known
/// unmapped).
pub fn title_memory_remapped(at: usize, len: usize) {
    // A placed texture there arrives, goes or changes with no descriptor
    // written.
    crate::descriptor::host::remapped(at, len);
    // The views the runtime told us of there are no longer what it shows.
    crate::inplace::views_gone(at, len);
    // And nothing stays in place over it.
    crate::inplace::title_memory_freeing(at, len);
    let mut g = guards();
    let (first, last) = (at / BLOCK, at.saturating_add(len).div_ceil(BLOCK));
    g.absent.remove(first, last);
    g.clean.remove(first, last);
    // New memory counts whole, whatever was open of what was there.
    let partly: Vec<usize> = g.parts.range(first..last).map(|(&block, _)| block).collect();
    for block in partly {
        g.parts.remove(&block);
    }
    let aliased: Vec<usize> = g.aliased.range(first..last).copied().collect();
    for block in aliased {
        g.aliased.remove(&block);
    }
    let held: Vec<usize> = g.held.range(first..last).copied().collect();
    for block in held {
        g.held.remove(&block);
    }
    g.hot.forget(first, last);
    if view_known() {
        // The runtime calls this for two things only (`xmem`: `map` and
        // `unmap`): a range it has just mapped a section's pages into,
        // and a view it is about to make a placeholder again. Either way
        // what stands there is view memory or a hole in it, which is what
        // `is_view` would answer for a range that begins there; and the
        // write watch's asking can only fail there. The title remaps some
        // 83 times a frame, and each time the blocks were forgotten, so
        // that the next asking about each range went to the kernel twice
        // (run 235: 266 to 565 failed askings a frame with the blocks
        // remembered from their guards alone, no fewer than without).
        g.viewed.extend(first..last);
        return;
    }
    // What the range shows is new: no longer known to be a view
    // (`VICTORY_VIEW_KNOWN`).
    let viewed: Vec<usize> = g.viewed.range(first..last).copied().collect();
    for block in viewed {
        g.viewed.remove(&block);
    }
    drop(g);
    // Mapping does not change what is write-watched; what was learned is
    // let go all the same, and learned again if still so (placeholders,
    // where the views are, stay known).
    kinds().unwatched.remove(first, last);
}

/// Registers the handler that takes writes to guarded blocks, once, ahead
/// of the crash reporter's.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH,
        EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_write(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        const WRITE: usize = 1;
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS
            || r.NumberParameters < 2
            || r.ExceptionInformation[0] != WRITE
        {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1];
        // Held across both checks: another thread may have opened the block
        // since this one faulted, and a copy may have guarded it again since
        // (then it is ours again, not read-only for some other reason).
        let mut g = guards();
        if open_fault(&mut g, at) {
            tally(S::Faults, 1);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::GuardFaults, 1);
            crate::framelog::add(crate::framelog::T::GuardFaults, 1);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        if writable(at) {
            EXCEPTION_CONTINUE_EXECUTION
        } else {
            EXCEPTION_CONTINUE_SEARCH
        }
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our guard state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_write));
    });
}

/// Whether `at` is committed read-write memory now.
fn writable(at: usize) -> bool {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_READWRITE, VirtualQuery,
    };
    // SAFETY: plain data filled in by the query.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a valid out pointer of the right size.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of_val(&info)) };
    n != 0 && info.State == MEM_COMMIT && info.Protect == PAGE_READWRITE
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    #[test]
    fn written_pages_are_reported_to_every_asker() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(
                std::ptr::null(),
                4 * PAGE,
                MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
                PAGE_READWRITE,
            )
        } as usize;
        assert_ne!(p, 0);
        // As the runtime tells of its reservations.
        title_memory_reserved(p, 4 * PAGE, false);
        let (start, end) = (p, p + 4 * PAGE);
        let (_, before) = writes(start, end, false).expect("watched");
        assert!(before.is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write(1) };
        let (now, after) = writes(start, end, false).expect("watched");
        assert_eq!(after, [(p / PAGE + 1, now)]);
        // Another asker after the kernel's bit was cleared still sees it.
        let (later, again) = writes(p + PAGE, p + 2 * PAGE, false).expect("watched");
        assert!(later > now);
        assert_eq!(again, [(p / PAGE + 1, now)]);
        let mut plain = vec![0u8; PAGE];
        assert!(
            writes(
                plain.as_mut_ptr() as usize,
                plain.as_ptr() as usize + PAGE,
                true
            )
            .is_none()
        );
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// A peek reports written pages without resetting the kernel's bits:
    /// asked again it says the same, and the next asking still finds them.
    #[test]
    fn a_peek_takes_nothing() {
        // SAFETY: a fresh write-watched allocation, freed at the end.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 2 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        title_memory_reserved(p, 2 * PAGE, false);
        let (start, end) = (p, p + 2 * PAGE);
        let _ = writes(start, end, false).expect("watched");
        assert!(peek_written(start, end).is_empty());
        // SAFETY: committed pages of ours.
        unsafe { ((p + PAGE + 8) as *mut u8).write_volatile(1) };
        assert_eq!(peek_written(start, end), [p / PAGE + 1]);
        assert_eq!(peek_written(start, end), [p / PAGE + 1]);
        let (now, written) = writes(start, end, false).expect("watched");
        assert!(written.contains(&(p / PAGE + 1, now)));
        assert!(peek_written(start, end).is_empty());
        // SAFETY: our allocation.
        unsafe { VirtualFree(p as *mut _, 0, MEM_RELEASE) };
    }

    /// The journal keeps which 4 KiB parts were written ([`fine`], on in
    /// tests): a copy that took a page at one asking is told only the parts
    /// written after it, by the kernel or as we noted them ([`wrote`]).
    #[test]
    fn written_parts_are_kept_per_asking() {
        let _turn = super::asking_test();
        // SAFETY: a fresh write-watched allocation, kept.
        let p = unsafe {
            VirtualAlloc(std::ptr::null(), 3 * PAGE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE)
        } as usize;
        assert_ne!(p, 0);
        let (start, end) = (p, p + 3 * PAGE);
        let page = p / PAGE;
        let (first, _) = writes(start, end, false).expect("watched");
        // SAFETY: committed pages of ours: part 3 of page 0, the last byte
        // of part 15 and the first of page 1 (two pages, one part each).
        unsafe {
            ((p + 3 * SUB + 9) as *mut u8).write_volatile(1);
            ((p + PAGE - 1) as *mut u8).write_volatile(1);
            ((p + PAGE) as *mut u8).write_volatile(1);
        }
        let (now, written) = writes(start, end, false).expect("watched");
        // (The journal is the whole process's: pages another test wrote
        // and freed before may be ours now, under older askings.)
        let written: Vec<(usize, u64)> = written.into_iter().filter(|&(_, w)| w > first).collect();
        assert_eq!(written, [(page, now), (page + 1, now)]);
        let mut parts = Vec::new();
        // A page the journal has no parts of (never written, so no copy
        // finds it stale) is all parts; a copy that never took a page,
        // too.
        subs_after(&[(page, first), (page + 1, first), (page + 2, first), (page, 0), (page, now)], &mut parts);
        assert_eq!(parts, [1 << 3 | 1 << 15, 1, ALL_SUBS, ALL_SUBS, 0]);
        // Written later in another part: a copy that took the page at
        // `now` is told that part alone, one that took it before both.
        // SAFETY: as above.
        unsafe { ((p + 7 * SUB) as *mut u8).write_volatile(2) };
        let (later, _) = writes(start, end, false).expect("watched");
        subs_after(&[(page, now), (page, first), (page, later)], &mut parts);
        assert_eq!(parts, [1 << 7, 1 << 3 | 1 << 7 | 1 << 15, 0]);
        // A write noted by us while a submission's one asking stands marks
        // the parts it touches.
        ask_once(&[(start, end)]);
        wrote(p + 2 * PAGE + 4 * SUB - 2, 4);
        ask_once_end();
        subs_after(&[(page + 2, later)], &mut parts);
        assert_eq!(parts, [1 << 3 | 1 << 4]);
        assert_eq!(parts_in(page, p + SUB, PAGE), !1);
        assert_eq!(parts_in(page, p + PAGE, 5), 0);
        // Not freed: the journal remembers these pages, and another test
        // given the same addresses would find them written.
    }

    /// With the guards' parts ([`fine_guards`], on in tests): a write into
    /// a guarded block opens its 4 KiB part and the block counts as written
    /// there; a write into a second part opens the block whole; our exact
    /// writes open the parts they write, and the kernel's write lands; a
    /// write announced for the kernel ([`writing`]) and a remapping count
    /// whole.
    #[test]
    fn guarded_blocks_open_a_part_at_a_time() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile};
        assert!(fine_guards());
        const SIZE: usize = 4 * BLOCK;
        // SAFETY: a pagefile section of four blocks, mapped read-write, kept
        // (the guards and the journal remember its addresses).
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, SIZE as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize
        };
        assert_ne!(view, 0);
        let page = view / PAGE;
        let all: Vec<(usize, u16)> = (page..page + 4).map(|p| (p, ALL_SUBS)).collect();
        assert_eq!(guard_parts(view, view + SIZE), Some(all), "never guarded: whole");
        assert_eq!(guard_parts(view, view + SIZE), Some(Vec::new()));
        let poke = |at: usize| {
            // SAFETY: inside our view; a guarded part faults and opens.
            unsafe { (at as *mut u8).write_volatile(0x5a) };
        };
        // One part of block 0, two parts of block 1 (one twice).
        poke(view + 5 * SUB + 17);
        poke(view + 5 * SUB + 18);
        poke(view + BLOCK + 2 * SUB);
        poke(view + BLOCK + 9 * SUB);
        poke(view + BLOCK + 9 * SUB + 1);
        assert_eq!(guard_parts(view, view + SIZE), Some(vec![(page, 1 << 5), (page + 1, ALL_SUBS)]));
        // Guarded again whole: the next write faults again.
        poke(view + 15 * SUB + 4095);
        assert_eq!(guard_parts(view, view + SIZE), Some(vec![(page, 1 << 15)]));
        // Our exact write across a part line in block 2: the kernel's
        // write lands, and only those parts count.
        let bytes = [7u8; 100];
        writing_exact(view + 2 * BLOCK + 4 * SUB - 50, bytes.len());
        crate::host::write_mapped((view + 2 * BLOCK + 4 * SUB - 50) as *mut u8, bytes.as_ptr(), bytes.len());
        // SAFETY: the bytes just written.
        let landed = unsafe { std::slice::from_raw_parts((view + 2 * BLOCK + 4 * SUB - 50) as *const u8, 100) };
        assert_eq!(landed, bytes);
        // A title write into a part opened that way does not open more.
        poke(view + 2 * BLOCK + 3 * SUB);
        assert_eq!(guard_parts(view, view + SIZE), Some(vec![(page + 2, 1 << 3 | 1 << 4)]));
        // A block opened in part, then announced for the kernel: whole.
        poke(view + 3 * BLOCK + 8);
        writing(view + 3 * BLOCK + 8 * SUB, 1);
        assert_eq!(guard_parts(view, view + SIZE), Some(vec![(page + 3, ALL_SUBS)]));
        // A block opened in part, then remapped: whole.
        poke(view + 6 * SUB);
        title_memory_remapped(view, BLOCK);
        assert_eq!(guard_parts(view, view + SIZE), Some(vec![(page, ALL_SUBS)]));
        // Into the journal with their parts, through the guards' askings.
        let (before, _) = writes(view, view + SIZE, true).expect("a view");
        poke(view + BLOCK + 11 * SUB);
        let (now, written) = writes(view, view + SIZE, true).expect("a view");
        assert!(written.contains(&(page + 1, now)));
        let mut parts = Vec::new();
        subs_after(&[(page + 1, before)], &mut parts);
        assert_eq!(parts, [1 << 11]);
    }

    /// A kernel read into a guarded view block fails; opened first, it
    /// lands and the block counts as written.
    #[test]
    fn kernel_reads_into_guarded_blocks_after_opening() {
        use std::io::{Read, Seek, Write};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, UnmapViewOfFile,
        };
        // SAFETY: a pagefile section of two blocks, mapped read-write.
        let view = unsafe {
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, 2 * BLOCK).Value as usize
        };
        assert_ne!(view, 0);
        assert!(guard(view, view + 2 * BLOCK).is_some()); // now read-only
        let mut file = tempfile::tempfile().unwrap();
        file.write_all(b"from the disk").unwrap();
        // `ReadFile` inside: the kernel writes the view.
        let read = |file: &mut std::fs::File| {
            file.rewind().unwrap();
            // SAFETY: 13 bytes of our view, nothing else refers to them.
            let into = unsafe { std::slice::from_raw_parts_mut((view + 100) as *mut u8, 13) };
            file.read(into).is_ok_and(|n| n == 13)
        };
        assert!(
            !read(&mut file),
            "the kernel does not fault into our handler"
        );
        writing(view + 100, 13);
        assert!(read(&mut file));
        // SAFETY: the bytes just read.
        let landed = unsafe { std::slice::from_raw_parts((view + 100) as *const u8, 13) };
        assert_eq!(landed, b"from the disk");
        assert_eq!(
            guard(view, view + 2 * BLOCK),
            Some((view / PAGE..view / PAGE + 1).collect())
        );
        // SAFETY: our view.
        unsafe {
            UnmapViewOfFile(
                windows_sys::Win32::System::Memory::MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                },
            )
        };
        title_memory_remapped(view, 2 * BLOCK);
    }

    #[test]
    fn writes_in_partly_committed_view_blocks_are_reported() {
        use windows_sys::Win32::Foundation::{CloseHandle, INVALID_HANDLE_VALUE};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile,
            SEC_RESERVE, UnmapViewOfFile,
        };

        const SIZE: usize = 3 * BLOCK;
        struct View(MEMORY_MAPPED_VIEW_ADDRESS);
        impl Drop for View {
            fn drop(&mut self) {
                title_memory_remapped(self.0.Value as usize, SIZE);
                // SAFETY: our view is still mapped.
                unsafe { UnmapViewOfFile(self.0) };
            }
        }
        // SAFETY: a new reserved pagefile section. The view holds its own
        // reference after we close the handle, including on assertion failure.
        let view = unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE | SEC_RESERVE,
                0,
                SIZE as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let view = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE);
            CloseHandle(section);
            assert!(!view.Value.is_null());
            View(view)
        };
        let start = view.0.Value as usize;
        let commit = |at: usize, len| {
            // SAFETY: these pages belong to our reserved view.
            let committed = unsafe { VirtualAlloc(at as _, len, MEM_COMMIT, PAGE_READWRITE) };
            assert_eq!(committed as usize, at);
        };
        let early = start + BLOCK + 3 * KERNEL_PAGE;
        let late = start + 2 * BLOCK + 3 * KERNEL_PAGE;
        commit(start, BLOCK);
        commit(early, KERNEL_PAGE);
        let refresh = || guard(start, start + SIZE).unwrap();
        assert_eq!(refresh(), [start / PAGE, early / PAGE, late / PAGE]);
        // The last block is still entirely reserved.
        assert!(!refresh().contains(&(late / PAGE)));

        commit(late, KERNEL_PAGE); // Ordinary commit, without a runtime mapping notification.
        for at in [early, late] {
            assert!(writable(at));
            // SAFETY: each address is inside its committed read-write page.
            unsafe { (at as *mut u8).write_volatile(1) };
        }
        assert_eq!(refresh(), [early / PAGE, late / PAGE]);
    }

    /// A range of view memory whose first block is not mapped is a view
    /// all the same: its blocks are reported once, then the ones written
    /// or mapped since.
    #[test]
    fn a_range_that_begins_in_an_unmapped_block_is_guarded() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of three; the middle block is mapped first.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (3 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        // Nothing mapped at all: a view with every block absent.
        assert!(is_view(p, p + 3 * BLOCK));
        let middle = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        // The range begins in the placeholder, its middle block is a view.
        assert!(is_view(p, p + 3 * BLOCK));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()), "nothing written since");
        // SAFETY: inside the mapped middle block.
        unsafe { ((p + BLOCK + 8) as *mut u8).write_volatile(1) };
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        let first = map(0);
        title_memory_remapped(p, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(0).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // Memory of ours that is no view stays none.
        let mut local = [0u8; 16];
        let at = local.as_mut_ptr() as usize;
        assert!(!is_view(at, at + local.len()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), middle, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// Blocks of a placeholder between views are reported once, then not
    /// again (they cannot be guarded) until the runtime maps them.
    #[test]
    fn unmapped_blocks_are_reported_once_until_mapped() {
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, MEM_PRESERVE_PLACEHOLDER, MEM_REPLACE_PLACEHOLDER,
            MEM_RESERVE_PLACEHOLDER, MapViewOfFile3, PAGE_NOACCESS, UnmapViewOfFile2,
            VirtualAlloc2,
        };
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        // SAFETY: a fresh placeholder of three blocks and a pagefile
        // section of two; block 0 shows the section's first block, as the
        // runtime maps physical pages.
        let (p, section) = unsafe {
            let p = VirtualAlloc2(
                GetCurrentProcess(),
                std::ptr::null(),
                3 * BLOCK,
                MEM_RESERVE | MEM_RESERVE_PLACEHOLDER,
                PAGE_NOACCESS,
                std::ptr::null_mut(),
                0,
            ) as usize;
            assert_ne!(p, 0);
            let section = CreateFileMappingW(
                -1isize as _,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                (2 * BLOCK) as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            (p, section)
        };
        let map = |block: usize| {
            // SAFETY: splits the placeholder at the block and maps the
            // section's block there.
            let view = unsafe {
                VirtualFree(
                    (p + block * BLOCK) as *mut _,
                    BLOCK,
                    MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER,
                );
                MapViewOfFile3(
                    section,
                    GetCurrentProcess(),
                    (p + block * BLOCK) as *const _,
                    (block * BLOCK) as u64,
                    BLOCK,
                    MEM_REPLACE_PLACEHOLDER,
                    PAGE_READWRITE,
                    std::ptr::null_mut(),
                    0,
                )
            };
            assert_eq!(view.Value as usize, p + block * BLOCK);
            view
        };
        let first = map(0);
        let pages = |block: usize| (p + block * BLOCK) / PAGE..(p + (block + 1) * BLOCK) / PAGE;
        let all: Vec<usize> = (0..3).flat_map(pages).collect();
        assert_eq!(guard(p, p + 3 * BLOCK), Some(all));
        assert_eq!(
            guard(p, p + 3 * BLOCK),
            Some(Vec::new()),
            "nothing written since"
        );
        let second = map(1);
        title_memory_remapped(p + BLOCK, BLOCK);
        assert_eq!(guard(p, p + 3 * BLOCK), Some(pages(1).collect()));
        assert_eq!(guard(p, p + 3 * BLOCK), Some(Vec::new()));
        // SAFETY: our views and placeholder.
        unsafe {
            UnmapViewOfFile2(GetCurrentProcess(), first, MEM_PRESERVE_PLACEHOLDER);
            UnmapViewOfFile2(GetCurrentProcess(), second, MEM_PRESERVE_PLACEHOLDER);
            for block in 0..3 {
                VirtualFree((p + block * BLOCK) as *mut _, 0, MEM_RELEASE);
            }
            windows_sys::Win32::Foundation::CloseHandle(section);
        }
        title_memory_remapped(p, 3 * BLOCK);
    }

    /// A pagefile section mapped twice: two views of the same physical
    /// pages, the second left for the test; the section handle, the two
    /// addresses.
    fn two_views(bytes: usize) -> (usize, usize, usize) {
        use windows_sys::Win32::Foundation::INVALID_HANDLE_VALUE;
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile,
        };
        // SAFETY: a fresh pagefile section, mapped twice; the views keep
        // the section alive and the handle is closed by the caller.
        unsafe {
            let section = CreateFileMappingW(
                INVALID_HANDLE_VALUE,
                std::ptr::null(),
                PAGE_READWRITE,
                0,
                bytes as u32,
                std::ptr::null(),
            );
            assert!(!section.is_null());
            let one = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            let two = MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, bytes).Value as usize;
            assert_ne!(one, 0);
            assert_ne!(two, 0);
            crate::inplace::title_memory_view(one, bytes, section as usize, 0);
            crate::inplace::title_memory_view(two, bytes, section as usize, 0);
            (section as usize, one, two)
        }
    }

    /// The section of [`two_views`] is closed and its views unmapped; the
    /// guard is told they are gone first.
    fn drop_views(section: usize, views: &[usize], bytes: usize) {
        use windows_sys::Win32::Foundation::CloseHandle;
        use windows_sys::Win32::System::Memory::{MEMORY_MAPPED_VIEW_ADDRESS, UnmapViewOfFile};
        for &view in views {
            title_memory_remapped(view, bytes);
            // SAFETY: a view of the section mapped by `two_views`.
            unsafe {
                UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS {
                    Value: view as *mut _,
                })
            };
        }
        // SAFETY: the section's last reference.
        unsafe { CloseHandle(section as *mut std::ffi::c_void) };
    }

    /// The title wrote the pages of a guarded texture through another view
    /// of them: the fault at that view is the guard's to see, and the
    /// guarded view's next check reports the pages written.
    #[test]
    fn a_write_through_another_view_is_reported_at_the_guarded_one() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        // The first check takes it in (and every view of its pages).
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        // Writing through the other view faults into the handler, which
        // opens that view and the guarded one: the next check sees it.
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((other + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A view of guarded pages is mapped anew: it is writable, so the
    /// guarded view checks its pages again (and guards the new view).
    #[test]
    fn a_view_mapped_beside_a_guarded_one_makes_it_check_again() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        title_memory_remapped(other, SIZE);
        crate::inplace::title_memory_view(other, SIZE, section, 0);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// A write of the runtime's own, told at one view of the pages
    /// (`writing`): the other views' next checks report the pages written.
    #[test]
    fn a_write_told_at_one_view_is_reported_at_the_others() {
        const SIZE: usize = 2 * BLOCK;
        let (section, guarded, other) = two_views(SIZE);
        let pages = |block: usize| (guarded + block * BLOCK) / PAGE;
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()), "nothing written yet");
        writing(other + BLOCK, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(1)]));
        assert_eq!(guard(guarded, guarded + SIZE), Some(Vec::new()));
        // Half a page and the next: both blocks, not just the first's.
        writing(other + BLOCK / 2, BLOCK);
        assert_eq!(guard(guarded, guarded + SIZE), Some(vec![pages(0), pages(1)]));
        drop_views(section, &[guarded, other], SIZE);
    }

    /// Both views of the pages are checked, each for a copy of its own
    /// (the title shows one physical page at several addresses, and each
    /// address has its host copy): a write through either is reported at
    /// both, whichever is checked first. Guarding the pages for the first
    /// view's check must not note the second view clean, or its copy would
    /// keep the bytes from before the write.
    #[test]
    fn a_write_is_reported_at_every_checked_view() {
        const SIZE: usize = 2 * BLOCK;
        let (section, one, two) = two_views(SIZE);
        let pages = |view: usize, block: usize| (view + block * BLOCK) / PAGE;
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0), pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0), pages(two, 1)]), "the second view's first check");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()), "nothing written yet");
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()), "nothing written yet");
        // SAFETY: the second view, read-write once the handler has opened it.
        unsafe { ((two + BLOCK + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 1)]));
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 1)]), "written through this view, checked second");
        // SAFETY: the first view, as above.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(two, two + SIZE), Some(vec![pages(two, 0)]));
        assert_eq!(guard(one, one + SIZE), Some(vec![pages(one, 0)]), "written through this view, checked second");
        assert_eq!(guard(one, one + SIZE), Some(Vec::new()));
        assert_eq!(guard(two, two + SIZE), Some(Vec::new()));
        drop_views(section, &[one, two], SIZE);
    }

    /// A page shown in more than `MANY_VIEWS` views is guarded at each by
    /// itself: the others stay writable, and nothing looks through them all.
    #[test]
    fn a_page_of_many_views_is_guarded_one_view_at_a_time() {
        let (section, one, two) = two_views(BLOCK);
        let filler = one + (1 << 45);
        for n in 0..MANY_VIEWS {
            crate::inplace::title_memory_view(filler + n * BLOCK, BLOCK, section, 0);
        }
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        assert!(writable(two), "another view of it is left alone");
        // SAFETY: the guarded view, opened by the handler.
        unsafe { ((one + 8) as *mut u8).write(1) };
        assert_eq!(guard(one, one + BLOCK), Some(vec![one / PAGE]));
        title_memory_remapped(filler, MANY_VIEWS * BLOCK);
        drop_views(section, &[one, two], BLOCK);
    }

    /// A write or mapping that runs past the end of the address space (a
    /// title's `fread` with a huge count) covers up to the end; the sum used
    /// to overflow. Up there no test has guarded blocks.
    #[test]
    fn ranges_past_the_address_space_end_do_not_overflow() {
        let top = usize::MAX - 16;
        writing(top, PAGE);
        title_memory_remapped(top, usize::MAX);
    }
}

#[cfg(test)]
mod mapping_tests {
    use super::{title_memory_reading, title_memory_remapping};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, mpsc};
    use std::time::Duration;

    /// A change of mapping waits for a read in place that is under way, and
    /// a read in place waits for a change of mapping.
    #[test]
    fn reads_in_place_and_remappings_wait_for_each_other() {
        // A reader holds; the change does not begin until it lets go.
        let changed = Arc::new(AtomicBool::new(false));
        let reading = title_memory_reading();
        let (began, wait) = mpsc::channel();
        let change = {
            let changed = changed.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _changing = title_memory_remapping();
                changed.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!changed.load(Ordering::SeqCst), "the mapping changed under a read in place");
        drop(reading);
        change.join().unwrap();
        assert!(changed.load(Ordering::SeqCst));
        // A change holds; the read does not begin until it is done.
        let read = Arc::new(AtomicBool::new(false));
        let changing = title_memory_remapping();
        let (began, wait) = mpsc::channel();
        let reader = {
            let read = read.clone();
            std::thread::spawn(move || {
                began.send(()).unwrap();
                let _reading = title_memory_reading();
                read.store(true, Ordering::SeqCst);
            })
        };
        wait.recv().unwrap();
        std::thread::sleep(Duration::from_millis(100));
        assert!(!read.load(Ordering::SeqCst), "a read in place began during a change of mapping");
        drop(changing);
        reader.join().unwrap();
        assert!(read.load(Ordering::SeqCst));
    }

    /// A read that waits for a change of mapping begins when that change
    /// ends, before the next one, however soon the runtime begins it (the
    /// standard lock let a run of changes go first: a texture refresh
    /// waited for all of them).
    #[test]
    fn a_read_waits_for_one_change_not_a_run_of_them() {
        if !super::fair() {
            return;
        }
        use std::sync::atomic::AtomicUsize;
        let ended = Arc::new(AtomicUsize::new(0));
        let changes = {
            let ended = ended.clone();
            std::thread::spawn(move || {
                for _ in 0..300 {
                    let changing = title_memory_remapping();
                    std::thread::sleep(Duration::from_millis(1));
                    drop(changing);
                    ended.fetch_add(1, Ordering::SeqCst);
                }
            })
        };
        while ended.load(Ordering::SeqCst) < 3 {
            std::thread::yield_now();
        }
        for _ in 0..20 {
            let before = ended.load(Ordering::SeqCst);
            let reading = title_memory_reading();
            let after = ended.load(Ordering::SeqCst);
            drop(reading);
            // The change under way when the read asked (the count is
            // raised just after a change lets go), and none more.
            assert!(after - before <= 2, "a read waited for {} changes", after - before);
        }
        changes.join().unwrap();
    }

    /// A read asked for without waiting, and refused, ends no read (the
    /// first fair lock made the refused guard and dropped it: the count of
    /// reads went below zero and every change after waited for ever).
    #[test]
    fn a_refused_read_ends_nothing() {
        if !super::fair() {
            return;
        }
        let changing = title_memory_remapping();
        assert!(super::title_memory_reading_now().is_none());
        drop(changing);
        // Other tests may be reading now, a few at most.
        assert!(super::FAIR.state().readers < 1000, "the count of reads went below zero");
    }

    /// Reads and changes from several threads at once: no read meets a
    /// change, and all of them end.
    #[test]
    fn reads_and_changes_never_overlap() {
        use std::sync::atomic::AtomicIsize;
        let inside = Arc::new(AtomicIsize::new(0));
        let threads: Vec<_> = (0..6)
            .map(|t| {
                let inside = inside.clone();
                std::thread::spawn(move || {
                    for i in 0..2000 {
                        if (t + i) % 3 == 0 {
                            let _changing = title_memory_remapping();
                            assert_eq!(inside.swap(-1, Ordering::SeqCst), 0, "a change met a read or a change");
                            std::hint::spin_loop();
                            assert_eq!(inside.swap(0, Ordering::SeqCst), -1);
                        } else if (t + i) % 3 == 1 {
                            let _reading = title_memory_reading();
                            assert!(inside.fetch_add(1, Ordering::SeqCst) >= 0, "a read met a change");
                            std::hint::spin_loop();
                            inside.fetch_sub(1, Ordering::SeqCst);
                        } else if let Some(_reading) = super::title_memory_reading_now() {
                            assert!(inside.fetch_add(1, Ordering::SeqCst) >= 0, "a read met a change");
                            inside.fetch_sub(1, Ordering::SeqCst);
                        }
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
    }
}

#[cfg(test)]
mod walk_tests {
    //! The guards' run sets and walk against what they replaced: sets of
    //! single blocks and a walk that looked at every block.
    use super::{Ahead, Guards, Runs, walk};
    use std::collections::BTreeSet;

    /// A small xorshift: the same cases at every run.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: usize) -> usize {
            (self.next() % n as u64) as usize
        }
    }

    fn members(runs: &Runs) -> BTreeSet<usize> {
        runs.0.iter().flat_map(|(&a, &b)| a..b).collect()
    }

    /// Runs apart (not touching), in order, none empty.
    fn well_formed(runs: &Runs) {
        let mut last_end = None;
        for (&a, &b) in &runs.0 {
            assert!(a < b, "empty run {a}..{b}");
            if let Some(end) = last_end {
                assert!(a > end, "runs touch or overlap at {a}");
            }
            last_end = Some(b);
        }
    }

    #[test]
    fn runs_behave_as_a_set_of_blocks() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        for _ in 0..200 {
            let mut runs = Runs::new();
            let mut model = BTreeSet::new();
            for _ in 0..200 {
                let a = rng.below(120);
                let b = a + rng.below(16);
                if rng.below(2) == 0 {
                    runs.insert(a, b);
                    model.extend(a..b);
                } else {
                    runs.remove(a, b);
                    for k in a..b {
                        model.remove(&k);
                    }
                }
                well_formed(&runs);
                assert_eq!(members(&runs), model);
                let (x, y) = (rng.below(130), rng.below(20));
                let y = x + y;
                assert_eq!(runs.contains(x), model.contains(&x));
                if x < y {
                    assert_eq!(runs.covers(x, y), (x..y).all(|k| model.contains(&k)));
                }
                let within: Vec<usize> = runs.within(x, y).into_iter().flat_map(|(a, b)| a..b).collect();
                let expected: Vec<usize> = model.range(x..y.max(x)).copied().collect();
                assert_eq!(within, expected);
                // Cut to the range and apart: neighbours inside it are one run.
                for pair in runs.within(x, y).windows(2) {
                    assert!(pair[0].1 < pair[1].0);
                }
            }
        }
    }

    /// Memory as the walk sees it: per block, committed or not, and the
    /// region (view or placeholder) it lies in. Protection fails across
    /// regions and on uncommitted blocks; a question about an uncommitted
    /// block answers to the end of its uncommitted region.
    struct Memory {
        committed: Vec<bool>,
        region: Vec<usize>,
    }

    impl Memory {
        fn uncommitted_end(&self, block: usize) -> usize {
            if block >= self.committed.len() || self.committed[block] {
                return block;
            }
            let mut end = block;
            while end < self.committed.len() && !self.committed[end] && self.region[end] == self.region[block] {
                end += 1;
            }
            end
        }

        fn settable(&self, from: usize, to: usize) -> bool {
            (from..to).all(|k| k < self.committed.len() && self.committed[k] && self.region[k] == self.region[from])
        }
    }

    /// The guards as they were, and their walk block by block (the code
    /// before the run sets, with the questions passed in).
    #[derive(Clone, Default)]
    struct Old {
        protected: BTreeSet<usize>,
        clean: BTreeSet<usize>,
        absent: BTreeSet<usize>,
    }

    fn old_protect(g: &mut Old, m: &Memory, from: usize, to: usize, calls: &mut Vec<(usize, usize)>) {
        calls.push((from, to));
        if m.settable(from, to) {
            g.protected.extend(from..to);
            g.clean.extend(from..to);
            return;
        }
        for block in from..to {
            if m.settable(block, block + 1) {
                g.protected.insert(block);
                g.clean.insert(block);
            } else if block < m.uncommitted_end(block) {
                g.absent.insert(block);
            }
        }
    }

    fn old_walk(g: &mut Old, m: &Memory, first: usize, last: usize, calls: &mut Vec<(usize, usize)>) -> Vec<usize> {
        let mut written = Vec::new();
        let mut run: Option<usize> = None;
        let clean: Vec<usize> = g.clean.range(first..last).copied().collect();
        if clean.len() == last - first {
            return written;
        }
        let mut next_clean = 0;
        let mut uncommitted_until = first;
        for block in first..=last {
            while next_clean < clean.len() && clean[next_clean] < block {
                next_clean += 1;
            }
            let is_clean = next_clean < clean.len() && clean[next_clean] == block;
            let mut open = block < last && !is_clean;
            if open && g.absent.contains(&block) {
                if block >= uncommitted_until {
                    uncommitted_until = m.uncommitted_end(block);
                }
                open = block >= uncommitted_until;
                if open {
                    g.absent.remove(&block);
                }
            }
            if open {
                written.push(block);
                if g.protected.contains(&block) {
                    g.clean.insert(block);
                } else {
                    run.get_or_insert(block);
                    continue;
                }
            }
            if let Some(from) = run.take() {
                old_protect(g, m, from, block, calls);
            }
        }
        written
    }

    fn new_protect(g: &mut Guards, m: &Memory, from: usize, to: usize, calls: &mut Vec<(usize, usize)>) {
        calls.push((from, to));
        if m.settable(from, to) {
            g.clean.insert(from, to);
            return;
        }
        for block in from..to {
            if m.settable(block, block + 1) {
                g.clean.insert(block, block + 1);
            } else if block < m.uncommitted_end(block) {
                g.absent.insert(block, block + 1);
            }
        }
    }

    fn new_walk(g: &mut Guards, m: &Memory, first: usize, last: usize, calls: &mut Vec<(usize, usize)>) -> Vec<usize> {
        let mut written = Vec::new();
        if first >= last || g.clean.covers(first, last) {
            return written;
        }
        let mut query = |block: usize| m.uncommitted_end(block);
        let mut protect = |g: &mut Guards, from: usize, to: usize| new_protect(g, m, from, to, calls);
        walk(g, first, last, &mut written, &mut query, &mut protect);
        written
    }

    /// Over random memory, guard states and sequences of askings, writes,
    /// commits and remaps: the same blocks reported written, the same
    /// protections asked for, the same blocks clean and absent after.
    #[test]
    fn the_walk_decides_as_the_block_by_block_walk_did() {
        const N: usize = 96;
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        for case in 0..400 {
            let mut region = Vec::with_capacity(N);
            let mut id = 0;
            for _ in 0..N {
                if rng.below(6) == 0 {
                    id += 1;
                }
                region.push(id);
            }
            let committed: Vec<bool> = (0..N).map(|_| rng.below(3) != 0).collect();
            let mut m = Memory { committed, region };
            let mut old = Old::default();
            let mut new = Guards {
                clean: Runs::new(),
                absent: Runs::new(),
                parts: std::collections::BTreeMap::new(),
                aliased: BTreeSet::new(),
                held: BTreeSet::new(),
                hot: super::Hot::new(),
                viewed: BTreeSet::new(),
            };
            for k in 0..N {
                match rng.below(5) {
                    0 | 1 if m.committed[k] => {
                        old.protected.insert(k);
                        old.clean.insert(k);
                        new.clean.insert(k, k + 1);
                    }
                    2 if !m.committed[k] => {
                        old.absent.insert(k);
                        new.absent.insert(k, k + 1);
                    }
                    _ => {}
                }
            }
            for step in 0..40 {
                match rng.below(4) {
                    // A write opens a block.
                    0 => {
                        let k = rng.below(N);
                        old.protected.remove(&k);
                        old.clean.remove(&k);
                        new.clean.remove(k, k + 1);
                    }
                    // An ordinary commit, unannounced.
                    1 => {
                        let k = rng.below(N);
                        m.committed[k] = true;
                    }
                    // A remap: new and read-write, nothing noted.
                    2 => {
                        let a = rng.below(N);
                        let b = (a + rng.below(8)).min(N);
                        for k in a..b {
                            m.committed[k] = rng.below(2) == 0;
                            old.protected.remove(&k);
                            old.clean.remove(&k);
                            old.absent.remove(&k);
                        }
                        new.clean.remove(a, b);
                        new.absent.remove(a, b);
                    }
                    _ => {}
                }
                let first = rng.below(N);
                let last = (first + 1 + rng.below(48)).min(N);
                let (mut old_calls, mut new_calls) = (Vec::new(), Vec::new());
                let a = old_walk(&mut old, &m, first, last, &mut old_calls);
                let b = new_walk(&mut new, &m, first, last, &mut new_calls);
                let at = format!("case {case} step {step} [{first}, {last})");
                assert_eq!(a, b, "written, {at}");
                assert_eq!(old_calls, new_calls, "protections, {at}");
                assert_eq!(old.protected, old.clean, "the old sets are one, {at}");
                assert_eq!(old.clean, members(&new.clean), "clean, {at}");
                assert_eq!(old.absent, members(&new.absent), "absent, {at}");
                well_formed(&new.clean);
                well_formed(&new.absent);
            }
        }
    }

    /// With view memory in the submission's one asking: its refreshes
    /// answer from the journal, a title write after the asking is seen by
    /// the next asking, and an asker that does not ask the guards (a
    /// texture's first look) is not answered for view memory.
    #[test]
    fn view_memory_asked_once_answers_from_the_journal() {
        let _turn = super::asking_test();
        use super::{BLOCK, PAGE, ask_once_end, ask_views, title_memory_remapped, writes};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE,
            UnmapViewOfFile,
        };
        const SIZE: usize = 4 * BLOCK;
        // SAFETY: a pagefile section of four blocks, mapped read-write.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, SIZE as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize
        };
        assert_ne!(view, 0);
        let page = view / PAGE;
        ask_once_end();
        ask_views(&[(view, view + SIZE), (view + BLOCK, view + 2 * BLOCK)]);
        let (now, written) = writes(view, view + SIZE, true).expect("asked ahead");
        // Never guarded before: every block is reported, under the asking.
        assert_eq!(written, (page..page + 4).map(|p| (p, now)).collect::<Vec<_>>());
        assert_eq!(writes(view + BLOCK, view + 2 * BLOCK, true).map(|w| w.0), Some(now), "from the journal");
        assert!(writes(view, view + SIZE, false).is_none(), "no write watch, and not asked of the guards");
        // SAFETY: inside the view; the write faults and opens the block.
        unsafe { ((view + 2 * BLOCK + 8) as *mut u8).write_volatile(7) };
        let (same, again) = writes(view, view + SIZE, true).unwrap();
        assert_eq!((same, again.iter().all(|&(_, w)| w == now)), (now, true), "the standing asking");
        ask_once_end();
        ask_views(&[(view, view + SIZE)]);
        let (next, written) = writes(view, view + SIZE, true).unwrap();
        assert!(next > now);
        let fresh: Vec<usize> = written.iter().filter(|&&(_, w)| w == next).map(|&(p, _)| p).collect();
        assert_eq!(fresh, [page + 2], "the next asking sees the write");
        ask_once_end();
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, SIZE);
    }

    /// A write-watched allocation of `blocks` blocks, told as the runtime
    /// tells of its reservations; freed when dropped.
    struct Watched(usize);
    impl Watched {
        fn new(blocks: usize) -> Self {
            use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
            // SAFETY: a fresh allocation, freed on drop.
            let p = unsafe {
                VirtualAlloc(std::ptr::null(), blocks * super::BLOCK, MEM_RESERVE | MEM_COMMIT | 0x20_0000, PAGE_READWRITE)
            } as usize;
            assert_ne!(p, 0);
            super::title_memory_reserved(p, blocks * super::BLOCK, false);
            Watched(p)
        }
    }
    impl Drop for Watched {
        fn drop(&mut self) {
            use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
            // SAFETY: our allocation.
            unsafe { VirtualFree(self.0 as *mut _, 0, MEM_RELEASE) };
        }
    }

    /// Memory found unwatched is not asked about again; a range across
    /// two reservations is known to fail, and so is one holding it, not
    /// one inside it; a reservation anew forgets what was known.
    #[test]
    fn unwatched_memory_is_learned_and_forgotten_when_reserved_anew() {
        use super::{BLOCK, PAGE, known_unwatched, learn_unwatched, title_memory_reserved, writes};
        let a = Watched::new(2);
        let b = Watched::new(2);
        let (a0, b0) = (a.0 / BLOCK, b.0 / BLOCK);
        assert!(!known_unwatched(a0, a0 + 2));
        assert!(writes(a.0, a.0 + 2 * PAGE, false).is_some());
        // Across both (whatever lies between): both ends are watched.
        let (lo, hi) = (a0.min(b0), a0.max(b0) + 2);
        learn_unwatched(lo, hi);
        assert!(known_unwatched(lo, hi));
        assert!(known_unwatched(lo.saturating_sub(1), hi + 1));
        assert!(!known_unwatched(a0, a0 + 2), "each reservation alone is watched");
        assert!(writes(b.0, b.0 + 2 * PAGE, false).is_some());
        // Our heap is not watched: learned at the first failure.
        let plain = vec![0u8; 3 * PAGE];
        let block = (plain.as_ptr() as usize).div_ceil(PAGE);
        assert!(writes(block * PAGE, block * PAGE + PAGE, false).is_none());
        assert!(known_unwatched(block, block + 1));
        // Told a placeholder: not asked; told write-watched again: asked.
        title_memory_reserved(a.0, 2 * BLOCK, true);
        assert!(known_unwatched(a0, a0 + 1));
        assert!(writes(a.0, a.0 + PAGE, false).is_none(), "taken for unwatched, as told");
        title_memory_reserved(a.0, 2 * BLOCK, false);
        assert!(writes(a.0, a.0 + PAGE, false).is_some());
        // What the test learned is let go for the tests after it.
        title_memory_reserved(lo * BLOCK, (hi - lo) * BLOCK, false);
        super::kinds().unwatched.remove(block, block + 1);
        drop(plain);
    }

    /// With the placeholders trusted, an absent block of a runtime
    /// placeholder is not asked about, and is reported once the runtime
    /// maps it; without, each walk asks.
    #[test]
    fn absent_blocks_of_placeholders_wait_for_the_runtime() {
        use super::{BLOCK, PAGE, Guards, Runs, kinds, walk};
        // A placeholder of four blocks the runtime told of, at block
        // numbers no other test uses (nothing is mapped or asked there).
        let base = 0x7ff0_0000_0000usize / BLOCK;
        kinds().placeholders.insert(base, base + 4);
        let mut g = Guards {
                clean: Runs::new(),
                absent: Runs::new(),
                parts: std::collections::BTreeMap::new(),
                aliased: BTreeSet::new(),
                held: BTreeSet::new(),
                hot: super::Hot::new(),
                viewed: BTreeSet::new(),
            };
        g.absent.insert(base, base + 4);
        let asked = std::cell::Cell::new(0);
        let protected = std::cell::RefCell::new(Vec::new());
        let mut written = Vec::new();
        // The walk as `guard` makes it with the switch on: the placeholder
        // answers.
        let mut trusted = |block: usize| match kinds().placeholders.run_end(block) {
            Some(end) => end,
            None => {
                asked.set(asked.get() + 1);
                block
            }
        };
        let mut protect = |_: &mut Guards, a: usize, b: usize| protected.borrow_mut().push((a, b));
        walk(&mut g, base, base + 4, &mut written, &mut trusted, &mut protect);
        assert_eq!((asked.get(), written.len(), protected.borrow().len()), (0, 0, 0));
        // The runtime maps block 1: no longer absent, it is reported.
        g.absent.remove(base + 1, base + 2);
        walk(&mut g, base, base + 4, &mut written, &mut trusted, &mut protect);
        assert_eq!(written, ((base + 1) * BLOCK / PAGE..(base + 2) * BLOCK / PAGE).collect::<Vec<_>>());
        assert_eq!(*protected.borrow(), [(base + 1, base + 2)]);
        assert_eq!(asked.get(), 0);
        super::title_memory_reserved(base * BLOCK, 4 * BLOCK, false);
        assert!(kinds().placeholders.run_end(base).is_none());
    }

    /// Overlapping view ranges of a submission are guarded as their union,
    /// once, and every one of them answers from the journal.
    #[test]
    fn overlapping_view_ranges_are_guarded_once() {
        let _turn = super::asking_test();
        use super::{BLOCK, PAGE, ask_once_end, ask_views, title_memory_remapped, writes};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE,
            UnmapViewOfFile,
        };
        const SIZE: usize = 6 * BLOCK;
        // SAFETY: a pagefile section of six blocks, mapped read-write.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, SIZE as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize
        };
        assert_ne!(view, 0);
        let page = view / PAGE;
        // Guarded once alone, so the guards know it.
        ask_once_end();
        ask_views(&[(view, view + SIZE)]);
        ask_once_end();
        // SAFETY: inside the view; the writes fault and open the blocks.
        unsafe {
            ((view + BLOCK) as *mut u8).write_volatile(1);
            ((view + 4 * BLOCK) as *mut u8).write_volatile(1);
        }
        let ranges = [(view, view + 3 * BLOCK), (view + 2 * BLOCK, view + 5 * BLOCK), (view + 5 * BLOCK, view + 6 * BLOCK)];
        ask_views(&ranges);
        let (now, written) = writes(view, view + SIZE, true).expect("the union was asked");
        let fresh: Vec<usize> = written.iter().filter(|&&(_, w)| w == now).map(|&(p, _)| p).collect();
        assert_eq!(fresh, [page + 1, page + 4]);
        for (a, b) in ranges {
            assert_eq!(writes(a, b, true).map(|w| w.0), Some(now));
        }
        ask_once_end();
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, SIZE);
    }

    /// The index of the ahead asking says what a look at each range said.
    #[test]
    fn the_ahead_index_answers_as_a_look_at_each_range() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for _ in 0..200 {
            let mut ahead = Ahead::new();
            let mut plain: Vec<(usize, usize, bool)> = Vec::new();
            // Kept sorted at each push (`VICTORY_AHEAD_INDEX=1`) or not.
            let sorted = rng.below(2) == 0;
            for _ in 0..rng.below(12) {
                let a = rng.below(100);
                let b = a + 1 + rng.below(30);
                let view = rng.below(3) == 0;
                ahead.push_as(a, b, view, sorted);
                plain.push((a, b, view));
                for _ in 0..20 {
                    let first = rng.below(110);
                    let last = first + 1 + rng.below(20);
                    for views in [false, true] {
                        let expected = plain.iter().any(|&(a, b, v)| (views || !v) && a <= first && last <= b);
                        assert_eq!(ahead.covers(first, last, views), expected);
                    }
                }
            }
            ahead.clear();
            assert!(!ahead.covers(0, 1, true));
        }
    }

    /// `VICTORY_HOT_BLOCKS`: a block found written at N askings in a row is
    /// left open and reported at every asking with no guard, until it is
    /// guarded again after the recheck; one not written stays clean.
    #[test]
    fn hot_blocks_are_reported_without_a_guard() {
        let m = Memory { committed: vec![true; 8], region: vec![0; 8] };
        let mut g = Guards {
            clean: Runs::new(),
            absent: Runs::new(),
            parts: std::collections::BTreeMap::new(),
            aliased: BTreeSet::new(),
            held: BTreeSet::new(),
            hot: super::Hot::new(),
            viewed: BTreeSet::new(),
        };
        g.clean.insert(0, 8);
        g.hot.n = 2;
        g.hot.recheck = 4;
        // Per asking: whether the title writes blocks 2 and 3 before it, and
        // the protections the walk asks for.
        let rounds: [(u64, bool, &[(usize, usize)]); 8] = [
            (1, true, &[(2, 4)]),
            // The second in a row: left open.
            (2, true, &[]),
            (3, true, &[]),
            (4, false, &[]),
            (5, true, &[]),
            // Four askings after: guarded again.
            (6, true, &[(2, 4)]),
            // Not written since: clean, not reported.
            (7, false, &[]),
            (8, true, &[(2, 4)]),
        ];
        for (round, write, protects) in rounds {
            g.hot.round = round;
            if write {
                // A write opens a guarded block (an open one stays open).
                g.clean.remove(2, 4);
            }
            let mut calls = Vec::new();
            let written = new_walk(&mut g, &m, 0, 8, &mut calls);
            let expected: Vec<usize> = if write || g.hot.is_open(2) { vec![2, 3] } else { vec![] };
            assert_eq!(written, expected, "reported, asking {round}");
            assert_eq!(calls, protects, "protections, asking {round}");
            // Open blocks are never noted clean.
            assert_eq!(g.clean.contains(2), !g.hot.is_open(2), "asking {round}");
            assert!(g.clean.contains(5), "never written, asking {round}");
        }
        // New memory: nothing learned stands.
        g.hot.round = 9;
        g.clean.remove(2, 4);
        let mut calls = Vec::new();
        new_walk(&mut g, &m, 0, 8, &mut calls);
        assert!(g.hot.is_open(2));
        g.hot.forget(0, 8);
        assert!(!g.hot.is_open(2) && g.hot.streak.is_empty());
    }

    /// `VICTORY_VIEWS_TRIM`: ranges inside one the guards know are passed
    /// over, and the asking notes what it notes without the switch.
    #[test]
    fn the_trimmed_view_asking_notes_the_same() {
        let _turn = super::asking_test();
        use super::{BLOCK, PAGE, ask_once_end, ask_views_as, title_memory_remapped, writes};
        use windows_sys::Win32::System::Memory::{
            CreateFileMappingW, FILE_MAP_WRITE, MEMORY_MAPPED_VIEW_ADDRESS, MapViewOfFile, PAGE_READWRITE,
            UnmapViewOfFile,
        };
        const SIZE: usize = 8 * BLOCK;
        // SAFETY: a pagefile section of eight blocks, mapped read-write.
        let view = unsafe {
            let section =
                CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, SIZE as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize
        };
        assert_ne!(view, 0);
        let page = view / PAGE;
        let b = |k: usize| view + k * BLOCK;
        let ranges = [(b(0), b(4)), (b(1), b(2)), (b(2), b(3) + 100), (b(3), b(8)), (b(4), b(5)), (b(6), b(7))];
        for trim in [true, false, true] {
            // Guarded once, so the guards know it.
            ask_once_end();
            ask_views_as(&[(view, view + SIZE)], trim);
            ask_once_end();
            // SAFETY: inside the view; the writes fault and open the blocks.
            unsafe {
                (b(1) as *mut u8).write_volatile(1);
                ((b(5) + 9) as *mut u8).write_volatile(1);
            }
            ask_views_as(&ranges, trim);
            let (now, written) = writes(view, view + SIZE, true).expect("the union was asked");
            let fresh: Vec<usize> = written.iter().filter(|&&(_, w)| w == now).map(|&(p, _)| p).collect();
            assert_eq!(fresh, [page + 1, page + 5], "trim {trim}");
            for (a, e) in ranges {
                assert_eq!(writes(a, e, true).map(|w| w.0), Some(now), "trim {trim}: from the journal");
            }
        }
        ask_once_end();
        // SAFETY: our view.
        unsafe { UnmapViewOfFile(MEMORY_MAPPED_VIEW_ADDRESS { Value: view as *mut _ }) };
        title_memory_remapped(view, SIZE);
    }
}
