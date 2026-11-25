//! Lazy copy-back (`VICTORY_LAZY_BACK`, off by default): pages of title
//! memory that only the card's shaders write are left on the card instead
//! of being merged into title memory at every landing. See
//! `docs/lazy-copyback.md`.
//!
//! Step 1 (`VICTORY_LAZY_BACK=1`): bookkeeping only. Every piece the
//! copy-back fetches is noted per 64 KiB page of title memory; a page that
//! nothing on the CPU's side touched for [`QUIET_FRAMES`] frames is a
//! *candidate* (it would be card-only). Our own reads and writes of title
//! memory through the kernel (`crate::host`) count as touches, except the
//! copy-back's own ([`merging`]). Nothing is skipped and nothing is
//! protected; the frame line says what step 2 would skip and how often a
//! candidate would have been materialized.
//!
//! Step 2 (`VICTORY_LAZY_BACK=2`): a candidate page, right after a normal
//! merge, becomes **card-only**: every title address that shows it is made
//! `PAGE_NOACCESS` (once), and the guards (`crate::watch`) let it go. A
//! later landing does not merge its piece; it keeps the fetched bytes
//! instead ([`kept`]). The page is given back (**materialized**) when
//! anything touches it: the title's CPU code (an access violation, taken
//! by our handler) or our own kernel read or write ([`touched`]). Its kept
//! bytes go into title memory through our own view while the title views
//! are still protected, then their protection comes back.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Pages of title memory as the copy-back cuts them.
const PAGE: usize = 64 << 10;

/// A page untouched on the CPU's side for this many frames is a candidate.
pub(crate) const QUIET_FRAMES: u64 = 60;

/// The step `VICTORY_LAZY_BACK` asks for; 0: off.
pub(crate) fn step() -> u8 {
    static STEP: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *STEP.get_or_init(|| {
        let step = std::env::var("VICTORY_LAZY_BACK").ok().and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(0).min(2);
        if step > 0 {
            tracing::info!(
                step,
                "lazy copy-back (VICTORY_LAZY_BACK): 1 notes the pages only the card writes; 2 also keeps them from the CPU and \
                 skips their merges"
            );
        }
        step
    })
}

#[derive(Clone, Copy, Default)]
struct Page {
    /// The frame a fetch first brought it; 0: never.
    first: u64,
    /// The frame a fetch last brought it; 0: never.
    fetched: u64,
    /// The frame something on the CPU's side last touched it; 0: never.
    touched: u64,
}

impl Page {
    /// Fetched for [`QUIET_FRAMES`] frames at least, and nothing on the
    /// CPU's side touched it in the last [`QUIET_FRAMES`]. (Counted from the
    /// first fetch: the fetch being merged is noted before its merge, and a
    /// page made card-only at its first fetch was the first world frame's
    /// pages, all at once.)
    fn candidate(&self, frame: u64) -> bool {
        self.first != 0
            && frame >= self.first + QUIET_FRAMES
            && (self.touched == 0 || frame >= self.touched + QUIET_FRAMES)
    }
}

static PAGES: Mutex<Option<HashMap<usize, Page>>> = Mutex::new(None);

/// Since the last frame line: pieces fetched, those whose every page was a
/// candidate, their bytes; touches of candidates (each one a materialize
/// in step 2), touches of other noted pages.
static COUNT: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

thread_local! {
    /// Set while this thread runs the copy-back's merge: its own reads and
    /// writes of title memory are not touches.
    static MERGING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// Set while this thread gives a page back ([`materialize`]): what it
    /// reads and writes meanwhile does not give pages back in turn.
    static GIVING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Runs `f` as the copy-back's merge ([`MERGING`]).
pub(crate) fn merging<R>(f: impl FnOnce() -> R) -> R {
    let was = MERGING.with(|m| m.replace(true));
    let r = f();
    MERGING.with(|m| m.set(was));
    r
}

/// The copy-back is about to fetch title memory `[from, to)` (a piece).
pub(crate) fn fetched(from: usize, to: usize) {
    if step() == 0 || from >= to {
        return;
    }
    let frame = crate::frame::current().max(1);
    let mut pages = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    let pages = pages.get_or_insert_with(HashMap::new);
    let mut all = true;
    for page in from / PAGE..to.div_ceil(PAGE) {
        let p = pages.entry(page).or_default();
        all &= p.candidate(frame);
        if p.first == 0 {
            p.first = frame;
        }
        p.fetched = frame;
    }
    COUNT[0].fetch_add(1, Relaxed);
    if all {
        COUNT[1].fetch_add(1, Relaxed);
        COUNT[2].fetch_add((to - from) as u64, Relaxed);
    }
}

/// Our own read or write of title memory `[at, at + len)` through the
/// kernel, outside the copy-back's merge: a touch of the noted pages there.
pub(crate) fn touched(at: usize, len: usize) {
    if step() == 0 || len == 0 {
        return;
    }
    let (first, last) = (at / PAGE, at.saturating_add(len).div_ceil(PAGE));
    if step() >= 2 && any_card_only() && !GIVING.with(std::cell::Cell::get) {
        // The card-only views in the range: looked up page by page for a
        // short range, picked out of the (smaller) lookup for a long one (a
        // kernel write of the title's can name gigabytes).
        let keys: Vec<usize> = {
            let shown = SHOWN.read().unwrap_or_else(|e| e.into_inner());
            let Some(shown) = shown.as_ref() else { return };
            if last - first <= shown.len() {
                (first..last).filter_map(|p| shown.get(&(p * PAGE)).copied()).collect()
            } else {
                shown.iter().filter(|(v, _)| (first..last).contains(&(*v / PAGE))).map(|(_, k)| *k).collect()
            }
        };
        for key in keys {
            materialize(key, Why::Ours);
        }
    }
    if MERGING.with(std::cell::Cell::get) {
        return;
    }
    let frame = crate::frame::current().max(1);
    let mut pages = PAGES.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pages) = pages.as_mut() else { return };
    let touch = |p: &mut Page| {
        COUNT[if p.candidate(frame) { 3 } else { 4 }].fetch_add(1, Relaxed);
        p.touched = frame;
    };
    if last - first <= pages.len() {
        for page in first..last {
            if let Some(p) = pages.get_mut(&page) {
                touch(p);
            }
        }
    } else {
        for (_, p) in pages.iter_mut().filter(|(page, _)| (first..last).contains(*page)) {
            touch(p);
        }
    }
}

/// A card-only page: the copy it is of, the copy's offset and the last
/// fetched bytes of the piece there, and every title address that shows
/// it with the protection to give it back.
struct CardOnly {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    lo: usize,
    bytes: Vec<u8>,
    /// The send the fetch of `bytes` was after (as the merge is told).
    after: u64,
    shown: Vec<(usize, u32)>,
}

/// Card-only pages by the title address of the piece's page. Held while a
/// page is made card-only or given back, so that a fault meanwhile waits.
static CARD: Mutex<Option<HashMap<usize, CardOnly>>> = Mutex::new(None);

/// Every title address of a card-only page (all its views), to its key in
/// [`CARD`]. Its own lock, which the guards take under theirs: never held
/// while one of ours is taken.
static SHOWN: std::sync::RwLock<Option<HashMap<usize, usize>>> = std::sync::RwLock::new(None);

/// Pages that were ever card-only: a fault there after the page was given
/// back is a retry, not someone else's.
static EVER: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

static ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether any page is card-only now (cheap).
pub(crate) fn any_card_only() -> bool {
    ANY.load(Relaxed)
}

/// Whether the 64 KiB page at title address `at` is card-only.
pub(crate) fn is_card_only(at: usize) -> bool {
    any_card_only() && card_key(at).is_some()
}

fn card_key(at: usize) -> Option<usize> {
    SHOWN.read().unwrap_or_else(|e| e.into_inner()).as_ref()?.get(&(at / PAGE * PAGE)).copied()
}

/// Since the last frame line (step 2): pages made card-only, pieces kept
/// instead of merged and their bytes, pages given back for the title's
/// CPU, for our own access, and faults taken where the page had been
/// given back already.
static CARD_COUNT: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Since the last frame line (step 2): fetched pieces whose page did not
/// become card-only, by why: a piece over two pages; not a candidate; no
/// own views (`VICTORY_BACK_OWN_VIEW=0`); not view memory; too many views
/// or the page not among them; a view's protection not ours to change.
static REFUSED: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Step 2, at a landing, before merging a fetched piece of `shadow` at
/// offsets `[lo, hi)`: if its page is card-only, its bytes are kept for
/// when it is given back and the merge is skipped (true).
pub(crate) fn kept(shadow: &crate::shadow::Shadow, lo: usize, hi: usize, gpu: &[u8], after: u64) -> bool {
    if step() < 2 || !any_card_only() {
        return false;
    }
    let at = shadow.title_start() + lo;
    let Some(key) = card_key(at) else { return false };
    let mut card = CARD.lock().unwrap_or_else(|e| e.into_inner());
    let Some(page) = card.as_mut().and_then(|c| c.get_mut(&key)) else { return false };
    if page.lo != lo || page.bytes.len() != hi - lo || !std::ptr::eq(std::sync::Arc::as_ptr(&page.shadow), shadow) {
        return false;
    }
    page.bytes.copy_from_slice(gpu);
    page.after = after;
    CARD_COUNT[1].fetch_add(1, Relaxed);
    CARD_COUNT[2].fetch_add(gpu.len() as u64, Relaxed);
    true
}

/// Step 2, right after a landing merged a fetched piece of `shadow` at
/// offsets `[lo, hi)` (`gpu` its bytes): if its page is a candidate, it
/// becomes card-only.
pub(crate) fn merged(shadow: &std::sync::Arc<crate::shadow::Shadow>, lo: usize, hi: usize, gpu: &[u8], after: u64) {
    use windows_sys::Win32::System::Memory::{PAGE_NOACCESS, PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    if step() < 2 || hi <= lo {
        return;
    }
    let at = shadow.title_start() + lo;
    let page = at / PAGE * PAGE;
    let no = |why: usize| {
        REFUSED[why].fetch_add(1, Relaxed);
    };
    // One piece, one page: what the copy-back cuts.
    if (shadow.title_start() + hi - 1) / PAGE * PAGE != page {
        return no(0);
    }
    if card_key(page).is_some() {
        return;
    }
    let frame = crate::frame::current().max(1);
    let candidate = PAGES
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .as_ref()
        .and_then(|p| p.get(&(page / PAGE)).copied())
        .is_some_and(|p| p.candidate(frame));
    if !candidate {
        return no(1);
    }
    if !crate::inplace::back_own() {
        return no(2);
    }
    // Only view memory, which our own view can write while the title's
    // views are kept from the CPU; every view of the page with it.
    let views = crate::inplace::pages_of(page);
    if views.is_empty() {
        return no(3);
    }
    if views.len() > crate::watch::MANY_VIEWS || !views.contains(&page) {
        return no(4);
    }
    install();
    let mut card = CARD.lock().unwrap_or_else(|e| e.into_inner());
    let mut shown = Vec::with_capacity(views.len());
    // Each view's protection now, and that it is one we may take over:
    // looked at before anything changes, so that a page we leave alone is
    // left exactly as it was (the guards' too).
    let mut was = Vec::with_capacity(views.len());
    for &view in &views {
        match protection(view) {
            Some(p) if p == PAGE_READWRITE || p == PAGE_READONLY => was.push((view, p)),
            _ => return no(5),
        }
    }
    // In the lookup before any view is protected: a fault on one from now
    // on is ours, and waits for [`CARD`]'s lock (held here) before it is
    // given back or let go.
    {
        let mut map = SHOWN.write().unwrap_or_else(|e| e.into_inner());
        let map = map.get_or_insert_with(HashMap::new);
        for &(view, _) in &was {
            map.insert(view, page);
        }
    }
    EVER.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).extend(was.iter().map(|w| w.0));
    ANY.store(true, Relaxed);
    let forget = || {
        let mut map = SHOWN.write().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut() {
            for &(view, _) in &was {
                map.remove(&view);
            }
            ANY.store(!map.is_empty(), Relaxed);
        }
    };
    for &(view, protect) in &was {
        let mut old = 0;
        // SAFETY: a committed page of title view memory; only its
        // protection changes, and it is given back on a touch.
        if unsafe { VirtualProtect(view as *const _, PAGE, PAGE_NOACCESS, &mut old) } == 0 {
            break;
        }
        shown.push((view, protect));
    }
    if shown.len() != views.len() {
        no(5);
        // Not every view: none, each back to exactly what it was.
        for &(view, protect) in &shown {
            let mut old = 0;
            // SAFETY: as above.
            unsafe { VirtualProtect(view as *const _, PAGE, protect, &mut old) };
        }
        forget();
        return;
    }
    // Only now do the guards let the page go: a block they had read-only
    // comes back read-write on a give-back (they no longer watch it, and
    // the bytes it gets back count as written).
    for (view, back) in shown.iter_mut() {
        if crate::watch::release(*view) && *back == PAGE_READONLY {
            *back = PAGE_READWRITE;
        }
    }
    card.get_or_insert_with(HashMap::new).insert(
        page,
        CardOnly {
            shadow: shadow.clone(),
            lo,
            bytes: gpu.to_vec(),
            after,
            shown,
        },
    );
    let n = CARD_COUNT[0].fetch_add(1, Relaxed);
    static MADE: AtomicU64 = AtomicU64::new(0);
    if MADE.fetch_add(1, Relaxed) < 64 {
        tracing::info!(
            page = format_args!("{page:#x}"),
            buffer = format_args!("{:#x}", shadow.title_start()),
            views = views.len(),
            frame,
            this_line = n + 1,
            "lazy copy-back: a page made card-only"
        );
    }
}

#[derive(Clone, Copy)]
enum Why {
    Cpu,
    Ours,
}

/// Gives the card-only page `key` back: its kept bytes into title memory,
/// the views' protection back. Under [`CARD`]'s lock, so that a fault on
/// it meanwhile waits and then finds it given back.
fn materialize(key: usize, why: Why) {
    let mut card = CARD.lock().unwrap_or_else(|e| e.into_inner());
    let Some(page) = card.as_mut().and_then(|c| c.remove(&key)) else { return };
    // Out of the lookup first: what the give-back reads and writes must not
    // find the page card-only. A fault on it meanwhile waits for this lock
    // (the handler), then finds the protection given back.
    {
        let mut map = SHOWN.write().unwrap_or_else(|e| e.into_inner());
        if let Some(map) = map.as_mut() {
            for &(view, _) in &page.shown {
                map.remove(&view);
            }
            ANY.store(!map.is_empty(), Relaxed);
        }
    }
    let was = GIVING.with(|g| g.replace(true));
    // Merged as the copy-back merges, and through our own view while the
    // title's are still kept from the CPU: no title thread sees the page
    // half given back.
    let merged = page.shadow.card_materialize(page.lo, &page.bytes, page.after, true);
    for &(view, back) in &page.shown {
        let mut old = 0;
        // SAFETY: a page we kept from the CPU, given its protection back.
        unsafe { windows_sys::Win32::System::Memory::VirtualProtect(view as *const _, PAGE, back, &mut old) };
    }
    if let Some((bytes, false)) = &merged {
        page.shadow.card_materialize_kernel(page.lo, bytes);
    }
    static GIVEN: AtomicU64 = AtomicU64::new(0);
    if GIVEN.fetch_add(1, Relaxed) < 64 {
        tracing::info!(
            page = format_args!("{key:#x}"),
            why = match why {
                Why::Cpu => "the title's CPU",
                Why::Ours => "our own access",
            },
            way = match &merged {
                Some((_, true)) => "own view",
                Some((_, false)) => "kernel",
                None => "NOT MERGED",
            },
            frame = crate::frame::current(),
            "lazy copy-back: a card-only page given back"
        );
    }
    GIVING.with(|g| g.set(was));
    drop(card);
    // It is touched now: not a candidate again for a while.
    if let Some(p) = PAGES.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|p| p.get_mut(&(key / PAGE))) {
        p.touched = crate::frame::current().max(1);
    }
    CARD_COUNT[match why {
        Why::Cpu => 3,
        Why::Ours => 4,
    }]
    .fetch_add(1, Relaxed);
}

fn protection(at: usize) -> Option<u32> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    // SAFETY: a query into our own structure.
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: as above.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
    (n != 0 && info.State == MEM_COMMIT).then_some(info.Protect)
}

/// The handler of access violations on card-only pages: the page is given
/// back and the access goes on.
fn install() {
    use windows_sys::Win32::Foundation::{EXCEPTION_ACCESS_VIOLATION, NTSTATUS};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    use windows_sys::Win32::System::Memory::{PAGE_READONLY, PAGE_READWRITE};
    unsafe extern "system" fn on_fault(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        if r.ExceptionCode != EXCEPTION_ACCESS_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let at = r.ExceptionInformation[1] / PAGE * PAGE;
        if let Some(key) = card_key(at) {
            materialize(key, Why::Cpu);
            return EXCEPTION_CONTINUE_EXECUTION;
        }
        // Given back by another thread since this one faulted, or being
        // given back now: wait for that (its lock), then go again if the
        // page lets the access through.
        if EVER.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|e| e.contains(&at)) {
            drop(CARD.lock().unwrap_or_else(|e| e.into_inner()));
            const WRITE: usize = 1;
            let ok = match protection(at) {
                Some(p) if p == PAGE_READWRITE => true,
                Some(p) if p == PAGE_READONLY => r.ExceptionInformation[0] != WRITE,
                _ => false,
            };
            if ok {
                CARD_COUNT[5].fetch_add(1, Relaxed);
                return EXCEPTION_CONTINUE_EXECUTION;
            }
        }
        EXCEPTION_CONTINUE_SEARCH
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches our own state and protections.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_fault));
    });
}

/// With the frame line.
pub(crate) fn line(frames: f64) {
    if step() == 0 {
        return;
    }
    let [pieces, skipped, bytes, candidates, others] = [0, 1, 2, 3, 4].map(|i| COUNT[i].swap(0, Relaxed) as f64 / frames);
    let (noted, now) = {
        let frame = crate::frame::current().max(1);
        let pages = PAGES.lock().unwrap_or_else(|e| e.into_inner());
        pages.as_ref().map_or((0, 0), |p| (p.len(), p.values().filter(|p| p.candidate(frame)).count()))
    };
    tracing::info!(
        "lazy copy-back, noting only (VICTORY_LAZY_BACK=1), per frame: {pieces:.0} pieces fetched, {skipped:.0} of them all on \
         card-only candidates ({:.1} MB step 2 would not merge); our own touches of candidates {candidates:.1} (each a \
         materialize), of other noted pages {others:.1}; {noted} pages noted, {now} candidates now",
        bytes / (1 << 20) as f64
    );
    if step() >= 2 {
        let [made, kept, kept_bytes, cpu, ours, retried] = [0, 1, 2, 3, 4, 5].map(|i| CARD_COUNT[i].swap(0, Relaxed) as f64 / frames);
        let now = CARD.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, HashMap::len);
        tracing::info!(
            "lazy copy-back, card-only pages (VICTORY_LAZY_BACK=2), per frame: {made:.1} made card-only, {kept:.0} pieces kept \
             instead of merged ({:.1} MB), given back for the title's CPU {cpu:.1}, for our own access {ours:.1}, faults after \
             a give-back {retried:.1}; {now} card-only now",
            kept_bytes / (1 << 20) as f64
        );
        let r = [0, 1, 2, 3, 4, 5].map(|i| REFUSED[i].swap(0, Relaxed) as f64 / frames);
        tracing::info!(
            "lazy copy-back, pieces whose page stayed as it is, per frame: over two pages {:.1}, not a candidate {:.1}, no own \
             views {:.1}, not view memory {:.1}, too many views or not among them {:.1}, a view's protection not ours {:.1}",
            r[0],
            r[1],
            r[2],
            r[3],
            r[4],
            r[5]
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The rule on one page, by its counters: fetched twice and untouched
    /// is a candidate; a touch of ours (not the merge's) takes it out.
    #[test]
    fn untouched_fetched_pages_are_candidates_and_a_touch_is_not_the_merges() {
        assert!(!Page::default().candidate(1000), "never fetched");
        // Counted from the first fetch, not the last one.
        let fresh = Page { first: 10, fetched: 10, touched: 0 };
        assert!(!fresh.candidate(10), "not at its first fetch");
        assert!(!fresh.candidate(10 + QUIET_FRAMES - 1));
        assert!(fresh.candidate(10 + QUIET_FRAMES));
        let touched = Page { first: 1, fetched: 10, touched: 10 };
        assert!(!touched.candidate(10 + QUIET_FRAMES - 1));
        assert!(touched.candidate(10 + QUIET_FRAMES));
        // The merge's own accesses are not touches.
        assert!(!MERGING.with(std::cell::Cell::get));
        let inside = merging(|| MERGING.with(std::cell::Cell::get));
        assert!(inside);
        assert!(!MERGING.with(std::cell::Cell::get));
    }
}
