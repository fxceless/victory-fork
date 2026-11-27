//! What the submitting thread waits on, per frame, and who ends each wait
//! (`VICTORY_LINE_WAITS=1`, diagnostics builds; off by default, and then
//! each hook is one load of a flag).
//!
//! The thread followed is the line's (`victory-submit`, which prepares and
//! sends the submissions with `VICTORY_ASYNC_SUBMIT=1`), or without it the
//! thread that submits. Every place it blocks is timed where it blocks
//! ([`began`], [`waited`]); the sums go out with the frame-rate line
//! ([`line`]). Who ended a wait is a thread by its id: the one that held
//! the lock when the wait began ([`Holder`], marked by each thread that
//! takes it), or the one that signalled the end; the card for fence
//! events. Ids are named at the line, by the threads' descriptions, their
//! numbers left out ("victory-wide-3" is "victory-wide").
//!
//! The waits for helpers of `crate::shadow::wide` are split: the part
//! before the last job of the call had begun (helpers not started yet:
//! waking, or busy with others' jobs) and the part after (helpers still
//! working); and how soon the first helper began, and how many jobs the
//! line ran itself. Those for the heaps' pass likewise: before the pass
//! for the submission began (the thread still on an earlier one, or not
//! woken yet), and in it.

use std::cell::Cell;
use std::sync::atomic::{AtomicU32, AtomicU64, AtomicUsize, Ordering::Relaxed};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

/// The pass over the title's view heaps on its own thread
/// (`crate::descriptor::sync_begin`, `victory-heap-sync`).
pub(crate) const HEAP_PASS: usize = 0;
/// Helpers of `crate::shadow::wide`.
pub(crate) const WIDE: usize = 1;
/// The guards' lock (`crate::watch`).
pub(crate) const GUARDS: usize = 2;
/// The write watch's journal lock (`crate::watch`).
pub(crate) const JOURNAL: usize = 3;
/// The runtime changing what title memory shows
/// (`crate::watch::title_memory_reading`).
pub(crate) const REMAPPING: usize = 4;
/// Flights landing (`relay_wait`: the thread that copies back).
pub(crate) const LANDINGS: usize = 5;
/// The card, at a fence's event.
pub(crate) const CARD: usize = 6;
/// The submission lock.
pub(crate) const SUBMISSION_LOCK: usize = 7;
/// Another thread's pass over the title's view heaps
/// (`crate::descriptor::sync_heaps`, one at a time).
pub(crate) const HEAP_TURN: usize = 8;
/// Nothing for the line to run (between frames, or the title not ahead).
pub(crate) const NOTHING_IN_LINE: usize = 9;
const SITES: usize = 10;

const NAMES: [&str; SITES] = [
    "the heaps' pass",
    "helpers of shadow::wide",
    "the guards' lock",
    "the journal's lock",
    "title memory being remapped",
    "landings (relay_wait)",
    "the card (fence events)",
    "the submission lock",
    "another thread's heaps' pass",
    "nothing in the line",
];

/// A wait the card ends.
pub(crate) const BY_CARD: u32 = u32::MAX;

/// Whether the account is kept.
#[inline]
pub(crate) fn on() -> bool {
    #[cfg(feature = "diagnostics")]
    {
        static ON: OnceLock<bool> = OnceLock::new();
        let on = *ON.get_or_init(|| std::env::var_os("VICTORY_LINE_WAITS").is_some_and(|v| v == "1"));
        #[cfg(test)]
        let on = on || ON_IN_TESTS.load(Relaxed);
        on
    }
    #[cfg(not(feature = "diagnostics"))]
    false
}

/// For tests: [`on`] from when this is set.
#[cfg(all(test, feature = "diagnostics"))]
static ON_IN_TESTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

thread_local! {
    static LINE: Cell<bool> = const { Cell::new(false) };
}

/// The line's thread, by its id.
static LINE_ID: AtomicU32 = AtomicU32::new(0);

/// This thread is the line's.
pub(crate) fn mark_line() {
    if !on() {
        return;
    }
    LINE.with(|l| l.set(true));
    LINE_ID.store(me(), Relaxed);
}

/// Without the line's own thread: the thread that submits is followed
/// (the first one to, alone).
#[inline]
pub(crate) fn mark_submitting() {
    if on() && LINE_ID.load(Relaxed) == 0 && LINE_ID.compare_exchange(0, me(), Relaxed, Relaxed).is_ok() {
        LINE.with(|l| l.set(true));
    }
}

/// Nanoseconds the line's thread has run on a processor (user and kernel).
fn line_cpu_ns() -> Option<u64> {
    use windows_sys::Win32::Foundation::{CloseHandle, FILETIME};
    use windows_sys::Win32::System::Threading::{GetThreadTimes, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    let id = LINE_ID.load(Relaxed);
    if id == 0 {
        return None;
    }
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut made, mut ended, mut kernel, mut user) = (zero, zero, zero, zero);
    // SAFETY: a thread of this process, opened to be asked its times and
    // closed after; the answers go into our own variables.
    let got = unsafe {
        let handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, id);
        if handle.is_null() {
            return None;
        }
        let got = GetThreadTimes(handle, &mut made, &mut ended, &mut kernel, &mut user) != 0;
        CloseHandle(handle);
        got
    };
    let of = |t: FILETIME| ((u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime)) * 100;
    got.then(|| of(kernel) + of(user))
}

fn line_thread() -> bool {
    LINE.try_with(Cell::get).unwrap_or(false)
}

/// This thread's id (a read of its own block, no call into the kernel).
pub(crate) fn me() -> u32 {
    // SAFETY: no arguments.
    unsafe { windows_sys::Win32::System::Threading::GetCurrentThreadId() }
}

/// The thread that took a lock (or signalled) last.
pub(crate) struct Holder(AtomicU32);

impl Holder {
    pub(crate) const fn new() -> Self {
        Holder(AtomicU32::new(0))
    }

    /// This thread has it (a store, only while the account is kept).
    #[inline]
    pub(crate) fn mark(&self) {
        if on() {
            self.0.store(me(), Relaxed);
        }
    }

    pub(crate) fn get(&self) -> u32 {
        self.0.load(Relaxed)
    }
}

/// When a wait of the line's thread begins; `None` on other threads or
/// with the account off.
#[inline]
pub(crate) fn began() -> Option<Instant> {
    (on() && line_thread()).then(Instant::now)
}

/// The wait at `site` that [`began`] is over; `by` ended it.
#[inline]
pub(crate) fn waited(site: usize, began: Option<Instant>, by: u32) {
    let Some(began) = began else {
        return;
    };
    let ns = began.elapsed().as_nanos() as u64;
    with_stats(|s| s.sites[site].add(ns, by));
}

#[derive(Default)]
struct Site {
    n: u64,
    ns: u64,
    longest: u64,
    /// Nanoseconds by who ended them.
    by: Vec<(u32, u64)>,
}

impl Site {
    fn add(&mut self, ns: u64, by: u32) {
        self.n += 1;
        self.ns += ns;
        self.longest = self.longest.max(ns);
        match self.by.iter_mut().find(|b| b.0 == by) {
            Some(b) => b.1 += ns,
            None => self.by.push((by, ns)),
        }
    }
}

#[derive(Default)]
struct Stats {
    sites: [Site; SITES],
    /// Calls of `wide`, their jobs, those the line ran; the first helper's
    /// start after the jobs were queued (summed, and the calls with one);
    /// the waiting before the call's last job had begun.
    wide: [u64; 6],
    /// The heaps' pass: waited before it began, in it.
    heap: [u64; 2],
}

static STATS: Mutex<Option<Stats>> = Mutex::new(None);

fn with_stats(f: impl FnOnce(&mut Stats)) {
    let mut s = STATS.lock().unwrap_or_else(|e| e.into_inner());
    f(s.get_or_insert_with(Stats::default));
}

/// Nanoseconds since the account began (for stamps other threads leave;
/// never 0).
pub(crate) fn stamp() -> u64 {
    static AT: OnceLock<Instant> = OnceLock::new();
    AT.get_or_init(Instant::now).elapsed().as_nanos() as u64 + 1
}

/// A call of `crate::shadow::wide` by the line's thread, followed.
pub(crate) struct WideCall {
    queued: u64,
    first: AtomicU64,
    last_start: AtomicU64,
    line_jobs: AtomicUsize,
    ended_by: AtomicU32,
}

/// For a call of `wide`, on the line's thread (`None` elsewhere).
#[inline]
pub(crate) fn wide_call() -> Option<Arc<WideCall>> {
    began()?;
    Some(Arc::new(WideCall {
        queued: stamp(),
        first: AtomicU64::new(u64::MAX),
        last_start: AtomicU64::new(0),
        line_jobs: AtomicUsize::new(0),
        ended_by: AtomicU32::new(0),
    }))
}

impl WideCall {
    /// A job of the call begins, on this thread.
    pub(crate) fn job_begins(&self) {
        let at = stamp().saturating_sub(self.queued);
        self.last_start.fetch_max(at, Relaxed);
        if line_thread() {
            self.line_jobs.fetch_add(1, Relaxed);
        } else {
            self.first.fetch_min(at, Relaxed);
        }
    }

    /// The call's last job was done by this thread.
    pub(crate) fn last_done(&self) {
        self.ended_by.store(me(), Relaxed);
    }

    /// The line waited from `waiting` (a [`stamp`]) to now for the call's
    /// `jobs` jobs.
    pub(crate) fn waited(&self, jobs: usize, waiting: u64) {
        let ns = stamp().saturating_sub(waiting);
        let unstarted = unstarted(self.queued, self.last_start.load(Relaxed), waiting, ns);
        let first = self.first.load(Relaxed);
        let by = self.ended_by.load(Relaxed);
        with_stats(|s| {
            s.sites[WIDE].add(ns, by);
            s.wide[0] += 1;
            s.wide[1] += jobs as u64;
            s.wide[2] += self.line_jobs.load(Relaxed) as u64;
            if first != u64::MAX {
                s.wide[3] += first;
                s.wide[4] += 1;
            }
            s.wide[5] += unstarted;
        });
    }
}

/// Of a wait of `ns` from `waiting` for jobs queued at `queued` (stamps),
/// the last of them begun `last_start` after: the part before that job
/// began (the helpers had not started it).
fn unstarted(queued: u64, last_start: u64, waiting: u64, ns: u64) -> u64 {
    (queued + last_start).saturating_sub(waiting).min(ns)
}

/// Of a wait of `ns` from `waiting` for a pass that began at `pass`
/// (stamps; 0 not known): the part before it began.
fn before_pass(waiting: u64, pass: u64, ns: u64) -> u64 {
    pass.saturating_sub(waiting).min(ns)
}

/// The heaps' pass was waited for from `waiting` to now (stamps); the pass
/// waited for began at `pass` (a stamp, 0 if not known).
pub(crate) fn heap_waited(waiting: u64, pass: u64) {
    let ns = stamp().saturating_sub(waiting);
    let before = before_pass(waiting, pass, ns);
    with_stats(|s| {
        s.sites[HEAP_PASS].add(ns, HEAP_THREAD.get());
        s.heap[0] += before;
        s.heap[1] += ns - before;
    });
}

/// The thread that runs the heaps' pass.
pub(crate) static HEAP_THREAD: Holder = Holder::new();

/// What `by` is called: its thread's description without its number.
fn name_of(by: u32, names: &mut std::collections::HashMap<u32, String>) -> String {
    match by {
        0 => "not known".into(),
        BY_CARD => "the card".into(),
        _ => names.entry(by).or_insert_with(|| group(&describe(by).unwrap_or_default())).clone(),
    }
}

/// A thread's name without its number ("victory-list-3": "victory-list").
fn group(name: &str) -> String {
    let name = name.trim_end_matches(|c: char| c.is_ascii_digit() || c == '-' || c == '_' || c == ' ' || c == '#');
    if name.is_empty() { "an unnamed thread".into() } else { name.into() }
}

/// The description of the thread `id`, if it is there and has one.
fn describe(id: u32) -> Option<String> {
    use windows_sys::Win32::Foundation::{CloseHandle, LocalFree};
    use windows_sys::Win32::System::Threading::{GetThreadDescription, OpenThread, THREAD_QUERY_LIMITED_INFORMATION};
    // SAFETY: a thread of this process, opened to be asked its name, closed
    // after; the system allocates the text, read up to its end and freed.
    unsafe {
        let handle = OpenThread(THREAD_QUERY_LIMITED_INFORMATION, 0, id);
        if handle.is_null() {
            return None;
        }
        let mut text: windows_sys::core::PWSTR = std::ptr::null_mut();
        let got = GetThreadDescription(handle, &mut text) >= 0 && !text.is_null();
        CloseHandle(handle);
        if !got {
            return None;
        }
        let mut len = 0;
        while *text.add(len) != 0 {
            len += 1;
        }
        let name = String::from_utf16_lossy(std::slice::from_raw_parts(text, len));
        LocalFree(text.cast());
        Some(name)
    }
}

/// The account since the last call, per frame, for the frame-rate line.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let Some(s) = STATS.lock().unwrap_or_else(|e| e.into_inner()).take() else {
        return;
    };
    static NAMES_SEEN: Mutex<Option<std::collections::HashMap<u32, String>>> = Mutex::new(None);
    let mut seen = NAMES_SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let names = seen.get_or_insert_with(Default::default);
    let ms = |ns: u64| ns as f64 / 1e6 / frames;
    let all: u64 = s.sites[..NOTHING_IN_LINE].iter().map(|site| site.ns).sum();
    let still = s.sites[NOTHING_IN_LINE].ns;
    // The wall time and the line's processor time since the last line: what
    // is neither waited nor run is the thread ready and not running (other
    // threads on its processors), or blocked where nothing here times it.
    static LAST: Mutex<Option<(Instant, u64)>> = Mutex::new(None);
    let now = (Instant::now(), line_cpu_ns().unwrap_or(0));
    let last = LAST.lock().unwrap_or_else(|e| e.into_inner()).replace(now);
    let mut parts = Vec::new();
    for (i, site) in s.sites.iter().enumerate() {
        if site.n == 0 {
            continue;
        }
        let mut by: Vec<(String, u64)> = Vec::new();
        for &(id, ns) in &site.by {
            let name = name_of(id, names);
            match by.iter_mut().find(|b| b.0 == name) {
                Some(b) => b.1 += ns,
                None => by.push((name, ns)),
            }
        }
        by.sort_unstable_by_key(|b| std::cmp::Reverse(b.1));
        let by: Vec<String> = by.iter().take(3).map(|(name, ns)| format!("{name} {:.2}", ms(*ns))).collect();
        parts.push(format!(
            "{} {:.2} ms ({:.1} waits, longest {:.2} ms, ended by {})",
            NAMES[i],
            ms(site.ns),
            site.n as f64 / frames,
            site.longest as f64 / 1e6,
            by.join(", ")
        ));
    }
    tracing::info!(
        "line waits per frame (VICTORY_LINE_WAITS=1): {:.2} ms in its work, {:.2} ms with nothing to run; {}",
        ms(all),
        ms(still),
        parts.join("; ")
    );
    if let Some((at, cpu)) = last {
        let wall = (now.0 - at).as_nanos() as u64;
        let ran = now.1.saturating_sub(cpu);
        tracing::info!(
            "  of a frame's {:.2} ms the line's thread ran {:.2} ms on a processor, waited {:.2} ms in its work and {:.2} ms for \
             work, and {:.2} ms neither (ready and not running, or blocked where nothing here times it)",
            ms(wall),
            ms(ran),
            ms(all),
            ms(still),
            ms(wall.saturating_sub(ran + all + still))
        );
    }
    let w = s.wide;
    let wide = s.sites[WIDE].ns;
    tracing::info!(
        "  of them, helpers of shadow::wide: {:.1} calls of {:.1} jobs a frame, {:.1} of the jobs run by the line itself; the \
         first helper began {:.0} us after the jobs were queued; of the {:.2} ms waited, {:.2} before the call's last job had \
         begun (helpers not started: waking, or on others' jobs), {:.2} with all begun (helpers still working); the heaps' \
         pass: {:.2} ms before it began (its thread on an earlier one, or waking), {:.2} in it",
        w[0] as f64 / frames,
        w[1] as f64 / frames,
        w[2] as f64 / frames,
        if w[4] > 0 { w[3] as f64 / w[4] as f64 / 1e3 } else { 0.0 },
        ms(wide),
        ms(w[5]),
        ms(wide.saturating_sub(w[5])),
        ms(s.heap[0]),
        ms(s.heap[1]),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_are_named_without_their_numbers() {
        assert_eq!(group("victory-wide-3"), "victory-wide");
        assert_eq!(group("victory-submit"), "victory-submit");
        assert_eq!(group("worker #12"), "worker");
        assert_eq!(group(""), "an unnamed thread");
        let mut names = std::collections::HashMap::new();
        assert_eq!(name_of(BY_CARD, &mut names), "the card");
        assert_eq!(name_of(0, &mut names), "not known");
        let named = std::thread::Builder::new()
            .name("victory-test-7".into())
            .spawn(|| {
                let id = me();
                (describe(id), name_of(id, &mut std::collections::HashMap::new()))
            })
            .unwrap()
            .join()
            .unwrap();
        assert_eq!(named.0.as_deref(), Some("victory-test-7"));
        assert_eq!(named.1, "victory-test");
    }

    #[test]
    fn a_wait_is_counted_with_who_ended_it() {
        let mut site = Site::default();
        site.add(5, 8);
        site.add(7, 8);
        site.add(3, 12);
        assert_eq!((site.n, site.ns, site.longest), (3, 15, 7));
        assert_eq!(site.by, [(8, 12), (12, 3)]);
    }

    #[test]
    fn waits_are_split_where_the_work_began() {
        // Jobs queued at 1000, the line waits from 1500 for 3000; the last
        // job began 900 after queueing: 400 of the wait before it.
        assert_eq!(unstarted(1000, 900, 1500, 3000), 400);
        // All begun before the wait: none of it.
        assert_eq!(unstarted(1000, 100, 1500, 3000), 0);
        // Never more than the wait.
        assert_eq!(unstarted(1000, 9000, 1500, 3000), 3000);
        // A pass that began 250 into a wait of 1000, one begun before it,
        // one not known.
        assert_eq!(before_pass(5000, 5250, 1000), 250);
        assert_eq!(before_pass(5000, 4000, 1000), 0);
        assert_eq!(before_pass(5000, 0, 1000), 0);
    }

    #[test]
    #[cfg(feature = "diagnostics")]
    fn the_lines_waits_are_followed_and_named() {
        ON_IN_TESTS.store(true, Relaxed);
        // Not the line's thread: nothing is timed.
        assert!(began().is_none());
        std::thread::Builder::new()
            .name("victory-test-line".into())
            .spawn(|| {
                mark_line();
                assert!(began().is_some());
                // A wide call: one job the line runs, one a helper runs late.
                let call = wide_call().unwrap();
                let waiting = stamp();
                call.job_begins();
                let c = call.clone();
                std::thread::Builder::new()
                    .name("victory-test-wide-2".into())
                    .spawn(move || {
                        std::thread::sleep(std::time::Duration::from_millis(2));
                        c.job_begins();
                        c.last_done();
                    })
                    .unwrap()
                    .join()
                    .unwrap();
                call.waited(2, waiting);
                // A lock another thread holds.
                static HELD: Holder = Holder::new();
                let lock = Arc::new(Mutex::new(()));
                let (taken, other) = (Arc::new(std::sync::Barrier::new(2)), lock.clone());
                let (t, done) = (taken.clone(), Arc::new(std::sync::Barrier::new(2)));
                let d = done.clone();
                let holder = std::thread::Builder::new()
                    .name("victory-test-holder".into())
                    .spawn(move || {
                        let g = other.lock().unwrap();
                        HELD.mark();
                        t.wait();
                        std::thread::sleep(std::time::Duration::from_millis(5));
                        drop(g);
                        // (Named while it is there.)
                        d.wait();
                    })
                    .unwrap();
                taken.wait();
                let (line, by) = (began(), HELD.get());
                drop(lock.lock().unwrap());
                waited(GUARDS, line, by);
                assert_eq!(name_of(by, &mut std::collections::HashMap::new()), "victory-test-holder");
                done.wait();
                holder.join().unwrap();
            })
            .unwrap()
            .join()
            .unwrap();
        let s = STATS.lock().unwrap();
        let s = s.as_ref().unwrap();
        assert!(s.wide[0] >= 1 && s.wide[1] >= 2 && s.wide[2] >= 1 && s.wide[4] >= 1, "{:?}", s.wide);
        // The helper's job began some 2 ms into the wait.
        assert!(s.wide[5] >= 1_000_000, "{:?}", s.wide);
        let guards = &s.sites[GUARDS];
        assert!(guards.n >= 1 && guards.ns >= 3_000_000);
    }
}
