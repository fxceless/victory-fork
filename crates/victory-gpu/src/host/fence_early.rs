//! How many flights the landing thread takes at once, cut where a flight
//! needs a fetch of card copies' pieces (`VICTORY_FENCE_EARLY`; 0, the
//! default: off; `1` on; `2` counts only, the takes as before), and the
//! times from "the card ran the flight" to "the title's fence behind it is
//! set", for the frame line.
//!
//! Run n53 (`VICTORY_RELAY_FREE_RAN=1`): the title's own time between
//! submissions 107-121 ms a frame, waiting on fences we set only when the
//! flights ahead of them land. The landing thread took every flight the
//! card had run (some 6 a landing, 28 a frame), and a landing whose flights
//! needed a fetch (4.5 a frame) set every fence in it after that fetch,
//! which stood behind 23-25 later flights on the host queue (70 ms). A
//! flight whose compare flagged nothing, with no card plan or nothing in
//! the card's memory, needs no fetch.
//!
//! With `1` a take is cut so that nothing that needs no fetch waits for
//! one: the things ahead of the first flight that needs one land on their
//! own (its fences set at once), then that flight with what follows it as
//! before (with `VICTORY_RELAY_LAND_ONE` that flight alone; flights that
//! need no fetch at the front of a take land together, not one a landing).
//!
//! Order is kept: a cut take is a take the landing thread could have made
//! anyway, had it looked when the card had run only those flights; every
//! landing still goes through `Host::relay_land` whole. Fences are only
//! ever set in the line's order: `crate::fence::set` stores the value as
//! given (no maximum) and wakes every waiter at or below it, and the title
//! waits for given values of fences of its own queues, so a later value
//! set early would both wake waits for work not merged yet and could move
//! a fence back when the earlier value came after it. Whether a flight
//! needs a fetch decides only where the take is cut, never what a landing
//! does: a wrong guess costs time, not correctness.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use super::{Flight, Late};

/// `VICTORY_FENCE_EARLY`: 0 off, 1 on, 2 counts only.
pub(super) fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level = std::env::var("VICTORY_FENCE_EARLY").ok().and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(0).min(2);
        if level > 0 && super::relay() {
            tracing::info!(
                level,
                "the landing thread's takes are cut ahead of the first flight that needs a fetch of card pieces \
                 (VICTORY_FENCE_EARLY=1; =2 counts only)"
            );
        }
        level
    })
}

/// The counters since the last frame line ([`n`] names them).
static STATS: [AtomicU64; n::ALL] = [const { AtomicU64::new(0) }; n::ALL];

mod n {
    /// Takes by the landing thread, flights in them, and of those flights
    /// the ones that needed a fetch.
    pub const TAKES: usize = 0;
    pub const FLIGHTS: usize = 1;
    pub const FETCHING: usize = 2;
    /// Ahead of the first flight needing a fetch in a take (with `2`: what
    /// a cut would have landed at once): flights, signals.
    pub const AHEAD_FLIGHTS: usize = 3;
    pub const AHEAD_SIGNALS: usize = 4;
    /// Takes cut (`1`), and of them those with no flight at all (signals
    /// and writes behind flights landed already).
    pub const CUTS: usize = 5;
    pub const CUTS_BARE: usize = 6;
    /// Signals set by landings; of them with a flight ahead in their
    /// landing: microseconds from the card having run it to the set, how
    /// many, the longest. The same for signals in cut takes.
    pub const SIGNALS: usize = 7;
    pub const SIG_US: usize = 8;
    pub const SIG_N: usize = 9;
    pub const SIG_MOST: usize = 10;
    pub const CUT_SIG_US: usize = 11;
    pub const CUT_SIG_N: usize = 12;
    /// Every signal: microseconds from the take to the set.
    pub const TAKE_SIG_US: usize = 13;
    /// Flights landed: microseconds from the card having run them to the
    /// end of their merge, how many.
    pub const LAND_US: usize = 14;
    pub const LAND_N: usize = 15;
    /// Takes made longer than one flight's with `VICTORY_RELAY_LAND_ONE`:
    /// flights needing no fetch at the front, landed together.
    pub const EXTENDS: usize = 16;
    pub const ALL: usize = 17;
}

fn add(i: usize, v: u64) {
    STATS[i].fetch_add(v, Relaxed);
}

/// The take being landed was cut ([`cut`]), and when it was taken
/// ([`super::now_us`]).
static CUT: AtomicBool = AtomicBool::new(false);
static TAKEN: AtomicU64 = AtomicU64::new(0);
/// The host fence value of the last flight landed, for a signal with no
/// flight ahead in its own landing.
static LAST: AtomicU64 = AtomicU64::new(0);

/// The compare's results (address, dwords), looked up outside the line's
/// lock (they are made once and stay).
pub(super) fn differs() -> Option<(usize, usize)> {
    if level() == 0 {
        return None;
    }
    let work = super::CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
    work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
}

/// Whether landing `flight` fetches pieces of card copies, the old way or
/// from its slots: some piece of its plan flagged by the compare (or in a
/// range not compared) with bytes in the copy, as `Host::relay_card_back`
/// and `Host::relay_card_inflight` pick them. Without the results: no
/// (nothing would be fetched).
fn needs_fetch(flight: &Flight, differs: Option<&[u32]>) -> bool {
    let Some(plan) = &flight.card else {
        return false;
    };
    let Some(differs) = differs else {
        return false;
    };
    let flags: Vec<(usize, u32, bool)> = plan
        .ranges
        .iter()
        .flat_map(|r| {
            (0..r.span.pieces).map(move |p| {
                let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0);
                (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, flagged)
            })
        })
        .collect();
    if !flags.iter().any(|f| f.2) {
        return false;
    }
    let fetched = super::flagged_pieces(&flags);
    plan.ranges.iter().any(|r| {
        let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
        (0..r.span.pieces).any(|p| {
            fetched.contains(&(copy, r.span.first + p)) && {
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                lo < hi
            }
        })
    })
}

/// What [`cut_at`] knows of a flight: whether its landing fetches, and
/// whether it is plain: no read-backs and no bottom-of-pipe writes behind
/// it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct Kind {
    pub fetching: bool,
    pub plain: bool,
}

/// Where a take of the first things of the line is cut, given what the
/// flights at their places are (`None` for a signal or write), `n` things
/// landable and `base` the take without the cut (`n`, or with
/// `VICTORY_RELAY_LAND_ONE` the first flight and what follows it).
///
/// - no flight: all of them;
/// - the first flight needs a fetch: what is ahead of it alone, if
///   anything is, else `base`;
/// - it needs none: up to the first later flight that needs one or that
///   follows a flight that is not plain (all, if there is none).
///
/// Why plain (run n58, the room unlit with `LAND_ONE`, flights needing
/// no fetch landed together): a landing merges the shader writes of all
/// its flights first and only then, flight by flight, does the
/// read-backs and bottom-of-pipe writes. Together, a later flight's
/// merges go into title memory before an earlier flight's read-backs
/// (the exposure's 1x1 and 5x1 targets) and writes, which then lie over
/// them. A flight is taken in with the ones before it only when none of
/// them is followed by anything of that kind, so no take puts anything
/// in another order than one flight a landing would.
pub(crate) fn cut_at(kinds: &[Option<Kind>], n: usize, base: usize) -> usize {
    let kinds = &kinds[..n];
    let Some(first) = kinds.iter().position(Option::is_some) else {
        return n;
    };
    if kinds[first].is_some_and(|k| k.fetching) {
        return if first > 0 { first } else { base };
    }
    let mut before = kinds[first].is_some_and(|k| k.plain);
    for (i, kind) in kinds.iter().enumerate().skip(first + 1) {
        let Some(kind) = kind else {
            continue;
        };
        if kind.fetching || !before {
            return i;
        }
        before = kind.plain;
    }
    n
}

/// How many of the `n` landable things at the front of `items` the
/// landing thread takes; `base` is the take without the cut.
pub(super) fn cut(items: &[Late], n: usize, base: usize, results: Option<(usize, usize)>) -> usize {
    let level = level();
    if level == 0 {
        return base;
    }
    // SAFETY: the compare's persistent mapping; the card has run the
    // compares of the flights looked at (they are landable), and their
    // dwords go to no other plan before they land (`Host::card_done`).
    let differs = results.map(|(at, words)| unsafe { std::slice::from_raw_parts(at as *const u32, words) });
    let kinds: Vec<Option<Kind>> = items[..n]
        .iter()
        .enumerate()
        .map(|(i, item)| match item {
            Late::Run(flight) => Some(Kind {
                fetching: needs_fetch(flight, differs),
                plain: !flight.order.iter().any(|&l| !flight.lists[l].host().deferred().backs.is_empty())
                    && !items[i + 1..].iter().take_while(|l| !matches!(l, Late::Run(_))).any(|l| matches!(l, Late::Write(..))),
            }),
            _ => None,
        })
        .collect();
    let fetching: Vec<Option<bool>> = kinds.iter().map(|k| k.map(|k| k.fetching)).collect();
    let first = fetching.iter().position(|f| *f == Some(true)).unwrap_or(n);
    let take = if level == 1 { cut_at(&kinds, n, base) } else { base };
    let cut = take < base;
    CUT.store(cut, Relaxed);
    TAKEN.store(super::now_us(), Relaxed);
    add(n::TAKES, 1);
    let taken = &fetching[..take];
    add(n::FLIGHTS, taken.iter().filter(|f| f.is_some()).count() as u64);
    add(n::FETCHING, taken.iter().filter(|f| **f == Some(true)).count() as u64);
    // What is ahead of the first flight needing a fetch, counted once: in
    // the take that lands it.
    if first < n && first <= take {
        let ahead = &items[..first];
        add(n::AHEAD_FLIGHTS, ahead.iter().filter(|i| matches!(i, Late::Run(_))).count() as u64);
        add(n::AHEAD_SIGNALS, ahead.iter().filter(|i| matches!(i, Late::Signal(..))).count() as u64);
    }
    if take > base {
        add(n::EXTENDS, 1);
    }
    if cut {
        add(n::CUTS, 1);
        if !taken.iter().any(Option::is_some) {
            add(n::CUTS_BARE, 1);
        }
    }
    take
}

/// When the host fence reached a value ([`super::now_us`]), in order of
/// value; the oldest go past [`KEEP`].
static DONE: Mutex<VecDeque<(u64, u64)>> = Mutex::new(VecDeque::new());
const KEEP: usize = 4096;

/// The thread that notes when the host fence moves (with [`level`] above
/// 0): it waits for the value after the one it saw, so each flight's run
/// is noted within a wake of its end.
pub(super) fn start_clock() {
    if level() == 0 {
        return;
    }
    let spawned = std::thread::Builder::new().name("victory-fence-clock".into()).spawn(|| {
        use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
        let Some(host) = super::host() else {
            return;
        };
        // SAFETY: a plain event of our own.
        let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
            tracing::warn!("no event for the clock of the host fence (VICTORY_FENCE_EARLY)");
            return;
        };
        let mut seen = 0u64;
        loop {
            // SAFETY: a plain query.
            let now = unsafe { host.fence.GetCompletedValue() };
            if now == u64::MAX {
                // Device removed: others say so.
                return;
            }
            if now > seen {
                seen = now;
                let mut done = DONE.lock().unwrap_or_else(|e| e.into_inner());
                done.push_back((now, super::now_us()));
                if done.len() > KEEP {
                    done.pop_front();
                }
            }
            // SAFETY: the host's fence and this thread's event.
            let waited = unsafe { host.fence.SetEventOnCompletion(seen + 1, event).map(|()| WaitForSingleObject(event, INFINITE)) };
            if waited.is_err() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        }
    });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "no clock of the host fence (VICTORY_FENCE_EARLY): no times card done -> fence set");
    }
}

/// When the card had run up to host fence value `fence`, if noted.
fn done_at(fence: u64) -> Option<u64> {
    let done = DONE.lock().unwrap_or_else(|e| e.into_inner());
    // The first note at or past it (the clock waits for each next value,
    // so that is within a wake of the run's end).
    done.get(done.partition_point(|d| d.0 < fence)).map(|d| d.1)
}

/// A flight with host fence value `fence` is landed (merged; its
/// read-backs, writes and signals follow).
pub(super) fn landed(fence: u64) {
    if level() == 0 {
        return;
    }
    LAST.store(fence, Relaxed);
    if let Some(done) = done_at(fence) {
        add(n::LAND_US, super::now_us().saturating_sub(done));
        add(n::LAND_N, 1);
    }
}

/// A title fence is set by a landing; `ahead`: the host fence value of
/// the last flight ahead of it in that landing.
pub(super) fn signalled(ahead: Option<u64>) {
    if level() == 0 {
        return;
    }
    let now = super::now_us();
    add(n::SIGNALS, 1);
    add(n::TAKE_SIG_US, now.saturating_sub(TAKEN.load(Relaxed)));
    // With no flight ahead in its landing it stood behind the last flight
    // landed before: from its run.
    let fence = ahead.unwrap_or_else(|| LAST.load(Relaxed));
    if fence == 0 {
        return;
    }
    let Some(done) = done_at(fence) else {
        return;
    };
    let us = now.saturating_sub(done);
    add(n::SIG_US, us);
    add(n::SIG_N, 1);
    STATS[n::SIG_MOST].fetch_max(us, Relaxed);
    if CUT.load(Relaxed) {
        add(n::CUT_SIG_US, us);
        add(n::CUT_SIG_N, 1);
    }
}

/// For the frame-rate line.
pub(super) fn line(frames: f64) {
    if level() == 0 {
        return;
    }
    let s: Vec<u64> = STATS.iter().map(|s| s.swap(0, Relaxed)).collect();
    let per = |i: usize| s[i] as f64 / frames;
    let avg = |us: usize, n: usize| if s[n] > 0 { s[us] as f64 / s[n] as f64 / 1000.0 } else { 0.0 };
    tracing::info!(
        "title fences behind flights per frame (VICTORY_FENCE_EARLY={}): the landing thread took {:.1} flights in {:.1} takes, \
         {:.1} of them needing a fetch of card pieces, {:.1} not; ahead of the first such flight of a take {:.1} flights and \
         {:.1} signals; takes cut ahead of a fetching flight {:.1} ({:.1} with no flight), takes of several flights needing none (VICTORY_RELAY_LAND_ONE) {:.1}; {:.1} signals set, card done -> \
         fence set {:.1} ms on average over {:.1} (the longest {:.1}), in cut takes {:.1} ms over {:.1}; take -> set {:.1} ms; \
         flights card done -> landed {:.1} ms on average over {:.1}",
        level(),
        per(n::FLIGHTS),
        per(n::TAKES),
        per(n::FETCHING),
        per(n::FLIGHTS) - per(n::FETCHING),
        per(n::AHEAD_FLIGHTS),
        per(n::AHEAD_SIGNALS),
        per(n::CUTS),
        per(n::CUTS_BARE),
        per(n::EXTENDS),
        per(n::SIGNALS),
        avg(n::SIG_US, n::SIG_N),
        per(n::SIG_N),
        s[n::SIG_MOST] as f64 / 1000.0,
        avg(n::CUT_SIG_US, n::CUT_SIG_N),
        per(n::CUT_SIG_N),
        if s[n::SIGNALS] > 0 { s[n::TAKE_SIG_US] as f64 / s[n::SIGNALS] as f64 / 1000.0 } else { 0.0 },
        avg(n::LAND_US, n::LAND_N),
        per(n::LAND_N),
    );
}

#[cfg(test)]
mod tests {
    use super::{Kind, cut_at};

    /// Nothing that needs no fetch waits behind a flight that needs one,
    /// and the order is never broken: a take is always a front of the
    /// landable things.
    #[test]
    fn takes_are_cut_ahead_of_the_first_fetch() {
        let kind = |fetching| Some(Kind { fetching, plain: true });
        let (f, t, s) = (kind(false), kind(true), None);
        // Flights needing none, then one that does: those first.
        assert_eq!(cut_at(&[f, s, f, t, s, f], 6, 6), 3);
        // Signals behind flights landed already, then a fetching flight:
        // the signals alone.
        assert_eq!(cut_at(&[s, s, t, f], 4, 4), 2);
        // The first flight fetches: the take as without the cut (all, or
        // one flight with what follows it).
        assert_eq!(cut_at(&[t, s, f, t], 4, 4), 4);
        assert_eq!(cut_at(&[t, s, f, t], 4, 2), 2);
        // With one flight a landing, flights needing none at the front
        // land together.
        assert_eq!(cut_at(&[f, s, f, f, s, t], 6, 2), 5);
        // None fetches, or no flight: everything landable.
        assert_eq!(cut_at(&[f, f, s], 3, 3), 3);
        assert_eq!(cut_at(&[s, s], 2, 2), 2);
        // Only the landable ones are looked at.
        assert_eq!(cut_at(&[f, f, t], 2, 2), 2);
    }

    /// A flight with read-backs or bottom-of-pipe writes behind it ends a
    /// take of flights needing no fetch: a later flight's merges would go
    /// in before them (run n58's unlit room).
    #[test]
    fn nothing_is_merged_ahead_of_an_earlier_flights_read_backs() {
        let (f, s) = (Some(Kind { fetching: false, plain: true }), None);
        let backs = Some(Kind { fetching: false, plain: false });
        // One flight a landing: the flight with read-backs lands alone
        // (as without the cut) or last in its take.
        assert_eq!(cut_at(&[backs, s, f, f], 4, 2), 2);
        assert_eq!(cut_at(&[f, s, backs, s, f, f], 6, 2), 4);
        assert_eq!(cut_at(&[f, f, backs], 3, 1), 3);
        // Without one flight a landing a take is never longer than before.
        assert_eq!(cut_at(&[f, backs, f], 3, 3), 2);
    }
}
