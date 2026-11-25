//! When the title may read what the card wrote for it: the flip wait
//! (`VICTORY_FLIP_LANDED`), and the counters that show how late those
//! results reach title memory (with `VICTORY_WALLS_TRACE=1` or
//! `VICTORY_RESULTS_TRACE=1`).
//!
//! What the card writes for the title's processor reaches title memory
//! when a flight lands (`crate::host`'s relay): what its shaders wrote,
//! its texture read-backs, its bottom-of-pipe values and query results,
//! and the fences behind it. A title that waits for a fence or a value
//! sees it in order. One that reads such memory after its frame pacing
//! alone does not wait for anything of ours: our frame events are a clock
//! (`crate::frame`), not the card.
//!
//! The title is paced by a blocking wait for the flip (frame event kind 2)
//! once a frame. On the console that event comes when a presented frame
//! has been shown, after the card has run all of that frame's work; ours
//! came on the next tick of a clock, whatever the card or our submit
//! thread had done. A title that reads what a frame's work wrote once the
//! flip of that frame has passed, without a fence, reads it from older
//! frames here whenever the work has not landed by then.
//!
//! Without `VICTORY_ASYNC_SUBMIT` the submissions of the frame before were
//! prepared and sent inside the title's own submit calls, so by its next
//! flip wait the card had nearly always run them and they had landed (and
//! sometimes not: the first room's left wall goes missing in some runs).
//! With it, a frame's submissions stand in the queues' lines and are
//! prepared by our thread some 60 ms (up to 140 ms) after the title made
//! them (runs n51, n55), and the present goes behind them: at the flip
//! wait the frame before is far from prepared, every frame. Those runs
//! lost the first room's walls every time.
//!
//! With the switch the flip wait also waits until everything the title
//! submitted before the present of the frame before the one it is
//! recording has been sent and has landed ([`FRAME_BEFORE`]). The title's
//! own frame still runs beside our thread's preparing of the frame before
//! (what async submit is for); only the end of the frame before is waited
//! for, where the console's flip waits for it too.
//!
//! What it assumes: that this is what the title relies on. The trace's
//! counters say how often the title gives a result slot to the card again
//! before the result before had landed ([`submitted`]), how often its flip
//! wait found the frame before not landed, and (with
//! `VICTORY_ASYNC_RACE=1`, `crate::host::race_note`) whether the title
//! rewrites memory a submission reads after its submit call instead.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// Whether a flip wait waits until the work of the frame before has landed
/// ([`flip_wait`]; `VICTORY_FLIP_LANDED`: `1` on, `0` off; unset: on with
/// `VICTORY_ASYNC_SUBMIT=1`, off without it; needs `VICTORY_LATE_WAIT`).
/// Read once.
pub(crate) fn flip_landed() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = match std::env::var("VICTORY_FLIP_LANDED").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => crate::queue::behind(),
        } && crate::host::late();
        if on {
            tracing::info!(
                "a flip wait waits until the work of the frame before has landed (VICTORY_FLIP_LANDED)"
            );
        }
        on
    })
}

/// Whether the results trace is on (`VICTORY_WALLS_TRACE=1|2` or
/// `VICTORY_RESULTS_TRACE=1`). Read once.
pub(crate) fn trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        crate::host::walls_trace() || std::env::var_os("VICTORY_RESULTS_TRACE").is_some_and(|v| v == "1")
    })
}

/// The frame event the title waits for before each present (`crate::frame`).
pub(crate) const FLIP: u32 = 2;

/// The title's last present call: when, and the submissions started then
/// (`crate::host::started`).
static PRESENT_AT: Mutex<Option<(Instant, u64)>> = Mutex::new(None);

/// [`PRESENT_AT`] as it was at the title's last frame-origin wait: the
/// present of the frame before the one the title is recording, whether it
/// waits for the flip before its present or after it (`None`: no origin
/// wait yet, [`PRESENT_AT`] is taken).
static FRAME_BEFORE: Mutex<Option<(Instant, u64)>> = Mutex::new(None);

/// Parts of the queues' lines sent so far, the latest last: when the title
/// put each in its line, and the submissions started when it had been sent
/// (its own number or above). The flip wait takes from it the last number
/// of the parts put in before a present.
static SENT: Mutex<VecDeque<(Instant, u64)>> = Mutex::new(VecDeque::new());

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// The title calls a present (its thread, before anything of it runs).
pub(crate) fn present_called() {
    let now = Instant::now();
    *lock(&PRESENT_AT) = Some((now, crate::host::started()));
    if trace() {
        event('P');
    }
}

/// Our thread (or a title thread running the lines) has sent a part of a
/// queue's line that the title put in at `at`.
pub(crate) fn sent(at: Instant) {
    let mut sent = lock(&SENT);
    sent.push_back((at, crate::host::started()));
    while sent.len() > 1024 {
        sent.pop_front();
    }
}

/// The highest submission number among the parts put in before `t` that
/// have been sent (0: none known).
fn sent_before(t: Instant) -> u64 {
    lock(&SENT).iter().filter(|(at, _)| *at < t).map(|&(_, n)| n).max().unwrap_or(0)
}

/// Since the last line ([`line`]); [`s`] names them.
static STATS: [AtomicU64; s::ALL] = [const { AtomicU64::new(0) }; s::ALL];

mod s {
    /// Flip waits; of them those that found work submitted before the last
    /// present not landed; the parts of the lines put in before it (summed
    /// over those waits); submissions sent and not landed (summed).
    pub const FLIPS: usize = 0;
    pub const FOUND: usize = 1;
    pub const UNSENT: usize = 2;
    pub const IN_AIR: usize = 3;
    /// Waited (microseconds), and the waits that gave up (nothing moved,
    /// or a second and a half went by).
    pub const WAITED_US: usize = 4;
    pub const GAVE_UP: usize = 5;
    /// Results submitted, by kind (bottom-of-pipe values, timestamps,
    /// read-backs), and of them those submitted to an address whose
    /// result before had not reached title memory yet.
    pub const SUBMITTED: usize = 6;
    pub const AGAIN: usize = 9;
    /// Results landed, by frames between their submission and the landing
    /// (0, 1, 2, 3 or more).
    pub const LAG: usize = 12;
    /// Frame waits by kind: origin, 1, the flip, others.
    pub const WAITS: usize = 16;
    /// `VICTORY_ASYNC_RACE=1` (`crate::host::race_note`): reads compared,
    /// and those the title changed between its submit call and our
    /// preparing, by kind (fills, card fills, title copies' sources, small
    /// refreshes), and their bytes.
    pub const RACE_N: usize = 20;
    pub const RACE_CHANGED: usize = 21;
    pub const RACE_BYTES: usize = 25;
    pub const ALL: usize = 26;
}

fn add(i: usize, n: u64) {
    STATS[i].fetch_add(n, Ordering::Relaxed);
}

/// A frame wait of `kind`: an origin wait notes the present before it
/// ([`FRAME_BEFORE`]); counted with the trace.
pub(crate) fn frame_wait(kind: u32) {
    if kind == 0 {
        let last = *lock(&PRESENT_AT);
        *lock(&FRAME_BEFORE) = last;
    }
    if trace() {
        add(s::WAITS + (kind as usize).min(3), 1);
        event(match kind {
            0 => 'O',
            FLIP => 'F',
            _ => 'v',
        });
    }
}

/// A blocking flip wait of the title's (`timeout_ms`, as it passed it):
/// counted, and with [`flip_landed`] waited until what the title submitted
/// before the present of the frame before the one it records has landed
/// ([`FRAME_BEFORE`]). Runs before the wait for the tick.
pub(crate) fn flip_wait(timeout_ms: u32) {
    let counting = trace();
    let waiting = flip_landed() && timeout_ms > 0;
    if !counting && !waiting {
        return;
    }
    let before = *lock(&FRAME_BEFORE);
    let Some((t, started_then)) = before.or(*lock(&PRESENT_AT)) else {
        return;
    };
    let target = |sent_then: u64| started_then.max(sent_then);
    let unsent = crate::queue::parts_put_before(t);
    let mut goal = target(sent_before(t));
    let finished = crate::host::finished();
    let behind = unsent > 0 || finished < goal;
    if counting {
        add(s::FLIPS, 1);
        if behind {
            add(s::FOUND, 1);
            add(s::UNSENT, unsent as u64);
            add(s::IN_AIR, goal.saturating_sub(finished));
        }
    }
    if !waiting || !behind {
        return;
    }
    // The title's wait may be bounded: so is ours (and never past a second
    // and a half, the time the sweep settles everything anyway).
    let limit = Duration::from_millis(u64::from(timeout_ms).min(1500));
    let began = Instant::now();
    let mut still = 0;
    // Once: our thread is woken again by whatever moves a line (a landing,
    // a signal, a submission), and the look for "nothing moves" below must
    // not find it woken by us.
    crate::queue::nudge();
    let met = loop {
        std::thread::sleep(Duration::from_millis(1));
        if crate::queue::parts_put_before(t) == 0 {
            goal = target(sent_before(t));
            if crate::host::finished() >= goal {
                break true;
            }
        }
        // Nothing moves any more: a line stands at a wait nothing of ours
        // meets (the title's next present forces it), and nothing is in
        // the air or being landed.
        let idle = !crate::queue::lines_moving() && !crate::host::line_moving();
        still = if idle { still + 1 } else { 0 };
        if still >= 3 || began.elapsed() >= limit {
            break false;
        }
    };
    add(s::WAITED_US, began.elapsed().as_micros() as u64);
    if !met {
        add(s::GAVE_UP, 1);
    }
}

/// Lines logged for reads the title changed after submitting them.
static RACE_LOGGED: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_ASYNC_RACE=1`: a read of `kind` (`n` bytes) compared at our
/// preparing with what it held at the title's submit call; `changed`: it
/// differed (`what` says where, for the first lines).
pub(crate) fn race(kind: u8, n: usize, changed: bool, what: impl FnOnce() -> String) {
    add(s::RACE_N, 1);
    if !changed {
        return;
    }
    let kind = usize::from(kind.min(3));
    add(s::RACE_CHANGED + kind, 1);
    add(s::RACE_BYTES, n as u64);
    if RACE_LOGGED.fetch_add(1, Ordering::Relaxed) < 32 {
        tracing::info!(
            frame = crate::frame::current(),
            kind = ["fill", "card fill", "title copy source", "small refresh"][kind],
            read = what(),
            "async race: title memory a submission reads changed between the title's submit call and our preparing (VICTORY_ASYNC_RACE=1)"
        );
    }
}

/// Results the title is given, not landed yet: by address, the frames
/// they were submitted in, oldest first.
static PENDING: Mutex<Option<HashMap<usize, VecDeque<u64>>>> = Mutex::new(None);

/// Of a result: a bottom-of-pipe value (query results among them), a
/// timestamp, a texture read-back.
pub(crate) const VALUE: u8 = 0;
pub(crate) const TIMESTAMP: u8 = 1;
pub(crate) const READ_BACK: u8 = 2;

/// Lines logged for results submitted again before the one before landed.
static AGAIN_LOGGED: AtomicU64 = AtomicU64::new(0);

/// The title submitted lists whose results go to `spans` (address, kind)
/// when they land (with the trace).
pub(crate) fn submitted(spans: &[(usize, u8)]) {
    if spans.is_empty() {
        return;
    }
    let frame = crate::frame::current();
    let mut pending = lock(&PENDING);
    let pending = pending.get_or_insert_with(HashMap::new);
    for &(at, kind) in spans {
        let kind = usize::from(kind.min(2));
        add(s::SUBMITTED + kind, 1);
        let line = pending.entry(at).or_default();
        if let Some(&oldest) = line.front() {
            add(s::AGAIN + kind, 1);
            if AGAIN_LOGGED.fetch_add(1, Ordering::Relaxed) < 24 {
                tracing::info!(
                    frame,
                    at = format_args!("{at:#x}"),
                    kind = ["value", "timestamp", "read-back"][kind],
                    pending = line.len(),
                    oldest_frame = oldest,
                    "walls trace: a result submitted again before the one before reached title memory (the title has read the slot or given up on it)"
                );
            }
        }
        // (Bounded: a list that never lands must not grow it for good.)
        if line.len() >= 16 {
            line.pop_front();
        }
        line.push_back(frame);
    }
}

/// A result reached title memory at `at` (with the trace).
pub(crate) fn landed(at: usize) {
    let frame = crate::frame::current();
    let mut pending = lock(&PENDING);
    let Some(pending) = pending.as_mut() else {
        return;
    };
    let Some(line) = pending.get_mut(&at) else {
        return;
    };
    if let Some(then) = line.pop_front() {
        add(s::LAG + (frame.saturating_sub(then) as usize).min(3), 1);
    }
    if line.is_empty() {
        pending.remove(&at);
    }
}

/// The order of the title's frame events and presents (with the trace):
/// kept from frame 400 on for 96 events, logged once.
static EVENTS: Mutex<Vec<(char, u64)>> = Mutex::new(Vec::new());

fn event(what: char) {
    if crate::frame::current() < 400 {
        return;
    }
    let mut events = lock(&EVENTS);
    if events.len() >= 96 {
        return;
    }
    events.push((what, crate::host::now_ms()));
    if events.len() == 96 {
        let first = events[0].1;
        let text: Vec<String> = events.iter().map(|(c, ms)| format!("{c}+{}", ms - first)).collect();
        tracing::info!(
            events = text.join(" "),
            "walls trace: the title's frame waits (O origin, F flip, v other) and presents (P) in order, ms from the first"
        );
    }
}

/// With the frame-rate line.
pub(crate) fn line(frames: f64) {
    if !trace() && !flip_landed() {
        return;
    }
    let v: Vec<f64> = STATS.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64).collect();
    let f = |i: usize| v[i] / frames;
    let found = v[s::FOUND].max(1.0);
    let (polls, below) = crate::fence::take_polls();
    tracing::info!(
        "the title's flip waits per frame (VICTORY_FLIP_LANDED={}): {:.2}, {:.2} of them found the work of the frame before not \
         landed ({:.1} parts of the lines not sent yet, {:.1} submissions in the air, on average there); \
         waited {:.1} ms for it, {:.2} gave up; results given to the title per frame (values, timestamps, read-backs): \
         {:.1}/{:.1}/{:.1} submitted, {:.2}/{:.2}/{:.2} submitted again before the one before had landed; landed in the frame \
         they were submitted in {:.1}, one frame later {:.1}, two {:.1}, three or more {:.1}; fence values asked for \
         {:.1}, below a value a queue was told to signal {:.1}; frame waits: origin {:.2}, kind 1 {:.2}, flip {:.2}, others {:.2};          reads of submissions in the lines compared at our preparing (VICTORY_ASYNC_RACE=1) {:.1}, the title had changed          {:.2} fills, {:.2} card fills, {:.2} title copies' sources, {:.2} small refreshes ({:.3} MB)",
        u8::from(flip_landed()),
        f(s::FLIPS),
        f(s::FOUND),
        v[s::UNSENT] / found,
        v[s::IN_AIR] / found,
        v[s::WAITED_US] / 1e3 / frames,
        f(s::GAVE_UP),
        f(s::SUBMITTED),
        f(s::SUBMITTED + 1),
        f(s::SUBMITTED + 2),
        f(s::AGAIN),
        f(s::AGAIN + 1),
        f(s::AGAIN + 2),
        f(s::LAG),
        f(s::LAG + 1),
        f(s::LAG + 2),
        f(s::LAG + 3),
        polls as f64 / frames,
        below as f64 / frames,
        f(s::WAITS),
        f(s::WAITS + 1),
        f(s::WAITS + 2),
        f(s::WAITS + 3),
        f(s::RACE_N),
        f(s::RACE_CHANGED),
        f(s::RACE_CHANGED + 1),
        f(s::RACE_CHANGED + 2),
        f(s::RACE_CHANGED + 3),
        v[s::RACE_BYTES] / 1e6 / frames,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn results_lag_and_resubmissions_are_counted() {
        let a = 0x1000usize;
        submitted(&[(a, VALUE), (a + 8, READ_BACK)]);
        submitted(&[(a, VALUE)]);
        assert_eq!(STATS[s::AGAIN].load(Ordering::Relaxed), 1);
        assert_eq!(STATS[s::AGAIN + 2].load(Ordering::Relaxed), 0);
        landed(a);
        landed(a);
        landed(a + 8);
        landed(a + 16);
        assert!(lock(&PENDING).as_ref().is_some_and(|p| p.is_empty()));
        let landed_n: u64 = (0..4).map(|i| STATS[s::LAG + i].load(Ordering::Relaxed)).sum();
        assert_eq!(landed_n, 3);
    }

    #[test]
    fn sent_parts_give_the_last_number_before_a_moment() {
        let early = Instant::now();
        sent(early);
        let mid = Instant::now() + Duration::from_millis(1);
        assert!(sent_before(mid) >= sent_before(early));
        assert_eq!(sent_before(early), 0, "only parts put in strictly before");
    }
}
