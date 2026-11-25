//! Lists in the air with the relay: what bounds them, and a switch that
//! gives back what bounds them once the card has run them, not once the
//! landing thread has landed them (`VICTORY_RELAY_FREE_RAN=1`; off by
//! default).
//!
//! Run n47: the submitting thread stood 22-28 ms a frame in the relay's
//! "lists in the air" wait (at 256 host lists in the air), and for none of
//! it was the card behind: every flight the wait needed was in the landing
//! thread's hands already, run by the card, its landing standing on the
//! host's queue behind the flights sent since (some 7 fetches a frame,
//! 50 ms of them waiting there).
//!
//! What a host list in the air holds (`crate::list::Detached`):
//!
//! - its draw state, with the dynamic descriptor slots its tables were
//!   made in (`crate::draw::State::dynamic`): what the bound is for (run 45
//!   ran the slots out with a frame's lists held at once). The card is done
//!   with them once it has run the flight; nothing of the landing reads them.
//! - the host list itself: its read-backs (done at the landing), its
//!   scratch chunks and the upload buffers of our steps.
//! - the upload memory handed to the title (`allocate_upload`), which is
//!   title-visible and may lie under a copy whose copy-back is still to come.
//!
//! With the switch, the draw state of every flight the card has run goes
//! back at once: by the landing thread when it takes flights to land (all
//! of them run), by the submitting thread when the line holds
//! `VICTORY_LATE_WAIT_LISTS` host lists that still hold theirs (it waits
//! for the card to run as many as take the count under the bound, as
//! `VICTORY_SETTLE_LISTS=1` does without the relay). The states are dropped
//! outside the line's lock, by whoever calls `crate::host`'s `land` next
//! (the submitting thread after each submission, the helper thread after
//! each landing): never by the landing thread, which takes no lock a
//! waiter may hold. The rest stays until the landing, so the line is
//! bounded in host lists all the same, at `VICTORY_RELAY_HELD_LISTS` (4
//! times the bound by default), and at that the submitting thread waits
//! for the landing as before. Nothing reaches the title earlier: results,
//! read-backs, bottom-of-pipe writes and fences still wait for the landing.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Whether flights' draw states go back when the card has run them
/// (`VICTORY_RELAY_FREE_RAN=1`). Only with the relay's second stage; the
/// caller checks that.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_RELAY_FREE_RAN").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "lists in the air give back their descriptor slots once the card has run them, not once they \
                 have landed (VICTORY_RELAY_FREE_RAN=1)"
            );
        }
        on
    })
}

/// With [`on`]: the most host lists in the line, those that gave their
/// state back included, before the submitting thread waits for the
/// landing (`VICTORY_RELAY_HELD_LISTS`; 4 times `most`, the bound on lists
/// that hold their state, by default). Each still holds its host list,
/// scratch and upload memory.
pub(crate) fn held_most(most: usize) -> usize {
    static MOST: OnceLock<Option<usize>> = OnceLock::new();
    MOST.get_or_init(|| std::env::var("VICTORY_RELAY_HELD_LISTS").ok().and_then(|v| v.trim().parse().ok()))
        .unwrap_or(most.saturating_mul(4))
        .max(most)
}

/// The counters, since the last frame line.
pub(crate) mod n {
    /// Flights and host lists that gave their state back as the landing
    /// thread took them, and as the submitting thread found them run.
    pub const LAND_FLIGHTS: usize = 0;
    pub const LAND_LISTS: usize = 1;
    pub const RAN_FLIGHTS: usize = 2;
    pub const RAN_LISTS: usize = 3;
    /// Times the line held the bound in lists that hold their state; of
    /// them, those the card had to run more flights for, and the
    /// microseconds it took.
    pub const AT_BOUND: usize = 4;
    pub const CARD_WAITS: usize = 5;
    pub const CARD_US: usize = 6;
    /// Times the line held `VICTORY_RELAY_HELD_LISTS` and the submitting
    /// thread waited for the landing, and the microseconds.
    pub const HELD_WAITS: usize = 7;
    pub const HELD_US: usize = 8;
    /// The most host lists in the line (given back or not) at a submission.
    pub const HELD_TOP: usize = 9;
    /// Settles because the compare's result area of card copies was half
    /// taken by plans in the air (`Host::card_results_low`), and their
    /// microseconds: more plans in the air reach it sooner.
    pub const AREA: usize = 10;
    pub const AREA_US: usize = 11;
    pub const ALL: usize = 12;
}

static N: [AtomicU64; n::ALL] = [const { AtomicU64::new(0) }; n::ALL];

pub(crate) fn add(what: usize, k: u64) {
    N[what].fetch_add(k, Ordering::Relaxed);
}

pub(crate) fn top(what: usize, k: u64) {
    N[what].fetch_max(k, Ordering::Relaxed);
}

/// For the frame's path line, per frame (`frames` of them since the last);
/// `most` is the bound on lists that hold their state.
pub(crate) fn text(frames: f64, most: usize) -> String {
    let v: Vec<u64> = N.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
    let f = |i: usize| v[i] as f64 / frames;
    format!(
        "; lists in the air given back once run (VICTORY_RELAY_FREE_RAN={}): as the landing took them {:.1} flights / \
         {:.0} lists, found run by the submitting thread {:.1} / {:.0}; at the bound ({most}) {:.1}x, {:.1}x of them \
         waited for the card {:.1} ms; held lists at most {} (VICTORY_RELAY_HELD_LISTS={}), waited for the landing \
         there {:.1}x {:.1} ms; settles for the compare's area {:.1}x {:.1} ms",
        u8::from(on()),
        f(n::LAND_FLIGHTS),
        f(n::LAND_LISTS),
        f(n::RAN_FLIGHTS),
        f(n::RAN_LISTS),
        f(n::AT_BOUND),
        f(n::CARD_WAITS),
        f(n::CARD_US) / 1e3,
        v[n::HELD_TOP],
        held_most(most),
        f(n::HELD_WAITS),
        f(n::HELD_US) / 1e3,
        f(n::AREA),
        f(n::AREA_US) / 1e3,
    )
}
