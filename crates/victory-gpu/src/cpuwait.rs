//! Where the thread that submits stands waiting with the relay, and what the
//! landing thread's fetches of card copies' pieces are made of: counters for
//! the frame's path line, and two switches (both off by default).
//!
//! Run n43 (card 61 ms a frame, frames 166-190 ms): the submission lock was
//! held 129 ms a frame, 39 of them standing; those 39 were relay waits for
//! the landing thread, "lists in the air" (`WHY_MOST`, until fewer than the
//! bound are in the air) and "title wrote under shader writes" (`WHY_STALE`,
//! until nothing is in the air), in shares that change from one second to
//! the next (33 + 6 ms, 7 + 26 ms). The landing thread was busy 78-100 ms a
//! frame, 55-67 of it in some 7 fetches of 64 MB that stood on the host's
//! one queue behind every flight sent since the ones being landed.
//!
//! What the counters split:
//!
//! - each relay wait of the two kinds: how many flights and lists were in
//!   the air when it began, how many of those the card had already run (so
//!   only the landing thread was behind), and how long the card took to run
//!   the flights the wait needed; the rest of the wait is the landing;
//! - why a "title wrote under shader writes" settle could not be spared by
//!   putting the title's bytes in beside the shaders' writes
//!   (`crate::shadow::merge_under_writes`);
//! - each fetch: pieces and bytes, of them in a copy's picked span (bytes
//!   that shaders picking views from the heap may write unannounced), bytes
//!   the merge found changed (the rest came down for nothing), the flights
//!   it stood behind and how long the card took to run those.
//!
//! The switches:
//!
//! - `VICTORY_STALE_EXACT=1` ([`stale_exact`]);
//! - `VICTORY_QUEUE2_PICKED=1|2` ([`queue2_picked`]).

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// A "title wrote under shader writes" settle with the relay's second stage
/// waits only until the flights that declared shader writes in what the
/// submission refreshes have landed, not until nothing is in the air
/// (`VICTORY_STALE_EXACT=1`; off by default).
///
/// The settle is there because a refresh leaves the ranges shaders may be
/// writing alone (`Shadow::update`): the title's bytes there would reach the
/// card a submission late. Once the flights that declared those ranges have
/// landed, the ranges are no longer pending in their copies and the refresh
/// takes the title's bytes as without flights; their copy-back is in title
/// memory, and the asking is made again after the wait, as after a settle.
/// Flights still in the air declared nothing there: by what the relay
/// assumes (shaders write only what flights declared) they do not write it,
/// the same assumption a title-read wait of the second stage makes.
pub(crate) fn stale_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_STALE_EXACT").is_some_and(|v| v == "1");
        if on {
            tracing::info!(
                "settles for title writes under shader writes wait only for the flights that declared them (VICTORY_STALE_EXACT=1)"
            );
        }
        on
    })
}

/// With `VICTORY_QUEUE2` bit 1 (fetches of card copies' pieces on a copy
/// queue): what becomes of pieces in a copy's picked span
/// (`VICTORY_QUEUE2_PICKED`; 0, the default: nothing, as upstream has it).
///
/// Why: a fetch on the copy queue stands behind the flights being landed
/// and nothing else, so later flights may be writing what it reads. For
/// writes a later flight declared, that flight fetches the piece again at
/// its own landing (`VICTORY_QUEUE2_REFETCH`). Shaders that pick unordered
/// access views from the heap by index write copies where no flight
/// declares it (runs n21 and n24 with our own copy-queue fetch: the strip
/// at the bottom and the stair-steps on the shoulder), and no landing
/// fetches those again: half-written bytes would stay in title memory.
/// Every such view's bytes lie in its copy's picked span
/// (`Shadow::mark_picked_uav`, from the heap pass).
///
/// - `1`: a piece that meets the picked span comes down on the host's
///   queue, behind every flight sent before it, where it holds whole
///   flights only, as without the copy queue. A landing with no such piece
///   waits for no later flight at all.
/// - `2`: such pieces come down on the copy queue with the others; once
///   that has run, if any flight that may have run beside it binds a
///   stage that can store through picked views (`crate::draw::PICKED_STORE_KEY`,
///   from the program analysis behind `Translated::picked_stores`: the BVH
///   skinning, cable, rope, tree and index unpacking programs), the picked
///   pieces come down again on the host's queue and only those bytes are
///   merged. A landing with no such flight beside it waits for no later
///   flight.
///
/// What it gains depends on the share of fetched bytes in picked spans and
/// how often picking flights fly, which the frame line counts.
pub(crate) fn queue2_picked() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let level =
            std::env::var("VICTORY_QUEUE2_PICKED").ok().and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(0).min(2);
        if level > 0 {
            tracing::info!(
                level,
                "pieces of card copies behind the heap's unordered access views are kept from torn copy-queue fetches \
                 (VICTORY_QUEUE2_PICKED)"
            );
        }
        level
    })
}

/// The relay waits split here: "lists in the air" and "title wrote under
/// shader writes".
pub(crate) const MOST: usize = 0;
pub(crate) const STALE: usize = 1;

/// Per kind of relay wait ([`MOST`], [`STALE`]), since the last frame line.
pub(crate) mod wait {
    /// Waits, flights in the air at their start, of them run by the card
    /// already, lists in the air, flights the wait needed the card to run,
    /// microseconds in all, microseconds until the card had run the needed
    /// flights (the rest was the landing thread's), waits that found the
    /// needed flights run already.
    pub const WAITS: usize = 0;
    pub const IN_AIR: usize = 1;
    pub const RUN: usize = 2;
    pub const LISTS: usize = 3;
    pub const NEEDED: usize = 4;
    pub const US: usize = 5;
    pub const CARD_US: usize = 6;
    pub const CARD_DONE: usize = 7;
    pub const ALL: usize = 8;
}

pub(crate) static WAITS: [[AtomicU64; wait::ALL]; 2] = [const { [const { AtomicU64::new(0) }; wait::ALL] }; 2];

/// Adds `n` at `what` of the wait kind `kind`.
pub(crate) fn wait_add(kind: usize, what: usize, n: u64) {
    WAITS[kind][what].fetch_add(n, Ordering::Relaxed);
}

/// The landing thread's fetches, since the last frame line.
pub(crate) mod fetch {
    /// Fetches, pieces, bytes; of them in a copy's picked span: pieces,
    /// bytes; bytes the merge found changed; fetches that changed nothing.
    pub const FETCHES: usize = 0;
    pub const PIECES: usize = 1;
    pub const BYTES: usize = 2;
    pub const PICKED_PIECES: usize = 3;
    pub const PICKED_BYTES: usize = 4;
    pub const MERGED: usize = 5;
    pub const IDLE: usize = 6;
    /// On the host's queue: flights in the line sent after the landed
    /// ones (the fetch stands behind them), microseconds until the card
    /// had run them, microseconds of the fetch in all.
    pub const AHEAD: usize = 7;
    pub const AHEAD_US: usize = 8;
    pub const HOST_US: usize = 9;
    pub const HOST_N: usize = 10;
    /// With [`super::queue2_picked`]: fetches on the copy queue, their
    /// bytes, microseconds; landings that fetched on both.
    pub const LANE_N: usize = 11;
    pub const LANE_BYTES: usize = 12;
    pub const LANE_US: usize = 13;
    pub const BOTH: usize = 14;
    /// With `VICTORY_QUEUE2_PICKED=2`: landings whose picked pieces came
    /// down again on the host's queue (a picking flight was beside).
    pub const AGAIN: usize = 15;
    pub const ALL: usize = 16;
}

pub(crate) static FETCH: [AtomicU64; fetch::ALL] = [const { AtomicU64::new(0) }; fetch::ALL];

pub(crate) fn fetch_add(what: usize, n: u64) {
    FETCH[what].fetch_add(n, Ordering::Relaxed);
}

/// Why `Shadow::merge_under_writes` asked for the settle: memory not
/// write-watched, the journal and the kernel could not say what was
/// written, a piece of a card copy's CPU side not held, too many changed
/// runs, no snapshot; and settles that the stale check asked for.
pub(crate) mod stale {
    pub const UNWATCHED: usize = 0;
    pub const NO_JOURNAL: usize = 1;
    pub const NOT_HELD: usize = 2;
    pub const RUNS: usize = 3;
    pub const NO_SNAPSHOT: usize = 4;
    pub const ALL: usize = 5;
}

pub(crate) static STALE_WHY: [AtomicU64; stale::ALL] = [const { AtomicU64::new(0) }; stale::ALL];

pub(crate) fn stale_why(what: usize) {
    STALE_WHY[what].fetch_add(1, Ordering::Relaxed);
}

/// With [`stale_exact`]: waits for the declaring flights after which the
/// check still asked for a settle (and everything was settled).
static STALE_FALLBACK: AtomicU64 = AtomicU64::new(0);

pub(crate) fn stale_fallback() {
    STALE_FALLBACK.fetch_add(1, Ordering::Relaxed);
}

/// For the frame's path line, per frame (`frames` of them since the last).
pub(crate) fn text(frames: f64) -> String {
    let mb = |b: f64| b / (1 << 20) as f64;
    let w: Vec<Vec<f64>> = WAITS
        .iter()
        .map(|kind| kind.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect())
        .collect();
    let f: Vec<f64> = FETCH.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let s: Vec<f64> = STALE_WHY.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let waits = [(MOST, "lists in the air"), (STALE, "title wrote under shader writes")]
        .iter()
        .map(|&(kind, name)| {
            let w = &w[kind];
            format!(
                "{name} {:.1}x {:.1} ms (at their start {:.1} flights / {:.0} lists in the air, {:.1} of the flights run by the \
                 card already; {:.1} flights needed, the card had run them after {:.1} ms, {:.1}x at once; the rest was the \
                 landing)",
                w[wait::WAITS],
                w[wait::US] / 1e3,
                w[wait::IN_AIR],
                w[wait::LISTS],
                w[wait::RUN],
                w[wait::NEEDED],
                w[wait::CARD_US] / 1e3,
                w[wait::CARD_DONE],
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "; the relay's waits split (VICTORY_STALE_EXACT={}, {:.1} settled all the same): {waits}; title writes under shader writes not merged beside \
         them: {:.1} unwatched, {:.1} the journal could not say, {:.1} card copies' CPU side not held, {:.1} over \
         VICTORY_MERGE_RUNS, {:.1} no snapshot; the landing thread's fetches: {:.1} with {:.0} pieces {:.1} MB, of them in \
         picked spans {:.0} pieces {:.1} MB, merged changed {:.1} MB ({:.1} fetches changed nothing); on the host's queue \
         {:.1}x {:.1} ms, behind {:.1} later flights the card took {:.1} ms to run; on the copy queue \
         (VICTORY_QUEUE2_PICKED={}) {:.1}x {:.1} MB {:.1} ms, {:.1} landings on both queues, {:.1} of them as a picking \
         flight was beside",
        u8::from(stale_exact()),
        STALE_FALLBACK.swap(0, Ordering::Relaxed) as f64 / frames,
        s[stale::UNWATCHED],
        s[stale::NO_JOURNAL],
        s[stale::NOT_HELD],
        s[stale::RUNS],
        s[stale::NO_SNAPSHOT],
        f[fetch::FETCHES],
        f[fetch::PIECES],
        mb(f[fetch::BYTES]),
        f[fetch::PICKED_PIECES],
        mb(f[fetch::PICKED_BYTES]),
        mb(f[fetch::MERGED]),
        f[fetch::IDLE],
        f[fetch::HOST_N],
        f[fetch::HOST_US] / 1e3,
        f[fetch::AHEAD],
        f[fetch::AHEAD_US] / 1e3,
        queue2_picked(),
        f[fetch::LANE_N],
        mb(f[fetch::LANE_BYTES]),
        f[fetch::LANE_US] / 1e3,
        f[fetch::BOTH],
        f[fetch::AGAIN],
    )
}
