//! What shaders touch, on the card's side, in one switch
//! (`VICTORY_CARD_SIDE=1`; off by default).
//!
//! A host copy in the PC's memory is read and written by shaders across the
//! bus. Copies of title buffers under the least size of `card_limits` (1 MiB
//! by default) are such copies, and so are the windows of title memory that
//! no room was found for on the card. Measured on an RTX 5090 (one scene of
//! about 4,850 draws a frame, the Series X profile, GPU timestamps by
//! command, in another tree of this layer, not on this one yet): the card's
//! work went from 53.8 to 26.6 ms a frame with every title buffer's copy and
//! the windows on the card; compute dispatches from 29.9 to 7.5 ms; two
//! programs that looked expensive went from 5.6 and 2.1 ms to 0.2 and 0.3 ms
//! (their loads reach 4 MiB windows: they were waiting for the PC's memory).
//!
//! With the switch, where no switch of their own says otherwise:
//! - every title buffer's copy goes to the card (a copy is 64 KiB at least,
//!   so a least of 64 KiB takes them all), up to a total by the card's
//!   budget ([`set_for`]);
//! - windows outside buffers go to the card up to a total by the budget, the
//!   way they go today: one buffer in a GPU upload heap, which the CPU writes
//!   across the bus, where the device has such heaps (`window_upload`), and
//!   two sides (the CPU's in the PC's memory, the card's that shaders read)
//!   where it has none or with `VICTORY_WINDOW_UPLOAD=0`. In the same tree
//!   and scene, the median frame of each run's last minute, runs alternated:
//!   85.5 ms without the placement; with it, 62.1, 71.8 and 74.0 ms with the
//!   windows through the bus, 66.7, 69.6, 83.7 and 74.5 ms with two sides
//!   (the later runs of both had other programs on the card), and the thread
//!   that prepares submissions about 7 ms less at work through the bus;
//! - a window goes to the card up to 97 hundredths of the budget, as a title
//!   buffer's copy does, and the texture queue keeps the room of both totals,
//!   less what they hold already (`card_room_kept`), so that copies made
//!   after the textures came still find it.
//!
//! Every switch of its own still wins: `VICTORY_CARD_BUFFERS_MIN_KB`,
//! `VICTORY_CARD_BUFFERS_MIN_MB`, `VICTORY_CARD_BUFFERS_MB`,
//! `VICTORY_CARD_WINDOWS_MB`, `VICTORY_WINDOW_UPLOAD`, and
//! `VICTORY_CARD_ROOM_FOR_BUFFERS=0` (no room kept, and the line for windows
//! stays at eight tenths).

use std::sync::OnceLock;

/// The switches the placement gives way to: with any of them set, the
/// start-up line names it.
const OWN_SWITCHES: [&str; 6] = [
    "VICTORY_CARD_BUFFERS_MIN_KB",
    "VICTORY_CARD_BUFFERS_MIN_MB",
    "VICTORY_CARD_BUFFERS_MB",
    "VICTORY_CARD_WINDOWS_MB",
    "VICTORY_WINDOW_UPLOAD",
    "VICTORY_CARD_ROOM_FOR_BUFFERS",
];

/// Whether the placement is asked for (`VICTORY_CARD_SIDE=1`).
fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_CARD_SIDE").is_ok_and(|v| v.trim() == "1"))
}

/// The sizes of the placement, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Set {
    /// The least size of a title buffer whose copy goes to the card.
    pub(crate) buffers_least: u64,
    /// The most bytes of such copies.
    pub(crate) buffers_most: u64,
    /// The most bytes of windows on the card.
    pub(crate) windows_most: u64,
}

/// The set for a card whose budget is `budget` bytes (`None`: not known,
/// and nothing is changed). By the size of the card it is on (the budget
/// the system gives is about nine tenths of it):
///
/// | card | budget | buffers | windows |
/// |---|---|---|---|
/// | 8 GB | up to 9 GiB | 768 MB | 512 MB |
/// | 12 GB | up to 13 GiB | 1 GB | 1 GB |
/// | 16 GB | up to 20 GiB | 1.5 GB | 1.5 GB |
/// | 24 GB | up to 26 GiB | 2 GB | 2 GB |
/// | 32 GB and more | more | 3 GB | 3 GB |
///
/// What was seen on the 32 GB card: title buffers of 64 KiB and more held
/// 1.3 GB in 3,714 copies, the windows 2 GB of a 2 GB limit (3.3 GB were
/// asked for in a captured frame); the textures picked by index took the
/// rest. On the smaller cards the copies made first take the room and the
/// later ones stay in the PC's memory, as today. Only the 32 GB row is
/// measured; the others are guesses to be measured.
pub(crate) fn set_for(budget: Option<u64>) -> Option<Set> {
    const MIB: u64 = 1 << 20;
    const GIB: u64 = 1 << 30;
    let budget = budget?;
    let (buffers, windows) = match budget {
        b if b <= 9 * GIB => (768, 512),
        b if b <= 13 * GIB => (1024, 1024),
        b if b <= 20 * GIB => (1536, 1536),
        b if b <= 26 * GIB => (2048, 2048),
        _ => (3072, 3072),
    };
    Some(Set {
        buffers_least: 64 << 10,
        buffers_most: buffers * MIB,
        windows_most: windows * MIB,
    })
}

/// The set in force: [`set_for`] the card's budget with the switch on, read
/// once and said once. `None` with the switch off (the host is then not
/// asked), without a host, or with no budget known.
pub(crate) fn set() -> Option<Set> {
    static SET: OnceLock<Option<Set>> = OnceLock::new();
    *SET.get_or_init(|| {
        if !on() {
            tracing::info!(
                "what shaders touch is placed as before, by its own switches (VICTORY_CARD_SIDE=1: all of it on the card's \
                 side, sizes by the card's budget)"
            );
            return None;
        }
        let host = crate::host::host()?;
        let budget = host.video_memory().map(|[budget, ..]| budget);
        let set = set_for(budget);
        let own: Vec<&str> = OWN_SWITCHES.into_iter().filter(|name| std::env::var_os(name).is_some()).collect();
        // (Asked of the device when it was made: `Host::gpu_upload`.)
        let bus = super::window_upload() && host.gpu_upload();
        match set {
            Some(set) => tracing::info!(
                "what shaders touch is on the card's side (VICTORY_CARD_SIDE=1): the card's budget {} MB; title buffers' copies \
                 from {} KiB up to {} MB, windows up to {} MB {}, both up to 97 % of the budget with their room kept by the \
                 texture queue; switches of their own set, which win: {}",
                budget.unwrap_or(0) >> 20,
                set.buffers_least >> 10,
                set.buffers_most >> 20,
                set.windows_most >> 20,
                if bus { "through the bus, in GPU upload heaps" } else { "with two sides" },
                if own.is_empty() { "none".to_owned() } else { own.join(", ") }
            ),
            None => tracing::info!(
                "VICTORY_CARD_SIDE=1, but the card's budget is not known: what shaders touch is placed as before"
            ),
        }
        set
    })
}

/// The size in bytes a switch of its own (`name`, in MiB) or the set in
/// force gives, if either does.
pub(crate) fn mb_or(name: &str, from_set: impl FnOnce(Set) -> u64) -> Option<u64> {
    match std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
    {
        Some(mb) => Some(mb << 20),
        None => set().map(from_set),
    }
}

/// `card_limits` from its switches (`VICTORY_CARD_BUFFERS_MIN_KB`, `_MIN_MB`
/// and `_MB`, as numbers) and the set in force: a size a switch says is that
/// size, one none says is the set's, and without a set 1 MiB and 1,024 MB as
/// before.
pub(crate) fn buffer_limits(
    min_kb: Option<u64>,
    min_mb: Option<u64>,
    most_mb: Option<u64>,
    set: Option<Set>,
) -> (u64, u64) {
    let least = match (min_kb, min_mb, set) {
        (Some(kib), ..) if kib > 0 => kib << 10,
        (_, None, Some(set)) => set.buffers_least,
        (_, min_mb, _) => (min_mb.unwrap_or(1) << 20).max(1 << 20),
    };
    let most = match (most_mb, set) {
        (Some(mb), _) => mb << 20,
        (None, Some(set)) => set.buffers_most,
        (None, None) => 1024 << 20,
    };
    (least, most)
}

/// The room the texture queue keeps (`card_room_kept`): the buffers' total
/// (`buffers_most`) less the `buffers_taken` bytes of copies made or turned
/// down. With the set (`windows`: the windows' total and the bytes they hold)
/// the windows' room is added, and each is kept up to what it came to with
/// every title buffer and window on the 32 GB card ([`set_for`]: 1.3 GB of
/// buffers; the windows filled a 2 GB limit), so that a total sized for a
/// big card does not keep room nothing comes for.
pub(crate) fn kept_room(buffers_most: u64, buffers_taken: u64, windows: Option<(u64, u64)>) -> u64 {
    const BUFFERS_SEEN: u64 = 3 << 29;
    const WINDOWS_SEEN: u64 = 2 << 30;
    match windows {
        None => buffers_most.saturating_sub(buffers_taken),
        Some((windows_most, windows_held)) => {
            buffers_most.min(BUFFERS_SEEN).saturating_sub(buffers_taken)
                + windows_most.min(WINDOWS_SEEN).saturating_sub(windows_held)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_set_grows_with_the_card() {
        const GIB: u64 = 1 << 30;
        const MIB: u64 = 1 << 20;
        assert_eq!(set_for(None), None);
        let sizes = |gib: u64| {
            let set = set_for(Some(gib * GIB)).unwrap();
            assert_eq!(set.buffers_least, 64 << 10, "every title buffer");
            (set.buffers_most / MIB, set.windows_most / MIB)
        };
        // Budgets as the system gives them for cards of 8, 12, 16, 24 and 32 GB.
        assert_eq!(sizes(7), (768, 512));
        assert_eq!(sizes(11), (1024, 1024));
        assert_eq!(sizes(15), (1536, 1536));
        assert_eq!(sizes(22), (2048, 2048));
        assert_eq!(sizes(30), (3072, 3072));
    }

    /// Without the set the buffers' limits are as before; with it, every
    /// size no switch says is the set's, and a switch always wins.
    #[test]
    fn a_switch_of_its_own_wins_over_the_set() {
        const MIB: u64 = 1 << 20;
        let set = set_for(Some(30 << 30));
        // As before: 1 MiB and 1,024 MB; the switches as they were read.
        assert_eq!(buffer_limits(None, None, None, None), (MIB, 1024 * MIB));
        assert_eq!(
            buffer_limits(Some(64), None, Some(1280), None),
            (64 << 10, 1280 * MIB)
        );
        assert_eq!(
            buffer_limits(Some(0), Some(16), Some(512), None),
            (16 * MIB, 512 * MIB)
        );
        assert_eq!(buffer_limits(None, Some(0), None, None), (MIB, 1024 * MIB));
        // The set: every title buffer, 3 GB on this card.
        assert_eq!(buffer_limits(None, None, None, set), (64 << 10, 3072 * MIB));
        // Each switch still wins.
        assert_eq!(
            buffer_limits(Some(256), None, None, set),
            (256 << 10, 3072 * MIB)
        );
        assert_eq!(
            buffer_limits(None, Some(4), None, set),
            (4 * MIB, 3072 * MIB)
        );
        assert_eq!(
            buffer_limits(None, None, Some(1024), set),
            (64 << 10, 1024 * MIB)
        );
        // A least of 0 KiB says nothing, as before.
        assert_eq!(
            buffer_limits(Some(0), None, None, set),
            (64 << 10, 3072 * MIB)
        );
    }

    /// The room kept: as before without the set; with it the windows' room
    /// is added, both capped, and nothing goes under zero.
    #[test]
    fn the_room_kept_counts_the_windows_with_the_set() {
        const MIB: u64 = 1 << 20;
        // As before: the buffers' total less what was made or turned down.
        assert_eq!(kept_room(1024 * MIB, 300 * MIB, None), 724 * MIB);
        assert_eq!(kept_room(1024 * MIB, 2048 * MIB, None), 0);
        // 3 GB and 3 GB on a 32 GB card: 1.5 GB and 2 GB kept at most.
        assert_eq!(kept_room(3072 * MIB, 0, Some((3072 * MIB, 0))), 3584 * MIB);
        assert_eq!(
            kept_room(3072 * MIB, 1000 * MIB, Some((3072 * MIB, 1500 * MIB))),
            536 * MIB + 548 * MIB
        );
        // 1 GB and 1 GB on a 12 GB card: kept whole until made.
        assert_eq!(kept_room(1024 * MIB, 0, Some((1024 * MIB, 0))), 2048 * MIB);
        assert_eq!(
            kept_room(1024 * MIB, 2048 * MIB, Some((1024 * MIB, 4096 * MIB))),
            0
        );
    }
}
