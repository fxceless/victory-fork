//! The barriers we record on the card, by why, for the frame-rate line;
//! and the per-list tracking behind `VICTORY_BARRIER_LAZY_UAV`.
//!
//! A barrier between two draws costs the card 4-6 us on an RTX 5070 Ti
//! whatever it names (`victory-shader`'s `mesh_bench_tests`, 2b2a36d: a
//! draw of 0.3 us alone): the card drains before it. Every
//! `ResourceBarrier` call recorded into a host list goes through
//! [`record`] (or `HostList::barrier`), counted by [`Why`]: calls (one
//! drain each at most) and the barriers in them by type.
//!
//! `VICTORY_BARRIER_LAZY_UAV=1`: a wait for every earlier shader write (an
//! unordered access barrier, on every resource or on one) that the title
//! or our amplified mesh draws ask for is not recorded when no draw or
//! dispatch that can write through unordered access (buffer or image
//! stores, atomics, or code not all known) and no title copy, clear or
//! resolve was recorded into the list since the list's last such wait
//! (its "writes"): nothing earlier is left for it to wait for. It is owed
//! instead, and recorded just before the next draw or dispatch that can
//! write, title copy, clear, resolve or the list's close, so what the
//! draws before it read is read before anything after writes (the
//! title's barrier orders reads before writes too). Several owed waits are
//! one. Transitions in the same title barrier are recorded where they are.
//! A list starts with writes (earlier lists of the same submission may
//! have written). Our own passes that write (the indirect clamp, runs)
//! wait for their own writes and end with such a wait.
//!
//! `VICTORY_BARRIER_STICKY_READS=1`: a title texture in a read-only state
//! (shader resource, depth read, copy source, ...) that a use wants in
//! another read-only state goes to both (their union) instead of from one
//! to the other, and a later use that wants a state it is in already
//! records nothing: depth read and depth read with shader read, pixel and
//! non-pixel reads, no longer move back and forth.

use std::sync::atomic::{AtomicU8, AtomicU32, AtomicU64, Ordering::Relaxed};
use windows::Win32::Graphics::Direct3D12::*;

/// Why a barrier was recorded.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Why {
    /// The title's transitions of textures (`ResourceBarrier`, kind 0).
    TitleTexture,
    /// The title's aliasing barriers (kind 1).
    TitleAlias,
    /// The title's unordered access barriers (kind 2), on a resource or all.
    TitleUav,
    /// The wait for every shader write a title barrier on a buffer becomes
    /// (buffers have host copies the barrier cannot name).
    TitleBuffer,
    /// The title's enhanced barrier groups: one wait for every shader write.
    TitleGroups,
    /// Our moves of title textures to the state a use needs (targets,
    /// shader reads, depth, unordered access, copies, clears).
    State,
    /// Depth textures' colour twins: their moves and copies.
    Twin,
    /// Copies of textures aliased in another format.
    Alias,
    /// An indirect call's records clamped alone: waits and moves around the pass.
    Clamp,
    /// The clamp's scratch buffer back to the common state after its call.
    ClampDone,
    /// A run of indirect calls clamped in one pass (`VICTORY_INDIRECT_BATCH`).
    Run,
    /// A wait before an indirect call that joined a run after a dispatch
    /// (`VICTORY_INDIRECT_BATCH=3`, `crate::indirect::RunRules::order`).
    RunOrder,
    /// The waits of indirect mesh draws launched by the amplification
    /// stage (`VICTORY_MESH_AMPLIFY`).
    AmplifyWait,
    /// Title buffer copies recorded on the card (`VICTORY_TABLE_GPU_COPY=2`).
    TitleCopy,
    /// An owed wait (`VICTORY_BARRIER_LAZY_UAV`) recorded before a write.
    Owed,
    /// The lists of their own before a list: state preludes, table
    /// constants copied on the card.
    Prelude,
    /// Lists made at submission: uploads, hashes and compares of copies.
    Submission,
    /// The present (and `VICTORY_BRIGHTEN`).
    Present,
}

const WHYS: usize = Why::Present as usize + 1;
const NAMES: [&str; WHYS] = [
    "title texture",
    "title alias",
    "title uav",
    "title buffer",
    "title groups",
    "state",
    "twin",
    "alias",
    "clamp",
    "clamp done",
    "run",
    "run order",
    "amplify wait",
    "title copy",
    "owed",
    "prelude",
    "submission",
    "present",
];

/// Since the last line, by [`Why`]: calls, then barriers in them.
static CALLS: [AtomicU64; WHYS] = [const { AtomicU64::new(0) }; WHYS];
static BARRIERS: [AtomicU64; WHYS] = [const { AtomicU64::new(0) }; WHYS];
/// Since the last line, by type: transitions, aliasing, unordered access
/// on all resources, on one.
static KINDS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The title's own calls since the last line ([`title`]).
pub(crate) mod title {
    /// `ResourceBarrier` calls, and barriers in them: transitions of
    /// textures, of buffers, split barriers' first halves, aliasing,
    /// unordered access, others; transitions of textures recorded nothing
    /// (no host texture, or the state it is in); barrier group calls;
    /// pipeline syncs.
    pub(crate) const CALLS: usize = 0;
    pub(crate) const TEXTURE: usize = 1;
    pub(crate) const BUFFER: usize = 2;
    pub(crate) const BEGIN_ONLY: usize = 3;
    pub(crate) const ALIAS: usize = 4;
    pub(crate) const UAV: usize = 5;
    pub(crate) const OTHER: usize = 6;
    pub(crate) const TEXTURE_NONE: usize = 7;
    pub(crate) const GROUPS: usize = 8;
    pub(crate) const SYNCS: usize = 9;
    pub(crate) const N: usize = 10;
}
pub(crate) static TITLE: [AtomicU64; title::N] = [const { AtomicU64::new(0) }; title::N];

/// Since the last line: with `VICTORY_BARRIER_LAZY_UAV`, waits not
/// recorded where asked (owed), owed waits recorded later, and moves left
/// out by `VICTORY_BARRIER_STICKY_READS` (the texture was readable as
/// wanted), and moves to a union of read states; GPU timestamps of
/// `VICTORY_GPU_PROFILE`.
pub(crate) static SKIPPED: AtomicU64 = AtomicU64::new(0);
pub(crate) static FLUSHED: AtomicU64 = AtomicU64::new(0);
pub(crate) static STICKY_KEPT: AtomicU64 = AtomicU64::new(0);
pub(crate) static STICKY_UNION: AtomicU64 = AtomicU64::new(0);
pub(crate) static TIMESTAMPS: AtomicU64 = AtomicU64::new(0);

/// Lists closed since the last line by barrier calls recorded into them
/// (none, 1-9, 10-99, 100-999, more), the calls and draws of all, and the
/// list with the most calls: its calls and draws.
static LISTS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
static LIST_CALLS: AtomicU64 = AtomicU64::new(0);
static LIST_DRAWS: AtomicU64 = AtomicU64::new(0);
static MOST: std::sync::Mutex<(u32, u32)> = std::sync::Mutex::new((0, 0));

fn count(why: Why, barriers: &[D3D12_RESOURCE_BARRIER]) {
    CALLS[why as usize].fetch_add(1, Relaxed);
    BARRIERS[why as usize].fetch_add(barriers.len() as u64, Relaxed);
    for b in barriers {
        let kind = match b.Type {
            D3D12_RESOURCE_BARRIER_TYPE_TRANSITION => 0,
            D3D12_RESOURCE_BARRIER_TYPE_ALIASING => 1,
            _ if is_global_uav(b) => 2,
            _ => 3,
        };
        KINDS[kind].fetch_add(1, Relaxed);
    }
}

/// Whether `b` is an unordered access barrier on every resource.
pub(crate) fn is_global_uav(b: &D3D12_RESOURCE_BARRIER) -> bool {
    // SAFETY: the union member the type names.
    b.Type == D3D12_RESOURCE_BARRIER_TYPE_UAV && unsafe { b.Anonymous.UAV.pResource.is_none() }
}

/// Records `barriers` into `list` (none: nothing), counted under `why`.
pub(crate) fn record(
    list: &ID3D12GraphicsCommandList,
    why: Why,
    barriers: &[D3D12_RESOURCE_BARRIER],
) {
    if barriers.is_empty() {
        return;
    }
    count(why, barriers);
    // SAFETY: an open list and live resources (the caller's).
    unsafe { list.ResourceBarrier(barriers) };
}

/// An unordered access barrier on every resource.
pub(crate) fn uav_all() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

/// `VICTORY_BARRIER_LAZY_UAV=1` (see the module).
pub(crate) fn lazy_uav() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BARRIER_LAZY_UAV").is_ok_and(|v| v.trim() == "1"))
}

/// `VICTORY_BARRIER_STICKY_READS=1` (see the module).
pub(crate) fn sticky_reads() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_BARRIER_STICKY_READS").is_ok_and(|v| v.trim() == "1"))
}

/// Host resource states that only read: vertex and constant, index, depth
/// read, non-pixel and pixel shader resource, indirect argument, copy
/// source, resolve source, shading rate source.
const READ_ONLY: u32 = 0x1 | 0x2 | 0x20 | 0x40 | 0x80 | 0x200 | 0x800 | 0x2000 | 0x0100_0000;

/// Whether host state `state` only reads (the common state does not
/// count: it is the one copies and promotions start from).
pub(crate) fn read_only(state: u32) -> bool {
    state != 0 && state & !READ_ONLY == 0
}

/// With [`sticky_reads`]: the state to move a texture in `before` to for a
/// use that wants `wanted` (both legal for it), or `None` for no move:
/// `wanted` if either state writes; else `before` if it holds `wanted`
/// (nothing to record), else both.
pub(crate) fn sticky(before: u32, wanted: u32) -> Option<u32> {
    if before == wanted {
        return None;
    }
    if !sticky_reads() || !read_only(before) || !read_only(wanted) {
        return Some(wanted);
    }
    if before & wanted == wanted {
        STICKY_KEPT.fetch_add(1, Relaxed);
        return None;
    }
    STICKY_UNION.fetch_add(1, Relaxed);
    Some(before | wanted)
}

/// A host list's barrier tracking: calls and draws recorded into it, and
/// with [`lazy_uav`] whether it has writes no wait has followed yet, owes
/// a wait, and binds a pipeline that can write.
#[derive(Default)]
pub(crate) struct Track {
    calls: AtomicU32,
    draws: AtomicU32,
    flags: AtomicU8,
}

const WRITES: u8 = 1;
const OWED: u8 = 2;
const BOUND_WRITES: u8 = 4;

impl Track {
    pub(crate) fn new() -> Track {
        Track {
            calls: AtomicU32::new(0),
            draws: AtomicU32::new(0),
            flags: AtomicU8::new(WRITES),
        }
    }

    /// The list was reset: as new.
    pub(crate) fn reset(&self) {
        self.calls.store(0, Relaxed);
        self.draws.store(0, Relaxed);
        self.flags.store(WRITES, Relaxed);
    }

    /// `barriers` were recorded into the list.
    pub(crate) fn recorded(&self, barriers: &[D3D12_RESOURCE_BARRIER]) {
        self.calls.fetch_add(1, Relaxed);
        if barriers.iter().any(is_global_uav) {
            self.fenced();
        }
    }

    /// A wait for every earlier shader write was recorded (by one of our
    /// passes, after its own writes): no writes, nothing owed.
    pub(crate) fn fenced(&self) {
        self.flags.fetch_and(!(WRITES | OWED), Relaxed);
    }

    /// Calls recorded without [`Track::recorded`] (our passes on the raw list).
    pub(crate) fn add_calls(&self, n: u32) {
        self.calls.fetch_add(n, Relaxed);
    }

    /// A wait for every earlier shader write is asked for: whether it must
    /// be recorded now (else it is owed).
    pub(crate) fn ask(&self) -> bool {
        if !lazy_uav() || self.flags.load(Relaxed) & WRITES != 0 {
            return true;
        }
        self.flags.fetch_or(OWED, Relaxed);
        SKIPPED.fetch_add(1, Relaxed);
        false
    }

    /// Something that may write is about to be recorded: whether an owed
    /// wait must be recorded first. The list has writes after it.
    pub(crate) fn write(&self) -> bool {
        let owed = self.flags.fetch_or(WRITES, Relaxed) & OWED != 0;
        if owed {
            self.flags.fetch_and(!OWED, Relaxed);
            FLUSHED.fetch_add(1, Relaxed);
        }
        owed
    }

    /// The list closes: whether an owed wait must be recorded.
    pub(crate) fn closing(&self) -> bool {
        let owed = self.flags.fetch_and(!OWED, Relaxed) & OWED != 0;
        if owed {
            FLUSHED.fetch_add(1, Relaxed);
        }
        owed
    }

    /// The pipeline bound for the next draws or dispatches can write
    /// through unordered access (or not).
    pub(crate) fn bound(&self, writes: bool) {
        if writes {
            self.flags.fetch_or(BOUND_WRITES, Relaxed);
        } else {
            self.flags.fetch_and(!BOUND_WRITES, Relaxed);
        }
    }

    /// Just before a draw or dispatch: whether it can write.
    pub(crate) fn draw(&self) -> bool {
        self.draws.fetch_add(1, Relaxed);
        self.flags.load(Relaxed) & BOUND_WRITES != 0
    }

    /// The list closed: its counts go to the line's.
    pub(crate) fn closed(&self) {
        let (calls, draws) = (self.calls.load(Relaxed), self.draws.load(Relaxed));
        let class = match calls {
            0 => 0,
            1..10 => 1,
            10..100 => 2,
            100..1000 => 3,
            _ => 4,
        };
        LISTS[class].fetch_add(1, Relaxed);
        LIST_CALLS.fetch_add(u64::from(calls), Relaxed);
        LIST_DRAWS.fetch_add(u64::from(draws), Relaxed);
        let mut most = MOST.lock().unwrap_or_else(|e| e.into_inner());
        if calls > most.0 {
            *most = (calls, draws);
        }
    }
}

/// The frame-rate line's barrier part, per frame, taken.
pub(crate) fn line(frames: f64) -> String {
    let frames = frames.max(1.0);
    let take = |a: &AtomicU64| a.swap(0, Relaxed) as f64 / frames;
    let calls: Vec<f64> = CALLS.iter().map(take).collect();
    let barriers: Vec<f64> = BARRIERS.iter().map(take).collect();
    let by: Vec<String> = (0..WHYS)
        .filter(|&i| calls[i] > 0.0)
        .map(|i| format!("{} {:.1} ({:.1})", NAMES[i], calls[i], barriers[i]))
        .collect();
    let kinds = KINDS.each_ref().map(take);
    let t = TITLE.each_ref().map(take);
    let lists = LISTS.each_ref().map(take);
    let (list_calls, list_draws) = (take(&LIST_CALLS), take(&LIST_DRAWS));
    let most = std::mem::take(&mut *MOST.lock().unwrap_or_else(|e| e.into_inner()));
    format!(
        "barriers per frame: {:.0} calls ({:.0} barriers: {:.0} transitions, {:.0} aliasing, {:.0} waits on all, {:.0} on one) by reason calls (barriers): {}; \
         the title's calls: {:.0} barrier calls with {:.0} texture transitions ({:.0} recorded nothing), {:.0} buffer, {:.0} split first halves, {:.0} aliasing, {:.0} uav, {:.0} other; {:.0} barrier groups, {:.0} syncs; \
         lists closed {:.0}: {:.0} with no barrier call, {:.0} 1-9, {:.0} 10-99, {:.0} 100-999, {:.0} more ({:.0} calls, {:.0} draws in them); most calls in one list {} ({} draws); \
         VICTORY_BARRIER_LAZY_UAV {}: {:.1} waits owed, {:.1} recorded later; VICTORY_BARRIER_STICKY_READS {}: {:.1} moves left out, {:.1} to unions; {:.0} profile timestamps",
        calls.iter().sum::<f64>(),
        barriers.iter().sum::<f64>(),
        kinds[0],
        kinds[1],
        kinds[2],
        kinds[3],
        if by.is_empty() {
            "none".to_owned()
        } else {
            by.join(", ")
        },
        t[title::CALLS],
        t[title::TEXTURE],
        t[title::TEXTURE_NONE],
        t[title::BUFFER],
        t[title::BEGIN_ONLY],
        t[title::ALIAS],
        t[title::UAV],
        t[title::OTHER],
        t[title::GROUPS],
        t[title::SYNCS],
        lists.iter().sum::<f64>(),
        lists[0],
        lists[1],
        lists[2],
        lists[3],
        lists[4],
        list_calls,
        list_draws,
        most.0,
        most.1,
        if lazy_uav() { "on" } else { "off" },
        take(&SKIPPED),
        take(&FLUSHED),
        if sticky_reads() { "on" } else { "off" },
        take(&STICKY_KEPT),
        take(&STICKY_UNION),
        take(&TIMESTAMPS),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_only_states() {
        assert!(read_only(0x80));
        assert!(read_only(0x20 | 0x80));
        assert!(read_only(0x800));
        assert!(!read_only(0));
        assert!(!read_only(0x4));
        assert!(!read_only(0x8 | 0x80));
        assert!(!read_only(0x10));
        assert!(!read_only(0x400));
    }

    #[test]
    fn a_list_owes_waits_only_without_writes() {
        // The switch is read once from the environment: the tracking is
        // tested through its flags here.
        let t = Track::new();
        assert_ne!(
            t.flags.load(Relaxed) & WRITES,
            0,
            "a list starts with writes"
        );
        t.recorded(&[uav_all()]);
        assert_eq!(t.flags.load(Relaxed) & (WRITES | OWED), 0);
        // Nothing written: a wait asked for is owed (with the switch on).
        if lazy_uav() {
            assert!(!t.ask());
            assert!(t.write(), "an owed wait goes before a write");
            assert!(t.ask(), "after a write a wait is recorded");
        } else {
            assert!(t.ask());
        }
        t.fenced();
        t.flags.fetch_or(OWED, Relaxed);
        assert!(t.closing());
        assert!(!t.closing());
        t.bound(true);
        assert!(t.draw());
        t.bound(false);
        assert!(!t.draw());
        t.reset();
        assert_eq!(t.flags.load(Relaxed), WRITES);
    }
}
