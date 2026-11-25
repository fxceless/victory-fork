//! Host copies of title buffers (decision 0007, step 3 and "Copies of title
//! memory"). Title buffers are title memory, which the host GPU cannot
//! read, so a buffer a draw uses gets a persistent host buffer of its size
//! on first use (system memory shaders may also write), and the ranges each
//! submission reads (vertex, index and constant data, buffer views) are
//! brought up to date from title memory before it runs: the 64 KiB pages
//! the title wrote since they were last copied ([`crate::watch`]: write
//! watching, or read-only guards on physical-page views), all of it where
//! neither works. A title GPU address inside the buffer is the same offset
//! in its copy. Ranges shaders write go back to title memory after the
//! submission ([`Shadow::gpu_writes`]).
//!
//! shortcut: only the ranges draws are known to read are brought up to
//! date (vertex and index buffer views, constant buffers in tables, root
//! descriptors, buffer views in tables); buffers read through directly
//! indexed heaps keep what their first use copied.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::watch::{PAGE, SUB};

/// A title buffer's host copy.
pub(crate) struct Shadow {
    pub(crate) buffer: ID3D12Resource,
    data: *mut u8,
    gpu: u64,
    /// Where the title buffer starts and how long it is.
    start: u64,
    len: u64,
    /// Per page of title memory from `start`'s on: the write-watch asking
    /// after which it was last copied (0: never).
    copied: Mutex<Vec<u64>>,
    /// Whether the memory is write-watched: unknown, yes, no.
    watched: AtomicU8,
    /// Ranges shaders of a submission not yet copied back may be writing:
    /// title memory there is older than the copy.
    pending: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: how many submissions not yet copied back declared
    /// each range of `pending` ([`Flights`]). Locked inside `pending`'s
    /// lock.
    flights: Mutex<Flights>,
    /// The copy's bytes before the submission ran, for the ranges its
    /// shaders write (`pending`), each at its offset in the copy: what the
    /// GPU wrote differs. Room for the whole copy; only those ranges' bytes
    /// are set (and read).
    before: Mutex<Option<Vec<u8>>>,
    /// With snapshots kept ([`keeps`]): which 64 KiB pieces of the snapshot
    /// hold the copy's bytes as they are now, and whether the snapshot is
    /// counted in [`KEPT_BYTES`].
    kept: Mutex<(Vec<bool>, bool)>,
    /// When the copy was last asked for (`crate::host::moment`): a window
    /// outside buffers nothing asked for in a while, and nothing holds, is
    /// let go ([`let_go_idle`]).
    used: AtomicU64,
    /// The lowest and highest title address asked for in the copy (a
    /// window is 4 MiB whatever was asked: measured for the memory line).
    asked: [AtomicU64; 2],
    /// Asks for the copy that gave no length (root views: they read from
    /// their address to the copy's end, whatever lies there).
    unsized_asks: AtomicU64,
    /// A number no other copy has had. Its address is no name for a copy:
    /// one let go and another made may share it.
    id: u64,
    /// With [`cards`], for a big title buffer: `buffer` is in the card's
    /// own memory, and this is the rest of the copy.
    card: Option<Card>,
    /// A window of title memory outside buffers, not a title buffer's copy.
    window: bool,
    /// With `crate::inplace`: not a copy at all. `buffer` is placed in a
    /// heap opened on the title memory itself, so shaders read and write
    /// the title's own bytes and nothing is copied either way. The heap,
    /// and the bytes from the buffer's first byte to `start` (the heap
    /// begins on a 64 KiB line; a title buffer need not).
    place: Option<(windows::Win32::Graphics::Direct3D12::ID3D12Heap, u64)>,
    /// With `place`: a run of the page pool's section, not private memory.
    pooled: bool,
    /// With [`gpu_upload_copies`]: `buffer` is in the card's memory, mapped
    /// for the CPU (a GPU upload heap), not a shared buffer in the PC's;
    /// its bytes.
    bar: Option<u64>,
    /// Offsets `[lowest, highest)` of the copy behind unordered access
    /// buffer views of the view heap (`crate::draw`'s heap pass): shaders
    /// that pick views by index may write there with no flight declaring
    /// it ([`picked_guard`]). Only grows; `u64::MAX, 0` while none.
    picked: [AtomicU64; 2],
    /// With [`picked_guard`] and no card side: the title's bytes as last
    /// copied into the picked span ([`Reference`]).
    reference: Mutex<Option<Reference>>,
}

/// For a copy with no card side whose picked span ([`Shadow::picked`])
/// shaders may write unannounced: the title's bytes as the refresh last
/// put them there, per 4 KiB part (from `lo` on) with whether the part
/// holds them yet. The copy itself cannot tell a title change from a
/// shader's write (it holds both); this can.
struct Reference {
    lo: usize,
    bytes: Vec<u8>,
    valid: Vec<bool>,
}

/// What a copy in the card's memory ([`cards`]) has beside the buffer
/// shaders see.
struct Card {
    /// The CPU's side, in the PC's memory: what `data` maps. Title memory
    /// is copied into it as into any copy, and what changed in it goes to
    /// the card's side before the next submission. `None` with
    /// [`no_pc_copy`]: then `mirror` is the CPU's side.
    staging: Option<ID3D12Resource>,
    /// With [`no_pc_copy`]: the CPU's side as address space, held only for
    /// the pieces of `written`; what `data` points at.
    mirror: Option<Mirror>,
    /// Per 64 KiB piece of the copy (cut as [`pieces_of`] cuts): whether a
    /// range shaders may write ever held it. With a mirror, exactly the
    /// pieces it holds.
    written: Vec<std::sync::atomic::AtomicBool>,
    /// The card side's bytes as they were when the submission's shaders
    /// started, in the card's memory too: made when shader writes are
    /// first declared for the copy, and compared with the card's side, on
    /// the card, after each submission that declares some.
    before: Mutex<Option<ID3D12Resource>>,
    /// Offsets of the copy whose bytes changed on the CPU's side and have
    /// not gone to the card's side yet.
    dirty: Mutex<Vec<(usize, usize)>>,
    /// With [`relay`]: the offsets that went from `dirty` to the card's
    /// side, each with the number of the send that took it
    /// ([`card_dirty`]), kept until a fetch sent after it has been merged.
    /// Locked inside `dirty`'s lock.
    sent: Mutex<Vec<(usize, usize, u64)>>,
    /// The buffers' bytes (the title buffer's, rounded up to 64 KiB).
    size: u64,
}

/// Whether the host copies of big title buffers are in the card's own
/// memory (on by default since run 75 showed it holds: the BVH skinning
/// list went from 50.5 to 4.5 ms on the card; `VICTORY_CARD_BUFFERS=0`
/// keeps every copy in the PC's memory, as before. Read when a copy is
/// made, so it takes a fresh start).
///
/// Every host copy was a buffer in the PC's memory that the card reads
/// and writes over the bus, so that our code can fill it and compare it.
/// The title's shaders read almost all their buffer data from a few big
/// title buffers (in the gameplay frame of run 69, 190-330 MB of distinct
/// ranges of the 174 MB heap per list against 0-13 MB elsewhere), and a
/// skinning dispatch of ONE group of 32 threads took 1.8 ms on the card
/// (run 70): the time of some 170 loads one after the other, not of the
/// arithmetic.
///
/// With the switch, a title buffer of [`card_limits`]' size or more has two
/// sides. Shaders see `Shadow::buffer`, in the card's memory. The CPU's
/// side (`Card::staging`, what `Shadow::data` maps) takes title memory as
/// before, and the ranges that changed in it are copied to the card's side
/// before each submission (`Host::card_before`). What shaders wrote is
/// found on the card: after the submission's lists a compute pass of ours
/// compares each declared range, by 64 KiB piece, with the card side's
/// bytes from before (`Card::before`), and only the pieces that differ are
/// fetched and merged into title memory, byte by byte as
/// [`Shadow::copy_back`] does (`Host::card_back`, [`Shadow::card_merge`]).
/// No snapshot is taken and nothing is compared on the CPU for such a copy.
///
/// What it assumes is what [`keeps`] assumes: shaders write a copy only in
/// ranges declared as written. A write outside them is seen by later
/// shaders, as before, and goes back to the title the next time a declared
/// range holds its piece (before: never).
pub(crate) fn cards() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_BUFFERS").is_none_or(|v| v != "0"))
}

/// The smallest title buffer whose copy goes to the card's memory, and the
/// most bytes of such copies (their "before" twins not counted: a copy
/// shaders write takes its size twice), both in bytes:
/// `VICTORY_CARD_BUFFERS_MIN_MB` (16) and `VICTORY_CARD_BUFFERS_MB` (512).
/// Run 69's gameplay has ten title buffers of 16 MiB or more, 494 MB, and
/// 1,650 MB of the card's budget free.
fn card_limits() -> (u64, u64) {
    static LIMITS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *LIMITS.get_or_init(|| {
        let mb = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default) << 20
        };
        // From 1 MiB up and 1,024 MB in all by default (it was 16 and 512):
        // the shader writes of the smaller buffers are then found by the
        // card's hashes too, and with that the first room's walls were
        // drawn in every run.
        (mb("VICTORY_CARD_BUFFERS_MIN_MB", 1).max(1 << 20), mb("VICTORY_CARD_BUFFERS_MB", 1024))
    })
}

/// Bytes alive of copies' card sides, and of their "before" twins.
static CARD_BYTES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the bytes alive of title buffers' card sides, of
/// windows' card sides (each has a side of the same size in the PC's
/// memory), and of the hashes kept on the card for their compares.
pub(crate) fn card_bytes() -> [u64; 3] {
    [
        CARD_BYTES[0].load(Ordering::Relaxed),
        WINDOW_CARD_BYTES.load(Ordering::Relaxed),
        CARD_BYTES[1].load(Ordering::Relaxed),
    ]
}

/// Whether copies in the card's memory keep no whole CPU side in the PC's
/// memory (`VICTORY_CARD_NO_PC_COPY=1`; off by default. Read when a copy is
/// made, so it takes a fresh start).
///
/// Each copy in the card's memory had a CPU side of its size in the PC's
/// memory (a shared buffer, committed whole): 941-945 MB of the system's
/// commit for the 97 copies made at world entry with
/// `VICTORY_CARD_BUFFERS_MIN_MB=1`. The CPU side is there for two things:
/// the bytes going up to the card are copied from it, and it holds what the
/// card's side held when a submission's shaders started, which the merge
/// of what they wrote needs byte by byte ([`Shadow::card_merge`]) and the
/// merge of title writes under shader writes too
/// ([`Shadow::merge_under_writes`]). Only the second needs bytes kept, and
/// only in the pieces some range shaders may write holds.
///
/// With the switch the CPU side is address space reserved for the copy
/// ([`Mirror`]), and a 64 KiB piece of it is committed and held only once
/// a range shaders may write holds the piece ([`Shadow::writing`]): its
/// bytes are read from title memory then and sent up whole, so the card's
/// side and the CPU's agree there from that send on, as they did before.
/// Elsewhere a refresh only notes what changed, and the bytes go up from
/// title memory itself when the submission goes to the card
/// (`Host::card_before`, through upload chunks shared with the lists'
/// scratch and given back when the submission has run). What is fetched
/// and merged after a submission lies in declared ranges, so in held
/// pieces: the merges are unchanged.
///
/// What it changes: bytes that go up from title memory are read when the
/// submission is sent, not at the refresh (newer bytes, which the next
/// refresh takes again as before); [`skip_same`] skips only in held pieces
/// (elsewhere there is nothing to compare with); and a shader write outside
/// declared ranges (the hole [`cards`] names) is undone when a declared
/// range first takes its piece, rather than put back to the title.
///
/// Not with the relay (`crate::host::relay`): the mirror was made and run
/// only with settles, and upstream 9376fd6 (c96b4f2) turned the relay on by
/// default, so `=1` from the run settings put mirrors under the relay's
/// merges for the first time (run n6: magenta patches, garbage blocks).
/// `=1` is then refused with a warning and the CPU's side is a shared
/// buffer, as without the switch; `VICTORY_CARD_NO_PC_COPY=2` takes the
/// mirror with the relay all the same.
pub(crate) fn no_pc_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| no_pc_copy_from(std::env::var("VICTORY_CARD_NO_PC_COPY").ok().as_deref(), crate::host::relay()))
}

/// [`no_pc_copy`] for the switch's value `asked` and whether the relay is
/// on.
fn no_pc_copy_from(asked: Option<&str>, relay: bool) -> bool {
    match asked.map(str::trim) {
        Some("2") => true,
        Some("1") if relay => {
            tracing::warn!(
                "VICTORY_CARD_NO_PC_COPY=1 is not taken with the relay (on by default since the merge of upstream \
                 9376fd6): copies in the card's memory keep a CPU side in the PC's memory. \
                 VICTORY_CARD_NO_PC_COPY=2 takes the mirror anyway, VICTORY_RELAY=0 turns the relay off"
            );
            false
        }
        Some("1") => true,
        _ => false,
    }
}

/// For the memory line, of copies in the card's memory: [0] the bytes of
/// their CPU sides that are whole shared buffers; [1] the bytes of pieces
/// ever held by a range shaders may write (with or without the switch);
/// [2] with [`no_pc_copy`], the copies with a mirror and [3] the bytes
/// their mirrors hold (committed); [4] mirror pieces that could not be
/// committed; [5] bytes noted to go up from title memory since the start.
static PC_SIDES: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// [`PC_SIDES`] as they are now.
pub(crate) fn pc_sides() -> [u64; 6] {
    PC_SIDES.each_ref().map(|a| a.load(Ordering::Relaxed))
}

/// With [`no_pc_copy`]: the CPU side of a copy in the card's memory as
/// address space reserved for all of it, committed piece by piece (see
/// there). Only pieces committed are ever read or written.
struct Mirror {
    base: *mut u8,
    size: usize,
    /// Bytes committed (the pieces held).
    committed: AtomicU64,
}

impl Mirror {
    /// Address space for `size` bytes, nothing committed.
    fn reserve(size: usize) -> Option<Mirror> {
        use windows_sys::Win32::System::Memory::{MEM_RESERVE, PAGE_NOACCESS, VirtualAlloc};
        // SAFETY: a fresh reservation of our own.
        let base = unsafe { VirtualAlloc(std::ptr::null(), size, MEM_RESERVE, PAGE_NOACCESS) }.cast::<u8>();
        if base.is_null() {
            return None;
        }
        PC_SIDES[2].fetch_add(1, Ordering::Relaxed);
        Some(Mirror {
            base,
            size,
            committed: AtomicU64::new(0),
        })
    }

    /// Commits offsets `[lo, hi)` (whole pages around them): whether they
    /// are readable and writable now. Committed pages start as zeros.
    fn commit(&self, lo: usize, hi: usize) -> bool {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, PAGE_READWRITE, VirtualAlloc};
        let a = lo & !(PAGE_BYTES - 1);
        let b = hi.next_multiple_of(PAGE_BYTES).min(self.size);
        if a >= b {
            return true;
        }
        // SAFETY: inside our own reservation.
        let at = unsafe { VirtualAlloc(self.base.add(a).cast(), b - a, MEM_COMMIT, PAGE_READWRITE) };
        if at.is_null() {
            return false;
        }
        self.committed.fetch_add((hi - lo) as u64, Ordering::Relaxed);
        PC_SIDES[3].fetch_add((hi - lo) as u64, Ordering::Relaxed);
        true
    }
}

/// The system's page, as the mirror commits it.
const PAGE_BYTES: usize = 4096;

impl Drop for Mirror {
    fn drop(&mut self) {
        use windows_sys::Win32::System::Memory::{MEM_RELEASE, VirtualFree};
        PC_SIDES[2].fetch_sub(1, Ordering::Relaxed);
        PC_SIDES[3].fetch_sub(*self.committed.get_mut(), Ordering::Relaxed);
        // SAFETY: our reservation; nothing reads it once its copy is gone.
        unsafe { VirtualFree(self.base.cast(), 0, MEM_RELEASE) };
    }
}

/// The CPU's side for a copy of `size` bytes whose card side is made: with
/// [`no_pc_copy`] a mirror (or, if no address space, a shared buffer as
/// without), else a shared buffer; as (shared buffer, mirror, mapping).
fn cpu_side(size: u64) -> Option<(Option<ID3D12Resource>, Option<Mirror>, *mut u8)> {
    if no_pc_copy()
        && let Some(mirror) = Mirror::reserve(size as usize)
    {
        let base = mirror.base;
        return Some((None, Some(mirror), base));
    }
    let (buffer, data) = host()?.shared_buffer(size)?;
    PC_SIDES[0].fetch_add(size, Ordering::Relaxed);
    Some((Some(buffer), None, data))
}

/// Offsets `[from, to)` of a copy of `len` bytes of title memory from
/// `start` on, cut where `held` (per piece, as [`pieces_of`] cuts) changes:
/// (from, to, held), in order, each run as long as it can be.
fn side_runs_of(
    start: usize,
    len: usize,
    held: &[std::sync::atomic::AtomicBool],
    from: usize,
    to: usize,
) -> Vec<(usize, usize, bool)> {
    let mut runs: Vec<(usize, usize, bool)> = Vec::new();
    if from >= to {
        return runs;
    }
    for piece in pieces_of(start, from, to) {
        let (lo, hi) = piece_at(start, len, piece);
        let (lo, hi) = (lo.max(from), hi.min(to));
        if lo >= hi {
            continue;
        }
        let is = held.get(piece).is_some_and(|w| w.load(Ordering::Acquire));
        match runs.last_mut() {
            Some(last) if last.2 == is && last.1 == lo => last.1 = hi,
            _ => runs.push((lo, hi, is)),
        }
    }
    runs
}

#[cfg(test)]
mod mirror_tests {
    use super::{Mirror, PIECE, PC_SIDES, no_pc_copy_from, piece_at, pieces_of, side_runs_of, written_notes};
    use std::sync::atomic::Ordering;

    /// `=1` takes the mirror only without the relay; `=2` with it too;
    /// anything else never.
    #[test]
    fn the_mirror_is_refused_under_the_relay_unless_forced() {
        assert!(no_pc_copy_from(Some("1"), false));
        assert!(!no_pc_copy_from(Some("1"), true));
        assert!(no_pc_copy_from(Some("2"), true));
        assert!(no_pc_copy_from(Some("2"), false));
        for relay in [false, true] {
            assert!(!no_pc_copy_from(None, relay));
            assert!(!no_pc_copy_from(Some("0"), relay));
        }
    }

    /// The runs cover the range once, in order, each piece's bytes with
    /// its own note, and neighbours with the same note are one run.
    #[test]
    fn side_runs_cut_where_the_notes_change() {
        for start in [0usize, 1, PIECE - 3, PIECE + 4096, 7 * PIECE + 123] {
            let len = 9 * PIECE + 77;
            let notes = written_notes(start as u64, len as u64);
            assert_eq!(notes.len(), pieces_of(start, 0, len).end);
            for (i, n) in notes.iter().enumerate() {
                n.store(i % 3 == 1 || i == 4, Ordering::Relaxed);
            }
            for (from, to) in [(0, len), (5, 6), (PIECE / 2, 5 * PIECE + 9), (len - 1, len), (3 * PIECE, 3 * PIECE)] {
                let runs = side_runs_of(start, len, &notes, from, to);
                let mut at = from;
                for (i, &(a, b, held)) in runs.iter().enumerate() {
                    assert_eq!(a, at, "{start} {from} {to}");
                    assert!(a < b);
                    for piece in pieces_of(start, a, b) {
                        assert_eq!(notes[piece].load(Ordering::Relaxed), held);
                    }
                    if i > 0 {
                        assert_ne!(runs[i - 1].2, held, "neighbours with one note are one run");
                    }
                    at = b;
                }
                assert_eq!(at, to.max(from), "{start} {from} {to}");
            }
        }
    }

    /// A mirror commits only what it is asked to, counts it, and gives it
    /// back when dropped.
    #[test]
    fn a_mirror_holds_only_the_pieces_committed() {
        let size = 8 * PIECE;
        let before = PC_SIDES[3].load(Ordering::Relaxed);
        let mirror = Mirror::reserve(size).expect("address space");
        let start = PIECE + 100;
        let (lo, hi) = piece_at(start, size - 100, 2);
        assert!(mirror.commit(lo, hi));
        assert_eq!(mirror.committed.load(Ordering::Relaxed), (hi - lo) as u64);
        // SAFETY: committed just now; fresh pages are zeros.
        unsafe {
            let bytes = std::slice::from_raw_parts_mut(mirror.base.add(lo), hi - lo);
            assert!(bytes.iter().all(|&b| b == 0));
            bytes.fill(0x5a);
        }
        // Asking about a page outside: not committed.
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
        // SAFETY: plain queries of our own reservation.
        let state = |at: usize| unsafe {
            let mut m: MEMORY_BASIC_INFORMATION = std::mem::zeroed();
            VirtualQuery(mirror.base.add(at).cast(), &mut m, size_of_val(&m));
            m.State
        };
        assert_eq!(state(lo), MEM_COMMIT);
        assert_ne!(state(hi.next_multiple_of(4096) + 4096), MEM_COMMIT);
        assert_ne!(state(0), MEM_COMMIT);
        drop(mirror);
        // Other tests may commit meanwhile; this one's bytes are given back.
        assert!(PC_SIDES[3].load(Ordering::Relaxed) <= before + (8 * PIECE) as u64);
    }
}

/// The per-piece notes of [`Card::written`] for a copy of `len` bytes of
/// title memory from `start` on.
fn written_notes(start: u64, len: u64) -> Vec<std::sync::atomic::AtomicBool> {
    let pieces = pieces_of(start as usize, 0, len.max(1) as usize).end;
    (0..pieces).map(|_| std::sync::atomic::AtomicBool::new(false)).collect()
}

/// Whether the texture queue leaves room on the card for the card copies
/// of title buffers that are not made yet
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=1`; off by default).
///
/// The copies are made in one burst when the world loads (run 189: 94 of
/// its 95 within two seconds, right before the first world frame), and
/// [`card_for`] makes one only while the card is under eight tenths of its
/// budget. That held as long as the textures picked by index filled the
/// card after the burst. Since the texture queue keeps packed neighbours
/// (52b8060) it fills the card while the world loads: 85% and more at the
/// burst in runs 190 to 192, so one copy of 95 was made, the CPU looked
/// for shader writes in 600 to 720 MB a frame where it had been 30, and
/// the frame went from about 240 to 370-470 ms.
///
/// With the switch, the texture queue's rule (`resource::room`) counts the
/// part of [`card_limits`]' second number that no copy has taken yet as
/// used, and a copy may be made up to nine tenths of the budget: the room
/// it takes was kept for it. After the burst what is kept is what the
/// burst left of the limit (87 MB of 1,024 in run 189).
///
/// On by default from the build after run 199
/// (`VICTORY_CARD_ROOM_FOR_BUFFERS=0`: off, as before). Run 193 had it on and still made 31 copies of 95: the copies
/// and the world's render targets (+1,603 MB) arrive in the same first
/// world frame and raced for the 448 MB left under the line, and a copy
/// turned down is not asked for again (`lookup` asks once), while its room
/// went on being kept (616 MB to the end of the run). So now: the texture
/// queue also keeps room for the targets (`resource::targets_room_kept`),
/// a copy is made up to 97 hundredths of the budget (its room was kept),
/// and the room of a copy turned down is given up ([`CARD_REFUSED`]).
fn card_room_kept_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ROOM_FOR_BUFFERS").is_none_or(|v| v != "0"))
}

/// Title buffers big enough for a copy in the card's memory that were
/// turned down for want of room, since the start, and their bytes: their
/// copies stay in the PC's memory for as long as the title keeps the
/// buffers, so no room is kept for them any more.
static CARD_REFUSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// The bytes of the card's memory the texture queue leaves free for card
/// copies of title buffers not made yet ([`card_room_kept_on`]): 0 with
/// the switch off or without card copies.
pub(crate) fn card_room_kept() -> u64 {
    if !card_room_kept_on() || !cards() {
        return 0;
    }
    card_limits().1.saturating_sub(CARD_BYTES[0].load(Ordering::Relaxed) + CARD_REFUSED[1].load(Ordering::Relaxed))
}

/// The copies in the card's memory.
static CARDED: Mutex<Vec<std::sync::Weak<Shadow>>> = Mutex::new(Vec::new());

/// Since the last frame line: bytes copied to card sides and the ranges
/// they were; pieces compared on the card, pieces fetched and their
/// bytes; the waits for the card the fetches added and their microseconds.
pub(crate) static CARD_STATS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// [`CARD_STATS`], taken; then the copies in the card's memory, their
/// bytes and their "before" twins' bytes.
pub(crate) fn card_stats() -> [u64; 10] {
    let mut out = [0u64; 10];
    for (i, stat) in CARD_STATS.iter().enumerate() {
        out[i] = stat.swap(0, Ordering::Relaxed);
    }
    out[7] = CARDED.lock().unwrap_or_else(|e| e.into_inner()).iter().filter(|w| w.strong_count() > 0).count() as u64;
    out[8] = CARD_BYTES[0].load(Ordering::Relaxed) + WINDOW_CARD_BYTES.load(Ordering::Relaxed);
    out[9] = CARD_BYTES[1].load(Ordering::Relaxed);
    out
}

/// The bytes of the hashes kept on the card for a copy of `size` bytes: 16
/// for each 64 KiB piece (a copy that does not start on a piece boundary
/// has one piece more).
pub(crate) fn card_hash_bytes(size: u64) -> u64 {
    (size / PIECE as u64 + 2) * 16
}

/// A buffer of `size` bytes in the card's memory for a title buffer's
/// copy, if the copy is to be there: [`cards`] is on, the size is within
/// [`card_limits`] and below 1 GiB (the compare counts in 32 bits), and the
/// host has what it takes (`Host::card_ready`).
fn card_for(size: u64) -> Option<ID3D12Resource> {
    if !cards() {
        return None;
    }
    let (least, most) = card_limits();
    if size < least || size >= 1 << 30 {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    // Not under the commit floor (`crate::memline`): the copy stays in the
    // PC's memory alone, as one below the size limit does. (Under
    // `VICTORY_CARD_FLOOR_MB` instead when it is set: the card side is not
    // the system's commit, and the CPU's side is there either way.)
    if crate::memline::card_pressed(crate::memline::past::CARD_COPIES) {
        crate::memline::HELD_BACK[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    // Only while the card has room: with this copy on it, a fifth of the
    // budget the system gives us must still be free. Copies are made when
    // the world loads, before the textures fill the card; the texture
    // queue's own rule (`resource::room`: nine tenths of the budget) counts
    // them, so the card's use stays where it was and the textures' share
    // shrinks by the copies' bytes, at most `card_limits`' second number.
    // (With `card_room_kept_on` the texture queue kept this copy's room
    // free, and the copy may go up to the queue's own line.)
    let hundredths = if card_room_kept_on() { 97 } else { 80 };
    if let Some([budget, used, ..]) = host.video_memory()
        && used + size > budget / 100 * hundredths
    {
        // Said, with the numbers it was decided by: run 193 made 31 copies
        // of 95 and no line named the others. The first 64, then every
        // 32nd.
        let n = CARD_REFUSED[0].fetch_add(1, Ordering::Relaxed) + 1;
        let bytes = CARD_REFUSED[1].fetch_add(size, Ordering::Relaxed) + size;
        if n <= 64 || n.is_multiple_of(32) {
            tracing::info!(
                "no card copy for a title buffer of {} KiB: the card has {} MB in use of a budget of {} MB and a copy is made up to \
                 {} MB; the copies made hold {} MB of at most {} MB; {} buffers turned down so far ({} MB), frame {}",
                size >> 10,
                used >> 20,
                budget >> 20,
                (budget / 100 * hundredths) >> 20,
                CARD_BYTES[0].load(Ordering::Relaxed) >> 20,
                most >> 20,
                n,
                bytes >> 20,
                crate::frame::current()
            );
        }
        return None;
    }
    if CARD_BYTES[0].fetch_add(size, Ordering::Relaxed) + size > most {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
    }
    buffer
}

/// Whether host copies with no card side are made in the card's memory,
/// mapped for the CPU (a GPU upload heap: resizable BAR), instead of a
/// shared buffer in the PC's memory: `VICTORY_GPU_UPLOAD_COPIES` `1`
/// windows outside buffers, `2` also title buffers and views' copies of
/// their own bytes; 0 (unset) none. At most `VICTORY_GPU_UPLOAD_COPIES_MB`
/// (2048 unless set) of them, and only while the card is under eight
/// tenths of its budget, as windows' card sides are. Read when a copy is
/// made, so it takes a fresh start; where the card has no such heap, or
/// with `VICTORY_SKIP_SAME=1` (which reads copies back), nothing changes.
///
/// Such a copy is one buffer, as the shared buffer was: the CPU writes
/// title memory into it at each refresh, through a write-combined mapping
/// (64 MiB in 3 ms, measured on this PC), and shaders read it in the
/// card's memory, not through the bus. No card side, no sends, no hashes:
/// nothing of `Card` is involved. And the same commit: a buffer of the
/// card's own memory is charged to the system's commit as one in the PC's
/// is (`crate::bus` `probe_heaps`: +256 MB for 256 MiB in a default heap,
/// a GPU upload heap and a shared buffer alike), so the commit floor has no
/// reason to hold these back, where it holds back card sides (they cost a
/// copy's size twice).
///
/// What it costs: the copies' bytes come out of the card's budget (the
/// texture queue's room); and reading such a copy on the CPU crosses the
/// bus uncached, some 40 MB/s (`probe_heaps`: 4 MiB in 94 ms), so a copy
/// whose ranges shaders write (its snapshot and compare read it) is slow:
/// the memory line counts the bytes so declared, and the first is said.
/// Windows: shaders wrote none in run 91 ("0.0 MB compared in 0 ranges of
/// windows"), hence `1` for them alone.
pub(crate) fn gpu_upload_copies() -> (u8, u64) {
    static ON: std::sync::OnceLock<(u8, u64)> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let level = std::env::var("VICTORY_GPU_UPLOAD_COPIES").ok().and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(0).min(2);
        let most = std::env::var("VICTORY_GPU_UPLOAD_COPIES_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(2048)
            << 20;
        if skip_same() { (0, 0) } else { (level, most) }
    })
}

/// [`gpu_upload_copies`]' copies alive and their bytes; and since the start,
/// bytes of ranges shaders were declared to write in them.
static BAR_COPIES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// For the memory line: [`BAR_COPIES`].
pub(crate) fn bar_copies() -> [u64; 3] {
    BAR_COPIES.each_ref().map(|b| b.load(Ordering::Relaxed))
}

/// The buffer of a host copy of `size` bytes that has no card side (a
/// `window` or not): in a GPU upload heap with [`gpu_upload_copies`], else
/// a shared buffer in the PC's memory; and with the first, its bytes.
fn pc_copy_buffer(size: u64, window: bool) -> Option<(ID3D12Resource, *mut u8, Option<u64>)> {
    let (level, most) = gpu_upload_copies();
    if level >= if window { 1 } else { 2 }
        && let Some(host) = host()
        && host.video_memory().is_none_or(|[budget, used, ..]| used + size <= budget / 10 * 8)
    {
        if BAR_COPIES[1].fetch_add(size, Ordering::Relaxed) + size <= most
            && let Some((buffer, data)) = host.card_shared_buffer(size)
        {
            BAR_COPIES[0].fetch_add(1, Ordering::Relaxed);
            return Some((buffer, data, Some(size)));
        }
        BAR_COPIES[1].fetch_sub(size, Ordering::Relaxed);
    }
    let (buffer, data) = host()?.shared_buffer(size)?;
    Some((buffer, data, None))
}

/// The most bytes of windows outside buffers whose copies go to the card's
/// memory (`VICTORY_CARD_WINDOWS_MB`; 0, the default, none).
///
/// Windows are two thirds of the PC memory the card shares (run 104: 861
/// windows, 3,552 MB of 5,400), and every draw that takes vertices, indices
/// or constants from the title's own allocator reads them over the bus.
/// Shaders write none of them (run 91: "0.0 MB compared in 0 ranges of
/// windows"), so a window on the card needs only what a card copy has
/// already: the CPU's side takes title memory as before, and what changed
/// in it goes up before each submission. No hashes, no compare, no fetch,
/// unless a shader does declare a write to it one day.
///
/// A window goes to the card while there is room under this number and
/// while the textures keep theirs ([`window_card_for`]). The CPU's side
/// stays in the PC's memory, so the memory shared with the card does not
/// shrink by this alone; what moves is what shaders read.
///
/// The textures' room was not kept before (runs n12, n13 with 4,096 MB):
/// windows went to the card while it was under four fifths of its budget,
/// 2.5 GB of them during the world's load, and the texture queue, whose
/// line counts every byte on the card (`resource::room`), was then full
/// for the rest of the run: 2.2 GB of textures the title fills on the card
/// against 5.2 GB without windows there (run n11), 28,000 textures picked
/// by index uploaded against 66,000. The first room's walls, its window
/// frame and the player's clothes lost theirs and were drawn dark. Window
/// bytes were right: nothing writes a window but the CPU's side.
fn window_card_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(0) << 20
    })
}

/// The card's memory a window leaves free for textures and the rest
/// (`VICTORY_CARD_WINDOWS_ROOM_MB`, 3,072): a window goes to the card only
/// while the card's use with it stays this far under the texture queue's
/// line (`resource::room_percent` of the budget). Measured: the world
/// makes 2.3-3.1 GB of render targets in its first frames, after the
/// windows of the load (`resource::targets_room_kept`), and textures fill
/// what is left.
fn window_card_room() -> u64 {
    static ROOM: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *ROOM.get_or_init(|| {
        std::env::var("VICTORY_CARD_WINDOWS_ROOM_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(3072)
            << 20
    })
}

/// Bytes alive of windows' card sides ([`window_card_most`]).
static WINDOW_CARD_BYTES: AtomicU64 = AtomicU64::new(0);

/// For the memory line, since the start, of windows the switch would have
/// put on the card: turned down because textures were waiting for room,
/// because the card would have come within [`window_card_room`] of the
/// texture line, or past [`window_card_most`] (each a count and bytes);
/// then the bytes sent up to windows' card sides, ranges of windows
/// declared written by shaders and their bytes, and windows made on the
/// card.
pub(crate) static WINDOW_CARD_STATS: [AtomicU64; 10] = [const { AtomicU64::new(0) }; 10];

/// [`WINDOW_CARD_STATS`]' places.
pub(crate) mod window_stats {
    pub(crate) const TEXTURES_WAIT: usize = 0;
    pub(crate) const NEAR_LINE: usize = 2;
    pub(crate) const PAST_MOST: usize = 4;
    pub(crate) const SENT: usize = 6;
    pub(crate) const DECLARED: usize = 7;
    pub(crate) const DECLARED_BYTES: usize = 8;
    pub(crate) const MADE: usize = 9;
}

fn window_refused(at: usize, size: u64) {
    WINDOW_CARD_STATS[at].fetch_add(1, Ordering::Relaxed);
    WINDOW_CARD_STATS[at + 1].fetch_add(size, Ordering::Relaxed);
}

/// A buffer in the card's memory for a window of `size` bytes, if it is to
/// be there ([`window_card_most`]): not while textures picked by index
/// wait for room (`resource::room_lately`), nor if the card's use with it
/// would come within [`window_card_room`] of the texture queue's line or
/// past four fifths of the budget.
fn window_card_for(size: u64) -> Option<ID3D12Resource> {
    use window_stats as w;
    let most = window_card_most();
    if most == 0 || !cards() {
        return None;
    }
    let host = host()?;
    if !host.card_ready() {
        return None;
    }
    if crate::memline::card_pressed(crate::memline::past::CARD_COPIES) {
        crate::memline::HELD_BACK[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    if !crate::resource::room_lately() {
        window_refused(w::TEXTURES_WAIT, size);
        return None;
    }
    if let Some([budget, used, ..]) = host.video_memory() {
        let line = (budget / 100 * crate::resource::room_percent()).saturating_sub(window_card_room());
        if used + size > line.min(budget / 10 * 8) {
            window_refused(w::NEAR_LINE, size);
            return None;
        }
    }
    if WINDOW_CARD_BYTES.fetch_add(size, Ordering::Relaxed) + size > most {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
        window_refused(w::PAST_MOST, size);
        return None;
    }
    let buffer = host.uav_buffer(size);
    if buffer.is_none() {
        WINDOW_CARD_BYTES.fetch_sub(size, Ordering::Relaxed);
    } else {
        WINDOW_CARD_STATS[w::MADE].fetch_add(1, Ordering::Relaxed);
    }
    buffer
}

/// For the memory line: windows alive on the card and their bytes, in the
/// PC's memory (shaders read them over the bus) and their bytes, and in
/// place and their bytes.
pub(crate) fn window_sides() -> [u64; 6] {
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = [0u64; 6];
    let mut seen = std::collections::HashSet::new();
    for s in loose.values() {
        // Each copy once: a window may stand under several keys.
        if !seen.insert(Arc::as_ptr(s) as usize) {
            continue;
        }
        let at = if s.card.is_some() {
            0
        } else if s.place.is_some() {
            4
        } else {
            2
        };
        out[at] += 1;
        out[at + 1] += s.len;
    }
    out
}

/// For the memory line, of the windows outside buffers that are copies
/// (not in place): [0] the bytes from each one's lowest asked address to
/// its highest (what a window cut to its asks would hold; a window is its
/// grid's cell whatever is read, `VICTORY_WINDOW_KB`); [1] the windows
/// asked without a length at least once (root views: they read to the
/// cell's end, `VICTORY_WINDOW_TAILS`) and [2] their bytes.
pub(crate) fn window_asked() -> [u64; 3] {
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = [0u64; 3];
    let mut seen = std::collections::HashSet::new();
    for s in loose.values() {
        if !seen.insert(Arc::as_ptr(s) as usize) || s.place.is_some() {
            continue;
        }
        let lo = s.asked[0].load(Ordering::Relaxed).max(s.start);
        let hi = s.asked[1].load(Ordering::Relaxed).min(s.start + s.len);
        out[0] += hi.saturating_sub(lo);
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[1] += 1;
            out[2] += s.len;
        }
    }
    out
}

/// The copies in the card's memory whose CPU side changed since they were
/// last asked: each with the changed offsets, in order and joined where
/// they touch. The ranges are taken: the caller copies them to the card's
/// side (`Host::card_before`).
///
/// With [`relay`], `sent_as` is the number of the send that takes them
/// (`crate::host`'s order of the queue): each range is noted with it, in
/// the same lock that took it, so that a merge of pieces fetched before
/// that send knows the card did not hold these bytes yet
/// ([`Shadow::card_merge_relay`]). Notes of sends that every fetch still to
/// be merged came after are dropped.
pub(crate) fn card_dirty(sent_as: u64) -> Vec<(Arc<Shadow>, Vec<(usize, usize)>)> {
    let mut out = Vec::new();
    let mut carded = CARDED.lock().unwrap_or_else(|e| e.into_inner());
    let fetched = RELAY_FETCHED.load(Ordering::Acquire);
    carded.retain(|weak| {
        let Some(shadow) = weak.upgrade() else {
            return false;
        };
        let ranges = shadow
            .card
            .as_ref()
            .map(|c| {
                let mut dirty = c.dirty.lock().unwrap_or_else(|e| e.into_inner());
                let ranges = std::mem::take(&mut *dirty);
                if relay() {
                    let mut sent = c.sent.lock().unwrap_or_else(|e| e.into_inner());
                    sent.retain(|s| s.2 > fetched);
                    sent.extend(ranges.iter().map(|&(a, b)| (a, b, sent_as)));
                }
                ranges
            })
            .unwrap_or_default();
        if !ranges.is_empty() {
            let ranges = joined(ranges);
            for &(a, b) in &ranges {
                count_skinned(skinned::SENT, shadow.start + a as u64, shadow.start + b as u64);
            }
            if let Some((lo, hi)) = shadow.picked_span() {
                let bytes: usize = ranges.iter().map(|&(a, b)| b.min(hi).saturating_sub(a.max(lo))).sum();
                PICKED_GUARD[picked_stat::SENT].fetch_add(bytes as u64, Ordering::Relaxed);
            }
            if shadow.window {
                let bytes: usize = ranges.iter().map(|&(a, b)| b - a).sum();
                WINDOW_CARD_STATS[window_stats::SENT].fetch_add(bytes as u64, Ordering::Relaxed);
            }
            out.push((shadow, ranges));
        }
        true
    });
    out
}

/// Ranges in order, those that overlap or touch as one.
fn joined(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for (from, to) in ranges {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Runs of pieces to hash again (`Host::card_before`), those that overlap
/// or touch as one, in order: every piece of `runs` is in the result once
/// and no other piece is. Ranges sent up lie close together (514 a frame
/// for 87 MB in run 102), and each had a pass of its own over its pieces,
/// so a piece that held several ranges was hashed once for each.
pub(crate) fn joined_piece_runs(mut runs: Vec<CardSpan>) -> Vec<CardSpan> {
    runs.sort_unstable_by_key(|r| r.first);
    let mut out: Vec<CardSpan> = Vec::with_capacity(runs.len());
    for run in runs {
        match out.last_mut() {
            Some(last) if run.first <= last.first + last.pieces => {
                last.pieces = last.pieces.max(run.first + run.pieces - last.first);
                last.to = last.to.max(run.to);
            }
            _ => out.push(run),
        }
    }
    out
}

/// A declared range of a copy in the card's memory, as the compare on the
/// card takes it: its offsets in the copy, its first 64 KiB piece (pieces
/// are cut at multiples of 64 KiB of title memory; piece 0 holds offset
/// 0), how many pieces hold it, and where the copy starts in piece 0.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CardSpan {
    pub(crate) from: u32,
    pub(crate) to: u32,
    pub(crate) first: u32,
    pub(crate) pieces: u32,
    pub(crate) skew: u32,
}

/// [`CardSpan`] for offsets `[from, to)` (`from < to`) of a copy of title
/// memory from `start` on.
fn card_span_of(start: usize, from: usize, to: usize) -> CardSpan {
    let pieces = pieces_of(start, from, to);
    CardSpan {
        from: from as u32,
        to: to as u32,
        first: pieces.start as u32,
        pieces: pieces.len() as u32,
        skew: (start % PIECE) as u32,
    }
}

/// The offsets of piece `i` of `span` that lie in the span, for a copy of
/// `len` bytes of title memory from `start` on: what is fetched and merged
/// of a piece the compare found differing.
fn card_piece_of(start: usize, len: usize, span: &CardSpan, i: u32) -> (usize, usize) {
    let (lo, hi) = piece_at(start, len, (span.first + i) as usize);
    (lo.max(span.from as usize), hi.min(span.to as usize))
}

#[cfg(test)]
mod card_tests {
    use super::{CardSpan, PIECE, card_piece_of, card_span_of, joined, joined_piece_runs};

    /// Joined runs hold every piece of the runs given, once, and no other.
    #[test]
    fn piece_runs_join_without_losing_or_adding_pieces() {
        let run = |first: u32, pieces: u32| CardSpan {
            from: first * 65536,
            to: (first + pieces) * 65536,
            first,
            pieces,
            skew: 7,
        };
        let runs = vec![run(40, 1), run(3, 2), run(4, 1), run(5, 3), run(40, 1), run(9, 1), run(41, 2), run(100, 1)];
        let mut wanted: Vec<u32> = runs.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        wanted.sort_unstable();
        wanted.dedup();
        let joined = joined_piece_runs(runs);
        let got: Vec<u32> = joined.iter().flat_map(|r| r.first..r.first + r.pieces).collect();
        assert_eq!(got, wanted);
        assert_eq!(joined.iter().map(|r| (r.first, r.pieces)).collect::<Vec<_>>(), [(3, 5), (9, 1), (40, 3), (100, 1)]);
        assert!(joined.iter().all(|r| r.skew == 7));
        assert!(joined_piece_runs(Vec::new()).is_empty());
    }

    /// The offsets the compare's shader takes for a piece (`CARD_COMPARE`
    /// in host.rs), before it widens them to 16 bytes.
    fn shader_piece(span: &CardSpan, i: u32) -> (u32, u32) {
        let piece = span.first + i;
        let lo = piece * 65536;
        let lo = if lo > span.skew { lo - span.skew } else { 0 };
        let hi = (piece + 1) * 65536 - span.skew;
        (lo.max(span.from), hi.min(span.to))
    }

    /// The pieces of a span cover it without gaps, and the shader's
    /// arithmetic names the same bytes as ours.
    #[test]
    fn the_shader_and_the_merge_cut_the_same_pieces() {
        for start in [0usize, 4096, PIECE - 16, PIECE, 3 * PIECE + 0x1f00, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                let mut at = from;
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    assert_eq!(lo, at, "{start:#x} {from} {to} piece {i}");
                    assert!(hi > lo && hi - lo <= PIECE);
                    assert_eq!(shader_piece(&span, i), (lo as u32, hi as u32), "{start:#x} {from} {to} piece {i}");
                    at = hi;
                }
                assert_eq!(at, to, "{start:#x} {from} {to}");
            }
        }
    }

    /// The fetch inside flights (`CARD_FETCH` in host.rs) copies each piece
    /// from `inflight_piece_from` on: what the merge takes of the piece lies
    /// in that slot.
    #[test]
    fn a_piece_fetched_in_flight_fits_its_slot() {
        let slot = crate::host::INFLIGHT_SLOT as usize;
        for start in [0usize, 4096, 7, PIECE - 16, PIECE - 3, 3 * PIECE + 0x1f09, 0xd_b790_df00] {
            let len = 5 * PIECE + 4096;
            for (from, to) in [(0, len), (0, 1), (16, PIECE), (PIECE - 4, PIECE + 4), (2 * PIECE, len), (12345, 4 * PIECE + 7)] {
                let span = card_span_of(start, from, to);
                for i in 0..span.pieces {
                    let (lo, hi) = card_piece_of(start, len, &span, i);
                    let at = crate::host::inflight_piece_from(&span, i);
                    assert_eq!(at % 16, 0);
                    assert!(lo >= at && hi - at <= slot, "{start:#x} {from} {to} piece {i}: {lo}..{hi} from {at}");
                }
            }
        }
    }

    /// The slots are a power of two, one per 64 KiB asked, within the
    /// shader's 32-bit offsets.
    #[test]
    fn inflight_slots_are_a_power_of_two() {
        use crate::host::{INFLIGHT_SLOT, inflight_slots};
        assert_eq!(inflight_slots(64 << 20), 1024);
        assert_eq!(inflight_slots(100 << 20), 1024);
        assert_eq!(inflight_slots(1), 1);
        assert_eq!(inflight_slots(1 << 20), 16);
        let most = inflight_slots(u64::MAX);
        assert!(most.is_power_of_two() && u64::from(most) * INFLIGHT_SLOT <= 1 << 32);
    }

    /// Tickets drawn and landed in order, the mark raised as the landing
    /// raises it (every ticket a landed flight drew), through the counter's
    /// wrap with the slots the default gives: no two tickets in the air
    /// that got a slot share one, slots are still given after the wrap, and
    /// the mark follows the counter.
    #[test]
    fn inflight_tickets_never_share_a_slot_across_the_wrap() {
        use crate::host::{inflight_gets_slot, inflight_raise, inflight_slots};
        let slots = inflight_slots(64 << 20);
        let mut landed = u32::MAX - 3000;
        let mut next = landed;
        // Flights of 1 to 700 flagged pieces, up to three in the air:
        // (tickets drawn, those that got a slot).
        let mut air: std::collections::VecDeque<(Vec<u32>, Vec<u32>)> = Default::default();
        let mut after_wrap = 0;
        for flight in 0..40u32 {
            let drawn: Vec<u32> = (0..1 + flight * 37 % 700).map(|i| next.wrapping_add(i)).collect();
            next = next.wrapping_add(drawn.len() as u32);
            let slotted: Vec<u32> = drawn.iter().copied().filter(|&k| inflight_gets_slot(k, landed, slots)).collect();
            after_wrap += slotted.iter().filter(|&&k| k < 1 << 20).count();
            air.push_back((drawn, slotted));
            let mut used: Vec<u32> = air.iter().flat_map(|a| a.1.iter()).map(|k| k % slots).collect();
            let n = used.len();
            used.sort_unstable();
            used.dedup();
            assert_eq!(used.len(), n, "flight {flight}: two tickets in the air share a slot");
            if air.len() == 3
                && let Some((drawn, _)) = air.pop_front()
            {
                for k in drawn {
                    landed = inflight_raise(landed, k);
                }
            }
        }
        while let Some((drawn, _)) = air.pop_front() {
            for k in drawn {
                landed = inflight_raise(landed, k);
            }
        }
        assert!(next < 1 << 20, "the counter wrapped");
        assert!(after_wrap > 0, "slots after the wrap");
        assert_eq!(landed, next);
        // Raised in wrapping order, never back.
        assert_eq!(inflight_raise(u32::MAX - 1, u32::MAX), 0);
        assert_eq!(inflight_raise(5, 2), 5);
        assert_eq!(inflight_raise(u32::MAX - 1, 3), 4);
        assert!(!inflight_gets_slot(3, 4, 8));
    }

    /// Flights 5, 6, 7 in the air with the fetch inside flights: 5 lands
    /// the old way with the queue's count at 7. The title's byte that went
    /// up in send 7 must still be told apart when 6 lands from its slot
    /// (its merge leaves out sends after 6), so send 7's note must outlive
    /// the mark (`card_dirty` keeps notes above it, `card_merge_relay`
    /// leaves out those above its "fetched after").
    #[test]
    fn a_fetch_the_old_way_keeps_the_notes_flights_in_the_air_need() {
        use crate::host::fallback_fetched_mark;
        let notes = [5u64, 6, 7];
        let mark = fallback_fetched_mark(true, 7, Some(5));
        let kept: Vec<u64> = notes.iter().copied().filter(|&s| s > mark).collect();
        assert!(kept.iter().any(|&s| s > 6), "send 7's note was dropped before flight 6 merged");
        assert_eq!(mark, 5);
        // Without the switch every later merge fetches after 7: as before.
        assert_eq!(fallback_fetched_mark(false, 7, Some(5)), 7);
        assert_eq!(fallback_fetched_mark(true, 7, None), 0);
    }

    /// Changed ranges are joined where they overlap or touch, and only there.
    #[test]
    fn ranges_join_where_they_touch() {
        assert_eq!(joined(vec![(8, 12), (0, 4), (4, 6), (10, 20), (30, 31)]), [(0, 6), (8, 20), (30, 31)]);
        assert!(joined(Vec::new()).is_empty());
    }
}

/// Whether the copy-back of what shaders wrote runs on a thread of its own,
/// beside the submitting thread and outside the submission lock
/// (`VICTORY_RELAY=1` or `2`; off by default; not with kept snapshots,
/// [`keeps`]). `crate::host::relay` says what it does; here it changes
/// what a copy keeps:
///
/// - the ranges shaders may be writing are counted per submission
///   ([`Flights`]): a range two flights declared stays closed to refreshes
///   until both have been copied back;
/// - the merge of what the card wrote never stores into memory the card
///   may be writing ([`merge_piece`]): what it learns goes into title
///   memory and into the copy's reference (the snapshot, or the CPU's side
///   of a copy in the card's memory), under that reference's lock;
/// - what goes up to a copy in the card's memory is numbered
///   ([`card_dirty`]), so that a merge can tell the title's bytes the card
///   had not received from bytes the card wrote.
pub(crate) fn relay() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default (stage 2, `crate::host::relay_exact`); `VICTORY_RELAY=0`
    // turns it off, `=1` asks for the first stage.
    // The crate's own tests run without it unless they ask: several of
    // them look at title memory right after a submission, which the
    // relay's landing thread may still be writing.
    *ON.get_or_init(|| std::env::var("VICTORY_RELAY").map_or(!cfg!(test), |v| v == "1" || v == "2") && !keeps())
}

/// The submission being prepared (its number among those started): what
/// [`Shadow::writing`] counts a declared range for ([`Flights`]).
static PREPARED: AtomicU64 = AtomicU64::new(0);

/// A submission's preparing begins (`crate::host`, under the submission
/// lock).
pub(crate) fn preparing(number: u64) {
    PREPARED.store(number, Ordering::Relaxed);
}

/// With [`relay`]: the number of the last send of changed ranges to the
/// card (`crate::host`'s order of the queue) that every fetch still to be
/// merged came after. Notes of sends up to it are of no use any more.
static RELAY_FETCHED: AtomicU64 = AtomicU64::new(0);

/// A fetch sent after send `number` is the oldest one that may still be
/// merged ([`RELAY_FETCHED`]).
pub(crate) fn relay_fetched(number: u64) {
    RELAY_FETCHED.fetch_max(number, Ordering::AcqRel);
}

/// With [`relay`]: how many submissions that are not copied back yet
/// declared each range shaders may be writing (range, count, the last
/// submission that declared it).
///
/// Without it a range declared by two flights was one entry, and the end
/// of the first flight's copy-back removed it while the second was still
/// writing: the next refresh would have put title bytes over what the
/// second flight's shaders were writing.
#[derive(Default)]
struct Flights(Vec<((usize, usize), u32, u64)>);

impl Flights {
    /// Submission `number` declares `range`: counted once for it, however
    /// often it declares it.
    fn declare(&mut self, range: (usize, usize), number: u64) {
        match self.0.iter_mut().find(|f| f.0 == range) {
            Some(f) => {
                if f.2 != number {
                    f.1 += 1;
                    f.2 = number;
                }
            }
            None => self.0.push((range, 1, number)),
        }
    }

    /// One submission that declared `ranges` (each once) is copied back:
    /// the ranges no other submission still holds.
    fn done(&mut self, ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let mut free = Vec::new();
        for range in ranges {
            match self.0.iter().position(|f| f.0 == *range) {
                Some(at) => {
                    self.0[at].1 = self.0[at].1.saturating_sub(1);
                    if self.0[at].1 == 0 {
                        self.0.swap_remove(at);
                        free.push(*range);
                    }
                }
                // Not counted (declared before anything counted): free.
                None => free.push(*range),
            }
        }
        free
    }
}

/// The relay's merge of a stretch of a copy ([`relay`]). `now` is what the
/// card's memory holds, read once; `held` is the copy's reference for the
/// stretch (its snapshot, or the CPU's side of a copy in the card's
/// memory): what the card's memory held when it was last looked at;
/// `title` is title memory as just read. Every byte the card changed
/// (`now` differs from `held`) goes into `title` and into `held`; every
/// other byte of both stays as it is, so a byte the title wrote and the
/// card did not keeps the title's value. Whether `title` changed.
///
/// Nothing is stored into `now`'s source: the card may be writing it for a
/// later flight at this moment, and a store there, even of the value just
/// read, could undo that write. (The merge of a waited-for submission
/// stores every byte of a differing stretch back into the copy.)
fn merge_piece(now: &[u8], held: &mut [u8], title: &mut [u8]) -> bool {
    let mut differs = 0u8;
    for ((n, h), t) in now.iter().zip(held.iter_mut()).zip(title.iter_mut()) {
        let wrote = if *n != *h { 0xffu8 } else { 0 };
        let merged = (*t & !wrote) | (*n & wrote);
        differs |= merged ^ *t;
        *t = merged;
        *h = (*h & !wrote) | (*n & wrote);
    }
    differs != 0
}

/// `[from, to)` without the ranges of `skip`, in order.
pub(crate) fn outside(from: usize, to: usize, mut skip: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    skip.sort_unstable();
    let mut out = Vec::new();
    let mut at = from;
    for (a, b) in skip {
        if b <= at {
            continue;
        }
        if a >= to {
            break;
        }
        if a > at {
            out.push((at, a));
        }
        at = at.max(b);
    }
    if at < to {
        out.push((at, to));
    }
    out
}

/// The runs of 4-byte words of title memory (cut on its addresses;
/// `read` and `merged` are of the memory at `address`) that hold a byte in
/// which `merged` differs from `read`, as (offset, bytes) in order to
/// `put`. A shader stores words, so these are what the card wrote; every
/// other byte of `merged` is the title's as it was read.
fn word_runs(address: usize, read: &[u8], merged: &[u8], mut put: impl FnMut(usize, usize)) {
    let n = read.len().min(merged.len());
    let mut i = 0;
    while i < n {
        // Equal bytes are passed over, 32 at a time while they last.
        match first_difference(&read[i..n], &merged[i..n]) {
            Some(d) => i += d,
            None => break,
        }
        // From the word that holds this byte (or the stretch's first
        // byte) to the end of the last word in a row that holds one.
        let first = i - ((address + i) & 3).min(i);
        let mut end = first;
        while end < n {
            let word = (((address + end) | 3) + 1 - address).min(n);
            if read[end..word] == merged[end..word] {
                break;
            }
            end = word;
        }
        put(first, end - first);
        i = end;
    }
}

/// Whether the merges that write a stretch of title memory back
/// ([`Shadow::relay_merge`], [`Shadow::card_merge`],
/// [`Shadow::card_merge_relay`]) look, right before that write, for title
/// bytes it is about to put back to older values (`VICTORY_MERGE_RACE=1`;
/// off by default). Logging only: [`merge_race`].
fn merge_race_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_RACE").is_some_and(|v| v == "1"))
}

/// [`merge_race`]'s sums. Of frame `frame`, and in `total` since the
/// start: stretches written back, their bytes, title bytes put back to
/// older values, bytes the title and the card both changed, stretches with
/// either, the runs a write of the card's words alone makes
/// ([`word_runs`]) and their bytes. `said`: the frame's first stretches
/// with either. `quiet`: frames since the last line.
struct MergeRace {
    frame: u64,
    now: [u64; 7],
    said: Vec<String>,
    total: [u64; 7],
    quiet: u64,
}

static MERGE_RACE: Mutex<MergeRace> =
    Mutex::new(MergeRace { frame: 0, now: [0; 7], said: Vec::new(), total: [0; 7], quiet: 0 });

/// With [`merge_race_on`], right before a merge writes `merged` over title
/// memory at `address` (a stretch of the copy of the title buffer `copy`:
/// start and bytes): title memory is read once more and compared with
/// `read`, what the merge read of it before. A byte that changed in
/// between and that the write-back stores with its value as first read
/// (the card did not change it) is a title byte put back to an older
/// value; one the write-back stores with the card's byte is one both
/// changed. Only bytes the write-back stores are looked at. Summed per
/// frame; a line for every frame that has any, and one every 100 frames
/// whatever it holds, so that "none" is said, not assumed.
///
/// A lower bound: a title write that lands after this read (while the
/// write-back itself runs) is not seen, nor one that wrote a byte and then
/// its old value again. Costs one more read and two compares of every
/// stretch written back, on the thread that merges.
fn merge_race(read: Option<&[u8]>, copy: (u64, u64), address: usize, merged: &[u8], which: &'static str) {
    if !merge_race_on() {
        return;
    }
    let Some(read) = read else {
        return;
    };
    let n = read.len().min(merged.len());
    let (read, merged) = (&read[..n], &merged[..n]);
    let mut runs: Vec<(usize, usize)> = Vec::new();
    word_runs(address, read, merged, |off, len| runs.push((off, len)));
    let whole = [(0, n)];
    let looked: &[(usize, usize)] = &whole;
    let mut again = vec![0u8; n];
    let (mut put_back, mut both, mut first) = (0u64, 0u64, None);
    if crate::host::read_into(address as *const u8, &mut again) {
        for &(off, len) in looked {
            for i in off..off + len {
                if again[i] == read[i] || again[i] == merged[i] {
                    continue;
                }
                if merged[i] == read[i] {
                    put_back += 1;
                    first.get_or_insert(i);
                } else {
                    both += 1;
                }
            }
        }
    }
    let frame = crate::frame::current();
    let mut race = MERGE_RACE.lock().unwrap_or_else(|e| e.into_inner());
    if race.frame != frame {
        race.quiet += 1;
        if race.now[4] > 0 || race.quiet >= 100 {
            tracing::info!(
                frame = race.frame,
                stretches = race.now[0],
                stretch_bytes = race.now[1],
                put_back = race.now[2],
                both_changed = race.now[3],
                stretches_hit = race.now[4],
                word_runs = race.now[5],
                word_run_bytes = race.now[6],
                stretches_since_start = race.total[0],
                stretch_bytes_since_start = race.total[1],
                put_back_since_start = race.total[2],
                both_changed_since_start = race.total[3],
                stretches_hit_since_start = race.total[4],
                word_runs_since_start = race.total[5],
                word_run_bytes_since_start = race.total[6],
                first = ?race.said,
                "merge race: title bytes that changed between a merge's read of title memory and its write-back of what shaders wrote, in this frame (VICTORY_MERGE_RACE=1)"
            );
            race.quiet = 0;
        }
        race.frame = frame;
        race.now = [0; 7];
        race.said.clear();
    }
    let hit = u64::from(put_back + both > 0);
    let run_bytes: u64 = runs.iter().map(|r| r.1 as u64).sum();
    let add = [1, n as u64, put_back, both, hit, runs.len() as u64, run_bytes];
    for (i, v) in add.iter().enumerate() {
        race.now[i] += v;
        race.total[i] += v;
    }
    if hit > 0 && race.said.len() < 8 {
        race.said.push(format!(
            "{which}: copy {:#x}+{:#x} stretch {address:#x}+{n:#x}: {put_back} put back (first at {:#x}), {both} both changed",
            copy.0,
            copy.1,
            address + first.unwrap_or(0)
        ));
    }
}

#[cfg(test)]
mod relay_tests {
    use super::{Flights, changed_runs, merge_piece, outside};

    /// A range two flights declared is free only when both are done, in
    /// either order; a submission that declares it many times counts once.
    #[test]
    fn a_range_stays_closed_until_every_flight_that_declared_it_is_done() {
        let (a, b) = ((0x1000, 0x2000), (0x1800, 0x3000));
        let mut flights = Flights::default();
        for _ in 0..3 {
            flights.declare(a, 7);
        }
        flights.declare(b, 7);
        flights.declare(a, 8);
        // Flight 7 lands: `a` is still flight 8's, `b` was only flight 7's.
        assert_eq!(flights.done(&[a, b]), vec![b]);
        // Declared again meanwhile by a third.
        flights.declare(a, 9);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        assert!(flights.0.is_empty());
        // The later flight first: the same.
        flights.declare(a, 1);
        flights.declare(a, 2);
        assert_eq!(flights.done(&[a]), vec![]);
        assert_eq!(flights.done(&[a]), vec![a]);
        // A range never counted is free at once.
        assert_eq!(flights.done(&[b]), vec![b]);
    }

    /// The merge moves exactly the bytes the card changed, into title
    /// memory and into the reference, and stores nothing else.
    #[test]
    fn the_merge_stores_only_what_the_card_changed() {
        let held0: Vec<u8> = (0..200u32).map(|i| (i * 5 + 1) as u8).collect();
        let mut now = held0.clone();
        for at in [0usize, 3, 4, 5, 64, 199] {
            now[at] ^= 0x33;
        }
        // The title wrote other bytes meanwhile, and one the card wrote too.
        let mut title = held0.clone();
        for at in [1usize, 5, 100] {
            title[at] = 0xee;
        }
        let title0 = title.clone();
        let mut held = held0.clone();
        assert!(merge_piece(&now, &mut held, &mut title));
        for at in 0..200 {
            if now[at] != held0[at] {
                assert_eq!((title[at], held[at]), (now[at], now[at]), "the card's byte at {at}");
            } else {
                assert_eq!((title[at], held[at]), (title0[at], held0[at]), "untouched at {at}");
            }
        }
        // Again with nothing new: nothing changes and nothing is reported.
        let (title1, held1) = (title.clone(), held.clone());
        assert!(!merge_piece(&now, &mut held, &mut title));
        assert_eq!((title, held), (title1, held1));
    }

    #[test]
    fn ranges_are_cut_out() {
        assert_eq!(outside(10, 50, vec![]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(0, 10), (50, 60)]), vec![(10, 50)]);
        assert_eq!(outside(10, 50, vec![(30, 35), (0, 12), (34, 40), (48, 99)]), vec![(12, 30), (40, 48)]);
        assert_eq!(outside(10, 50, vec![(0, 99)]), vec![]);
    }

    /// A model of a copy in the PC's memory under the relay: the card
    /// writes `copy` directly; `snap` is the snapshot; `title` is title
    /// memory. Flight A and flight B declare overlapping ranges and both
    /// are in the air; the title's CPU writes in between and its bytes are
    /// put in beside the flights' (`merge_under_writes`: into the copy and
    /// the snapshot); A lands while B is still writing; then B lands.
    #[test]
    fn two_flights_overlapping_ranges_and_a_cpu_write_in_between() {
        const N: usize = 64;
        let base: Vec<u8> = (0..N as u32).map(|i| i as u8).collect();
        let (mut copy, mut title) = (base.clone(), base.clone());
        // A declares [0, 40), B declares [24, 64): each byte's snapshot is
        // taken when the first range that holds it is declared.
        let (a, b) = ((0usize, 40usize), (24usize, 64usize));
        let mut snap = base.clone();
        // The card runs A: writes 4..12 and 30..34.
        for at in (4..12).chain(30..34) {
            copy[at] = 0xa0;
        }
        // The title's CPU writes 8..10 (over A's bytes), 20 and 50, after
        // A's refresh: the next submission's preparing finds them against
        // the snapshot and puts them in, into the copy and the snapshot.
        for at in [8usize, 9, 20, 50] {
            title[at] = 0x77;
        }
        for (at, n) in changed_runs(&title, &snap) {
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // The card starts B: it has written 32..36 so far, over A's 32..34.
        for at in 32..36 {
            copy[at] = 0xb0;
        }
        // A lands (B still on the card): its range is merged. The copy is
        // read once and not written.
        let seen = copy.clone();
        {
            let mut t = title[a.0..a.1].to_vec();
            assert!(merge_piece(&seen[a.0..a.1], &mut snap[a.0..a.1], &mut t));
            title[a.0..a.1].copy_from_slice(&t);
        }
        assert_eq!(copy, seen, "the merge stores nothing into the copy");
        // A's bytes are in title memory, except where the title wrote
        // after A's shaders did (8, 9: the title's); B's first bytes came
        // along; nothing of B's range beyond A's was touched.
        for at in (4..8).chain(10..12).chain(30..32) {
            assert_eq!(title[at], 0xa0, "A's byte at {at}");
        }
        assert_eq!((title[8], title[9], title[20], title[50]), (0x77, 0x77, 0x77, 0x77));
        for at in 32..36 {
            assert_eq!(title[at], 0xb0, "B's byte so far at {at}");
        }
        // The card finishes B: 36..44.
        for at in 36..44 {
            copy[at] = 0xb1;
        }
        // The title's CPU writes 40 after B's shader did: found against
        // the snapshot at the next preparing, which is before B lands.
        title[40] = 0x55;
        for (at, n) in changed_runs(&title[b.0..b.1], &snap[b.0..b.1]) {
            let at = at + b.0;
            copy[at..at + n].copy_from_slice(&title[at..at + n]);
            snap[at..at + n].copy_from_slice(&title[at..at + n]);
        }
        // B lands.
        let seen = copy.clone();
        {
            let mut t = title[b.0..b.1].to_vec();
            assert!(merge_piece(&seen[b.0..b.1], &mut snap[b.0..b.1], &mut t));
            title[b.0..b.1].copy_from_slice(&t);
        }
        // B's results are there, never replaced by A's older ones.
        for at in 32..36 {
            assert_eq!(title[at], 0xb0);
        }
        for at in (36..40).chain(41..44) {
            assert_eq!(title[at], 0xb1, "B's byte at {at}");
        }
        assert_eq!(title[40], 0x55, "the title wrote it last");
        // In the declared ranges the copy, the snapshot and title memory
        // now agree byte for byte: nothing is left to hand back, and
        // nothing was lost.
        assert_eq!(copy[a.0..b.1], title[a.0..b.1]);
        assert_eq!(snap[a.0..b.1], title[a.0..b.1]);
        // A third merge finds nothing.
        let mut t = title.clone();
        assert!(!merge_piece(&copy, &mut snap, &mut t));
    }

    /// A model of a copy in the card's memory under the relay: `card` is
    /// the card's side, `cpu` the CPU's side, `title` title memory. The
    /// title's changed bytes go into the CPU's side at once and up to the
    /// card with a later send; pieces fetched before that send must not
    /// take the card's older bytes there for something the card wrote.
    #[test]
    fn title_bytes_not_yet_on_the_card_are_not_taken_for_shader_writes() {
        const N: usize = 48;
        let base: Vec<u8> = (0..N as u32).map(|i| (i + 100) as u8).collect();
        let (mut card, mut cpu, mut title) = (base.clone(), base.clone(), base.clone());
        // Flight A's shaders write 10..14 on the card.
        for at in 10..14 {
            card[at] = 0xa0;
        }
        // The title's CPU writes 12 (over A's) and 30; the preparing of
        // the next submission puts both into the CPU's side and notes them
        // to go up (send 5). They are not on the card yet.
        title[12] = 0x77;
        title[30] = 0x78;
        let mut noted = Vec::new();
        for (at, n) in changed_runs(&title, &cpu) {
            cpu[at..at + n].copy_from_slice(&title[at..at + n]);
            noted.push((at, at + n, 5u64));
        }
        // A's pieces are fetched after send 4, before send 5: the fetch
        // holds A's bytes and the card's old bytes at 12 and 30.
        let fetched = card.clone();
        let fetched_after = 4u64;
        // Send 5 goes up behind it.
        for &(a, b, _) in &noted {
            card[a..b].copy_from_slice(&cpu[a..b]);
        }
        // The merge leaves out what went up after the fetch.
        let skip: Vec<(usize, usize)> = noted.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)).collect();
        let mut changed = false;
        for (a, b) in outside(0, N, skip) {
            let mut t = title[a..b].to_vec();
            changed |= merge_piece(&fetched[a..b], &mut cpu[a..b], &mut t);
            title[a..b].copy_from_slice(&t);
        }
        assert!(changed);
        // A's bytes arrived; the title's two writes stand, in title memory
        // and on both sides of the copy.
        for at in [10usize, 11, 13] {
            assert_eq!((title[at], cpu[at]), (0xa0, 0xa0));
        }
        assert_eq!((title[12], cpu[12], card[12]), (0x77, 0x77, 0x77));
        assert_eq!((title[30], cpu[30], card[30]), (0x78, 0x78, 0x78));
        assert_eq!(card, cpu);
        assert_eq!(card, title);
        // Without the leaving out, the card's old byte at 30 would have
        // been taken for a shader write and put over the title's.
        let mut wrong_title = title.clone();
        let mut wrong_cpu = cpu.clone();
        merge_piece(&fetched, &mut wrong_cpu, &mut wrong_title);
        assert_eq!(wrong_title[30], base[30]);
    }
}

/// Snapshot memory kept for reuse (already touched, so no page faults).
static POOL: Mutex<Vec<Vec<u8>>> = Mutex::new(Vec::new());

/// Snapshot buffers out of the pool (held by copies) and their bytes: a
/// snapshot has room for its whole copy, all of it committed, whatever
/// part of it the declared ranges fill.
static SNAPSHOTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: snapshot buffers held by copies, and their bytes.
pub(crate) fn snapshots() -> [u64; 2] {
    [SNAPSHOTS[0].load(Ordering::Relaxed), SNAPSHOTS[1].load(Ordering::Relaxed)]
}

/// A snapshot buffer of `bytes` (its capacity) is no longer held by a copy.
fn snapshot_gone(bytes: usize) {
    SNAPSHOTS[0].fetch_sub(1, Ordering::Relaxed);
    SNAPSHOTS[1].fetch_sub(bytes as u64, Ordering::Relaxed);
}

/// An empty buffer with room for `n` bytes, from the pool if it has one
/// of about that size: a small copy's snapshot took the 174 MB buffer kept
/// for the big heap's, which then got another, until eight of them were
/// reserved (1.4 GB in run 24).
fn pooled(n: usize) -> Vec<u8> {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let most = n.saturating_mul(4).max(1 << 20);
    let fit = pool
        .iter()
        .enumerate()
        .filter(|(_, v)| v.capacity() >= n && v.capacity() <= most)
        .min_by_key(|(_, v)| v.capacity())
        .map(|(i, _)| i);
    let mut v = match fit {
        Some(i) => pool.swap_remove(i),
        None => Vec::with_capacity(n),
    };
    v.clear();
    SNAPSHOTS[0].fetch_add(1, Ordering::Relaxed);
    SNAPSHOTS[1].fetch_add(v.capacity() as u64, Ordering::Relaxed);
    v
}

/// Returns a snapshot buffer to the pool: at most 8 kept, the largest, and
/// of those the largest one and the next ones up to `KEPT` bytes in all;
/// none under the commit floor.
fn give_back(v: Vec<u8>) {
    const KEPT: usize = 512 << 20;
    snapshot_gone(v.capacity());
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    // Under the commit floor (`crate::memline`) none is kept: the pool goes
    // and the next snapshot is allocated anew.
    if crate::memline::pressed() {
        let freed = std::mem::take(&mut *pool);
        drop(pool);
        drop((v, freed));
        return;
    }
    pool.push(v);
    pool.sort_by_key(|v| std::cmp::Reverse(v.capacity()));
    pool.truncate(8);
    let (mut seen, mut total) = (0, 0usize);
    pool.retain(|v| {
        seen += 1;
        total = total.saturating_add(v.capacity());
        seen == 1 || total <= KEPT
    });
}

const NOT_WATCHED: u8 = 2;

// Fixed-size equality compiles to two SIMD comparisons on Windows x64.
// Only the first or last unequal chunk needs a byte-wise search.
const DIFF_CHUNK: usize = 32;

/// Where `a` and `b` first differ within their common length.
fn first_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    let (offset, a, b) = match a_chunks.iter().zip(b_chunks).position(|(x, y)| x != y) {
        Some(i) => (i * DIFF_CHUNK, &a_chunks[i][..], &b_chunks[i][..]),
        None => (len - a_tail.len(), a_tail, b_tail),
    };
    a.iter().zip(b).position(|(x, y)| x != y).map(|i| offset + i)
}

/// Where `a` and `b` last differ within their common length.
fn last_difference(a: &[u8], b: &[u8]) -> Option<usize> {
    let len = a.len().min(b.len());
    let (a_chunks, a_tail) = a[..len].as_chunks::<DIFF_CHUNK>();
    let (b_chunks, b_tail) = b[..len].as_chunks::<DIFF_CHUNK>();
    if let Some(i) = a_tail.iter().zip(b_tail).rposition(|(x, y)| x != y) {
        return Some(len - a_tail.len() + i);
    }
    let chunk = a_chunks.iter().zip(b_chunks).rposition(|(x, y)| x != y)?;
    a_chunks[chunk]
        .iter()
        .zip(&b_chunks[chunk])
        .rposition(|(x, y)| x != y)
        .map(|i| chunk * DIFF_CHUNK + i)
}

/// [`Shadow::read_written`] on plain memory: `copy` and `old` are a copy's
/// bytes and its snapshot's for title memory from `at` on, and `dst` holds
/// title memory's bytes there as a fill read them. By 64 KiB piece of title
/// memory, from the first byte where the copy differs from its snapshot to
/// the last, `title` reads title memory (false where it is not all mapped:
/// the piece is left as it is) and `dst` takes the copy's byte where it
/// differs and the title's elsewhere: what [`Shadow::copy_back`] puts into
/// title memory there. Returns the bytes taken from the copy.
fn merge_written(
    at: usize,
    copy: &[u8],
    old: &[u8],
    dst: &mut [u8],
    mut title: impl FnMut(usize, &mut [u8]) -> bool,
) -> usize {
    let mut taken = 0;
    let mut read = Vec::new();
    let mut i = 0;
    while i < copy.len() {
        let n = (PIECE - (at + i) % PIECE).min(copy.len() - i);
        let (c, o) = (&copy[i..i + n], &old[i..i + n]);
        if let Some(first) = first_difference(c, o) {
            let last = last_difference(c, o).unwrap_or(first);
            read.clear();
            read.resize(last + 1 - first, 0);
            if title(at + i + first, &mut read) {
                for (k, t) in read.iter_mut().enumerate() {
                    if c[first + k] != o[first + k] {
                        *t = c[first + k];
                        taken += 1;
                    }
                }
                dst[i + first..=i + last].copy_from_slice(&read);
            }
        }
        i += n;
    }
    taken
}

#[cfg(test)]
mod read_written_tests {
    use super::{PIECE, merge_written};

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

    /// The copy-back of `Shadow::copy_back_with` over the declared range,
    /// title memory from `base` on: by 64 KiB piece, from the first byte
    /// where the copy differs from the snapshot to the last, title memory
    /// takes the copy's byte where it differs, unless a page there is not
    /// mapped.
    fn copy_back(base: usize, title: &mut [u8], mapped: &dyn Fn(usize) -> bool, copy: &[u8], old: &[u8]) {
        let mut at = 0;
        while at < copy.len() {
            let n = (PIECE - (base + at) % PIECE).min(copy.len() - at);
            let (c, o) = (&copy[at..at + n], &old[at..at + n]);
            if let (Some(first), Some(last)) =
                (c.iter().zip(o).position(|(x, y)| x != y), c.iter().zip(o).rposition(|(x, y)| x != y))
                && (at + first..=at + last).all(mapped)
            {
                for b in at + first..=at + last {
                    if copy[b] != old[b] {
                        title[b] = copy[b];
                    }
                }
            }
            at += n;
        }
    }

    /// A fill of title memory (zeros where it is not mapped, as
    /// `crate::host::copy_mapped`).
    fn fill(title: &[u8], mapped: &dyn Fn(usize) -> bool, from: usize, to: usize) -> Vec<u8> {
        (from..to).map(|b| if mapped(b) { title[b] } else { 0 }).collect()
    }

    /// A fill under shaders' writes, read from the copy, is the fill after
    /// the copy-back: for the whole declared range and parts of it, with
    /// title memory unmapped by whole pieces, shader writes that put back
    /// the snapshot's bytes, and title writes after the snapshot.
    #[test]
    fn a_fill_read_from_the_copy_is_the_fill_after_the_copy_back() {
        let mut rng = Rng(0x2545_f491_4f6c_dd1d);
        // On a page's line, not a piece's.
        let base = 7 * PIECE + 0x3000;
        let len = 5 * PIECE;
        for round in 0..400 {
            let mut title: Vec<u8> = (0..len).map(|_| rng.next() as u8).collect();
            let old = title.clone();
            let mut copy = old.clone();
            // Shaders write runs, some with the bytes that were there.
            for _ in 0..rng.below(12) {
                let at = rng.below(len);
                let n = (1 + rng.below(3 * 4096)).min(len - at);
                if rng.below(4) != 0 {
                    for b in &mut copy[at..at + n] {
                        *b = rng.next() as u8;
                    }
                }
            }
            // The title writes some bytes after the snapshot.
            for _ in 0..rng.below(8) {
                let at = rng.below(len);
                let n = (1 + rng.below(64)).min(len - at);
                for b in &mut title[at..at + n] {
                    *b = rng.next() as u8;
                }
            }
            // Now and then a whole piece of title memory is not mapped.
            let hole = (rng.below(3) == 0).then(|| {
                let lo = (rng.below(4) + 1) * PIECE - base % PIECE;
                (lo, (lo + PIECE).min(len))
            });
            let mapped = |b: usize| hole.is_none_or(|(lo, hi)| b < lo || b >= hi);
            let (from, to) = if rng.below(2) == 0 {
                (0, len)
            } else {
                let from = rng.below(len - 1);
                (from, from + 1 + rng.below(len - from - 1))
            };
            let mut after = title.clone();
            copy_back(base, &mut after, &mapped, &copy, &old);
            let want = fill(&after, &mapped, from, to);
            let mut got = fill(&title, &mapped, from, to);
            let taken = merge_written(base + from, &copy[from..to], &old[from..to], &mut got, |at, out| {
                let b = at - base;
                if !(b..b + out.len()).all(mapped) {
                    return false;
                }
                out.copy_from_slice(&title[b..b + out.len()]);
                true
            });
            assert_eq!(got, want, "round {round}, fill {from:#x}..{to:#x}");
            let written = (from..to).filter(|&b| copy[b] != old[b] && mapped(b)).count();
            assert_eq!(taken, written, "round {round}");
        }
    }
}

#[cfg(test)]
mod difference_tests {
    use super::{first_difference, last_difference};

    /// The chunk searches agree with a byte-wise scan across chunk boundaries.
    #[test]
    fn differences_are_found_as_byte_by_byte() {
        let base: Vec<u8> = (0..98u32).map(|i| (i * 7 + 3) as u8).collect();
        for len in [0, 1, 31, 32, 33, 63, 64, 65, 97] {
            let a = &base[1..1 + len];
            let same = a.to_vec();
            assert_eq!(first_difference(&same, a), None);
            assert_eq!(last_difference(&same, a), None);
            for at in 0..len {
                for other in at..len {
                    let mut b = a.to_vec();
                    b[at] ^= 0x80;
                    b[other] ^= 0x01;
                    let first = a.iter().zip(&b).position(|(x, y)| x != y);
                    let last = a.iter().zip(&b).rposition(|(x, y)| x != y);
                    assert_eq!(first_difference(a, &b), first, "{len} {at} {other}");
                    assert_eq!(last_difference(a, &b), last, "{len} {at} {other}");
                }
            }
        }
    }

    #[test]
    fn differences_ignore_bytes_past_the_shorter_slice() {
        let a = [0; 65];
        let mut b = [0; 67];
        b[65] = 1;
        assert_eq!(first_difference(&a, &b), None);
        assert_eq!(last_difference(&b, &a), None);
        b[32] = 1;
        b[64] = 1;
        assert_eq!(first_difference(&b, &a), Some(32));
        assert_eq!(last_difference(&a, &b), Some(64));
    }
}

/// The unit shader writes are looked for and put back in.
const PIECE: usize = 64 << 10;

/// Copies `len` bytes, on several threads if they are many: a snapshot of
/// the world's big heap is 100-170 MB per submission, 2-4 GB a frame, and
/// one thread copies some 8 GB/s.
///
/// # Safety
/// Both ranges are valid for `len` bytes and do not overlap.
unsafe fn copy_wide(src: *const u8, dst: *mut u8, len: usize) {
    const PER_THREAD: usize = 16 << 20;
    let threads = (len / PER_THREAD).clamp(1, 6);
    if threads == 1 {
        // SAFETY: the caller's.
        unsafe { std::ptr::copy_nonoverlapping(src, dst, len) };
        return;
    }
    let (src, dst, each) = (src as usize, dst as usize, len.div_ceil(threads));
    std::thread::scope(|s| {
        for i in 0..threads {
            let at = (i * each).min(len);
            let n = each.min(len - at);
            s.spawn(move || {
                // SAFETY: a part of the caller's ranges.
                unsafe { std::ptr::copy_nonoverlapping((src + at) as *const u8, (dst + at) as *mut u8, n) };
            });
        }
    });
}

/// Whether the snapshots of a submission's shader-written ranges, and the
/// compare that finds what the shaders wrote, run on several threads
/// across ranges (`VICTORY_WIDE_BACK=1`; off by default).
///
/// Both ran range by range on the submitting thread, a range on several
/// threads only if it was 16 MiB or more. The ranges left on the CPU's side
/// since the big buffers went to the card are small and many: in run 86,
/// 7,024 declared ranges a frame, 375 MB of them compared after the card
/// ran in 563 pieces of work, none of 16 MiB: 51 ms a frame of snapshots
/// and 35 ms of comparing, one thread, bound by one core's memory speed.
///
/// With the switch a submission's snapshots are noted as it is prepared
/// and taken together when its refreshes are done, just before it goes to
/// the card ([`take_snapshots`]), and the compares of everything declared
/// run together after the card ([`find_all`]), both cut into pieces of
/// work of about 1 MiB for a few threads that are kept ([`wide`]). The
/// bytes taken and the bytes compared are the same ones: a range's bytes
/// cannot change between the moment it is declared and the end of the
/// preparing (a declared range is left alone by every refresh, the title's
/// own copies write title memory and not the copy, and the card runs the
/// lists only afterwards).
pub(crate) fn wide_back() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // On by default; `VICTORY_WIDE_BACK=0` turns it off.
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_BACK").is_none_or(|v| v != "0"))
}

type Job = Box<dyn FnOnce() + Send>;

/// Threads kept for [`wide`]: what they are to do, and the word that there
/// is something.
struct Workers {
    jobs: Mutex<std::collections::VecDeque<Job>>,
    more: std::sync::Condvar,
}

/// How many threads [`workers`] keeps.
fn kept_threads() -> usize {
    std::thread::available_parallelism().map_or(4, |n| n.get() / 4).clamp(2, 6)
}

/// The threads a [`wide_run`] runs its jobs on: the kept ones and the
/// caller.
pub(crate) fn wide_threads() -> usize {
    kept_threads() + 1
}

fn workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    pool(&WORKERS, &STARTED, kept_threads(), "victory-wide")
}

/// The threads of [`wide_textures`]: a pool of their own, so that a
/// texture's check (which takes the reading of title memory, the journal
/// and the guards) never runs inside another caller's [`wide`] that holds a
/// lock of its own meanwhile (the heap sync holds the shader heaps' lock
/// while its jobs run).
fn texture_workers() -> &'static Workers {
    static WORKERS: std::sync::OnceLock<Workers> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    pool(&WORKERS, &STARTED, kept_threads().min(4), "victory-textures")
}

/// A pool of `n` kept threads named `name-i`, made once.
fn pool(
    cell: &'static std::sync::OnceLock<Workers>,
    started: &'static std::sync::Once,
    n: usize,
    name: &'static str,
) -> &'static Workers {
    let workers = cell.get_or_init(|| Workers {
        jobs: Mutex::new(std::collections::VecDeque::new()),
        more: std::sync::Condvar::new(),
    });
    started.call_once(|| {
        for i in 0..n {
            let _ = std::thread::Builder::new().name(format!("{name}-{i}")).spawn(move || {
                let workers = cell.get().expect("made above");
                loop {
                    let job = {
                        let mut jobs = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
                        loop {
                            if let Some(job) = jobs.pop_front() {
                                break job;
                            }
                            jobs = workers.more.wait(jobs).unwrap_or_else(|e| e.into_inner());
                        }
                    };
                    job();
                }
            });
        }
    });
    workers
}

/// Runs `jobs` on the kept threads and on this one, and returns when all
/// of them are done. The jobs copy and compare memory; they do not panic.
fn wide(jobs: Vec<Job>) {
    wide_on(workers(), jobs);
}

/// [`wide`] on the threads of [`texture_workers`] and this one: the
/// submission's texture checks gathered by `crate::host`'s
/// `VICTORY_WIDE_TEXTURES`.
pub(crate) fn wide_textures(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide_on(texture_workers(), jobs);
}

/// [`wide`] on `workers`.
fn wide_on(workers: &'static Workers, jobs: Vec<Job>) {
    if jobs.len() < 2 {
        for job in jobs {
            job();
        }
        return;
    }
    let left = Arc::new((Mutex::new(jobs.len()), std::sync::Condvar::new()));
    {
        let mut queue = workers.jobs.lock().unwrap_or_else(|e| e.into_inner());
        for job in jobs {
            let left = left.clone();
            queue.push_back(Box::new(move || {
                job();
                let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
                *n -= 1;
                if *n == 0 {
                    left.1.notify_all();
                }
            }));
        }
    }
    workers.more.notify_all();
    loop {
        let job = workers.jobs.lock().unwrap_or_else(|e| e.into_inner()).pop_front();
        match job {
            Some(job) => job(),
            None => break,
        }
    }
    let mut n = left.0.lock().unwrap_or_else(|e| e.into_inner());
    while *n > 0 {
        n = left.1.wait(n).unwrap_or_else(|e| e.into_inner());
    }
}

/// [`wide`], for the host's own work on host copies (a submission's
/// refreshes, the merge of what shaders wrote to card copies).
pub(crate) fn wide_run(jobs: Vec<Box<dyn FnOnce() + Send>>) {
    wide(jobs);
}

/// Since the last frame-rate line, of the refreshes of host copies (the
/// frame line's "copies"), in microseconds on the submitting thread: asking
/// which pages the title wrote ([`crate::watch::writes`], the journal or
/// the kernel, and the guards of view memory); moving title memory into
/// the copies (with [`wide_copy`] the wall time of the moves on all
/// threads; with `VICTORY_WIDE_REFRESH` the sum of its threads' times);
/// then bytes found the same and left alone ([`skip_same`]); and
/// the asking made once per submission before its refreshes
/// (`crate::watch::ask_once`, not part of "copies").
pub(crate) static REFRESH_SPLIT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// Whether a refresh writes only the 4 KiB blocks of a host copy whose
/// bytes differ from title memory (`VICTORY_SKIP_SAME=1`; off by default).
///
/// The write watch and the guards name 64 KiB pages the title wrote, not
/// bytes that changed, and a page is copied whole: 64-160 MB a frame in the
/// opening cutscene. Much of it is not new: the title rewrites buffers with
/// what they held, and every piece we put back into title memory after the
/// card wrote it (`Host::card_back`, [`Shadow::copy_back`]) shows as a
/// title write and is copied once more at the next refresh (the comment in
/// `copy_back_with`: "some dozens of MB a frame"). For a copy in the card's
/// memory each such page also went up to the card again and was hashed
/// there again.
///
/// With the switch title memory is read into scratch memory that stays in
/// the core's cache (256 KiB at a time, through the kernel as before), then
/// compared with the copy, 4 KiB at a time; only blocks that differ are
/// written, and only they count as changed: a kept snapshot stays current
/// elsewhere, and a copy in the card's memory sends up only them. The copy
/// ends with the same bytes as a plain copy gives it, so nothing stale
/// reaches the card through a copy in the PC's memory (the card reads that
/// buffer itself). A page that cannot be read whole is copied as before.
///
/// What it assumes for copies in the card's memory: their CPU side equals
/// their card side outside the bytes still to go up (`Card::dirty`). That
/// holds but for shader writes outside declared ranges, which the card
/// keeps until a declared range brings them back (the hole [`cards`]
/// already names): a page the title rewrites with the bytes the CPU side
/// holds used to go up again and overwrite such a write on the card, and
/// with the switch does not. And a new copy's two sides start as zeros
/// (committed resources are zeroed), so a zero block of title memory is
/// left alone on both from the first refresh on.
pub(crate) fn skip_same() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SKIP_SAME").is_some_and(|v| v == "1"))
}

/// Whether a submission's refreshes move their bytes on several threads
/// (`VICTORY_WIDE_COPY=1`; off by default).
///
/// A flush of a submission's refreshes asked and copied copy by copy on
/// the submitting thread, through the kernel (`ReadProcessMemory`), about
/// 4-5 GB/s (opening cutscene: 26-57 ms a frame for 64-160 MB).
/// `VICTORY_WIDE_REFRESH` spreads whole copies over threads, but a frame's
/// bytes lie mostly in a few big copies. With this switch the flush first
/// asks, for every copy, which pages to take ([`Shadow::plan`], on the
/// submitting thread, the notes made as before), then moves all the runs
/// on the threads kept for [`wide`], cut into pieces of work of about
/// 1 MiB, and then tells each copy what changed ([`Shadow::moved`]). The
/// moves are done before the flush returns, as before, so nothing reads a
/// copy or writes title memory before its bytes are in. With
/// [`skip_same`] each piece of work writes only what differs.
pub(crate) fn wide_copy() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_COPY").is_some_and(|v| v == "1"))
}

/// Whether, after the card wrote a copy in its memory, only the bytes in
/// which the merge left the CPU's side other than the card's go up again
/// (`VICTORY_MERGE_EXACT=1`; off by default).
///
/// [`Shadow::card_merge`] marked every merged span (from the first byte
/// the card changed to the last) to be sent up before the next submission,
/// though the card holds those bytes already: 50-70 MB a frame sent up and
/// hashed again for nothing (opening cutscene: 1,000 pieces, 66 MB fetched
/// a frame, 130 MB sent up). The card's side holds what was fetched (every
/// flight has run, and nothing runs between the fetch and the merge, both
/// under the submission lock); the CPU's side holds the merge; they differ
/// only where the title's bytes won (bytes the title changed and the card
/// did not), and only those go up.
///
/// A piece not sent up is not hashed again, so the compare after the lists
/// keeps the hash it makes (as with the relay: `CARD_COMPARE`'s mode 2),
/// and the next compare finds only what shaders write from then on. Its
/// hash is then of the card's bytes, as a hash made after a send is.
pub(crate) fn merge_exact() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_MERGE_EXACT").is_some_and(|v| v == "1"))
}

/// What host uploads do in the picked spans of copies
/// ([`Shadow::picked`]): the title memory behind unordered access buffer
/// views of the view heap, which the 10 programs that store through views
/// they pick by index (BVH skinning, cable and rope, tree animation,
/// `CSUnpackIB*`) may write with no flight declaring it
/// (`crate::draw::picked_uav_writes`, off). `VICTORY_PICKED_GUARD`, read
/// once: unset or `0` counts the bytes the refreshes put there
/// ([`PICKED_GUARD`]); `1` also compares each such byte with what the
/// title had there before, to count the ones the title did not change but
/// the refresh writes (and a copy in the card's memory sends up) all the
/// same; `2` writes and sends up only the bytes the title changed.
///
/// Why: the refresh copies whatever the write watch names, 4 KiB parts
/// (`VICTORY_FINE_PAGES=1`) or 64 KiB pages, whole; all of memory that is
/// not write-watched. A part is named when the title wrote any byte of it,
/// and also after our own writes into title memory: every merge of what
/// the card wrote (`Shadow::card_merge_relay`, `copy_back`) goes through
/// the kernel and shows as a title write. Each such part goes from title
/// memory into the copy (for a copy in the card's memory: into its CPU
/// side, and up to the card before the next submission, behind the
/// flights in the air). Where a shader wrote that part through a picked
/// view, title memory and the CPU side never had its bytes (no flight
/// declared them, so nothing fetched them), and the refresh puts the bytes
/// from before over them: older skinned vertices, indices from before an
/// unpack. Which parts, and whether a draw reads them before the next
/// skinning pass writes them again, depends on when flights land and
/// merges run, so on how fast the card is.
///
/// With `2`, a copy in the card's memory compares title memory with its
/// CPU side, which holds what the card has outside the shaders' writes
/// (both start as zeros; every change goes up), and only the runs that
/// differ (closer than 16 bytes as one) go into it and up. A copy without
/// a card side (shared buffer, GPU upload heap) cannot tell (it holds the
/// shaders' bytes): it keeps the title's bytes as last copied there
/// ([`Reference`], made at the first refresh with the switch, a part at a
/// time) and writes only bytes the title changed since; a part it has not
/// taken yet is copied whole once.
///
/// What `2` assumes: a title write of a byte with the value it already had
/// need not reach the card. Where a shader changed that byte unannounced
/// the title's rewrite of the old value is lost (a counter the title
/// clears to the zero it last wrote): the console would have taken it.
/// `VICTORY_SKIP_SAME` assumes the same for whole 4 KiB blocks.
pub(crate) fn picked_guard() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let level = std::env::var("VICTORY_PICKED_GUARD").ok().and_then(|v| v.trim().parse::<u8>().ok()).unwrap_or(0).min(2);
        if level > 0 {
            tracing::warn!(
                level,
                "VICTORY_PICKED_GUARD: refreshes into title memory behind the heap's unordered access buffer views are compared                  with the title's bytes from before{}",
                if level == 2 { ", and only the bytes the title changed are written and sent up" } else { " (counted only)" }
            );
        }
        level
    })
}

/// Since the last frame-rate line ([`picked_guard`]), bytes: refreshed into
/// picked spans of copies in the card's memory, and of them the ones the
/// title had not changed (with `1` or `2`); refreshed into picked spans of
/// copies without a card side, and of them the unchanged ones (parts the
/// reference held); sent up to the card's side in picked spans (taken from
/// `Card::dirty`); not written or sent thanks to `2`; of the first and
/// third, copied whole because the memory is not write-watched; runs
/// written by `2`. Kept, not reset: copies marked so far.
pub(crate) static PICKED_GUARD: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];
mod picked_stat {
    pub(super) const CARD: usize = 0;
    pub(super) const CARD_SAME: usize = 1;
    pub(super) const PC: usize = 2;
    pub(super) const PC_SAME: usize = 3;
    pub(super) const SENT: usize = 4;
    pub(super) const SPARED: usize = 5;
    pub(super) const WHOLE: usize = 6;
    pub(super) const MARKED: usize = 7;
    pub(super) const RUNS: usize = 8;
}

/// Runs of `exact` closer than this many bytes go as one (the bytes between
/// are the CPU side's either way; a few more bytes beat another copy).
const PICKED_GAP: usize = 16;

/// [`changed_runs`]' runs joined where fewer than [`PICKED_GAP`] bytes
/// lie between them.
fn close_runs(runs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (at, n) in runs {
        match out.last_mut() {
            Some(last) if at < last.0 + last.1 + PICKED_GAP => last.1 = at + n - last.0,
            _ => out.push((at, n)),
        }
    }
    out
}

/// The frame-rate line of [`picked_guard`].
pub(crate) fn picked_guard_line(frames: f64) {
    use picked_stat::*;
    let n: [u64; 9] = std::array::from_fn(|i| if i == MARKED { PICKED_GUARD[i].load(Ordering::Relaxed) } else { PICKED_GUARD[i].swap(0, Ordering::Relaxed) });
    let mb = |b: u64| b as f64 / 1e6 / frames;
    tracing::info!(
        "host uploads into title memory behind the heap's unordered access buffer views (picked spans of {} copies;          VICTORY_PICKED_GUARD {}), per frame: {:.2} MB refreshed into copies in the card's memory ({:.2} MB of it the title had          not changed), {:.2} MB sent up there; {:.2} MB refreshed into copies without a card side ({:.2} MB unchanged);          {:.2} MB of both copied whole (not write-watched); {:.2} MB kept from being written or sent ({:.0} runs written)",
        n[MARKED],
        picked_guard(),
        mb(n[CARD]),
        mb(n[CARD_SAME]),
        mb(n[SENT]),
        mb(n[PC]),
        mb(n[PC_SAME]),
        mb(n[WHOLE]),
        mb(n[SPARED]),
        n[RUNS] as f64 / frames,
    );
    let e = PICKED_ELSEWHERE.each_ref().map(|n| n.load(Ordering::Relaxed));
    tracing::info!(
        "unordered access buffer views of the heap whose copy is not the one draws find for the same bytes (what shaders \
         write through them never reaches what draws read), at the last measure: {} of {} views, {:.2} MB; of them {} in \
         windows, {} in copies of their own bytes; {} views whose bytes draws would find no copy for yet",
        e[0],
        e[5],
        e[1] as f64 / 1e6,
        e[2],
        e[3],
        e[4],
    );
}

/// The title memory the programs that store through views picked from
/// the heap were last found to write (`crate::draw`'s `picked_exact`:
/// the views their dispatches' job records and root constants name, as
/// bound), by start address: (end, frame last named, the number of the
/// host copy the view's slot writes, [`MIXED`] where ranges of several
/// copies joined). Ranges that overlap or touch are one. Kept for the
/// counters ([`SKINNED_OVER`]) whatever `VICTORY_PICKED_EXACT` says; a
/// range not named for [`SKINNED_KEEP`] frames is let go.
static SKINNED: Mutex<BTreeMap<u64, (u64, u64, u64)>> = Mutex::new(BTreeMap::new());

/// [`SKINNED`]'s copy number where joined ranges were written in several
/// copies (or the copy is not known).
const MIXED: u64 = u64::MAX;

/// Whether [`SKINNED`] holds anything (a cheap first look).
static SKINNED_ANY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Frames a range of [`SKINNED`] stays without being named again.
const SKINNED_KEEP: u64 = 64;

/// Since the last frame-rate line, bytes of [`SKINNED`] ranges (what the
/// skinning, cable, rope, tree and index-unpacking programs write through
/// picked views): sent up from a copy's CPU side to its card side
/// ([`card_dirty`]: refreshes, merges, title bytes put in under writes),
/// written into copies without a card side by refreshes (the card reads
/// those buffers itself), read from title memory by title buffer copies
/// run on the CPU, read from title memory by fills of upload memory, put
/// into title memory by merges of fetched pieces, and bound by draws as
/// vertex or index buffers in a copy other than the one the shaders wrote.
pub(crate) static SKINNED_OVER: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
pub(crate) mod skinned {
    pub const SENT: usize = 0;
    pub const PC: usize = 1;
    pub const COPY_READ: usize = 2;
    pub const FILL_READ: usize = 3;
    pub const MERGED: usize = 4;
    pub const DRAWN_ELSEWHERE: usize = 5;
}

/// Of the heap's buffer views that only read, at the heap pass's last
/// measure: views over [`SKINNED`] memory, of them views whose slot shows
/// another copy than the one the shaders wrote, and those views' bytes of
/// it (what draws reading skinned vertices through the heap find there is
/// title memory as some refresh left it). Set, not summed.
pub(crate) static SKINNED_READERS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Title memory `[address, address + len)` is written by a dispatch of a
/// program that stores through views it picks from the heap, in the host
/// copy numbered `copy` (`MIXED` if not known) ([`SKINNED`]).
pub(crate) fn note_skinned(address: u64, len: u64, copy: u64) {
    if len == 0 {
        return;
    }
    let frame = crate::frame::current();
    let (mut from, mut to, mut copy) = (address, address + len, copy);
    let mut map = SKINNED.lock().unwrap_or_else(|e| e.into_inner());
    // Already held whole: only its frame changes.
    if let Some((_, entry)) = map.range_mut(..=from).next_back()
        && entry.0 >= to
    {
        entry.1 = frame;
        if entry.2 != copy {
            entry.2 = MIXED;
        }
        return;
    }
    // Joined with every range it meets or touches.
    let meets: Vec<u64> = map
        .range(..=to)
        .rev()
        .take_while(|(_, (end, _, _))| *end >= from)
        .map(|(&start, _)| start)
        .collect();
    for start in meets {
        if let Some((end, _, other)) = map.remove(&start) {
            from = from.min(start);
            to = to.max(end);
            if other != copy {
                copy = MIXED;
            }
        }
    }
    map.insert(from, (to, frame, copy));
    // Let go of what no dispatch named for a while (at most every 256th
    // insert looks).
    if map.len() % 256 == 0 {
        map.retain(|_, (_, seen, _)| seen.saturating_add(SKINNED_KEEP) >= frame);
    }
    SKINNED_ANY.store(!map.is_empty(), Ordering::Relaxed);
}

/// Bytes of title memory `[from, to)` inside [`SKINNED`] ranges; with
/// `copy`, only of ranges written in another copy than that one (and in a
/// known one).
fn skinned_in(from: u64, to: u64, copy: Option<u64>) -> u64 {
    if from >= to || !SKINNED_ANY.load(Ordering::Relaxed) {
        return 0;
    }
    let map = SKINNED.lock().unwrap_or_else(|e| e.into_inner());
    map.range(..to)
        .rev()
        .take_while(|(_, (end, _, _))| *end > from)
        .filter(|(_, (_, _, wrote))| copy.is_none_or(|c| *wrote != c && *wrote != MIXED))
        .map(|(&start, &(end, _, _))| end.min(to).saturating_sub(start.max(from)))
        .sum()
}

/// Bytes of title memory `[from, to)` inside [`SKINNED`] ranges.
pub(crate) fn skinned_bytes(from: u64, to: u64) -> u64 {
    skinned_in(from, to, None)
}

/// Bytes of title memory `[from, to)` inside [`SKINNED`] ranges the
/// shaders wrote in another copy than `shadow` (a reader of them that
/// finds `shadow` reads what some refresh left there).
pub(crate) fn skinned_elsewhere(from: u64, to: u64, shadow: &Shadow) -> u64 {
    skinned_in(from, to, Some(shadow.id))
}

/// Counts `[from, to)`'s bytes inside [`SKINNED`] ranges under `what`
/// ([`skinned`]).
pub(crate) fn count_skinned(what: usize, from: u64, to: u64) {
    let n = skinned_bytes(from, to);
    if n > 0 {
        SKINNED_OVER[what].fetch_add(n, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod skinned_tests {
    use super::{note_skinned, skinned_bytes, skinned_in};

    // The registry is shared; these addresses are this test's alone.
    #[test]
    fn noted_ranges_join_and_count_once() {
        const BASE: u64 = 0x7a00_0000_0000;
        note_skinned(BASE, 0x100, 7);
        note_skinned(BASE + 0x200, 0x100, 7);
        assert_eq!(skinned_bytes(BASE, BASE + 0x400), 0x200);
        assert_eq!(skinned_bytes(BASE + 0x80, BASE + 0x280), 0x80 + 0x80);
        // Read through copy 8: all of it is elsewhere; through 7, none.
        assert_eq!(skinned_in(BASE, BASE + 0x400, Some(8)), 0x200);
        assert_eq!(skinned_in(BASE, BASE + 0x400, Some(7)), 0);
        // One that bridges both makes them one range, counted once.
        note_skinned(BASE + 0x80, 0x200, 7);
        assert_eq!(skinned_bytes(BASE, BASE + 0x400), 0x300);
        // Inside: nothing changes but the copy (now two: not known).
        note_skinned(BASE + 0x10, 0x10, 9);
        assert_eq!(skinned_bytes(BASE, BASE + 0x400), 0x300);
        assert_eq!(skinned_in(BASE, BASE + 0x400, Some(8)), 0);
        assert_eq!(skinned_bytes(BASE + 0x300, BASE + 0x400), 0);
        assert_eq!(skinned_bytes(BASE + 0x2ff, BASE + 0x400), 1);
    }
}

/// [`SKINNED`]'s ranges and bytes now.
pub(crate) fn skinned_known() -> (usize, u64) {
    let map = SKINNED.lock().unwrap_or_else(|e| e.into_inner());
    (map.len(), map.iter().map(|(&start, &(end, _, _))| end - start).sum())
}

/// Since the last frame-rate line, with [`merge_exact`]: bytes the merges
/// still send up to the card, and the merged spans' bytes (what went up
/// without the switch).
pub(crate) static MERGE_SENT: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Title memory read at a time by [`move_same`]: scratch memory that stays
/// in the core's cache, and few enough kernel calls.
const SAME_CHUNK: usize = 256 << 10;
/// What [`move_same`] compares and writes: the kernel's page.
const SAME_BLOCK: usize = 4 << 10;
/// Changed blocks this close are written as one run (the bytes between
/// are the same either way): fewer, longer sends to the card.
const SAME_GAP: usize = 4 * SAME_BLOCK;

/// Appends `[from, to)` to `runs` (in order), joined with the last one if
/// they touch.
fn push_run(runs: &mut Vec<(usize, usize)>, from: usize, to: usize) {
    match runs.last_mut() {
        Some(last) if from <= last.1 => last.1 = last.1.max(to),
        _ => runs.push((from, to)),
    }
}

/// [`skip_same`]'s move: title memory `[src, src + len)` into a host
/// copy's mapping at `dst`, writing only 4 KiB blocks (counted from `src`)
/// whose bytes differ. Appends the runs written, as offsets from `src`, in
/// order, to `runs`, and returns their bytes. A part that cannot be read
/// whole is copied as [`crate::host::copy_mapped`] copies it (zeros where
/// nothing is mapped) and counts as written.
pub(crate) fn move_same(dst: usize, src: usize, len: usize, runs: &mut Vec<(usize, usize)>) -> usize {
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    SCRATCH.with(|scratch| {
        let mut scratch = scratch.borrow_mut();
        if scratch.len() < SAME_CHUNK {
            scratch.resize(SAME_CHUNK, 0);
        }
        let mut written = 0;
        let mut at = 0;
        while at < len {
            let n = (len - at).min(SAME_CHUNK);
            let title = &mut scratch[..n];
            if !crate::host::read_into((src + at) as *const u8, title) {
                crate::host::copy_mapped((dst + at) as *mut u8, (src + at) as *const u8, n, false);
                push_run(runs, at, at + n);
                written += n;
                at += n;
                continue;
            }
            // SAFETY: inside the caller's mapping of `len` bytes at `dst`,
            // which only this thread writes now.
            let host = unsafe { std::slice::from_raw_parts_mut((dst + at) as *mut u8, n) };
            written += write_differing(host, title, |a, b| push_run(runs, at + a, at + b));
            at += n;
        }
        written
    })
}

/// Writes into `host` the 4 KiB blocks of `title` (of one length) whose
/// bytes differ, runs of them closer than [`SAME_GAP`] as one; tells
/// `wrote` each run written (offsets, in order) and returns their bytes.
fn write_differing(host: &mut [u8], title: &[u8], mut wrote: impl FnMut(usize, usize)) -> usize {
    let n = host.len().min(title.len());
    let mut written = 0;
    let mut open: Option<(usize, usize)> = None;
    let mut put = |host: &mut [u8], (a, b): (usize, usize)| {
        host[a..b].copy_from_slice(&title[a..b]);
        wrote(a, b);
        b - a
    };
    let mut b = 0;
    while b < n {
        let e = (b + SAME_BLOCK).min(n);
        if host[b..e] != title[b..e] {
            match &mut open {
                Some(run) if b <= run.1 + SAME_GAP => run.1 = e,
                _ => {
                    if let Some(run) = open.take() {
                        written += put(host, run);
                    }
                    open = Some((b, e));
                }
            }
        }
        b = e;
    }
    if let Some(run) = open {
        written += put(host, run);
    }
    written
}

/// [`wide_copy`]'s moves: for each of `plans` (a copy and the title ranges
/// [`Shadow::plan`] gave for it), the ranges moved into the copy on the
/// threads kept for [`wide`], with [`skip_same`] only what differs. Returns
/// per plan the title ranges written (in order, joined where they touch)
/// and the bytes written in all; the caller tells each copy
/// ([`Shadow::moved`]).
pub(crate) fn move_wide(plans: &[(Arc<Shadow>, Vec<(usize, usize)>)]) -> (Vec<Vec<(usize, usize)>>, usize) {
    // A piece of work's parts: which plan, where in the copy, where in
    // title memory, how many bytes.
    type Part = (usize, usize, usize, usize);
    let same = skip_same();
    let found: Arc<Mutex<Vec<(usize, usize, usize)>>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine = Vec::with_capacity(group.len());
            for (which, dst, src, len) in group {
                if same {
                    let mut runs = Vec::new();
                    move_same(dst, src, len, &mut runs);
                    mine.extend(runs.into_iter().map(|(a, b)| (which, src + a, src + b)));
                } else {
                    crate::host::copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    mine.push((which, src, src + len));
                }
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    for (which, (shadow, runs)) in plans.iter().enumerate() {
        for &(from, to) in runs {
            let mut at = from;
            while at < to {
                let n = (WIDE_JOB - bytes).min(to - at);
                group.push((which, shadow.host_at(at), at, n));
                at += n;
                bytes += n;
                if bytes >= WIDE_JOB {
                    flush(&mut group, &mut jobs);
                    bytes = 0;
                }
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable();
    let mut out = vec![Vec::new(); plans.len()];
    let mut written = 0;
    for (which, from, to) in found {
        written += to - from;
        push_run(&mut out[which], from, to);
    }
    (out, written)
}

/// About this many bytes to a piece of work for [`wide`].
const WIDE_JOB: usize = 1 << 20;

/// With [`wide_back`]: the snapshots noted while a submission is prepared
/// (from, to, bytes), taken by [`take_snapshots`].
static SNAPS: Mutex<Vec<(usize, usize, usize)>> = Mutex::new(Vec::new());

/// With [`wide_back`], when a submission's refreshes are done and before
/// its lists go to the card: takes the snapshots [`Shadow::writing`] noted.
/// Nothing to do without the switch.
pub(crate) fn take_snapshots() {
    let snaps = std::mem::take(&mut *SNAPS.lock().unwrap_or_else(|e| e.into_inner()));
    if snaps.is_empty() {
        return;
    }
    wide(copy_jobs(snaps));
}

/// `copies` (from, to, bytes) as pieces of work of about [`WIDE_JOB`]
/// bytes each: long ones cut, short ones together.
fn copy_jobs(copies: Vec<(usize, usize, usize)>) -> Vec<Job> {
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<(usize, usize, usize)>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if !group.is_empty() {
            jobs.push(Box::new(move || {
                for (src, dst, len) in group {
                    // SAFETY: the caller's ranges: a host copy's mapping and
                    // its snapshot, which do not overlap.
                    unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, len) };
                }
            }));
        }
    };
    for (src, dst, len) in copies {
        let mut at = 0;
        while at < len {
            let n = (WIDE_JOB - bytes).min(len - at);
            group.push((src + at, dst + at, n));
            at += n;
            bytes += n;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    jobs
}

/// With [`wide_back`]: for each of `work` (a copy not in the card's memory
/// and a range `[address, end)` its shaders may have written, declared and
/// not yet done), the 64 KiB pieces that differ from the snapshot, as
/// [`Shadow::copy_back_with`] takes them; found on several threads.
pub(crate) fn find_all(work: &[(Arc<Shadow>, usize, usize)]) -> Vec<Vec<(usize, usize)>> {
    let scans = work
        .iter()
        .map(|(shadow, address, end)| {
            let (from, to) = shadow.inside(*address, end.saturating_sub(*address));
            // The snapshot's memory stays until the range is done with
            // (`copy_back_done`), which comes after this.
            match shadow.before.lock().unwrap_or_else(|e| e.into_inner()).as_ref() {
                Some(before) => (shadow.data as usize, before.as_ptr() as usize, shadow.start as usize, from, to),
                None => (0, 0, 0, 0, 0),
            }
        })
        .collect();
    find_pieces(scans)
}

/// For each of `scans` (where a copy's bytes are, where its snapshot's
/// are, the title address the copy starts at, offsets from and to): the
/// 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
/// [`Shadow::differing`] cuts them) of `[from, to)` whose bytes differ, as
/// (offset, bytes) in order. On several threads ([`wide`]), in pieces of
/// work of about [`WIDE_JOB`] bytes that hold whole pieces.
fn find_pieces(scans: Vec<(usize, usize, usize, usize, usize)>) -> Vec<Vec<(usize, usize)>> {
    // What a job finds: (which scan, its place among that scan's jobs, the
    // pieces).
    type Found = Vec<(usize, usize, Vec<(usize, usize)>)>;
    // A job's part of a scan: the scan, then which and place.
    type Part = (usize, usize, usize, usize, usize, usize, usize);
    let found: Arc<Mutex<Found>> = Arc::default();
    let mut jobs: Vec<Job> = Vec::new();
    let mut group: Vec<Part> = Vec::new();
    let mut bytes = 0;
    let flush = |group: &mut Vec<Part>, jobs: &mut Vec<Job>| {
        let group = std::mem::take(group);
        if group.is_empty() {
            return;
        }
        let found = found.clone();
        jobs.push(Box::new(move || {
            let mut mine: Found = Vec::with_capacity(group.len());
            for (data, before, start, from, to, which, place) in group {
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    // SAFETY: inside the copy's mapping and its snapshot
                    // (which holds the copy's bytes at the copy's offsets);
                    // nothing writes either while a submission's results
                    // are compared.
                    let (now, old) = unsafe {
                        (
                            std::slice::from_raw_parts((data + at) as *const u8, n),
                            std::slice::from_raw_parts((before + at) as *const u8, n),
                        )
                    };
                    if now != old {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                mine.push((which, place, pieces));
            }
            found.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
        }));
    };
    let count = scans.len();
    for (which, (data, before, start, from, to)) in scans.into_iter().enumerate() {
        let (mut at, mut place) = (from, 0);
        while at < to {
            // Cut where pieces end, so that every job sees whole pieces.
            let room = (WIDE_JOB - bytes).max(PIECE);
            let mut next = (at + room).min(to);
            if next < to {
                next -= (start + next) % PIECE;
                if next <= at {
                    next = (at + PIECE - (start + at) % PIECE).min(to);
                }
            }
            group.push((data, before, start, at, next, which, place));
            bytes += next - at;
            place += 1;
            at = next;
            if bytes >= WIDE_JOB {
                flush(&mut group, &mut jobs);
                bytes = 0;
            }
        }
    }
    flush(&mut group, &mut jobs);
    wide(jobs);
    let mut found = std::mem::take(&mut *found.lock().unwrap_or_else(|e| e.into_inner()));
    found.sort_unstable_by_key(|f| (f.0, f.1));
    let mut out = vec![Vec::new(); count];
    for (which, _, pieces) in found {
        out[which].extend(pieces);
    }
    out
}

/// Since the last frame-rate line, of the shader-written ranges looked for
/// on the CPU (copies not in the card's memory): bytes snapshotted in
/// windows outside buffers and in title buffers' copies; bytes compared in
/// windows, in buffers' copies, and of those in copies of 1 MiB or more;
/// ranges compared in windows and in buffers' copies.
static BACK_CLASS: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];

/// For the frame-rate line: where the CPU's search for shader writes goes
/// (per frame). A copy of a title buffer can go to the card's memory,
/// where the card does the search (`VICTORY_CARD_BUFFERS_MIN_MB`); a
/// window outside buffers cannot as the code stands.
pub(crate) fn back_line(frames: f64) {
    let c: Vec<f64> = BACK_CLASS.iter().map(|c| c.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if c.iter().all(|&v| v == 0.0) {
        return;
    }
    let mb = |v: f64| v / (1 << 20) as f64;
    tracing::info!(
        "shader writes looked for on the CPU per frame: {:.1} MB compared in {:.0} ranges of windows outside buffers ({:.1} MB \
         snapshotted), {:.1} MB in {:.0} ranges of title buffers' copies ({:.1} MB snapshotted), of them {:.1} MB in copies \
         of 1 MiB or more (VICTORY_CARD_BUFFERS_MIN_MB=1 would have the card look there)",
        mb(c[2]),
        c[5],
        mb(c[0]),
        mb(c[3]),
        c[6],
        mb(c[1]),
        mb(c[4]),
    );
}

#[cfg(test)]
mod same_tests {
    use super::{SAME_BLOCK, SAME_CHUNK, SAME_GAP, move_same, push_run, write_differing};

    /// A pseudo-random byte stream (no crate for it here).
    fn bytes(n: usize, seed: u64) -> Vec<u8> {
        let mut x = seed | 1;
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                x as u8
            })
            .collect()
    }

    #[test]
    fn only_differing_blocks_are_written_and_the_copy_ends_equal() {
        // Over several chunks, not a whole number of blocks.
        let len = 2 * SAME_CHUNK + 3 * SAME_BLOCK + 123;
        let title = bytes(len, 7);
        let mut host = title.clone();
        // Changes: a byte in block 3, blocks 10-11, a byte in block 13
        // (within the gap of 11: one run), one far away across a chunk
        // boundary, and the last (short) block.
        let changed = [
            3 * SAME_BLOCK + 5,
            10 * SAME_BLOCK,
            12 * SAME_BLOCK - 1,
            13 * SAME_BLOCK + 9,
            SAME_CHUNK - 1,
            SAME_CHUNK,
            len - 1,
        ];
        for &at in &changed {
            host[at] ^= 0x5a;
        }
        let mut runs = Vec::new();
        let written = move_same(host.as_mut_ptr() as usize, title.as_ptr() as usize, len, &mut runs);
        assert_eq!(host, title, "the copy holds title memory");
        assert_eq!(written, runs.iter().map(|r| r.1 - r.0).sum::<usize>());
        // Every change is in a run, and every run holds a change at its
        // first and its last block.
        for &at in &changed {
            assert!(runs.iter().any(|&(a, b)| a <= at && at < b), "change at {at} written");
        }
        for &(a, b) in &runs {
            assert!(changed.iter().any(|&c| c >= a && c < a + SAME_BLOCK), "run {a}..{b} starts at a change");
            assert!(changed.iter().any(|&c| c < b && c + SAME_BLOCK >= b), "run {a}..{b} ends at a change");
        }
        // In order and apart.
        assert!(runs.windows(2).all(|w| w[0].1 < w[1].0));
        // Far less than the whole was written.
        assert!(written <= 12 * SAME_BLOCK + SAME_GAP, "{written} bytes written");
    }

    #[test]
    fn the_same_bytes_write_nothing() {
        let title = bytes(SAME_CHUNK + 5000, 3);
        let mut host = title.clone();
        let mut runs = Vec::new();
        assert_eq!(move_same(host.as_mut_ptr() as usize, title.as_ptr() as usize, title.len(), &mut runs), 0);
        assert!(runs.is_empty());
        assert_eq!(host, title);
    }

    #[test]
    fn a_whole_new_range_is_one_run() {
        let title = bytes(3 * SAME_CHUNK / 2, 11);
        let mut host = vec![0u8; title.len()];
        let mut runs = Vec::new();
        let written = move_same(host.as_mut_ptr() as usize, title.as_ptr() as usize, title.len(), &mut runs);
        assert_eq!(host, title);
        assert_eq!(written, title.len());
        assert_eq!(runs, vec![(0, title.len())], "chunks join where they touch");
    }

    #[test]
    fn runs_closer_than_the_gap_are_one() {
        let n = 20 * SAME_BLOCK;
        let title = bytes(n, 5);
        let mut host = title.clone();
        host[0] ^= 1;
        host[SAME_BLOCK + SAME_GAP] ^= 1; // within the gap of block 0's run
        host[n - 1] ^= 1; // far
        let mut seen = Vec::new();
        let written = write_differing(&mut host, &title, |a, b| seen.push((a, b)));
        assert_eq!(host, title);
        assert_eq!(seen, vec![(0, 2 * SAME_BLOCK + SAME_GAP), (n - SAME_BLOCK, n)]);
        assert_eq!(written, 2 * SAME_BLOCK + SAME_GAP + SAME_BLOCK);
    }

    #[test]
    fn pushed_runs_join_only_where_they_touch() {
        let mut runs = Vec::new();
        push_run(&mut runs, 0, 10);
        push_run(&mut runs, 10, 20);
        push_run(&mut runs, 25, 30);
        push_run(&mut runs, 28, 40);
        assert_eq!(runs, vec![(0, 20), (25, 40)]);
    }
}

#[cfg(test)]
mod wide_tests {
    use super::{Job, PIECE, WIDE_JOB, copy_jobs, find_pieces, wide, wide_textures};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Every job runs once, however many there are, and `wide` returns
    /// only when all have.
    #[test]
    fn all_jobs_run_before_wide_returns() {
        for n in [0usize, 1, 2, 7, 300] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
    }

    /// The texture checks' pool, as [`wide`]: every job once, all before
    /// it returns, also while the copies' pool is busy on another thread.
    #[test]
    fn texture_jobs_run_on_their_own_pool() {
        let busy = std::thread::spawn(|| {
            let jobs: Vec<Job> = (0..64).map(|_| Box::new(|| std::thread::yield_now()) as Job).collect();
            wide(jobs);
        });
        for n in [0usize, 1, 2, 9, 200] {
            let ran = Arc::new(AtomicUsize::new(0));
            let jobs: Vec<Job> = (0..n)
                .map(|i| {
                    let ran = ran.clone();
                    Box::new(move || {
                        ran.fetch_add(i + 1, Ordering::Relaxed);
                    }) as Job
                })
                .collect();
            wide_textures(jobs);
            assert_eq!(ran.load(Ordering::Relaxed), n * (n + 1) / 2, "{n} jobs");
        }
        busy.join().unwrap();
    }

    /// Copies cut into jobs copy every byte once: long ones, short ones
    /// and empty ones.
    #[test]
    fn copies_cut_into_jobs_copy_the_same_bytes() {
        let len = 3 * WIDE_JOB + 12345;
        let src: Vec<u8> = (0..len).map(|i| (i * 31 + i / 977) as u8).collect();
        let mut dst = vec![0u8; len];
        let (s, d) = (src.as_ptr() as usize, dst.as_mut_ptr() as usize);
        // Ranges of the buffer, each to the same offsets of the other.
        let cuts = [0, 1, 1, 4096, 70_000, WIDE_JOB + 3, 2 * WIDE_JOB + 3, 2 * WIDE_JOB + 9, len];
        let copies: Vec<(usize, usize, usize)> = cuts.windows(2).map(|w| (s + w[0], d + w[0], w[1] - w[0])).collect();
        let jobs = copy_jobs(copies);
        assert!(jobs.len() >= 3);
        wide(jobs);
        assert!(src == dst);
    }

    /// The pieces found on several threads are the ones a walk piece by
    /// piece finds, for copies that start anywhere in a 64 KiB piece of
    /// title memory and ranges that start and end anywhere.
    #[test]
    fn pieces_found_wide_are_those_found_one_by_one() {
        let len = 5 * WIDE_JOB / 2 + 777;
        let old: Vec<u8> = (0..len).map(|i| (i * 7 + i / 513) as u8).collect();
        let mut now = old.clone();
        // Changes: single bytes at piece edges, a run across two pieces,
        // the last byte, and nothing in between.
        for at in [0usize, 1, PIECE - 1, PIECE, 3 * PIECE + 5, WIDE_JOB - 1, WIDE_JOB, WIDE_JOB + PIECE / 2, 2 * WIDE_JOB + 9, len - 1] {
            now[at] ^= 0x5a;
        }
        for byte in &mut now[9 * PIECE - 40..9 * PIECE + 40] {
            *byte ^= 0x01;
        }
        let (data, before) = (now.as_ptr() as usize, old.as_ptr() as usize);
        let mut scans = Vec::new();
        let mut wanted = Vec::new();
        for start in [0usize, 4096, PIECE - 16, 0xd_b790_df00] {
            for (from, to) in [(0, len), (0, 1), (1, 2), (PIECE - 1, PIECE + 1), (3 * PIECE, 3 * PIECE), (12345, len - 1), (WIDE_JOB - 3, 2 * WIDE_JOB + 10)] {
                scans.push((data, before, start, from, to));
                let mut pieces = Vec::new();
                let mut at = from;
                while at < to {
                    let n = (PIECE - (start + at) % PIECE).min(to - at);
                    if now[at..at + n] != old[at..at + n] {
                        pieces.push((at, n));
                    }
                    at += n;
                }
                wanted.push(pieces);
            }
        }
        // A scan of nothing (a range whose snapshot is gone).
        scans.push((0, 0, 0, 0, 0));
        wanted.push(Vec::new());
        assert_eq!(find_pieces(scans), wanted);
    }
}

/// What [`Shadow::copy_back`] does with a piece that differs from the
/// snapshot: `host` is the copy's bytes there, `old` the snapshot's. From
/// the first byte that differs to the last, title memory is read into
/// `title` (scratch, sized to that stretch) by `read` (the stretch's offset
/// in the piece, where to), the GPU's bytes are put in and `host` takes the
/// result. The stretch's offset in the piece and its length if that changed
/// title memory; none if nothing differs or title memory is not there.
/// `runs` takes the runs of the stretch (offsets into `title`, lengths)
/// whose bytes are the GPU's and differ from the title's as read: only
/// those are to be written, so a title write that lands between the read
/// and the write elsewhere in the stretch stays (upstream VIctory #2009).
/// `mask` is scratch.
fn back_piece(
    host: &mut [u8],
    old: &[u8],
    title: &mut Vec<u8>,
    mask: &mut Vec<u8>,
    runs: &mut Vec<(usize, usize)>,
    read: impl FnOnce(usize, &mut [u8]) -> bool,
) -> Option<(usize, usize)> {
    runs.clear();
    // From the first byte that differs to the last: title memory is read
    // and written there only (a shader that sets a few words of a piece
    // cost a read, a walk and a write of all 64 KiB of it).
    let first = first_difference(host, old)?;
    let last = last_difference(host, old).unwrap_or(first);
    let n = last + 1 - first;
    let (host, old) = (&mut host[first..=last], &old[first..=last]);
    title.clear();
    title.resize(n, 0);
    if !read(first, title) {
        return None;
    }
    // Title memory with the GPU's bytes put in; the copy takes the title's
    // bytes in turn.
    // Without a branch, so the loop runs many bytes at a time: the GPU's
    // byte where the copy's byte is not what it was before the submission
    // and the title left its byte as it was, else the title's. (With the
    // two searches above this was 44 ms a frame for 970 pieces.)
    // Where the shader and the title both changed a byte while the GPU ran,
    // the title's stays: the shader computed its byte from the copy's older
    // data (one cleared a bit of a record the title had already replaced,
    // and copying it back broke the title's object: an access violation
    // through 0xffffffff00000000; PalmVIDev e0b70476, VIctory f39c6c2).
    mask.clear();
    mask.resize(n, 0);
    let mut differs = 0u8;
    for (((t, h), o), m) in title.iter_mut().zip(host.iter_mut()).zip(old).zip(mask.iter_mut()) {
        let gpu = if (*h != *o) & (*t == *o) { 0xffu8 } else { 0 };
        let merged = (*t & !gpu) | (*h & gpu);
        *m = merged ^ *t;
        differs |= *m;
        *t = merged;
        *h = merged;
    }
    if differs == 0 {
        return None;
    }
    let mut i = 0;
    while i < n {
        while i < n && mask[i] == 0 {
            i += 1;
        }
        let start = i;
        while i < n && mask[i] != 0 {
            i += 1;
        }
        if i > start {
            runs.push((start, i - start));
        }
    }
    Some((first, n))
}

/// Since the last frame-rate line, of [`Shadow::merge_under_writes`]: pages
/// compared, changed runs put in, and their bytes.
static MERGED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`MERGED`], taken.
pub(crate) fn merged_stats() -> [u64; 3] {
    [0, 1, 2].map(|i| MERGED[i].swap(0, Ordering::Relaxed))
}

/// `runs` (in order) with those a gap of `gap` bytes or fewer apart made
/// one run, the gap's bytes included.
fn joined_over(runs: Vec<(usize, usize)>, gap: usize) -> Vec<(usize, usize)> {
    if gap == 0 || runs.len() < 2 {
        return runs;
    }
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (at, n) in runs {
        match out.last_mut() {
            Some(last) if at - (last.0 + last.1) <= gap => last.1 = at + n - last.0,
            _ => out.push((at, n)),
        }
    }
    out
}

/// The gap, in bytes, over which the title bytes put into a copy in the
/// card's memory under flights' shader writes are joined into one run
/// (`VICTORY_MERGE_GAP`, 0 to 64; 0 by default: only the bytes the title
/// changed go up, as before).
///
/// Run 219: the title rewrites some hundred pages of its big buffer a
/// frame while flights that may write that buffer are in the air, and
/// byte by byte the new content differs from the old in 38,000 runs a
/// frame (equal bytes in between, mostly zeros), far over the 512 runs a
/// submission may send, so the flights were settled twice a frame, 50 ms
/// of the submitting thread. Joined over small gaps the same pages are a
/// few runs.
///
/// What it gives up: a gap byte is equal in title memory and on the CPU
/// side of the copy, and goes up with its run. If a flight in the air
/// wrote exactly that byte on the card (a shader store within `gap` bytes
/// between two bytes the title changed in the same frame), the store is
/// overwritten with the old value, and no compare finds it. Exact (0) is
/// the default for that reason.
fn merge_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| std::env::var("VICTORY_MERGE_GAP").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(0, |n| n.min(64)))
}

/// The runs of bytes where `now` differs from `was` (of one length), as
/// (offset, bytes) in order: every differing byte is in a run and no equal
/// byte is.
fn changed_runs(now: &[u8], was: &[u8]) -> Vec<(usize, usize)> {
    let n = now.len().min(was.len());
    let mut runs = Vec::new();
    let mut i = 0;
    while i < n {
        // Equal bytes, eight at a time while they last.
        while i + 8 <= n && now[i..i + 8] == was[i..i + 8] {
            i += 8;
        }
        while i < n && now[i] == was[i] {
            i += 1;
        }
        if i >= n {
            break;
        }
        let first = i;
        while i < n && now[i] != was[i] {
            i += 1;
        }
        runs.push((first, i - first));
    }
    runs
}

#[cfg(test)]
mod merge_tests {
    use super::{PIECE, back_piece, changed_runs};

    /// A piece put back takes the GPU's bytes where the copy changed and
    /// the title's everywhere else, reads and writes title memory only from
    /// the first change to the last, and its scratch holds that stretch and
    /// no more: not the 64 KiB per range it once was.
    #[test]
    fn a_piece_put_back_reads_only_its_stretch() {
        let old: Vec<u8> = (0..PIECE).map(|i| (i * 7 + i / 251) as u8).collect();
        // The title wrote 7 and 40,000 since the snapshot; the GPU wrote 5,
        // 9 and 40,000 (the GPU's byte wins where both wrote).
        let mut title_memory = old.clone();
        title_memory[7] ^= 0x11;
        title_memory[40_000] ^= 0x22;
        let mut host = old.clone();
        for at in [5, 9] {
            host[at] ^= 0x33;
        }
        let gpu = host.clone();
        let mut scratch = Vec::new();
        let (mut mask, mut runs) = (Vec::new(), Vec::new());
        let mut asked = None;
        let read = |first: usize, into: &mut [u8]| {
            asked = Some((first, into.len()));
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((5, 5)));
        assert_eq!(asked, Some((5, 5)));
        let mut wanted = title_memory[5..10].to_vec();
        wanted[0] = gpu[5];
        wanted[4] = gpu[9];
        assert_eq!(scratch, wanted);
        assert_eq!(host[5..10], wanted);
        // Only the GPU's two bytes are written back (upstream #2009).
        assert_eq!(runs, [(0, 1), (4, 1)]);
        // The copy outside the stretch is left as it was.
        assert!(host[..5] == gpu[..5] && host[10..] == gpu[10..]);
        assert!(scratch.capacity() < PIECE / 2, "scratch of {} bytes for a stretch of 5", scratch.capacity());

        // Nothing differs: title memory is not read.
        let mut same = old.clone();
        assert_eq!(back_piece(&mut same, &old, &mut scratch, &mut mask, &mut runs, |_, _| unreachable!()), None);
        // Title memory not there: nothing to write, the copy unchanged.
        let mut host = gpu.clone();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, _| false), None);
        assert!(host == gpu);
        // The GPU wrote what the title holds already: nothing to write.
        let mut title_memory = old.clone();
        title_memory[5] = gpu[5];
        title_memory[9] = gpu[9];
        let mut host = gpu.clone();
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), None);
        // A whole piece changed: the scratch grows to it.
        let mut host: Vec<u8> = old.iter().map(|b| !b).collect();
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, |_, into| into.len() == PIECE), Some((0, PIECE)));
        assert_eq!(scratch.len(), PIECE);
    }

    /// A byte the shader and the title both changed while the GPU ran keeps
    /// the title's (the shader computed it from the copy's older data); a
    /// byte only the shader changed goes back, and only such bytes are in
    /// the runs to write (PalmVIDev e0b70476, upstream #2009).
    #[test]
    fn a_byte_both_changed_keeps_the_titles_and_only_gpu_bytes_are_written() {
        let old = vec![1u8, 2, 3, 4, 5, 6];
        let mut host = old.clone();
        host[1] = 20; // the GPU alone
        host[3] = 40; // both
        host[4] = 50; // the GPU alone
        let mut title_memory = old.clone();
        title_memory[3] = 44;
        title_memory[5] = 66; // the title alone
        let (mut scratch, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
        let read = |first: usize, into: &mut [u8]| {
            into.copy_from_slice(&title_memory[first..first + into.len()]);
            true
        };
        assert_eq!(back_piece(&mut host, &old, &mut scratch, &mut mask, &mut runs, read), Some((1, 4)));
        assert_eq!(scratch, [20, 3, 44, 50], "the title's byte where both changed");
        assert_eq!(host[1..5], [20, 3, 44, 50], "the copy takes the result");
        assert_eq!(runs, [(0, 1), (3, 1)], "the GPU's bytes only");
    }

    /// The runs hold exactly the bytes that differ: putting them into the
    /// old bytes gives the new ones, and no run holds an equal byte.
    #[test]
    fn changed_runs_are_exactly_the_bytes_that_differ() {
        let was: Vec<u8> = (0..300u32).map(|i| (i * 13 + 5) as u8).collect();
        for changes in [vec![], vec![0], vec![299], vec![7, 8, 9], vec![7, 9], vec![15, 16, 17, 23, 24, 64, 65, 66, 67, 68, 69, 70, 71, 72, 298, 299]] {
            let mut now = was.clone();
            for &at in &changes {
                now[at] ^= 0x41;
            }
            let runs = changed_runs(&now, &was);
            let mut put = was.clone();
            let mut held = Vec::new();
            for &(at, n) in &runs {
                assert!(n > 0);
                put[at..at + n].copy_from_slice(&now[at..at + n]);
                held.extend(at..at + n);
            }
            assert_eq!(put, now, "{changes:?}");
            assert_eq!(held, changes, "{changes:?}");
            // Runs do not touch: two that did would be one.
            assert!(runs.windows(2).all(|w| w[0].0 + w[0].1 < w[1].0), "{changes:?}");
        }
        assert!(changed_runs(&[], &[]).is_empty());
    }
}

/// Host copies alive (also those only a list, a descriptor pass or a
/// snapshot still holds) and the bytes of title memory they copy.
static LIVE: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

impl Drop for Shadow {
    fn drop(&mut self) {
        if self.place.is_some() {
            // Never counted with the copies (`Shadow::placed`).
            crate::inplace::gone(self.window, self.len, self.pooled);
            return;
        }
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(self.len, Ordering::Relaxed);
        if let Some(bytes) = self.bar {
            BAR_COPIES[0].fetch_sub(1, Ordering::Relaxed);
            BAR_COPIES[1].fetch_sub(bytes, Ordering::Relaxed);
        }
        if let Some(snapshot) = self.before.get_mut().unwrap_or_else(|e| e.into_inner()).as_ref() {
            snapshot_gone(snapshot.capacity());
        }
        if self.kept.get_mut().unwrap_or_else(|e| e.into_inner()).1 {
            KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
        }
        let window = self.window;
        let (start, len) = (self.start as usize, self.len as usize);
        if let Some(card) = &mut self.card {
            if card.staging.is_some() {
                PC_SIDES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            let held: u64 = card
                .written
                .iter()
                .enumerate()
                .filter(|(_, w)| w.load(Ordering::Relaxed))
                .map(|(piece, _)| {
                    let (lo, hi) = piece_at(start, len, piece);
                    hi.saturating_sub(lo) as u64
                })
                .sum();
            PC_SIDES[1].fetch_sub(held, Ordering::Relaxed);
            if window {
                WINDOW_CARD_BYTES.fetch_sub(card.size, Ordering::Relaxed);
            } else {
                CARD_BYTES[0].fetch_sub(card.size, Ordering::Relaxed);
            }
            if card.before.get_mut().unwrap_or_else(|e| e.into_inner()).is_some() {
                CARD_BYTES[1].fetch_sub(card_hash_bytes(card.size), Ordering::Relaxed);
            }
        }
    }
}

/// Whether a copy's snapshot is kept from one submission to the next and
/// only the pieces that changed in between are taken again
/// (`VICTORY_KEEP_SNAPSHOTS=1`; off by default).
///
/// A snapshot is the copy's bytes before a submission, to find what its
/// shaders wrote. Taken anew for every submission it is 1,050 MB of copying
/// a frame (72-86 ms in gameplay, runs 61 and 62) for buffers of which
/// 100 MB changed. Kept, a piece of it is taken again only after something
/// changed the copy there: a refresh from title memory ([`Shadow::copy`]),
/// shader writes found after a submission, or the bytes put back then
/// ([`Shadow::copy_back`]).
///
/// What it assumes, and why it is not on by default: that nothing else
/// writes a copy. A shader that writes a buffer through a view we did not
/// declare as written (we declare the unordered-access views of shaders
/// that can store) changes the copy unseen; with a fresh snapshot that
/// write is simply never put back, with a kept one it would be taken for a
/// write of a later submission and put back then, over whatever the title
/// wrote there since.
fn keeps() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_some_and(|v| v == "1"))
}

/// Bytes of the snapshots kept, and the most that may be.
static KEPT_BYTES: AtomicU64 = AtomicU64::new(0);
/// `VICTORY_KEEP_SNAPSHOTS_MB`, 256 unless set (it was 768: run 80, with
/// kept snapshots beside copies in the card's memory, ran the PC out of
/// memory in 27 s; kept snapshots are the PC's memory, all of it resident).
fn kept_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_KEEP_SNAPSHOTS_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(256) << 20
    })
}

/// The 64 KiB pieces, cut at multiples of 64 KiB of title memory, that
/// hold offsets `[from, to)` of a copy of memory from `start` on
/// (`from < to`); piece 0 holds offset 0.
fn pieces_of(start: usize, from: usize, to: usize) -> std::ops::Range<usize> {
    let first = start / PIECE;
    (start + from) / PIECE - first..(start + to - 1) / PIECE - first + 1
}

/// The offsets that piece `piece` holds of a copy of `len` bytes of memory
/// from `start` on.
fn piece_at(start: usize, len: usize, piece: usize) -> (usize, usize) {
    let first = start / PIECE;
    let lo = ((first + piece) * PIECE).saturating_sub(start);
    let hi = ((first + piece + 1) * PIECE - start).min(len);
    (lo, hi)
}

#[cfg(test)]
mod piece_tests {
    use super::{PIECE, piece_at, pieces_of};

    /// The pieces cover a copy without gaps, each cut where the compare
    /// cuts (at multiples of 64 KiB of title memory).
    #[test]
    fn pieces_cover_the_copy_as_the_compare_cuts_it() {
        for start in [0usize, 1, 4096, PIECE - 1, PIECE, PIECE + 77, 5 * PIECE + 12345] {
            for len in [1usize, 100, PIECE - 1, PIECE, PIECE + 1, 3 * PIECE + 500] {
                let count = pieces_of(start, 0, len).end;
                let mut at = 0;
                for piece in 0..count {
                    let (lo, hi) = piece_at(start, len, piece);
                    assert_eq!(lo, at, "{start} {len} {piece}");
                    // As `Shadow::differing` cuts: to the next multiple.
                    let n = (PIECE - (start + at) % PIECE).min(len - at);
                    assert_eq!(hi, at + n, "{start} {len} {piece}");
                    assert_eq!(pieces_of(start, lo, hi), piece..piece + 1);
                    assert_eq!(pieces_of(start, hi - 1, hi), piece..piece + 1);
                    at = hi;
                }
                assert_eq!(at, len, "{start} {len}");
                assert_eq!(pieces_of(start, 0, len), 0..count);
            }
        }
    }
}

/// The pages of `[a, e)` (title addresses, `a < e`) that a copy whose
/// first page is `base` must take from title memory before it holds the
/// title's bytes: those it never took (`copied` 0), whether or not the
/// title wrote them since tracking began, and those `written` (as
/// `crate::watch::writes` gives it, by page in order) shows written after
/// the copy took them. As [`Shadow::copy_stale`] decides it.
///
/// [`Shadow::stale_under_writes`] and [`Shadow::merge_under_writes`] looked
/// at `written` alone, so a page of a range shaders may be writing that the
/// copy never took, and the title had not written lately, was neither
/// merged nor a reason to settle; the refresh leaves such ranges alone, and
/// shaders read zeros there (fxceless/victory-fork#206: missing geometry
/// with `VICTORY_LATE_WAIT`/`VICTORY_RELAY`).
fn stale_pages_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> Vec<usize> {
    (a / PAGE..e.div_ceil(PAGE))
        .filter(|&page| page >= base)
        .filter(|&page| match copied.get(page - base) {
            None => false,
            Some(0) => true,
            Some(&at) => written
                .binary_search_by_key(&page, |&(p, _)| p)
                .is_ok_and(|i| written[i].1 > at),
        })
        .collect()
}

/// Whether [`stale_pages_in`] would name any page, looking only until it
/// finds the first.
fn any_stale_page_in(base: usize, copied: &[u64], a: usize, e: usize, written: &[(usize, u64)]) -> bool {
    (a / PAGE..e.div_ceil(PAGE)).filter(|&page| page >= base).any(|page| match copied.get(page - base) {
        None => false,
        Some(0) => true,
        Some(&at) => written.binary_search_by_key(&page, |&(p, _)| p).is_ok_and(|i| written[i].1 > at),
    })
}

#[cfg(test)]
mod stale_tests {
    use super::*;

    #[test]
    fn the_first_stale_page_answers_as_the_list_does() {
        let base = 0x100;
        let at = |page: usize| page * PAGE;
        // Pages 0x100..0x108: never taken, taken at 5, taken at 9.
        let copied = [5, 0, 5, 9, 9, 5, 5, 9];
        for written in [vec![], vec![(0x100, 4)], vec![(0x103, 10)], vec![(0x105, 6), (0x107, 9)], vec![(0x106, 5)]] {
            for a in 0x0fe..0x109 {
                for e in a + 1..0x10a {
                    let (from, to) = (at(a) + 8, at(e) - 8);
                    if from >= to {
                        continue;
                    }
                    assert_eq!(
                        any_stale_page_in(base, &copied, from, to, &written),
                        !stale_pages_in(base, &copied, from, to, &written).is_empty(),
                        "{a:#x}..{e:#x} {written:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn pages_never_taken_are_stale_written_or_not() {
        let base = 0x100;
        let copied = [5, 0, 5, 5];
        let at = |page: usize| page * PAGE;
        // Page 0x101 was never taken; page 0x102 was written after it was
        // taken, page 0x103 before.
        let written = [(0x102, 6), (0x103, 4)];
        assert_eq!(stale_pages_in(base, &copied, at(0x100), at(0x104), &written), [0x101, 0x102]);
        // Part of a page counts as the page; pages outside the copy do not.
        assert_eq!(stale_pages_in(base, &copied, at(0x101) + 8, at(0x101) + 16, &[]), [0x101]);
        assert!(stale_pages_in(base, &copied, at(0x104), at(0x106), &[(0x104, 9)]).is_empty());
    }
}

/// Copies of windows outside buffers let go since the start, and their
/// bytes.
static LET_GO: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// For the memory line: the windows let go since the start and their
/// bytes, and of the bytes of the windows alive, how many lie between the
/// lowest and the highest address ever asked for in each.
#[cfg(feature = "diagnostics")]
pub(crate) fn windows() -> [u64; 3] {
    let asked: u64 = {
        let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        loose
            .values()
            .map(|s| {
                let low = s.asked[0].load(Ordering::Relaxed);
                let high = s.asked[1].load(Ordering::Relaxed);
                high.saturating_sub(low)
            })
            .sum()
    };
    [LET_GO[0].load(Ordering::Relaxed), LET_GO[1].load(Ordering::Relaxed), asked]
}

/// For the memory line, of the windows alive: the bytes of their pages
/// ever copied from title memory (the rest of each window is memory the
/// host zeroed and nothing read: what a window made no bigger than what is
/// read would not hold); the windows a descriptor slot shows and their
/// bytes (they stay until the title writes the slot again); the windows a
/// list or a pass over the heap holds and their bytes; the windows asked
/// for without a length (root views, which keep the 4 MiB grid) and their
/// bytes; and the window grid in bytes ([`window`]).
#[cfg(feature = "diagnostics")]
pub(crate) fn window_use() -> [u64; 8] {
    let mut out = [0u64; 8];
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    for s in loose.values() {
        let copied = s.copied.lock().unwrap_or_else(|e| e.into_inner());
        let pages = copied.iter().filter(|&&at| at != 0).count() as u64;
        out[0] += (pages * PAGE as u64).min(s.len);
        if crate::host::references(&s.buffer) > 1 {
            out[1] += 1;
            out[2] += s.len;
        }
        if Arc::strong_count(s) > 1 {
            out[3] += 1;
            out[4] += s.len;
        }
        if s.unsized_asks.load(Ordering::Relaxed) > 0 {
            out[5] += 1;
            out[6] += s.len;
        }
    }
    out[7] = window();
    out
}

/// Lets go of the copies of windows outside buffers that nothing names any
/// more. They were never freed, and the world's streaming keeps asking for
/// new ones: 648 windows (2.8 GB) when the cutscene began, 974 (4.3 GB)
/// eight minutes later (run 24). A window goes when
/// - only the table of windows holds its copy: no list not yet reset reads
///   or writes it, no pass over the heap keeps its range, no submission is
///   comparing it;
/// - only the copy holds its host buffer: no descriptor slot views it (a
///   slot keeps the buffer it views, `HostHeap::kept`), so no draw can read
///   it through a slot made earlier;
/// - nothing asked for it for 10 s (2 s under the commit floor, `crate::memline::idle_ms`) and `IDLE_FRAMES` frames.
///
/// The next read of that memory makes a new copy and copies the window
/// again. Called before a submission starts, by the thread that submits.
pub(crate) fn let_go_idle() {
    const IDLE_FRAMES: u64 = 4;
    let gone: Vec<Arc<Shadow>> = {
        // Held from the look to the removal: `loose` hands a copy out
        // under the same lock, so none is handed out in between.
        let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        if map.is_empty() {
            return;
        }
        // 10 s, shorter when free RAM is low (`crate::host::ram_idle_ms`)
        // and under the commit floor (`crate::memline::idle_ms`).
        let idle_ms = crate::memline::idle_ms(crate::host::ram_idle_ms(crate::host::available_physical_ram()));
        let idle: Vec<(u64, u64)> = map
            .iter()
            .filter(|(_, s)| {
                Arc::strong_count(s) == 1
                    && crate::host::long_ago(s.used.load(Ordering::Relaxed), idle_ms, IDLE_FRAMES)
                    && crate::host::references(&s.buffer) == 1
            })
            .map(|(key, _)| *key)
            .collect();
        idle.into_iter().filter_map(|key| map.remove(&key)).collect()
    };
    if !gone.is_empty() {
        let (went, bytes) = (gone.len(), gone.iter().map(|s| s.len).sum::<u64>());
        crate::host::trail(|| {
            format!("sweep: {went} host copies of windows of title memory let go ({} KiB), held by no list and no slot", bytes >> 10)
        });
    }
    for shadow in gone {
        LET_GO[0].fetch_add(1, Ordering::Relaxed);
        LET_GO[1].fetch_add(shadow.len, Ordering::Relaxed);
        // Nothing runs now and nothing names it: freed at once unless a
        // submission is running after all.
        crate::host::retire(shadow);
    }
}

/// For the memory line: host copies alive and their bytes, of them the
/// windows of memory outside buffers and their bytes, and the snapshot
/// buffers kept for reuse and the bytes they reserve.
pub(crate) fn memory() -> [u64; 6] {
    let (windows, window_bytes) = {
        let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        // Each copy once: a window may stand under several keys.
        let mut seen = std::collections::HashSet::new();
        let bytes: u64 = loose
            .values()
            .filter(|s| seen.insert(Arc::as_ptr(s) as usize))
            .map(|s| s.len)
            .sum();
        (seen.len() as u64, bytes)
    };
    let (kept, kept_bytes) = {
        let pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
        (pool.len() as u64, pool.iter().map(|v| v.capacity() as u64).sum::<u64>())
    };
    [
        LIVE[0].load(Ordering::Relaxed),
        LIVE[1].load(Ordering::Relaxed),
        windows,
        window_bytes,
        kept,
        kept_bytes,
    ]
}

/// Since the last frame line, with [`crate::watch::fine`]: bytes of the
/// stale 64 KiB pages refreshes found (cut to their ranges: what they
/// copied without the switch), bytes of the runs they took instead, and
/// those runs; with the check (`VICTORY_FINE_PAGES=2`), bytes of parts
/// passed over that were compared with title memory, and bytes that
/// differed (the parts holding them were taken after all).
pub(crate) static FINE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// For the frame line: what [`crate::watch::fine`] saved, and what the
/// askings noted written, per frame (taken).
pub(crate) fn fine_line(frames: f64) -> String {
    let frames = frames.max(1.0);
    let f = FINE.each_ref().map(|a| a.swap(0, Ordering::Relaxed) as f64 / frames);
    let n = crate::watch::NOTED.each_ref().map(|a| a.swap(0, Ordering::Relaxed) as f64 / frames / 1e6);
    let noted = format!(
        "title writes noted {:.1} MB in 4 KiB pages, {:.1} MB as 64 KiB pages, {:.1} MB in guarded 64 KiB blocks ({:.1} MB          in their parts written, VICTORY_FINE_GUARDS {})",
        n[0],
        n[1],
        n[2],
        n[3],
        if crate::watch::fine_guards() { "on" } else { "off" }
    );
    match crate::watch::fine_level() {
        0 => format!("fine pages off ({noted})"),
        level => {
            let mut line = format!(
                "fine pages: {:.1} MB taken of {:.1} MB in stale 64 KiB pages, in {:.0} runs ({noted})",
                f[1] / 1e6,
                f[0] / 1e6,
                f[2]
            );
            if level >= 2 {
                line.push_str(&format!(
                    "; check: {:.1} MB passed over compared, {:.1} KB of them differed and were taken",
                    f[3] / 1e6,
                    f[4] / 1e3
                ));
            }
            line
        }
    }
}

/// With [`crate::watch::fine`]: parts of stale pages this close (bytes)
/// are taken as one run, the parts between them too (they hold what title
/// memory holds, so taking them changes nothing): fewer, longer moves and
/// sends up to the card. `VICTORY_FINE_GAP_KB` (8; 0 joins only parts that
/// touch).
fn fine_gap() -> usize {
    static GAP: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *GAP.get_or_init(|| {
        std::env::var("VICTORY_FINE_GAP_KB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(8) << 10
    })
}

/// What [`subs_after`](crate::watch::subs_after) answers: per (page, the
/// asking after which the copy last took it), the parts written since.
type SubsAfter<'a> = dyn Fn(&[(usize, u64)], &mut Vec<u16>) + 'a;

/// The runs a refresh of `[from, to)` copies, for a copy of title memory
/// `[start, end)` (`span`) whose page notes are `copied` (per 64 KiB page
/// from `start`'s on, the asking after which it was last taken, 0 never),
/// at asking `now`, with `written` the journal's pages of the range (page,
/// asking that last found it written; in page order). Notes the pages
/// taken whole in the range as taken at `now`, and appends the runs (title
/// addresses, in order, apart by more than `gap`) to `runs`.
///
/// A page is stale if never taken or written since. Without `subs` it is
/// copied whole (cut to the range), neighbours as one run: what refreshes
/// always did. With `subs` (the journal's parts, [`crate::watch::fine`])
/// only its parts written since the copy last took it are copied: a part
/// last written at or before that asking was taken then or before and is
/// current; a page never taken is copied whole. Every part of the page
/// is then current, so the note is right as it is without the switch.
/// Runs closer than `gap` are one, with the bytes between (current ones).
///
/// With `skipped` the parts of stale pages in the range that are not taken
/// are appended there (for the check). Returns the bytes of the stale
/// pages in the range and the bytes of the runs.
#[allow(clippy::too_many_arguments)]
fn stale_runs_of(
    (start, end): (usize, usize),
    copied: &mut [u64],
    (from, to): (usize, usize),
    now: u64,
    written: &[(usize, u64)],
    subs: Option<&SubsAfter<'_>>,
    gap: usize,
    runs: &mut Vec<(usize, usize)>,
    mut skipped: Option<&mut Vec<(usize, usize)>>,
) -> [usize; 2] {
    use crate::watch::{ALL_SUBS, SUB, SUBS};
    if from >= to {
        return [0, 0];
    }
    let base = start / PAGE;
    let mut writes = written.iter().peekable();
    // The stale pages, each with the asking after which it was last taken.
    let mut stale: Vec<(usize, u64)> = Vec::new();
    let last = (to - 1) / PAGE;
    for page in from / PAGE..=last {
        while writes.peek().is_some_and(|&&(p, _)| p < page) {
            writes.next();
        }
        let write = match writes.peek() {
            Some(&&(p, w)) if p == page => w,
            _ => 0,
        };
        let at = &mut copied[page - base];
        if *at == 0 || write > *at {
            stale.push((page, *at));
            // A page only partly in the range is copied in part: it stays
            // stale for the rest.
            if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                *at = now;
            }
        }
    }
    let mut parts = Vec::new();
    match subs {
        Some(subs) if stale.iter().any(|s| s.1 != 0) => subs(&stale, &mut parts),
        _ => parts.resize(stale.len(), ALL_SUBS),
    }
    let first_new = runs.len();
    let put = |runs: &mut Vec<(usize, usize)>, a: usize, b: usize, gap: usize| {
        if a >= b {
            return;
        }
        // Not joined with the caller's runs from before.
        let ours = runs.len() > first_new;
        match runs.last_mut() {
            Some(last) if ours && a <= last.1 + gap => last.1 = last.1.max(b),
            _ => runs.push((a, b)),
        }
    };
    let mut stale_bytes = 0;
    for (&(page, _), &mask) in stale.iter().zip(&parts) {
        // Cut to the range: next to it shaders may be writing the copy.
        let (lo, hi) = ((page * PAGE).max(from), ((page + 1) * PAGE).min(to));
        stale_bytes += hi - lo;
        if mask == ALL_SUBS {
            put(runs, lo, hi, if subs.is_some() { gap } else { 0 });
            continue;
        }
        let mut i = 0;
        while i < SUBS {
            let taken = mask >> i & 1 != 0;
            let mut k = i + 1;
            while k < SUBS && (mask >> k & 1 != 0) == taken {
                k += 1;
            }
            let (a, b) = ((page * PAGE + i * SUB).max(lo), (page * PAGE + k * SUB).min(hi));
            if taken {
                put(runs, a, b, gap);
            } else if let Some(skipped) = skipped.as_deref_mut()
                && a < b
            {
                skipped.push((a, b));
            }
            i = k;
        }
    }
    let run_bytes = runs[first_new..].iter().map(|r| r.1 - r.0).sum();
    [stale_bytes, run_bytes]
}

#[cfg(test)]
mod fine_tests {
    //! The refreshes' runs with the journal's 4 KiB parts
    //! (`VICTORY_FINE_PAGES`) against title memory itself: after every
    //! refresh a copy's CPU side and its card side hold what title memory
    //! holds, for random title writes, overlapping copies, and shader writes
    //! on the card merged back into title memory as `Shadow::card_merge`
    //! merges them.
    use super::{PAGE, SubsAfter, stale_runs_of};
    use crate::watch::{ALL_SUBS, SUB, SUBS, subs_after, writes};

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
            (self.next() % n.max(1) as u64) as usize
        }
    }

    /// The runs as refreshes made them before the parts (whole stale
    /// pages, neighbours as one run, cut to the range): the reference.
    fn old_runs(start: usize, end: usize, copied: &mut [u64], from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> Vec<(usize, usize)> {
        let base = start / PAGE;
        let mut runs = Vec::new();
        let mut writes = written.iter().peekable();
        let mut run = None;
        let last = (to - 1) / PAGE;
        for page in from / PAGE..=last {
            let write = match writes.peek() {
                Some(&&(p, w)) if p == page => {
                    writes.next();
                    w
                }
                _ => 0,
            };
            let at = &mut copied[page - base];
            if *at == 0 || write > *at {
                if (page * PAGE).max(start) >= from && ((page + 1) * PAGE).min(end) <= to {
                    *at = now;
                }
                run.get_or_insert(page);
            } else if let Some(first) = run.take() {
                runs.push(((first * PAGE).max(from), (page * PAGE).min(to)));
            }
        }
        if let Some(first) = run {
            runs.push(((first * PAGE).max(from), ((last + 1) * PAGE).min(to)));
        }
        runs
    }

    /// Without the parts the runs and notes are exactly the old ones; with
    /// them the runs lie inside the old ones, hold every part written after
    /// the copy's asking, and the notes are the same.
    #[test]
    fn runs_are_the_old_ones_or_their_written_parts() {
        let mut rng = Rng(0x5851_f42d_4c95_7f2d);
        for case in 0..2000 {
            let base = 0x7_0000_0000usize;
            let start = base + rng.below(3 * PAGE);
            let end = start + 1 + rng.below(9 * PAGE);
            let pages = end.div_ceil(PAGE) - start / PAGE;
            let notes: Vec<u64> = (0..pages).map(|_| [0, 3, 5, 9][rng.below(4)]).collect();
            // The journal: per page an asking and per part an asking, the
            // page's being the latest of its parts' (as `Written::mark`
            // keeps them).
            let mut parts: Vec<[u64; SUBS]> = vec![[0; SUBS]; pages];
            let mut written = Vec::new();
            for (i, subs) in parts.iter_mut().enumerate() {
                if rng.below(3) == 0 {
                    continue;
                }
                for sub in subs.iter_mut() {
                    if rng.below(4) == 0 {
                        *sub = [1, 4, 6, 10, 11][rng.below(5)];
                    }
                }
                let w = subs.iter().copied().max().unwrap_or(0);
                if w > 0 {
                    written.push((start / PAGE + i, w));
                }
            }
            let journal = |asks: &[(usize, u64)], out: &mut Vec<u16>| {
                out.clear();
                out.extend(asks.iter().map(|&(page, since)| {
                    if since == 0 {
                        return ALL_SUBS;
                    }
                    let subs = &parts[page - start / PAGE];
                    (0..SUBS).filter(|&i| subs[i] > since).fold(0u16, |m, i| m | 1 << i)
                }));
            };
            let from = start + rng.below(end - start);
            let to = from + 1 + rng.below(end - from);
            let to = to.min(end);
            let now = 12;
            // As `crate::watch::writes` answers: the pages of the range.
            written.retain(|&(page, _)| page >= from / PAGE && page < to.div_ceil(PAGE));
            let (mut a, mut b) = (notes.clone(), notes.clone());
            let old = old_runs(start, end, &mut a, from, to, now, &written);
            let mut plain = Vec::new();
            stale_runs_of((start, end), &mut b, (from, to), now, &written, None, 0, &mut plain, None);
            assert_eq!(plain, old, "case {case}");
            assert_eq!(a, b, "case {case}");
            for gap in [0usize, 8 << 10] {
                let mut fine = Vec::new();
                let mut skipped = Vec::new();
                let mut c2 = notes.clone();
                let [stale, taken] = stale_runs_of(
                    (start, end),
                    &mut c2,
                    (from, to),
                    now,
                    &written,
                    Some(&journal as &SubsAfter<'_>),
                    gap,
                    &mut fine,
                    Some(&mut skipped),
                );
                assert_eq!(c2, a, "the notes, case {case}");
                assert_eq!(stale, old.iter().map(|r| r.1 - r.0).sum::<usize>(), "case {case}");
                assert_eq!(taken, fine.iter().map(|r| r.1 - r.0).sum::<usize>(), "case {case}");
                // In order, apart by more than the gap, inside the old runs.
                assert!(fine.windows(2).all(|w| w[1].0 > w[0].1 + gap), "case {case}");
                for &(x, y) in &fine {
                    assert!(old.iter().any(|&(p, q)| p <= x && y <= q), "case {case}: {x:#x}..{y:#x} outside");
                }
                // Every byte of a part written after the copy's asking (or
                // of a page never taken) is in a run, and every byte of a
                // stale page in the range is taken or passed over.
                for &(p, q) in &old {
                    for at in (p..q).step_by(97).chain([p, q - 1]) {
                        let page = at / PAGE;
                        let since = notes[page - start / PAGE];
                        let part = (at % PAGE) / SUB;
                        let due = since == 0 || parts[page - start / PAGE][part] > since;
                        let taken = fine.iter().any(|&(x, y)| x <= at && at < y);
                        let passed = skipped.iter().any(|&(x, y)| x <= at && at < y);
                        if due {
                            assert!(taken, "case {case}: written byte {at:#x} not taken");
                        }
                        assert!(taken || passed, "case {case}: byte {at:#x} neither taken nor passed over");
                        assert!(!(passed && due), "case {case}: written byte {at:#x} passed over");
                    }
                }
            }
        }
    }

    /// A host copy as the tests keep it: title memory `[start, end)`, its
    /// CPU side, its card side, and its page notes.
    struct Copy {
        start: usize,
        end: usize,
        cpu: Vec<u8>,
        card: Vec<u8>,
        copied: Vec<u64>,
        fine: bool,
        gap: usize,
        /// Bytes of stale pages, bytes taken.
        bytes: [usize; 2],
    }

    impl Copy {
        fn new(start: usize, end: usize, fine: bool, gap: usize) -> Copy {
            Copy {
                start,
                end,
                cpu: vec![0; end - start],
                card: vec![0; end - start],
                copied: vec![0; end.div_ceil(PAGE) - start / PAGE],
                fine,
                gap,
                bytes: [0; 2],
            }
        }

        /// A refresh of `[from, to)`, as `Shadow::update` does it: the
        /// runs are copied into the CPU side, and each run then goes up to
        /// the card's side (`Shadow::card_changed`, `Host::card_before`).
        /// The parts passed over must hold title memory's bytes already.
        fn refresh(&mut self, from: usize, to: usize) {
            if from >= to {
                return;
            }
            // View memory is asked of the guards, as refreshes ask.
            let (now, written) = writes(from, to, true).expect("write-watched or a view");
            let mut runs = Vec::new();
            let mut skipped = Vec::new();
            let counts = stale_runs_of(
                (self.start, self.end),
                &mut self.copied,
                (from, to),
                now,
                &written,
                if self.fine { Some(&subs_after as &SubsAfter<'_>) } else { None },
                self.gap,
                &mut runs,
                Some(&mut skipped),
            );
            self.bytes[0] += counts[0];
            self.bytes[1] += counts[1];
            for &(a, b) in &skipped {
                assert!(self.cpu[a - self.start..b - self.start] == title(a, b), "a part passed over at {a:#x}..{b:#x} is stale");
            }
            for (a, b) in runs {
                let (x, y) = (a - self.start, b - self.start);
                crate::host::copy_mapped(self.cpu[x..].as_mut_ptr(), a as *const u8, b - a, false);
                self.card[x..y].copy_from_slice(&self.cpu[x..y]);
            }
        }

        /// Both sides hold title memory's bytes in `[from, to)`.
        fn holds_title(&self, from: usize, to: usize, what: &str) {
            let (x, y) = (from - self.start, to - self.start);
            let now = title(from, to);
            if let Some(i) = (0..y - x).find(|&i| self.cpu[x + i] != now[i]) {
                panic!("{what}: the CPU side differs from title memory at {:#x}", from + i);
            }
            if let Some(i) = (0..y - x).find(|&i| self.card[x + i] != now[i]) {
                panic!("{what}: the card's side differs from title memory at {:#x}", from + i);
            }
        }
    }

    /// Title memory `[a, b)`, read as a copy reads it.
    fn title(a: usize, b: usize) -> Vec<u8> {
        let mut out = vec![0u8; b - a];
        assert!(crate::host::read_into(a as *const u8, &mut out));
        out
    }

    /// Random title writes into write-watched memory, three copies over
    /// it (two of the same range, one with the parts and one without, and
    /// one overlapping both), and in most rounds a range the card's shaders
    /// write in the first: the refresh leaves it alone, the card writes in
    /// it while the title writes too, and the merge puts the card's bytes
    /// into title memory (through the kernel, as `write_mapped`) and into
    /// both sides. After each refresh and merge every copy holds what title
    /// memory holds, on both sides; with the parts far fewer bytes moved.
    #[test]
    fn copies_hold_title_memory_after_every_refresh_and_merge() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x20_0000;
        // SAFETY: a fresh write-watched allocation, kept to the end.
        let base = unsafe { VirtualAlloc(std::ptr::null(), SIZE, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as usize;
        assert_ne!(base, 0);
        copies_hold_title_memory(base, 0x2545_f491_4f6c_dd1d);
    }

    /// The same over view memory (a pagefile section's view, as xmem maps
    /// physical pages), which the guards watch, with their 4 KiB parts
    /// (`crate::watch::fine_guards`, on in tests): the title's writes fault
    /// and open a part (a second one the whole block), the merges open the
    /// parts they write (`crate::watch::writing_exact`).
    #[test]
    fn view_copies_hold_title_memory_after_every_refresh_and_merge() {
        use windows_sys::Win32::System::Memory::{CreateFileMappingW, FILE_MAP_WRITE, MapViewOfFile, PAGE_READWRITE};
        // SAFETY: a pagefile section mapped read-write, kept to the end.
        let base = unsafe {
            let section = CreateFileMappingW(-1isize as _, std::ptr::null(), PAGE_READWRITE, 0, SIZE as u32, std::ptr::null());
            assert!(!section.is_null());
            MapViewOfFile(section, FILE_MAP_WRITE, 0, 0, SIZE).Value as usize
        };
        assert_ne!(base, 0);
        copies_hold_title_memory(base, 0x9e37_79b9_7f4a_7c15);
    }

    /// Title memory of the tests above.
    const SIZE: usize = 12 * PAGE;

    /// [`copies_hold_title_memory_after_every_refresh_and_merge`] over
    /// `SIZE` bytes of title memory at `base`.
    fn copies_hold_title_memory(base: usize, seed: u64) {
        let mut rng = Rng(seed);
        let poke = |rng: &mut Rng, at: usize, len: usize| {
            for i in 0..len {
                // SAFETY: inside our allocation (callers keep to it).
                unsafe { ((at + i) as *mut u8).write_volatile(rng.next() as u8) };
            }
        };
        // Title memory starts with something in it.
        poke(&mut rng, base, SIZE);
        let (a0, a1) = (base + 0x1234, base + 7 * PAGE + 77);
        let (b0, b1) = (base + 3 * PAGE + 5, base + 11 * PAGE + 4000);
        let mut fine = Copy::new(a0, a1, true, 8 << 10);
        let mut coarse = Copy::new(a0, a1, false, 0);
        let mut other = Copy::new(b0, b1, true, 0);
        for round in 0..400 {
            // The title writes: a few bytes, a run across parts or pages,
            // now and then a long one, anywhere in its memory.
            for _ in 0..1 + rng.below(6) {
                let len = match rng.below(10) {
                    0 => 1 + rng.below(3 * PAGE),
                    1..=3 => 1 + rng.below(2 * SUB),
                    _ => 1 + rng.below(16),
                };
                let at = base + rng.below(SIZE - len);
                poke(&mut rng, at, len);
            }
            // A range the card's shaders write, in three rounds of four.
            let declared = (rng.below(4) != 0).then(|| {
                let d0 = a0 + rng.below(a1 - a0 - 1);
                (d0, (d0 + 1 + rng.below(2 * PAGE)).min(a1))
            });
            match declared {
                Some((d0, d1)) => {
                    fine.refresh(a0, d0);
                    fine.refresh(d1, a1);
                    fine.holds_title(a0, d0, &format!("round {round}, before the range"));
                    fine.holds_title(d1, a1, &format!("round {round}, after the range"));
                }
                None => {
                    fine.refresh(a0, a1);
                    fine.holds_title(a0, a1, &format!("round {round}"));
                }
            }
            if let Some((d0, d1)) = declared {
                // The card writes some of the range; the title writes some
                // of it meanwhile.
                let (x, y) = (d0 - a0, d1 - a0);
                for _ in 0..rng.below(4) {
                    let at = x + rng.below(y - x);
                    let n = (1 + rng.below(3 * SUB)).min(y - at);
                    for byte in &mut fine.card[at..at + n] {
                        *byte = rng.next() as u8;
                    }
                }
                if rng.below(2) == 0 {
                    let at = d0 + rng.below(d1 - d0);
                    let n = (1 + rng.below(64)).min(d1 - at);
                    poke(&mut rng, at, n);
                }
                // The merge (`Shadow::card_merge`): the card's byte where it
                // is not what the CPU side held, else the title's.
                let now = title(d0, d1);
                let gpu = fine.card[x..y].to_vec();
                let old = &mut fine.cpu[x..y];
                let mut merged = now.clone();
                for i in 0..merged.len() {
                    if gpu[i] != old[i] {
                        merged[i] = gpu[i];
                    }
                }
                if merged != now {
                    crate::watch::writing_exact(d0, merged.len());
                    crate::host::write_mapped(d0 as *mut u8, merged.as_ptr(), merged.len());
                }
                old.copy_from_slice(&merged);
                fine.card[x..y].copy_from_slice(&merged);
                fine.holds_title(a0, a1, &format!("round {round}, after the merge"));
            }
            coarse.refresh(a0, a1);
            coarse.holds_title(a0, a1, &format!("round {round}, without the parts"));
            if round % 2 == 0 {
                other.refresh(b0, b1);
                other.holds_title(b0, b1, &format!("round {round}, the overlapping copy"));
            }
        }
        // The same bytes held, with fewer moved.
        assert_eq!(fine.cpu, coarse.cpu);
        // (View memory saves less: a second part written opens the block
        // whole.)
        assert!(fine.bytes[1] * 4 < coarse.bytes[1] * 3, "taken with the parts {:?}, without {:?}", fine.bytes, coarse.bytes);
        assert!(other.bytes[1] < other.bytes[0], "{:?}", other.bytes);
        // Not freed: the journal remembers these pages, and another test
        // given the same addresses would find them written.
    }
}

// SAFETY: D3D12 resources are free-threaded; `data` is a persistent mapping
// written only on the submitting thread.
unsafe impl Send for Shadow {}
// SAFETY: as above.
unsafe impl Sync for Shadow {}

impl Shadow {
    /// The host GPU address of a title address inside the buffer.
    pub(crate) fn gpu(&self, address: u64) -> u64 {
        self.gpu + (address - self.start)
    }

    /// The title memory the copy is of: its start and length.
    pub(crate) fn span(&self) -> (u64, u64) {
        (self.start, self.len)
    }

    /// The offset of a title address in the copy, if it lies in it.
    pub(crate) fn offset_in(&self, address: u64) -> Option<u64> {
        address.checked_sub(self.start).filter(|&at| at < self.len)
    }

    /// Bytes from a title address to the buffer's end.
    pub(crate) fn left(&self, address: u64) -> u64 {
        (self.start + self.len).saturating_sub(address)
    }

    /// Whether the memory behind the copy turned out not to be write-watched:
    /// every refresh then copies its whole range.
    pub(crate) fn unwatched(&self) -> bool {
        self.watched.load(Ordering::Relaxed) == NOT_WATCHED
    }

    /// What a refresh of `[address, address + len)` asks about at most
    /// (`crate::watch::writes`): the range cut to the buffer; `None` if
    /// nothing (outside it, or the copy is title memory in place).
    pub(crate) fn refreshed(&self, address: usize, len: usize) -> Option<(usize, usize)> {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), address.saturating_add(len).min(end));
        (self.place.is_none() && from < to).then_some((from, to))
    }

    /// A window of title memory outside buffers.
    pub(crate) fn is_window(&self) -> bool {
        self.window
    }

    /// Title memory in place (nothing is copied for it).
    pub(crate) fn is_placed(&self) -> bool {
        self.place.is_some()
    }

    /// The copy's own number (see the field).
    pub(crate) fn id(&self) -> u64 {
        self.id
    }

    /// Offset of a title address from the buffer's start.
    pub(crate) fn offset(&self, address: u64) -> u64 {
        address - self.start + self.place.as_ref().map_or(0, |place| place.1)
    }

    /// Title memory `[start, start + len)` in place: `placed` is a buffer
    /// over it in a heap opened on it. Shaped as a copy so that everything
    /// that binds, views or keeps a copy works on it unchanged; what
    /// copies does nothing ([`Shadow::update`], [`Shadow::gpu_writes`]).
    fn placed(placed: crate::inplace::Placed, start: u64, len: u64, window: bool) -> Shadow {
        crate::host::name(&placed.buffer, || format!("title memory {start:#x}+{len:#x} in place"));
        let mut shadow = Shadow::new(placed.buffer, start as *mut u8, placed.gpu, start, len);
        // Not a copy: not counted with them, and nothing asks what the
        // title wrote there.
        LIVE[0].fetch_sub(1, Ordering::Relaxed);
        LIVE[1].fetch_sub(len, Ordering::Relaxed);
        *shadow.watched.get_mut() = NOT_WATCHED;
        shadow.window = window;
        shadow.place = Some((placed.heap, placed.bias));
        shadow.pooled = placed.pooled;
        shadow
    }

    /// Brings `len` bytes at `address` up to date when `list` is submitted,
    /// and keeps the copy alive until the list is reset.
    pub(crate) fn refresh(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        let len = len.min(self.start + self.len - address);
        crate::bus::bound(self.bus_kind(), len);
        list.defer_refresh(self, address as usize, len as usize);
    }

    /// Where shaders read the copy: [`crate::bus::kind`].
    fn bus_kind(&self) -> usize {
        use crate::bus::kind;
        if self.card.is_some() {
            kind::CARD
        } else if self.bar.is_some() {
            kind::CARD_BAR
        } else if self.place.is_some() {
            kind::IN_PLACE
        } else if self.window {
            kind::PC_WINDOW
        } else {
            kind::PC_BUFFER
        }
    }

    fn new(buffer: ID3D12Resource, data: *mut u8, gpu: u64, start: u64, len: u64) -> Shadow {
        let pages = (start + len).div_ceil(PAGE as u64) - start / PAGE as u64;
        LIVE[0].fetch_add(1, Ordering::Relaxed);
        LIVE[1].fetch_add(len, Ordering::Relaxed);
        Shadow {
            buffer,
            data,
            gpu,
            start,
            len,
            copied: Mutex::new(vec![0; pages as usize]),
            watched: AtomicU8::new(0),
            pending: Mutex::default(),
            flights: Mutex::default(),
            before: Mutex::default(),
            kept: Mutex::default(),
            used: AtomicU64::new(crate::host::moment()),
            asked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            unsized_asks: AtomicU64::new(0),
            id: {
                static NEXT: AtomicU64 = AtomicU64::new(1);
                NEXT.fetch_add(1, Ordering::Relaxed)
            },
            card: None,
            window: false,
            place: None,
            pooled: false,
            bar: None,
            picked: [AtomicU64::new(u64::MAX), AtomicU64::new(0)],
            reference: Mutex::new(None),
        }
    }

    /// `[address, address + len)` cut to the buffer, as offsets in the copy.
    fn inside(&self, address: usize, len: usize) -> (usize, usize) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let from = address.clamp(start, end);
        (from - start, (address + len).clamp(from, end) - start)
    }

    /// Shaders of `list` may write `[address, address + len)` of the copy
    /// (unordered access): after the list's submission has run, the range
    /// is copied back into title memory, where the title reads it; until
    /// then no refresh copies title memory over it. Call after the
    /// [`Shadow::refresh`] that gives the shaders the title's data.
    ///
    /// shortcut: what shaders wrote is found by keeping the range's bytes
    /// before the submission (a copy per declared range and submission);
    /// a GPU-side dirty mask would save that if big ranges make it slow.
    pub(crate) fn gpu_writes(self: &Arc<Self>, list: &HostList, address: u64, len: u64) {
        // In place the shaders write title memory itself: no snapshot, no
        // compare, nothing to copy back, and no wait on its account.
        // shortcut: another host copy of the same bytes (a title buffer
        // inside this one with a copy of its own, a texture in it) is not
        // told: the card's writes are not seen by write watching.
        if self.place.is_some() {
            return;
        }
        let len = len.min(self.left(address));
        if self.bar.is_some() {
            // Their snapshot and compare read the copy back through the
            // bus, uncached (`gpu_upload_copies`): counted, said once.
            BAR_COPIES[2].fetch_add(len, Ordering::Relaxed);
            static SAID: std::sync::Once = std::sync::Once::new();
            SAID.call_once(|| {
                tracing::warn!(
                    start = format_args!("{:#x}", self.start),
                    len = self.len,
                    window = self.window,
                    "shaders write a host copy in a GPU upload heap (VICTORY_GPU_UPLOAD_COPIES): its compare reads it back \
                     through the bus at some 40 MB/s"
                );
            });
        }
        list.defer_gpu_write(self, address as usize, len as usize);
    }

    /// At submission, after the range's refresh: a submitted list's shaders
    /// write this range. Its bytes are kept to tell later what the GPU wrote:
    /// one snapshot per copy and submission, each byte taken once, when the
    /// first range holding it is declared (the world declares overlapping
    /// views of its 174 MB heap dozens of times, from different offsets: a
    /// snapshot grown to cover each was copied whole every time), in reused
    /// memory (fresh snapshots cost ~2.3 million page faults per world
    /// frame).
    pub(crate) fn writing(&self, address: usize, len: usize) {
        if self.place.is_some() {
            return;
        }
        let range = (address, address + len);
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        // With the copy-back on its own thread: counted for this submission
        // (once), whether or not an earlier one still holds the range.
        if relay() {
            self.flights.lock().unwrap_or_else(|e| e.into_inner()).declare(range, PREPARED.load(Ordering::Relaxed));
        }
        // Dispatches declare the same ranges thousands of times.
        if pending.contains(&range) {
            return;
        }
        if self.window {
            WINDOW_CARD_STATS[window_stats::DECLARED].fetch_add(1, Ordering::Relaxed);
            WINDOW_CARD_STATS[window_stats::DECLARED_BYTES].fetch_add(len as u64, Ordering::Relaxed);
        }
        // In the card's memory: no snapshot. What shaders wrote is found on
        // the card (`Host::card_before` reads the declared ranges from the
        // lists).
        if self.card.is_some() {
            pending.push(range);
            drop(pending);
            let (from, to) = self.inside(address, len);
            self.card_declare(from, to);
            return;
        }
        // The parts no declared range holds yet: their bytes are taken now,
        // just after the range's refresh (a part declared earlier has not
        // been refreshed since, and keeps the bytes taken then).
        let mut parts = vec![self.inside(address, len)];
        for &(a, e) in pending.iter() {
            let (a, e) = self.inside(a, e - a);
            let mut rest = Vec::with_capacity(parts.len() + 1);
            for (from, to) in parts {
                if e <= from || a >= to {
                    rest.push((from, to));
                    continue;
                }
                if from < a {
                    rest.push((from, a));
                }
                if e < to {
                    rest.push((e, to));
                }
            }
            parts = rest;
        }
        pending.push(range);
        let taken: usize = parts.iter().map(|&(from, to)| to.saturating_sub(from)).sum();
        BACK_CLASS[usize::from(!self.window)].fetch_add(taken as u64, Ordering::Relaxed);
        let total = self.len as usize;
        let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let snapshot = before.get_or_insert_with(|| pooled(total));
        if keeps() {
            // Piece by piece, and only the pieces not known current. A
            // piece the part holds whole is current from now on; of a
            // piece it holds partly only the part's bytes are taken, as
            // without the switch, and it is taken again next time (the
            // rest of it may belong to a range declared earlier, whose
            // bytes stay as they were taken).
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if kept.0.is_empty() {
                kept.0 = vec![false; self.pieces(0, total.max(1)).end];
            }
            for (from, to) in parts {
                if from >= to {
                    continue;
                }
                for piece in self.pieces(from, to) {
                    if kept.0[piece] {
                        continue;
                    }
                    let (lo, hi) = self.piece(piece);
                    let whole = from <= lo && hi <= to;
                    let (lo, hi) = (lo.max(from), hi.min(to));
                    // SAFETY: as below; `hi <= total`.
                    unsafe {
                        std::ptr::copy_nonoverlapping(self.data.add(lo), snapshot.as_mut_ptr().add(lo), hi - lo);
                    }
                    kept.0[piece] = whole;
                }
            }
            return;
        }
        if wide_back() {
            // Noted, and taken with the submission's other snapshots when
            // its refreshes are done (`take_snapshots`): the range is
            // declared from here on, so nothing changes its bytes before
            // then. The snapshot's memory stays where it is until the
            // range is done with (`copy_back_done`).
            let mut snaps = SNAPS.lock().unwrap_or_else(|e| e.into_inner());
            let (data, snapshot) = (self.data as usize, snapshot.as_mut_ptr() as usize);
            snaps.extend(parts.into_iter().filter(|&(from, to)| from < to).map(|(from, to)| (data + from, snapshot + from, to - from)));
            return;
        }
        for (from, to) in parts {
            if from < to {
                // SAFETY: the copy's mapping and the snapshot's allocation
                // (`pooled`) both hold `total` bytes, and `to <= total`.
                unsafe {
                    copy_wide(self.data.add(from), snapshot.as_mut_ptr().add(from), to - from);
                }
            }
        }
    }

    /// After the submission ran: copies the bytes its shaders wrote back
    /// into title memory (unmapped pages left alone), and only those: the
    /// title's threads write the same buffers while the GPU runs (its
    /// command memory shares the big GPU heap with what shaders write), and
    /// putting back the copy's older bytes there broke the title's own
    /// records (a fence list read as -1, title+0x4e9b54a). Returns the bytes
    /// written. The range must lie in ranges declared with
    /// [`Shadow::writing`] and not yet done: `Host::execute` passes the
    /// union of a submission's ranges, each byte once, then calls
    /// [`Shadow::copy_back_done`].
    pub(crate) fn copy_back(&self, address: usize, len: usize) -> usize {
        self.copy_back_with(address, len, None)
    }

    /// [`Shadow::copy_back`]; `found`, if given, is the pieces of the range
    /// that differ from the snapshot, found beforehand ([`find_all`]).
    pub(crate) fn copy_back_with(&self, address: usize, len: usize, found: Option<Vec<(usize, usize)>>) -> usize {
        if self.place.is_some() {
            return 0;
        }
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let mut written = 0;
        let (from, to) = self.inside(address, len);
        let start = self.start as usize;
        self.compared(from, to);
        if let Some(before) = &*snapshot
            && from < to
        {
            // SAFETY: `writing` set the snapshot's bytes of every declared
            // range, at the copy's offsets.
            let before = unsafe { std::slice::from_raw_parts(before.as_ptr().add(from), to - from) };
            // By 64 KiB piece: the copy compared with its bytes before the
            // submission (on several threads: gigabytes a frame, bound by
            // memory speed), title memory read only where the GPU wrote.
            #[cfg(feature = "diagnostics")]
            let scanning = std::time::Instant::now();
            let differing = match found {
                Some(found) => found,
                None => self.differing(from, to, before),
            };
            #[cfg(feature = "diagnostics")]
            let syncing = {
                use crate::copystat::{C, add};
                add(C::BackScanUs, scanning.elapsed().as_micros() as u64);
                add(C::BackDiffer, differing.len() as u64);
                std::time::Instant::now()
            };
            // Where the copy changes (the GPU wrote, and below the title's
            // bytes come in), a kept snapshot is no longer current.
            for &(at, n) in &differing {
                self.changed(at, at + n);
            }
            // Sized to each stretch read (`back_piece`): a zeroed 64 KiB per
            // range, most of whose ranges differ nowhere, was most of the
            // work for small ranges.
            let (mut title, mut mask, mut runs) = (Vec::new(), Vec::new(), Vec::new());
            for (at, n) in differing {
                // SAFETY: inside the mapping; only the submitting thread
                // writes it.
                let host = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                let old = &before[at - from..at - from + n];
                let read = |first: usize, title: &mut [u8]| crate::host::read_into((start + at + first) as *const u8, title);
                let Some((first, n)) = back_piece(host, old, &mut title, &mut mask, &mut runs, read) else {
                    continue;
                };
                let at = at + first;
                count_skinned(skinned::MERGED, (start + at) as u64, (start + at + n) as u64);
                crate::host::shaders_write(start + at, &title[..n.min(title.len())]);
                crate::watch::writing_exact(start + at, n);
                // Only the GPU's bytes (upstream #2009): the rest of the
                // stretch is the title's as read, and a title write landing
                // there since stays.
                for &(off, len) in &runs {
                    crate::host::write_mapped((start + at + off) as *mut u8, title[off..].as_ptr(), len);
                    written += len;
                }
            }
            // Our write shows as a title write at the copy's next refresh,
            // which copies those pages once more (some dozens of MB a
            // frame). Asking about the whole range here, to note them
            // current, cost more: 19-31 askings a frame over 110-174 MB of
            // view memory, each a walk of its guards.
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::BackSyncUs, syncing.elapsed().as_micros() as u64);
        }
        written
    }

    /// Offsets `[from, to)` of the copy are compared with the snapshot: for
    /// the frame line's counts.
    fn compared(&self, from: usize, to: usize) {
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackCompared, (to - from) as u64);
        let (bytes, ranges) = if self.window { (2, 5) } else { (3, 6) };
        BACK_CLASS[bytes].fetch_add((to - from) as u64, Ordering::Relaxed);
        BACK_CLASS[ranges].fetch_add(1, Ordering::Relaxed);
        if !self.window && self.len >= 1 << 20 {
            BACK_CLASS[4].fetch_add((to - from) as u64, Ordering::Relaxed);
        }
    }

    /// [`relay`]: the declared range `[address, address + len)` of a copy
    /// in the PC's memory is about to be looked through
    /// (`crate::shadow::find_all`) by the thread that settles: counted as
    /// [`Shadow::copy_back_with`] counts it.
    pub(crate) fn relay_compared(&self, address: usize, len: usize) {
        let (from, to) = self.inside(address, len);
        self.compared(from, to);
    }

    /// [`relay`], for a copy in the PC's memory: offsets `[at, at + n)`
    /// are a 64 KiB piece of a declared range that differed from the
    /// snapshot when it was looked through. What the card wrote there
    /// since the snapshot was last brought up goes into title memory, from
    /// the first to the last such byte, and into the snapshot, so that the
    /// next flight's merge finds only what the card writes from now on.
    /// The copy is read once and never written: a later flight's shaders
    /// may be writing it at this moment. Under the snapshot's lock, which
    /// [`Shadow::merge_under_writes`] holds while it reads title memory
    /// and puts the title's bytes into the copy and the snapshot: the two
    /// never see each other half done. `now` and `title` are scratch
    /// memory. Returns the bytes written to title memory.
    pub(crate) fn relay_merge(&self, at: usize, n: usize, now: &mut Vec<u8>, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_some() || n == 0 || at + n > self.len as usize {
            return 0;
        }
        let mut snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = snapshot.as_mut() else {
            return 0;
        };
        now.clear();
        now.reserve(n);
        // SAFETY: inside the copy's mapping (checked above), read into
        // room just reserved; the card may write the mapping meanwhile, and
        // each byte read is one it held at some moment.
        unsafe {
            std::ptr::copy_nonoverlapping(self.data.add(at), now.as_mut_ptr(), n);
            now.set_len(n);
        }
        // SAFETY: the snapshot's allocation holds the copy's length
        // (`writing`), and these offsets are in a range declared and not
        // done with, whose bytes it holds; nothing else touches them
        // without the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(before.as_mut_ptr().add(at), n) };
        let Some(first) = first_difference(now, held) else {
            return 0;
        };
        let last = last_difference(now, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (now, held) = (&now[first..=last], &mut held[first..=last]);
        title.clear();
        title.resize(n, 0);
        // Title memory not there: looked at again when the next flight
        // that declared the range lands (the snapshot stays as it is).
        if !crate::host::read_into((start + at) as *const u8, title) {
            return 0;
        }
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = merge_race_on().then(|| title.clone());
        if !merge_piece(now, held, title) {
            return 0;
        }
        count_skinned(skinned::MERGED, (start + at) as u64, (start + at + title.len()) as u64);
        crate::host::shaders_write(start + at, &title[..]);
        if self.put_back_own(start + at, &title[..]) {
            return n;
        }
        crate::watch::writing_exact(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "pc relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }

    /// With `crate::inplace::back_own` (`VICTORY_BACK_OWN_VIEW=1`): stores
    /// what a merge made of title memory `[at, at + bytes.len())` (inside
    /// one 64 KiB page, as every merged stretch is) through a view of our
    /// own, tells the journal, and notes the page current for this copy if
    /// it was current before: the copy holds the same bytes (the merge put
    /// them into both), so the next refresh need not take the page again.
    /// The guards are not touched: a title write to the page still faults
    /// and is reported at a later asking, which every copy then takes,
    /// this one too. False with nothing done if the switch is off or the
    /// memory cannot be written that way (the caller writes as before).
    fn put_back_own(&self, at: usize, bytes: &[u8]) -> bool {
        /// Since the start: stretches and bytes stored this way, pages
        /// noted current for their copy, stretches written the old way;
        /// and with `VICTORY_BACK_OWN_CHECK=1`: stores the title's own
        /// address did not show afterwards, pages noted current whose
        /// CPU side in the copy differs from title memory, and those of
        /// them whose guard was clean (a hole in the rule: a title write
        /// on its way leaves the guard open).
        static OWN_WRITES: [AtomicU64; 7] = [const { AtomicU64::new(0) }; 7];
        static CHECK: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let check = *CHECK.get_or_init(|| std::env::var_os("VICTORY_BACK_OWN_CHECK").is_some_and(|v| v == "1"));
        if !crate::inplace::back_own() {
            return false;
        }
        let Some(shown) = crate::inplace::write_own(at, bytes) else {
            OWN_WRITES[3].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        if check {
            // The title's own address must show what went in through ours.
            let mut back = vec![0u8; bytes.len()];
            if crate::host::read_into(at as *const u8, &mut back) && back != bytes {
                OWN_WRITES[4].fetch_add(1, Ordering::Relaxed);
            }
        }
        let page = at / PAGE;
        let others: Vec<usize> = shown.iter().map(|&a| a / PAGE).filter(|&p| p != page).collect();
        let (before, now) = crate::watch::wrote_own(page, &others);
        // A placed texture there changed with no descriptor written (what
        // `crate::watch::writing` tells the texture table on the old way).
        crate::descriptor::host::remapped(at, bytes.len());
        let base = self.start as usize / PAGE;
        {
            let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(taken) = page.checked_sub(base).and_then(|i| copied.get_mut(i))
                && *taken != 0
                && *taken >= before
            {
                *taken = (*taken).max(now);
                OWN_WRITES[2].fetch_add(1, Ordering::Relaxed);
                // A copy in the card's memory only: its CPU side is never
                // written by the card, so it must equal title memory now.
                if check && self.card.is_some() {
                    let (start, end) = (self.start as usize, (self.start + self.len) as usize);
                    let (lo, hi) = ((page * PAGE).max(start), ((page + 1) * PAGE).min(end));
                    if lo < hi {
                        let mut title = vec![0u8; hi - lo];
                        // SAFETY: inside the copy's mapping; read only.
                        let held = unsafe { std::slice::from_raw_parts(self.data.add(lo - start).cast_const(), hi - lo) };
                        if crate::host::read_into(lo as *const u8, &mut title) && title != held {
                            OWN_WRITES[5].fetch_add(1, Ordering::Relaxed);
                            if crate::watch::clean(page) {
                                OWN_WRITES[6].fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    }
                }
            }
        }
        OWN_WRITES[1].fetch_add(bytes.len() as u64, Ordering::Relaxed);
        let n = OWN_WRITES[0].fetch_add(1, Ordering::Relaxed) + 1;
        if n % 16384 == 0 {
            tracing::info!(
                "what shaders wrote, stored through views of our own (VICTORY_BACK_OWN_VIEW=1), since the start: {n} stretches {} MB, {} pages noted current for the copy that wrote them (not copied in again), {} stretches written the old way; with VICTORY_BACK_OWN_CHECK=1: {} stores the title's address did not show, {} pages noted current whose CPU side differs from title memory, {} of those with the guard clean",
                OWN_WRITES[1].load(Ordering::Relaxed) >> 20,
                OWN_WRITES[2].load(Ordering::Relaxed),
                OWN_WRITES[3].load(Ordering::Relaxed),
                OWN_WRITES[4].load(Ordering::Relaxed),
                OWN_WRITES[5].load(Ordering::Relaxed),
                OWN_WRITES[6].load(Ordering::Relaxed)
            );
        }
        true
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory) of
    /// `[from, to)` of the copy that differ from `before` (the bytes from
    /// `from` on), as (offset, bytes) in order; compared on several threads
    /// when the range is big.
    fn differing(&self, from: usize, to: usize, before: &[u8]) -> Vec<(usize, usize)> {
        let (start, data) = (self.start as usize, self.data as usize);
        let scan = |a: usize, b: usize| -> Vec<(usize, usize)> {
            let mut found = Vec::new();
            let mut at = a;
            while at < b {
                let n = (PIECE - (start + at) % PIECE).min(b - at);
                // SAFETY: inside the mapping, which nothing writes while a
                // submission's results are compared.
                let host = unsafe { std::slice::from_raw_parts((data + at) as *const u8, n) };
                if host != &before[at - from..at - from + n] {
                    found.push((at, n));
                }
                at += n;
            }
            found
        };
        const PER_THREAD: usize = 16 << 20;
        let threads = ((to - from) / PER_THREAD).clamp(1, 8);
        if threads == 1 {
            return scan(from, to);
        }
        // Cut on piece boundaries, so that every thread sees whole pieces.
        let cut = |i: usize| {
            if i == threads {
                return to;
            }
            let at = from + (to - from) / threads * i;
            (at - (start + at) % PIECE).max(from)
        };
        let scan = &scan;
        std::thread::scope(|s| {
            let parts: Vec<_> = (0..threads)
                .map(|i| {
                    let (a, b) = (cut(i), cut(i + 1));
                    s.spawn(move || scan(a, b))
                })
                .collect();
            parts
                .into_iter()
                .flat_map(|part| match part.join() {
                    Ok(found) => found,
                    Err(panic) => std::panic::resume_unwind(panic),
                })
                .collect()
        })
    }

    /// After [`Shadow::copy_back`] of a submission's ranges: every submitted
    /// step that declared one of `ranges` (`(address, end)`, as declared) is
    /// done with it (one submission runs at a time), and the snapshot with
    /// the last range.
    ///
    /// With [`relay`] several submissions may hold a range: `ranges` are
    /// those of one of them (each once), and a range is done with when the
    /// last submission that declared it is ([`Flights`]).
    pub(crate) fn copy_back_done(&self, ranges: &[(usize, usize)]) {
        if self.place.is_some() {
            return;
        }
        let mut pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if relay() {
            let free = self.flights.lock().unwrap_or_else(|e| e.into_inner()).done(ranges);
            pending.retain(|r| !free.contains(r));
        } else {
            pending.retain(|r| !ranges.contains(r));
        }
        if !pending.is_empty() || self.card.is_some() {
            return;
        }
        if keeps() {
            // Kept for the next submission, if it is worth it (a megabyte
            // or more) and there is room.
            let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
            if !kept.1 && self.len >= 1 << 20 {
                if KEPT_BYTES.fetch_add(self.len, Ordering::Relaxed) + self.len <= kept_most() {
                    kept.1 = true;
                } else {
                    KEPT_BYTES.fetch_sub(self.len, Ordering::Relaxed);
                }
            }
            if kept.1 {
                return;
            }
            // Given back: nothing of it is current next time. (The lock
            // on the pieces ends here: `copy_back` takes it while it holds
            // the snapshot's.)
            kept.0.clear();
        }
        if let Some(bytes) = self.before.lock().unwrap_or_else(|e| e.into_inner()).take() {
            give_back(bytes);
        }
    }

    /// `VICTORY_FILL_FROM_COPY` (`crate::host::fill_from_copy`): whether
    /// what shaders wrote to `[from, to)` (title addresses, inside the
    /// declared range `declared` that is not copied back yet) can be read
    /// out of this copy without a copy-back, once the card has run every
    /// submission that declared it ([`Shadow::read_written`]). Only a copy
    /// in the PC's memory (not in the card's, not in place), with fresh
    /// snapshots (not [`keeps`]), without the relay (which merges on a
    /// thread of its own), whose snapshot is there and whose pending
    /// ranges still hold `declared`.
    pub(crate) fn written_readable(&self, declared: (usize, usize), from: usize, to: usize) -> bool {
        if self.place.is_some() || self.card.is_some() || keeps() || relay() || from >= to {
            return false;
        }
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        if from < start || to > end || from < declared.0 || to > declared.1 {
            return false;
        }
        let pending = self.pending.lock().unwrap_or_else(|e| e.into_inner());
        if !pending.contains(&declared) {
            return false;
        }
        // (The pending ranges' lock, then the snapshot's: as `writing` and
        // `copy_back_done` take them.)
        self.before.lock().unwrap_or_else(|e| e.into_inner()).is_some()
    }

    /// `VICTORY_FILL_FROM_COPY`: `dst` holds `to - from` bytes of title
    /// memory from `from` on (a fill of upload memory read them); where
    /// the shaders of submissions on the card wrote this copy there, their
    /// bytes go into `dst` instead, as [`Shadow::copy_back`] would put them
    /// into title memory: by 64 KiB piece, from the first byte where the
    /// copy differs from its snapshot to the last, the copy's byte where it
    /// differs and title memory's elsewhere, and nothing where title memory
    /// is not all mapped. Nothing is written but `dst`: the copy-back that
    /// follows at the next settle finds the same bytes and puts them into
    /// title memory.
    ///
    /// The caller has checked [`Shadow::written_readable`] and waited for
    /// the card to run every submission that declared the range; no
    /// copy-back has run since (under the submission lock, and a settle in
    /// between drops the plan). Returns the bytes taken from the copy.
    pub(crate) fn read_written(&self, from: usize, to: usize, dst: *mut u8) -> usize {
        let snapshot = self.before.lock().unwrap_or_else(|e| e.into_inner());
        let Some(before) = &*snapshot else {
            return 0;
        };
        let start = self.start as usize;
        let (lo, hi) = self.inside(from, to.saturating_sub(from));
        if lo >= hi || lo + start != from {
            return 0;
        }
        // SAFETY: inside the copy's mapping; the card has run what writes
        // it, and only the submitting thread (which holds the submission
        // lock) writes it otherwise. The snapshot holds the copy's length,
        // and these offsets lie in a declared range whose bytes it holds
        // (`written_readable`). `dst` holds `to - from` bytes.
        let (copy, old, dst) = unsafe {
            (
                std::slice::from_raw_parts(self.data.add(lo), hi - lo),
                std::slice::from_raw_parts(before.as_ptr().add(lo), hi - lo),
                std::slice::from_raw_parts_mut(dst, hi - lo),
            )
        };
        merge_written(start + lo, copy, old, dst, |at, title| crate::host::read_into(at as *const u8, title))
    }

    /// The 64 KiB pieces (cut at multiples of 64 KiB of title memory, as
    /// [`Shadow::differing`] cuts them) that hold offsets `[from, to)` of
    /// the copy; `from < to`.
    fn pieces(&self, from: usize, to: usize) -> std::ops::Range<usize> {
        pieces_of(self.start as usize, from, to)
    }

    /// The offsets in the copy that piece `piece` holds.
    fn piece(&self, piece: usize) -> (usize, usize) {
        piece_at(self.start as usize, self.len as usize, piece)
    }

    /// The copy's bytes at offsets `[from, to)` change: a kept snapshot is
    /// not current there any more.
    fn changed(&self, from: usize, to: usize) {
        if !keeps() || from >= to {
            return;
        }
        let mut kept = self.kept.lock().unwrap_or_else(|e| e.into_inner());
        if kept.0.is_empty() {
            return;
        }
        for piece in self.pieces(from, to) {
            kept.0[piece] = false;
        }
    }

    /// Whether shaders of a submission not yet copied back may be writing
    /// the copy.
    pub(crate) fn writes_pending(&self) -> bool {
        !self.pending.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Whether ranges shaders of submissions not yet copied back may be
    /// writing in this copy cover title memory `[from, to)` whole: what
    /// they write there is in this copy (`crate::host::table_gpu_copy`).
    pub(crate) fn pending_covers(&self, from: usize, to: usize) -> bool {
        crate::host::covered(&self.pending.lock().unwrap_or_else(|e| e.into_inner()), from, to)
    }

    /// Whether [`Shadow::update`] of `[address, address + len)` would leave
    /// title bytes behind: a range shaders may be writing holds a page the
    /// title wrote since the copy last took it. The answer comes from the
    /// write-watch journal alone, as the submission's one asking left it
    /// (`crate::watch::asked`); where the journal cannot say (memory not
    /// write-watched, a range not asked about) it is yes. A page the copy
    /// never took whole counts as written if the title ever wrote it.
    ///
    /// For submissions sent to the card without a wait
    /// (`crate::host::late`): the ranges are those of flights in the air,
    /// which are then settled before this refresh.
    pub(crate) fn stale_under_writes(&self, address: usize, len: usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            return true;
        }
        let base = start / PAGE;
        let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let first_only = crate::host::stale_once();
        busy.into_iter().any(|(a, e)| match crate::watch::asked(a, e) {
            // With `VICTORY_STALE_ONCE`: the first stale page answers.
            Some((_, written)) if first_only => any_stale_page_in(base, &copied, a, e, &written),
            Some((_, written)) => !stale_pages_in(base, &copied, a, e, &written).is_empty(),
            None => true,
        })
    }

    /// Where [`Shadow::stale_under_writes`] says yes: puts the bytes the
    /// title changed into the copy, in the ranges shaders may be writing,
    /// and only those bytes, so that what the shaders write beside them
    /// stays (`crate::host::late_exact`). For each page of such a range
    /// that the title wrote since the copy last took it, title memory as
    /// it is now is compared with what the copy held when the range was
    /// given to the shaders: the copy's CPU side for a copy in the card's
    /// memory (the changed runs go up before the next submission, each a
    /// copy of its own bytes, `runs_left` of them at most), the snapshot
    /// for the others (the changed bytes go into the copy and into the
    /// snapshot, so the compare after the card ran takes them for the
    /// title's and not the shaders').
    ///
    /// Whether the caller must still settle the flights: the journal
    /// cannot say what was written, a snapshot is missing, or `runs_left`
    /// ran out. Pages done before that stay done.
    pub(crate) fn merge_under_writes(&self, address: usize, len: usize, runs_left: &mut usize) -> bool {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return false;
        }
        let busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .filter(|&&(a, e)| a < to && e > from)
            .map(|&(a, e)| (a.max(from), e.min(to)))
            .collect();
        if busy.is_empty() {
            return false;
        }
        if self.watched.load(Ordering::Relaxed) == NOT_WATCHED {
            crate::cpuwait::stale_why(crate::cpuwait::stale::UNWATCHED);
            return true;
        }
        let base = start / PAGE;
        let mut title = vec![0u8; PAGE];
        for (a, e) in busy {
            // As the refresh itself asks (`update_free`): the journal if
            // the submission's asking covered the range, else the kernel,
            // else the guards. The journal alone said nothing for copies
            // of view memory, which is guarded and not write-watched, and
            // those were all of run 99's cases: the flights were settled
            // 4-5 times a frame for "0 pages looked at".
            let Some((now, written)) = crate::watch::writes(a, e, true) else {
                crate::cpuwait::stale_why(crate::cpuwait::stale::NO_JOURNAL);
                return true;
            };
            let stale: Vec<usize> = {
                let copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                stale_pages_in(base, &copied, a, e, &written)
            };
            for page in stale {
                let (lo, hi) = ((page * PAGE).max(a), ((page + 1) * PAGE).min(e));
                if lo >= hi {
                    continue;
                }
                let n = hi - lo;
                let title = &mut title[..n];
                // The lock of the copy's reference (the snapshot, or the
                // CPU's side of a copy in the card's memory), from before
                // title memory is read until the title's bytes are in: with
                // the copy-back on its own thread ([`relay`]) that thread
                // puts what the card wrote into title memory and into the
                // reference under the same lock, and a page read before
                // such a merge and compared after it would take the card's
                // bytes for the title's.
                let mut before = self.before.lock().unwrap_or_else(|e| e.into_inner());
                // A page that is not there is left as it is, and looked at
                // again next time.
                if !crate::host::read_into(lo as *const u8, title) {
                    continue;
                }
                MERGED[0].fetch_add(1, Ordering::Relaxed);
                let at = lo - start;
                if self.card.is_some() {
                    // A mirror's piece not held (`no_pc_copy`): nothing to
                    // compare with, so the flights are settled.
                    if !self.held(at, at + n) {
                        crate::cpuwait::stale_why(crate::cpuwait::stale::NOT_HELD);
                        return true;
                    }
                    // SAFETY: inside the mapping of the copy's CPU side,
                    // which only the submitting thread writes.
                    let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), n) };
                    let runs = joined_over(changed_runs(title, held), merge_gap());
                    if runs.len() > *runs_left {
                        crate::cpuwait::stale_why(crate::cpuwait::stale::RUNS);
                        return true;
                    }
                    *runs_left -= runs.len();
                    for (r, m) in runs {
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        self.card_changed(at + r, at + r + m);
                        MERGED[1].fetch_add(1, Ordering::Relaxed);
                        MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                    }
                } else {
                    let Some(snapshot) = before.as_mut() else {
                        crate::cpuwait::stale_why(crate::cpuwait::stale::NO_SNAPSHOT);
                        return true;
                    };
                    // SAFETY: the snapshot's allocation holds the copy's
                    // bytes at the copy's offsets (`writing`), and these
                    // are in a declared range.
                    let held = unsafe { std::slice::from_raw_parts_mut(snapshot.as_mut_ptr().add(at), n) };
                    for (r, m) in changed_runs(title, held) {
                        // SAFETY: inside the copy's mapping. The card may be
                        // writing the copy beside these bytes; they are
                        // written here and nothing around them.
                        unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(r), self.data.add(at + r), m) };
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        MERGED[1].fetch_add(1, Ordering::Relaxed);
                        MERGED[2].fetch_add(m as u64, Ordering::Relaxed);
                    }
                }
                drop(before);
                // Taken, if the range holds all of the page that is in the
                // copy; else the rest of it is the refresh's, which notes
                // only whole pages, and the page is looked at again.
                if (page * PAGE).max(start) >= a && ((page + 1) * PAGE).min(end) <= e {
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(at) = copied.get_mut(page - base) {
                        *at = now;
                    }
                }
            }
        }
        false
    }

    /// At submission: copies what changed of `[address, address + len)`
    /// since it was last copied (all of it if the memory is not
    /// write-watched). Returns the bytes copied.
    pub(crate) fn update(&self, address: usize, len: usize) -> usize {
        // In place: the card reads title memory itself.
        if self.place.is_some() {
            return 0;
        }
        self.free_parts(address, len).into_iter().map(|(a, b)| self.update_free(a, b)).sum()
    }

    /// `[address, address + len)` cut to the buffer, less the ranges shaders
    /// of a submission not yet copied back may be writing: the parts a
    /// refresh may copy, in order.
    pub(crate) fn free_parts(&self, address: usize, len: usize) -> Vec<(usize, usize)> {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (address.max(start), (address + len).min(end));
        if from >= to {
            return Vec::new();
        }
        let mut busy: Vec<(usize, usize)> = self
            .pending
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .copied()
            .filter(|&(a, e)| a < to && e > from)
            .collect();
        if busy.is_empty() {
            return vec![(from, to)];
        }
        // Around ranges shaders are writing: the gaps, each as below (in
        // the world most submissions have such ranges; copying the gaps
        // whole cost ~1 s per frame).
        busy.sort_unstable();
        let mut parts = Vec::new();
        let mut at = from;
        for (a, e) in busy {
            if a > at {
                parts.push((at, a));
            }
            at = at.max(e);
        }
        if at < to {
            parts.push((at, to));
        }
        parts
    }

    /// [`Shadow::update`] of `[from, to)` (inside the buffer) where no
    /// shader writes.
    fn update_free(&self, from: usize, to: usize) -> usize {
        if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
            let asking = std::time::Instant::now();
            let writes = crate::watch::writes(from, to, true);
            REFRESH_SPLIT[0].fetch_add(asking.elapsed().as_micros() as u64, Ordering::Relaxed);
            if let Some((now, written)) = writes {
                self.watched.store(1, Ordering::Relaxed);
                let copied = self.copy_stale(from, to, now, &written);
                self.refresh_check(from, to);
                #[cfg(feature = "diagnostics")]
                {
                    use crate::copystat::{C, add};
                    add(C::Checked, (to - from) as u64);
                    add(C::Unchanged, (to - from).saturating_sub(copied) as u64);
                }
                return copied;
            }
            self.watched.store(NOT_WATCHED, Ordering::Relaxed);
        }
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        if let Some((a, b)) = self.picked_cut(from, to) {
            PICKED_GUARD[picked_stat::WHOLE].fetch_add((b - a) as u64, Ordering::Relaxed);
        }
        let copied = self.copy(from, to);
        #[cfg(feature = "diagnostics")]
        crate::copystat::whole(from, to - from, started.elapsed().as_micros() as u64);
        copied
    }

    /// Diagnostic (`VICTORY_REFRESH_CHECK=1`; off by default): after a
    /// refresh of `[from, to)` that trusted the write watch or the guards,
    /// the first 256 bytes of every 64 KiB block of it are read from title
    /// memory and compared with the copy. A block that differs on three
    /// refreshes in a row was changed with nothing to tell of it (a write
    /// through another view of its pages, as DirectStorage's page
    /// destinations make) and the card reads old bytes there: said with
    /// the copy and the address. One that differs once is a write of the
    /// title's that arrived between the refresh and the look.
    fn refresh_check(&self, from: usize, to: usize) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_REFRESH_CHECK").is_some_and(|v| v == "1")) {
            return;
        }
        const SAMPLE: usize = 256;
        // Blocks found differing: how many refreshes in a row.
        static DIFFERING: Mutex<Option<std::collections::HashMap<usize, u32>>> = Mutex::new(None);
        static SAID: AtomicU64 = AtomicU64::new(0);
        let start = self.start as usize;
        let mut title = [0u8; SAMPLE];
        let mut at = from;
        while at < to {
            let n = SAMPLE.min(to - at);
            if crate::host::read_into(at as *const u8, &mut title[..n]) {
                // SAFETY: `[from, to)` lies inside the buffer (`update`),
                // so these bytes are inside the copy's mapping.
                let copy = unsafe { std::slice::from_raw_parts(self.data.add(at - start).cast_const(), n) };
                let mut map = DIFFERING.lock().unwrap_or_else(|e| e.into_inner());
                let map = map.get_or_insert_with(Default::default);
                if copy == &title[..n] {
                    map.remove(&at);
                } else {
                    let times = map.entry(at).or_insert(0);
                    *times += 1;
                    if *times == 3 {
                        let said = SAID.fetch_add(1, Ordering::Relaxed) + 1;
                        if said <= 64 || said.is_power_of_two() {
                            tracing::warn!(
                                said,
                                copy = self.id,
                                copy_start = format_args!("{start:#x}"),
                                copy_bytes = self.len,
                                address = format_args!("{at:#x}"),
                                refreshed = format_args!("{from:#x}..{to:#x}"),
                                on_the_card = self.card.is_some(),
                                window = self.window,
                                "a host copy of title memory is stale after its refresh, three times in a row:                                  the memory changed with nothing to tell of it (VICTORY_REFRESH_CHECK=1)"
                            );
                        }
                    }
                }
            }
            // The first bytes of the next 64 KiB block.
            at = (at / PAGE + 1) * PAGE;
        }
    }

    /// Copies the pages of `[from, to)` written after their last copy
    /// (`written`: last writes by page, in page order; `now`: this
    /// asking), whole pages within the buffer.
    fn copy_stale(&self, from: usize, to: usize, now: u64, written: &[(usize, u64)]) -> usize {
        let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
        let mut runs = Vec::new();
        self.stale_runs(&mut copied, from, to, now, written, &mut runs);
        // With the notes locked until the bytes are in.
        runs.into_iter().map(|(a, b)| self.copy(a, b)).sum()
    }

    /// The runs of `[from, to)` that [`Shadow::copy_stale`] copies (title
    /// addresses, in order, appended to `runs`), with the pages' notes
    /// (`copied`) made as it makes them: whole stale pages, or with
    /// [`crate::watch::fine`] their parts written since ([`stale_runs_of`]).
    /// With the check (`VICTORY_FINE_PAGES=2`) the parts passed over are
    /// compared with title memory first, and any that differ are taken too
    /// ([`Shadow::fine_check`]).
    fn stale_runs(
        &self,
        copied: &mut [u64],
        from: usize,
        to: usize,
        now: u64,
        written: &[(usize, u64)],
        runs: &mut Vec<(usize, usize)>,
    ) {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let fine = crate::watch::fine();
        let check = fine && crate::watch::fine_level() >= 2;
        let mut mine = Vec::new();
        let mut skipped = Vec::new();
        let counts = stale_runs_of(
            (start, end),
            copied,
            (from, to),
            now,
            written,
            if fine { Some(&crate::watch::subs_after as &SubsAfter<'_>) } else { None },
            fine_gap(),
            &mut mine,
            check.then_some(&mut skipped),
        );
        if fine {
            FINE[0].fetch_add(counts[0] as u64, Ordering::Relaxed);
            FINE[1].fetch_add(counts[1] as u64, Ordering::Relaxed);
            FINE[2].fetch_add(mine.len() as u64, Ordering::Relaxed);
        }
        if !skipped.is_empty() {
            let repairs = self.fine_check(&skipped);
            if !repairs.is_empty() {
                mine.extend(repairs);
                mine = joined(mine);
            }
        }
        runs.extend(mine);
    }

    /// `VICTORY_FINE_PAGES=2`: of `skipped` (title ranges inside the copy
    /// that the journal says were not written since the copy took them),
    /// the ones whose bytes on the CPU's side differ from title memory now;
    /// counted ([`FINE`]). A title write racing the copy also counts (it is
    /// copied at the next refresh either way). Pieces of a mirror not held
    /// (`no_pc_copy`) have nothing to compare with and are passed over.
    fn fine_check(&self, skipped: &[(usize, usize)]) -> Vec<(usize, usize)> {
        let start = self.start as usize;
        let mut out = Vec::new();
        let mut title = Vec::new();
        for &(a, b) in skipped {
            if a >= b || !self.held(a - start, b - start) {
                continue;
            }
            title.clear();
            title.resize(b - a, 0);
            if !crate::host::read_into(a as *const u8, &mut title) {
                continue;
            }
            FINE[3].fetch_add((b - a) as u64, Ordering::Relaxed);
            // SAFETY: inside the CPU side's mapping (`a`, `b` lie in the
            // copy, and it holds them); nothing writes it meanwhile but this
            // thread (they lie outside ranges shaders may be writing).
            let host = unsafe { std::slice::from_raw_parts(self.data.add(a - start), b - a) };
            let differ: usize = changed_runs(&title, host).iter().map(|r| r.1).sum();
            if differ > 0 {
                FINE[4].fetch_add(differ as u64, Ordering::Relaxed);
                out.push((a, b));
            }
        }
        out
    }

    /// With [`wide_copy`]: [`Shadow::update`] without its moves. The title
    /// ranges it would copy are appended to `runs` (in order, cut to the
    /// buffer), and the pages' notes are made as it makes them; the caller
    /// moves the bytes ([`move_wide`]) before anything reads the copy or
    /// writes title memory, and then tells the copy which bytes changed
    /// ([`Shadow::moved`]). Only under the submission lock, as `update`.
    pub(crate) fn plan(&self, address: usize, len: usize, runs: &mut Vec<(usize, usize)>) {
        if self.place.is_some() {
            return;
        }
        let parts = self.free_parts(address, len);
        self.plan_parts(&parts, runs);
    }

    /// [`Shadow::plan`] of `parts` (title ranges inside the buffer, in
    /// order), taken as free of shader writes: what [`Shadow::free_parts`]
    /// gave when they were found (`crate::host`'s `VICTORY_DECLARED_WIDE`
    /// finds them at a shader-write declaration and plans them at the next
    /// flush of all, when the range just declared is pending and would be
    /// left out).
    pub(crate) fn plan_parts(&self, parts: &[(usize, usize)], runs: &mut Vec<(usize, usize)>) {
        if self.place.is_some() {
            return;
        }
        if !self.mirrored() {
            if self.picked_span().is_none() {
                self.plan_runs(parts, runs);
                return;
            }
            // The parts in the picked span: with `VICTORY_PICKED_GUARD` they
            // are copied here, as `copy_held` copies them; the rest is moved
            // wide.
            let mut mine = Vec::new();
            self.plan_runs(parts, &mut mine);
            for (from, to) in mine {
                let Some((a, b)) = self.picked_cut(from, to) else {
                    runs.push((from, to));
                    continue;
                };
                if from < a {
                    runs.push((from, a));
                }
                if picked_guard() == 0 {
                    let stat = if self.card.is_some() { picked_stat::CARD } else { picked_stat::PC };
                    PICKED_GUARD[stat].fetch_add((b - a) as u64, Ordering::Relaxed);
                    runs.push((a, b));
                } else {
                    self.copy_picked(a, b);
                }
                if b < to {
                    runs.push((b, to));
                }
            }
            return;
        }
        // With a mirror (`no_pc_copy`): only what it holds is moved; the
        // rest is noted to go up from title memory, as `copy` notes it.
        let start = self.start as usize;
        let mut planned = Vec::new();
        self.plan_runs(parts, &mut planned);
        for (from, to) in planned {
            for (a, b, held) in self.side_runs(from - start, to - start) {
                if held {
                    runs.push((start + a, start + b));
                } else {
                    self.card_changed(a, b);
                    PC_SIDES[5].fetch_add((b - a) as u64, Ordering::Relaxed);
                }
            }
        }
    }

    /// [`Shadow::plan`]'s runs, whatever holds the CPU's side.
    fn plan_runs(&self, parts: &[(usize, usize)], runs: &mut Vec<(usize, usize)>) {
        for &(from, to) in parts {
            if self.watched.load(Ordering::Relaxed) != NOT_WATCHED {
                let asking = std::time::Instant::now();
                let writes = crate::watch::writes(from, to, true);
                REFRESH_SPLIT[0].fetch_add(asking.elapsed().as_micros() as u64, Ordering::Relaxed);
                if let Some((now, written)) = writes {
                    self.watched.store(1, Ordering::Relaxed);
                    let mut copied = self.copied.lock().unwrap_or_else(|e| e.into_inner());
                    self.stale_runs(&mut copied, from, to, now, &written, runs);
                    continue;
                }
                self.watched.store(NOT_WATCHED, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::WholeBytes, (to - from) as u64);
            if let Some((a, b)) = self.picked_cut(from, to) {
                PICKED_GUARD[picked_stat::WHOLE].fetch_add((b - a) as u64, Ordering::Relaxed);
            }
            runs.push((from, to));
        }
    }

    /// With [`wide_copy`]: where title address `address` (inside the
    /// buffer) is in the copy's mapping.
    pub(crate) fn host_at(&self, address: usize) -> usize {
        self.data as usize + (address - self.start as usize)
    }

    /// With [`wide_copy`]: the bytes of title ranges `ranges` (inside the
    /// buffer, as [`Shadow::plan`] gave them) were moved into the copy and
    /// changed: a kept snapshot is no longer current there, and a copy in
    /// the card's memory sends them up.
    pub(crate) fn moved(&self, ranges: &[(usize, usize)]) {
        let start = self.start as usize;
        for &(from, to) in ranges {
            if self.card.is_none() {
                count_skinned(skinned::PC, from as u64, to as u64);
            }
            self.changed(from - start, to - start);
            self.card_changed(from - start, to - start);
        }
    }

    /// Copies title memory `[from, to)`, cut to the buffer, into the copy;
    /// with [`skip_same`] only the 4 KiB blocks whose bytes differ.
    /// Returns the bytes written.
    fn copy(&self, from: usize, to: usize) -> usize {
        let (start, end) = (self.start as usize, (self.start + self.len) as usize);
        let (from, to) = (from.max(start), to.min(end));
        if from >= to {
            return 0;
        }
        if self.mirrored() {
            // Into the mirror where it holds the bytes; elsewhere only
            // noted, and the bytes go up from title memory when the
            // submission is sent (`no_pc_copy`).
            let mut written = 0;
            for (a, b, held) in self.side_runs(from - start, to - start) {
                if held {
                    written += self.copy_held(start + a, start + b);
                } else {
                    self.card_changed(a, b);
                    PC_SIDES[5].fetch_add((b - a) as u64, Ordering::Relaxed);
                    written += b - a;
                }
            }
            return written;
        }
        self.copy_held(from, to)
    }

    /// [`Shadow::copy`] of title memory `[from, to)` (inside the buffer)
    /// into the CPU's side, which holds those bytes: what lies in the
    /// picked span as [`picked_guard`] says.
    fn copy_held(&self, from: usize, to: usize) -> usize {
        let Some((a, b)) = self.picked_cut(from, to) else {
            return self.copy_plain(from, to);
        };
        let mut written = 0;
        if from < a {
            written += self.copy_plain(from, a);
        }
        written += self.copy_picked(a, b);
        if b < to {
            written += self.copy_plain(b, to);
        }
        written
    }

    /// Title addresses `[address, address + len)` of the copy behind
    /// unordered access buffer views of the view heap, which shaders that
    /// pick views by index may write unannounced: its picked span grows to
    /// hold them ([`picked_guard`]). From the heap pass, for every such
    /// view at every pass.
    pub(crate) fn mark_picked_uav(&self, address: u64, len: u64) {
        if self.place.is_some() {
            return;
        }
        let from = address.saturating_sub(self.start).min(self.len);
        let to = address.saturating_add(len).saturating_sub(self.start).min(self.len);
        if from >= to {
            return;
        }
        if self.picked[0].fetch_min(from, Ordering::Relaxed) == u64::MAX {
            PICKED_GUARD[picked_stat::MARKED].fetch_add(1, Ordering::Relaxed);
        }
        self.picked[1].fetch_max(to, Ordering::Relaxed);
    }

    /// The picked span as offsets in the copy, if it has one.
    fn picked_span(&self) -> Option<(usize, usize)> {
        let lo = self.picked[0].load(Ordering::Relaxed);
        let hi = self.picked[1].load(Ordering::Relaxed);
        (lo < hi).then_some((lo as usize, hi as usize))
    }

    /// Whether offsets `[lo, hi)` of the copy meet its picked span: bytes
    /// shaders that pick views from the heap may write unannounced
    /// (`crate::cpuwait::queue2_picked`).
    pub(crate) fn picked_meets(&self, lo: usize, hi: usize) -> bool {
        self.picked_span().is_some_and(|(a, b)| a < hi && b > lo)
    }

    /// Of title addresses `[from, to)`, the part in the picked span, if any.
    fn picked_cut(&self, from: usize, to: usize) -> Option<(usize, usize)> {
        let (lo, hi) = self.picked_span()?;
        let start = self.start as usize;
        let (a, b) = (from.max(start + lo), to.min(start + hi));
        (a < b).then_some((a, b))
    }

    /// Title memory `[from, to)`, inside the picked span, into the copy, as
    /// [`picked_guard`] says. Returns the bytes written.
    fn copy_picked(&self, from: usize, to: usize) -> usize {
        let carded = self.card.is_some();
        let stat = if carded { picked_stat::CARD } else { picked_stat::PC };
        PICKED_GUARD[stat].fetch_add((to - from) as u64, Ordering::Relaxed);
        // A mirror's pieces not held go up from title memory: nothing to
        // compare with.
        if picked_guard() == 0 || self.mirrored() {
            return self.copy_plain(from, to);
        }
        let exact = picked_guard() == 2;
        if carded { self.copy_picked_card(from, to, exact) } else { self.copy_picked_pc(from, to, exact) }
    }

    /// [`Shadow::copy_picked`] for a copy in the card's memory: its CPU
    /// side holds what the card has but for the shaders' unannounced
    /// writes, so a byte equal to title memory there was not changed by
    /// the title, and with `exact` (`2`) it is neither written nor sent up.
    fn copy_picked_card(&self, from: usize, to: usize, exact: bool) -> usize {
        thread_local! {
            static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
        }
        let start = self.start as usize;
        SCRATCH.with(|scratch| {
            let mut scratch = scratch.borrow_mut();
            let mut written = 0;
            let mut at = from;
            while at < to {
                let n = (to - at).min(SAME_CHUNK);
                if scratch.len() < n {
                    scratch.resize(n, 0);
                }
                let title = &mut scratch[..n];
                if !crate::host::read_into(at as *const u8, title) {
                    written += self.copy_plain(at, at + n);
                    at += n;
                    continue;
                }
                let off = at - start;
                // SAFETY: inside the CPU side's mapping (`[from, to)` lies in
                // the copy); only the submitting thread writes it outside
                // ranges flights declared, and refreshes keep out of those.
                let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(off), n) };
                let runs = changed_runs(title, held);
                let differ: usize = runs.iter().map(|r| r.1).sum();
                PICKED_GUARD[picked_stat::CARD_SAME].fetch_add((n - differ) as u64, Ordering::Relaxed);
                if exact {
                    let runs = close_runs(runs);
                    let mut sent = 0;
                    for &(r, m) in &runs {
                        held[r..r + m].copy_from_slice(&title[r..r + m]);
                        self.changed(off + r, off + r + m);
                        self.card_changed(off + r, off + r + m);
                        sent += m;
                    }
                    PICKED_GUARD[picked_stat::SPARED].fetch_add((n - sent) as u64, Ordering::Relaxed);
                    PICKED_GUARD[picked_stat::RUNS].fetch_add(runs.len() as u64, Ordering::Relaxed);
                    written += sent;
                } else {
                    held.copy_from_slice(title);
                    self.changed(off, off + n);
                    self.card_changed(off, off + n);
                    written += n;
                }
                at += n;
            }
            written
        })
    }

    /// [`Shadow::copy_picked`] for a copy without a card side: the copy
    /// holds the shaders' bytes beside the title's, so the title's bytes as
    /// last copied are kept apart ([`Reference`]); with `exact` (`2`) only
    /// bytes that differ from them are written. A part the reference does
    /// not hold yet is copied whole and taken into it.
    fn copy_picked_pc(&self, from: usize, to: usize, exact: bool) -> usize {
        let Some((lo, hi)) = self.picked_span() else {
            return self.copy_plain(from, to);
        };
        let start = self.start as usize;
        let mut reference = self.reference.lock().unwrap_or_else(|e| e.into_inner());
        // Made again whole when the span grew: every part not held yet.
        if reference.as_ref().is_none_or(|r| r.lo != lo || r.bytes.len() != hi - lo) {
            *reference = Some(Reference { lo, bytes: vec![0; hi - lo], valid: vec![false; (hi - lo).div_ceil(SUB)] });
        }
        let Some(r) = reference.as_mut() else {
            return self.copy_plain(from, to);
        };
        let mut title = vec![0u8; SUB];
        let mut written = 0;
        let (mut off, end) = (from - start, to - start);
        while off < end {
            // The part of the reference `off` is in, cut to the span and to
            // `[from, to)`.
            let part = (off - lo) / SUB;
            let (part_lo, part_hi) = (lo + part * SUB, (lo + (part + 1) * SUB).min(hi));
            let next = part_hi.min(end);
            let n = next - off;
            let title = &mut title[..n];
            if !crate::host::read_into((start + off) as *const u8, title) {
                written += self.copy_plain(start + off, start + next);
                r.valid[part] = false;
                off = next;
                continue;
            }
            let kept = &mut r.bytes[off - lo..next - lo];
            if r.valid[part] {
                let runs = changed_runs(title, kept);
                let differ: usize = runs.iter().map(|r| r.1).sum();
                PICKED_GUARD[picked_stat::PC_SAME].fetch_add((n - differ) as u64, Ordering::Relaxed);
                if exact {
                    let runs = close_runs(runs);
                    let mut put = 0;
                    for &(a, m) in &runs {
                        // SAFETY: inside the copy's mapping; the card may be
                        // writing beside these bytes, and only these are
                        // written.
                        unsafe { std::ptr::copy_nonoverlapping(title.as_ptr().add(a), self.data.add(off + a), m) };
                        count_skinned(skinned::PC, (start + off + a) as u64, (start + off + a + m) as u64);
                        self.changed(off + a, off + a + m);
                        put += m;
                    }
                    kept.copy_from_slice(title);
                    PICKED_GUARD[picked_stat::SPARED].fetch_add((n - put) as u64, Ordering::Relaxed);
                    PICKED_GUARD[picked_stat::RUNS].fetch_add(runs.len() as u64, Ordering::Relaxed);
                    written += put;
                    off = next;
                    continue;
                }
            }
            // SAFETY: inside the copy's mapping.
            unsafe { std::ptr::copy_nonoverlapping(title.as_ptr(), self.data.add(off), n) };
            count_skinned(skinned::PC, (start + off) as u64, (start + off + n) as u64);
            self.changed(off, off + n);
            kept.copy_from_slice(title);
            // Held from now on if the whole part was taken.
            if off == part_lo && next == part_hi {
                r.valid[part] = true;
            }
            written += n;
            off = next;
        }
        written
    }

    /// [`Shadow::copy_held`] without the picked span's rule: as a refresh
    /// always copied.
    fn copy_plain(&self, from: usize, to: usize) -> usize {
        let start = self.start as usize;
        let moving = std::time::Instant::now();
        if skip_same() {
            let mut runs = Vec::new();
            let written = move_same(self.data as usize + (from - start), from, to - from, &mut runs);
            REFRESH_SPLIT[1].fetch_add(moving.elapsed().as_micros() as u64, Ordering::Relaxed);
            REFRESH_SPLIT[2].fetch_add((to - from - written) as u64, Ordering::Relaxed);
            // After the bytes are in: a snapshot taken meanwhile is not
            // current there, and the card's side takes them.
            for (a, b) in runs {
                if self.card.is_none() {
                    count_skinned(skinned::PC, (from + a) as u64, (from + b) as u64);
                }
                self.changed(from - start + a, from - start + b);
                self.card_changed(from - start + a, from - start + b);
            }
            return written;
        }
        // SAFETY: inside the mapping (`from` and `to` are within the buffer).
        let at = unsafe { self.data.add(from - start) };
        crate::host::copy_mapped(at, from as *const u8, to - from, false);
        REFRESH_SPLIT[1].fetch_add(moving.elapsed().as_micros() as u64, Ordering::Relaxed);
        if self.card.is_none() {
            count_skinned(skinned::PC, from as u64, to as u64);
        }
        // After the bytes are in: a snapshot taken meanwhile is not current.
        self.changed(from - start, to - start);
        self.card_changed(from - start, to - start);
        to - from
    }

    /// Whether shaders read and write this copy in the card's own memory
    /// ([`cards`]).
    pub(crate) fn carded(&self) -> bool {
        self.card.is_some()
    }

    /// Whether the copy is in a GPU upload heap ([`gpu_upload_copies`]):
    /// declared writes are compared through the bus.
    pub(crate) fn bar_mapped(&self) -> bool {
        self.bar.is_some()
    }

    /// The CPU side's bytes at offsets `[from, to)` changed: they go to the
    /// card's side before the next submission.
    fn card_changed(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let mut dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
        match dirty.last_mut() {
            Some(last) if from <= last.1 && to >= last.0 => *last = (last.0.min(from), last.1.max(to)),
            _ => dirty.push((from, to)),
        }
    }

    /// Offsets `[from, to)` of a copy in the card's memory lie in a range
    /// its shaders may write ([`Shadow::writing`]): their pieces are noted
    /// ([`Card::written`]), and with a mirror ([`no_pc_copy`]) each piece
    /// not held yet is committed, takes title memory's bytes and goes up
    /// whole, so that the card's side and the mirror agree there. On the
    /// submitting thread, before the submission's sends are taken.
    fn card_declare(&self, from: usize, to: usize) {
        let Some(card) = &self.card else {
            return;
        };
        if from >= to {
            return;
        }
        let start = self.start as usize;
        for piece in self.pieces(from, to) {
            let Some(note) = card.written.get(piece) else {
                continue;
            };
            if note.load(Ordering::Acquire) {
                continue;
            }
            let (lo, hi) = self.piece(piece);
            if lo >= hi {
                continue;
            }
            if let Some(mirror) = &card.mirror {
                if !mirror.commit(lo, hi) {
                    // Left unheld: merges there are passed over
                    // (`Shadow::held`), and the piece is tried again at the
                    // next declaration.
                    if PC_SIDES[4].fetch_add(1, Ordering::Relaxed) == 0 {
                        tracing::warn!(
                            start = format_args!("{:#x}", self.start),
                            "a piece of a copy in the card's memory could not be committed: what shaders write there does not reach title memory"
                        );
                    }
                    continue;
                }
                // SAFETY: committed just now, inside the reservation (the
                // card side's size is at least the copy's length).
                crate::host::copy_mapped(unsafe { self.data.add(lo) }, (start + lo) as *const u8, hi - lo, false);
                self.card_changed(lo, hi);
            }
            PC_SIDES[1].fetch_add((hi - lo) as u64, Ordering::Relaxed);
            note.store(true, Ordering::Release);
        }
    }

    /// Whether the CPU's side holds offsets `[from, to)`: always, but for
    /// a mirror's pieces not held ([`no_pc_copy`]).
    fn held(&self, from: usize, to: usize) -> bool {
        let Some(card) = &self.card else {
            return true;
        };
        if card.mirror.is_none() || from >= to {
            return true;
        }
        self.pieces(from, to).all(|piece| card.written.get(piece).is_some_and(|w| w.load(Ordering::Acquire)))
    }

    /// Whether the copy's CPU side is a mirror ([`no_pc_copy`]).
    fn mirrored(&self) -> bool {
        self.card.as_ref().is_some_and(|c| c.mirror.is_some())
    }

    /// Offsets `[from, to)` of the copy cut where the CPU's side starts or
    /// stops holding them: (from, to, held), in order. One run, held,
    /// without a mirror.
    fn side_runs(&self, from: usize, to: usize) -> Vec<(usize, usize, bool)> {
        let Some(card) = self.card.as_ref().filter(|c| c.mirror.is_some()) else {
            return vec![(from, to, true)];
        };
        side_runs_of(self.start as usize, self.len as usize, &card.written, from, to)
    }

    /// With a mirror ([`no_pc_copy`]): where the bytes of offsets
    /// `[from, to)`, about to go up, are read from, as (from, to, address,
    /// in title memory): the mirror where it holds them, title memory
    /// elsewhere. `None` for a copy whose CPU side is a shared buffer.
    pub(crate) fn card_sources(&self, from: usize, to: usize) -> Option<Vec<(usize, usize, usize, bool)>> {
        if !self.mirrored() {
            return None;
        }
        let to = to.min(self.len as usize);
        let (start, data) = (self.start as usize, self.data as usize);
        Some(
            self.side_runs(from, to)
                .into_iter()
                .map(|(a, b, held)| if held { (a, b, data + a, false) } else { (a, b, start + a, true) })
                .collect(),
        )
    }

    /// Of a copy in the card's memory: the CPU's side if it is a shared
    /// buffer (with [`no_pc_copy`] none: [`Shadow::card_sources`]), the
    /// card side's bytes from before if made yet, and their size.
    pub(crate) fn card_sides(&self) -> Option<(Option<&ID3D12Resource>, Option<ID3D12Resource>, u64)> {
        let card = self.card.as_ref()?;
        let before = card.before.lock().unwrap_or_else(|e| e.into_inner()).clone();
        Some((card.staging.as_ref(), before, card.size))
    }

    /// Shader writes are declared for a copy in the card's memory: makes
    /// its "before" twin with `make` (bytes to buffer) if it has none.
    /// Whether one was made now: the caller fills it, whole, from the
    /// card's side before the submission's lists run. Without one (no
    /// memory) every piece of a declared range is fetched.
    pub(crate) fn card_make_before(&self, make: impl FnOnce(u64) -> Option<ID3D12Resource>) -> bool {
        let Some(card) = &self.card else {
            return false;
        };
        let mut before = card.before.lock().unwrap_or_else(|e| e.into_inner());
        if before.is_some() {
            return false;
        }
        *before = make(card.size);
        if before.is_some() {
            CARD_BYTES[1].fetch_add(card_hash_bytes(card.size), Ordering::Relaxed);
        }
        before.is_some()
    }

    /// The declared range `[address, end)` as the compare on the card
    /// takes it; `None` if none of it lies in the copy.
    pub(crate) fn card_span(&self, address: usize, end: usize) -> Option<CardSpan> {
        let (from, to) = self.inside(address, end.saturating_sub(address));
        (from < to).then(|| card_span_of(self.start as usize, from, to))
    }

    /// The pieces that hold offsets `[from, to)` of the copy (cut to it),
    /// as the card's hashing takes them.
    pub(crate) fn card_run(&self, from: usize, to: usize) -> CardSpan {
        let to = to.min(self.len as usize);
        card_span_of(self.start as usize, from.min(to.saturating_sub(1)), to)
    }

    /// The offsets of piece `i` of `span` that lie in the span.
    pub(crate) fn card_piece(&self, span: &CardSpan, i: u32) -> (usize, usize) {
        card_piece_of(self.start as usize, self.len as usize, span, i)
    }

    /// The walls trace (`crate::host::walls_trace`): a piece of `fetched`
    /// bytes of this copy came back from the card and `written` of them
    /// went into title memory. Summed per buffer and logged once a frame:
    /// which shader-written buffers the title's CPU could see fresh in
    /// that frame, and how much of each changed.
    pub(crate) fn walls_merged(&self, fetched: usize, written: usize) {
        if !crate::host::walls_trace() {
            return;
        }
        static SEEN: Mutex<(u64, BTreeMap<u64, (u64, u32, u64, u64)>)> = Mutex::new((0, BTreeMap::new()));
        let frame = crate::frame::current();
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        if seen.0 != frame {
            let buffers: Vec<String> = seen
                .1
                .iter()
                .map(|(at, v)| format!("{at:#x}+{:#x}: {} pieces, {} fetched, {} written", v.0, v.1, v.2, v.3))
                .take(32)
                .collect();
            if !buffers.is_empty() {
                tracing::info!(
                    frame = seen.0,
                    count = seen.1.len(),
                    buffers = ?buffers,
                    "walls trace: shader-written card buffers merged into title memory in this frame"
                );
            }
            seen.1.clear();
            seen.0 = frame;
        }
        let entry = seen.1.entry(self.start).or_insert((self.len, 0, 0, 0));
        entry.1 += 1;
        entry.2 += fetched as u64;
        entry.3 += written as u64;
    }

    /// After a submission, for a copy in the card's memory: `gpu` is what
    /// the card's side holds at offset `at` on, in a piece the compare
    /// found differing. The bytes shaders wrote (those that differ from the
    /// CPU's side, which is what the card's side held when they started)
    /// go into title memory, from the first to the last of them, as
    /// [`Shadow::copy_back`] puts them; the CPU's side takes the result and
    /// sends it up before the next submission. `title` is scratch memory.
    /// Returns the bytes written to title memory.
    pub(crate) fn card_merge(&self, at: usize, gpu: &[u8], title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        if self.card.is_none() || at + gpu.len() > self.len as usize || !self.held(at, at + gpu.len()) {
            return 0;
        }
        // SAFETY: inside the mapping (checked above); only the submitting
        // thread writes it.
        let old = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, old) else {
            return 0;
        };
        let last = last_difference(gpu, old).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, old) = (&gpu[first..=last], &mut old[first..=last]);
        title.clear();
        title.resize(n, 0);
        let mut written = 0;
        if crate::host::read_into((start + at) as *const u8, title) {
            // The stretch as read: what the card did not change is the
            // title's, and may have been written again since.
            let as_read = merge_race_on().then(|| title.clone());
            // As in `copy_back`: the card's byte where it is not what it
            // was before the submission, else the title's.
            let mut differs = 0u8;
            for ((t, g), o) in title.iter_mut().zip(gpu).zip(old.iter_mut()) {
                let wrote = if *g != *o { 0xffu8 } else { 0 };
                let merged = (*t & !wrote) | (*g & wrote);
                differs |= merged ^ *t;
                *t = merged;
                *o = merged;
            }
            if differs != 0 {
                count_skinned(skinned::MERGED, (start + at) as u64, (start + at + title.len()) as u64);
                crate::host::shaders_write(start + at, &title[..]);
                crate::watch::writing_exact(start + at, n);
                merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card");
                crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
                written = n;
            }
        } else {
            // Title memory is not mapped there: the copy keeps what the
            // card wrote, as before.
            old.copy_from_slice(gpu);
        }
        if merge_exact() {
            // The card's side holds `gpu`: only where the merge put other
            // bytes into the CPU's side does it take them.
            if let Some(first) = first_difference(gpu, old) {
                let last = last_difference(gpu, old).unwrap_or(first);
                self.card_changed(at + first, at + last + 1);
                MERGE_SENT[0].fetch_add((last + 1 - first) as u64, Ordering::Relaxed);
            }
            MERGE_SENT[1].fetch_add(n as u64, Ordering::Relaxed);
            return written;
        }
        // The card's side and its "before" take the result.
        self.card_changed(at, at + n);
        written
    }

    /// [`relay`], for a copy in the card's memory: `gpu` is what the
    /// card's side held at offset `at` on when a fetch sent after send
    /// `fetched_after` (`crate::host`'s order of the queue) ran, in a piece
    /// the compare found differing. The bytes that differ from the CPU's
    /// side are what shaders wrote; they go into title memory, from the
    /// first to the last of them, and into the CPU's side. Nothing is sent
    /// up afterwards (the card holds these bytes or newer ones: sending
    /// them up could undo a later flight's write), and the title's other
    /// bytes are left to the refresh.
    ///
    /// Left out: bytes of the CPU's side that hold title bytes the card
    /// had not received when the fetch ran, those still to go up and those
    /// that went up with a later send. There the card's side is the older
    /// one, and taking it for a shader's write would put it over what the
    /// title wrote. Under the CPU side's lock (the snapshot's, which such
    /// a copy has no other use for), as [`Shadow::merge_under_writes`].
    /// `title` is scratch memory. Returns the bytes written to title
    /// memory.
    pub(crate) fn card_merge_relay(&self, at: usize, gpu: &[u8], fetched_after: u64, title: &mut Vec<u8>) -> usize {
        let start = self.start as usize;
        let Some(card) = &self.card else {
            return 0;
        };
        if at + gpu.len() > self.len as usize || !self.held(at, at + gpu.len()) {
            return 0;
        }
        let _side = self.before.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: inside the mapping (checked above). The offsets are in a
        // range declared and not done with, which no refresh writes; what
        // does write it (the title's bytes put in under shader writes)
        // holds the lock held here.
        let held = unsafe { std::slice::from_raw_parts_mut(self.data.add(at), gpu.len()) };
        let Some(first) = first_difference(gpu, held) else {
            return 0;
        };
        let last = last_difference(gpu, held).unwrap_or(first);
        let (at, n) = (at + first, last + 1 - first);
        let (gpu, held) = (&gpu[first..=last], &mut held[first..=last]);
        let skip: Vec<(usize, usize)> = {
            let dirty = card.dirty.lock().unwrap_or_else(|e| e.into_inner());
            let sent = card.sent.lock().unwrap_or_else(|e| e.into_inner());
            dirty
                .iter()
                .copied()
                .chain(sent.iter().filter(|s| s.2 > fetched_after).map(|s| (s.0, s.1)))
                .filter(|&(a, b)| a < at + n && b > at)
                .collect()
        };
        title.clear();
        title.resize(n, 0);
        let mapped = crate::host::read_into((start + at) as *const u8, title);
        // The stretch as read: what the card did not change is the
        // title's, and may have been written again since.
        let as_read = (mapped && merge_race_on()).then(|| title.clone());
        let mut changed = false;
        for (a, b) in outside(at, at + n, skip) {
            let (a, b) = (a - at, b - at);
            // Title memory that is not mapped: the CPU's side keeps what
            // the card wrote, as before; `title` is scratch then.
            changed |= merge_piece(&gpu[a..b], &mut held[a..b], &mut title[a..b]);
        }
        if !mapped || !changed {
            return 0;
        }
        count_skinned(skinned::MERGED, (start + at) as u64, (start + at + title.len()) as u64);
        crate::host::shaders_write(start + at, &title[..]);
        if self.put_back_own(start + at, &title[..]) {
            return n;
        }
        crate::watch::writing_exact(start + at, n);
        merge_race(as_read.as_deref(), (self.start, self.len), start + at, &title[..], "card relay");
        crate::host::write_mapped((start + at) as *mut u8, title.as_ptr(), n);
        n
    }
}

struct Entry {
    end: u64,
    shadow: Option<Arc<Shadow>>,
}

/// Title buffers by (start address, resource), with their copy once made.
static BUFFERS: Mutex<BTreeMap<(u64, usize), Entry>> = Mutex::new(BTreeMap::new());

fn buffers() -> std::sync::MutexGuard<'static, BTreeMap<(u64, usize), Entry>> {
    BUFFERS.lock().unwrap_or_else(|e| e.into_inner())
}

/// A title buffer exists at `start` (`len` bytes).
pub(crate) fn register(resource: usize, start: u64, len: u64) {
    buffers().insert(
        (start, resource),
        Entry {
            end: start + len,
            shadow: None,
        },
    );
}

/// The title released it; lists that still hold its copy keep it alive.
pub(crate) fn unregister(resource: usize, start: u64) {
    buffers().remove(&(start, resource));
}

/// The title buffer holding `address`, as (start, length), without making
/// its copy (diagnostics: which buffer a dropped program's root view names).
#[cfg(feature = "diagnostics")]
pub(crate) fn extent(address: u64) -> Option<(u64, u64)> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(k, e)| (k.0, e.end - k.0))
}

/// Title memory `[at, at + len)` is about to show something else or be
/// given up: what is in place there (`crate::inplace`) is taken out of the
/// tables, so that the next ask makes a copy or opens the memory anew, and
/// goes when no list and no slot holds it any more. Returns how many.
pub(crate) fn in_place_let_go(at: u64, len: u64) -> u64 {
    let end = at.saturating_add(len);
    let mut gone: Vec<Arc<Shadow>> = Vec::new();
    {
        let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        let keys: Vec<(u64, u64)> = map
            .range((at.saturating_sub(WINDOW), 0)..(end, 0))
            .filter(|(key, shadow)| shadow.place.is_some() && key.0 + key.1 > at)
            .map(|(key, _)| *key)
            .collect();
        gone.extend(keys.into_iter().filter_map(|key| map.remove(&key)));
    }
    {
        // As `lookup` finds a buffer: among the 256 that start below.
        let mut map = buffers();
        for (_, entry) in map.range_mut(..(end, 0)).rev().take(256) {
            if entry.end > at && entry.shadow.as_ref().is_some_and(|s| s.place.is_some()) {
                gone.extend(entry.shadow.take());
            }
        }
    }
    let n = gone.len() as u64;
    for shadow in gone {
        crate::host::retire(shadow);
    }
    n
}

/// For the in-place census (`crate::inplace`): the title buffers (start,
/// bytes, false) and the windows outside buffers that have a copy (start,
/// bytes, true).
pub(crate) fn spans() -> Vec<(u64, u64, bool)> {
    let mut spans: Vec<(u64, u64, bool)> = buffers().iter().map(|(key, entry)| (key.0, entry.end - key.0, false)).collect();
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    spans.extend(loose.keys().map(|&(start, len)| (start, len, true)));
    spans
}

/// The copy of the innermost title buffer holding `[address, address +
/// len)`, made on first use. A `len` of 0 or 1 is an ask without a length
/// (a root view: the caller reads on to the copy's end).
pub(crate) fn find(address: u64, len: u64) -> Option<Arc<Shadow>> {
    lookup(address, len, false)
}

/// As [`find`], for an ask that is done with the copy when its list has
/// run (a draw's vertex and index buffers, indirect arguments): outside
/// buffer objects it takes a window that is there already and holds the
/// range ([`holder`]), and makes one of its own only if none does.
///
/// [`find`] names a window by the ask alone, which a descriptor slot needs
/// (the slot shows the window its view found when it was made, and the
/// pass that brings it up to date must find the same one). A draw needs no
/// such thing: it binds the copy it found and brings that copy up to date.
/// Named by the ask alone, every draw's range was a window of its own
/// wherever the windows are cut finer than 4 MiB: draws that bind
/// different ranges of the same memory then each copied their range, and
/// windows nothing asked for again for ten seconds were let go and made
/// (and copied) anew. Run 71, on a 256 KiB grid: 350 MB copied per frame
/// against 120 MB, 7,087 write-watch askings against 2,600, and 5,263
/// windows (4.5 GB) let go in two minutes against 393 in a whole run.
pub(crate) fn find_any(address: u64, len: u64) -> Option<Arc<Shadow>> {
    // With whole windows (the default) there is one window to a cell and
    // nothing to choose: exactly `find`.
    if !fine() {
        return find(address, len);
    }
    lookup(address, len, true)
}

/// Why [`sole_buffer_copy`] found none: the range lies in no title buffer
/// (memory outside buffer objects, whose windows it does not take), or
/// another copy holds some of its bytes.
pub(crate) enum NotSole {
    NoBuffer,
    OtherCopy,
}

/// For bytes the host would otherwise copy out of title memory on the CPU
/// (a constant view of a descriptor table, `crate::draw`): the copy of the
/// innermost title buffer holding `[address, address + len)`, as [`find`]
/// gives it (made on first use), if no other copy holds any of those bytes
/// (no other title buffer's copy over them, no window outside buffers).
/// Shaders that write those bytes then wrote this copy, so the card reads
/// what they wrote there, in the order it runs the lists.
/// shortcut: as [`in_place_let_go`], among the 256 buffers that start
/// below the range's end, and among windows that start at most 64 MiB
/// below it.
pub(crate) fn sole_buffer_copy(address: u64, len: u64) -> Result<Arc<Shadow>, NotSole> {
    let end = address.checked_add(len.max(1)).ok_or(NotSole::NoBuffer)?;
    {
        let map = buffers();
        // `lookup`'s choice, exactly.
        let key = map
            .range(..=(address, usize::MAX))
            .rev()
            .take(64)
            .find(|(_, e)| e.end >= end)
            .map(|(k, _)| *k)
            .ok_or(NotSole::NoBuffer)?;
        if map
            .range(..(end, 0))
            .rev()
            .take(256)
            .any(|(k, e)| *k != key && e.end > address && e.shadow.is_some())
        {
            return Err(NotSole::OtherCopy);
        }
    }
    {
        let map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
        if map
            .range((address.saturating_sub(64 << 20), 0)..(end, 0))
            .any(|(k, _)| k.0 + k.1 > address)
        {
            return Err(NotSole::OtherCopy);
        }
    }
    find(address, len).ok_or(NotSole::NoBuffer)
}

/// As [`sole_buffer_copy`], for bytes the card copies out of title
/// buffers' copies (`crate::host::table_gpu_copy`): the copy holding
/// `[address, address + len)` as one piece, or, where no one title buffer
/// holds the range, two pieces in two buffers side by side, each the sole
/// copy of its part (address, bytes). Memory outside buffers in the range
/// is [`NotSole::NoBuffer`]; a part some other copy holds too, or more
/// than two buffers, [`NotSole::OtherCopy`].
pub(crate) fn card_pieces(address: u64, len: u64) -> Result<Vec<(Arc<Shadow>, u64, u64)>, NotSole> {
    match sole_buffer_copy(address, len) {
        Ok(shadow) => return Ok(vec![(shadow, address, len.max(1))]),
        Err(NotSole::OtherCopy) => return Err(NotSole::OtherCopy),
        Err(NotSole::NoBuffer) => {}
    }
    let end = address.checked_add(len).ok_or(NotSole::NoBuffer)?;
    // The innermost buffer holding the first byte (the one starting last,
    // as `lookup` takes it) ends inside the range: the rest must lie in
    // one buffer from there on.
    let mid = extent_of(address).ok_or(NotSole::NoBuffer)?;
    if mid <= address || mid >= end {
        return Err(NotSole::NoBuffer);
    }
    let first = sole_buffer_copy(address, mid - address)?;
    let second = match sole_buffer_copy(mid, end - mid) {
        // Neither does a buffer from `mid` on hold the rest: three or more.
        Err(NotSole::NoBuffer) if extent_of(mid).is_some_and(|e| e < end) => return Err(NotSole::OtherCopy),
        other => other?,
    };
    if Arc::ptr_eq(&first, &second) {
        return Err(NotSole::OtherCopy);
    }
    Ok(vec![(first, address, mid - address), (second, mid, end - mid)])
}

/// The end of the innermost title buffer holding `address`, if one does.
fn extent_of(address: u64) -> Option<u64> {
    buffers()
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end > address)
        .map(|(_, e)| e.end)
}

fn lookup(address: u64, len: u64, any: bool) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    let mut map = buffers();
    let key = map
        .range(..=(address, usize::MAX))
        .rev()
        .take(64)
        .find(|(_, e)| e.end >= end)
        .map(|(k, _)| *k);
    let Some(key) = key else {
        drop(map);
        return loose(address, end, len <= 1, any);
    };
    let entry = map.get_mut(&key)?;
    if entry.shadow.is_none() {
        let len = entry.end - key.0;
        // Rounded up so that views ending in the buffer's last 256 bytes fit.
        // Shaders write some buffers (unordered access), so not upload memory.
        let size = len.next_multiple_of(1 << 16);
        // In place, if asked for and the memory can be opened
        // (`crate::inplace::place_for`): the buffer is the title's own
        // memory. Not the big ones `cards` puts in the card's memory: those
        // are read too often to stay in the PC's.
        let big = cards() && size >= card_limits().0;
        if !big && let Some(placed) = crate::inplace::place_for(key.0, len, false) {
            entry.shadow = Some(Arc::new(Shadow::placed(placed, key.0, len, false)));
            return entry.shadow.clone();
        }
        // A big one, with `cards`: shaders get a buffer in the card's own
        // memory, and the CPU's side is made beside it (`cpu_side`).
        let (buffer, data, card, bar) = match card_for(size) {
            Some(on_card) => {
                let Some((staging, mirror, data)) = cpu_side(size) else {
                    CARD_BYTES[0].fetch_sub(size, Ordering::Relaxed);
                    return None;
                };
                if let Some(staging) = &staging {
                    crate::host::name(staging, || format!("title buffer {:#x}+{len:#x}", key.0));
                }
                crate::host::name(&on_card, || format!("title buffer {:#x}+{len:#x} on the card", key.0));
                tracing::info!(
                    start = format_args!("{:#x}", key.0),
                    len,
                    mirror = mirror.is_some(),
                    "a buffer's host copy is in the card's memory"
                );
                let card = Card {
                    staging,
                    mirror,
                    written: written_notes(key.0, len),
                    before: Mutex::default(),
                    dirty: Mutex::default(),
                    sent: Mutex::default(),
                    size,
                };
                (on_card, data, Some(card), None)
            }
            None => {
                let (buffer, data, bar) = pc_copy_buffer(size, false)?;
                crate::host::name(&buffer, || {
                    format!("title buffer {:#x}+{len:#x}{}", key.0, if bar.is_some() { " (GPU upload heap)" } else { "" })
                });
                (buffer, data, None, bar)
            }
        };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        if len >= 16 << 20 {
            tracing::info!(
                start = format_args!("{:#x}", key.0),
                len,
                "large buffer copied to the host"
            );
        } else {
            tracing::debug!(
                start = format_args!("{:#x}", key.0),
                len,
                "buffer copied to the host"
            );
        }
        let mut shadow = Shadow::new(buffer, data, gpu, key.0, len);
        shadow.card = card;
        shadow.bar = bar;
        let shadow = Arc::new(shadow);
        if shadow.card.is_some() {
            CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
        }
        entry.shadow = Some(shadow);
    }
    entry.shadow.clone()
}

/// Titles also use memory outside any buffer object as GPU memory (
/// the title's interface vertices, from its own allocator). Such memory
/// gets copies of the aligned windows it touches.
/// A window nothing names any more is let go ([`let_go_idle`]).
/// shortcut: a window is 4 MiB however little of it is read (the memory
/// line says how much was ever asked for, and how much was ever copied);
/// [`window`] cuts the windows of asks that give a length on a finer grid.
const WINDOW: u64 = 4 << 20;

/// The grid the windows of asks with a length are cut on, in bytes: 4 MiB
/// unless `VICTORY_WINDOW_KB` names a smaller power of two (64 at least).
/// A window is the PC's memory, all of it, however little of it is read:
/// 803 windows held 3.3 GB in run 60's gameplay, and all that was ever
/// asked for in them lay within 1.6 GB. On a grid of 1 MiB or 256 KiB a
/// window ends nearer to what is read. An ask without a length keeps the
/// 4 MiB grid whatever is set: a root view reads from its address to its
/// window's end, and mesh streams bound that way reach megabytes.
/// shortcut: the default stays 4 MiB until a run with the memory line's
/// window counts has shown what the finer grids give; the same memory
/// asked for with different ranges then lies in more than one window.
fn window() -> u64 {
    static GRID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *GRID.get_or_init(|| {
        let kb = std::env::var("VICTORY_WINDOW_KB").ok().and_then(|v| v.trim().parse::<u64>().ok());
        grid(kb)
    })
}

/// The window grid in bytes for a setting in KiB: a power of two from
/// 64 KiB to 4 MiB, 4 MiB for anything else.
fn grid(kb: Option<u64>) -> u64 {
    match kb {
        Some(kb) if kb.is_power_of_two() && (64..=4096).contains(&kb) => kb << 10,
        _ => WINDOW,
    }
}

/// The window (start, bytes) of title memory for an ask of `[address,
/// end)` on a grid of `grid` bytes.
fn window_of(address: u64, end: u64, grid: u64) -> (u64, u64) {
    let start = address & !(grid - 1);
    (start, end.next_multiple_of(grid) - start)
}

/// Whether the window of an ask without a length begins at the ask instead
/// of at the 4 MiB line below it. Such an ask (a root view) reads from its
/// address to the end of its 4 MiB cell; what lies below the lowest
/// address ever asked for in the cell is memory the host zeroed and nothing
/// reads (513 of run 65's 782 windows were asked for that way, 2,052 MB;
/// of all windows' 3,236 MB, 1,988 MB were ever copied). With this on, the
/// window is the cell's tail from the ask's step on ([`TAIL_STEP`]); an ask
/// lower in the cell later makes a longer tail, which every ask from then
/// on takes, and the shorter one goes when nothing holds it any more.
/// shortcut: off unless `VICTORY_WINDOW_TAILS=1`. A longer tail is another
/// host buffer than the shorter one: what was bound to the shorter stays
/// right only because every list brings up to date the copy it bound
/// (`draw.rs` keeps a root view's copy with its slot). It pays only
/// together with `VICTORY_WINDOW_KB` below 4096: on the 4 MiB grid an ask
/// with a length in the same cell makes the whole cell's window, and the
/// tail is that.
fn tails() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WINDOW_TAILS").is_some_and(|v| v == "1"))
}

/// Tails begin on multiples of this. A longer tail is a new host buffer,
/// filled from title memory from its first byte to its last: begun on the
/// 64 KiB page of each ask, a cell whose asks came from the top down made
/// up to 63 of them, one after the other (part of run 71's copies). On
/// this step it is 15 at most, and a tail holds up to 192 KiB it did not
/// need.
const TAIL_STEP: u64 = 256 << 10;

/// The window (start, bytes) a tail ask at `address` makes: from the ask's
/// step ([`TAIL_STEP`]) to the end of its 4 MiB cell.
fn tail_of(address: u64, end: u64) -> (u64, u64) {
    let (cell, len) = window_of(address, end, WINDOW);
    let start = address & !(TAIL_STEP - 1);
    (start, cell + len - start)
}

/// Of `windows` (start, bytes), the one an ask of `[address, end)` that
/// may take any window takes ([`find_any`]): the longest that holds the
/// whole range.
fn holder(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    windows
        .filter(|&(start, bytes)| start <= address && start + bytes >= end)
        .max_by_key(|&(start, bytes)| (bytes, std::cmp::Reverse(start)))
}

/// Whether windows may be smaller than their 4 MiB cell (a finer grid, or
/// tails): only then is there more than one window to choose from.
fn fine() -> bool {
    tails() || window() < WINDOW
}

/// For the memory line: the bytes of title memory the windows alive lie
/// over (each byte once), beside the windows' own bytes in
/// [`memory`]: the difference is memory that is in more than one window,
/// which is copied once for each.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_cover() -> u64 {
    let loose = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    cover(loose.keys().copied())
}

/// The bytes `windows` (start, bytes; in order of start) lie over, each
/// byte once.
#[cfg_attr(not(any(test, feature = "diagnostics")), allow(dead_code))]
fn cover(windows: impl Iterator<Item = (u64, u64)>) -> u64 {
    let (mut covered, mut reached) = (0, 0);
    for (start, bytes) in windows {
        let end = start + bytes;
        if end > reached {
            covered += end - start.max(reached);
            reached = end;
        }
    }
    covered
}

/// Of `windows` (start, bytes; in order), the longest tail a tail ask at
/// `address` can take: the first that begins in the ask's cell at or below
/// the ask's page and ends where the cell ends.
fn longest_tail(windows: impl Iterator<Item = (u64, u64)>, address: u64, end: u64) -> Option<(u64, u64)> {
    let (cell, len) = window_of(address, end, WINDOW);
    let step = address & !(TAIL_STEP - 1);
    windows
        .filter(|&(start, bytes)| start >= cell && start <= step && start + bytes == cell + len)
        .min()
}

/// The copy a draw's ask for `[address, address + len)` finds now
/// ([`find_any`]), with none made: the innermost title buffer's copy if it
/// has one (none if not made yet), else the window that would be taken.
/// For the heap pass's look at where picked views write
/// ([`picked_elsewhere`]).
pub(crate) fn peek(address: u64, len: u64) -> Option<Arc<Shadow>> {
    let end = address.checked_add(len.max(1))?;
    {
        let map = buffers();
        if let Some((_, entry)) = map.range(..=(address, usize::MAX)).rev().take(64).find(|(_, e)| e.end >= end) {
            return entry.shadow.clone();
        }
    }
    let map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, window());
    let key = if fine() {
        holder(map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key), address, end).unwrap_or(named)
    } else {
        named
    };
    map.get(&key).cloned()
}

/// Whether `shadow` is a copy of a single view's own bytes ([`exact`]).
pub(crate) fn is_own_bytes(shadow: &Arc<Shadow>) -> bool {
    let (start, len) = shadow.span();
    EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.get(&(start, len)).is_some_and(|(s, _)| Arc::ptr_eq(s, shadow))
}

/// Of the heap's unordered access buffer views at the last measure of the
/// heap pass: views whose copy is not the one a draw asking for the same
/// bytes finds now ([`peek`]), so that what shaders write through them
/// never reaches what draws bind (nothing copies one host copy into
/// another; only declared writes go back through title memory); their
/// bytes; of them views in a window, in a copy of their own bytes
/// ([`exact`]), and views whose bytes draws would find no copy for yet;
/// views looked at. Set, not summed.
pub(crate) static PICKED_ELSEWHERE: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Copies of title memory outside buffers, by (start, length).
static LOOSE: Mutex<BTreeMap<(u64, u64), Arc<Shadow>>> = Mutex::new(BTreeMap::new());

fn loose(address: u64, end: u64, unsized_ask: bool, any: bool) -> Option<Arc<Shadow>> {
    let mut map = LOOSE.lock().unwrap_or_else(|e| e.into_inner());
    let (cell, _) = window_of(address, end, WINDOW);
    let named = window_of(address, end, if unsized_ask { WINDOW } else { window() });
    let (start, len) = if unsized_ask && tails() {
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        longest_tail(there, address, end).unwrap_or_else(|| tail_of(address, end))
    } else if any && fine() && !unsized_ask {
        // The windows that begin in the ask's cell at or below the ask (one
        // that begins in a cell below and reaches this far is passed over:
        // the ask then makes its own).
        let there = map.range((cell, 0)..=(address, u64::MAX)).map(|(key, _)| *key);
        holder(there, address, end).unwrap_or(named)
    } else {
        named
    };
    // Under the lock, with the handing out: see `let_go_idle`.
    let note = |shadow: &Shadow| {
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        shadow.asked[0].fetch_min(address, Ordering::Relaxed);
        shadow.asked[1].fetch_max(end, Ordering::Relaxed);
        if unsized_ask {
            shadow.unsized_asks.fetch_add(1, Ordering::Relaxed);
        }
    };
    if let Some(shadow) = map.get(&(start, len)) {
        note(shadow);
        return Some(shadow.clone());
    }
    // In place, if asked for and the window's memory can be opened: the
    // window is the title's own memory, with nothing to copy.
    if let Some(placed) = crate::inplace::place_for(start, len, true) {
        let shadow = Arc::new(Shadow::placed(placed, start, len, true));
        note(&shadow);
        map.insert((start, len), shadow.clone());
        return Some(shadow);
    }
    // With room for windows on the card (`window_card_most`): shaders get a
    // buffer there, and the CPU's side is made beside it (`cpu_side`).
    let (buffer, data, card, bar) = match window_card_for(len) {
        Some(on_card) => {
            let Some((staging, mirror, data)) = cpu_side(len) else {
                WINDOW_CARD_BYTES.fetch_sub(len, Ordering::Relaxed);
                return None;
            };
            if let Some(staging) = &staging {
                crate::host::name(staging, || format!("title memory {start:#x}+{len:#x}"));
            }
            crate::host::name(&on_card, || format!("title memory {start:#x}+{len:#x} on the card"));
            let card = Card {
                staging,
                mirror,
                written: written_notes(start, len),
                before: Mutex::default(),
                dirty: Mutex::default(),
                sent: Mutex::default(),
                size: len,
            };
            (on_card, data, Some(card), None)
        }
        None => {
            let (buffer, data, bar) = pc_copy_buffer(len, true)?;
            crate::host::name(&buffer, || {
                format!("title memory {start:#x}+{len:#x}{}", if bar.is_some() { " (GPU upload heap)" } else { "" })
            });
            (buffer, data, None, bar)
        }
    };
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    tracing::debug!(
        start = format_args!("{start:#x}"),
        len,
        "loose memory copied to the host"
    );
    let mut shadow = Shadow::new(buffer, data, gpu, start, len);
    shadow.window = true;
    shadow.card = card;
    shadow.bar = bar;
    let shadow = Arc::new(shadow);
    if shadow.card.is_some() {
        CARDED.lock().unwrap_or_else(|e| e.into_inner()).push(Arc::downgrade(&shadow));
    }
    note(&shadow);
    map.insert((start, len), shadow.clone());
    Some(shadow)
}

/// Copies of single buffer views' own bytes ([`exact`]), by (start,
/// bytes), each with the asking that last took it; and the askings so far.
#[allow(clippy::type_complexity)]
static EXACT: Mutex<(u64, BTreeMap<(u64, u64), (Arc<Shadow>, u64)>)> = Mutex::new((0, BTreeMap::new()));

/// Since the last frame line: askings for such copies, and copies made.
static EXACT_STATS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// A copy of the title memory `[address, address + len)` that starts at
/// `address` itself: for a buffer view whose bytes do not start on an
/// element of the copy its buffer has (`descriptor/host.rs` `own_bytes`).
/// A host view counts its first element from its buffer's start; a console
/// view starts at any address, and the title places views at ring
/// positions inside its big GPU heap. Such a view was refused and its slot
/// read zeros ("buffer view not on an element of its host copy", in every
/// run; upstream VIctory 86a5ed6 found its meshlet culling constants and
/// colour-bake parameter blocks among them and gives each a copy of its
/// own bytes, as here).
///
/// The copy is brought up to date and written back like any other (a list
/// refreshes the copy its view's slot shows, `buffer_copy`). Kept by
/// (start, bytes) for as long as a list or a slot holds it; of the others,
/// those not asked for lately go once there are many (ring positions
/// change every frame).
pub(crate) fn exact(address: u64, len: u64) -> Option<Arc<Shadow>> {
    /// Copies kept before the idle ones are looked for, and the askings an
    /// idle one stays for.
    const MANY: usize = 256;
    const LATELY: u64 = 512;
    if len == 0 {
        return None;
    }
    EXACT_STATS[0].fetch_add(1, Ordering::Relaxed);
    let mut guard = EXACT.lock().unwrap_or_else(|e| e.into_inner());
    let (tick, map) = &mut *guard;
    *tick += 1;
    let now = *tick;
    if let Some((shadow, used)) = map.get_mut(&(address, len)) {
        *used = now;
        shadow.used.store(crate::host::moment(), Ordering::Relaxed);
        return Some(shadow.clone());
    }
    if map.len() >= MANY {
        // Held by the map alone (no list), its buffer by the copy alone (no
        // slot shows it), and not asked for lately.
        map.retain(|_, (shadow, used)| {
            Arc::strong_count(shadow) > 1 || crate::host::references(&shadow.buffer) > 1 || *used + LATELY > now
        });
    }
    let (buffer, data, bar) = pc_copy_buffer(len.next_multiple_of(256), false)?;
    crate::host::name(&buffer, || format!("title view {address:#x}+{len:#x}"));
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    let mut shadow = Shadow::new(buffer, data, gpu, address, len);
    shadow.bar = bar;
    let shadow = Arc::new(shadow);
    map.insert((address, len), (shadow.clone(), now));
    EXACT_STATS[1].fetch_add(1, Ordering::Relaxed);
    Some(shadow)
}

/// With the frame line: the buffer views that got a copy of their own
/// bytes ([`exact`]), per frame over `frames`, and the copies kept now.
/// Nothing while there are none.
pub(crate) fn exact_line(frames: f64) {
    let [asked, made] = [0, 1].map(|i| EXACT_STATS[i].swap(0, Ordering::Relaxed));
    if asked == 0 {
        return;
    }
    let kept = EXACT.lock().unwrap_or_else(|e| e.into_inner()).1.len();
    tracing::info!(
        "buffer views off their copy's elements (they were refused and read zeros; VICTORY_VIEW_OWN_COPY=0 refuses them again): \
         {:.1} asked for per frame, {:.1} copies of their own bytes made per frame, {kept} kept",
        asked as f64 / frames,
        made as f64 / frames
    );
}

#[cfg(test)]
mod window_tests {
    use super::{WINDOW, cover, grid, holder, longest_tail, tail_of, window_of};

    /// An ask that may take any window takes the longest that holds all of
    /// it, and none that holds only a part.
    #[test]
    fn an_ask_takes_the_longest_window_that_holds_it() {
        let cell = 0xd_1200_0000u64;
        let (ask, end) = (cell + 0x9_0000, cell + 0xb_0000);
        assert_eq!(holder([].into_iter(), ask, end), None);
        let there = [
            (cell, 0x4_0000),            // ends below the ask
            (cell + 0x8_0000, 0x2_0000), // holds its first half only
            (cell + 0x8_0000, 0x4_0000), // holds it
            (cell + 0x4_0000, 0x10_0000), // holds it, longer
            (cell + 0xa_0000, 0x40_0000), // begins above it
        ];
        assert_eq!(holder(there.into_iter(), ask, end), Some((cell + 0x4_0000, 0x10_0000)));
        // Of two as long, the one that begins lower.
        let two = [(cell + 0x8_0000, 0x8_0000), (cell + 0x4_0000, 0x8_0000)];
        assert_eq!(holder(two.into_iter(), ask, end), Some((cell + 0x4_0000, 0x8_0000)));
    }

    /// What windows lie over counts each byte once.
    #[test]
    fn overlapping_windows_cover_their_memory_once() {
        assert_eq!(cover([].into_iter()), 0);
        let apart = [(0x10_0000u64, 0x4_0000u64), (0x20_0000, 0x4_0000)];
        assert_eq!(cover(apart.into_iter()), 0x8_0000);
        // One inside another, one overlapping its end, one touching.
        let over = [
            (0x10_0000u64, 0x10_0000u64),
            (0x14_0000, 0x4_0000),
            (0x1c_0000, 0x8_0000),
            (0x24_0000, 0x4_0000),
        ];
        assert_eq!(cover(over.into_iter()), 0x18_0000);
    }

    /// A tail runs from the ask's page to the end of its 4 MiB cell, and an
    /// ask takes the longest tail there is that holds it.
    #[test]
    fn tails_run_to_the_cells_end_and_the_longest_is_taken() {
        let cell = 0xd_1200_0000u64;
        let ask = cell + 0x12_3456;
        assert_eq!(tail_of(ask, ask + 1), (cell + 0x10_0000, WINDOW - 0x10_0000));
        // The cell's first page: the whole cell, the window there always was.
        assert_eq!(tail_of(cell + 0xffff, cell + 0x1_0000), (cell, WINDOW));
        let end = cell + WINDOW;
        let tail = |from: u64| (cell + from, WINDOW - from);
        // Nothing there, or only tails that begin above the ask.
        assert_eq!(longest_tail([].into_iter(), ask, ask + 1), None);
        assert_eq!(longest_tail([tail(0x20_0000)].into_iter(), ask, ask + 1), None);
        // The longest of those that hold the ask; windows that do not reach
        // the cell's end (cut on a finer grid for asks with a length) and
        // windows of other cells are not tails.
        let there = [
            (cell - WINDOW, WINDOW),
            (cell, 0x4_0000),
            tail(0x8_0000),
            tail(0x10_0000),
            (cell + 0x10_0000, 0x4_0000),
            tail(0x20_0000),
        ];
        assert_eq!(longest_tail(there.into_iter(), ask, ask + 1), Some(tail(0x8_0000)));
        let (start, bytes) = tail(0x8_0000);
        assert!(start <= ask && start + bytes == end);
        // The whole cell's window is the longest tail of all.
        let with_whole = [(cell, WINDOW), tail(0x8_0000)];
        assert_eq!(longest_tail(with_whole.into_iter(), ask, ask + 1), Some((cell, WINDOW)));
    }

    /// The setting takes powers of two from 64 KiB to 4 MiB; anything else
    /// leaves the grid as it was.
    #[test]
    fn the_grid_is_a_power_of_two_or_the_default() {
        assert_eq!(grid(None), WINDOW);
        assert_eq!(grid(Some(4096)), WINDOW);
        assert_eq!(grid(Some(1024)), 1 << 20);
        assert_eq!(grid(Some(256)), 256 << 10);
        assert_eq!(grid(Some(64)), 64 << 10);
        for bad in [0, 32, 100, 1000, 8192] {
            assert_eq!(grid(Some(bad)), WINDOW, "{bad}");
        }
    }

    /// A window holds its ask and is cut on the grid; on the default grid
    /// it is the window there always was.
    #[test]
    fn windows_hold_their_ask_on_the_grid() {
        let (address, end) = (0xd_1234_5678u64, 0xd_1234_5678u64 + 0x9000);
        assert_eq!(window_of(address, end, WINDOW), (0xd_1200_0000, 4 << 20));
        assert_eq!(window_of(address, end, 1 << 20), (0xd_1230_0000, 1 << 20));
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 256 << 10));
        // An ask across a grid line takes both cells.
        let end = address + 0x4_0000;
        assert_eq!(window_of(address, end, 256 << 10), (0xd_1234_0000, 512 << 10));
        for g in [64u64 << 10, 256 << 10, 1 << 20, WINDOW] {
            let (start, len) = window_of(address, end, g);
            assert!(start <= address && start + len >= end);
            assert!(start % g == 0 && len % g == 0);
        }
    }
}

#[cfg(test)]
mod card_merge_tests {
    use super::{Card, Shadow};
    use std::sync::Mutex;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;

    /// A plain buffer on the software adapter: what a copy's `buffer` and
    /// its card's `staging` are; the merge never touches them.
    fn warp_buffer() -> Option<ID3D12Resource> {
        // SAFETY: plain DXGI and D3D12 creation calls on the software adapter.
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
            let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device).ok()?;
            let device = device?;
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 64 << 10,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .ok()?;
            buffer
        }
    }

    /// The mixed landing the fetch inside flights must survive, on a real
    /// copy in the card's memory and its merge (`Shadow::card_merge_relay`):
    /// flights 5, 6, 7 in the air; 5 lands the old way with the queue's
    /// count at 7; the title wrote a byte that went up in send 7 into a
    /// piece flight 6's shaders write; 6 then lands from its own slot, which
    /// holds that byte as it was before send 7. The notes of sends are
    /// dropped as `card_dirty` drops them, below the mark the old-way fetch
    /// sets (`crate::host::fallback_fetched_mark`): with the mark at the
    /// landed plans' send the title's byte survives and the shader's write
    /// lands; with the mark at the fetch's count (as before the fix) the
    /// title's byte is put back to the card's older one.
    #[test]
    fn a_title_byte_sent_after_a_flight_survives_its_slot() {
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        let (title_byte, shader_byte) = (100usize, 200usize);
        let content: Vec<u8> = (0..LEN).map(|i| (i * 7 + 3) as u8).collect();
        let land = |mark: u64| -> Vec<u8> {
            let mut title = content.clone();
            let mut side = content.clone();
            // The title's byte, refreshed into the CPU's side and sent up in send 7.
            title[title_byte] ^= 0x5a;
            side[title_byte] = title[title_byte];
            let mut shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64);
            shadow.card = Some(Card {
                staging: Some(staging.clone()),
                mirror: None,
                written: super::written_notes(title.as_ptr() as u64, LEN as u64),
                before: Mutex::new(None),
                dirty: Mutex::default(),
                sent: Mutex::new(vec![(title_byte, title_byte + 1, 7)]),
                // 0: no card bytes counted (`Drop` takes them off again).
                size: 0,
            });
            // As `card_dirty` with the mark set by the old-way fetch.
            if let Some(card) = &shadow.card {
                card.sent.lock().unwrap().retain(|s| s.2 > mark);
            }
            // Flight 6's slot: the card's bytes behind its lists, before send 7.
            let mut slot = content.clone();
            slot[shader_byte] ^= 0xc3;
            let written = shadow.card_merge_relay(0, &slot, 6, &mut Vec::new());
            assert!(written > 0);
            drop(shadow);
            title
        };
        let fixed = land(crate::host::fallback_fetched_mark(true, 7, Some(5)));
        assert_eq!(fixed[title_byte], content[title_byte] ^ 0x5a, "the title's byte was lost");
        assert_eq!(fixed[shader_byte], content[shader_byte] ^ 0xc3, "the shader's write did not land");
        // The mark at the fetch's count: the note of send 7 is gone and the
        // older card byte is taken for a shader's write.
        let before = land(7);
        assert_eq!(before[title_byte], content[title_byte]);
    }

    /// The refresh of a 4 KiB part the title wrote one byte of, in a copy
    /// in the card's memory whose card side a shader wrote elsewhere in
    /// that part through a picked view: as before (`VICTORY_PICKED_GUARD`
    /// below 2) the whole part goes up and the shader's byte on the card
    /// is overwritten with the old one; with `2` only the title's byte
    /// goes up.
    #[test]
    fn a_refresh_sends_up_only_what_the_title_changed_in_a_picked_span() {
        let (Some(buffer), Some(staging)) = (warp_buffer(), warp_buffer()) else {
            return;
        };
        const LEN: usize = 64 << 10;
        const PART: usize = 4096;
        let content: Vec<u8> = (0..LEN).map(|i| (i * 13 + 5) as u8).collect();
        let sent = |exact: bool| -> (Vec<(usize, usize)>, Vec<u8>) {
            let mut title = content.clone();
            // The CPU side holds what went up; the card's side has a
            // shader's byte at 200 besides (not seen here).
            let mut side = content.clone();
            title[100] ^= 0x5a;
            let mut shadow = Shadow::new(buffer.clone(), side.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64);
            shadow.card = Some(Card {
                staging: Some(staging.clone()),
                mirror: None,
                written: super::written_notes(title.as_ptr() as u64, LEN as u64),
                before: Mutex::new(None),
                dirty: Mutex::default(),
                sent: Mutex::default(),
                size: 0,
            });
            let start = title.as_ptr() as usize;
            shadow.mark_picked_uav(start as u64, LEN as u64);
            shadow.copy_picked_card(start, start + PART, exact);
            let dirty = shadow.card.as_ref().map(|c| c.dirty.lock().unwrap().clone()).unwrap_or_default();
            drop(shadow);
            (dirty, side)
        };
        let (dirty, side) = sent(false);
        assert_eq!(dirty, vec![(0, PART)], "the whole part went up, over the shader's byte");
        assert_eq!(side[100], content[100] ^ 0x5a);
        let (dirty, side) = sent(true);
        assert_eq!(dirty, vec![(100, 101)], "only the title's byte goes up");
        assert_eq!(side[100], content[100] ^ 0x5a);
        assert_eq!(&side[..100], &content[..100]);
    }

    /// A copy without a card side holds the shaders' bytes beside the
    /// title's: the first refresh of a part copies it whole and keeps the
    /// title's bytes apart; at the next, with `exact`, only what the title
    /// changed since is written, and a shader's byte beside it stays. As
    /// before (not exact) the shader's byte is overwritten.
    #[test]
    fn a_copy_without_a_card_side_keeps_shader_bytes_the_title_did_not_touch() {
        let Some(buffer) = warp_buffer() else {
            return;
        };
        const LEN: usize = 64 << 10;
        const PART: usize = 4096;
        let content: Vec<u8> = (0..LEN).map(|i| (i * 31 + 9) as u8).collect();
        let refreshed = |exact: bool| -> Vec<u8> {
            let mut title = content.clone();
            let mut data = vec![0u8; LEN];
            let shadow = Shadow::new(buffer.clone(), data.as_mut_ptr(), 0, title.as_ptr() as u64, LEN as u64);
            let start = title.as_ptr() as usize;
            shadow.mark_picked_uav(start as u64 + 16, (LEN - 16) as u64);
            // The first refresh takes the part whole (from the span's start on).
            shadow.copy_picked_pc(start + 16, start + 16 + PART, exact);
            assert_eq!(&data[16..16 + PART], &title[16..16 + PART]);
            // A shader writes 200, the title writes 100; the part is refreshed.
            data[200] ^= 0xc3;
            title[100] ^= 0x5a;
            shadow.copy_picked_pc(start + 16, start + 16 + PART, exact);
            drop(shadow);
            assert_eq!(data[100], title[100], "the title's byte is in");
            data
        };
        let data = refreshed(true);
        assert_eq!(data[200], content[200] ^ 0xc3, "the shader's byte stays");
        let data = refreshed(false);
        assert_eq!(data[200], content[200], "as before: the shader's byte is overwritten");
    }

    /// `VICTORY_DECLARED_WIDE`: the parts of a span free of shader writes
    /// when a declaration is made, planned after it, are moved whole, the
    /// declared range among them; a plan of the span after the declaration
    /// leaves the declared range out. A range an earlier flight writes
    /// stays out of both.
    #[test]
    fn parts_found_at_a_declaration_are_moved_after_it() {
        let Some(buffer) = warp_buffer() else {
            return;
        };
        const LEN: usize = 64 << 10;
        let title: Vec<u8> = (0..LEN).map(|i| (i * 13 + 7) as u8).collect();
        let mut data = vec![0u8; LEN];
        let start = title.as_ptr() as usize;
        let shadow = std::sync::Arc::new(Shadow::new(buffer, data.as_mut_ptr(), 0, start as u64, LEN as u64));
        let flight = (start + 100, start + 200);
        let declared = (start + 1000, start + 3000);
        shadow.pending.lock().unwrap().push(flight);
        let parts = shadow.free_parts(start, LEN);
        assert_eq!(parts, vec![(start, flight.0), (flight.1, start + LEN)]);
        // The declaration.
        shadow.pending.lock().unwrap().push(declared);
        let mut after = Vec::new();
        shadow.plan(start, LEN, &mut after);
        assert!(!after.is_empty());
        assert!(after.iter().all(|&(a, b)| b <= declared.0 || a >= declared.1), "{after:?}");
        let mut runs = Vec::new();
        shadow.plan_parts(&parts, &mut runs);
        let plans = vec![(shadow.clone(), runs)];
        let (moved, _) = super::move_wide(&plans);
        shadow.moved(&moved[0]);
        drop(plans);
        drop(shadow);
        assert_eq!(&data[..100], &title[..100]);
        assert!(data[100..200].iter().all(|&b| b == 0), "the flight's range is left alone");
        assert_eq!(&data[200..], &title[200..], "the declared range is in");
    }

    #[test]
    fn runs_closer_than_the_gap_go_as_one() {
        assert_eq!(super::close_runs(vec![(0, 4), (10, 2), (40, 1)]), vec![(0, 12), (40, 1)]);
        assert_eq!(super::close_runs(Vec::new()), Vec::new());
    }
}

/// For device-removed reports: the copy of title memory whose host range
/// starts closest below host GPU address `va`, and where `va` is in it.
pub(crate) fn near(va: u64) -> String {
    let mut all: Vec<(u64, u64, u64)> = buffers()
        .values()
        .filter_map(|e| e.shadow.as_deref().map(|s| (s.gpu, s.start, s.len)))
        .collect();
    all.extend(
        LOOSE
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .map(|s| (s.gpu, s.start, s.len)),
    );
    let below = all.iter().filter(|c| c.0 <= va).max();
    let above = all.iter().filter(|c| c.0 > va).min();
    let mut text = match below {
        Some(&(gpu, start, len)) => format!(
            "; the nearest title memory copy below it: {start:#x}+{len:#x} at host {gpu:#x}, offset {:#x}",
            va - gpu
        ),
        None => "; no title memory copy below it".into(),
    };
    if let Some(&(gpu, start, len)) = above {
        text.push_str(&format!(
            "; the nearest above: {start:#x}+{len:#x} at host {gpu:#x} ({} copies in all)",
            all.len()
        ));
    }
    text
}
