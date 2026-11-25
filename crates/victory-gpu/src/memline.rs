//! The host memory line, in every build: where the process's memory and
//! the system's commit go, by what holds them, logged with the frame-rate
//! line every five seconds (`crate::host`'s meter). Diagnostics builds
//! log a longer account beside it (`crate::copystat`).
//!
//! What the system charges for the run, beside other programs:
//! - the process's private bytes: the title's own heaps (`xmem`'s
//!   allocations) and everything of ours in the PC's memory (host copies
//!   of title memory, snapshots, staging, lists, shader tables);
//! - its shared commit: the title's page pool (`xmem`'s physical pages, a
//!   section committed page by page as the title maps them), not private;
//! - what the card holds for us (host textures, copies in the card's
//!   memory, twins) IS in the private bytes and the system's commit, byte
//!   for byte: `crate::bus` `probe_heaps` on this PC (RTX 5070 Ti) saw
//!   +256 MB of private bytes and of the system's commit for 256 MiB in a
//!   default heap, as for a shared buffer, an upload heap and a GPU upload
//!   heap. So of runs n48-n52's 25.2 GB private, 12.6 GB is the card's in
//!   use (the line's "the card: ..."), and the PC's own ~12.6 GB.
//!
//! Where the PC's own ~12.5 GB goes, from a walk of run n56's address space
//! from outside at its end (24.7 GB private, the card 12.3 GB):
//! - 4.6 GB the title's own heaps (`xmem` `XMemVirtualAlloc`, committed
//!   whole by the title: 2,199, 1,172, 596 and 563 MB at console addresses);
//! - 3.7 GB windows of title memory outside buffers, copies of those heaps'
//!   bytes: 2.55 GB of them in the card's memory (GPU upload heaps,
//!   `VICTORY_GPU_UPLOAD_COPIES`), the rest shared buffers in the PC's
//!   memory. One copy each: a window in a GPU upload heap has no second
//!   side in the PC's memory. `VICTORY_INPLACE=1` opens the heaps' parts
//!   in place instead (no copy at all); the line's "asked within them"
//!   says what windows cut to their asks would hold;
//!   `inplace` `card_probe_in_place` on this PC's card: parts of private
//!   memory open (`ID3D12Device13`), the card reads and writes them, a
//!   64 MiB part opens in 14 ms and copies out as fast as a shared buffer;
//!   while open, the title cannot make those pages read-only or decommit
//!   them. A default-heap buffer of 256 MiB adds 256 MB of write-combined
//!   private memory (`probe_list_memory`): the card's memory is the
//!   memory map's write-combined kind;
//! - 2.0 GB the driver's command memory, in private allocations of 1 GiB
//!   (three of them): `crate::bus` `probe_list_memory` saw 0.8 MB per
//!   allocator and list made and 1.55 MB more per 10,000 commands
//!   recorded, kept over a reset and NOT given back when the lists go (the
//!   driver keeps it for the next ones). So what counts is the most host
//!   lists alive at once with their biggest recordings: 1,062 in n55 (the
//!   title's 632 lists' host sides, the idle pool of `crate::list` up to
//!   512, the flights'), reached at the world's load and flat after;
//! - 1.1 GB process heap segments (ours: shader containers and
//!   translations 0.37 GB, the tables of copies; and the title's C runtime);
//! - 0.94 GB the CPU sides of title buffers' copies in the card's memory
//!   (shared buffers; `VICTORY_CARD_NO_PC_COPY` is refused under the
//!   relay); 0.6 GB upload and read-back heaps (texture staging 256 + 128
//!   pool, lists' scratch 128, fetches 64); one block of 401 MB of the
//!   global allocator (in every run since cap1; owner not named yet);
//!   0.84 GB the driver's blocks of 32 and 8 MiB.
//!
//! Over time: private bytes stay within +-0.5 GB from 120 s to 300 s in
//! runs n49-n56, the page pool at 7,190 MB: nothing grows. What brings the
//! system under the commit floor is the rest of the PC: in n55 the system
//! had 58.7 GB committed of 63.7 GB, ours 32.3 GB of it with the page
//! pool, other programs and the kernel the other 26 GB.
//!
//! The commit floor: while less than `VICTORY_COMMIT_FLOOR_MB` (6144
//! unless set; 0 turns it off) of the system's commit is left, the host
//! makes no new host copies of textures for shaders that pick them by
//! index and no new copies in the card's memory, and lets go of idle
//! copies sooner ([`idle_ms`]). Both only wait or drop what can be made
//! again from title memory; above the floor nothing changes.
//! `VICTORY_CARD_FLOOR_MB` (unset: off) gives what lives in the card's
//! memory alone (host textures, card copies of title buffers and windows,
//! the textures' sweep) a lower floor of its own ([`card_pressed`]). That
//! was on the ground that the card's memory is not the system's commit,
//! which the probe above disproves: what goes past the commit floor under
//! it costs the commit all the same (said once when the switch is set).
//!
//! What the floor holds back, and how it looks: a texture picked by index
//! (`draw.rs` `heap_textures`) with no host copy is refused one, and its
//! slot keeps the null view it had (`descriptor/host.rs` `unmake`): a
//! shader reads zeros from it, alpha 0 too, so an alpha-tested mesh is
//! drawn as nothing and a material's colour or mask reads black. The
//! slot is looked at again after about four seconds. [`SLOTS_PRESSED`]
//! counts those refusals (the line's "held back ... textures" counts the
//! times [`crate::resource::room`] said no, asked per pass and per 32 MiB,
//! not textures). `VICTORY_PRESSED_LEVELS_KB` (`crate::tiles::pressed_cap`)
//! makes such a texture anyway with its small levels alone.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

const MB: f64 = (1u64 << 20) as f64;

/// The process's private bytes, its shared commit and its working set.
pub(crate) fn process_memory() -> [u64; 3] {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX2,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS_EX2 {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX2>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a structure of the size passed. Systems
    // before Windows 10 1809 refuse this size: then the plain counters.
    let ok = unsafe {
        K32GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(), c.cb)
    } != 0;
    if !ok {
        let mut p = PROCESS_MEMORY_COUNTERS { cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32, ..Default::default() };
        // SAFETY: as above.
        unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut p, p.cb) };
        return [p.PagefileUsage as u64, 0, p.WorkingSetSize as u64];
    }
    [c.PrivateUsage as u64, c.SharedCommitUsage, c.WorkingSetSize as u64]
}

/// The system's commit in bytes: in use, and its limit.
pub(crate) fn system_commit() -> Option<(u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
    let mut info = PERFORMANCE_INFORMATION { cb: size_of::<PERFORMANCE_INFORMATION>() as u32, ..Default::default() };
    // SAFETY: a structure of the size passed.
    if unsafe { K32GetPerformanceInfo(&mut info, info.cb) } == 0 {
        return None;
    }
    let page = info.PageSize as u64;
    Some((info.CommitTotal as u64 * page, info.CommitLimit as u64 * page))
}

/// The system's commit left, in bytes, asked at most every 250 ms.
pub(crate) fn commit_left() -> Option<u64> {
    static ASKED: AtomicU64 = AtomicU64::new(u64::MAX);
    static LEFT: AtomicU64 = AtomicU64::new(u64::MAX);
    let now = crate::host::now_ms();
    let asked = ASKED.load(Relaxed);
    if asked == u64::MAX || now >= asked + 250 {
        ASKED.store(now, Relaxed);
        let left = system_commit().map_or(u64::MAX, |(used, limit)| limit.saturating_sub(used));
        LEFT.store(left, Relaxed);
    }
    Some(LEFT.load(Relaxed)).filter(|&left| left != u64::MAX)
}

/// The commit floor in bytes (`VICTORY_COMMIT_FLOOR_MB`, 6144 unless set;
/// 0: no floor). A guard that stops the run at 4 GB left, or the system
/// refusing commit (the title's page pool fails then, error 1450), comes
/// before anything else the floor would cost.
fn floor() -> u64 {
    static FLOOR: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        std::env::var("VICTORY_COMMIT_FLOOR_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(6144) << 20
    })
}

/// Times the floor held something back since the start: host copies of
/// textures not made, copies not put in the card's memory, sweeps with the
/// shorter idle time.
pub(crate) static HELD_BACK: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Since the start: texture slots picked by index refused a new host copy
/// for lack of room (the commit floor or the card's budget), which then
/// show a null view (zeros); and ones made with their small levels alone
/// instead (`VICTORY_PRESSED_LEVELS_KB`).
pub(crate) static SLOTS_PRESSED: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether less of the system's commit is left than the floor ([`floor`]).
pub(crate) fn pressed() -> bool {
    let floor = floor();
    floor > 0 && commit_left().is_some_and(|left| left < floor)
}

/// How long a host copy nothing names must have been idle to be let go:
/// `normal` milliseconds, 2 s while the commit floor is pressed.
pub(crate) fn idle_ms(normal: u64) -> u64 {
    const PRESSED: u64 = 2_000;
    if normal > PRESSED && pressed() {
        HELD_BACK[2].fetch_add(1, Relaxed);
        PRESSED
    } else {
        normal
    }
}

/// The floor for what lives in the card's memory alone, in bytes
/// (`VICTORY_CARD_FLOOR_MB`; unset: none of its own, the commit floor
/// [`floor`] holds those back too, as before).
///
/// The card's memory is not charged to the system's commit. Runs n15-n17
/// (5 s memory lines from the menus to two minutes into the world): the
/// system's commit less the process's private bytes and its shared commit
/// stayed at 24.1-25.5 GB while the card's use went from 1.7 GB to 12.2
/// GB and back to 9.6. Yet the commit floor was pressed from the first
/// world frame to the end in all three, and held back every host texture
/// of the texture queue (`resource::room`), every card copy of a title
/// buffer or window (`crate::shadow`), and let textures nothing named go
/// after 2 s instead of 5: the card stayed at 9.1-10.2 GB of a 15.2 GB
/// budget (the queue's line 13.7 GB) while the copies of textures the
/// title fills fell from 5.9-7.5 GB at the load to 2.1-4.8 GB. What making
/// such a copy does cost of the commit is its staging buffer, a few
/// hundred MB at most in those runs (texture.rs, which keeps the commit
/// floor for them); a lower floor of its own still stops it above the run
/// script's 4 GB guard.
fn card_floor() -> Option<u64> {
    static FLOOR: std::sync::OnceLock<Option<u64>> = std::sync::OnceLock::new();
    *FLOOR.get_or_init(|| {
        let floor =
            std::env::var("VICTORY_CARD_FLOOR_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map(|mb| mb << 20);
        if floor.is_some() {
            tracing::warn!(
                "VICTORY_CARD_FLOOR_MB: the card's memory costs the system's commit as the PC's does (probe_heaps:                  +256 MB for 256 MiB in a default heap), so what this lets past the commit floor brings the run nearer                  the commit limit"
            );
        }
        floor
    })
}

/// Times since the start the commit floor was pressed but what is in the
/// card's memory alone went ahead under [`card_floor`]: host textures of
/// the texture queue, card copies of title buffers and windows, sweeps of
/// host textures at the normal idle time.
pub(crate) static LET_PAST: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// [`LET_PAST`]'s places.
pub(crate) mod past {
    pub(crate) const TEXTURES: usize = 0;
    pub(crate) const CARD_COPIES: usize = 1;
    pub(crate) const SWEEPS: usize = 2;
}

/// [`pressed`] for what lives in the card's memory alone (`what`, a place
/// of [`LET_PAST`]): with [`card_floor`] set, whether less than it is left;
/// a pass under the commit floor that this lets through is counted.
pub(crate) fn card_pressed(what: usize) -> bool {
    let Some(card) = card_floor() else {
        return pressed();
    };
    let left = commit_left();
    let under = |floor: u64| floor > 0 && left.is_some_and(|left| left < floor);
    if under(card) {
        return true;
    }
    if under(floor()) {
        LET_PAST[what].fetch_add(1, Relaxed);
    }
    false
}

/// [`idle_ms`] for host textures, which live in the card's memory alone:
/// 2 s only under [`card_floor`] when it is set.
pub(crate) fn card_idle_ms(normal: u64) -> u64 {
    if card_floor().is_none() {
        return idle_ms(normal);
    }
    const PRESSED: u64 = 2_000;
    if normal > PRESSED && card_pressed(past::SWEEPS) {
        HELD_BACK[2].fetch_add(1, Relaxed);
        PRESSED
    } else {
        normal
    }
}

/// Whether the address space is walked for the memory map lines every
/// sixth memory line: `VICTORY_MEMORY_MAP=1` (off in a plain release
/// build: a walk takes about 0.6 s on a thread of its own and holds up
/// other threads' memory calls while it asks about big regions; on in
/// diagnostics builds, which walk from their own line).
fn memory_map() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_MEMORY_MAP").is_some_and(|v| v == "1"))
}

/// Logs the memory line (see the module).
pub(crate) fn log() {
    let mb = |v: u64| v as f64 / MB;
    let [private, shared, working] = process_memory();
    let (used, limit) = system_commit().unwrap_or_default();
    let card = crate::host::host().and_then(|h| h.video_memory()).unwrap_or_default();
    let copies = crate::shadow::memory();
    let [card_bytes, window_card_bytes, card_hashes] = crate::shadow::card_bytes();
    let [snaps, snap_bytes] = crate::shadow::snapshots();
    let staged = crate::texture::memory();
    let spare = crate::texture::spare_bytes();
    let ahead = crate::prepare::held();
    let textures = [0, 1].map(|i| crate::resource::HOST_TEXTURES[i].load(Relaxed));
    let filled = [0, 1].map(|i| crate::resource::KINDS[i].load(Relaxed));
    let alive: Vec<u64> = crate::host::ALIVE.iter().map(|a| a.load(Relaxed)).collect();
    let kept = crate::pipeline::kept();
    let numbered = crate::memory::numbered();
    let held = HELD_BACK.each_ref().map(|h| h.load(Relaxed));
    let past = LET_PAST.each_ref().map(|h| h.load(Relaxed));
    let card_floor = match card_floor() {
        Some(f) => format!(
            "card floor {:.0} MB (VICTORY_CARD_FLOOR_MB): let past the commit floor {} textures, {} card copies, {} sweeps",
            mb(f),
            past[0],
            past[1],
            past[2]
        ),
        None => "no card floor of its own (VICTORY_CARD_FLOOR_MB)".to_string(),
    };
    let slots = SLOTS_PRESSED.each_ref().map(|h| h.load(Relaxed));
    let pressed_made = crate::resource::PRESSED_MADE.each_ref().map(|h| h.load(Relaxed));
    let pressed_tiles = crate::tiles::PRESSED.each_ref().map(|h| h.load(Relaxed));
    let card_floor = format!(
        "{card_floor}; texture slots refused a host copy for room {} (null views until made), made with small levels          instead {} (VICTORY_PRESSED_LEVELS_KB: {}): {} textures made without memory, {} the card would not cut made          whole, {} uploads gave small levels {:.0} MB",
        slots[0],
        slots[1],
        crate::tiles::pressed_cap().map_or("off".to_string(), |cap| format!("{} KiB", cap >> 10)),
        pressed_made[0],
        pressed_made[1],
        pressed_tiles[0],
        mb(pressed_tiles[1]),
    );
    let small = crate::resource::small_textures();
    let twice = crate::resource::duplicates();
    let small = format!(
        "{} host copies of textures the title fills at the same title memory as another {:.0} MB; \
         small textures (VICTORY_SMALL_TEXTURES: {}): {} placed {:.0} MB in heaps of {:.0} MB, saving {:.0} MB of 64 KiB \
         padding; {} committed that could be placed, {:.0} MB of them padding",
        twice[0],
        mb(twice[1]),
        if crate::resource::small_on() { "on" } else { "off" },
        small[0],
        mb(small[1]),
        mb(small[2]),
        mb(small[3]),
        small[4],
        mb(small[5]),
    );
    let sides = crate::shadow::pc_sides();
    let lists = {
        let idle = crate::list::idle();
        let asked = crate::shadow::window_asked();
        let places = crate::shadow::window_sides();
        format!(
            "; host lists idle {} ({:.0} MB of upload memory kept; every host list holds the driver's command memory of              its biggest recording, kept by the driver when the list goes); windows copied: asked within them {:.0} MB,              {} asked without a length {:.0} MB (VICTORY_WINDOW_KB, VICTORY_WINDOW_TAILS); windows in place {} {:.0} MB              (VICTORY_INPLACE {})",
            idle[0],
            mb(idle[1]),
            mb(asked[0]),
            asked[1],
            mb(asked[2]),
            places[4],
            mb(places[5]),
            crate::inplace::level(),
        )
    };
    let stage = crate::host::CARD_STAGE.each_ref().map(|s| s.load(Relaxed));
    let fetch = crate::host::CARD_FETCH_BYTES.each_ref().map(|f| f.load(Relaxed));
    tracing::info!(
        "host memory: {:.0} MB private ({:.0} MB in memory), {:.0} MB shared commit (the title's page pool); the system: \
         {:.0} of {:.0} MB committed, {:.0} MB left (floor {:.0} MB: held back {} textures, {} card copies, {} sweeps; {}); \
         the card: {:.0} of {:.0} MB ({}), the PC's memory shared with it {:.0} MB (our upload, read-back and shared buffers); \
         host copies of title memory: {} alive {:.0} MB, of them {} windows outside buffers {:.0} MB; in the card's memory \
         {:.0} MB of buffers and {:.0} MB of windows, and {:.0} MB of hashes; their CPU sides in the PC's memory: {:.0} MB \
         whole (shared buffers), {} mirrors holding {:.0} MB (VICTORY_CARD_NO_PC_COPY: {}), {:.0} MB of pieces shaders may \
         write ({} pieces not committed), {:.0} MB noted to go up from title memory since the start; their sends' upload \
         chunks {} held {:.0} MB, {} waits, {:.0} MB moved since the start; fetch buffers {:.0} MB and {:.0} MB (relay), {} \
         let go; snapshots: {} out {:.0} MB, {} kept for reuse {:.0} MB; staging of textures: {} alive {:.0} MB, \
         {} kept by their texture, {:.0} MB spare, {:.0} MB made ready ahead; host textures: {} alive {:.0} MB, of them {} \
         the title fills {:.0} MB; host lists: {} alive, scratch {} chunks {:.0} MB ({} spare {:.0} MB), other upload \
         buffers {} {:.0} MB; shaders: {} containers {:.0} MB, {} translations {:.0} MB, the pipeline library's file \
         {:.0} MB; title memory by address: {} copies held {:.0} MB; {}{}",
        mb(private),
        mb(working),
        mb(shared),
        mb(used),
        mb(limit),
        mb(limit.saturating_sub(used)),
        mb(floor()),
        held[0],
        held[1],
        held[2],
        card_floor,
        mb(card[1]),
        mb(card[0]),
        small,
        mb(card[3]),
        copies[0],
        mb(copies[1]),
        copies[2],
        mb(copies[3]),
        mb(card_bytes),
        mb(window_card_bytes),
        mb(card_hashes),
        mb(sides[0]),
        sides[2],
        mb(sides[3]),
        if crate::shadow::no_pc_copy() { "on" } else { "off" },
        mb(sides[1]),
        sides[4],
        mb(sides[5]),
        stage[0],
        mb(stage[1]),
        stage[2],
        mb(stage[3]),
        mb(fetch[0]),
        mb(fetch[1]),
        fetch[2],
        snaps,
        mb(snap_bytes),
        copies[4],
        mb(copies[5]),
        staged[0],
        mb(staged[1]),
        staged[2],
        mb(spare),
        mb(ahead),
        textures[0],
        mb(textures[1]),
        filled[0],
        mb(filled[1]),
        alive[0],
        alive[1],
        mb(alive[2]),
        alive[3],
        mb(alive[4]),
        alive[6],
        mb(alive[7]),
        kept[0],
        mb(kept[1]),
        kept[2],
        mb(kept[3]),
        mb(kept[4]),
        numbered[0],
        mb(numbered[1]),
        crate::bus::memory_text(),
        lists,
    );
    window_line();
    static LINES: AtomicU64 = AtomicU64::new(0);
    static WALKING: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if memory_map() && LINES.fetch_add(1, Relaxed) % 6 == 0 && !WALKING.swap(true, Relaxed) {
        let walk = std::thread::Builder::new().name("victory-memory-map".into()).spawn(|| {
            log_memory_map();
            WALKING.store(false, Relaxed);
        });
        if walk.is_err() {
            WALKING.store(false, Relaxed);
        }
    }
}

/// Beside the memory line, while windows may go to the card
/// (`VICTORY_CARD_WINDOWS_MB`) or shaders declare writes to windows: where
/// the windows are, what went up to their card sides per frame since the
/// last line, what shaders declared written in windows, and the windows
/// turned down for the card and why.
fn window_line() {
    use crate::shadow::{WINDOW_CARD_STATS as S, window_stats as w};
    static LAST: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];
    let frame = crate::frame::current();
    let sent = S[w::SENT].load(Relaxed);
    let declared = S[w::DECLARED].load(Relaxed);
    let declared_bytes = S[w::DECLARED_BYTES].load(Relaxed);
    let last = [
        LAST[0].swap(frame, Relaxed),
        LAST[1].swap(sent, Relaxed),
        LAST[2].swap(declared, Relaxed),
        LAST[3].swap(declared_bytes, Relaxed),
    ];
    let made = S[w::MADE].load(Relaxed);
    let refused: u64 = [w::TEXTURES_WAIT, w::NEAR_LINE, w::PAST_MOST].iter().map(|&i| S[i].load(Relaxed)).sum();
    if made == 0 && refused == 0 && declared == 0 {
        return;
    }
    let frames = frame.saturating_sub(last[0]).max(1) as f64;
    let sides = crate::shadow::window_sides();
    tracing::info!(
        "windows outside buffers: {} on the card {:.0} MB, {} in the PC's memory {:.0} MB (shaders read those over the bus), \
         {} in place {:.0} MB; per frame {:.1} MB sent up to windows on the card, {:.1} ranges of windows declared written \
         by shaders ({:.2} MB); since the start {} made on the card, turned down for the card: {} while textures waited for \
         room ({:.0} MB), {} within VICTORY_CARD_WINDOWS_ROOM_MB of the texture line ({:.0} MB), {} past \
         VICTORY_CARD_WINDOWS_MB ({:.0} MB), {} ranges declared written ({:.0} MB)",
        sides[0],
        sides[1] as f64 / MB,
        sides[2],
        sides[3] as f64 / MB,
        sides[4],
        sides[5] as f64 / MB,
        sent.saturating_sub(last[1]) as f64 / MB / frames,
        declared.saturating_sub(last[2]) as f64 / frames,
        declared_bytes.saturating_sub(last[3]) as f64 / MB / frames,
        made,
        S[w::TEXTURES_WAIT].load(Relaxed),
        S[w::TEXTURES_WAIT + 1].load(Relaxed) as f64 / MB,
        S[w::NEAR_LINE].load(Relaxed),
        S[w::NEAR_LINE + 1].load(Relaxed) as f64 / MB,
        S[w::PAST_MOST].load(Relaxed),
        S[w::PAST_MOST + 1].load(Relaxed) as f64 / MB,
        declared,
        declared_bytes as f64 / MB,
    );
}

/// Asks which of the pages in `asked` (address, answer) are in physical
/// memory and adds the bytes each stands for (kind, bytes) to its kind.
fn in_memory(asked: &mut Vec<[usize; 2]>, stands: &mut Vec<(usize, u64)>, present: &mut [u64; 11]) {
    use windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: our own process; `asked` holds that many entries of two
    // words each, as the call fills them in.
    let ok = unsafe {
        K32QueryWorkingSetEx(GetCurrentProcess(), asked.as_mut_ptr().cast(), (asked.len() * 16) as u32)
    } != 0;
    if ok {
        for (a, &(kind, bytes)) in asked.iter().zip(stands.iter()) {
            // Bit 0 of the answer: the page is in the process's memory.
            if a[1] & 1 != 0 {
                present[kind] += bytes;
            }
        }
    }
    asked.clear();
    stands.clear();
}

/// Logs what the process has committed by kind, from a walk of its address
/// space, and how much of each kind is in physical memory (one page in
/// sixteen is asked about). The kinds tell the title's own memory from
/// ours: the title's heaps and page views lie at console addresses (below
/// 1 TiB), everything the host allocates lies far above. Run 30, seen
/// from outside at 18.0 GB private: 4.6 GB the title's heaps, 2.6 GB in
/// 162 segments of the process heap, 2.7 GB in 707 allocations of 4 MiB
/// (windows of title memory), 0.46 GB the pipeline library's file; mapped
/// and so not private: 6.2 GB the title's page pool, 2.0 GB the driver's
/// shader cache file.
/// shortcut: the walk takes about 0.6 s on a thread of its own, and a
/// query of a region of gigabytes holds up other threads' memory calls
/// for its length (89 ms for the 6.2 GB pool), so it runs with every
/// sixth memory line only.
pub(crate) fn log_memory_map() {
    use std::collections::HashMap;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED, MEMORY_BASIC_INFORMATION, PAGE_WRITECOMBINE,
        VirtualQuery,
    };
    const KINDS: [&str; 11] = [
        "the title's heaps (private, console addresses)",
        "views of the title's pages (mapped, console addresses)",
        "the title's page pool (mapped, 1 GB and more)",
        "process heap segments (private, 15.8 MB each)",
        "windows of title memory and other 4 MiB allocations",
        "write-combined (the card's own memory as it shows here, upload heaps, GPU upload heaps)",
        "other private allocations of 1 MB and more",
        "other private allocations under 1 MB",
        "files and other mappings at host addresses",
        "programs and libraries",
        "the driver's command memory (private allocations of 1 GiB, committed as lists record, never given back)",
    ];
    /// Console addresses lie below this.
    const HOST: usize = 1 << 40;
    /// One page asked about stands for this much.
    const STEP: usize = 64 << 10;
    let started = std::time::Instant::now();
    // Allocation, start, bytes, committed, type, protection.
    type Region = (usize, usize, usize, bool, u32, u32);
    let mut regions: Vec<Region> = Vec::new();
    let mut at = 0usize;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
            break;
        }
        if m.State != MEM_FREE {
            regions.push((
                m.AllocationBase as usize,
                m.BaseAddress as usize,
                m.RegionSize,
                m.State == MEM_COMMIT,
                m.Type,
                m.Protect,
            ));
        }
        let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        if next <= at {
            break;
        }
        at = next;
    }
    // Each allocation's whole size, reserved parts included.
    let mut whole: HashMap<usize, usize> = HashMap::new();
    for r in &regions {
        *whole.entry(r.0).or_default() += r.2;
    }
    let kind = |r: &Region| -> usize {
        let size = whole.get(&r.0).copied().unwrap_or(0);
        let console = r.0 < HOST;
        if r.4 == MEM_IMAGE {
            9
        } else if r.4 == MEM_MAPPED {
            match (console, size >= 1 << 30) {
                (false, _) => 8,
                (true, true) => 2,
                (true, false) => 1,
            }
        } else if console {
            0
        } else if size == 0xfd_0000 {
            3
        } else if size == 4 << 20 {
            4
        } else if size == 1 << 30 && r.5 & PAGE_WRITECOMBINE == 0 {
            10
        } else if r.5 & PAGE_WRITECOMBINE != 0 {
            5
        } else if size >= 1 << 20 {
            6
        } else {
            7
        }
    };
    let mut committed = [0u64; 11];
    let mut present = [0u64; 11];
    let mut asked: Vec<[usize; 2]> = Vec::with_capacity(4096);
    let mut stands: Vec<(usize, u64)> = Vec::with_capacity(4096);
    for r in regions.iter().filter(|r| r.3) {
        let k = kind(r);
        committed[k] += r.2 as u64;
        let mut offset = 0;
        while offset < r.2 {
            asked.push([r.1 + offset, 0]);
            stands.push((k, STEP.min(r.2 - offset) as u64));
            if asked.len() == 4096 {
                in_memory(&mut asked, &mut stands, &mut present);
            }
            offset += STEP;
        }
    }
    if !asked.is_empty() {
        in_memory(&mut asked, &mut stands, &mut present);
    }
    let text: Vec<String> = (0..KINDS.len())
        .filter(|&k| committed[k] > 0)
        .map(|k| format!("{} {} MB ({} MB in memory)", KINDS[k], committed[k] >> 20, present[k] >> 20))
        .collect();
    tracing::info!(
        "memory map ({} regions, {} ms): {}",
        regions.len(),
        started.elapsed().as_millis(),
        text.join("; ")
    );
    // The private allocations the kinds above leave as "other" (and the
    // write-combined ones), grouped by their size: the sizes name their
    // owners (the pipeline library's file to the byte, a snapshot buffer
    // of a title heap's size, the driver's blocks). Committed bytes by
    // allocation first, then allocations by (kind, whole size).
    let mut by_allocation: HashMap<usize, (usize, u64)> = HashMap::new();
    for r in regions.iter().filter(|r| r.3) {
        let k = kind(r);
        if k == 5 || k == 6 {
            by_allocation.entry(r.0).or_insert((k, 0)).1 += r.2 as u64;
        }
    }
    let mut groups: HashMap<(usize, usize), (u64, u64)> = HashMap::new();
    for (base, &(k, bytes)) in &by_allocation {
        let group = groups.entry((k, whole.get(base).copied().unwrap_or(0))).or_default();
        group.0 += 1;
        group.1 += bytes;
    }
    let mut groups: Vec<((usize, usize), (u64, u64))> = groups.into_iter().collect();
    groups.sort_by_key(|g| std::cmp::Reverse(g.1.1));
    let biggest: Vec<String> = groups
        .iter()
        .take(20)
        .map(|&((k, size), (n, bytes))| {
            format!(
                "{n} of {:.2} MB{}: {} MB",
                size as f64 / MB,
                if k == 5 { " write-combined" } else { "" },
                bytes >> 20
            )
        })
        .collect();
    tracing::info!(
        "memory map, other private allocations by size, the {} biggest groups of {} (committed): {}",
        biggest.len(),
        groups.len(),
        biggest.join("; ")
    );
}

#[cfg(test)]
mod tests {
    /// The counters the line and the floor read answer on this PC.
    #[test]
    fn the_process_and_the_system_are_measured() {
        let [private, _, working] = super::process_memory();
        assert!(private > 0 && working > 0);
        let (used, limit) = super::system_commit().expect("the system's commit");
        assert!(used <= limit);
        let left = super::commit_left().expect("the commit left");
        assert!(left <= limit);
        assert_eq!(super::idle_ms(1_000), 1_000, "a time under the pressed one stays as it is");
    }
}
