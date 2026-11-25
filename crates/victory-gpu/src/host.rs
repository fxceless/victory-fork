//! The host GPU: one D3D12 device and direct queue (decision 0007), and a
//! window to present on.
//!
//! shortcut: every submission is waited for, so CPU and GPU never overlap
//! (unless `VICTORY_LATE_WAIT=1`, see [`late`]);
//! overlap them with per-submission fences once more than clears run.

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Foundation::{HANDLE, HWND, LPARAM, LRESULT, RECT, WPARAM};
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
use windows::Win32::UI::WindowsAndMessaging::*;
use windows::core::{Interface, PCWSTR, w};

/// Resource states without the console's own bits (compression hints from
/// bit 25 up), which the host rejects.
pub(crate) fn pc_state(state: u32) -> u32 {
    state & 0x01ff_ffff
}

/// A console state as a host texture state: PC bits only, and none of the
/// buffer-only ones the console allows on textures (vertex, constant and
/// index buffer, stream out, indirect arguments, as in "generic read").
pub(crate) fn texture_state(state: u32) -> u32 {
    const BUFFER_ONLY: u32 = 0x1 | 0x2 | 0x100 | 0x200;
    pc_state(state) & !BUFFER_ONLY
}

/// Render target views live only until their command is recorded, so a
/// ring of host descriptors is enough.
const RTV_RING: u32 = 1024;

pub(crate) struct Host {
    pub(crate) device: ID3D12Device,
    queue: ID3D12CommandQueue,
    fence: ID3D12Fence,
    /// The last value signalled on `fence`, behind the submission lock.
    submitted: Mutex<u64>,
    event: HANDLE,
    rtvs: ID3D12DescriptorHeap,
    rtv_size: u32,
    /// A ring of depth views, used as `rtvs`.
    dsvs: ID3D12DescriptorHeap,
    dsv_size: u32,
    next_dsv: AtomicU32,
    /// With `VICTORY_D3D12_DEBUG` set (diagnostics builds): the debug
    /// layer's messages.
    messages: Option<ID3D12InfoQueue>,
    next_rtv: AtomicU32,
    presenter: Mutex<Option<Presenter>>,
    /// The adapter the device is on, for its memory budget
    /// ([`Host::video_memory`]); `None` if the system does not name it.
    adapter: Option<IDXGIAdapter3>,
}

// SAFETY: D3D12 devices, queues, fences and descriptor heaps are
// free-threaded; the event handle is only waited on under `submitted`.
unsafe impl Send for Host {}
// SAFETY: as above.
unsafe impl Sync for Host {}

/// Host objects the title let go of while a submission may still use them
/// (a texture its descriptors name, a descriptor slot's old resource), by
/// the submission count they wait for. The title's descriptor writes and
/// releases reach the host at once, from its own threads, while the host
/// runs a submission: freeing there is a GPU page fault on the freed
/// texture (a device hang).
static RETIRED: Mutex<Vec<(u64, Retired)>> = Mutex::new(Vec::new());
/// Submissions started and finished (waited for).
static STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FINISHED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Submissions started so far (their numbers run from 1; a flight's is
/// taken when its preparing begins).
pub(crate) fn started() -> u64 {
    STARTED.load(Ordering::Acquire)
}

/// The number of the last submission waited for or landed: every one
/// numbered up to it has run, and what it wrote for the title (shader
/// writes, read-backs, bottom-of-pipe values, the fences behind it) is in
/// title memory.
pub(crate) fn finished() -> u64 {
    FINISHED.load(Ordering::Acquire)
}

/// A retired object, dropped on whatever thread finishes the submission.
struct Retired(#[allow(dead_code)] Box<dyn std::any::Any>);
// SAFETY: what is retired is D3D12 objects (free-threaded) and the
// reference-counted wrappers holding them, dropped once.
unsafe impl Send for Retired {}

/// Milliseconds since the first call.
pub(crate) fn now_ms() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_millis() as u64
}

/// This moment, for telling later how long ago it was ([`long_ago`]): the
/// frames begun so far (high half, `crate::frame::current`) and tenths of a
/// second since the start. Host copies of title memory, staging buffers and
/// host textures note when they were last used with it.
pub(crate) fn moment() -> u64 {
    (crate::frame::current() << 32) | ((now_ms() / 100) & 0xffff_ffff)
}

/// Whether `moment` lies at least `ms` milliseconds and `frames` frames
/// back. Both: at seconds per frame, what every frame uses is not touched
/// for seconds, and at many frames per second not for many frames. (A title
/// that never waits for frames has no frame count: nothing is then idle.)
pub(crate) fn long_ago(moment: u64, ms: u64, frames: u64) -> bool {
    let now = self::moment();
    let tenths = (now & 0xffff_ffff).saturating_sub(moment & 0xffff_ffff);
    (now >> 32).saturating_sub(moment >> 32) >= frames && tenths * 100 >= ms
}

/// Available physical RAM on the host in bytes, from `GlobalMemoryStatusEx`.
pub(crate) fn available_physical_ram() -> Option<u64> {
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the call fills in.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = std::mem::size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a sized structure.
    let ok = unsafe { GlobalMemoryStatusEx(&mut m) } != 0;
    ok.then_some(m.ullAvailPhys)
}

/// Effective idle timeout (in milliseconds) for VRAM-backed host textures,
/// scaled down under VRAM pressure. Never returns less than 1,000 ms so
/// handles not yet recorded into a list or slot remain protected.
pub(crate) fn vram_idle_ms(video_memory: Option<[u64; 4]>) -> u64 {
    const NORMAL_MS: u64 = 5_000;
    const MODERATE_MS: u64 = 2_500;
    const HIGH_MS: u64 = 1_000;
    let Some([budget, used, ..]) = video_memory else {
        return NORMAL_MS;
    };
    if budget == 0 {
        return NORMAL_MS;
    }
    let used_pct = (u128::from(used) * 100) / u128::from(budget);
    if used_pct >= 85 {
        HIGH_MS
    } else if used_pct >= 75 {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// Effective idle timeout (in milliseconds) for system-RAM caches (staging
/// buffers, loose shadows, and the mapped pipeline library), scaled down
/// when free physical RAM is low. Never returns less than 2,000 ms.
pub(crate) fn ram_idle_ms(avail_phys: Option<u64>) -> u64 {
    const GIB: u64 = 1024 * 1024 * 1024;
    const NORMAL_MS: u64 = 10_000;
    const MODERATE_MS: u64 = 5_000;
    const HIGH_MS: u64 = 2_000;
    let Some(avail) = avail_phys else {
        return NORMAL_MS;
    };
    if avail < 2 * GIB {
        HIGH_MS
    } else if avail < 4 * GIB {
        MODERATE_MS
    } else {
        NORMAL_MS
    }
}

/// How many references are held on a host object (its COM count): ours
/// all, since the host's views and commands hold none. A buffer or texture
/// with one is held by its owner alone: no descriptor slot keeps it
/// (`HostHeap::kept`) and no list (`HostList::keep`).
pub(crate) fn references(object: &impl Interface) -> u32 {
    type Method = unsafe extern "system" fn(*mut std::ffi::c_void) -> u32;
    let raw = object.as_raw();
    // SAFETY: a live COM object, whose first three methods are those of
    // `IUnknown` (query, add a reference, release one); the pair leaves the
    // count as it was, and the release returns it. The first is not called.
    unsafe {
        let methods = *raw.cast::<*const [Method; 3]>();
        ((*methods)[1])(raw);
        ((*methods)[2])(raw)
    }
}

/// Once a second, before a submission starts: lets go of what nothing
/// names any more (copies of windows of title memory, staging buffers,
/// host copies of textures the title fills).
/// `VICTORY_SWEEP=0` keeps everything, as before.
fn sweep() {
    // The in-place probe and census, if asked for (`VICTORY_INPLACE_PROBE`).
    crate::inplace::tick();
    static ON: OnceLock<bool> = OnceLock::new();
    if !*ON.get_or_init(|| std::env::var_os("VICTORY_SWEEP").is_none_or(|v| v != "0")) {
        return;
    }
    static LAST: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let now = now_ms();
    let last = LAST.load(Ordering::Relaxed);
    if now < last + 1000
        || LAST
            .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
            .is_err()
    {
        return;
    }
    crate::pipeline::trim_library();
    crate::shadow::let_go_idle();
    crate::texture::let_go_idle();
    crate::resource::let_go_idle();
}

/// Drops `object` once every submission running now has finished: at once
/// if none is.
pub(crate) fn retire<T: 'static>(object: T) {
    let started = STARTED.load(Ordering::Acquire);
    if started <= FINISHED.load(Ordering::Acquire) {
        RETIREMENTS[0].fetch_add(1, Ordering::Relaxed);
        drop(object);
        return;
    }
    RETIREMENTS[1].fetch_add(1, Ordering::Relaxed);
    RETIRED
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .push((started, Retired(Box::new(object))));
}

/// Drops what waited for submissions up to `finished`.
fn bury(finished: u64) {
    let done: Vec<(u64, Retired)> = {
        let mut retired = RETIRED.lock().unwrap_or_else(|e| e.into_inner());
        if retired.is_empty() {
            return;
        }
        let (done, keep) = std::mem::take(&mut *retired)
            .into_iter()
            .partition(|(at, _)| *at <= finished);
        *retired = keep;
        done
    };
    drop(done);
}

/// Objects [`retire`] dropped at once (no submission was running) and
/// kept until a running submission finished, since the start.
static RETIREMENTS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// What the layer did to the host's objects lately, newest last (sweeps
/// that let host copies go, pipeline library calls, saves and trims): said
/// once when the host device is removed ([`log_removal_context`]), with
/// what the submitting thread and the pipeline workers were doing at that
/// moment. A device the driver removes "internally" names no command, and
/// the pass that happens to notice first names only the view it was at:
/// the three such losses of runs 72, 73 and 82 each came within 5 ms of a
/// pipeline library call that no line of the log tied to them.
/// `VICTORY_TRAIL=0`: nothing is kept and nothing is said.
static TRAIL: Mutex<std::collections::VecDeque<(u64, u32, String)>> = Mutex::new(std::collections::VecDeque::new());
/// The entries kept.
const TRAIL_MOST: usize = 96;

fn trail_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TRAIL").is_none_or(|v| v != "0"))
}

/// Microseconds since the first call.
pub(crate) fn now_us() -> u64 {
    static EPOCH: OnceLock<std::time::Instant> = OnceLock::new();
    EPOCH.get_or_init(std::time::Instant::now).elapsed().as_micros() as u64
}

/// The calling thread, as the log's other lines name threads.
pub(crate) fn thread_id() -> u32 {
    // SAFETY: a plain query.
    unsafe { windows::Win32::System::Threading::GetCurrentThreadId() }
}

/// Notes something done to the host's objects ([`TRAIL`]); `what` is only
/// run when the trail is kept.
pub(crate) fn trail(what: impl FnOnce() -> String) {
    if !trail_on() {
        return;
    }
    let entry = (now_us(), thread_id(), what());
    // Sweeps also go to the log as they happen: the trail is only said at
    // a loss, and the driver's errors in the system's event log (run 224's
    // first fell inside a sweep of 4,711 host textures) need the sweeps of
    // clean runs beside them.
    if entry.2.starts_with("sweep:") {
        tracing::info!("{}", entry.2);
    }
    let mut trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    if trail.len() >= TRAIL_MOST {
        trail.pop_front();
    }
    trail.push_back(entry);
}

/// What the submitting thread is doing (0 nothing: between submissions,
/// 1 preparing one, 2 waiting for the card to run it, 3 copying back what
/// it wrote), the lists of that submission, and since when ([`now_us`]).
static SUBMITTING: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

fn submitting(phase: u64, lists: usize) {
    SUBMITTING[1].store(lists as u64, Ordering::Relaxed);
    SUBMITTING[2].store(now_us(), Ordering::Relaxed);
    SUBMITTING[0].store(phase, Ordering::Relaxed);
}

/// What was going on when the host device went: whether a submission was
/// on the card, the pipeline library's state and the calls running in it,
/// and the trail.
pub(crate) fn removal_context() -> String {
    use std::fmt::Write;
    let now = now_us();
    let ms = |at: u64| now.saturating_sub(at) as f64 / 1000.0;
    let phase = match SUBMITTING[0].load(Ordering::Relaxed) {
        0 => "between submissions (nothing of ours was running on the card)",
        1 => "preparing a submission (nothing of ours was running on the card)",
        2 => "waiting for the card to run a submission",
        _ => "copying back what a submission wrote (nothing of ours was running on the card)",
    };
    let mut text = format!(
        "the submitting thread was {phase}, of {} lists, for {:.1} ms; {}; objects let go at once {} and after a running submission {}; submissions started {} finished {}",
        SUBMITTING[1].load(Ordering::Relaxed),
        ms(SUBMITTING[2].load(Ordering::Relaxed)),
        crate::pipeline::library_removal_note(),
        RETIREMENTS[0].load(Ordering::Relaxed),
        RETIREMENTS[1].load(Ordering::Relaxed),
        STARTED.load(Ordering::Relaxed),
        FINISHED.load(Ordering::Relaxed),
    );
    let trail = TRAIL.lock().unwrap_or_else(|e| e.into_inner());
    let _ = write!(text, "; the last {} things done, newest last:", trail.len());
    for (at, thread, what) in trail.iter() {
        let _ = write!(text, "\n  {:.1} ms before, thread {thread}: {what}", ms(*at));
    }
    text
}

/// Says [`removal_context`] in the log, once: called by whoever finds the
/// host device removed.
pub(crate) fn log_removal_context() {
    static SAID: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    if !trail_on() || SAID.swap(true, Ordering::Relaxed) {
        return;
    }
    tracing::error!("when the host device went: {}", removal_context());
}

/// Time spent in submissions since the last frame-rate line, in
/// microseconds: preparing them on the CPU (title memory copies, texture
/// detiling), then waiting for the GPU.
static SUBMISSION_US: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// Adds to [`SUBMISSION_US`] and to the frame log's totals.
fn submission_us(i: usize, us: u64) {
    SUBMISSION_US[i].fetch_add(us, Ordering::Relaxed);
    crate::framelog::add([crate::framelog::T::PrepareUs, crate::framelog::T::CardWaitUs][i], us);
    if i == 0 {
        crate::framelog::add(crate::framelog::T::Submissions, 1);
    }
}

/// Adds to [`PREPARING`]'s copies (time, bytes) and to the frame log's totals.
fn preparing(i: usize, n: u64) {
    PREPARING[i].fetch_add(n, Ordering::Relaxed);
    crate::framelog::add([crate::framelog::T::CopyUs, crate::framelog::T::CopyBytes][i], n);
}

/// Adds to [`COPY_BACK`] and to the frame log's totals.
fn copy_back(i: usize, n: u64) {
    COPY_BACK[i].fetch_add(n, Ordering::Relaxed);
    crate::framelog::add([crate::framelog::T::CopyBackUs, crate::framelog::T::CopyBackBytes][i], n);
}

/// Adds to [`CARD_CLOCK_US`] and, for the card's idle times, to the frame
/// log's totals.
fn card_clock_us(i: usize, us: u64) {
    CARD_CLOCK_US[i].fetch_add(us, Ordering::Relaxed);
    match i {
        0 => crate::framelog::add(crate::framelog::T::CardIdleUs, us),
        1 => crate::framelog::add(crate::framelog::T::CardIdleBehindUs, us),
        _ => {}
    }
}

/// After submissions: shader-written ranges compared and copied back into
/// title memory (microseconds, bytes that differed), since the last
/// frame-rate line.
static COPY_BACK: [std::sync::atomic::AtomicU64; 2] =
    [std::sync::atomic::AtomicU64::new(0), std::sync::atomic::AtomicU64::new(0)];

/// The card's clock, for the frame line of every run, in microseconds since
/// the last one: how long the card had nothing of ours queued; how much of
/// that while the thread that holds the submission lock was working (the
/// card idle behind us: preparing and copying back are done with the card
/// standing); and how long that thread stood waiting for the card.
///
/// Every send of lists to the queue is followed by a signal on a fence of
/// the clock's own ([`Host::card_sent`]). A thread of the clock waits for
/// the newest signal and notes the moment the card reached it with nothing
/// newer sent; the next send ends that idle stretch.
/// Then (index 3) how much of the idle stretches fell while that thread
/// held the lock and stood waiting (for the card or for the landing
/// thread: with the relay the card may stand idle while the landing thread
/// fetches and merges and the submitting thread waits for it).
static CARD_CLOCK_US: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// Of the thread that holds the submission lock: the microseconds it has
/// worked so far (preparing, copying back; not its waits for the card), and
/// since when it works now ([`now_us`] plus one; 0: it does not).
static WORKING: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// The thread that holds the submission lock starts (`on`) or stops
/// working; whether it worked until now.
fn working(on: bool) -> bool {
    let now = now_us() + 1;
    let since = WORKING[1].swap(if on { now } else { 0 }, Ordering::Relaxed);
    if since != 0 {
        WORKING[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
    }
    since != 0
}

/// [`WORKING`] as of now: microseconds worked so far.
fn worked() -> u64 {
    let since = WORKING[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    WORKING[0].load(Ordering::Relaxed) + now
}

/// Of the submission lock: the microseconds it was held so far (by
/// whichever thread), since when it is held now ([`now_us`] plus one; 0:
/// it is not), and when it was last let go ([`now_us`] plus one; 0: never).
/// For the frame's path ([`path_line`]) and the card's clock.
static HELD: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The thread that took the submission lock last ([`thread_id`]).
static SUBMITTER: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Whether the calling thread is the one that took the submission lock
/// last: for the heap pass's split by thread on the frame's path.
pub(crate) fn on_submitting_thread() -> bool {
    SUBMITTER.load(Ordering::Relaxed) == thread_id()
}

/// The submission lock is taken (`on`) or let go: [`HELD`], and with a
/// take the gap since the last let-go in [`PATH`] (the title's own time
/// between submissions of ours).
fn held(on: bool) {
    let now = now_us() + 1;
    if on {
        SUBMITTER.store(thread_id(), Ordering::Relaxed);
        let last = HELD[2].load(Ordering::Relaxed);
        if last != 0 {
            let gap = now.saturating_sub(last);
            PATH[path::GAPS].fetch_add(gap, Ordering::Relaxed);
            PATH[path::GAPS_N].fetch_add(1, Ordering::Relaxed);
            PATH[path::GAP_MOST].fetch_max(gap, Ordering::Relaxed);
        }
        HELD[1].store(now, Ordering::Relaxed);
    } else {
        let since = HELD[1].swap(0, Ordering::Relaxed);
        if since != 0 {
            HELD[0].fetch_add(now.saturating_sub(since), Ordering::Relaxed);
        }
        HELD[2].store(now, Ordering::Relaxed);
    }
}

/// [`HELD`] as of now: microseconds the lock was held so far.
fn held_so_far() -> u64 {
    let since = HELD[1].load(Ordering::Relaxed);
    let now = if since != 0 { (now_us() + 1).saturating_sub(since) } else { 0 };
    HELD[0].load(Ordering::Relaxed) + now
}

/// The submission lock, timed while it is held ([`held`]): every taker
/// goes through [`Host::lock_submitted`].
pub(crate) struct Submitted<'a>(std::sync::MutexGuard<'a, u64>);

impl std::ops::Deref for Submitted<'_> {
    type Target = u64;
    fn deref(&self) -> &u64 {
        &self.0
    }
}

impl std::ops::DerefMut for Submitted<'_> {
    fn deref_mut(&mut self) -> &mut u64 {
        &mut self.0
    }
}

impl Drop for Submitted<'_> {
    fn drop(&mut self) {
        held(false);
    }
}

/// Since the last frame-rate line, for the frame's path ([`path_line`]),
/// in microseconds or counts ([`path`] names them).
pub(crate) static PATH: [std::sync::atomic::AtomicU64; path::ALL] = [const { std::sync::atomic::AtomicU64::new(0) }; path::ALL];

/// The parts of [`PATH`].
pub(crate) mod path {
    /// Gaps between a let-go of the submission lock and the next take
    /// (the title's own time: no submission of ours under way), how many,
    /// and the longest.
    pub const GAPS: usize = 0;
    pub const GAPS_N: usize = 1;
    pub const GAP_MOST: usize = 2;
    /// The sync of descriptor records the title copied into its heaps
    /// (`crate::descriptor::sync_heaps`) on the submitting thread, and of
    /// it the part done before a title-read wait (`VICTORY_SYNC_IN_WAIT`).
    pub const SYNC: usize = 3;
    pub const SYNC_EARLY: usize = 4;
    /// Of the landing thread (relay): landing in all, and waiting for
    /// fetches of card copies' pieces; and with `VICTORY_RELAY_COPY_QUEUE`
    /// fetches on its copy queue, pieces there copied behind a later
    /// flight that declared them, and pieces fetched again on the host
    /// queue (a flight that declared them may have run beside the copy).
    pub const LAND: usize = 5;
    pub const FETCH: usize = 6;
    pub const FETCH_COPY_N: usize = 7;
    pub const BEHIND_N: usize = 8;
    /// The heap pass over buffer views (`crate::draw`'s "heap buffers") on
    /// the thread that submits and on others.
    pub const HEAP_SUBMITTER: usize = 9;
    pub const HEAP_OTHER: usize = 10;
    pub const AGAIN_N: usize = 11;
    /// With `VICTORY_SYNC_OVERLAP=1` (`super::sync_overlap`): the heap
    /// sync's time on its own thread beside the preparing, the submitting
    /// thread's wait for it where it is joined, the submissions whose sync
    /// ran beside the preparing, and all submissions.
    pub const SYNC_BESIDE: usize = 12;
    pub const SYNC_JOIN: usize = 13;
    pub const SYNC_BESIDE_N: usize = 14;
    pub const SUBMITS_N: usize = 15;
    /// With `VICTORY_SYNC_OVERLAP=1`: the joins before the end of the
    /// preparing (before a sweep, before a settle), and the time waited
    /// at the join at the end of the preparing.
    pub const JOIN_EARLY_N: usize = 16;
    pub const JOIN_LATE_US: usize = 17;
    /// With `VICTORY_SYNC_OVERLAP=2`: the passes of the sync that runs all
    /// along, and those that found pages written.
    pub const ALONG_N: usize = 18;
    pub const ALONG_FOUND_N: usize = 19;
    pub const ALL: usize = 20;
}

/// Of the submitting thread's working time ([`WORKING`]: not its waits for
/// the card or the landing thread), by part of [`Host::submit`], in
/// microseconds since the last frame-rate line ([`work`] names them; what
/// none holds is worked outside `submit`, "elsewhere" on the frame's path).
pub(crate) static WORK_PARTS: [std::sync::atomic::AtomicU64; work::ALL] =
    [const { std::sync::atomic::AtomicU64::new(0) }; work::ALL];

/// The parts of [`WORK_PARTS`], in the order they run in a submission.
pub(crate) mod work {
    /// From the lock to the heap sync: the looks at tables on the card,
    /// the settles' and read waits' own work.
    pub const BEFORE: usize = 0;
    /// The heap sync on the submitting thread, or (`VICTORY_SYNC_OVERLAP`)
    /// the wait where the sync beside the preparing is joined.
    pub const SYNC: usize = 1;
    /// The sweep and the slots that waited for a texture's host copy.
    pub const SWEEP: usize = 2;
    /// The submission's one asking with its spans, and the stale checks.
    pub const ASK: usize = 3;
    /// The lists' steps (fills, refresh flushes, declarations, texture
    /// checks), gathering the lists, the flush of all at their end, the
    /// asking's end.
    pub const STEPS: usize = 4;
    /// The snapshots, and the copies in the card's memory.
    pub const CARD: usize = 5;
    /// Sending to the queue, and (a submission waited for) the copy-back.
    pub const SEND: usize = 6;
    pub const ALL: usize = 7;
}

/// A point on the submitting thread's working time ([`worked`]): each
/// [`WorkMark::to`] gives the time worked since the last point to a part
/// of [`WORK_PARTS`].
struct WorkMark(u64);

impl WorkMark {
    fn new() -> Self {
        Self(worked())
    }

    fn to(&mut self, part: usize) {
        let now = worked();
        WORK_PARTS[part].fetch_add(now.saturating_sub(self.0), Ordering::Relaxed);
        self.0 = now;
    }
}

/// Whether the heap sync (`crate::descriptor::sync_heaps`) of a submission
/// sent ahead runs on a thread of its own beside the submission's preparing
/// (`VICTORY_SYNC_OVERLAP=1`; off by default), joined before the copies in
/// the card's memory and the send.
///
/// It was 30-35 ms of the submitting thread's 110-120 ms of work a frame
/// (run n37), all of it before the preparing, with the card idle behind it.
/// What the sync writes are host descriptor slots, read by the card when
/// the lists run, and its own records of what it saw; the preparing writes
/// host copies, upload memory and title memory (title copies), and reads
/// no host slot. Title threads mirror the views they make beside the
/// preparing already ([`crate::descriptor`]'s `mirror`). So the sync runs
/// beside it, as it ran before it, over the same pages and in the same
/// order, and is joined:
/// - before a sweep (which runs after it as before);
/// - before a settle inside the preparing without the relay (its copy-back
///   writes title memory the sync may read; with the relay the landing
///   thread writes it beside the sync either way);
/// - before the copies in the card's memory (which gather what the sync's
///   slots may have named since) and so before the lists go to the queue.
///
/// The slots that waited for a texture's host copy
/// (`crate::descriptor::make_waiting`) are made after the join, not before
/// the preparing: the waiting ones the sync adds are made in the same
/// submission, as they were after a sync before it. Not with levels on
/// demand (`crate::tiles::wanted`): the texture checks read what the sync
/// writes there (the first level a placed view shows).
///
/// Run n42 found the join waiting 28 of the sync's 35 ms a frame: most of
/// the sync's work is pages the title's other threads wrote while this
/// submission waited for the card or the landing thread, about the same in
/// every submission, while most of the preparing is in a few of them.
///
/// `VICTORY_SYNC_OVERLAP=2` (with the relay) instead syncs all along, on a
/// thread of its own, every [`sync_along_ms`] milliseconds, whether a
/// submission is under way or not ([`sync_along`]): the pages the title
/// wrote since the last pass, but those under flights' declared writes
/// (left for the next sync, as `VICTORY_SYNC_IN_WAIT`'s sync before a wait
/// leaves them), and the records left to tables' draws that are due. The
/// sync before each submission stays where it was, on the submitting
/// thread, with nothing changed: it takes what the passes left (the pages
/// written in the last milliseconds, those under flights' writes), so
/// everything the title wrote before the submission is mirrored before its
/// lists go to the card, as before. A pass and that sync do not run at
/// once (both hold the heaps' lock throughout); a pass and the sweep do not
/// either ([`ALONG_GATE`]). A pass beside the preparing is what `1` did.
pub(crate) fn sync_overlap() -> bool {
    sync_overlap_level() == 1
}

/// `VICTORY_SYNC_OVERLAP`: 0 (off), 1 (beside the preparing), 2 (all
/// along; only with the relay), 3 (the heap pages read and fingerprinted
/// all along, mirrored by the sync before each submission:
/// `crate::descriptor::sync_heaps_preread`).
///
/// `1` and `2` garbled the picture (runs n42, n45) and `2` slowed the
/// title's threads (20 to 63 ms a frame of their own time): both make
/// host slots, and the sets the submission's steps read beside them,
/// while a submission's steps run, and `2` held the heaps' lock and took
/// the locks title threads' view calls take, record by record, all along.
/// `3` makes nothing off the submitting thread: what host slots and those
/// sets hold when each submission's steps run is what the sync before the
/// preparing made, as with `0`.
fn sync_overlap_level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VICTORY_SYNC_OVERLAP").ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        Some("3") => 3,
        _ => 0,
    })
}

/// [`sync_overlap_level`] 2: how often a pass runs
/// (`VICTORY_SYNC_ALONG_MS`, 1 to 50; 2 by default).
fn sync_along_ms() -> u64 {
    static MS: OnceLock<u64> = OnceLock::new();
    *MS.get_or_init(|| {
        std::env::var("VICTORY_SYNC_ALONG_MS").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(2).clamp(1, 50)
    })
}

/// Held by each pass of [`sync_along`] and by the submitting thread around
/// a sweep: what the sweep lets go of was looked at by a sync before it.
static ALONG_GATE: Mutex<()> = Mutex::new(());

/// [`sync_overlap_level`] 2: starts the thread that syncs all along, once
/// (with the relay; without it, a settle's copy-back on the submitting
/// thread writes title memory and nothing is changed).
fn sync_along() {
    static STARTED: std::sync::Once = std::sync::Once::new();
    if sync_overlap_level() == 3 {
        // The pre-read: no relay needed (a copy-back that writes heap pages
        // sets their write watch, and the sync reads them again), nothing
        // made, so nothing for the sweep to wait for.
        STARTED.call_once(|| {
            let spawned = std::thread::Builder::new().name("victory-heap-preread".into()).spawn(|| {
                let every = std::time::Duration::from_millis(sync_along_ms());
                loop {
                    std::thread::sleep(every);
                    let started = std::time::Instant::now();
                    let found = std::panic::catch_unwind(crate::descriptor::sync_heaps_preread).unwrap_or(false);
                    PATH[path::SYNC_BESIDE].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                    PATH[path::ALONG_N].fetch_add(1, Ordering::Relaxed);
                    if found {
                        PATH[path::ALONG_FOUND_N].fetch_add(1, Ordering::Relaxed);
                    }
                }
            });
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "the heap pre-read's thread could not be started: the sync reads its pages itself");
            }
        });
        return;
    }
    if sync_overlap_level() != 2 || !relay() {
        return;
    }
    STARTED.call_once(|| {
        let spawned = std::thread::Builder::new().name("victory-heap-sync-along".into()).spawn(|| {
            let every = std::time::Duration::from_millis(sync_along_ms());
            loop {
                std::thread::sleep(every);
                let _gate = ALONG_GATE.lock().unwrap_or_else(|e| e.into_inner());
                // Pages under flights' declared writes wait for the sync
                // before a submission (the line's lock is let go first).
                let written = line().written.clone();
                let pages = crate::descriptor::SYNCED[0].load(Ordering::Relaxed);
                let started = std::time::Instant::now();
                let _ = std::panic::catch_unwind(|| crate::descriptor::sync_heaps_beside(&written));
                PATH[path::SYNC_BESIDE].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                PATH[path::ALONG_N].fetch_add(1, Ordering::Relaxed);
                if crate::descriptor::SYNCED[0].load(Ordering::Relaxed) != pages {
                    PATH[path::ALONG_FOUND_N].fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "the heap sync's own thread could not be started: the sync runs before each submission only");
        }
    });
}

/// The thread of [`sync_overlap`]: syncs asked for and done (counts), and
/// the words that one is asked for and that one is done.
struct Beside {
    state: Mutex<(u64, u64)>,
    asked: std::sync::Condvar,
    done: std::sync::Condvar,
}

/// [`Beside`], its thread started on first use (`None`: it could not be).
fn beside() -> Option<&'static Beside> {
    static BESIDE: OnceLock<Option<&'static Beside>> = OnceLock::new();
    *BESIDE.get_or_init(|| {
        let beside: &'static Beside = Box::leak(Box::new(Beside {
            state: Mutex::new((0, 0)),
            asked: std::sync::Condvar::new(),
            done: std::sync::Condvar::new(),
        }));
        let spawned = std::thread::Builder::new().name("victory-heap-sync".into()).spawn(move || {
            loop {
                let target = {
                    let mut state = beside.state.lock().unwrap_or_else(|e| e.into_inner());
                    while state.0 == state.1 {
                        state = beside.asked.wait(state).unwrap_or_else(|e| e.into_inner());
                    }
                    state.0
                };
                let started = std::time::Instant::now();
                // A panic must not leave the submitting thread waiting.
                let _ = std::panic::catch_unwind(crate::descriptor::sync_heaps);
                PATH[path::SYNC_BESIDE].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                let mut state = beside.state.lock().unwrap_or_else(|e| e.into_inner());
                state.1 = target;
                beside.done.notify_all();
            }
        });
        if let Err(e) = spawned {
            tracing::warn!(error = %e, "the heap sync's own thread could not be started: it runs before the preparing");
            return None;
        }
        Some(beside)
    })
}

/// [`sync_overlap`]: starts a heap sync on its thread; whether it did (if
/// not, the caller syncs itself).
pub(crate) fn sync_beside_start() -> bool {
    let Some(beside) = beside() else {
        return false;
    };
    let mut state = beside.state.lock().unwrap_or_else(|e| e.into_inner());
    // One at a time: one asked before and not joined is waited for first.
    while state.1 < state.0 {
        state = beside.done.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    state.0 += 1;
    beside.asked.notify_one();
    true
}

/// [`sync_overlap`]: if a sync runs beside (`on`), waits until it is done
/// (and notes it is not on any more).
pub(crate) fn sync_beside_join(on: &mut bool) {
    if !std::mem::take(on) {
        return;
    }
    let Some(beside) = beside() else {
        return;
    };
    let waiting = std::time::Instant::now();
    let mut state = beside.state.lock().unwrap_or_else(|e| e.into_inner());
    while state.1 < state.0 {
        state = beside.done.wait(state).unwrap_or_else(|e| e.into_inner());
    }
    drop(state);
    PATH[path::SYNC_JOIN].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
}

/// The thread that holds the submission lock stood waiting for the card
/// from `since` until now ([`CARD_CLOCK_US`]).
fn stood(since: std::time::Instant) {
    card_clock_us(2, since.elapsed().as_micros() as u64);
}

/// What [`CARD_CLOCK_US`] is measured with.
struct CardClock {
    fence: ID3D12Fence,
    /// Signals sent so far; and, while the card has reached the last of
    /// them, since when and what [`worked`] and [`held_so_far`] said then.
    state: Mutex<(u64, Option<(std::time::Instant, u64, u64)>)>,
    sent: std::sync::Condvar,
}

// SAFETY: a fence (free-threaded) behind a lock.
unsafe impl Send for CardClock {}
// SAFETY: as above.
unsafe impl Sync for CardClock {}

/// The clock's thread: waits for the newest signal sent and notes when the
/// card got there with nothing newer behind it.
fn card_clock_watch(clock: &'static CardClock) {
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        return;
    };
    let mut seen = 0;
    loop {
        let target = {
            let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
            while state.0 == seen {
                state = clock.sent.wait(state).unwrap_or_else(|e| e.into_inner());
            }
            state.0
        };
        // SAFETY: the clock's fence and this thread's event.
        unsafe {
            if clock.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        seen = target;
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        if state.0 == target && state.1.is_none() {
            state.1 = Some((std::time::Instant::now(), worked(), held_so_far()));
        }
    }
}

/// For the frame-rate line: [`CARD_CLOCK_US`] per frame, `ms` being the
/// frame's time.
fn card_clock_line(frames: f64, ms: f64) -> [f64; 4] {
    let [idle, behind, stood, waits] = [0, 1, 2, 3].map(|i| CARD_CLOCK_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
    tracing::info!(
        "the card's clock per frame: nothing of ours queued {idle:.1} ms (the card about {:.0}% busy by this count), \
         {behind:.1} ms of that while the submitting thread was working (the card idle behind us), {waits:.1} while it \
         held the submission lock and stood waiting (for the landing thread), the rest while the title had not submitted; \
         the submitting thread stood waiting for the card {stood:.1} ms",
        if ms > 0.0 { (100.0 * (1.0 - idle / ms)).clamp(0.0, 100.0) } else { 0.0 }
    );
    [idle, behind, stood, waits]
}

/// The frame's path, per frame, for the frame-rate line's period: where a
/// frame of `ms` goes on the thread that submits (the submission lock held,
/// of it working and waiting; the title's own time between submissions),
/// where the card stands idle (`clock`: [`card_clock_line`]'s idle, behind
/// our work, stood, behind our waits), and what the landing thread and the
/// heap pass take. "Serial" is the time the card stood idle while a thread
/// of ours worked or stood waiting for our own landing: what the next cut
/// is to overlap.
fn path_line(frames: f64, ms: f64, clock: [f64; 4]) {
    static LAST: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];
    let (held_now, worked_now) = (held_so_far(), worked());
    let held = held_now.saturating_sub(LAST[0].swap(held_now, Ordering::Relaxed)) as f64 / 1e3 / frames;
    let work = worked_now.saturating_sub(LAST[1].swap(worked_now, Ordering::Relaxed)) as f64 / 1e3 / frames;
    let v: Vec<u64> = PATH.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
    let t = |i: usize| v[i] as f64 / 1e3 / frames;
    let n = |i: usize| v[i] as f64 / frames;
    let [idle, behind, _, waits] = clock;
    let parts = {
        let p: Vec<f64> = WORK_PARTS.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames).collect();
        let named = [
            "before the sync",
            "the heap sync (or the wait for it beside)",
            "sweep and waiting slots",
            "asking once and stale checks",
            "the lists' steps and flushes",
            "snapshots and card copies",
            "sending and after",
        ];
        let mut text = named.iter().zip(&p).map(|(name, ms)| format!("{name} {ms:.1}")).collect::<Vec<_>>().join(", ");
        text.push_str(&format!(", elsewhere {:.1}", (work - p.iter().sum::<f64>()).max(0.0)));
        text
    };
    tracing::info!(
        "the frame's path per frame: {ms:.1} ms; the submission lock held {held:.1} (working {:.1}, of it the heap sync \
         {:.1} [{:.1} of it before title-read waits over {:.1} pages, {:.1} pages left for after them as flights' shaders \
         may write them, VICTORY_SYNC_IN_WAIT={}, VICTORY_SYNC_WIDE={}]; working by part: {}; the heap sync beside the \
         preparing (VICTORY_SYNC_OVERLAP={}): {:.1} of {:.1} submissions, {:.1} on its own thread, {:.1} waited for where \
         joined ({:.1} joins before a sweep or a settle, {:.1} ms waited at the end of the preparing); all along \
         (VICTORY_SYNC_OVERLAP=2, every {} ms): {:.1} passes, {:.1} of them found pages written; fills on several threads (VICTORY_WIDE_FILLS={}): {}; standing waiting for the \
         card or the landing thread {:.1}); between submissions, the title's own time {:.1} in {:.1} gaps (the longest {:.1}); \
         the card idle {idle:.1}: behind our work {behind:.1}, behind our waits {waits:.1}, the title not submitting {:.1}; \
         serial (the card idle behind us) {:.1}; the landing thread busy {:.1}, fetching card pieces {:.1} ({:.1} fetches on \
         its copy queue, {:.1} pieces copied there behind a later flight that declared them, {:.1} fetched again on the host \
         queue, VICTORY_RELAY_COPY_QUEUE={}); heap buffer passes {:.1} on the submitting thread, {:.1} on others{}{}",
        work.min(held),
        t(path::SYNC),
        t(path::SYNC_EARLY),
        crate::descriptor::EARLY[0].swap(0, Ordering::Relaxed) as f64 / frames,
        crate::descriptor::EARLY[1].swap(0, Ordering::Relaxed) as f64 / frames,
        u8::from(sync_in_wait()),
        crate::descriptor::sync_wide(),
        parts,
        sync_overlap_level(),
        n(path::SYNC_BESIDE_N),
        n(path::SUBMITS_N),
        t(path::SYNC_BESIDE),
        t(path::SYNC_JOIN),
        n(path::JOIN_EARLY_N),
        t(path::JOIN_LATE_US),
        sync_along_ms(),
        n(path::ALONG_N),
        n(path::ALONG_FOUND_N),
        u8::from(wide_fills()),
        fills_line(frames),
        (held - work).max(0.0),
        t(path::GAPS),
        n(path::GAPS_N),
        v[path::GAP_MOST] as f64 / 1e3,
        (idle - behind - waits).max(0.0),
        behind + waits,
        t(path::LAND),
        t(path::FETCH),
        n(path::FETCH_COPY_N),
        n(path::BEHIND_N),
        n(path::AGAIN_N),
        u8::from(relay_copy_queue()),
        t(path::HEAP_SUBMITTER),
        t(path::HEAP_OTHER),
        crate::cpuwait::text(frames),
        crate::airlists::text(frames, late_most()),
    );
}

/// Of the preparing: title memory copies (microseconds, bytes) and texture
/// refreshes (microseconds, bytes hashed), since the last frame-rate line.
pub(crate) static PREPARING: [std::sync::atomic::AtomicU64; 4] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];

/// The whole of a submission's preparing (the frame line's "preparing",
/// `started` to `prepared` in [`Host::submit`]), timed where each part runs
/// on the submitting thread, in nanoseconds since the last frame-rate line
/// ([`prep`] names the parts); what no part holds is the line's "the rest".
/// Then the thread's time around the preparing, which "preparing" does not
/// hold, and the steps run, by kind.
pub(crate) static PREP_SPLIT: [std::sync::atomic::AtomicU64; prep::ALL] =
    [const { std::sync::atomic::AtomicU64::new(0) }; prep::ALL];

/// The parts of [`PREP_SPLIT`].
pub(crate) mod prep {
    /// Steps of the lists, each without the refresh flushes it set off:
    /// queuing refreshes (and the lists' own bookkeeping, their lock and
    /// the loop), the title's buffer copies, fills of upload memory, our
    /// work that may write title memory, our work on host memory only,
    /// declarations of what shaders write, texture checks.
    pub const QUEUE: usize = 0;
    pub const COPY: usize = 1;
    pub const FILL: usize = 2;
    pub const RUN: usize = 3;
    pub const HOST: usize = 4;
    pub const DECLARE: usize = 5;
    pub const TEXTURE: usize = 6;
    /// Refresh flushes (the write-watch asking ahead, the moves), wherever
    /// a step or the end of the steps set them off.
    pub const REFRESH: usize = 7;
    /// The one asking a submission (`VICTORY_ASK_ONCE`) with the spans it
    /// is built from, without a settle inside it.
    pub const ONCE: usize = 8;
    /// Whether a refresh would put stale bytes under flights' writes.
    pub const STALE: usize = 9;
    /// The lists (and preludes) gathered for the queue.
    pub const LISTS: usize = 10;
    /// The end of the submission's asking.
    pub const ASK_END: usize = 11;
    /// The snapshots left for the end (`crate::shadow::take_snapshots`).
    pub const SNAPSHOTS: usize = 12;
    /// Copies in the card's memory (`Host::card_before`): gathering the
    /// declared ranges; taking the changed ranges (and first-time copies
    /// of what was before); recording the sends and hash passes; filling
    /// upload memory for them; waiting for the card between segments of
    /// the sends (and for the queue's order, with the relay); the compare
    /// after the lists; the check whether the
    /// compare's results are half taken.
    pub const CARD_GATHER: usize = 13;
    pub const CARD_DIRTY: usize = 14;
    pub const CARD_SENDS: usize = 15;
    pub const CARD_FILLS: usize = 16;
    pub const CARD_WAITS: usize = 17;
    pub const CARD_COMPARE: usize = 18;
    pub const CARD_LOW: usize = 19;
    /// Inside "preparing" but taken out of it: settles of flights.
    pub const SETTLED_IN: usize = 20;
    /// Around the preparing (not in it): the wait for the submission lock,
    /// settles before it (with the check whether one is due), the sweep,
    /// slots waiting for textures, and after it the sending to the queue
    /// with what a submission sent ahead takes along.
    pub const LOCK: usize = 21;
    pub const SETTLE_BEFORE: usize = 22;
    pub const SWEEP: usize = 23;
    pub const WAITING_SLOTS: usize = 24;
    pub const SEND: usize = 25;
    /// Steps run, by the kinds above (`QUEUE` counts refreshes queued);
    /// then of the declarations, those first in their submission.
    pub const COUNTS: usize = 26;
    pub const FIRST_DECLARED: usize = COUNTS + 7;
    pub const ALL: usize = FIRST_DECLARED + 1;
}

/// Of some parts of [`PREP_SPLIT`], since the last frame-rate line, on the
/// submitting thread: nanoseconds, and counts ([`more`] names them).
pub(crate) static PREP_MORE: [std::sync::atomic::AtomicU64; more::ALL] =
    [const { std::sync::atomic::AtomicU64::new(0) }; more::ALL];

/// The parts of [`PREP_MORE`].
pub(crate) mod more {
    /// Of "asking once": gathering the spans (and with views each range)
    /// from the lists' steps; the asking of the spans
    /// (`crate::watch::ask_once`); the asking of the ranges in view memory
    /// (`crate::watch::ask_once_views`).
    pub const ONCE_SPANS: usize = 0;
    pub const ONCE_ASK: usize = 1;
    pub const ONCE_VIEWS: usize = 2;
    /// Of the card copies' "sends": recording the copies up (with the
    /// upload room they take), of that taking new upload chunks (and how
    /// many), recording the hash passes, and closing the list and sending
    /// it to the queue (and how many times).
    pub const SEND_COPIES: usize = 3;
    pub const SEND_CHUNKS: usize = 4;
    pub const SEND_CHUNKS_N: usize = 5;
    pub const SEND_HASH: usize = 6;
    pub const SEND_SUBMIT: usize = 7;
    pub const SEND_SUBMIT_N: usize = 8;
    /// Of "refresh flushes": one copy's before a shader-write declaration
    /// (and how many found a refresh pending), all of them (and how many
    /// flushes had any).
    pub const FLUSH_ONE: usize = 9;
    pub const FLUSH_ONE_N: usize = 10;
    pub const FLUSH_ALL: usize = 11;
    pub const FLUSH_ALL_N: usize = 12;
    /// Sends' lists that went to the queue with the submission's own lists
    /// (`VICTORY_CARD_SYNC_JOIN`).
    pub const SYNC_JOINED_N: usize = 13;
    /// Of "one copy before a declaration", with `VICTORY_FLUSH_ONE_PART`
    /// (`super::flush_one_part`): the flushes that refreshed only the part
    /// declared, and that part's bytes (what they asked about; the rest of
    /// each copy's pending range is left to the flush of all).
    pub const FLUSH_PART_N: usize = 14;
    pub const FLUSH_PART_BYTES: usize = 15;
    /// With `VICTORY_DECLARED_WIDE` (`super::declared_wide`): declarations
    /// whose copy's refresh was left to the flush of all, and the bytes of
    /// the parts they left; of the flushes of all with `VICTORY_WIDE_COPY`,
    /// the planning on the submitting thread and the moves (wall time).
    pub const DECLARED_N: usize = 16;
    pub const DECLARED_BYTES: usize = 17;
    pub const FLUSH_PLAN: usize = 18;
    pub const FLUSH_MOVE: usize = 19;
    /// With `VICTORY_WIDE_TEXTURES` (`super::wide_textures`): texture checks
    /// gathered, the flushes that did two or more on several threads, and
    /// all the flushes' time on the submitting thread.
    pub const TEX_WIDE_N: usize = 20;
    pub const TEX_WIDE_FLUSHES: usize = 21;
    pub const TEX_WIDE: usize = 22;
    /// Of the stale checks (`super::stale_under_flights`): refresh steps
    /// looked at, copies among them, copies with shader writes pending
    /// (the spans checked).
    pub const STALE_STEPS: usize = 23;
    pub const STALE_COPIES: usize = 24;
    pub const STALE_SPANS: usize = 25;
    pub const ALL: usize = 26;
}

fn more_add(part: usize, ns: u64) {
    PREP_MORE[part].fetch_add(ns, Ordering::Relaxed);
}

/// The frame line's further split of some parts of the preparing, per frame.
fn more_line(frames: f64) -> String {
    use more::*;
    let v: Vec<u64> = PREP_MORE.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
    let ms = |i: usize| v[i] as f64 / 1e6 / frames;
    let n = |i: usize| v[i] as f64 / frames;
    format!(
        "asking once [spans {:.1}, asking {:.1}, view memory {:.1}], sends [copies {:.1} (new upload chunks {:.0}, {:.1}), \
         hash passes {:.1}, close and to the queue {:.0}x {:.1}, {:.0} joined to the submission], refresh flushes \
         [one copy before a declaration {:.0}x {:.1} (VICTORY_FLUSH_ONE_PART {}: {:.0} the declared part only, \
         {:.1} MB of it asked about; VICTORY_DECLARED_WIDE {}: {:.0} left to the flush of all, {:.1} MB), all {:.0}x {:.1} \
         (planning {:.1}, moves {:.1})], texture checks (VICTORY_WIDE_TEXTURES {}): {:.0} gathered, {:.1} flushes on \
         several threads, {:.1} ms; stale checks (VICTORY_STALE_ONCE {}): {:.0} refresh steps, {:.1} copies, {:.1} with \
         shader writes pending; {}; {}; {}",
        ms(ONCE_SPANS),
        ms(ONCE_ASK),
        ms(ONCE_VIEWS),
        ms(SEND_COPIES),
        n(SEND_CHUNKS_N),
        ms(SEND_CHUNKS),
        ms(SEND_HASH),
        n(SEND_SUBMIT_N),
        ms(SEND_SUBMIT),
        n(SYNC_JOINED_N),
        n(FLUSH_ONE_N),
        ms(FLUSH_ONE),
        if flush_one_part() { "on" } else { "off" },
        n(FLUSH_PART_N),
        n(FLUSH_PART_BYTES) / 1e6,
        if declared_wide() { "on" } else { "off" },
        n(DECLARED_N),
        n(DECLARED_BYTES) / 1e6,
        n(FLUSH_ALL_N),
        ms(FLUSH_ALL),
        ms(FLUSH_PLAN),
        ms(FLUSH_MOVE),
        if wide_textures() { "on" } else { "off" },
        n(TEX_WIDE_N),
        n(TEX_WIDE_FLUSHES),
        ms(TEX_WIDE),
        if stale_once() { "on" } else { "off" },
        n(STALE_STEPS),
        n(STALE_COPIES),
        n(STALE_SPANS),
        crate::watch::asking_cost_line(frames),
        String::new(),
        crate::draw::record_line(frames),
    )
}

/// Adds `since`'s time to part `part` of [`PREP_SPLIT`].
fn prep_add(part: usize, since: std::time::Instant) {
    PREP_SPLIT[part].fetch_add(since.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

/// The frame line's split of the preparing, `cpu` ms per frame: the parts
/// of [`PREP_SPLIT`] per frame in ms, and what no part holds.
fn prep_line(frames: f64, cpu: f64) -> String {
    use prep::*;
    let v: Vec<f64> = PREP_SPLIT
        .iter()
        .enumerate()
        .map(|(i, n)| {
            let n = n.swap(0, Ordering::Relaxed) as f64;
            if i >= COUNTS { n / frames } else { n / 1e6 / frames }
        })
        .collect();
    // Submissions started since the last line (their fixed costs repeat).
    static LAST_STARTED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let started = STARTED.load(Ordering::Relaxed);
    let submissions = started.saturating_sub(LAST_STARTED.swap(started, Ordering::Relaxed)) as f64 / frames;
    let card: f64 = v[CARD_GATHER..=CARD_LOW].iter().sum();
    let parts: f64 = v[..=CARD_LOW].iter().sum();
    let c = |k: usize| v[COUNTS + k];
    format!(
        "{submissions:.1} submissions; refresh flushes {:.1}, steps [{:.0} refreshes queued and bookkeeping {:.1}, {:.0} title copies {:.1}, {:.0} fills {:.1}, \
         {:.0} our steps that may write title memory {:.1}, {:.0} host steps {:.1}, {:.0} shader-write declarations ({:.0} first in their submission) {:.1}, \
         {:.0} texture checks {:.1}], asking once {:.1}, stale checks {:.1}, lists {:.1}, asking's end {:.1}, snapshots {:.1}, \
         card copies {card:.1} [gathering {:.1}, changed ranges {:.1}, sends {:.1}, fills {:.1}, waits for the card or the queue's order {:.1}, \
         compare {:.1}, results check {:.1}], the rest {:.1}; not in it: settles inside {:.1}, before it lock {:.1}, \
         settles {:.1}, sweep {:.1}, waiting slots {:.1}, after it sending {:.1}",
        v[REFRESH],
        c(QUEUE),
        v[QUEUE],
        c(COPY),
        v[COPY],
        c(FILL),
        v[FILL],
        c(RUN),
        v[RUN],
        c(HOST),
        v[HOST],
        c(DECLARE),
        v[FIRST_DECLARED],
        v[DECLARE],
        c(TEXTURE),
        v[TEXTURE],
        v[ONCE],
        v[STALE],
        v[LISTS],
        v[ASK_END],
        v[SNAPSHOTS],
        v[CARD_GATHER],
        v[CARD_DIRTY],
        v[CARD_SENDS],
        v[CARD_FILLS],
        v[CARD_WAITS],
        v[CARD_COMPARE],
        v[CARD_LOW],
        (cpu - parts).max(0.0),
        v[SETTLED_IN],
        v[LOCK],
        v[SETTLE_BEFORE],
        v[SWEEP],
        v[WAITING_SLOTS],
        v[SEND],
    )
}

/// Host buffers made since the start, as (count, bytes): in the upload heap
/// (staging, list scratch), shared with the processor (host copies of title
/// memory) and the graphics card's own (indirect arguments, twins' copies).
/// For the memory line: against what is alive, they show what was let go.
pub(crate) static MADE: [std::sync::atomic::AtomicU64; 6] = [const { std::sync::atomic::AtomicU64::new(0) }; 6];

/// What of that is alive now, for the memory line (`MADE` counts what was
/// ever made: 42,607 upload buffers, 77.9 GB, in five minutes of run 60,
/// and nothing said how many were alive): host lists; the lists' scratch
/// chunks and their bytes; of those, the chunks no list holds (kept for the
/// next list, [`Spare`]) and their bytes; chunks taken from those kept
/// instead of made, since the start; and the lists' other upload buffers
/// (texture data copied from buffers, scratch too big for a chunk) and
/// their bytes.
pub(crate) static ALIVE: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// About host lists, for the memory line: lists made by the host since the
/// start; preludes made (a list's second allocator and list, made when it
/// first needs one, [`lazy_prelude`]); and, in diagnostics builds, the
/// bytes the process's private memory grew by while lists were being made
/// (what a list costs: host objects with the driver's command memory; 631
/// lists were alive in run 65, each with a prelude it may never use).
/// Then the lists alive that were never reset since they were made: the
/// host sides of title lists the title made and has not used (a host list
/// cost 0.68 MB of private memory in run 69: 431 MB for the 632 the title
/// makes at its start).
pub(crate) static LISTS: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The process's private bytes (diagnostics: around the making of lists).
#[cfg(feature = "diagnostics")]
fn private_bytes() -> u64 {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut c = PROCESS_MEMORY_COUNTERS {
        cb: size_of::<PROCESS_MEMORY_COUNTERS>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), &mut c, c.cb) };
    c.PagefileUsage as u64
}

/// What the runtime does when the host GPU fails for good: report it like a
/// crash and end the run. The console has no such state, and carrying on
/// draws nothing at seconds per frame.
static ON_FAILURE: OnceLock<fn(&str) -> !> = OnceLock::new();

/// Sets what a host GPU failure does (the runtime's crash report).
pub fn on_failure(report: fn(&str) -> !) {
    let _ = ON_FAILURE.set(report);
}

/// Ends the run on a host GPU failure: what failed, the removal reason and
/// the debug layer's first error, if any (`VICTORY_D3D12_DEBUG=1` in a
/// diagnostics build names the command).
pub(crate) fn fail(what: std::fmt::Arguments) -> ! {
    let mut text = what.to_string();
    if let Some(host) = host() {
        // SAFETY: a plain query.
        if let Err(e) = unsafe { host.device.GetDeviceRemovedReason() } {
            text.push_str(&format!("; device removed: {e}"));
            log_removal_context();
        }
        match host.log_messages() {
            Some(first) => text.push_str(&format!("; debug layer: {first}")),
            None if host.messages.is_none() => text.push_str(
                "; a diagnostics build with VICTORY_D3D12_DEBUG=1 names the command",
            ),
            None => {}
        }
    }
    tracing::error!("{text}");
    // Tests have no log: the reason goes to their output.
    #[cfg(test)]
    eprintln!("host GPU failure: {text}");
    match ON_FAILURE.get() {
        Some(report) => report(&text),
        None => std::process::abort(),
    }
}

/// The host GPU, created on first use; `None` without one (a run then
/// behaves as before: nothing executes).
pub(crate) fn host() -> Option<&'static Host> {
    static HOST: OnceLock<Option<Host>> = OnceLock::new();
    HOST.get_or_init(|| match create() {
        Ok(host) => Some(host),
        Err(e) => {
            tracing::warn!(error = %e, "no host D3D12 device; command lists will not run");
            None
        }
    })
    .as_ref()
}

fn create() -> windows::core::Result<Host> {
    crate::drvwatch::start();
    crate::spread::start();
    // SAFETY: plain D3D12 object creation with valid descriptions.
    unsafe {
        // The debug layer only in builds with diagnostics.
        let debug =
            cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DEBUG").is_some();
        if debug {
            let mut layer: Option<ID3D12Debug> = None;
            match D3D12GetDebugInterface(&mut layer) {
                Ok(()) => layer.iter().for_each(|l| l.EnableDebugLayer()),
                Err(e) => tracing::warn!(error = %e, "no D3D12 debug layer"),
            }
            // GPU-based validation (`VICTORY_D3D12_GBV`): out-of-range
            // descriptor heap indices and the like become messages naming
            // the command instead of page faults. Without state tracking
            // (directly indexed textures are not transitioned).
            if std::env::var_os("VICTORY_D3D12_GBV").is_some()
                && let Some(l) = &layer
            {
                if let Ok(d1) = l.cast::<ID3D12Debug1>() {
                    d1.SetEnableGPUBasedValidation(true);
                }
                if let Ok(d2) = l.cast::<ID3D12Debug2>() {
                    d2.SetGPUBasedValidationFlags(
                        D3D12_GPU_BASED_VALIDATION_FLAGS_DISABLE_STATE_TRACKING,
                    );
                }
                tracing::info!("GPU-based validation on");
            }
        }
        // Diagnostics: device-removed data (the command that faulted, the
        // faulting address), read when the device is removed.
        if dred_on() {
            let mut dred: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
            if D3D12GetDebugInterface(&mut dred).is_ok()
                && let Some(dred) = dred
            {
                dred.SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
                dred.SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            }
        }
        let mut device: Option<ID3D12Device> = None;
        // Diagnostics: the software adapter, on which a shader's wild access
        // is an access violation in this process (a crash report with the
        // address and instruction) instead of a GPU fault and driver reset.
        if cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_WARP").is_some() {
            let adapter = warp_adapter()?;
            D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device)?;
            tracing::info!("host device on the software adapter (WARP)");
        } else {
            D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device)?;
        }
        let device = device.ok_or_else(windows::core::Error::empty)?;
        // `VICTORY_D3D12_BACKGROUND=0`: the runtime's background work (the
        // driver compiling shaders again on the "D3D Background Thread"s,
        // for minutes after pipelines were made or loaded) is switched off.
        // An experiment: while those threads ran, the same shot took
        // 1,000-1,200 ms a frame, and 830-950 ms from the moment they
        // stopped (run 54, 08:32:01 to 08:32:06). Left as it is by default:
        // whether the card is slower with the shaders it then keeps is not
        // known.
        if std::env::var_os("VICTORY_D3D12_BACKGROUND").is_some_and(|v| v == "0")
            && let Ok(device6) = device.cast::<ID3D12Device6>()
        {
            let set = device6.SetBackgroundProcessingMode(
                D3D12_BACKGROUND_PROCESSING_MODE_DISABLE_BACKGROUND_WORK,
                D3D12_MEASUREMENTS_ACTION_KEEP_ALL,
                None,
                None,
            );
            tracing::info!(ok = set.is_ok(), "the host's background shader work is switched off (VICTORY_D3D12_BACKGROUND=0)");
        }
        // the widest wave the GPU runs: 32 (NVIDIA) gets console wave64 compute programs whose
        // cross-lane code stays in 32-lane halves translated for waves of 32
        let mut o1 = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
        if device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS1, (&raw mut o1).cast(), size_of_val(&o1) as u32)
            .is_ok()
        {
            tracing::info!(min = o1.WaveLaneCountMin, max = o1.WaveLaneCountMax, "host wave lanes");
            victory_shader::emit::set_host_max_wave(o1.WaveLaneCountMax);
        }
        let messages = debug
            .then(|| device.cast::<ID3D12InfoQueue>().ok())
            .flatten();
        let queue: ID3D12CommandQueue = device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
            Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
            ..Default::default()
        })?;
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
        let event = CreateEventW(None, false, false, None)?;
        let rtvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let rtv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
        let dsvs: ID3D12DescriptorHeap =
            device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                NumDescriptors: RTV_RING,
                ..Default::default()
            })?;
        let dsv_size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV);
        let adapter = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
            .ok();
        tracing::info!("host D3D12 device ready");
        Ok(Host {
            device,
            queue,
            fence,
            submitted: Mutex::new(0),
            event,
            rtvs,
            rtv_size,
            dsvs,
            dsv_size,
            next_dsv: AtomicU32::new(0),
            messages,
            next_rtv: AtomicU32::new(0),
            presenter: Mutex::new(None),
            adapter,
        })
    }
}

/// DXGI may return hardware for `EnumWarpAdapter` (notably under Proton).
/// Reject it before creating a device in diagnostics' software-only mode.
pub(crate) fn warp_adapter() -> windows::core::Result<IDXGIAdapter> {
    // SAFETY: plain DXGI adapter queries.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
        let adapter: IDXGIAdapter = factory.EnumWarpAdapter()?;
        let desc = adapter.cast::<IDXGIAdapter1>()?.GetDesc1()?;
        if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
            tracing::warn!(
                vendor = format_args!("{:#x}", desc.VendorId),
                flags = format_args!("{:#x}", desc.Flags),
                "DXGI did not identify the WARP adapter as software; refusing device"
            );
            return Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_FAIL,
            ));
        }
        Ok(adapter)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn sparse_copies_preserve_islands_and_observe_remaps() {
        use windows_sys::Win32::System::Memory::*;
        const N: usize = 5 << 16;
        // SAFETY: we reserve a private region, commit selected pages inside it,
        // and only directly access the pages while they are writable.
        unsafe {
            let base = VirtualAlloc(std::ptr::null(), N, MEM_RESERVE, PAGE_NOACCESS).cast::<u8>();
            assert!(!base.is_null());
            let mut expected = vec![0; N];
            for (offset, value) in [(0, 9), (2 << 16, 37), (N - 4096, 85)] {
                let at = base.add(offset);
                assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                at.write_bytes(value, 4096);
                expected[offset..offset + 4096].fill(value);
            }
            for remap in [false, true] {
                if remap {
                    let mut old = 0;
                    assert_ne!(VirtualProtect(base.cast(), 4096, PAGE_NOACCESS, &mut old), 0);
                    expected[..4096].fill(0);
                    let at = base.add(1 << 16);
                    assert_eq!(VirtualAlloc(at.cast(), 4096, MEM_COMMIT, PAGE_READWRITE), at.cast());
                    at.write_bytes(0xc7, 4096);
                    expected[1 << 16..(1 << 16) + 4096].fill(0xc7);
                }
                let mut out = vec![0xee; N];
                super::copy_mapped(out.as_mut_ptr().add(27), base.add(27), N - 54, false);
                assert_eq!(out[27..N - 27], expected[27..N - 27]);
                assert!(out[..27].iter().chain(&out[N - 27..]).all(|&v| v == 0xee));
            }
            assert_ne!(VirtualFree(base.cast(), 0, MEM_RELEASE), 0);
        }
    }

    use super::*;

    #[test]
    fn warp_adapter_accepts_only_software() {
        // This probes DXGI without creating a D3D12 device or running a title.
        // SAFETY: plain DXGI adapter queries.
        let raw = unsafe {
            let Ok(factory) = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            else {
                return;
            };
            factory.EnumWarpAdapter::<IDXGIAdapter>()
        };
        let Ok(raw) = raw else { return };
        // SAFETY: EnumWarpAdapter returns a DXGI adapter, which implements
        // IDXGIAdapter1; `raw` owns that interface.
        let desc = unsafe { raw.cast::<IDXGIAdapter1>().unwrap().GetDesc1().unwrap() };
        eprintln!(
            "EnumWarpAdapter: vendor={:#x}, flags={:#x}",
            desc.VendorId, desc.Flags
        );
        assert_eq!(
            warp_adapter().is_ok(),
            desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0,
        );
    }
}

/// A host command list and its allocator, always used together.
pub(crate) struct HostList {
    allocator: ID3D12CommandAllocator,
    pub(crate) list: ID3D12GraphicsCommandList,
    /// Work for submission time, in recorded order.
    deferred: Mutex<Deferred>,
    /// Upload memory for data the list's draws make (expanded vertices),
    /// reused after a reset.
    scratch: Mutex<Scratch>,
    /// Host texture states the list assumes, by host copy: at its start and
    /// at its end. Lists are recorded on many threads and run in submission
    /// order, so the transitions from a texture's real state are made at
    /// submission, in `prelude`. Holding the copies keeps every texture the
    /// list names alive until it is reset, even if the title releases the
    /// resource first.
    states: Mutex<States>,
    /// What the list does with depth textures that have a colour twin, by
    /// twin ([`TwinUse`]).
    twins: Mutex<std::collections::HashMap<usize, TwinUse>>,
    /// The list that runs those transitions just before this one, made
    /// when the list first has such transitions ([`lazy_prelude`]).
    prelude: OnceLock<Option<Prelude>>,
    /// With [`table_gpu_copy`]: card memory its tables' constant views are
    /// copied into, and the list that copies them, run after the prelude
    /// (made when first needed).
    card_ring: Mutex<CardRing>,
    card_pre: OnceLock<Option<Prelude>>,
    /// The pipelines the list binds, for reports of slow submissions.
    pipelines: Mutex<std::collections::BTreeSet<String>>,
    /// The work its indirect executions may still do (`indirect`): a
    /// shader-writable word, zeroed when the list is submitted, and after
    /// it what the clamp notes about mesh records. In memory the CPU maps.
    budget: OnceLock<Option<(ID3D12Resource, usize)>>,
    /// The same words in the card's own memory ([`card_budget`]), which the
    /// clamp uses instead when there is one: zeroed in the list before its
    /// first clamp, its notes copied to `budget` when the list closes.
    card_budget: OnceLock<Option<ID3D12Resource>>,
    /// Device-local scratch buffer for clamped indirect arguments (`indirect`),
    /// grown to the largest indirect execution recorded on the list.
    indirect_args: Mutex<Option<(ID3D12Resource, u64)>>,
    /// The buffer runs of indirect calls clamp their records into
    /// (`VICTORY_INDIRECT_BATCH`, `indirect::Run`).
    batch_args: Mutex<crate::indirect::BatchArgs>,
    /// The buffers of `crate::indirect::window_check`, made at its first use.
    window_check: OnceLock<Option<crate::indirect::WindowCheck>>,
    /// Counts the list in [`ALIVE`] while it lives.
    _alive: Counted,
    /// With `VICTORY_GPU_PROFILE=1`: what the recording times
    /// ([`crate::gpuprof`]).
    prof: Mutex<crate::gpuprof::Recording>,
    /// Its barrier calls and draws, and what `VICTORY_BARRIER_LAZY_UAV`
    /// tracks ([`crate::barstat::Track`]).
    bar: crate::barstat::Track,
    /// The pair of timestamps its run is timed with since its last reset
    /// (`crate::cardmeter`), or none.
    meter: AtomicU32,
    /// With `crate::cardmeter::pipelines`: the pair of the pipeline bound
    /// last, open until the next one is bound or the list closes.
    pipeline_meter: AtomicU32,
    /// The texts of the markers `note_op` set, in order (for device-removed
    /// data without marker texts), with the host GPU range a copy reads.
    #[cfg(feature = "diagnostics")]
    notes: Mutex<Vec<Note>>,
    /// With `VICTORY_GPU_TIMING`: a GPU timestamp before each noted command
    /// and at the end, and where they are read back.
    #[cfg(feature = "diagnostics")]
    stamps: OnceLock<Option<(ID3D12QueryHeap, ID3D12Resource)>>,
}

/// A list's prelude: an allocator and a closed list.
type Prelude = (ID3D12CommandAllocator, ID3D12GraphicsCommandList);

/// Whether a host list's prelude is made when the list first needs one. A
/// prelude runs the texture transitions and twin copies a list needs before
/// it; lists that name no texture never have one, and every host list was
/// made with one all the same: twice the host objects per list, for the
/// title's 630 lists and for every list made anew.
/// `VICTORY_LAZY_PRELUDE=0`: made with the list, as before.
fn lazy_prelude() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LAZY_PRELUDE").is_none_or(|v| v != "0"))
}

/// Timestamps per list (diagnostics); commands past them are not timed.
#[cfg(feature = "diagnostics")]
const STAMPS: u32 = 4096;

/// A marker's text and the host range its command reads (diagnostics).
#[cfg(feature = "diagnostics")]
type Note = (String, Option<(u64, u64)>);

type States = std::collections::HashMap<
    usize,
    (u32, u32, std::sync::Arc<crate::resource::HostTexture>),
>;

/// What a list does with a depth texture that has a colour twin
/// ([`crate::resource::Twin`]). Lists are recorded on many threads and run
/// in submission order, so a list cannot know which side an earlier list
/// leaves behind: it notes which sides it uses as it finds them, and its
/// prelude copies one side into the other at submission if that side is
/// behind then. From its first use of a side on, the list knows.
struct TwinUse {
    /// The depth texture (the title resource's host copy) and its twin.
    depth: std::sync::Arc<crate::resource::HostTexture>,
    twin: std::sync::Arc<crate::resource::Twin>,
    /// The sides (depth texture, twin) the list uses as it finds them.
    needs: [bool; 2],
    /// Whether each side lacks what the other was written last, where the
    /// list is now; `None`: as the list found it.
    behind: [Option<bool>; 2],
    /// The draw or dispatch being prepared writes the twin.
    writing: bool,
    /// The depth plane's twin holds what the plane lacks although the list
    /// copied the twins since: the plane's pass waits for the list's close.
    late: bool,
}

/// The pass that draws the twin of a depth plane into the plane: one
/// triangle over each mip of each slice, its pixel shader writing the
/// twin's texel as the depth, with the depth test "always" and no stencil
/// (the stencil plane is copied through a buffer and not touched here).
///
/// A 16-bit twin holds whole steps of 1/65535, and the depth written is
/// the step times the float nearest 1/65535 (0x37800080). That is how
/// The title's tiled lighting computes the depth of the
/// rectangles it then draws with the test "equal", one class of materials
/// a draw (all six of its vertex shaders multiply the class number by that
/// constant), so the pass and the draw give the host the same float, and
/// the host turns the same float into the same 24 bits. The step divided
/// by 65535 is another float for 512 of the 65,536 steps (none under 257),
/// and lands in another 24-bit step for 127 of them (none under 385).
const PLANE_CORNERS: &str = r#"
float4 corner(uint i : SV_VertexID) : SV_Position {
    return float4(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0, 0.0, 1.0);
}
"#;
const PLANE_STEPS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<uint> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return float(twin.Load(int4(int2(p.xy), 0, 0))) * asfloat(0x37800080);
}
"#;
const PLANE_FLOATS: &str = r#"
#define RS "DescriptorTable(SRV(t0), visibility = SHADER_VISIBILITY_PIXEL)"
Texture2DArray<float> twin : register(t0);
[RootSignature(RS)]
float plane(float4 p : SV_Position) : SV_Depth {
    return twin.Load(int4(int2(p.xy), 0, 0));
}
"#;

/// Views of twins the pass reads, in a heap of ours used as a ring.
/// shortcut: a view must last until its list has run; with one or two
/// passes a frame the ring is far from coming round.
const PLANE_VIEWS: u32 = 256;

/// What the pass draws with: the heap of its views of twins, and per kind
/// of twin (16-bit steps or floats) and depth format the root signature
/// and pipeline, made when first needed (`None`: the host refused).
struct PlanePass {
    views: ID3D12DescriptorHeap,
    cpu: usize,
    gpu: u64,
    step: u32,
    next: AtomicU32,
    made: Mutex<Vec<(bool, i32, Option<PlanePipeline>)>>,
}

/// The pass's root signature and pipeline for one kind of twin and one
/// depth format.
type PlanePipeline = (ID3D12RootSignature, ID3D12PipelineState);

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for PlanePass {}
// SAFETY: as above.
unsafe impl Sync for PlanePass {}

fn plane_pass() -> Option<&'static PlanePass> {
    static PASS: OnceLock<Option<PlanePass>> = OnceLock::new();
    PASS.get_or_init(|| {
        let host = host()?;
        let kind = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
        // SAFETY: plain heap creation and queries.
        unsafe {
            let views: ID3D12DescriptorHeap = host
                .device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: PLANE_VIEWS,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    NodeMask: 0,
                })
                .ok()?;
            Some(PlanePass {
                cpu: views.GetCPUDescriptorHandleForHeapStart().ptr,
                gpu: views.GetGPUDescriptorHandleForHeapStart().ptr,
                step: host.device.GetDescriptorHandleIncrementSize(kind),
                views,
                next: AtomicU32::new(0),
                made: Mutex::default(),
            })
        }
    })
    .as_ref()
}

/// Compiles function `entry` of `source` for `target` with the system's
/// compiler, as the indirect-argument clamp is (`crate::indirect`).
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            windows::core::s!("victory"),
            None,
            None,
            entry,
            target,
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("a shader of ours: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes.
    let bytes = unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    };
    Ok(bytes.to_vec())
}

impl PlanePass {
    /// The root signature and pipeline for a twin of 16-bit steps (`steps`)
    /// or of floats and the depth format `format`.
    fn pipeline(&self, steps: bool, format: DXGI_FORMAT) -> Option<PlanePipeline> {
        let mut made = self.made.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, _, pipeline)) = made.iter().find(|m| (m.0, m.1) == (steps, format.0)) {
            return pipeline.clone();
        }
        let pipeline = match Self::make(steps, format) {
            Ok(pipeline) => Some(pipeline),
            Err(e) => {
                tracing::warn!(
                    error = %e,
                    steps,
                    format = format.0,
                    "no pipeline to draw a depth plane from its twin: such planes stay as they are"
                );
                None
            }
        };
        made.push((steps, format.0, pipeline.clone()));
        pipeline
    }

    fn make(steps: bool, format: DXGI_FORMAT) -> windows::core::Result<PlanePipeline> {
        use windows::core::s;
        let device = &host().ok_or_else(windows::core::Error::empty)?.device;
        let vs = compile(PLANE_CORNERS, s!("corner"), s!("vs_5_1"))?;
        let source = if steps { PLANE_STEPS } else { PLANE_FLOATS };
        let ps = compile(source, s!("plane"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        // SAFETY: the root signature is inside the pixel shader; the
        // description points at live data and never releases its pointer.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, &ps[..])?;
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Every texel is written; stencil stays off.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                    DepthFunc: D3D12_COMPARISON_FUNC_ALWAYS,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                DSVFormat: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            Ok((root, pipeline))
        }
    }
}

/// Records into `list` the pass that draws `plane`, the twin of the depth
/// plane of `depth`, into that plane: `depth` in the depth-write state,
/// `plane` readable by pixel shaders. The pass sets the list's descriptor
/// heap, graphics root signature, pipeline, depth target, viewport,
/// scissor and topology and puts none of them back: it is for a list that
/// records nothing of the title's after it.
fn record_plane_pass(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    plane: &ID3D12Resource,
) {
    let (Some(host), Some(pass)) = (host(), plane_pass()) else {
        return;
    };
    // SAFETY: live resources.
    let (of, twin) = unsafe { (depth.GetDesc(), plane.GetDesc()) };
    // An R16 typeless twin (53) holds steps, read as R16_UINT (57); an R32
    // typeless one floats, read as R32_FLOAT (41). The depth view names
    // the depth-stencil format of the texture's family (44: 45, 19: 20).
    let steps = twin.Format.0 == 53;
    let format = DXGI_FORMAT(match of.Format.0 {
        44 => 45,
        19 => 20,
        typed => typed,
    });
    let Some((root, pipeline)) = pass.pipeline(steps, format) else {
        return;
    };
    for slice in 0..u32::from(of.DepthOrArraySize) {
        for mip in 0..u32::from(of.MipLevels) {
            let slot = pass.next.fetch_add(1, Ordering::Relaxed) % PLANE_VIEWS;
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: DXGI_FORMAT(if steps { 57 } else { 41 }),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2DARRAY,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_SRV {
                        MostDetailedMip: mip,
                        MipLevels: 1,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                        ..Default::default()
                    },
                },
            };
            let target = D3D12_DEPTH_STENCIL_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2DARRAY,
                Anonymous: D3D12_DEPTH_STENCIL_VIEW_DESC_0 {
                    Texture2DArray: D3D12_TEX2D_ARRAY_DSV {
                        MipSlice: mip,
                        FirstArraySlice: slice,
                        ArraySize: 1,
                    },
                },
                ..Default::default()
            };
            let target = host.dsv(depth, Some(&target));
            let (width, height) = ((of.Width >> mip).max(1) as u32, (of.Height >> mip).max(1));
            let cpu = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: pass.cpu + (slot * pass.step) as usize,
            };
            let gpu = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: pass.gpu + u64::from(slot * pass.step),
            };
            // SAFETY: an open list; a slot of our heap; live resources and
            // objects (the pass's own live as long as the process).
            unsafe {
                host.device
                    .CreateShaderResourceView(plane, Some(&view), cpu);
                list.SetDescriptorHeaps(&[Some(pass.views.clone())]);
                list.SetGraphicsRootSignature(&root);
                list.SetPipelineState(&pipeline);
                list.SetGraphicsRootDescriptorTable(0, gpu);
                list.OMSetRenderTargets(0, None, false, Some(&target));
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: width as f32,
                    Height: height as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                list.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: width as i32,
                    bottom: height as i32,
                }]);
                list.IASetPrimitiveTopology(
                    windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY(4),
                );
                list.DrawInstanced(3, 1, 0, 0);
            }
            crate::resource::PLANES[0].fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Records into `list` the copy of a depth texture into its colour twin
/// (`colour`) or of the twin into the depth texture, both already in their
/// copy states: whole textures of one size in one format family, which the
/// host copies as they are; or, for the twin of a stencil plane, that plane
/// alone, through the twin's buffer.
fn record_twin_copy(
    list: &ID3D12GraphicsCommandList,
    depth: &ID3D12Resource,
    twin: &crate::resource::Twin,
    colour: bool,
) {
    let (from, to) = if colour {
        (depth, &twin.colour.texture)
    } else {
        (&twin.colour.texture, depth)
    };
    crate::resource::TWINS[2 + usize::from(colour)].fetch_add(1, Ordering::Relaxed);
    // shortcut: the depth plane's twin is not given the plane (that is a
    // pass the other way: the plane read by a shader, the twin as its
    // target). The twin then lacks what draws and clears wrote to the
    // plane since shaders last wrote the twin. In the title
    // nothing writes that plane but the twin's pass (its 192 draws against
    // it only test, run 29), so the twin lacks nothing. Counted.
    if colour && twin.plane.is_some() {
        crate::resource::PLANES[1].fetch_add(1, Ordering::Relaxed);
    }
    let Some(through) = &twin.through else {
        // SAFETY: an open list and two live textures.
        unsafe { list.CopyResource(to, from) };
        return;
    };
    // A stencil plane and its twin: each mip and slice of the plane goes
    // into the buffer, laid out as the host lays the plane out for copies,
    // and from the buffer into the other texture, whole subresources both
    // times. In the depth texture the stencil plane's subresources follow
    // the depth plane's; the twin has the one plane.
    let Some(host) = host() else { return };
    // SAFETY: a live texture.
    let of = unsafe { depth.GetDesc() };
    let planes = u32::from(of.MipLevels) * u32::from(of.DepthOrArraySize);
    let mut rows = vec![D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); planes as usize];
    // SAFETY: a plain query writing `planes` footprints.
    unsafe {
        host.device.GetCopyableFootprints(
            &of,
            planes,
            planes,
            0,
            Some(rows.as_mut_ptr()),
            None,
            None,
            None,
        );
    }
    let (first_from, first_to) = if colour { (planes, 0) } else { (0, planes) };
    let subresource = |texture: &ID3D12Resource, index: u32| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: a borrowed pointer the location never releases.
        pResource: unsafe { std::mem::transmute_copy(texture) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: index,
        },
    };
    let placed = |row: &D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(through) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: *row,
        },
    };
    let (common, source, dest) = (
        D3D12_RESOURCE_STATE_COMMON,
        D3D12_RESOURCE_STATE_COPY_SOURCE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    // SAFETY: an open list and live resources; the buffer was made as big
    // as these footprints, and rests in the common state between copies.
    unsafe {
        crate::barstat::record(&list, crate::barstat::Why::Twin, &[transition(through, common, dest)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_from + i as u32;
            list.CopyTextureRegion(&placed(row), 0, 0, 0, &subresource(from, index), None);
        }
        crate::barstat::record(&list, crate::barstat::Why::Twin, &[transition(through, dest, source)]);
        for (i, row) in rows.iter().enumerate() {
            let index = first_to + i as u32;
            list.CopyTextureRegion(&subresource(to, index), 0, 0, 0, &placed(row), None);
        }
        crate::barstat::record(&list, crate::barstat::Why::Twin, &[transition(through, source, common)]);
    }
}

/// Whether device-removed data is on (`VICTORY_D3D12_DRED`, diagnostics
/// builds).
pub(crate) fn dred_on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        cfg!(feature = "diagnostics") && std::env::var_os("VICTORY_D3D12_DRED").is_some()
    })
}

/// Indirect mesh dispatches since the last frame line: the largest group
/// count a record asked for, and the groups asked in all (from the lists'
/// budget buffers, a run late).
static MESH_GROUPS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether a list's indirect-work budget lives in the card's own memory
/// (`VICTORY_INDIRECT_CARD_BUDGET=0` keeps it in memory the CPU maps, as
/// before). The clamp charges it with atomics, at least two in a row for
/// every clamp; in mapped memory on a card of its own each of them
/// crosses the bus.
fn card_budget() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_CARD_BUDGET").is_none_or(|v| v != "0"))
}

/// Whether each list of a submission runs and is timed on its own, slow
/// ones logged with their commands (`VICTORY_GPU_TIMING`, diagnostics
/// builds; the commands are noted with `VICTORY_D3D12_DRED` on).
#[cfg(feature = "diagnostics")]
fn timing() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_TIMING").is_some())
}

/// Names a host object for the device-removed report, which names the
/// allocations a page fault hit (only with device-removed data on).
pub(crate) fn name(object: &impl Interface, text: impl FnOnce() -> String) {
    if dred_on()
        && let Ok(object) = object.cast::<ID3D12Object>()
    {
        // SAFETY: a plain call on a live object.
        let _ = unsafe { object.SetName(&windows::core::HSTRING::from(text())) };
    }
}

/// One in [`ALIVE`]`[0]` while it lives (a host list), and one in
/// [`LISTS`]`[3]` until it is first reset.
struct Counted {
    /// Not reset since it was made.
    fresh: std::sync::atomic::AtomicBool,
}

impl Counted {
    fn new() -> Counted {
        ALIVE[0].fetch_add(1, Ordering::Relaxed);
        LISTS[3].fetch_add(1, Ordering::Relaxed);
        Counted {
            fresh: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// The list is reset: it is in use.
    fn used(&self) {
        if self.fresh.swap(false, Ordering::Relaxed) {
            LISTS[3].fetch_sub(1, Ordering::Relaxed);
        }
    }
}

impl Drop for Counted {
    fn drop(&mut self) {
        ALIVE[0].fetch_sub(1, Ordering::Relaxed);
        self.used();
    }
}

/// A chunk of upload memory for lists' scratch data: the buffer, its
/// mapping and its host GPU address.
struct Chunk {
    _buffer: ID3D12Resource,
    data: usize,
    gpu: u64,
    /// In the card's memory (`crate::bus::gpu_upload_asked`), not the
    /// upload heap; then its bytes, for the memory line.
    card: Option<u64>,
}

impl Drop for Chunk {
    fn drop(&mut self) {
        if let Some(bytes) = self.card {
            crate::bus::ALIVE[0].fetch_sub(1, Ordering::Relaxed);
            crate::bus::ALIVE[1].fetch_sub(bytes, Ordering::Relaxed);
        }
    }
}

// SAFETY: a D3D12 resource (free-threaded) and its persistent mapping,
// written only by the list that holds the chunk.
unsafe impl Send for Chunk {}

/// Upload chunks handed out front to back: the chunks (each with its
/// bytes), the one in use and the bytes used in it.
#[derive(Default)]
struct Scratch {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
}

// SAFETY: the chunk pointers are persistent mappings of upload buffers.
unsafe impl Send for Scratch {}

impl Drop for Scratch {
    fn drop(&mut self) {
        give_chunks(std::mem::take(&mut self.chunks));
    }
}

const SCRATCH_CHUNK: u64 = 4 << 20;
/// A list's first chunk when the chunks are shared ([`scratch_pool`]): most
/// lists put a few constants or a rectangle's corners there, and every
/// list that did took 4 MiB of the PC's memory for it.
const SCRATCH_FIRST: u64 = 256 << 10;
/// The most bytes of chunks kept while no list holds them, unless set.
const SPARE_MOST: u64 = 128 << 20;

/// The most bytes of scratch chunks kept while no list (and no send to a
/// copy in the card's memory) holds them: `VICTORY_SCRATCH_SPARE_MB`
/// ([`SPARE_MOST`], 128, unless set). Kept chunks are upload memory, all
/// of it committed; the memory line gives what is kept ("scratch ...
/// spare").
fn spare_most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_SCRATCH_SPARE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map_or(SPARE_MOST, |mb| mb << 20)
    })
}

/// Whether lists share their scratch chunks: a list takes a chunk when it
/// first needs one, a small one first, and gives all of them back when it
/// is reset or dropped, for the next list. Before, a list kept every chunk
/// it ever took (4 MiB each) for as long as it lived, used or not, and the
/// host lists of held submissions, which live for one submission, each
/// made theirs anew: some 57 upload buffers of 4 MiB made, zeroed and
/// freed per frame of gameplay (run 60).
/// `VICTORY_SCRATCH_POOL=0`: chunks of 4 MiB kept by their list, as before.
fn scratch_pool() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SCRATCH_POOL").is_none_or(|v| v != "0"))
}

/// Things of some sizes that nothing holds now, kept for whoever next asks
/// for one of the same size.
struct Spare<T> {
    kept: Vec<(T, u64)>,
    bytes: u64,
}

impl<T> Spare<T> {
    const fn new() -> Self {
        Spare { kept: Vec::new(), bytes: 0 }
    }

    /// A kept thing of `size` bytes, the one given last.
    fn take(&mut self, size: u64) -> Option<T> {
        let at = self.kept.iter().rposition(|(_, s)| *s == size)?;
        self.bytes -= size;
        Some(self.kept.remove(at).0)
    }

    /// Keeps `thing` (`size` bytes) unless that takes the kept bytes past
    /// `most`: then it comes back.
    fn give(&mut self, thing: T, size: u64, most: u64) -> Option<T> {
        if self.bytes + size > most {
            return Some(thing);
        }
        self.bytes += size;
        self.kept.push((thing, size));
        None
    }
}

/// Scratch chunks no list holds.
static SPARE: Mutex<Spare<Chunk>> = Mutex::new(Spare::new());

/// A scratch chunk of `size` bytes: one kept ([`SPARE`]) or a new one.
fn take_chunk(host: &Host, size: u64) -> Option<Chunk> {
    if scratch_pool()
        && let Some(chunk) = SPARE.lock().unwrap_or_else(|e| e.into_inner()).take(size)
    {
        ALIVE[3].fetch_sub(1, Ordering::Relaxed);
        ALIVE[4].fetch_sub(size, Ordering::Relaxed);
        ALIVE[5].fetch_add(1, Ordering::Relaxed);
        return Some(chunk);
    }
    let (buffer, data, card) = host.cpu_written_buffer(size)?;
    name(&buffer, || if card { "list scratch on the card".into() } else { "list scratch".into() });
    // SAFETY: a live buffer.
    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
    ALIVE[1].fetch_add(1, Ordering::Relaxed);
    ALIVE[2].fetch_add(size, Ordering::Relaxed);
    if card {
        crate::bus::ALIVE[0].fetch_add(1, Ordering::Relaxed);
        crate::bus::ALIVE[1].fetch_add(size, Ordering::Relaxed);
    }
    Some(Chunk {
        _buffer: buffer,
        data: data as usize,
        gpu,
        card: card.then_some(size),
    })
}

/// A list is done with `chunks` (it was reset after its submissions ran,
/// or dropped): kept for the next list, or freed past [`spare_most`] and
/// without the pool.
fn give_chunks(chunks: Vec<(Chunk, u64)>) {
    if chunks.is_empty() {
        return;
    }
    let mut freed = Vec::new();
    {
        let mut spare = SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for (chunk, size) in chunks {
            let back = if scratch_pool() {
                spare.give(chunk, size, spare_most())
            } else {
                Some(chunk)
            };
            match back {
                None => {
                    ALIVE[3].fetch_add(1, Ordering::Relaxed);
                    ALIVE[4].fetch_add(size, Ordering::Relaxed);
                }
                Some(chunk) => {
                    ALIVE[1].fetch_sub(1, Ordering::Relaxed);
                    ALIVE[2].fetch_sub(size, Ordering::Relaxed);
                    freed.push(chunk);
                }
            }
        }
    }
    // Freed outside the lock.
    drop(freed);
}

/// What the host keeps for copies in the card's own memory
/// ([`crate::shadow::cards`]): three lists of ours and the compare.
struct CardWork {
    /// Before a submission's lists: what changed on the copies' CPU side
    /// goes to their card side. Those not in use: a plan takes one and
    /// gives it back when its submission has run (`Host::card_done`), so
    /// that submissions sent ahead ([`late`]) each have their own.
    sync: Vec<Prelude>,
    /// After them, in the same submission: the compare of the declared
    /// ranges with their bytes from before. As `sync`.
    check: Vec<Prelude>,
    /// After it ran: the pieces that differ come down.
    fetch: Prelude,
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// One dword per piece compared, set where the piece differs: PC
    /// memory, mapped (buffer, mapping, bytes).
    differs: (ID3D12Resource, usize, u64),
    /// Where the pieces that differ are fetched to: PC memory, mapped.
    fetched: Option<(ID3D12Resource, usize, u64)>,
    /// The dwords of `differs` taken by plans whose results were not read
    /// yet.
    taken: u32,
    /// Plans made and not done with (`Host::card_done`).
    out: u32,
    /// With mirrors (`crate::shadow::no_pc_copy`): the fence, its event and
    /// its last value for the waits of [`stage_most`].
    stage_wait: Option<(ID3D12Fence, HANDLE, u64)>,
    /// The fetch inside flights ([`inflight_fetch`]), if it is on and was
    /// made.
    inflight: Option<CardInflight>,
}

/// What the fetch inside flights needs ([`inflight_fetch`], [`CARD_FETCH`]).
struct CardInflight {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    /// PC memory, mapped (buffer, mapping, bytes): dword 0 the counter of
    /// tickets, then one dword per dword of `CardWork::differs`, the entry
    /// of that piece.
    table: (ID3D12Resource, usize, u64),
    /// PC memory, mapped (buffer, mapping, slots of [`INFLIGHT_SLOT`]
    /// bytes): where the pieces come down. With [`inflight_on_card`] the
    /// card's own memory, no mapping (0): the landing copies slots down.
    fetched: (ID3D12Resource, usize, u32),
    /// Every ticket below it belongs to a landed flight: its slot may be
    /// written over. Raised by the thread that lands flights; read when a
    /// compare's list is recorded (an older value only gives fewer slots).
    landed: u32,
    /// What fetches the old way brought (copy, offsets, and the highest
    /// send whose lists were surely on the queue ahead of the fetch: its
    /// order of the queue less one). A flight sent at or before that ran
    /// ahead of the fetch, so the CPU's side holds those bytes as the card
    /// had them after the flight, or newer: the flight's own slot is older
    /// there and is not merged there. Dropped once every flight sent up to
    /// it has landed.
    brought: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)>,
}

// SAFETY: D3D12 objects (free-threaded) and persistent mappings, used only
// under the submission lock.
unsafe impl Send for CardWork {}

static CARD_WORK: Mutex<Option<CardWork>> = Mutex::new(None);

/// The compare of a declared range of a copy in the card's memory with
/// its bytes from before, one thread per 64 KiB piece (cut at multiples of
/// 64 KiB of title memory, as `Shadow::card_piece` cuts them): `differs`
/// gets a dword per piece, 1 where a byte differs. The offsets are widened
/// to 16 bytes; buffers are multiples of 64 KiB, and a piece flagged for a
/// byte just outside the range costs a fetch that finds nothing.
pub(crate) const CARD_COMPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint compare;  // 0: the hashes are stored; else compared with those stored (2: and then stored)
};
ByteAddressBuffer now : register(t0);
// 16 bytes per piece of the copy: the hash of its bytes when they were
// last sent up (or hashed whole, the first time).
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pieces) return;
    uint piece = first + id.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [loop]
    for (uint at = lo; at < hi; at += 16u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (compare != 0u) {
        differs.Store(out_at + id.x * 4u, any(h != hashes.Load4(piece * 16u)) ? 1u : 0u);
        // With the copy-back on its own thread: the next compare of the
        // piece finds what is written from here on.
        if (compare == 2u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
"#;

/// Whether the pieces a flight's compare flags come down to the PC inside
/// the flight itself (`VICTORY_CARD_FETCH_INFLIGHT=1`; off by default, and
/// only with [`relay`]).
///
/// With the relay the thread that lands flights fetched the flagged pieces
/// with a list of its own on the host's queue, the one the submitting
/// thread keeps filling: the fetch ran behind every flight sent meanwhile.
/// Measured on an RTX 3070 Laptop: 140.9 ms a frame of that thread waiting
/// for fetches, for 63 pieces (3.2 MB) in about five fetches, some 28 ms a
/// fetch spent queued behind newer flights; and the submitting thread
/// waited 81.7 ms a frame in "title memory read" settles for that thread.
///
/// With the switch the compare's list does the fetch: right behind the
/// compare, [`CARD_FETCH`] copies each piece it flagged into a slot of a
/// buffer in the PC's memory ([`CardInflight`]), so when the flight's fence
/// is reached its bytes are there, and the landing merges them
/// (`Shadow::card_merge_relay`) with no list, signal or wait of its own.
/// The merge's "fetched after" is the flight's own send number
/// (`CardPlan::sent_as`): see [`Host::relay_card_back`] for why that is
/// exact and when a landing falls back to the old fetch.
fn inflight_fetch() -> bool {
    inflight_mode() != 0
}

/// [`inflight_fetch`]'s mode: 0 off, 1 the slots in the PC's memory, 2 the
/// slots in the card's memory ([`inflight_on_card`]).
fn inflight_mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| {
        if !relay() {
            return 0;
        }
        match std::env::var("VICTORY_CARD_FETCH_INFLIGHT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        }
    })
}

/// Whether the slots of the fetch inside flights are in the card's own
/// memory (`VICTORY_CARD_FETCH_INFLIGHT=2`; off by default), the landing
/// thread copying the landed flights' slots down to the PC on a copy queue
/// of its own.
///
/// Why `=1` made the card so much busier (run n10: the card about 86% busy
/// at 413 ms a frame, where n9 had 77% at 219): [`CARD_FETCH`]'s stores go
/// straight to the PC's memory (a write-back custom heap), and on the RTX
/// 5070 Ti those stores fall off a cliff past some 20 MB in a pass:
/// `card_fetch_timings` (indirect.rs, run by hand) timed 300 flagged
/// pieces (19 MB) at 1.05 ms, 400 at 178 ms, 1,024 (64 MiB) at 590-610
/// ms, about 0.6 ms a piece past the cliff; into the card's own memory the
/// same 1,024 pieces took 0.74 ms, and a copy list of 64 MiB from the
/// card to the PC 2.4-2.6 ms. So the shader copies within the card and a
/// copy list brings the bytes down.
///
/// The copy down is ordered rightly though it does not wait for later
/// flights: a slot is written only by the [`CARD_FETCH`] of the flight
/// whose ticket it holds, right behind that flight's own lists (after its
/// writes, before any later flight's), and no later ticket gets the slot
/// before that one is landed ([`inflight_gets_slot`]), which is after the
/// copy down has been waited for. What later flights write to the copies
/// themselves never reaches the slot, so nothing is torn. The copy queue
/// waits on the host fence behind the landed flights (reached already)
/// so their writes to the slots are seen.
fn inflight_on_card() -> bool {
    inflight_mode() == 2
}

/// The bytes of memory pieces fetched inside flights go to
/// ([`inflight_fetch`]; `VICTORY_CARD_FETCH_INFLIGHT_MB`; 64, or 256 with
/// the slots in the card's memory): 1,024 slots, or 4,096
/// ([`inflight_slots`]). A flight whose flagged pieces find no free slot
/// lands the old way. A flight's tickets are tested against the landed
/// mark of when its list was recorded, up to a frame of flights before
/// it ran (some 1,100-1,200 flagged pieces a frame, runs n47-n50), so
/// 1,024 slots overflow.
fn inflight_fetch_bytes() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        let default = if inflight_on_card() { 256 } else { 64 };
        std::env::var("VICTORY_CARD_FETCH_INFLIGHT_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default).max(1)
            << 20
    })
}

/// A slot of [`CARD_FETCH`]: a 64 KiB piece as the compare cuts it, widened
/// to 16 bytes at both ends (one more 16-byte step where the copy does not
/// start at a multiple of 16 in its piece).
pub(crate) const INFLIGHT_SLOT: u64 = (64 << 10) + 16;

/// The slots for `bytes` asked of [`inflight_fetch_bytes`]: one per 64 KiB,
/// rounded down to a power of two (at least 1, at most 2^15: the shader's
/// byte offsets into the slots are 32-bit), so that
/// `ticket % slots` runs on without a jump when the 32-bit ticket wraps
/// (2^32 is a multiple of it; with 1,023 slots it would jump by 4 and two
/// tickets in the air could share a slot). The buffer is `slots` times
/// [`INFLIGHT_SLOT`], a little over `bytes`.
pub(crate) fn inflight_slots(bytes: u64) -> u32 {
    let wanted = (bytes >> 16).clamp(1, 1 << 15);
    1u32 << (63 - wanted.leading_zeros())
}

/// [`CARD_FETCH`]'s table entry for a piece the compare did not flag.
pub(crate) const INFLIGHT_NONE: u32 = 0xffff_ffff;

/// Where piece `i` of `span` starts in its [`CARD_FETCH`] slot, as an
/// offset of the copy: the compare's lower bound of the piece, widened down
/// to 16 bytes (`CARD_COMPARE`, before clamping to the declared range).
/// What `Shadow::card_piece` gives for the piece lies at or above it, and
/// within [`INFLIGHT_SLOT`] of it.
pub(crate) fn inflight_piece_from(span: &crate::shadow::CardSpan, i: u32) -> usize {
    let lo = (span.first + i) as usize * 65536;
    let lo = lo.saturating_sub(span.skew as usize);
    lo & !15
}

/// Whether the ticket `k` of [`CARD_FETCH`]'s counter gets a slot, with
/// every ticket below `floor` landed and `slots` slots: the slot is
/// `k % slots`. The tickets of flights still in the air are at or above
/// `floor` (they are taken in the queue's order and flights land in that
/// order), so two that get a slot are fewer than `slots` apart and never
/// share one. The arithmetic of the shader, for the tests.
pub(crate) fn inflight_gets_slot(k: u32, floor: u32, slots: u32) -> bool {
    k.wrapping_sub(floor) < slots
}

/// A piece [`Host::relay_card_inflight`] merges: its copy, offsets, its
/// place in the slots' buffer, and its "fetched after".
type InflightPiece = (std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64);

/// With [`inflight_on_card`]: the copies that bring pieces (their place in
/// the slots' buffer, length) down one after the other: (from, to,
/// length), a piece that starts where the one before ended in the slots
/// joining its copy.
pub(crate) fn slots_down(pieces: impl IntoIterator<Item = (u64, u64)>) -> Vec<(u64, u64, u64)> {
    let mut copies: Vec<(u64, u64, u64)> = Vec::new();
    let mut to = 0u64;
    for (from, len) in pieces {
        if len == 0 {
            continue;
        }
        match copies.last_mut() {
            Some(last) if last.0 + last.2 == from && last.1 + last.2 == to => last.2 += len,
            _ => copies.push((from, to, len)),
        }
        to += len;
    }
    copies
}

/// Records [`slots_down`]'s `copies` from the slots' buffer `ring` into
/// `down`, on an open list (the landing thread's copy queue's, or a
/// test's).
pub(crate) fn record_slots_down(list: &ID3D12GraphicsCommandList, down: &ID3D12Resource, ring: &ID3D12Resource, copies: &[(u64, u64, u64)]) {
    for &(from, to, len) in copies {
        // SAFETY: an open list; live buffers; the caller sized `down`.
        unsafe { list.CopyBufferRegion(down, to, ring, from, len) };
    }
}

/// The fetch inside a flight ([`inflight_fetch`]), run in the compare's list
/// right behind the compare of a declared range (a UAV barrier between):
/// one group per piece (looping where a range has more than 65,535); where
/// `differs` flags the piece, its first thread takes a ticket of the
/// counter (`table`'s dword 0) and, if [`inflight_gets_slot`], the group
/// copies the piece, cut and widened as the compare hashes it, into slot
/// `ticket % slots` of `fetched`. Every piece's entry in `table` (dword 1 +
/// its dword in `differs`) is written: the ticket, also one that got no
/// slot (the landing repeats the test with the plan's `landed`), or
/// [`INFLIGHT_NONE`].
pub(crate) const CARD_FETCH: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), UAV(u0), UAV(u1), UAV(u2)"
#define SLOT 65552u
#define NONE 0xffffffffu
cbuffer Span : register(b0) {
    uint first;    // the first piece
    uint skew;     // where the copy starts in its piece 0
    uint size;     // the copy's buffer, in bytes (a multiple of 64 KiB)
    uint pieces;
    uint out_at;   // the first piece's dword in `differs`, in bytes
    uint landed;   // every ticket below it is landed
    uint slots;    // slots of `fetched`
    uint groups;   // groups dispatched
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer differs : register(u0);
RWByteAddressBuffer table : register(u1);
RWByteAddressBuffer fetched : register(u2);
groupshared uint ticket;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group : SV_GroupID, uint t : SV_GroupIndex) {
    [loop]
    for (uint i = group.x; i < pieces; i += groups) {
        if (t == 0u) {
            uint entry = NONE;
            if (differs.Load(out_at + i * 4u) != 0u) {
                uint k;
                table.InterlockedAdd(0, 1u, k);
                entry = k;
            }
            table.Store(4u + out_at + i * 4u, entry);
            ticket = entry;
        }
        GroupMemoryBarrierWithGroupSync();
        uint k = ticket;
        GroupMemoryBarrierWithGroupSync();
        if (k != NONE && (k - landed) < slots) {
            uint piece = first + i;
            uint lo = piece * 65536u;
            lo = lo > skew ? lo - skew : 0u;
            uint hi = min((piece + 1u) * 65536u - skew, size);
            lo = lo & ~15u;
            hi = (hi + 15u) & ~15u;
            uint into = (k % slots) * SLOT;
            [loop]
            for (uint at = lo + t * 16u; at < hi; at += 1024u) {
                fetched.Store4(into + (at - lo), now.Load4(at));
            }
        }
    }
}
"#;

/// Whether the pieces hashed again after ranges were sent up to a copy in
/// the card's memory are hashed in runs of pieces that touch, each piece
/// once (`VICTORY_CARD_REHASH_RUNS=1`; off by default), and not in a pass
/// per range sent up.
///
/// These passes are ours, they run on the card before every submission's
/// lists, and no timing so far names them: each walks its 64 KiB pieces
/// 16 bytes at a time, 4,096 steps one after the other, so a pass takes
/// the time of one piece however few its pieces are. Run 102 sent up 514
/// ranges a frame: 514 such passes. The same pieces are hashed either way;
/// a piece hashed when it need not be, or not hashed when it should, costs
/// a fetch that finds nothing and loses no shader write.
///
/// `VICTORY_CARD_REHASH_RUNS=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log: one run then measures both on the
/// same scene (the card meter's "send-and-hash lists" in the frame log).
/// Runs 105-110 found no change in the frame, but the card was not the
/// frame's limit then.
fn rehash_runs() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_CARD_REHASH_RUNS").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            let joined = (now_us() / 20_000_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(joined), Ordering::Relaxed) != u8::from(joined) {
                tracing::info!(
                    "card rehash passes from now (VICTORY_CARD_REHASH_RUNS=alt): {}",
                    if joined { "JOINED, one pass a run of touching pieces" } else { "one pass a range sent up, as usual" }
                );
            }
            joined
        }
    }
}

/// Since the last frame-rate line, of our own passes over copies in the
/// card's memory: hash passes after ranges were sent up, the pieces they
/// hashed, and compare passes after the lists.
static CARD_PASSES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// Whether what shaders wrote to copies in the card's memory is merged
/// into title memory on several threads (`VICTORY_WIDE_MERGE=1`; off by
/// default). The pieces the card's compare found differing come down in
/// one fetch and were then merged one after the other on the thread that
/// settles: about 630 pieces and 40 MB a frame (run 104), each compared
/// with the copy's CPU side byte by byte and put into title memory through
/// the kernel. A piece is its own bytes of its copy and of title memory,
/// so eight threads take an eighth of them each.
fn wide_merge() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_MERGE").is_some_and(|v| v == "1"))
}

/// Whether the flights settled together fetch what their shaders wrote to
/// copies in the card's memory in one go (`VICTORY_JOIN_FETCH=1`; off by
/// default).
///
/// A settle waits for the last flight, then each flight with such copies
/// fetched its differing pieces with a list of its own and one more wait
/// for the card (opening cutscene: 24-27 such waits a frame, 12-13 ms, in
/// 4-5 flights a settle). Every flight has run before the first fetch, and
/// nothing else runs on the card until the settle ends, so each of those
/// fetches read the card's last bytes: one fetch of all the pieces (each
/// once, as before) reads the same, with one wait. The merges follow in the
/// flights' order, as before.
fn join_fetch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_FETCH").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line, of the copying back, in microseconds:
/// merging what was fetched from copies in the card's memory (the waits for
/// the fetches not counted: `crate::shadow::CARD_STATS`), and comparing
/// and putting back the ranges of copies in the PC's memory (`hand_back`).
static BACK_SPLIT: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// One declared range of a copy in the card's memory, in a submission.
struct CardRange {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    /// As declared (title addresses), for the statistics.
    declared: (usize, usize),
    span: crate::shadow::CardSpan,
    /// Its first piece's dword in the compare's results.
    at: u32,
    /// Whether the card compares it; else every piece is fetched.
    compared: bool,
}

/// What a submission does for copies in the card's memory.
struct CardPlan {
    ranges: Vec<CardRange>,
    /// The compare, run after the submission's lists.
    check: Option<ID3D12CommandList>,
    /// The copies whose card side was brought up to date, alive until the
    /// submission has run.
    _synced: Vec<std::sync::Arc<crate::shadow::Shadow>>,
    /// The lists of ours the plan recorded (whether the compare, the
    /// list), given back by `Host::card_done`.
    lists: Vec<(bool, Prelude)>,
    /// With [`relay`], the number of this plan's send of changed ranges
    /// (`QUEUE_ORDER`); else 0.
    sent_as: u64,
    /// Whether the compare's list fetches the flagged pieces itself
    /// ([`inflight_fetch`]): every compared range has its [`CARD_FETCH`].
    inflight: bool,
    /// The `CardInflight::landed` its [`CARD_FETCH`] was told: which of its
    /// tickets got a slot ([`inflight_gets_slot`]).
    inflight_landed: u32,
    /// With mirrors (`crate::shadow::no_pc_copy`): the upload chunks its
    /// sends read from, given back by `Host::card_done`.
    stages: Vec<(Chunk, u64)>,
    /// With [`card_sync_join`]: the sends' list, closed and not sent yet;
    /// it goes to the queue first in the submission's own call.
    sync: Option<ID3D12CommandList>,
}

/// Whether the list that sends changed ranges up to copies in the card's
/// memory goes to the queue in the submission's own `ExecuteCommandLists`,
/// first, instead of in a call of its own just before it
/// (`VICTORY_CARD_SYNC_JOIN=1`; off by default; only with mirrors,
/// `VICTORY_CARD_NO_PC_COPY=1`, and without the relay).
///
/// Every submission with something to send made two calls to the queue,
/// back to back. One call runs the same lists in the same order; what the
/// call boundary did for the buffers (they fall back to the common state
/// when a call's work is done, its writes seen by what follows) is done by
/// barriers at the end of the sends' list: every buffer it wrote or read
/// goes back to the common state there, so the title's lists that follow
/// find them as they did. A list cut at `VICTORY_CARD_STAGE_MB` still
/// sends its first parts, and waits for them, on their own.
fn card_sync_join() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_CARD_SYNC_JOIN").is_some_and(|v| v == "1") && crate::shadow::no_pc_copy() && !relay()
    })
}

/// The most bytes a submission's sends to copies in the card's memory
/// with mirrors (`crate::shadow::no_pc_copy`) take in upload chunks before
/// the list so far goes to the card, is waited for, and its chunks are
/// used again: `VICTORY_CARD_STAGE_MB`, 64 unless set (at least one
/// chunk). Without a bound the first sends of the world's big copies
/// would take hundreds of MB of upload memory at once, at the moment the
/// title's own memory peaks; past it the list waits (at world entry and
/// big streaming, seldom in play).
fn stage_most() -> u64 {
    static MOST: OnceLock<u64> = OnceLock::new();
    *MOST.get_or_init(|| {
        (std::env::var("VICTORY_CARD_STAGE_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(64) << 20)
            .max(SCRATCH_CHUNK)
    })
}

/// For the memory line: upload chunks plans of copies in the card's memory
/// hold now and their bytes; the waits [`stage_most`] made since the start;
/// bytes moved into chunks since the start.
pub(crate) static CARD_STAGE: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// The most bytes a buffer that fetches what shaders wrote to copies in
/// the card's memory keeps once its fetch is merged:
/// `VICTORY_CARD_FETCH_KEEP_MB` (unset: none is let go, as before).
///
/// The fetch buffer (and the relay's own) is PC memory shared with the
/// card, grown to the biggest fetch so far in steps of 16 MiB and kept for
/// the rest of the run: one frame that fetched 300 MB holds 304 MB from
/// then on. With the switch a buffer bigger than this goes once its
/// fetch is merged, and the next fetch makes one of its own size. The
/// memory line gives the buffers' sizes either way.
fn fetch_keep() -> Option<u64> {
    static KEEP: OnceLock<Option<u64>> = OnceLock::new();
    *KEEP.get_or_init(|| {
        std::env::var("VICTORY_CARD_FETCH_KEEP_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).map(|mb| mb << 20)
    })
}

/// For the memory line: the bytes of the fetch buffer and of the relay's;
/// fetch buffers let go since the start ([`fetch_keep`]).
pub(crate) static CARD_FETCH_BYTES: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// One submission's sends to copies in the card's memory with mirrors:
/// upload chunks (from the lists' scratch pool, [`take_chunk`]) handed out
/// front to back, and the moves into them still to make.
#[derive(Default)]
struct Stager {
    chunks: Vec<(Chunk, u64)>,
    current: usize,
    used: u64,
    /// Bytes taken since the list was last sent and waited for.
    segment: u64,
    /// Times the list was sent and waited for ([`stage_most`]).
    segments: u32,
    /// (where in a chunk's mapping, source address, bytes, whether the
    /// source is title memory): made before the list runs.
    fills: Vec<(usize, usize, usize, bool)>,
}

impl Stager {
    /// Room for up to `want` bytes (`want > 0`): the chunk, the offset in
    /// it and the bytes given. `None` if no chunk could be had.
    fn take(&mut self, host: &Host, want: u64) -> Option<(usize, u64, u64)> {
        loop {
            if let Some(&(_, size)) = self.chunks.get(self.current) {
                if self.used < size {
                    let n = want.min(size - self.used);
                    let at = self.used;
                    // Each send starts on 16 bytes in its chunk.
                    self.used = (self.used + n).next_multiple_of(16).min(size);
                    self.segment += n;
                    return Some((self.current, at, n));
                }
                self.current += 1;
                self.used = 0;
                continue;
            }
            let taking = std::time::Instant::now();
            let chunk = take_chunk(host, SCRATCH_CHUNK);
            more_add(more::SEND_CHUNKS, taking.elapsed().as_nanos() as u64);
            more_add(more::SEND_CHUNKS_N, 1);
            let chunk = chunk?;
            CARD_STAGE[0].fetch_add(1, Ordering::Relaxed);
            CARD_STAGE[1].fetch_add(SCRATCH_CHUNK, Ordering::Relaxed);
            self.chunks.push((chunk, SCRATCH_CHUNK));
        }
    }

    /// The moves into the chunks, made now: before their list runs.
    fn fill(&mut self) {
        let fills = std::mem::take(&mut self.fills);
        let bytes: usize = fills.iter().map(|f| f.2).sum();
        CARD_STAGE[3].fetch_add(bytes as u64, Ordering::Relaxed);
        let fill = |(dst, src, n, title): (usize, usize, usize, bool)| {
            if title {
                copy_mapped(dst as *mut u8, src as *const u8, n, false);
            } else {
                // SAFETY: a mirror's held bytes (`Shadow::card_sources`)
                // into a chunk's mapping with room for them.
                unsafe { std::ptr::copy_nonoverlapping(src as *const u8, dst as *mut u8, n) };
            }
        };
        if crate::shadow::wide_copy() && bytes >= 8 << 20 {
            // Several at a time, in jobs of about a megabyte: each writes
            // its own bytes of a chunk.
            let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::new();
            let mut group = Vec::new();
            let mut grouped = 0;
            for f in fills {
                grouped += f.2;
                group.push(f);
                if grouped >= 1 << 20 {
                    let g = std::mem::take(&mut group);
                    jobs.push(Box::new(move || g.into_iter().for_each(fill)));
                    grouped = 0;
                }
            }
            if !group.is_empty() {
                jobs.push(Box::new(move || group.into_iter().for_each(fill)));
            }
            crate::shadow::wide_run(jobs);
        } else {
            fills.into_iter().for_each(fill);
        }
    }

    /// The list ran and was waited for: its chunks are free again.
    fn rewind(&mut self) {
        self.current = 0;
        self.used = 0;
        self.segment = 0;
        self.segments += 1;
    }

    /// The chunks, for the plan to hold until its submission has run.
    fn keep(self) -> Vec<(Chunk, u64)> {
        self.chunks
    }
}

/// One piece of submission-time work, in recorded order.
enum Step {
    /// A copy into title memory (destination, source, bytes): the title's
    /// buffer copies, which later refreshes must see.
    Copy(usize, usize, usize),
    /// A copy of title memory into the list's own upload memory
    /// (destination, source, bytes, what it is for: a `FILL_` kind).
    Fill(usize, usize, usize, u8),
    /// As `Fill`, for a table constant view bound at a slot of the list's
    /// card memory ([`table_gpu_copy`]): bytes under flights' writes are
    /// put over it on the card ([`HostList::card_classify`]), so it is
    /// not a read that waits for them (destination, source, bytes).
    CardFill(usize, usize, usize),
    /// Brings a range (address, bytes) of a buffer's host copy up to date:
    /// once per submission, whichever lists ask, until a `Copy` or `Run`
    /// intervenes.
    Refresh(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Work that may write title memory.
    Run(Box<dyn Fn() + Send>),
    /// Work that writes host memory only; whether the texture checks
    /// gathered before it are done first (`VICTORY_WIDE_TEXTURES`): false
    /// for work that touches no staging buffer, or only a staging buffer's
    /// flags, which order with its checks as [`HostList::defer_host_flags`]
    /// says.
    Host(Box<dyn Fn() + Send>, bool),
    /// Shaders write a range (address, bytes) of a buffer's host copy.
    GpuWrite(std::sync::Arc<crate::shadow::Shadow>, usize, usize),
    /// Brings a texture's staging buffer up to date: once per submission,
    /// whichever lists sample it, until a `Copy` or `Run` intervenes.
    Texture(std::sync::Arc<crate::texture::Staging>),
}

/// What a list does on the CPU when it is submitted: copies between host
/// memory regions (title buffers, upload buffers and buffer copies), and
/// the upload buffers its GPU commands read, kept until the list is reset.
#[derive(Default)]
struct Deferred {
    /// Destination, source, bytes.
    copies: Vec<Step>,
    uploads: Vec<ID3D12Resource>,
    /// The bytes of `uploads` (counted in [`ALIVE`] while they are held).
    upload_bytes: u64,
    /// Host objects the list's commands name (pipelines, root signatures,
    /// descriptor heaps): the title may release its objects before it
    /// submits the list.
    kept: Vec<windows::core::IUnknown>,
    /// Buffer ranges refreshed since the last title copy (host copy,
    /// address, bytes): refreshing them again would find the same bytes.
    refreshed: std::collections::HashSet<(usize, usize, usize)>,
    /// Textures whose contents the list brings up to date.
    textures: std::collections::HashSet<usize>,
    /// Bottom-of-pipe writes, done once the list has run: address, what,
    /// bytes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// Textures read back (texture to buffer copies), done once the list
    /// has run: the host buffer the list copies the texels into, the title
    /// memory they go to, the bytes of a row, the rows, the title's row
    /// pitch.
    backs: Vec<(ID3D12Resource, usize, [usize; 3])>,
    /// Whether the indirect-work budget is zeroed for this recording (a
    /// step on the CPU, or in the list for the card's copy).
    budget_zeroed: bool,
    /// With [`table_on_card`]: the title memory (address, bytes) of the
    /// constant views of descriptor tables bound in host copies instead of
    /// filled, for the frame line's count of reads left to the card.
    on_card: Vec<(usize, usize)>,
    /// With [`table_gpu_copy`]: the copies on the card run just before the
    /// list ([`HostList::card_pre`]) from the views' CPU fills (scratch)
    /// into their slots, joined where both go on.
    card_copies: Vec<CardCopy>,
    /// With [`table_gpu_copy`]: the views bound at slots, each once (later
    /// views of the same bytes share it), for the parts copied from the
    /// buffers' copies at submission ([`HostList::card_classify`]).
    card_views: Vec<CardView>,
    /// With [`table_gpu_copy`]: the title memory (address, bytes) read on
    /// the card instead of on the CPU, for the frame line.
    gpu_reads: Vec<(usize, usize)>,
    /// With [`table_gpu_copy`] at `2`: the title copies recorded on the card
    /// (destination, source, bytes), in order.
    gpu_copied: Vec<(usize, usize, usize)>,
    /// With [`table_gpu_copy`]: the title memory the list's shaders write
    /// so far (as declared).
    declared: Joined,
    /// With [`table_gpu_copy`]: the card slot each constant view (address,
    /// bytes) took since the list's last step that writes title memory on
    /// the CPU (title copy, step of ours), for later views of the same
    /// bytes: their fill would read what the first one's read.
    card_slots: std::collections::HashMap<(u64, u32), u64>,
    /// The indirect scratch buffer (`HostList::indirect_args`, by its
    /// interface pointer) that a clamp in this recording left readable as
    /// indirect arguments; 0 if none. It stays readable until the next
    /// clamp into it ([`HostList::indirect_state`]).
    indirect_readable: usize,
    /// With `VICTORY_ASYNC_RACE=1` ([`race_note`]): the title memory its
    /// steps read on the CPU as it was when the title submitted the list
    /// (address, bytes, kind, fingerprint), and the key of its entries in
    /// [`RACE_PENDING`].
    race: Vec<(usize, usize, u8, u64)>,
    race_key: u64,
}

impl Deferred {
    /// Holds `buffer` (`size` bytes of upload memory) until the list is
    /// reset.
    fn hold_upload(&mut self, buffer: ID3D12Resource, size: u64) {
        ALIVE[6].fetch_add(1, Ordering::Relaxed);
        ALIVE[7].fetch_add(size, Ordering::Relaxed);
        self.upload_bytes += size;
        self.uploads.push(buffer);
    }
}

impl Drop for Deferred {
    fn drop(&mut self) {
        ALIVE[6].fetch_sub(self.uploads.len() as u64, Ordering::Relaxed);
        ALIVE[7].fetch_sub(self.upload_bytes, Ordering::Relaxed);
    }
}

/// Whether the title memory a part of a queue's line reads on the CPU is
/// fingerprinted when the title submits it and compared when our thread
/// prepares it (`VICTORY_ASYNC_RACE=1`, with `VICTORY_ASYNC_SUBMIT=1`;
/// off by default: it reads that memory twice, some milliseconds a frame).
///
/// What it tells: whether the title rewrites memory a submission reads
/// after its submit call (fills of constant views, title copies' sources,
/// small refreshes), which async submit lets the submission see where the
/// title's own thread, preparing at once, did not. Reads that our own work
/// may change between the two moments are not compared: under a title
/// copy or a declared shader write of a part before it in the lines, or of
/// a flight in the air (its landing merges them), or under a read-back.
pub(crate) fn async_race() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASYNC_RACE").is_some_and(|v| v == "1") && crate::queue::behind())
}

/// [`async_race`]: what the parts in the lines write themselves (their
/// title copies' destinations, their declared shader writes), (address,
/// end, the part's key), until they are prepared.
static RACE_PENDING: Mutex<Vec<(usize, usize, u64)>> = Mutex::new(Vec::new());

/// [`async_race`]: the title submitted `lists` (a part of a queue's line).
pub(crate) fn race_note(lists: &[crate::list::Detached]) {
    static KEYS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let key = KEYS.fetch_add(1, Ordering::Relaxed);
    let pending: Vec<(usize, usize)> = RACE_PENDING.lock().unwrap_or_else(|e| e.into_inner()).iter().map(|&(a, b, _)| (a, b)).collect();
    let (written, read_back) = {
        let line = line();
        (line.written.clone(), line.read_back.clone())
    };
    let mut mine: Vec<(usize, usize, u64)> = Vec::new();
    let mut budget: usize = 16 << 20;
    let mut scratch: Vec<u8> = Vec::new();
    for list in lists {
        let mut deferred = list.host().deferred();
        let mut prints = Vec::new();
        for step in &deferred.copies {
            let (src, n, kind) = match step {
                Step::Fill(_, src, n, _) => (*src, *n, 0u8),
                Step::CardFill(_, src, n) => (*src, *n, 1),
                Step::Copy(dst, src, n) => {
                    mine.push((*dst, *dst + *n, key));
                    (*src, *n, 2)
                }
                Step::Refresh(_, at, n) if *n <= 64 << 10 => (*at, *n, 3),
                Step::GpuWrite(_, at, n) => {
                    mine.push((*at, *at + *n, key));
                    continue;
                }
                _ => continue,
            };
            let end = src + n;
            if n == 0
                || n > budget
                || pending.iter().any(|&(a, b)| a < end && src < b)
                || mine.iter().any(|&(a, b, _)| a < end && src < b)
                || touches(&written, src, end)
                || touches(&read_back, src, end)
            {
                continue;
            }
            budget -= n;
            scratch.resize(n, 0);
            if read_into(src as *const u8, &mut scratch) {
                prints.push((src, n, kind, fingerprint(&scratch)));
            }
        }
        deferred.race = prints;
        deferred.race_key = key;
    }
    RACE_PENDING.lock().unwrap_or_else(|e| e.into_inner()).extend(mine);
}

/// [`async_race`]: our thread is about to prepare `lists`, which the title
/// put in its line at `at`: their reads compared with [`race_note`]'s.
pub(crate) fn race_check(lists: &[crate::list::Detached], at: std::time::Instant) {
    let age_ms = at.elapsed().as_secs_f64() * 1e3;
    let mut scratch: Vec<u8> = Vec::new();
    let mut keys = Vec::new();
    for list in lists {
        let (prints, key) = {
            let mut deferred = list.host().deferred();
            (std::mem::take(&mut deferred.race), deferred.race_key)
        };
        keys.push(key);
        for (src, n, kind, print) in prints {
            scratch.resize(n, 0);
            let changed = !read_into(src as *const u8, &mut scratch) || fingerprint(&scratch) != print;
            crate::results::race(kind, n, changed, || format!("{src:#x}+{n:#x}, put in the line {age_ms:.1} ms before"));
        }
    }
    RACE_PENDING.lock().unwrap_or_else(|e| e.into_inner()).retain(|&(_, _, k)| !keys.contains(&k));
}

/// A fingerprint of `bytes` for [`race_note`] (not a cryptographic one).
fn fingerprint(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ bytes.len() as u64;
    let mut words = bytes.chunks_exact(8);
    for w in &mut words {
        let v = u64::from_le_bytes(w.try_into().unwrap_or([0; 8]));
        h = (h ^ v).wrapping_mul(0x0000_0100_0000_01b3).rotate_left(29);
    }
    for &b in words.remainder() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Whether a queue's submissions go to the card without a wait for each
/// (`VICTORY_LATE_WAIT=1`; off by default, and off in the timing run, which
/// waits for every list on its own).
///
/// Every submission was prepared, sent, waited for and copied back before
/// the next one was looked at, so the card stood idle while we prepared
/// and we stood idle while it ran: in run 86 a frame of 446 ms is 134 ms of
/// preparing, 194 ms of waiting and 78 ms of copying back, one after the
/// other, on the one thread that submits.
///
/// With the switch a queue's submission is sent and joins a line of
/// "flights" ([`Flight`]); the thread goes on to the next one while the
/// card runs. The flights are settled together ([`Host::settle`]: one wait
/// for the card, what their shaders wrote copied back once per byte, then
/// their bottom-of-pipe writes and the fence signals behind them, in
/// order) at the next point where the title, or our own preparing, can
/// observe what they do. Those points, exactly:
///
/// 1. a synchronous submission of ours ([`Host::execute`]): a present, a
///    read-back or map of a texture, a clear or upload run at once;
/// 2. the next submission reads title memory that a flight's shaders may
///    write: a title buffer copy, a fill of upload memory, a texture's
///    memory, any step that runs code of ours on title memory
///    ([`late_blocked`]; with [`fill_from_copy`] fills whose bytes there
///    lie in copies in the PC's memory wait only for the flights that
///    declared them and read what they wrote from the copies);
/// 3. the next submission refreshes a range of a host copy that a flight's
///    shaders may write, and the title wrote a page of it since it was last
///    copied ([`stale_under_flights`]): the refresh leaves such a range
///    alone until the flight's writes are back, as it always did within a
///    submission;
/// 4. a queue stands at a wait for a memory value that no flight's
///    bottom-of-pipe write is known to satisfy, at an address a flight may
///    write ([`late_met`]);
/// 5. the title blocks on a fence without an event;
/// 6. the card has run every flight (a thread of ours waits for that:
///    [`late_helper`]), or the flights hold [`late_most`] host lists.
///
/// What the title is told (bottom-of-pipe writes, fence values) is told
/// when the card has really run the work, later than before and never
/// earlier; a fence signalled on a queue while flights are in the air is
/// set behind them. A wait for a memory value that a flight's
/// bottom-of-pipe write satisfies counts as met: the waiting part goes to
/// the card behind that flight, which is the order the wait asks for.
///
/// What it assumes beyond today: memory the title rewrites between two of
/// its submissions, without waiting for the first, may be seen by the
/// first's shaders (as on the console, where the card reads title memory
/// when it runs, not when the list is submitted).
pub(crate) fn late() -> bool {
    late_stage() > 0
}

/// `VICTORY_LATE_WAIT`: 0 (off), 1, or 2 ([`late_exact`]).
fn late_stage() -> u8 {
    static STAGE: OnceLock<u8> = OnceLock::new();
    *STAGE.get_or_init(|| {
        if std::env::var_os("VICTORY_GPU_TIMING").is_some() {
            return 0;
        }
        match std::env::var("VICTORY_LATE_WAIT").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            // The relay alone: on top of stage 3.
            Err(_) if crate::shadow::relay() => 3,
            _ => 0,
        }
    })
}

/// Stage 2 of [`late`] (`VICTORY_LATE_WAIT=2`): point 3 of its list is no
/// longer a point. Where the title wrote a page that lies in a range a
/// flight's shaders may write, the bytes the title changed are found (the
/// page as it is now against what the copy held when the flight was
/// prepared) and put into the copy, and only those: what the flight's
/// shaders write beside them stays (`Shadow::merge_under_writes`). The
/// flights are settled for it only where that cannot be done: memory that
/// is not write-watched, or a copy in the card's memory with more changed
/// runs than a submission should send up one by one.
///
/// In run 92 (stage 1) this point settled the flights five times a frame
/// of twelve, each time with the card's work still to wait for: 172 ms a
/// frame of waiting were left of 200.
///
/// What differs from settling first: a byte both the title and a flight's
/// shader wrote is the title's if the shader wrote it first, the shader's
/// if it wrote it afterwards; settled first it was always the shader's.
/// The title writing a byte a shader of its own is writing is a race on
/// the console too.
fn late_exact() -> bool {
    late_stage() > 1
}

/// Stage 3 of [`late`] (`VICTORY_LATE_WAIT=3`: stage 2, and the present is
/// no longer a point where flights are settled). The copy to the back
/// buffer is sent as a flight of its own (`Host::execute_own`, a list of
/// its own for each present in the air) and the swap chain is told to
/// present behind it; the queue's call returns, and the card finishes the
/// frame while the title records the next one. What the title is told
/// about the frame (bottom-of-pipe writes, fences) is told when the card
/// has run it, by the settling thread.
///
/// What is left out with it: the look whether the presented frame is black
/// (diagnostics; it reads the frame back at once). A frame that is saved
/// or listed is presented the old way. Once every second and a half a
/// submission settles the flights all the same, for the sweep that lets
/// idle host objects go (it needs nothing in the air).
pub(crate) fn late_free_present() -> bool {
    late_stage() > 2
}

/// The relay's first stage (`VICTORY_RELAY=1`; off by default; it turns
/// [`late`] on at stage 3 unless `VICTORY_LATE_WAIT` says another stage,
/// and is off with it, in the timing run and with kept snapshots).
///
/// With [`late`] alone every settle does three things on the thread that
/// holds the submission lock: it waits for everything in the air, copies
/// back with the card standing, and only then goes on preparing. The frame
/// stays a relay: the card works about 242 ms of it and one thread
/// prepares, waits and copies back for about 220.
///
/// With the switch a thread of its own ([`relay_worker`]) lands flights as
/// soon as the card has run them, in the order they were sent and without
/// the submission lock: it fetches and merges what their shaders wrote to
/// copies in the card's memory, compares and merges the other copies,
/// frees the flights' ranges, and then does their read-backs, their
/// bottom-of-pipe writes and the fence signals behind them, in the line's
/// order. The submitting thread meanwhile prepares and sends the next
/// submissions. [`Host::settle`] no longer settles anything: it waits until
/// that thread has landed everything in the air, at the same points as
/// before (the list in [`late`]), so whoever can observe title memory, a
/// fence or a present still does so only after the flights before it are
/// merged.
///
/// What makes that safe is in `crate::shadow::relay` (ranges counted per
/// flight; a merge that never stores into memory the card may be writing;
/// sends to card copies numbered) and here: flights land in the order of
/// [`LINE`] by one thread, so a later flight's results never reach title
/// memory before an earlier flight's; the compare on the card stores the
/// hash it made, so each flight's compare flags exactly what changed since
/// the compare before it, however late the pieces are fetched; the fetch
/// goes to the queue under [`QUEUE_ORDER`], which says which sends of
/// title bytes the card had received when it ran.
pub(crate) fn relay() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| late_stage() > 0 && crate::shadow::relay())
}

/// The relay's second stage (`VICTORY_RELAY=2`): a submission that reads
/// title memory a flight's shaders may write (a title buffer copy, a fill
/// of upload memory, a texture) waits only until the flights that declared
/// that memory have landed, not for everything in the air; and at
/// [`late_most`] lists in the air it waits until there are fewer. At `1`
/// both wait for everything.
fn relay_exact() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    // Stage 2 unless `VICTORY_RELAY=1` asks for the first one (the default
    // since the walls of the first room were found drawn in every run with
    // this set and in fewer than half without it).
    *ON.get_or_init(|| relay() && std::env::var_os("VICTORY_RELAY").is_none_or(|v| v == "2"))
}

/// Whether a submission that waits for flights to land before it reads
/// title memory under their writes ([`Host::relay_read_wait`]) first syncs
/// the descriptor records the title copied into its heaps
/// (`crate::descriptor::sync_heaps`), which it did after the wait
/// (`VICTORY_SYNC_IN_WAIT=1`; off by default).
///
/// In run n15 the wait was 70-78 ms a frame (one wait, every flight in the
/// air needed), the card running the needed flights for 55-62 ms of it;
/// the sync after it took 29 ms a frame with nothing of ours on the card.
/// With the switch it runs while the card still works, and the sync after
/// the wait, as before, takes only the heap pages written since.
///
/// What it changes: host slots are made from the title's records some
/// milliseconds earlier, with flights in the air, as they are already for
/// every submission sent while flights fly (relay stage 2): the title
/// reuses a slot only once the fence of the work that read it is set, and
/// we set fences when flights land. Not done where a flight in the air
/// declared shader writes into a heap's memory (the records would not be
/// the ones the flight leaves); the sync after the wait still runs, so
/// everything the title wrote before the submission is mirrored before
/// its lists go to the card, as before.
fn sync_in_wait() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| relay_exact() && std::env::var_os("VICTORY_SYNC_IN_WAIT").is_some_and(|v| v == "1"))
}

/// How often, in milliseconds, [`sync_in_wait`] syncs the heap pages the
/// title wrote since during a title-read wait.
///
/// Run n24 showed the sync before the wait finding nothing (0 pages, 0.3
/// left for after it): the title's other threads write the records the
/// sync after the wait took 28 ms over while this thread waited (the
/// previous submission's sync had taken everything before). So the pages
/// are synced as they come, during the wait, and the sync after it takes
/// only the last few milliseconds' worth.
const SYNC_IN_WAIT_MS: u64 = 3;

/// [`sync_in_wait`], before and during the wait: the sync of the pages
/// written since the last, but those under flights' declared writes.
fn sync_before_wait() {
    if !sync_in_wait() {
        return;
    }
    // Pages under flights' writes are left to the sync after the wait.
    let written = line().written.clone();
    let syncing = std::time::Instant::now();
    crate::descriptor::sync_heaps_early(&written);
    let us = syncing.elapsed().as_micros() as u64;
    PATH[path::SYNC].fetch_add(us, Ordering::Relaxed);
    PATH[path::SYNC_EARLY].fetch_add(us, Ordering::Relaxed);
}

/// Whether the thread that lands flights ([`relay`]) fetches the pieces of
/// card copies their shaders wrote with a copy queue of its own
/// (`VICTORY_RELAY_COPY_QUEUE=1`; off by default; not with
/// `VICTORY_CARD_FETCH_INFLIGHT`, whose fetches ride in the flights).
///
/// The fetch went to the host's one queue, behind every flight sent since
/// the landed ones: in run n15 the landing thread waited 100-112 ms a frame
/// for 6-7 fetches of some 60 MB, while the submission that read under
/// the flights' writes waited for it (the card had run the flights it
/// needed 12-15 ms before they had landed). On a copy queue a fetch waits
/// only for its own copies.
///
/// No byte a running flight may be writing is merged (run n21 merged
/// such torn bytes, from copies that ran beside later flights, and showed
/// garbage): a piece that no flight in the line declared shader writes for
/// is copied at once; one that flights in the line declared is copied
/// behind the last of them (the copy queue waits on the host fence), so it
/// holds that flight's complete bytes, as the host queue's fetch brought
/// them. After the copy has run, a piece that a flight which may have run
/// beside it declared (one sent since the pieces were looked at; for the
/// second kind one behind the flight waited for) is not merged from it but
/// fetched on the host queue as without the switch. What the card may not
/// have had of the title's bytes when a copy ran (sends after the last
/// flight known to have run before it) is left out of the merge
/// (`Shadow::card_merge_relay`'s mark), and the notes of sends are kept
/// from that mark on (`crate::shadow::relay_fetched`).
///
/// What it assumes, as the relay does: shaders write card copies only in
/// the ranges flights declared.
///
/// Refused at `1` since run n24 (see below): it takes effect only as
/// `VICTORY_RELAY_COPY_QUEUE=declared`, for experiments.
///
/// Why it is not safe, even with no byte of a declared write of a running
/// flight merged (runs n21 and n24 both showed the garbage strip at the
/// bottom of the frame and stair-steps on shoulders, which the same runs
/// without the switch did not): shaders also write card copies where no
/// flight declared it, through buffer views they pick from the heap by
/// index (`crate::draw`'s heap pass: "what such a shader writes through a
/// picked unordered access view stays in the host copy"). The fetch on the
/// host queue runs behind every flight sent before it, so it brings those
/// writes too, complete; a fetch that does not wait for the later flights
/// (the point of the copy queue) brings them part-written or not at all,
/// and nothing flags those pieces again: the merge puts stale or torn
/// bytes into title memory and the CPU's side, which the title reads and
/// later refreshes send back up. Waiting for every flight sent before it,
/// as needed for that, is the host queue's fetch, and saves nothing.
fn relay_copy_queue() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if !relay() {
            return false;
        }
        match std::env::var("VICTORY_RELAY_COPY_QUEUE").as_deref() {
            Ok("declared") => true,
            Ok("1") => {
                tracing::warn!(
                    "VICTORY_RELAY_COPY_QUEUE=1 is refused: a fetch that does not wait for later flights misses their writes \
                     through views picked by index, which no flight declares (the garbage strip of runs n21 and n24); \
                     the landing thread fetches on the host queue (VICTORY_RELAY_COPY_QUEUE=declared still takes it)"
                );
                false
            }
            _ => false,
        }
    })
}

/// Told when something joins [`LINE`], for the thread that lands flights
/// ([`relay`]); [`LINE_MOVED`] stays the word for the thread that looks at
/// what queues hold.
static RELAY_MOVED: std::sync::Condvar = std::sync::Condvar::new();

/// Told when flights have landed ([`relay`]): who waits for that looks
/// again.
static LINE_LANDED: std::sync::Condvar = std::sync::Condvar::new();

/// With [`relay`], the order of what we put on the queue: the number of
/// sends of changed ranges to copies in the card's memory so far
/// (`Host::card_before`). Held from the taking of the ranges until their
/// list is on the queue, and while a fetch of pieces goes to the queue: a
/// fetch sent with the count at N ran behind sends 1..N and ahead of every
/// later one.
static QUEUE_ORDER: Mutex<u64> = Mutex::new(0);

/// Since the last frame-rate line, of the thread that lands flights
/// ([`relay`]): landings, flights landed, microseconds it waited for the
/// card with nothing to land, microseconds it waited for fetches of card
/// copies' pieces, microseconds of landing in all.
static RELAY_STATS: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// Since the last frame-rate line, with [`inflight_fetch`]: pieces merged
/// from the slots their flights fetched them to, their bytes, landings that
/// went the old way because a plan's pieces overflowed the slots, and
/// landings that fetched the old way (for any reason, overflow too).
static INFLIGHT_STATS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// After a fetch the old way taken with the queue's count at `after`, for
/// plans whose highest send is `landed`: up to which send the notes of
/// sends to card copies may go (`crate::shadow::relay_fetched`).
///
/// Without [`inflight_fetch`] every later merge is of a later fetch, whose
/// count is `after` or more: notes up to `after` are of no use, as before.
/// With it, a flight still in the air, sent before the fetch, merges its
/// own slot with "fetched after" its own send, which may be below `after`:
/// it needs the notes of the sends between, or it takes the title's bytes
/// that went up in them for its shaders' writes and puts its older card
/// bytes over them. (Flights 5, 6, 7 in the air; 5 lands the old way with
/// the count at 7; a title byte went up in send 7 into a piece 6's shaders
/// write; 6 lands from its slot, which holds that byte from before send 7.)
/// So the mark goes only to the landed plans' highest send: every flight
/// still to land was sent after it, and every later fetch the old way
/// takes a count at least that high. (`CardInflight::brought` is another
/// matter: the card's bytes the fetch put in the CPU's side, not the
/// title's.)
pub(crate) fn fallback_fetched_mark(inflight: bool, after: u64, landed: Option<u64>) -> u64 {
    match (inflight, landed) {
        (false, _) => after,
        (true, Some(landed)) => landed.min(after),
        (true, None) => 0,
    }
}

/// The landed mark of [`CARD_FETCH`]'s tickets after ticket `k` landed: one
/// past `k` if that is ahead of `landed`, in wrapping order (tickets in the
/// air are always within 2^31 of the mark), else `landed`.
pub(crate) fn inflight_raise(landed: u32, k: u32) -> u32 {
    let next = k.wrapping_add(1);
    if (next.wrapping_sub(landed) as i32) > 0 { next } else { landed }
}

/// With [`inflight_fetch`], after a landing merged (either way): the tickets
/// these plans' fetches took are landed, and their slots may be written
/// over. Plans land in the line's order and tickets are taken in the
/// queue's (a UAV barrier opens every compare's list), so every ticket of a
/// plan in the air is ahead of these.
///
/// The counter is never reset and wraps after 2^32 tickets: the mark is
/// raised in wrapping order ([`inflight_raise`]), and the slots are a power
/// of two, so `k % slots` runs on across the wrap. shortcut: ticket
/// 0xffff_ffff reads as [`INFLIGHT_NONE`]; a flagged piece that draws it
/// copies nothing and its landing goes the old way, once every 2^32.
fn inflight_tickets_landed(plans: &[&CardPlan]) {
    let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
    let Some(work) = work.as_mut() else {
        return;
    };
    let dwords = (work.differs.2 / 4) as usize;
    let Some(inflight) = work.inflight.as_mut() else {
        return;
    };
    // SAFETY: the mapping of the tickets; these plans' fetches have run.
    let table = unsafe { std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords) };
    for plan in plans.iter().filter(|p| p.inflight) {
        for r in plan.ranges.iter().filter(|r| r.compared) {
            for p in 0..r.span.pieces {
                if let Some(&k) = table.get((r.at + p) as usize)
                    && k != INFLIGHT_NONE
                {
                    inflight.landed = inflight_raise(inflight.landed, k);
                }
            }
        }
    }
    // Every flight still to land was sent after these: what fetches the old
    // way brought up to them concerns none of them.
    if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
        inflight.brought.retain(|b| b.3 > last);
    }
}

/// Of the line's things in order (a flight: the value signalled on the
/// host's fence behind it; a write or a signal behind a flight: `None`):
/// how many, from the front, can be landed when the card has reached
/// `completed`: everything ahead of the first flight it has not run. A
/// later flight never lands before an earlier one, and what follows a
/// flight (its bottom-of-pipe writes, signals) never before the flight.
fn landable(line: &[Option<u64>], completed: u64) -> usize {
    line.iter().position(|f| f.is_some_and(|f| f > completed)).unwrap_or(line.len())
}

/// With [`crate::airlists::on`]: the flights at the front of `items` that
/// the card has run (host fence value up to `completed`; it runs them in
/// order) give their lists' draw states back into `out`, each flight once
/// (`Flight::freed`). Flights, host lists.
fn give_back_ran(items: &mut [Late], completed: u64, out: &mut Vec<crate::draw::State>) -> (u64, usize) {
    let (mut flights, mut lists) = (0, 0);
    for item in items.iter_mut() {
        let Late::Run(flight) = item else {
            continue;
        };
        if flight.fence > completed {
            break;
        }
        if flight.freed {
            continue;
        }
        flight.freed = true;
        flights += 1;
        lists += flight.order.len().max(1);
        out.extend(flight.lists.iter_mut().map(crate::list::Detached::release));
    }
    (flights, lists)
}

/// For one plan of copies in the card's memory, with [`relay`]: `flags`
/// holds, for every piece of every range of the plan, the copy, the
/// piece's number in the copy and whether the compare flagged it there (or
/// did not run). The pieces to fetch, under whichever range: all of them
/// flagged under any. Two ranges of a plan may hold parts of the same
/// 64 KiB piece; the compare hashes the whole piece for each and stores
/// the hash, so the second of them may find it stored already and not
/// flag it.
fn flagged_pieces(flags: &[(usize, u32, bool)]) -> std::collections::HashSet<(usize, u32)> {
    flags.iter().filter(|f| f.2).map(|f| (f.0, f.1)).collect()
}

/// How a landing's fetch of pieces is split ([`Host::relay_card_back`]):
/// whether each piece lies in its copy's picked span (`picked`), whether
/// there is a copy queue for fetches (`lane`), and whether picked pieces
/// stay on the host's queue (`split`, `crate::cpuwait::queue2_picked`).
/// The parts in the order they are fetched, each with whether it goes to
/// the copy queue and the pieces' places: the copy queue's first, as it
/// waits for nothing later than the landed flights; no empty part.
fn fetch_parts(picked: &[bool], lane: bool, split: bool) -> Vec<(bool, Vec<usize>)> {
    let on_lane = |p: bool| lane && !(split && p);
    [true, false]
        .into_iter()
        .map(|part| (part, (0..picked.len()).filter(|&i| on_lane(picked[i]) == part).collect::<Vec<usize>>()))
        .filter(|(_, at)| !at.is_empty())
        .collect()
}

/// Whether pieces fetched on a copy queue beside later flights are fetched
/// again when those flights land (`VICTORY_QUEUE2_REFETCH=0`: not, as
/// before; on by default with `VICTORY_QUEUE2` bit 1).
fn refetch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE2_REFETCH").is_none_or(|v| v != "0"))
}

/// With fetches on a copy queue (`crate::queue2`, bit 1): the pieces (copy,
/// piece number) a fetch brought down while later submissions may have been
/// running, each with the number of the last submission started when the
/// fetch had run ([`STARTED`]).
///
/// Such a fetch stands behind the flights being landed and nothing else,
/// so it runs beside whatever was sent after them. A later flight that
/// writes the piece (a buffer cleared and filled again) may be half way
/// through it: the bytes that come down are neither the earlier flight's
/// nor the later one's, and they go into title memory. If the later flight
/// leaves the piece as the earlier one left it (a still scene does, every
/// frame), its own compare finds the stored hash unchanged and flags
/// nothing: title memory kept the half-written bytes until the piece next
/// changed. On the direct queue a fetch only ever sees whole flights.
///
/// So a flight that declared such a piece and had started before the fetch
/// had run fetches it at its own landing whatever its compare says
/// ([`fetched_beside`]); that fetch stands behind the flight, and is noted
/// here in turn if still later flights were in the air. Between the two
/// landings title memory may hold the half-written bytes, as it does on
/// the console, where the title reads the memory the card is writing: what
/// the title is promised is the piece as the flight left it once that
/// flight's fence is signalled, and the landing comes before the signal.
static BESIDE: Mutex<Option<std::collections::HashMap<(usize, u32), u64>>> = Mutex::new(None);

/// Since the last frame-rate line: pieces fetched because an earlier fetch
/// of them ran beside their flight ([`BESIDE`]), and pieces noted there.
static BESIDE_STATS: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`BESIDE`]: whether the flight numbered `number` (among the submissions
/// started) must fetch a piece that an earlier landing's fetch brought
/// down when submissions up to `started` had been started: it had started
/// by then, so it may have been writing the piece under that fetch.
fn fetched_beside(started: Option<u64>, number: u64) -> bool {
    started.is_some_and(|started| started >= number)
}

/// [`BESIDE`], after a landing of flights up to number `landed` whose
/// fetch brought `fetched` down while submissions up to `started` had
/// been started: what is noted of the pieces afterwards, and how many were
/// noted now. A note is done with once every submission it names has
/// landed (each fetched the piece again if it declared it); the pieces
/// just fetched are noted if anything later than the flights landed had
/// started.
fn note_beside(noted: &mut std::collections::HashMap<(usize, u32), u64>, fetched: &[(usize, u32)], landed: u64, started: u64) -> u64 {
    noted.retain(|_, upto| *upto > landed);
    if started <= landed {
        return 0;
    }
    for piece in fetched {
        noted.insert(*piece, started);
    }
    fetched.len() as u64
}

#[cfg(test)]
mod relay_tests {
    use super::{fetch_parts, fetched_beside, flagged_pieces, join_ranges, landable, note_beside, read_back_span, touches};

    /// Without a copy queue every piece goes to the host's queue; with one
    /// all of them go there, unless picked pieces stay on the host's queue
    /// (`VICTORY_QUEUE2_PICKED`), which come second.
    #[test]
    fn picked_pieces_stay_on_the_host_queue_only_when_asked() {
        let picked = [false, true, false, true];
        assert_eq!(fetch_parts(&picked, false, false), vec![(false, vec![0, 1, 2, 3])]);
        assert_eq!(fetch_parts(&picked, false, true), vec![(false, vec![0, 1, 2, 3])]);
        assert_eq!(fetch_parts(&picked, true, false), vec![(true, vec![0, 1, 2, 3])]);
        assert_eq!(fetch_parts(&picked, true, true), vec![(true, vec![0, 2]), (false, vec![1, 3])]);
        assert_eq!(fetch_parts(&[true, true], true, true), vec![(false, vec![0, 1])]);
        assert!(fetch_parts(&[], true, true).is_empty());
    }

    /// A [`super::WHY_MOST`] wait needs the card to have run the flights
    /// whose landing takes the lists in the air under the bound.
    #[test]
    fn a_wait_for_fewer_lists_needs_the_first_flights_only() {
        assert_eq!(super::most_need_lists(&[(5, 100), (6, 100), (7, 100)], 256), 5);
        assert_eq!(super::most_need_lists(&[(5, 50), (6, 100), (7, 200)], 256), 6);
        assert_eq!(super::most_need_lists(&[(5, 10), (6, 10)], 256), 0);
    }

    /// Flights give their draw states back only once the card has run them,
    /// in order, each once, and only theirs are counted
    /// (`VICTORY_RELAY_FREE_RAN`).
    #[test]
    fn only_flights_the_card_ran_give_their_state_back_once() {
        let flight = |fence: u64, lists: usize| {
            super::Late::Run(super::Flight {
                number: fence,
                fence,
                lists: Vec::new(),
                order: (0..lists).collect(),
                card: None,
                back: Vec::new(),
                read_back: Vec::new(),
                freed: false,
                picks: false,
            })
        };
        let mut items = vec![flight(5, 10), super::Late::Signal(0, 1), flight(6, 0), flight(7, 30), flight(8, 4)];
        let mut out = Vec::new();
        // The empty flight (the present's copy) counts as one list.
        assert_eq!(super::give_back_ran(&mut items, 6, &mut out), (2, 11));
        assert_eq!(super::give_back_ran(&mut items, 6, &mut out), (0, 0));
        assert_eq!(super::give_back_ran(&mut items, 8, &mut out), (2, 34));
        let freed: Vec<bool> = items
            .iter()
            .filter_map(|i| match i {
                super::Late::Run(f) => Some(f.freed),
                _ => None,
            })
            .collect();
        assert_eq!(freed, [true; 4]);
        let mut later = vec![flight(9, 1), flight(10, 1)];
        assert_eq!(super::give_back_ran(&mut later, 8, &mut out), (0, 0));
    }

    /// A read-back's span ends with its last row, not a pitch past it, and
    /// a read of any byte of it is under it; one of the bytes between rows
    /// is under it too (a shortcut: the span is counted whole).
    #[test]
    fn a_read_back_covers_its_rows() {
        assert_eq!(read_back_span(0x1000, [80, 1, 256]), Some((0x1000, 0x1050)));
        assert_eq!(read_back_span(0x1000, [16, 3, 64]), Some((0x1000, 0x1000 + 2 * 64 + 16)));
        assert_eq!(read_back_span(0x1000, [0, 3, 64]), None);
        assert_eq!(read_back_span(0x1000, [16, 0, 64]), None);
        let mut spans = Vec::new();
        join_ranges(&mut spans, [read_back_span(0x2000, [16, 1, 0]).unwrap(), read_back_span(0x1000, [80, 1, 0]).unwrap()]);
        assert!(touches(&spans, 0x104f, 0x1100));
        assert!(!touches(&spans, 0x1050, 0x2000));
        assert!(touches(&spans, 0x1ff0, 0x2001));
    }

    /// A piece fetched on the copy queue while a later flight was in the
    /// air is fetched again by that flight, and by no flight sent after
    /// the fetch had run; the note goes when its flights have landed.
    #[test]
    fn a_piece_fetched_beside_a_later_flight_is_fetched_again_by_it() {
        let mut noted = std::collections::HashMap::new();
        let (a, p) = (0x1000usize, 7u32);
        // Flight 5 lands; its fetch ran while 6 and 7 had started.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 5, 7), 1);
        assert!(fetched_beside(noted.get(&(a, p)).copied(), 6));
        assert!(fetched_beside(noted.get(&(a, p)).copied(), 7));
        // Flight 8 was sent after the fetch had run: its compare decides.
        assert!(!fetched_beside(noted.get(&(a, p)).copied(), 8));
        // Another piece was never noted.
        assert!(!fetched_beside(noted.get(&(a, p + 1)).copied(), 6));
        // Flight 6 lands and fetches it again, with 7 still in the air and
        // nothing newer: the note stays for 7.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 6, 7), 1);
        assert!(fetched_beside(noted.get(&(a, p)).copied(), 7));
        // Flight 7 lands with nothing behind it: nothing is noted any more.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 7, 7), 0);
        assert!(noted.is_empty());
        // A fetch with nothing later in the air notes nothing.
        assert_eq!(note_beside(&mut noted, &[(a, p)], 9, 9), 0);
        assert!(noted.is_empty());
    }

    /// Two flights and what follows each: nothing of the second lands
    /// before the card has run it, and nothing behind a flight lands
    /// before the flight.
    #[test]
    fn flights_land_in_the_order_they_were_sent() {
        // Flight (fence 5), its write, a signal, flight (fence 6), its write.
        let line = [Some(5), None, None, Some(6), None];
        assert_eq!(landable(&line, 4), 0);
        assert_eq!(landable(&line, 5), 3);
        assert_eq!(landable(&line, 6), 5);
        assert_eq!(landable(&line, u64::MAX), 5);
        // Signals with no flight ahead land at once; behind one they wait.
        assert_eq!(landable(&[None, None], 0), 2);
        assert_eq!(landable(&[None, Some(9), None], 8), 1);
        assert_eq!(landable(&[], 3), 0);
        // One flight a landing: the first flight with what follows it up
        // to the next one (and what comes ahead of it).
        assert_eq!(super::first_landing(&[Some(3), None, Some(4), None]), 2);
        assert_eq!(super::first_landing(&[None, Some(3), Some(4)]), 2);
        assert_eq!(super::first_landing(&[Some(3), None, None]), 3);
        assert_eq!(super::first_landing(&[None, None]), 2);
        // After the first three landed, the rest when its turn comes.
        assert_eq!(landable(&line[3..], 5), 0);
        assert_eq!(landable(&line[3..], 6), 2);
    }

    /// A piece flagged under one range of a plan is fetched under every
    /// range that holds a part of it; pieces of other copies are not.
    #[test]
    fn a_piece_flagged_under_one_range_is_fetched_under_all() {
        let (a, b) = (0x1000usize, 0x2000usize);
        // Copy a: piece 3 under two ranges, flagged under the first only;
        // piece 4 not flagged; copy b: piece 3 not flagged, piece 9 flagged.
        let flags = [(a, 3, true), (a, 4, false), (a, 3, false), (b, 3, false), (b, 9, true)];
        let fetch = flagged_pieces(&flags);
        assert!(fetch.contains(&(a, 3)));
        assert!(!fetch.contains(&(a, 4)));
        assert!(!fetch.contains(&(b, 3)));
        assert!(fetch.contains(&(b, 9)));
        assert_eq!(fetch.len(), 2);
    }
}

/// The most host lists in the air before the flights are settled
/// (`VICTORY_LATE_WAIT_LISTS`, 256; `=128` is the value before: with the
/// relay it stopped the submitting thread twice a frame for 92-103 ms in
/// all while the card stood idle 64-113 ms a frame, runs 158, 159 and 162;
/// 32 settled the flights three times a
/// frame in run 92 and 64 still 1.3-1.7 times in run 99, with 12,919 tables
/// asked and none refused; watch "tables asked, N refused"): their dynamic
/// descriptor slots, upload
/// memory and scratch chunks are held until then, and a frame's lists held
/// all at once ran the slots out before (run 45).
fn late_most() -> usize {
    static MOST: OnceLock<usize> = OnceLock::new();
    *MOST.get_or_init(|| {
        std::env::var("VICTORY_LATE_WAIT_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(256).max(1)
    })
}

/// `VICTORY_SETTLE_LISTS` (0, the default: off; not with the relay):
///
/// `1`: at [`late_most`] host lists in the air the flights are no longer
/// settled. The bound is there for what the lists hold, above all their
/// dynamic descriptor slots (run 45 ran them out), and a list needs those
/// only until the card has run it. So the card is waited for only until the
/// first flights have run, as many as take the line under the bound (in run
/// n1 that wait was 0.0 ms of the 87 ms the settle waited: the rest was for
/// the flights behind them), and every flight the card has run by then
/// gives back its lists' draw state and slots there and then. The memory
/// handed to the title for the lists (`allocate_upload`) and the host lists
/// stay until the landing. The flights stay in the line otherwise: their
/// copy-back, read-backs, bottom-of-pipe writes and the fence signals behind
/// them wait for the next settle, at the points [`late`] lists, so the
/// title is told nothing earlier and sees nothing before it is merged.
/// The card keeps running the flights behind them while the next
/// submissions are prepared. The flights are settled as before when the
/// line holds [`SETTLE_LISTS_HELD`] times the bound in all, or when the
/// compare's result area of copies in the card's memory runs low.
///
/// `2`: also, a submission that reads title memory flights' shaders may
/// write ([`WHY_READ`]) and is not read from copies ([`fill_from_copy`])
/// waits only until the card has run the flights that declared it, and is
/// not settled for at all if those flights' compares on the card found
/// every piece it reads unchanged ([`reads_unwritten`]): nothing a shader
/// wrote is there, so title memory already holds what a settle would leave
/// there. Only copies in the card's memory say so; anything else (copies in
/// the PC's memory, textures' memory, our steps, a title copy into memory
/// flights may write, a range not compared) settles as before.
fn settle_lists() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        let asked: u8 = std::env::var("VICTORY_SETTLE_LISTS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0).min(2);
        if relay() {
            // Since upstream 9376fd6 (c96b4f2) the relay is on unless
            // `VICTORY_RELAY=0`: run settings that set this got the relay's
            // waits instead, with nothing said.
            if asked > 0 {
                tracing::warn!(
                    asked,
                    "VICTORY_SETTLE_LISTS is not used with the relay (on by default since the merge of upstream \
                     9376fd6; its stage 2 waits only for the flights that declared what a read needs, as =2 did): \
                     VICTORY_RELAY=0 turns the relay off"
                );
            }
            return 0;
        }
        asked
    })
}

/// With [`settle_lists`]: how many times [`late_most`] the line may hold
/// in host lists, those that gave back what they hold included, before it
/// is settled all the same (each still holds its host list and scratch).
const SETTLE_LISTS_HELD: usize = 4;

/// Since the last frame-rate line, of [`settle_lists`]: [0] times the
/// bound freed lists instead of a settle, [1] flights and [2] host lists
/// that gave back their descriptor slots, [3] microseconds waited for
/// the card for it, [4] settles all the same (the held bound or the
/// compare's area); [5] reads found unwritten instead of a settle, [6]
/// microseconds waited for the flights they needed, [7] reads looked at
/// and settled for after all.
static SETTLE_LISTS_STATS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// Whether a submission with bottom-of-pipe writes may fly
/// (`VICTORY_LATE_WAIT_WRITES=0`: it is settled as soon as it is sent, so
/// the title is told at once, as before; for telling a title that polls
/// such a value from one that does not).
fn late_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_LATE_WAIT_WRITES").is_none_or(|v| v != "0"))
}

/// Per host copy, the ranges a submission's shaders may write, as declared
/// (address, end).
type Declared = (std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>);

/// What [`Host::submit`] hands back for a submission it sent and did not
/// wait for.
struct Sent {
    /// Its number among the submissions started.
    number: u64,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// Its bottom-of-pipe writes.
    writes: Vec<(usize, crate::queue::Written, usize)>,
    /// The title memory its read-backs write once it has run
    /// ([`read_back_spans`]).
    read_back: Vec<(usize, usize)>,
}

/// A submission on the card that has not been waited for ([`late`]): its
/// host lists, taken from the title's (which it may reset meanwhile), and
/// what is to be done once it has run.
struct Flight {
    number: u64,
    /// The value signalled on the host's fence behind it.
    fence: u64,
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    card: Option<CardPlan>,
    back: Vec<Declared>,
    /// The title memory its read-backs write when it lands
    /// ([`read_back_spans`]); not in `back`, so not in `Line::written`.
    read_back: Vec<(usize, usize)>,
    /// The card has run it and its lists gave back their descriptor slots
    /// ([`settle_lists`]); not counted in `Line::lists`.
    freed: bool,
    /// A list of it binds a stage that can store through buffer views it
    /// picks from the heap (`crate::draw::PICKED_STORE_KEY`): it may write
    /// copies' picked spans where it declared nothing.
    picks: bool,
}

/// One thing in the line of what waits for the card, in order.
enum Late {
    Run(Flight),
    /// A bottom-of-pipe write of the flight before it: address, what, bytes.
    Write(usize, crate::queue::Written, usize),
    /// A fence (its address) signalled behind the flights before it.
    Signal(usize, u64),
}

struct Line {
    line: Vec<Late>,
    /// The title memory the flights' shaders may write (address, end), in
    /// order and joined.
    written: Vec<(usize, usize)>,
    /// Host lists in the flights that still hold what they hold (all of
    /// them, without [`settle_lists`]).
    lists: usize,
    /// Host lists in the flights, those freed ([`settle_lists`]) included.
    held: usize,
    /// The value signalled on the host's fence behind the last flight.
    target: u64,
    /// A settle has taken the line and is not done: a signal still goes
    /// behind it.
    settling: bool,
    /// Something was settled away from a queue's own calls: the helper
    /// thread looks at what queues hold.
    poke: bool,
    /// Host lists of settled flights, reset and given back outside the
    /// submission lock ([`land`]: a settle may run under a caller's own
    /// locks).
    landed: Vec<(Vec<crate::list::Detached>, Vec<usize>)>,
    /// With [`relay`]: the bottom-of-pipe writes of the flights being
    /// landed (taken out of `line`, not done yet), for [`late_met`].
    in_hand: Vec<(usize, crate::queue::Written, usize)>,
    /// With [`relay`]: the flights being landed (taken out of `line`), for
    /// [`READ_WAIT`].
    landing: usize,
    /// The title memory the read-backs of the flights in `line` and of
    /// those being landed write when they land (in order, joined): written
    /// besides their declared shader writes (`written`), for
    /// [`READ_EARLY`].
    read_back: Vec<(usize, usize)>,
    /// With [`crate::airlists::on`]: the draw states (dynamic descriptor
    /// slots) of flights the card has run, taken out of their lists before
    /// the landing, to be dropped outside the lock by [`land`].
    released: Vec<crate::draw::State>,
}

// SAFETY: D3D12 objects (free-threaded) and host lists nothing records
// into; touched under the lock.
unsafe impl Send for Line {}

static LINE: Mutex<Line> = Mutex::new(Line {
    line: Vec::new(),
    written: Vec::new(),
    lists: 0,
    held: 0,
    target: 0,
    settling: false,
    poke: false,
    landed: Vec::new(),
    in_hand: Vec::new(),
    landing: 0,
    read_back: Vec::new(),
    released: Vec::new(),
});
/// Told when a flight joins [`LINE`] or it is poked.
static LINE_MOVED: std::sync::Condvar = std::sync::Condvar::new();

fn line() -> std::sync::MutexGuard<'static, Line> {
    LINE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Why flights were settled ([`LATE_STATS`]' places from 2 on).
const WHY_SYNC: usize = 2;
const WHY_CARD_DONE: usize = 3;
const WHY_VALUE: usize = 4;
const WHY_STALE: usize = 5;
const WHY_READ: usize = 6;
const WHY_RUN: usize = 7;
const WHY_MOST: usize = 8;
const WHY_WRITES: usize = 9;
const WHY_FENCE: usize = 10;

/// Since the last frame-rate line: submissions sent without a wait;
/// settles; settles by reason ([`WHY_SYNC`]..); flights settled;
/// microseconds settles waited for the card; value waits met by a flight's
/// bottom-of-pipe write; fence signals set behind flights.
/// Then: the settles for title memory read, by what read it (a buffer
/// copy, a fill of upload memory, a texture).
static LATE_STATS: [std::sync::atomic::AtomicU64; 18] = [const { std::sync::atomic::AtomicU64::new(0) }; 18];

/// Since the last frame-rate line: the microseconds settles waited for the
/// card, by why they settled (the `WHY_` places).
static LATE_WAITED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// Since the last frame-rate line, by why flights were settled (the `WHY_`
/// places), without the relay: microseconds copying back after the wait.
static LATE_BACK: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// Since the last frame-rate line, by why flights were settled, without
/// the relay: of the microseconds waited for the card ([`LATE_WAITED`]),
/// those until the flights the settle needed had run. For title memory
/// read ([`WHY_READ`]): the last flight that declared memory the
/// submission reads; for lists in the air ([`WHY_MOST`]): the first
/// flights, as many as take the line under [`late_most`]; for the others
/// all of it. The rest was spent waiting for later flights: what a wait
/// for the needed flights alone (`VICTORY_RELAY=2`,
/// `VICTORY_FILL_FROM_COPY=1`) could leave to the card.
static LATE_NEED: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];

/// Settles that landed something, since the start: a plan made before one
/// ([`ReadPlan`]) is void after it.
static SETTLES_DONE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// What a fill of upload memory is for ([`Step::Fill`]): a root constant
/// view the card cannot take where it is (not 256-byte aligned: 64 KiB of
/// title memory from its address, as much as a constant buffer may be);
/// a constant view of a descriptor table (its own size); a texture's
/// texels copied from a buffer.
pub(crate) const FILL_ROOT: u8 = 0;
pub(crate) const FILL_TABLE: u8 = 1;
pub(crate) const FILL_TEXTURE: u8 = 2;

/// Places of [`READ_SPLIT`]: of the submissions found reading title memory
/// that flights' shaders may write, those with such a read by: a root
/// constant fill whose first 256 bytes are under flights' writes, one
/// where only bytes past those are, a table constant fill, a texture fill,
/// something else (a title buffer copy, a texture's memory, a step of
/// ours); those whose reads are all fills with every byte under flights'
/// writes in copies in the PC's memory; of the others, those with such
/// bytes in a copy in the card's memory, and those where a copy could not
/// be read for another reason (two copies of the same bytes, a snapshot
/// or range gone, a copy in place, the relay, kept snapshots). Then, with
/// [`fill_from_copy`]: submissions that read from copies instead of a
/// settle, fills that took bytes from copies, the bytes taken, and
/// microseconds waited for the flights they needed.
mod reads {
    pub const ROOT_HEAD: usize = 0;
    pub const ROOT_TAIL: usize = 1;
    pub const TABLE: usize = 2;
    pub const TEXTURE: usize = 3;
    pub const OTHER: usize = 4;
    pub const READABLE: usize = 5;
    pub const CARD: usize = 6;
    pub const UNREADABLE: usize = 7;
    pub const FROM_COPY: usize = 8;
    pub const FILLS: usize = 9;
    pub const BYTES: usize = 10;
    pub const WAITED_US: usize = 11;
    pub const ALL: usize = 12;
}
static READ_SPLIT: [std::sync::atomic::AtomicU64; reads::ALL] = [const { std::sync::atomic::AtomicU64::new(0) }; reads::ALL];

/// Places of [`READ_WAIT`]: with the relay ([`relay_exact`]), the waits of
/// a submission whose preparing reads title memory flights' shaders may
/// write ("title memory read" on the frame line) until the flights that
/// declared it have landed.
///
/// Run n7 showed one such wait a frame, 106-112 ms, with every place of
/// [`READ_SPLIT`] at 0: those are counted only without the relay
/// ([`read_plan`]), so the frame line said nothing of what read. The relay
/// lands flights in order, and each landing without
/// [`inflight_fetch`] fetches card copies' pieces with a list of its own on
/// the host's queue, behind every flight sent meanwhile: a wait for one
/// flight is then a wait for the card to run everything in the air. These
/// places say how much of a wait was the card (until it had run the
/// flights the read needed) and how much the landing behind it.
mod read_wait {
    /// Waits that blocked.
    pub const WAITS: usize = 0;
    /// Microseconds of them.
    pub const US: usize = 1;
    /// Of them, microseconds until the card had run every flight that
    /// declared memory the submission reads (seen at most 1 ms late).
    pub const CARD_US: usize = 2;
    /// Flights in the line at the start of the waits (not counting those
    /// being landed).
    pub const IN_AIR: usize = 3;
    /// Of them, the flights up to the last that declared memory read: the
    /// ones the wait needs landed (flights land in order).
    pub const NEEDED: usize = 4;
    /// Flights being landed at the start of the waits.
    pub const LANDING: usize = 5;
    /// Waits whose needed flights the card had run at the start: all of
    /// the wait was the landing.
    pub const CARD_DONE: usize = 6;
    /// Waits for flights being landed only (none in the line declared
    /// memory read).
    pub const IN_HAND: usize = 7;
    /// Microseconds the landing thread waited for its fetches of card
    /// copies' pieces, across the waits (its own counter's growth).
    pub const FETCH_US: usize = 8;
    /// Host lists in the line at the start of the waits.
    pub const LISTS: usize = 9;
    pub const ALL: usize = 10;
}
static READ_WAIT: [std::sync::atomic::AtomicU64; read_wait::ALL] = [const { std::sync::atomic::AtomicU64::new(0) }; read_wait::ALL];

/// Title memory a flight writes when it lands that is not among its
/// declared shader writes (`Flight::back`, `Line::written`): its
/// read-backs ([`read_backs`]) and its bottom-of-pipe writes. Neither
/// blocks a submission's preparing ([`late_blocked`] looks at `written`
/// only), so a submission sent ahead that reads such memory (a title copy,
/// a fill, a texture) while the flight is still on the card or being
/// landed reads the bytes from before it.
///
/// The old settles landed every flight in the air whenever a read was
/// under any declared write; the relay's second stage ([`relay_exact`])
/// waits only for the flights that declared what is read, so a later
/// flight's read-back over the same memory may still be in the air when
/// the read goes through. Counted right after the submission's waits for
/// flights (a flight may still land before the read itself: an upper
/// bound). Places: submissions that read under a read-back still to land,
/// under a bottom-of-pipe write still to land, and of either, those that
/// had waited in [`Host::relay_read_wait`] (released by the declared
/// writes alone).
///
/// Run n7-n14: the strip of garbage at the bottom of the scene is there
/// with the relay off too (n14), so it is not this; the count says whether
/// the hole is ever taken.
static READ_EARLY: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];

/// The title memory a read-back writes: `rows` rows of `row` bytes,
/// `pitch` apart, from `at` on ([`HostList::defer_read_back`]).
fn read_back_span(at: usize, [row, rows, pitch]: [usize; 3]) -> Option<(usize, usize)> {
    (row > 0 && rows > 0).then(|| (at, at + (rows - 1) * pitch + row))
}

/// The title memory the read-backs of `runs` write (in order, joined).
fn read_back_spans(runs: &[&HostList]) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    for list in runs {
        for (_, at, shape) in &list.deferred().backs {
            spans.extend(read_back_span(*at, *shape));
        }
    }
    let mut out = Vec::new();
    join_ranges(&mut out, spans);
    out
}

/// [`READ_EARLY`] for `runs`, a submission sent ahead, after its waits for
/// flights; `targeted` whether one was [`Host::relay_read_wait`].
fn read_early_look(runs: &[&HostList], targeted: bool) {
    let (backs, writes) = {
        let line = line();
        let mut writes = Vec::new();
        join_ranges(
            &mut writes,
            line.in_hand.iter().map(|&(at, _, n)| (at, at + n)).chain(line.line.iter().filter_map(|l| match l {
                Late::Write(at, _, n) => Some((*at, *at + *n)),
                _ => None,
            })),
        );
        if line.read_back.is_empty() && writes.is_empty() {
            return;
        }
        (line.read_back.clone(), writes)
    };
    let back = blocked_by(&backs, runs, false) == Some(WHY_READ);
    let write = blocked_by(&writes, runs, false) == Some(WHY_READ);
    if back {
        READ_EARLY[0].fetch_add(1, Ordering::Relaxed);
    }
    if write {
        READ_EARLY[1].fetch_add(1, Ordering::Relaxed);
    }
    if targeted && (back || write) {
        READ_EARLY[2].fetch_add(1, Ordering::Relaxed);
    }
}

/// [`READ_WAIT`]'s part of the frame line.
fn read_wait_line(frames: f64) -> String {
    use read_wait as w;
    let v: Vec<f64> = READ_WAIT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64).collect();
    let waits = v[w::WAITS];
    let each = |x: f64| if waits > 0.0 { x / waits } else { 0.0 };
    format!(
        "; title-read waits (relay): {:.1}x {:.1} ms, each with {:.1} flights in the air ({:.1} lists) and {:.1} being landed, \
         {:.1} of them needed (up to the last that declared what is read); the card had run them after {:.1} ms of the {:.1} \
         ({:.1}x already at the start, {:.1}x for flights being landed only), the landing thread fetching card pieces \
         {:.1} ms meanwhile (VICTORY_CARD_FETCH_INFLIGHT={})",
        waits / frames,
        v[w::US] / 1e3 / frames,
        each(v[w::IN_AIR]),
        each(v[w::LISTS]),
        each(v[w::LANDING]),
        each(v[w::NEEDED]),
        each(v[w::CARD_US] / 1e3),
        each(v[w::US] / 1e3),
        v[w::CARD_DONE] / frames,
        v[w::IN_HAND] / frames,
        v[w::FETCH_US] / 1e3 / frames,
        inflight_mode(),
    )
}

/// With the relay, a submission's reads of title memory under flights'
/// shader writes, logged for the first few and then every 512th wait: per
/// read its kind, range and the flights in the line that declared memory
/// it touches (number, host fence value, place in the line, the declared
/// range, and whether its copy is in the card's memory).
fn read_wait_detail(runs: &[&HostList], need: u64, completed: u64) {
    static SEEN: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let n = SEEN.fetch_add(1, Ordering::Relaxed);
    if n >= 8 && n % 512 != 0 {
        return;
    }
    let mut reads: Vec<(usize, usize, &'static str)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    reads.push((src, src + len, "title copy (source)"));
                    reads.push((dst, dst + len, "title copy (destination)"));
                }
                &Step::Fill(_, src, len, kind) => reads.push((
                    src,
                    src + len,
                    match kind {
                        FILL_ROOT => "root constants fill",
                        FILL_TABLE => "table constants fill",
                        FILL_TEXTURE => "texture fill",
                        _ => "other fill",
                    },
                )),
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    reads.push((at, at + len, "texture memory"));
                }
                _ => {}
            }
        }
    }
    let line = line();
    let flights = line.line.iter().filter(|l| matches!(l, Late::Run(_))).count();
    let mut out = String::new();
    let mut shown = 0;
    for &(from, to, what) in &reads {
        if !touches(&line.written, from, to) {
            continue;
        }
        shown += 1;
        if shown > 6 {
            continue;
        }
        out += &format!("\n  {what} {from:#x}+{:#x}:", to - from);
        let mut place = 0;
        let mut any = false;
        for item in &line.line {
            let Late::Run(flight) = item else {
                continue;
            };
            place += 1;
            for (shadow, ranges) in &flight.back {
                for &(a, e) in ranges {
                    if a < to && e > from {
                        any = true;
                        let (start, len) = shadow.span();
                        out += &format!(
                            " [flight {} fence {} at {place} of {flights}: declared {a:#x}+{:#x} in a {} copy {start:#x}+{len:#x}]",
                            flight.number,
                            flight.fence,
                            e - a,
                            if shadow.carded() { "card" } else { "PC" },
                        );
                    }
                }
            }
        }
        if !any {
            out += " (only flights being landed declared it)";
        }
    }
    drop(line);
    tracing::info!(
        wait = n + 1,
        reads_under_writes = shown,
        need,
        completed,
        "title memory read under flights' writes waits for their landing (relay):{out}"
    );
}

/// Whether a submission whose only reads of title memory under flights'
/// shader writes are fills of upload memory, and whose bytes there lie in
/// copies in the PC's memory, waits only for the flights that declared
/// those bytes and reads what they wrote out of the copies, instead of a
/// settle (`VICTORY_FILL_FROM_COPY=1`; off by default; not with the relay,
/// whose `VICTORY_RELAY=2` waits for those flights to land instead).
///
/// A settle waits for everything in the air and copies it back, and the
/// card stands idle from then until the next submission goes up: in run
/// play1 (stage 3) fills settled the flights 15 times a frame (4 flights
/// each time), 33 ms of waiting a frame and their copy-back besides.
///
/// What is read is what the fill would read after the settle, byte for
/// byte: the copy-back puts into title memory, by 64 KiB piece from the
/// first byte where a copy differs from its snapshot to the last, the
/// copy's byte where it differs and the title's elsewhere, and leaves
/// pieces whose title memory is not all mapped alone; the fill takes the
/// title's bytes and then, where the flights wrote, the same merge
/// (`Shadow::read_written`). The flights stay in the air and are copied
/// back at the next settle, as before. Anything else under flights' writes
/// (a title buffer copy, a texture's memory, a step of ours, a copy in the
/// card's memory, bytes two copies hold, a settle inside the preparing)
/// settles as before. Where the merge differs at all: a piece whose title
/// memory is mapped only in part, which the copy-back passes over whole and
/// the fill reads where it is mapped (the shaders' bytes there, not the
/// older title bytes).
fn fill_from_copy() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !relay() && std::env::var_os("VICTORY_FILL_FROM_COPY").is_some_and(|v| v == "1"))
}

/// Whether the constant views of descriptor tables are bound in the host
/// copy of the title buffer they lie in, as an aligned root view is, instead
/// of a copy in the list's upload memory filled from title memory when the
/// list is submitted (`VICTORY_TABLE_ON_CARD`: `1` copies in the card's
/// memory, `2` any copy; off by default; see
/// `crate::draw::table_constant_on_card`).
///
/// GPU-driven rendering: the title's compute passes write constants that
/// later draws read through their tables. The fill reads them on the CPU,
/// so a submission with such a fill settled every flight in the air first
/// ([`WHY_READ`], "table constants" on the frame line): in run n4 one
/// settle a frame, 64 ms waiting for the card and 14 ms copying back, all
/// of it for the flights that wrote them. The host does nothing with the
/// bytes but hand them to the card, and the copy the flights' shaders
/// wrote holds them: bound there, the draw reads what they wrote when the
/// card runs it, after them in the queue's order, and nothing waits. The
/// title's bytes there go up with the copy's refresh, as for any buffer
/// view. A view that lies in memory outside buffer objects, in more than
/// one copy, not 256-byte aligned in its copy, or (at `1`) in a copy in
/// the PC's memory is filled as before.
pub(crate) fn table_on_card() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("VICTORY_TABLE_ON_CARD").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0).min(2)
    })
}

/// [`table_on_card`]'s counts since the last frame-rate line.
pub(crate) mod table_card {
    use std::sync::atomic::{AtomicU64, Ordering};
    /// Constant views of tables bound in a copy.
    pub const BOUND: usize = 0;
    /// Filled: in memory outside buffer objects.
    pub const NO_BUFFER: usize = 1;
    /// Filled: another copy holds some of the bytes.
    pub const OTHER_COPY: usize = 2;
    /// Filled: the copy is in the PC's memory (at `1`).
    pub const NOT_CARDED: usize = 3;
    /// Filled: not 256-byte aligned in the copy, or at its end.
    pub const UNALIGNED: usize = 4;
    /// At submission: views bound in a copy that flights' shaders may
    /// write (each a fill that would have read under flights' writes).
    pub const UNDER: usize = 5;
    /// Submissions with such views.
    pub const SUBMISSIONS: usize = 6;
    /// Of them, those that read nothing else under flights' writes: the
    /// settles the fills would have made.
    pub const AVOIDED: usize = 7;
    /// Of those, the ones whose flights the card had run already.
    pub const ALREADY: usize = 8;
    /// Microseconds from the others until the card had run the flights
    /// their fills would have waited for, seen at later submissions (so at
    /// most that): the waits left to the card.
    pub const WAITED_US: usize = 9;
    pub const ALL: usize = 10;
    pub static STATS: [AtomicU64; ALL] = [const { AtomicU64::new(0) }; ALL];

    pub fn count(place: usize) {
        STATS[place].fetch_add(1, Ordering::Relaxed);
    }
}

/// [`table_card::WAITED_US`]: per settle left out, the host fence value of
/// the last flight its fills would have waited for, and when.
static TABLE_CARD_PENDING: std::sync::Mutex<Vec<(u64, std::time::Instant)>> = std::sync::Mutex::new(Vec::new());

/// [`table_on_card`]'s part of the frame line.
fn table_card_line(frames: f64) -> String {
    use table_card as t;
    let v: Vec<f64> = t::STATS.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    format!(
        "; table constants on the card (VICTORY_TABLE_ON_CARD={}): {:.1} bound in copies, filled {:.1} outside buffers, \
         {:.1} in two copies, {:.1} in PC copies, {:.1} unaligned; {:.1} under flights' writes in {:.1} submissions, \
         {:.1} settles left out ({:.1} with the flights run), {:.1} ms the card still had to run their flights",
        table_on_card(),
        v[t::BOUND],
        v[t::NO_BUFFER],
        v[t::OTHER_COPY],
        v[t::NOT_CARDED],
        v[t::UNALIGNED],
        v[t::UNDER],
        v[t::SUBMISSIONS],
        v[t::AVOIDED],
        v[t::ALREADY],
        v[t::WAITED_US] / 1e3,
    )
}

/// Whether the bytes a submission's preparing would copy out of title
/// memory on the CPU, where earlier flights' shaders may have written them,
/// are copied on the card instead, in queue order after those flights
/// (`VICTORY_TABLE_GPU_COPY`: `1` constant views of descriptor tables, `2`
/// those and the title's buffer copies; off by default).
///
/// Run n7 traced the frame's one "title memory read" wait (106-116 ms
/// under the relay) to such CPU reads: fills of upload memory for tables'
/// constant views (GPU-driven constants), and with [`table_on_card`] a
/// title buffer copy. [`table_on_card`] binds a view in its buffer's copy
/// only when it is 256-byte aligned there and in one copy: 12,066 views a
/// frame were not, and were filled.
///
/// At `1` such a view (any alignment, in one title buffer's copy in the
/// card's memory or split over two side by side) is bound at a 256-byte
/// aligned slot of card memory the list holds ([`CardRing`]). Its bytes are
/// filled on the CPU as before, in the list's order, into scratch memory
/// ([`Step::CardFill`]), and a list run just before the list
/// ([`HostList::card_pre`]) copies the scratch into the slot. Only the bytes
/// that lie under flights' shader writes at submission are copied into the
/// slot from the buffer's copy on the card instead (after the scratch, in
/// queue order after those flights, [`HostList::card_classify`]); the fill
/// does not wait for them. A view whose bytes this list's own shaders or
/// title copies on the card write before the draw is filled as before (the
/// copy would run ahead of them); so is a view in memory outside buffers,
/// in bytes two copies hold, or in a copy in the PC's memory (later
/// preparing writes that while the card may not have copied it yet). A view
/// under flights' writes made in another copy than its own (a texture's,
/// say) is filled with the wait, as before.
///
/// Run n19 (an earlier form: every byte copied from the buffer's copy on
/// the card) drew garbage blocks and lost the room's walls: that copy holds
/// the title's bytes as the whole submission's preparing leaves them, sent
/// ahead of every list (a later list's title copy into the bytes, a step
/// of ours, a write the write watch did not see), where the fill reads them
/// at the view's place in the list. With the fill as the source for every
/// byte no flight writes, the picture is the fill's (run n20's) but for
/// those bytes.
///
/// At `2` also a title buffer copy (`HostList::defer_copy`) whose bytes lie
/// under flights' writes (or this list's shaders', or an earlier copy on the
/// card) is recorded in the list as a copy between the two buffers' copies,
/// in order, and its destination declared as written by the list's shaders:
/// title memory gets it when the flight lands, as for a dispatch's writes.
/// Later CPU reads of that destination in the same list (fills, title
/// copies left on the CPU) read the copy's source instead.
pub(crate) fn table_gpu_copy() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("VICTORY_TABLE_GPU_COPY").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0).min(2)
    })
}

/// [`table_gpu_copy`]'s counts since the last frame-rate line.
pub(crate) mod table_gpu {
    use std::sync::atomic::{AtomicU64, Ordering};
    /// Table constant views bound at a slot the card copies into.
    pub const VIEWS: usize = 0;
    /// Of them, those split over two buffers' copies.
    pub const SPLIT: usize = 1;
    /// Copies recorded for them from their CPU fills (scratch).
    pub const PIECES: usize = 2;
    /// Their bytes.
    pub const BYTES: usize = 3;
    /// Lists run ahead of a list for those copies.
    pub const PRE_LISTS: usize = 4;
    /// Filled: in memory outside buffer objects (or partly).
    pub const NO_BUFFER: usize = 5;
    /// Filled: two copies hold some of the bytes, or more than two buffers.
    pub const OTHER_COPY: usize = 6;
    /// Filled: the list's shaders write some of the bytes before the draw.
    pub const IN_LIST: usize = 7;
    /// Filled: a title copy on the card in the list writes some of them.
    pub const LIST_COPY: usize = 8;
    /// Filled: no card memory for the slot.
    pub const NO_RING: usize = 9;
    /// Title buffer copies recorded on the card.
    pub const TITLE_COPIES: usize = 10;
    /// Their bytes.
    pub const TITLE_BYTES: usize = 11;
    /// Title copies under writes left on the CPU: source or destination
    /// not in one buffer's copy alone.
    pub const TITLE_NOT_SOLE: usize = 12;
    /// ... destination is title memory in place.
    pub const TITLE_PLACED: usize = 13;
    /// ... source and destination in the same copy.
    pub const TITLE_SAME: usize = 14;
    /// Title copies not under writes (left on the CPU).
    pub const TITLE_FREE: usize = 15;
    /// CPU reads (fills, title copies) in a list taken from the source of
    /// an earlier title copy on the card in it.
    pub const REMAPPED: usize = 16;
    /// Title copies left on the CPU into bytes an earlier title copy on the
    /// card in the list wrote (the landing puts the card's bytes back over
    /// them).
    pub const CONFLICTS: usize = 17;
    /// At submission: reads done on the card that lie under flights' writes
    /// (each a CPU read that would have waited).
    pub const UNDER: usize = 18;
    /// Submissions with such reads.
    pub const SUBMISSIONS: usize = 19;
    /// Of them, those that read nothing else under flights' writes on the
    /// CPU: the title-memory-read waits left out.
    pub const AVOIDED: usize = 20;
    /// Table views bound at a slot an earlier view of the same bytes in the
    /// list took (no copy of their own).
    pub const REUSED: usize = 21;
    /// Copies joined to the one before (source and slot both go on).
    pub const JOINED: usize = 22;
    /// Filled: a piece lies in a copy in the PC's memory.
    pub const PC_COPY: usize = 23;
    /// At submission: views (slots) with bytes under flights' writes,
    /// copied from the buffer's copy on the card over their fill.
    pub const CARD_VIEWS: usize = 24;
    /// The copies for those bytes, and the bytes.
    pub const CARD_COPIES: usize = 25;
    pub const CARD_BYTES: usize = 26;
    /// At submission: views under flights' writes made in another copy
    /// than the view's: their fill waits for the flights, as before.
    pub const ELSEWHERE: usize = 27;
    pub const ALL: usize = 28;
    pub static STATS: [AtomicU64; ALL] = [const { AtomicU64::new(0) }; ALL];

    pub fn count(place: usize) {
        STATS[place].fetch_add(1, Ordering::Relaxed);
    }

    pub fn add(place: usize, n: u64) {
        STATS[place].fetch_add(n, Ordering::Relaxed);
    }
}

/// [`table_gpu_copy`]'s part of the frame line.
fn table_gpu_line(frames: f64) -> String {
    use table_gpu as g;
    let v: Vec<f64> = g::STATS.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    format!(
        "; copies on the card (VICTORY_TABLE_GPU_COPY={}): {:.1} table views ({:.1} split over two copies, {:.1} in a slot \
         taken already) filled into slots by {:.1} copies ({:.1} joined), {:.1} KB, {:.1} lists ahead; {:.1} slots with bytes \
         under flights' writes taken from the card in {:.1} copies ({:.1} KB), {:.1} filled with a wait (written in another \
         copy); filled on the CPU {:.1} outside buffers, {:.1} in bytes two copies hold, {:.1} in PC copies, {:.1} written \
         by the list's shaders first, {:.1} by its card title copies first, {:.1} without card memory; {:.1} title copies \
         on the card ({:.1} KB), left on the CPU {:.1} not in one copy, {:.1} into memory in place, {:.1} within one copy, \
         {:.1} not under writes; {:.1} CPU reads taken from a card copy's source, {:.1} CPU copies over a card copy; \
         {:.1} reads on the card under flights' writes in {:.1} submissions, {:.1} title-memory-read waits left out",
        table_gpu_copy(),
        v[g::VIEWS],
        v[g::SPLIT],
        v[g::REUSED],
        v[g::PIECES],
        v[g::JOINED],
        v[g::BYTES] / 1024.0,
        v[g::PRE_LISTS],
        v[g::CARD_VIEWS],
        v[g::CARD_COPIES],
        v[g::CARD_BYTES] / 1024.0,
        v[g::ELSEWHERE],
        v[g::NO_BUFFER],
        v[g::OTHER_COPY],
        v[g::PC_COPY],
        v[g::IN_LIST],
        v[g::LIST_COPY],
        v[g::NO_RING],
        v[g::TITLE_COPIES],
        v[g::TITLE_BYTES] / 1024.0,
        v[g::TITLE_NOT_SOLE],
        v[g::TITLE_PLACED],
        v[g::TITLE_SAME],
        v[g::TITLE_FREE],
        v[g::REMAPPED],
        v[g::CONFLICTS],
        v[g::UNDER],
        v[g::SUBMISSIONS],
        v[g::AVOIDED],
    )
}

/// Title buffer copies (`CopyBufferRegion` between two buffers) recorded
/// on the card, in the list's order, instead of run on the CPU when the
/// submission is prepared (`VICTORY_COPY_ON_CARD`; off by default).
///
/// The CPU copy runs before any of its submission's lists and, unless it
/// waits (`VICTORY_COPY_AHEAD` unset), after every flight in the air has
/// landed; on the console the copy runs on the card after every command
/// before it in queue order, and never holds the title's submitting thread.
/// Run n49 (copies under flights' writes waiting for them, 1x 28 ms a
/// frame for all 14 flights in the air) lost the character's ponytail; n50 (the
/// same copies run ahead) had it, and its early-read check found that
/// none of the 8-14 copies a frame that went ahead read bytes a landing
/// then changed. On the card the copy needs neither: its source's copy
/// holds what earlier flights' shaders wrote and what the title wrote
/// (the refresh steps before it), in queue order.
///
/// - `1`: every title copy whose source and destination each lie in one
///   buffer's host copy alone, not the same one, the destination not title
///   memory in place, and the source's copy in the card's memory (a copy in
///   the PC's memory or an upload heap is written by later submissions'
///   preparing at once, maybe before the card has copied out of it);
/// - `2`: of those, only the ones whose bytes lie under flights' writes
///   (or the list's own shaders', or an earlier copy on the card) when the
///   title records the copy.
///
/// The destination is declared as written by the list's shaders: title
/// memory gets it when the flight lands, before the fences behind it are
/// set (the title's CPU reads it after its fence). Later CPU reads of it in
/// the same list (fills, title copies left on the CPU) read the copy's
/// source instead ([`through_card_copies`]); CPU reads of it in later lists
/// of the same submission are counted ([`card_copy::LATER_READS`]): they
/// read title memory from before the copy. Copies left on the CPU are
/// waited for (or not) as before.
pub(crate) fn copy_on_card() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("VICTORY_COPY_ON_CARD").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(0).min(2)
    })
}

/// [`copy_on_card`]'s counts since the last frame-rate line.
pub(crate) mod card_copy {
    use std::sync::atomic::{AtomicU64, Ordering};
    /// Title copies recorded on the card.
    pub const COPIES: usize = 0;
    /// Their bytes.
    pub const BYTES: usize = 1;
    /// Of them, those under flights' writes (or the list's) when recorded.
    pub const UNDER: usize = 2;
    /// Left on the CPU: source or destination not in one buffer's copy.
    pub const NOT_SOLE: usize = 3;
    /// ... destination is title memory in place.
    pub const PLACED: usize = 4;
    /// ... source and destination in the same copy.
    pub const SAME: usize = 5;
    /// ... source's copy not in the card's memory.
    pub const NOT_CARD: usize = 6;
    /// ... (`2`) not under writes when recorded.
    pub const FREE: usize = 7;
    /// At submission: CPU reads (fills, title copies) in a later list of
    /// the submission of bytes a card copy of an earlier list writes.
    pub const LATER_READS: usize = 8;
    /// At submission: title copies left on the CPU in submissions that also
    /// hold copies on the card.
    pub const CPU_BESIDE: usize = 9;
    pub const ALL: usize = 10;
    pub static STATS: [AtomicU64; ALL] = [const { AtomicU64::new(0) }; ALL];

    pub fn count(place: usize) {
        STATS[place].fetch_add(1, Ordering::Relaxed);
    }

    pub fn add(place: usize, n: u64) {
        STATS[place].fetch_add(n, Ordering::Relaxed);
    }
}

/// [`copy_on_card`]'s part of the frame line.
fn card_copy_line(frames: f64) -> String {
    use card_copy as c;
    let v: Vec<f64> = c::STATS.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    format!(
        "; title copies on the card (VICTORY_COPY_ON_CARD={}): {:.1} ({:.1} KB, {:.1} under writes when recorded); left on \
         the CPU {:.1} not in one copy, {:.1} into memory in place, {:.1} within one copy, {:.1} source not in the card's \
         memory, {:.1} not under writes; at submission {:.1} CPU reads in later lists of what a card copy writes, {:.1} CPU \
         copies beside card copies",
        copy_on_card(),
        v[c::COPIES],
        v[c::BYTES] / 1024.0,
        v[c::UNDER],
        v[c::NOT_SOLE],
        v[c::PLACED],
        v[c::SAME],
        v[c::NOT_CARD],
        v[c::FREE],
        v[c::LATER_READS],
        v[c::CPU_BESIDE],
    )
}

/// [`copy_on_card`], at a submission: the CPU reads of its lists that lie
/// in what a title copy on the card in an earlier list of it writes (they
/// run before any list), and the CPU title copies beside such copies.
fn card_copy_look(runs: &[&HostList]) {
    use card_copy as c;
    let mut written: Vec<(usize, usize)> = Vec::new();
    let (mut later, mut beside, mut any) = (0u64, 0u64, false);
    for list in runs {
        let deferred = list.deferred();
        for step in &deferred.copies {
            let (src, len) = match step {
                &Step::Copy(_, src, len) => {
                    beside += 1;
                    (src, len)
                }
                &Step::Fill(_, src, len, _) | &Step::CardFill(_, src, len) => (src, len),
                _ => continue,
            };
            if !written.is_empty() && written.iter().any(|&(a, e)| a < src + len && e > src) {
                later += 1;
            }
        }
        for &(d, _, n) in &deferred.gpu_copied {
            any = true;
            written.push((d, d + n));
        }
    }
    if any {
        c::add(c::LATER_READS, later);
        c::add(c::CPU_BESIDE, beside);
    }
}

/// [`table_gpu_copy`], for the frame line, before a submission sent ahead
/// looks for reads under flights' writes: the reads its lists left to the
/// card that lie under them, and whether nothing else it reads on the CPU
/// does (a wait left out).
fn table_gpu_look(runs: &[&HostList]) {
    use table_gpu as g;
    let reads: Vec<(usize, usize)> = runs.iter().flat_map(|list| list.deferred().gpu_reads.clone()).collect();
    if reads.is_empty() {
        return;
    }
    let written = {
        let line = line();
        if line.written.is_empty() {
            return;
        }
        line.written.clone()
    };
    let under = reads.iter().filter(|&&(a, n)| touches(&written, a, a + n)).count() as u64;
    if under == 0 {
        return;
    }
    g::add(g::UNDER, under);
    g::count(g::SUBMISSIONS);
    if blocked_by(&written, runs, false).is_none() {
        g::count(g::AVOIDED);
    }
}

/// `[from, from + len)` of title memory as a list's CPU steps read it at
/// submission, given the title copies recorded on the card before them in
/// the list (`copied`: destination, source, bytes, in order; see
/// [`table_gpu_copy`]): pieces (offset in the range, address to read,
/// bytes), in order. Bytes a copy on the card wrote are read at that
/// copy's source (through earlier copies again); the others where they are.
fn through_card_copies(copied: &[(usize, usize, usize)], from: usize, len: usize, base: usize, out: &mut Vec<(usize, usize, usize)>) {
    if len == 0 {
        return;
    }
    let to = from + len;
    let Some(i) = copied.iter().rposition(|&(d, _, n)| d < to && d + n > from) else {
        out.push((base, from, len));
        return;
    };
    let (d, s, n) = copied[i];
    let (lo, hi) = (from.max(d), to.min(d + n));
    // No later copy touches the range (the last one that does is `i`):
    // the parts beside it see the same copies, the part inside only those
    // before `i`.
    through_card_copies(copied, from, lo - from, base, out);
    through_card_copies(&copied[..i], s + (lo - d), hi - lo, base + (lo - from), out);
    through_card_copies(copied, hi, to - hi, base + (hi - from), out);
}

/// Ranges in order and joined, for [`table_gpu_copy`]'s per-list record of
/// what the list's shaders write.
#[derive(Default)]
struct Joined(Vec<(usize, usize)>);

impl Joined {
    fn add(&mut self, from: usize, to: usize) {
        if from >= to {
            return;
        }
        // The first range that ends at or after `from` and the first that
        // starts after `to`: those between join the new one.
        let a = self.0.partition_point(|r| r.1 < from);
        let b = self.0.partition_point(|r| r.0 <= to);
        if a == b {
            self.0.insert(a, (from, to));
            return;
        }
        let joined = (self.0[a].0.min(from), self.0[b - 1].1.max(to));
        self.0.splice(a..b, [joined]);
    }

    fn touches(&self, from: usize, to: usize) -> bool {
        touches(&self.0, from, to)
    }
}

/// The parts of `[from, to)` that lie in `ranges` (in order, joined), in
/// order.
fn parts_in(ranges: &[(usize, usize)], from: usize, to: usize) -> Vec<(usize, usize)> {
    if from >= to {
        return Vec::new();
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges[at..]
        .iter()
        .take_while(|r| r.0 < to)
        .map(|r| (r.0.max(from), r.1.min(to)))
        .filter(|r| r.0 < r.1)
        .collect()
}

/// Whether `ranges` (in any order, overlapping or not) cover `[from, to)`
/// whole.
pub(crate) fn covered(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    let mut near: Vec<(usize, usize)> = ranges.iter().copied().filter(|r| r.0 < to && r.1 > from).collect();
    near.sort_unstable();
    let mut at = from;
    for (a, e) in near {
        if a > at {
            return false;
        }
        at = at.max(e);
        if at >= to {
            return true;
        }
    }
    at >= to
}

/// [`table_gpu_copy`]: what a step does to a slot's fill and copies on the
/// card, as the step scanners see it. Whether `step` reads title memory
/// that flights' shaders may write (`written`, in order, joined) on the
/// CPU, so that preparing must wait for them; and why ([`WHY_READ`] or
/// [`WHY_RUN`]). A [`Step::CardFill`] is not such a read: what flights
/// write in its bytes is put over it on the card.
fn step_blocks(step: &Step, written: &[(usize, usize)]) -> Option<(usize, usize)> {
    match step {
        &Step::Copy(dst, src, len) => {
            (touches(written, src, src + len) || touches(written, dst, dst + len)).then_some((WHY_READ, 15))
        }
        &Step::Fill(_, src, len, _) => touches(written, src, src + len).then_some((WHY_READ, 16)),
        Step::Texture(staging) => {
            let (at, len) = staging.memory();
            touches(written, at, at + len).then_some((WHY_READ, 17))
        }
        Step::Run(_) => Some((WHY_RUN, usize::MAX)),
        Step::CardFill(..) | Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(..) => None,
    }
}

/// A copy of `len` bytes between two buffers (resource and offset each),
/// as the list ahead of a list records it ([`HostList::card_pre`]).
#[derive(Clone, Debug, PartialEq)]
struct Region<R> {
    src: R,
    src_at: u64,
    dst: R,
    dst_at: u64,
    len: u64,
}

/// The copy of title bytes `[lo, hi)` of a view at a card slot (`dst` at
/// `ring_at`, the view's bytes from title address `from` on) out of the
/// buffer copy `src` that holds them, `src_at` being where `lo` lies there.
fn part_region<R>(src: R, src_at: u64, dst: R, ring_at: u64, from: usize, lo: usize, hi: usize) -> Region<R> {
    Region { src, src_at, dst, dst_at: ring_at + (lo - from) as u64, len: (hi - lo) as u64 }
}

/// Puts `next` after `copies`, joined to the last one where both its source
/// and destination go on from it in the same buffers. Whether it joined.
fn push_region<R: PartialEq>(copies: &mut Vec<Region<R>>, next: Region<R>) -> bool {
    if let Some(last) = copies.last_mut()
        && last.src == next.src
        && last.dst == next.dst
        && last.src_at + last.len == next.src_at
        && last.dst_at + last.len == next.dst_at
    {
        last.len += next.len;
        return true;
    }
    copies.push(next);
    false
}

/// Card memory a list's table constant views are copied into
/// ([`table_gpu_copy`]): chunks handed out front to back in 256-byte
/// aligned slots, reused after a reset (every submission of the list has
/// run then), given back to [`CARD_RING_SPARE`] when the list goes.
#[derive(Default)]
struct CardRing {
    chunks: Vec<RingChunk>,
    current: usize,
    used: u64,
}

/// A device-local buffer of [`CARD_RING_CHUNK`] bytes and its address.
#[derive(Clone)]
struct RingChunk {
    buffer: ID3D12Resource,
    gpu: u64,
}

// SAFETY: a D3D12 resource (free-threaded).
unsafe impl Send for RingChunk {}

const CARD_RING_CHUNK: u64 = 1 << 20;
/// The most chunks kept while no list holds them.
const CARD_RING_SPARE_MOST: usize = 64;
static CARD_RING_SPARE: Mutex<Vec<RingChunk>> = Mutex::new(Vec::new());

impl CardRing {
    /// A slot of `len` bytes (rounded up to 256): its chunk, the offset in
    /// it and its host GPU address.
    fn take(&mut self, len: u64) -> Option<(ID3D12Resource, u64, u64)> {
        let len = len.max(1).next_multiple_of(256);
        if len > CARD_RING_CHUNK {
            return None;
        }
        while self.current < self.chunks.len() && self.used + len > CARD_RING_CHUNK {
            (self.current, self.used) = (self.current + 1, 0);
        }
        if self.current == self.chunks.len() {
            let spare = CARD_RING_SPARE.lock().unwrap_or_else(|e| e.into_inner()).pop();
            let chunk = match spare {
                Some(chunk) => chunk,
                None => {
                    let buffer = host()?.uav_buffer(CARD_RING_CHUNK)?;
                    name(&buffer, || "list card constants".into());
                    // SAFETY: a live buffer.
                    let gpu = unsafe { buffer.GetGPUVirtualAddress() };
                    RingChunk { buffer, gpu }
                }
            };
            self.chunks.push(chunk);
            self.used = 0;
        }
        let chunk = &self.chunks[self.current];
        let at = self.used;
        self.used += len;
        Some((chunk.buffer.clone(), at, chunk.gpu + at))
    }

    /// After a reset: every slot free again.
    fn rewind(&mut self) {
        (self.current, self.used) = (0, 0);
    }
}

impl Drop for CardRing {
    fn drop(&mut self) {
        let mut spare = CARD_RING_SPARE.lock().unwrap_or_else(|e| e.into_inner());
        for chunk in self.chunks.drain(..) {
            if spare.len() < CARD_RING_SPARE_MOST {
                spare.push(chunk);
            }
        }
    }
}

/// A copy on the card ahead of a list ([`HostList::card_pre`]) from the
/// list's scratch (upload memory a [`Step::CardFill`] fills, which stays in
/// its one state) into a slot of its [`CardRing`]; the buffers are held
/// until the list is reset (the scratch by the list, the slot's chunk here).
type CardCopy = Region<ID3D12Resource>;

/// A table constant view bound at a slot of a list's [`CardRing`]
/// ([`table_gpu_copy`]), once per slot.
struct CardView {
    /// The slot: its chunk and offset there.
    ring: ID3D12Resource,
    ring_at: u64,
    /// The title memory the view reads.
    from: usize,
    to: usize,
    /// The title buffers' copies in the card's memory that hold it (one,
    /// or two side by side): copy, from, to (title addresses).
    pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Its fill's place in the list's steps.
    step: usize,
    /// Under flights' writes made in another copy: its fill became a
    /// [`Step::Fill`] that waits for them, as before.
    waits: bool,
    /// At the last submission ([`HostList::card_classify`]): the bytes
    /// under flights' writes, copied over the fill from the copy that
    /// holds them (piece, from, to).
    on_card: Vec<(usize, usize, usize)>,
}

// SAFETY: D3D12 resources (free-threaded) and copies; touched under the
// list's lock.
unsafe impl Send for CardView {}

impl CardView {
    /// [`under_writes`] of the view's pieces.
    fn under_writes(
        &self,
        written: &[(usize, usize)],
        pending: impl Fn(usize, usize, usize) -> bool,
    ) -> Result<Vec<(usize, usize, usize)>, ()> {
        let pieces: Vec<(usize, usize)> = self.pieces.iter().map(|p| (p.1, p.2)).collect();
        under_writes(&pieces, written, pending)
    }
}

/// At a submission, for a view at a card slot held by `pieces` (title
/// ranges, each in one copy), with `written` the title memory flights'
/// shaders may write (in order, joined) and `pending(piece, from, to)`
/// whether those writes in `[from, to)` are made in the piece's own copy:
/// the parts of the view to copy from the copies (piece, from, to), in
/// order. `Err` if some part under flights' writes is written in another
/// copy, which the view's copy does not hold.
fn under_writes(
    pieces: &[(usize, usize)],
    written: &[(usize, usize)],
    pending: impl Fn(usize, usize, usize) -> bool,
) -> Result<Vec<(usize, usize, usize)>, ()> {
    let mut out = Vec::new();
    for (i, &(from, to)) in pieces.iter().enumerate() {
        for (lo, hi) in parts_in(written, from, to) {
            if !pending(i, lo, hi) {
                return Err(());
            }
            out.push((i, lo, hi));
        }
    }
    Ok(out)
}

/// For a submission that reads title memory flights' shaders may write
/// ([`WHY_READ`]): the host fence value of the last flight that declared
/// memory it reads ([`LATE_NEED`]), and whether it can read from copies
/// instead of a settle ([`fill_from_copy`]): per copy, the parts of its
/// fills under flights' writes (copy, from, to: title addresses).
struct ReadPlan {
    need: u64,
    from_copy: Option<Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>>,
}

/// [`ReadPlan`] for `runs`, counted in [`READ_SPLIT`].
fn read_plan(runs: &[&HostList]) -> ReadPlan {
    // The reads of title memory, as (from, to, kind): a `FILL_` kind, or
    // `u8::MAX` for anything that is not a fill.
    const NOT_FILL: u8 = u8::MAX;
    let mut read: Vec<(usize, usize, u8)> = Vec::new();
    let mut ours = false;
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    read.push((src, src + len, NOT_FILL));
                    read.push((dst, dst + len, NOT_FILL));
                }
                &Step::Fill(_, src, len, kind) => read.push((src, src + len, kind)),
                Step::Texture(staging) => {
                    let (at, len) = staging.memory();
                    read.push((at, at + len, NOT_FILL));
                }
                Step::Run(_) => ours = true,
                Step::CardFill(..) | Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(..) => {}
            }
        }
    }
    let mut seen = [false; reads::READABLE];
    let mut need = if ours { u64::MAX } else { 0 };
    // Per fill under flights' writes: (copy, declared range, from, to).
    let mut parts: Vec<(std::sync::Arc<crate::shadow::Shadow>, (usize, usize), usize, usize)> = Vec::new();
    {
        let line = line();
        for &(from, to, kind) in &read {
            if !touches(&line.written, from, to) {
                continue;
            }
            seen[match kind {
                FILL_ROOT if touches(&line.written, from, (from + 256).min(to)) => reads::ROOT_HEAD,
                FILL_ROOT => reads::ROOT_TAIL,
                FILL_TABLE => reads::TABLE,
                FILL_TEXTURE => reads::TEXTURE,
                _ => reads::OTHER,
            }] = true;
            for item in &line.line {
                let Late::Run(flight) = item else {
                    continue;
                };
                for (shadow, ranges) in &flight.back {
                    for &(a, e) in ranges {
                        if a < to && e > from {
                            need = need.max(flight.fence);
                            if kind != NOT_FILL {
                                parts.push((shadow.clone(), (a, e), a.max(from), e.min(to)));
                            }
                        }
                    }
                }
            }
        }
    }
    if ours {
        seen[reads::OTHER] = true;
    }
    for (place, &seen) in seen.iter().enumerate() {
        if seen {
            READ_SPLIT[place].fetch_add(1, Ordering::Relaxed);
        }
    }
    let readable = || -> Result<Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>, usize> {
        if ours || seen[reads::OTHER] {
            return Err(reads::OTHER);
        }
        let mut out = Vec::with_capacity(parts.len());
        for (shadow, declared, from, to) in &parts {
            // Only the bytes the copy holds: shaders write nothing past it.
            let (start, len) = shadow.span();
            let (from, to) = ((*from).max(start as usize), (*to).min((start + len) as usize));
            if from >= to {
                continue;
            }
            if shadow.carded() {
                return Err(reads::CARD);
            }
            if !shadow.written_readable(*declared, from, to) {
                return Err(reads::UNREADABLE);
            }
            out.push((shadow.clone(), from, to));
        }
        // Bytes two copies hold: whichever copy-back came last would put
        // its bytes there.
        out.sort_by_key(|p| p.1);
        for (i, a) in out.iter().enumerate() {
            for b in out[i + 1..].iter().take_while(|b| b.1 < a.2) {
                if !std::sync::Arc::ptr_eq(&a.0, &b.0) {
                    return Err(reads::UNREADABLE);
                }
            }
        }
        Ok(out)
    };
    let from_copy = match readable() {
        Ok(parts) => {
            READ_SPLIT[reads::READABLE].fetch_add(1, Ordering::Relaxed);
            Some(parts)
        }
        Err(why) => {
            if why != reads::OTHER {
                READ_SPLIT[why].fetch_add(1, Ordering::Relaxed);
            }
            None
        }
    };
    ReadPlan { need, from_copy }
}

/// [`fill_from_copy`]: the fill of `len` bytes of title memory at `src`
/// into `dst` has run; where `parts` (copy, from, to) lie in it, what
/// flights' shaders wrote is taken from the copies.
fn from_copies(parts: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)], dst: usize, src: usize, len: usize) {
    let end = src + len;
    let mut taken = 0;
    let mut any = false;
    for (shadow, from, to) in parts {
        let (from, to) = ((*from).max(src), (*to).min(end));
        if from < to {
            any = true;
            taken += shadow.read_written(from, to, (dst + (from - src)) as *mut u8);
        }
    }
    if any {
        READ_SPLIT[reads::FILLS].fetch_add(1, Ordering::Relaxed);
        READ_SPLIT[reads::BYTES].fetch_add(taken as u64, Ordering::Relaxed);
    }
}

/// For a settle for [`WHY_MOST`]: the host fence value of the flight up to
/// which the line must land to hold fewer than [`late_most`] host lists.
fn most_need() -> u64 {
    let line = line();
    let most = late_most();
    let mut left = line.lists;
    for item in &line.line {
        if let Late::Run(flight) = item
            && !flight.freed
        {
            left = left.saturating_sub(flight.order.len().max(1));
            if left < most {
                return flight.fence;
            }
        }
    }
    u64::MAX
}

/// [`settle_lists`] at 2, for a submission whose preparing reads title
/// memory flights' shaders may write ([`WHY_READ`]), once the card has run
/// every flight up to host fence value `need` (all that declared memory it
/// reads, [`ReadPlan::need`]): whether no shader of theirs wrote any byte
/// it reads, so title memory holds what a settle would leave there and
/// none is due.
///
/// Yes only where every flight that declared a range under a read is known
/// to have left it alone: the range lies in a copy in the card's memory,
/// the flight's compare took every piece the read holds of it, and found
/// each unchanged (a piece's hash after the flight's lists against the
/// hash kept for it: a shader that wrote it changed it). The compare runs
/// in the flight's own submission, before the fence value it signals, so
/// its results are there once the card reached `need`; they stay until the
/// flight lands (`Host::card_done`). Anything not known so says no: copies
/// in the PC's memory, a flight without the plan or the piece, a range the
/// compare's area had no room for, textures' memory, steps of ours, and a
/// title copy into memory flights may write (it would put title bytes
/// where a later merge looks for the shaders').
fn reads_unwritten(runs: &[&HostList], need: u64) -> bool {
    let mut read: Vec<(usize, usize)> = Vec::new();
    let mut into: Vec<(usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            match step {
                &Step::Copy(dst, src, len) => {
                    read.push((src, src + len));
                    into.push((dst, dst + len));
                }
                &Step::Fill(_, src, len, _) => read.push((src, src + len)),
                Step::Texture(_) | Step::Run(_) => return false,
                Step::CardFill(..) | Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(..) => {}
            }
        }
    }
    // The compare's results: one dword per piece. Under the submission
    // lock, which the caller holds, nothing makes them anew.
    let (differs, words) = {
        let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        match work.as_ref() {
            Some(work) => (work.differs.1 as *const u32, (work.differs.2 / 4) as usize),
            None => return false,
        }
    };
    // SAFETY: the compare's persistent mapping, `words` dwords; the card
    // has run the compares read below.
    let differs = unsafe { std::slice::from_raw_parts(differs, words) };
    let line = line();
    if into.iter().any(|&(from, to)| touches(&line.written, from, to)) {
        return false;
    }
    for &(from, to) in &read {
        if !touches(&line.written, from, to) {
            continue;
        }
        for item in &line.line {
            let Late::Run(flight) = item else {
                continue;
            };
            for (shadow, ranges) in &flight.back {
                for &(a, e) in ranges {
                    if a >= to || e <= from {
                        continue;
                    }
                    if flight.fence > need || !shadow.carded() {
                        return false;
                    }
                    // Only the bytes the copy holds: shaders write nothing
                    // past it.
                    let (start, len) = shadow.span();
                    let (start, end) = (start as usize, (start + len) as usize);
                    let (lo, hi) = (a.max(from).max(start), e.min(to).min(end));
                    if lo >= hi {
                        continue;
                    }
                    let Some(card) = &flight.card else {
                        return false;
                    };
                    if !card_unwritten(card, shadow, lo - start, hi - start, differs) {
                        return false;
                    }
                }
            }
        }
    }
    true
}

/// For [`reads_unwritten`]: whether `plan`'s compares took every byte of
/// offsets `[from, to)` of `shadow` (a copy in the card's memory) in
/// pieces they found unchanged.
fn card_unwritten(plan: &CardPlan, shadow: &std::sync::Arc<crate::shadow::Shadow>, from: usize, to: usize, differs: &[u32]) -> bool {
    let mut taken: Vec<(usize, usize)> = Vec::new();
    for r in plan.ranges.iter().filter(|r| std::sync::Arc::ptr_eq(&r.shadow, shadow)) {
        for p in 0..r.span.pieces {
            let (lo, hi) = r.shadow.card_piece(&r.span, p);
            if lo >= to || hi <= from || lo >= hi {
                continue;
            }
            if !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0) {
                return false;
            }
            taken.push((lo.max(from), hi.min(to)));
        }
    }
    covers(&mut taken, from, to)
}

/// Whether `parts` (in any order) cover `[from, to)` without a gap.
fn covers(parts: &mut [(usize, usize)], from: usize, to: usize) -> bool {
    parts.sort_unstable();
    let mut at = from;
    for &(lo, hi) in parts.iter() {
        if lo > at {
            break;
        }
        at = at.max(hi);
        if at >= to {
            break;
        }
    }
    at >= to
}

/// The settles since the last frame-rate line, for the frame line: per
/// frame, by why they settled, how many, ms waiting for the card (of it,
/// without the relay, ms until the flights they needed had run) and ms
/// copying back; then the reads of title memory under flights' writes
/// ([`READ_SPLIT`]). Read before [`late_line`], which takes the counts.
fn settle_split(frames: f64) -> String {
    let s = |i: usize| LATE_STATS[i].load(Ordering::Relaxed) as f64 / frames;
    let w = |i: usize| LATE_WAITED[i].load(Ordering::Relaxed) as f64 / 1e3 / frames;
    let b: Vec<f64> = LATE_BACK.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames).collect();
    let n: Vec<f64> = LATE_NEED.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames).collect();
    let r: Vec<f64> = READ_SPLIT.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let mut out = format!("{:.1} settles", s(1));
    for (why, name) in [
        (WHY_READ, "title memory read"),
        (WHY_MOST, "lists in the air"),
        (WHY_SYNC, "synchronous or present"),
        (WHY_CARD_DONE, "card done"),
        (WHY_VALUE, "memory value"),
        (WHY_STALE, "title wrote under shader writes"),
        (WHY_RUN, "a step of ours"),
        (WHY_WRITES, "bottom-of-pipe writes"),
        (WHY_FENCE, "blocking fence wait"),
    ] {
        if s(why) == 0.0 && w(why) == 0.0 {
            continue;
        }
        out += &format!(", {name} {:.1}x {:.1} ms waiting", s(why), w(why));
        if !relay() {
            out += &format!(" ({:.1} until the flights needed) + {:.1} copying back", n[why], b[why]);
        }
    }
    let f: Vec<f64> = SETTLE_LISTS_STATS.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    if settle_lists() > 0 {
        out += &format!(
            "; instead of settles (VICTORY_SETTLE_LISTS={}): lists in the air given back {:.1}x ({:.1} flights, {:.1} lists, \
             {:.1} ms waiting for the card), settled all the same {:.1}x; reads found unwritten {:.1}, settled after the look {:.1}, \
             {:.1} ms waiting for the flights they needed",
            settle_lists(),
            f[0],
            f[1],
            f[2],
            f[3] / 1e3,
            f[4],
            f[5],
            f[7],
            f[6] / 1e3,
        );
    }
    if table_on_card() > 0 {
        out += &table_card_line(frames);
    }
    if table_gpu_copy() > 0 {
        out += &table_gpu_line(frames);
    }
    if copy_on_card() > 0 {
        out += &card_copy_line(frames);
    }
    if relay_exact() {
        out += &read_wait_line(frames);
    }
    out += &format!(
        "; reads under flights' writes: {:.1} root constants in their first 256 bytes, {:.1} only past them, {:.1} table constants, \
         {:.1} textures' texels, {:.1} other (title copies, textures' memory, our steps); {:.1} all fills in PC copies, \
         {:.1} in a card copy, {:.1} unreadable; from copies (VICTORY_FILL_FROM_COPY) {:.1} instead of a settle, {:.1} fills \
         {:.1} KB, {:.1} ms waiting for the flights they needed",
        r[reads::ROOT_HEAD],
        r[reads::ROOT_TAIL],
        r[reads::TABLE],
        r[reads::TEXTURE],
        r[reads::OTHER],
        r[reads::READABLE],
        r[reads::CARD],
        r[reads::UNREADABLE],
        r[reads::FROM_COPY],
        r[reads::FILLS],
        r[reads::BYTES] / 1024.0,
        r[reads::WAITED_US] / 1e3,
    );
    out
}

/// Whether `[from, to)` touches one of `ranges` (in order, joined).
fn touches(ranges: &[(usize, usize)], from: usize, to: usize) -> bool {
    if from >= to {
        return false;
    }
    let at = ranges.partition_point(|r| r.1 <= from);
    ranges.get(at).is_some_and(|r| r.0 < to)
}

/// `ranges` with `more`, in order, joined where they overlap or touch.
fn join_ranges(ranges: &mut Vec<(usize, usize)>, more: impl IntoIterator<Item = (usize, usize)>) {
    ranges.extend(more.into_iter().filter(|r| r.0 < r.1));
    ranges.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(ranges.len());
    for &(from, to) in ranges.iter() {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    *ranges = out;
}

/// Whether the line of flights holds anything (flights, bottom-of-pipe
/// writes, signals) or is being landed or settled: something the landing
/// or settling thread will still do without being asked
/// (`crate::queue`'s waits on its lines, `VICTORY_ASYNC_SUBMIT=1`).
pub(crate) fn line_moving() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    line.settling || !line.line.is_empty() || line.landing > 0
}

/// Whether anything is in the air.
fn flying() -> bool {
    if !late() {
        return false;
    }
    let line = line();
    // (With the relay: also while flights are being landed.)
    line.settling || line.line.iter().any(|l| matches!(l, Late::Run(_)))
}

/// With flights in the air ([`late`]): why `runs` cannot be prepared
/// before they are settled, if so. Preparing reads title memory on the
/// CPU (the title's buffer copies, fills of upload memory, textures) and
/// runs steps of ours that may read or write it anywhere; what the
/// flights' shaders write is not there yet.
fn late_blocked(runs: &[&HostList], count: bool) -> Option<usize> {
    let written = {
        let line = line();
        if line.written.is_empty() {
            return None;
        }
        line.written.clone()
    };
    blocked_by(&written, runs, count)
}

/// [`late_blocked`] against `written`, the title memory flights' shaders
/// may write (in order, joined); `count` whether the kind of read is
/// counted for the frame line.
fn blocked_by(written: &[(usize, usize)], runs: &[&HostList], count: bool) -> Option<usize> {
    if written.is_empty() {
        return None;
    }
    let ahead = fill_ahead::on();
    for list in runs {
        for step in &list.deferred().copies {
            // Upstream's `fill_ahead` (on unless `VICTORY_FILL_WAIT=1`): a
            // fill, or a title copy that only reads where flights may
            // write, is not waited for; [`note_reads_ahead`] has it
            // checked when the flights land.
            if ahead && read_ahead(step, written).is_some() {
                continue;
            }
            if let Some((why, stat)) = step_blocks(step, written) {
                if count && let Some(n) = LATE_STATS.get(stat) {
                    n.fetch_add(1, Ordering::Relaxed);
                }
                #[cfg(feature = "diagnostics")]
                if count {
                    match step {
                        &Step::Copy(dst, src, len) => fill_check::blocked_copy(written, dst, src, len),
                        &Step::Fill(_, src, len, _) => fill_check::blocked(written, src, len),
                        _ => {}
                    }
                }
                return Some(why);
            }
        }
    }
    None
}

/// Upstream's `fill_ahead`: the title memory `step` reads on the CPU
/// ahead of flights that may write it, if it is a fill of upload memory
/// that touches `written`, or (with [`copy_ahead`]) a title copy whose
/// source does and whose destination does not (a copy INTO such a range
/// always waits).
fn read_ahead(step: &Step, written: &[(usize, usize)]) -> Option<(usize, usize)> {
    match step {
        &Step::Copy(dst, src, len) => (copy_ahead() && touches(written, src, src + len) && !touches(written, dst, dst + len))
            .then_some((src, len)),
        &Step::Fill(_, src, len, _) => touches(written, src, src + len).then_some((src, len)),
        _ => None,
    }
}

/// Whether a title buffer copy whose source lies under flights' declared
/// shader writes goes ahead without waiting for them, as upstream's
/// `fill_ahead` let it (`VICTORY_COPY_AHEAD=0` makes such a copy wait until
/// those flights have landed, as before upstream 0b0074e; on by default:
/// with the wait one test run lost the character's hair, and the early-read
/// check below found 0 copies that read bytes a flight then changed).
///
/// A fill read ahead of the shaders' writes is used by one submission's
/// draws and is gone; a copy's destination stays in title memory, and the
/// title reads it there: on the CPU once its fence is set (a read-back of
/// what the card counted or found visible), or from later GPU work
/// (indirect arguments). Copied ahead of the flight that writes the
/// source, it holds the bytes from before, and nothing copies them again.
/// Whether the flight had landed when the copy ran depended on timing, run
/// to run. The check upstream leaves behind (`fill_ahead::wrote`) finds
/// such a copy only after it ran, and found none: it never saw the reads
/// of a submission that had waited for other flights first (they were
/// noted only when nothing waited). `crate::early` reports them now.
fn copy_ahead() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_AHEAD").is_none_or(|v| v != "0"))
}

/// Right before a submission's steps run, after every wait for flights:
/// with [`fill_ahead`] on, its fills (and copies, [`copy_ahead`]) under
/// the writes of flights still in the air ([`read_ahead`]) are noted, to be
/// compared with what those flights write when they land; and its title
/// copies whose source lies under shader writes not in title memory yet
/// (flights in the air, or this submission's own lists before the copy)
/// are given to `crate::early`. `number` is the submission's.
///
/// Upstream noted the reads inside `blocked_by`; this fork asked
/// `blocked_by` uncounted first and noted them only when nothing waited,
/// so a submission that waited for some flights (the relay's targeted
/// wait, [`Host::relay_read_wait`]) and then read ahead of others was
/// never looked at. Here they are noted where they read: after the waits,
/// with the flights in the air numbered below `number`.
fn note_reads_ahead(runs: &[&HostList], number: u64) {
    let ahead = fill_ahead::on();
    let early = crate::early::on();
    if !ahead && !early {
        return;
    }
    let written = line().written.clone();
    // The ranges this submission's shaders declared, in step order.
    let mut own: Vec<(usize, usize)> = Vec::new();
    for list in runs {
        for step in &list.deferred().copies {
            if ahead
                && !written.is_empty()
                && let Some((src, len)) = read_ahead(step, &written)
            {
                fill_ahead::read(src, len, number.saturating_sub(1));
            }
            if !early {
                continue;
            }
            match step {
                &Step::Copy(dst, src, len) => {
                    let end = src + len;
                    let under_own = own.iter().any(|&(a, e)| a < end && e > src);
                    crate::early::copy(dst, src, len, number, touches(&written, src, end), under_own);
                }
                Step::GpuWrite(_, address, len) => own.push((*address, *address + *len)),
                _ => {}
            }
        }
    }
}

/// Fills of upload memory (a constant buffer a table names, copied into
/// the list's own memory when the list is submitted) that read title memory
/// flights "may write" are no longer waited for; what the flights then
/// write is checked against them, and the first fill that read bytes a
/// flight changed afterwards turns the waiting back on for good.
///
/// Why: a flight declares every range its shaders can store to, and the
/// title has an unordered view over the whole of its 174 MB buffer, whose
/// copy is in the card's memory. Its per-draw constants lie in the same
/// buffer, at new addresses every frame. So the first such fill of a frame
/// waited until those flights had landed: one wait a frame, 56 to 95 ms of
/// the submitting thread standing still (runs 202 and 206; 145 ms while
/// the world loaded in 208), the largest single wait of the frame. Run 208
/// looked at every one of them: 44 fills of 480 bytes, and after the wait
/// the bytes were the same 44 times.
///
/// What it risks, and how it is bounded: a fill that reads bytes a flight
/// in the air is about to change takes the old bytes, and the lists of
/// that one submission draw with them. `wrote` sees it when the flight
/// lands (the bytes going into title memory are compared with the ones
/// there, over every fill that read ahead of them) and from then on fills
/// wait as before. `VICTORY_FILL_WAIT=1`: they wait from the start.
mod fill_ahead {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};

    /// Fills are looked up by the 64 KiB of title memory they read.
    const PIECE: usize = 1 << 16;
    /// A fill is checked against flights that land this many frames after
    /// it at most (flights are settled at 256 lists in the air, and by
    /// every present that waits).
    const KEEP: u64 = 16;

    /// A flight changed bytes a fill had read ahead of it: fills wait.
    static WAIT: AtomicBool = AtomicBool::new(false);
    /// Since the last frame line: fills that read ahead; landings' writes
    /// that touched one's piece. Since the start: fills that read bytes a
    /// flight then changed.
    static COUNT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

    #[derive(Default)]
    struct Piece {
        /// The frame its old fills were last dropped in.
        pruned: u64,
        /// From, to, frame, and the number of the last submission sent
        /// when it read: the flights in the air then have that number or
        /// a lower one, and only their writes can be ones it missed.
        fills: Vec<(usize, usize, u64, u64)>,
    }

    /// The lowest submission number among the flights the landing thread
    /// is landing now; 0: not known (a landing by another way), and then
    /// every write counts. (Run 211: a read of frame 1381 was taken for
    /// missed because shaders wrote the same 80 bytes in frame 1397, in a
    /// flight sent sixteen frames after it.)
    static LANDING: AtomicU64 = AtomicU64::new(0);

    pub(super) fn landing(first: u64) {
        LANDING.store(first, Relaxed);
    }

    static READ: Mutex<Option<HashMap<usize, Piece>>> = Mutex::new(None);

    pub(super) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_WAIT").is_none_or(|v| v != "1")) && !WAIT.load(Relaxed)
    }

    /// A fill reads `len` bytes at `src` ahead of flights that may write
    /// them; `upto`: the highest number among the flights in the air.
    pub(super) fn read(src: usize, len: usize, upto: u64) {
        if len == 0 {
            return;
        }
        COUNT[0].fetch_add(1, Relaxed);
        let frame = crate::frame::current();
        let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
        let map = read.get_or_insert_with(HashMap::new);
        for piece in src / PIECE..=(src + len - 1) / PIECE {
            let p = map.entry(piece).or_default();
            if p.pruned != frame {
                p.pruned = frame;
                p.fills.retain(|f| f.2 + KEEP >= frame);
            }
            p.fills.push((src, src + len, frame, upto));
        }
    }

    /// A landing is about to put `new` into title memory at `at`: bytes
    /// shaders wrote, merged into the title's.
    pub(crate) fn wrote(at: usize, new: &[u8]) {
        crate::early::wrote(at, new);
        if new.is_empty() || WAIT.load(Relaxed) {
            return;
        }
        let frame = crate::frame::current();
        let landing = LANDING.load(Relaxed);
        let hits: Vec<(usize, usize, u64)> = {
            let read = READ.lock().unwrap_or_else(|e| e.into_inner());
            let Some(map) = read.as_ref() else { return };
            let end = at + new.len();
            (at / PIECE..=(end - 1) / PIECE)
                .filter_map(|piece| map.get(&piece))
                .flat_map(|p| p.fills.iter().copied())
                .filter(|f| f.0 < end && f.1 > at && f.2 + KEEP >= frame && (landing == 0 || landing <= f.3))
                .map(|f| (f.0, f.1, f.2))
                .collect()
        };
        if hits.is_empty() {
            return;
        }
        COUNT[1].fetch_add(1, Relaxed);
        let mut now = Vec::new();
        for (from, to, read_in) in hits {
            let (a, b) = (from.max(at), to.min(at + new.len()));
            now.clear();
            now.resize(b - a, 0);
            // Title memory as it is, which is what the fill read unless the
            // title wrote it since; not there: nothing to say.
            if !super::read_into(a as *const u8, &mut now) || now[..] == new[a - at..b - at] {
                continue;
            }
            let n = COUNT[2].fetch_add(1, Relaxed) + 1;
            let first = now.iter().zip(&new[a - at..b - at]).position(|(x, y)| x != y).unwrap_or(0);
            if !WAIT.swap(true, Relaxed) {
                tracing::warn!(
                    "a fill of upload memory read {} bytes at {from:#x} in frame {read_in} ahead of a flight whose shaders changed \
                     them (first at {:#x}, landed in frame {frame}): the lists of that submission drew with the bytes from before. \
                     Fills wait for flights again from here on, as with VICTORY_FILL_WAIT=1 ({n} such fills)",
                    to - from,
                    a + first
                );
            }
            return;
        }
    }

    /// With the frame line.
    pub(super) fn line(frames: f64) {
        let [ahead, looked] = [0, 1].map(|i| COUNT[i].swap(0, Relaxed));
        if ahead == 0 && looked == 0 {
            return;
        }
        let kept = {
            let frame = crate::frame::current();
            let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
            read.as_mut().map_or(0, |map| {
                map.retain(|_, p| {
                    p.fills.retain(|f| f.2 + KEEP >= frame);
                    !p.fills.is_empty()
                });
                map.values().map(|p| p.fills.len()).sum::<usize>()
            })
        };
        tracing::info!(
            "fills of upload memory that read title memory flights may write, not waited for (VICTORY_FILL_WAIT=1 waits): {:.1} per \
             frame; {:.1} landings' writes per frame fell in the same 64 KiB as one and were compared with it; fills that read \
             bytes a flight then changed, since the start: {}{}; {kept} fills kept to check",
            ahead as f64 / frames,
            looked as f64 / frames,
            COUNT[2].load(Relaxed),
            if WAIT.load(Relaxed) { " (fills wait again)" } else { "" }
        );
    }
}

pub(crate) use fill_ahead::wrote as shaders_write;

/// Diagnostics: the fills of upload memory that made a submission wait for
/// flights (`blocked_by`), which one they are, and whether the wait was
/// needed: the bytes the fill reads are kept when it blocks and compared
/// with what it reads once the flights have landed.
///
/// Runs 202 and 206: one such wait a frame, 56 to 95 ms of the submitting
/// thread standing still, the largest single wait of the frame. The ranges
/// flights "may write" are whole views shaders can store to; if the bytes
/// a fill reads never change under the wait, the wait is for nothing.
#[cfg(feature = "diagnostics")]
mod fill_check {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// The fill waited for now: source, length, its bytes when it blocked.
    static PENDING: Mutex<Option<(usize, usize, Vec<u8>)>> = Mutex::new(None);
    /// Fills that blocked by (source page, length): how often.
    static SEEN: Mutex<Vec<(usize, usize, u64)>> = Mutex::new(Vec::new());
    /// Fills checked after their wait; those whose bytes had changed; the
    /// bytes that had.
    static CHECKED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

    /// A fill of `len` bytes at `src` blocks on `written`.
    pub(super) fn blocked(written: &[(usize, usize)], src: usize, len: usize) {
        let at = written.partition_point(|r| r.1 <= src);
        let touched = written.get(at).copied().unwrap_or((0, 0));
        let key = (src & !0xfff, len);
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let known = seen.iter().position(|s| (s.0, s.1) == key);
        match known {
            Some(k) => seen[k].2 += 1,
            None if seen.len() < 256 => {
                seen.push((key.0, key.1, 1));
                if seen.len() <= 12 {
                    tracing::info!(
                        "a fill of upload memory waits for flights: {len} bytes at {src:#x}; the range shaders may write that it                          touches: {:#x}+{:#x} ({} ranges in the air); frame {}",
                        touched.0,
                        touched.1 - touched.0,
                        written.len(),
                        crate::frame::current()
                    );
                }
            }
            None => {}
        }
        drop(seen);
        let mut bytes = vec![0u8; len.min(1 << 20)];
        if super::read_into(src as *const u8, &mut bytes) {
            *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((src, len, bytes));
        }
    }

    /// The fill of `len` bytes at `src` runs now.
    pub(super) fn runs(src: usize, len: usize) {
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        if !pending.as_ref().is_some_and(|p| (p.0, p.1) == (src, len)) {
            return;
        }
        let Some((_, _, before)) = pending.take() else { return };
        drop(pending);
        let mut now = vec![0u8; before.len()];
        if !super::read_into(src as *const u8, &mut now) {
            return;
        }
        let differ = before.iter().zip(&now).filter(|(a, b)| a != b).count() as u64;
        let n = CHECKED[0].fetch_add(1, Relaxed) + 1;
        if differ > 0 {
            let bad = CHECKED[1].fetch_add(1, Relaxed) + 1;
            CHECKED[2].fetch_add(differ, Relaxed);
            if bad <= 12 {
                let first = before.iter().zip(&now).position(|(a, b)| a != b).unwrap_or(0);
                tracing::info!(
                    "a fill of upload memory that waited for flights read other bytes after the wait: {differ} of {len} bytes at                      {src:#x} changed, the first at +{first:#x} (check {n}, frame {})",
                    crate::frame::current()
                );
            }
        }
    }

    /// The title buffer copy waited for now: destination, source, length,
    /// the bytes of both when it blocked.
    static PENDING_COPY: Mutex<Option<(usize, usize, usize, Vec<u8>, Vec<u8>)>> = Mutex::new(None);
    /// Copies checked after their wait; those whose source had changed;
    /// those whose destination had.
    static COPIES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    /// Copies that blocked by (source page, destination page, length).
    static SEEN_COPIES: Mutex<Vec<(usize, usize, usize, u64)>> = Mutex::new(Vec::new());

    /// A title buffer copy of `len` bytes from `src` to `dst` blocks on
    /// `written`.
    pub(super) fn blocked_copy(written: &[(usize, usize)], dst: usize, src: usize, len: usize) {
        let by_src = super::touches(written, src, src + len);
        let by_dst = super::touches(written, dst, dst + len);
        let key = (src & !0xfff, dst & !0xfff, len);
        let mut seen = SEEN_COPIES.lock().unwrap_or_else(|e| e.into_inner());
        let known = seen.iter().position(|s| (s.0, s.1, s.2) == key);
        match known {
            Some(k) => seen[k].3 += 1,
            None if seen.len() < 256 => {
                seen.push((key.0, key.1, key.2, 1));
                if seen.len() <= 16 {
                    let range = |at: usize| {
                        let k = written.partition_point(|r| r.1 <= at);
                        written.get(k).copied().unwrap_or((0, 0))
                    };
                    let (s, d) = (range(src), range(dst));
                    tracing::info!(
                        "a title buffer copy waits for flights: {len} bytes from {src:#x} to {dst:#x}; source in a range shaders may                          write: {by_src} ({:#x}+{:#x}); destination: {by_dst} ({:#x}+{:#x}); {} ranges in the air; frame {}",
                        s.0,
                        s.1 - s.0,
                        d.0,
                        d.1 - d.0,
                        written.len(),
                        crate::frame::current()
                    );
                }
            }
            None => {}
        }
        drop(seen);
        let n = len.min(1 << 20);
        let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
        if super::read_into(src as *const u8, &mut a) && super::read_into(dst as *const u8, &mut b) {
            *PENDING_COPY.lock().unwrap_or_else(|e| e.into_inner()) = Some((dst, src, len, a, b));
        }
    }

    /// The title buffer copy runs now.
    pub(super) fn copy_runs(dst: usize, src: usize, len: usize) {
        let mut pending = PENDING_COPY.lock().unwrap_or_else(|e| e.into_inner());
        if !pending.as_ref().is_some_and(|p| (p.0, p.1, p.2) == (dst, src, len)) {
            return;
        }
        let Some((_, _, _, src_before, dst_before)) = pending.take() else { return };
        drop(pending);
        let (mut a, mut b) = (vec![0u8; src_before.len()], vec![0u8; dst_before.len()]);
        if !super::read_into(src as *const u8, &mut a) || !super::read_into(dst as *const u8, &mut b) {
            return;
        }
        let n = COPIES[0].fetch_add(1, Relaxed) + 1;
        let (src_changed, dst_changed) = (a != src_before, b != dst_before);
        if src_changed {
            COPIES[1].fetch_add(1, Relaxed);
        }
        if dst_changed {
            COPIES[2].fetch_add(1, Relaxed);
        }
        if (src_changed || dst_changed) && COPIES[1].load(Relaxed) + COPIES[2].load(Relaxed) <= 16 {
            let differ = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(p, q)| p != q).count();
            tracing::info!(
                "a title buffer copy that waited for flights found other bytes after the wait: {len} bytes from {src:#x} to {dst:#x};                  {} source bytes changed, {} destination bytes changed (check {n}, frame {})",
                differ(&a, &src_before),
                differ(&b, &dst_before),
                crate::frame::current()
            );
        }
    }

    /// With the frame line.
    pub(super) fn line() {
        let copies = COPIES[0].load(Relaxed);
        if copies > 0 {
            let seen = SEEN_COPIES.lock().unwrap_or_else(|e| e.into_inner());
            let mut most: Vec<_> = seen.clone();
            most.sort_unstable_by_key(|s| std::cmp::Reverse(s.3));
            let most: Vec<String> =
                most.iter().take(6).map(|s| format!("{:#x}->{:#x}+{:#x} {}x", s.0, s.1, s.2, s.3)).collect();
            tracing::info!(
                "title buffer copies that waited for flights, since the start: {copies} checked after their wait; the source had                  other bytes after it {} times, the destination {} times; {} distinct copies, the most frequent: {}",
                COPIES[1].load(Relaxed),
                COPIES[2].load(Relaxed),
                seen.len(),
                most.join(", ")
            );
        }
        let n = CHECKED[0].load(Relaxed);
        if n == 0 {
            return;
        }
        let seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let mut most: Vec<_> = seen.clone();
        most.sort_unstable_by_key(|s| std::cmp::Reverse(s.2));
        let most: Vec<String> = most.iter().take(6).map(|s| format!("{:#x}+{:#x} {}x", s.0, s.1, s.2)).collect();
        tracing::info!(
            "fills of upload memory that waited for flights, since the start: {n} checked after their wait, {} of them read other              bytes than before it ({} bytes in all); {} distinct fills (page, length), the most frequent: {}",
            CHECKED[1].load(Relaxed),
            CHECKED[2].load(Relaxed),
            seen.len(),
            most.join(", ")
        );
    }
}

/// Per copy that shaders of flights may be writing (`Shadow::writes_pending`),
/// the span of what `runs` refresh in it (title addresses; a flush brings
/// that much up to date at most).
fn refresh_spans(runs: &[&HostList]) -> Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> {
    use std::sync::Arc;
    let mut spans: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    if stale_once() {
        // Each copy looked up once: where its span is, or none if no
        // shader writes are pending in it ([`stale_once`]).
        const IDLE: usize = usize::MAX;
        let mut at: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
        let mut steps = 0u64;
        for list in runs {
            for step in &list.deferred().copies {
                let Step::Refresh(shadow, address, len) = step else {
                    continue;
                };
                steps += 1;
                let (from, to) = (*address, *address + *len);
                match at.entry(Arc::as_ptr(shadow) as usize) {
                    std::collections::hash_map::Entry::Occupied(e) => {
                        if *e.get() != IDLE {
                            let s = &mut spans[*e.get()];
                            (s.1, s.2) = (s.1.min(from), s.2.max(to));
                        }
                    }
                    std::collections::hash_map::Entry::Vacant(e) => {
                        if shadow.writes_pending() {
                            e.insert(spans.len());
                            spans.push((shadow.clone(), from, to));
                        } else {
                            e.insert(IDLE);
                        }
                    }
                }
            }
        }
        more_add(more::STALE_STEPS, steps);
        more_add(more::STALE_COPIES, at.len() as u64);
        more_add(more::STALE_SPANS, spans.len() as u64);
    } else {
        for list in runs {
            for step in &list.deferred().copies {
                if let Step::Refresh(shadow, address, len) = step
                    && shadow.writes_pending()
                {
                    match spans.iter_mut().find(|s| Arc::ptr_eq(&s.0, shadow)) {
                        Some(s) => (s.1, s.2) = (s.1.min(*address), s.2.max(*address + *len)),
                        None => spans.push((shadow.clone(), *address, *address + *len)),
                    }
                }
            }
        }
    }
    spans
}

/// With flights in the air, after the submission's write-watch asking:
/// whether `runs` refresh a range of a host copy that a flight's shaders
/// may write and that the title wrote since it was last copied. The
/// refresh leaves ranges shaders are writing alone (`Shadow::update`), so
/// the title's bytes there would reach the card a submission late.
fn stale_under_flights(runs: &[&HostList]) -> bool {
    let spans = refresh_spans(runs);
    if late_exact() {
        // The title's bytes are put in beside the flights' writes; a
        // settle is still due where that could not be done. At most this
        // many changed runs go up to copies in the card's memory, each a
        // copy of its own there.
        // (`VICTORY_MERGE_RUNS`: the cap. Run 218 had 500 to 5,400 changed
        // runs a frame of a few bytes each, and the cap of 512 settled the
        // flights twice a frame, 50 ms of the submitting thread.)
        static RUNS: OnceLock<usize> = OnceLock::new();
        let mut runs_left = *RUNS.get_or_init(|| {
            std::env::var("VICTORY_MERGE_RUNS").ok().and_then(|v| v.trim().parse::<usize>().ok()).map_or(512, |n| n.clamp(1, 1 << 16))
        });
        return spans.iter().any(|(shadow, from, to)| shadow.merge_under_writes(*from, *to - *from, &mut runs_left));
    }
    spans.iter().any(|(shadow, from, to)| shadow.stale_under_writes(*from, *to - *from))
}

/// [`stale_under_flights`], timed into `ns` and [`PREP_SPLIT`].
fn stale_timed(runs: &[&HostList], ns: &mut u64) -> bool {
    let checking = std::time::Instant::now();
    let stale = stale_under_flights(runs);
    let took = checking.elapsed().as_nanos() as u64;
    *ns += took;
    PREP_SPLIT[prep::STALE].fetch_add(took, Ordering::Relaxed);
    stale
}

/// A fence signal on a queue ([`late`]): goes behind the flights in the
/// air, if there are any, and is set when they are settled. Whether it
/// did; if not the caller sets the fence now.
pub(crate) fn late_signal(fence: usize, value: u64) -> bool {
    if !late() {
        return false;
    }
    let mut line = line();
    if line.line.is_empty() && !line.settling {
        return false;
    }
    line.line.push(Late::Signal(fence, value));
    LATE_STATS[14].fetch_add(1, Ordering::Relaxed);
    drop(line);
    // A signal alone (behind a settle under way) has no flight to wake
    // the helper thread for it.
    LINE_MOVED.notify_one();
    RELAY_MOVED.notify_one();
    true
}

/// A queue stands at `wait`, which memory does not meet now ([`late`]):
/// whether it is met once the flights in the air have run. A flight's
/// bottom-of-pipe write of a value that meets it says yes without a wait
/// for the card: the part behind the wait goes to the card behind that
/// flight, which is the order the wait asks for. If the flights may write
/// the value in a way not known here (shaders, a timestamp, another
/// width), they are settled and memory is asked again.
pub(crate) fn late_met(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    {
        let line = line();
        if line.line.is_empty() && line.in_hand.is_empty() {
            return false;
        }
        let mut unsure = touches(&line.written, address, address + bytes);
        // (With the relay: first the writes of flights being landed.)
        let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
            Late::Write(at, what, n) => Some((at, what, n)),
            _ => None,
        }));
        for (at, what, n) in writes {
            if *at >= address + bytes || address >= *at + *n {
                continue;
            }
            match what {
                crate::queue::Written::Value(v) if *at == address && *n >= bytes => {
                    if wait.holds_for(*v) {
                        LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
                        return true;
                    }
                }
                _ => unsure = true,
            }
        }
        if !unsure {
            return false;
        }
    }
    settle_now(WHY_VALUE);
    wait.met()
}

/// [`late_met`] without its settle: whether a bottom-of-pipe write of a
/// flight in the air, or of one being landed, is known to meet `wait`.
/// False says nothing about what else the flights may write there.
pub(crate) fn late_met_known(wait: &crate::list::ValueWait) -> bool {
    let (address, bytes) = wait.place();
    let line = line();
    let writes = line.in_hand.iter().map(|(at, what, n)| (at, what, n)).chain(line.line.iter().filter_map(|item| match item {
        Late::Write(at, what, n) => Some((at, what, n)),
        _ => None,
    }));
    for (at, what, n) in writes {
        if let crate::queue::Written::Value(v) = what
            && *at == address
            && *n >= bytes
            && wait.holds_for(*v)
        {
            LATE_STATS[13].fetch_add(1, Ordering::Relaxed);
            return true;
        }
    }
    false
}

/// Settles the flights in the air, if any ([`Host::settle`]), for reason
/// `why` (one of the `WHY_` places). Not from inside a submission.
pub(crate) fn settle_now(why: usize) {
    if !late() {
        return;
    }
    let Some(host) = host() else {
        return;
    };
    let mut submitted = host.lock_submitted();
    host.settle(&mut submitted, why);
}

/// [`settle_now`] at a present, and for a fence the title blocks on.
pub(crate) fn settle_for_present() {
    settle_now(WHY_SYNC);
}
pub(crate) fn settle_for_fence() {
    settle_now(WHY_FENCE);
}

/// Resets the host lists of settled flights and keeps them for reuse, as
/// a queue does for the lists it ran (`crate::list::recycle`).
fn land() {
    let (landed, released) = {
        let mut line = line();
        (std::mem::take(&mut line.landed), std::mem::take(&mut line.released))
    };
    // Their descriptor slots go back (the heaps' own lock).
    drop(released);
    for (lists, order) in landed {
        crate::list::recycle(lists, &order);
    }
}

/// The thread that settles the flights once the card has run them all
/// ([`late`]): without it a bottom-of-pipe write would reach the title only
/// at the next call that settles, and a title thread that polls such a
/// value while the submitting thread waits for that thread would never
/// see it.
fn late_helper() {
    let Some(host) = host() else {
        return;
    };
    crate::framelog::keep_on_fast_cores("victory-late-wait");
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        tracing::warn!("no event for the thread that settles submissions sent ahead");
        return;
    };
    loop {
        let target = {
            let mut line = line();
            loop {
                if line.poke {
                    line.poke = false;
                    break None;
                }
                // (With the relay the flights are another thread's.)
                if !relay() && !line.line.is_empty() && !line.settling {
                    break Some(line.target);
                }
                line = LINE_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some(target) = target else {
            land();
            crate::queue::release_held(false);
            continue;
        };
        // The card first, without the submission lock: the submitting
        // thread prepares and sends meanwhile.
        // SAFETY: our fence and event.
        unsafe {
            if host.fence.SetEventOnCompletion(target, event).is_ok() {
                WaitForSingleObject(event, INFINITE);
            }
        }
        {
            let mut submitted = host.lock_submitted();
            // Something newer went up meanwhile: the card is waited for
            // again, the submitting thread is not held up for it.
            if line().target != target {
                continue;
            }
            host.settle(&mut submitted, WHY_CARD_DONE);
        }
        land();
        crate::queue::release_held(false);
    }
}

/// Starts the threads of [`late`], once: the one that settles flights when
/// the card has run them and looks at what queues hold, and with [`relay`]
/// the one that lands flights.
fn late_threads() {
    static HELPER: std::sync::Once = std::sync::Once::new();
    HELPER.call_once(|| {
        if let Err(e) = std::thread::Builder::new().name("victory-late-wait".into()).spawn(late_helper) {
            tracing::warn!(error = %e, "no thread to settle submissions sent ahead: they are settled by the calls that follow");
        }
        tracing::info!(
            most = late_most(),
            writes = late_writes(),
            "submissions are sent to the card without a wait for each (VICTORY_LATE_WAIT=1)"
        );
        if relay() {
            // Without it nothing would ever land.
            if let Err(e) = std::thread::Builder::new().name("victory-relay".into()).spawn(relay_worker) {
                fail(format_args!("no thread to land submissions sent ahead (VICTORY_RELAY=1): {e}"));
            }
            fence_early::start_clock();
            tracing::info!(
                exact = relay_exact(),
                "what shaders wrote is copied back by a thread of its own, beside the submitting thread (VICTORY_RELAY=1)"
            );
        }
    });
}

/// With [`relay`]: waits until `done` says so of the line, looked at each
/// time flights have landed. Counted as a settle for `why`, and its time
/// as waiting for the card. The caller may hold the submission lock: the
/// thread that lands flights never takes it.
fn relay_wait(why: usize, done: impl Fn(&Line) -> bool) {
    relay_wait_split(why, None, done);
}

/// [`relay_wait`], split for the frame's path line (`crate::cpuwait`) when
/// `split` names the kind of wait and gives the host fence value up to
/// which the card must have run the flights for `done` to come true (the
/// rest of the wait is the landing thread's): what was in the air at the
/// start, and when the card had run the needed flights. Looked at every
/// millisecond until then, as [`Host::relay_read_wait`] does.
fn relay_wait_split(why: usize, split: Option<(usize, &dyn Fn(&Line) -> u64)>, done: impl Fn(&Line) -> bool) {
    use crate::cpuwait::{wait, wait_add};
    let mut line = line();
    if done(&line) {
        return;
    }
    LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
    LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
    // SAFETY (both): a plain query.
    let completed = || host().map_or(u64::MAX, |h| unsafe { h.fence.GetCompletedValue() });
    let split = split.map(|(kind, need)| {
        let need = need(&line);
        let now = completed();
        let (mut in_air, mut run, mut needed) = (0u64, 0u64, 0u64);
        for item in &line.line {
            if let Late::Run(flight) = item {
                in_air += 1;
                run += u64::from(flight.fence <= now);
                needed += u64::from(flight.fence <= need);
            }
        }
        wait_add(kind, wait::WAITS, 1);
        wait_add(kind, wait::IN_AIR, in_air);
        wait_add(kind, wait::RUN, run);
        wait_add(kind, wait::LISTS, line.lists as u64);
        wait_add(kind, wait::NEEDED, needed);
        if now >= need {
            wait_add(kind, wait::CARD_DONE, 1);
        }
        (kind, need, now >= need)
    });
    let was_working = working(false);
    let waiting = std::time::Instant::now();
    let mut card = split.and_then(|s| s.2.then_some(std::time::Duration::ZERO));
    while !done(&line) {
        match split {
            Some((_, need, _)) if card.is_none() => {
                line = LINE_LANDED.wait_timeout(line, std::time::Duration::from_millis(1)).unwrap_or_else(|e| e.into_inner()).0;
                if completed() >= need {
                    card = Some(waiting.elapsed());
                }
            }
            _ => line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner()),
        }
    }
    drop(line);
    stood(waiting);
    working(was_working);
    if let Some((kind, ..)) = split {
        let all = waiting.elapsed();
        wait_add(kind, wait::US, all.as_micros() as u64);
        wait_add(kind, wait::CARD_US, card.unwrap_or(all).as_micros() as u64);
    }
    let us = waiting.elapsed().as_micros() as u64;
    LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
    LATE_WAITED[why.min(10)].fetch_add(us, Ordering::Relaxed);
    submission_us(1, us);
}

/// For a [`WHY_MOST`] wait with [`relay_exact`] ([`relay_wait_split`]): the
/// host fence value up to which the card must have run the flights for
/// fewer than [`late_most`] lists to be in the air once they have landed
/// (0: the flights being landed are enough).
fn most_need_relay(line: &Line) -> u64 {
    let flights: Vec<(u64, usize)> = line
        .line
        .iter()
        .filter_map(|item| match item {
            Late::Run(flight) => Some((flight.fence, flight.order.len().max(1))),
            _ => None,
        })
        .collect();
    most_need_lists(&flights, late_most())
}

/// [`most_need_relay`] of flights (host fence value, lists) in the line's
/// order: the fence value of the last flight that must land for fewer than
/// `most` lists to stay (0: none).
fn most_need_lists(flights: &[(u64, usize)], most: usize) -> u64 {
    let mut left: usize = flights.iter().map(|f| f.1).sum();
    let mut need = 0;
    for &(fence, lists) in flights {
        if left < most {
            break;
        }
        left -= lists;
        need = fence;
    }
    need
}

/// What the thread that lands flights fetches card copies' pieces with
/// ([`relay`]): a list, a fence and an event of its own (the host's are
/// behind the submission lock), and the memory the pieces come down to.
struct RelayFetch {
    list: Prelude,
    fence: ID3D12Fence,
    event: HANDLE,
    /// The last value signalled on `fence`.
    sent: u64,
    /// Buffer, mapping, bytes.
    buffer: Option<(ID3D12Resource, usize, u64)>,
    /// With [`relay_copy_queue`]: the copy queue, its allocator and lists,
    /// made at the first fetch there (they signal `fence` too).
    copy: Option<CopyFetch>,
    /// With [`relay_copy_queue`]: the highest send of title bytes that the
    /// card had when the flights landed so far ran.
    mark: u64,
}

/// [`RelayFetch::copy`]: a copy queue, an allocator, and the lists for the
/// pieces copied at once and those copied behind a flight.
type CopyFetch = (ID3D12CommandQueue, ID3D12CommandAllocator, ID3D12GraphicsCommandList, ID3D12GraphicsCommandList);

/// Submissions sent ahead whose lists went to the queue, and those of them
/// that joined [`LINE`] since ([`flights_in_line`]). Both rise under the
/// submission lock, the first right before the lists go to the queue.
static FLIGHTS_SENT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static FLIGHTS_LINED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// With `VICTORY_QUEUE2_PICKED=2`, after a fetch on the copy queue that
/// stood behind host fence value `behind` has run: whether a flight that
/// may have run beside it (any in the line after `behind`, every flight
/// whose lists went to the queue so far being there) binds a stage that
/// can store through buffer views it picks from the heap (`Flight::picks`).
/// Such a flight may have been writing a picked span where it declared
/// nothing, and no landing would fetch those bytes again.
fn picks_beside(behind: u64) -> bool {
    flights_in_line();
    line().line.iter().any(|item| matches!(item, Late::Run(flight) if flight.fence > behind && flight.picks))
}

/// Waits until every flight whose lists went to the queue so far is in
/// [`LINE`] (the submitting thread puts it there right after sending it,
/// with no wait between): what the line holds then is every flight the
/// card may be running.
fn flights_in_line() {
    let sent = FLIGHTS_SENT.load(Ordering::SeqCst);
    while FLIGHTS_LINED.load(Ordering::SeqCst) < sent {
        std::thread::yield_now();
    }
}

/// The host fence value of the last flight in `line` after host fence
/// value `after` that declared shader writes in offsets `lo..hi` of
/// `shadow` (a card copy), if one did.
fn declared_by(line: &Line, shadow: &std::sync::Arc<crate::shadow::Shadow>, lo: usize, hi: usize, after: u64) -> Option<u64> {
    let start = shadow.span().0 as usize;
    let (a, b) = (start + lo, start + hi);
    line.line
        .iter()
        .filter_map(|item| match item {
            Late::Run(flight)
                if flight.fence > after
                    && flight.back.iter().any(|(s, ranges)| {
                        std::sync::Arc::ptr_eq(s, shadow) && ranges.iter().any(|&(x, y)| x < b && y > a)
                    }) =>
            {
                Some(flight.fence)
            }
            _ => None,
        })
        .max()
}

/// The last send of title bytes to card copies that the card has had once
/// the flights in `line` up to host fence value `upto` have run (0: none of
/// them sent any).
fn sends_through(line: &Line, upto: u64) -> u64 {
    line.line
        .iter()
        .filter_map(|item| match item {
            Late::Run(flight) if flight.fence <= upto => flight.card.as_ref().map(|c| c.sent_as),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// Whether the landing thread lands one flight at a time (with what
/// follows it in the line) rather than every flight the card has run
/// (`VICTORY_RELAY_LAND_ONE`: on with the slots of the fetch inside
/// flights in the card's memory, [`inflight_on_card`], unless `=0`; `=1`
/// on in any case).
///
/// A landing signals its flights' fences only once all of them are
/// merged; run n53 (`VICTORY_RELAY_FREE_RAN=1`) had the title waiting for
/// those fences, 121 ms a frame of its own time between submissions. With
/// the slots on the card a flight's pieces come down in a copy of their
/// own that waits for no later flight, so a flight's fences go out as soon
/// as its own pieces are merged. With the old fetch every landing waits
/// behind the flights sent since, and one flight a landing would multiply
/// those waits: not on unless asked.
fn land_one() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_RELAY_LAND_ONE").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => inflight_on_card(),
    })
}

/// Of the first `fences.len()` things of the line, all landable
/// ([`landable`]): how many make the first landing with [`land_one`],
/// the first flight and what follows it up to the next flight.
pub(crate) fn first_landing(fences: &[Option<u64>]) -> usize {
    match fences.iter().position(Option::is_some) {
        Some(first) => fences[first + 1..].iter().position(Option::is_some).map_or(fences.len(), |next| first + 1 + next),
        None => fences.len(),
    }
}

/// The thread that lands flights ([`relay`]): as soon as the card has run
/// the first flights of the line it takes them, with what follows them,
/// and does what they left to do ([`Host::relay_land`]); else it waits for
/// the card, or for a flight. It takes neither the submission lock nor
/// anything a caller of [`Host::settle`] may hold: resetting the landed
/// lists and looking at what queues hold are left to [`late_helper`].
mod fence_early;

fn relay_worker() {
    let Some(host) = host() else {
        return;
    };
    crate::framelog::keep_on_fast_cores("victory-relay");
    // SAFETY: a plain event of our own.
    let Ok(event) = (unsafe { CreateEventW(None, false, false, None) }) else {
        fail(format_args!("no event for the thread that lands submissions sent ahead (VICTORY_RELAY=1)"));
    };
    let mut fetch: Option<RelayFetch> = None;
    loop {
        // (Outside the line's lock.)
        let results = fence_early::differs();
        let next = {
            let mut line = line();
            loop {
                if line.line.is_empty() {
                    line = RELAY_MOVED.wait(line).unwrap_or_else(|e| e.into_inner());
                    continue;
                }
                let fences: Vec<Option<u64>> = line
                    .line
                    .iter()
                    .map(|l| match l {
                        Late::Run(flight) => Some(flight.fence),
                        _ => None,
                    })
                    .collect();
                // SAFETY: a plain query.
                let completed = unsafe { host.fence.GetCompletedValue() };
                let n = landable(&fences, completed);
                if n == 0 {
                    // The first thing is a flight the card has not run.
                    break Err(fences[0].unwrap_or(0));
                }
                let base = if land_one() { first_landing(&fences[..n]) } else { n };
                // Nothing that needs no fetch waits for one (`VICTORY_FENCE_EARLY`).
                let n = fence_early::cut(&line.line, n, base, results);
                line.settling = true;
                let mut items: Vec<Late> = line.line.drain(..n).collect();
                if crate::airlists::on() && relay_exact() {
                    // The card has run all of them: their descriptor
                    // slots need not wait for the landing (dropped by
                    // `land`, not here).
                    let mut released = std::mem::take(&mut line.released);
                    let (flights, lists) = give_back_ran(&mut items, u64::MAX, &mut released);
                    line.released = released;
                    line.lists = line.lists.saturating_sub(lists);
                    crate::airlists::add(crate::airlists::n::LAND_FLIGHTS, flights);
                    crate::airlists::add(crate::airlists::n::LAND_LISTS, lists as u64);
                    if lists > 0 {
                        LINE_LANDED.notify_all();
                    }
                }
                line.landing = items.iter().filter(|l| matches!(l, Late::Run(_))).count();
                line.in_hand = items
                    .iter()
                    .filter_map(|l| match l {
                        Late::Write(at, what, bytes) => Some((*at, *what, *bytes)),
                        _ => None,
                    })
                    .collect();
                break Ok(items);
            }
        };
        match next {
            Ok(items) => host.relay_land(items, &mut fetch),
            Err(target) => {
                let waiting = std::time::Instant::now();
                // SAFETY: the host's fence and this thread's event.
                let waited = unsafe {
                    host.fence.SetEventOnCompletion(target, event).map(|()| {
                        WaitForSingleObject(event, INFINITE);
                    })
                };
                if waited.is_err() {
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                RELAY_STATS[2].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
        }
    }
}

/// [`relay`]: what shaders wrote to the declared ranges `back` of copies
/// in the PC's memory goes into title memory. As [`hand_back`], each byte
/// once, the pieces that differ found on several threads; but the merge is
/// `Shadow::relay_merge`, which does not write the copy, and the ranges
/// are not done with here (each flight's are, afterwards).
fn relay_hand_back(back: &[Declared]) {
    let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
    for (shadow, declared) in back {
        if shadow.carded() {
            continue;
        }
        let mut spans: Vec<(usize, usize)> = Vec::new();
        join_ranges(&mut spans, declared.iter().copied());
        for &(from, to) in &spans {
            shadow.relay_compared(from, to - from);
        }
        work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
    }
    if work.is_empty() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    let scanning = std::time::Instant::now();
    // Looked through without a lock: a piece that changes meanwhile (a
    // later flight writing, the title's bytes going in) is found now or by
    // that flight's landing, and the merge looks again under the lock.
    let found = crate::shadow::find_all(&work);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
    let (mut now, mut title) = (Vec::new(), Vec::new());
    for ((shadow, from, to), found) in work.iter().zip(found) {
        let mut written = 0;
        for (at, n) in found {
            written += shadow.relay_merge(at, n, &mut now, &mut title);
        }
        copy_back(1, written as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::compared(*from, to - from, written);
        let _ = (from, to);
    }
}

/// For the frame-rate line, with [`late`]: what the line of flights did
/// per frame since the last one.
fn late_line(frames: f64) {
    if !late() {
        return;
    }
    #[cfg(feature = "diagnostics")]
    fill_check::line();
    fill_ahead::line(frames);
    crate::early::line(frames);
    let s: Vec<f64> = LATE_STATS.iter().map(|s| s.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    tracing::info!(
        "submissions sent ahead per frame: {:.1} in {:.1} settles ({:.1} flights each time), {:.1} ms waiting for the card in them; \
         settled for: a synchronous submission or present {:.1}, the card had run them {:.1}, a wait for a memory value {:.1}, \
         title memory changed under shader writes {:.1}, title memory read {:.1} (by a buffer copy {:.1}, a fill of upload memory {:.1}, \
         a texture {:.1}), a step of ours {:.1}, {} lists in the air {:.1}, \
         bottom-of-pipe writes (VICTORY_LATE_WAIT_WRITES=0) {:.1}, a blocking fence wait {:.1}; \
         {:.1} waits for memory values met by a flight's bottom-of-pipe write, {:.1} fence signals set behind flights",
        s[0],
        s[1],
        if s[1] > 0.0 { s[11] / s[1] } else { 0.0 },
        s[12] / 1000.0,
        s[WHY_SYNC],
        s[WHY_CARD_DONE],
        s[WHY_VALUE],
        s[WHY_STALE],
        s[WHY_READ],
        s[15],
        s[16],
        s[17],
        s[WHY_RUN],
        late_most(),
        s[WHY_MOST],
        s[WHY_WRITES],
        s[WHY_FENCE],
        s[13],
        s[14],
    );
    {
        let w: Vec<f64> = LATE_WAITED.iter().map(|w| w.swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames).collect();
        tracing::info!(
            "of the waiting for the card in settles, per frame, by what settled: a synchronous submission or present {:.1} ms, \
             the card had run them {:.1}, a wait for a memory value {:.1}, title memory changed under shader writes {:.1}, \
             title memory read {:.1}, a step of ours {:.1}, lists in the air {:.1}, bottom-of-pipe writes {:.1}, a blocking fence wait {:.1}",
            w[WHY_SYNC],
            w[WHY_CARD_DONE],
            w[WHY_VALUE],
            w[WHY_STALE],
            w[WHY_READ],
            w[WHY_RUN],
            w[WHY_MOST],
            w[WHY_WRITES],
            w[WHY_FENCE],
        );
    }
    {
        let e: Vec<f64> = READ_EARLY.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        tracing::info!(
            "submissions sent ahead per frame that read title memory a flight not landed yet writes besides its declared shader              writes (read early): {:.2} under its read-backs, {:.2} under its bottom-of-pipe writes, {:.2} of them right after              a relay title-read wait (VICTORY_RELAY=2)",
            e[0],
            e[1],
            e[2],
        );
    }
    if relay() {
        fence_early::line(frames);
        let r: Vec<f64> = RELAY_STATS.iter().map(|r| r.swap(0, Ordering::Relaxed) as f64 / frames).collect();
        let [again, noted] = [0, 1].map(|i| BESIDE_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        tracing::info!(
            "the relay per frame (VICTORY_RELAY={}): {:.1} flights landed in {:.1} landings by the thread that copies back, \
             beside the submitting thread; that thread waited {:.1} ms for the card with nothing to land and {:.1} ms for \
             fetches of card copies' pieces, and was {:.1} ms landing in all (the frame line's \"copying back\" is its time; \
             \"settles\" above are waits of other threads for it); pieces fetched on the copy queue beside later \
             submissions: {noted:.1}, pieces those fetched again at their own landing whatever the compare said: {again:.1} \
             (VICTORY_QUEUE2_REFETCH)",
            if relay_exact() { 2 } else { 1 },
            r[1],
            r[0],
            r[2] / 1000.0,
            r[3] / 1000.0,
            r[4] / 1000.0,
        );
        if inflight_fetch() {
            let i: Vec<f64> = INFLIGHT_STATS.iter().map(|i| i.swap(0, Ordering::Relaxed) as f64 / frames).collect();
            tracing::info!(
                "card copies' pieces fetched inside their flights per frame (VICTORY_CARD_FETCH_INFLIGHT={}): {:.1} pieces \
                 ({:.2} MB) merged with no fetch of their own; {:.1} landings fetched the old way (the wait above), {:.1} of \
                 them because a flight's pieces overflowed the slots; slots in the card's memory copied down on the landing \
                 thread's copy queue: {:.1}x {:.2} MB in {:.0} copies, {:.1} ms",
                inflight_mode(),
                i[0],
                i[1] / (1 << 20) as f64,
                i[3],
                i[2],
                i[4],
                i[5] / (1 << 20) as f64,
                i[7],
                i[6] / 1000.0,
            );
        }
    }
    if late_exact() {
        let [pages, runs, bytes] = crate::shadow::merged_stats().map(|n| n as f64 / frames);
        tracing::info!(
            "title bytes put in under flights' shader writes per frame (VICTORY_LATE_WAIT=2): {pages:.0} pages looked at, \
             {runs:.0} changed runs, {:.2} MB",
            bytes / (1 << 20) as f64
        );
    }
}

#[cfg(test)]
mod late_tests {
    use super::{continues, covers, join_ranges, touches};

    /// A read is spared a settle only where the compared pieces cover it
    /// whole: a gap anywhere, at the ends included, says no.
    #[test]
    fn pieces_cover_a_read_only_without_gaps() {
        assert!(covers(&mut [(0, 10), (10, 20)], 0, 20));
        assert!(covers(&mut [(10, 20), (0, 12)], 3, 18));
        assert!(covers(&mut [(0, 100)], 40, 60));
        assert!(!covers(&mut [(0, 10), (11, 20)], 0, 20));
        assert!(!covers(&mut [(1, 20)], 0, 20));
        assert!(!covers(&mut [(0, 19)], 0, 20));
        assert!(!covers(&mut [], 0, 1));
    }

    /// Copies join only when both ends continue and the whole does not
    /// overlap itself.
    #[test]
    fn copies_join_only_end_to_end_and_apart() {
        // (destination, source, bytes)
        assert!(continues((1000, 5000, 16), (1016, 5016, 32)));
        assert!(continues((5000, 1000, 16), (5016, 1016, 32)));
        // A gap, or only one side continues.
        assert!(!continues((1000, 5000, 16), (1020, 5016, 32)));
        assert!(!continues((1000, 5000, 16), (1016, 5020, 32)));
        assert!(!continues((1000, 5000, 16), (1000, 5000, 16)));
        // Together they would overlap: the second reads what the first wrote.
        assert!(!continues((1000, 1016, 16), (1016, 1032, 16)));
        assert!(!continues((1016, 1000, 16), (1032, 1016, 16)));
        // They touch without overlapping.
        assert!(continues((1000, 1032, 16), (1016, 1048, 16)));
        assert!(!continues((usize::MAX - 8, 0, 8), (0, 8, 8)));
    }

    /// Ranges are found by what they hold, not by where they start, and a
    /// range that only touches another's end does not touch it.
    #[test]
    fn ranges_touch_where_they_overlap() {
        let mut ranges = Vec::new();
        join_ranges(&mut ranges, [(40, 50), (10, 20), (20, 25), (45, 60), (7, 7)]);
        assert_eq!(ranges, [(10, 25), (40, 60)]);
        join_ranges(&mut ranges, [(25, 40)]);
        assert_eq!(ranges, [(10, 60)]);
        let ranges = [(10usize, 25usize), (40, 60), (100, 101)];
        for (from, to, hit) in [
            (0, 10, false),
            (0, 11, true),
            (24, 25, true),
            (25, 40, false),
            (25, 41, true),
            (59, 200, true),
            (60, 100, false),
            (100, 101, true),
            (101, 500, false),
            (30, 30, false),
            (50, 50, false),
        ] {
            assert_eq!(touches(&ranges, from, to), hit, "{from}..{to}");
        }
        assert!(!touches(&[], 0, usize::MAX));
    }
}

/// Whether a title buffer copy that continues the one recorded before it
/// in its list becomes one step with it (`VICTORY_JOIN_COPIES=1`; off by
/// default).
///
/// Each copy is done at submission through the kernel, which reports a
/// page that is not there instead of faulting (`copy_mapped`), tells the
/// write watch and the guards, and ends the refreshes gathered so far: a
/// call into the kernel for some 480 bytes, 14,042 times a frame in run 86
/// (6.8 MB; "moves 19 ms" with the refreshes' own reads). How many of them
/// continue the one before is counted either way and said on its own line
/// with the frame rate; the switch is worth that many kernel calls.
///
/// Joined, the same bytes move: [`continues`] joins only copies whose
/// ranges, taken together, do not overlap, and nothing is recorded between
/// two steps that follow each other. What differs: a page that is not
/// mapped now fails the joined copy as a whole (it then goes page by page,
/// zeros for what cannot be read, as a single copy does), where before
/// only the copy that held it did.
fn join_copies() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_JOIN_COPIES").is_some_and(|v| v == "1"))
}

/// Since the last frame-rate line: title buffer copies recorded, and those
/// that continue the one before ([`join_copies`]).
static COPIES_JOINED: [std::sync::atomic::AtomicU64; 2] = [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// Whether the copy `next` (destination, source, bytes) goes on exactly
/// where `last` ended, source and destination both, and the two as one do
/// not overlap (source over destination): then one copy of both moves the
/// bytes the two would move in turn.
fn continues(last: (usize, usize, usize), next: (usize, usize, usize)) -> bool {
    let ((d, s, n), (dst, src, len)) = (last, next);
    let Some(all) = n.checked_add(len) else {
        return false;
    };
    d.checked_add(n) == Some(dst)
        && s.checked_add(n) == Some(src)
        && (d.checked_add(all).is_some_and(|end| end <= s) || s.checked_add(all).is_some_and(|end| end <= d))
}

/// The ranges the shaders of `runs` may write, per host copy, as declared.
/// One list's lock at a time: a submission may name a list twice.
fn declared_writes(runs: &[&HostList]) -> Vec<Declared> {
    gather_declared(runs, |_| true)
}

/// The ranges the shaders of `runs` may write, of the host copies `keep`
/// takes: per copy in the order first declared, each range once in the
/// order first declared. One list's lock at a time.
///
/// The copies and their ranges were searched from the start for each of
/// the thousands of declarations a submission makes; they are looked up.
fn gather_declared(runs: &[&HostList], keep: impl Fn(&crate::shadow::Shadow) -> bool) -> Vec<Declared> {
    let mut out: Vec<Declared> = Vec::new();
    let mut at: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut seen: std::collections::HashSet<(usize, usize, usize)> = std::collections::HashSet::new();
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::GpuWrite(shadow, address, len) = step
                && keep(shadow)
            {
                let key = std::sync::Arc::as_ptr(shadow) as usize;
                let range = (*address, *address + *len);
                if !seen.insert((key, range.0, range.1)) {
                    continue;
                }
                match at.get(&key) {
                    Some(&i) => out[i].1.push(range),
                    None => {
                        at.insert(key, out.len());
                        out.push((shadow.clone(), vec![range]));
                    }
                }
            }
        }
    }
    out
}

/// The walls trace (`VICTORY_WALLS_TRACE=1` or `2`; off by default): log
/// lines that say what the title's CPU was given while it decided whether
/// the camera is in a room. The values of the small read-backs
/// ([`read_backs`]), the frame's main pass by scissor rectangle with its
/// culling dispatches (`draw.rs`, `walls_census`), the shader-written
/// buffers merged into title memory (`shadow.rs`, `walls_merged`), host
/// copies found stale after their refresh (`shadow.rs`; only with
/// `VICTORY_REFRESH_CHECK=1` as well: it costs about 100 ms a frame), and, in the
/// runtime, the read requests that fail (with 2: every request).
pub(crate) fn walls_trace() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WALLS_TRACE").is_some_and(|v| v == "1" || v == "2"))
}

/// After `list` has run: the texels its read-backs copied into host
/// buffers of ours go to the title's memory (`HostList::defer_read_back`).
///
/// A submission that is waited for did this itself. One sent ahead
/// ([`late`]) did not, at any stage: the flights' lists were settled
/// without it, so the title read a buffer nobody wrote, as before read-backs
/// were made (the exposure metering's two small float targets, every frame).
fn read_backs(list: &HostList) {
    for (buffer, at, [row, rows, pitch]) in &list.deferred().backs {
        let mut data = std::ptr::null_mut();
        // SAFETY: a read-back buffer of ours, written by the list that has
        // run; the title's destination holds every row (checked when the
        // copy was recorded).
        unsafe {
            if buffer.Map(0, None, Some(&mut data)).is_ok() && !data.is_null() {
                let from = data.cast::<u8>().cast_const();
                for line in 0..*rows {
                    std::ptr::copy_nonoverlapping(from.add(line * row.next_multiple_of(256)), (*at as *mut u8).add(line * pitch), *row);
                }
                if crate::results::trace() {
                    crate::results::landed(*at);
                }
                // The walls trace: what the title is handed (one row of at
                // most 24 values: its 5x1 and 1x1 float targets).
                if walls_trace() && *rows == 1 && *row <= 96 {
                    let mut values = [0f32; 24];
                    std::ptr::copy_nonoverlapping(from, values.as_mut_ptr().cast::<u8>(), *row & !3);
                    tracing::info!(
                        frame = crate::frame::current(),
                        to = format_args!("{:#x}", *at),
                        bytes = *row,
                        values = ?&values[..*row / 4],
                        "walls trace: a read-back lands in title memory"
                    );
                }
                buffer.Unmap(0, None);
            }
        }
    }
}

/// After the card ran what declared `back`: what shaders wrote there goes
/// into title memory, and the ranges are done with.
fn hand_back(back: &[Declared]) {
    if crate::shadow::wide_back() {
        // Every range first (the pieces that differ, found on several
        // threads), then the merges, in order, as below.
        let mut work: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (shadow, declared) in back {
            if shadow.carded() {
                continue;
            }
            let mut sorted = declared.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            work.extend(spans.into_iter().map(|(from, to)| (shadow.clone(), from, to)));
        }
        #[cfg(feature = "diagnostics")]
        let scanning = std::time::Instant::now();
        let found = crate::shadow::find_all(&work);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::BackScanUs, scanning.elapsed().as_micros() as u64);
        for ((shadow, from, to), found) in work.iter().zip(found) {
            let written = shadow.copy_back_with(*from, to - from, Some(found));
            copy_back(1, written as u64);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(*from, to - from, written);
        }
        for (shadow, declared) in back {
            shadow.copy_back_done(declared);
        }
        return;
    }
    for (shadow, declared) in back {
        if shadow.carded() {
            // Done before (`Host::card_back`).
            shadow.copy_back_done(declared);
            continue;
        }
        // Each byte compared once: the world declares dozens of views
        // of its 174 MB heap that overlap almost entirely, and comparing
        // every one whole was gigabytes a frame (0.3-0.5 s) for the few
        // MB shaders wrote. Only ranges that overlap or touch are
        // joined: what lies between two was not given to shaders (the
        // snapshot does not hold it).
        let mut sorted = declared.clone();
        sorted.sort_unstable();
        let mut spans: Vec<(usize, usize)> = Vec::new();
        for (from, to) in sorted {
            match spans.last_mut() {
                Some(last) if from <= last.1 => last.1 = last.1.max(to),
                _ => spans.push((from, to)),
            }
        }
        for (from, to) in spans {
            let written = shadow.copy_back(from, to - from);
            copy_back(1, written as u64);
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
        }
        shadow.copy_back_done(declared);
    }
}

impl Host {
    /// Whether the host runs mesh shaders at shader model 6.6 (translated
    /// mesh shaders need both); logged once.
    pub(crate) fn mesh_shaders(&self) -> bool {
        static MESH: OnceLock<bool> = OnceLock::new();
        *MESH.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS7::default();
            let mut model = D3D12_FEATURE_DATA_SHADER_MODEL {
                HighestShaderModel: D3D_SHADER_MODEL_6_6,
            };
            // SAFETY: each query writes its own structure of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS7,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
                    && self
                        .device
                        .CheckFeatureSupport(
                            D3D12_FEATURE_SHADER_MODEL,
                            (&raw mut model).cast(),
                            size_of_val(&model) as u32,
                        )
                        .is_ok()
            };
            let mesh = ok
                && options.MeshShaderTier != D3D12_MESH_SHADER_TIER_NOT_SUPPORTED
                && model.HighestShaderModel.0 >= D3D_SHADER_MODEL_6_6.0;
            tracing::info!(mesh, "host mesh shader support");
            mesh
        })
    }

    /// Whether the host GPU supports the depth bounds test; logged once.
    pub(crate) fn depth_bounds(&self) -> bool {
        static BOUNDS: OnceLock<bool> = OnceLock::new();
        *BOUNDS.get_or_init(|| {
            let mut options = D3D12_FEATURE_DATA_D3D12_OPTIONS2::default();
            // SAFETY: writes `options` of the given size.
            let ok = unsafe {
                self.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS2,
                        (&raw mut options).cast(),
                        size_of_val(&options) as u32,
                    )
                    .is_ok()
            };
            let supported = ok && options.DepthBoundsTestSupported.as_bool();
            tracing::info!(supported, "host depth bounds support");
            supported
        })
    }

    /// A new list, open for recording.
    pub(crate) fn new_list(&self) -> windows::core::Result<HostList> {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let before = private_bytes();
        let list = self.make_list();
        if let Ok(list) = &list {
            list.prof_start(self);
        }
        LISTS[0].fetch_add(1, Ordering::Relaxed);
        // (May shrink while a list is made, on another thread's account:
        // added as it is, the sum is what counts.)
        #[cfg(feature = "diagnostics")]
        LISTS[2].fetch_add(private_bytes().wrapping_sub(before), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        {
            use crate::copystat::{C, add};
            add(C::HostLists, 1);
            add(C::HostListNs, started.elapsed().as_nanos() as u64);
        }
        list
    }

    /// An allocator and a closed list for a prelude.
    fn make_prelude(&self) -> Option<Prelude> {
        // SAFETY: plain object creation.
        let made: windows::core::Result<Prelude> = unsafe {
            (|| {
                let allocator: ID3D12CommandAllocator =
                    self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
                let list: ID3D12GraphicsCommandList =
                    self.device
                        .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)?;
                list.Close()?;
                Ok((allocator, list))
            })()
        };
        LISTS[1].fetch_add(1, Ordering::Relaxed);
        match made {
            Ok(prelude) => Some(prelude),
            Err(e) => {
                tracing::warn!(error = %e, "no host list for a prelude");
                None
            }
        }
    }

    fn make_list(&self) -> windows::core::Result<HostList> {
        // SAFETY: plain object creation.
        unsafe {
            let allocator: ID3D12CommandAllocator = self
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list: ID3D12GraphicsCommandList = self.device.CreateCommandList(
                0,
                D3D12_COMMAND_LIST_TYPE_DIRECT,
                &allocator,
                None,
            )?;
            let prelude = OnceLock::new();
            if !lazy_prelude() {
                let _ = prelude.set(self.make_prelude());
            }
            Ok(HostList {
                allocator,
                list,
                deferred: Mutex::default(),
                scratch: Mutex::default(),
                states: Mutex::default(),
                twins: Mutex::default(),
                prelude,
                card_ring: Mutex::default(),
                card_pre: OnceLock::new(),
                pipelines: Mutex::default(),
                budget: OnceLock::new(),
                card_budget: OnceLock::new(),
                indirect_args: Mutex::default(),
                batch_args: Mutex::default(),
                window_check: OnceLock::new(),
                _alive: Counted::new(),
                prof: Mutex::default(),
                bar: crate::barstat::Track::new(),
                meter: AtomicU32::new(u32::MAX),
                pipeline_meter: AtomicU32::new(u32::MAX),
                #[cfg(feature = "diagnostics")]
                notes: Mutex::default(),
                #[cfg(feature = "diagnostics")]
                stamps: OnceLock::new(),
            })
        }
    }

    /// The card's clock ([`CARD_CLOCK_US`]), made at the first send.
    fn card_clock(&self) -> Option<&'static CardClock> {
        static CLOCK: OnceLock<Option<CardClock>> = OnceLock::new();
        static WATCH: std::sync::Once = std::sync::Once::new();
        let clock = CLOCK
            .get_or_init(|| {
                // SAFETY: a plain fence on our device.
                let fence = unsafe { self.device.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) }.ok()?;
                Some(CardClock {
                    fence,
                    state: Mutex::new((0, None)),
                    sent: std::sync::Condvar::new(),
                })
            })
            .as_ref()?;
        WATCH.call_once(|| {
            if let Err(e) = std::thread::Builder::new().name("victory-card-clock".into()).spawn(move || card_clock_watch(clock)) {
                tracing::warn!(error = %e, "no thread for the card's clock: the frame line's idle times stay 0");
            }
        });
        Some(clock)
    }

    /// Lists of ours just went to the queue: the clock's signal goes
    /// behind them, and an idle stretch of the card, if one was under way,
    /// ends here ([`CARD_CLOCK_US`]).
    fn card_sent(&self) {
        let Some(clock) = self.card_clock() else {
            return;
        };
        let mut state = clock.state.lock().unwrap_or_else(|e| e.into_inner());
        state.0 += 1;
        // SAFETY: our queue and the clock's fence.
        let _ = unsafe { self.queue.Signal(&clock.fence, state.0) };
        if let Some((since, worked_then, held_then)) = state.1.take() {
            let idle = since.elapsed().as_micros() as u64;
            let work = worked().saturating_sub(worked_then).min(idle);
            let held = held_so_far().saturating_sub(held_then).min(idle);
            card_clock_us(0, idle);
            card_clock_us(1, work);
            CARD_CLOCK_US[3].fetch_add(held.saturating_sub(work), Ordering::Relaxed);
        }
        drop(state);
        clock.sent.notify_one();
    }

    /// Whether the host has what copies in the card's memory need (three
    /// lists and the compare); made at the first ask. A copy goes to the
    /// card's memory only if it has.
    pub(crate) fn card_ready(&self) -> bool {
        static READY: OnceLock<bool> = OnceLock::new();
        *READY.get_or_init(|| match self.make_card_work() {
            Ok(work) => {
                *CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()) = Some(work);
                true
            }
            Err(e) => {
                tracing::warn!(error = %e, "no host copies in the card's memory: the compare could not be made");
                false
            }
        })
    }

    fn make_card_work(&self) -> windows::core::Result<CardWork> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: the source and names live through the call; the out
        // pointer is ours.
        unsafe {
            D3DCompile(
                CARD_COMPARE.as_ptr().cast(),
                CARD_COMPARE.len(),
                windows::core::s!("card compare"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
        }
        let code = code.ok_or_else(windows::core::Error::empty)?;
        let list = || self.make_prelude().ok_or_else(windows::core::Error::empty);
        // A megabyte of results: 262,144 pieces, 16 GB of declared ranges
        // in one submission.
        const RESULTS: u64 = 1 << 20;
        let (differs, data) = self.shared_buffer(RESULTS).ok_or_else(windows::core::Error::empty)?;
        name(&differs, || "card compare results".into());
        // SAFETY: the blob's own bytes; the root signature is inside the
        // shader.
        unsafe {
            let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
            let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
            let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(CardWork {
                sync: vec![list()?],
                check: vec![list()?],
                fetch: list()?,
                root,
                pso,
                differs: (differs, data as usize, RESULTS),
                fetched: None,
                taken: 0,
                out: 0,
                stage_wait: None,
                inflight: if inflight_fetch() { self.make_card_inflight(RESULTS) } else { None },
            })
        }
    }

    /// [`CardInflight`] for `results` bytes of the compare's results; `None`
    /// (and the old fetch) if the shader or the memory cannot be had.
    fn make_card_inflight(&self, results: u64) -> Option<CardInflight> {
        use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
        use windows::Win32::Graphics::Direct3D::ID3DBlob;
        let made = (|| -> windows::core::Result<CardInflight> {
            let mut code: Option<ID3DBlob> = None;
            // SAFETY: the source and names live through the call; the out
            // pointer is ours.
            unsafe {
                D3DCompile(
                    CARD_FETCH.as_ptr().cast(),
                    CARD_FETCH.len(),
                    windows::core::s!("card fetch in flight"),
                    None,
                    None,
                    windows::core::s!("main"),
                    windows::core::s!("cs_5_1"),
                    0,
                    0,
                    &mut code,
                    None,
                )?;
            }
            let code = code.ok_or_else(windows::core::Error::empty)?;
            let slots = inflight_slots(inflight_fetch_bytes());
            let (table, table_data) = self.shared_buffer(4 + results).ok_or_else(windows::core::Error::empty)?;
            name(&table, || "card fetch in flight: tickets".into());
            let fetched_bytes = u64::from(slots) * INFLIGHT_SLOT;
            let (fetched, fetched_data) = if inflight_on_card() {
                (self.uav_buffer(fetched_bytes).ok_or_else(windows::core::Error::empty)?, std::ptr::null_mut())
            } else {
                self.shared_buffer(fetched_bytes).ok_or_else(windows::core::Error::empty)?
            };
            name(&fetched, || "card fetch in flight: pieces".into());
            // SAFETY: the mapping holds the counter; nothing of ours ran
            // on the card with it yet.
            unsafe { table_data.cast::<u32>().write_volatile(0) };
            // SAFETY: the blob's own bytes; the root signature is inside
            // the shader.
            unsafe {
                let bytes = std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
                let root: ID3D12RootSignature = self.device.CreateRootSignature(0, bytes)?;
                let pso: ID3D12PipelineState = self.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: bytes.as_ptr().cast(),
                        BytecodeLength: bytes.len(),
                    },
                    ..Default::default()
                })?;
                Ok(CardInflight {
                    root,
                    pso,
                    table: (table, table_data as usize, 4 + results),
                    fetched: (fetched, fetched_data as usize, slots),
                    landed: 0,
                    brought: Vec::new(),
                })
            }
        })();
        match made {
            Ok(inflight) => {
                tracing::info!(
                    slots = inflight.fetched.2,
                    on_card = inflight_on_card(),
                    "card copies' flagged pieces come down inside their flights (VICTORY_CARD_FETCH_INFLIGHT={}{})",
                    inflight_mode(),
                    if inflight_on_card() { ": slots in the card's memory, copied down at the landing on a copy queue" } else { "" }
                );
                Some(inflight)
            }
            Err(e) => {
                tracing::warn!(error = %e, "no fetch inside flights (VICTORY_CARD_FETCH_INFLIGHT=1): pieces are fetched the old way");
                None
            }
        }
    }

    /// Before a submission's lists run, for copies in the card's memory
    /// ([`crate::shadow::cards`]): what changed on their CPU side since
    /// the last submission goes to their card side (a list of ours, run
    /// now; the submission's lists follow it on the queue), and the ranges
    /// this submission's shaders may write get a compare to run after its
    /// lists. `None` if there is nothing to do.
    ///
    /// `flown`: the submission is sent ahead with [`relay`]. Its compare
    /// then stores the hashes it makes (see [`flagged_pieces`] and
    /// `relay`).
    ///
    /// `sent`: the last value signalled on the host's fence; the caller
    /// signals the next one behind this submission's lists.
    ///
    /// With `VICTORY_QUEUE2` bit 2 (`crate::queue2`; off by default) the
    /// changed bytes cross from the PC's memory to the card's on a copy
    /// queue, on the card's copy engine, while the direct queue still runs
    /// what was sent before: they go into a landing buffer in the card's
    /// memory that nothing else uses. The direct queue is told to wait for
    /// that copy and then moves the bytes from the landing buffer into the
    /// copies, inside the card's memory, in the same place of its order as
    /// before. So the copies change exactly when they did on one queue; what
    /// leaves the direct queue is the crossing from the PC's memory.
    fn card_before(&self, runs: &[&HostList], flown: bool, sent: u64) -> Option<CardPlan> {
        use std::sync::Arc;
        if !crate::shadow::cards() {
            return None;
        }
        let gathering = std::time::Instant::now();
        // The ranges this submission's shaders may write, per copy.
        let declared = gather_declared(runs, |shadow| shadow.carded());
        prep_add(prep::CARD_GATHER, gathering);
        let taking = std::time::Instant::now();
        // With the relay: the order of the queue is held from here until
        // what is taken is on the queue, and the taking is numbered.
        let mut order = relay().then(|| QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        // The wait for the queue's order (the relay's thread holds it while
        // it sends) is a wait, not the taking.
        let ordered = taking.elapsed();
        PREP_SPLIT[prep::CARD_WAITS].fetch_add(ordered.as_nanos() as u64, Ordering::Relaxed);
        let taking = std::time::Instant::now();
        let sent_as = match order.as_mut() {
            Some(order) => {
                **order += 1;
                **order
            }
            None => 0,
        };
        let mut dirty = crate::shadow::card_dirty(sent_as);
        // With fetches on a copy queue (`crate::queue2`, bit 1 of
        // `VICTORY_QUEUE2`): the send's number is signalled on the direct
        // queue behind the send, so that a fetch knows which sends the card
        // has finished (`Second::order_done`).
        let numbered = || {
            if sent_as != 0
                && let Some(second) = crate::queue2::second().filter(|s| s.fetches())
            {
                second.sent_up(&self.queue, sent_as);
            }
        };
        if declared.is_empty() && dirty.is_empty() {
            prep_add(prep::CARD_DIRTY, taking);
            numbered();
            return None;
        }
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let work = work.as_mut()?;
        // A copy whose shader writes are declared for the first time gets
        // its bytes from before, whole.
        let mut whole: Vec<Arc<crate::shadow::Shadow>> = Vec::new();
        for (shadow, _) in &declared {
            if shadow.card_make_before(|size| {
                let before = self.uav_buffer(crate::shadow::card_hash_bytes(size))?;
                name(&before, || "title buffer on the card, hashes of its pieces".into());
                Some(before)
            }) {
                whole.push(shadow.clone());
                if !dirty.iter().any(|(s, _)| Arc::ptr_eq(s, shadow)) {
                    dirty.push((shadow.clone(), Vec::new()));
                }
            }
        }
        prep_add(prep::CARD_DIRTY, taking);
        let sending = std::time::Instant::now();
        // Of the sends, the fills and the waits (nanoseconds), counted on
        // their own.
        let sends_apart = std::cell::Cell::new(0u64);
        // Of the sends: the hash passes' recording, the list's closing and
        // calls to the queue (and how many), nanoseconds.
        let (hash_ns, submit_ns, submit_n) =
            (std::cell::Cell::new(0u64), std::cell::Cell::new(0u64), std::cell::Cell::new(0u64));
        let mut used: Vec<(bool, Prelude)> = Vec::new();
        let mut stage = Stager::default();
        let mut joined: Option<ID3D12CommandList> = None;
        // With [`card_sync_join`]: the buffers the list so far left out of
        // the common state, and the state each is in (one entry each).
        let touched: std::cell::RefCell<Vec<(ID3D12Resource, D3D12_RESOURCE_STATES)>> = std::cell::RefCell::default();
        // The changed bytes through the copy queue (`VICTORY_QUEUE2` bit
        // 2): the landing buffer they are in, in the order of `dirty`'s
        // ranges, once the direct queue has waited. `None`: straight from
        // the PC's memory on the direct queue, as before.
        let mut landed: Option<ID3D12Resource> = None;
        // Not with a copy that has no PC side (a mirror,
        // `VICTORY_CARD_NO_PC_COPY`): its bytes go up through upload chunks.
        let all_staged = dirty.iter().filter(|(shadow, _)| shadow.carded()).all(|(shadow, _)| shadow.card_sides().is_some_and(|s| s.0.is_some()));
        if all_staged && let Some(second) = crate::queue2::second().filter(|s| s.uploads()) {
            let bytes: u64 = dirty
                .iter()
                .filter(|(shadow, _)| shadow.carded())
                .flat_map(|(_, ranges)| ranges.iter().map(|&(a, b)| (b - a) as u64))
                .sum();
            // SAFETY: a plain query.
            let completed = unsafe { self.fence.GetCompletedValue() };
            // Held until the card has passed this submission's signal: the
            // direct queue's copy out of it runs ahead of that.
            if bytes > 0
                && let Some(landing) = second.landing(bytes, completed, sent + 1)
            {
                // SAFETY: an open copy list of the lane's; live buffers,
                // and the ranges lie in them; the host's queue.
                let sent_up = unsafe {
                    (|| -> windows::core::Result<()> {
                        let copy = second.up.list()?;
                        let mut at = 0u64;
                        for (shadow, ranges) in &dirty {
                            let Some((Some(staging), _, _)) = shadow.card_sides() else {
                                continue;
                            };
                            for &(a, b) in ranges {
                                copy.1.CopyBufferRegion(&landing, at, staging, a as u64, (b - a) as u64);
                                at += (b - a) as u64;
                            }
                        }
                        let job = second.up.send(copy, &[], bytes)?;
                        // Nothing sent to the direct queue from here on
                        // runs before the bytes have landed.
                        second.hold(&self.queue, job)
                    })()
                };
                if let Err(e) = sent_up {
                    fail(format_args!(
                        "host copies in the card's memory could not be brought up to date through the copy queue ({e}, VICTORY_QUEUE2)"
                    ));
                }
                landed = Some(landing);
            }
        }
        if !dirty.is_empty() {
            let Some(sync) = work.sync.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list to bring host copies in the card's memory up to date"));
            };
            let (allocator, list) = &sync;
            // With mirrors: past `stage_most` bytes the list so far goes to
            // the card and is waited for, and its chunks are used again.
            let stage_wait = &mut work.stage_wait;
            let sends_apart = &sends_apart;
            let (submit_ns, submit_n, touched) = (&submit_ns, &submit_n, &touched);
            let mut stage_flush = |stage: &mut Stager| -> windows::core::Result<()> {
                let filling = std::time::Instant::now();
                stage.fill();
                let filled = filling.elapsed().as_nanos() as u64;
                PREP_SPLIT[prep::CARD_FILLS].fetch_add(filled, Ordering::Relaxed);
                sends_apart.set(sends_apart.get() + filled);
                // The call's end puts every buffer back in the common state.
                touched.borrow_mut().clear();
                // SAFETY: our list, closed and run; the fence and the event
                // are ours, under the submission lock.
                unsafe {
                    let submitting = std::time::Instant::now();
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    submit_ns.set(submit_ns.get() + submitting.elapsed().as_nanos() as u64);
                    submit_n.set(submit_n.get() + 1);
                    if stage_wait.is_none() {
                        let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE)?;
                        let event = CreateEventW(None, false, false, None)?;
                        *stage_wait = Some((fence, event, 0));
                    }
                    let Some((fence, event, value)) = stage_wait.as_mut() else {
                        return Err(windows::core::Error::empty());
                    };
                    *value += 1;
                    let waiting = std::time::Instant::now();
                    self.queue.Signal(&*fence, *value)?;
                    fence.SetEventOnCompletion(*value, *event)?;
                    let was_working = working(false);
                    WaitForSingleObject(*event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    let waited = waiting.elapsed().as_nanos() as u64;
                    PREP_SPLIT[prep::CARD_WAITS].fetch_add(waited, Ordering::Relaxed);
                    sends_apart.set(sends_apart.get() + waited);
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                }
                CARD_STAGE[2].fetch_add(1, Ordering::Relaxed);
                stage.rewind();
                Ok(())
            };
            // SAFETY: the list's last run finished (it ran ahead of a
            // submission that was waited for, and came back after that);
            // live buffers, and the ranges lie in them.
            let join = card_sync_join();
            // A buffer's state after the list so far, for the barriers back
            // to the common state (with `join`).
            let note = |resource: &ID3D12Resource, state: D3D12_RESOURCE_STATES| {
                if !join {
                    return;
                }
                let mut touched = touched.borrow_mut();
                match touched.iter_mut().find(|(r, _)| r == resource) {
                    Some(entry) => entry.1 = state,
                    None => touched.push((resource.clone(), state)),
                }
            };
            let recorded = unsafe {
                (|| -> windows::core::Result<Option<ID3D12CommandList>> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Sync);
                    // Where the next range lies in the landing buffer.
                    let mut at = 0u64;
                    for (shadow, ranges) in &dirty {
                        let Some((staging, before, size)) = shadow.card_sides() else {
                            continue;
                        };
                        // The segment of the list the copy's last send is in.
                        let mut last_send = None;
                        if let Some(staging) = staging {
                            for &(a, b) in ranges {
                                // Upstream's `VICTORY_QUEUE2` bit 2: out of
                                // the landing buffer the copy queue filled.
                                match &landed {
                                    Some(landing) => list.CopyBufferRegion(&shadow.buffer, a as u64, landing, at, (b - a) as u64),
                                    None => list.CopyBufferRegion(&shadow.buffer, a as u64, staging, a as u64, (b - a) as u64),
                                }
                                at += (b - a) as u64;
                                crate::shadow::CARD_STATS[0].fetch_add((b - a) as u64, Ordering::Relaxed);
                                last_send = Some(stage.segments);
                            }
                        } else {
                            // A mirror (`crate::shadow::no_pc_copy`): the bytes
                            // go up through upload chunks, from the mirror
                            // where it holds them and from title memory
                            // elsewhere, moved there before the list runs.
                            for &(a, b) in ranges {
                                for (lo, hi, src, title) in shadow.card_sources(a, b).unwrap_or_default() {
                                    let mut at = lo;
                                    while at < hi {
                                        if stage.segment >= stage_most() {
                                            stage_flush(&mut stage)?;
                                        }
                                        let (chunk, offset, n) = match stage.take(self, (hi - at) as u64) {
                                            Some(room) => room,
                                            None if stage.segment > 0 => {
                                                stage_flush(&mut stage)?;
                                                continue;
                                            }
                                            None => fail(format_args!(
                                                "no upload memory to bring a host copy in the card's memory up to date (VICTORY_CARD_NO_PC_COPY=1)"
                                            )),
                                        };
                                        let (buffer, data) = (&stage.chunks[chunk].0._buffer, stage.chunks[chunk].0.data);
                                        list.CopyBufferRegion(&shadow.buffer, at as u64, buffer, offset, n);
                                        stage.fills.push((data + offset as usize, src + (at - lo), n as usize, title));
                                        last_send = Some(stage.segments);
                                        note(&shadow.buffer, D3D12_RESOURCE_STATE_COPY_DEST);
                                        at += n as usize;
                                    }
                                }
                                crate::shadow::CARD_STATS[0].fetch_add((b - a) as u64, Ordering::Relaxed);
                            }
                        }
                        crate::shadow::CARD_STATS[1].fetch_add(ranges.len() as u64, Ordering::Relaxed);
                        let Some(before) = before else {
                            continue;
                        };
                        let hashing = std::time::Instant::now();
                        // No twin (it cost the card the copy's size again,
                        // 399 MB in run 75): `before` holds a 128-bit hash
                        // of each 64 KiB piece. The pieces just sent up are
                        // hashed again, all of them the first time; the
                        // compare after the lists hashes the declared
                        // pieces and flags those whose hash is another.
                        // Sent up in an earlier segment of the list, which
                        // ran and was waited for: back in the common state,
                        // which the hashing's read promotes from.
                        if last_send == Some(stage.segments) {
                            crate::barstat::record(&list, crate::barstat::Why::Submission, &[transition(
                                &shadow.buffer,
                                D3D12_RESOURCE_STATE_COPY_DEST,
                                D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                            )]);
                        }
                        list.SetComputeRootSignature(&work.root);
                        list.SetPipelineState(&work.pso);
                        list.SetComputeRootShaderResourceView(1, shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                        let runs: Vec<crate::shadow::CardSpan> = if whole.iter().any(|s| Arc::ptr_eq(s, shadow)) {
                            vec![shadow.card_run(0, usize::MAX)]
                        } else {
                            ranges.iter().map(|&(a, b)| shadow.card_run(a, b)).collect()
                        };
                        // The same pieces in fewer passes (`rehash_runs`).
                        let runs = if rehash_runs() { crate::shadow::joined_piece_runs(runs) } else { runs };
                        CARD_PASSES[0].fetch_add(runs.len() as u64, Ordering::Relaxed);
                        CARD_PASSES[1].fetch_add(runs.iter().map(|r| u64::from(r.pieces)).sum::<u64>(), Ordering::Relaxed);
                        for run in runs {
                            let constants = [run.first, run.skew, size as u32, run.pieces, 0, 0];
                            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                            list.Dispatch(run.pieces.div_ceil(64), 1, 1);
                        }
                        // Read by the hashing (after the barrier above, or
                        // promoted from the common state); its hashes and
                        // the results written.
                        note(&shadow.buffer, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
                        note(&before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                        note(&work.differs.0, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                        hash_ns.set(hash_ns.get() + hashing.elapsed().as_nanos() as u64);
                    }
                    let filling = std::time::Instant::now();
                    stage.fill();
                    let filled = filling.elapsed().as_nanos() as u64;
                    PREP_SPLIT[prep::CARD_FILLS].fetch_add(filled, Ordering::Relaxed);
                    sends_apart.set(sends_apart.get() + filled);
                    let submitting = std::time::Instant::now();
                    if join {
                        // What the call's end did: every buffer back in the
                        // common state, its writes seen by what follows.
                        let barriers: Vec<D3D12_RESOURCE_BARRIER> = touched
                            .borrow()
                            .iter()
                            .map(|(r, state)| transition(r, *state, D3D12_RESOURCE_STATE_COMMON))
                            .collect();
                        if !barriers.is_empty() {
                            crate::barstat::record(&list, crate::barstat::Why::Submission, &barriers);
                        }
                        drop(barriers);
                        list.Close()?;
                        submit_ns.set(submit_ns.get() + submitting.elapsed().as_nanos() as u64);
                        return Ok(Some(list.cast()?));
                    }
                    crate::cardmeter::end(list, timed);
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    submit_ns.set(submit_ns.get() + submitting.elapsed().as_nanos() as u64);
                    submit_n.set(submit_n.get() + 1);
                    Ok(None)
                })()
            };
            match recorded {
                Ok(list) => joined = list,
                Err(e) => fail(format_args!("host copies in the card's memory could not be brought up to date ({e})")),
            }
            used.push((false, sync));
        }
        numbered();
        drop(order);
        let sends = (sending.elapsed().as_nanos() as u64).saturating_sub(sends_apart.get());
        PREP_SPLIT[prep::CARD_SENDS].fetch_add(sends, Ordering::Relaxed);
        more_add(more::SEND_HASH, hash_ns.get());
        more_add(more::SEND_SUBMIT, submit_ns.get());
        more_add(more::SEND_SUBMIT_N, submit_n.get());
        more_add(more::SEND_COPIES, sends.saturating_sub(hash_ns.get() + submit_ns.get()));
        let comparing = std::time::Instant::now();
        let mut plan = CardPlan {
            ranges: Vec::new(),
            check: None,
            _synced: dirty.into_iter().map(|(shadow, _)| shadow).collect(),
            lists: used,
            stages: stage.keep(),
            sync: joined,
            sent_as,
            inflight: false,
            inflight_landed: 0,
        };
        // Each byte once, as the copy-back joins the declared ranges. The
        // results go behind those of plans still in the air.
        let mut at = work.taken;
        for (shadow, ranges) in &declared {
            let mut sorted = ranges.clone();
            sorted.sort_unstable();
            let mut spans: Vec<(usize, usize)> = Vec::new();
            for (from, to) in sorted {
                match spans.last_mut() {
                    Some(last) if from <= last.1 => last.1 = last.1.max(to),
                    _ => spans.push((from, to)),
                }
            }
            let twin = shadow.card_sides().is_some_and(|(_, before, _)| before.is_some());
            for (from, to) in spans {
                let Some(span) = shadow.card_span(from, to) else {
                    continue;
                };
                let compared = twin && u64::from(at + span.pieces) * 4 <= work.differs.2;
                plan.ranges.push(CardRange {
                    shadow: shadow.clone(),
                    declared: (from, to),
                    span,
                    at,
                    compared,
                });
                if compared {
                    at += span.pieces;
                }
            }
        }
        work.taken = at;
        if plan.ranges.iter().any(|r| r.compared) {
            let Some(check) = work.check.pop().or_else(|| self.make_prelude()) else {
                fail(format_args!("no host list for the compare of host copies in the card's memory"));
            };
            let (allocator, list) = &check;
            // SAFETY: the list's last run finished (its submission was
            // waited for, and it came back after that); live buffers; the
            // compare's own root signature.
            let recorded = unsafe {
                (|| -> windows::core::Result<ID3D12CommandList> {
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    let timed = crate::cardmeter::begin(list, crate::cardmeter::Kind::Check);
                    // What the submission's shaders wrote, first.
                    crate::barstat::record(&list, crate::barstat::Why::Submission, &[D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                    }]);
                    list.SetComputeRootSignature(&work.root);
                    list.SetPipelineState(&work.pso);
                    list.SetComputeRootUnorderedAccessView(3, work.differs.0.GetGPUVirtualAddress());
                    for r in plan.ranges.iter().filter(|r| r.compared) {
                        let Some((_, Some(before), size)) = r.shadow.card_sides() else {
                            continue;
                        };
                        // Mode 2 keeps the hash it makes: with the relay, and when
                        // merged pieces are not all sent up again (`merge_exact`).
                        let mode = if flown || crate::shadow::merge_exact() { 2 } else { 1 };
                        let constants = [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, mode];
                        list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                        list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, before.GetGPUVirtualAddress());
                        list.Dispatch(r.span.pieces.div_ceil(64), 1, 1);
                        CARD_PASSES[2].fetch_add(1, Ordering::Relaxed);
                    }
                    // The fetch inside the flight (`inflight_fetch`): behind
                    // every compare of the list, the pieces they flagged go
                    // to slots in the PC's memory.
                    if flown && let Some(inflight) = &work.inflight {
                        crate::barstat::record(&list, crate::barstat::Why::Submission, &[D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                            Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                    pResource: std::mem::ManuallyDrop::new(None),
                                }),
                            },
                        }]);
                        list.SetComputeRootSignature(&inflight.root);
                        list.SetPipelineState(&inflight.pso);
                        list.SetComputeRootUnorderedAccessView(2, work.differs.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(3, inflight.table.0.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(4, inflight.fetched.0.GetGPUVirtualAddress());
                        let mut every = true;
                        for r in plan.ranges.iter().filter(|r| r.compared) {
                            let Some((_, Some(_), size)) = r.shadow.card_sides() else {
                                every = false;
                                continue;
                            };
                            let groups = r.span.pieces.clamp(1, 65535);
                            let constants =
                                [r.span.first, r.span.skew, size as u32, r.span.pieces, r.at * 4, inflight.landed, inflight.fetched.2, groups];
                            list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                            list.SetComputeRootShaderResourceView(1, r.shadow.buffer.GetGPUVirtualAddress());
                            list.Dispatch(groups, 1, 1);
                        }
                        plan.inflight = every;
                        plan.inflight_landed = inflight.landed;
                    }
                    crate::cardmeter::end(list, timed);
                    list.Close()?;
                    list.cast()
                })()
            };
            match recorded {
                Ok(list) => plan.check = Some(list),
                Err(e) => fail(format_args!("the compare of host copies in the card's memory could not be recorded ({e})")),
            }
            plan.lists.push((true, check));
        }
        work.out += 1;
        prep_add(prep::CARD_COMPARE, comparing);
        Some(plan)
    }

    /// With [`relay`]: whether the compare's result area is half taken by
    /// plans in the air. It starts from the beginning only when no plan is
    /// out, so the caller lets the flights land first.
    fn card_results_low(&self) -> bool {
        CARD_WORK
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .as_ref()
            .is_some_and(|work| u64::from(work.taken) * 8 > work.differs.2)
    }

    /// The plan's submission has run and its results were read
    /// ([`Host::card_back`]): its lists are free for the next plan. Called
    /// when nothing of ours is on the card, so the compare's results start
    /// from the beginning again.
    fn card_done(&self, plan: CardPlan) {
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        for (check, list) in plan.lists {
            if check { work.check.push(list) } else { work.sync.push(list) }
        }
        if !plan.stages.is_empty() {
            CARD_STAGE[0].fetch_sub(plan.stages.len() as u64, Ordering::Relaxed);
            CARD_STAGE[1].fetch_sub(plan.stages.iter().map(|c| c.1).sum::<u64>(), Ordering::Relaxed);
            give_chunks(plan.stages);
        }
        work.out = work.out.saturating_sub(1);
        // With the relay other plans may be on the card, their results
        // not read yet: the area starts again only when none is out.
        if !relay() || work.out == 0 {
            work.taken = 0;
        }
    }

    /// The relay's own list, fence, event for [`Host::relay_card_back`].
    fn make_relay_fetch(&self) -> Option<RelayFetch> {
        let list = self.make_prelude()?;
        // SAFETY: plain objects of our own.
        unsafe {
            let fence: ID3D12Fence = self.device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
            let event = CreateEventW(None, false, false, None).ok()?;
            Some(RelayFetch {
                list,
                fence,
                event,
                sent: 0,
                buffer: None,
                copy: None,
                mark: 0,
            })
        }
    }

    /// [`relay`], on the thread that lands flights: for the plans of the
    /// flights being landed (the card has run them), the pieces their
    /// compares flagged are fetched, once each, with a list and a wait of
    /// this thread's own, and what shaders wrote in them goes into title
    /// memory (`Shadow::card_merge_relay`). As [`Host::card_back`], but:
    /// no submission lock; a piece flagged under one range of a plan is
    /// fetched under all ([`flagged_pieces`]); the fetch runs behind
    /// whatever was sent meanwhile, so it may bring a later flight's
    /// bytes, which that flight's landing then finds merged already; and
    /// nothing is sent back up.
    ///
    /// With [`inflight_fetch`] the pieces are first looked for where the
    /// flights' own compare lists put them ([`Host::relay_card_inflight`]);
    /// only a landing that cannot take them from there fetches, as here.
    /// `behind`: the value signalled on the host's fence behind the last of
    /// the flights. With `VICTORY_QUEUE2` bit 1 (`crate::queue2`) the fetch
    /// goes to a copy queue, behind that value and nothing else: it does
    /// not stand behind what the submitting thread sent meanwhile (run
    /// 118: 114-157 ms a frame of this thread's time). It then reads the
    /// card's side while later flights may be writing it: a byte is the
    /// old one or the new one, and the flight that writes it has a compare
    /// and a landing of its own behind it.
    ///
    /// `plans` come with their flights' numbers and `landed` is the number
    /// of the last flight being landed: a piece an earlier fetch on the
    /// copy queue brought down beside one of these flights is fetched
    /// whatever the compare says ([`BESIDE`]).
    fn relay_card_back(&self, plans: &[(u64, &CardPlan)], fetch: &mut Option<RelayFetch>, behind: u64, landed: u64) {
        if inflight_fetch() {
            let only: Vec<&CardPlan> = plans.iter().map(|p| p.1).collect();
            let taken = self.relay_card_inflight(&only, fetch, behind);
            inflight_tickets_landed(&only);
            if taken {
                return;
            }
        }
        let lane = crate::queue2::second().filter(|s| s.fetches());
        // The pieces fetched beside flights that had started, as noted by
        // earlier landings: looked at for these plans, renewed below.
        let beside = lane.is_some() && refetch();
        let noted: std::collections::HashMap<(usize, u32), u64> = if beside {
            BESIDE.lock().unwrap_or_else(|e| e.into_inner()).clone().unwrap_or_default()
        } else {
            Default::default()
        };
        let mut again = 0u64;
        // Every piece fetched now: (copy, piece number).
        let mut numbers: Vec<(usize, u32)> = Vec::new();
        let differs = {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            work.as_ref().map(|work| (work.differs.1, (work.differs.2 / 4) as usize))
        };
        let mut pieces: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, u64)> = Vec::new();
        // Each piece's (copy, piece number), as `numbers` holds them.
        let mut piece_numbers: Vec<(usize, u32)> = Vec::new();
        let mut total = 0u64;
        if let Some((differs, dwords)) = differs {
            // SAFETY: the mapping of the results. The compares of these
            // plans have run, and their dwords are given to no other plan
            // before these are done with (`card_done`).
            let differs = unsafe { std::slice::from_raw_parts(differs as *const u32, dwords) };
            let mut seen = std::collections::HashSet::new();
            for &(number, plan) in plans {
                let mut flags: Vec<(usize, u32, bool)> = plan
                    .ranges
                    .iter()
                    .flat_map(|r| {
                        (0..r.span.pieces).map(move |p| {
                            let flagged = !r.compared || differs.get((r.at + p) as usize).is_none_or(|&d| d != 0);
                            (std::sync::Arc::as_ptr(&r.shadow) as usize, r.span.first + p, flagged)
                        })
                    })
                    .collect();
                // A piece an earlier fetch brought down while this flight
                // may have been writing it: fetched, flagged or not.
                for flag in flags.iter_mut().filter(|f| !f.2) {
                    if fetched_beside(noted.get(&(flag.0, flag.1)).copied(), number) {
                        flag.2 = true;
                        again += 1;
                    }
                }
                let fetched = flagged_pieces(&flags);
                numbers.extend(fetched.iter().copied());
                for r in &plan.ranges {
                    let copy = std::sync::Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        if !fetched.contains(&(copy, r.span.first + p)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo < hi && seen.insert((copy, lo, hi)) {
                            pieces.push((r.shadow.clone(), lo, hi, total));
                            piece_numbers.push((copy, r.span.first + p));
                            total += (hi - lo) as u64;
                        }
                    }
                    crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
                }
            }
        }
        BESIDE_STATS[0].fetch_add(again, Ordering::Relaxed);
        if relay_copy_queue() && !inflight_fetch() {
            let last_sent = plans.iter().map(|p| p.1.sent_as).max();
            self.relay_copy_fetch(pieces.into_iter().map(|(shadow, lo, hi, _)| (shadow, lo, hi)).collect(), last_sent, fetch);
            return;
        }
        if pieces.is_empty() {
            if inflight_fetch() {
                // The next merge may be of a flight in the air now, with
                // bytes from when it ran: only sends up to these plans' are
                // older than it.
                if let Some(last) = plans.iter().map(|p| p.1.sent_as).max() {
                    crate::shadow::relay_fetched(last);
                }
                return;
            }
            if beside {
                // Notes whose flights have all landed are done with.
                let mut all = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
                note_beside(all.get_or_insert_with(Default::default), &[], landed, landed);
            }
            // Nothing to fetch: every send so far is older than the next
            // fetch (on a copy queue: every send the card has finished).
            crate::shadow::relay_fetched(match lane {
                Some(second) => second.order_done(),
                None => *QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()),
            });
            return;
        }
        if inflight_fetch() {
            INFLIGHT_STATS[3].fetch_add(1, Ordering::Relaxed);
        }
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY=1)"));
        };
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            CARD_FETCH_BYTES[1].store(f.buffer.as_ref().map_or(0, |b| b.2), Ordering::Relaxed);
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, VICTORY_RELAY=1)"));
        };
        let event = f.event;
        // The frame line's split (`crate::cpuwait`): what comes down, and
        // how much of it lies where shaders picking views from the heap may
        // write unannounced.
        let picked: Vec<bool> = pieces.iter().map(|(shadow, lo, hi, _)| shadow.picked_meets(*lo, *hi)).collect();
        {
            use crate::cpuwait::{fetch as c, fetch_add};
            fetch_add(c::FETCHES, 1);
            fetch_add(c::PIECES, pieces.len() as u64);
            fetch_add(c::BYTES, total);
            for (piece, &p) in pieces.iter().zip(&picked) {
                if p {
                    fetch_add(c::PICKED_PIECES, 1);
                    fetch_add(c::PICKED_BYTES, (piece.2 - piece.1) as u64);
                }
            }
        }
        // Which pieces go to the copy queue: all of them with it, none
        // without. With `crate::cpuwait::queue2_picked` at 1 none in a
        // picked span: those come down on the host's queue behind every
        // flight sent before them, after the others. At 2 they come down on
        // the copy queue with the others, and again on the host's queue if
        // a flight that may have run beside that fetch can store through
        // picked views ([`picks_beside`]).
        let level = if lane.is_some() { crate::cpuwait::queue2_picked() } else { 0 };
        let split = level == 1;
        let mut parts = fetch_parts(&picked, lane.is_some(), split);
        // Pieces whose bytes from the copy queue are not merged: they come
        // again on the host's queue.
        let mut again = vec![false; pieces.len()];
        let waiting = std::time::Instant::now();
        let mut title = Vec::new();
        let mut written = 0;
        // The lowest queue count of the parts' fetches: notes of sends are
        // kept from there.
        let mut after_all = u64::MAX;
        let mut part = 0;
        while part < parts.len() {
            let (lane_part, at) = parts[part].clone();
            let (lane_part, at) = (&lane_part, &at);
            part += 1;
            let part_bytes: u64 = at.iter().map(|&i| (pieces[i].2 - pieces[i].1) as u64).sum();
            let part_started = std::time::Instant::now();
            // SAFETY: the list's last run was waited for by this thread;
            // live buffers (the copies are held by the plans); the fence and
            // the event are this thread's.
            let ran = unsafe {
                (|| -> windows::core::Result<u64> {
                    if *lane_part && let Some(second) = lane {
                        // The sends of title bytes the card has finished are in
                        // what comes down. A later one may be too, if the card
                        // runs it before this fetch: the merge leaves its bytes
                        // out as a send that came after, and they are the
                        // title's either way.
                        let after = second.order_done();
                        let copy = second.down.list()?;
                        for &i in at {
                            let (shadow, lo, hi, to) = &pieces[i];
                            copy.1.CopyBufferRegion(&buffer, *to, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                        }
                        let job = second.down.send(copy, &[(&self.fence, behind)], part_bytes)?;
                        second.down.wait(job, event)?;
                        if beside {
                            // Whatever had started by now may have been
                            // writing these pieces while they came down
                            // (counted before a submission's lists go to the
                            // queue, `Host::submit`: nothing later ran yet).
                            // Only the pieces that came down here: the host
                            // queue's saw whole flights.
                            let started = STARTED.load(Ordering::Acquire);
                            let here: Vec<(usize, u32)> =
                                if split { at.iter().map(|&i| piece_numbers[i]).collect() } else { numbers.clone() };
                            let mut all = BESIDE.lock().unwrap_or_else(|e| e.into_inner());
                            let noted = note_beside(all.get_or_insert_with(Default::default), &here, landed, started);
                            BESIDE_STATS[1].fetch_add(noted, Ordering::Relaxed);
                        }
                        second.down.stats.waited_us.fetch_add(part_started.elapsed().as_micros() as u64, Ordering::Relaxed);
                        return Ok(after);
                    }
                    let f = fetch.as_mut().expect("the fetch's list was made above");
                    f.sent += 1;
                    let ((allocator, list), fence, sent) = (f.list.clone(), f.fence.clone(), f.sent);
                    allocator.Reset()?;
                    list.Reset(&allocator, None)?;
                    for &i in at {
                        let (shadow, lo, hi, to) = &pieces[i];
                        list.CopyBufferRegion(&buffer, *to, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                    }
                    list.Close()?;
                    let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                    // What this fetch stands behind on the host's queue: the
                    // flights sent after the ones landed (the value behind
                    // the last of them is on the queue ahead of it).
                    let (ahead, ahead_to) = {
                        let line = line();
                        let n = line.line.iter().filter(|l| matches!(l, Late::Run(flight) if flight.fence > behind)).count();
                        (n as u64, line.target)
                    };
                    // In the order of the queue: the sends of title bytes so
                    // far are ahead of this fetch, every later one behind it.
                    let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                    self.queue.ExecuteCommandLists(&lists);
                    let after = *order;
                    drop(order);
                    self.card_sent();
                    self.queue.Signal(&fence, sent)?;
                    // The flights ahead first (for the frame line), then the
                    // fetch's own copies.
                    if ahead_to > behind && self.fence.GetCompletedValue() < ahead_to {
                        self.fence.SetEventOnCompletion(ahead_to, event)?;
                        WaitForSingleObject(event, INFINITE);
                    }
                    crate::cpuwait::fetch_add(crate::cpuwait::fetch::AHEAD, ahead);
                    crate::cpuwait::fetch_add(crate::cpuwait::fetch::AHEAD_US, part_started.elapsed().as_micros() as u64);
                    fence.SetEventOnCompletion(sent, event)?;
                    WaitForSingleObject(event, INFINITE);
                    Ok(after)
                })()
            };
            let after = match ran {
                Ok(after) => after,
                Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY=1)")),
            };
            {
                use crate::cpuwait::{fetch as c, fetch_add};
                let us = part_started.elapsed().as_micros() as u64;
                if *lane_part {
                    fetch_add(c::LANE_N, 1);
                    fetch_add(c::LANE_BYTES, part_bytes);
                    fetch_add(c::LANE_US, us);
                } else {
                    fetch_add(c::HOST_N, 1);
                    fetch_add(c::HOST_US, us);
                }
            }
            // This fetch may bring bytes of flights in the air now: their own
            // slots are older than what it puts in the CPU's side there
            // (`relay_card_inflight`). Sends up to `after` are numbered, and the
            // submission numbered `after` may not have its lists on the queue
            // yet; every one before it has (the submission lock).
            if inflight_fetch()
                && after > 1
                && let Some(inflight) = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|w| w.inflight.as_mut())
            {
                inflight.brought.extend(at.iter().map(|&i| {
                    let (shadow, lo, hi, _) = &pieces[i];
                    (shadow.clone(), *lo, *hi, after - 1)
                }));
            }
            if *lane_part && level == 2 && picked.iter().any(|&p| p) && picks_beside(behind) {
                let picked_here: Vec<usize> = at.iter().copied().filter(|&i| picked[i]).collect();
                for &i in &picked_here {
                    again[i] = true;
                }
                crate::cpuwait::fetch_add(crate::cpuwait::fetch::AGAIN, 1);
                parts.push((false, picked_here));
            }
            for &i in at {
                if *lane_part && again[i] {
                    continue;
                }
                let (shadow, lo, hi, to) = &pieces[i];
                // SAFETY: the mapping of the fetched bytes, `total` of them;
                // the copy into it has run.
                let gpu = unsafe { std::slice::from_raw_parts((data + *to as usize) as *const u8, hi - lo) };
                let merged = shadow.card_merge_relay(*lo, gpu, after, &mut title);
                shadow.walls_merged(gpu.len(), merged);
                written += merged;
            }
            after_all = after_all.min(after);
        }
        if parts.len() > 1 {
            crate::cpuwait::fetch_add(crate::cpuwait::fetch::BOTH, 1);
        }
        let after = after_all;
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        PATH[path::FETCH].fetch_add(us, Ordering::Relaxed);
        {
            use crate::shadow::CARD_STATS;
            CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
            CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
            CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
            CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
        }
        crate::cpuwait::fetch_add(crate::cpuwait::fetch::MERGED, written as u64);
        if written == 0 {
            crate::cpuwait::fetch_add(crate::cpuwait::fetch::IDLE, 1);
        }
        let Some(f) = fetch.as_mut() else {
            return;
        };
        copy_back(1, written as u64);
        // Sends up to `after` are of no use to any later merge; with the
        // fetch inside flights only those up to these plans' are
        // (`fallback_fetched_mark`).
        let landed = plans.iter().map(|p| p.1.sent_as).max();
        crate::shadow::relay_fetched(fallback_fetched_mark(inflight_fetch(), after, landed));
        // Waited for and merged: a buffer past the bound goes
        // (`fetch_keep`).
        if fetch_keep().is_some_and(|keep| f.buffer.as_ref().is_some_and(|b| b.2 > keep)) {
            f.buffer = None;
            CARD_FETCH_BYTES[1].store(0, Ordering::Relaxed);
            CARD_FETCH_BYTES[2].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// [`Host::relay_card_back`] with [`relay_copy_queue`]: `flagged` (copy,
    /// offsets), pieces of flights the card has run, are fetched on the
    /// landing thread's copy queue and merged, never with bytes a running
    /// flight may be writing:
    ///
    /// - a piece no flight in the line declared is copied at once;
    /// - a piece flights in the line declared is copied behind the last of
    ///   them (the copy queue waits on the host fence for it), so its bytes
    ///   are that flight's complete ones, as the host queue's fetch brought;
    /// - after a copy has run, a piece a flight that could have run beside
    ///   it declared (one sent since it was looked at, or for the second
    ///   kind one behind the flight waited for) is not merged from it but
    ///   fetched again on the host queue, behind everything sent so far.
    ///
    /// `landed` is the landed plans' last send of title bytes (none: no
    /// plan landed). The merge leaves out title bytes the card may not have
    /// had when a copy ran: those sent after the last flight known to have
    /// run before it.
    fn relay_copy_fetch(
        &self,
        flagged: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        landed: Option<u64>,
        fetch: &mut Option<RelayFetch>,
    ) {
        use std::sync::Arc;
        type Piece = (Arc<crate::shadow::Shadow>, usize, usize, u64);
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let Some(f) = fetch.as_mut() else {
            fail(format_args!("no list, fence or event to fetch what shaders wrote to the card's memory (VICTORY_RELAY_COPY_QUEUE=1)"));
        };
        if let Some(landed) = landed {
            f.mark = f.mark.max(landed);
        }
        let mark = f.mark;
        let mut seen = std::collections::HashSet::new();
        let mut pieces: Vec<Piece> = Vec::new();
        let mut total = 0u64;
        for (shadow, lo, hi) in flagged {
            if lo < hi && seen.insert((Arc::as_ptr(&shadow) as usize, lo, hi)) {
                pieces.push((shadow, lo, hi, total));
                total += (hi - lo) as u64;
            }
        }
        if pieces.is_empty() {
            // Only sends the card had when the landed flights ran are older
            // than the next fetch: it does not wait for the host's queue.
            crate::shadow::relay_fetched(mark);
            return;
        }
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            CARD_FETCH_BYTES[1].store(f.buffer.as_ref().map_or(0, |b| b.2), Ordering::Relaxed);
        }
        let Some((buffer, data, _)) = f.buffer.clone() else {
            fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes, VICTORY_RELAY_COPY_QUEUE=1)"));
        };
        if self.relay_copy_lane(f).is_none() {
            fail(format_args!("no copy queue for the landing thread's fetches (VICTORY_RELAY_COPY_QUEUE=1)"));
        }
        let Some((queue, allocator, now_list, behind_list)) = f.copy.clone() else {
            return;
        };
        // Which pieces flights in the line declared, and the last of those
        // flights; and the last send of title bytes the card has had once
        // that flight has run.
        let (busy, wait_for, mark_behind) = {
            flights_in_line();
            let line = line();
            let busy: Vec<Option<u64>> = pieces.iter().map(|(shadow, lo, hi, _)| declared_by(&line, shadow, *lo, *hi, 0)).collect();
            let wait_for = busy.iter().flatten().copied().max();
            let mark_behind = wait_for.map_or(mark, |w| sends_through(&line, w).max(mark));
            (busy, wait_for, mark_behind)
        };
        let record = |list: &ID3D12GraphicsCommandList, behind: bool| -> windows::core::Result<bool> {
            let mut any = false;
            // SAFETY: an open list of this thread's; live buffers (the
            // copies are held by the plans being landed).
            unsafe {
                list.Reset(&allocator, None)?;
                for ((shadow, lo, hi, at), busy) in pieces.iter().zip(&busy) {
                    if busy.is_some() == behind {
                        list.CopyBufferRegion(&buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                        any = true;
                    }
                }
                list.Close()?;
            }
            Ok(any)
        };
        let waiting = std::time::Instant::now();
        let (first, second) = (f.sent + 1, f.sent + 2);
        f.sent += 2;
        let (fence, event) = (f.fence.clone(), f.event);
        // SAFETY: the lists' last runs were waited for by this thread; the
        // queue, fence and event are this thread's; the host fence is ours.
        let sent = unsafe {
            (|| -> windows::core::Result<(bool, bool)> {
                allocator.Reset()?;
                let now = record(&now_list, false)?;
                let behind = record(&behind_list, true)?;
                if now {
                    queue.ExecuteCommandLists(&[Some(now_list.cast::<ID3D12CommandList>()?)]);
                }
                queue.Signal(&fence, first)?;
                if let Some(w) = wait_for
                    && behind
                {
                    queue.Wait(&self.fence, w)?;
                    queue.ExecuteCommandLists(&[Some(behind_list.cast::<ID3D12CommandList>()?)]);
                }
                queue.Signal(&fence, second)?;
                Ok((now, behind))
            })()
        };
        let (now, behind) = match sent {
            Ok(sent) => sent,
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY_COPY_QUEUE=1)")),
        };
        let wait = |value: u64| {
            // SAFETY: this thread's fence and event.
            let waited = unsafe { fence.SetEventOnCompletion(value, event).map(|()| WaitForSingleObject(event, INFINITE)) };
            if let Err(e) = waited {
                fail(format_args!("the landing thread's copy queue could not be waited for ({e}, VICTORY_RELAY_COPY_QUEUE=1)"));
            }
        };
        let mut title = Vec::new();
        let mut written = 0;
        // Pieces whose copy may hold a running flight's bytes.
        let mut again: Vec<(Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        for (behind_kind, done, merge_mark, ran) in [(false, first, mark, now), (true, second, mark_behind, behind)] {
            if !ran {
                continue;
            }
            wait(done);
            // Every flight sent before the copy ended is in the line now.
            flights_in_line();
            let line = line();
            // A flight that may have run beside the copy: for pieces copied
            // at once any in the line (none declared them when looked at),
            // for those copied behind a flight any behind it.
            let after = if behind_kind { wait_for.unwrap_or(0) } else { 0 };
            let suspect: Vec<bool> = pieces
                .iter()
                .zip(&busy)
                .map(|((shadow, lo, hi, _), busy)| busy.is_some() == behind_kind && declared_by(&line, shadow, *lo, *hi, after).is_some())
                .collect();
            drop(line);
            for (((shadow, lo, hi, at), busy), suspect) in pieces.iter().zip(&busy).zip(suspect) {
                if busy.is_some() != behind_kind {
                    continue;
                }
                if suspect {
                    again.push((shadow.clone(), *lo, *hi));
                    continue;
                }
                // SAFETY: the mapping of the fetched bytes, `total` of them;
                // the copy into it has run.
                let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
                let merged = shadow.card_merge_relay(*lo, gpu, merge_mark, &mut title);
                shadow.walls_merged(gpu.len(), merged);
                written += merged;
            }
        }
        let us = waiting.elapsed().as_micros() as u64;
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        PATH[path::FETCH].fetch_add(us, Ordering::Relaxed);
        PATH[path::FETCH_COPY_N].fetch_add(1, Ordering::Relaxed);
        PATH[path::BEHIND_N].fetch_add(busy.iter().filter(|b| b.is_some()).count() as u64, Ordering::Relaxed);
        {
            use crate::shadow::CARD_STATS;
            CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
            CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
            CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
            CARD_STATS[6].fetch_add(us, Ordering::Relaxed);
        }
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
        if !again.is_empty() {
            PATH[path::AGAIN_N].fetch_add(again.len() as u64, Ordering::Relaxed);
            self.relay_queue_fetch(f, &again, &buffer, data);
        }
        // Only up to the landed plans' sends: a later landing's copy may
        // merge with that mark, and needs the notes of every send after it.
        crate::shadow::relay_fetched(mark);
        if fetch_keep().is_some_and(|keep| f.buffer.as_ref().is_some_and(|b| b.2 > keep)) {
            f.buffer = None;
            CARD_FETCH_BYTES[1].store(0, Ordering::Relaxed);
            CARD_FETCH_BYTES[2].fetch_add(1, Ordering::Relaxed);
        }
    }

    /// [`Host::relay_copy_fetch`]'s pieces whose copy may hold a running
    /// flight's bytes: fetched as without the switch, on the host queue
    /// behind everything sent so far, into the start of `buffer` (mapped
    /// at `data`; the copy queue's bytes there are merged already), and
    /// merged with the queue's order at the send.
    fn relay_queue_fetch(
        &self,
        f: &mut RelayFetch,
        pieces: &[(std::sync::Arc<crate::shadow::Shadow>, usize, usize)],
        buffer: &ID3D12Resource,
        data: usize,
    ) {
        f.sent += 1;
        let ((allocator, list), fence, event, sent) = (f.list.clone(), f.fence.clone(), f.event, f.sent);
        let mut offsets = Vec::with_capacity(pieces.len());
        let mut total = 0u64;
        for (_, lo, hi) in pieces {
            offsets.push(total);
            total += (hi - lo) as u64;
        }
        // SAFETY: as in `relay_card_back`: this thread's list, fence and
        // event; live buffers; the pieces fit (they are some of those the
        // buffer was sized for).
        let ran = unsafe {
            (|| -> windows::core::Result<u64> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                for ((shadow, lo, hi), at) in pieces.iter().zip(&offsets) {
                    list.CopyBufferRegion(buffer, *at, &shadow.buffer, *lo as u64, (*hi - *lo) as u64);
                }
                list.Close()?;
                let lists = [Some(list.cast::<ID3D12CommandList>()?)];
                let order = QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner());
                self.queue.ExecuteCommandLists(&lists);
                let after = *order;
                drop(order);
                self.card_sent();
                self.queue.Signal(&fence, sent)?;
                fence.SetEventOnCompletion(sent, event)?;
                WaitForSingleObject(event, INFINITE);
                Ok(after)
            })()
        };
        let after = match ran {
            Ok(after) => after,
            Err(e) => fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e}, VICTORY_RELAY_COPY_QUEUE=1)")),
        };
        let mut title = Vec::new();
        let mut written = 0;
        for ((shadow, lo, hi), at) in pieces.iter().zip(&offsets) {
            // SAFETY: the mapping of the fetched bytes; the copy has run.
            let gpu = unsafe { std::slice::from_raw_parts((data + *at as usize) as *const u8, hi - lo) };
            let merged = shadow.card_merge_relay(*lo, gpu, after, &mut title);
            shadow.walls_merged(gpu.len(), merged);
            written += merged;
        }
        COPY_BACK[1].fetch_add(written as u64, Ordering::Relaxed);
    }

    /// [`Host::relay_card_back`] with [`inflight_fetch`]: the pieces these
    /// plans' compares flagged are taken from the slots their own compare
    /// lists copied them to ([`CARD_FETCH`]) and merged, plan by plan in the
    /// line's order, with no list or wait of ours. `false` (nothing merged)
    /// if any plan cannot land so; the caller then lands them all the old
    /// way, as without the switch.
    ///
    /// The merge's "fetched after" (`Shadow::card_merge_relay`) is the
    /// plan's own `sent_as`. `QUEUE_ORDER` is raised only in
    /// `Host::card_before`, under the submission lock, and that lock is held
    /// until the submission's lists and its compare's list are on the queue
    /// behind its sync list: so when the compare's list ran, the card had
    /// received exactly the sends 1..=`sent_as` of title bytes and no later
    /// one, which is what the merge's skip of later sends asks. (The old
    /// fetch, by contrast, ran whenever it reached the queue, and took the
    /// count at its own ExecuteCommandLists.)
    ///
    /// When a landing goes the old way instead, all of it does: a fetch the
    /// old way brings the card's bytes as they are now, newer than the slots
    /// of plans in the air then, and merging an older slot after it would
    /// put a flight's bytes back over a later flight's. For the same reason
    /// the bytes such a fetch brought (`CardInflight::brought`) are left
    /// out of the slots of flights whose lists were on the queue ahead of
    /// it: the CPU's side holds what the card held after them there. The
    /// rest of their slots is merged, newer than anything merged there
    /// before. (Left out whole, a flight sent before such a fetch would have
    /// had to land the old way, its fetch taking a later count still: one
    /// overflow kept every landing on the old way while flights were in
    /// the air.) Plans land in the line's order, each plan's slot as it was
    /// right behind its own lists, so every merge here is of bytes at least
    /// as new as the CPU's side holds.
    ///
    /// shortcut: all or nothing per landing, and a plan whose pieces overflow
    /// the slots lands the old way for all its pieces, not only the ones
    /// left out.
    ///
    /// With [`inflight_on_card`] the slots are in the card's memory: the
    /// pieces to merge are copied from them to a buffer in the PC's memory
    /// on the landing thread's copy queue, behind `behind` (the host fence
    /// value behind the last of these flights, reached already), and
    /// waited for, before the merge and before the tickets are landed.
    fn relay_card_inflight(&self, plans: &[&CardPlan], fetch: &mut Option<RelayFetch>, behind: u64) -> bool {
        use std::sync::Arc;
        // (copy, offsets, the piece's place in the slots' buffer, "fetched after").
        let mut pieces: Vec<InflightPiece> = Vec::new();
        let ring: (ID3D12Resource, usize);
        {
            let work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
            let Some(work) = work.as_ref() else {
                return false;
            };
            let Some(inflight) = &work.inflight else {
                return false;
            };
            let dwords = (work.differs.2 / 4) as usize;
            // SAFETY: the mappings of the results and the tickets. The
            // compares and fetches of these plans have run, and their dwords
            // are given to no other plan before these are done with
            // (`card_done`).
            let (differs, table) = unsafe {
                (
                    std::slice::from_raw_parts(work.differs.1 as *const u32, dwords),
                    std::slice::from_raw_parts((inflight.table.1 as *const u32).add(1), dwords),
                )
            };
            let slots = inflight.fetched.2;
            ring = (inflight.fetched.0.clone(), inflight.fetched.1);
            for plan in plans {
                if plan.ranges.is_empty() {
                    continue;
                }
                if !plan.inflight || plan.ranges.iter().any(|r| !r.compared) {
                    return false;
                }
                // Pieces (copy, number) the compare flagged, with their ticket.
                let mut tickets: std::collections::HashMap<(usize, u32), u32> = std::collections::HashMap::new();
                let mut flags: Vec<(usize, u32, bool)> = Vec::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let at = (r.at + p) as usize;
                        let (Some(&flag), Some(&entry)) = (differs.get(at), table.get(at)) else {
                            return false;
                        };
                        if flag != 0 {
                            if entry == INFLIGHT_NONE {
                                return false;
                            }
                            if !inflight_gets_slot(entry, plan.inflight_landed, slots) {
                                INFLIGHT_STATS[2].fetch_add(1, Ordering::Relaxed);
                                return false;
                            }
                            tickets.insert((copy, r.span.first + p), entry);
                        }
                        flags.push((copy, r.span.first + p, flag != 0));
                    }
                }
                let fetched = flagged_pieces(&flags);
                let mut seen = std::collections::HashSet::new();
                for r in &plan.ranges {
                    let copy = Arc::as_ptr(&r.shadow) as usize;
                    for p in 0..r.span.pieces {
                        let piece = r.span.first + p;
                        if !fetched.contains(&(copy, piece)) {
                            continue;
                        }
                        let (lo, hi) = r.shadow.card_piece(&r.span, p);
                        if lo >= hi || !seen.insert((copy, lo, hi)) {
                            continue;
                        }
                        let from = inflight_piece_from(&r.span, p);
                        let Some(&k) = tickets.get(&(copy, piece)) else {
                            return false;
                        };
                        if lo < from || (hi - from) as u64 > INFLIGHT_SLOT {
                            return false;
                        }
                        let slot = (k % slots) as usize * INFLIGHT_SLOT as usize;
                        // Bytes a fetch the old way brought after this flight ran.
                        let newer: Vec<(usize, usize)> = inflight
                            .brought
                            .iter()
                            .filter(|b| b.3 >= plan.sent_as && Arc::ptr_eq(&b.0, &r.shadow) && b.1 < hi && b.2 > lo)
                            .map(|b| (b.1, b.2))
                            .collect();
                        for (a, b) in crate::shadow::outside(lo, hi, newer) {
                            pieces.push((r.shadow.clone(), a, b, slot + (a - from), plan.sent_as));
                        }
                    }
                }
            }
        }
        // Outside the lock: no slot of these plans is written over before
        // their tickets are landed (`inflight_tickets_landed`, after this).
        // With the slots in the card's memory they come down first, into
        // `base` (mapped), one after the other; else `base` is the slots'
        // own mapping.
        let base = if inflight_on_card() {
            match self.inflight_copy_down(&ring.0, &pieces, fetch, behind) {
                Some(data) => data,
                None => return false,
            }
        } else {
            ring.1
        };
        let mut title = Vec::new();
        let (mut written, mut bytes) = (0, 0u64);
        for (shadow, lo, hi, at, after) in &pieces {
            let at = if inflight_on_card() { base + bytes as usize } else { base + at };
            // SAFETY: inside the mapping of the slots (checked above), or of
            // the buffer they were copied down to (`inflight_copy_down`
            // puts them one after the other, and has waited); the copy into
            // the slot ran in the plan's flight, which the card has run.
            let gpu = unsafe { std::slice::from_raw_parts(at as *const u8, hi - lo) };
            written += shadow.card_merge_relay(*lo, gpu, *after, &mut title);
            bytes += (hi - lo) as u64;
        }
        for plan in plans {
            for r in &plan.ranges {
                crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
            }
        }
        copy_back(1, written as u64);
        INFLIGHT_STATS[0].fetch_add(pieces.len() as u64, Ordering::Relaxed);
        INFLIGHT_STATS[1].fetch_add(bytes, Ordering::Relaxed);
        // The next merge is of a later plan, or a fetch the old way sent now.
        if let Some(last) = plans.iter().map(|p| p.sent_as).max() {
            crate::shadow::relay_fetched(last);
        }
        true
    }

    /// [`Host::relay_card_inflight`] with [`inflight_on_card`]: `pieces`'
    /// bytes, at their places in the slots' buffer `ring`, are copied one
    /// after the other into the landing thread's buffer in the PC's memory
    /// ([`RelayFetch::buffer`]) on its copy queue, which first waits on the
    /// host fence for `behind`; and waited for. The mapping of the buffer,
    /// or `None` (the landing then goes the old way) if the queue or the
    /// memory cannot be had.
    fn inflight_copy_down(
        &self,
        ring: &ID3D12Resource,
        pieces: &[InflightPiece],
        fetch: &mut Option<RelayFetch>,
        behind: u64,
    ) -> Option<usize> {
        let started = std::time::Instant::now();
        if fetch.is_none() {
            *fetch = self.make_relay_fetch();
        }
        let f = fetch.as_mut()?;
        let copies = slots_down(pieces.iter().map(|(_, lo, hi, at, _)| (*at as u64, (hi - lo) as u64)));
        let total: u64 = copies.iter().map(|c| c.2).sum();
        if total == 0 {
            return Some(0);
        }
        if f.buffer.as_ref().is_none_or(|b| b.2 < total) {
            let size = total.next_multiple_of(16 << 20);
            f.buffer = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
            CARD_FETCH_BYTES[1].store(f.buffer.as_ref().map_or(0, |b| b.2), Ordering::Relaxed);
        }
        let (buffer, data, _) = f.buffer.clone()?;
        let (queue, allocator, list, _) = self.relay_copy_lane(f)?;
        f.sent += 1;
        let (fence, event, sent) = (f.fence.clone(), f.event, f.sent);
        // SAFETY: the list's last run was waited for by this thread; live
        // buffers; the queue, fence and event are this thread's; the host
        // fence is ours.
        let ran = unsafe {
            (|| -> windows::core::Result<()> {
                allocator.Reset()?;
                list.Reset(&allocator, None)?;
                record_slots_down(&list, &buffer, ring, &copies);
                list.Close()?;
                queue.Wait(&self.fence, behind)?;
                queue.ExecuteCommandLists(&[Some(list.cast::<ID3D12CommandList>()?)]);
                queue.Signal(&fence, sent)?;
                fence.SetEventOnCompletion(sent, event)?;
                WaitForSingleObject(event, INFINITE);
                Ok(())
            })()
        };
        if let Err(e) = ran {
            fail(format_args!("the slots of the fetch inside flights could not be copied down ({e}, VICTORY_CARD_FETCH_INFLIGHT=2)"));
        }
        let us = started.elapsed().as_micros() as u64;
        INFLIGHT_STATS[4].fetch_add(1, Ordering::Relaxed);
        INFLIGHT_STATS[5].fetch_add(total, Ordering::Relaxed);
        INFLIGHT_STATS[6].fetch_add(us, Ordering::Relaxed);
        INFLIGHT_STATS[7].fetch_add(copies.len() as u64, Ordering::Relaxed);
        RELAY_STATS[3].fetch_add(us, Ordering::Relaxed);
        PATH[path::FETCH].fetch_add(us, Ordering::Relaxed);
        Some(data)
    }

    /// The landing thread's copy queue ([`RelayFetch::copy`]), made the
    /// first time.
    fn relay_copy_lane(&self, f: &mut RelayFetch) -> Option<CopyFetch> {
        if f.copy.is_none() {
            // SAFETY: plain object creation on our device.
            let made = unsafe {
                (|| -> windows::core::Result<CopyFetch> {
                    let queue: ID3D12CommandQueue = self.device.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_COPY,
                        ..Default::default()
                    })?;
                    let allocator: ID3D12CommandAllocator = self.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COPY)?;
                    let list = |allocator: &ID3D12CommandAllocator| -> windows::core::Result<ID3D12GraphicsCommandList> {
                        let list: ID3D12GraphicsCommandList =
                            self.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COPY, allocator, None)?;
                        list.Close()?;
                        Ok(list)
                    };
                    let (now, behind) = (list(&allocator)?, list(&allocator)?);
                    Ok((queue, allocator, now, behind))
                })()
            };
            match made {
                Ok(made) => {
                    name(&made.0, || "victory relay copy queue".into());
                    tracing::info!(
                        "the landing thread copies card pieces down on a copy queue of its own (VICTORY_RELAY_COPY_QUEUE=1 or \
                         VICTORY_CARD_FETCH_INFLIGHT=2)"
                    );
                    f.copy = Some(made);
                }
                Err(e) => {
                    tracing::warn!(error = %e, "no copy queue for the landing thread");
                    return None;
                }
            }
        }
        f.copy.clone()
    }

    /// [`relay`], on the thread that lands flights: `items` are the first
    /// things of the line, flights the card has run and what follows them.
    /// What their shaders wrote goes into title memory (copies in the
    /// card's memory first, then the others, each byte once whichever
    /// flights declared it); then every flight's ranges are done with;
    /// then, in the line's order, each flight's read-backs, its
    /// bottom-of-pipe writes and the fence signals behind it. Title memory
    /// is complete before the title is told anything.
    fn relay_land(&self, items: Vec<Late>, fetch: &mut Option<RelayFetch>) {
        let landing = std::time::Instant::now();
        {
            let flights: Vec<&Flight> = items
                .iter()
                .filter_map(|l| match l {
                    Late::Run(flight) => Some(flight),
                    _ => None,
                })
                .collect();
            if !flights.is_empty() {
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = flights.iter().flat_map(|f| f.order.iter().map(|&i| f.lists[i].host())).collect();
                    fail(format_args!(
                        "host GPU device removed after {} submissions sent ahead (VICTORY_RELAY=1) of {} lists{}{}",
                        flights.len(),
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                fill_ahead::landing(flights.iter().map(|f| f.number).min().unwrap_or(0));
                crate::early::landing(
                    flights.iter().map(|f| f.number).min().unwrap_or(0),
                    flights.iter().map(|f| f.number).max().unwrap_or(0),
                );
                let plans: Vec<(u64, &CardPlan)> = flights.iter().filter_map(|f| f.card.as_ref().map(|card| (f.number, card))).collect();
                let behind = flights.iter().map(|f| f.fence).max().unwrap_or(0);
                let landed = flights.iter().map(|f| f.number).max().unwrap_or(0);
                self.relay_card_back(&plans, fetch, behind, landed);
                let mut back: Vec<Declared> = Vec::new();
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                relay_hand_back(&back);
                // Only now, and flight by flight: a range another flight
                // in the air declared too stays closed to refreshes.
                for flight in &flights {
                    for (shadow, ranges) in &flight.back {
                        shadow.copy_back_done(ranges);
                    }
                }
                copy_back(0, landing.elapsed().as_micros() as u64);
                RELAY_STATS[0].fetch_add(1, Ordering::Relaxed);
                RELAY_STATS[1].fetch_add(flights.len() as u64, Ordering::Relaxed);
            }
        }
        // In the order given: what tells the title the card is done.
        let mut last = 0;
        let mut landed = Vec::new();
        // The host fence value of the last flight ahead (`VICTORY_FENCE_EARLY`'s times).
        let mut ahead = None;
        for item in items {
            match item {
                Late::Run(flight) => {
                    last = flight.number;
                    fence_early::landed(flight.fence);
                    ahead = Some(flight.fence);
                    for &i in &flight.order {
                        read_backs(flight.lists[i].host());
                    }
                    if let Some(card) = flight.card {
                        self.card_done(card);
                    }
                    landed.push((flight.lists, flight.order));
                }
                Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                Late::Signal(fence, value) => {
                    crate::fence::set(fence as crate::object::This, value);
                    fence_early::signalled(ahead);
                }
            }
        }
        if last > 0 {
            FINISHED.store(last, Ordering::Release);
            bury(last);
        }
        self.log_messages();
        fill_ahead::landing(0);
        crate::early::landing(0, 0);
        RELAY_STATS[4].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        PATH[path::LAND].fetch_add(landing.elapsed().as_micros() as u64, Ordering::Relaxed);
        {
            let mut line = line();
            let Line {
                line: items,
                written,
                lists,
                held,
                settling,
                poke,
                landed: kept,
                in_hand,
                landing,
                read_back,
                ..
            } = &mut *line;
            kept.extend(landed);
            in_hand.clear();
            *landing = 0;
            // What is still in the air, from the flights that are.
            written.clear();
            read_back.clear();
            *lists = 0;
            *held = 0;
            let mut ranges = Vec::new();
            let mut backs = Vec::new();
            for item in items.iter() {
                if let Late::Run(flight) = item {
                    let n = flight.order.len().max(1);
                    // A flight that gave its state back is not counted
                    // against the bound (`crate::airlists`).
                    if !flight.freed {
                        *lists += n;
                    }
                    *held += n;
                    ranges.extend(flight.back.iter().flat_map(|(_, r)| r.iter().copied()));
                    backs.extend(flight.read_back.iter().copied());
                }
            }
            join_ranges(written, ranges);
            join_ranges(read_back, backs);
            *settling = false;
            // The landed lists are reset, and what queues hold looked at,
            // by the helper thread (they take locks a waiter may hold).
            *poke = true;
        }
        LINE_MOVED.notify_all();
        LINE_LANDED.notify_all();
    }

    /// After a submission ran, for copies in the card's memory: the pieces
    /// of the declared ranges that the compare found differing (all of a
    /// range it did not compare) are fetched, with one more wait for the
    /// card, and what shaders wrote in them goes into title memory.
    /// `submitted` is the last value signalled on the fence. `seen` holds
    /// the pieces fetched already since the card was last waited for
    /// (several submissions settled together name the same ones).
    fn card_back(
        &self,
        plan: &CardPlan,
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        self.card_back_all(&[plan], submitted, seen);
    }

    /// [`Host::card_back`] for the plans of several submissions settled
    /// together ([`join_fetch`]): their pieces come down in one fetch, with
    /// one wait for the card, and are merged in the order of the plans.
    /// Every flight has run when the fetch does, so each piece is the card's
    /// last bytes either way.
    fn card_back_all(
        &self,
        plans: &[&CardPlan],
        submitted: &mut u64,
        seen: &mut std::collections::HashSet<(usize, usize, usize)>,
    ) {
        let ranges: Vec<&CardRange> = plans.iter().flat_map(|plan| plan.ranges.iter()).collect();
        if ranges.is_empty() {
            return;
        }
        let merging = std::time::Instant::now();
        let mut waited = std::time::Duration::ZERO;
        let mut work = CARD_WORK.lock().unwrap_or_else(|e| e.into_inner());
        let Some(work) = work.as_mut() else {
            return;
        };
        // SAFETY: the mapping of the results, which the compare (it has
        // run) wrote.
        let differs = unsafe { std::slice::from_raw_parts(work.differs.1 as *const u32, (work.differs.2 / 4) as usize) };
        // The pieces to fetch: the range, the offsets in its copy, and
        // where the bytes come down.
        let mut pieces: Vec<(usize, usize, usize, u64)> = Vec::new();
        let mut total = 0u64;
        for (i, r) in ranges.iter().enumerate() {
            for p in 0..r.span.pieces {
                if r.compared && differs[(r.at + p) as usize] == 0 {
                    continue;
                }
                let (lo, hi) = r.shadow.card_piece(&r.span, p);
                if lo < hi && seen.insert((std::sync::Arc::as_ptr(&r.shadow) as usize, lo, hi)) {
                    pieces.push((i, lo, hi, total));
                    total += (hi - lo) as u64;
                }
            }
            crate::shadow::CARD_STATS[2].fetch_add(u64::from(r.span.pieces), Ordering::Relaxed);
        }
        let mut written = vec![0usize; ranges.len()];
        if !pieces.is_empty() {
            if work.fetched.as_ref().is_none_or(|f| f.2 < total) {
                let size = total.next_multiple_of(16 << 20);
                work.fetched = self.shared_buffer(size).map(|(buffer, data)| (buffer, data as usize, size));
                CARD_FETCH_BYTES[0].store(work.fetched.as_ref().map_or(0, |f| f.2), Ordering::Relaxed);
            }
            let Some((fetched, data, _)) = &work.fetched else {
                fail(format_args!("no memory to fetch what shaders wrote to the card's memory ({total} bytes)"));
            };
            let waiting = std::time::Instant::now();
            let (allocator, list) = &work.fetch;
            // SAFETY: the list's last run was waited for; live buffers; the
            // fence and event are ours, under the submission lock.
            let ran = unsafe {
                (|| -> windows::core::Result<()> {
                    // With `VICTORY_QUEUE2` bit 1 the fetch goes to the copy
                    // queue here too, behind everything sent to the direct
                    // queue. Nothing is gained (the card has run it all
                    // when this is reached); the path is the same one the
                    // relay uses, in the plainest run there is.
                    if let Some(second) = crate::queue2::second().filter(|s| s.fetches()) {
                        let copy = second.down.list()?;
                        for &(i, lo, hi, at) in &pieces {
                            copy.1.CopyBufferRegion(fetched, at, &ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                        }
                        let job = second.down.send(copy, &[(&self.fence, *submitted)], total)?;
                        let was_working = working(false);
                        second.down.wait(job, self.event)?;
                        stood(waiting);
                        working(was_working);
                        second.down.stats.waited_us.fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                        return Ok(());
                    }
                    allocator.Reset()?;
                    list.Reset(allocator, None)?;
                    for &(i, lo, hi, at) in &pieces {
                        list.CopyBufferRegion(fetched, at, &ranges[i].shadow.buffer, lo as u64, (hi - lo) as u64);
                    }
                    list.Close()?;
                    self.queue.ExecuteCommandLists(&[Some(list.cast()?)]);
                    self.card_sent();
                    *submitted += 1;
                    self.queue.Signal(&self.fence, *submitted)?;
                    self.fence.SetEventOnCompletion(*submitted, self.event)?;
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    stood(waiting);
                    working(was_working);
                    Ok(())
                })()
            };
            if let Err(e) = ran {
                fail(format_args!("what shaders wrote to the card's memory could not be fetched ({e})"));
            }
            {
                use crate::shadow::CARD_STATS;
                CARD_STATS[3].fetch_add(pieces.len() as u64, Ordering::Relaxed);
                CARD_STATS[4].fetch_add(total, Ordering::Relaxed);
                CARD_STATS[5].fetch_add(1, Ordering::Relaxed);
                CARD_STATS[6].fetch_add(waiting.elapsed().as_micros() as u64, Ordering::Relaxed);
                waited = waiting.elapsed();
            }
            if wide_merge() && pieces.len() >= 8 {
                // The pieces are apart from one another (each its own bytes
                // of its copy and of title memory): several at a time.
                let fetched = *data;
                let counts: std::sync::Arc<Vec<std::sync::atomic::AtomicUsize>> =
                    std::sync::Arc::new((0..ranges.len()).map(|_| std::sync::atomic::AtomicUsize::new(0)).collect());
                let each = pieces.len().div_ceil(8);
                let jobs: Vec<Box<dyn FnOnce() + Send>> = pieces
                    .chunks(each)
                    .map(|chunk| {
                        let chunk: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize, usize, u64)> =
                            chunk.iter().map(|&(i, lo, hi, at)| (ranges[i].shadow.clone(), i, lo, hi, at)).collect();
                        let counts = counts.clone();
                        Box::new(move || {
                            let mut title = Vec::new();
                            for (shadow, i, lo, hi, at) in chunk {
                                // SAFETY: the mapping of the fetched bytes;
                                // the copy into it has run, and the caller
                                // waits for this job.
                                let gpu = unsafe { std::slice::from_raw_parts((fetched + at as usize) as *const u8, hi - lo) };
                                let merged = shadow.card_merge(lo, gpu, &mut title);
                                shadow.walls_merged(gpu.len(), merged);
                                counts[i].fetch_add(merged, Ordering::Relaxed);
                            }
                        }) as Box<dyn FnOnce() + Send>
                    })
                    .collect();
                crate::shadow::wide_run(jobs);
                for (written, count) in written.iter_mut().zip(counts.iter()) {
                    *written += count.load(Ordering::Relaxed);
                }
            } else {
                let mut title = Vec::new();
                for &(i, lo, hi, at) in &pieces {
                    // SAFETY: the mapping of the fetched bytes, `total` of
                    // them; the copy into it has run.
                    let gpu = unsafe { std::slice::from_raw_parts((*data + at as usize) as *const u8, hi - lo) };
                    let merged = ranges[i].shadow.card_merge(lo, gpu, &mut title);
                    ranges[i].shadow.walls_merged(gpu.len(), merged);
                    written[i] += merged;
                }
            }
        }
        // Waited for and merged: a buffer past the bound goes
        // (`fetch_keep`).
        if fetch_keep().is_some_and(|keep| work.fetched.as_ref().is_some_and(|f| f.2 > keep)) {
            work.fetched = None;
            CARD_FETCH_BYTES[0].store(0, Ordering::Relaxed);
            CARD_FETCH_BYTES[2].fetch_add(1, Ordering::Relaxed);
        }
        BACK_SPLIT[0].fetch_add(merging.elapsed().saturating_sub(waited).as_micros() as u64, Ordering::Relaxed);
        for (r, written) in ranges.iter().zip(written) {
            copy_back(1, written as u64);
            let (from, to) = r.declared;
            #[cfg(feature = "diagnostics")]
            crate::copystat::compared(from, to - from, written);
            let _ = (from, to);
        }
    }

    /// Runs closed lists and waits for them, and with [`late`] for
    /// everything sent to the card before them, which is settled first.
    pub(crate) fn execute(&self, runs: &[&HostList]) {
        // Behind what the queues were given before it and have not run
        // yet (`crate::queue::behind`).
        crate::queue::before_sync();
        let _ = self.submit(runs, false);
    }

    /// A queue's submission with [`late`]: the host lists `lists`, taken
    /// from the title's, go to the card in `order` and are not waited for.
    /// They join the line of flights and are settled, with the others in
    /// the air, at the next point where what they do can be observed
    /// ([`late`] lists those points); their host lists are reset and kept
    /// for reuse then, as a queue does for the lists it ran.
    pub(crate) fn execute_late(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        self.execute_late_cut(lists, order, &[]);
    }

    /// Several parts of one queue's line ([`crate::queue`]'s
    /// `VICTORY_JOIN_PARTS`), each `(lists, order)` as [`Host::execute_late`]
    /// takes them, prepared as one submission and sent as one flight: one
    /// heap sync, one asking, one flush of the refreshes, one snapshot and
    /// compare, one signal of the host fence. Each part still goes to the
    /// queue in an `ExecuteCommandLists` call of its own, so the card sees
    /// the same boundaries between them as before.
    pub(crate) fn execute_late_joined(&self, parts: Vec<(Vec<crate::list::Detached>, Vec<usize>)>) {
        let mut lists = Vec::new();
        let mut order = Vec::new();
        let mut cuts = Vec::new();
        for (part, part_order) in parts {
            if !order.is_empty() && !part_order.is_empty() {
                cuts.push(order.len());
            }
            let base = lists.len();
            order.extend(part_order.into_iter().map(|i| i + base));
            lists.extend(part);
        }
        self.execute_late_cut(lists, order, &cuts);
    }

    /// [`Host::execute_late`], with `cuts` the places in `order` where a
    /// part joined to the ones before it begins ([`Host::execute_late_joined`]).
    fn execute_late_cut(&self, lists: Vec<crate::list::Detached>, order: Vec<usize>, cuts: &[usize]) {
        if order.is_empty() {
            crate::list::recycle(lists, &order);
            return;
        }
        late_threads();
        let (picks, sent) = {
            let runs: Vec<&HostList> = order.iter().map(|&i| lists[i].host()).collect();
            // As recorded, before the steps run.
            let picks = runs.iter().any(|list| list.deferred().textures.contains(&crate::draw::PICKED_STORE_KEY));
            (picks, self.submit_cut(&runs, true, cuts))
        };
        let Some((mut submitted, sent)) = sent else {
            // (Not reached: a submission sent ahead always comes back.)
            crate::list::recycle(lists, &order);
            return;
        };
        let told = !sent.writes.is_empty();
        let most = {
            let mut line = line();
            let Line { line: items, written, lists: flying, held, target, read_back, .. } = &mut *line;
            join_ranges(written, sent.back.iter().flat_map(|(_, ranges)| ranges.iter().copied()));
            join_ranges(read_back, sent.read_back.iter().copied());
            *flying += order.len();
            *held += order.len();
            *target = *submitted;
            items.push(Late::Run(Flight {
                number: sent.number,
                fence: *target,
                lists,
                order,
                card: sent.card,
                back: sent.back,
                read_back: sent.read_back,
                freed: false,
                picks,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            FLIGHTS_LINED.fetch_add(1, Ordering::SeqCst);
            let free_ran = crate::airlists::on() && relay_exact();
            if free_ran {
                crate::airlists::top(crate::airlists::n::HELD_TOP, *held as u64);
            }
            *flying >= late_most() || (free_ran && *held >= crate::airlists::held_most(late_most()))
        };
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        RELAY_MOVED.notify_one();
        if told && !late_writes() {
            self.settle(&mut submitted, WHY_WRITES);
        } else if most && relay_exact() && crate::airlists::on() {
            self.relay_give_back(&mut submitted);
        } else if most && relay_exact() {
            // Until there are fewer, not until there are none.
            relay_wait_split(WHY_MOST, Some((crate::cpuwait::MOST, &most_need_relay)), |line| line.lists < late_most());
        } else if most && settle_lists() > 0 {
            self.free_ran(&mut submitted);
        } else if most {
            let need = if relay() { u64::MAX } else { most_need() };
            self.settle_needing(&mut submitted, WHY_MOST, need);
        } else {
            LINE_MOVED.notify_one();
        }
        drop(submitted);
        land();
    }

    /// Sends `list`, a closed list of ours (the present's copy), as a
    /// flight ([`late_free_present`]): not waited for, settled with the
    /// others. The caller keeps the list and does not reset it before the
    /// submission whose number comes back has finished ([`FINISHED`]); 0 if
    /// it ran and was waited for after all.
    fn execute_own(&self, list: &HostList) -> u64 {
        late_threads();
        let Some((submitted, sent)) = self.submit(&[list], true) else {
            return 0;
        };
        let number = sent.number;
        {
            let mut line = line();
            let Line { line: items, written, lists: flying, held, target, read_back, .. } = &mut *line;
            join_ranges(written, sent.back.iter().flat_map(|(_, ranges)| ranges.iter().copied()));
            join_ranges(read_back, sent.read_back.iter().copied());
            *flying += 1;
            *held += 1;
            *target = *submitted;
            items.push(Late::Run(Flight {
                number,
                fence: *target,
                lists: Vec::new(),
                order: Vec::new(),
                card: sent.card,
                back: sent.back,
                read_back: sent.read_back,
                freed: false,
                picks: false,
            }));
            items.extend(sent.writes.into_iter().map(|(at, what, bytes)| Late::Write(at, what, bytes)));
            FLIGHTS_LINED.fetch_add(1, Ordering::SeqCst);
        }
        LATE_STATS[0].fetch_add(1, Ordering::Relaxed);
        LINE_MOVED.notify_one();
        RELAY_MOVED.notify_one();
        drop(submitted);
        number
    }

    /// Waits for the card to run every flight in the air ([`late`]) and
    /// does what they left to do, as [`Host::execute`] does for one
    /// submission after its wait: the pieces their shaders wrote in copies
    /// in the card's memory are fetched and merged (each once), the ranges
    /// they declared in the other copies are compared and copied back
    /// (each byte once, whichever flights declared it), and then, in the
    /// order they were given: their bottom-of-pipe writes and the fence
    /// signals behind them. `submitted` is the submission lock's value;
    /// `why` one of the `WHY_` places. Nothing happens if nothing flies.
    ///
    /// Everything in the air is settled, never the first flights alone:
    /// the merge writes the copies' CPU side, which a flight still on the
    /// card may be writing too.
    fn settle(&self, submitted: &mut u64, why: usize) {
        self.settle_needing(submitted, why, u64::MAX);
    }

    /// [`settle_lists`] at 2: waits until the card has run the flights up
    /// to `need` and says whether [`reads_unwritten`] spares the settle.
    fn read_unwritten(&self, runs: &[&HostList], need: u64) -> bool {
        let waiting = std::time::Instant::now();
        self.card_reached(need, true);
        let us = waiting.elapsed().as_micros() as u64;
        SETTLE_LISTS_STATS[6].fetch_add(us, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
        let unwritten = reads_unwritten(runs, need);
        SETTLE_LISTS_STATS[if unwritten { 5 } else { 7 }].fetch_add(1, Ordering::Relaxed);
        unwritten
    }

    /// [`settle_lists`]: the line holds [`late_most`] host lists that still
    /// hold their descriptor slots. The card is waited for until the
    /// first of those flights have run, as many as take the count under the
    /// bound, and every flight the card has run by then gives them back; the
    /// flights stay in the line for the next settle. Settled as without the
    /// switch when the line is too long in all or the compare's result area
    /// of copies in the card's memory runs low (it starts again only when
    /// no plan is out).
    fn free_ran(&self, submitted: &mut u64) {
        let held = line().held;
        if held >= late_most().saturating_mul(SETTLE_LISTS_HELD) || self.card_results_low() {
            SETTLE_LISTS_STATS[4].fetch_add(1, Ordering::Relaxed);
            let need = most_need();
            self.settle_needing(submitted, WHY_MOST, need);
            return;
        }
        let need = most_need();
        let waiting = std::time::Instant::now();
        self.card_reached(need.min(*submitted), true);
        let us = waiting.elapsed().as_micros() as u64;
        SETTLE_LISTS_STATS[3].fetch_add(us, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
        // SAFETY: a plain query.
        let completed = unsafe { self.fence.GetCompletedValue() };
        // Dropped outside the line's lock: giving slots back takes the
        // descriptor heaps' own.
        let mut gone = Vec::new();
        let (mut flights, mut lists) = (0u64, 0u64);
        {
            let mut line = line();
            let Line { line: items, lists: flying, .. } = &mut *line;
            for item in items.iter_mut() {
                let Late::Run(flight) = item else {
                    continue;
                };
                if flight.fence > completed {
                    // Later flights were sent after it: the card runs them
                    // in order.
                    break;
                }
                if flight.freed {
                    continue;
                }
                flight.freed = true;
                let n = flight.order.len().max(1);
                *flying = flying.saturating_sub(n);
                flights += 1;
                lists += n as u64;
                gone.extend(flight.lists.iter_mut().map(crate::list::Detached::release));
            }
        }
        drop(gone);
        SETTLE_LISTS_STATS[0].fetch_add(1, Ordering::Relaxed);
        SETTLE_LISTS_STATS[1].fetch_add(flights, Ordering::Relaxed);
        SETTLE_LISTS_STATS[2].fetch_add(lists, Ordering::Relaxed);
        LINE_MOVED.notify_one();
    }

    /// [`crate::airlists::on`], with [`relay_exact`]: the line holds
    /// [`late_most`] host lists that still hold their draw state, or
    /// [`crate::airlists::held_most`] in all. At the second the submitting
    /// thread waits for the landing, as without the switch (host lists,
    /// scratch and upload memory stay until then). At the first the card
    /// is waited for until it has run as many of the first flights as take
    /// the count under the bound, and every flight it has run by then gives
    /// its state back ([`give_back_ran`]); the flights stay in the line for
    /// the landing thread.
    fn relay_give_back(&self, submitted: &mut u64) {
        use crate::airlists::{add, held_most, n};
        let most = late_most();
        let held = held_most(most);
        if line().held >= held {
            let waiting = std::time::Instant::now();
            let need = |line: &Line| {
                let flights: Vec<(u64, usize)> = line
                    .line
                    .iter()
                    .filter_map(|item| match item {
                        Late::Run(flight) => Some((flight.fence, flight.order.len().max(1))),
                        _ => None,
                    })
                    .collect();
                most_need_lists(&flights, held)
            };
            relay_wait_split(WHY_MOST, Some((crate::cpuwait::MOST, &need)), |line| line.held < held);
            add(n::HELD_WAITS, 1);
            add(n::HELD_US, waiting.elapsed().as_micros() as u64);
        }
        if line().lists < most {
            return;
        }
        add(n::AT_BOUND, 1);
        // Dropped outside the line's lock: giving slots back takes the
        // descriptor heaps' own.
        let mut gone = Vec::new();
        let give_back = |gone: &mut Vec<crate::draw::State>| {
            // SAFETY: a plain query.
            let completed = unsafe { self.fence.GetCompletedValue() };
            let mut line = line();
            let (flights, lists) = give_back_ran(&mut line.line, completed, gone);
            line.lists = line.lists.saturating_sub(lists);
            add(n::RAN_FLIGHTS, flights);
            add(n::RAN_LISTS, lists as u64);
            line.lists
        };
        if give_back(&mut gone) >= most {
            let need = most_need();
            let waiting = std::time::Instant::now();
            self.card_reached(need.min(*submitted), true);
            let us = waiting.elapsed().as_micros() as u64;
            add(n::CARD_WAITS, 1);
            add(n::CARD_US, us);
            SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
            give_back(&mut gone);
        }
        drop(gone);
    }

    /// [`table_on_card`], for the frame line, before a submission sent
    /// ahead looks for reads under flights' writes: whether its tables'
    /// constant views bound in copies lie under them (their fills would
    /// have read there), and whether that alone would have settled.
    fn table_card_look(&self, runs: &[&HostList]) {
        use table_card as t;
        // SAFETY: a plain query.
        let completed = unsafe { self.fence.GetCompletedValue() };
        TABLE_CARD_PENDING.lock().unwrap_or_else(|e| e.into_inner()).retain(|&(need, at)| {
            if need > completed {
                return true;
            }
            t::STATS[t::WAITED_US].fetch_add(at.elapsed().as_micros() as u64, Ordering::Relaxed);
            false
        });
        let reads: Vec<(usize, usize)> = runs.iter().flat_map(|list| list.deferred().on_card.clone()).collect();
        if reads.is_empty() {
            return;
        }
        let (written, need, under) = {
            let line = line();
            if line.written.is_empty() {
                return;
            }
            let (mut need, mut under) = (0u64, 0u64);
            for &(from, len) in &reads {
                let to = from + len;
                if !touches(&line.written, from, to) {
                    continue;
                }
                under += 1;
                for item in &line.line {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    for (_, ranges) in &flight.back {
                        if ranges.iter().any(|&(a, e)| a < to && e > from) {
                            need = need.max(flight.fence);
                        }
                    }
                }
            }
            if under == 0 {
                return;
            }
            (line.written.clone(), need, under)
        };
        t::STATS[t::UNDER].fetch_add(under, Ordering::Relaxed);
        t::count(t::SUBMISSIONS);
        if blocked_by(&written, runs, false).is_none() {
            t::count(t::AVOIDED);
            if need <= completed {
                t::count(t::ALREADY);
            } else {
                TABLE_CARD_PENDING
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .push((need, std::time::Instant::now()));
            }
        }
    }

    /// Waits until the card has run what went up before host fence value
    /// `value` was signalled; with `stand`, the time counts as the
    /// submitting thread standing for the card ([`stood`]). Under the
    /// submission lock: the host's event is the settles'.
    fn card_reached(&self, value: u64, stand: bool) {
        // SAFETY: our fence and event, under the submission lock.
        unsafe {
            if self.fence.GetCompletedValue() >= value {
                return;
            }
            match self.fence.SetEventOnCompletion(value, self.event) {
                Ok(()) => {
                    let waiting = std::time::Instant::now();
                    let was_working = working(false);
                    WaitForSingleObject(self.event, INFINITE);
                    if stand {
                        stood(waiting);
                    }
                    working(was_working);
                }
                Err(e) => tracing::warn!(error = %e, "submissions sent ahead not waited for"),
            }
        }
    }

    /// [`Host::settle`]; `need` is the host fence value up to which the
    /// settle needed the card to have run (`u64::MAX`: all of it), for the
    /// frame line ([`LATE_NEED`]): the card is waited for up to it first,
    /// then for the rest, and nothing else differs.
    fn settle_needing(&self, submitted: &mut u64, why: usize, need: u64) {
        if !late() {
            return;
        }
        if relay() {
            // Nothing is settled here: the thread that lands flights does
            // it, in order, and this waits until nothing is in the air and
            // nothing is being landed. Every point that settled before
            // comes here, so each still sees title memory, fences and
            // bottom-of-pipe values as after a settle.
            let done = |line: &Line| line.line.is_empty() && !line.settling;
            if why == WHY_STALE {
                relay_wait_split(why, Some((crate::cpuwait::STALE, &|line: &Line| line.target)), done);
            } else {
                relay_wait(why, done);
            }
            return;
        }
        let was_working = working(true);
        let mut counted = false;
        loop {
            let items = {
                let mut line = line();
                if line.line.is_empty() {
                    line.settling = false;
                    break;
                }
                line.settling = true;
                line.written.clear();
                line.read_back.clear();
                line.lists = 0;
                line.held = 0;
                std::mem::take(&mut line.line)
            };
            let flights = items.iter().filter(|i| matches!(i, Late::Run(_))).count();
            let first = !counted;
            if !counted {
                counted = true;
                LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
                LATE_STATS[why].fetch_add(1, Ordering::Relaxed);
            }
            SETTLES_DONE.fetch_add(1, Ordering::Relaxed);
            LATE_STATS[11].fetch_add(flights as u64, Ordering::Relaxed);
            if flights > 0 {
                submitting(2, flights);
                let waiting = std::time::Instant::now();
                // The flights the settle needed first (the frame line's
                // split of the wait); then all of them, as without it.
                let needed = if first && need < *submitted {
                    self.card_reached(need, false);
                    waiting.elapsed()
                } else {
                    std::time::Duration::MAX
                };
                // SAFETY: the fence and event are ours, under the
                // submission lock; `submitted` is the last value signalled.
                unsafe {
                    match self.fence.SetEventOnCompletion(*submitted, self.event) {
                        Ok(()) => {
                            working(false);
                            WaitForSingleObject(self.event, INFINITE);
                            stood(waiting);
                            working(true);
                        }
                        Err(e) => tracing::warn!(error = %e, "submissions sent ahead not waited for"),
                    }
                }
                let gpu = waiting.elapsed();
                LATE_STATS[12].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                LATE_WAITED[why.min(10)].fetch_add(gpu.as_micros() as u64, Ordering::Relaxed);
                if first {
                    LATE_NEED[why.min(10)].fetch_add(needed.min(gpu).as_micros() as u64, Ordering::Relaxed);
                }
                submission_us(1, gpu.as_micros() as u64);
                #[cfg(feature = "diagnostics")]
                crate::copystat::card_waited(gpu.as_micros() as u64, || {
                    format!("{flights} submissions sent ahead, settled together")
                });
                // SAFETY: a plain query.
                if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
                    let runs: Vec<&HostList> = items
                        .iter()
                        .filter_map(|i| match i {
                            Late::Run(f) => Some(f),
                            _ => None,
                        })
                        .flat_map(|f| f.order.iter().map(|&i| f.lists[i].host()))
                        .collect();
                    fail(format_args!(
                        "host GPU device removed after {flights} submissions sent ahead (VICTORY_LATE_WAIT=1) of {} lists{}{}",
                        runs.len(),
                        self.removal_data(&runs),
                        last_notes(&runs)
                    ));
                }
                submitting(3, flights);
                let copying = std::time::Instant::now();
                let mut seen = std::collections::HashSet::new();
                let mut back: Vec<Declared> = Vec::new();
                if join_fetch() {
                    let plans: Vec<&CardPlan> = items
                        .iter()
                        .filter_map(|item| match item {
                            Late::Run(flight) => flight.card.as_ref(),
                            _ => None,
                        })
                        .collect();
                    self.card_back_all(&plans, submitted, &mut seen);
                }
                for item in &items {
                    let Late::Run(flight) = item else {
                        continue;
                    };
                    if let Some(card) = &flight.card
                        && !join_fetch()
                    {
                        self.card_back(card, submitted, &mut seen);
                    }
                    for (shadow, ranges) in &flight.back {
                        match back.iter_mut().find(|(s, _)| std::sync::Arc::ptr_eq(s, shadow)) {
                            Some((_, all)) => {
                                for range in ranges {
                                    if !all.contains(range) {
                                        all.push(*range);
                                    }
                                }
                            }
                            None => back.push((shadow.clone(), ranges.clone())),
                        }
                    }
                }
                let comparing = std::time::Instant::now();
                hand_back(&back);
                BACK_SPLIT[1].fetch_add(comparing.elapsed().as_micros() as u64, Ordering::Relaxed);
                copy_back(0, copying.elapsed().as_micros() as u64);
                LATE_BACK[why.min(10)].fetch_add(copying.elapsed().as_micros() as u64, Ordering::Relaxed);
            }
            // In the order given: what tells the title the card is done.
            let mut last = 0;
            let mut landed = Vec::new();
            for item in items {
                match item {
                    Late::Run(flight) => {
                        last = flight.number;
                        for &i in &flight.order {
                            read_backs(flight.lists[i].host());
                        }
                        if let Some(card) = flight.card {
                            self.card_done(card);
                        }
                        landed.push((flight.lists, flight.order));
                    }
                    Late::Write(at, what, bytes) => what.write(at as *mut u8, bytes),
                    Late::Signal(fence, value) => crate::fence::set(fence as crate::object::This, value),
                }
            }
            if last > 0 {
                FINISHED.store(last, Ordering::Release);
                bury(last);
            }
            line().landed.extend(landed);
        }
        if counted {
            self.log_messages();
            submitting(0, 0);
            // What was settled may be what a queue stands at, and the
            // caller may not be one that looks (a read-back, a present).
            if why != WHY_CARD_DONE {
                line().poke = true;
                LINE_MOVED.notify_one();
            }
        }
        working(was_working);
    }

    /// The settle for title writes under shader writes ([`WHY_STALE`]) that
    /// `runs` refresh. With [`crate::cpuwait::stale_exact`] (and the relay's
    /// second stage) only until the flights that declared shader writes in
    /// what they refresh have landed; if the check still finds such writes
    /// after it, everything is settled as without the switch. `stale_ns`:
    /// the stale checks' time, as [`stale_timed`] keeps it.
    fn settle_stale(&self, submitted: &mut u64, runs: &[&HostList], stale_ns: &mut u64) {
        if !(relay_exact() && crate::cpuwait::stale_exact()) {
            self.settle(submitted, WHY_STALE);
            return;
        }
        let spans: Vec<(usize, usize)> = refresh_spans(runs).into_iter().map(|(_, a, e)| (a, e)).collect();
        let need = |line: &Line| {
            line.line
                .iter()
                .filter_map(|item| match item {
                    Late::Run(flight)
                        if flight
                            .back
                            .iter()
                            .any(|(_, ranges)| ranges.iter().any(|&(x, y)| spans.iter().any(|&(a, e)| x < e && y > a))) =>
                    {
                        Some(flight.fence)
                    }
                    _ => None,
                })
                .max()
                .unwrap_or(0)
        };
        let done = |line: &Line| !spans.iter().any(|&(a, e)| touches(&line.written, a, e));
        relay_wait_split(WHY_STALE, Some((crate::cpuwait::STALE, &need)), done);
        if stale_timed(runs, stale_ns) {
            crate::cpuwait::stale_fallback();
            self.settle(submitted, WHY_STALE);
        }
    }

    /// [`relay_wait`] for a submission whose preparing reads title memory
    /// flights' shaders may write ([`WHY_READ`], with [`relay_exact`]):
    /// until the flights that declared it have landed, as before, counted
    /// besides in [`READ_SPLIT`] (what read) and [`READ_WAIT`] (how many
    /// flights were in the air and needed, and when the card had run the
    /// needed ones: the rest of the wait was their landing).
    fn relay_read_wait(&self, runs: &[&HostList]) {
        use read_wait as w;
        let done = |line: &Line| blocked_by(&line.written, runs, false).is_none();
        // The host fence value of the last flight in the line that declared
        // memory read (0: only flights being landed did, which the card has
        // run; `u64::MAX`: a step of ours reads anywhere).
        let need = read_plan(runs).need;
        // SAFETY: a plain query.
        let completed = unsafe { self.fence.GetCompletedValue() };
        if done(&line()) {
            return;
        }
        read_wait_detail(runs, need, completed);
        let mut line = line();
        if done(&line) {
            return;
        }
        let target = if need == u64::MAX { line.target } else { need };
        let (mut in_air, mut needed) = (0u64, 0u64);
        for item in &line.line {
            if let Late::Run(flight) = item {
                in_air += 1;
                if flight.fence <= target {
                    needed += 1;
                }
            }
        }
        READ_WAIT[w::WAITS].fetch_add(1, Ordering::Relaxed);
        READ_WAIT[w::IN_AIR].fetch_add(in_air, Ordering::Relaxed);
        READ_WAIT[w::NEEDED].fetch_add(needed, Ordering::Relaxed);
        READ_WAIT[w::LANDING].fetch_add(line.landing as u64, Ordering::Relaxed);
        READ_WAIT[w::LISTS].fetch_add(line.lists as u64, Ordering::Relaxed);
        if needed == 0 {
            READ_WAIT[w::IN_HAND].fetch_add(1, Ordering::Relaxed);
        }
        let fetched_before = RELAY_STATS[3].load(Ordering::Relaxed);
        LATE_STATS[1].fetch_add(1, Ordering::Relaxed);
        LATE_STATS[WHY_READ].fetch_add(1, Ordering::Relaxed);
        let was_working = working(false);
        let waiting = std::time::Instant::now();
        // SAFETY (both): a plain query.
        let reached = || unsafe { self.fence.GetCompletedValue() } >= target;
        let mut card = reached().then_some(std::time::Duration::ZERO);
        if card.is_some() {
            READ_WAIT[w::CARD_DONE].fetch_add(1, Ordering::Relaxed);
        }
        // With `sync_in_wait`: when the heap pages the title wrote since
        // were last synced, every few milliseconds of the wait.
        let mut synced = std::time::Instant::now();
        while !done(&line) {
            if sync_in_wait() && synced.elapsed() >= std::time::Duration::from_millis(SYNC_IN_WAIT_MS) {
                drop(line);
                sync_before_wait();
                synced = std::time::Instant::now();
                line = self::line();
                if card.is_none() && reached() {
                    card = Some(waiting.elapsed());
                }
                continue;
            }
            if card.is_none() {
                // Looked at every millisecond until the card has run the
                // needed flights (nothing tells this thread of that), then
                // only when flights have landed.
                line = LINE_LANDED.wait_timeout(line, std::time::Duration::from_millis(1)).unwrap_or_else(|e| e.into_inner()).0;
                if reached() {
                    card = Some(waiting.elapsed());
                }
            } else if sync_in_wait() {
                line = LINE_LANDED.wait_timeout(line, std::time::Duration::from_millis(1)).unwrap_or_else(|e| e.into_inner()).0;
            } else {
                line = LINE_LANDED.wait(line).unwrap_or_else(|e| e.into_inner());
            }
        }
        drop(line);
        stood(waiting);
        working(was_working);
        let all = waiting.elapsed();
        let us = all.as_micros() as u64;
        READ_WAIT[w::US].fetch_add(us, Ordering::Relaxed);
        READ_WAIT[w::CARD_US].fetch_add(card.unwrap_or(all).as_micros() as u64, Ordering::Relaxed);
        READ_WAIT[w::FETCH_US].fetch_add(RELAY_STATS[3].load(Ordering::Relaxed).saturating_sub(fetched_before), Ordering::Relaxed);
        LATE_STATS[12].fetch_add(us, Ordering::Relaxed);
        LATE_WAITED[WHY_READ].fetch_add(us, Ordering::Relaxed);
        SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
    }

    /// [`Host::execute`], or with `late` the first half of
    /// [`Host::execute_late`]: the lists are prepared and sent, and what is
    /// to be done once they have run comes back with the submission lock
    /// still held.
    /// The submission lock, timed while held ([`Submitted`]).
    fn lock_submitted(&self) -> Submitted<'_> {
        let guard = self.submitted.lock().unwrap_or_else(|e| e.into_inner());
        held(true);
        Submitted(guard)
    }

    fn submit<'a>(&'a self, runs: &[&HostList], late: bool) -> Option<(Submitted<'a>, Sent)> {
        self.submit_cut(runs, late, &[])
    }

    /// [`Host::submit`]; with `late`, `cuts` are the places in `runs` where
    /// a part of a joined submission begins ([`Host::execute_late_joined`]):
    /// the lists go to the queue in one call per part, one signal after all.
    fn submit_cut<'a>(&'a self, runs: &[&HostList], late: bool, cuts: &[usize]) -> Option<(Submitted<'a>, Sent)> {
        #[cfg(feature = "diagnostics")]
        let called = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_begins();
        let locking = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        crate::profile::seen(crate::profile::Seen::Submit);
        // One submission at a time from preparation to copy-back: snapshots
        // and the ranges shaders write are per host copy, not per submission.
        let mut submitted = self.lock_submitted();
        prep_add(prep::LOCK, locking);
        let was_working = working(true);
        let mut mark = WorkMark::new();
        PATH[path::SUBMITS_N].fetch_add(1, Ordering::Relaxed);
        sync_along();
        submitting(1, runs.len());
        let settling_before = std::time::Instant::now();
        // Flights in the air ([`late`]): a submission that is waited for
        // settles them first, one sent ahead only if preparing it reads
        // what they write.
        // With [`fill_from_copy`]: the parts of fills read from copies, and
        // the settles so far (one after this voids them).
        let mut from_copy: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)> = Vec::new();
        let settles_before = SETTLES_DONE.load(Ordering::Relaxed);
        // Whether the waits below were the relay's for declared writes only.
        let mut targeted = false;
        if late && table_on_card() > 0 {
            self.table_card_look(runs);
        }
        // The table constant views bound at card slots: which of their
        // bytes the card copies from the buffers' copies (under flights'
        // writes), before anything looks at what preparing reads; their
        // fills read none of those. No submission is sent from here until
        // the lists' steps have run (the submission lock): no flight joins.
        if table_gpu_copy() > 0 {
            let written = line().written.clone();
            for list in runs {
                list.card_classify(&written);
            }
        }
        if late && table_gpu_copy() > 0 {
            table_gpu_look(runs);
        }
        if copy_on_card() > 0 {
            card_copy_look(runs);
        }
        let blocked = if late { late_blocked(runs, false) } else { None };
        if !late {
            self.settle(&mut submitted, WHY_SYNC);
        } else if let Some(why) = blocked {
            if relay_exact() && why == WHY_READ {
                // Only until the flights that may write what is read have
                // landed: their copy-back's end, not the card's other work.
                let _ = late_blocked(runs, true);
                sync_before_wait();
                self.relay_read_wait(runs);
                targeted = true;
            } else if why == WHY_READ && !relay() {
                let plan = read_plan(runs);
                match plan.from_copy {
                    Some(parts) if fill_from_copy() => {
                        // Only until the card has run the flights that
                        // declared what the fills read; they stay in the air.
                        let waiting = std::time::Instant::now();
                        self.card_reached(plan.need, true);
                        let us = waiting.elapsed().as_micros() as u64;
                        READ_SPLIT[reads::FROM_COPY].fetch_add(1, Ordering::Relaxed);
                        READ_SPLIT[reads::WAITED_US].fetch_add(us, Ordering::Relaxed);
                        SUBMISSION_US[1].fetch_add(us, Ordering::Relaxed);
                        from_copy = parts;
                    }
                    _ if settle_lists() > 1 && plan.need != u64::MAX && self.read_unwritten(runs, plan.need) => {}
                    _ => {
                        let _ = late_blocked(runs, true);
                        self.settle_needing(&mut submitted, why, plan.need);
                    }
                }
            } else {
                if why == WHY_READ {
                    let _ = late_blocked(runs, true);
                }
                self.settle(&mut submitted, why);
            }
        }
        if late {
            read_early_look(runs, targeted);
        }
        mark.to(work::BEFORE);
        // Descriptors the title copied into placed heaps directly, before
        // command lists that index those slots are submitted. With
        // `sync_overlap` on a thread of its own beside the preparing, joined
        // before anything that must come after it.
        let syncing = std::time::Instant::now();
        let mut beside = late && sync_overlap() && !crate::tiles::wanted() && sync_beside_start();
        if beside {
            PATH[path::SYNC_BESIDE_N].fetch_add(1, Ordering::Relaxed);
        } else {
            crate::descriptor::sync_heaps();
            PATH[path::SYNC].fetch_add(syncing.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        mark.to(work::SYNC);
        let behind = late && flying();
        prep_add(prep::SETTLE_BEFORE, settling_before);
        let sweeping = std::time::Instant::now();
        // Before this submission counts as started: what the sweep lets go
        // of is named by no list, and nothing runs now (behind flights the
        // sweep waits for a submission that finds none in the air: there is
        // one at every present).
        static SWEPT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        if !behind {
            // After the sync, as before.
            mark.to(work::SWEEP);
            if beside {
                PATH[path::JOIN_EARLY_N].fetch_add(1, Ordering::Relaxed);
            }
            sync_beside_join(&mut beside);
            mark.to(work::SYNC);
            let _gate = ALONG_GATE.lock().unwrap_or_else(|e| e.into_inner());
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        } else if late_free_present() && now_ms() > SWEPT.load(Ordering::Relaxed) + 1500 {
            mark.to(work::SWEEP);
            if beside {
                PATH[path::JOIN_EARLY_N].fetch_add(1, Ordering::Relaxed);
            }
            sync_beside_join(&mut beside);
            mark.to(work::SYNC);
            // With the present sent ahead nothing else brings a moment
            // with nothing in the air: one is made for the sweep.
            self.settle(&mut submitted, WHY_SYNC);
            let _gate = ALONG_GATE.lock().unwrap_or_else(|e| e.into_inner());
            sweep();
            SWEPT.store(now_ms(), Ordering::Relaxed);
        }
        let behind = late && flying();
        prep_add(prep::SWEEP, sweeping);
        let waiting_slots = std::time::Instant::now();
        // The slots that waited for a texture's host copy, before the
        // lists that may read them run (with the sync beside, after it is
        // joined: the ones it adds are made then).
        if !beside {
            crate::descriptor::make_waiting();
        }
        prep_add(prep::WAITING_SLOTS, waiting_slots);
        mark.to(work::SWEEP);
        let this_submission = STARTED.fetch_add(1, Ordering::AcqRel) + 1;
        // The ranges shaders write are counted for it (with the relay).
        crate::shadow::preparing(this_submission);
        let started = std::time::Instant::now();
        #[cfg(feature = "diagnostics")]
        let lock = (started - called).as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::Submissions, 1);
        let mut lists: Vec<Option<ID3D12CommandList>> = Vec::with_capacity(2 * runs.len());
        // Where each run's lists start in `lists`.
        let mut firsts = Vec::with_capacity(runs.len());
        let mut batch = Batch::default();
        // The time a settle of flights takes inside this preparing: the
        // wait for the card and the copy-back have their own places on the
        // frame line (run 92 showed them twice, "preparing" 250 ms for 135).
        let mut settled = std::time::Duration::ZERO;
        let once_started = std::time::Instant::now();
        // The stale checks' time, which the asking's does not hold.
        let mut stale_ns = 0u64;
        // One write-watch asking for everything the submission refreshes
        // (`VICTORY_ASK_ONCE`): per copy, the span of its refreshed ranges.
        if crate::watch::once() {
            let mut spans: std::collections::HashMap<usize, (usize, usize)> = std::collections::HashMap::new();
            // With `VICTORY_ASK_ONCE_VIEWS`: each refreshed range as its
            // refresh asks (cut to its copy), for the view memory the
            // spans' asking cannot take in (`crate::watch::views_once`).
            let views = crate::watch::views_once();
            let mut exact: Vec<(usize, usize)> = Vec::new();
            for list in runs {
                for step in &list.deferred().copies {
                    if let Step::Refresh(shadow, address, len) = step
                        && !shadow.unwatched()
                    {
                        let span = spans
                            .entry(std::sync::Arc::as_ptr(shadow) as usize)
                            .or_insert((*address, *address + *len));
                        *span = (span.0.min(*address), span.1.max(*address + *len));
                        if views && let Some(range) = shadow.refreshed(*address, *len) {
                            exact.push(range);
                        }
                    }
                }
            }
            let ranges: Vec<(usize, usize)> = spans.into_values().collect();
            let asking = std::time::Instant::now();
            more_add(more::ONCE_SPANS, (asking - once_started).as_nanos() as u64);
            crate::watch::ask_once(&ranges);
            let viewing = std::time::Instant::now();
            more_add(more::ONCE_ASK, (viewing - asking).as_nanos() as u64);
            crate::watch::ask_once_views(&exact);
            more_add(more::ONCE_VIEWS, viewing.elapsed().as_nanos() as u64);
            crate::shadow::REFRESH_SPLIT[3].fetch_add(asking.elapsed().as_micros() as u64, Ordering::Relaxed);
            // Behind flights: the title wrote memory that a flight's shaders
            // may write too, and this submission refreshes it. The flights
            // are settled, and the asking made again: the copy-back wrote
            // title memory, which other copies of it must see.
            if behind && stale_timed(runs, &mut stale_ns) {
                let settling = std::time::Instant::now();
                // Its copy-back writes title memory the sync may read.
                if !relay() {
                    sync_beside_join(&mut beside);
                }
                self.settle_stale(&mut submitted, runs, &mut stale_ns);
                settled += settling.elapsed();
                crate::watch::ask_once(&ranges);
                crate::watch::ask_once_views(&exact);
            }
            PREP_SPLIT[prep::ONCE].fetch_add(
                (once_started.elapsed().saturating_sub(settled).as_nanos() as u64).saturating_sub(stale_ns),
                Ordering::Relaxed,
            );
        } else if behind && stale_timed(runs, &mut stale_ns) {
            let settling = std::time::Instant::now();
            if !relay() {
                sync_beside_join(&mut beside);
            }
            self.settle_stale(&mut submitted, runs, &mut stale_ns);
            settled += settling.elapsed();
        }
        mark.to(work::ASK);
        // A settle since the plan copied everything back: title memory
        // holds what the flights wrote, and the fills read it there.
        if SETTLES_DONE.load(Ordering::Relaxed) == settles_before {
            batch.from_copy = from_copy;
        }
        // What the steps read ahead of shaders' writes, after every wait
        // for flights above (`note_reads_ahead`).
        if late {
            note_reads_ahead(runs, this_submission);
        }
        for list in runs {
            list.run_deferred(&mut batch);
            let gathering = std::time::Instant::now();
            firsts.push(lists.len());
            if let Some(prelude) = list.prelude() {
                lists.push(Some(prelude));
            }
            // Its table constants copied on the card ([`table_gpu_copy`]),
            // after everything before it in the queue.
            if let Some(pre) = list.card_pre() {
                lists.push(Some(pre));
            }
            lists.push(list.list.cast().ok());
            prep_add(prep::LISTS, gathering);
        }
        batch.flush();
        PREP_SPLIT[prep::REFRESH].fetch_add(batch.flush_ns, Ordering::Relaxed);
        PREP_SPLIT[prep::FILL].fetch_add(batch.fill_ns, Ordering::Relaxed);
        PREP_SPLIT[prep::TEXTURE].fetch_add(batch.tex_ns, Ordering::Relaxed);
        let ending = std::time::Instant::now();
        crate::watch::ask_once_end();
        prep_add(prep::ASK_END, ending);
        mark.to(work::STEPS);
        // The snapshots of the ranges shaders may write, if they were left
        // for now (`crate::shadow::wide_back`): the refreshes are done, the
        // lists have not gone to the card.
        {
            let snapshots = std::time::Instant::now();
            crate::shadow::take_snapshots();
            prep_add(prep::SNAPSHOTS, snapshots);
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::SnapshotUs, snapshots.elapsed().as_micros() as u64);
        }
        mark.to(work::CARD);
        // The sync beside the preparing ends here: before the copies in
        // the card's memory (they gather what its slots may name) and the
        // lists going to the queue; then the slots that waited for a
        // texture's host copy, those it added among them.
        if beside {
            let joining = std::time::Instant::now();
            sync_beside_join(&mut beside);
            PATH[path::JOIN_LATE_US].fetch_add(joining.elapsed().as_micros() as u64, Ordering::Relaxed);
            mark.to(work::SYNC);
            let waiting_slots = std::time::Instant::now();
            crate::descriptor::make_waiting();
            prep_add(prep::WAITING_SLOTS, waiting_slots);
            mark.to(work::SWEEP);
        }
        // Copies in the card's memory: what changed on their CPU side goes
        // up ahead of the lists, and a compare of what the shaders may
        // write follows them in the same submission.
        let flown = late && relay();
        let checking = std::time::Instant::now();
        if flown && self.card_results_low() {
            prep_add(prep::CARD_LOW, checking);
            let settling = std::time::Instant::now();
            self.settle(&mut submitted, WHY_MOST);
            settled += settling.elapsed();
            crate::airlists::add(crate::airlists::n::AREA, 1);
            crate::airlists::add(crate::airlists::n::AREA_US, settling.elapsed().as_micros() as u64);
        } else {
            prep_add(prep::CARD_LOW, checking);
        }
        let mut card = self.card_before(runs, flown, *submitted);
        // The sends' list first, in the same call (`card_sync_join`): the
        // first group holds it.
        if let Some(sync) = card.as_mut().and_then(|c| c.sync.take()) {
            lists.insert(0, Some(sync));
            for first in firsts.iter_mut().skip(1) {
                *first += 1;
            }
            more_add(more::SYNC_JOINED_N, 1);
        }
        if let Some(check) = card.as_ref().and_then(|c| c.check.clone()) {
            lists.push(Some(check));
        }
        mark.to(work::CARD);
        let prepared = std::time::Instant::now();
        PREP_SPLIT[prep::SETTLED_IN].fetch_add(settled.as_nanos() as u64, Ordering::Relaxed);
        if late {
            *submitted += 1;
            FLIGHTS_SENT.fetch_add(1, Ordering::SeqCst);
            // SAFETY: closed lists; the fence is ours.
            let signalled = unsafe {
                // A joined submission: a call per part, as when each was a
                // submission of its own (the card's boundaries between
                // them stay); the sends' list is in the first, the compare
                // in the last.
                let mut from = 0;
                for &cut in cuts {
                    let at = firsts.get(cut).copied().unwrap_or(lists.len());
                    if at > from && at < lists.len() {
                        self.queue.ExecuteCommandLists(&lists[from..at]);
                        from = at;
                    }
                }
                self.queue.ExecuteCommandLists(&lists[from..]);
                self.queue.Signal(&self.fence, *submitted)
            };
            self.card_sent();
            if let Err(e) = signalled {
                fail(format_args!("a submission sent ahead could not be followed by a signal ({e})"));
            }
            for list in runs {
                list.prof_submitted(*submitted);
            }
            let back = declared_writes(runs);
            let writes = runs.iter().flat_map(|l| l.deferred().writes.clone()).collect();
            let read_back = read_back_spans(runs);
            prep_add(prep::SEND, prepared);
            submission_us(0, (prepared - started).saturating_sub(settled).as_micros() as u64);
            submitting(0, 0);
            #[cfg(feature = "diagnostics")]
            crate::copystat::submission_ends(lock);
            let sent = Sent {
                number: this_submission,
                card,
                back,
                writes,
                read_back,
            };
            mark.to(work::SEND);
            working(was_working);
            return Some((submitted, sent));
        }
        // Diagnostics: each list on its own, timed (`VICTORY_GPU_TIMING`).
        #[cfg(feature = "diagnostics")]
        let groups: Vec<&[Option<ID3D12CommandList>]> = if timing() {
            firsts
                .iter()
                .zip(firsts.iter().skip(1).chain([&lists.len()]))
                .map(|(&a, &b)| &lists[a..b])
                .collect()
        } else {
            vec![&lists[..]]
        };
        #[cfg(not(feature = "diagnostics"))]
        let groups = [&lists[..]];
        submitting(2, runs.len());
        for (i, group) in groups.iter().enumerate() {
            *submitted += 1;
            let started = std::time::Instant::now();
            // SAFETY: closed lists; the fence and event are ours.
            unsafe {
                self.queue.ExecuteCommandLists(group);
                self.card_sent();
                let waited = self
                    .queue
                    .Signal(&self.fence, *submitted)
                    .and_then(|()| self.fence.SetEventOnCompletion(*submitted, self.event));
                match waited {
                    Ok(()) => {
                        working(false);
                        WaitForSingleObject(self.event, INFINITE);
                        stood(started);
                        working(true);
                    }
                    Err(e) => tracing::warn!(error = %e, "host submission not waited for"),
                }
            }
            // With timing on each group is one of `runs` (and the compare
            // of copies in the card's memory, with the last): every
            // command's time on the card goes to the sums by pipeline.
            #[cfg(feature = "diagnostics")]
            if timing() {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                for (ms, op) in runs[i].slowest(frequency, usize::MAX) {
                    crate::copystat::command_took(&op, ms);
                }
            }
            // A submission of one list was never named here (`groups.len()
            // > 1` was the test): the two 50 ms lists of the gameplay
            // frame, each a submission of its own, printed nothing.
            #[cfg(feature = "diagnostics")]
            if timing() && started.elapsed() > std::time::Duration::from_millis(50) {
                // SAFETY: a plain query.
                let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1);
                let commands = runs[i].notes.lock().unwrap_or_else(|e| e.into_inner()).len();
                tracing::warn!(ms = started.elapsed().as_millis(), commands, "slow host list");
                for (ms, op) in runs[i].slowest(frequency, 6) {
                    tracing::warn!("slow list command, {ms:.1} ms: {op}");
                }
            }
            let _ = (i, started);
        }
        let _ = &firsts;
        for list in runs {
            list.prof_submitted(*submitted);
        }
        FINISHED.store(this_submission, Ordering::Release);
        bury(this_submission);
        let gpu = prepared.elapsed();
        submission_us(0, (prepared - started).as_micros() as u64);
        submission_us(1, gpu.as_micros() as u64);
        #[cfg(feature = "diagnostics")]
        crate::copystat::card_waited(gpu.as_micros() as u64, || {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            let text = format!("{} lists, pipelines {pipelines:?}", runs.len());
            text.chars().take(1200).collect()
        });
        // The host resets a device whose submission runs for about 2 s.
        if gpu > std::time::Duration::from_millis(250) {
            let pipelines: std::collections::BTreeSet<String> = runs
                .iter()
                .flat_map(|l| l.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clone())
                .collect();
            tracing::warn!(ms = gpu.as_millis(), lists = runs.len(), ?pipelines, "slow host submission");
        }
        // SAFETY: a plain query.
        if unsafe { self.device.GetDeviceRemovedReason() }.is_err() {
            fail(format_args!(
                "host GPU device removed after a submission of {} lists{}{}",
                runs.len(),
                self.removal_data(runs),
                last_notes(runs)
            ));
        }
        submitting(3, runs.len());
        // What shaders wrote goes back before the bottom-of-pipe writes,
        // which tell the title the GPU is done; each range once per
        // submission, however many dispatches declared it (the world's
        // passes declare the same big ranges dozens of times).
        let copying = std::time::Instant::now();
        // For copies in the card's memory the compare has run there: the
        // pieces that differ are fetched and merged.
        if let Some(card) = &card {
            self.card_back(card, &mut submitted, &mut std::collections::HashSet::new());
        }
        let back = declared_writes(runs);
        let comparing = std::time::Instant::now();
        hand_back(&back);
        BACK_SPLIT[1].fetch_add(comparing.elapsed().as_micros() as u64, Ordering::Relaxed);
        if let Some(card) = card {
            self.card_done(card);
        }
        #[cfg(feature = "diagnostics")]
        if !back.is_empty() {
            crate::copystat::add(crate::copystat::C::BackSubmissions, 1);
        }
        // Nothing of this submission goes back to the title: no range of
        // shader writes, no value written when it is done. The wait for
        // such a one is only for order and for what the card still holds.
        #[cfg(feature = "diagnostics")]
        if back.is_empty() && runs.iter().all(|l| l.deferred().writes.is_empty()) {
            use crate::copystat::{C, add};
            add(C::NothingBack, 1);
            add(C::NothingBackUs, gpu.as_micros() as u64);
        }
        copy_back(0, copying.elapsed().as_micros() as u64);
        for list in runs {
            // Read backs first: a value written at the bottom of the pipe
            // may be what tells the title that they are there.
            read_backs(list);
            for &(at, what, bytes) in &list.deferred().writes {
                what.write(at as *mut u8, bytes);
            }
        }
        self.log_messages();
        submitting(0, 0);
        #[cfg(feature = "diagnostics")]
        crate::copystat::submission_ends(lock);
        // With the relay: nothing is in the air now, so no fetch still to
        // be merged is older than the sends to card copies so far, and
        // their notes can go.
        if relay() {
            crate::shadow::relay_fetched(*QUEUE_ORDER.lock().unwrap_or_else(|e| e.into_inner()));
        }
        mark.to(work::SEND);
        working(was_working);
        None
    }

    /// With `VICTORY_D3D12_DRED`: the faulting address and the allocations
    /// it lies in (by the names we give them) or the title memory copy just
    /// below it; for each list that did not finish, the command it stopped
    /// at and what the draws and dispatches in flight there run (or, in
    /// lists that never started, the pipelines they bind).
    fn removal_data(&self, runs: &[&HostList]) -> String {
        let Ok(dred) = self.device.cast::<ID3D12DeviceRemovedExtendedData1>() else {
            return String::new();
        };
        let mut text = String::new();
        let mut fault_va = None;
        // SAFETY: the outputs are the runtime's, valid while the device
        // lives; nodes are walked as linked, histories have their counts.
        unsafe {
            if let Ok(fault) = dred.GetPageFaultAllocationOutput1()
                && fault.PageFaultVA != 0
            {
                let va = fault.PageFaultVA;
                fault_va = Some(va);
                text.push_str(&format!("; page fault at {va:#x}"));
                for (what, head) in [
                    ("inside", fault.pHeadExistingAllocationNode),
                    ("inside freed", fault.pHeadRecentFreedAllocationNode),
                ] {
                    let mut names = Vec::new();
                    let mut node = head;
                    while let Some(n) = node.as_ref()
                        && names.len() < 8
                    {
                        let name = if n.ObjectNameW.is_null() {
                            "unnamed".to_owned()
                        } else {
                            n.ObjectNameW.to_string().unwrap_or_default()
                        };
                        names.push(format!("{name} (type {})", n.AllocationType.0));
                        node = n.pNext;
                    }
                    if !names.is_empty() {
                        text.push_str(&format!("; {what} {names:?}"));
                    }
                }
                text.push_str(&crate::shadow::near(va));
                text.push_str(&crate::descriptor::host::near_heaps(va));
            }
            // The plain output: the extended one (with marker texts) came
            // back without stopped lists on this host.
            let crumbs = match dred.GetAutoBreadcrumbsOutput() {
                Ok(crumbs) => crumbs,
                Err(e) => {
                    text.push_str(&format!("; no breadcrumbs ({e})"));
                    return text;
                }
            };
            // Every list's progress (done/count), in the runtime's order.
            let mut progress = Vec::new();
            let mut node = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = node.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                let ours = p.pCommandList.as_ref().map(|l| l.as_raw());
                let index = runs.iter().position(|r| Some(r.list.as_raw()) == ours);
                progress.push(format!("{done}/{} {index:?}", p.BreadcrumbCount));
                node = p.pNext;
            }
            text.push_str(&format!("; breadcrumbs (done/count, list) {progress:?}"));
            let mut plain = crumbs.pHeadAutoBreadcrumbNode;
            while let Some(p) = plain.as_ref() {
                let done = p.pLastBreadcrumbValue.as_ref().copied().unwrap_or(0);
                if done < p.BreadcrumbCount && !p.pCommandHistory.is_null() {
                    let history =
                        std::slice::from_raw_parts(p.pCommandHistory, p.BreadcrumbCount as usize);
                    text.push_str(&format!(
                        "; a list stopped at command {done} of {} (op {})",
                        p.BreadcrumbCount, history[done as usize].0
                    ));
                    let list = p.pCommandList.as_ref().map(|l| l.as_raw());
                    let run = runs.iter().find(|r| Some(r.list.as_raw()) == list);
                    // The k-th marker command is the k-th note.
                    #[cfg(feature = "diagnostics")]
                    let markers: Vec<(u32, String)> = run.map_or_else(Vec::new, |r| {
                        let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
                        history
                            .iter()
                            .enumerate()
                            .filter(|(_, op)| **op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
                            .zip(notes.iter())
                            .map(|((at, _), (note, range))| {
                                let place = fault_va.zip(*range).map(|(va, r)| fault_in(va, r));
                                (at as u32, format!("{note}{}", place.unwrap_or_default()))
                            })
                            .collect()
                    });
                    #[cfg(not(feature = "diagnostics"))]
                    let markers: Vec<(u32, String)> = Vec::new();
                    #[cfg(not(feature = "diagnostics"))]
                    let _ = fault_va;
                    if done > 0 {
                        text.push_str(&in_flight(history, &markers, done as usize));
                    } else if let Some(r) = run {
                        let pipelines = r.pipelines.lock().unwrap_or_else(|e| e.into_inner());
                        text.push_str(&format!(", its pipelines {pipelines:?}"));
                    }
                }
                plain = p.pNext;
            }
        }
        text
    }

    /// With the frame line (`VICTORY_D3D12_DEBUG=1`, diagnostics builds):
    /// the debug layer's messages since the last call go to our log, the
    /// first five of each kind in full, the rest counted, with one line of
    /// the counts by kind.
    ///
    /// They were only read when the device was lost ([`fail`]): a run that
    /// survived (run 205) said nothing of what the layer had found, and
    /// what it finds is what a driver's fault would have to be traced back
    /// to (an object let go while the card uses it, a resource in the
    /// wrong state, a view of something gone).
    pub(crate) fn drain_messages(&self) {
        let Some(queue) = &self.messages else {
            return;
        };
        static SEEN: Mutex<std::collections::BTreeMap<i32, (u64, i32)>> = Mutex::new(std::collections::BTreeMap::new());
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        let stored = unsafe {
            let stored = queue.GetNumStoredMessages();
            for i in 0..stored {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let kind = seen.entry(m.ID.0).or_insert((0, m.Severity.0));
                    kind.0 += 1;
                    if kind.0 <= 5 {
                        let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let text = String::from_utf8_lossy(text);
                        let text = text.trim_end_matches('\0');
                        tracing::warn!(id = m.ID.0, severity = m.Severity.0, nth = kind.0, "D3D12 debug layer: {text}");
                    }
                }
            }
            queue.ClearStoredMessages();
            stored
        };
        if stored > 0 {
            let counts: Vec<String> = seen.iter().map(|(id, (n, severity))| format!("id {id} (severity {severity}) x{n}")).collect();
            tracing::info!(
                "D3D12 debug layer messages so far by kind (severity 0 corruption, 1 error, 2 warning, 3 info, 4 message): {}",
                counts.join(", ")
            );
        }
    }

    /// The host device (`crate::cardmeter`).
    pub(crate) fn device(&self) -> &ID3D12Device {
        &self.device
    }

    /// Ticks a second of the queue's timestamps; 0 if it has none.
    pub(crate) fn timestamp_frequency(&self) -> u64 {
        // SAFETY: a plain query.
        unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(0)
    }

    /// Moves the debug layer's messages into our log; returns the first
    /// error among them.
    pub(crate) fn log_messages(&self) -> Option<String> {
        let Some(queue) = &self.messages else {
            return None;
        };
        let mut first_error = None;
        // SAFETY: each message is read into a buffer of the size the queue
        // reports, then the queue is emptied.
        unsafe {
            for i in 0..queue.GetNumStoredMessages() {
                let mut size = 0;
                if queue.GetMessage(i, None, &mut size).is_err() || size == 0 {
                    continue;
                }
                let mut buffer = vec![0u64; size.div_ceil(8)];
                let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
                if queue.GetMessage(i, Some(message), &mut size).is_ok() {
                    let m = &*message;
                    let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                    let text = String::from_utf8_lossy(text);
                    let text = text.trim_end_matches('\0');
                    tracing::warn!(id = m.ID.0, severity = m.Severity.0, "D3D12: {text}");
                    if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 && first_error.is_none() {
                        first_error = Some(text.to_owned());
                    }
                }
            }
            queue.ClearStoredMessages();
        }
        first_error
    }

    /// A host render target view of `resource`, valid until it is recorded.
    pub(crate) fn rtv(
        &self,
        resource: Option<&ID3D12Resource>,
        desc: Option<&D3D12_RENDER_TARGET_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_rtv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource, or none with a
        // description (a typed null view).
        unsafe {
            let mut handle = self.rtvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.rtv_size) as usize;
            self.device
                .CreateRenderTargetView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }

    /// A host depth view of `resource`, valid until it is recorded.
    pub(crate) fn dsv(
        &self,
        resource: &ID3D12Resource,
        desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>,
    ) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        let slot = self.next_dsv.fetch_add(1, Ordering::Relaxed) % RTV_RING;
        // SAFETY: a slot of our heap and a live resource.
        unsafe {
            let mut handle = self.dsvs.GetCPUDescriptorHandleForHeapStart();
            handle.ptr += (slot * self.dsv_size) as usize;
            self.device
                .CreateDepthStencilView(resource, desc.map(|d| d as *const _), handle);
            handle
        }
    }
}

impl HostList {
    /// The timestamp heap and its read-back buffer (diagnostics, timing on).
    #[cfg(feature = "diagnostics")]
    fn stamps(&self) -> Option<&(ID3D12QueryHeap, ID3D12Resource)> {
        if !timing() {
            return None;
        }
        self.stamps
            .get_or_init(|| {
                let device = &host()?.device;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(STAMPS) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation.
                unsafe {
                    device
                        .CreateQueryHeap(
                            &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: STAMPS, NodeMask: 0 },
                            &mut heap,
                        )
                        .ok()?;
                    device
                        .CreateCommittedResource(
                            &readback,
                            D3D12_HEAP_FLAG_NONE,
                            &desc,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            None,
                            &mut back,
                        )
                        .ok()?;
                }
                Some((heap?, back?))
            })
            .as_ref()
    }

    /// The noted commands that took longest on the GPU, longest first, as
    /// (milliseconds, text) (diagnostics, timing on, after the list ran).
    #[cfg(feature = "diagnostics")]
    fn slowest(&self, frequency: u64, n: usize) -> Vec<(f64, String)> {
        let notes = self.notes.lock().unwrap_or_else(|e| e.into_inner());
        let (Some((_, back)), count) = (self.stamps(), notes.len().min(STAMPS as usize - 1)) else {
            return Vec::new();
        };
        let mut data = std::ptr::null_mut();
        let range = D3D12_RANGE { Begin: 0, End: (count + 1) * 8 };
        // SAFETY: a read-back buffer the finished list resolved into.
        let stamps: Vec<u64> = unsafe {
            if back.Map(0, Some(&range), Some(&mut data)).is_err() {
                return Vec::new();
            }
            let s = std::slice::from_raw_parts(data.cast::<u64>(), count + 1).to_vec();
            back.Unmap(0, None);
            s
        };
        let mut times: Vec<(f64, String)> = (0..count)
            .map(|i| {
                let ticks = stamps[i + 1].saturating_sub(stamps[i]);
                (ticks as f64 * 1e3 / frequency.max(1) as f64, notes[i].0.chars().take(300).collect())
            })
            .collect();
        times.sort_by(|a, b| b.0.total_cmp(&a.0));
        times.truncate(n);
        times
    }

    pub(crate) fn close(&self) {
        // A wait owed (`VICTORY_BARRIER_LAZY_UAV`): what the list read goes
        // before what later lists write.
        if self.bar.closing() {
            self.barrier(crate::barstat::Why::Owed, &[crate::barstat::uav_all()]);
        }
        self.twins_to_depth();
        self.bar.closed();
        self.prof_finish();
        let scratch = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner()).clone();
        if let Some((scratch, _)) = scratch {
            self.indirect_common(&scratch);
        }
        self.card_budget_out();
        crate::cardmeter::end(&self.list, crate::cardmeter::Span::from_raw(self.pipeline_meter.swap(u32::MAX, Ordering::Relaxed)));
        crate::cardmeter::end_named(
            &self.list,
            crate::cardmeter::Span::from_raw(self.meter.swap(u32::MAX, Ordering::Relaxed)),
            &self.pipelines.lock().unwrap_or_else(|e| e.into_inner()),
        );
        #[cfg(feature = "diagnostics")]
        if let Some((heap, back)) = self.stamps() {
            let count = (self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32).min(STAMPS - 1);
            // SAFETY: an open list, our heap and read-back buffer.
            unsafe {
                self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, count);
                self.list.ResolveQueryData(heap, D3D12_QUERY_TYPE_TIMESTAMP, 0, count + 1, back, 0);
            }
        }
        // SAFETY: our list.
        if let Err(e) = unsafe { self.list.Close() } {
            fail(format_args!(
                "a host command list did not close ({e}): a command recorded into it was invalid"
            ));
        }
    }

    /// Only after the list's last submission finished (every one is waited
    /// for).
    pub(crate) fn reset(&self) {
        #[cfg(feature = "diagnostics")]
        let started = std::time::Instant::now();
        self._alive.used();
        self.bar.reset();
        *self.deferred() = Deferred::default();
        if crate::gpuprof::on() {
            self.prof.lock().unwrap_or_else(|e| e.into_inner()).clear();
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.twins.lock().unwrap_or_else(|e| e.into_inner()).clear();
        self.pipelines.lock().unwrap_or_else(|e| e.into_inner()).clear();
        #[cfg(feature = "diagnostics")]
        self.notes.lock().unwrap_or_else(|e| e.into_inner()).clear();
        {
            let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            (scratch.current, scratch.used) = (0, 0);
            // Every submission of the list has run: nothing reads its
            // scratch memory any more, and the next list to ask takes it.
            if scratch_pool() {
                give_chunks(std::mem::take(&mut scratch.chunks));
            }
        }
        self.card_ring.lock().unwrap_or_else(|e| e.into_inner()).rewind();
        self.batch_args().rewind();
        // SAFETY: the list is not executing.
        let reset = unsafe {
            self.allocator
                .Reset()
                .and_then(|()| self.list.Reset(&self.allocator, None))
        };
        if let Err(e) = reset {
            fail(format_args!("a host command list did not reset ({e})"));
        }
        if let Some(host) = crate::gpuprof::on().then(host).flatten() {
            self.prof_start(host);
        }
        self.meter
            .store(crate::cardmeter::begin(&self.list, crate::cardmeter::Kind::Lists).raw(), Ordering::Relaxed);
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::ResetNs, started.elapsed().as_nanos() as u64);
    }

    /// Moves a title texture's host copy to `wanted` (a host state) in this
    /// list: the host texture and the state it leaves, if that is a change.
    /// The list's first use of a texture records nothing: its prelude brings
    /// the texture there at submission.
    pub(crate) fn transition(
        &self,
        resource: crate::object::This,
        wanted: u32,
    ) -> Option<(&'static ID3D12Resource, u32, u32)> {
        let texture = crate::resource::host_resource(resource)?;
        let copy = crate::resource::host_copy(resource)?;
        let wanted = legal_state(texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        match states.entry(std::sync::Arc::as_ptr(&copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                // `VICTORY_BARRIER_STICKY_READS`: a read state that holds
                // the one wanted stays (or both are taken).
                let before = o.get().1;
                let after = crate::barstat::sticky(before, wanted)?;
                o.get_mut().1 = after;
                Some((texture, before, after))
            }
        }
    }

    /// Records `barriers` into the list, counted under `why`.
    pub(crate) fn barrier(&self, why: crate::barstat::Why, barriers: &[D3D12_RESOURCE_BARRIER]) {
        if barriers.is_empty() {
            return;
        }
        self.bar.recorded(barriers);
        crate::barstat::record(&self.list, why, barriers);
    }

    /// A wait for every earlier shader write is asked for (`why`): recorded,
    /// or owed with `VICTORY_BARRIER_LAZY_UAV` when the list has no write
    /// since its last such wait ([`crate::barstat`]). `with`: barriers of
    /// the same call recorded either way (in one call with the wait).
    pub(crate) fn wait_for_writes(&self, why: crate::barstat::Why, with: &[D3D12_RESOURCE_BARRIER]) {
        if self.bar.ask() {
            let mut all = with.to_vec();
            all.push(crate::barstat::uav_all());
            self.barrier(why, &all);
        } else {
            self.barrier(why, with);
        }
    }

    /// The title's barrier asks for a wait for every earlier shader write:
    /// whether it is recorded (false: owed, [`crate::barstat`]).
    pub(crate) fn wait_asked(&self) -> bool {
        self.bar.ask()
    }

    /// Something that may write (a title copy, clear, resolve) is about to
    /// be recorded: a wait owed goes first ([`crate::barstat`]).
    pub(crate) fn may_write(&self) {
        if self.bar.write() {
            self.barrier(crate::barstat::Why::Owed, &[crate::barstat::uav_all()]);
            self.bar.write();
        }
    }

    /// The pipeline bound for the next draws or dispatches can write through
    /// unordered access ([`crate::barstat`]).
    pub(crate) fn bound_writes(&self, writes: bool) {
        self.bar.bound(writes);
    }

    /// One of our passes recorded on the raw list `calls` barrier calls and
    /// ends with a wait for every shader write, its own included.
    pub(crate) fn fenced_by_pass(&self, calls: u32) {
        self.bar.add_calls(calls);
        self.bar.fenced();
    }

    /// Moves host texture `copy` to `wanted` (a host state) in this list,
    /// as [`HostList::transition`] does a title resource's host copy, and
    /// records the barrier: for textures that are not a title resource's
    /// host copy (colour twins) and for the copies between the two.
    fn settle(&self, copy: &std::sync::Arc<crate::resource::HostTexture>, wanted: u32) {
        let wanted = legal_state(&copy.texture, wanted);
        let mut states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        let before = match states.entry(std::sync::Arc::as_ptr(copy) as usize) {
            std::collections::hash_map::Entry::Vacant(v) => {
                v.insert((wanted, wanted, copy.clone()));
                None
            }
            std::collections::hash_map::Entry::Occupied(mut o) => {
                let before = std::mem::replace(&mut o.get_mut().1, wanted);
                (before != wanted).then_some(before)
            }
        };
        drop(states);
        if let Some(before) = before {
            let barrier = transition(
                &copy.texture,
                D3D12_RESOURCE_STATES(before as i32),
                D3D12_RESOURCE_STATES(wanted as i32),
            );
            self.barrier(crate::barstat::Why::Twin, &[barrier]);
        }
    }

    /// A use of title texture `resource` about to be recorded, if it is a
    /// depth texture with a colour twin (false if it is not): of the twin
    /// (`colour`: unordered access) or of the depth texture (depth target,
    /// clear, shader read, copy, resolve), writing it or only reading. The
    /// side used is first brought up to date: by a copy recorded here if
    /// this list wrote the other side since, by the list's prelude if the
    /// list uses the side as it finds it. The twin is left in the
    /// unordered access state; the caller moves the depth texture to the
    /// state its use needs, as for any texture. A twin still ahead when the
    /// list closes is copied then (`twins_to_depth`).
    /// shortcut: every depth target bound for a draw counts as written
    /// (also when the draw only tests), and every unordered access view a
    /// shader names as written; views a shader picks from the heap by a
    /// computed index are not seen here at all.
    pub(crate) fn twin_use(
        &self,
        resource: crate::object::This,
        colour: bool,
        write: bool,
    ) -> bool {
        const UNORDERED_ACCESS: u32 = 0x8;
        // Asked for every texture a draw reads: most have no twin.
        let Some(twin) = crate::resource::twin(resource) else {
            return false;
        };
        let Some(depth) = crate::resource::host_copy(resource) else {
            return false;
        };
        let side = usize::from(colour);
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let u = twins
            .entry(std::sync::Arc::as_ptr(&twin) as usize)
            .or_insert_with(|| TwinUse {
                depth: depth.clone(),
                twin: twin.clone(),
                needs: [false; 2],
                behind: [None; 2],
                writing: false,
                late: false,
            });
        let copy = match u.behind[side] {
            Some(behind) => behind,
            None => {
                u.needs[side] = true;
                false
            }
        };
        u.behind[side] = Some(false);
        match (write, colour) {
            // Shaders write the twin when the draw or dispatch runs: what
            // the same one reads of the depth texture is prepared before
            // (`twins_written`).
            (true, true) => u.writing = true,
            (true, false) => u.behind[1] = Some(true),
            (false, _) => {}
        }
        // shortcut: the depth plane's twin is not drawn into the plane
        // here: the pass sets a pipeline and a target of ours, which a list
        // still recording the title's commands cannot take. This use sees
        // the plane as it was; the pass runs when the list closes
        // (`twins_to_depth`). Counted.
        if copy && !colour && twin.plane.is_some() {
            u.late = true;
            crate::resource::PLANES[2].fetch_add(1, Ordering::Relaxed);
        }
        drop(twins);
        if copy {
            self.twin_copy(&depth, &twin, colour);
        } else if colour {
            self.settle(&twin.colour, UNORDERED_ACCESS);
        }
        // The depth plane's twin is written through unordered access too.
        if colour && let Some(plane) = &twin.plane {
            self.settle(plane, UNORDERED_ACCESS);
        }
        true
    }

    /// Records the pass that draws the twin of the depth plane of `depth`
    /// into the plane, if `twin` has one, with the moves to the states the
    /// pass needs; the twin is left in the unordered access state. Only
    /// for a list that records nothing of the title's afterwards
    /// (`record_plane_pass`).
    fn pass_plane(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const DEPTH_WRITE: u32 = 0x10;
        const PIXEL_SHADER_RESOURCE: u32 = 0x80;
        let Some(plane) = &twin.plane else { return };
        self.settle(depth, DEPTH_WRITE);
        self.settle(plane, PIXEL_SHADER_RESOURCE);
        record_plane_pass(&self.list, &depth.texture, &plane.texture);
        self.settle(plane, UNORDERED_ACCESS);
    }

    /// Records the copy of a depth texture into its colour twin (`colour`)
    /// or of the twin into the depth texture, with the moves to the copy
    /// states. The twin is left in the unordered access state.
    fn twin_copy(
        &self,
        depth: &std::sync::Arc<crate::resource::HostTexture>,
        twin: &crate::resource::Twin,
        colour: bool,
    ) {
        const UNORDERED_ACCESS: u32 = 0x8;
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let (from, to) = if colour {
            (depth, &twin.colour)
        } else {
            (&twin.colour, depth)
        };
        self.settle(from, COPY_SOURCE);
        self.settle(to, COPY_DEST);
        record_twin_copy(&self.list, &depth.texture, twin, colour);
        self.settle(&twin.colour, UNORDERED_ACCESS);
    }

    /// Before the list closes: the depth textures whose twins it wrote last
    /// are given what was written, and left readable by shaders. Shaders
    /// that pick a view from the heap by a computed index are not seen by
    /// `twin_use` and read the depth texture in lists that never name it
    /// (in a captured frame of the title no table names a shadow
    /// map for reading but the pass that fills the next one).
    fn twins_to_depth(&self) {
        const SHADER_RESOURCE: u32 = 0x40 | 0x80;
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        let written: Vec<_> = twins
            .values_mut()
            .filter(|u| u.behind[0] == Some(true) || u.late)
            .map(|u| {
                // Not copied again if the list copied since the last write
                // and only the depth plane's pass is left.
                let copy = u.behind[0] == Some(true);
                u.behind[0] = Some(false);
                u.late = false;
                (u.depth.clone(), u.twin.clone(), copy)
            })
            .collect();
        drop(twins);
        for (depth, twin, copy) in &written {
            if *copy {
                self.twin_copy(depth, twin, false);
            }
            // Nothing of the title's is recorded after this: the pass may
            // leave its own pipeline and target set.
            self.pass_plane(depth, twin);
            self.settle(depth, SHADER_RESOURCE);
        }
    }

    /// The draw or dispatch just prepared is recorded: the twins it writes
    /// now hold what their depth textures lack.
    pub(crate) fn twins_written(&self) {
        let mut twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values_mut().filter(|u| u.writing) {
            u.writing = false;
            u.behind[0] = Some(true);
        }
    }

    /// At submission: the list that moves each texture this list uses from
    /// its real state to the one the list starts with; the textures' real
    /// states become those the list ends with. For depth textures with a
    /// colour twin it first copies the side the list uses as it finds it
    /// from the other, if a list submitted earlier left it behind.
    fn prelude(&self) -> Option<ID3D12CommandList> {
        const COPY_DEST: u32 = 0x400;
        const COPY_SOURCE: u32 = 0x800;
        let host_state = |state: u32| D3D12_RESOURCE_STATES(state as i32);
        let states = self.states.lock().unwrap_or_else(|e| e.into_inner());
        // The copies (depth texture, twin, whether into the twin); the
        // sides are then as this list leaves them.
        let mut syncs = Vec::new();
        let twins = self.twins.lock().unwrap_or_else(|e| e.into_inner());
        for u in twins.values() {
            let real = u.twin.behind.load(Ordering::Relaxed);
            let mut behind = [real & 1 != 0, real & 2 != 0];
            for side in [0, 1] {
                if u.needs[side] && behind[side] {
                    syncs.push((u.depth.clone(), u.twin.clone(), side == 1));
                    behind = [false; 2];
                }
            }
            let left = [0, 1].map(|side| u.behind[side].unwrap_or(behind[side]));
            let left = u32::from(left[0]) | (u32::from(left[1]) << 1);
            u.twin.behind.store(left, Ordering::Relaxed);
        }
        drop(twins);
        // A copy runs with its textures in the copy states: each moves
        // there from its real state, and from there to the state the list
        // starts with (back to its real state if the list does not use it).
        let mut to_copy = Vec::new();
        let mut barriers = Vec::new();
        let mut copying = std::collections::HashMap::new();
        for (depth, twin, colour) in &syncs {
            let (from, to) = if *colour {
                (depth, &twin.colour)
            } else {
                (&twin.colour, depth)
            };
            for (texture, state) in [(from, COPY_SOURCE), (to, COPY_DEST)] {
                let key = std::sync::Arc::as_ptr(texture) as usize;
                let real = texture.state.load(Ordering::Relaxed);
                copying.insert(key, state);
                if real != state {
                    to_copy.push(transition(
                        &texture.texture,
                        host_state(real),
                        host_state(state),
                    ));
                    if !states.contains_key(&key) {
                        barriers.push(transition(
                            &texture.texture,
                            host_state(state),
                            host_state(real),
                        ));
                    }
                }
            }
        }
        barriers.extend(states.iter().filter_map(|(key, (first, last, copy))| {
            let real = copy.state.swap(*last, Ordering::Relaxed);
            let real = copying.get(key).copied().unwrap_or(real);
            (real != *first)
                .then(|| transition(&copy.texture, host_state(real), host_state(*first)))
        }));
        if barriers.is_empty() && syncs.is_empty() {
            return None;
        }
        let Some((allocator, list)) = self
            .prelude
            .get_or_init(|| host().and_then(|h| h.make_prelude()))
            .as_ref()
        else {
            fail(format_args!("no host list for a state prelude"));
        };
        // SAFETY: the prelude is not executing (every submission is waited
        // for); the textures are live.
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                if !to_copy.is_empty() {
                    crate::barstat::record(&list, crate::barstat::Why::Prelude, &to_copy);
                }
                for (depth, twin, colour) in &syncs {
                    record_twin_copy(list, &depth.texture, twin, *colour);
                }
                if !barriers.is_empty() {
                    crate::barstat::record(&list, crate::barstat::Why::Prelude, &barriers);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("a host state prelude did not record ({e})"));
        }
        list.cast().ok()
    }

    /// Notes what the command about to be recorded does, as a marker
    /// command whose text the list keeps (diagnostics, with device-removed
    /// data on): the report pairs the breadcrumbs' marker commands with the
    /// texts in order and names the commands a fault stopped among.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_op(&self, what: impl FnOnce() -> String) {
        self.note(what, None);
    }

    /// As [`HostList::note_op`], for a copy that reads the host GPU range
    /// `(address, bytes)`: the report places the faulting address in it.
    #[cfg(feature = "diagnostics")]
    pub(crate) fn note_read(&self, what: impl FnOnce() -> String, range: (u64, u64)) {
        self.note(what, Some(range));
    }

    #[cfg(feature = "diagnostics")]
    fn note(&self, what: impl FnOnce() -> String, range: Option<(u64, u64)>) {
        if !dred_on() {
            crate::capture::op(std::ptr::from_ref(self) as usize, what);
            return;
        }
        {
            let started = std::time::Instant::now();
            let at = self.notes.lock().unwrap_or_else(|e| e.into_inner()).len() as u32;
            if let Some((heap, _)) = self.stamps() && at < STAMPS - 1 {
                // SAFETY: an open list and our query heap.
                unsafe { self.list.EndQuery(heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            }
            let what = what();
            crate::capture::op(std::ptr::from_ref(self) as usize, || what.clone());
            let text: Vec<u16> = what.encode_utf16().chain([0]).collect();
            const UNICODE: u32 = 0;
            // SAFETY: an open list; the marker is copied when recorded.
            unsafe {
                self.list.SetMarker(UNICODE, Some(text.as_ptr().cast()), 2 * text.len() as u32)
            };
            self.notes
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .push((what, range));
            crate::copystat::add(crate::copystat::C::NoteNs, started.elapsed().as_nanos() as u64);
        }
    }

    /// Whether nothing recorded since the list was opened does any work:
    /// no draw or dispatch (the first keeps its pipeline), no copy, clear,
    /// upload or resolve (they defer a step or move a texture's state), no
    /// bottom-of-pipe write. State set on the list does not count.
    pub(crate) fn untouched(&self) -> bool {
        {
            let deferred = self.deferred();
            if !(deferred.copies.is_empty()
                && deferred.uploads.is_empty()
                && deferred.kept.is_empty()
                && deferred.textures.is_empty()
                && deferred.writes.is_empty()
                && deferred.backs.is_empty())
            {
                return false;
            }
        }
        self.states.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
            && self.twins.lock().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Notes a pipeline the list binds; called where it is bound.
    pub(crate) fn used_pipeline(&self, name: &str) {
        {
            let mut pipelines = self.pipelines.lock().unwrap_or_else(|e| e.into_inner());
            if !pipelines.contains(name) {
                pipelines.insert(name.to_owned());
            }
        }
        if crate::cardmeter::pipelines() {
            // The pipeline before it ends here on the card's clock.
            crate::cardmeter::end(&self.list, crate::cardmeter::Span::from_raw(self.pipeline_meter.swap(u32::MAX, Ordering::Relaxed)));
            self.pipeline_meter.store(crate::cardmeter::begin_named(&self.list, name).raw(), Ordering::Relaxed);
        }
    }

    /// `VICTORY_GPU_PROFILE`: the list opened (made or reset): its first
    /// timestamp.
    fn prof_start(&self, host: &Host) {
        let Some(stamps) = crate::gpuprof::stamps(&host.device, &host.queue) else {
            return;
        };
        if let Some(at) = self.prof.lock().unwrap_or_else(|e| e.into_inner()).start(stamps.pool) {
            // SAFETY: an open list and the profile's heap.
            unsafe { self.list.EndQuery(&stamps.heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            crate::barstat::TIMESTAMPS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// `VICTORY_GPU_PROFILE`: the pipeline the next draws or dispatches run
    /// (by host object), named by its shaders.
    pub(crate) fn prof_bound(&self, key: usize, name: &str) {
        if crate::gpuprof::on() {
            self.prof.lock().unwrap_or_else(|e| e.into_inner()).bound(key, name);
        }
    }

    /// `VICTORY_GPU_PROFILE`: just before a draw or dispatch is recorded
    /// (`after` false) and just after (true).
    pub(crate) fn prof_draw(&self, after: bool) {
        // Every draw and dispatch of the title's comes here first: one that
        // can write is preceded by a wait owed (`VICTORY_BARRIER_LAZY_UAV`).
        if !after && self.bar.draw() {
            self.may_write();
        }
        if !crate::gpuprof::on() {
            return;
        }
        let Some(stamps) = host().and_then(|h| crate::gpuprof::stamps(&h.device, &h.queue)) else {
            return;
        };
        let mut prof = self.prof.lock().unwrap_or_else(|e| e.into_inner());
        let at = if after { prof.after(stamps.pool) } else { prof.before(stamps.pool) };
        if let Some(at) = at {
            // SAFETY: an open list and the profile's heap.
            unsafe { self.list.EndQuery(&stamps.heap, D3D12_QUERY_TYPE_TIMESTAMP, at) };
            crate::barstat::TIMESTAMPS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// `VICTORY_GPU_PROFILE`: the list closes: its last timestamp, and all
    /// of them resolved into the read-back buffer.
    fn prof_finish(&self) {
        if !crate::gpuprof::on() {
            return;
        }
        let Some(stamps) = host().and_then(|h| crate::gpuprof::stamps(&h.device, &h.queue)) else {
            return;
        };
        let mut prof = self.prof.lock().unwrap_or_else(|e| e.into_inner());
        let (end, ranges) = prof.finish(stamps.pool);
        // SAFETY: an open list, the profile's heap and read-back buffer;
        // each range holds only timestamps this list wrote.
        unsafe {
            if let Some(at) = end {
                self.list.EndQuery(&stamps.heap, D3D12_QUERY_TYPE_TIMESTAMP, at);
                crate::barstat::TIMESTAMPS.fetch_add(1, Ordering::Relaxed);
            }
            for (first, count) in ranges {
                self.list.ResolveQueryData(
                    &stamps.heap,
                    D3D12_QUERY_TYPE_TIMESTAMP,
                    first,
                    count,
                    &stamps.back,
                    u64::from(first) * 8,
                );
            }
        }
    }

    /// `VICTORY_GPU_PROFILE`: the list goes to the card in the submission
    /// that signals `value`.
    fn prof_submitted(&self, value: u64) {
        if crate::gpuprof::on()
            && let Some(record) = self.prof.lock().unwrap_or_else(|e| e.into_inner()).record(value)
        {
            crate::gpuprof::submitted(record);
        }
    }

    fn deferred(&self) -> std::sync::MutexGuard<'_, Deferred> {
        self.deferred.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copies `len` bytes from `src` to `dst` when the list is submitted.
    pub(crate) fn defer_copy(&self, dst: *mut u8, src: *const u8, len: usize) {
        if (table_gpu_copy() >= 2 || copy_on_card() > 0) && self.title_copy_on_card(dst as usize, src as usize, len) {
            return;
        }
        let mut deferred = self.deferred();
        // Later table views of bytes it may write fill anew, at their place
        // ([`HostList::card_constant`]).
        deferred.card_slots.clear();
        // After a title copy on the card in this list ([`table_gpu_copy`]):
        // bytes it wrote are read at its source, and a copy into them is
        // counted (the landing puts the card's bytes back over it).
        if !deferred.gpu_copied.is_empty() {
            let (d0, d1) = (dst as usize, dst as usize + len);
            if deferred.gpu_copied.iter().any(|&(d, _, n)| d < d1 && d + n > d0) {
                table_gpu::count(table_gpu::CONFLICTS);
            }
            let mut pieces = Vec::new();
            through_card_copies(&deferred.gpu_copied, src as usize, len, 0, &mut pieces);
            if pieces.len() > 1 || pieces.first().is_some_and(|p| p.1 != src as usize) {
                table_gpu::count(table_gpu::REMAPPED);
                for (at, from, n) in pieces {
                    deferred.copies.push(Step::Copy(dst as usize + at, from, n));
                }
                deferred.refreshed.clear();
                return;
            }
        }
        COPIES_JOINED[0].fetch_add(1, Ordering::Relaxed);
        // A copy that goes on where the one recorded just before it ended,
        // source and destination both: with `join_copies` the two are one
        // step.
        if let Some(Step::Copy(d, s, n)) = deferred.copies.last_mut()
            && continues((*d, *s, *n), (dst as usize, src as usize, len))
        {
            COPIES_JOINED[1].fetch_add(1, Ordering::Relaxed);
            if join_copies() {
                *n += len;
                return;
            }
        }
        deferred
            .copies
            .push(Step::Copy(dst as usize, src as usize, len));
        deferred.refreshed.clear();
    }

    /// Copies `len` bytes of title memory at `src` into the list's upload
    /// memory at `dst` when the list is submitted; `kind` (a `FILL_` kind)
    /// says what for, for the frame line's count of settles.
    pub(crate) fn defer_fill(&self, dst: *mut u8, src: *const u8, len: usize, kind: u8) {
        let mut deferred = self.deferred();
        // Bytes a title copy on the card in this list wrote
        // ([`table_gpu_copy`]): read at its source.
        if !deferred.gpu_copied.is_empty() {
            let mut pieces = Vec::new();
            through_card_copies(&deferred.gpu_copied, src as usize, len, 0, &mut pieces);
            if pieces.len() > 1 || pieces.first().is_some_and(|p| p.1 != src as usize) {
                table_gpu::count(table_gpu::REMAPPED);
                for (at, from, n) in pieces {
                    deferred.copies.push(Step::Fill(dst as usize + at, from, n, kind));
                }
                return;
            }
        }
        deferred.copies.push(Step::Fill(dst as usize, src as usize, len, kind));
    }

    /// [`table_gpu_copy`]: the host address a constant view of a descriptor
    /// table (`size` bytes of title memory at `address`) is bound at, in a
    /// slot of the list's card memory. Its bytes are filled on the CPU as
    /// before, at this place in the list's steps, into scratch
    /// ([`Step::CardFill`]); a list run just before the list copies the
    /// scratch into the slot ([`HostList::card_pre`]), and over it the
    /// bytes that lie under flights' writes at submission, from the
    /// buffer's copy on the card ([`HostList::card_classify`]). `None` (the
    /// fill, as before) when the bytes lie outside buffers, two copies hold
    /// some of them, a copy in the PC's memory holds them, the list itself
    /// writes some of them before the draw (its shaders, or its title copies
    /// on the card: the copy would run ahead of them), or there is no card
    /// memory or scratch.
    pub(crate) fn card_constant(&self, address: u64, size: u32) -> Option<u64> {
        use table_gpu as g;
        if table_gpu_copy() == 0 || size == 0 {
            return None;
        }
        let len = u64::from(size);
        let (from, to) = (address as usize, (address + len) as usize);
        {
            let deferred = self.deferred();
            if deferred.gpu_copied.iter().any(|&(d, _, n)| d < to && d + n > from) {
                g::count(g::LIST_COPY);
                return None;
            }
            if deferred.declared.touches(from, to) {
                g::count(g::IN_LIST);
                return None;
            }
            // The same bytes, with nothing in the list writing them since
            // (its shaders: checked above; a title copy or a step of ours
            // forgets the slots): the same slot, whose fill read them.
            if let Some(&gpu) = deferred.card_slots.get(&(address, size)) {
                g::count(g::VIEWS);
                g::count(g::REUSED);
                return Some(gpu);
            }
        }
        let pieces = match crate::shadow::card_pieces(address, len) {
            Ok(pieces) => pieces,
            Err(crate::shadow::NotSole::NoBuffer) => {
                g::count(g::NO_BUFFER);
                return None;
            }
            Err(crate::shadow::NotSole::OtherCopy) => {
                g::count(g::OTHER_COPY);
                return None;
            }
        };
        // A copy in the PC's memory is written by later submissions'
        // preparing at once, maybe before the card has copied out of it.
        if pieces.iter().any(|(shadow, ..)| !shadow.carded()) {
            g::count(g::PC_COPY);
            return None;
        }
        let Some((ring, ring_at, gpu)) = self.card_ring.lock().unwrap_or_else(|e| e.into_inner()).take(len) else {
            g::count(g::NO_RING);
            return None;
        };
        let Some((scratch, scratch_at, _, data)) = self.scratch_place(len) else {
            g::count(g::NO_RING);
            return None;
        };
        // The title's bytes go into the copies at submission: those under
        // flights' writes are copied from there, and the refreshes are what
        // the checks for title writes under flights' writes look at.
        for (shadow, at, n) in &pieces {
            shadow.refresh(self, *at, *n);
        }
        let mut deferred = self.deferred();
        g::count(g::VIEWS);
        if pieces.len() > 1 {
            g::count(g::SPLIT);
        }
        g::add(g::BYTES, len);
        let step = deferred.copies.len();
        deferred.copies.push(Step::CardFill(data as usize, from, size as usize));
        // The slot whole (scratch and slot are both 256-byte rounded), so
        // that views filled one after the other make one copy.
        let copy = Region { src: scratch, src_at: scratch_at, dst: ring.clone(), dst_at: ring_at, len: len.next_multiple_of(256) };
        if push_region(&mut deferred.card_copies, copy) {
            g::count(g::JOINED);
        } else {
            g::count(g::PIECES);
        }
        deferred.card_views.push(CardView {
            ring,
            ring_at,
            from,
            to,
            pieces: pieces.into_iter().map(|(shadow, at, n)| (shadow, at as usize, (at + n) as usize)).collect(),
            step,
            waits: false,
            on_card: Vec::new(),
        });
        deferred.gpu_reads.push((from, len as usize));
        deferred.card_slots.insert((address, size), gpu);
        Some(gpu)
    }

    /// At a submission ([`table_gpu_copy`]), before anything looks at what
    /// its preparing reads, with `written` the title memory flights'
    /// shaders may write (in order, joined): for each view at a slot, the
    /// bytes under those writes. Its fill reads them as title memory holds
    /// them before the flights land; the list ahead copies them over the
    /// fill from the buffer's copy on the card, which the flights wrote, in
    /// queue order after them. Where those writes are made in another copy
    /// of the memory (a texture's, a copy in the PC's memory), the view's
    /// copy does not hold them: its fill becomes one that waits for them,
    /// as before.
    fn card_classify(&self, written: &[(usize, usize)]) {
        use table_gpu as g;
        let mut deferred = self.deferred();
        let Deferred { card_views, copies, .. } = &mut *deferred;
        for view in card_views.iter_mut() {
            view.on_card.clear();
            if view.waits || !touches(written, view.from, view.to) {
                continue;
            }
            match view.under_writes(written, |i, lo, hi| view.pieces[i].0.pending_covers(lo, hi)) {
                Ok(parts) => {
                    if !parts.is_empty() {
                        g::count(g::CARD_VIEWS);
                    }
                    view.on_card = parts;
                }
                Err(()) => {
                    view.waits = true;
                    if let Some(step) = copies.get_mut(view.step)
                        && let Step::CardFill(dst, src, len) = *step
                    {
                        *step = Step::Fill(dst, src, len, FILL_TABLE);
                    }
                    g::count(g::ELSEWHERE);
                }
            }
        }
    }

    /// At submission, with [`HostList::card_constant`]'s slots: the list
    /// that fills them, just before this one (after its prelude). The
    /// fills' scratch goes into the slots; then, past a barrier (two copies
    /// into the same bytes are not ordered without one), the bytes under
    /// flights' writes from the buffers' copies, every shader write before
    /// the list first. The buffers it touched are back in the common state
    /// at its end, its writes seen by what follows (as the card sends'
    /// list leaves them, `Host::card_before`).
    fn card_pre(&self) -> Option<ID3D12CommandList> {
        use table_gpu as g;
        let deferred = self.deferred();
        if deferred.card_copies.is_empty() {
            return None;
        }
        let mut from_card: Vec<CardCopy> = Vec::new();
        for view in deferred.card_views.iter().filter(|v| !v.waits) {
            for &(i, lo, hi) in &view.on_card {
                let shadow = &view.pieces[i].0;
                g::add(g::CARD_BYTES, (hi - lo) as u64);
                let part = part_region(shadow.buffer.clone(), shadow.offset(lo as u64), view.ring.clone(), view.ring_at, view.from, lo, hi);
                push_region(&mut from_card, part);
            }
        }
        g::add(g::CARD_COPIES, from_card.len() as u64);
        let Some((allocator, list)) = self.card_pre.get_or_init(|| host().and_then(|h| h.make_prelude())).as_ref() else {
            fail(format_args!("no host list for the copies of table constants on the card (VICTORY_TABLE_GPU_COPY)"));
        };
        let unique = |copies: &[CardCopy], src: bool| -> Vec<ID3D12Resource> {
            let mut out: Vec<ID3D12Resource> = Vec::new();
            for copy in copies {
                let r = if src { &copy.src } else { &copy.dst };
                if !out.contains(r) {
                    out.push(r.clone());
                }
            }
            out
        };
        let back = |resources: &[ID3D12Resource], state: D3D12_RESOURCE_STATES| -> Vec<D3D12_RESOURCE_BARRIER> {
            resources.iter().map(|r| transition(r, state, D3D12_RESOURCE_STATE_COMMON)).collect()
        };
        let slots = back(&unique(&deferred.card_copies, false), D3D12_RESOURCE_STATE_COPY_DEST);
        let mut after = back(&unique(&from_card, false), D3D12_RESOURCE_STATE_COPY_DEST);
        after.extend(back(&unique(&from_card, true), D3D12_RESOURCE_STATE_COPY_SOURCE));
        // SAFETY: the list is not executing (every submission of this host
        // list has run: as its prelude); live buffers, held by the list
        // (scratch), its ring and its views (the copies).
        let recorded = unsafe {
            allocator.Reset().and_then(|()| {
                list.Reset(allocator, None)?;
                crate::barstat::record(&list, crate::barstat::Why::Prelude, &[uav_barrier_all()]);
                // Buffers are in the common state here (as the lists before
                // leave them or decay to it): the copies promote them. The
                // scratch is upload memory, in its one state.
                for copy in &deferred.card_copies {
                    list.CopyBufferRegion(&copy.dst, copy.dst_at, &copy.src, copy.src_at, copy.len);
                }
                crate::barstat::record(&list, crate::barstat::Why::Prelude, &slots);
                if !from_card.is_empty() {
                    for copy in &from_card {
                        list.CopyBufferRegion(&copy.dst, copy.dst_at, &copy.src, copy.src_at, copy.len);
                    }
                    crate::barstat::record(&list, crate::barstat::Why::Prelude, &after);
                }
                list.Close()
            })
        };
        if let Err(e) = recorded {
            fail(format_args!("the copies of table constants on the card did not record ({e})"));
        }
        g::count(g::PRE_LISTS);
        list.cast().ok()
    }

    /// [`table_gpu_copy`] at `2`: a title buffer copy (`len` bytes from
    /// `src` to `dst`, title addresses) recorded here, in the list's order,
    /// as a copy between the two buffers' host copies, when preparing would
    /// otherwise read bytes flights' shaders may write (or the list's own
    /// shaders, or an earlier copy on the card writes them). The source's
    /// and destination's title bytes go into the copies first (refreshes);
    /// the destination is declared as written by the list's shaders, so
    /// title memory gets it when the flight lands and what reads it waits
    /// for that. Whether it was; if not the caller copies on the CPU.
    fn title_copy_on_card(&self, dst: usize, src: usize, len: usize) -> bool {
        use table_gpu as g;
        if len == 0 {
            return false;
        }
        let (s1, d1) = (src + len, dst + len);
        let under = {
            let deferred = self.deferred();
            deferred.gpu_copied.iter().any(|&(d, _, n)| (d < s1 && d + n > src) || (d < d1 && d + n > dst))
                || deferred.declared.touches(src, s1)
        } || {
            let line = line();
            touches(&line.written, src, s1) || touches(&line.written, dst, d1)
        };
        // [`copy_on_card`] counts its own; at `1` a copy goes on the card
        // whether or not it lies under writes now.
        let own = copy_on_card();
        let left = |table: usize, card: usize| {
            if own > 0 {
                card_copy::count(card);
            } else {
                g::count(table);
            }
            false
        };
        if !under && own != 1 {
            return left(g::TITLE_FREE, card_copy::FREE);
        }
        let (Ok(from), Ok(to)) = (
            crate::shadow::sole_buffer_copy(src as u64, len as u64),
            crate::shadow::sole_buffer_copy(dst as u64, len as u64),
        ) else {
            return left(g::TITLE_NOT_SOLE, card_copy::NOT_SOLE);
        };
        if to.is_placed() {
            return left(g::TITLE_PLACED, card_copy::PLACED);
        }
        if std::sync::Arc::ptr_eq(&from, &to) {
            return left(g::TITLE_SAME, card_copy::SAME);
        }
        // The source's bytes must be in the card's memory, which only the
        // sends' list (queue order) writes: a copy in the PC's memory or an
        // upload heap takes later submissions' refreshes at once.
        if own > 0 && !from.carded() {
            card_copy::count(card_copy::NOT_CARD);
            return false;
        }
        from.refresh(self, src as u64, len as u64);
        to.refresh(self, dst as u64, len as u64);
        to.gpu_writes(self, dst as u64, len as u64);
        // It may write the records of indirect calls a run clamped already.
        self.batch_args().end_runs();
        #[cfg(feature = "diagnostics")]
        self.note_op(|| format!("title copy on the card {src:#x} to {dst:#x}, {len:#x} bytes"));
        // SAFETY: an open list; live buffers, held by the refresh steps.
        // Shader writes before it first; both buffers back in the common
        // state after, its writes seen by what follows.
        unsafe {
            self.barrier(crate::barstat::Why::TitleCopy, &[uav_barrier_all()]);
            self.list.CopyBufferRegion(&to.buffer, to.offset(dst as u64), &from.buffer, from.offset(src as u64), len as u64);
            self.barrier(crate::barstat::Why::TitleCopy, &[
                transition(&to.buffer, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COMMON),
                transition(&from.buffer, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COMMON),
            ]);
        }
        // A write the list's next waits are for (`VICTORY_BARRIER_LAZY_UAV`).
        self.may_write();
        let mut deferred = self.deferred();
        deferred.gpu_copied.push((dst, src, len));
        deferred.gpu_reads.push((src, len));
        deferred.gpu_reads.push((dst, len));
        if own > 0 {
            card_copy::count(card_copy::COPIES);
            card_copy::add(card_copy::BYTES, len as u64);
            if under {
                card_copy::count(card_copy::UNDER);
            }
        } else {
            g::count(g::TITLE_COPIES);
            g::add(g::TITLE_BYTES, len as u64);
        }
        true
    }

    /// [`table_on_card`]: a constant view of a descriptor table reads
    /// `len` bytes of title memory at `address` in a host copy (no fill).
    pub(crate) fn note_table_on_card(&self, address: usize, len: usize) {
        self.deferred().on_card.push((address, len));
    }

    /// Brings `[address, address + len)` of a buffer's host copy up to date
    /// when the list is submitted (once per range until a title copy
    /// intervenes), keeping the copy alive until the list is reset.
    pub(crate) fn defer_refresh(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        let key = (std::sync::Arc::as_ptr(shadow) as usize, address, len);
        if deferred.refreshed.insert(key) {
            if shadow.is_placed() {
                crate::inplace::served(len as u64);
            }
            deferred
                .copies
                .push(Step::Refresh(shadow.clone(), address, len));
        }
    }

    /// Shaders of the list write `[address, address + len)` of a buffer's
    /// host copy: copied back into title memory once the list has run.
    pub(crate) fn defer_gpu_write(
        &self,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        address: usize,
        len: usize,
    ) {
        let mut deferred = self.deferred();
        if table_gpu_copy() > 0 || copy_on_card() > 0 {
            deferred.declared.add(address, address + len);
        }
        deferred.copies.push(Step::GpuWrite(shadow.clone(), address, len));
    }

    /// A read back: once the list has run, `rows` rows of `row` bytes go
    /// from `buffer` (rows 256-byte pitched, see `PlacedFootprint::host_rows`)
    /// to title memory at `dst`, pitched `pitch`. The buffer lives until
    /// the list is reset.
    pub(crate) fn defer_read_back(&self, buffer: ID3D12Resource, dst: *mut u8, [row, rows, pitch]: [usize; 3]) {
        self.deferred().backs.push((buffer, dst as usize, [row, rows, pitch]));
    }

    /// Stores `what` (`bytes` of it) at `dst` once the list has run.
    pub(crate) fn defer_write(&self, dst: *mut u8, what: crate::queue::Written, bytes: usize) {
        self.deferred().writes.push((dst as usize, what, bytes));
    }

    /// Where the list's results for the title's processor go when it has
    /// run (`crate::results`): its bottom-of-pipe writes (values, query
    /// results among them, and timestamps) and its read-backs' first rows.
    pub(crate) fn result_spans(&self) -> Vec<(usize, u8)> {
        let deferred = self.deferred();
        let writes = deferred.writes.iter().map(|&(at, what, _)| {
            let kind = match what {
                crate::queue::Written::Value(_) => crate::results::VALUE,
                _ => crate::results::TIMESTAMP,
            };
            (at, kind)
        });
        let backs = deferred.backs.iter().map(|(_, at, _)| (*at, crate::results::READ_BACK));
        writes.chain(backs).collect()
    }

    /// Whether the list, once it has run, stores at `address` a value of
    /// `bytes` bytes or more for which `holds` says yes (one of its
    /// bottom-of-pipe writes, as [`late_met`] reads them of a flight).
    pub(crate) fn will_write(&self, address: usize, bytes: usize, holds: impl Fn(u64) -> bool) -> bool {
        self.deferred()
            .writes
            .iter()
            .any(|(at, what, n)| *at == address && *n >= bytes && matches!(what, crate::queue::Written::Value(v) if holds(*v)))
    }

    /// A mapped upload buffer of `size` bytes, alive until the list is
    /// reset.
    ///
    /// shortcut: one buffer per copy; pool them once uploads are frequent.
    pub(crate) fn upload(&self, host: &Host, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        let (buffer, data) = host.upload_buffer(size)?;
        name(&buffer, || format!("list upload {size:#x}"));
        self.deferred().hold_upload(buffer.clone(), size);
        Some((buffer, data))
    }

    /// `len` bytes of upload memory (256-byte aligned) until the list is
    /// reset: the host GPU address and the CPU pointer.
    pub(crate) fn scratch(&self, len: u64) -> Option<(u64, *mut u8)> {
        self.scratch_place(len).map(|(_, _, gpu, data)| (gpu, data))
    }

    /// As [`HostList::scratch`], with the buffer the bytes lie in and their
    /// offset there (for copies out of them on the card).
    fn scratch_place(&self, len: u64) -> Option<(ID3D12Resource, u64, u64, *mut u8)> {
        let host = host()?;
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let len = len.max(1).next_multiple_of(256);
        if len > SCRATCH_CHUNK {
            let (buffer, data) = host.upload_buffer(len)?;
            name(&buffer, || format!("list scratch {len:#x}"));
            // SAFETY: a live buffer.
            let gpu = unsafe { buffer.GetGPUVirtualAddress() };
            self.deferred().hold_upload(buffer.clone(), len);
            crate::bus::bound(crate::bus::kind::SCRATCH_PC, len);
            return Some((buffer, 0, gpu, data));
        }
        // The chunk in use while `len` fits in what is left of it, else the
        // next one (a list's first chunk may be a small one).
        while s.current < s.chunks.len() && s.used + len > s.chunks[s.current].1 {
            (s.current, s.used) = (s.current + 1, 0);
        }
        if s.current == s.chunks.len() {
            let size = if scratch_pool() && s.chunks.is_empty() && len <= SCRATCH_FIRST {
                SCRATCH_FIRST
            } else {
                SCRATCH_CHUNK
            };
            let chunk = take_chunk(host, size)?;
            s.chunks.push((chunk, size));
            s.used = 0;
        }
        let chunk = &s.chunks[s.current].0;
        let (data, gpu, buffer) = (chunk.data, chunk.gpu, chunk._buffer.clone());
        let on_card = chunk.card.is_some();
        crate::bus::bound(if on_card { crate::bus::kind::SCRATCH_CARD } else { crate::bus::kind::SCRATCH_PC }, len);
        let at = s.used;
        s.used += len;
        Some((buffer, at, gpu + at, (data + at as usize) as *mut u8))
    }

    /// The bytes of scratch chunks the list holds now (after a reset:
    /// none while lists share their chunks).
    pub(crate) fn scratch_kept(&self) -> u64 {
        let upload: u64 = {
            let scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
            scratch.chunks.iter().map(|c| c.1).sum()
        };
        // And what this fork's switches keep with the list in the card's
        // memory (`VICTORY_TABLE_GPU_COPY`'s ring, `VICTORY_INDIRECT_BATCH`'s
        // run buffer), so that upstream's pool of idle lists stays bounded
        // with them on.
        let ring = self.card_ring.lock().unwrap_or_else(|e| e.into_inner()).chunks.len() as u64 * CARD_RING_CHUNK;
        upload + ring + self.batch_args().kept()
    }

    /// The host address of the list's indirect-work budget, zeroed at each
    /// submission of the list from the first use after a reset on.
    pub(crate) fn budget(&self) -> Option<u64> {
        let (buffer, data) = self
            .budget
            .get_or_init(|| {
                let (b, d) = host()?.shared_buffer(256)?;
                Some((b, d as usize))
            })
            .as_ref()?;
        let card = self.card_budget.get_or_init(|| self.make_card_budget()).as_ref();
        let mut deferred = self.deferred();
        if !deferred.budget_zeroed {
            deferred.budget_zeroed = true;
            let at = *data;
            // SAFETY: the list's own mapped words, written before it runs.
            // The words after the budget hold what the clamp noted about the
            // list's last run (mesh dispatches: the largest group count
            // asked for, the groups asked in all; with the card's copy, as
            // the list's end copied them here): taken, then zeroed.
            deferred.copies.push(Step::Host(
                Box::new(move || unsafe {
                let words = at as *mut u32;
                let (largest, all) = (words.add(1).read_volatile(), words.add(2).read_volatile());
                MESH_GROUPS[0].fetch_max(u64::from(largest), Ordering::Relaxed);
                MESH_GROUPS[1].fetch_add(u64::from(all), Ordering::Relaxed);
                for k in 0..3 {
                    words.add(k).write_volatile(0);
                }
            }),
                // The list's own words: no staging buffer.
                false,
            ));
            drop(deferred);
            if let Some(card) = card {
                self.zero_card_budget(card);
            }
        }
        // SAFETY: a live buffer.
        Some(unsafe { card.unwrap_or(buffer).GetGPUVirtualAddress() })
    }

    /// The budget's words in the card's own memory ([`card_budget`]), if
    /// the list can zero them itself (`WriteBufferImmediate`).
    fn make_card_budget(&self) -> Option<ID3D12Resource> {
        if !card_budget() {
            return None;
        }
        self.list.cast::<ID3D12GraphicsCommandList2>().ok()?;
        let buffer = host()?.uav_buffer(256)?;
        name(&buffer, || "indirect budget".to_owned());
        Some(buffer)
    }

    /// Records, before the list's first clamp, the zeroing of the card's
    /// copy of the budget: common, written as a copy, then left for
    /// unordered access until the list closes ([`HostList::card_budget_out`]).
    fn zero_card_budget(&self, card: &ID3D12Resource) {
        let Ok(list2) = self.list.cast::<ID3D12GraphicsCommandList2>() else {
            return;
        };
        // SAFETY: a live buffer.
        let at = unsafe { card.GetGPUVirtualAddress() };
        let zeros = [0u64, 4, 8].map(|offset| D3D12_WRITEBUFFERIMMEDIATE_PARAMETER {
            Dest: at + offset,
            Value: 0,
        });
        // SAFETY: an open list; the buffer is the list's own and in the
        // common state at the start of every recording.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            list2.WriteBufferImmediate(zeros.len() as u32, zeros.as_ptr(), None);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_DEST,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            )]);
        }
    }

    /// Before the list closes, if this recording charged the card's copy
    /// of the budget: its notes on mesh records go to the mapped words
    /// (read at the next submission), and it goes back to common.
    fn card_budget_out(&self) {
        if !self.deferred().budget_zeroed {
            return;
        }
        let (Some(Some(card)), Some(Some((mapped, _)))) =
            (self.card_budget.get(), self.budget.get())
        else {
            return;
        };
        // SAFETY: an open list; both buffers are the list's own. The
        // mapped one is promoted from common to the copy destination.
        unsafe {
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            self.list.CopyBufferRegion(mapped, 4, card, 4, 8);
            self.list.ResourceBarrier(&[transition(
                card,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_COMMON,
            )]);
        }
    }

    /// A device-local unordered-access buffer of at least `size` bytes for
    /// clamped indirect arguments, reused across executions and resets of the
    /// list. Smaller buffers replaced mid-list stay alive until
    /// [`HostList::reset`].
    pub(crate) fn indirect_scratch(&self, size: u64) -> Option<ID3D12Resource> {
        let needed = size.max(64 << 10);
        let mut slot = self.indirect_args.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((buffer, cap)) = &*slot
            && *cap >= needed
        {
            return Some(buffer.clone());
        }
        let prev = slot.as_ref().map_or(0, |(_, cap)| *cap);
        let alloc = prev
            .saturating_mul(2)
            .max(needed)
            .next_multiple_of(64 << 10);
        let buffer = host()?.uav_buffer(alloc)?;
        name(&buffer, || format!("indirect arguments {alloc:#x}"));
        let old = slot.replace((buffer.clone(), alloc));
        drop(slot);
        if let Some((old_buf, _)) = old {
            // It ends the list common, as every scratch buffer does.
            self.indirect_common(&old_buf);
            self.keep(&old_buf);
        }
        Some(buffer)
    }

    /// The list's run buffer (`indirect::Run`).
    pub(crate) fn batch_args(&self) -> std::sync::MutexGuard<'_, crate::indirect::BatchArgs> {
        self.batch_args.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Moves `scratch` back to the common state if a clamp left it readable
    /// as indirect arguments ([`HostList::indirect_readable`]): where the
    /// list's first clamp expects it. Done before the list closes and before
    /// a bigger scratch buffer replaces it: one barrier per list (where
    /// there was one per indirect execution), and the list ends as it
    /// begins however it is submitted.
    fn indirect_common(&self, scratch: &ID3D12Resource) {
        {
            let mut deferred = self.deferred();
            if deferred.indirect_readable != scratch.as_raw() as usize {
                return;
            }
            deferred.indirect_readable = 0;
        }
        let barrier = transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        );
        // An open list and a live buffer this recording left in
        // `INDIRECT_ARGUMENT`; counted with the barriers by why.
        self.barrier(crate::barstat::Why::ClampDone, &[barrier]);
    }

    /// The state the list's indirect scratch buffer `scratch` is in at this
    /// point of the recording: indirect arguments if a clamp in this
    /// recording left it so ([`HostList::indirect_readable`]), otherwise
    /// common: the state a buffer is made in, and the one a recording
    /// leaves it in when it closes or replaces it ([`HostList::indirect_common`]).
    pub(crate) fn indirect_state(&self, scratch: &ID3D12Resource) -> D3D12_RESOURCE_STATES {
        if self.deferred().indirect_readable == scratch.as_raw() as usize {
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        } else {
            D3D12_RESOURCE_STATE_COMMON
        }
    }

    /// A clamp recorded in this list left `scratch` readable as indirect
    /// arguments. It stays so for every execution that reads it, until the
    /// next clamp moves it back to unordered access.
    pub(crate) fn indirect_readable(&self, scratch: &ID3D12Resource) {
        self.deferred().indirect_readable = scratch.as_raw() as usize;
    }

    /// The list's buffers for `crate::indirect::window_check`.
    pub(crate) fn window_check(&self) -> Option<&crate::indirect::WindowCheck> {
        self.window_check.get_or_init(crate::indirect::WindowCheck::new).as_ref()
    }

    /// Keeps `object` alive until the list is reset.
    pub(crate) fn keep(&self, object: &impl Interface) {
        if let Ok(object) = object.cast::<windows::core::IUnknown>() {
            self.deferred().kept.push(object);
        }
    }

    /// Whether this is the first time since the reset that the list asks
    /// about `key` (a texture to bring up to date).
    pub(crate) fn first_use(&self, key: usize) -> bool {
        self.deferred().textures.insert(key)
    }

    /// Runs `work` when the list is submitted, in order with its copies.
    /// `work` may write title memory (refreshes after it copy again).
    pub(crate) fn defer(&self, work: impl Fn() + Send + 'static) {
        let mut deferred = self.deferred();
        deferred.copies.push(Step::Run(Box::new(work)));
        deferred.refreshed.clear();
        deferred.card_slots.clear();
    }

    /// As [`HostList::defer`], for work that writes only host memory.
    pub(crate) fn defer_host(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work), true));
    }

    /// As [`HostList::defer_host`], for work that writes only a texture
    /// staging buffer's flags (that the list's upload takes it, that it is
    /// clean): with `VICTORY_WIDE_TEXTURES` the texture checks gathered
    /// before it are not done first. A check of that staging buffer still
    /// pending then runs after the work; it can only find the texture
    /// changed and mark it dirty, and the list's upload, which runs after
    /// every check of the submission, takes what the check put there (the
    /// next list uploads once more).
    pub(crate) fn defer_host_flags(&self, work: impl Fn() + Send + 'static) {
        self.deferred().copies.push(Step::Host(Box::new(work), !wide_textures()));
    }

    /// Brings a texture's staging buffer up to date when the list is
    /// submitted, keeping it alive until the list is reset.
    pub(crate) fn defer_texture(&self, staging: std::sync::Arc<crate::texture::Staging>) {
        self.deferred().copies.push(Step::Texture(staging));
    }

    /// Runs the list's submission-time steps; refreshes go into `batch`
    /// (the title's big buffers, a 174 MB one among them, are read by most
    /// lists of a submission, through many ranges).
    fn run_deferred(&self, batch: &mut Batch) {
        #[cfg(feature = "diagnostics")]
        use crate::copystat::{C, add};
        // The time from one step's start to the next one's goes to the
        // kind of the first ([`prep`]), less the refresh flushes it set off
        // (those are the flushes' own); what comes before the first step,
        // the lists' lock, is bookkeeping.
        let mut split = [0u64; 7];
        let mut counts = [0u64; 7];
        // The fills' flushes ([`wide_fills`]) are taken out as the refresh
        // flushes are, and counted as fills at the submission's end.
        let mut last = (std::time::Instant::now(), batch.taken_ns(), prep::QUEUE);
        let deferred = self.deferred();
        for step in &deferred.copies {
            let started = std::time::Instant::now();
            {
                let (at, flushed, kind) = last;
                split[kind] += ((started - at).as_nanos() as u64).saturating_sub(batch.taken_ns() - flushed);
            }
            let kind = match step {
                Step::Copy(..) => prep::COPY,
                Step::Fill(..) | Step::CardFill(..) => prep::FILL,
                Step::Refresh(..) => prep::QUEUE,
                Step::Run(_) => prep::RUN,
                Step::Host(..) => prep::HOST,
                Step::GpuWrite(..) => prep::DECLARE,
                Step::Texture(_) => prep::TEXTURE,
            };
            counts[kind] += 1;
            last = (started, batch.taken_ns(), kind);
            // Reads of title memory where picked views' programs write
            // (`crate::shadow::SKINNED_OVER`): title memory there has only
            // what a fetch last happened to bring back.
            match step {
                &Step::Copy(_, src, len) => {
                    crate::shadow::count_skinned(crate::shadow::skinned::COPY_READ, src as u64, (src + len) as u64)
                }
                &Step::Fill(_, src, len, _) | &Step::CardFill(_, src, len) => {
                    crate::shadow::count_skinned(crate::shadow::skinned::FILL_READ, src as u64, (src + len) as u64)
                }
                _ => {}
            }
            let (copied, since) = match step {
                &Step::Copy(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.title_wrote(dst, len);
                    #[cfg(feature = "diagnostics")]
                    fill_check::copy_runs(dst, src, len);
                    // Timed from here: the flush above counts its own time.
                    let since = std::time::Instant::now();
                    copy_mapped(dst as *mut u8, src as *const u8, len, true);
                    // With the submission's one asking: the journal learns it.
                    crate::watch::wrote(dst, len);
                    (len, since)
                }
                &Step::Fill(dst, src, len, _) if wide_fills() && batch.from_copy.is_empty() => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.fills.push((dst, src, len));
                    continue;
                }
                &Step::CardFill(dst, src, len) if wide_fills() => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    batch.fills.push((dst, src, len));
                    continue;
                }
                &Step::Fill(dst, src, len, _) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                        fill_check::runs(src, len);
                    }
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    // Under flights' shader writes ([`fill_from_copy`]):
                    // what they wrote comes from the copies they wrote.
                    if !batch.from_copy.is_empty() {
                        from_copies(&batch.from_copy, dst, src, len);
                    }
                    (len, started)
                }
                &Step::CardFill(dst, src, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Copies, 1);
                        add(C::CopyBytes, len as u64);
                    }
                    // What flights write in it is copied over it on the
                    // card ([`HostList::card_pre`]).
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                    (len, started)
                }
                Step::Refresh(shadow, address, len) => {
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Refreshes, 1);
                        add(C::RefreshBytes, *len as u64);
                    }
                    batch.add(shadow, *address, *len);
                    continue;
                }
                Step::Run(work) | Step::Host(work, _) => {
                    if matches!(step, Step::Run(_)) {
                        batch.title_written();
                    } else {
                        // Our host work may read or write what a fill
                        // writes: the fills before it are done first; and
                        // what a texture check writes, unless it says not.
                        batch.flush_fills();
                        if matches!(step, Step::Host(_, true)) {
                            batch.flush_textures();
                        }
                    }
                    work();
                    // It wrote title memory, where is not known here: the
                    // submission's one asking ends, later refreshes ask anew.
                    if matches!(step, Step::Run(_)) {
                        crate::watch::ask_once_end();
                    }
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::Runs, 1);
                        add(C::RunUs, started.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::GpuWrite(shadow, address, len) => {
                    // The shaders start from the title's data: this copy's
                    // pending refresh first. The other copies' wait for
                    // the next flush of all (flushing all of them at every
                    // range a dispatch writes made thousands of write-watch
                    // askings a frame).
                    batch.flush_one(shadow, *address, *len);
                    // Declared again in the same submission: nothing to do
                    // ([`declare_once`]).
                    if !batch.first_declared(shadow, *address, *len) {
                        continue;
                    }
                    PREP_SPLIT[prep::FIRST_DECLARED].fetch_add(1, Ordering::Relaxed);
                    #[cfg(feature = "diagnostics")]
                    let snapshot = std::time::Instant::now();
                    shadow.writing(*address, *len);
                    #[cfg(feature = "diagnostics")]
                    {
                        add(C::GpuWrites, 1);
                        add(C::SnapshotUs, snapshot.elapsed().as_micros() as u64);
                    }
                    continue;
                }
                Step::Texture(staging) => {
                    if batch.texture_due(staging) {
                        if wide_textures() {
                            batch.textures_due.push(staging.clone());
                        } else {
                            staging.refresh();
                        }
                    }
                    continue;
                }
            };
            preparing(0, since.elapsed().as_micros() as u64);
            preparing(1, copied as u64);
        }
        drop(deferred);
        {
            let (at, flushed, kind) = last;
            split[kind] += (at.elapsed().as_nanos() as u64).saturating_sub(batch.taken_ns() - flushed);
        }
        for (i, (&ns, &n)) in split.iter().zip(&counts).enumerate() {
            if ns != 0 {
                PREP_SPLIT[i].fetch_add(ns, Ordering::Relaxed);
            }
            if n != 0 {
                PREP_SPLIT[prep::COUNTS + i].fetch_add(n, Ordering::Relaxed);
            }
        }
    }
}

/// Whether a range shaders write that a submission declared already is
/// passed over when the submission declares it again (on unless
/// `VICTORY_DECLARE_ONCE=0`).
///
/// Dispatches declare the same ranges thousands of times a submission,
/// and each declaration ([`crate::shadow::Shadow::writing`]) took the
/// copy's locks and searched its pending ranges (and, with the relay, the
/// flights' counts) from the start. A repeat is a no-op there: the count
/// is per submission (the flights' entry already names this one), and the
/// range stays pending while this submission holds it (only a copy-back
/// takes ranges out, and with the relay only ranges no submission still
/// holds; without it no copy-back runs while a submission's steps do).
fn declare_once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DECLARE_ONCE").is_none_or(|v| v != "0"))
}

/// Whether a shader-write declaration refreshes only the part of its copy's
/// pending refresh that lies in the range declared, and leaves the rest
/// pending (`VICTORY_FLUSH_ONE_PART=1`; off by default).
///
/// A declaration ([`Step::GpuWrite`]) did its copy's whole pending refresh
/// first, on the submitting thread, one copy at a time: some 536 a frame,
/// 5.1 ms in n28, of ranges the submission's refreshes of that copy had
/// merged into one span. Only the declared range needs it then: from the
/// declaration on, refreshes leave a declared range alone (its bytes are
/// what the shaders start from, and what the snapshot holds). The rest of
/// the span is refreshed by the flush of all that follows (before a step
/// that writes title memory, and before the lists go: on several threads
/// with `VICTORY_WIDE_COPY`), as the other copies' refreshes are; a range
/// the shaders read is refreshed before the lists run either way.
///
/// What it costs elsewhere (found after run n34): a 64 KiB page across an
/// edge of the declared range is refreshed in two parts, the declared part
/// here and the rest after the declaration, and neither holds the page
/// whole, so the page is never noted taken. While submissions declare the
/// range again with flights in the air it stays stale: copied again at
/// every refresh, and at every stale check a reason to merge or settle
/// under the flights. `VICTORY_DECLARED_WIDE` takes the whole free span
/// as one, as without either switch.
fn flush_one_part() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FLUSH_ONE_PART").is_some_and(|v| v == "1"))
}

/// Whether a submission's refreshes of host copies run on several threads,
/// a copy to a thread (`VICTORY_WIDE_REFRESH=1`; off by default).
///
/// A submission brings some sixty copies up to date, one after the other
/// on the submitting thread: for each, the pages the title wrote are asked
/// for and read through the kernel into the copy. That is the frame line's
/// "copies", 45-50 ms a frame, and the copies have nothing to do with one
/// another: each refresh writes its own copy and its own notes, and reads
/// title memory. With the switch the copies of a flush (four or more) are
/// spread over the threads kept for the snapshots (`crate::shadow::wide`).
/// The write-watch journal and the guards are behind their own locks, as
/// they are for the title's threads.
fn wide_refresh() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_REFRESH").is_some_and(|v| v == "1"))
}

/// Whether a range of a host copy brought up to date in a submission is
/// left alone for the rest of that submission, until a step of ours writes
/// title memory there (`VICTORY_REFRESH_ONCE=1`; off by default). After
/// upstream VIctory 6d021fe ("a copy's range brought up to date is not
/// copied again in the submission until a step writes title memory").
///
/// A list of dispatches records "refresh this range, shaders write it" for
/// every dispatch, and each of those refreshes was done on its own
/// (`Batch::flush_one`): a walk of the guards or a failed kernel asking
/// for a copy of view memory, some 1,800 to 2,700 a frame and 23 to 29 ms
/// (runs 222 and 225, "watches"), nearly all finding nothing new.
///
/// What changes: what the title's other threads write into a range while
/// its submission is being prepared is taken by the next submission, not
/// by a later step of this one, as `crate::watch::once` already has it for
/// write-watched memory. What this submission's lists read was written
/// before they were submitted. A title buffer copy of ours takes its
/// bytes out of what counts as up to date ([`Batch::title_wrote`]); any
/// other step that writes title memory takes everything out.
///
/// `VICTORY_REFRESH_ONCE=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log, so that one run measures both in
/// the same scene (two runs do not stand in the same scene at the same
/// second: runs 228 and 229).
fn refresh_once() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: OnceLock<u8> = OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_REFRESH_ONCE").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            let on = (now_us() / 20_000_000) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "refreshes once a submission from now (VICTORY_REFRESH_ONCE=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// A submission's refreshes not done yet, merged per buffer copy into the
/// range that covers them: done before a step that writes title memory or
/// gives a range to shaders, and before the lists run.
///
/// Also the textures brought up to date in the submission so far: most
/// lists of a world submission sample the same textures, and a texture in
/// physical-page memory is hashed whole each time it is checked. Checked
/// once, it is checked again only after a step that writes title memory.
#[derive(Default)]
struct Batch {
    refreshes: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Where each copy (by address) is in `refreshes`: the list that makes
    /// the heap's buffers ready adds some 4,100 copies, and searching them
    /// for every refresh of a submission was tens of millions of
    /// comparisons a frame.
    index: std::collections::HashMap<usize, usize>,
    /// With [`refresh_once`], by copy (its address; the copy is held, so
    /// that the address names no other): the range of title addresses
    /// brought up to date in this submission that no step of ours has
    /// written since.
    current: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// Staging buffers (by address) refreshed since the last title write.
    textures: std::collections::HashSet<usize>,
    /// Nanoseconds spent in [`Batch::flush`] and [`Batch::flush_one`] on
    /// the submitting thread ([`PREP_SPLIT`]).
    flush_ns: u64,
    /// The ranges shaders write that the submission declared so far (copy
    /// by address, address, bytes; [`declare_once`]).
    declared: std::collections::HashSet<(usize, usize, usize)>,
    /// With [`fill_from_copy`]: the parts of the submission's fills that
    /// flights on the card (run, not copied back) wrote, by copy (copy,
    /// from, to: title addresses), read from those copies.
    from_copy: Vec<(std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
    /// With [`wide_fills`]: the fills not done yet (destination, source,
    /// bytes), done together before a step that writes title memory or
    /// runs work of ours, and before the lists go ([`Batch::flush_fills`]);
    /// and the nanoseconds that took on the submitting thread.
    fills: Vec<(usize, usize, usize)>,
    fill_ns: u64,
    /// With [`declared_wide`]: per copy, the parts of its pending refresh
    /// that were free of shader writes when a declaration of the copy
    /// would have refreshed them on its own ([`Batch::flush_one`]), planned
    /// and moved with the rest at the next flush of all.
    declared_parts: Vec<(std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)>,
    /// With [`wide_textures`]: the texture checks not done yet, done
    /// together before a step that writes title memory or host work that
    /// may read a staging buffer, and before the lists go
    /// ([`Batch::flush_textures`]); and the nanoseconds that took on the
    /// submitting thread.
    textures_due: Vec<std::sync::Arc<crate::texture::Staging>>,
    tex_ns: u64,
}

/// Whether a submission's fills of upload memory (`Step::Fill`,
/// `Step::CardFill`) are gathered and done together on several threads
/// (`VICTORY_WIDE_FILLS=1`; off by default).
///
/// Each fill is one read of title memory through the kernel, 0.7 us
/// however small: 13,000 a frame were 9.5 ms on the submitting thread
/// (run n37). A fill reads title memory and writes the list's own upload
/// memory, which nothing else of the preparing reads or writes. So fills
/// wait, in order, until a step that writes title memory (a title copy,
/// work of ours: [`Batch::title_written`]), a step of our host work, or
/// the end of the steps, and are then done on the kept threads
/// (`crate::shadow::wide_run`). The bytes each fill writes are the ones it
/// wrote in place: nothing between it and the flush writes title memory.
/// Fills whose destinations overlap are done one after the other, in
/// order; and not with [`fill_from_copy`]'s parts (their pieces are
/// written over the fill in place).
fn wide_fills() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_FILLS").is_some_and(|v| v == "1"))
}

/// Whether the refresh a shader-write declaration sets off for its copy
/// ([`Batch::flush_one`]) is left to the next flush of all and moved there
/// with the other copies' on several threads (`VICTORY_DECLARED_WIDE=1`;
/// off by default; only with `VICTORY_WIDE_COPY`, with the snapshots left
/// for the end of the steps (`VICTORY_WIDE_BACK`, on unless 0), and not
/// with `VICTORY_KEEP_SNAPSHOTS`).
///
/// Some 540 declarations a frame found their copy's refresh pending and
/// did it there, one copy at a time on the submitting thread: 2.7-3.6 ms a
/// frame in n43, while the flush of all moves its bytes wide. What the
/// declaration needs is that the bytes the shaders start from, and the
/// snapshot that tells later what they wrote, are the title's. So the
/// parts of the copy's pending span free of shader writes at that moment
/// ([`crate::shadow::Shadow::free_parts`], the declared range among them)
/// are noted, and at the next flush of all planned as they are
/// ([`crate::shadow::Shadow::plan_parts`]: the declared range is pending
/// by then, and a plan of the span would leave it out) and moved with the
/// rest. Nothing reads the copy or writes title memory in between: a title
/// buffer copy, work of ours that may write title memory and the end of
/// the steps all flush first; the snapshots are taken after the last flush
/// (`crate::shadow::take_snapshots`), and a copy in the card's memory sends
/// its changed bytes up after it as well. A range a later declaration of
/// the submission takes is moved with the parts and its snapshot taken
/// after them, as it would have been after the declaration's refresh. What
/// the title writes in between is read now and not then; the next asking
/// sees it either way.
fn declared_wide() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_DECLARED_WIDE").is_some_and(|v| v == "1")
            && crate::shadow::wide_copy()
            && crate::shadow::wide_back()
            && std::env::var_os("VICTORY_KEEP_SNAPSHOTS").is_none_or(|v| v != "1")
    })
}

/// Whether a submission's texture checks (`Step::Texture`) are gathered
/// and done together on threads of their own (`VICTORY_WIDE_TEXTURES=1`;
/// off by default).
///
/// Each check asks the journal or the guards about the texture's memory
/// and, if it may have changed, hashes and detiles it into its staging
/// buffer: 363 checks a frame, 2.2-4.5 ms on the submitting thread in n43,
/// most of it a few textures read whole. A check reads title memory and
/// writes its own staging buffer (one check per buffer between title
/// writes, [`Batch::texture_due`]); the async textures and the preparing
/// threads (`crate::texture::prepare`) check buffers off the submitting
/// thread already. So checks wait, in order, until a step that writes
/// title memory, host work that may read a staging buffer (all host work
/// but the flags a list's upload sets, [`HostList::defer_host_flags`]),
/// or the end of the steps, and are then done on the threads of
/// `crate::shadow::wide_textures` (a pool apart from the copies', so that
/// no check runs inside another thread's wide work that holds a lock).
pub(crate) fn wide_textures() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WIDE_TEXTURES").is_some_and(|v| v == "1"))
}

/// Whether the check before a submission's refreshes that flights' shader
/// writes lie over pages the title wrote ([`stale_under_flights`], the
/// "stale checks") gathers its spans by copy, with one look at each copy's
/// pending ranges, and stops at a range's first stale page
/// (`VICTORY_STALE_ONCE=1`; off by default). The spans and the answer are
/// the same: the steps were matched to their copy through a list of spans,
/// each refresh step locked its copy's pending ranges, and each range
/// listed all its stale pages to see whether there was one.
pub(crate) fn stale_once() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_STALE_ONCE").is_some_and(|v| v == "1"))
}

/// Since the last frame line, of [`wide_fills`]: fills done on several
/// threads, the flushes that did so, fills of flushes done in order (too
/// few, or overlapping destinations), and the nanoseconds of all flushes
/// on the submitting thread.
static FILLS_WIDE: [std::sync::atomic::AtomicU64; 4] = [const { std::sync::atomic::AtomicU64::new(0) }; 4];

/// [`FILLS_WIDE`] per frame, for the frame's path.
fn fills_line(frames: f64) -> String {
    let v: Vec<f64> = FILLS_WIDE.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    format!(
        "{:.0} fills in {:.1} flushes on several threads, {:.0} in order, {:.2} ms",
        v[0],
        v[1],
        v[2],
        v[3] / 1e6
    )
}

/// Fewer fills than this are done in order on the submitting thread.
const WIDE_FILLS_MIN: usize = 64;

/// The fills `fills` (destination, source, bytes), as `Step::Fill` does
/// them: on several threads when [`fills_apart`], in order otherwise.
/// Returns the bytes and whether they went wide.
fn do_fills(fills: &[(usize, usize, usize)]) -> (usize, bool) {
    let bytes: usize = fills.iter().map(|f| f.2).sum();
    if fills.len() < WIDE_FILLS_MIN || !fills_apart(fills) {
        for &(dst, src, len) in fills {
            copy_mapped(dst as *mut u8, src as *const u8, len, false);
        }
        return (bytes, false);
    }
    // A job's share: bytes plus the cost of a kernel read, so that many
    // small fills and a few big ones spread alike.
    const CALL: usize = 4096;
    let jobs_n = crate::shadow::wide_threads().max(2);
    let each = (bytes + CALL * fills.len()).div_ceil(jobs_n).max(1);
    let mut jobs: Vec<Box<dyn FnOnce() + Send>> = Vec::with_capacity(jobs_n + 1);
    let mut group: Vec<(usize, usize, usize)> = Vec::new();
    let mut weight = 0;
    for &fill in fills {
        group.push(fill);
        weight += fill.2 + CALL;
        if weight >= each {
            let mine = std::mem::take(&mut group);
            jobs.push(Box::new(move || {
                for (dst, src, len) in mine {
                    copy_mapped(dst as *mut u8, src as *const u8, len, false);
                }
            }));
            weight = 0;
        }
    }
    if !group.is_empty() {
        jobs.push(Box::new(move || {
            for (dst, src, len) in group {
                copy_mapped(dst as *mut u8, src as *const u8, len, false);
            }
        }));
    }
    crate::shadow::wide_run(jobs);
    (bytes, true)
}

/// `runs` (title ranges) in order, those that overlap or touch as one.
fn joined_runs(mut runs: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    runs.sort_unstable();
    let mut out: Vec<(usize, usize)> = Vec::with_capacity(runs.len());
    for (from, to) in runs {
        match out.last_mut() {
            Some(last) if from <= last.1 => last.1 = last.1.max(to),
            _ => out.push((from, to)),
        }
    }
    out
}

/// Whether no two of `fills` write the same byte: then the order they are
/// done in does not matter.
fn fills_apart(fills: &[(usize, usize, usize)]) -> bool {
    let mut at: Vec<(usize, usize)> = fills.iter().filter(|f| f.2 > 0).map(|f| (f.0, f.0 + f.2)).collect();
    at.sort_unstable();
    at.windows(2).all(|w| w[0].1 <= w[1].0)
}

impl Batch {
    fn add(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let end = address + len;
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        // Brought up to date in this submission already ([`refresh_once`];
        // nothing is in `current` without the switch).
        if let Some(&(_, from, to)) = self.current.get(&key)
            && from <= address
            && end <= to
        {
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::add(crate::copystat::C::Skipped, 1);
                crate::copystat::add(crate::copystat::C::SkippedBytes, len as u64);
            }
            return;
        }
        match self.index.get(&key) {
            Some(&at) => {
                let r = &mut self.refreshes[at];
                (r.1, r.2) = (r.1.min(address), r.2.max(end));
            }
            None => {
                self.index.insert(key, self.refreshes.len());
                self.refreshes.push((shadow.clone(), address, end));
            }
        }
    }

    /// Before shaders are given `[address, address + len)` of `shadow`'s
    /// copy: its pending refresh is done (with [`flush_one_part`] only
    /// where it meets that range), the others stay pending.
    fn flush_one(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) {
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        if declared_wide() {
            let Some(at) = self.index.remove(&key) else {
                return;
            };
            let started = std::time::Instant::now();
            let (shadow, from, to) = self.refreshes.swap_remove(at);
            if let Some(moved) = self.refreshes.get(at) {
                self.index.insert(std::sync::Arc::as_ptr(&moved.0) as usize, at);
            }
            // What is free of shader writes now: after the declaration the
            // declared range is not.
            let parts = shadow.free_parts(from, to - from);
            let bytes: usize = parts.iter().map(|p| p.1 - p.0).sum();
            Self::brought(&mut self.current, &shadow, from, to);
            if !parts.is_empty() {
                self.declared_parts.push((shadow, parts));
            }
            let ns = started.elapsed().as_nanos() as u64;
            self.flush_ns += ns;
            more_add(more::FLUSH_ONE, ns);
            more_add(more::FLUSH_ONE_N, 1);
            more_add(more::DECLARED_N, 1);
            more_add(more::DECLARED_BYTES, bytes as u64);
            return;
        }
        if flush_one_part() {
            let Some(&at) = self.index.get(&key) else {
                return;
            };
            let started = std::time::Instant::now();
            let (from, to) = (self.refreshes[at].1, self.refreshes[at].2);
            let (a, b) = (address.max(from), address.saturating_add(len).min(to));
            if a < b {
                Self::update(shadow, a, b);
            }
            let ns = started.elapsed().as_nanos() as u64;
            self.flush_ns += ns;
            more_add(more::FLUSH_ONE, ns);
            more_add(more::FLUSH_ONE_N, 1);
            more_add(more::FLUSH_PART_N, 1);
            more_add(more::FLUSH_PART_BYTES, b.saturating_sub(a) as u64);
            return;
        }
        let Some(at) = self.index.remove(&key) else {
            return;
        };
        let started = std::time::Instant::now();
        let (shadow, from, to) = self.refreshes.swap_remove(at);
        // The last entry took its place.
        if let Some(moved) = self.refreshes.get(at) {
            self.index.insert(std::sync::Arc::as_ptr(&moved.0) as usize, at);
        }
        Self::update(&shadow, from, to);
        let ns = started.elapsed().as_nanos() as u64;
        self.flush_ns += ns;
        more_add(more::FLUSH_ONE, ns);
        more_add(more::FLUSH_ONE_N, 1);
        Self::brought(&mut self.current, &shadow, from, to);
    }

    /// With [`refresh_once`]: `[from, to)` of `shadow`'s copy was brought
    /// up to date. Kept: its union with what was up to date if they touch,
    /// else the larger of the two.
    fn brought(
        current: &mut std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, usize, usize)>,
        shadow: &std::sync::Arc<crate::shadow::Shadow>,
        from: usize,
        to: usize,
    ) {
        if !refresh_once() || from >= to {
            return;
        }
        let entry = current
            .entry(std::sync::Arc::as_ptr(shadow) as usize)
            .or_insert_with(|| (shadow.clone(), from, to));
        if from <= entry.2 && entry.1 <= to {
            (entry.1, entry.2) = (entry.1.min(from), entry.2.max(to));
        } else if to - from > entry.2 - entry.1 {
            (entry.1, entry.2) = (from, to);
        }
    }

    /// Brings `[from, to)` of a copy up to date, timed.
    fn update(shadow: &crate::shadow::Shadow, from: usize, to: usize) {
        let started = std::time::Instant::now();
        let copied = shadow.update(from, to - from);
        let us = started.elapsed().as_micros() as u64;
        #[cfg(feature = "diagnostics")]
        crate::copystat::range(from, to - from, us, 0);
        preparing(0, us);
        preparing(1, copied as u64);
    }

    /// Whether `[address, address + len)` of `shadow`'s copy is declared
    /// for the first time in this submission (and notes it declared).
    fn first_declared(&mut self, shadow: &std::sync::Arc<crate::shadow::Shadow>, address: usize, len: usize) -> bool {
        !declare_once() || self.declared.insert((std::sync::Arc::as_ptr(shadow) as usize, address, len))
    }

    /// Whether `staging` is still to be refreshed in this submission (and
    /// notes it refreshed).
    fn texture_due<T>(&mut self, staging: &std::sync::Arc<T>) -> bool {
        self.textures.insert(std::sync::Arc::as_ptr(staging) as *const () as usize)
    }

    /// Before a step that writes title memory: pending refreshes are done
    /// first, and textures are checked again after it.
    fn title_written(&mut self) {
        self.flush();
        self.textures.clear();
        // Where it writes is not known: no copy counts as up to date.
        self.current.clear();
    }

    /// [`Batch::title_written`] for a write of ours whose place is known
    /// (a title buffer copy into `[at, at + len)`): of a copy up to date
    /// in this submission that holds some of those bytes, the larger part
    /// beside them stays so.
    fn title_wrote(&mut self, at: usize, len: usize) {
        self.flush();
        self.textures.clear();
        let end = at + len;
        self.current.retain(|_, entry| {
            let (from, to) = (entry.1, entry.2);
            if end <= from || to <= at {
                return true;
            }
            let (left, right) = (at.saturating_sub(from), to.saturating_sub(end));
            if left >= right {
                entry.2 = from + left;
            } else {
                entry.1 = end;
            }
            entry.1 < entry.2
        });
    }

    /// The nanoseconds of flushes the steps set off so far (refreshes,
    /// fills, texture checks): taken out of the steps' own time.
    fn taken_ns(&self) -> u64 {
        self.flush_ns + self.fill_ns + self.tex_ns
    }

    /// [`wide_textures`]: the texture checks gathered so far, done.
    fn flush_textures(&mut self) {
        if self.textures_due.is_empty() {
            return;
        }
        let started = std::time::Instant::now();
        let due = std::mem::take(&mut self.textures_due);
        let n = due.len();
        if n < 2 {
            for staging in &due {
                staging.refresh();
            }
        } else {
            let jobs: Vec<Box<dyn FnOnce() + Send>> = due
                .into_iter()
                .map(|staging| {
                    Box::new(move || {
                        // A pool thread that panicked would leave the
                        // submitting thread waiting for its job.
                        if std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| staging.refresh())).is_err() {
                            tracing::error!("a texture check panicked on a texture thread (VICTORY_WIDE_TEXTURES)");
                        }
                    }) as Box<dyn FnOnce() + Send>
                })
                .collect();
            crate::shadow::wide_textures(jobs);
            more_add(more::TEX_WIDE_FLUSHES, 1);
        }
        let ns = started.elapsed().as_nanos() as u64;
        self.tex_ns += ns;
        more_add(more::TEX_WIDE_N, n as u64);
        more_add(more::TEX_WIDE, ns);
    }

    /// [`wide_fills`]: the fills gathered so far, done.
    fn flush_fills(&mut self) {
        if self.fills.is_empty() {
            return;
        }
        let started = std::time::Instant::now();
        let mut fills = std::mem::take(&mut self.fills);
        let (bytes, wide) = do_fills(&fills);
        let ns = started.elapsed().as_nanos() as u64;
        self.fill_ns += ns;
        PREPARING[0].fetch_add(ns / 1000, Ordering::Relaxed);
        PREPARING[1].fetch_add(bytes as u64, Ordering::Relaxed);
        FILLS_WIDE[if wide { 0 } else { 2 }].fetch_add(fills.len() as u64, Ordering::Relaxed);
        if wide {
            FILLS_WIDE[1].fetch_add(1, Ordering::Relaxed);
        }
        FILLS_WIDE[3].fetch_add(ns, Ordering::Relaxed);
        // The vector's room is kept for the next ones.
        fills.clear();
        self.fills = fills;
    }

    fn flush(&mut self) {
        self.flush_fills();
        self.flush_textures();
        if self.refreshes.is_empty() && self.declared_parts.is_empty() {
            self.index.clear();
            return;
        }
        let started = std::time::Instant::now();
        self.flush_all();
        let ns = started.elapsed().as_nanos() as u64;
        self.flush_ns += ns;
        more_add(more::FLUSH_ALL, ns);
        more_add(more::FLUSH_ALL_N, 1);
    }

    fn flush_all(&mut self) {
        self.index.clear();
        // One write-watch asking for the copies that lie close together,
        // not one per copy.
        // (Not while the submission's one asking stands, with
        // `VICTORY_AHEAD_INDEX`: it would ask nothing.)
        let ahead = self.refreshes.len() + self.declared_parts.len() > 1
            && !(crate::watch::ahead_index() && crate::watch::kept());
        if ahead {
            let started = std::time::Instant::now();
            let mut ranges: Vec<(usize, usize)> = self
                .refreshes
                .iter()
                .filter(|r| !r.0.unwatched())
                .map(|r| (r.1, r.2))
                .collect();
            // The parts declarations left ([`declared_wide`]), as the span
            // each copy's lie in.
            ranges.extend(
                self.declared_parts
                    .iter()
                    .filter(|d| !d.0.unwatched() && !d.1.is_empty())
                    .map(|d| (d.1[0].0, d.1[d.1.len() - 1].1)),
            );
            crate::watch::ask_ahead(&ranges);
            preparing(0, started.elapsed().as_micros() as u64);
            crate::shadow::REFRESH_SPLIT[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        if crate::shadow::wide_copy() && !(self.refreshes.is_empty() && self.declared_parts.is_empty()) {
            // Every copy's pages first, on this thread; then all the moves
            // together on several (`crate::shadow::wide_copy`).
            let started = std::time::Instant::now();
            let mut plans: Vec<(std::sync::Arc<crate::shadow::Shadow>, Vec<(usize, usize)>)> = Vec::new();
            // With [`declared_wide`]: the parts declarations left, first. A
            // copy planned twice (parts, then a span queued after its
            // declaration) gets one plan, its runs joined, so that no two
            // pieces of work write the same bytes.
            let mut which: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
            for (shadow, parts) in self.declared_parts.drain(..) {
                let mut runs = Vec::new();
                shadow.plan_parts(&parts, &mut runs);
                let key = std::sync::Arc::as_ptr(&shadow) as usize;
                match which.get(&key) {
                    Some(&i) => plans[i].1.extend(runs),
                    None => {
                        which.insert(key, plans.len());
                        plans.push((shadow, runs));
                    }
                }
            }
            let twice = !which.is_empty();
            for (shadow, from, to) in self.refreshes.drain(..) {
                let mut runs = Vec::new();
                shadow.plan(from, to - from, &mut runs);
                let key = std::sync::Arc::as_ptr(&shadow) as usize;
                match which.get(&key) {
                    Some(&i) => plans[i].1.extend(runs),
                    None if !runs.is_empty() => plans.push((shadow, runs)),
                    None => {}
                }
            }
            if twice {
                for (_, runs) in &mut plans {
                    *runs = joined_runs(std::mem::take(runs));
                }
                plans.retain(|p| !p.1.is_empty());
            }
            let planned: usize = plans.iter().flat_map(|(_, runs)| runs.iter().map(|r| r.1 - r.0)).sum();
            let moving = std::time::Instant::now();
            more_add(more::FLUSH_PLAN, (moving - started).as_nanos() as u64);
            let (moved, written) = crate::shadow::move_wide(&plans);
            more_add(more::FLUSH_MOVE, moving.elapsed().as_nanos() as u64);
            crate::shadow::REFRESH_SPLIT[1].fetch_add(moving.elapsed().as_micros() as u64, Ordering::Relaxed);
            crate::shadow::REFRESH_SPLIT[2].fetch_add(planned.saturating_sub(written) as u64, Ordering::Relaxed);
            for ((shadow, _), ranges) in plans.iter().zip(&moved) {
                shadow.moved(ranges);
            }
            PREPARING[0].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
            PREPARING[1].fetch_add(written as u64, Ordering::Relaxed);
        } else if wide_refresh() && self.refreshes.len() >= 4 {
            // Each copy on its own, several at a time: a copy's refresh
            // writes that copy and reads title memory, nothing of another's.
            // The time counted is the time it took, not the threads' sum.
            let started = std::time::Instant::now();
            let before = PREPARING[0].load(Ordering::Relaxed);
            // Noted now: nothing reads the notes before the jobs are done.
            for (shadow, from, to) in &self.refreshes {
                Self::brought(&mut self.current, shadow, *from, *to);
            }
            let jobs: Vec<Box<dyn FnOnce() + Send>> = self
                .refreshes
                .drain(..)
                .map(|(shadow, from, to)| Box::new(move || Self::update(&shadow, from, to)) as Box<dyn FnOnce() + Send>)
                .collect();
            crate::shadow::wide_run(jobs);
            let summed = PREPARING[0].load(Ordering::Relaxed).saturating_sub(before);
            let over = summed.saturating_sub(started.elapsed().as_micros() as u64);
            PREPARING[0].fetch_sub(over, Ordering::Relaxed);
            crate::framelog::sub(crate::framelog::T::CopyUs, over);
        } else {
            for (shadow, from, to) in self.refreshes.drain(..) {
                Self::update(&shadow, from, to);
                Self::brought(&mut self.current, &shadow, from, to);
            }
        }
        if ahead {
            crate::watch::ask_ahead_done();
        }
    }
}

/// Copies title memory at submission. The ranges were checked against their
/// resources when recorded, but the title may since have unmapped pages of
/// them (tiled memory, page mappings), which read as zeros here; a title
/// destination (`title_dst`) that is not all mapped is left alone.
///
/// The source is read through the kernel (`ReadProcessMemory`), which
/// reports an unmapped page instead of faulting. Asking `VirtualQuery` first
/// cost about 2 ms per copy in the title's big pools (it walks the page
/// tables to the region's end), which made copies run at 100-200 MB/s.
pub(crate) fn copy_mapped(dst: *mut u8, src: *const u8, len: usize, title_dst: bool) {
    #[cfg(feature = "diagnostics")]
    match crate::copystat::mode() {
        1 => {
            // SAFETY: timing only: the caller vouches for the source.
            unsafe { std::ptr::copy(src, dst, len) };
            return;
        }
        2 => return,
        _ => {}
    }
    if len == 0 {
        return;
    }
    // A title destination that is not all mapped is left alone. Asking
    // Windows first (`VirtualQuery`) walks the page tables to the end of
    // the destination's region: in the title's big blocks that is
    // milliseconds a copy, and it was a seventh of the submitting thread's
    // time in gameplay (run 60: 71 of 500 looks). The kernel's copy below
    // says the same when it fails, so the question is only asked then.
    // `VICTORY_COPY_ASK=1` asks first, as before.
    static ASK: OnceLock<bool> = OnceLock::new();
    let ask_first = title_dst
        && *ASK.get_or_init(|| std::env::var_os("VICTORY_COPY_ASK").is_some_and(|v| v == "1"));
    if ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    if title_dst {
        // The kernel writes the destination: it must not be guarded.
        crate::watch::writing(dst as usize, len);
    }
    #[cfg(feature = "diagnostics")]
    let moved = std::time::Instant::now();
    // Into host memory: in place first, if asked for (`crate::plaincopy`);
    // through the kernel if that is off or met a page that was not there.
    let whole = (!title_dst && crate::plaincopy::copy(dst, src, len)) || read(src, dst, len);
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::MoveUs, moved.elapsed().as_micros() as u64);
    if whole {
        return;
    }
    // The copy failed: a source page is not there, or the destination is
    // not all mapped. Then the destination stays as it is (the kernel may
    // have written its first pages).
    if title_dst && !ask_first && !crate::texture::committed(dst, len) {
        return;
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::Fallbacks, 1);
    #[cfg(feature = "diagnostics")]
    let sparse = std::time::Instant::now();
    if !title_dst && len >= 64 << 10 {
        read_sparse(src, dst, len);
    } else {
        read_pieces(src, dst, len);
    }
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::SparseUs, sparse.elapsed().as_micros() as u64);
}

/// After a failed bulk read into host memory, skip contiguous holes. Never
/// cache the query across copies: the title may remap the same address.
fn read_sparse(src: *const u8, dst: *mut u8, len: usize) {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEMORY_BASIC_INFORMATION, PAGE_NOACCESS, VirtualQuery,
    };
    let mut offset = 0;
    while offset < len {
        let at = src.wrapping_add(offset);
        let to = dst.wrapping_add(offset);
        let remaining = len - offset;
        // SAFETY: the query inspects an address and fills a local structure.
        let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        let size = unsafe { VirtualQuery(at.cast(), &mut info, size_of_val(&info)) };
        let n = (info.BaseAddress as usize).saturating_add(info.RegionSize)
            .saturating_sub(at as usize).min(remaining);
        if size != size_of_val(&info) || n == 0 {
            read_pieces(at, to, remaining);
            break;
        }
        if info.State != MEM_COMMIT || info.Protect == PAGE_NOACCESS {
            #[cfg(feature = "diagnostics")]
            crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
            // SAFETY: this path is only used with a writable host destination.
            unsafe { to.write_bytes(0, n) };
        } else if !read(at, to, n) {
            // A mapping may change between query and read.
            read_pieces(at, to, n);
        }
        offset += n;
    }
}

fn read_pieces(src: *const u8, dst: *mut u8, len: usize) {
    // Something is unmapped: page by page (64 KiB, then 4 KiB in a piece
    // that fails), zeros for what cannot be read.
    for_pieces(src as usize, len, 64 << 10, |at, n| {
        let to = dst.wrapping_add(at - src as usize);
        if !read(at as *const u8, to, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                let to = dst.wrapping_add(at - src as usize);
                if !read(at as *const u8, to, n) {
                    #[cfg(feature = "diagnostics")]
                    crate::copystat::add(crate::copystat::C::Unmapped, n as u64);
                    // SAFETY: inside the destination, which is mapped.
                    unsafe { to.write_bytes(0, n) };
                }
            });
        }
    });
}

/// Reads title memory at `src` into `dst`; false if a page of it is not
/// readable.
pub(crate) fn read_into(src: *const u8, dst: &mut [u8]) -> bool {
    // In place first, if asked for (`crate::plaincopy`): the destination is
    // memory of ours, as in `copy_mapped`.
    crate::plaincopy::copy(dst.as_mut_ptr(), src, dst.len()) || read(src, dst.as_mut_ptr(), dst.len())
}

/// Copies `len` bytes from `src` to `dst` unless a source page is not
/// readable.
fn read(src: *const u8, dst: *mut u8, len: usize) -> bool {
    use windows_sys::Win32::System::Diagnostics::Debug::ReadProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut done = 0;
    // SAFETY: the destination is mapped writable memory of `len` bytes; the
    // kernel checks the source.
    unsafe { ReadProcessMemory(GetCurrentProcess(), src.cast(), dst.cast(), len, &mut done) != 0 }
}

#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    /// The kernel's write into a process's memory, `(process, address,
    /// bytes, length, written out)`: an NTSTATUS, negative when a page of
    /// the range is not there or not writable. It changes no protection.
    fn NtWriteVirtualMemory(
        process: *mut std::ffi::c_void,
        base: *mut std::ffi::c_void,
        buffer: *const std::ffi::c_void,
        size: usize,
        written: *mut usize,
    ) -> i32;
}

/// Whether title memory is written with the kernel's own call
/// (`NtWriteVirtualMemory`) and not `WriteProcessMemory`, which first asks
/// what the pages are (as `VirtualQuery` does: about 2 ms a call in the
/// title's big pools, see [`copy_mapped`]): putting back what shaders wrote
/// cost 1.2-1.9 s a frame for 430-800 pieces, 2 ms each whatever their size
/// (run 24). On by default; `VICTORY_DIRECT_WRITES=0` writes as before.
fn direct_writes() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_DIRECT_WRITES").is_none_or(|v| v != "0"))
}

/// Copies host memory into title memory after a submission (what shaders
/// wrote), leaving pages that are not mapped alone.
pub(crate) fn write_mapped(dst: *mut u8, src: *const u8, len: usize) {
    use windows_sys::Win32::System::Diagnostics::Debug::WriteProcessMemory;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let direct = direct_writes();
    let write = |at: usize, n: usize| {
        let from = src.wrapping_add(at - dst as usize);
        let mut done = 0;
        if direct {
            // SAFETY: as below; the kernel checks the destination and
            // writes nothing it cannot (the caller opened its guards:
            // `crate::watch::writing`).
            return unsafe {
                NtWriteVirtualMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) >= 0
            };
        }
        // SAFETY: the source is our mapped host copy; the kernel checks the
        // destination.
        unsafe { WriteProcessMemory(GetCurrentProcess(), at as _, from.cast(), n, &mut done) != 0 }
    };
    if len == 0 || write(dst as usize, len) {
        return;
    }
    for_pieces(dst as usize, len, 64 << 10, |at, n| {
        if !write(at, n) {
            for_pieces(at, n, 4 << 10, |at, n| {
                write(at, n);
            });
        }
    });
}

/// Calls `f` with the pieces of `[start, start + len)` cut at multiples of
/// `step`.
fn for_pieces(start: usize, len: usize, step: usize, mut f: impl FnMut(usize, usize)) {
    let end = start + len;
    let mut at = start;
    while at < end {
        let next = ((at / step + 1) * step).min(end);
        f(at, next - at);
        at = next;
    }
}

impl Host {
    /// The queue everything of ours runs on, for a call that must go to it
    /// in order with the lists (`crate::tiles`: a texture's levels are
    /// given memory ahead of the list that uploads them).
    pub(crate) fn queue(&self) -> &ID3D12CommandQueue {
        &self.queue
    }

    /// A buffer in the upload heap, mapped for writing.
    /// A buffer of `size` bytes the card writes and we read (a read back),
    /// in the state copies into it need.
    pub(crate) fn read_back_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a read-back buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COPY_DEST,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no read-back buffer");
                None
            }
        }
    }

    /// Whether the card and the runtime have GPU upload heaps (the card's
    /// memory mapped for the CPU: resizable BAR), asked once and said.
    fn gpu_upload_heap(&self) -> bool {
        static HAS: OnceLock<bool> = OnceLock::new();
        *HAS.get_or_init(|| {
            let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
            // SAFETY: a structure of the size passed.
            let asked = unsafe {
                self.device.CheckFeatureSupport(
                    D3D12_FEATURE_D3D12_OPTIONS16,
                    (&raw mut o16).cast(),
                    size_of_val(&o16) as u32,
                )
            };
            let has = asked.is_ok() && o16.GPUUploadHeapSupported.as_bool();
            tracing::info!(
                asked = asked.is_ok(),
                has,
                "GPU upload heaps (the card's memory mapped for the CPU; resizable BAR): VICTORY_GPU_UPLOAD {}",
                if has { "puts scratch and strides tables there" } else { "falls back to the upload heap" }
            );
            has
        })
    }

    /// A buffer of `size` bytes the CPU writes and shaders only read,
    /// mapped for writing, and whether it is in the card's memory: there
    /// with `VICTORY_GPU_UPLOAD=1` where the card has GPU upload heaps
    /// (`crate::bus::gpu_upload_asked`), else in the upload heap as
    /// [`Self::upload_buffer`] makes it. Either way in the state shaders and
    /// copies read it in, for good.
    pub(crate) fn cpu_written_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8, bool)> {
        if crate::bus::gpu_upload_asked() && self.gpu_upload_heap() {
            match self.mapped_buffer(D3D12_HEAP_TYPE_GPU_UPLOAD, size) {
                Ok(made) => {
                    crate::bus::MADE[0].fetch_add(1, Ordering::Relaxed);
                    crate::bus::MADE[1].fetch_add(size, Ordering::Relaxed);
                    return Some((made.0, made.1, true));
                }
                Err(e) => {
                    let n = crate::bus::MADE[2].fetch_add(1, Ordering::Relaxed);
                    if n < 8 {
                        tracing::warn!(error = %e, size, "no buffer in a GPU upload heap: the upload heap instead");
                    }
                }
            }
        }
        self.upload_buffer(size).map(|(buffer, data)| (buffer, data, false))
    }

    /// A buffer of `size` bytes in a heap of `kind` the CPU maps, in the
    /// generic read state, and its mapping.
    fn mapped_buffer(&self, kind: D3D12_HEAP_TYPE, size: u64) -> windows::core::Result<(ID3D12Resource, *mut u8)> {
        let heap = D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a buffer the CPU writes.
        unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut buffer,
            )?;
            let b = buffer.ok_or_else(windows::core::Error::empty)?;
            // Nothing is read through the mapping.
            b.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }), Some(&mut data))?;
            Ok((b, data.cast()))
        }
    }

    pub(crate) fn upload_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[0].fetch_add(1, Ordering::Relaxed);
        MADE[1].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_UPLOAD,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of an upload buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_GENERIC_READ,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no upload buffer");
                None
            }
        }
    }
}

/// The swap chain formats a presented texture can be copied into: its own
/// family's.
fn swap_format(format: DXGI_FORMAT) -> Option<DXGI_FORMAT> {
    Some(match format.0 {
        27..=32 => DXGI_FORMAT_R8G8B8A8_UNORM,
        87 | 90 | 91 => DXGI_FORMAT_B8G8R8A8_UNORM,
        23..=25 => DXGI_FORMAT_R10G10B10A2_UNORM,
        9..=14 => DXGI_FORMAT_R16G16B16A16_FLOAT,
        _ => return None,
    })
}

const BUFFERS: u32 = 2;

struct Presenter {
    swap_chain: IDXGISwapChain3,
    width: u32,
    height: u32,
    format: DXGI_FORMAT,
    copy: HostList,
    /// With presents sent ahead ([`late_free_present`]): a list for each
    /// present in the air, and the submission it last flew as (it is used
    /// again once that one has finished).
    ahead: Vec<(HostList, u64)>,
    hwnd: HWND,
    meter: Meter,
}

/// The frame rate, shown in the window's caption every second and logged
/// every few seconds.
struct Meter {
    since: std::time::Instant,
    frames: u32,
    #[cfg(feature = "diagnostics")]
    copy_stats_frames: u64,
    drawn: u64,
    logged: std::time::Instant,
    /// The caption the window had, once read.
    caption: Option<Vec<u16>>,
}

impl Meter {
    fn new() -> Self {
        let now = std::time::Instant::now();
        Meter {
            since: now,
            frames: 0,
            #[cfg(feature = "diagnostics")]
            copy_stats_frames: 0,
            drawn: crate::draw::drawn(),
            logged: now,
            caption: None,
        }
    }

    /// Counts a presented frame.
    fn frame(&mut self, hwnd: HWND) {
        #[cfg(feature = "diagnostics")]
        tracing::debug!(target: "victory_gpu::frame", token = crate::frame::current(), "present");
        #[cfg(feature = "diagnostics")]
        crate::copystat::presented();
        self.frames += 1;
        if let Some(host) = crate::gpuprof::on().then(host).flatten()
            && let Some(stamps) = crate::gpuprof::stamps(&host.device, &host.queue)
        {
            // SAFETY: a plain query of our fence.
            crate::gpuprof::presented(stamps, unsafe { host.fence.GetCompletedValue() });
        }
        #[cfg(feature = "diagnostics")]
        {
            self.copy_stats_frames += 1;
        }
        let elapsed = self.since.elapsed();
        if elapsed < std::time::Duration::from_secs(1) {
            return;
        }
        let drawn = crate::draw::drawn();
        let frames = f64::from(self.frames);
        let fps = frames / elapsed.as_secs_f64();
        let ms = elapsed.as_secs_f64() * 1000.0 / frames;
        let draws = (drawn - self.drawn) as f64 / frames;
        (self.since, self.frames, self.drawn) = (std::time::Instant::now(), 0, drawn);
        // Of a frame, ms preparing submissions and ms waiting for the GPU.
        let [cpu, gpu] = [0, 1].map(|i| {
            SUBMISSION_US[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames
        });
        // All of the preparing, part by part (`PREP_SPLIT`).
        let prep_split = format!("{}; further: {}", prep_line(frames, cpu), more_line(frames));
        let [copy_ms, copy_mb, texture_ms, texture_mb] = [0, 1, 2, 3].map(|i| {
            let scale = if i % 2 == 0 { 1e3 } else { 1e6 };
            PREPARING[i].swap(0, Ordering::Relaxed) as f64 / scale / frames
        });
        let [back_ms, back_mb] = [0, 1].map(|i| {
            COPY_BACK[i].swap(0, Ordering::Relaxed) as f64 / [1e3, 1e6][i] / frames
        });
        // Of the copies (`crate::shadow::REFRESH_SPLIT`), per frame: ms
        // asking which pages the title wrote, ms moving bytes, MB found the
        // same (`VICTORY_SKIP_SAME`); and ms of the asking made once per
        // submission before its refreshes (not part of "copies").
        let [ask_ms, move_ms, same_mb, once_ms] = [0, 1, 2, 3].map(|i| {
            let n = crate::shadow::REFRESH_SPLIT[i].swap(0, Ordering::Relaxed) as f64;
            if i == 2 { n / 1e6 / frames } else { n / 1e3 / frames }
        });
        // Where the asking goes (`crate::watch::split_line`), on all threads.
        let watch_split = crate::watch::split_line(frames);
        // What the journal's 4 KiB parts saved (`VICTORY_FINE_PAGES`), and
        // what the askings noted written.
        let fine_line = crate::shadow::fine_line(frames);
        // Of the copying back (`BACK_SPLIT`), per frame: ms merging what
        // was fetched from copies in the card's memory, ms comparing copies
        // in the PC's memory; with `VICTORY_MERGE_EXACT`, MB the merges
        // still send up of the MB they merged.
        let [merge_ms, compare_ms] =
            [0, 1].map(|i| BACK_SPLIT[i].swap(0, Ordering::Relaxed) as f64 / 1e3 / frames);
        let [exact_up_mb, exact_merged_mb] =
            [0, 1].map(|i| crate::shadow::MERGE_SENT[i].swap(0, Ordering::Relaxed) as f64 / 1e6 / frames);
        let dropped = crate::draw::take_dropped() as f64 / frames;
        let indirect = crate::list::take_indirect() as f64 / frames;
        let mesh_largest = MESH_GROUPS[0].swap(0, Ordering::Relaxed);
        let mesh_groups = MESH_GROUPS[1].swap(0, Ordering::Relaxed) as f64 / frames;
        let amplified = crate::draw::AMPLIFIED.swap(0, Ordering::Relaxed) as f64 / frames;
        let amplified_waits = crate::draw::AMPLIFIED_WAITS.swap(0, Ordering::Relaxed) as f64 / frames;
        tracing::info!(
            "indirect mesh dispatches: the largest group count asked for {mesh_largest} (run: at most 65535), {mesh_groups:.0} groups asked per frame; {amplified:.0} draws a frame launched by the amplification stage, without a clamp pass (VICTORY_MESH_AMPLIFY={}), {amplified_waits:.1} waits a frame where their runs end (VICTORY_MESH_AMPLIFY_WAIT={})",
            u8::from(victory_shader::emit::mesh_amplify()),
            crate::draw::amplify_wait()
        );
        // Read before `late_line` takes the counts.
        let settles_text = settle_split(frames);
        // (Run 200 counted indirect compute records asking for 77,000 to
        // 98,000 groups in x, 0.2 to 0.8 a frame, cut to the 65,535 one
        // dispatch runs; the counter went when the clamp's budget moved to
        // the card's memory.)
        tracing::info!("{}", crate::indirect::calls_line(frames));
        tracing::info!("{}", crate::barstat::line(frames));
        {
            let [texture_us, slots] = [0, 1].map(|i| crate::draw::HEAP_PASS_PARTS[i].swap(0, Ordering::Relaxed));
            tracing::info!(
                "the pass over the heap per frame (the frame line's \"heap buffers ... ms\" is both parts): {:.1} ms in its                  pass over textures picked by index; {:.0} buffer slots looked at by its pass over buffer views",
                texture_us as f64 / 1e3 / frames,
                slots as f64 / frames
            );
        }
        {
            use crate::draw::{PICKED_SET, PICKED_STATS};
            let [draws, lists, ranges, bytes, readers, over] =
                [0, 1, 2, 3, 4, 5].map(|i| PICKED_STATS[i].swap(0, Ordering::Relaxed) as f64 / frames);
            let set = PICKED_SET.each_ref().map(|n| n.load(Ordering::Relaxed));
            let analysis = crate::pipeline::PICKED_ANALYSIS.each_ref().map(|n| n.load(Ordering::Relaxed));
            let mb = |b: u64| b as f64 / 1e6;
            tracing::info!(
                "views picked from the heap by index, per frame: {draws:.0} draws and dispatches that can store through them \
                 in {lists:.1} lists ({readers:.0} that only read them); {ranges:.0} ranges {:.1} MB declared written for them \
                 (VICTORY_PICKED_UAV_WRITES {}; {over:.1} lists declared none, the set over VICTORY_PICKED_UAV_WRITES_MB); the heap's unordered access buffer views at the last measure ({} so far): {} views \
                 in {} copies, {:.1} MB joined in {} ranges ({:.1} MB first to last byte per copy), of it {:.1} MB in the card's \
                 memory, {:.1} MB in GPU upload heaps, {:.1} MB in place; programs analysed so far {} ({} store through picked \
                 views, {} not told, {:.1} ms)",
                bytes / 1e6,
                if crate::draw::picked_uav_writes() { "on" } else { "off: counted only" },
                set[7],
                set[0],
                set[1],
                mb(set[2]),
                set[8],
                mb(set[3]),
                mb(set[4]),
                mb(set[5]),
                mb(set[6]),
                analysis[0],
                analysis[1],
                analysis[2],
                analysis[3] as f64 / 1e3,
            );
        }
        crate::shadow::picked_guard_line(frames);
        crate::draw::picked_exact_line(frames);
        crate::shadow::exact_line(frames);
        crate::bus::line(frames);
        crate::descriptor::sync_line(frames);
        late_line(frames);
        let clock = card_clock_line(frames, ms);
        path_line(frames, ms, clock);
        crate::queue2::line(frames);
        crate::queue::async_line(frames);
        crate::results::line(frames);
        crate::plaincopy::line(frames);
        if let Some(host) = host() {
            host.drain_messages();
        }
        {
            let held = crate::pipeline::WALKS_HELD.swap(0, Ordering::Relaxed);
            if held > 0 {
                tracing::info!(
                    "dispatches of programs that walk ray-tracing structures held back per frame (VICTORY_BVH_HOLD=1; the                      builder has not built them): {:.1}",
                    held as f64 / frames
                );
            }
        }
        crate::shadow::back_line(frames);
        {
            let [passes, pieces, compares] = [0, 1, 2].map(|i| CARD_PASSES[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if passes + compares > 0.0 {
                tracing::info!(
                    "our passes over copies in the card's memory per frame: {passes:.0} hash passes over {pieces:.0} pieces after \
                     ranges were sent up (VICTORY_CARD_REHASH_RUNS=1 joins them: now {}), {compares:.0} compare passes",
                    if rehash_runs() { "on" } else { "off" }
                );
            }
        }
        // Of the texture refreshes (`crate::texture::REFRESH_WAITS`), per
        // frame in ms on all threads: waiting for a texture another thread
        // refreshes, asking the write watch and the guards, waiting for
        // title memory to hold still, and all of it; for the frame line too.
        let [own, asking, still, all] =
            [0, 1, 2, 3].map(|i| crate::texture::REFRESH_WAITS[i].swap(0, Ordering::Relaxed) as f64 / 1000.0 / frames);
        // How they ended (`crate::texture::REFRESH_ENDS`), per frame; the
        // longest wait for title memory in ms, and ms detiling per frame.
        let ends = {
            use crate::texture::{REFRESH_ENDS, ends};
            let mut n = [0.0f64; 12];
            for (i, n) in n.iter_mut().enumerate() {
                let v = REFRESH_ENDS[i].swap(0, Ordering::Relaxed) as f64;
                *n = match i {
                    ends::LONGEST_WAIT => v / 1e3,
                    ends::DETILE_US => v / 1e3 / frames,
                    _ => v / frames,
                };
            }
            n
        };
        // Changes of mapping by the runtime (`crate::watch::REMAPS`): per
        // frame, and ms per frame they held title memory, the longest in ms,
        // and ms per frame the runtime waited to begin them.
        let remaps = {
            let [n, held, longest, waited] =
                [0, 1, 2, 3].map(|i| crate::watch::REMAPS[i].swap(0, Ordering::Relaxed) as f64);
            [n / frames, held / 1e3 / frames, longest / 1e3, waited / 1e3 / frames]
        };
        {
            if all >= 1.0 {
                tracing::info!(
                    "texture refreshes per frame, all threads: {all:.1} ms, of them {own:.1} ms waiting for a texture another thread \
                     is refreshing, {asking:.1} ms asking the write watch and the guards, {still:.1} ms waiting for title memory to \
                     hold still (the runtime remapping it); the rest is reading, hashing and detiling"
                );
            }
            let [through, left] = [0, 1].map(|i| crate::texture::UNLOCKED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if through + left > 0.0 {
                tracing::info!(
                    "texture refreshes that did not wait for title memory to hold still, per frame (VICTORY_TEXTURE_NO_WAIT=1): \
                     {through:.1} read through the kernel, {left:.1} left for the next refresh (a page was not there)"
                );
            }
        }
        {
            let [copies, joined] = [0, 1].map(|i| COPIES_JOINED[i].swap(0, Ordering::Relaxed) as f64 / frames);
            if copies > 0.0 {
                tracing::info!(
                    "title buffer copies recorded per frame: {copies:.0}, of them {joined:.0} continue the one before \
                     (VICTORY_JOIN_COPIES=1 makes those one copy with it: now {})",
                    if join_copies() { "on" } else { "off" }
                );
            }
        }
        // Passes over the heap's buffer views for shaders that pick them by index.
        let picked = crate::draw::HEAP_BUFFER_PASSES.swap(0, Ordering::Relaxed) as f64 / frames;
        let picked_ms =
            crate::draw::HEAP_BUFFER_MICROS.swap(0, Ordering::Relaxed) as f64 / 1e3 / frames;
        // Of them, per frame (`HEAP_BUFFER_SPLIT`): ms walking the slots,
        // building the spans and deferring their refreshes, and in the
        // texture side; slots walked; walks of every slot and their ms.
        let [walk_ms, spans_ms, tex_ms, walked, full, full_ms] = [0, 1, 2, 3, 4, 5].map(|i| {
            let n = crate::draw::HEAP_BUFFER_SPLIT[i].swap(0, Ordering::Relaxed) as f64;
            if matches!(i, 3 | 4) { n / frames } else { n / 1e6 / frames }
        });
        // Of the walk, per frame (`HEAP_WALK_SPLIT`): ms checking kept
        // resolutions (`VICTORY_HEAP_SLOT_CACHE`), in `mirror`, in the copy
        // lookup; slots kept, resolved, of those failed checks; slots of
        // 0.1 ms or more and their ms; ms of the spans building them.
        let [w_check_ms, w_mirror_ms, w_copy_ms, w_kept, w_resolved, w_failed, w_slow, w_slow_ms, w_build_ms] =
            [0, 1, 2, 3, 4, 5, 6, 7, 8].map(|i| {
                let n = crate::draw::HEAP_WALK_SPLIT[i].swap(0, Ordering::Relaxed) as f64;
                if matches!(i, 3..=6) { n / frames } else { n / 1e6 / frames }
            });
        // Of the texture side, per frame (`HEAP_TEXTURE_SPLIT`): ms taking
        // the queue, asking what is mapped, making, staging, checking;
        // slots looked at; passes stopped by the time and the byte budget.
        let [tq_ms, tlook_ms, tmake_ms, tstage_ms, tcheck_ms, tlooked, tby_time, tby_bytes] =
            [0, 1, 2, 3, 4, 5, 6, 7].map(|i| {
                let n = crate::draw::HEAP_TEXTURE_SPLIT[i].swap(0, Ordering::Relaxed) as f64;
                if i >= 5 { n / frames } else { n / 1e6 / frames }
            });
        // Staging buffers taken from those kept (`VICTORY_STAGING_POOL_MB`)
        // and made anew, per frame.
        let [pool_hits, pool_made] =
            [0, 1].map(|i| crate::texture::POOL_USE[i].swap(0, Ordering::Relaxed) as f64 / frames);
        // Textures made ready on worker threads (`VICTORY_ASYNC_TEXTURES`):
        // queued, prepared, recorded per frame, and what is staged waiting.
        let ahead = crate::prepare::line(frames);
        // Of the queue and the looks, per frame (`HEAP_LOOK_SPLIT`): ms
        // taking the remapped ranges, the due slots and census, and letting
        // go of untaken items; slots left at the cap and items taken without
        // a look (`VICTORY_CHEAP_LOOKS`). And what is mapped asked, on all
        // threads: questions and ms (`crate::texture::MAPPED_ASKS`).
        let [q_remapped, q_due, q_sweep, l_capped, l_ready] = [0, 1, 2, 3, 4].map(|i| {
            let n = crate::draw::HEAP_LOOK_SPLIT[i].swap(0, Ordering::Relaxed) as f64;
            if i >= 3 { n / frames } else { n / 1e6 / frames }
        });
        let [asks, asks_ms] = [0, 1].map(|i| {
            let n = crate::texture::MAPPED_ASKS[i].swap(0, Ordering::Relaxed) as f64;
            if i == 0 { n / frames } else { n / 1e6 / frames }
        });
        // Copies between depth textures and their colour twins, each way.
        let [twin_to_depth, twin_to_colour] =
            [2, 3].map(|i| crate::resource::TWINS[i].swap(0, Ordering::Relaxed) as f64 / frames);
        // Depth planes drawn from their twins.
        let planes = crate::resource::PLANES[0].swap(0, Ordering::Relaxed) as f64 / frames;
        // Copies in the card's memory (`crate::shadow::cards`).
        let card = crate::shadow::card_stats();
        // Of them, per frame: MB sent up and in how many ranges; fetches of
        // what shaders wrote there (each a wait for the card) and their ms.
        let [up_mb, up_ranges, fetches, fetch_ms] =
            [(0, 1e6), (1, 1.0), (5, 1.0), (6, 1e3)].map(|(i, scale)| card[i] as f64 / scale / frames);
        #[cfg(feature = "diagnostics")]
        let slow = crate::watch::slow_asks();
        // Over the same frames as the counts above.
        #[cfg(feature = "diagnostics")]
        let drops = crate::pipeline::take_drops();
        if self.logged.elapsed() >= std::time::Duration::from_secs(5) {
            self.logged = std::time::Instant::now();
            tracing::info!(
                "placed texture views: {} writes of another view of the texture a slot showed (it kept the old view until the new was made)",
                crate::descriptor::host::KEPT_VIEWS.load(Ordering::Relaxed)
            );
            tracing::info!(
                "copies left out since the start because both sides were one host texture: {}",
                crate::list::SELF_COPIES.load(Ordering::Relaxed)
            );
            if crate::drvwatch::errors() > 0 {
                tracing::warn!(
                    "errors the graphics driver reported to the system since the start: {} (the paths under suspicion are off)",
                    crate::drvwatch::errors()
                );
            }
            crate::tiles::status();
            {
                let [queued, due, away, partly, oldest] =
                    [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
                let [same, parts, tails] =
                    [0, 1, 2].map(|i| crate::draw::HEAP_TEXTURES_PARTLY[i].load(Ordering::Relaxed));
                tracing::info!(
                    "textures picked by index, the queue: {queued} slots, {due} of them due (the oldest written {oldest} frames ago), \
                     {away} waiting for their memory, {partly} uploaded in part; since the start: {} slots found uploaded already by \
                     the draws that read them through tables, {} writes of tables' slots (due some frames later), {} slots taken out \
                     as read through a table, {parts} uploads of textures not all mapped ({tails} of them without their first page), \
                     {same} looks that found no more mapped; {} slots made stale by a newer view of other memory there, \
                     {} ranges of memory mapped, unmapped or written under texture views ({} dropped), {} slots made due by them, \
                     {} made slots found unmapped and made null",
                    crate::draw::HEAP_TEXTURES_TABLED.load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[0].load(Ordering::Relaxed),
                    crate::descriptor::host::TABLED_WRITES[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[0].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[1].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[2].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[3].load(Ordering::Relaxed),
                    crate::descriptor::host::STALE[4].load(Ordering::Relaxed)
                );
            }
            {
                let mode = crate::watch::page_writes();
                let [writes, bytes, guarded, plain] = [0, 1, 2, 3].map(|i| crate::watch::WRITTEN[i].load(Ordering::Relaxed));
                let [kept, dropped, slots, uploaded] =
                    [0, 1, 2, 3].map(|i| crate::descriptor::host::PAGE_WRITES[i].load(Ordering::Relaxed));
                let [behind, oldest, refreshed] =
                    [0, 1, 2].map(|i| crate::descriptor::host::PAGE_BEHIND[i].load(Ordering::Relaxed));
                tracing::info!(
                    "writes behind the guards, since the start: {writes} DirectStorage writes into physical pages at addresses the                      title had mapped ({} MB); {guarded} guarded 64 KiB blocks of host copies lay under them ({}); {plain} reads to                      plain addresses finished ({}); {kept} ranges under texture views kept for the texture pass ({dropped} dropped);                      {slots} slots of placed textures held their bytes, {uploaded} of them uploaded before the write ({}),                      {refreshed} made due for it, {} worker reads from before it let go; textures older than the title's last write                      now: {behind} slots (the oldest write {oldest} frames ago)                      (VICTORY_PAGE_WRITES: unset or 1 the guards and the slots, guards the guards alone, 0 counts only, slots, all)",
                    bytes >> 20,
                    if mode & 1 != 0 { "opened, taken again" } else { "left as they were" },
                    if mode & 4 != 0 { "told as well" } else if mode & 2 != 0 { "their texture slots told" } else { "counted only" },
                    if mode & 2 != 0 { "made due again" } else { "left as they were" },
                    crate::prepare::FORGOTTEN.load(Ordering::Relaxed)
                );
            }
            #[cfg(feature = "diagnostics")]
            tracing::info!(
                "write-watch askings of a millisecond or more: {:.1} a frame, {:.1} ms a frame; the longest asking {:.1} ms",
                slow[0] as f64 / frames,
                slow[1] as f64 / 1e3 / frames,
                slow[2] as f64 / 1e3
            );
            if crate::shadow::cards() {
                let per_frame = |i: usize| card[i] as f64 / frames;
                tracing::info!(
                    "host copies in the card's memory: {} copies {} MB, their bytes from before {} MB; per frame: \
                     {:.1} MB sent up in {:.0} ranges, {:.0} pieces compared on the card, {:.0} of them fetched \
                     ({:.1} MB) with {:.1} more waits for the card ({:.1} ms)",
                    card[7],
                    card[8] >> 20,
                    card[9] >> 20,
                    per_frame(0) / 1e6,
                    per_frame(1),
                    per_frame(2),
                    per_frame(3),
                    per_frame(4) / 1e6,
                    per_frame(5),
                    per_frame(6) / 1e3
                );
            }
            tracing::info!(
                "{fps:.1} fps, {ms:.1} ms/frame ({cpu:.1} preparing [all of it: {prep_split}]: copies {copy_ms:.1} ms \
                 {copy_mb:.1} MB [asking {ask_ms:.1} ms + {once_ms:.1} ms once a submission ({watch_split}), moving {move_ms:.1} ms, \
                 {same_mb:.1} MB found the same; {fine_line}; {up_mb:.1} MB sent up to the card in {up_ranges:.0} ranges], \
                 textures {texture_ms:.1} ms {texture_mb:.1} MB [waiting for title memory to hold still {still:.1} ms \
                 ({long_waits:.1} waits of 1 ms or more, the longest {longest_wait:.1} ms), asking {asking:.1} ms, \
                 another thread's texture {own:.1} ms, detiling {detile_ms:.1} ms, the rest {tex_rest:.1} ms; \
                 {refreshes:.0} refreshes: {unwritten:.0} unwritten, {guarded:.0} guarded, {watched:.0} written, \
                 {partial_skipped:.0} partly mapped passed over ({partial_early:.0} without waiting), {partial_read:.0} \
                 partly mapped read, {same:.0} the same, {detiled:.0} detiled; the runtime changed mappings {remap_n:.0} \
                 times holding {remap_held:.1} ms (the longest {remap_longest:.1} ms) and waited {remap_waited:.1} ms \
                 to begin them]; {gpu:.1} on the \
                 GPU [{settles_text}]; {back_ms:.1} copying back {back_mb:.1} MB [{fetches:.1} fetches from the card {fetch_ms:.1} ms, \
                 merging {merge_ms:.1} ms, CPU compares {compare_ms:.1} ms; merges sent up {exact_up_mb:.1} of \
                 {exact_merged_mb:.1} MB]), {draws:.0} draws/frame, \
                 dropped {dropped:.0} without a host pipeline ({making} being made) and \
                 {indirect:.0} indirect; heap buffers {picked:.1} passes {picked_ms:.1} ms \
                 (walk {walk_ms:.1} ms over {walked:.0} slots [{w_kept:.0} kept, checking {w_check_ms:.1} ms; \
                 {w_resolved:.0} resolved, mirror {w_mirror_ms:.1} ms, lookups {w_copy_ms:.1} ms, {w_slow:.1} of 0.1 ms or more \
                 {w_slow_ms:.1} ms; {w_failed:.0} failed checks], {full:.2} full walks {full_ms:.1} ms; spans {spans_ms:.1} ms \
                 (building {w_build_ms:.1}); \
                 textures {tex_ms:.1} ms: queue {tq_ms:.1}, looks {tlook_ms:.1}, making {tmake_ms:.1}, staging {tstage_ms:.1}, \
                 checks {tcheck_ms:.1}, {tlooked:.0} slots, stopped by time {tby_time:.2} by bytes {tby_bytes:.2}; \
                 queue: remapped {q_remapped:.1} ms, due {q_due:.1} ms, sweep {q_sweep:.1} ms; looks: {asks:.0} mapped asks \
                 {asks_ms:.1} ms (all threads), {l_capped:.0} left at the cap and {l_ready:.0} ready taken unasked; \
                 staging pool {pool_hits:.0} reused {pool_made:.0} made; {ahead}); texture slots written so far: {placed} placed, \
                 {objects} of objects; \
                 textures picked by index: {picked_textures} uploaded so far ({picked_mb} MB), \
                 {picked_waiting} waiting, {picked_away} passed over unmapped; depth textures with                  unordered access: {du0} made, bound as depth {du1}x, cleared as depth {du2}x,                  unordered views refused {du3}x; buffer views left null as the host would refuse them: {refused}",
                long_waits = ends[crate::texture::ends::LONG_WAITS],
                longest_wait = ends[crate::texture::ends::LONGEST_WAIT],
                detile_ms = ends[crate::texture::ends::DETILE_US],
                tex_rest = (all - still - asking - own - ends[crate::texture::ends::DETILE_US]).max(0.0),
                refreshes = ends[crate::texture::ends::ALL],
                unwritten = ends[crate::texture::ends::UNWRITTEN],
                guarded = ends[crate::texture::ends::GUARDED],
                watched = ends[crate::texture::ends::WATCHED],
                partial_skipped = ends[crate::texture::ends::PARTIAL_SKIPPED],
                partial_early = ends[crate::texture::ends::PARTIAL_EARLY],
                partial_read = ends[crate::texture::ends::PARTIAL_READ],
                same = ends[crate::texture::ends::SAME],
                detiled = ends[crate::texture::ends::DETILED],
                remap_n = remaps[0],
                remap_held = remaps[1],
                remap_longest = remaps[2],
                remap_waited = remaps[3],
                picked_textures = crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
                picked_mb = crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
                picked_waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed),
                picked_away = crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed),
                du0 = crate::resource::DEPTH_UNORDERED[0].load(Ordering::Relaxed),
                du1 = crate::resource::DEPTH_UNORDERED[1].load(Ordering::Relaxed),
                du2 = crate::resource::DEPTH_UNORDERED[2].load(Ordering::Relaxed),
                du3 = crate::resource::DEPTH_UNORDERED[3].load(Ordering::Relaxed),
                refused = format!(
                    "{}; depth twins: {} alive, {} depth textures left without one; twin copies per frame: {twin_to_depth:.1} into depth textures, {twin_to_colour:.1} into twins; depth planes: {planes:.1} drawn from their twins per frame, {} times not given back to a twin, {} times used before the pass",
                    crate::descriptor::host::REFUSED_VIEWS.load(Ordering::Relaxed),
                    crate::resource::TWINS[0].load(Ordering::Relaxed),
                    crate::resource::TWINS[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[1].load(Ordering::Relaxed),
                    crate::resource::PLANES[2].load(Ordering::Relaxed)
                ),
                making = crate::pipeline::making(),
                placed = crate::descriptor::host::PLACED_TEXTURE_SLOTS.load(Ordering::Relaxed),
                objects = crate::descriptor::host::OBJECT_TEXTURE_SLOTS.load(Ordering::Relaxed)
            );
            // Where the memory is, by what holds it (`crate::memline`).
            crate::memline::log();
            crate::inplace::frame_line(frames, copy_mb, texture_mb);
            #[cfg(feature = "diagnostics")]
            {
                crate::copystat::log(std::mem::take(&mut self.copy_stats_frames) as f64);
                for (why, n) in drops.iter().take(40) {
                    tracing::info!(
                        "dropped {:.1}/frame without a host pipeline: {why}",
                        *n as f64 / frames
                    );
                }
            }
        }
        if self.caption.is_none() {
            // The window's own text, read without a message.
            let mut buffer = vec![0u16; 512];
            // SAFETY: a window handle and a buffer of that length.
            let n = unsafe { InternalGetWindowText(hwnd, &mut buffer) };
            buffer.truncate(n.max(0) as usize);
            self.caption = Some(buffer);
        }
        let Some(caption) = &self.caption else { return };
        let text = format!(" | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame");
        let mut text: Vec<u16> = caption.iter().copied().chain(text.encode_utf16()).collect();
        text.push(0);
        // Setting the text waits for the window's thread (a title thread,
        // which may not pump for a while), so a thread of ours does it and
        // presenting never waits; it takes the newest text when free.
        static PENDING: Mutex<Option<(usize, Vec<u16>)>> = Mutex::new(None);
        static SETTER: std::sync::Once = std::sync::Once::new();
        *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((hwnd.0 as usize, text));
        SETTER.call_once(|| {
            let _ = std::thread::Builder::new()
                .name("victory-caption".into())
                .spawn(|| {
                    loop {
                        let next = PENDING.lock().unwrap_or_else(|e| e.into_inner()).take();
                        match next {
                            // SAFETY: a window handle and a NUL-terminated text.
                            Some((hwnd, text)) => unsafe {
                                let _ = SetWindowTextW(HWND(hwnd as *mut _), PCWSTR(text.as_ptr()));
                            },
                            None => std::thread::sleep(std::time::Duration::from_millis(100)),
                        }
                    }
                });
        });
    }
}

impl Host {
    /// Shows `texture`, which is in `state` (a host state) and stays in it,
    /// as frame `frame` (saved if captures are on). `None` if nothing was
    /// shown; else, in diagnostics builds, whether the frame was black.
    pub(crate) fn present(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        let showing = std::time::Instant::now();
        let shown = self.present_now(texture, state, frame);
        crate::framelog::add(crate::framelog::T::PresentShowUs, showing.elapsed().as_micros() as u64);
        shown
    }

    fn present_now(&self, texture: &ID3D12Resource, state: u32, frame: u64) -> Option<bool> {
        // SAFETY: a live resource.
        let desc = unsafe { texture.GetDesc() };
        let Some(format) = swap_format(desc.Format) else {
            tracing::warn!(format = desc.Format.0, "cannot present this format");
            return None;
        };
        let (width, height) = (desc.Width as u32, desc.Height);
        let mut presenter = self.presenter.lock().unwrap_or_else(|e| e.into_inner());
        if presenter.is_none() {
            *presenter = match self.presenter(width, height, format) {
                Ok(p) => Some(p),
                Err(e) => {
                    tracing::warn!(error = %e, "no swap chain");
                    return None;
                }
            };
        }
        let p = presenter.as_mut()?;
        if (p.width, p.height, p.format) != (width, height, format) {
            // Presents sent ahead hold back buffers until they have run.
            settle_now(WHY_SYNC);
            // SAFETY: no back buffer is held; every submission was waited for.
            let resized = unsafe {
                p.swap_chain
                    .ResizeBuffers(BUFFERS, width, height, format, DXGI_SWAP_CHAIN_FLAG(0))
            };
            match resized {
                Ok(()) => {
                    (p.width, p.height, p.format) = (width, height, format);
                    tracing::info!(
                        width,
                        height,
                        format = format.0,
                        frame,
                        "the swap chain's buffers were made again for another size or format"
                    );
                    self.log_video_memory("right after the swap chain's buffers were made again");
                }
                Err(e) => {
                    tracing::warn!(error = %e, width, height, "swap chain not resized");
                    return None;
                }
            }
        }
        let state = D3D12_RESOURCE_STATES(state as i32);
        // SAFETY: our swap chain, list and a live source texture.
        unsafe {
            let index = p.swap_chain.GetCurrentBackBufferIndex();
            let Ok(back) = p.swap_chain.GetBuffer::<ID3D12Resource>(index) else {
                return None;
            };
            // Sent ahead ([`late_free_present`]) unless the frame is to be
            // read back at once (a saved or listed frame): then a list of
            // its own, one whose last flight has finished or a new one.
            #[cfg(feature = "diagnostics")]
            let read_now = crate::capture::wanted(frame) || crate::capture::listed(frame);
            #[cfg(not(feature = "diagnostics"))]
            let read_now = false;
            let slot = if late_free_present() && !read_now {
                let finished = FINISHED.load(Ordering::Acquire);
                p.ahead.iter().position(|(_, flown)| *flown <= finished).or_else(|| {
                    let list = self.new_list().ok()?;
                    list.close();
                    p.ahead.push((list, 0));
                    Some(p.ahead.len() - 1)
                })
            } else {
                None
            };
            let copy = match slot {
                Some(i) => &p.ahead[i].0,
                None => &p.copy,
            };
            copy.reset();
            let list = &copy.list;
            // `VICTORY_BRIGHTEN`: the image through the curve first, and the
            // window copies that (`crate::brighten`).
            let bright = crate::brighten::settings().and_then(|curve| {
                static BRIGHTEN: Mutex<Option<Option<crate::brighten::Brighten>>> = Mutex::new(None);
                let mut b = BRIGHTEN.lock().unwrap_or_else(|e| e.into_inner());
                let b = b.get_or_insert_with(|| {
                    crate::brighten::Brighten::new(&self.device)
                        .map_err(|e| tracing::warn!(error = %e, "VICTORY_BRIGHTEN: no pipeline"))
                        .ok()
                });
                // A typeless texture is read as the swap chain's format.
                let view = if matches!(desc.Format.0, 27 | 90 | 23 | 9) { format } else { desc.Format };
                b.as_mut()?.record(&self.device, list, texture, state, view, format, (width, height), curve)
            });
            match &bright {
                Some(bright) => {
                    crate::barstat::record(&list, crate::barstat::Why::Present, &[transition(
                        &back,
                        D3D12_RESOURCE_STATE_PRESENT,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                    )]);
                    list.CopyResource(&back, bright);
                }
                None => {
                    crate::barstat::record(&list, crate::barstat::Why::Present, &[
                        transition(texture, state, D3D12_RESOURCE_STATE_COPY_SOURCE),
                        transition(
                            &back,
                            D3D12_RESOURCE_STATE_PRESENT,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                        ),
                    ]);
                    list.CopyResource(&back, texture);
                }
            }
            // (the capture and the probe read the title's image in COPY_SOURCE:
            // not with `VICTORY_BRIGHTEN`, which leaves it in its own state)
            #[cfg(feature = "diagnostics")]
            let capture = (crate::capture::wanted(frame) && bright.is_none())
                .then(|| crate::capture::read_back(&self.device, list, texture))
                .flatten();
            #[cfg(feature = "diagnostics")]
            let probe = if slot.is_some() || bright.is_some() {
                None
            } else {
                crate::capture::probe(&self.device, list, texture)
            };
            #[cfg(not(feature = "diagnostics"))]
            let _ = frame;
            match &bright {
                Some(bright) => {
                    crate::brighten::Brighten::after_copy(list, bright);
                    crate::barstat::record(&list, crate::barstat::Why::Present, &[transition(
                        &back,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        D3D12_RESOURCE_STATE_PRESENT,
                    )]);
                }
                None => {
                    crate::barstat::record(&list, crate::barstat::Why::Present, &[
                        transition(texture, D3D12_RESOURCE_STATE_COPY_SOURCE, state),
                        transition(
                            &back,
                            D3D12_RESOURCE_STATE_COPY_DEST,
                            D3D12_RESOURCE_STATE_PRESENT,
                        ),
                    ]);
                }
            }
            copy.close();
            match slot {
                Some(i) => {
                    let flown = self.execute_own(copy);
                    p.ahead[i].1 = flown;
                }
                None => self.execute(&[copy]),
            }
            #[cfg(feature = "diagnostics")]
            if let Some((buffer, footprint)) = capture {
                crate::capture::save_read_back(frame, &buffer, &footprint);
            }
            let presented = p.swap_chain.Present(0, DXGI_PRESENT(0));
            if presented.is_err() {
                // A refused present shows nothing (the window keeps what it
                // had): said, a few times.
                static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
                if SAID.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::warn!(error = %windows::core::Error::from_hresult(presented), frame, "the swap chain refused a present");
                }
            }
            p.meter.frame(p.hwnd);
            #[cfg(feature = "diagnostics")]
            return Some(probe.is_some_and(|b| crate::capture::black(&b)));
            #[cfg(not(feature = "diagnostics"))]
            Some(false)
        }
    }

    fn presenter(
        &self,
        width: u32,
        height: u32,
        format: DXGI_FORMAT,
    ) -> windows::core::Result<Presenter> {
        let (hwnd, own) = match title_window() {
            Some(hwnd) => (hwnd, false),
            None => (window(width, height)?, true),
        };
        self.log_video_memory("right before the swap chain is made");
        // SAFETY: plain object creation for our window.
        unsafe {
            let factory: IDXGIFactory2 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0))?;
            let desc = DXGI_SWAP_CHAIN_DESC1 {
                Width: width,
                Height: height,
                Format: format,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                BufferUsage: DXGI_USAGE_RENDER_TARGET_OUTPUT,
                BufferCount: BUFFERS,
                SwapEffect: DXGI_SWAP_EFFECT_FLIP_DISCARD,
                ..Default::default()
            };
            let swap_chain = factory
                .CreateSwapChainForHwnd(&self.queue, hwnd, &desc, None, None)?
                .cast::<IDXGISwapChain3>()?;
            let copy = self.new_list()?;
            copy.close();
            tracing::info!(
                width,
                height,
                format = format.0,
                window = format_args!("{:#x}", hwnd.0 as usize),
                title_window = !own,
                "presenting to a window"
            );
            self.log_video_memory("right after the swap chain is made");
            Ok(Presenter {
                swap_chain,
                width,
                height,
                format,
                copy,
                ahead: Vec::new(),
                hwnd,
                meter: Meter::new(),
            })
        }
    }
}

/// The title's main window as the runtime saw it made
/// ([`title_window_made`]), and the area it was asked with: 0 if none.
static TITLE_WINDOW: Mutex<(usize, i64)> = Mutex::new((0, 0));

/// The title made a top-level window `hwnd` of `width` x `height` (as it
/// asked, before any fitting). The biggest is the one frames go to: a
/// window found by [`title_window`]'s walk had to be visible already when
/// the first frame was presented, and a title that shows its window only
/// after that got a window of ours instead, its own staying white for the
/// whole run (and its caption never showing the frame rate).
pub fn title_window_made(hwnd: usize, width: i32, height: i32) {
    if hwnd == 0 {
        return;
    }
    let area = i64::from(width.max(1)) * i64::from(height.max(1));
    let mut known = TITLE_WINDOW.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: a plain query on a handle that may be stale.
    let alive = known.0 != 0 && unsafe { IsWindow(Some(HWND(known.0 as *mut _))) }.as_bool();
    if !alive || area > known.1 {
        *known = (hwnd, area);
    }
}

/// The window frames are presented to: the title's main window if the
/// runtime saw it made ([`title_window_made`]) and it still stands, else a
/// visible top-level window of this process (the one the title made, if it
/// made one).
fn title_window() -> Option<HWND> {
    let known = TITLE_WINDOW.lock().unwrap_or_else(|e| e.into_inner()).0;
    // SAFETY: a plain query on a handle that may be stale.
    if known != 0 && unsafe { IsWindow(Some(HWND(known as *mut _))) }.as_bool() {
        return Some(HWND(known as *mut _));
    }
    visible_window()
}

/// A visible top-level window of this process.
fn visible_window() -> Option<HWND> {
    extern "system" fn visit(hwnd: HWND, found: LPARAM) -> windows::core::BOOL {
        let mut process = 0;
        // SAFETY: a window handle from the enumeration; `found` is ours.
        unsafe {
            GetWindowThreadProcessId(hwnd, Some(&mut process));
            if process == windows::Win32::System::Threading::GetCurrentProcessId()
                && IsWindowVisible(hwnd).as_bool()
            {
                *(found.0 as *mut Option<HWND>) = Some(hwnd);
                return false.into();
            }
        }
        true.into()
    }
    let mut found: Option<HWND> = None;
    // SAFETY: the callback only writes `found`, which outlives the call.
    let _ = unsafe { EnumWindows(Some(visit), LPARAM(&raw mut found as isize)) };
    found
}

/// `state` without the bits the host texture's flags forbid: the console
/// lets the title name states its own resources allow (unordered access on
/// a depth target, whose host copy cannot have both).
pub(crate) fn legal_state(texture: &ID3D12Resource, state: u32) -> u32 {
    // SAFETY: a plain query.
    legal_for(unsafe { texture.GetDesc() }.Flags.0 as u32, state)
}

/// [`legal_state`] for a texture with resource flags `flags`.
pub(crate) fn legal_for(flags: u32, state: u32) -> u32 {
    // (flag, set or clear forbids, state bits): render target, unordered
    // access, depth, denied shader resource.
    const FORBIDDEN: [(u32, bool, u32); 4] = [
        (0x1, false, 0x4),
        (0x4, false, 0x8),
        (0x2, false, 0x10 | 0x20),
        (0x8, true, 0x40 | 0x80),
    ];
    FORBIDDEN
        .iter()
        .filter(|&&(flag, deny, _)| (flags & flag != 0) == deny)
        .fold(state, |s, &(_, _, bits)| s & !bits)
}

impl Host {
    /// The card's memory as the system accounts it to this process, in
    /// bytes: the budget and the use of the card's own memory, then of the
    /// PC's memory shared with the card (where the system moves what does
    /// not fit on the card).
    pub(crate) fn video_memory(&self) -> Option<[u64; 4]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        // The card's budget moved by 256 MB or more since the last look:
        // said at once, with what was in use at the last look, so the log
        // has the moment. (It fell from 11,242 MB to some 10,300 MB with
        // the first world frame in runs 179 to 185, and the memory line,
        // seconds later, was the first to show it. We reserve nothing
        // and make the swap chain once, at the start.)
        static LAST: [std::sync::atomic::AtomicU64; 3] = [const { std::sync::atomic::AtomicU64::new(0) }; 3];
        let now = [own.Budget, own.CurrentUsage, shared.CurrentUsage];
        let last = [0, 1, 2].map(|i| LAST[i].swap(now[i], Ordering::Relaxed));
        if last[0] != 0 && last[0].abs_diff(own.Budget) >= (256u64 << 20) {
            tracing::info!(
                "the system changed the budget of the card's memory from {} MB to {} MB (frame {}); at the last look {} MB of the card's and {} MB of the PC's shared memory were in use",
                last[0] >> 20,
                own.Budget >> 20,
                crate::frame::current(),
                last[1] >> 20,
                last[2] >> 20
            );
            self.log_video_memory("right after that change");
        }
        Some([own.Budget, own.CurrentUsage, shared.Budget, shared.CurrentUsage])
    }

    /// Both of the system's memory groups for this process, in bytes: for
    /// the card's own memory and then for the PC's memory shared with the
    /// card, the budget, the use, what could still be reserved and what
    /// is reserved (the layer reserves none).
    pub(crate) fn video_memory_groups(&self) -> Option<[[u64; 4]; 2]> {
        let adapter = self.adapter.as_ref()?;
        let mut own = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        let mut shared = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
        // SAFETY: plain queries that fill our structures.
        unsafe {
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut own)
                .ok()?;
            adapter
                .QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut shared)
                .ok()?;
        }
        let all = |m: &DXGI_QUERY_VIDEO_MEMORY_INFO| {
            [m.Budget, m.CurrentUsage, m.AvailableForReservation, m.CurrentReservation]
        };
        Some([all(&own), all(&shared)])
    }

    /// Logs [`Self::video_memory_groups`], with `when` it is said.
    pub(crate) fn log_video_memory(&self, when: &str) {
        let Some([own, shared]) = self.video_memory_groups() else {
            return;
        };
        tracing::info!(
            "video memory as the system counts it, {}: the card's own: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved; the PC's shared with the card: budget {} MB, in use {} MB, {} MB could still be reserved, {} MB reserved (the layer reserves none)",
            when,
            own[0] >> 20,
            own[1] >> 20,
            own[2] >> 20,
            own[3] >> 20,
            shared[0] >> 20,
            shared[1] >> 20,
            shared[2] >> 20,
            shared[3] >> 20
        );
    }

    /// A buffer in system memory that the CPU maps (write-back, so reading
    /// it is fast too) and shaders read and write: for host copies of title
    /// memory that shaders write, which upload memory cannot be. Starts in
    /// the common state (buffers move out of it implicitly).
    pub(crate) fn shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        MADE[2].fetch_add(1, Ordering::Relaxed);
        MADE[3].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut buffer,
                )
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => Some((buffer?, data.cast())),
            Err(e) => {
                tracing::warn!(error = %e, size, "no shared buffer");
                None
            }
        }
    }

    /// [`Self::shared_buffer`] in the card's own memory: a buffer in a GPU
    /// upload heap (the card's memory mapped for the CPU through its window
    /// on the bus, write-combined), shaders may write it, in the common
    /// state. `None` where the card has no such heap
    /// ([`Self::gpu_upload_heap`]) or it could not be made. The CPU writes
    /// it as fast as upload memory; reading it back crosses the bus,
    /// uncached (`crate::shadow::gpu_upload_copies`).
    pub(crate) fn card_shared_buffer(&self, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
        if !self.gpu_upload_heap() {
            return None;
        }
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: plain creation and mapping of a CPU-visible buffer.
        let made = unsafe {
            self.device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut buffer)
                .and_then(|()| {
                    let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                    b.Map(0, None, Some(&mut data))
                })
        };
        match made {
            Ok(()) => {
                crate::bus::MADE[0].fetch_add(1, Ordering::Relaxed);
                crate::bus::MADE[1].fetch_add(size, Ordering::Relaxed);
                Some((buffer?, data.cast()))
            }
            Err(e) => {
                let n = crate::bus::MADE[2].fetch_add(1, Ordering::Relaxed);
                if n < 8 {
                    tracing::warn!(error = %e, size, "no shared buffer in a GPU upload heap: in the PC's memory instead");
                }
                None
            }
        }
    }

    /// A device-local buffer shaders may write (unordered access) and indirect
    /// execution may read, starting in the common state.
    pub(crate) fn uav_buffer(&self, size: u64) -> Option<ID3D12Resource> {
        MADE[4].fetch_add(1, Ordering::Relaxed);
        MADE[5].fetch_add(size, Ordering::Relaxed);
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size.max(1),
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ..Default::default()
        };
        let mut buffer: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a device-local buffer.
        let made = unsafe {
            self.device.CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
        };
        match made {
            Ok(()) => buffer,
            Err(e) => {
                tracing::warn!(error = %e, size, "no unordered-access buffer");
                None
            }
        }
    }
}

/// For adapters without device-removed data (the software one): the last
/// commands each list of the failed submission noted (diagnostics, with
/// `VICTORY_D3D12_DRED`).
fn last_notes(runs: &[&HostList]) -> String {
    #[cfg(feature = "diagnostics")]
    {
        let mut text = String::new();
        for (i, r) in runs.iter().enumerate() {
            let notes = r.notes.lock().unwrap_or_else(|e| e.into_inner());
            for (n, _) in notes.iter().rev().take(6).rev() {
                text.push_str(&format!("; list {i} of {}, of its {}: {n}", runs.len(), notes.len()));
            }
        }
        text
    }
    #[cfg(not(feature = "diagnostics"))]
    {
        let _ = runs;
        String::new()
    }
}

/// Where the faulting address `va` lies against a range `(start, bytes)` a
/// command reads (diagnostics).
#[cfg(feature = "diagnostics")]
fn fault_in(va: u64, (start, bytes): (u64, u64)) -> String {
    let end = start + bytes;
    if va < start {
        format!(" (fault {:#x} below its source)", start - va)
    } else if va < end {
        format!(" (fault at source +{:#x})", va - start)
    } else {
        format!(" (fault {:#x} past its source's end)", va - end)
    }
}

/// The commands that may have been running where a list stopped at
/// command `done`: those between the barriers around it (work between
/// barriers overlaps on the GPU), by kind, and what the markers just
/// before them say (`HostList::note_op`: diagnostics builds with
/// device-removed data on).
fn in_flight(history: &[D3D12_AUTO_BREADCRUMB_OP], markers: &[(u32, String)], done: usize) -> String {
    let barrier = |op: &D3D12_AUTO_BREADCRUMB_OP| *op == D3D12_AUTO_BREADCRUMB_OP_RESOURCEBARRIER;
    let from = history[..done].iter().rposition(barrier).map_or(0, |b| b + 1);
    let to = history[done..]
        .iter()
        .position(barrier)
        .map_or(history.len(), |b| (done + b).max(done + 1));
    let mut text = format!(", commands {from}..{to} between its barriers:");
    let mut shown = 0;
    // From just before the command the GPU had not finished: the first
    // twelve after the barrier named draws long done when a list of 79
    // mesh draws stopped at its 45th (run 41).
    for i in from.max(done.saturating_sub(4))..to {
        let op = history[i];
        if op == D3D12_AUTO_BREADCRUMB_OP_SETMARKER || shown == 12 {
            continue;
        }
        shown += 1;
        let marker = i
            .checked_sub(1)
            .filter(|&m| history[m] == D3D12_AUTO_BREADCRUMB_OP_SETMARKER)
            .and_then(|m| markers.iter().find(|(at, _)| *at as usize == m))
            .map_or("no marker", |(_, s)| s.as_str());
        text.push_str(&format!("; command {i} (op {}): {marker}", op.0));
    }
    text
}

/// An unordered access barrier on every resource: shader writes before it
/// are done and seen by what follows.
fn uav_barrier_all() -> D3D12_RESOURCE_BARRIER {
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

pub(crate) fn transition(
    resource: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: a borrowed pointer; the barrier never releases it.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

/// A window of the given client size, owned by its own message thread.
fn window(width: u32, height: u32) -> windows::core::Result<HWND> {
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::Builder::new()
        .name("victory-window".into())
        .spawn(move || {
            let made = make_window(width, height);
            let ok = made.is_ok();
            let _ = sender.send(made.map(|h| h.0 as usize));
            if ok {
                pump();
            }
        })
        .map_err(|_| windows::core::Error::empty())?;
    let hwnd = receiver
        .recv()
        .map_err(|_| windows::core::Error::empty())??;
    Ok(HWND(hwnd as *mut _))
}

extern "system" fn window_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    // SAFETY: the default handling for our window.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

fn make_window(width: u32, height: u32) -> windows::core::Result<HWND> {
    const CLASS: PCWSTR = w!("VIctory");
    // SAFETY: registers our class once and creates a window of it.
    unsafe {
        let instance = GetModuleHandleW(None)?;
        let class = WNDCLASSEXW {
            cbSize: size_of::<WNDCLASSEXW>() as u32,
            lpfnWndProc: Some(window_proc),
            hInstance: instance.into(),
            lpszClassName: CLASS,
            hCursor: LoadCursorW(None, IDC_ARROW)?,
            ..Default::default()
        };
        RegisterClassExW(&class);
        let mut rect = RECT {
            left: 0,
            top: 0,
            right: width as i32,
            bottom: height as i32,
        };
        AdjustWindowRect(&mut rect, WS_OVERLAPPEDWINDOW, false)?;
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            CLASS,
            w!("VIctory"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            rect.right - rect.left,
            rect.bottom - rect.top,
            None,
            None,
            Some(instance.into()),
            None,
        )
    }
}

fn pump() {
    let mut message = MSG::default();
    // SAFETY: the standard message loop of this thread's window.
    unsafe {
        while GetMessageW(&mut message, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&message);
            DispatchMessageW(&message);
        }
    }
}

#[cfg(test)]
mod trail {
    /// The trail keeps the newest things, and the report made when the
    /// host device is removed says them and the pipeline library's state
    /// (no card is asked for any of it).
    #[test]
    fn the_removal_report_says_the_newest_things_done() {
        for n in 0..super::TRAIL_MOST + 5 {
            super::trail(|| format!("thing {n} done"));
        }
        let text = super::removal_context();
        assert!(text.contains("the submitting thread was"));
        assert!(text.contains("pipeline library: no call running"));
        assert!(text.contains(&format!("thing {} done", super::TRAIL_MOST + 4)));
        assert!(!text.contains("thing 4 done"), "the oldest entries are dropped");
    }
}

#[cfg(test)]
mod card_copies {
    use super::{Joined, through_card_copies};

    fn pieces(copied: &[(usize, usize, usize)], from: usize, len: usize) -> Vec<(usize, usize, usize)> {
        let mut out = Vec::new();
        through_card_copies(copied, from, len, 0, &mut out);
        out
    }

    /// CPU reads after title copies on the card in the same list read each
    /// byte where the last copy that wrote it read it, through earlier
    /// copies again, and the rest where it is.
    #[test]
    fn reads_after_card_copies_take_their_sources() {
        // Nothing copied, or the read beside the copy: as it is.
        assert_eq!(pieces(&[], 100, 10), vec![(0, 100, 10)]);
        assert_eq!(pieces(&[(200, 500, 16)], 100, 10), vec![(0, 100, 10)]);
        // Inside a copy (0x200 bytes from 1000 to 2000).
        assert_eq!(pieces(&[(2000, 1000, 0x200)], 2010, 4), vec![(0, 1010, 4)]);
        // Over its start and end: before, inside, after.
        assert_eq!(pieces(&[(2000, 1000, 16)], 1996, 24), vec![(0, 1996, 4), (4, 1000, 16), (20, 2016, 4)]);
        // A chain: 1000 -> 2000, then 2000 -> 3000; reading 3000 reads 1000.
        assert_eq!(pieces(&[(2000, 1000, 16), (3000, 2000, 16)], 3004, 8), vec![(0, 1004, 8)]);
        // The later of two copies into the same bytes wins.
        assert_eq!(pieces(&[(2000, 1000, 16), (2000, 5000, 8)], 2000, 16), vec![(0, 5000, 8), (8, 1008, 8)]);
        // A copy over its own earlier source: read before that copy ran.
        assert_eq!(pieces(&[(2000, 1000, 16), (1000, 7000, 16)], 2000, 4), vec![(0, 1000, 4)]);
    }

    /// The list's declared writes stay in order and joined.
    #[test]
    fn joined_ranges_join_and_answer_overlaps() {
        let mut j = Joined::default();
        j.add(10, 20);
        j.add(40, 50);
        j.add(30, 35);
        assert_eq!(j.0, vec![(10, 20), (30, 35), (40, 50)]);
        assert!(j.touches(19, 21) && !j.touches(20, 30) && !j.touches(35, 40));
        j.add(20, 30);
        assert_eq!(j.0, vec![(10, 35), (40, 50)]);
        j.add(5, 60);
        assert_eq!(j.0, vec![(5, 60)]);
        j.add(70, 70);
        assert_eq!(j.0, vec![(5, 60)]);
    }

    use super::{Region, Step, WHY_READ, WHY_RUN, covered, part_region, parts_in, push_region, step_blocks, under_writes};

    /// The parts of a range inside ranges in order and joined.
    #[test]
    fn parts_in_cuts_to_the_range() {
        let ranges = [(10, 20), (30, 40), (50, 60)];
        assert_eq!(parts_in(&ranges, 0, 10), vec![]);
        assert_eq!(parts_in(&ranges, 15, 35), vec![(15, 20), (30, 35)]);
        assert_eq!(parts_in(&ranges, 20, 30), vec![]);
        assert_eq!(parts_in(&ranges, 0, 100), vec![(10, 20), (30, 40), (50, 60)]);
        assert_eq!(parts_in(&ranges, 55, 55), vec![]);
        assert_eq!(parts_in(&[], 0, 100), vec![]);
    }

    /// Ranges in any order, overlapping, cover a range whole or not.
    #[test]
    fn covered_needs_every_byte() {
        let pending = [(30, 50), (0, 20), (15, 31)];
        assert!(covered(&pending, 0, 50));
        assert!(covered(&pending, 10, 40));
        assert!(!covered(&pending, 0, 51));
        assert!(!covered(&[(0, 10), (11, 20)], 0, 20));
        assert!(covered(&[(0, 10), (10, 20)], 5, 15));
        assert!(!covered(&[], 0, 1));
    }

    /// A slot's fill is not a read that waits for flights: what they write
    /// there comes from the card. Every other read under their writes is.
    #[test]
    fn card_fills_do_not_wait_for_flights() {
        let written = [(100, 200)];
        assert_eq!(step_blocks(&Step::CardFill(0x9000, 150, 16), &written), None);
        assert_eq!(step_blocks(&Step::Fill(0x9000, 150, 16, super::FILL_TABLE), &written).map(|b| b.0), Some(WHY_READ));
        assert_eq!(step_blocks(&Step::Fill(0x9000, 200, 16, super::FILL_TABLE), &written), None);
        assert_eq!(step_blocks(&Step::Copy(300, 190, 16), &written).map(|b| b.0), Some(WHY_READ));
        assert_eq!(step_blocks(&Step::Copy(190, 300, 16), &written).map(|b| b.0), Some(WHY_READ));
        assert_eq!(step_blocks(&Step::Run(Box::new(|| {})), &[]).map(|b| b.0), Some(WHY_RUN));
        assert_eq!(step_blocks(&Step::Host(Box::new(|| {}), true), &written), None);
    }

    /// Fills read ahead of flights' writes; a title copy out of memory they
    /// write waits for them (unless `VICTORY_COPY_AHEAD=1`): its
    /// destination keeps what it copied.
    #[test]
    fn title_copies_do_not_read_ahead_of_flights() {
        let written = [(100, 200)];
        assert_eq!(super::read_ahead(&Step::Fill(0x9000, 150, 16, super::FILL_TABLE), &written), Some((150, 16)));
        if !super::copy_ahead() {
            assert_eq!(super::read_ahead(&Step::Copy(300, 190, 16), &written), None);
        }
        assert_eq!(super::read_ahead(&Step::Copy(190, 300, 16), &written), None, "a copy into such memory never did");
    }

    /// Copies whose source and destination both go on join; any other
    /// does not.
    #[test]
    fn regions_join_only_where_both_sides_go_on() {
        let r = |src: u32, src_at: u64, dst: u32, dst_at: u64, len: u64| Region { src, src_at, dst, dst_at, len };
        let mut copies = Vec::new();
        assert!(!push_region(&mut copies, r(0, 0, 9, 512, 256)));
        assert!(push_region(&mut copies, r(0, 256, 9, 768, 256)));
        assert_eq!(copies, vec![r(0, 0, 9, 512, 512)]);
        // Source goes on, the slot does not (a reused slot between).
        assert!(!push_region(&mut copies, r(0, 512, 9, 1280, 256)));
        // Another scratch chunk.
        assert!(!push_region(&mut copies, r(1, 1536, 9, 1536, 256)));
        assert_eq!(copies.len(), 3);
    }

    /// The parts of a view under flights' writes, per piece; a part
    /// written in another copy than the piece's leaves the view to a fill
    /// that waits.
    #[test]
    fn parts_under_writes_by_piece() {
        let written = [(0x104, 0x108), (0x1f0, 0x210)];
        // One piece, its own copy pending there.
        assert_eq!(under_writes(&[(0x100, 0x110)], &written, |_, _, _| true), Ok(vec![(0, 0x104, 0x108)]));
        // Two pieces side by side (a buffer ends at 0x200).
        assert_eq!(
            under_writes(&[(0x1e0, 0x200), (0x200, 0x220)], &written, |_, _, _| true),
            Ok(vec![(0, 0x1f0, 0x200), (1, 0x200, 0x210)])
        );
        // Nothing under writes.
        assert_eq!(under_writes(&[(0x110, 0x1f0)], &written, |_, _, _| false), Ok(vec![]));
        // Written there in another copy.
        assert_eq!(under_writes(&[(0x100, 0x110)], &written, |_, _, _| false), Err(()));
        assert_eq!(under_writes(&[(0x1e0, 0x200), (0x200, 0x220)], &written, |i, _, _| i == 0), Err(()));
    }

    /// Run n19's picture, in small: a view's bytes change after its place
    /// in the submission's preparing (a later list's title copy), so the
    /// buffer's copy on the card holds the later bytes (its refresh goes
    /// up ahead of every list); a flight wrote 4 of its bytes there. The
    /// slot gets the fill's bytes, the flight's over them; copied whole
    /// from the buffer's copy (as before) it got the later bytes.
    #[test]
    fn slot_holds_the_fill_and_the_flights_bytes() {
        const SCRATCH: usize = 0;
        const COPY: usize = 1;
        const RING: usize = 2;
        let apply = |memory: &mut [Vec<u8>], copies: &[Region<usize>]| {
            for c in copies {
                let bytes = memory[c.src][c.src_at as usize..(c.src_at + c.len) as usize].to_vec();
                memory[c.dst][c.dst_at as usize..(c.dst_at + c.len) as usize].copy_from_slice(&bytes);
            }
        };
        // The view: 16 bytes of title memory at 0x1010 (unaligned), in a
        // buffer copy starting at title 0x1000; the flight wrote
        // 0x1014..0x1018.
        let (from, to, copy_start) = (0x1010usize, 0x1020usize, 0x1000usize);
        let written = [(0x1014, 0x1018)];
        let mut memory = vec![vec![0u8; 512], vec![0u8; 256], vec![0u8; 1024]];
        // The fill, at the view's place: title bytes then (the flight's
        // bytes are not in title memory yet).
        memory[SCRATCH][256..272].copy_from_slice(&[0xa0; 16]);
        // The buffer's copy as the card sees it: the later title bytes
        // (sent ahead of the lists), the flight's where it wrote.
        memory[COPY][0x10..0x20].copy_from_slice(&[0xb0; 16]);
        memory[COPY][0x14..0x18].copy_from_slice(&[0xc0; 4]);
        let ring_at = 768u64;
        let mut copies = vec![Region { src: SCRATCH, src_at: 256, dst: RING, dst_at: ring_at, len: 256 }];
        let parts = under_writes(&[(from, to)], &written, |_, _, _| true).unwrap();
        for (_, lo, hi) in parts {
            push_region(&mut copies, part_region(COPY, (lo - copy_start) as u64, RING, ring_at, from, lo, hi));
        }
        let mut new = memory.clone();
        apply(&mut new, &copies);
        let slot = &new[RING][ring_at as usize..ring_at as usize + 16];
        assert_eq!(&slot[..4], &[0xa0; 4]);
        assert_eq!(&slot[4..8], &[0xc0; 4]);
        assert_eq!(&slot[8..], &[0xa0; 8]);
        // Before: the whole view from the buffer's copy.
        let mut old = memory.clone();
        apply(&mut old, &[Region { src: COPY, src_at: 0x10, dst: RING, dst_at: ring_at, len: 16 }]);
        assert_eq!(&old[RING][ring_at as usize..ring_at as usize + 4], &[0xb0; 4]);
    }
}

#[cfg(test)]
mod spare {
    use super::Spare;

    /// Kept things come back by size, the last given first, and nothing is
    /// kept past the limit.
    #[test]
    fn kept_things_come_back_by_size_within_the_limit() {
        let mut spare: Spare<u32> = Spare::new();
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.give(1, 4, 10), None);
        assert_eq!(spare.give(2, 1, 10), None);
        assert_eq!(spare.give(3, 4, 10), None);
        assert_eq!(spare.bytes, 9);
        // Past the limit: handed back, nothing changes.
        assert_eq!(spare.give(4, 4, 10), Some(4));
        assert_eq!(spare.bytes, 9);
        assert_eq!(spare.take(2), None, "no thing of that size");
        assert_eq!(spare.take(4), Some(3));
        assert_eq!(spare.take(4), Some(1));
        assert_eq!(spare.take(4), None);
        assert_eq!(spare.take(1), Some(2));
        assert_eq!(spare.bytes, 0);
        assert!(spare.kept.is_empty());
    }
}

#[cfg(test)]
mod batch {   // Batch::texture_due lives here; `mod tests` is already taken at the top of this file
    use super::Batch;
    use std::sync::Arc;

    #[test]
    fn a_texture_is_checked_once_per_submission() {
        let mut batch = Batch::default();
        let (a, b) = (Arc::new([0u8; 16]), Arc::new([0u8; 16]));
        // The first list sampling each texture checks it ...
        assert!(batch.texture_due(&a));
        assert!(batch.texture_due(&b));
        // ... later lists of the same submission do not.
        assert!(!batch.texture_due(&a));
        assert!(!batch.texture_due(&a.clone()));
        assert!(!batch.texture_due(&b));
    }

    #[test]
    fn title_writes_make_textures_due_again() {
        let mut batch = Batch::default();
        let a = Arc::new(1u32);
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A copy into title memory or work that may write it may have
        // changed the texture's memory.
        batch.title_written();
        assert!(batch.texture_due(&a));
        assert!(!batch.texture_due(&a));
        // A fresh submission starts over.
        let mut next = Batch::default();
        assert!(next.texture_due(&a));
    }

    /// The hashing a submission does for textures in physical-page memory,
    /// with every list checking the textures it samples (before) and with
    /// each checked once (after). Run with
    /// `cargo test --release -p victory-gpu -- --ignored --nocapture texture_checks`.
    #[test]
    #[ignore = "benchmark"]
    fn texture_checks_per_submission() {
        const TEXTURES: usize = 48;
        const SIZE: usize = 4 << 20;
        const LISTS: usize = 12;
        let textures: Vec<Arc<Vec<u8>>> = (0..TEXTURES)
            .map(|t| Arc::new((0..SIZE).map(|i| (i * 31 + t) as u8).collect()))
            .collect();
        let submit = |dedupe: bool| {
            let mut batch = Batch::default();
            let mut hashed = 0usize;
            let started = std::time::Instant::now();
            for _ in 0..LISTS {
                for t in &textures {
                    if !dedupe || batch.texture_due(t) {
                        std::hint::black_box(blake3::hash(t));
                        hashed += 1;
                    }
                }
            }
            (started.elapsed(), hashed)
        };
        submit(false); // warm up
        let (before, n_before) = submit(false);
        let (after, n_after) = submit(true);
        assert_eq!((n_before, n_after), (TEXTURES * LISTS, TEXTURES));
        println!(
            "{LISTS} lists sampling {TEXTURES} textures of {} MiB: \
             before {n_before} hashes in {before:?}, after {n_after} in {after:?} ({:.1}x)",
            SIZE >> 20,
            before.as_secs_f64() / after.as_secs_f64()
        );
    }

    #[test]
    fn vram_idle_ms_scales_with_pressure() {
        use super::vram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (< 75%)
        assert_eq!(vram_idle_ms(Some([10 * gib, 0, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 7 * gib, 0, 0])), 5_000);
        assert_eq!(vram_idle_ms(Some([100, 74, 0, 0])), 5_000);
        // Moderate (75%..85%)
        assert_eq!(vram_idle_ms(Some([100, 75, 0, 0])), 2_500);
        assert_eq!(vram_idle_ms(Some([100, 84, 0, 0])), 2_500);
        // High (>= 85%, including over-budget)
        assert_eq!(vram_idle_ms(Some([100, 85, 0, 0])), 1_000);
        assert_eq!(vram_idle_ms(Some([10 * gib, 12 * gib, 0, 0])), 1_000);
        // Missing or zero budget falls back to normal
        assert_eq!(vram_idle_ms(None), 5_000);
        assert_eq!(vram_idle_ms(Some([0, 100, 0, 0])), 5_000);
    }

    #[test]
    fn ram_idle_ms_scales_with_pressure() {
        use super::ram_idle_ms;
        let gib = 1024 * 1024 * 1024;
        // Normal (>= 4 GiB)
        assert_eq!(ram_idle_ms(Some(8 * gib)), 10_000);
        assert_eq!(ram_idle_ms(Some(4 * gib)), 10_000);
        // Moderate (2 GiB..4 GiB)
        assert_eq!(ram_idle_ms(Some(4 * gib - 1)), 5_000);
        assert_eq!(ram_idle_ms(Some(2 * gib)), 5_000);
        // High (< 2 GiB)
        assert_eq!(ram_idle_ms(Some(2 * gib - 1)), 2_000);
        assert_eq!(ram_idle_ms(Some(0)), 2_000);
        // Missing falls back to normal
        assert_eq!(ram_idle_ms(None), 10_000);
    }
}

#[cfg(test)]
mod wide_fill_tests {
    use super::{do_fills, fills_apart, joined_runs};

    /// `VICTORY_DECLARED_WIDE`'s joined plans: runs that overlap or touch
    /// become one, the rest stay apart, in order.
    #[test]
    fn runs_planned_twice_are_joined() {
        assert_eq!(joined_runs(vec![(50, 60), (0, 10), (10, 20), (5, 8), (30, 40), (35, 45)]), vec![(0, 20), (30, 45), (50, 60)]);
        assert!(joined_runs(Vec::new()).is_empty());
    }

    /// Title-like source bytes and fills of many sizes into one upload
    /// buffer, laid out apart: the fills done on several threads leave the
    /// same bytes as the fills done one after the other.
    #[test]
    fn wide_fills_write_what_fills_in_order_write() {
        let source: Vec<u8> = (0..1 << 20).map(|i: u32| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        let mut fills_at = Vec::new();
        let (mut src, mut dst) = (0usize, 0usize);
        for i in 0..3000usize {
            let len = [4, 16, 64, 256, 300, 1024, 4096, 17][i % 8];
            fills_at.push((dst, src % (source.len() - len), len));
            src += 7919 * (i + 1);
            dst += len + (i % 3) * 16;
        }
        let size = dst;
        let run = |wide: bool| {
            let mut upload = vec![0xAAu8; size];
            let fills: Vec<(usize, usize, usize)> = fills_at
                .iter()
                .map(|&(d, s, n)| (upload.as_mut_ptr() as usize + d, source.as_ptr() as usize + s, n))
                .collect();
            if wide {
                let (bytes, went) = do_fills(&fills);
                assert!(went);
                assert_eq!(bytes, fills_at.iter().map(|f| f.2).sum::<usize>());
            } else {
                for &(d, s, n) in &fills {
                    super::copy_mapped(d as *mut u8, s as *const u8, n, false);
                }
            }
            upload
        };
        assert_eq!(run(true), run(false));
    }

    /// Fills whose destinations overlap are done in order: the later one
    /// wins, as in place.
    #[test]
    fn overlapping_fills_keep_their_order() {
        let source: Vec<u8> = (0..=255).collect();
        let mut upload = vec![0u8; 4096];
        let base = upload.as_mut_ptr() as usize;
        let mut fills: Vec<(usize, usize, usize)> =
            (0..100).map(|i| (base + i * 16, source.as_ptr() as usize + i, 16)).collect();
        // The last one writes over the first.
        fills.push((base + 8, source.as_ptr() as usize + 200, 16));
        assert!(!fills_apart(&fills));
        let (_, went) = do_fills(&fills);
        assert!(!went);
        assert_eq!(&upload[8..24], &source[200..216]);
        assert_eq!(&upload[0..8], &source[0..8]);
        assert!(fills_apart(&fills[..100]));
    }
}

/// The host's own paths for a copy in the card's memory, end to end, on
/// the software adapter only: the title's bytes go up, a shader reads them
/// and writes, and what it wrote comes back into title memory. Run it with
/// the switches under test in the environment (they are read once per
/// process); it does nothing unless the host device is the software
/// adapter (`VICTORY_D3D12_WARP=1` in a diagnostics build) and the copy is
/// in the card's memory (`VICTORY_CARD_BUFFERS_MIN_MB=1`):
///
/// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 [VICTORY_QUEUE2=n]
///  [VICTORY_RELAY=2 | VICTORY_LATE_WAIT=3] cargo test -p victory-gpu
///  --features diagnostics queue2_tests`
///
/// The same title memory must come out whatever the switches: through the
/// direct queue alone, and through the copy queues.
#[cfg(test)]
mod queue2_tests {
    use super::*;

    /// `memory[to] = memory[from] + add`, after `spin` steps.
    const ADD: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer What : register(b0) { uint to; uint from; uint add; uint spin; };
RWByteAddressBuffer memory : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint h = 0u;
    [loop]
    for (uint i = 0u; i < spin; i++) {
        h = (h ^ memory.Load(from)) * 0x01000193u;
    }
    memory.Store(to, memory.Load(from) + add + (h & 0u));
}
"#;

    #[test]
    fn warp_title_bytes_go_up_and_shader_writes_come_back_whatever_the_queues() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // Never on a real card.
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("queue2_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        const ROUNDS: usize = 12;
        // "Title memory": write-watched, as the title's is.
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x51_5151, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("queue2_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(ADD, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        // Round k: the title writes a dword in piece 1+k; a submission's
        // shader adds k to it and stores the sum in piece 30+k, a range
        // declared as shader-written. Nothing is waited for between the
        // rounds where submissions are sent ahead.
        let from = |k: usize| (1 + k) * PIECE + 4 * k + 64;
        let to = |k: usize| (30 + k) * PIECE + 8 * k + 128;
        let titles = |k: usize| 0x0100_0000 + 0x1000 * k as u32 + 7;
        let mut lists = Vec::new();
        for k in 0..ROUNDS {
            // SAFETY: inside the allocation.
            unsafe { dword(from(k)).write_volatile(titles(k)) };
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + (30 + k) * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [to(k) as u32, from(k) as u32, k as u32, 30_000];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            if late() {
                host.execute_own(&list);
            } else {
                host.execute(&[&list]);
            }
            lists.push(list);
        }
        settle_now(WHY_SYNC);
        for k in 0..ROUNDS {
            // SAFETY: inside the allocation.
            let (sum, kept) = unsafe { (dword(to(k)).read_volatile(), dword(from(k)).read_volatile()) };
            assert_eq!(kept, titles(k), "round {k}: the title's own dword changed");
            assert_eq!(
                sum,
                titles(k) + k as u32,
                "round {k}: the shader read {:#x} where the title had written {:#x}, or its write did not come back",
                sum.wrapping_sub(k as u32),
                titles(k)
            );
        }
        // Nothing else moved.
        let touched: std::collections::HashSet<usize> = (0..ROUNDS).flat_map(|k| [from(k), to(k)]).collect();
        for i in (0..LEN / 4).filter(|i| !touched.contains(&(i * 4))) {
            // SAFETY: inside the allocation.
            let got = unsafe { dword(i * 4).read_volatile() };
            assert_eq!(got, 0x5A00_0000 | i as u32, "dword {i} of title memory changed");
        }
        // The queues the switches name did the work.
        if let Some(second) = crate::queue2::second() {
            let (down, up) = (second.down.stats.jobs.load(Ordering::Relaxed), second.up.stats.jobs.load(Ordering::Relaxed));
            // (Sends that found no landing buffer went the old way.)
            let no_room = second.landing[1].load(Ordering::Relaxed);
            eprintln!(
                "queue2_tests: VICTORY_QUEUE2={} late {} relay {}: {down} fetches and {up} sends through the copy queues ({no_room} more found no landing buffer), the direct queue told to wait {} times",
                crate::queue2::bits(),
                late(),
                relay(),
                second.held[0].load(Ordering::Relaxed)
            );
            assert_eq!(down > 0, second.fetches(), "fetches through the copy queue");
            assert_eq!(up + no_room > 0, second.uploads(), "sends through the copy queue");
            assert_eq!(up, second.held[0].load(Ordering::Relaxed), "the direct queue waits for every send");
        } else {
            eprintln!("queue2_tests: the direct queue alone (late {}, relay {})", late(), relay());
        }
        drop(lists);
    }

    /// `memory[to] = first` at once; then `spin` steps, each a store to
    /// `memory[to + 4]`; then `memory[to + 4] = 0` and `memory[to] = last`.
    /// Whatever `spin` is, the bytes it leaves are the same.
    const TWICE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), UAV(u0)"
cbuffer What : register(b0) { uint to; uint first; uint last; uint spin; };
RWByteAddressBuffer memory : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    memory.Store(to, first);
    [loop]
    for (uint i = 0u; i < spin; i++) {
        memory.Store(to + 4u, i);
    }
    memory.Store(to + 4u, 0u);
    memory.Store(to, last);
}
"#;

    /// The hole a fetch on the copy queue had (`VICTORY_QUEUE2` bit 1 with
    /// the relay): flight A writes a piece; flight B, sent behind it,
    /// writes the same piece through a half-way value and leaves it as A
    /// left it. A is landed while B is half way: the fetch, beside B on
    /// the copy queue, brings B's half-way bytes into title memory. B's
    /// compare finds the piece as A's compare left it and flags nothing,
    /// so B's landing fetched nothing and title memory kept the half-way
    /// bytes. With the fix B's landing fetches the piece all the same
    /// (`BESIDE`).
    ///
    /// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_QUEUE2=1
    ///  VICTORY_RELAY=1 cargo test -p victory-gpu --features diagnostics
    ///  fetched_again -- --nocapture` (alone: it lands its two flights
    /// itself; `VICTORY_QUEUE2_REFETCH=0` shows the fault).
    #[test]
    fn warp_a_piece_fetched_while_a_later_flight_writes_it_is_fetched_again() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // Never on a real card.
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("queue2_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        if !relay() || !crate::queue2::second().is_some_and(|s| s.fetches()) {
            eprintln!("queue2_tests: skipped, fetches do not go to a copy queue beside the relay (VICTORY_RELAY=1 VICTORY_QUEUE2=1)");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x52_5252, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("queue2_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(TWICE, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        // A list whose shader writes the dword at 256 of `piece`, declared.
        let record = |piece: usize, first: u32, last: u32, spin: u32| {
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + piece * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [(piece * PIECE + 256) as u32, first, last, spin];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            list
        };
        // SAFETY: a plain query.
        let completed = || unsafe { host.fence.GetCompletedValue() };
        let wait_for = |value: u64| {
            while completed() < value {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        // How many steps keep the card busy for some seconds: timed on
        // another piece, waited for.
        const TRIAL: u32 = 1 << 20;
        let trial = record(40, 1, 2, TRIAL);
        let timing = std::time::Instant::now();
        host.execute(&[&trial]);
        let took = timing.elapsed().as_secs_f64().max(1e-4);
        let spin = (f64::from(TRIAL) * 3.0 / took).clamp(f64::from(TRIAL), 2.0e9) as u32;
        // SAFETY: inside the allocation.
        assert_eq!(unsafe { dword(40 * PIECE + 256).read_volatile() }, 2, "the trial's write did not come back");
        const AT: usize = 20 * PIECE + 256;
        const HALF_WAY: u32 = 0xDEAD_0001;
        const LEFT: u32 = 0x600D_0002;
        // Flight A, and the card has run it.
        let a = record(20, HALF_WAY, LEFT, 16);
        let (submitted, mut sent_a) = host.submit(&[&a], true).expect("flight A sent ahead");
        let fence_a = *submitted;
        drop(submitted);
        wait_for(fence_a);
        // Flight B: the same piece, through the half-way value, for seconds.
        let b = record(20, HALF_WAY, LEFT, spin);
        let (submitted, mut sent_b) = host.submit(&[&b], true).expect("flight B sent ahead");
        let fence_b = *submitted;
        drop(submitted);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(
            completed() < fence_b,
            "flight B had run after 0.3 s ({spin} steps; the trial's {TRIAL} took {took:.3} s): nothing to show"
        );
        // A lands, as `Host::relay_land` lands it, while B is half way.
        let mut fetch = None;
        {
            let plan = sent_a.card.as_ref().expect("flight A has a plan for the card's copies");
            host.relay_card_back(&[(sent_a.number, plan)], &mut fetch, fence_a, sent_a.number);
        }
        for (copy, ranges) in &sent_a.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let seen = unsafe { dword(AT).read_volatile() };
        let b_ran = completed() >= fence_b;
        eprintln!(
            "queue2_tests: after A's landing title memory holds {seen:#x} (half way {HALF_WAY:#x}, left {LEFT:#x}); B had run: {b_ran}; {spin} steps"
        );
        assert_eq!(
            seen, HALF_WAY,
            "the fetch for A did not come down beside B (it waited for B, or B had not begun): this machine cannot show the fault"
        );
        // B has run; it left the piece as A left it, so its compare flags
        // nothing. Its landing must bring the piece down all the same.
        wait_for(fence_b);
        {
            let plan = sent_b.card.as_ref().expect("flight B has a plan for the card's copies");
            host.relay_card_back(&[(sent_b.number, plan)], &mut fetch, fence_b, sent_b.number);
        }
        for (copy, ranges) in &sent_b.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let (left, beside) = unsafe { (dword(AT).read_volatile(), dword(AT + 4).read_volatile()) };
        assert_eq!(
            left, LEFT,
            "title memory kept flight B's half-way bytes: B's compare found the piece as A left it and B's landing fetched nothing"
        );
        assert_eq!(beside, 0, "the dword beside it holds a half-way count");
        for sent in [&mut sent_a, &mut sent_b] {
            if let Some(plan) = sent.card.take() {
                host.card_done(plan);
            }
        }
        drop((a, b, trial));
    }

    /// The fetch inside flights with its slots in the card's memory
    /// (`VICTORY_CARD_FETCH_INFLIGHT=2`): flight A leaves a value in a
    /// piece; flight B, sent behind it, writes the same piece through a
    /// half-way value for seconds. A lands while B is half way: its pieces
    /// come down from A's own slot on the landing thread's copy queue,
    /// without waiting for B, and hold what A left, neither B's half-way
    /// bytes (the tear of a copy-queue fetch of the copy itself) nor B's
    /// last ones. Then B lands with its own.
    ///
    /// `VICTORY_D3D12_WARP=1 VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_RELAY=1
    ///  VICTORY_CARD_FETCH_INFLIGHT=2 cargo test -p victory-gpu --features
    ///  diagnostics --lib slots_on_the_card -- --nocapture` (alone: it lands
    /// its two flights itself).
    #[test]
    fn warp_slots_on_the_card_hold_their_flights_bytes_while_a_later_flight_writes() {
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc};
        const MEM_WRITE_WATCH: u32 = 0x0020_0000;
        let Some(host) = host() else {
            return;
        };
        // SAFETY: a plain query.
        let software = host
            .adapter
            .as_ref()
            .and_then(|a| unsafe { a.GetDesc1() }.ok())
            .is_some_and(|d| d.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0);
        if !software {
            eprintln!("inflight_tests: skipped, the host device is not the software adapter (VICTORY_D3D12_WARP=1, diagnostics)");
            return;
        }
        if !relay() || !inflight_on_card() {
            eprintln!("inflight_tests: skipped, needs VICTORY_RELAY=1 VICTORY_CARD_FETCH_INFLIGHT=2");
            return;
        }
        const PIECE: usize = 64 << 10;
        const LEN: usize = 4 << 20;
        // SAFETY: a plain allocation of our own.
        let title = unsafe { VirtualAlloc(std::ptr::null(), LEN, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) } as *mut u8;
        assert!(!title.is_null());
        let dword = |at: usize| unsafe { title.add(at).cast::<u32>() };
        for i in 0..LEN / 4 {
            // SAFETY: inside the allocation.
            unsafe { dword(i * 4).write_volatile(0x5A00_0000 | i as u32) };
        }
        crate::shadow::register(0x52_5253, title as u64, LEN as u64);
        let shadow = crate::shadow::find(title as u64, LEN as u64).expect("a host copy");
        if !shadow.carded() {
            eprintln!("inflight_tests: skipped, the copy is not in the card's memory (VICTORY_CARD_BUFFERS_MIN_MB=1)");
            return;
        }
        let code = compile(TWICE, windows::core::s!("main"), windows::core::s!("cs_5_1")).expect("the test's shader");
        // SAFETY: the shader's own root signature; plain creation.
        let (root, pso): (ID3D12RootSignature, ID3D12PipelineState) = unsafe {
            let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).expect("a root signature");
            let pso = host
                .device
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: code.as_ptr().cast(),
                        BytecodeLength: code.len(),
                    },
                    ..Default::default()
                })
                .expect("a pipeline");
            (root, pso)
        };
        let record = |piece: usize, first: u32, last: u32, spin: u32| {
            let list = host.new_list().expect("a host list");
            shadow.refresh(&list, title as u64, LEN as u64);
            shadow.gpu_writes(&list, (title as usize + piece * PIECE) as u64, PIECE as u64);
            // SAFETY: an open list; the copy's buffer, with unordered
            // access; the shader's own root signature.
            unsafe {
                list.list.SetComputeRootSignature(&root);
                list.list.SetPipelineState(&pso);
                let constants = [(piece * PIECE + 256) as u32, first, last, spin];
                list.list.SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
                list.list.SetComputeRootUnorderedAccessView(1, shadow.buffer.GetGPUVirtualAddress());
                list.list.Dispatch(1, 1, 1);
            }
            list.close();
            list
        };
        // SAFETY: a plain query.
        let completed = || unsafe { host.fence.GetCompletedValue() };
        let wait_for = |value: u64| {
            while completed() < value {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
        };
        const TRIAL: u32 = 1 << 20;
        let trial = record(40, 1, 2, TRIAL);
        let timing = std::time::Instant::now();
        host.execute(&[&trial]);
        let took = timing.elapsed().as_secs_f64().max(1e-4);
        let spin = (f64::from(TRIAL) * 3.0 / took).clamp(f64::from(TRIAL), 2.0e9) as u32;
        const AT: usize = 20 * PIECE + 256;
        const A_LEFT: u32 = 0xA000_0001;
        const B_HALF_WAY: u32 = 0xDEAD_0002;
        const B_LEFT: u32 = 0xB000_0003;
        let stats = || [3, 4].map(|i| INFLIGHT_STATS[i].load(Ordering::Relaxed));
        let before = stats();
        let a = record(20, A_LEFT, A_LEFT, 16);
        let (submitted, mut sent_a) = host.submit(&[&a], true).expect("flight A sent ahead");
        let fence_a = *submitted;
        drop(submitted);
        let b = record(20, B_HALF_WAY, B_LEFT, spin);
        let (submitted, mut sent_b) = host.submit(&[&b], true).expect("flight B sent ahead");
        let fence_b = *submitted;
        drop(submitted);
        wait_for(fence_a);
        std::thread::sleep(std::time::Duration::from_millis(300));
        assert!(completed() < fence_b, "flight B had run after 0.3 s ({spin} steps): nothing to show");
        let mut fetch = None;
        {
            let plan = sent_a.card.as_ref().expect("flight A has a plan for the card's copies");
            assert!(plan.inflight, "flight A's compare list has no fetch inside it");
            host.relay_card_back(&[(sent_a.number, plan)], &mut fetch, fence_a, sent_a.number);
        }
        for (copy, ranges) in &sent_a.back {
            copy.copy_back_done(ranges);
        }
        let b_ran = completed() >= fence_b;
        // SAFETY: inside the allocation.
        let seen = unsafe { dword(AT).read_volatile() };
        eprintln!("inflight_tests: after A's landing title memory holds {seen:#x} (A left {A_LEFT:#x}); B had run: {b_ran}");
        assert!(!b_ran, "A's landing waited for the later flight B");
        assert_eq!(seen, A_LEFT, "A's landing did not bring A's own bytes (B's half way {B_HALF_WAY:#x}, B's last {B_LEFT:#x})");
        let after_a = stats();
        assert_eq!(after_a[0], before[0], "A landed the old way");
        assert_eq!(after_a[1], before[1] + 1, "A's slot did not come down on the copy queue");
        wait_for(fence_b);
        {
            let plan = sent_b.card.as_ref().expect("flight B has a plan for the card's copies");
            host.relay_card_back(&[(sent_b.number, plan)], &mut fetch, fence_b, sent_b.number);
        }
        for (copy, ranges) in &sent_b.back {
            copy.copy_back_done(ranges);
        }
        // SAFETY: inside the allocation.
        let (left, beside) = unsafe { (dword(AT).read_volatile(), dword(AT + 4).read_volatile()) };
        assert_eq!(left, B_LEFT, "B's landing did not bring B's bytes");
        assert_eq!(beside, 0, "the dword beside it holds a half-way count");
        assert_eq!(stats()[0], before[0], "B landed the old way");
        for sent in [&mut sent_a, &mut sent_b] {
            if let Some(plan) = sent.card.take() {
                host.card_done(plan);
            }
        }
        drop((a, b, trial));
    }

    /// The copies down of slots: pieces that follow each other in the slots
    /// join one copy, the others start one of their own; empty ones none.
    #[test]
    fn slots_down_join_pieces_that_follow_each_other() {
        use super::slots_down;
        assert!(slots_down([]).is_empty());
        assert_eq!(slots_down([(100, 10), (110, 5), (0, 0), (500, 7), (507, 1), (0, 3)]), vec![
            (100, 0, 15),
            (500, 15, 8),
            (0, 23, 3)
        ]);
    }
}
