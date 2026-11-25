//! Textures picked by index made ready on worker threads
//! (`VICTORY_ASYNC_TEXTURES=<workers>`; off by default).
//!
//! The pass over the view heap's textures (`draw.rs` `heap_textures`) runs
//! on the thread that records the frame. For each texture it uploads it
//! made the host texture (a committed resource), made a staging buffer (an
//! upload buffer, another committed resource) and recorded the copy; the
//! submission then read and detiled the title's memory into the staging
//! buffer. At a budget of 30 ms a pass that was some 55 textures a frame
//! (making 10-13 ms, staging 16-18 ms) with 20,700 waiting: the world
//! streamed in over minutes, blurred.
//!
//! With the switch, the pass hands a placed texture it would upload to a
//! worker instead and leaves its slot due. The worker makes the host
//! texture and the staging buffer and reads and detiles the title's memory
//! into it, as the submission's check would. A later pass finds it ready:
//! the host texture is there (making the slot's view is cheap), the staging
//! buffer goes in the table (`crate::texture::adopt`), and the upload is
//! recorded as before, the copy and its barriers.
//!
//! What the texture shows is still as of the submission that uploads it:
//! that submission checks the staging buffer as it checks every one
//! (`HostList::defer_texture`), and the worker's read noted the write-watch
//! asking (or hash, or guard) it was taken at. What the title wrote since
//! is read again there; what it did not is not. Streamed textures are
//! written once, by the loader, before their views are written, so that
//! check almost never finds anything; reading them a frame or two earlier
//! than the submission reads the same bytes. Pages mapped after the worker
//! asked what was mapped make the slot due again, as they do now (the pass
//! is told the bytes mapped when the worker asked, not when it took the
//! result).
//!
//! Texture objects (not placed) are left to the pass as before: the title
//! may release one while a worker holds it, and they are few.
//!
//! What is staged and not yet recorded is held to `VICTORY_ASYNC_TEXTURES_MB`
//! (512 by default; queued jobs count by their title memory): no more jobs
//! are queued past it. A job that cannot be done (no host texture, no
//! staging buffer) is done by the pass itself, as before.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, Condvar, Mutex};

use crate::object::This;

/// Worker threads: `VICTORY_ASYNC_TEXTURES` (read once), 0 (off) by default.
pub(crate) fn workers() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("VICTORY_ASYNC_TEXTURES")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok())
            .unwrap_or(0)
            .min(16)
    })
}

/// Bytes staged and not yet recorded, at most: `VICTORY_ASYNC_TEXTURES_MB`
/// (read once), 512 by default. The machine has little memory to spare
/// (the game commits some 22 GB), and upload memory is the processor's.
fn cap() -> u64 {
    static MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MB.get_or_init(|| {
        std::env::var("VICTORY_ASYNC_TEXTURES_MB")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(512)
    }) << 20
}

/// A finished item no pass took in this long is let go: its slot went
/// away, was written with another view, or was found tabled. Under the five
/// seconds after which an unused host copy may be let go
/// (`crate::resource::let_go_idle`), which an item holds off anyway.
const EXPIRE: std::time::Duration = std::time::Duration::from_secs(4);

/// Bytes held: jobs queued or under way (by their title memory) and items
/// finished and not yet taken (by their staging buffers).
static HELD: AtomicU64 = AtomicU64::new(0);

/// Since the frame line last took them: jobs queued, items made ready,
/// items the pass recorded, jobs the pass did itself (a worker could not),
/// items let go untaken, and slots left for a later pass at the cap.
static COUNTS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// Bytes counted in [`HELD`] while it lives.
struct Held(u64);

impl Held {
    fn new(bytes: u64) -> Held {
        HELD.fetch_add(bytes, Relaxed);
        Held(bytes)
    }
}

impl Drop for Held {
    fn drop(&mut self) {
        HELD.fetch_sub(self.0, Relaxed);
    }
}

/// A texture a worker made ready.
pub(crate) struct Prepared {
    /// The title texture.
    pub(crate) resource: usize,
    /// Its staging buffer, detiled, not in the table yet.
    pub(crate) staging: Arc<crate::texture::Staging>,
    /// Bytes of its title memory mapped when the worker asked, before it
    /// read: what more is mapped by the time of the upload is uploaded at
    /// the next look.
    pub(crate) mapped: u64,
    /// Held so that the host copy is not let go as idle while it waits
    /// (`crate::resource::let_go_idle` leaves a held one alone).
    _host: Arc<crate::resource::HostTexture>,
    _held: Held,
}

/// What became of a job.
enum Outcome {
    Ready(Prepared),
    /// The worker could not make it: the pass does it itself.
    Failed,
    /// Nothing to do now (no room on the card, nothing mapped): the pass
    /// looks at the slot as it would have.
    Later,
}

struct Job {
    index: u32,
    resource: usize,
    address: u64,
    len: u64,
    _held: Held,
}

struct State {
    queue: VecDeque<Job>,
    /// Slots with a job queued or under way.
    busy: std::collections::HashSet<u32>,
    /// Finished jobs by slot: the title texture, the outcome, and when.
    done: HashMap<u32, (usize, Outcome, std::time::Instant)>,
    /// Slots whose job is under way while the title's memory under them
    /// was written ([`forget`]): what the worker read is from before the
    /// write, and its item is let go when it finishes.
    spoiled: std::collections::HashSet<u32>,
}

struct Pool {
    state: Mutex<State>,
    more: Condvar,
}

fn pool() -> &'static Pool {
    static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
    static STARTED: std::sync::Once = std::sync::Once::new();
    let pool = POOL.get_or_init(|| Pool {
        state: Mutex::new(State {
            queue: VecDeque::new(),
            busy: Default::default(),
            done: HashMap::new(),
            spoiled: Default::default(),
        }),
        more: Condvar::new(),
    });
    STARTED.call_once(|| {
        let n = workers();
        tracing::info!(workers = n, cap_mb = cap() >> 20, "textures picked by index are made ready on worker threads");
        for i in 0..n {
            let spawned = std::thread::Builder::new()
                .name(format!("victory-texture-{i}"))
                .spawn(work);
            if let Err(e) = spawned {
                tracing::warn!(error = %e, "no texture worker thread");
            }
        }
    });
    pool
}

fn lock(pool: &Pool) -> std::sync::MutexGuard<'_, State> {
    pool.state.lock().unwrap_or_else(|e| e.into_inner())
}

/// A worker thread: takes jobs for good.
fn work() {
    // Below the title's threads and the one that records and submits the
    // frame: the work here is what they need not wait for.
    // SAFETY: plain calls on this thread's own handle.
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
    }
    let pool = pool();
    loop {
        let job = {
            let mut state = lock(pool);
            loop {
                if let Some(job) = state.queue.pop_front() {
                    break job;
                }
                state = pool.more.wait(state).unwrap_or_else(|e| e.into_inner());
            }
        };
        // A panic in the detiling must not leave the slot busy for good.
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepare(&job)))
            .unwrap_or(Outcome::Failed);
        if matches!(outcome, Outcome::Ready(_)) {
            COUNTS[1].fetch_add(1, Relaxed);
        }
        let (replaced, spoiled) = {
            let mut state = lock(pool);
            state.busy.remove(&job.index);
            // Read before a write the title made under it: the slot is
            // looked at again (asked again, as after `Later`).
            let (outcome, spoiled) = if state.spoiled.remove(&job.index) {
                (Outcome::Later, Some(outcome))
            } else {
                (outcome, None)
            };
            (state.done.insert(job.index, (job.resource, outcome, std::time::Instant::now())), spoiled)
        };
        // Let go outside the lock (a staging buffer goes back to the spares).
        drop(replaced);
        drop(spoiled);
        drop(job);
    }
}

/// Makes `job`'s host texture and staging buffer, and reads and detiles
/// its title memory into it.
fn prepare(job: &Job) -> Outcome {
    // The pass asked before it queued the job; it may have waited since.
    if !crate::resource::room() {
        return Outcome::Later;
    }
    // Before the read: what is mapped after this is uploaded at a later look.
    let mapped = crate::texture::mapped_bytes(job.address as *const u8, job.len as usize);
    if mapped == 0 {
        return Outcome::Later;
    }
    let Some(host) = crate::resource::host_copy(job.resource as This) else {
        return Outcome::Failed;
    };
    if !crate::texture::title_written(&host.texture) {
        return Outcome::Failed;
    }
    let Some(staging) = crate::texture::prepare(job.resource as This, &host.texture) else {
        return Outcome::Failed;
    };
    let held = Held::new(staging.bytes());
    Outcome::Ready(Prepared {
        resource: job.resource,
        staging,
        mapped,
        _host: host,
        _held: held,
    })
}

/// What the pass is to do with a slot.
pub(crate) enum Look {
    /// Upload this: made ready by a worker.
    Ready(Prepared),
    /// Do it here, as before (a worker could not).
    Inline,
    /// Leave the slot due: a worker has it (just queued, or waiting for
    /// room under the cap).
    Wait,
}

/// What [`look`] would do with a slot, asked before the pass asks what is
/// mapped of its texture (`VICTORY_CHEAP_LOOKS`, `draw.rs`).
pub(crate) enum Ahead {
    /// A worker made a texture ready for it: [`look`] hands it over
    /// (or, made for another texture, queues a job).
    Ready,
    /// No job can be queued for it at the cap: [`look`] would leave it due.
    Capped,
    /// Anything else: as before, the look first.
    Look,
}

/// For slot `index`, whose placed texture takes `len` bytes: [`Ahead`].
/// Counts a slot left at the cap as [`look`] does.
pub(crate) fn ahead_of_look(index: u32, len: u64) -> Ahead {
    let state = lock(pool());
    match state.done.get(&index) {
        Some((_, Outcome::Ready(_), _)) => return Ahead::Ready,
        Some(_) => return Ahead::Look,
        None => {}
    }
    if state.busy.contains(&index) {
        return Ahead::Look;
    }
    let held = HELD.load(Relaxed);
    if held > 0 && held + len > cap() {
        COUNTS[5].fetch_add(1, Relaxed);
        return Ahead::Capped;
    }
    Ahead::Look
}

/// Whether a worker has a job for slot `index` queued or under way: the
/// pass need not look at it.
pub(crate) fn busy(index: u32) -> bool {
    lock(pool()).busy.contains(&index)
}

/// For slot `index`, whose view shows title texture `resource` placed at
/// `address` (`len` bytes): what a worker made of it, or a job queued for it.
pub(crate) fn look(index: u32, resource: usize, address: u64, len: u64) -> Look {
    let pool = pool();
    let mut state = lock(pool);
    if let Some((of, outcome, _)) = state.done.remove(&index) {
        if of == resource {
            match outcome {
                Outcome::Ready(prepared) => return Look::Ready(prepared),
                Outcome::Failed => {
                    COUNTS[3].fetch_add(1, Relaxed);
                    return Look::Inline;
                }
                // Asked again below.
                Outcome::Later => {}
            }
        } else {
            // Made for the texture the slot showed before.
            COUNTS[4].fetch_add(1, Relaxed);
        }
    }
    if state.busy.contains(&index) {
        return Look::Wait;
    }
    // At the cap, nothing more is queued; one texture bigger than the cap
    // is queued when nothing else is held.
    let held = HELD.load(Relaxed);
    if held > 0 && held + len > cap() {
        COUNTS[5].fetch_add(1, Relaxed);
        return Look::Wait;
    }
    state.busy.insert(index);
    state.queue.push_back(Job {
        index,
        resource,
        address,
        len,
        _held: Held::new(len),
    });
    drop(state);
    pool.more.notify_one();
    COUNTS[0].fetch_add(1, Relaxed);
    Look::Wait
}

/// Title memory under slot `index`'s texture was written after a worker
/// may have read it (a DirectStorage write into its pages, or a kernel
/// read that finished): an item made ready for it is let go, and a job
/// under way gives none (its read may be from before the write). The slot
/// is then looked at and read again as one no worker has. Without this the
/// item was uploaded at the slot's next look and the slot left the queue
/// with the bytes from before the write for the rest of the run: some
/// other texture's that the title's pages held before, a different look
/// from run to run (runs n15, n23: the character's prison top pink-red, the window
/// mesh gone). Returns whether a worker had anything of the slot.
pub(crate) fn forget(index: u32) -> bool {
    if workers() == 0 {
        return false;
    }
    let (item, busy) = {
        let mut state = lock(pool());
        let item = state.done.remove(&index);
        let busy = state.busy.contains(&index);
        if busy {
            state.spoiled.insert(index);
        }
        (item, busy)
    };
    let ready = matches!(item, Some((_, Outcome::Ready(_), _)));
    if ready || busy {
        FORGOTTEN.fetch_add(1, Relaxed);
    }
    // Let go outside the lock.
    drop(item);
    ready || busy
}

/// Since the start: worker items let go, or jobs under way spoiled, by
/// [`forget`].
pub(crate) static FORGOTTEN: AtomicU64 = AtomicU64::new(0);

/// The pass recorded the upload of an item a worker made ready.
pub(crate) fn recorded() {
    COUNTS[2].fetch_add(1, Relaxed);
}

/// Lets go of finished items no pass took within [`EXPIRE`]. Once a pass.
pub(crate) fn sweep() {
    let now = std::time::Instant::now();
    let mut gone = Vec::new();
    {
        let mut state = lock(pool());
        state.done.retain(|_, (_, outcome, at)| {
            let keep = now.duration_since(*at) < EXPIRE;
            if !keep && matches!(outcome, Outcome::Ready(_)) {
                gone.push(std::mem::replace(outcome, Outcome::Later));
            }
            keep
        });
    }
    COUNTS[4].fetch_add(gone.len() as u64, Relaxed);
    // Let go outside the lock.
    drop(gone);
}

/// For the memory line: the bytes of staging buffers made ready and not
/// yet taken.
pub(crate) fn held() -> u64 {
    HELD.load(Relaxed)
}

/// For the frame line, per frame over `frames`: what the workers did.
pub(crate) fn line(frames: f64) -> String {
    let n = workers();
    if n == 0 {
        return "async textures off".into();
    }
    let [queued, ready, recorded, inline, expired, capped] =
        [0, 1, 2, 3, 4, 5].map(|i| COUNTS[i].swap(0, Relaxed) as f64 / frames);
    let (busy, waiting) = {
        let state = lock(pool());
        (state.busy.len(), state.done.len())
    };
    format!(
        "async textures ({n} workers): {queued:.0} queued, {ready:.0} prepared, {recorded:.0} recorded, \
         {inline:.1} done inline, {expired:.1} let go untaken, {capped:.0} left at the cap per frame; \
         {busy} queued or under way, {waiting} finished, {:.0} MB staged waiting (cap {} MB)",
        HELD.load(Relaxed) as f64 / f64::from(1u32 << 20),
        cap() >> 20
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Without workers (the switch is not set in tests) jobs stay queued:
    /// a slot is queued once and then busy; past the cap nothing more is
    /// queued; a job that failed is done inline once; one made for the
    /// texture a slot showed before is let go and the slot queued anew. The
    /// bytes held go with their holders.
    #[test]
    fn slots_are_queued_once_within_the_cap() {
        const MB: u64 = 1 << 20;
        assert_eq!(workers(), 0, "run without VICTORY_ASYNC_TEXTURES");
        assert_eq!(HELD.load(Relaxed), 0);
        assert!(matches!(look(1, 0x10, 0x1000_0000, 100 * MB), Look::Wait));
        assert!(busy(1));
        assert_eq!(HELD.load(Relaxed), 100 * MB);
        assert!(matches!(look(1, 0x10, 0x1000_0000, 100 * MB), Look::Wait));
        assert_eq!(lock(pool()).queue.len(), 1, "queued once");
        // 100 + 600 MB is past the 512 MB cap: left due, not queued.
        assert!(matches!(look(2, 0x20, 0x2000_0000, 600 * MB), Look::Wait));
        assert!(!busy(2));
        // What a worker would leave: failed for this texture, made for
        // another one.
        let at = std::time::Instant::now();
        lock(pool()).done.insert(3, (0x30, Outcome::Failed, at));
        assert!(matches!(look(3, 0x30, 0x3000_0000, MB), Look::Inline));
        lock(pool()).done.insert(4, (0x40, Outcome::Failed, at));
        assert!(matches!(look(4, 0x41, 0x4000_0000, MB), Look::Wait));
        assert!(busy(4), "queued for the texture the slot shows now");
        // The queue let go: nothing held.
        let mut state = lock(pool());
        state.queue.clear();
        state.busy.clear();
        drop(state);
        assert_eq!(HELD.load(Relaxed), 0);
    }
}
