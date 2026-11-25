//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;

const QUEUE_IIDS: [Iid; 4] = [
    guid(
        0x0ec870a6,
        0x5d7e,
        0x4c22,
        [0x8c, 0xfc, 0x5b, 0xaa, 0xe0, 0x76, 0x16, 0xed],
    ),
    PAGEABLE_IIDS[0],
    PAGEABLE_IIDS[1],
    PAGEABLE_IIDS[2],
];

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDesc {
    kind: u32,
    priority: i32,
    flags: u32,
    nodes: u32,
}

/// The title reads the list type right after the header.
#[repr(C)]
struct Queue {
    object: Object,
    kind: u32,
    desc: QueueDesc,
    /// What the queue was given after a wait for a fence that had not been
    /// reached, in order; it runs when the fence gets there.
    #[cfg(windows)]
    held: std::sync::Mutex<Vec<Held>>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

/// One thing a waiting queue holds back.
#[cfg(windows)]
enum Held {
    /// The queue waits for the fence (its address) to reach the value.
    Wait(usize, u64),
    /// A list of the next submission waits for a memory value.
    Value(crate::list::ValueWait),
    /// A submission: the lists' host sides, taken from the title's lists
    /// (which it may reset meanwhile), the order they run in, and when it
    /// was put in the line (for [`async_line`]).
    Run(Vec<crate::list::Detached>, Vec<usize>, std::time::Instant),
    /// A signal of the queue's, due after what it holds before it.
    Signal(usize, u64),
    /// With [`behind`]: the submission at the head of the line is being
    /// prepared and sent by our thread, with the line unlocked, so the
    /// title can add to its end meanwhile; whatever is added stays behind
    /// this (the line is not empty), signals too.
    Running,
    /// With [`present_behind`]: the title's present of a frame, due after
    /// what the queue holds before it: the host texture shown, the state
    /// the title's lists leave it in, and the frame's number.
    Present(windows::Win32::Graphics::Direct3D12::ID3D12Resource, u32, u64),
}

/// Presents that are in a queue's line and have not run ([`Held::Present`]).
#[cfg(windows)]
static PRESENTS_HELD: AtomicU64 = AtomicU64::new(0);

/// Whether a present goes behind the queue's line and returns at once
/// (`VICTORY_ASYNC_PRESENT=1`; needs [`behind`] and presents that do not
/// settle the flights, `crate::host::late_free_present`).
///
/// Why: with [`behind`] our thread prepares and sends the submissions, but
/// the title's present still ran the line itself first, that is: it stood
/// until our thread had prepared every submission of the frame. Run 225's
/// frame log (the cell, 136 ms a frame): the title's render thread is 77 ms
/// of every frame inside the present waiting for that, using 3 ms of
/// processor itself, and only then begins to record the next frame (about
/// 57 ms), during which our thread has nothing to prepare. The two phases
/// ran one after the other.
///
/// With the switch the present is one more thing in the line
/// ([`Held::Present`]): our thread shows the frame when it gets there, in
/// order, and the title records the next frame meanwhile.
///
/// The title is one frame ahead at most: a present that finds the one
/// before it still in a line runs the old way (the lines first, then
/// itself), and so does a frame that is saved or listed, and a present
/// that finds its queue's line empty. A wait nothing ever meets is
/// therefore still forced, by the present after it.
///
/// What it assumes, beyond what [`behind`] does: title memory a submission
/// reads stays as it was until the submission's fence is signalled, also
/// across the title's present. On the console the card reads that memory
/// when it runs the list, after the present call has returned.
///
/// `VICTORY_ASYNC_PRESENT=alt`: twenty seconds with, twenty without, in
/// turn, each change said in the log, for measuring it in one scene.
#[cfg(windows)]
fn present_behind() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| {
        let asked = match std::env::var("VICTORY_ASYNC_PRESENT").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => 0,
        };
        let mode = if asked != 0 && behind() && crate::host::late_free_present() { asked } else { 0 };
        if mode != 0 {
            tracing::info!("presents go behind the queue's line and return at once (VICTORY_ASYNC_PRESENT=1; alt: in turn)");
        }
        mode
    });
    match mode {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "presents behind the line from now (VICTORY_ASYNC_PRESENT=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// With [`present_behind`]: puts the present of the next frame behind what
/// the queue `this` holds. False if it has to run the old way.
#[cfg(windows)]
fn send_behind(this: This, planes: u32, plane_parameters: *const Plane) -> bool {
    if this.is_null() || planes == 0 || (PRESENTS_HELD.load(Ordering::Acquire) != 0 && !present_wait()) {
        if !this.is_null() && planes != 0 {
            stat(a::PRESENT_OLD_HELD, 1);
        }
        return false;
    }
    let frame = FRAMES.load(Ordering::Relaxed) + 1;
    // A frame that is saved or listed is read back at once.
    #[cfg(feature = "diagnostics")]
    if crate::capture::wanted(frame) || crate::capture::listed(frame) {
        return false;
    }
    // SAFETY: the title passes `planes` plane descriptions.
    let Some(plane) = (unsafe { plane_parameters.as_ref() }) else {
        return false;
    };
    if plane.count == 0 || plane.resources.is_null() {
        return false;
    }
    // SAFETY: the plane lists `count` resources.
    let resource = unsafe { *plane.resources };
    let Some(texture) = crate::resource::host_resource(resource) else {
        return false;
    };
    // The present before is still in a line ([`present_wait`]): until our
    // thread has shown it.
    if PRESENTS_HELD.load(Ordering::Acquire) != 0 {
        let event = thread_event();
        lock(&PRESENT_WAITERS).push(event);
        let shown = wait_on_line(a::PRESENT_US, event, &|| PRESENTS_HELD.load(Ordering::Acquire) == 0);
        lock(&PRESENT_WAITERS).retain(|&e| e != event);
        if !shown {
            stat(a::PRESENT_OLD_HELD, 1);
            return false;
        }
        stat(a::PRESENT_WAITED, 1);
    }
    // A wait is not kept across frames: the old way forces it.
    let before = lock(&LAST_PRESENT).replace(std::time::Instant::now());
    if present_wait() && before.is_some_and(stands_across_frames) {
        stat(a::PRESENT_OLD_STANDS, 1);
        return false;
    }
    // The state the lists recorded so far leave the texture in: what our
    // copy finds it in when it runs behind them (looked at, not changed).
    const COPY_SOURCE: u32 = 0x800;
    let state = crate::resource::host_transition(resource, COPY_SOURCE).map_or(COPY_SOURCE, |(_, before)| before);
    crate::resource::host_transition(resource, state);
    let texture = texture.clone();
    let sent = hold(this, || {
        PRESENTS_HELD.fetch_add(1, Ordering::Relaxed);
        Held::Present(texture, state, frame)
    });
    if sent {
        PRESENTS_BEHIND.fetch_add(1, Ordering::Relaxed);
        wake();
    } else {
        stat(a::PRESENT_OLD_EMPTY, 1);
    }
    sent
}

/// Presents sent behind their queue's line since the start
/// ([`send_behind`]), for the line every 512 of them.
#[cfg(windows)]
static PRESENTS_BEHIND: AtomicU64 = AtomicU64::new(0);

/// The queues that hold something (their addresses).
#[cfg(windows)]
static WAITING: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

#[cfg(windows)]
fn lock<T>(mutex: &std::sync::Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether queues wait for fences. `VICTORY_QUEUE_WAIT=0`: they do not,
/// and every submission runs when it is made (as before).
#[cfg(windows)]
fn waits() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUEUE_WAIT").is_none_or(|v| v != "0"))
}

/// Whether a submission is held list by list: the lists before the first
/// one whose wait for a memory value is not met run at once, and each list
/// that waits is held with the lists after it, behind its own waits.
/// Holding a submission whole made the title's queues wait for each other:
/// a graphics submission was held for a value a compute list writes, and
/// that compute list for a value an earlier list of the same graphics
/// submission writes. Nothing moved until the present forced it, and the
/// title reached its present only after waiting about 2.5 s, every frame
/// (runs 27-30: 2.4-2.9 s from the last submission to the present call,
/// 13-32 ms before waits were honoured; the frame's main graphics
/// submissions, 88 lists, ran forced, the compute lists before them).
/// On by default; `VICTORY_SPLIT_WAITS=0` holds whole submissions, as
/// before.
#[cfg(windows)]
fn split() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPLIT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a list is cut into host lists at its waits for memory values
/// (`crate::list::cut`), and a queue runs a submission part by part: it
/// stands at each wait until the value is in memory, with everything
/// recorded before the wait run and its bottom-of-pipe writes done, as on
/// the console. Held list by list ([`split`]) a wait still held its whole
/// list: the title's lists are chains of jobs, each behind a wait for a
/// value another queue's list writes in the middle of its own chain
/// (frame 858 of run 33: a compute list of 433 commands with 13 waits;
/// graphics lists of the form "341 dispatches, wait, draws"), so two lists
/// each waited for a write behind the other's wait and only the present
/// moved them, in the wrong order.
/// On by default where waits are honoured list by list;
/// `VICTORY_CUT_WAITS=0` holds whole lists, as before.
#[cfg(windows)]
pub(crate) fn cuts() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    waits() && split() && *ON.get_or_init(|| std::env::var_os("VICTORY_CUT_WAITS").is_none_or(|v| v != "0"))
}

/// Whether a present runs what queues still hold (on by default: a wait
/// is not kept across frames). `VICTORY_WAIT_FORCE=0` keeps it held: for
/// finding out whether the title's waits reach across its presents; a
/// wait nothing ever meets then stops the title for good.
#[cfg(windows)]
fn forces() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WAIT_FORCE").is_none_or(|v| v != "0"))
}

/// A submission to a queue whose lists are cut at their waits ([`cuts`]).
/// The lists before the first one that waits run at once if the queue
/// holds nothing; the rest goes behind what the queue holds, part by part
/// (`crate::list::detach_parts`): the waits of a part, then its host
/// lists. [`release_held`] runs the queue's line as far as the values are
/// there, now and whenever a submission has run.
#[cfg(windows)]
fn submit_cut(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let first = if !behind() && lock(&queue(this).held).is_empty() {
        lists
            .iter()
            .position(|&l| crate::list::cut_list(l))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {first} run at once, the rest part by part)",
            Some(queue(this).kind),
            first < lists.len()
        )
    });
    if first > 0 && crate::host::late() {
        // Sent to the card and not waited for: their host sides are taken
        // from the title's lists (which it may reset before they have run)
        // and come back, with their tables' slots, when they are settled.
        let (taken, order) = crate::list::detach(&lists[..first]);
        host.execute_late(taken, order);
    } else if first > 0 {
        let now: Vec<_> = lists[..first]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
        // They have run: their descriptor tables' slots are free again, as
        // on the path without cuts ([`execute`]). This path replaced that
        // one without the call, so the lists that run at once kept their
        // slots until the title's reset and the ring ran out as before
        // (run 45: 3,143 of 15,942 tables refused in one frame).
        crate::list::tables_done(&lists[..first]);
    }
    if first < lists.len() {
        let parts = crate::list::detach_parts(&lists[first..]);
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = first,
                parts = parts.len(),
                waits = parts.iter().map(|p| p.0.len()).sum::<usize>(),
                "a submission is given to its queue part by part (its lists wait for memory values)"
            );
        }
        {
            let mut held = lock(&queue(this).held);
            for (waits, taken, order) in parts {
                if crate::host::async_race() {
                    crate::host::race_note(&taken);
                }
                held.extend(waits.into_iter().map(Held::Value));
                held.push(Held::Run(taken, order, std::time::Instant::now()));
                lined(1);
            }
        }
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just
    // given to the queue may be free to run already.
    release_held(false);
}

/// A submission to a queue that honours waits, list by list ([`split`]).
/// shortcut: a wait still holds its whole list, not only the commands
/// after it; and lists named twice among those held stay in one piece.
#[cfg(windows)]
fn submit_split(this: This, lists: &[This]) {
    let Some(host) = crate::host::host() else {
        return;
    };
    let by = crate::list::waits_by_list(lists);
    // Behind what the queue already holds, nothing runs now.
    let cut = if lock(&queue(this).held).is_empty() {
        (0..lists.len())
            .find(|&i| by[i].iter().any(|w| !w.met()))
            .unwrap_or(lists.len())
    } else {
        0
    };
    #[cfg(feature = "diagnostics")]
    crate::capture::op(this as usize, || {
        let named: Vec<String> = lists
            .iter()
            .map(|&l| match crate::list::host_list(l) {
                Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                None => format!("title {l:?}"),
            })
            .collect();
        format!(
            "SUBMIT queue kind {:?} held {} lists {named:?} (the first {cut} run at once)",
            Some(queue(this).kind),
            cut < lists.len()
        )
    });
    if cut > 0 {
        let now: Vec<_> = lists[..cut]
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&now);
    }
    if cut < lists.len() {
        static SEEN: AtomicU64 = AtomicU64::new(0);
        if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
            tracing::info!(
                kind = queue(this).kind,
                lists = lists.len(),
                run_at_once = cut,
                waits = ?by[cut],
                "lists of a submission are held for a memory value"
            );
        }
        let rest = &lists[cut..];
        // A list's host side can be taken once.
        let twice = rest.iter().enumerate().any(|(i, l)| rest[..i].contains(l));
        let mut pieces: Vec<Held> = Vec::new();
        let mut from = cut;
        while from < lists.len() {
            // Up to the next list that waits for anything (met or not: it
            // is looked at again when its turn comes).
            let to = if twice {
                lists.len()
            } else {
                (from + 1..lists.len())
                    .find(|&i| !by[i].is_empty())
                    .unwrap_or(lists.len())
            };
            let waits: Vec<crate::list::ValueWait> = by[from..to].concat();
            pieces.extend(waits.into_iter().map(Held::Value));
            let (taken, order) = crate::list::detach(&lists[from..to]);
            pieces.push(Held::Run(taken, order, std::time::Instant::now()));
            lined(1);
            from = to;
        }
        lock(&queue(this).held).extend(pieces);
        let mut waiting = lock(&WAITING);
        if !waiting.contains(&(this as usize)) {
            waiting.push(this as usize);
        }
    }
    // What ran may be what another queue waits for, and what was just held
    // may be free to run already.
    release_held(false);
}

/// Adds `what()` to what the queue holds, if it holds anything (it waits
/// for a fence): whether it was added.
#[cfg(windows)]
fn hold(this: This, what: impl FnOnce() -> Held) -> bool {
    if this.is_null() {
        return false;
    }
    let mut held = lock(&queue(this).held);
    if held.is_empty() {
        return false;
    }
    held.push(what());
    true
}

/// Runs what waiting queues hold as far as their fences have been reached
/// (`all`: all of it), each queue's in the order it was given. Called
/// where a fence may have moved: after a signal and after a submission (a
/// list's bottom-of-pipe write may be into a fence), never from inside
/// one.
#[cfg(windows)]
pub(crate) fn release_held(all: bool) {
    if !all && behind() {
        // A thread of ours runs what is free to run ([`behind`]).
        wake();
        return;
    }
    release_held_now(all);
}

/// Whether a submission is only put in its queue's line by the thread
/// that submits it, and prepared and sent by a thread of ours
/// (`VICTORY_ASYNC_SUBMIT=1`; needs flights and lists cut at their waits,
/// both defaults).
///
/// Why: the profile of run 213 (292 ms a frame) has the title's render
/// thread inside `queue::execute` for 202 ms of every frame: 155 ms in
/// `Host::submit` (preparing: the copies of title memory, the write
/// watch, descriptors, textures; and waiting for flights), while its five
/// recording threads stand idle 78 % of the time and the card 43 %. The
/// title's thread does nothing with the result of that work: it only has
/// to be done before the lists run, in order.
///
/// With the switch a submission's lists are taken from the title's lists
/// (as they already are when the queue holds anything) and put behind
/// the queue's line; one thread of ours ([`worker`]) runs the line in
/// order. A signal of the queue goes behind them, as it does behind
/// anything held. Everything that must see earlier submissions done runs
/// the line itself first: a present, and every submission of ours that
/// is waited for ([`before_sync`]).
///
/// What it assumes: title memory a submission reads is not rewritten by
/// the title between its submit call and the moment our thread prepares
/// it, some milliseconds later. On the console the card reads that
/// memory when it runs the list, later still, so a title that rewrote it
/// earlier would be wrong there too.
#[cfg(windows)]
pub(crate) fn behind() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_ASYNC_SUBMIT").is_some_and(|v| v == "1") && crate::host::late() && cuts();
        if on {
            tracing::info!("submissions are prepared and sent by a thread of ours, not by the thread that submits (VICTORY_ASYNC_SUBMIT=1)");
            let _ = std::thread::Builder::new().name("victory-submit".into()).spawn(worker);
        }
        on
    })
}

/// Whether there is something for [`worker`] to look at.
#[cfg(windows)]
static WORK: (std::sync::Mutex<bool>, std::sync::Condvar) = (std::sync::Mutex::new(false), std::sync::Condvar::new());

#[cfg(windows)]
fn wake() {
    *lock(&WORK.0) = true;
    WORK.1.notify_one();
}

/// The thread of [`behind`]: runs what queues hold as far as it is free
/// to run, each time something was added or a fence moved.
#[cfg(windows)]
fn worker() {
    crate::framelog::keep_on_fast_cores("victory-submit");
    WORKER_HERE.set(true);
    loop {
        {
            let idle = std::time::Instant::now();
            let mut work = lock(&WORK.0);
            while !*work {
                work = WORK.1.wait(work).unwrap_or_else(|e| e.into_inner());
            }
            *work = false;
            drop(work);
            stat(a::IDLE_US, idle.elapsed().as_micros() as u64);
        }
        let started = std::time::Instant::now();
        let (user, kernel) = crate::framelog::thread_times();
        if crate::framelog::on() {
            crate::framelog::add(crate::framelog::T::WorkerLooks, 1);
            if crate::framelog::on_slow_core() {
                crate::framelog::add(crate::framelog::T::WorkerSlow, 1);
            }
        }
        release_held_now(false);
        let us = started.elapsed().as_micros() as u64;
        stat(a::PASS_US, us);
        let (user_now, kernel_now) = crate::framelog::thread_times();
        crate::framelog::add(crate::framelog::T::WorkerUs, us);
        crate::framelog::add(crate::framelog::T::WorkerUserUs, user_now.saturating_sub(user));
        crate::framelog::add(crate::framelog::T::WorkerKernelUs, kernel_now.saturating_sub(kernel));
    }
}

#[cfg(windows)]
thread_local! {
    /// This thread is inside [`release_held_now`].
    static RELEASING_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// This thread is [`worker`].
    static WORKER_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// An event of this thread's for [`wait_on_line`] (0: not made yet).
    static EVENT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// With [`behind`], since the last frame-rate line ([`async_line`]):
/// microseconds or counts, [`a`] names them.
#[cfg(windows)]
static ASYNC: [AtomicU64; a::ALL] = [const { AtomicU64::new(0) }; a::ALL];

/// The parts of [`ASYNC`].
#[cfg(windows)]
mod a {
    /// Parts of submissions put in a queue's line, the parts in the lines
    /// then (summed over the puts, for the mean depth), and the most.
    pub const LINED: usize = 0;
    pub const DEPTH_SUM: usize = 1;
    pub const DEPTH_MOST: usize = 2;
    /// From a part's put in the line to its preparing (summed; the most).
    pub const AGE_US: usize = 3;
    pub const AGE_MOST: usize = 4;
    /// Our thread: its passes over the lines, its preparing and sending
    /// in them (how many parts), and of that the time no title thread
    /// waited on the lines (the preparing done beside the title's own
    /// work); waiting for work.
    pub const PASS_US: usize = 5;
    pub const PREP_US: usize = 6;
    pub const PREP_N: usize = 7;
    pub const PREP_ALONE_US: usize = 8;
    pub const IDLE_US: usize = 9;
    /// A title thread waiting on the lines, by cause (time, calls): its
    /// present (the old way, or waiting for the present before,
    /// [`super::present_wait`]), a fence it blocks on, a waited-for
    /// submission of ours, a page mapping ([`super::maps_drain`]).
    pub const PRESENT_US: usize = 10;
    pub const PRESENT_N: usize = 11;
    pub const FENCE_US: usize = 12;
    pub const FENCE_N: usize = 13;
    pub const OURS_US: usize = 14;
    pub const OURS_N: usize = 15;
    pub const MAPS_US: usize = 16;
    pub const MAPS_N: usize = 17;
    /// Of those waits: standing for the lines' lock (our thread's pass);
    /// parts a title thread prepared itself.
    pub const LOCK_US: usize = 18;
    pub const RAN_BY_TITLE: usize = 19;
    /// Presents: put behind the line after waiting for the one before
    /// ([`super::present_wait`]), the old way with the one before still
    /// in a line, the old way with the line empty.
    pub const PRESENT_WAITED: usize = 20;
    pub const PRESENT_OLD_HELD: usize = 21;
    pub const PRESENT_OLD_EMPTY: usize = 22;
    /// Blocking fence waits met by waiting for the fence
    /// ([`super::fence_wait`]), and those that ran the lines and settled
    /// as before.
    pub const FENCE_MET: usize = 23;
    pub const FENCE_DRAINED: usize = 24;
    /// The title's submit calls: time.
    pub const SUBMIT_US: usize = 25;
    /// Presents the old way (which forces what the lines hold) because a
    /// line stood at a wait not met with a part put in before the present
    /// before ([`super::stands_across_frames`]).
    pub const PRESENT_OLD_STANDS: usize = 26;
    pub const ALL: usize = 27;
}

#[cfg(windows)]
fn stat(i: usize, n: u64) {
    ASYNC[i].fetch_add(n, Ordering::Relaxed);
}

/// Parts of submissions in the queues' lines now (put in, not taken yet).
#[cfg(windows)]
static LINED_NOW: AtomicU64 = AtomicU64::new(0);

/// `n` parts were put in a line.
#[cfg(windows)]
fn lined(n: u64) {
    let now = LINED_NOW.fetch_add(n, Ordering::Relaxed) + n;
    stat(a::LINED, n);
    stat(a::DEPTH_SUM, n * now);
    ASYNC[a::DEPTH_MOST].fetch_max(now, Ordering::Relaxed);
}

/// Title threads waiting on the lines now ([`drain`], [`wait_on_line`]).
#[cfg(windows)]
static TITLE_WAITING: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Threads running the lines now (inside [`release_held_now`]'s lock).
#[cfg(windows)]
static RUNNING_LINES: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);

/// Whether parts that follow one another in the queues' lines go to the
/// host as one submission (`VICTORY_JOIN_PARTS=1`; `alt`: twenty seconds
/// with, twenty without, in turn, each change said in the log). Off by
/// default. Needs [`behind`].
///
/// Why: the title's lists wait for memory values (19 to 20 waits a frame
/// in the first rooms, each met by the bottom-of-pipe write of the part
/// before it), and every wait cuts a submission into parts: 29 host
/// submissions a frame from about 15 of the title's (run 238). Each part
/// is prepared on its own: 82 askings of the write watch, 666 refresh
/// steps, a heap sync, a merge under the flights in the air, and on the
/// card one list of ours that sends changed bytes and one that compares.
/// The title hands a frame over in one burst at the end of its recording,
/// then stands still until the line has finished it (its fence wait of
/// the next frame: run 238), so the parts are all in the lines when our
/// thread gets to them.
///
/// With the switch the thread that runs the lines takes parts off them
/// in the order they would have run and gives them to the host together:
/// the same host lists in the same order, prepared once. Between two
/// parts a wait passes if memory meets it, if a write of a flight in the
/// air is known to meet it, or if a bottom-of-pipe write of a part taken
/// before it meets it (that part is ahead of it in the joined
/// submission, which is the order the wait asks for). A signal met
/// behind parts not sent yet is delivered right behind the joined
/// submission: never before the part it followed has run. Anything else
/// (a wait not known to be met, a present, the end of what is free to
/// run) sends what was taken first. At most [`join_most`] parts or
/// [`join_lists`] lists are joined, so that the card has work while the
/// next ones are prepared. Each queue with a part taken keeps the mark
/// [`Held::Running`] at the head of its line until the part is sent:
/// what the title gives the queue meanwhile, signals too, stays behind.
#[cfg(windows)]
fn join_parts() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    let mode = *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_JOIN_PARTS").as_deref() {
            Ok("1") => 1,
            Ok("alt") => 2,
            _ => 0,
        };
        if mode != 0 {
            tracing::info!(
                "parts that follow one another in the queues' lines go to the host as one submission, at most {} parts or {} lists (VICTORY_JOIN_PARTS=1; alt: in turn)",
                join_most(),
                join_lists()
            );
        }
        mode
    });
    match mode {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!("parts joined into one host submission from now (VICTORY_JOIN_PARTS=alt): {}", if on { "ON" } else { "off, as usual" });
            }
            on
        }
    }
}

/// The most parts joined into one host submission (`VICTORY_JOIN_MOST`, 4).
#[cfg(windows)]
fn join_most() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_JOIN_MOST").ok().and_then(|v| v.parse().ok()).filter(|&n| n >= 1).unwrap_or(4))
}

/// The most host lists joined into one host submission (`VICTORY_JOIN_LISTS`, 48).
#[cfg(windows)]
fn join_lists() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| std::env::var("VICTORY_JOIN_LISTS").ok().and_then(|v| v.parse().ok()).filter(|&n| n >= 1).unwrap_or(48))
}

/// With [`join_parts`]: parts taken off the queues' lines and not yet
/// given to the host, in the order they run.
#[cfg(windows)]
#[derive(Default)]
struct Joined {
    lists: Vec<crate::list::Detached>,
    order: Vec<usize>,
    parts: usize,
    /// Signals met behind those parts: delivered when the parts are sent.
    signals: Vec<(usize, u64)>,
    /// The queues with a part here: each has [`Held::Running`] at the
    /// head of its line.
    marked: Vec<usize>,
}

#[cfg(windows)]
impl Joined {
    fn is_empty(&self) -> bool {
        self.parts == 0
    }

    /// One more part behind those taken: its lists keep their order.
    fn add(&mut self, lists: Vec<crate::list::Detached>, order: Vec<usize>) {
        let (lists, order) = join(std::mem::take(&mut self.lists), std::mem::take(&mut self.order), lists, order);
        self.lists = lists;
        self.order = order;
        self.parts += 1;
    }

    fn full(&self) -> bool {
        self.parts >= join_most() || self.order.len() >= join_lists()
    }

    /// Whether a bottom-of-pipe write of a part taken meets `wait`.
    fn meets(&self, wait: &crate::list::ValueWait) -> bool {
        let (address, bytes) = wait.place();
        self.order.iter().any(|&i| self.lists[i].host().will_write(address, bytes, |v| wait.holds_for(v)))
    }
}

#[cfg(windows)]
/// The things of one part behind those of another: `order` names places in
/// `lists`, so the second part's places move up by the first's count.
fn join<T>(mut lists: Vec<T>, mut order: Vec<usize>, more: Vec<T>, more_order: Vec<usize>) -> (Vec<T>, Vec<usize>) {
    let base = lists.len();
    lists.extend(more);
    order.extend(more_order.into_iter().map(|i| i + base));
    (lists, order)
}

/// Parts joined, host submissions they went in, their lists, signals
/// delivered behind them, and waits passed for a write of a part taken.
#[cfg(windows)]
static JOINED: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Gives the host what [`Joined`] holds, as one submission, delivers the
/// signals met behind it, and takes the marks off the lines. Called with
/// no line locked.
#[cfg(windows)]
fn send_joined(joined: &mut Joined) {
    if joined.parts > 0 {
        let (lists, order) = (std::mem::take(&mut joined.lists), std::mem::take(&mut joined.order));
        JOINED[0].fetch_add(joined.parts as u64, Ordering::Relaxed);
        let sent = JOINED[1].fetch_add(1, Ordering::Relaxed) + 1;
        JOINED[2].fetch_add(order.len() as u64, Ordering::Relaxed);
        JOINED[3].fetch_add(joined.signals.len() as u64, Ordering::Relaxed);
        if sent % 2048 == 0 {
            let n: Vec<u64> = JOINED.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
            tracing::info!(
                "parts joined (VICTORY_JOIN_PARTS=1): {} parts went to the host in {} submissions ({:.1} parts, {:.1} lists each), {} signals delivered behind them, {} waits passed for a write of a part not sent yet",
                n[0],
                n[1],
                n[0] as f64 / n[1].max(1) as f64,
                n[2] as f64 / n[1].max(1) as f64,
                n[3],
                n[4]
            );
        }
        match crate::host::host() {
            Some(host) => host.execute_late(lists, order),
            // (Not reached: parts are only taken with a host.)
            None => crate::list::recycle(lists, &order),
        }
        joined.parts = 0;
    }
    // Behind what was just sent, in the order they were met.
    for (fence, value) in joined.signals.drain(..) {
        if !crate::host::late_signal(fence, value) {
            crate::fence::set(fence as This, value);
        }
    }
    for q in joined.marked.drain(..) {
        // SAFETY: a live queue, as in the loop that marked it.
        let line = unsafe { &(*(q as *const Queue)).held };
        let mut held = lock(line);
        if matches!(held.first(), Some(Held::Running)) {
            held.remove(0);
        }
        if held.is_empty() {
            lock(&WAITING).retain(|&w| w != q);
        }
    }
}

/// Before a submission of ours that is waited for, with [`behind`]: what
/// the queues hold and is free to run runs first, on this thread, so the
/// submission keeps its place behind them.
#[cfg(windows)]
pub(crate) fn before_sync() {
    if behind() && !RELEASING_HERE.get() {
        drain(a::OURS_US);
    }
}

/// When the title put in the part our thread is preparing now, with
/// [`behind`] (it is out of its line meanwhile; one releaser at a time).
#[cfg(windows)]
static PREPARING_AT: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

/// Parts of submissions the title put in the queues' lines before `t` that
/// have not been sent yet: still in a line, or being prepared
/// (`crate::results`' flip wait).
#[cfg(windows)]
pub(crate) fn parts_put_before(t: std::time::Instant) -> usize {
    let mut n = usize::from(lock(&PREPARING_AT).is_some_and(|at| at < t));
    let queues = lock(&WAITING).clone();
    for q in queues {
        // SAFETY: a live queue: one released is taken off the list.
        let held = lock(unsafe { &(*(q as *const Queue)).held });
        n += held.iter().filter(|h| matches!(h, Held::Run(_, _, at) if *at < t)).count();
    }
    n
}

/// Whether a thread runs the lines now or our thread has been asked to
/// look at them (`crate::results`' flip wait: something still moves).
#[cfg(windows)]
pub(crate) fn lines_moving() -> bool {
    RUNNING_LINES.load(Ordering::Acquire) > 0 || (behind() && *lock(&WORK.0))
}

/// With [`behind`]: our thread looks at the lines.
#[cfg(windows)]
pub(crate) fn nudge() {
    if behind() {
        wake();
    }
}

/// [`release_held_now`] on a thread of the title's, which waits for it
/// (counted under `cause`, a pair of [`a`]: time, then calls).
#[cfg(windows)]
fn drain(cause: usize) {
    let started = std::time::Instant::now();
    let here = !WORKER_HERE.get();
    if here {
        TITLE_WAITING.fetch_add(1, Ordering::Relaxed);
    }
    release_held_now(false);
    if here {
        TITLE_WAITING.fetch_sub(1, Ordering::Relaxed);
        stat(cause, started.elapsed().as_micros() as u64);
        stat(cause + 1, 1);
    }
}

/// This thread's event for [`wait_on_line`] (made on first use; 0 if it
/// could not be).
#[cfg(windows)]
pub(crate) fn thread_event() -> usize {
    let made = EVENT.get();
    if made != 0 {
        return made;
    }
    // SAFETY: a plain auto-reset event, kept for the thread's life.
    let event = unsafe { windows_sys::Win32::System::Threading::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) } as usize;
    EVENT.set(event);
    event
}

/// Events of title threads waiting for a present of a line to be shown
/// ([`present_wait`]).
#[cfg(windows)]
static PRESENT_WAITERS: std::sync::Mutex<Vec<usize>> = std::sync::Mutex::new(Vec::new());

/// With [`behind`], on a title thread: waits until `done` says so, our
/// thread and the landing thread working meanwhile; `event` is set when
/// something `done` looks at may have changed (it is also looked at every
/// few milliseconds). False, with `done` not true, if nothing moves any
/// more: no thread runs the lines, our thread has nothing to look at, and
/// nothing is in the air or being landed (a line stands at a wait that
/// nothing of ours meets); the caller then does what it did before.
/// Counted under `cause` (a pair of [`a`]).
#[cfg(windows)]
fn wait_on_line(cause: usize, event: usize, done: &dyn Fn() -> bool) -> bool {
    use windows_sys::Win32::System::Threading::WaitForSingleObject;
    if done() {
        return true;
    }
    let started = std::time::Instant::now();
    TITLE_WAITING.fetch_add(1, Ordering::Relaxed);
    wake();
    let mut still = 0;
    let met = loop {
        if event != 0 {
            // SAFETY: an event of this thread's.
            unsafe { WaitForSingleObject(event as _, 4) };
        } else {
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        if done() {
            break true;
        }
        let idle = RUNNING_LINES.load(Ordering::Acquire) == 0 && !*lock(&WORK.0) && !crate::host::line_moving();
        still = if idle { still + 1 } else { 0 };
        // Found so three times, some milliseconds apart.
        if still >= 3 {
            break done();
        }
    };
    TITLE_WAITING.fetch_sub(1, Ordering::Relaxed);
    stat(cause, started.elapsed().as_micros() as u64);
    stat(cause + 1, 1);
    met
}

/// Whether a fence the title blocks on (a wait without an event) is
/// waited for until it is reached, with [`behind`]: our thread sends what
/// the lines hold and the landing thread sets the fence behind it, and
/// the call returns as soon as the fence gets there
/// (`VICTORY_ASYNC_FENCE_WAIT=0`: the lines are run on the title's thread
/// and everything in the air is settled, as before).
///
/// Why: before, the call ran every line as far as it was free on the
/// title's thread (also what the title submitted after the signal it
/// waits for), and then waited for every flight in the air to land
/// (`crate::host::settle_for_fence`), not only those before the signal.
/// What the title sees is the same: the fence is set where the flights
/// before its signal have landed, which is what it waits for.
#[cfg(windows)]
pub(crate) fn fence_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    behind() && *ON.get_or_init(|| std::env::var_os("VICTORY_ASYNC_FENCE_WAIT").is_none_or(|v| v != "0"))
}

/// [`fence_wait`]: waits until `reached` says the fence is there;
/// `event` is set by the fence when it gets there. False if nothing
/// moves any more ([`wait_on_line`]).
#[cfg(windows)]
pub(crate) fn wait_for_fence(event: usize, reached: &dyn Fn() -> bool) -> bool {
    let met = wait_on_line(a::FENCE_US, event, reached);
    stat(if met { a::FENCE_MET } else { a::FENCE_DRAINED }, 1);
    met
}

/// A blocking fence wait that runs the lines first, as before
/// ([`fence_wait`] off, or it found nothing moving).
#[cfg(windows)]
pub(crate) fn before_fence() {
    if behind() && !RELEASING_HERE.get() {
        drain(a::FENCE_US);
    }
}

/// Whether a page mapping (`copy_page_mappings`) first runs what the
/// lines hold and is free to run, with [`behind`]
/// (`VICTORY_ASYNC_MAPS=0`: it does not).
///
/// Why: the mapping changes at once what title addresses show, and a
/// submission still in a line reads title memory when our thread
/// prepares it (a texture in the tiled heap). On the console the mapping
/// is ordered behind the queue's earlier submissions; without
/// [`behind`] every submission the title made before it had been
/// prepared by then, so it read the pages as they were mapped when it
/// was made. With [`behind`] and without this it could be prepared after
/// the mapping and read the new pages (or none: zeros where a range was
/// unmapped).
#[cfg(windows)]
fn maps_drain() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    behind() && *ON.get_or_init(|| std::env::var_os("VICTORY_ASYNC_MAPS").is_none_or(|v| v != "0"))
}

/// Whether a present that finds the present before it still in a line
/// waits until our thread has shown that one and then goes behind the
/// line too, with [`present_behind`] (`VICTORY_ASYNC_PRESENT_WAIT=0`: it
/// runs the old way, as before: the lines first, then itself).
///
/// Why: our thread prepares more slowly than the title records, so the
/// title's present nearly always found the one before still in a line,
/// and every other present ran the old way: the title's thread stood
/// until every submission it had made was prepared, its own frame's
/// last ones too, and our thread then had nothing to prepare until the
/// title's next submissions came. With the switch it stands only until
/// the frame before is shown, and its own frame's last submissions are
/// prepared while it records the next one. The title is one frame ahead
/// at most, as with every present that goes behind.
#[cfg(windows)]
fn present_wait() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_ASYNC_PRESENT_WAIT").is_none_or(|v| v != "0"))
}

/// When the title's last present that went through [`send_behind`] was
/// called.
#[cfg(windows)]
static LAST_PRESENT: std::sync::Mutex<Option<std::time::Instant>> = std::sync::Mutex::new(None);

/// With [`present_wait`]: whether a queue's line stands at a wait that is
/// not met (a fence, a memory value) and holds a part put in before
/// `since` (the present before). Every other present ran the old way
/// before the switch and forced such a wait; with it nearly every present
/// goes behind the line, and a line that stands for good (a value the
/// card never writes) would never run again.
#[cfg(windows)]
fn stands_across_frames(since: std::time::Instant) -> bool {
    let queues = lock(&WAITING).clone();
    queues.into_iter().any(|q| {
        // SAFETY: a live queue: one released is taken off the list.
        let held = lock(unsafe { &(*(q as *const Queue)).held });
        let stands = match held.first() {
            Some(&Held::Wait(fence, value)) => crate::fence::completed(fence as This) < value,
            Some(Held::Value(wait)) => !wait.met(),
            _ => false,
        };
        stands && held.iter().any(|h| matches!(h, Held::Run(_, _, at) if *at < since))
    })
}

/// A present of a line was shown: title threads waiting for that
/// ([`present_wait`]) look again.
#[cfg(windows)]
fn presents_moved() {
    for &event in lock(&PRESENT_WAITERS).iter() {
        // SAFETY: a waiting thread's event (kept for its life).
        unsafe { windows_sys::Win32::System::Threading::SetEvent(event as _) };
    }
}

/// With [`behind`], at the frame-rate line: what the queues' lines and
/// our thread did per frame since the last one ([`ASYNC`]).
#[cfg(windows)]
pub(crate) fn async_line(frames: f64) {
    if !behind() || frames <= 0.0 {
        return;
    }
    let v: Vec<u64> = ASYNC.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
    let ms = |i: usize| v[i] as f64 / 1e3 / frames;
    let n = |i: usize| v[i] as f64 / frames;
    let per = |sum: usize, of: usize| if v[of] == 0 { 0.0 } else { v[sum] as f64 / v[of] as f64 };
    let waited = ms(a::PRESENT_US) + ms(a::FENCE_US) + ms(a::OURS_US) + ms(a::MAPS_US);
    tracing::info!(
        "submissions prepared by our thread per frame (VICTORY_ASYNC_SUBMIT=1): {:.1} parts put in the lines, {:.1} in them          then on average (the most {}), taken {:.1} ms after (the most {:.1}); our thread: passes {:.1} ms, preparing and          sending {:.1} ms in {:.1} parts, {:.1} of it beside the title (no title thread waiting on the lines), waiting for          work {:.1}; the title: in its submit calls {:.1} ms, waiting on the lines {:.1} ms: in presents {:.1} in {:.2},          blocking fence waits {:.1} in {:.2} ({:.2} met as the fence got there, {:.2} ran the lines and settled,          VICTORY_ASYNC_FENCE_WAIT={}), our own waited submissions {:.1} in {:.2}, page mappings {:.1} in {:.2}          (VICTORY_ASYNC_MAPS={}); of it standing for our thread's pass {:.1}, {:.2} parts prepared on a title thread;          presents: {:.2} behind the line after the one before was shown (VICTORY_ASYNC_PRESENT_WAIT={}), {:.2} the old way          with the one before still in a line, {:.2} the old way with the line empty, {:.2} the old way to force a line          that stood at a wait since the present before",
        n(a::LINED),
        per(a::DEPTH_SUM, a::LINED),
        v[a::DEPTH_MOST],
        per(a::AGE_US, a::LINED) / 1e3,
        v[a::AGE_MOST] as f64 / 1e3,
        ms(a::PASS_US),
        ms(a::PREP_US),
        n(a::PREP_N),
        ms(a::PREP_ALONE_US),
        ms(a::IDLE_US),
        ms(a::SUBMIT_US),
        waited,
        ms(a::PRESENT_US),
        n(a::PRESENT_N),
        ms(a::FENCE_US),
        n(a::FENCE_N),
        n(a::FENCE_MET),
        n(a::FENCE_DRAINED),
        u8::from(fence_wait()),
        ms(a::OURS_US),
        n(a::OURS_N),
        ms(a::MAPS_US),
        n(a::MAPS_N),
        u8::from(maps_drain()),
        ms(a::LOCK_US),
        n(a::RAN_BY_TITLE),
        n(a::PRESENT_WAITED),
        u8::from(present_behind() && present_wait()),
        n(a::PRESENT_OLD_HELD),
        n(a::PRESENT_OLD_EMPTY),
        n(a::PRESENT_OLD_STANDS),
    );
}

#[cfg(windows)]
fn release_held_now(all: bool) {
    // One caller at a time, so a queue's submissions keep their order.
    static RELEASING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if lock(&WAITING).is_empty() {
        return;
    }
    let locking = std::time::Instant::now();
    let _one = lock(&RELEASING);
    if behind() && !WORKER_HERE.get() {
        stat(a::LOCK_US, locking.elapsed().as_micros() as u64);
    }
    RUNNING_LINES.fetch_add(1, Ordering::AcqRel);
    let was_here = RELEASING_HERE.replace(true);
    struct Leave(bool);
    impl Drop for Leave {
        fn drop(&mut self) {
            RELEASING_HERE.set(self.0);
            RUNNING_LINES.fetch_sub(1, Ordering::AcqRel);
        }
    }
    let _leave = Leave(was_here);
    // Parts taken and not sent yet ([`join_parts`]): none when the lines
    // are forced, and none is left when this returns.
    let joining = !all && behind() && crate::host::late() && crate::host::host().is_some() && join_parts();
    let mut joined = Joined::default();
    loop {
        let queues = lock(&WAITING).clone();
        let mut moved = false;
        for q in queues {
            // SAFETY: a live queue: one released is taken off the list.
            let line = unsafe { &(*(q as *const Queue)).held };
            let mut held = lock(line);
            loop {
                // A queue with a part in `joined` has our mark at the head
                // of its line: what is looked at is behind the mark.
                let base = usize::from(joined.marked.contains(&q));
                match held.get(base) {
                    None | Some(Held::Running) => break,
                    Some(&Held::Wait(fence, value))
                        if !all && crate::fence::completed(fence as This) < value =>
                    {
                        break;
                    }
                    // With parts taken and not sent: only what is known
                    // without a wait for the card. A wait that stays is
                    // looked at again when the parts have been sent.
                    Some(Held::Value(wait)) if !all && !joined.is_empty() => {
                        if !wait.met() && !crate::host::late_met_known(wait) {
                            if !joined.meets(wait) {
                                break;
                            }
                            JOINED[4].fetch_add(1, Ordering::Relaxed);
                        }
                    }
                    Some(Held::Value(wait)) if !all && !value_met(wait) => break,
                    // The frame is shown behind the parts taken: they are
                    // sent first.
                    Some(Held::Present(..)) if !joined.is_empty() => break,
                    Some(_) => {}
                }
                match held.remove(base) {
                    Held::Wait(..) | Held::Running => {}
                    Held::Value(wait) => {
                        // Passed at a present without being met: what the
                        // queue stood at.
                        if all && !value_met(&wait) {
                            static SEEN: AtomicU64 = AtomicU64::new(0);
                            if SEEN.fetch_add(1, Ordering::Relaxed) < 16 {
                                tracing::warn!(
                                    queue = format_args!("{q:#x}"),
                                    ?wait,
                                    "a wait for a memory value was not met at a present: the queue runs on"
                                );
                            }
                        }
                    }
                    Held::Run(lists, order, at) => {
                        LINED_NOW.fetch_sub(1, Ordering::Relaxed);
                        let age = at.elapsed().as_micros() as u64;
                        stat(a::AGE_US, age);
                        ASYNC[a::AGE_MOST].fetch_max(age, Ordering::Relaxed);
                        if !WORKER_HERE.get() {
                            stat(a::RAN_BY_TITLE, 1);
                        }
                        #[cfg(feature = "diagnostics")]
                        crate::capture::op(q, || {
                            format!(
                                "RUN held submission of {} lists (forced {all})",
                                order.len()
                            )
                        });
                        // Diagnostics: the part behind this one stands at
                        // waits that are met already, so the two could have
                        // been one host submission (one wait for the card,
                        // each range shaders write compared once).
                        #[cfg(feature = "diagnostics")]
                        {
                            let waits = held.iter().take_while(|h| matches!(h, Held::Value(_))).count();
                            if waits > 0
                                && matches!(held.get(waits), Some(Held::Run(..)))
                                && held[..waits].iter().all(|h| matches!(h, Held::Value(w) if w.met()))
                            {
                                crate::copystat::add(crate::copystat::C::PartsTogether, 1);
                            }
                        }
                        if joining {
                            // Taken, to go with the parts that follow it.
                            // The line stays open behind our mark.
                            if !joined.marked.contains(&q) {
                                held.insert(0, Held::Running);
                                joined.marked.push(q);
                            }
                            joined.add(lists, order);
                            if joined.full() {
                                drop(held);
                                send_joined(&mut joined);
                                held = lock(line);
                            }
                            moved = true;
                            continue;
                        }
                        if crate::host::late()
                            && let Some(host) = crate::host::host()
                        {
                            // The parts right behind this one that can go
                            // with it as one submission (`VICTORY_JOIN_PARTS`),
                            // taken from the line now, with the signals
                            // between them, to be set behind the joined flight.
                            let (joined, signals): (Vec<(Vec<crate::list::Detached>, Vec<usize>)>, Vec<(usize, u64)>) = (Vec::new(), Vec::new());
                            let send = |lists: Vec<crate::list::Detached>, order: Vec<usize>, joined: Vec<(Vec<crate::list::Detached>, Vec<usize>)>| {
                                if joined.is_empty() {
                                    host.execute_late(lists, order);
                                } else {
                                    let mut parts = Vec::with_capacity(joined.len() + 1);
                                    parts.push((lists, order));
                                    parts.extend(joined);
                                    host.execute_late_joined(parts);
                                }
                            };
                            // Not waited for: the host keeps the lists
                            // until they are settled.
                            if behind() && !all {
                                // The line is open while this part is
                                // prepared and sent: only this thread takes
                                // from its head (one releaser at a time).
                                held.insert(0, Held::Running);
                                drop(held);
                                let alone = TITLE_WAITING.load(Ordering::Relaxed) == 0;
                                let preparing = std::time::Instant::now();
                                *lock(&PREPARING_AT) = Some(at);
                                if crate::host::async_race() {
                                    crate::host::race_check(&lists, at);
                                    for (part, _) in &joined {
                                        crate::host::race_check(part, at);
                                    }
                                }
                                send(lists, order, joined);
                                *lock(&PREPARING_AT) = None;
                                crate::results::sent(at);
                                if WORKER_HERE.get() {
                                    let us = preparing.elapsed().as_micros() as u64;
                                    stat(a::PREP_US, us);
                                    stat(a::PREP_N, 1);
                                    // No title thread waited on the lines when
                                    // it began or ended: done beside the title.
                                    if alone && TITLE_WAITING.load(Ordering::Relaxed) == 0 {
                                        stat(a::PREP_ALONE_US, us);
                                    }
                                }
                                held = lock(line);
                                if matches!(held.first(), Some(Held::Running)) {
                                    held.remove(0);
                                }
                            } else {
                                *lock(&PREPARING_AT) = Some(at);
                                if crate::host::async_race() {
                                    crate::host::race_check(&lists, at);
                                    for (part, _) in &joined {
                                        crate::host::race_check(part, at);
                                    }
                                }
                                send(lists, order, joined);
                                *lock(&PREPARING_AT) = None;
                                crate::results::sent(at);
                            }
                            // Behind the joined flight, in their order.
                            for (fence, value) in signals {
                                if !crate::host::late_signal(fence, value) {
                                    crate::fence::set(fence as This, value);
                                }
                            }
                        } else {
                            let ran = {
                                let runs: Vec<&crate::host::HostList> =
                                    order.iter().map(|&i| lists[i].host()).collect();
                                match crate::host::host() {
                                    Some(host) => {
                                        host.execute(&runs);
                                        true
                                    }
                                    None => false,
                                }
                            };
                            // They have run: their host lists are used again.
                            if ran {
                                crate::list::recycle(lists, &order);
                            }
                        }
                        if all {
                            #[cfg(feature = "diagnostics")]
                            crate::copystat::add(crate::copystat::C::Forced, 1);
                            static ONCE: AtomicBool = AtomicBool::new(false);
                            if !ONCE.swap(true, Ordering::Relaxed) {
                                tracing::warn!(
                                    "a queue still held a submission at a present: its fence was not reached"
                                );
                            }
                        }
                    }
                    Held::Present(texture, state, frame) => {
                        let show = || {
                            if let Some(host) = crate::host::host() {
                                let _ = host.present(&texture, state, frame);
                            }
                        };
                        if all {
                            show();
                        } else {
                            // The line is open while the frame is shown,
                            // as it is while a submission is prepared.
                            held.insert(0, Held::Running);
                            drop(held);
                            show();
                            held = lock(line);
                            if matches!(held.first(), Some(Held::Running)) {
                                held.remove(0);
                            }
                        }
                        PRESENTS_HELD.fetch_sub(1, Ordering::Relaxed);
                        presents_moved();
                    }
                    // Behind what was sent to the card and not waited
                    // for, if anything was.
                    Held::Signal(fence, value) => {
                        if !joined.is_empty() {
                            // Behind parts not sent yet: with them.
                            joined.signals.push((fence, value));
                        } else if !crate::host::late_signal(fence, value) {
                            crate::fence::set(fence as This, value);
                        }
                    }
                }
                moved = true;
            }
            if held.is_empty() {
                lock(&WAITING).retain(|&w| w != q);
            }
        }
        if !moved {
            if joined.is_empty() {
                break;
            }
            // Nothing more is free to run behind the parts taken: they go
            // to the host, and what stood at a wait is looked at again.
            send_joined(&mut joined);
        }
    }
}

/// Whether a queue may pass `wait`: memory holds the value, or, with
/// submissions sent to the card without a wait (`crate::host::late`), it
/// will once those in the air have run.
#[cfg(windows)]
fn value_met(wait: &crate::list::ValueWait) -> bool {
    wait.met() || (crate::host::late() && crate::host::late_met(wait))
}

fn queue<'a>(this: This) -> &'a Queue {
    // SAFETY: called from queue methods only.
    unsafe { &*this.cast::<Queue>() }
}

pub(crate) extern "system" fn create(
    _: This,
    desc: *const QueueDesc,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&desc) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(?desc, "command queue");
    let queue = Box::new(Queue {
        object: if crate::device::newer() {
            Object::new(&NEWER_FUNCTIONS)
        } else {
            Object::new(&FUNCTIONS)
        },
        kind: desc.kind,
        desc,
        #[cfg(windows)]
        held: Default::default(),
    });
    object::hand_out(QUEUE, queue, iid, &QUEUE_IIDS, out)
}

/// The console's queue description: an engine or pipe and a queue on it
/// in place of PC's priority and node mask.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct QueueDescX {
    kind: u32,
    flags: u32,
    engine_or_pipe: u32,
    queue: u32,
}

/// Device method `CreateCommandQueueX`. The host picks engines itself, so
/// the queue is created as on PC.
pub(crate) extern "system" fn create_x(
    this: This,
    desc: *const QueueDescX,
    iid: *const Iid,
    out: *mut This,
) -> HRESULT {
    // SAFETY: the title passes a description (null is tolerated).
    let Some(&x) = (unsafe { desc.as_ref() }) else {
        return E_INVALIDARG;
    };
    tracing::debug!(desc = ?x, "console command queue");
    let desc = QueueDesc {
        kind: x.kind,
        priority: 0,
        flags: x.flags,
        nodes: 1,
    };
    create(this, &desc, iid, out)
}

extern "system" fn query(this: This, iid: *const Iid, out: *mut This) -> HRESULT {
    object::answer(
        QUEUE,
        iid,
        object::is(iid, &QUEUE_IIDS).then_some(this),
        out,
    )
}
extern "system" fn release(this: This) -> u32 {
    let left = object::release_boxed::<Queue>(this);
    #[cfg(windows)]
    if left == 0 {
        lock(&WAITING).retain(|&q| q != this as usize);
    }
    left
}

/// Runs the lists' host lists and waits for them (decision 0007). A queue
/// that waits for a fence keeps them until the fence is reached.
extern "system" fn execute(this: This, count: u32, lists: *const This) {
    #[cfg(windows)]
    let started = std::time::Instant::now();
    execute_now(this, count, lists);
    #[cfg(windows)]
    {
        let us = started.elapsed().as_micros() as u64;
        if behind() {
            stat(a::SUBMIT_US, us);
        }
        crate::framelog::add(crate::framelog::T::ExecuteUs, us);
        crate::framelog::here(crate::framelog::H::ExecuteUs, us);
    }
}

fn execute_now(this: This, count: u32, lists: *const This) {
    if lists.is_null() {
        return;
    }
    #[cfg(windows)]
    if let Some(host) = crate::host::host() {
        // SAFETY: the title passes `count` lists.
        let lists = unsafe { std::slice::from_raw_parts(lists, count as usize) };
        // The walls trace: where the lists' results for the title's
        // processor go when they land (`crate::results`).
        if crate::results::trace() {
            let spans: Vec<(usize, u8)> =
                lists.iter().filter_map(|&l| crate::list::host_list(l)).flat_map(|h| h.result_spans()).collect();
            crate::results::submitted(&spans);
        }
        if cuts() && !this.is_null() {
            submit_cut(this, lists);
            return;
        }
        if waits() && split() && !this.is_null() {
            submit_split(this, lists);
            return;
        }
        // A list that waits for a memory value not there yet: the queue
        // holds the submission, as it does behind a fence.
        if waits() && !this.is_null() {
            let unmet = crate::list::unmet_waits(lists);
            if !unmet.is_empty() {
                static SEEN: AtomicU64 = AtomicU64::new(0);
                if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::info!(kind = queue(this).kind, ?unmet, "a submission is held for a memory value");
                }
                lock(&queue(this).held).extend(unmet.into_iter().map(Held::Value));
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
        }
        let kept = hold(this, || {
            let (taken, order) = crate::list::detach(lists);
            if crate::host::async_race() {
                crate::host::race_note(&taken);
            }
            lined(1);
            Held::Run(taken, order, std::time::Instant::now())
        });
        // Diagnostics: the order submissions are made in, next to the
        // commands of the frame's `ops` file (which are in recording order).
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || {
            let kind = (!this.is_null()).then(|| queue(this).kind);
            // The lists as the frame's other lines name them (their host sides).
            let named: Vec<String> = lists
                .iter()
                .map(|&l| match crate::list::host_list(l) {
                    Some(h) => format!("{:#x}", std::ptr::from_ref(h) as usize),
                    None => format!("title {l:?}"),
                })
                .collect();
            format!("SUBMIT queue kind {kind:?} held {kept} lists {named:?}")
        });
        if kept {
            return;
        }
        let host_lists: Vec<_> = lists
            .iter()
            .filter_map(|&l| crate::list::host_list(l))
            .collect();
        host.execute(&host_lists);
        crate::list::tables_done(lists);
        // What the lists wrote at the bottom of the pipe may be a fence
        // another queue waits for.
        release_held(false);
        return;
    }
    let _ = this;
    static ONCE: AtomicBool = AtomicBool::new(false);
    if !ONCE.swap(true, Ordering::Relaxed) {
        tracing::warn!(count, "command lists submitted; no host GPU runs them");
    }
}

/// The start of a present plane: frame token, then the plane's resources.
#[repr(C)]
pub(crate) struct Plane {
    token: u64,
    count: u32,
    resources: *const This,
}
/// Submission tagged with the frame it belongs to `(count, lists, frame
/// token, flags)`; frames are paced elsewhere, so it runs as a plain one.
extern "system" fn execute_x(this: This, count: u32, lists: *const This, _token: u64, _flags: u32) {
    execute(this, count, lists);
}
/// Signal with sync settings for the top and bottom of the pipe; our
/// submissions are complete when they return, so it signals at once.
extern "system" fn signal2_x(
    this: This,
    fence: This,
    value: u64,
    _bottom: *const u8,
    _top: *const u8,
) -> HRESULT {
    signal(this, fence, value)
}
/// Markers and events only matter to profilers.
extern "system" fn marker(_: This) {}
extern "system" fn signal(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    crate::fence::ask(fence, value);
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::fence::pace::signal_called(fence as usize, value);
    // A queue that waits signals after what it holds.
    #[cfg(windows)]
    if hold(this, || Held::Signal(fence as usize, value)) {
        return S_OK;
    }
    // Behind the submissions sent to the card and not waited for: the
    // fence is set when they are settled.
    #[cfg(windows)]
    if crate::host::late_signal(fence as usize, value) {
        return S_OK;
    }
    let _ = this;
    crate::fence::signal(fence, value)
}
/// From here on the queue runs nothing it is given until `fence` reaches
/// `value`: how the console orders work between queues (the title's
/// compute queue waits for the depth the graphics queue draws). Our
/// submissions run when they are made, so a wait for a value already
/// reached is nothing; otherwise what the queue is given next is held
/// ([`Held`]) until a signal or a list's bottom-of-pipe write gets the
/// fence there.
extern "system" fn wait(this: This, fence: This, value: u64) -> HRESULT {
    if fence.is_null() {
        return E_INVALIDARG;
    }
    #[cfg(windows)]
    if !this.is_null() {
        let now = crate::fence::completed(fence);
        // Whether the title uses this at all (the first calls).
        {
            static CALLS: AtomicU64 = AtomicU64::new(0);
            if CALLS.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(kind = queue(this).kind, now, value, "a queue is told to wait for a fence");
            }
        }
        #[cfg(feature = "diagnostics")]
        crate::capture::op(this as usize, || format!("WAIT queue kind {} for fence {fence:?} value {value} (now {now})", queue(this).kind));
        let behind = {
            let mut held = lock(&queue(this).held);
            let behind = !held.is_empty() || now < value;
            if behind && waits() {
                held.push(Held::Wait(fence as usize, value));
            }
            behind
        };
        if behind {
            static SEEN: AtomicU64 = AtomicU64::new(0);
            if SEEN.fetch_add(1, Ordering::Relaxed) < 8 {
                tracing::info!(
                    kind = queue(this).kind,
                    now,
                    value,
                    held = waits(),
                    "a queue waits for a fence not reached yet"
                );
            }
        }
        if behind && waits() {
            {
                let mut waiting = lock(&WAITING);
                if !waiting.contains(&(this as usize)) {
                    waiting.push(this as usize);
                }
            }
            // The fence may have got there meanwhile.
            release_held(false);
        }
    }
    let _ = (this, value);
    S_OK
}
/// Ticks per second of GPU timestamps (see [`Written::GpuTimestamp`]).
extern "system" fn timestamp_frequency(_: This, out: *mut u64) -> HRESULT {
    if out.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: checked non-null.
    unsafe { out.write(TIMESTAMP_FREQUENCY) };
    S_OK
}
const TIMESTAMP_FREQUENCY: u64 = 100_000_000;

/// What a list's bottom-of-pipe write stores when the list has run.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Written {
    Value(u64),
    /// The GPU clock: [`TIMESTAMP_FREQUENCY`] ticks per second since the
    /// process started (our GPU work is done on the CPU's timeline).
    GpuTimestamp,
    /// The CPU's performance counter.
    SystemTimestamp,
}

impl Written {
    /// Stores the low `bytes` (4 or 8) of the value at `at`.
    pub(crate) fn write(self, at: *mut u8, bytes: usize) {
        let value = match self {
            Written::Value(v) => v,
            Written::GpuTimestamp => {
                static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
                let elapsed = START.get_or_init(std::time::Instant::now).elapsed();
                (elapsed.as_nanos() / (1_000_000_000 / u128::from(TIMESTAMP_FREQUENCY))) as u64
            }
            Written::SystemTimestamp => system_timestamp(),
        };
        #[cfg(windows)]
        if crate::results::trace() {
            crate::results::landed(at as usize);
        }
        // SAFETY: the title names writable GPU memory (unified with ours).
        unsafe {
            if bytes == 4 {
                at.cast::<u32>().write_unaligned(value as u32);
            } else {
                at.cast::<u64>().write_unaligned(value);
            }
        }
    }
}

fn system_timestamp() -> u64 {
    #[cfg(windows)]
    {
        let mut count = 0;
        // SAFETY: a valid out pointer.
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut count) };
        count as u64
    }
    #[cfg(not(windows))]
    0
}

/// Frames presented so far.
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// Title images drawn into (as render targets) or copied to: `.0` since they
/// were last presented, `.1` ever. the title's opening video and
/// title cards present its two images in turn but draw into only every
/// other one (the frames between show an image it did not touch): shown as
/// they are, the window flicked between the newest picture and one from
/// long before. Such an image is now passed over for the one shown last
/// (`VICTORY_PRESENT_EVERY=1`: every image as the title names it). An image
/// written only by shaders (an upscaler's output) is never in `.1`, so it is
/// always shown.
static WRITTEN: std::sync::Mutex<Option<(std::collections::HashSet<usize>, std::collections::HashSet<usize>)>> =
    std::sync::Mutex::new(None);

/// `resource` was drawn into or copied to (see [`WRITTEN`]).
pub(crate) fn note_written(resource: This) {
    if resource.is_null() {
        return;
    }
    let mut w = WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
    let (since, ever) = w.get_or_insert_with(Default::default);
    // (bounded: render targets come and go; a clear only lets one stale
    // image through)
    if since.len() > 8192 {
        since.clear();
    }
    if ever.len() > 8192 {
        ever.clear();
    }
    since.insert(resource as usize);
    ever.insert(resource as usize);
}

/// The image to show for a present of `resource`: itself, or the one shown
/// last when `resource` is an image the title draws into and has not since
/// it was last shown (see [`WRITTEN`]).
#[cfg(windows)]
fn fresh_or_last(resource: This) -> This {
    static EVERY: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    static LAST: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if *EVERY.get_or_init(|| std::env::var_os("VICTORY_PRESENT_EVERY").is_some()) {
        return resource;
    }
    let key = resource as usize;
    let stale = {
        let mut w = WRITTEN.lock().unwrap_or_else(|e| e.into_inner());
        let (since, ever) = w.get_or_insert_with(Default::default);
        let fresh = since.remove(&key);
        !fresh && ever.contains(&key)
    };
    let last = LAST.load(Ordering::Relaxed);
    if stale && last != 0 && last != key {
        static N: AtomicU64 = AtomicU64::new(0);
        if N.fetch_add(1, Ordering::Relaxed) < 4 {
            tracing::info!("a present of an image not drawn since it was last shown: the last image is shown again");
        }
        return last as This;
    }
    LAST.store(key, Ordering::Relaxed);
    resource
}

#[cfg(feature = "diagnostics")]
pub(crate) fn presented() -> u64 {
    FRAMES.load(Ordering::Relaxed)
}

/// Shows the first plane's first resource in our window.
///
/// shortcut: one plane, no scaling or placement, the source expected in
/// the common state.
extern "system" fn present(
    this: This,
    planes: u32,
    plane_parameters: *const Plane,
    _parameters: *const u8,
) -> HRESULT {
    #[cfg(windows)]
    let presenting = std::time::Instant::now();
    // What the title submitted before this present is what its flip wait
    // before the next one waits for (`crate::results`).
    #[cfg(windows)]
    crate::results::present_called();
    let hr = present_now(this, planes, plane_parameters, _parameters);
    #[cfg(windows)]
    {
        crate::framelog::add(crate::framelog::T::PresentUs, presenting.elapsed().as_micros() as u64);
        crate::framelog::frame();
        #[cfg(feature = "diagnostics")]
        crate::fence::pace::frame();
    }
    hr
}

fn present_now(this: This, planes: u32, plane_parameters: *const Plane, _parameters: *const u8) -> HRESULT {
    #[cfg(windows)]
    let presenting = std::time::Instant::now();
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::copystat::present_begins();
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::profile::frame();
    // With `present_behind`: the frame is shown by our thread, behind what
    // the queue holds, and nothing is run or waited for here.
    #[cfg(windows)]
    let sent_behind = present_behind() && send_behind(this, planes, plane_parameters);
    #[cfg(not(windows))]
    let _ = this;
    // With `behind`: the title's thread waits on the lines from here
    // (counted).
    #[cfg(windows)]
    let draining = (behind() && !sent_behind).then(|| {
        TITLE_WAITING.fetch_add(1, Ordering::Relaxed);
        std::time::Instant::now()
    });
    // Submissions sent to the card and not waited for are settled first,
    // and what that frees runs in its own order, before anything is forced.
    #[cfg(windows)]
    if sent_behind {
    } else if crate::host::late() && !crate::host::late_free_present() {
        // (With `behind`: what the queues hold runs here first, and what
        // it sent is settled with the rest.)
        if behind() {
            release_held_now(false);
        }
        crate::host::settle_for_present();
        release_held_now(false);
        crate::host::settle_for_present();
    } else if crate::host::late() {
        // The present is sent ahead too: what waits for a value a flight
        // will write runs behind that flight, and nothing is settled here.
        release_held_now(false);
    }
    // shortcut: what a queue still holds at a present runs now, whatever
    // its fence: a wait is not kept across frames.
    #[cfg(windows)]
    if !sent_behind {
        release_held(forces());
    }
    #[cfg(windows)]
    if let Some(draining) = draining {
        TITLE_WAITING.fetch_sub(1, Ordering::Relaxed);
        stat(a::PRESENT_US, draining.elapsed().as_micros() as u64);
        stat(a::PRESENT_N, 1);
    }
    #[cfg(windows)]
    crate::framelog::add(crate::framelog::T::PresentLineUs, presenting.elapsed().as_micros() as u64);
    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    #[cfg(windows)]
    if sent_behind && PRESENTS_BEHIND.load(Ordering::Relaxed).is_multiple_of(512) {
        tracing::info!(
            frame,
            behind = PRESENTS_BEHIND.load(Ordering::Relaxed),
            "presents sent behind their queue's line so far (VICTORY_ASYNC_PRESENT=1); the others ran the old way"
        );
    }
    if frame.is_power_of_two() {
        let (draws, dispatches) = crate::list::take_counts();
        #[cfg(windows)]
        let (drawn, dispatched) = {
            static DRAWN: AtomicU64 = AtomicU64::new(0);
            static DISPATCHED: AtomicU64 = AtomicU64::new(0);
            let since = |last: &AtomicU64, now: u64| now - last.swap(now, Ordering::Relaxed);
            (
                since(&DRAWN, crate::draw::drawn()),
                since(&DISPATCHED, crate::draw::dispatched()),
            )
        };
        #[cfg(not(windows))]
        let (drawn, dispatched) = (0, 0);
        tracing::info!(
            frame,
            draws,
            drawn,
            dispatches,
            dispatched,
            "frame presented (since the last line: draws and dispatches, and those run on the host)"
        );
        #[cfg(windows)]
        if frame >= 256 {
            let (hits, misses) = crate::shader_cache::counts();
            tracing::info!(hits, misses, "shaders from the cache and translated so far");
        }
    }
    #[cfg(all(windows, feature = "diagnostics"))]
    {
        crate::capture::frame_drew(frame, crate::draw::drawn());
        if crate::capture::listed(frame) {
            crate::capture::dump_targets(frame);
        }
    }
    // `VICTORY_PRESENT_LOG=1`: what the title passes with its first 600
    // presents (planes, each plane's token and resources, the first words of
    // the parameters), to see which image it means to show.
    #[cfg(windows)]
    if std::env::var_os("VICTORY_PRESENT_LOG").is_some() && frame <= 600 && !plane_parameters.is_null() {
        let mut text = String::new();
        for i in 0..planes.min(4) as usize {
            // SAFETY: the title passes `planes` plane descriptions.
            let p = unsafe { &*plane_parameters.add(i) };
            let res: Vec<String> = (0..p.count.min(4) as usize)
                // SAFETY: the plane lists `count` resources.
                .map(|r| format!("{:#x}", unsafe { *p.resources.add(r) } as usize))
                .collect();
            text.push_str(&format!(" plane{i}: token {:#x} count {} [{}]", p.token, p.count, res.join(" ")));
        }
        let params = if _parameters.is_null() {
            String::from("none")
        } else {
            // SAFETY: the title's parameter block is at least 32 bytes.
            let w = unsafe { std::slice::from_raw_parts(_parameters.cast::<u32>(), 8) };
            format!("{w:08x?}")
        };
        // draws and dispatches run on the host since the last present
        static LAST: std::sync::Mutex<(u64, u64)> = std::sync::Mutex::new((0, 0));
        let (drawn, dispatched) = (crate::draw::drawn(), crate::draw::dispatched());
        let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
        let (dd, dx) = (drawn - last.0, dispatched - last.1);
        *last = (drawn, dispatched);
        tracing::info!(frame, planes, drawn = dd, dispatched = dx, "present:{text}; parameters {params}");
    }
    #[cfg(windows)]
    if !sent_behind
        && planes > 0
        // SAFETY: the title passes `planes` plane descriptions.
        && let Some(plane) = unsafe { plane_parameters.as_ref() }
        && plane.count > 0
        && !plane.resources.is_null()
        // SAFETY: the plane lists `count` resources.
        && let resource = fresh_or_last(unsafe { *plane.resources })
        && let Some(texture) = crate::resource::host_resource(resource)
        && let Some(host) = crate::host::host()
    {
        const COPY_SOURCE: u32 = 0x800;
        let state = crate::resource::host_transition(resource, COPY_SOURCE)
            .map_or(COPY_SOURCE, |(_, before)| before);
        let shown = host.present(texture, state, frame);
        crate::resource::host_transition(resource, state);
        #[cfg(feature = "diagnostics")]
        blinks(frame, resource, state, shown);
        #[cfg(not(feature = "diagnostics"))]
        let _ = shown;
    }
    #[cfg(not(windows))]
    let _ = (planes, plane_parameters);
    S_OK
}
/// Diagnostics: logs a presented frame that is black after one that was
/// not, with what was presented and how many draws ran since.
#[cfg(all(windows, feature = "diagnostics"))]
fn blinks(frame: u64, resource: This, state: u32, shown: Option<bool>) {
    static LAST: std::sync::Mutex<(bool, u64, u64)> = std::sync::Mutex::new((false, 0, 0));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let drawn = crate::draw::drawn();
    let black = shown.unwrap_or(false);
    let draws = crate::draw::take_frame();
    if black && !last.0 {
        last.2 += 1;
        tracing::warn!(
            frame,
            resource = ?resource,
            state = format_args!("{state:#x}"),
            draws = drawn - last.1,
            copied = shown.is_some(),
            black_frames = last.2,
            levels = ?crate::capture::recent_levels(),
            "a black frame after a frame that was not"
        );
        for d in draws {
            tracing::warn!("black frame {frame}: {d}");
        }
        AFTER.store(frame + 6, Ordering::Relaxed);
    }
    // Six frames on: the levels around it (a fade or a single drop).
    static AFTER: AtomicU64 = AtomicU64::new(0);
    if AFTER.load(Ordering::Relaxed) == frame {
        tracing::warn!(frame, levels = ?crate::capture::recent_levels(), "levels after the black frame");
    }
    *last = (black, drawn, last.2);
}

/// A run of destination pages: how they are mapped (0 pool pages from
/// `start` on, 1 pool page `start` each, 2 unmapped, 3 left as they are)
/// and how many.
#[repr(C)]
pub(crate) struct PageRange {
    kind: u32,
    start: u32,
    count: u32,
}

/// The ranges mapped from a destination address on.
#[repr(C)]
pub(crate) struct PageBatch {
    destination: u64,
    count: u32,
    ranges: *const PageRange,
}

/// Bytes per mapped page.
const PAGE: u64 = 64 << 10;

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. The title streams textures into pool pages
/// and maps them into its tiled heap this way (title+0xf2c470, ranges of
/// one constant page or of unmapped pages). Done at once: every earlier
/// submission has run (with `VICTORY_ASYNC_SUBMIT=1`, has been prepared
/// once what the lines hold has run: [`maps_drain`]).
extern "system" fn copy_page_mappings(
    _: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if batches.is_null() && count > 0 {
        return E_INVALIDARG;
    }
    // Submissions made before it read the pages as they are mapped now.
    #[cfg(windows)]
    if maps_drain() && count > 0 {
        drain(a::MAPS_US);
    }
    let batches = (count > 0).then(|| {
        // SAFETY: the title passes `count` batches.
        unsafe { std::slice::from_raw_parts(batches, count as usize) }
    });
    let mut ok = true;
    for b in batches.into_iter().flatten() {
        if b.ranges.is_null() {
            continue;
        }
        // SAFETY: each batch holds `count` ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        let mut at = b.destination;
        for r in ranges {
            let len = u64::from(r.count) * PAGE;
            let inside = |pages: u32| r.start.checked_add(pages).is_some_and(|e| e <= pool_pages);
            let source = pool + u64::from(r.start) * PAGE;
            ok &= match r.kind {
                0 if inside(r.count) => map(at, source, len),
                1 if inside(1) => (0..u64::from(r.count)).all(|k| map(at + k * PAGE, source, PAGE)),
                2 => map(at, 0, len),
                3 => true,
                _ => false,
            };
            at += len;
        }
    }
    if !ok {
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!(pool = format_args!("{pool:#x}"), "a page mapping failed"));
    }
    S_OK
}

extern "system" fn desc(this: This, out: *mut QueueDesc) {
    if !out.is_null() {
        // SAFETY: checked non-null; the title passes room for a description.
        unsafe { out.write(queue(this).desc) };
    }
}

/// The queue methods by their slot in older titles' queues (the 2020-era
/// layout): IUnknown 3, object 4, device child 1, queue 26 (12 as on PC,
/// then console ones).
const METHODS: [(usize, *const ()); 15] = [
    (0, query as *const ()),
    (2, release as *const ()),
    (7, object::parent_device as *const ()), // 0x38
    (10, execute as *const ()),              // 0x50
    (11, marker as *const ()),               // 0x58 set marker
    (12, marker as *const ()),               // 0x60 begin event
    (13, marker as *const ()),               // 0x68 end event
    (14, signal as *const ()),               // 0x70
    (15, wait as *const ()),                 // 0x78
    (16, timestamp_frequency as *const ()),  // 0x80
    (26, desc as *const ()),                 // 0xd0
    (29, copy_page_mappings as *const ()),   // 0xe8 page mapping batch
    (30, present as *const ()),              // 0xf0
    (33, execute_x as *const ()),            // 0x108
    (32, signal2_x as *const ()),            // 0x100
];

/// Newer titles' queues have two more methods before Signal, which moved
/// from 0x70 to 0x80: every slot from 0x70 on is two later (taken here as
/// starting exactly at 0x70, the methods before it kept in place).
/// Inferred from the title's calls: Signal 0x80 `(fence,
/// value)`, timestamp frequency 0x90 (title+0x24841a6), suspend and resume
/// 0xb8/0xc0, PresentX 0x100, the frame-token submission 0x118 `(count,
/// lists, token, flags)` (title+0x24821dd).
const fn newer_slot(slot: usize) -> usize {
    if slot >= 14 { slot + 2 } else { slot }
}

const fn queue_table(newer: bool) -> [*const (); 36] {
    let mut f = object::d3d12_object(object::missing_table::<QUEUE, 36>());
    let mut i = 0;
    while i < METHODS.len() {
        let (slot, method) = METHODS[i];
        f[if newer { newer_slot(slot) } else { slot }] = method;
        i += 1;
    }
    f
}

static FUNCTIONS: Table<36> = Table(queue_table(false));
static NEWER_FUNCTIONS: Table<36> = Table(queue_table(true));

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(windows)]
    #[test]
    fn joined_parts_keep_their_lists_and_their_order() {
        // a part of three lists run as 2, 0, 1; then one of two run as 1, 0; then one of one
        let (lists, order) = join(vec!["a0", "a1", "a2"], vec![2, 0, 1], vec!["b0", "b1"], vec![1, 0]);
        let (lists, order) = join(lists, order, vec!["c0"], vec![0]);
        assert_eq!(lists, ["a0", "a1", "a2", "b0", "b1", "c0"]);
        let run: Vec<&str> = order.iter().map(|&i| lists[i]).collect();
        assert_eq!(run, ["a2", "a0", "a1", "b1", "b0", "c0"], "each part's lists in its own order, the parts one after the other");
        // a part with a list named twice, and an empty one
        let (lists, order) = join(vec!["a0"], vec![0, 0], Vec::new(), Vec::new());
        let (lists, order) = join(lists, order, vec!["b0"], vec![0, 0]);
        assert_eq!(order.iter().map(|&i| lists[i]).collect::<Vec<_>>(), ["a0", "a0", "b0", "b0"]);
    }
    use crate::object::slot;

    /// Where a queue's signal is: newer layouts have it two slots later.
    /// Another test may switch layouts while this one runs, so the queue's
    /// own table decides.
    fn signal_slot(q: This) -> usize {
        // SAFETY: one of our queues; its first field is the table.
        let table = unsafe { *q.cast::<*const *const ()>() };
        if table == NEWER_FUNCTIONS.0.as_ptr() {
            0x80
        } else {
            0x70
        }
    }

    #[test]
    fn queue_signals_complete_at_once() {
        let null = std::ptr::null();
        let desc = QueueDesc {
            kind: 0,
            priority: 0,
            flags: 0,
            nodes: 0,
        };
        let mut q = null;
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        // SAFETY: the title reads the type at +0x10.
        assert_eq!(unsafe { q.cast::<u8>().add(0x10).cast::<u32>().read() }, 0);
        let mut fence = null;
        let fence_iid = &crate::fence::FENCE_IIDS[0];
        assert_eq!(
            crate::fence::create(null, 0, 0, fence_iid, &mut fence),
            S_OK
        );
        let signal = slot::<extern "system" fn(This, This, u64) -> HRESULT>(q, signal_slot(q));
        assert_eq!(signal(q, fence, 7), S_OK);
        assert_eq!(
            slot::<extern "system" fn(This) -> u64>(fence, 0x40)(fence),
            7
        );
        assert_eq!(slot::<extern "system" fn(This) -> u32>(q, 0x10)(q), 0);
    }

    /// The title's compute queue: `{COMPUTE, 0, pipe 1, queue 0}`.
    #[test]
    fn console_queues_are_created_as_on_pc() {
        let null = std::ptr::null();
        let desc = QueueDescX {
            kind: 2,
            flags: 0,
            engine_or_pipe: 1,
            queue: 0,
        };
        let mut q = null;
        assert_eq!(create_x(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        assert_eq!(queue(q).desc.kind, 2);
        let none = std::ptr::null();
        assert_eq!(create_x(null, none, &QUEUE_IIDS[0], &mut q), E_INVALIDARG);
    }

    #[test]
    fn newer_queues_have_their_methods_two_slots_later() {
        let (old, new) = (queue_table(false), queue_table(true));
        assert_eq!(old[10], new[10]); // submission stays at 0x50
        assert_eq!(old[14], new[16]); // signal 0x70 → 0x80
        assert_eq!(old[30], new[32]); // present 0xf0 → 0x100
        assert_eq!(new[35], execute_x as *const ()); // 0x118
    }

    #[test]
    fn timestamps_and_values_are_written() {
        let mut at = [0u64; 2];
        Written::Value(0x1_0000_0002).write(at.as_mut_ptr().cast(), 4);
        assert_eq!(at[0], 2);
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        let first = at[1];
        std::thread::sleep(std::time::Duration::from_millis(2));
        Written::GpuTimestamp.write(at[1..].as_mut_ptr().cast(), 8);
        assert!(at[1] > first);
    }
}
