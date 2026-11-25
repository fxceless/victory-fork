//! Command queues. Submitted lists run on the host GPU and are waited for,
//! so fence signals complete at once.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::object::{self, E_INVALIDARG, HRESULT, Iid, Object, QUEUE, S_OK, Table, This, guid};
use crate::resource::PAGEABLE_IIDS;
#[cfg(windows)]
use crate::fence::FenceRef;

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
    held: std::sync::Mutex<QueueLine>,
}
const _: () = assert!(std::mem::offset_of!(Queue, kind) == 0x10);

#[cfg(windows)]
#[derive(Default)]
struct QueueLine {
    items: Vec<Held>,
    // A pending remap commits this segment to real queue ordering. Keep the
    // flag until the line drains, including after the remap has executed:
    // a dependent run may still be waiting when its present is called.
    ordered: bool,
}

#[cfg(windows)]
impl std::ops::Deref for QueueLine {
    type Target = Vec<Held>;

    fn deref(&self) -> &Self::Target {
        &self.items
    }
}

#[cfg(windows)]
impl std::ops::DerefMut for QueueLine {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.items
    }
}

#[cfg(windows)]
impl QueueLine {
    fn unregister_if_drained(&mut self, queue: usize) {
        if self.is_empty() {
            self.ordered = false;
            lock(&WAITING).retain(|w| w.0 != queue);
        }
    }
}

/// One thing a waiting queue holds back.
#[cfg(windows)]
enum Held {
    /// The queue waits for the fence to reach the value.
    Wait(FenceRef, u64),
    /// A list of the next submission waits for a memory value.
    Value(crate::list::ValueWait),
    /// A submission: the lists' host sides, taken from the title's lists
    /// (which it may reset meanwhile), and the order they run in.
    Run(Vec<crate::list::Detached>, Vec<usize>),
    /// A page mapping change, after preceding submissions and their copy-back.
    MapPages(QueuedPageMappings),
    /// A signal of the queue's, due after what it holds before it.
    Signal(FenceRef, u64),
    /// With [`behind`]: the submission at the head of the line is being
    /// prepared and sent by our thread, with the line unlocked, so the
    /// title can add to its end meanwhile; whatever is added stays behind
    /// this (the line is not empty), signals too.
    Running,
    /// A frame shown after the work already admitted to this queue.
    Present(PresentSource),
}

#[cfg(windows)]
enum PresentSource {
    Texture(std::sync::Arc<crate::resource::HostTexture>, u32, u64),
    #[cfg(test)]
    Observe(Box<dyn FnOnce() + Send>),
}

#[cfg(windows)]
impl PresentSource {
    fn show(self) {
        match self {
            Self::Texture(texture, state, frame) => {
                if let Some(host) = crate::host::host() {
                    let _ = host.present(&texture.texture, state, frame);
                }
            }
            #[cfg(test)]
            Self::Observe(observe) => observe(),
        }
    }
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
/// therefore still forced, by the present after it, except across ordered
/// mappings. These presents must queue regardless of the experiment switch.
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

/// Optional pipelining never overrides a mapping's ordering requirement.
#[cfg(windows)]
fn present_admission(line: &QueueLine, opportunistic: bool) -> Option<bool> {
    if line.ordered {
        Some(true)
    } else if opportunistic && !line.is_empty() && PRESENTS_HELD.load(Ordering::Relaxed) == 0 {
        Some(false)
    } else {
        None
    }
}

#[cfg(windows)]
fn enqueue_present(this: This, opportunistic: bool, source: impl FnOnce() -> Option<PresentSource>) -> bool {
    let line = &queue(this).held;
    if present_admission(&lock(line), opportunistic).is_none() {
        return false;
    }
    // Resolving a host texture can allocate. Never do it under the line lock.
    let Some(source) = source() else { return false };
    let mut held = lock(line);
    let Some(ordered) = present_admission(&held, opportunistic) else { return false };
    PRESENTS_HELD.fetch_add(1, Ordering::Relaxed);
    held.push(Held::Present(source));
    mark_waiting(this);
    drop(held);
    if !ordered {
        let behind = PRESENTS_BEHIND.fetch_add(1, Ordering::Relaxed) + 1;
        if behind.is_multiple_of(512) {
            tracing::info!(behind, "presents sent ahead of the caller by the optional async-present mode");
        }
        wake();
    }
    // In synchronous mode the operation ahead of an ordered present will
    // resume the line on return or when its wait is satisfied.
    true
}

#[cfg(windows)]
fn send_behind(this: This, resource: This, frame: u64, opportunistic: bool) -> bool {
    if this.is_null() || resource.is_null() {
        return false;
    }
    // Optional presents avoid synchronous captures; required ordering applies
    // to those frames too. Host::present still handles their readback.
    #[cfg(feature = "diagnostics")]
    let opportunistic = opportunistic && !crate::capture::wanted(frame) && !crate::capture::listed(frame);
    enqueue_present(this, opportunistic, || {
        let texture = crate::resource::host_copy(resource)?;
        let state = texture.state.load(Ordering::Relaxed);
        Some(PresentSource::Texture(texture, state, frame))
    })
}

#[cfg(windows)]
fn present_resource(planes: u32, plane_parameters: *const Plane) -> This {
    if planes == 0 {
        return std::ptr::null();
    }
    // SAFETY: the title passes `planes` plane descriptions.
    let Some(plane) = (unsafe { plane_parameters.as_ref() }) else {
        return std::ptr::null();
    };
    if plane.count == 0 || plane.resources.is_null() {
        return std::ptr::null();
    }
    // SAFETY: the plane lists `count` resources.
    unsafe { *plane.resources }
}

/// Presents sent behind their queue's line since the start
/// ([`send_behind`]), for the line every 512 of them.
#[cfg(windows)]
static PRESENTS_BEHIND: AtomicU64 = AtomicU64::new(0);

/// The queues that hold something, with a reference held until their line
/// is drained. The releaser also pins its snapshot before dropping the lock.
#[cfg(windows)]
static WAITING: std::sync::Mutex<Vec<QueueRef>> = std::sync::Mutex::new(Vec::new());

#[cfg(windows)]
struct QueueRef(usize);

#[cfg(windows)]
impl QueueRef {
    fn new(this: This) -> Self {
        object::add_ref(this);
        Self(this as usize)
    }
}

#[cfg(windows)]
impl Clone for QueueRef {
    fn clone(&self) -> Self {
        Self::new(self.0 as This)
    }
}

#[cfg(windows)]
impl Drop for QueueRef {
    fn drop(&mut self) {
        object::release_boxed::<Queue>(self.0 as This);
    }
}

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
    // A frame shot learns that its frame's lists are being submitted.
    #[cfg(feature = "diagnostics")]
    if crate::shot::recording() {
        for &l in lists {
            if let Some(h) = crate::list::host_list(l) {
                crate::shot::submitted(h);
            }
        }
    }
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
                held.extend(waits.into_iter().map(Held::Value));
                held.push(Held::Run(taken, order));
            }
            mark_waiting(this);
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
    // A frame shot learns that its frame's lists are being submitted.
    #[cfg(feature = "diagnostics")]
    if crate::shot::recording() {
        for &l in lists {
            if let Some(h) = crate::list::host_list(l) {
                crate::shot::submitted(h);
            }
        }
    }
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
            pieces.push(Held::Run(taken, order));
            from = to;
        }
        let mut held = lock(&queue(this).held);
        held.extend(pieces);
        mark_waiting(this);
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
    loop {
        {
            let mut work = lock(&WORK.0);
            while !*work {
                work = WORK.1.wait(work).unwrap_or_else(|e| e.into_inner());
            }
            *work = false;
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
        let (user_now, kernel_now) = crate::framelog::thread_times();
        crate::framelog::add(crate::framelog::T::WorkerUs, started.elapsed().as_micros() as u64);
        crate::framelog::add(crate::framelog::T::WorkerUserUs, user_now.saturating_sub(user));
        crate::framelog::add(crate::framelog::T::WorkerKernelUs, kernel_now.saturating_sub(kernel));
    }
}

#[cfg(windows)]
thread_local! {
    /// This thread is inside [`release_held_now`].
    static RELEASING_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

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
    signals: Vec<(FenceRef, u64)>,
    /// The queues with a part here: each has [`Held::Running`] at the
    /// head of its line.
    marked: Vec<QueueRef>,
}

#[cfg(windows)]
impl Joined {
    fn has_mark(&self, queue: usize) -> bool {
        self.marked.iter().any(|owner| owner.0 == queue)
    }

    fn mark(&mut self, owner: &QueueRef, line: &mut Vec<Held>) {
        if !self.has_mark(owner.0) {
            line.insert(0, Held::Running);
            self.marked.push(owner.clone());
        }
    }

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
        if !crate::host::late_signal(fence.as_ptr(), value) {
            crate::fence::set(fence.as_ptr(), value);
        }
    }
    for owner in joined.marked.drain(..) {
        // `owner` pins this queue independently of the waiting registry. In
        // particular the mutex must outlive removal of its registry entry.
        let line = &queue(owner.0 as This).held;
        let mut held = lock(line);
        if matches!(held.first(), Some(Held::Running)) {
            held.remove(0);
        }
        held.unregister_if_drained(owner.0);
    }
}

/// Before a submission of ours that is waited for, with [`behind`]: what
/// the queues hold and is free to run runs first, on this thread, so the
/// submission keeps its place behind them.
#[cfg(windows)]
pub(crate) fn before_sync() {
    if behind() && !RELEASING_HERE.get() {
        release_held_now(false);
    }
}

#[cfg(windows)]
fn release_held_now(all: bool) {
    // One caller at a time, so a queue's submissions keep their order.
    static RELEASING: std::sync::Mutex<()> = std::sync::Mutex::new(());
    if lock(&WAITING).is_empty() {
        return;
    }
    let _one = lock(&RELEASING);
    let was_here = RELEASING_HERE.replace(true);
    struct Leave(bool);
    impl Drop for Leave {
        fn drop(&mut self) {
            RELEASING_HERE.set(self.0);
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
        for owner in queues {
            let q = owner.0;
            // SAFETY: the snapshot holds an owned reference for this visit.
            let line = unsafe { &(*(q as *const Queue)).held };
            let mut held = lock(line);
            loop {
                // A queue with a part in `joined` has our mark at the head
                // of its line: what is looked at is behind the mark.
                let base = usize::from(joined.has_mark(q));
                match held.get(base) {
                    None | Some(Held::Running) => break,
                    Some(Held::Wait(fence, value))
                        if crate::fence::completed(fence.as_ptr()) < *value
                            && !may_force_queue(all, &held) =>
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
                    Some(Held::Value(wait))
                        if !may_force_queue(all, &held) && !value_met(wait) => break,
                    // Both are boundaries for joined submissions. In
                    // particular no list after a remap can prepare its
                    // memory reads using the previous mapping.
                    Some(Held::Present(..) | Held::MapPages(..)) if !joined.is_empty() => break,
                    Some(_) => {}
                }
                match held.remove(base) {
                    Held::Wait(..) => {}
                    Held::Running => unreachable!("in-flight queue work cannot be removed here"),
                    Held::Value(wait) => {
                        // Passed at a present without being met: what the
                        // queue stood at.
                        if may_force_queue(all, &held) && !value_met(&wait) {
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
                    Held::Run(lists, order) => {
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
                            joined.mark(&owner, &mut held);
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
                            // Not waited for: the host keeps the lists
                            // until they are settled.
                            if behind() && !all {
                                // The line is open while this part is
                                // prepared and sent: only this thread takes
                                // from its head (one releaser at a time).
                                held.insert(0, Held::Running);
                                drop(held);
                                host.execute_late(lists, order);
                                held = lock(line);
                                if matches!(held.first(), Some(Held::Running)) {
                                    held.remove(0);
                                }
                            } else {
                                host.execute_late(lists, order);
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
                        if all && may_force_queue(all, &held) {
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
                    Held::MapPages(work) => {
                        held.insert(0, Held::Running);
                        drop(held);
                        // Host flights are a shared line, not split by title
                        // queue. Finish prior copy-back before changing any
                        // backing. Do not hold a queue mutex during the wait.
                        crate::host::with_page_mapping(|| {
                            work.pool.apply(work.batches.iter().map(|(at, ranges)| (*at, ranges.as_slice())));
                        });
                        held = lock(line);
                        held.remove(0);
                    }
                    Held::Present(source) => {
                        held.insert(0, Held::Running);
                        drop(held);
                        source.show();
                        held = lock(line);
                        held.remove(0);
                        PRESENTS_HELD.fetch_sub(1, Ordering::Relaxed);
                    }
                    // Behind what was sent to the card and not waited
                    // for, if anything was.
                    Held::Signal(fence, value) => {
                        if !joined.is_empty() {
                            // Behind parts not sent yet: with them.
                            joined.signals.push((fence, value));
                        } else if !crate::host::late_signal(fence.as_ptr(), value) {
                            crate::fence::set(fence.as_ptr(), value);
                        }
                    }
                }
                moved = true;
            }
            held.unregister_if_drained(q);
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

/// The legacy present-time escape hatch must not force commands through a
/// real wait in front of a pending mapping change (or the remap itself).
#[cfg(windows)]
fn may_force_queue(all: bool, line: &QueueLine) -> bool {
    all && !line.ordered
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
    object::release_boxed::<Queue>(this)
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
                let mut held = lock(&queue(this).held);
                held.extend(unmet.into_iter().map(Held::Value));
                mark_waiting(this);
            }
        }
        let kept = hold(this, || {
            let (taken, order) = crate::list::detach(lists);
            Held::Run(taken, order)
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
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::fence::pace::signal_called(fence as usize, value);
    // A queue that waits signals after what it holds.
    #[cfg(windows)]
    if hold(this, || Held::Signal(FenceRef::new(fence), value)) {
        return S_OK;
    }
    // Behind the submissions sent to the card and not waited for: the
    // fence is set when they are settled.
    #[cfg(windows)]
    if crate::host::late_signal(fence, value) {
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
                held.push(Held::Wait(FenceRef::new(fence), value));
                mark_waiting(this);
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
    let hr = present_now(this, planes, plane_parameters);
    #[cfg(windows)]
    {
        crate::framelog::add(crate::framelog::T::PresentUs, presenting.elapsed().as_micros() as u64);
        crate::framelog::frame();
        #[cfg(feature = "diagnostics")]
        crate::fence::pace::frame();
    }
    hr
}

fn present_now(this: This, planes: u32, plane_parameters: *const Plane) -> HRESULT {
    #[cfg(windows)]
    let presenting = std::time::Instant::now();
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::copystat::present_begins();
    #[cfg(all(windows, feature = "diagnostics"))]
    crate::profile::frame();

    let frame = FRAMES.fetch_add(1, Ordering::Relaxed) + 1;
    // A mapping (even one running inline) and its dependent work are a
    // required presentation boundary, independent of the async-present mode.
    #[cfg(windows)]
    let resource = present_resource(planes, plane_parameters);
    #[cfg(windows)]
    let sent_behind = send_behind(this, resource, frame, present_behind());
    #[cfg(not(windows))]
    let _ = this;
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
    // The legacy force path may skip waits on other queue segments, but
    // cannot do so on a segment that contains or has executed a remap.
    #[cfg(windows)]
    if !sent_behind {
        release_held(forces());
    }
    // Check again if a remap arrived while the ordinary path drained queues.
    #[cfg(windows)]
    let sent_behind = sent_behind || send_behind(this, resource, frame, false);
    #[cfg(windows)]
    crate::framelog::add(crate::framelog::T::PresentLineUs, presenting.elapsed().as_micros() as u64);
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
        crate::renderdoc::frame_drew(frame, crate::draw::drawn());
        crate::shot::present(frame, crate::draw::drawn());
        if crate::capture::listed(frame) {
            crate::capture::dump_targets(frame);
        }
    }
    #[cfg(windows)]
    if !sent_behind
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
#[derive(Clone, Copy)]
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

#[derive(Clone, Copy)]
struct PagePool {
    map: crate::resource::PageMapper,
    base: u64,
    pages: u32,
}

impl PagePool {
    fn apply<'a>(self, batches: impl Iterator<Item = (u64, &'a [PageRange])>) {
        let mut ok = true;
        for (mut at, ranges) in batches {
            for r in ranges {
                if r.count == 0 {
                    continue;
                }
                let len = u64::from(r.count) * PAGE;
                let Some(end) = at.checked_add(len) else {
                    ok = false;
                    break;
                };
                let source = |count: u32| {
                    let end = r.start.checked_add(count).filter(|&e| e <= self.pages)?;
                    self.base.checked_add(u64::from(end) * PAGE)?;
                    self.base.checked_add(u64::from(r.start) * PAGE).filter(|&at| at != 0)
                };
                let map = self.map;
                ok &= match r.kind {
                    0 => source(r.count).is_some_and(|p| map(at, p, len)),
                    1 => source(1).is_some_and(|p| (0..u64::from(r.count)).all(|k| map(at + k * PAGE, p, PAGE))),
                    2 => map(at, 0, len),
                    3 => true,
                    _ => false,
                };
                at = end;
            }
        }
        if !ok {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!(pool = format_args!("{:#x}", self.base), "a page mapping failed"));
        }
    }
}

#[cfg(windows)]
struct QueuedPageMappings {
    pool: PagePool,
    batches: Vec<(u64, Vec<PageRange>)>,
}

fn page_ranges(batches: &[PageBatch]) -> impl Iterator<Item = (u64, &[PageRange])> {
    batches.iter().filter_map(|b| {
        if b.count == 0 || b.ranges.is_null() {
            return None;
        }
        // SAFETY: only used during the call, which borrows the title's ranges.
        let ranges = unsafe { std::slice::from_raw_parts(b.ranges, b.count as usize) };
        Some((b.destination, ranges))
    })
}

#[cfg(windows)]
fn mark_waiting(this: This) {
    let mut waiting = lock(&WAITING);
    if !waiting.iter().any(|q| q.0 == this as usize) {
        waiting.push(QueueRef::new(this));
    }
}

/// `CopyPageMappingsBatchX(count, batches, pool, pool pages, flags)`: maps
/// pages of a pool (memory the title mapped physical pages into) at the
/// batches' addresses. This is a queue operation: earlier work, including
/// copy-back into title addresses, must finish before the addresses change.
extern "system" fn copy_page_mappings(
    this: This,
    count: u32,
    batches: *const PageBatch,
    pool: u64,
    pool_pages: u32,
    _flags: u32,
) -> HRESULT {
    let Some(map) = crate::resource::page_mapper() else {
        return E_INVALIDARG;
    };
    if count == 0 {
        return S_OK;
    }
    if batches.is_null() {
        return E_INVALIDARG;
    }
    // SAFETY: the title passes `count` batches, valid for this call.
    let batches = unsafe { std::slice::from_raw_parts(batches, count as usize) };
    let pool = PagePool { map, base: pool, pages: pool_pages };
    #[cfg(windows)]
    if !this.is_null() {
        let line = &queue(this).held;
        let mut held = lock(line);
        held.ordered = true;
        if !held.is_empty() {
            // The caller's batch and range arrays need not outlive this call.
            held.push(Held::MapPages(QueuedPageMappings {
                pool,
                batches: page_ranges(batches).map(|(at, ranges)| (at, ranges.to_vec())).collect(),
            }));
            mark_waiting(this);
            drop(held);
            release_held(false);
            return S_OK;
        }
        // No older logical work is held. Run inline without copying the
        // descriptions, but don't let a new submission or signal pass us.
        held.push(Held::Running);
        drop(held);
        crate::host::with_page_mapping(|| pool.apply(page_ranges(batches)));
        let mut held = lock(line);
        held.remove(0);
        let more = !held.is_empty();
        if more {
            mark_waiting(this);
        } else {
            held.unregister_if_drained(this as usize);
        }
        drop(held);
        if more {
            release_held(false);
        }
        return S_OK;
    }
    let _ = this;
    pool.apply(page_ranges(batches));
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
    fn new_test_queue() -> QueueRef {
        let null = std::ptr::null();
        let mut q = null;
        let desc = QueueDesc { kind: 0, priority: 0, flags: 0, nodes: 0 };
        assert_eq!(create(null, &desc, &QUEUE_IIDS[0], &mut q), S_OK);
        QueueRef(q as usize)
    }

    #[cfg(windows)]
    fn new_test_fence() -> (This, FenceRef) {
        let null = std::ptr::null();
        let mut fence = null;
        assert_eq!(crate::fence::create(null, 0, 0, &crate::fence::FENCE_IIDS[0], &mut fence), S_OK);
        let owner = FenceRef::new(fence);
        let release = slot::<extern "system" fn(This) -> u32>(fence, 0x10);
        assert_eq!(release(fence), 1);
        (fence, owner)
    }

    #[cfg(windows)]
    #[test]
    fn remap_orders_presents_even_without_async_present() {
        // Real emulated queue waits and dispatch loop; an observer in place
        // of the window copy records when it is allowed to run.
        static REACHED: AtomicU64 = AtomicU64::new(0);
        fn map(to: u64, from: u64, len: u64) -> bool {
            assert_eq!((to, from, len), (2 * PAGE, PAGE, PAGE));
            assert_eq!(REACHED.swap(1, Ordering::SeqCst), 0);
            true
        }
        let owner = new_test_queue();
        let q = owner.0 as This;
        let (before, _before_owner) = new_test_fence();
        let (after, _after_owner) = new_test_fence();
        let (producer, _producer_owner) = new_test_fence();
        {
            let mut line = lock(&queue(q).held);
            line.ordered = true;
            line.push(Held::Wait(FenceRef::new(before), 1));
            line.push(Held::MapPages(QueuedPageMappings {
                pool: PagePool { map, base: PAGE, pages: 1 },
                batches: vec![(2 * PAGE, vec![PageRange { kind: 0, start: 0, count: 1 }])],
            }));
            line.push(Held::Wait(FenceRef::new(after), 1));
            line.push(Held::Signal(FenceRef::new(producer), 1));
            mark_waiting(q);
        }
        release_held_now(true);
        assert_eq!(REACHED.load(Ordering::SeqCst), 0, "the first wait must not be forced");
        crate::fence::set(before, 1);
        release_held_now(true);
        assert_eq!(REACHED.load(Ordering::SeqCst), 1, "only the remap has run");
        let producer = FenceRef::new(producer);
        // The first present arrives after the remap, while the dependent
        // work is still blocked. A queued map marker alone is not enough.
        assert!(enqueue_present(q, false, || Some(PresentSource::Observe(Box::new(move || {
            assert_eq!(crate::fence::completed(producer.as_ptr()), 1);
            assert_eq!(REACHED.fetch_add(1, Ordering::SeqCst), 1);
        })))));
        assert!(enqueue_present(q, false, || Some(PresentSource::Observe(Box::new(|| {
            assert_eq!(REACHED.fetch_add(1, Ordering::SeqCst), 2);
        })))));
        release_held_now(true);
        assert_eq!(REACHED.load(Ordering::SeqCst), 1, "neither present can force the second wait");
        crate::fence::set(after, 1);
        release_held_now(true);
        assert_eq!(REACHED.load(Ordering::SeqCst), 3);
        assert!(lock(&queue(q).held).is_empty());
        assert!(!lock(&queue(q).held).ordered);
    }

    #[cfg(windows)]
    #[test]
    fn present_cannot_pass_an_inline_mapping_marker() {
        let owner = new_test_queue();
        let q = owner.0 as This;
        let order = std::sync::Arc::new(AtomicU64::new(0));
        {
            let mut line = lock(&queue(q).held);
            line.ordered = true;
            line.push(Held::Running);
        }
        let observed = order.clone();
        assert!(enqueue_present(q, false, || Some(PresentSource::Observe(Box::new(move || {
            assert_eq!(observed.swap(2, Ordering::SeqCst), 1);
        })))));
        release_held_now(true);
        assert_eq!(order.load(Ordering::SeqCst), 0);
        {
            let mut line = lock(&queue(q).held);
            assert!(matches!(line.remove(0), Held::Running));
            order.store(1, Ordering::SeqCst);
        }
        release_held_now(false);
        assert_eq!(order.load(Ordering::SeqCst), 2);
    }

    #[cfg(windows)]
    #[test]
    fn joined_cleanup_pins_queue_after_client_and_snapshot_release() {
        let client = new_test_queue();
        let q = client.0 as This;
        let mut joined = Joined::default();
        {
            let snapshot = QueueRef::new(q);
            let mut line = lock(&queue(q).held);
            mark_waiting(q);
            joined.mark(&snapshot, &mut line);
            joined.mark(&snapshot, &mut line);
            assert_eq!(line.len(), 1, "one mark and owner per queue");
        }
        std::mem::forget(client);
        // No test/client/snapshot reference is left when cleanup runs;
        // WAITING and Joined must each have one and no more.
        assert_eq!(release(q), 2);
        send_joined(&mut joined);
        assert!(!lock(&WAITING).iter().any(|w| w.0 == q as usize));
        assert!(joined.marked.is_empty());
    }

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
