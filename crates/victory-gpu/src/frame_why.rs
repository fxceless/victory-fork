//! With `VICTORY_FRAME_WHY2=1` (off by default), every five seconds show
//! where each thread spent the time between present calls. A phase still open
//! at a present is split there, even a worker asleep for many frames.
use std::cell::Cell;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard};
use std::time::{Duration, Instant};

static ON: LazyLock<bool> =
    LazyLock::new(|| std::env::var_os("VICTORY_FRAME_WHY2").is_some_and(|v| v == "1"));

#[derive(Clone, Copy)]
pub(crate) enum Worker {
    Submit,
    Relay,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Phase {
    Present,
    Record,
    Submit,
    Pace,
    Prepare,
    WaitRelay,
    WaitFence,
    Idle,
    Land,
    Other,
}
use Phase::*;
type Times = [u64; Other as usize + 1];
const TITLE: &[(Phase, &str)] = &[
    (Present, "present"),
    (Record, "record"),
    (Submit, "submit"),
    (Pace, "pace"),
];
const SUBMIT: &[(Phase, &str)] = &[
    (Prepare, "prepare"),
    (WaitRelay, "relay wait"),
    (WaitFence, "fence wait"),
    (Idle, "idle"),
];
const RELAY: &[(Phase, &str)] = &[(WaitFence, "fence wait"), (Land, "land"), (Idle, "idle")];

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

struct Track {
    worker: Option<Worker>,
    phase: Phase,
    since: Instant,
    ns: Times,
}

impl Track {
    fn new(worker: Option<Worker>, now: Instant) -> Self {
        Self {
            worker,
            phase: Other,
            since: now,
            ns: Times::default(),
        }
    }

    fn charge(&mut self, now: Instant) {
        if self.phase != Other {
            self.ns[self.phase as usize] += (now - self.since).as_nanos() as u64;
        }
        self.since = now;
    }

    fn switch(&mut self, phase: Phase, now: Instant) -> Phase {
        self.charge(now);
        std::mem::replace(&mut self.phase, phase)
    }

    fn sample(&mut self, now: Instant) -> Times {
        self.charge(now);
        std::mem::take(&mut self.ns)
    }
}

static TRACKS: Mutex<Vec<Arc<Mutex<Track>>>> = Mutex::new(Vec::new());
thread_local! {
    static WORKER: Cell<Option<Worker>> = const { Cell::new(None) };
    static TRACK: Arc<Mutex<Track>> = {
        let t = Arc::new(Mutex::new(Track::new(WORKER.get(), Instant::now())));
        lock(&TRACKS).push(t.clone());
        t
    };
}

/// Called at the worker's entry, before its first phase.
pub(crate) fn worker(w: Worker) {
    if *ON {
        WORKER.set(Some(w));
    }
}

pub(crate) struct Span(Option<Phase>);

impl Drop for Span {
    fn drop(&mut self) {
        if let Some(previous) = self.0 {
            TRACK.with(|t| lock(t).switch(previous, Instant::now()));
        }
    }
}

pub(crate) fn span(phase: Phase) -> Span {
    if !*ON {
        return Span(None);
    }
    // Preparation may also run on a title thread, inside its present or
    // submit call. Charge the whole call there, and the worker phases only
    // on the workers whose work they describe.
    let phases = match WORKER.get() {
        None => TITLE,
        Some(Worker::Submit) => SUBMIT,
        Some(Worker::Relay) => RELAY,
    };
    if !phases.iter().any(|&(p, _)| p == phase) {
        return Span(None);
    }
    TRACK.with(|t| {
        let mut t = lock(t);
        if t.phase == Present {
            return Span(None);
        }
        Span(Some(t.switch(phase, Instant::now())))
    })
}

#[derive(Default)]
struct Reading {
    wall: u64,
    // Presenting caller, submit worker, relay worker, all other callers.
    lanes: [Times; 4],
    workers: [bool; 2],
}

impl Reading {
    fn finish(&mut self, wall: Duration) {
        self.wall = wall.as_nanos() as u64;
        for lane in &mut self.lanes[..3] {
            // Also covers the time before the worker started. Other callers
            // can overlap one another and have no single frame of wall time.
            lane[Other as usize] = self.wall.saturating_sub(lane.iter().sum::<u64>());
        }
    }

    fn add(&mut self, frame: &Self) {
        self.wall += frame.wall;
        for (sum, lane) in self.lanes.iter_mut().zip(&frame.lanes) {
            for (out, ns) in sum.iter_mut().zip(lane) {
                *out += ns;
            }
        }
        for (seen, active) in self.workers.iter_mut().zip(frame.workers) {
            *seen |= active;
        }
    }

    fn log(&self, what: &str, frames: u64) {
        let ms = |ns: u64| ns as f64 / frames as f64 / 1e6;
        let fields = |lane: usize, phases: &[(Phase, &str)]| {
            if (lane == 1 || lane == 2) && !self.workers[lane - 1] {
                return "no worker observed".to_owned();
            }
            let mut fields: Vec<_> = phases
                .iter()
                .map(|&(p, name)| format!("{name} {:.1}", ms(self.lanes[lane][p as usize])))
                .collect();
            if lane < 3 {
                fields.push(format!("other {:.1}", ms(self.lanes[lane][Other as usize])));
            }
            fields.join(", ")
        };
        tracing::info!(
            "why2 {what} (frames={frames}, ms): wall {:.1} | presenter: {} | submit worker: {} | relay: {} | other callers (thread-ms): {}",
            ms(self.wall),
            fields(0, TITLE),
            fields(1, SUBMIT),
            fields(2, RELAY),
            fields(3, TITLE)
        );
    }
}

struct Window {
    first: Instant,
    last: Instant,
    frames: u64,
    sum: Reading,
    worst: Reading,
}

impl Window {
    fn new(now: Instant) -> Self {
        Self {
            first: now,
            last: now,
            frames: 0,
            sum: Reading::default(),
            worst: Reading::default(),
        }
    }
}

/// Sample all the threads at the same instant, including callers that
/// left since the last present. Keep their last reading before dropping them.
fn sample(presenter: &Arc<Mutex<Track>>) -> (Instant, Reading) {
    let mut frame = Reading::default();
    let mut tracks = lock(&TRACKS);
    let mut guards: Vec<_> = tracks.iter().map(|t| lock(t)).collect();
    let now = Instant::now();
    for (track, t) in tracks.iter().zip(&mut guards) {
        let lane = match t.worker {
            Some(Worker::Submit) => 1,
            Some(Worker::Relay) => 2,
            None if Arc::ptr_eq(track, presenter) => 0,
            None => 3,
        };
        if t.worker.is_some() {
            frame.workers[lane - 1] = true;
        }
        for (out, ns) in frame.lanes[lane].iter_mut().zip(t.sample(now)) {
            *out += ns;
        }
    }
    drop(guards);
    // A caller can finish a last call and exit after we release the guards.
    // Keep any time it recorded then for the next frame.
    tracks.retain(|t| Arc::strong_count(t) > 1 || lock(t).ns != Times::default());
    (now, frame)
}

/// One interval per title present call, including a present queued for later.
pub(crate) fn frame() {
    static WINDOW: Mutex<Option<Window>> = Mutex::new(None);
    if !*ON {
        return;
    }
    let presenter = TRACK.with(Arc::clone);
    let mut window = lock(&WINDOW);
    let (now, mut frame) = sample(&presenter);
    let Some(w) = window.as_mut() else {
        *window = Some(Window::new(now));
        return;
    };
    frame.finish(now - w.last);
    w.last = now;
    w.frames += 1;
    w.sum.add(&frame);
    if frame.wall >= w.worst.wall {
        w.worst = frame;
    }
    if now - w.first >= Duration::from_secs(5) {
        let report = std::mem::replace(w, Window::new(now));
        drop(window);
        report.sum.log("mean", report.frames);
        report.worst.log("longest", 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_fence_wait_is_split_at_present_and_taken_out_of_its_parent() {
        let start = Instant::now();
        let at = |n| start + Duration::from_nanos(n);
        let mut relay = Track::new(Some(Worker::Relay), start);
        relay.switch(Land, at(1));
        let parent = relay.switch(WaitFence, at(4));
        let mut frame = Reading::default();
        frame.lanes[2] = relay.sample(at(7));
        frame.finish(at(7) - start);
        let lane = frame.lanes[2];
        assert_eq!(
            (
                lane[Land as usize],
                lane[WaitFence as usize],
                lane[Other as usize]
            ),
            (3, 3, 1)
        );
        relay.switch(parent, at(9));
        let next = relay.sample(at(10));
        assert_eq!((next[Land as usize], next[WaitFence as usize]), (1, 2));
    }

    #[test]
    fn worker_times_each_cover_the_frame_and_other_callers_can_exceed_it() {
        let mut frame = Reading::default();
        frame.lanes[0][Present as usize] = 8;
        frame.lanes[1][Prepare as usize] = 4;
        frame.lanes[2][WaitFence as usize] = 6;
        frame.lanes[3][Record as usize] = 22;
        frame.finish(Duration::from_nanos(10));
        assert_eq!(frame.lanes.map(|t| t[Other as usize]), [2, 6, 4, 0]);
        let mut window = Reading::default();
        window.add(&frame);
        window.add(&frame);
        assert_eq!(window.wall, 20);
        assert_eq!(
            window.lanes.map(|t| t.iter().sum::<u64>()),
            [20, 20, 20, 44]
        );
    }
}
