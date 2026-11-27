use std::collections::VecDeque;
use std::fmt::Write as _;
use std::io::Write as _;
use std::sync::Mutex;
use std::time::Instant;

const RING: usize = 4096;
const PENDING: usize = 64;
const ROWS: usize = 1 << 16;

pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = matches!(std::env::var("VICTORY_PACING").as_deref(), Ok("1"));
        if on {
            tracing::info!(
                "frame pacing is recorded (VICTORY_PACING=1): the title's Present, the host present and completion, a line each frame-rate period"
            );
        }
        on
    })
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Sample {
    frame: u64,
    title: Option<f64>,
    host: Option<f64>,
    complete: Option<f64>,
    display: Option<(u32, f64)>,
    depth: u64,
    dropped: u64,
    behind: bool,
}

pub(crate) struct Ring<T> {
    items: VecDeque<T>,
    cap: usize,
    pushed: u64,
}

impl<T> Ring<T> {
    pub(crate) fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        Ring {
            items: VecDeque::with_capacity(cap),
            cap,
            pushed: 0,
        }
    }
    pub(crate) fn push(&mut self, item: T) {
        if self.items.len() == self.cap {
            self.items.pop_front();
        }
        self.items.push_back(item);
        self.pushed += 1;
    }
    pub(crate) fn pushed(&self) -> u64 {
        self.pushed
    }
    pub(crate) fn since(&self, pushed: u64, before: usize) -> impl Iterator<Item = &T> {
        let new = (self.pushed.saturating_sub(pushed)).min(self.items.len() as u64) as usize;
        let take = (new + before).min(self.items.len());
        self.items.iter().skip(self.items.len() - take)
    }
}

pub(crate) fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let n = sorted.len();
    let rank = ((p / 100.0) * n as f64).ceil() as usize;
    Some(sorted[rank.clamp(1, n) - 1])
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Spread {
    count: usize,
    p50: f64,
    p95: f64,
    p99: f64,
    max: f64,
}

pub(crate) fn spread(mut values: Vec<f64>) -> Spread {
    values.retain(|v| v.is_finite());
    values.sort_by(f64::total_cmp);
    let at = |p| percentile(&values, p).unwrap_or(0.0);
    Spread {
        count: values.len(),
        p50: at(50.0),
        p95: at(95.0),
        p99: at(99.0),
        max: values.last().copied().unwrap_or(0.0),
    }
}

impl std::fmt::Display for Spread {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.count == 0 {
            return write!(f, "none");
        }
        write!(
            f,
            "{} p50 {:.1} p95 {:.1} p99 {:.1} max {:.1}",
            self.count, self.p50, self.p95, self.p99, self.max
        )
    }
}

fn intervals(times: impl Iterator<Item = f64>) -> Vec<f64> {
    let mut last = None;
    let mut out = Vec::new();
    for t in times {
        if let Some(l) = last {
            out.push(t - l);
        }
        last = Some(t);
    }
    out
}

fn display_intervals(points: impl Iterator<Item = (u32, f64)>) -> Vec<f64> {
    let mut last: Option<(u32, f64)> = None;
    let mut out = Vec::new();
    for (count, t) in points {
        match last {
            Some((c, _)) if count == c => continue,
            Some((c, l)) if count > c && t >= l => {
                let n = count - c;
                out.extend(std::iter::repeat_n((t - l) / f64::from(n), n as usize));
            }
            _ => {}
        }
        last = Some((count, t));
    }
    out
}

pub(crate) fn summary(samples: &[Sample], new: usize) -> String {
    let start = samples.len().saturating_sub(new);
    let fresh = &samples[start..];
    let title: Vec<f64> = samples
        .windows(2)
        .skip(start.saturating_sub(1))
        .filter(|pair| pair[1].frame == pair[0].frame + 1)
        .filter_map(|pair| Some(pair[1].title? - pair[0].title?))
        .collect();
    let host = intervals(samples.iter().filter_map(|s| s.host));
    let host = host[host.len().saturating_sub(fresh.len())..].to_vec();
    let display = display_intervals(samples.iter().filter_map(|s| s.display));
    let latency: Vec<f64> = fresh
        .iter()
        .filter_map(|s| Some(s.complete? - s.title?))
        .collect();
    let n = fresh.len().max(1) as f64;
    let depth = fresh.iter().map(|s| s.depth as f64).sum::<f64>() / n;
    let dropped = fresh.iter().map(|s| s.dropped as f64).sum::<f64>() / n;
    let behind = fresh.iter().filter(|s| s.behind).count() as f64 * 100.0 / n;
    let missing = fresh
        .iter()
        .filter(|s| s.host.is_none() || s.complete.is_none())
        .count();
    let mut line = String::new();
    let _ = write!(
        line,
        "frame pacing over {} frames (VICTORY_PACING, ms): title Present interval {}; host present interval {}; \
         title Present to completion {}; display interval {} (frame statistics); queue depth {depth:.1} mean; \
         dropped {dropped:.1}/frame; async {behind:.0}%; {missing} without a host present or completion",
        fresh.len(),
        spread(title),
        spread(host),
        spread(latency),
        spread(display)
    );
    line
}

struct State {
    epoch: Instant,
    qpc: Option<(i64, f64)>,
    pending: VecDeque<(Sample, u64)>,
    ring: Ring<Sample>,
    logged: u64,
    seen_dropped: u64,
    finished: u64,
    rows: Vec<Sample>,
    stats_failed: u64,
}

static STATE: Mutex<Option<State>> = Mutex::new(None);
static CSV: Mutex<Option<std::io::BufWriter<std::fs::File>>> = Mutex::new(None);

fn qpc_now() -> Option<(i64, f64)> {
    let (mut counter, mut frequency) = (0i64, 0i64);
    unsafe {
        windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut counter);
        windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut frequency);
    }
    (frequency > 0).then_some((counter, frequency as f64))
}

fn csv_open() {
    let Ok(path) = std::env::var("VICTORY_PACING_CSV") else {
        return;
    };
    if path.is_empty() {
        return;
    }
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path);
    match file {
        Ok(file) => {
            let empty = file.metadata().map(|m| m.len() == 0).unwrap_or(true);
            let mut writer = std::io::BufWriter::new(file);
            if empty {
                let _ = writeln!(
                    writer,
                    "frame,t_title_present_ms,t_host_present_ms,t_complete_ms,queue_depth,dropped,async,t_display_ms"
                );
            }
            tracing::info!(
                path,
                "frame pacing rows go to a CSV file (VICTORY_PACING_CSV)"
            );
            *CSV.lock().unwrap_or_else(|e| e.into_inner()) = Some(writer);
        }
        Err(e) => tracing::warn!(path, error = %e, "the frame pacing CSV file could not be opened"),
    }
}

fn with<R>(f: impl FnOnce(&mut State) -> R) -> R {
    let mut state = STATE.lock().unwrap_or_else(|e| e.into_inner());
    let state = state.get_or_insert_with(|| {
        csv_open();
        State {
            epoch: Instant::now(),
            qpc: qpc_now(),
            pending: VecDeque::with_capacity(PENDING + 1),
            ring: Ring::new(RING),
            logged: 0,
            seen_dropped: crate::draw::dropped_total(),
            finished: 0,
            rows: Vec::new(),
            stats_failed: 0,
        }
    });
    f(state)
}

impl State {
    fn ms(&self, at: Instant) -> f64 {
        at.saturating_duration_since(self.epoch).as_secs_f64() * 1e3
    }
    fn entry(&mut self, frame: u64) -> usize {
        if let Some(i) = self.pending.iter().position(|(s, _)| s.frame == frame) {
            return i;
        }
        if self.pending.len() >= PENDING {
            self.retire(1);
        }
        let at = self
            .pending
            .iter()
            .position(|(s, _)| s.frame > frame)
            .unwrap_or(self.pending.len());
        self.pending.insert(
            at,
            (
                Sample {
                    frame,
                    ..Sample::default()
                },
                0,
            ),
        );
        at
    }
    fn retire(&mut self, n: usize) {
        for _ in 0..n.min(self.pending.len()) {
            if let Some((sample, _)) = self.pending.pop_front() {
                self.ring.push(sample);
                if self.rows.len() < ROWS {
                    self.rows.push(sample);
                }
            }
        }
    }
    fn settle(&mut self) {
        if let Some(last) = self.pending.iter().rposition(|(s, _)| s.complete.is_some()) {
            self.retire(last + 1);
        }
    }
}

pub(crate) fn title(frame: u64, at: Instant, behind: bool, depth: u64) {
    let total = crate::draw::dropped_total();
    with(|state| {
        let dropped = total.saturating_sub(state.seen_dropped);
        state.seen_dropped = state.seen_dropped.max(total);
        let title = state.ms(at);
        let i = state.entry(frame);
        let sample = &mut state.pending[i].0;
        sample.title = Some(title);
        sample.behind = behind;
        sample.depth = depth;
        sample.dropped = dropped;
        state.settle();
    });
}

pub(crate) fn shown(frame: u64, flight: u64, stats: Option<(u32, i64)>) {
    let now = Instant::now();
    with(|state| {
        let host = state.ms(now);
        let display = match (stats, state.qpc) {
            (Some((count, sync)), Some((counter, frequency))) if sync != 0 && count != 0 => {
                Some((count, (sync - counter) as f64 * 1e3 / frequency))
            }
            _ => {
                state.stats_failed += 1;
                None
            }
        };
        let done = flight == 0 || flight <= state.finished;
        let i = state.entry(frame);
        let (sample, waits) = &mut state.pending[i];
        sample.host = Some(host);
        sample.display = display;
        *waits = flight;
        if done {
            sample.complete = Some(host);
        }
        state.settle();
    });
}

pub(crate) fn finished(last: u64) {
    if !on() {
        return;
    }
    let now = Instant::now();
    with(|state| {
        state.finished = state.finished.max(last);
        let ms = state.ms(now);
        let mut any = false;
        for (sample, waits) in state.pending.iter_mut() {
            if *waits != 0 && *waits <= last && sample.complete.is_none() {
                sample.complete = Some(ms);
                any = true;
            }
        }
        if any {
            state.settle();
        }
    });
}

pub(crate) fn log() {
    if !on() {
        return;
    }
    let (line, rows, failed) = with(|state| {
        let new = state.ring.pushed() - state.logged;
        let samples: Vec<Sample> = state.ring.since(state.logged, 1).copied().collect();
        let new = (new as usize).min(samples.len());
        state.logged = state.ring.pushed();
        let line = summary(&samples, new);
        (
            line,
            std::mem::take(&mut state.rows),
            std::mem::take(&mut state.stats_failed),
        )
    });
    tracing::info!("{line}; {failed} presents without frame statistics");
    let mut csv = CSV.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(writer) = csv.as_mut() {
        let cell = |v: Option<f64>| v.map(|v| format!("{v:.3}")).unwrap_or_default();
        for s in &rows {
            let _ = writeln!(
                writer,
                "{},{},{},{},{},{},{},{}",
                s.frame,
                cell(s.title),
                cell(s.host),
                cell(s.complete),
                s.depth,
                s.dropped,
                u8::from(s.behind),
                cell(s.display.map(|(_, t)| t))
            );
        }
        let _ = writer.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_nearest_rank() {
        let v: Vec<f64> = (1..=100).map(f64::from).collect();
        assert_eq!(percentile(&v, 50.0), Some(50.0));
        assert_eq!(percentile(&v, 95.0), Some(95.0));
        assert_eq!(percentile(&v, 99.0), Some(99.0));
        assert_eq!(percentile(&v, 100.0), Some(100.0));
        assert_eq!(percentile(&v, 0.0), Some(1.0));
        assert_eq!(percentile(&[], 50.0), None);
        assert_eq!(percentile(&[7.0], 99.0), Some(7.0));
    }

    #[test]
    fn spread_sorts_and_skips_non_finite() {
        let s = spread(vec![5.0, 1.0, f64::NAN, 3.0, 2.0, 4.0, f64::INFINITY]);
        assert_eq!(s.count, 5);
        assert_eq!(s.p50, 3.0);
        assert_eq!(s.p95, 5.0);
        assert_eq!(s.max, 5.0);
        assert_eq!(spread(Vec::new()).count, 0);
        assert_eq!(spread(Vec::new()).to_string(), "none");
    }

    #[test]
    fn ring_keeps_the_newest() {
        let mut ring = Ring::new(4);
        for i in 0..10 {
            ring.push(i);
        }
        assert_eq!(ring.items.len(), 4);
        assert_eq!(ring.pushed(), 10);
        assert_eq!(
            ring.since(0, 0).copied().collect::<Vec<_>>(),
            vec![6, 7, 8, 9]
        );
        assert_eq!(ring.since(8, 0).copied().collect::<Vec<_>>(), vec![8, 9]);
        assert_eq!(ring.since(8, 1).copied().collect::<Vec<_>>(), vec![7, 8, 9]);
        assert_eq!(ring.since(10, 0).count(), 0);
        assert_eq!(ring.since(10, 1).copied().collect::<Vec<_>>(), vec![9]);
        assert_eq!(Ring::<u8>::new(0).cap, 1);
    }

    #[test]
    fn intervals_and_display() {
        assert_eq!(intervals([0.0, 10.0, 25.0].into_iter()), vec![10.0, 15.0]);
        assert!(intervals(std::iter::once(1.0)).is_empty());
        let d = display_intervals([(1, 0.0), (1, 0.0), (2, 16.0), (4, 48.0)].into_iter());
        assert_eq!(d, vec![16.0, 16.0, 16.0]);
    }

    #[test]
    fn summary_counts_the_new_frames() {
        let samples: Vec<Sample> = (0..5)
            .map(|i| Sample {
                frame: i + 1,
                title: Some(i as f64 * 10.0),
                host: Some(i as f64 * 10.0 + 2.0),
                complete: Some(i as f64 * 10.0 + 30.0),
                display: None,
                depth: 2,
                dropped: 1,
                behind: i % 2 == 0,
            })
            .collect();
        let line = summary(&samples, 4);
        assert!(line.starts_with("frame pacing over 4 frames"), "{line}");
        assert!(line.contains("title Present interval 4 p50 10.0"), "{line}");
        assert!(
            line.contains("title Present to completion 4 p50 30.0"),
            "{line}"
        );
        assert!(line.contains("queue depth 2.0 mean"), "{line}");
        assert!(line.contains("async 50%"), "{line}");
    }
}
