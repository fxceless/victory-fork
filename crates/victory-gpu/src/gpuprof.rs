//! GPU time per pipeline, in release builds (`VICTORY_GPU_PROFILE=1`).
//!
//! Each host list writes GPU timestamps as it runs: one when it starts, one
//! before the first of a run of draws or dispatches of one pipeline, one
//! after each of them, and one at its close, where it resolves them into a
//! read-back buffer. A submission notes, per list, the fence value that
//! says it has run; the present collects the lists whose value the fence
//! has reached (a few frames later, never waiting for the card) and adds
//! their times to sums by pipeline, named by its shaders' stages and
//! hashes. Every ~10 s the 25 pipelines with the most GPU time per frame
//! are logged, with their calls and time per call, and the share of the
//! timed lists' time they and all draws cover.
//!
//! Lists are not run on their own (unlike `VICTORY_GPU_TIMING`): times of
//! commands the card overlaps are as the card ran them. A run of draws of
//! one pipeline also holds what was recorded between them (barriers,
//! uploads of expanded vertices); what lies between runs (copies, clears,
//! state moves) counts to the lists' time only.
//!
//! Bounded throughout: the query indices come from one heap of
//! [`CHUNKS`] chunks of [`CHUNK`]; a list that finds none free stops
//! timing (counted), and at most [`PENDING_MOST`] lists wait to be read.
//! Off, nothing is made and each draw pays one test of a flag.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// Whether GPU time is profiled per pipeline (`VICTORY_GPU_PROFILE=1`).
pub(crate) fn on() -> bool {
    mode() != 0
}

/// `VICTORY_GPU_PROFILE`: 0 off, 1 a timestamp after every draw or dispatch, 2 ("coarse") one
/// timestamp per run only, where the next run starts or the list closes (what lies between
/// runs then counts to the run before). A timestamp after every one of ~4,800 draws a frame
/// may cost the card time of its own (a drain of the pipeline on some GPUs): the same scene
/// under 1 and 2 shows how much.
fn mode() -> u8 {
    static MODE: OnceLock<u8> = OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_GPU_PROFILE").ok().as_deref().map(str::trim) {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    })
}

fn coarse() -> bool {
    #[cfg(test)]
    if let Some(c) = TEST_COARSE.get() {
        return c;
    }
    mode() == 2
}

#[cfg(test)]
thread_local! {
    static TEST_COARSE: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// Timestamps a list takes at a time.
pub(crate) const CHUNK: u32 = 32;
/// Chunks in the heap: 262144 timestamps, a read-back buffer of 2 MiB
/// (a list holds at least one from its reset until its submissions are
/// read: some 600 lists a frame, two or three frames in the air).
pub(crate) const CHUNKS: u32 = 8192;
/// Lists submitted and not yet read, at most (the oldest go first).
const PENDING_MOST: usize = 8192;
/// Seconds between reports.
const REPORT_SECS: u64 = 10;
/// Pipelines a report names.
const TOP: usize = 25;

/// `VICTORY_GPU_PROFILE_WATCH=<hashes>` (separated by `,`, `;` or spaces): pipelines whose
/// name holds one of them get a line of each report even when they are not in its top
/// [`TOP`], and a hash no pipeline of the report ran gets a line saying so. A pipeline
/// that drops out of the top can have become cheap (its draws launch nothing) or not be
/// drawn at all: its calls tell which.
fn watched() -> &'static [String] {
    static WATCH: OnceLock<Vec<String>> = OnceLock::new();
    WATCH.get_or_init(|| {
        std::env::var("VICTORY_GPU_PROFILE_WATCH")
            .unwrap_or_default()
            .split([',', ';', ' '])
            .map(|h| h.trim().to_ascii_lowercase())
            .filter(|h| !h.is_empty())
            .collect()
    })
}

/// The free chunks of the heap.
pub(crate) struct Pool {
    free: Mutex<Vec<u32>>,
}

impl Pool {
    pub(crate) fn new(chunks: u32) -> Pool {
        Pool { free: Mutex::new((0..chunks).rev().collect()) }
    }

    fn take(&'static self) -> Option<Arc<Lease>> {
        let chunk = self.free.lock().unwrap_or_else(|e| e.into_inner()).pop()?;
        Some(Arc::new(Lease { chunk, pool: self }))
    }

    #[cfg(test)]
    fn free(&self) -> usize {
        self.free.lock().unwrap().len()
    }
}

/// A chunk a list writes into. It goes back to the pool when the list let
/// go of it (reset: its runs are over) and every submission of it was read.
pub(crate) struct Lease {
    chunk: u32,
    pool: &'static Pool,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.pool.free.lock().unwrap_or_else(|e| e.into_inner()).push(self.chunk);
    }
}

/// A run of draws or dispatches of one pipeline: the timestamps before
/// the first and after the last (heap indices), and how many.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Segment {
    name: u32,
    begin: u32,
    end: u32,
    calls: u32,
}

/// Pipeline names (their shaders), by id; never let go (a few thousand).
static NAMES: Mutex<(Vec<Arc<str>>, Option<HashMap<Arc<str>, u32>>)> = Mutex::new((Vec::new(), None));

fn intern(name: &str) -> u32 {
    let mut names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    let (list, map) = &mut *names;
    let map = map.get_or_insert_with(HashMap::new);
    if let Some(&id) = map.get(name) {
        return id;
    }
    let id = list.len() as u32;
    let name: Arc<str> = Arc::from(name);
    list.push(name.clone());
    map.insert(name, id);
    id
}

fn name_of(id: u32) -> Arc<str> {
    let names = NAMES.lock().unwrap_or_else(|e| e.into_inner());
    names.0.get(id as usize).cloned().unwrap_or_else(|| Arc::from("?"))
}

/// What one host list's recording timed, as it records.
#[derive(Default)]
pub(crate) struct Recording {
    leases: Vec<Arc<Lease>>,
    /// The next index of the last lease, and its end.
    next: u32,
    limit: u32,
    begin: Option<u32>,
    end: Option<u32>,
    /// The pipeline bound last: its host object and name id.
    bound: Option<(usize, u32)>,
    by_key: HashMap<usize, u32>,
    segments: Vec<Segment>,
    /// The last segment may take the next draw (nothing timed since).
    open: bool,
    /// No index was free: nothing more is timed.
    full: bool,
    /// Coarse (`VICTORY_GPU_PROFILE=2`): the last segment's end is the next timestamp taken.
    pending: bool,
}

/// Lists that found no free chunk, since the last report.
static STARVED: AtomicU64 = AtomicU64::new(0);

impl Recording {
    fn take(&mut self, pool: &'static Pool) -> Option<u32> {
        if self.full {
            return None;
        }
        if self.next == self.limit {
            let Some(lease) = pool.take() else {
                self.full = true;
                STARVED.fetch_add(1, Ordering::Relaxed);
                return None;
            };
            self.next = lease.chunk * CHUNK;
            self.limit = self.next + CHUNK;
            self.leases.push(lease);
        }
        self.next += 1;
        Some(self.next - 1)
    }

    /// Lets go of what the last recording timed (the list was reset).
    pub(crate) fn clear(&mut self) {
        *self = Recording::default();
    }

    /// The list opened: the index of its first timestamp.
    pub(crate) fn start(&mut self, pool: &'static Pool) -> Option<u32> {
        self.clear();
        self.begin = self.take(pool);
        self.begin
    }

    /// The pipeline the next draws run: its host object and its name.
    pub(crate) fn bound(&mut self, key: usize, name: &str) {
        if self.bound.is_some_and(|(k, _)| k == key) {
            return;
        }
        let id = *self.by_key.entry(key).or_insert_with(|| intern(name));
        self.bound = Some((key, id));
    }

    /// Before a draw or dispatch: the index of a timestamp to write first,
    /// if it starts a new run.
    pub(crate) fn before(&mut self, pool: &'static Pool) -> Option<u32> {
        let (_, name) = self.bound?;
        if self.open && self.segments.last().is_some_and(|s| s.name == name) {
            return None;
        }
        self.open = false;
        let at = self.take(pool)?;
        if std::mem::take(&mut self.pending)
            && let Some(last) = self.segments.last_mut()
        {
            last.end = at;
        }
        self.segments.push(Segment { name, begin: at, end: at, calls: 0 });
        self.open = true;
        Some(at)
    }

    /// After a draw or dispatch: the index of the timestamp to write. An
    /// end that finds no index closes the run where its last draw ended.
    pub(crate) fn after(&mut self, pool: &'static Pool) -> Option<u32> {
        if !self.open {
            return None;
        }
        if coarse() {
            // Its end is stamped where the next run starts or the list closes.
            self.segments.last_mut()?.calls += 1;
            self.pending = true;
            return None;
        }
        let Some(at) = self.take(pool) else {
            self.open = false;
            return None;
        };
        let segment = self.segments.last_mut()?;
        segment.end = at;
        segment.calls += 1;
        Some(at)
    }

    /// The draws recorded from here on start a new run (something else
    /// timed came between).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn cut(&mut self) {
        self.open = false;
    }

    /// The list closes: the index of its last timestamp, and the ranges
    /// to resolve (first index, count).
    pub(crate) fn finish(&mut self, pool: &'static Pool) -> (Option<u32>, Vec<(u32, u32)>) {
        self.open = false;
        self.end = self.take(pool);
        if std::mem::take(&mut self.pending)
            && let (Some(end), Some(last)) = (self.end, self.segments.last_mut())
        {
            last.end = end;
        }
        let ranges = self
            .leases
            .iter()
            .enumerate()
            .map(|(i, lease)| {
                let first = lease.chunk * CHUNK;
                let count = if i + 1 == self.leases.len() { self.next - first } else { CHUNK };
                (first, count)
            })
            .filter(|&(_, count)| count > 0)
            .collect();
        (self.end, ranges)
    }

    /// What a submission of the closed list will have timed once the fence
    /// reaches `value`; `None` if it timed nothing.
    pub(crate) fn record(&self, value: u64) -> Option<Record> {
        if self.leases.is_empty() || self.end.is_none() {
            return None;
        }
        Some(Record {
            value,
            begin: self.begin,
            end: self.end,
            segments: self.segments.iter().copied().filter(|s| s.calls > 0).collect(),
            _leases: self.leases.clone(),
        })
    }
}

/// A submitted list's timestamps, read when the fence reaches `value`.
pub(crate) struct Record {
    value: u64,
    begin: Option<u32>,
    end: Option<u32>,
    segments: Vec<Segment>,
    _leases: Vec<Arc<Lease>>,
}

/// GPU time summed since the last report, in timestamp ticks.
#[derive(Default)]
pub(crate) struct Tally {
    /// By pipeline name id: ticks and calls.
    by: HashMap<u32, (u64, u64)>,
    /// The lists' time, first to last timestamp, and how many.
    list_ticks: u64,
    lists: u64,
    /// Of it, in runs of draws.
    draw_ticks: u64,
    /// Spans that made no sense (a later run of the list overwrote part of
    /// its timestamps before they were read, or one was not written).
    rejected: u64,
}

impl Tally {
    /// Adds what `record` timed; `read` gives a timestamp by heap index;
    /// spans longer than `most` ticks are rejected.
    pub(crate) fn add(&mut self, record: &Record, read: impl Fn(u32) -> u64, most: u64) {
        let span = |a: u32, b: u32| {
            let (a, b) = (read(a), read(b));
            (a != 0 && b >= a && b - a <= most).then(|| b - a)
        };
        let first = record.begin.or(record.segments.first().map(|s| s.begin));
        if let (Some(first), Some(end)) = (first, record.end) {
            match span(first, end) {
                Some(t) => {
                    self.list_ticks += t;
                    self.lists += 1;
                }
                None => self.rejected += 1,
            }
        }
        for s in &record.segments {
            match span(s.begin, s.end) {
                Some(t) => {
                    let entry = self.by.entry(s.name).or_default();
                    entry.0 += t;
                    entry.1 += u64::from(s.calls);
                    self.draw_ticks += t;
                }
                None => self.rejected += 1,
            }
        }
    }

    /// The report's lines, for `frames` frames over `secs` seconds with
    /// `frequency` ticks a second.
    pub(crate) fn report(&self, frequency: u64, frames: u64, secs: f64, name: impl Fn(u32) -> Arc<str>, watch: &[String]) -> Vec<String> {
        let frames_f = frames.max(1) as f64;
        let ms = |ticks: u64| ticks as f64 * 1e3 / frequency.max(1) as f64;
        let per_frame = |ticks: u64| ms(ticks) / frames_f;
        let mut top: Vec<(u32, u64, u64)> = self.by.iter().map(|(&id, &(t, c))| (id, t, c)).collect();
        top.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        // The watched pipelines past the top, in the same order.
        let rest: Vec<(usize, (u32, u64, u64))> = if watch.is_empty() {
            Vec::new()
        } else {
            top.iter()
                .enumerate()
                .skip(TOP)
                .filter(|(_, t)| watch.iter().any(|h| name(t.0).contains(h.as_str())))
                .map(|(i, &t)| (i, t))
                .collect()
        };
        let unseen: Vec<&String> =
            watch.iter().filter(|h| !self.by.keys().any(|&id| name(id).contains(h.as_str()))).collect();
        top.truncate(TOP);
        let top_ticks: u64 = top.iter().map(|t| t.1).sum();
        let share = |ticks: u64| {
            if self.list_ticks == 0 { 0.0 } else { 100.0 * ticks as f64 / self.list_ticks as f64 }
        };
        let mut lines = vec![format!(
            "GPU profile over {secs:.1} s, {frames} frames: {:.1} ms/frame on the card in {:.0} lists/frame; \
             runs of draws and dispatches {:.1} ms/frame ({:.0}%) of {} pipelines; the top {} cover {:.1} ms/frame ({:.0}%){}",
            per_frame(self.list_ticks),
            self.lists as f64 / frames_f,
            per_frame(self.draw_ticks),
            share(self.draw_ticks),
            self.by.len(),
            top.len(),
            per_frame(top_ticks),
            share(top_ticks),
            if self.rejected > 0 { format!(", {} spans rejected", self.rejected) } else { String::new() },
        )];
        for (i, &(id, ticks, calls)) in top.iter().enumerate() {
            lines.push(format!(
                "GPU profile #{:<2} {:7.2} ms/frame {:4.1}%  {:7.1} calls/frame {:8.4} ms/call  {}",
                i + 1,
                per_frame(ticks),
                share(ticks),
                calls as f64 / frames_f,
                ms(ticks) / calls.max(1) as f64,
                name(id),
            ));
        }
        for (i, (id, ticks, calls)) in rest {
            lines.push(format!(
                "GPU profile watch #{:<4} {:7.3} ms/frame {:4.1}%  {:7.1} calls/frame {:8.4} ms/call  {}",
                i + 1,
                per_frame(ticks),
                share(ticks),
                calls as f64 / frames_f,
                ms(ticks) / calls.max(1) as f64,
                name(id),
            ));
        }
        for h in unseen {
            lines.push(format!("GPU profile watch: no pipeline with {h} ran (0 calls)"));
        }
        lines
    }
}

/// The heap, its read-back buffer (mapped while the process lives) and
/// the pool of its chunks.
#[cfg(windows)]
pub(crate) struct Stamps {
    pub(crate) heap: windows::Win32::Graphics::Direct3D12::ID3D12QueryHeap,
    pub(crate) back: windows::Win32::Graphics::Direct3D12::ID3D12Resource,
    data: *const u64,
    frequency: u64,
    pub(crate) pool: &'static Pool,
}

// SAFETY: D3D12 objects are free-threaded; `data` is a mapping of `back`,
// only read, at indices whose writes the fence has seen.
#[cfg(windows)]
unsafe impl Send for Stamps {}
#[cfg(windows)]
unsafe impl Sync for Stamps {}

/// The heap, made at the first use (profiling on).
#[cfg(windows)]
pub(crate) fn stamps(
    device: &windows::Win32::Graphics::Direct3D12::ID3D12Device,
    queue: &windows::Win32::Graphics::Direct3D12::ID3D12CommandQueue,
) -> Option<&'static Stamps> {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    static STAMPS: OnceLock<Option<Stamps>> = OnceLock::new();
    if !on() {
        return None;
    }
    STAMPS
        .get_or_init(|| {
            let made = (|| -> windows::core::Result<Stamps> {
                let count = CHUNKS * CHUNK;
                let mut heap: Option<ID3D12QueryHeap> = None;
                let mut back: Option<ID3D12Resource> = None;
                let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: u64::from(count) * 8,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    ..Default::default()
                };
                // SAFETY: plain object creation and a map of a read-back
                // buffer of ours, kept for the process's life.
                unsafe {
                    device.CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: count, NodeMask: 0 },
                        &mut heap,
                    )?;
                    device.CreateCommittedResource(
                        &readback,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COPY_DEST,
                        None,
                        &mut back,
                    )?;
                    let back = back.ok_or_else(windows::core::Error::empty)?;
                    let mut data = std::ptr::null_mut();
                    back.Map(0, Some(&D3D12_RANGE { Begin: 0, End: desc.Width as usize }), Some(&mut data))?;
                    let frequency = queue.GetTimestampFrequency()?;
                    Ok(Stamps {
                        heap: heap.ok_or_else(windows::core::Error::empty)?,
                        back,
                        data: data.cast::<u64>().cast_const(),
                        frequency,
                        pool: Box::leak(Box::new(Pool::new(CHUNKS))),
                    })
                }
            })();
            match made {
                Ok(s) => {
                    tracing::info!(
                        "GPU profile on (VICTORY_GPU_PROFILE={}{}): {} timestamps, {} ticks a second; reports every {REPORT_SECS} s",
                        mode(),
                        if coarse() { ", a timestamp per run" } else { ", a timestamp per draw" },
                        CHUNKS * CHUNK,
                        s.frequency
                    );
                    Some(s)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "GPU profile: no timestamp heap; nothing is profiled");
                    None
                }
            }
        })
        .as_ref()
}

/// The lists submitted and not yet read, and what was read since the
/// last report.
struct Collector {
    pending: VecDeque<Record>,
    tally: Tally,
    frames: u64,
    since: Option<std::time::Instant>,
    /// Records let go unread (too many waiting).
    dropped: u64,
}

static COLLECTOR: std::sync::LazyLock<Mutex<Collector>> = std::sync::LazyLock::new(|| {
    Mutex::new(Collector { pending: VecDeque::new(), tally: Tally::default(), frames: 0, since: None, dropped: 0 })
});

/// A submission of a list that timed something: read once the fence
/// reaches its value.
pub(crate) fn submitted(record: Record) {
    let mut c = COLLECTOR.lock().unwrap_or_else(|e| e.into_inner());
    if c.pending.len() >= PENDING_MOST {
        c.pending.pop_front();
        c.dropped += 1;
    }
    c.pending.push_back(record);
}

/// At a present: reads the lists the fence has passed (`completed`, its
/// value) and logs a report every [`REPORT_SECS`].
#[cfg(windows)]
pub(crate) fn presented(stamps: &Stamps, completed: u64) {
    let mut c = COLLECTOR.lock().unwrap_or_else(|e| e.into_inner());
    c.frames += 1;
    let since = *c.since.get_or_insert_with(std::time::Instant::now);
    let data = stamps.data;
    let read = |i: u32| {
        if i >= CHUNKS * CHUNK {
            return 0;
        }
        // SAFETY: an index of the mapped buffer that a finished list
        // resolved into.
        unsafe { std::ptr::read_volatile(data.add(i as usize)) }
    };
    // Submissions go out in fence order: the ready ones are at the front.
    while c.pending.front().is_some_and(|r| r.value <= completed) {
        let record = c.pending.pop_front().unwrap();
        c.tally.add(&record, read, stamps.frequency);
    }
    let elapsed = since.elapsed();
    if elapsed < std::time::Duration::from_secs(REPORT_SECS) {
        return;
    }
    let tally = std::mem::take(&mut c.tally);
    let (frames, dropped) = (c.frames, std::mem::take(&mut c.dropped));
    let waiting = c.pending.len();
    c.frames = 0;
    c.since = Some(std::time::Instant::now());
    drop(c);
    let starved = STARVED.swap(0, Ordering::Relaxed);
    for line in tally.report(stamps.frequency, frames, elapsed.as_secs_f64(), name_of, watched()) {
        tracing::info!("{line}");
    }
    if starved + dropped > 0 {
        tracing::info!(
            "GPU profile: {starved} lists stopped timing for want of timestamps, {dropped} submissions let go unread; {waiting} waiting"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pool(chunks: u32) -> &'static Pool {
        Box::leak(Box::new(Pool::new(chunks)))
    }

    #[test]
    fn runs_of_one_pipeline_share_a_begin() {
        let p = pool(4);
        let mut r = Recording::default();
        assert_eq!(r.start(p), Some(0));
        // Nothing bound: nothing timed.
        assert_eq!(r.before(p), None);
        assert_eq!(r.after(p), None);
        r.bound(1, "[vs a, ps b]");
        let b = r.before(p).unwrap();
        let e1 = r.after(p).unwrap();
        assert_eq!(r.before(p), None, "same pipeline: no new begin");
        let e2 = r.after(p).unwrap();
        r.bound(2, "[cs c]");
        let b2 = r.before(p).unwrap();
        let e3 = r.after(p).unwrap();
        // Back to the first: a new run.
        r.bound(1, "[vs a, ps b]");
        assert!(r.before(p).is_some());
        r.after(p).unwrap();
        // Something else timed between: a new run of the same pipeline.
        r.cut();
        assert!(r.before(p).is_some());
        r.after(p).unwrap();
        assert_eq!(r.segments.len(), 4);
        assert_eq!((r.segments[0].begin, r.segments[0].end, r.segments[0].calls), (b, e2, 2));
        assert_eq!((r.segments[1].begin, r.segments[1].end), (b2, e3));
        assert_eq!(r.segments[0].name, r.segments[2].name);
        assert!(e1 < e2);
        let (end, ranges) = r.finish(p);
        assert!(end.is_some());
        let used: u32 = ranges.iter().map(|r| r.1).sum();
        assert_eq!(used, 1 + 3 + 2 + 2 + 2 + 1, "start, four runs, close");
        assert_eq!(ranges, vec![(0, used)]);
    }

    #[test]
    fn coarse_runs_end_where_the_next_begins() {
        TEST_COARSE.set(Some(true));
        let p = pool(4);
        let mut r = Recording::default();
        r.start(p);
        r.bound(1, "[vs a, ps b]");
        let b = r.before(p).unwrap();
        assert_eq!(r.after(p), None, "no timestamp after a draw");
        assert_eq!(r.before(p), None);
        assert_eq!(r.after(p), None);
        r.bound(2, "[cs c]");
        let b2 = r.before(p).unwrap();
        assert_eq!(r.after(p), None);
        let (end, ranges) = r.finish(p);
        TEST_COARSE.set(None);
        assert_eq!((r.segments[0].begin, r.segments[0].end, r.segments[0].calls), (b, b2, 2));
        assert_eq!((r.segments[1].begin, Some(r.segments[1].end), r.segments[1].calls), (b2, end, 1));
        assert_eq!(ranges.iter().map(|r| r.1).sum::<u32>(), 4, "start, two runs, close");
    }

    #[test]
    fn chunks_are_taken_in_turn_and_come_back() {
        let p = pool(2);
        let mut r = Recording::default();
        r.start(p);
        r.bound(7, "[cs x]");
        for _ in 0..CHUNK / 2 {
            r.cut();
            r.before(p);
            r.after(p);
        }
        let (_, ranges) = r.finish(p);
        assert_eq!(ranges, vec![(0, CHUNK), (CHUNK, 2)]);
        assert_eq!(p.free(), 0);
        let record = r.record(5).unwrap();
        assert_eq!(record.segments.len(), (CHUNK / 2) as usize);
        r.clear();
        assert_eq!(p.free(), 0, "the record still holds the chunks");
        drop(record);
        assert_eq!(p.free(), 2);
    }

    #[test]
    fn a_list_without_timestamps_stops() {
        let p = pool(1);
        let mut a = Recording::default();
        a.start(p);
        let mut b = Recording::default();
        assert_eq!(b.start(p), None);
        b.bound(1, "[vs a]");
        assert_eq!(b.before(p), None);
        assert_eq!(b.after(p), None);
        assert!(b.record(1).is_none());
        a.bound(1, "[vs a]");
        for _ in 0..CHUNK / 2 {
            a.cut();
            a.before(p);
            a.after(p);
        }
        assert!(a.full);
        // The run whose end found no index timed nothing.
        assert_eq!(a.segments.last().unwrap().calls, 0);
        let (end, ranges) = a.finish(p);
        assert_eq!(end, None);
        assert_eq!(ranges, vec![(0, CHUNK)]);
        assert!(a.record(1).is_none(), "no last timestamp: not read");
    }

    #[test]
    fn tally_sums_by_pipeline_and_rejects_nonsense() {
        let p = pool(1);
        let mut r = Recording::default();
        r.start(p); // 0
        r.bound(1, "[vs t1]");
        r.before(p); // 1
        r.after(p); // 2
        r.before(p);
        r.after(p); // 3 (a second draw of the run)
        r.bound(2, "[cs t2]");
        r.before(p); // 4
        r.after(p); // 5
        r.finish(p); // 6
        let record = r.record(9).unwrap();
        let ticks = [100u64, 110, 150, 200, 210, 260, 300];
        let mut tally = Tally::default();
        tally.add(&record, |i| ticks[i as usize], 1000);
        assert_eq!(tally.list_ticks, 200);
        assert_eq!(tally.draw_ticks, 90 + 50);
        let t1 = intern("[vs t1]");
        assert_eq!(tally.by[&t1], (90, 2));
        // A later run wrote part of them: spans going back are rejected.
        let mut tally = Tally::default();
        let torn = [100u64, 110, 150, 200, 900, 260, 300];
        tally.add(&record, |i| torn[i as usize], 1000);
        assert_eq!(tally.rejected, 1);
        // Never written (zero), or longer than the most: rejected too.
        let mut tally = Tally::default();
        tally.add(&record, |i| if i == 0 { 0 } else { ticks[i as usize] }, 1000);
        assert_eq!((tally.lists, tally.rejected), (0, 1));
        tally.add(&record, |i| ticks[i as usize] * 100, 1000);
        assert_eq!(tally.rejected, 1 + 3, "the list and both runs too long");
        let mut tally = Tally::default();
        tally.add(&record, |i| torn[i as usize], 1000);
        let lines = tally.report(1000, 2, 10.0, name_of, &[]);
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("2 frames") && lines[0].contains("1 spans rejected"), "{}", lines[0]);
        assert!(lines[1].contains("[vs t1]") && lines[1].contains("45.00 ms/frame"), "{}", lines[1]);
    }

    #[test]
    fn report_orders_by_time() {
        let mut tally = Tally::default();
        let (a, b) = (intern("[ps slow]"), intern("[ps fast]"));
        tally.by.insert(a, (3000, 3));
        tally.by.insert(b, (1000, 10));
        tally.list_ticks = 8000;
        tally.draw_ticks = 4000;
        tally.lists = 4;
        let lines = tally.report(1_000_000, 1, 10.0, name_of, &[]);
        assert!(lines[1].contains("[ps slow]") && lines[1].contains("37.5%"), "{}", lines[1]);
        assert!(lines[2].contains("[ps fast]") && lines[2].contains("0.1000 ms/call"), "{}", lines[2]);
        assert!(lines[0].contains("(50%)"), "{}", lines[0]);
    }

    #[test]
    fn watched_pipelines_are_named_past_the_top() {
        let mut tally = Tally::default();
        for i in 0..TOP as u64 + 3 {
            let id = intern(&format!("[mesh watchtest{i:04}]"));
            tally.by.insert(id, (10_000 - i, 2));
        }
        tally.list_ticks = 1_000_000;
        let watch = ["watchtest0027".to_owned(), "watchtest0003".to_owned(), "absent99".to_owned()];
        let lines = tally.report(1_000_000, 1, 10.0, name_of, &watch);
        // the header, the top, the one watched past it, the one that did not run
        assert_eq!(lines.len(), 1 + TOP + 2, "{lines:#?}");
        assert!(lines[1 + TOP].contains("watch #28") && lines[1 + TOP].contains("watchtest0027"), "{}", lines[1 + TOP]);
        assert!(lines[1 + TOP].contains("2.0 calls/frame"), "{}", lines[1 + TOP]);
        assert!(lines[2 + TOP].contains("no pipeline with absent99"), "{}", lines[2 + TOP]);
    }
}
