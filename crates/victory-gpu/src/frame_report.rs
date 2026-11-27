//! Optional bounded, nonblocking frame samples. Disk I/O belongs to the app.
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError};
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct FrameSample {
    pub sequence: u64,
    pub timestamp_us: u64,
    pub interval_us: u64,
}

static SINK: OnceLock<SyncSender<FrameSample>> = OnceLock::new();
static ACTIVE: AtomicBool = AtomicBool::new(false);
static DROPPED: AtomicU64 = AtomicU64::new(0);
static REJECTED: AtomicU64 = AtomicU64::new(0);
static EXPECTED: AtomicU64 = AtomicU64::new(0);
static FRAMEGEN_SEEN: AtomicBool = AtomicBool::new(false);

pub fn install(sink: SyncSender<FrameSample>) -> bool {
    if SINK.set(sink).is_err() { return false; }
    ACTIVE.store(true, Ordering::Release);
    true
}

pub fn stop() { ACTIVE.store(false, Ordering::Release); }
pub fn dropped() -> u64 { DROPPED.load(Ordering::Relaxed) }
pub fn rejected() -> u64 { REJECTED.load(Ordering::Relaxed) }
pub fn expected() -> u64 { EXPECTED.load(Ordering::Relaxed) }
pub fn framegen_seen() -> bool { FRAMEGEN_SEEN.load(Ordering::Relaxed) }

pub(crate) fn rejected_present() {
    if ACTIVE.load(Ordering::Acquire) { REJECTED.fetch_add(1, Ordering::Relaxed); }
}

fn send(sink: &SyncSender<FrameSample>, sample: FrameSample, dropped: &AtomicU64) {
    if let Err(error) = sink.try_send(sample) {
        dropped.fetch_add(1, Ordering::Relaxed);
        if matches!(error, TrySendError::Disconnected(_)) { stop(); }
    }
}

#[derive(Default)]
pub(crate) struct Sampler {
    start: Option<Instant>,
    last: Option<Instant>,
    sequence: u64,
}

impl Sampler {
    pub(crate) fn presented(&mut self) {
        if !ACTIVE.load(Ordering::Acquire) { return; }
        let Some(sink) = SINK.get() else { return };
        let now = Instant::now();
        let start = *self.start.get_or_insert(now);
        if let Some(last) = self.last.replace(now) {
            if !ACTIVE.load(Ordering::Acquire) { return; }
            if crate::streamline::framegen_live() { FRAMEGEN_SEEN.store(true, Ordering::Relaxed); }
            self.sequence += 1;
            EXPECTED.store(self.sequence, Ordering::Relaxed);
            send(sink, FrameSample {
                sequence: self.sequence,
                timestamp_us: now.duration_since(start).as_micros().min(u128::from(u64::MAX)) as u64,
                interval_us: now.duration_since(last).as_micros().min(u128::from(u64::MAX)) as u64,
            }, &DROPPED);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn a_full_report_queue_drops_samples_without_waiting_and_keeps_sequence_gaps() {
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        let dropped = AtomicU64::new(0);
        let sample = |sequence| FrameSample { sequence, timestamp_us: sequence * 10_000, interval_us: 10_000 };
        send(&tx, sample(1), &dropped);
        send(&tx, sample(2), &dropped);
        assert_eq!(dropped.load(Ordering::Relaxed), 1);
        assert_eq!(rx.try_recv().unwrap().sequence, 1);
        send(&tx, sample(3), &dropped);
        assert_eq!(rx.try_recv().unwrap().sequence, 3);
    }
}
