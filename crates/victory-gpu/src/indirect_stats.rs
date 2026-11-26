//! Indirect commands recorded during a diagnostics timing window.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

#[derive(Default)]
pub(crate) struct Counters {
    title_calls: AtomicU64,
    requested_max_records: AtomicU64,
    multi_record_calls: AtomicU64,
    host_calls: AtomicU64,
    clamp_dispatches: AtomicU64,
}

pub(crate) static STATS: Counters = Counters {
    title_calls: AtomicU64::new(0),
    requested_max_records: AtomicU64::new(0),
    multi_record_calls: AtomicU64::new(0),
    host_calls: AtomicU64::new(0),
    clamp_dispatches: AtomicU64::new(0),
};

impl Counters {
    pub(crate) fn title_call(&self, max: u32) {
        self.title_calls.fetch_add(1, Relaxed);
        // A GPU count buffer can reduce this maximum at execution time.
        self.requested_max_records.fetch_add(u64::from(max), Relaxed);
        if max > 1 {
            self.multi_record_calls.fetch_add(1, Relaxed);
        }
    }

    pub(crate) fn host_call(&self) {
        self.host_calls.fetch_add(1, Relaxed);
    }

    pub(crate) fn clamp_dispatch(&self) {
        self.clamp_dispatches.fetch_add(1, Relaxed);
    }

    fn take(&self) -> [u64; 5] {
        [
            self.title_calls.swap(0, Relaxed),
            self.requested_max_records.swap(0, Relaxed),
            self.multi_record_calls.swap(0, Relaxed),
            self.host_calls.swap(0, Relaxed),
            self.clamp_dispatches.swap(0, Relaxed),
        ]
    }
}

pub(crate) fn log(frames: u64) {
    let [title, records, multi, host, clamps] = STATS.take();
    if title == 0 && host == 0 && clamps == 0 {
        return;
    }
    tracing::info!(
        "indirect commands recorded over {frames} frames: {title} title calls,          {records} requested maximum records ({multi} multi-record calls),          {host} host ExecuteIndirect calls, {clamps} clamp dispatches"
    );
}

#[cfg(test)]
mod tests {
    use super::Counters;

    #[test]
    fn distinguishes_batches_single_calls_and_clamp_reuse() {
        let counters = Counters::default();
        counters.title_call(3);
        counters.clamp_dispatch();
        counters.host_call();
        assert_eq!(counters.take(), [1, 3, 1, 1, 1]);

        for _ in 0..3 {
            counters.title_call(1);
            counters.host_call();
        }
        counters.clamp_dispatch();
        assert_eq!(counters.take(), [3, 3, 0, 3, 1]);
        assert_eq!(counters.take(), [0; 5]);
    }
}
