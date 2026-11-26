//! Opt-in wall-clock instrumentation for selected title submission stages.
//! Adapted concept from Kelvaroq 0053, without moving any fence or queue work.
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
pub(super) enum Stage {
    DescriptorSync = 0,
    DescriptorWaiting = 1,
    DescriptorJoin = 2,
    Execute = 3,
}

#[derive(Default)]
struct Stats {
    elapsed: [Duration; 4],
    calls: u64,
    lists: u64,
}

impl Stats {
    fn record(&mut self, stage: Stage, elapsed: Duration, lists: usize) {
        self.elapsed[stage as usize] += elapsed;
        if matches!(stage, Stage::Execute) {
            self.calls += 1;
            self.lists += lists as u64;
        }
    }
}

static STATS: Mutex<Stats> = Mutex::new(Stats {
    elapsed: [Duration::ZERO; 4],
    calls: 0,
    lists: 0,
});

fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SUBMIT_STAGES").is_some_and(|v| v == "1"))
}

pub(super) fn start() -> Option<Instant> {
    enabled().then(Instant::now)
}

pub(super) fn finish(start: Option<Instant>, stage: Stage, lists: usize) {
    if let Some(start) = start {
        let elapsed = start.elapsed();
        STATS.lock().unwrap_or_else(|e| e.into_inner()).record(stage, elapsed, lists);
    }
}

pub(super) fn line(frames: f64) {
    if !enabled() || !frames.is_finite() || frames <= 0.0 {
        return;
    }
    let stats = std::mem::take(&mut *STATS.lock().unwrap_or_else(|e| e.into_inner()));
    let [sync, waiting, join, execute] = stats.elapsed.map(|d| d.as_secs_f64() * 1000.0 / frames);
    tracing::info!(
        descriptor_sync_ms = sync,
        descriptor_waiting_ms = waiting,
        descriptor_join_ms = join,
        execute_cpu_ms = execute,
        calls_per_frame = stats.calls as f64 / frames,
        lists_per_call = stats.lists as f64 / stats.calls.max(1) as f64,
        "title submission stages (wall ms/frame; ExecuteCommandLists excludes signal and fence wait)"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn submission_stages_keep_execute_counts_and_durations_separate() {
        let mut stats = Stats::default();
        stats.record(Stage::DescriptorSync, Duration::from_micros(20), 0);
        stats.record(Stage::DescriptorWaiting, Duration::from_micros(30), 0);
        stats.record(Stage::DescriptorJoin, Duration::from_micros(40), 0);
        stats.record(Stage::Execute, Duration::from_micros(80), 3);
        stats.record(Stage::Execute, Duration::from_micros(15), 2);
        assert_eq!(stats.elapsed.map(|d| d.as_micros()), [20, 30, 40, 95]);
        assert_eq!((stats.calls, stats.lists), (2, 5));
        let snapshot = std::mem::take(&mut stats);
        assert_eq!(snapshot.calls, 2);
        assert_eq!(stats.calls, 0);
        assert_eq!(stats.elapsed, [Duration::ZERO; 4]);
    }
}
