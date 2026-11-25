//! Display settings of the presenting window: a frame-rate limit and the
//! window mode (windowed or borderless), and what is measured about the
//! gaps between presents.
//!
//! The settings come from `victory.toml` (`[display]`) and the environment
//! (`VICTORY_FPS_LIMIT`, `VICTORY_WINDOW_MODE`), resolved by `victory-core`
//! and handed over with [`set`] before the first present. Nothing here
//! changes the title's own frame clock (see `frame.rs`): the limit only
//! delays our `Present`.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// How the presenting window is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WindowMode {
    #[default]
    Windowed,
    /// A popup over the monitor the window is on (no exclusive mode).
    Borderless,
}

/// What the user asked for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Settings {
    /// Presents per second at most; 0 is no limit.
    pub fps_limit: u32,
    pub mode: WindowMode,
}

static SETTINGS: OnceLock<Settings> = OnceLock::new();

/// Sets the settings, once, before the first present. A later call is
/// ignored.
pub fn set(settings: Settings) {
    let _ = SETTINGS.set(settings);
}

/// The settings in force (the defaults if [`set`] was never called).
pub fn settings() -> Settings {
    SETTINGS.get().copied().unwrap_or_default()
}

/// Spreads presents `1/limit` apart. A present that is late is not made up
/// for: the next deadline counts from the moment it was made, so a slow
/// frame is never followed by a burst.
#[derive(Debug)]
pub struct Pacer {
    period: Option<Duration>,
    next: Option<Instant>,
}

impl Pacer {
    pub fn new(fps_limit: u32) -> Self {
        let period = (fps_limit > 0).then(|| Duration::from_secs_f64(1.0 / f64::from(fps_limit)));
        Pacer { period, next: None }
    }

    /// The time to wait until before presenting at `now`, if any; it also
    /// books the deadline of the present after it.
    pub fn deadline(&mut self, now: Instant) -> Option<Instant> {
        let period = self.period?;
        let due = self.next.unwrap_or(now);
        let at = due.max(now);
        self.next = Some(at + period);
        (due > now).then_some(due)
    }

    /// Waits for the deadline: asleep until a moment before it (the system
    /// timer is coarse), then spinning. Returns the time spent waiting.
    pub fn wait(&mut self) -> Duration {
        let start = Instant::now();
        let Some(due) = self.deadline(start) else {
            return Duration::ZERO;
        };
        const SPIN: Duration = Duration::from_micros(1500);
        if let Some(sleep) = due.checked_duration_since(Instant::now()).and_then(|d| d.checked_sub(SPIN)) {
            std::thread::sleep(sleep);
        }
        while Instant::now() < due {
            std::hint::spin_loop();
        }
        start.elapsed()
    }
}

// Per second, reset by `take`: presents, the longest gap between two
// presents and the sum of gaps (microseconds), time the limiter waited.
static PRESENTS: AtomicU64 = AtomicU64::new(0);
static GAP_MAX_US: AtomicU64 = AtomicU64::new(0);
static GAP_SUM_US: AtomicU64 = AtomicU64::new(0);
static LIMITED_US: AtomicU64 = AtomicU64::new(0);

/// Notes a present `gap` after the one before it, and the time the limiter
/// held it.
pub fn note_present(gap: Option<Duration>, limited: Duration) {
    PRESENTS.fetch_add(1, Ordering::Relaxed);
    LIMITED_US.fetch_add(limited.as_micros() as u64, Ordering::Relaxed);
    if let Some(gap) = gap {
        let us = gap.as_micros() as u64;
        GAP_SUM_US.fetch_add(us, Ordering::Relaxed);
        GAP_MAX_US.fetch_max(us, Ordering::Relaxed);
    }
}

/// The line logged with the frame rate: gaps between presents and what the
/// limit held back, over the presents since the last call.
pub fn take_line() -> Option<String> {
    let presents = PRESENTS.swap(0, Ordering::Relaxed);
    let sum = GAP_SUM_US.swap(0, Ordering::Relaxed);
    let max = GAP_MAX_US.swap(0, Ordering::Relaxed);
    let limited = LIMITED_US.swap(0, Ordering::Relaxed);
    if presents == 0 {
        return None;
    }
    let limit = settings().fps_limit;
    Some(format!(
        "present gaps: mean {:.1} ms, longest {:.1} ms over {presents} presents; fps limit {} held back {:.1} ms per present",
        sum as f64 / 1000.0 / presents as f64,
        max as f64 / 1000.0,
        if limit == 0 { "off".to_string() } else { limit.to_string() },
        limited as f64 / 1000.0 / presents as f64,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_limit_never_waits() {
        let mut pacer = Pacer::new(0);
        let now = Instant::now();
        assert_eq!(pacer.deadline(now), None);
        assert_eq!(pacer.deadline(now), None);
    }

    #[test]
    fn presents_are_spread_one_period_apart() {
        let mut pacer = Pacer::new(100);
        let t0 = Instant::now();
        assert_eq!(pacer.deadline(t0), None, "the first present is not held");
        let due = pacer.deadline(t0 + Duration::from_millis(2)).unwrap();
        assert_eq!(due, t0 + Duration::from_millis(10));
        // Presented at that deadline: the next is 10 ms after it.
        assert_eq!(
            pacer.deadline(t0 + Duration::from_millis(10)),
            Some(t0 + Duration::from_millis(20))
        );
    }

    #[test]
    fn a_late_frame_is_not_made_up_for() {
        let mut pacer = Pacer::new(100);
        let t0 = Instant::now();
        pacer.deadline(t0);
        // 50 ms late: presented at once, and the next one is 10 ms after it.
        assert_eq!(pacer.deadline(t0 + Duration::from_millis(50)), None);
        assert_eq!(
            pacer.deadline(t0 + Duration::from_millis(51)),
            Some(t0 + Duration::from_millis(60))
        );
    }

    #[test]
    fn wait_holds_a_real_period() {
        let mut pacer = Pacer::new(50);
        pacer.wait();
        let start = Instant::now();
        pacer.wait();
        assert!(start.elapsed() >= Duration::from_millis(19), "{:?}", start.elapsed());
    }
}
