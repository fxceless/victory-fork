use std::sync::{Mutex, MutexGuard, TryLockError};
use std::time::{Duration, Instant};

struct State {
    frame: u64,
    spent: Duration,
    limit: Duration,
    tried_slot: bool,
}

static STATE: Mutex<State> = Mutex::new(State {
    frame: 0,
    spent: Duration::ZERO,
    limit: Duration::ZERO,
    tried_slot: false,
});

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_FRAME_BUDGET").is_some_and(|v| v == "1"))
}

pub(crate) fn limit(cap: Duration, due: u64) -> Duration {
    match due {
        0..=255 => cap / 4,
        256..=1023 => cap / 2,
        1024..=2047 => cap - cap / 4,
        _ => cap,
    }
}

/// Only one ordinary pass can charge the budget at a time. A competing
/// recorder leaves the slots queued instead of waiting on its peer.
pub(crate) struct Pass {
    state: MutexGuard<'static, State>,
    started: Instant,
}

impl Pass {
    pub(crate) fn start(frame: u64, limit: Duration) -> Option<Self> {
        let mut state = match STATE.try_lock() {
            Ok(state) => state,
            Err(TryLockError::WouldBlock) => return None,
            Err(TryLockError::Poisoned(error)) => error.into_inner(),
        };
        if frame < state.frame {
            return None;
        }
        if frame != state.frame || state.limit.is_zero() {
            *state = State {
                frame,
                spent: Duration::ZERO,
                limit,
                tried_slot: false,
            };
        }
        if state.spent >= state.limit {
            return None;
        }
        Some(Self {
            state,
            started: Instant::now(),
        })
    }

    pub(crate) fn oldest_first(&self) -> bool {
        !self.state.tried_slot && self.state.frame.is_multiple_of(8)
    }

    pub(crate) fn exhausted(&self) -> bool {
        self.state.tried_slot
            && self.state.spent.saturating_add(self.started.elapsed()) >= self.state.limit
    }

    pub(crate) fn try_slot(&mut self) {
        self.state.tried_slot = true;
    }
}

impl Drop for Pass {
    fn drop(&mut self) {
        self.state.spent = self.state.spent.saturating_add(self.started.elapsed());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordinary_passes_share_a_budget_and_new_frames_reset_it() {
        let limit = Duration::from_secs(60);
        let mut pass = Pass::start(8, limit).expect("first pass");
        assert!(pass.oldest_first());
        assert!(Pass::start(8, limit * 2).is_none());
        pass.state.spent = limit;
        assert!(!pass.exhausted(), "the first slot can make progress");
        pass.try_slot();
        assert!(pass.exhausted());
        assert!(!pass.oldest_first());
        drop(pass);
        assert!(Pass::start(8, limit).is_none());
        assert!(Pass::start(9, limit).is_some());
        assert!(Pass::start(8, limit).is_none());
    }

    #[test]
    fn backlog_uses_a_larger_share_without_exceeding_the_cap() {
        let cap = Duration::from_millis(80);
        assert_eq!(limit(cap, 0), Duration::from_millis(20));
        assert_eq!(limit(cap, 256), Duration::from_millis(40));
        assert_eq!(limit(cap, 1024), Duration::from_millis(60));
        assert_eq!(limit(cap, 2048), cap);
    }
}
