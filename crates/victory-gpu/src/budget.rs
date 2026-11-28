//! Pure admission rules shared by the host-memory and texture-pool paths.
pub(crate) const MIB: u64 = 1 << 20;
pub(crate) const GIB: u64 = 1 << 30;

pub(crate) fn small_host(total: Option<u64>, forced: bool) -> bool {
    forced || total.is_some_and(|bytes| bytes <= 16 * GIB)
}

pub(crate) fn prefetch(
    requested: u64, pressure: bool, critical: bool, limited: bool, staging: u64, small: bool,
) -> u64 {
    let cap = if small {
        if critical { 8 * MIB } else if pressure || staging >= 256 * MIB { 16 * MIB } else { 32 * MIB }
    } else if critical {
        16 * MIB
    } else if pressure || (limited && staging >= GIB) {
        64 * MIB
    } else {
        requested
    };
    requested.min(cap)
}

pub(crate) fn texture_spares(memory: Option<[u64; 4]>, keep: bool, small: bool) -> usize {
    match memory {
        Some([budget, used, ..]) if keep && u128::from(used) * 100 < u128::from(budget) * 85 => {
            if small { 4 } else if budget <= 8 * GIB { ((budget / 16) / (64 * MIB)).clamp(2, 8) as usize } else { 64 }
        }
        _ => 2,
    }
}

/// Account for reusable pool space and reservations without integer wraparound.
pub(crate) fn texture_usage(used: u64, free_pool: u64, buffers: u64, targets: u64) -> u64 {
    used.saturating_sub(free_pool).saturating_add(buffers).saturating_add(targets)
}

/// Supplement reported-memory admission with an explicit texture cap, never
/// substitute the cap for it. Unknown VRAM retains the fallback cap.
pub(crate) fn texture_room(filled: u64, limit: Option<u64>, under_budget: Option<bool>) -> bool {
    if limit.is_some_and(|limit| filled >= limit) { return false; }
    under_budget.unwrap_or_else(|| filled < limit.unwrap_or(4 * GIB).min(4 * GIB))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn texture_usage_accounts_for_free_pool_and_saturates_reservations() {
        assert_eq!(texture_usage(4 * GIB, GIB, GIB, 2 * GIB), 6 * GIB);
        assert_eq!(texture_usage(0, GIB, 0, 0), 0);
        assert_eq!(texture_usage(u64::MAX - 1, 0, 2, 2), u64::MAX);
    }

    #[test]
    fn small_profile_starts_before_memory_pressure() {
        assert!(small_host(Some(16 * GIB), false));
        assert!(small_host(Some(32 * GIB), true));
        assert!(!small_host(Some(32 * GIB), false));
        assert!(!small_host(None, false));
        assert_eq!(prefetch(512 * MIB, false, false, true, 0, true), 32 * MIB);
        assert_eq!(prefetch(512 * MIB, true, false, true, 0, true), 16 * MIB);
        assert_eq!(prefetch(512 * MIB, true, true, true, 0, true), 8 * MIB);
        assert_eq!(prefetch(4 * MIB, true, true, true, 0, true), 4 * MIB);
        assert_eq!(prefetch(0, true, true, true, 0, true), 0);
    }
    #[test]
    fn previous_large_host_prefetch_rules_are_preserved() {
        assert_eq!(prefetch(512 * MIB, false, false, false, 0, false), 512 * MIB);
        assert_eq!(prefetch(512 * MIB, false, false, true, GIB, false), 64 * MIB);
        assert_eq!(prefetch(512 * MIB, true, true, false, 0, false), 16 * MIB);
    }
    #[test]
    fn idle_pool_does_not_keep_four_gib_on_an_eight_gib_gpu() {
        assert_eq!(texture_spares(Some([8 * GIB, 4 * GIB, 0, 0]), true, false), 8);
        assert_eq!(texture_spares(Some([8 * GIB, 4 * GIB, 0, 0]), true, true), 4);
        assert_eq!(texture_spares(Some([24 * GIB, 12 * GIB, 0, 0]), true, false), 64);
        assert_eq!(texture_spares(Some([8 * GIB, 7 * GIB, 0, 0]), true, true), 2);
        assert_eq!(texture_spares(None, true, true), 2);
    }
    #[test]
    fn explicit_limit_never_bypasses_live_vram_or_reservations() {
        assert!(!texture_room(MIB, Some(4 * GIB), Some(false)));
        assert!(!texture_room(MIB, None, Some(false)));
        assert!(texture_room(GIB, Some(4 * GIB), Some(true)));
        assert!(!texture_room(4 * GIB, Some(4 * GIB), Some(true)));
        assert!(!texture_room(0, Some(0), Some(true)));
        assert!(texture_room(GIB, None, None));
        assert!(!texture_room(4 * GIB, Some(8 * GIB), None));
    }
}
