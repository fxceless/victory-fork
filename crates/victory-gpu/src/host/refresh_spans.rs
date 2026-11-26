//! Bounded range union for buffers known not to support write-watch.
//! Keep one job per shadow and retain all existing write barriers.
use crate::shadow::Shadow;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, OnceLock};

type Span = (usize, usize);
const GAP: usize = 65536;
const MAX_SPANS: usize = 8;
fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPARSE_REFRESH").is_some_and(|v| v == "1"))
}

enum Ranges {
    One(Span),
    Many(Vec<Span>),
}

struct Spans {
    ranges: Ranges,
    sparse: bool,
}
impl Spans {
    fn new(from: usize, to: usize, sparse: bool) -> Self {
        Self {
            ranges: Ranges::One((from, to)),
            sparse,
        }
    }
    fn ranges(&self) -> &[Span] {
        match &self.ranges {
            Ranges::One(r) => std::slice::from_ref(r),
            Ranges::Many(v) => v,
        }
    }
    fn add(&mut self, from: usize, to: usize) {
        if let Ranges::One(r) = &mut self.ranges {
            if !self.sparse || (from <= r.1.saturating_add(GAP) && r.0 <= to.saturating_add(GAP)) {
                *r = (r.0.min(from), r.1.max(to));
                return;
            }
            self.ranges = Ranges::Many(vec![*r, (from, to)]);
        } else if let Ranges::Many(v) = &mut self.ranges {
            v.push((from, to));
        }
        let Ranges::Many(v) = &mut self.ranges else {
            unreachable!()
        };
        v.sort_unstable_by_key(|r| r.0);
        let mut n = 0;
        for i in 0..v.len() {
            let r = v[i];
            if n > 0 && r.0 <= v[n - 1].1.saturating_add(GAP) {
                v[n - 1].1 = v[n - 1].1.max(r.1);
            } else {
                v[n] = r;
                n += 1;
            }
        }
        v.truncate(n);
        if n > MAX_SPANS {
            let union = (v[0].0, v[n - 1].1);
            self.ranges = Ranges::One(union);
            self.sparse = false;
        } else if n == 1 {
            self.ranges = Ranges::One(v[0]);
        }
    }
}

pub(super) struct Pending {
    pub(super) shadow: Arc<Shadow>,
    spans: Spans,
}
impl Pending {
    pub(super) fn new(shadow: &Arc<Shadow>, from: usize, to: usize) -> Self {
        Self {
            shadow: shadow.clone(),
            spans: Spans::new(from, to, enabled() && shadow.unwatched()),
        }
    }
    pub(super) fn add(&mut self, from: usize, to: usize) {
        self.spans.add(from, to);
    }
    pub(super) fn ranges(&self) -> &[Span] {
        self.spans.ranges()
    }
    pub(super) fn note(&self) {
        let r = self.ranges();
        if r.len() < 2 {
            return;
        }
        static GROUPS: AtomicU64 = AtomicU64::new(0);
        static OMITTED: AtomicU64 = AtomicU64::new(0);
        let total: usize = r.iter().map(|(a, b)| b - a).sum();
        let gap = r[r.len() - 1].1 - r[0].0 - total;
        let omitted = OMITTED.fetch_add(gap as u64, Relaxed) + gap as u64;
        let n = GROUPS.fetch_add(1, Relaxed) + 1;
        if n.is_power_of_two() {
            tracing::info!(
                "sparse refresh: {n} split groups, {omitted} logical gap bytes omitted (not measured bus traffic)"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse_ranges_keep_every_requested_byte_and_bound_fragments() {
        let mut r = Spans::new(0, 16, true);
        for i in (1..=MAX_SPANS).rev() {
            r.add(i * GAP * 2, i * GAP * 2 + 16);
        }
        assert!(!r.sparse);
        assert_eq!(r.ranges(), &[(0, MAX_SPANS * GAP * 2 + 16)]);
        r.add(100 * GAP, 100 * GAP + 16);
        assert_eq!(r.ranges(), &[(0, 100 * GAP + 16)]);
        let mut r = Spans::new(4 * GAP, 4 * GAP + 16, true);
        r.add(0, 16);
        assert_eq!(r.ranges().len(), 2);
        r.add(GAP, 3 * GAP);
        assert_eq!(r.ranges(), &[(0, 4 * GAP + 16)]);
        let mut r = Spans::new(0, 16, false);
        r.add(4 * GAP, 4 * GAP + 16);
        assert_eq!(r.ranges(), &[(0, 4 * GAP + 16)]);
    }
}
