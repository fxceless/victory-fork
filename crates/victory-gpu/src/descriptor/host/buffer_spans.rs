//! Maintain the exact enclosing range for each backing buffer from its live slots.
use std::collections::{BTreeSet, HashMap};

struct Span<T> {
    copy: T,
    starts: BTreeSet<(u64, u32)>,
    ends: BTreeSet<(u64, u32)>,
}

pub(crate) struct BufferSpans<T> {
    copies: HashMap<usize, Span<T>>,
}

impl<T> Default for BufferSpans<T> {
    fn default() -> Self {
        Self { copies: HashMap::new() }
    }
}

impl<T: Clone> BufferSpans<T> {
    pub(crate) fn insert(&mut self, key: usize, slot: u32, copy: &T, from: u64, to: u64) {
        let span = self.copies.entry(key).or_insert_with(|| Span {
            copy: copy.clone(),
            starts: BTreeSet::new(),
            ends: BTreeSet::new(),
        });
        span.starts.insert((from, slot));
        span.ends.insert((to, slot));
    }

    /// False means the caller must rebuild from its authoritative slot map.
    pub(crate) fn remove(&mut self, key: usize, slot: u32, from: u64, to: u64) -> bool {
        let Some(span) = self.copies.get_mut(&key) else {
            return false;
        };
        let start = span.starts.remove(&(from, slot));
        let end = span.ends.remove(&(to, slot));
        if span.starts.is_empty() && span.ends.is_empty() {
            self.copies.remove(&key);
        }
        start && end
    }

    /// Exactly the same per-buffer unions as walking all live descriptors.
    pub(crate) fn snapshot(&self) -> HashMap<usize, (T, u64, u64)> {
        self.copies.iter().map(|(&key, span)| {
            let &(from, _) = span.starts.first().expect("nonempty span");
            let &(to, _) = span.ends.last().expect("nonempty span");
            (key, (span.copy.clone(), from, to))
        }).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn span_index_matches_full_scan_after_slot_moves_and_removals() {
        let mut index = BufferSpans::default();
        let mut slots: HashMap<u32, (usize, u64, u64)> = HashMap::new();
        for n in 0..2048u32 {
            let slot = (n * 17) % 97;
            if let Some((key, from, to)) = slots.remove(&slot) {
                assert!(index.remove(key, slot, from, to));
            }
            if n % 5 != 0 {
                let key = ((n * 13) % 11) as usize;
                let from = u64::from(n % 7) * 64;
                let to = from + u64::from((n % 9) + 1) * 32;
                slots.insert(slot, (key, from, to));
                index.insert(key, slot, &key, from, to);
            }
            let mut expected = HashMap::new();
            for &(key, from, to) in slots.values() {
                let span = expected.entry(key).or_insert((key, from, to));
                span.1 = span.1.min(from);
                span.2 = span.2.max(to);
            }
            assert_eq!(index.snapshot(), expected);
        }
        for (slot, (key, from, to)) in slots {
            assert!(index.remove(key, slot, from, to));
        }
        assert!(index.snapshot().is_empty());
        assert!(!index.remove(0, 0, 0, 1));
    }

    #[test]
    fn duplicate_bounds_are_kept_until_the_last_slot_is_removed() {
        let mut index = BufferSpans::default();
        index.insert(1, 0, &1, 16, 64);
        index.insert(1, 1, &1, 16, 64);
        index.insert(1, 2, &1, 32, 48);
        assert!(index.remove(1, 0, 16, 64));
        assert_eq!(index.snapshot()[&1], (1, 16, 64));
        assert!(index.remove(1, 1, 16, 64));
        assert_eq!(index.snapshot()[&1], (1, 32, 48));
        assert!(index.remove(1, 2, 32, 48));
        assert!(index.snapshot().is_empty());
    }
}
