//! Virtual-to-physical descriptor paging for hosts with capped shader-visible
//! view heaps (NVIDIA `PORTABLE_VIEWS = 1_000_000`).
//!
//! Direct slots `[0..direct_count)` remain 1:1 with title heap slots; virtual
//! slots at or above `direct_count` are mapped dynamically via `SLOT_VIEW_MAP`
//! (`t252`) into a physical pool `[base .. base + capacity)` disjoint from both
//! the direct range and `DYNAMIC` table slots (`[base + capacity .. null_slot)`).

use std::collections::HashMap;

#[cfg(test)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct PagerStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub unmaps: u64,
    pub peak_resident: u32,
    pub highest_virt: u32,
}

#[cfg(not(test))]
#[derive(Default)]
struct PagerStats {
    hits: u64,
    misses: u64,
    evictions: u64,
    unmaps: u64,
    peak_resident: u32,
    highest_virt: u32,
}

pub(crate) struct VirtualDescriptorPager {
    base: u32,
    capacity: u32,
    null_slot: u32,
    virt_to_phys: HashMap<u32, u32>,
    phys_to_virt: Vec<Option<u32>>,
    used_bits: Vec<bool>,
    free: Vec<u32>,
    clock_hand: usize,
    stats: PagerStats,
}

impl VirtualDescriptorPager {
    pub(crate) fn new(base: u32, capacity: u32, null_slot: u32) -> Self {
        let cap = capacity as usize;
        let free = (0..capacity).rev().collect();
        Self {
            base,
            capacity,
            null_slot,
            virt_to_phys: HashMap::new(),
            phys_to_virt: vec![None; cap],
            used_bits: vec![false; cap],
            free,
            clock_hand: 0,
            stats: PagerStats::default(),
        }
    }

    /// Returns the physical heap slot for `virt` if currently resident and
    /// marks its second-chance reference bit.
    pub(crate) fn lookup(&mut self, virt: u32) -> Option<u32> {
        let &off = self.virt_to_phys.get(&virt)?;
        self.used_bits[off as usize] = true;
        self.stats.hits += 1;
        Some(self.base + off)
    }

    /// Resolves `virt` to a physical heap slot, allocating from the free pool
    /// or clock-evicting the oldest unreferenced resident slot if full.
    ///
    /// Returns `(phys_slot, evicted_virt, is_new_mapping)`.
    pub(crate) fn resolve_or_alloc(&mut self, virt: u32) -> (u32, Option<u32>, bool) {
        if let Some(phys) = self.lookup(virt) {
            return (phys, None, false);
        }
        self.stats.misses += 1;
        if virt > self.stats.highest_virt {
            self.stats.highest_virt = virt;
        }
        if self.capacity == 0 {
            return (self.null_slot, None, false);
        }

        if let Some(off) = self.free.pop() {
            self.virt_to_phys.insert(virt, off);
            self.phys_to_virt[off as usize] = Some(virt);
            self.used_bits[off as usize] = true;
            let resident = self.virt_to_phys.len() as u32;
            if resident > self.stats.peak_resident {
                self.stats.peak_resident = resident;
            }
            return (self.base + off, None, true);
        }

        let len = self.phys_to_virt.len();
        loop {
            let idx = self.clock_hand;
            self.clock_hand = (idx + 1) % len;
            if self.used_bits[idx] {
                self.used_bits[idx] = false;
                continue;
            }
            let evicted = self.phys_to_virt[idx].take();
            if let Some(old_virt) = evicted {
                self.virt_to_phys.remove(&old_virt);
            }
            let off = idx as u32;
            self.virt_to_phys.insert(virt, off);
            self.phys_to_virt[idx] = Some(virt);
            self.used_bits[idx] = true;
            self.stats.evictions += 1;
            return (self.base + off, evicted, true);
        }
    }

    /// Unmaps `virt`, returning its physical slot to the free list if resident.
    pub(crate) fn unmap(&mut self, virt: u32) -> Option<u32> {
        let off = self.virt_to_phys.remove(&virt)?;
        self.phys_to_virt[off as usize] = None;
        self.used_bits[off as usize] = false;
        self.free.push(off);
        self.stats.unmaps += 1;
        Some(self.base + off)
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> PagerStats {
        self.stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alloc_hit_and_unmap_reuse_physical_slot() {
        let mut pager = VirtualDescriptorPager::new(600_000, 4, 999_999);
        let (p0, ev0, new0) = pager.resolve_or_alloc(800_000);
        let (p1, ev1, new1) = pager.resolve_or_alloc(900_000);
        assert_eq!((p0, ev0, new0), (600_000, None, true));
        assert_eq!((p1, ev1, new1), (600_001, None, true));

        let (p0_hit, ev_hit, new_hit) = pager.resolve_or_alloc(800_000);
        assert_eq!((p0_hit, ev_hit, new_hit), (600_000, None, false));
        assert_eq!(pager.stats().hits, 1);

        assert_eq!(pager.unmap(800_000), Some(600_000));
        assert_eq!(pager.lookup(800_000), None);
        let (reused, ev2, new2) = pager.resolve_or_alloc(1_200_000);
        assert_eq!((reused, ev2, new2), (600_000, None, true));
        assert_eq!(pager.stats().unmaps, 1);
    }

    #[test]
    fn clock_eviction_preserves_hot_slots_and_reports_evicted_virt() {
        let mut pager = VirtualDescriptorPager::new(100, 3, 999);
        let (p0, _, _) = pager.resolve_or_alloc(1000);
        let (p1, _, _) = pager.resolve_or_alloc(1001);
        let (p2, _, _) = pager.resolve_or_alloc(1002);
        assert_eq!((p0, p1, p2), (100, 101, 102));

        // Clear second-chance bit on 1000/1001/1002 via one miss cycle? First keep 1000 hot:
        // Initial allocations set used=true; first clock sweep clears 1000 (off 0), 1001 (off 1),
        // 1002 (off 2), then evicts 1000. Re-touch 1000 after inserting 1003 so 1000 gets second chance.
        let (p3, ev3, _) = pager.resolve_or_alloc(1003);
        assert_eq!((p3, ev3), (100, Some(1000)));
        assert_eq!(pager.lookup(1000), None);

        // Touch 1001 so it gets second-chance protection on the next eviction pass;
        // clock hand is at 1 (1001 -> clear bit, skip), then 2 (1002 -> bit already false -> evict 1002!).
        assert_eq!(pager.lookup(1001), Some(101));
        let (p4, ev4, _) = pager.resolve_or_alloc(1004);
        assert_eq!((p4, ev4), (102, Some(1002)));
        assert_eq!(pager.lookup(1001), Some(101));
        assert_eq!(pager.stats().evictions, 2);
    }

    #[test]
    fn high_virtual_slots_stay_inside_pool_and_disjoint_from_dynamic() {
        let direct = 606_783u32;
        let pool = 131_072u32;
        let dynamic_base = direct + pool; // 737_855
        let null_slot = 999_999u32;
        let mut pager = VirtualDescriptorPager::new(direct, pool, null_slot);

        for v in [700_000u32, 950_000, 1_200_000, 1_507_328, 2_000_000] {
            let (phys, _, is_new) = pager.resolve_or_alloc(v);
            assert!(is_new);
            assert!(phys >= direct && phys < dynamic_base, "phys {phys} outside [{direct}..{dynamic_base})");
            assert!(phys < null_slot);
        }
        assert_eq!(pager.stats().highest_virt, 2_000_000);
        assert_eq!(pager.stats().peak_resident, 5);
    }
}
