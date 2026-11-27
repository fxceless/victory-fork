//! Exact conventional CBV reuse within one recording, never across lists.
use super::*;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CBV_REUSE_UPLOAD").is_some_and(|v| v == "1"))
}

#[derive(Default)]
pub(super) struct Cache {
    entries: std::collections::HashMap<(u64, u32), u64>,
    scanned: usize,
}

impl Cache {
    fn lookup(&mut self, copies: &[Step], src: u64, size: u32) -> Option<u64> {
        if self.scanned > copies.len()
            || copies[self.scanned..].iter().any(|s| !matches!(s, Step::Fill(..)))
        {
            self.entries.clear();
        }
        self.scanned = copies.len();
        self.entries.get(&(src, size)).copied()
    }

    fn remember(&mut self, src: u64, size: u32, gpu: u64, steps: usize) {
        if self.entries.len() >= 4096 {
            self.entries.clear();
        }
        self.entries.insert((src, size), gpu);
        self.scanned = steps;
    }
}

fn eligible(src: u64, size: u32) -> bool {
    src != 0 && size != 0 && size <= 65_536 && size.is_multiple_of(256)
        && src.checked_add(u64::from(size)).is_some()
}

impl HostList {
    /// Preserve the original scratch path for malformed or unaligned sizes.
    /// Card fills are still handled by the caller before this function.
    pub(crate) fn table_cbv_upload(&self, src: u64, size: u32) -> Option<u64> {
        let reuse = enabled() && eligible(src, size);
        if reuse {
            let mut d = self.deferred();
            let d = &mut *d;
            if let Some(gpu) = d.cbv_uploads.lookup(&d.copies, src, size) {
                return Some(gpu);
            }
        }
        let (gpu, to) = self.scratch(u64::from(size))?;
        self.defer_fill(to, src as *const u8, size as usize);
        if reuse {
            let mut d = self.deferred();
            let steps = d.copies.len();
            d.cbv_uploads.remember(src, size, gpu, steps);
        }
        Some(gpu)
    }
}

#[test]
fn exact_table_cbv_cache_stops_at_non_fill_steps() {
    let mut c = Cache::default();
    let mut steps = vec![Step::Fill(0x2000, 0x1000, 256)];
    c.remember(0x1000, 256, 0x3000, steps.len());
    steps.push(Step::Fill(0x4000, 0x5000, 256));
    assert_eq!(c.lookup(&steps, 0x1000, 256), Some(0x3000));
    assert_eq!(c.lookup(&steps, 0x1000, 512), None);
    assert_eq!(c.lookup(&steps, 0x1001, 256), None);
    steps.push(Step::Copy(0x1000, 0x2000, 256));
    assert_eq!(c.lookup(&steps, 0x1000, 256), None);
    c.remember(0x1000, 256, 0x3000, steps.len());
    steps.push(Step::Run(Box::new(|| {}), None));
    assert_eq!(c.lookup(&steps, 0x1000, 256), None);
    c.remember(0x1000, 256, 0x3000, steps.len());
    steps.push(Step::Host(Box::new(|| {})));
    assert_eq!(c.lookup(&steps, 0x1000, 256), None);
    c.remember(0x1000, 256, 0x3000, steps.len());
    assert_eq!(c.lookup(&[], 0x1000, 256), None);
}

#[test]
fn table_cbv_reuse_only_accepts_bounded_aligned_sizes() {
    assert!(eligible(0x1000, 256));
    assert!(eligible(0x1000, 65_536));
    assert!(!eligible(0, 256));
    assert!(!eligible(0x1000, 0));
    assert!(!eligible(0x1000, 4));
    assert!(!eligible(0x1000, 65_792));
    assert!(!eligible(u64::MAX - 127, 256));
}
