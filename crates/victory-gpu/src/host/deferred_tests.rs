    use super::Deferred;

    #[test]
    fn gpu_written_deduplicates_until_refresh_or_title_write() {
        let mut d = Deferred::default();
        let (s1, s2) = (0x1000usize, 0x2000usize);
        // Same (shadow, address, len) deduplicates.
        assert!(d.gpu_written.insert((s1, 0x1000, 64)));
        assert!(!d.gpu_written.insert((s1, 0x1000, 64)));
        // Distinct ranges or shadows stay distinct.
        assert!(d.gpu_written.insert((s1, 0x2000, 64)));
        assert!(d.gpu_written.insert((s2, 0x1000, 64)));

        // Intervening refresh on `s1` invalidates `s1`'s entries but keeps `s2`.
        d.gpu_written.retain(|k| k.0 != s1);
        assert!(d.gpu_written.insert((s1, 0x1000, 64)));
        assert!(!d.gpu_written.insert((s2, 0x1000, 64)));

        // Title copy/run clears all pending deduplication state.
        d.gpu_written.clear();
        assert!(d.gpu_written.insert((s2, 0x1000, 64)));
    }
