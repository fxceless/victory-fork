    use std::sync::Mutex;
    use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

    /// The fill waited for now: source, length, its bytes when it blocked.
    static PENDING: Mutex<Option<(usize, usize, Vec<u8>)>> = Mutex::new(None);
    /// Fills that blocked by (source page, length): how often.
    static SEEN: Mutex<Vec<(usize, usize, u64)>> = Mutex::new(Vec::new());
    /// Fills checked after their wait; those whose bytes had changed; the
    /// bytes that had.
    static CHECKED: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

    /// A fill of `len` bytes at `src` blocks on `written`.
    pub(super) fn blocked(written: &[(usize, usize)], src: usize, len: usize) {
        let at = written.partition_point(|r| r.1 <= src);
        let touched = written.get(at).copied().unwrap_or((0, 0));
        let key = (src & !0xfff, len);
        let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let known = seen.iter().position(|s| (s.0, s.1) == key);
        match known {
            Some(k) => seen[k].2 += 1,
            None if seen.len() < 256 => {
                seen.push((key.0, key.1, 1));
                if seen.len() <= 12 {
                    tracing::info!(
                        "a fill of upload memory waits for flights: {len} bytes at {src:#x}; the range shaders may write that it                          touches: {:#x}+{:#x} ({} ranges in the air); frame {}",
                        touched.0,
                        touched.1 - touched.0,
                        written.len(),
                        crate::frame::current()
                    );
                }
            }
            None => {}
        }
        drop(seen);
        let mut bytes = vec![0u8; len.min(1 << 20)];
        if super::read_into(src as *const u8, &mut bytes) {
            *PENDING.lock().unwrap_or_else(|e| e.into_inner()) = Some((src, len, bytes));
        }
    }

    /// The fill of `len` bytes at `src` runs now.
    pub(super) fn runs(src: usize, len: usize) {
        let mut pending = PENDING.lock().unwrap_or_else(|e| e.into_inner());
        if !pending.as_ref().is_some_and(|p| (p.0, p.1) == (src, len)) {
            return;
        }
        let Some((_, _, before)) = pending.take() else { return };
        drop(pending);
        let mut now = vec![0u8; before.len()];
        if !super::read_into(src as *const u8, &mut now) {
            return;
        }
        let differ = before.iter().zip(&now).filter(|(a, b)| a != b).count() as u64;
        let n = CHECKED[0].fetch_add(1, Relaxed) + 1;
        if differ > 0 {
            let bad = CHECKED[1].fetch_add(1, Relaxed) + 1;
            CHECKED[2].fetch_add(differ, Relaxed);
            if bad <= 12 {
                let first = before.iter().zip(&now).position(|(a, b)| a != b).unwrap_or(0);
                tracing::info!(
                    "a fill of upload memory that waited for flights read other bytes after the wait: {differ} of {len} bytes at                      {src:#x} changed, the first at +{first:#x} (check {n}, frame {})",
                    crate::frame::current()
                );
            }
        }
    }

    /// The title buffer copy waited for now: destination, source, length,
    /// the bytes of both when it blocked.
    static PENDING_COPY: Mutex<Option<(usize, usize, usize, Vec<u8>, Vec<u8>)>> = Mutex::new(None);
    /// Copies checked after their wait; those whose source had changed;
    /// those whose destination had.
    static COPIES: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
    /// Copies that blocked by (source page, destination page, length).
    static SEEN_COPIES: Mutex<Vec<(usize, usize, usize, u64)>> = Mutex::new(Vec::new());

    /// A title buffer copy of `len` bytes from `src` to `dst` blocks on
    /// `written`.
    pub(super) fn blocked_copy(written: &[(usize, usize)], dst: usize, src: usize, len: usize) {
        let by_src = super::touches(written, src, src + len);
        let by_dst = super::touches(written, dst, dst + len);
        let key = (src & !0xfff, dst & !0xfff, len);
        let mut seen = SEEN_COPIES.lock().unwrap_or_else(|e| e.into_inner());
        let known = seen.iter().position(|s| (s.0, s.1, s.2) == key);
        match known {
            Some(k) => seen[k].3 += 1,
            None if seen.len() < 256 => {
                seen.push((key.0, key.1, key.2, 1));
                if seen.len() <= 16 {
                    let range = |at: usize| {
                        let k = written.partition_point(|r| r.1 <= at);
                        written.get(k).copied().unwrap_or((0, 0))
                    };
                    let (s, d) = (range(src), range(dst));
                    tracing::info!(
                        "a title buffer copy waits for flights: {len} bytes from {src:#x} to {dst:#x}; source in a range shaders may                          write: {by_src} ({:#x}+{:#x}); destination: {by_dst} ({:#x}+{:#x}); {} ranges in the air; frame {}",
                        s.0,
                        s.1 - s.0,
                        d.0,
                        d.1 - d.0,
                        written.len(),
                        crate::frame::current()
                    );
                }
            }
            None => {}
        }
        drop(seen);
        let n = len.min(1 << 20);
        let (mut a, mut b) = (vec![0u8; n], vec![0u8; n]);
        if super::read_into(src as *const u8, &mut a) && super::read_into(dst as *const u8, &mut b) {
            *PENDING_COPY.lock().unwrap_or_else(|e| e.into_inner()) = Some((dst, src, len, a, b));
        }
    }

    /// The title buffer copy runs now.
    pub(super) fn copy_runs(dst: usize, src: usize, len: usize) {
        let mut pending = PENDING_COPY.lock().unwrap_or_else(|e| e.into_inner());
        if !pending.as_ref().is_some_and(|p| (p.0, p.1, p.2) == (dst, src, len)) {
            return;
        }
        let Some((_, _, _, src_before, dst_before)) = pending.take() else { return };
        drop(pending);
        let (mut a, mut b) = (vec![0u8; src_before.len()], vec![0u8; dst_before.len()]);
        if !super::read_into(src as *const u8, &mut a) || !super::read_into(dst as *const u8, &mut b) {
            return;
        }
        let n = COPIES[0].fetch_add(1, Relaxed) + 1;
        let (src_changed, dst_changed) = (a != src_before, b != dst_before);
        if src_changed {
            COPIES[1].fetch_add(1, Relaxed);
        }
        if dst_changed {
            COPIES[2].fetch_add(1, Relaxed);
        }
        if (src_changed || dst_changed) && COPIES[1].load(Relaxed) + COPIES[2].load(Relaxed) <= 16 {
            let differ = |x: &[u8], y: &[u8]| x.iter().zip(y).filter(|(p, q)| p != q).count();
            tracing::info!(
                "a title buffer copy that waited for flights found other bytes after the wait: {len} bytes from {src:#x} to {dst:#x};                  {} source bytes changed, {} destination bytes changed (check {n}, frame {})",
                differ(&a, &src_before),
                differ(&b, &dst_before),
                crate::frame::current()
            );
        }
    }

    /// With the frame line.
    pub(super) fn line() {
        let copies = COPIES[0].load(Relaxed);
        if copies > 0 {
            let seen = SEEN_COPIES.lock().unwrap_or_else(|e| e.into_inner());
            let mut most: Vec<_> = seen.clone();
            most.sort_unstable_by_key(|s| std::cmp::Reverse(s.3));
            let most: Vec<String> =
                most.iter().take(6).map(|s| format!("{:#x}->{:#x}+{:#x} {}x", s.0, s.1, s.2, s.3)).collect();
            tracing::info!(
                "title buffer copies that waited for flights, since the start: {copies} checked after their wait; the source had                  other bytes after it {} times, the destination {} times; {} distinct copies, the most frequent: {}",
                COPIES[1].load(Relaxed),
                COPIES[2].load(Relaxed),
                seen.len(),
                most.join(", ")
            );
        }
        let n = CHECKED[0].load(Relaxed);
        if n == 0 {
            return;
        }
        let seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
        let mut most: Vec<_> = seen.clone();
        most.sort_unstable_by_key(|s| std::cmp::Reverse(s.2));
        let most: Vec<String> = most.iter().take(6).map(|s| format!("{:#x}+{:#x} {}x", s.0, s.1, s.2)).collect();
        tracing::info!(
            "fills of upload memory that waited for flights, since the start: {n} checked after their wait, {} of them read other              bytes than before it ({} bytes in all); {} distinct fills (page, length), the most frequent: {}",
            CHECKED[1].load(Relaxed),
            CHECKED[2].load(Relaxed),
            seen.len(),
            most.join(", ")
        );
    }
