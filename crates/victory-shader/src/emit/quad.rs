use super::*;

impl Lower<'_> {
    /// `x` of lane `src` (in this lane's quad) through quad reads, whose
    /// lane must be a constant.
    pub(super) fn quad_pick(&mut self, x: V, src: V) -> V {
        self.s.flags |= flags::WAVE_OPS;
        let c = self.cur;
        let three = self.s.m.i32(3);
        let q = self.s.m.bin(c, Bin::And, src, three);
        let mut r = x;
        for n in 0..4 {
            let kn = self.s.m.i32(n);
            let read = self.s.op(
                c,
                "quadReadLaneAt.i32",
                Attr::NoUnwind,
                self.s.t.i32,
                122,
                &[x, kn],
            );
            let is = self.s.m.cmp(c, pred::EQ, q, kn);
            r = self.s.m.select(c, is, read, r);
        }
        r
    }

    pub(super) fn pixel_quad_mask(&mut self, input: Bool) -> Result<V, Why> {
        let c = self.cur;
        let on = self.boolean(input, c)?;
        if matches!(input, Bool::TRUE | Bool::FALSE) {
            return Ok(on);
        }
        self.s.flags |= flags::WAVE_OPS;
        let bits = self.s.m.cast(c, Cast::ZExt, on, self.s.t.i32);
        let k = self.s.m.i8(0);
        let other = self.s.op(c, "quadOp.i32", Attr::NoUnwind, self.s.t.i32, 123, &[bits, k]);
        let pair = self.s.m.bin(c, Bin::Or, bits, other);
        let k = self.s.m.i8(1);
        let other = self.s.op(c, "quadOp.i32", Attr::NoUnwind, self.s.t.i32, 123, &[pair, k]);
        let all = self.s.m.bin(c, Bin::Or, pair, other);
        let zero = self.s.m.i32(0);
        Ok(self.s.m.cmp(c, pred::NE, all, zero))
    }

    /// `x` of lane `src` of this lane's quad, where lane `q` of each quad
    /// reads lane `pattern[q]` of it (`src` is that lane's index in the
    /// wave), for a quad-local swizzle or DPP `quad_perm`: in a pixel
    /// shader and in a mesh shader through quad reads, else through a wave
    /// read. In a pixel shader the quad's other pixels at a triangle's edge
    /// are helper lanes, which wave operations leave out (DXIL without
    /// `WaveOpsIncludeHelperLanes`): a wave read of one gives what the host
    /// likes (0 on AMD), so a derivative taken by hand was the whole value
    /// there, the texture read that used it took its smallest level, and
    /// triangle edges showed as lines on car paint, glass and skin (from
    /// Belobug 03629f3f). Quad reads see helper lanes. A pattern that is a
    /// broadcast or a swap across x, y or the diagonal takes one quad read.
    pub(super) fn quad_read(&mut self, x: V, src: V, pattern: [u32; 4]) -> V {
        let quads = self.s.mesh.is_some() || self.s.pixel;
        if !quads {
            return self.read_lane(x, src);
        }
        #[cfg(test)]
        {
            self.s.quad_reads += 1;
        }
        let p = pattern.map(|k| k & 3);
        if p == [0, 1, 2, 3] {
            return x;
        }
        let c = self.cur;
        let i32t = self.s.t.i32;
        if p.iter().all(|&k| k == p[0]) {
            self.s.flags |= flags::WAVE_OPS;
            let k = self.s.m.i32(p[0]);
            return self.s.op(c, "quadReadLaneAt.i32", Attr::NoUnwind, i32t, 122, &[x, k]);
        }
        // QuadOpKind: 0 across x (lane ^ 1), 1 across y (^ 2), 2 diagonal (^ 3).
        for (kind, flip) in [(0u8, 1u32), (1, 2), (2, 3)] {
            if (0..4).all(|q| p[q as usize] == q ^ flip) {
                self.s.flags |= flags::WAVE_OPS;
                let k = self.s.m.i8(kind);
                return self.s.op(c, "quadOp.i32", Attr::NoUnwind, i32t, 123, &[x, k]);
            }
        }
        self.quad_pick(x, src)
    }
}
