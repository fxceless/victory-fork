//! Software IEEE-754 binary64 (`f64`) addition and wave reduction over
//! 32-bit integer DXIL operations (`i32` / `i1`), used when the host GPU lacks
//! `DoublePrecisionFloatShaderOps`.

use crate::emit::bitcode::{Attr, Bin, Cast, V, pred};
use crate::emit::shader::{Shader, flags};
use super::{Lower, Why};

struct F64Asm<'a> {
    s: &'a mut Shader,
    c: usize,
}

impl<'a> F64Asm<'a> {
    fn k(&mut self, v: u32) -> V {
        self.s.m.i32(v)
    }

    fn bin(&mut self, op: Bin, a: V, b: V) -> V {
        self.s.m.bin(self.c, op, a, b)
    }

    fn cmp(&mut self, p: u8, a: V, b: V) -> V {
        self.s.m.cmp(self.c, p, a, b)
    }

    fn sel(&mut self, cond: V, t: V, f: V) -> V {
        self.s.m.select(self.c, cond, t, f)
    }

    fn sel64(&mut self, cond: V, t: (V, V), f: (V, V)) -> (V, V) {
        (self.sel(cond, t.0, f.0), self.sel(cond, t.1, f.1))
    }

    fn zext(&mut self, b: V) -> V {
        self.s.m.cast(self.c, Cast::ZExt, b, self.s.t.i32)
    }

    fn ffbh(&mut self, x: V) -> V {
        self.s
            .op(self.c, "unaryBits.i32", Attr::ReadNone, self.s.t.i32, 33, &[x])
    }

    fn add64(&mut self, a: (V, V), b: (V, V)) -> (V, V) {
        let lo = self.bin(Bin::Add, a.0, b.0);
        let carry = self.cmp(pred::IULT, lo, a.0);
        let c = self.zext(carry);
        let hi = self.bin(Bin::Add, a.1, b.1);
        let hi = self.bin(Bin::Add, hi, c);
        (lo, hi)
    }

    fn sub64(&mut self, a: (V, V), b: (V, V)) -> (V, V) {
        let lo = self.bin(Bin::Sub, a.0, b.0);
        let borrow = self.cmp(pred::IULT, a.0, b.0);
        let br = self.zext(borrow);
        let hi = self.bin(Bin::Sub, a.1, b.1);
        let hi = self.bin(Bin::Sub, hi, br);
        (lo, hi)
    }

    fn lt64(&mut self, a: (V, V), b: (V, V)) -> V {
        let hi_lt = self.cmp(pred::IULT, a.1, b.1);
        let hi_eq = self.cmp(pred::EQ, a.1, b.1);
        let lo_lt = self.cmp(pred::IULT, a.0, b.0);
        let tie = self.bin(Bin::And, hi_eq, lo_lt);
        self.bin(Bin::Or, hi_lt, tie)
    }

    /// Shifts a 64-bit `(lo, hi)` pair left by `s` (`0 <= s < 64`).
    /// Keeps every 32-bit shift count in `0..=31` so DXIL never observes poison.
    fn shl64(&mut self, a: (V, V), s: V) -> (V, V) {
        let (k0, k1, k31, k32) = (self.k(0), self.k(1), self.k(31), self.k(32));
        let s_mod = self.bin(Bin::And, s, k31);
        let inv31 = self.bin(Bin::Sub, k31, s_mod);
        let lo_sh = self.bin(Bin::Shl, a.0, s_mod);
        let hi_sh = self.bin(Bin::Shl, a.1, s_mod);
        let carry = self.bin(Bin::LShr, a.0, k1);
        let carry = self.bin(Bin::LShr, carry, inv31);
        let small_hi = self.bin(Bin::Or, hi_sh, carry);
        let ge32 = self.cmp(pred::IUGE, s, k32);
        (self.sel(ge32, k0, lo_sh), self.sel(ge32, lo_sh, small_hi))
    }

    /// Shifts `(lo, hi)` right by `s` (`s >= 0`) and ORs any shifted-out non-zero
    /// bits into bit 0 so round-to-nearest-even sees an exact sticky bit.
    fn shr64_sticky(&mut self, a: (V, V), s: V) -> (V, V) {
        let (k0, k1, k31, k32, k64) =
            (self.k(0), self.k(1), self.k(31), self.k(32), self.k(64));
        let s_mod = self.bin(Bin::And, s, k31);
        let inv31 = self.bin(Bin::Sub, k31, s_mod);

        let hi_sh = self.bin(Bin::LShr, a.1, s_mod);
        let lo_sh = self.bin(Bin::LShr, a.0, s_mod);
        let carry = self.bin(Bin::Shl, a.1, k1);
        let carry = self.bin(Bin::Shl, carry, inv31);
        let small_lo = self.bin(Bin::Or, lo_sh, carry);
        let lost_lo = self.bin(Bin::Shl, a.0, k1);
        let lost_lo = self.bin(Bin::Shl, lost_lo, inv31);
        let sticky_small = self.cmp(pred::NE, lost_lo, k0);

        // When 32 <= s < 64, `carry` holds the low `s_mod` bits shifted out of `a.1`.
        let lost_mid = self.bin(Bin::Or, a.0, carry);
        let sticky_mid = self.cmp(pred::NE, lost_mid, k0);

        let any_all = self.bin(Bin::Or, a.0, a.1);
        let sticky_big = self.cmp(pred::NE, any_all, k0);

        let ge32 = self.cmp(pred::IUGE, s, k32);
        let ge64 = self.cmp(pred::IUGE, s, k64);

        let lo = self.sel(ge32, hi_sh, small_lo);
        let lo = self.sel(ge64, k0, lo);
        let hi = self.sel(ge32, k0, hi_sh);

        let sticky = self.sel(ge32, sticky_mid, sticky_small);
        let sticky = self.sel(ge64, sticky_big, sticky);
        let sticky = self.zext(sticky);
        let lo = self.bin(Bin::Or, lo, sticky);
        (lo, hi)
    }

    /// Unpacks `x` (`(lo, hi)`) into `(effective_exp, sig56_lo, sig56_hi)` with
    /// the significand positioned at bits `55..3` and GRS guard bits `2..0` zero.
    fn unpack_sig56(&mut self, lo: V, hi: V) -> (V, (V, V)) {
        let (k0, k1, k3, k20, k29) = (self.k(0), self.k(1), self.k(3), self.k(20), self.k(29));
        let k7ff = self.k(0x7FF);
        let frac_mask = self.k(0x000F_FFFF);
        let imp_bit = self.k(0x0010_0000);

        let exp_raw = self.bin(Bin::LShr, hi, k20);
        let exp_raw = self.bin(Bin::And, exp_raw, k7ff);
        let is_norm = self.cmp(pred::NE, exp_raw, k0);
        // Subnormals carry effective biased exponent 1 (`2^-1022`) without an implicit 1.
        let eff_exp = self.sel(is_norm, exp_raw, k1);
        let imp = self.sel(is_norm, imp_bit, k0);
        let frac_hi = self.bin(Bin::And, hi, frac_mask);
        let sig_hi = self.bin(Bin::Or, frac_hi, imp);

        let hi_s = self.bin(Bin::Shl, sig_hi, k3);
        let carry = self.bin(Bin::LShr, lo, k29);
        let hi_s = self.bin(Bin::Or, hi_s, carry);
        let lo_s = self.bin(Bin::Shl, lo, k3);
        (eff_exp, (lo_s, hi_s))
    }

    /// Rounds normalized `(norm_lo, norm_hi)` (at bits `55..3` with GRS in `2..0`)
    /// to nearest-even and packs `(lo, hi)` for finite non-zero values.
    fn round_and_pack(
        &mut self,
        sign: V,
        norm: (V, V),
        norm_exp: V,
    ) -> (V, (V, V)) {
        let (k0, k1, k3, k4, k7, k20, k29) = (
            self.k(0),
            self.k(1),
            self.k(3),
            self.k(4),
            self.k(7),
            self.k(20),
            self.k(29),
        );
        let sig_lo = self.bin(Bin::LShr, norm.0, k3);
        let sig_carry = self.bin(Bin::Shl, norm.1, k29);
        let sig_lo = self.bin(Bin::Or, sig_lo, sig_carry);
        let sig_hi = self.bin(Bin::LShr, norm.1, k3);

        let grs = self.bin(Bin::And, norm.0, k7);
        let gt_half = self.cmp(pred::IUGT, grs, k4);
        let eq_half = self.cmp(pred::EQ, grs, k4);
        let odd_bit = self.bin(Bin::And, sig_lo, k1);
        let odd = self.cmp(pred::NE, odd_bit, k0);
        let tie_even = self.bin(Bin::And, eq_half, odd);
        let round_up = self.bin(Bin::Or, gt_half, tie_even);
        let round_inc = self.zext(round_up);
        let rnd = self.add64((sig_lo, sig_hi), (round_inc, k0));

        let ov_mask = self.k(0x0020_0000);
        let ov_bits = self.bin(Bin::And, rnd.1, ov_mask);
        let rnd_ov = self.cmp(pred::NE, ov_bits, k0);
        let exp_p1 = self.bin(Bin::Add, norm_exp, k1);
        let exp_cand = self.sel(rnd_ov, exp_p1, norm_exp);

        let norm_mask = self.k(0x0030_0000);
        let norm_bits = self.bin(Bin::And, rnd.1, norm_mask);
        let is_norm = self.cmp(pred::NE, norm_bits, k0);
        let final_exp = self.sel(is_norm, exp_cand, k0);

        let frac_mask = self.k(0x000F_FFFF);
        let frac_hi = self.bin(Bin::And, rnd.1, frac_mask);
        let frac_hi = self.sel(rnd_ov, k0, frac_hi);
        let frac_lo = self.sel(rnd_ov, k0, rnd.0);

        let exp_sh = self.bin(Bin::Shl, final_exp, k20);
        let hi_body = self.bin(Bin::Or, exp_sh, frac_hi);
        let hi = self.bin(Bin::Or, sign, hi_body);

        let k7ff = self.k(0x7FF);
        let exp_ov = self.cmp(pred::IUGE, final_exp, k7ff);
        (exp_ov, (frac_lo, hi))
    }

    fn add(&mut self, lo_a: V, hi_a: V, lo_b: V, hi_b: V) -> (V, V) {
        let (k0, k1, k8, k20, k24) = (self.k(0), self.k(1), self.k(8), self.k(20), self.k(24));
        let sign_mask = self.k(0x8000_0000);
        let abs_mask = self.k(0x7FFF_FFFF);
        let frac_mask = self.k(0x000F_FFFF);
        let k7ff = self.k(0x7FF);

        let sign_a = self.bin(Bin::And, hi_a, sign_mask);
        let sign_b = self.bin(Bin::And, hi_b, sign_mask);
        let abs_hi_a = self.bin(Bin::And, hi_a, abs_mask);
        let abs_hi_b = self.bin(Bin::And, hi_b, abs_mask);

        // Special cases: NaN and Inf (`exp == 0x7FF`). Canonical qNaN keeps addition commutative.
        let exp_a = self.bin(Bin::LShr, abs_hi_a, k20);
        let exp_b = self.bin(Bin::LShr, abs_hi_b, k20);
        let a_exp_all = self.cmp(pred::EQ, exp_a, k7ff);
        let b_exp_all = self.cmp(pred::EQ, exp_b, k7ff);
        let a_fhi = self.bin(Bin::And, hi_a, frac_mask);
        let b_fhi = self.bin(Bin::And, hi_b, frac_mask);
        let a_fany = self.bin(Bin::Or, a_fhi, lo_a);
        let b_fany = self.bin(Bin::Or, b_fhi, lo_b);
        let a_fnz = self.cmp(pred::NE, a_fany, k0);
        let b_fnz = self.cmp(pred::NE, b_fany, k0);

        let a_nan = self.bin(Bin::And, a_exp_all, a_fnz);
        let b_nan = self.bin(Bin::And, b_exp_all, b_fnz);
        let b_true = self.s.m.bool(true);
        let a_fnz_not = self.bin(Bin::Xor, a_fnz, b_true);
        let b_fnz_not = self.bin(Bin::Xor, b_fnz, b_true);
        let a_inf = self.bin(Bin::And, a_exp_all, a_fnz_not);
        let b_inf = self.bin(Bin::And, b_exp_all, b_fnz_not);
        let diff_sign = self.cmp(pred::NE, sign_a, sign_b);
        let inf_both = self.bin(Bin::And, a_inf, b_inf);
        let inf_cancel = self.bin(Bin::And, inf_both, diff_sign);
        let nan_inputs = self.bin(Bin::Or, a_nan, b_nan);
        let is_nan_res = self.bin(Bin::Or, nan_inputs, inf_cancel);
        let any_inf = self.bin(Bin::Or, a_inf, b_inf);
        let inf_sign = self.sel(a_inf, sign_a, sign_b);

        // Order operands so `|big| >= |sml|` before subtraction.
        let a_lt_b = self.lt64((lo_a, abs_hi_a), (lo_b, abs_hi_b));
        let big = self.sel64(a_lt_b, (lo_b, hi_b), (lo_a, hi_a));
        let sml = self.sel64(a_lt_b, (lo_a, hi_a), (lo_b, hi_b));

        let big_sign = self.bin(Bin::And, big.1, sign_mask);
        let sml_sign = self.bin(Bin::And, sml.1, sign_mask);
        let eff_sub = self.cmp(pred::NE, big_sign, sml_sign);

        let (big_exp, big_s) = self.unpack_sig56(big.0, big.1);
        let (sml_exp, sml_s) = self.unpack_sig56(sml.0, sml.1);
        let exp_diff = self.bin(Bin::Sub, big_exp, sml_exp);
        let sml_aligned = self.shr64_sticky(sml_s, exp_diff);

        let sum = self.add64(big_s, sml_aligned);
        let diff = self.sub64(big_s, sml_aligned);
        let raw = self.sel64(eff_sub, diff, sum);

        let raw_any = self.bin(Bin::Or, raw.0, raw.1);
        let is_zero = self.cmp(pred::EQ, raw_any, k0);

        let ov_bit_mask = self.k(0x0100_0000);
        let ov_bits = self.bin(Bin::And, raw.1, ov_bit_mask);
        let overflow_bit = self.cmp(pred::NE, ov_bits, k0);

        // Addition carry into bit 56 shifts right by 1 while keeping the sticky bit.
        let k31 = self.k(31);
        let ov_lo = self.bin(Bin::LShr, raw.0, k1);
        let ov_carry = self.bin(Bin::Shl, raw.1, k31);
        let ov_sticky = self.bin(Bin::And, raw.0, k1);
        let ov_lo = self.bin(Bin::Or, ov_lo, ov_carry);
        let ov_lo = self.bin(Bin::Or, ov_lo, ov_sticky);
        let ov_hi = self.bin(Bin::LShr, raw.1, k1);
        let ov_exp = self.bin(Bin::Add, big_exp, k1);

        // Subtraction cancellation shifts left toward bit 55, clamped at the subnormal floor.
        let raw_hi_nz = self.cmp(pred::NE, raw.1, k0);
        let lz_hi = self.ffbh(raw.1);
        let lz_lo = self.ffbh(raw.0);
        let lsh_hi_amt = self.bin(Bin::Sub, lz_hi, k8);
        let lsh_lo_amt = self.bin(Bin::Add, lz_lo, k24);
        let need_lshift = self.sel(raw_hi_nz, lsh_hi_amt, lsh_lo_amt);
        let skip_lshift = self.bin(Bin::Or, overflow_bit, is_zero);
        let safe_need = self.sel(skip_lshift, k0, need_lshift);
        let max_lshift = self.bin(Bin::Sub, big_exp, k1);
        let clamp_to_max = self.cmp(pred::IUGT, safe_need, max_lshift);
        let actual_lshift = self.sel(clamp_to_max, max_lshift, safe_need);
        let lsh = self.shl64(raw, actual_lshift);
        let lsh_exp = self.bin(Bin::Sub, big_exp, actual_lshift);

        let norm = self.sel64(overflow_bit, (ov_lo, ov_hi), lsh);
        let norm_exp = self.sel(overflow_bit, ov_exp, lsh_exp);

        let (exp_ov, fin) = self.round_and_pack(big_sign, norm, norm_exp);

        let zero_hi = self.sel(eff_sub, k0, big_sign);
        let inf_bits = self.k(0x7FF0_0000);
        let ov_hi = self.bin(Bin::Or, big_sign, inf_bits);
        let inf_hi = self.bin(Bin::Or, inf_sign, inf_bits);
        let qnan_hi = self.k(0x7FF8_0000);

        let res = self.sel64(is_zero, (k0, zero_hi), fin);
        let res = self.sel64(exp_ov, (k0, ov_hi), res);
        let res = self.sel64(any_inf, (k0, inf_hi), res);
        self.sel64(is_nan_res, (k0, qnan_hi), res)
    }
}

/// Emits branchless software IEEE-754 binary64 addition `(lo_a, hi_a) + (lo_b, hi_b)`
/// into `block` using only 32-bit integer instructions.
pub(super) fn emit_f64_add(
    s: &mut Shader,
    block: usize,
    lo_a: V,
    hi_a: V,
    lo_b: V,
    hi_b: V,
) -> (V, V) {
    F64Asm { s, c: block }.add(lo_a, hi_a, lo_b, hi_b)
}

impl<'a> Lower<'a> {
    /// Software `WaveActiveSum` of 64-bit floats (`(lo, hi)` masked to `+0.0` outside the active
    /// set), using `waveActiveBallot` + `WaveReadLaneAt` over active lanes so partial thread
    /// groups and diverged lanes never read undefined inactive lanes.
    pub(super) fn sw_active_f64_sum(&mut self, lo: V, hi: V) -> Result<(V, V), Why> {
        let c = self.cur;
        let i32t = self.s.t.i32;
        self.s.flags |= flags::WAVE_OPS;
        let four = self.s.m.named_struct("dx.types.fouri32", vec![i32t; 4]);
        let here = self.s.m.bool(true);
        let ballot = self.s.op(c, "waveActiveBallot", Attr::NoUnwind, four, 116, &[here]);
        let lane = self.s.op(c, "waveGetLaneIndex", Attr::ReadOnly, i32t, 111, &[]);
        let steps = match self.s.wave_size {
            Some(w) => w.trailing_zeros().clamp(1, 7),
            None if super::host_max_wave() == 32 => 5,
            None => 7,
        };
        let b0 = self.s.m.extract(c, ballot, 0);
        let b1 = (steps >= 6).then(|| self.s.m.extract(c, ballot, 1));
        let b2 = (steps >= 7).then(|| self.s.m.extract(c, ballot, 2));
        let b3 = (steps >= 7).then(|| self.s.m.extract(c, ballot, 3));
        let lane_count = (steps >= 7)
            .then(|| self.s.op(c, "waveGetLaneCount", Attr::ReadOnly, i32t, 112, &[]));
        let [zero, k31, k32] = [0, 31, 32].map(|k| self.s.m.i32(k));
        let [k5, k3, k2, k64] = [5, 3, 2, 64].map(|k| self.s.m.i32(k));

        let mut sum_lo = lo;
        let mut sum_hi = hi;
        for step in 0..steps {
            let size = 1u32 << step;
            let (bit, run) = (self.s.m.i32(size), self.s.m.i32(!(size - 1)));
            let other = self.s.m.bin(c, Bin::Xor, lane, bit);
            let base = self.s.m.bin(c, Bin::And, other, run);
            let word = match (b1, b2, b3) {
                (None, ..) => b0,
                (Some(b1), None, _) => {
                    let upper = self.s.m.cmp(c, pred::IUGE, base, k32);
                    self.s.m.select(c, upper, b1, b0)
                }
                (Some(b1), Some(b2), Some(b3)) => {
                    let hi_half = self.s.m.cmp(c, pred::IUGE, base, k64);
                    let in_w1 = self.s.m.cmp(c, pred::IUGE, base, k32);
                    let w_lo = self.s.m.select(c, in_w1, b1, b0);
                    let w_idx = self.s.m.bin(c, Bin::LShr, base, k5);
                    let w_idx = self.s.m.bin(c, Bin::And, w_idx, k3);
                    let in_w3 = self.s.m.cmp(c, pred::NE, w_idx, k2);
                    let w_hi = self.s.m.select(c, in_w3, b3, b2);
                    self.s.m.select(c, hi_half, w_hi, w_lo)
                }
                _ => unreachable!(),
            };
            let at = self.s.m.bin(c, Bin::And, base, k31);
            let mut bits = self.s.m.bin(c, Bin::LShr, word, at);
            if size < 32 {
                let m = self.s.m.i32((1u32 << size) - 1);
                bits = self.s.m.bin(c, Bin::And, bits, m);
            }
            let first = self.s.op(c, "unaryBits.i32", Attr::ReadNone, i32t, 32, &[bits]);
            let mut any = self.s.m.cmp(c, pred::NE, bits, zero);
            if let Some(lc) = lane_count {
                let in_wave = self.s.m.cmp(c, pred::IULT, base, lc);
                any = self.s.m.bin(c, Bin::And, any, in_wave);
            }
            let first = self.s.m.select(c, any, first, zero);
            let from = self.s.m.bin(c, Bin::Add, base, first);
            // Fall back to our own active lane when the partner run is empty.
            let from = self.s.m.select(c, any, from, lane);
            let theirs_lo = self.read_lane(sum_lo, from);
            let theirs_hi = self.read_lane(sum_hi, from);
            let theirs_lo = self.s.m.select(c, any, theirs_lo, zero);
            let theirs_hi = self.s.m.select(c, any, theirs_hi, zero);
            (sum_lo, sum_hi) = emit_f64_add(&mut self.s, c, sum_lo, sum_hi, theirs_lo, theirs_hi);
        }

        if !self.local && self.exchange.as_ref().is_some_and(|e| !e.1) {
            if !self.all_lanes(self.cur_ir) {
                super::need(self.cur_ir);
                return Err("a wave reduction where not every lane runs");
            }
            let (lo0, lo1) = self.exchange_halves(self.cur, sum_lo);
            let (hi0, hi1) = self.exchange_halves(self.cur, sum_hi);
            (sum_lo, sum_hi) = emit_f64_add(&mut self.s, self.cur, lo0, hi0, lo1, hi1);
        }
        Ok((sum_lo, sum_hi))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emit::bitcode::Term;
    use crate::emit::host::{Dispatch, Host, Root, View};
    use crate::emit::shader::Shader;
    use crate::emit::{HostShader, validator};
    use crate::native::HwStage;

    /// Builds a 1-thread compute shader that executes `emit_f64_add` on four
    /// `b0` user-data dwords (`lo_a, hi_a, lo_b, hi_b`) and stores `(lo, hi)`
    /// into the raw UAV at heap slot 0.
    fn build_f64_add_shader() -> HostShader {
        let mut s = Shader::new(false);
        s.threads = Some([1, 1, 1]);
        s.user_data = 4;
        let ud_h = s.user_data_handle();
        let c = s.entry;
        let zero = s.m.i32(0);
        let row = s.op(
            c,
            "cbufferLoadLegacy.i32",
            Attr::ReadOnly,
            s.t.cbuf_ret,
            59,
            &[ud_h, zero],
        );
        let lo_a = s.m.extract(c, row, 0);
        let hi_a = s.m.extract(c, row, 1);
        let lo_b = s.m.extract(c, row, 2);
        let hi_b = s.m.extract(c, row, 3);
        let (r_lo, r_hi) = emit_f64_add(&mut s, c, lo_a, hi_a, lo_b, hi_b);

        let out_h = s.heap_handle(c, zero, false, false, (11 | 0x1000, 0));
        let undef_i = s.m.undef(s.t.i32);
        let undef_f = s.m.undef(s.t.f32);
        let f_lo = s.m.cast(c, Cast::Bitcast, r_lo, s.t.f32);
        let f_hi = s.m.cast(c, Cast::Bitcast, r_hi, s.t.f32);
        let mask3 = s.m.i8(0b0011);
        let align4 = s.m.i32(4);
        s.op(
            c,
            "rawBufferStore.f32",
            Attr::NoUnwind,
            s.t.void,
            140,
            &[out_h, zero, undef_i, f_lo, f_hi, undef_f, undef_f, mask3, align4],
        );
        s.m.terminate(c, Term::Ret);
        let dxil = s.finish();

        // Must never emit DOUBLES or INT64 feature bits.
        let container = crate::Container::parse(&dxil).unwrap();
        let sfi0 = u64::from_le_bytes(
            container
                .part(crate::FourCC::SFI0)
                .unwrap()
                .data
                .try_into()
                .unwrap(),
        );
        assert_eq!(sfi0 & 0x1, 0, "software f64_add emitted DOUBLES ({sfi0:#x})");
        assert_eq!(sfi0 & 0x8000, 0, "software f64_add emitted INT64 ({sfi0:#x})");

        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }

        HostShader {
            stage: HwStage::Cs,
            dxil,
            user_data: (0..4)
                .map(|i| crate::emit::UserData {
                    index: i,
                    slot: 0,
                    root: crate::RootType::Constants,
                    dword: i,
                })
                .collect(),
            samplers: Vec::new(),
            static_samplers: Vec::new(),
            root_cbvs: Vec::new(),
            root_srvs: Vec::new(),
            root_uavs: Vec::new(),
            root_indexed_views: Vec::new(),
        }
    }

    fn f64_edge_values() -> Vec<u64> {
        let mut vals = vec![
            0.0f64.to_bits(),
            (-0.0f64).to_bits(),
            1.0f64.to_bits(),
            (-1.0f64).to_bits(),
            1.25f64.to_bits(),
            (-0.75f64).to_bits(),
            0.5f64.to_bits(),
            (-0.5f64).to_bits(),
            2.0f64.to_bits(),
            (-2.0f64).to_bits(),
            f64::MAX.to_bits(),
            f64::MIN.to_bits(),
            f64::MIN_POSITIVE.to_bits(),
            (-f64::MIN_POSITIVE).to_bits(),
            f64::INFINITY.to_bits(),
            f64::NEG_INFINITY.to_bits(),
            // Largest subnormals and smallest subnormals (both signs).
            0x000F_FFFF_FFFF_FFFF,
            0x800F_FFFF_FFFF_FFFF,
            0x0008_0000_0000_0000,
            0x8008_0000_0000_0000,
            0x0000_0000_0000_0001,
            0x8000_0000_0000_0001,
            // Rounding ties and near-ties around 1.0 and f64::MAX.
            0x3FF0_0000_0000_0001, // 1 + 1 ulp (odd)
            0x3CA0_0000_0000_0000, // 2^-53 (0.5 ulp at 1.0)
            0x3CA0_0000_0000_0001, // > 0.5 ulp at 1.0
            0xBC90_0000_0000_0000, // -2^-54 (0.5 ulp below 1.0)
            0xBC80_0000_0000_0000, // -2^-55
            0xBCB8_0000_0000_0000, // -(2^-53 + 2^-54)
            0x7CA0_0000_0000_0000, // 0.5 ulp of f64::MAX (2^971)
            0x7C90_0000_0000_0000, // 0.25 ulp of f64::MAX
            // Cancellation down into subnormals.
            0x0010_0000_0000_0001,
            0x0020_0000_0000_0000,
            0x8010_0000_0000_0001,
            // NaNs (quiet and signalling).
            0x7FF8_0000_0000_0000,
            0xFFF8_0000_0000_0001,
            0x7FF0_0000_0000_0001,
        ];
        // Deterministic PRNG values across exponents.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..20 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            vals.push(seed);
        }
        vals
    }

    #[test]
    fn software_f64_add_matches_ieee754_on_gpu_and_warp() {
        let shader = build_f64_add_shader();
        let vals = f64_edge_values();
        let mut pairs = Vec::with_capacity(vals.len() * vals.len());
        for &a in &vals {
            for &b in &vals {
                pairs.push((a, b));
            }
        }
        let _one = crate::emit::valu_tests::ONE_AT_A_TIME
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let mut ran = 0;
        for (label, warp) in [("hardware", false), ("warp", true)] {
            let Ok(host) = Host::new(warp) else {
                continue;
            };
            let (root, pso) = host
                .compute_pipeline(&shader)
                .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
            for &(a, b) in &pairs {
                let inputs = vec![a as u32, (a >> 32) as u32, b as u32, (b >> 32) as u32];
                let out = host
                    .dispatch(
                        &root,
                        &pso,
                        &Dispatch {
                            roots: vec![(0, Root::Constants(inputs))],
                            views: &[View::RawUav(2)],
                            samplers: &[],
                            groups: [1, 1, 1],
                        },
                    )
                    .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
                let got = u64::from(out[0][0]) | (u64::from(out[0][1]) << 32);
                let fa = f64::from_bits(a);
                let fb = f64::from_bits(b);
                let want_f = fa + fb;
                if want_f.is_nan() {
                    let fg = f64::from_bits(got);
                    assert!(
                        fg.is_nan() && (got & 0x0008_0000_0000_0000) != 0,
                        "{label}: f64_add({a:#018x}, {b:#018x}) = {got:#018x}, expected qNaN",
                    );
                } else {
                    let want = want_f.to_bits();
                    assert_eq!(
                        got, want,
                        "{label}: f64_add({fa} [{a:#018x}], {fb} [{b:#018x}]) = {got:#018x}, want {want:#018x}",
                    );
                }
            }
            ran += 1;
        }
        assert!(ran > 0, "no SM 6.6 device available");
    }
}
