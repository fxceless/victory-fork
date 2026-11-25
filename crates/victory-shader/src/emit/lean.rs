//! Cheaper exact lowerings of common console instructions, each behind a
//! switch of its own and off by default. They are part of the translation,
//! so the shader cache keys on them (`victory-gpu`'s does).
//!
//! - `VICTORY_QUAD_OPS=1`: a DPP quad permutation (`quad_perm`) in a pixel
//!   program that every lane of a quad reads the same way is one host quad
//!   operation: a broadcast of one lane is `QuadReadLaneAt`, the swaps
//!   across x, y and the diagonal are `QuadReadAcrossX/Y/Diagonal`, the
//!   identity reads nothing. Otherwise each lane works out its source lane
//!   (five integer operations) and reads it with `WaveReadLaneAt`, which
//!   moreover leaves a helper lane's value undefined where quad operations
//!   read it. In mesh programs the broadcasts and the identity skip the
//!   four quad reads and their selects. (A sampled shader cache of Grand
//!   Theft Auto VI: some 12 such reads a pixel program.)
//! - `VICTORY_CLASS_RANGES=1`: `v_cmp_class_f32/f16` with a constant class
//!   mask tests the ranges of the magnitude's bits those classes cover
//!   (zero, denormals, normals, infinity, signalling and quiet NaNs lie in
//!   that order), with the sign where the mask asks for one: `±inf` is two
//!   instructions instead of some twenty (the general form works out the
//!   class number and shifts the mask by it).

use super::bitcode::{Attr, Bin, Module, V, pred};
use super::shader::{Shader, flags};

/// Whether DPP quad permutations become quad operations
/// (`VICTORY_QUAD_OPS=1`).
pub(super) fn quad_ops_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_QUAD_OPS.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_QUAD_OPS").is_some_and(|v| v == "1"))
}

/// Whether class tests with a constant mask test ranges
/// (`VICTORY_CLASS_RANGES=1`).
pub(super) fn class_ranges_on() -> bool {
    #[cfg(test)]
    if let Some(on) = TEST_CLASS_RANGES.get() {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CLASS_RANGES").is_some_and(|v| v == "1"))
}

#[cfg(test)]
thread_local! {
    /// A test's choices for the switches, for its thread only.
    pub(super) static TEST_QUAD_OPS: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
    pub(super) static TEST_CLASS_RANGES: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// A quad permutation as one quad operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum QuadRead {
    /// Each lane reads itself.
    Same,
    /// Every lane reads quad lane n (`QuadReadLaneAt`).
    Lane(u32),
    /// `QuadOp`: 0 across x (lane ^ 1), 1 across y (lane ^ 2), 2 the
    /// diagonal (lane ^ 3), quad lanes numbered as D3D numbers them.
    Across(u8),
}

/// `quad_perm` `ctrl` (0..=0xFF: lane i of a quad reads lane `ctrl >> 2i &
/// 3`) as one quad operation, if it is one.
pub(super) fn quad_read(ctrl: u16) -> Option<QuadRead> {
    if ctrl > 0xFF {
        return None;
    }
    let p = [0, 1, 2, 3].map(|i| u32::from(ctrl >> (2 * i) & 3));
    Some(match p {
        [0, 1, 2, 3] => QuadRead::Same,
        [n, b, c, d] if n == b && n == c && n == d => QuadRead::Lane(n),
        [1, 0, 3, 2] => QuadRead::Across(0),
        [2, 3, 0, 1] => QuadRead::Across(1),
        [3, 2, 1, 0] => QuadRead::Across(2),
        _ => return None,
    })
}

/// `x` (an i32) as quad read `q` reads it, in block `c`.
pub(super) fn quad_op(s: &mut Shader, c: usize, x: V, q: QuadRead) -> V {
    let i32t = s.t.i32;
    match q {
        QuadRead::Same => x,
        QuadRead::Lane(n) => {
            s.flags |= flags::WAVE_OPS;
            let n = s.m.i32(n);
            s.op(c, "quadReadLaneAt.i32", Attr::NoUnwind, i32t, 122, &[x, n])
        }
        QuadRead::Across(k) => {
            s.flags |= flags::WAVE_OPS;
            let k = s.m.i8(k);
            s.op(c, "quadOp.i32", Attr::NoUnwind, i32t, 123, &[x, k])
        }
    }
}

/// A run of the magnitude's bits `lo..=hi` and the sign it needs (`None`:
/// either).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Run {
    pub lo: u32,
    pub hi: u32,
    pub negative: Option<bool>,
}

/// The magnitudes of each class of a float, in their order: zero,
/// denormals, normals, infinity, signalling NaNs, quiet NaNs; with the mask
/// bits of the positive and the negative class.
fn kinds(f16: bool) -> [(u32, u32, u32, u32); 6] {
    let (inf, quiet, top) = if f16 { (0x7C00, 0x7E00, 0x7FFF) } else { (0x7F80_0000, 0x7FC0_0000, 0x7FFF_FFFF) };
    let first_normal = if f16 { 0x0400 } else { 0x0080_0000 };
    [
        (0, 0, 6, 5),
        (1, first_normal - 1, 7, 4),
        (first_normal, inf - 1, 8, 3),
        (inf, inf, 9, 2),
        (inf + 1, quiet - 1, 0, 0),
        (quiet, top, 1, 1),
    ]
}

/// The runs of magnitudes a class mask (bits 0-9: signalling NaN, quiet
/// NaN, -inf, -normal, -denormal, -0, +0, +denormal, +normal, +inf) takes,
/// adjacent classes with the same sign rule joined.
pub(super) fn class_runs(mask: u32, f16: bool) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for (lo, hi, pos, neg) in kinds(f16) {
        let (p, n) = (mask >> pos & 1 != 0, mask >> neg & 1 != 0);
        let negative = match (p, n) {
            (false, false) => continue,
            (true, true) => None,
            (true, false) => Some(false),
            (false, true) => Some(true),
        };
        match runs.last_mut() {
            Some(r) if r.hi + 1 == lo && r.negative == negative => r.hi = hi,
            _ => runs.push(Run { lo, hi, negative }),
        }
    }
    runs
}

/// The class test of `x` (an f32's bits, or an f16's in the low half) for
/// the constant `mask`, as the general form decides it: the magnitude is
/// the bits below the sign, the sign is anything at or above it.
pub(super) fn class_test(m: &mut Module, c: usize, x: V, mask: u32, f16: bool) -> V {
    let runs = class_runs(mask, f16);
    let (top, sign_shift) = if f16 { (0x7FFF, 15) } else { (0x7FFF_FFFF, 31) };
    if runs.is_empty() {
        return m.bool(false);
    }
    let k = |m: &mut Module, v: u32| m.i32(v);
    let km = k(m, top);
    let abs = m.bin(c, Bin::And, x, km);
    let mut sign = None;
    let mut result: Option<V> = None;
    for run in runs {
        let range = match (run.lo, run.hi) {
            (0, hi) if hi == top => None,
            (lo, hi) if lo == hi => {
                let v = k(m, lo);
                Some(m.cmp(c, pred::EQ, abs, v))
            }
            (0, hi) => {
                let v = k(m, hi);
                Some(m.cmp(c, pred::IULE, abs, v))
            }
            (lo, hi) if hi == top => {
                let v = k(m, lo);
                Some(m.cmp(c, pred::IUGE, abs, v))
            }
            (lo, hi) => {
                let (l, n) = (k(m, lo), k(m, hi - lo));
                let off = m.bin(c, Bin::Sub, abs, l);
                Some(m.cmp(c, pred::IULE, off, n))
            }
        };
        let signed = run.negative.map(|negative| {
            let s = *sign.get_or_insert_with(|| {
                let sh = k(m, sign_shift);
                let s = m.bin(c, Bin::LShr, x, sh);
                let z = k(m, 0);
                m.cmp(c, pred::NE, s, z)
            });
            if negative {
                s
            } else {
                let t = m.bool(true);
                m.bin(c, Bin::Xor, s, t)
            }
        });
        let term = match (range, signed) {
            (Some(a), Some(b)) => m.bin(c, Bin::And, a, b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => m.bool(true),
        };
        result = Some(match result {
            Some(r) => m.bin(c, Bin::Or, r, term),
            None => term,
        });
    }
    result.unwrap_or_else(|| m.bool(false))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The general form's class number of `x` (`Lower::compare`).
    fn class_of(x: u32, f16: bool) -> u32 {
        let (exp_sh, sign_sh, exp_mask, mant_mask, quiet_bit) =
            if f16 { (10, 15, 0x1F, 0x03FF, 0x0200) } else { (23, 31, 0xFF, 0x7F_FFFF, 0x40_0000) };
        let e = x >> exp_sh & exp_mask;
        let m = x & mant_mask;
        let neg = x >> sign_sh != 0;
        let t = if m == 0 { 6 } else { 7 };
        let t = if e == 0 { t } else { 8 };
        let t = if e == exp_mask { 9 } else { t };
        let t = if neg { 11 - t } else { t };
        if m != 0 && e == exp_mask { u32::from(x & quiet_bit != 0) } else { t }
    }

    /// `x` against the runs, as [`class_test`] builds it.
    fn runs_say(runs: &[Run], x: u32, f16: bool) -> bool {
        let (top, sign_shift) = if f16 { (0x7FFF, 15) } else { (0x7FFF_FFFF, 31) };
        let abs = x & top;
        let neg = x >> sign_shift != 0;
        runs.iter().any(|r| abs >= r.lo && abs <= r.hi && r.negative.is_none_or(|n| n == neg))
    }

    #[test]
    fn runs_agree_with_the_general_form_for_every_half_and_mask() {
        for mask in 0..1024u32 {
            let runs = class_runs(mask, true);
            for x in 0..=0xFFFFu32 {
                let want = mask >> class_of(x, true) & 1 != 0;
                assert_eq!(runs_say(&runs, x, true), want, "mask {mask:#x} x {x:#06x}");
            }
            // The f16 form reads bits above the half as a sign, as the
            // general one does.
            for x in [0x1_0000u32, 0x1_3C00, 0xFFFF_7C00] {
                let want = mask >> class_of(x, true) & 1 != 0;
                assert_eq!(runs_say(&runs, x, true), want, "mask {mask:#x} x {x:#x}");
            }
        }
    }

    #[test]
    fn runs_agree_with_the_general_form_for_floats() {
        // Every class edge and its neighbours, both signs.
        let mut xs = Vec::new();
        for edge in [0u32, 1, 0x7F_FFFF, 0x80_0000, 0x3F80_0000, 0x7F7F_FFFF, 0x7F80_0000, 0x7F80_0001, 0x7FBF_FFFF,
            0x7FC0_0000, 0x7FFF_FFFF]
        {
            for d in [0u32, 1, 2] {
                for x in [edge.wrapping_sub(d), edge.wrapping_add(d)] {
                    xs.push(x & 0x7FFF_FFFF);
                    xs.push(x | 0x8000_0000);
                }
            }
        }
        let mut seed = 0x9E37_79B9u32;
        for _ in 0..4096 {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            xs.push(seed);
        }
        for mask in 0..1024u32 {
            let runs = class_runs(mask, false);
            for &x in &xs {
                let want = mask >> class_of(x, false) & 1 != 0;
                assert_eq!(runs_say(&runs, x, false), want, "mask {mask:#x} x {x:#010x}");
            }
        }
    }

    #[test]
    fn common_masks_are_one_run() {
        let one = |mask, f16| class_runs(mask, f16).len();
        // ±inf, NaN, ±0, ±denormal, finite, not NaN.
        for mask in [0x204, 0x003, 0x060, 0x090, 0x1F8, 0x3FC] {
            assert_eq!(one(mask, false), 1, "{mask:#x}");
        }
        assert_eq!(class_runs(0x204, false), [Run { lo: 0x7F80_0000, hi: 0x7F80_0000, negative: None }]);
        assert_eq!(class_runs(0x200, true), [Run { lo: 0x7C00, hi: 0x7C00, negative: Some(false) }]);
        assert!(class_runs(0, false).is_empty());
        assert_eq!(class_runs(0x3FF, false), [Run { lo: 0, hi: 0x7FFF_FFFF, negative: None }]);
    }

    #[test]
    fn quad_reads_read_the_lanes_the_permutation_names() {
        let mut found = 0;
        for ctrl in 0..=0xFFu16 {
            let Some(q) = quad_read(ctrl) else { continue };
            found += 1;
            for i in 0..4u32 {
                let want = u32::from(ctrl) >> (2 * i) & 3;
                let got = match q {
                    QuadRead::Same => i,
                    QuadRead::Lane(n) => n,
                    QuadRead::Across(k) => i ^ (u32::from(k) + 1),
                };
                assert_eq!(got, want, "ctrl {ctrl:#x} lane {i}");
            }
        }
        // The identity, four broadcasts and three swaps.
        assert_eq!(found, 8);
        assert_eq!(quad_read(0xB1), Some(QuadRead::Across(0)));
        assert_eq!(quad_read(0x4E), Some(QuadRead::Across(1)));
        assert_eq!(quad_read(0x1B), Some(QuadRead::Across(2)));
        assert_eq!(quad_read(0xE4), Some(QuadRead::Same));
        assert_eq!(quad_read(0xAA), Some(QuadRead::Lane(2)));
        assert_eq!(quad_read(0x101), None);
    }

    /// A pixel program reading a texture coordinate's neighbour in its quad
    /// (`v_mov_b32_dpp quad_perm`) through `ctrl`, lowered with quad
    /// operations or not; validated by `dxil.dll` when present.
    #[cfg(windows)]
    fn pixel_dpp(ctrl: u16, quad_ops: bool) -> Vec<u8> {
        use super::super::container::SigElement;
        use super::super::lower::{Parts, lower};
        use crate::native::{ExportPatch, HwStage, NativeProgram, PsInput, StageIo, Usage};
        use crate::{io, ir, resource, structure};
        let words = [
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0x7E06_02FA, // v_mov_b32_dpp v3, v2 quad_perm row_mask:0xf bank_mask:0xf
            0xFF00_0002 | u32::from(ctrl) << 8,
            0xF800_180F, // exp mrt0, v3, v3, v3, v3 done vm
            0x0303_0303,
            0xBF81_0000, // s_endpgm
        ];
        let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let native = NativeProgram {
            stage: HwStage::Ps,
            wave64: false,
            vgprs: 256,
            layout: vec![],
            used: vec![],
            io: StageIo::Ps {
                inputs: vec![PsInput {
                    usage: Usage::Generic,
                    row: 1,
                    mask: 3,
                    attr: 0,
                    flat: false,
                    fp16: false,
                    pack2: false,
                    row2: 1,
                }],
                input_addr: 2,
                db_shader_control: 0,
                patches: Vec::<ExportPatch>::new(),
            },
            code,
        };
        let element = |name: &str, kind, interp, row, cols, usage| SigElement {
            name: name.into(),
            index: 0,
            kind,
            comp: 9,
            interp,
            row: Some(row),
            col: 0,
            cols,
            usage,
            half: false,
        };
        let p = ir::build(&native.code, false, 256).unwrap();
        let st = structure::structure(&p).unwrap();
        let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
        let io = io::raise(&p, &native).unwrap();
        let was = TEST_QUAD_OPS.replace(Some(quad_ops));
        let lowered = lower(&Parts {
            native: &native,
            p: &p,
            st: &st,
            io: &io,
            res: &res,
            inputs: vec![element("TEXCOORD", 0, 2, 1, 2, 1)],
            outputs: vec![element("SV_Target", 16, 0, 0, 4, 0xF)],
            prims: vec![],
            mesh_stride: 0,
        });
        TEST_QUAD_OPS.set(was);
        let dxil = lowered.unwrap_or_else(|e| panic!("ctrl {ctrl:#x}: {e}")).dxil;
        let path = super::super::validator::Validator::default_path();
        if let Ok(mut v) = super::super::validator::Validator::load(path.as_deref()) {
            let mut copy = dxil.clone();
            v.validate(&mut copy).unwrap_or_else(|e| panic!("ctrl {ctrl:#x} quad ops {quad_ops}: {e}"));
        }
        dxil
    }

    #[cfg(windows)]
    #[test]
    fn pixel_quad_permutations_validate_and_shrink() {
        // Across x, y, the diagonal; broadcasts; the identity; one that is no
        // quad operation (left as it was).
        for ctrl in [0xB1, 0x4E, 0x1B, 0x00, 0xFF, 0xE4] {
            let (quad, general) = (pixel_dpp(ctrl, true), pixel_dpp(ctrl, false));
            assert!(quad.len() < general.len(), "ctrl {ctrl:#x}: {} bytes, the general form {}", quad.len(), general.len());
        }
        assert_eq!(pixel_dpp(0x93, true), pixel_dpp(0x93, false));
    }
}
