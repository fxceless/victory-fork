//! End-to-end tests of synthetic GFX10 VALU shaders lowered to DXIL, validated
//! by `dxil.dll` when present, and executed on the host GPU and WARP.
//!
//! Covers RDNA2 hardware semantics that differ from naive DXIL lowering:
//! `v_med3_f32`/`v_med3_f16` with NaN operands, `v_fract_f32`/`v_fract_f16`
//! near `1.0`, turn-range reduction for `v_sin_*`/`v_cos_*`, and the LDS
//! float atomics `ds_min_f32`/`ds_max_f32`.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// SOP1 `s_mov_b32 sdst, src0`.
const fn s_mov_b32(sdst: u8, src0: u8) -> u32 {
    0xBE80_0000 | (sdst as u32) << 16 | 0x03 << 8 | src0 as u32
}

/// SOP2 `op sdst, src0, src1`.
const fn sop2(op: u8, sdst: u8, src0: u8, src1: u8) -> u32 {
    0x8000_0000
        | (op as u32 & 0x7F) << 23
        | (sdst as u32 & 0x7F) << 16
        | (src1 as u32) << 8
        | src0 as u32
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], 0` (soffset = null).
const fn s_load_dwordx4(dst: u8, base: u8) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        125 << 25,
    ]
}

/// VOP1 `op vdst, src0` (`src0`: 0..105 = SGPR, 256..511 = VGPR).
const fn vop1(op: u16, vdst: u8, src0: u16) -> u32 {
    0x7E00_0000 | (vdst as u32) << 17 | (op as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// VOP3 `op vdst, src0, src1, src2` (no modifiers).
const fn vop3(op: u16, vdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    vop3_mod(op, vdst, src0, src1, src2, 0, 0, 0, false)
}

/// VOP3 `op vdst, src0, src1, src2` with explicit `abs`, `neg`, `omod`, `clamp`.
#[allow(clippy::too_many_arguments)]
const fn vop3_mod(
    op: u16,
    vdst: u8,
    src0: u16,
    src1: u16,
    src2: u16,
    abs: u8,
    neg: u8,
    omod: u8,
    clamp: bool,
) -> [u32; 2] {
    [
        0x35 << 26
            | (op as u32 & 0x3FF) << 16
            | (clamp as u32) << 15
            | (abs as u32 & 7) << 8
            | vdst as u32,
        (neg as u32 & 7) << 29
            | (omod as u32 & 3) << 27
            | (src0 as u32 & 0x1FF)
            | (src1 as u32 & 0x1FF) << 9
            | (src2 as u32 & 0x1FF) << 18,
    ]
}

/// MUBUF `buffer_store_dword vdata, off, s[srsrc:srsrc+3], 0 offset:byte_off`.
const fn buffer_store_dword(vdata: u8, srsrc: u8, byte_off: u16) -> [u32; 2] {
    [
        0x38 << 26 | 0x1C << 18 | (byte_off as u32 & 0xFFF),
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8,
    ]
}

const fn sgpr(n: u8) -> u16 {
    n as u16
}

const fn vgpr(n: u8) -> u16 {
    256 + n as u16
}

/// Builds and lowers a 1-thread synthetic RDNA2 compute shader that receives
/// 4 dwords of root constants in `s0..s3`, runs `body` (leaving its result in
/// `v0`), and stores `v0` to dword 0 of the raw UAV at heap index 0.
fn build_valu_cs(body: &[u32]) -> HostShader {
    build_cs(body, 1, 0)
}

/// [`build_valu_cs`] with `threads` threads in x (thread index in `v0`
/// on entry, before `body`) and `lds_bytes` of LDS (a multiple of 512).
fn build_cs(body: &[u32], threads: u32, lds_bytes: u32) -> HostShader {
    let mut words = Vec::new();
    // s[4:5] = descriptor table address (heap index 0 -> byte offset 0).
    words.push(s_mov_b32(5, 0x80)); // inline constant 0
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Copy root constants s0..s3 into v1..v4 so VALU instructions operate on
    // VGPRs just as they do in real shaders.
    for i in 0..4u8 {
        words.push(vop1(0x001, i + 1, sgpr(i))); // v_mov_b32
    }
    words.extend_from_slice(body);
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: (5 << 1) | ((lds_bytes / 512) << 15), // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };

    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        let ours = copy[4..20].to_vec();
        v.validate(&mut copy).expect("DXIL validation failed");
        assert_eq!(copy[4..20], ours[..], "container hash mismatch");
    }

    shader
}

/// One GPU test at a time. With these tests on the hardware adapter and WARP
/// from several test threads at once, now and then none of their dispatches
/// ever completes and the test binary hangs; and two GPU tests at once on
/// WARP can end the test process with an access violation. Other GPU test
/// modules take this lock too.
pub(super) static ONE_AT_A_TIME: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Runs `shader` on every available SM 6.6 host (hardware GPU and WARP) for
/// each 4-dword input set and invokes `check(backend, case_index, inputs, result_dword)`.
fn run_cases(
    shader: &HostShader,
    cases: &[[u32; 4]],
    check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    run_cases_on(shader, cases, &[("hardware", false), ("warp", true)], check);
}

fn run_cases_on(
    shader: &HostShader,
    cases: &[[u32; 4]],
    devices: &[(&str, bool)],
    mut check: impl FnMut(&str, usize, [u32; 4], u32),
) {
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for &(label, warp) in devices {
        let host = match Host::new(warp) {
            Ok(host) => host,
            Err(e) => {
                eprintln!("no SM 6.6 {label} device ({e}); skipped");
                continue;
            }
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        for (i, &inputs) in cases.iter().enumerate() {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            check(label, i, inputs, out[0][0]);
        }
        ran += 1;
    }
    if ran == 0 && devices.iter().all(|&(_, warp)| !warp) {
        // A case for the hardware adapter alone, on a machine without one.
        eprintln!("no SM 6.6 hardware device; a hardware-only case not run");
        return;
    }
    assert!(ran > 0, "no SM 6.6 device available ({devices:?})");
}

#[test]
fn scalar_quad_masks_pack_nibbles_and_set_scc() {
    let pack_words = |opcode: u32| {
        [
            0xBE90_0000 | opcode << 8,   // s_quadmask s16, s0 (pairs for b64)
            sop2(0x0a, 18, 129, 128),  // s_cselect_b32 s18, 1, 0
            sop2(0x1e, 18, 18, 144),   // s_lshl_b32 s18, s18, 16
            sop2(0x10, 16, 16, 18),    // s_or_b32 s16, s16, s18
            vop1(1, 0, sgpr(16)),      // v_mov_b32 v0, s16
        ]
    };
    let cases = [
        ([0, 0, 0, 0], [0, 0]),
        ([0x8000_0420, 0x0100_0001, 0, 0], [0x1_0086, 0x1_4186]),
        ([0, 0x8000_0000, 0, 0], [0, 0x1_8000]),
        ([u32::MAX, u32::MAX, 0, 0], [0x1_00ff, 0x1_ffff]),
    ];
    let inputs = cases.map(|(input, _)| input);
    for (wide, opcode) in [0x2c, 0x2d].into_iter().enumerate() {
        let shader = build_valu_cs(&pack_words(opcode));
        run_cases(&shader, &inputs, |backend, i, _, got| {
            assert_eq!(got, cases[i].1[wide], "{backend}, opcode {opcode:#x}");
        });
    }
    let high = build_valu_cs(&[
        0xBE90_2D00, // s_quadmask_b64 s[16:17], s[0:1]
        vop1(1, 0, sgpr(17)),
    ]);
    run_cases(&high, &inputs, |backend, _, _, got| {
        assert_eq!(got, 0, "{backend}: high dword");
    });
}

fn is_f16_nan(bits: u16) -> bool {
    (bits & 0x7FFF) > 0x7C00
}

#[test]
fn a_branch_to_fallthrough_keeps_both_values_at_a_join() {
    let shader = build_valu_cs(&[
        0xBF06_0100,            // s_cmp_eq_u32 s0, s1
        0xBF85_0002,            // s_cbranch_scc1 -> the s2 = 1 path
        s_mov_b32(2, 128),      // s2 = 0
        0xBF82_0002,            // s_branch -> the join
        s_mov_b32(2, 129),      // s2 = 1
        0xBF85_0000,            // s_cbranch_scc1 -> the following instruction
        vop1(0x001, 0, sgpr(2)), // join: v0 = s2
    ]);
    run_cases(
        &shader,
        &[[7, 7, 0, 0], [7, 5, 0, 0]],
        |backend, _, inputs, actual| {
            assert_eq!(actual, u32::from(inputs[0] == inputs[1]), "{backend}");
        },
    );
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = u32::from(bits >> 15) << 31;
    let exp = (bits >> 10) & 0x1F;
    let mant = bits & 0x3FF;
    let f32_bits = match exp {
        0 => {
            if mant == 0 {
                sign
            } else {
                let mut m = u32::from(mant);
                let mut e = 127 - 14;
                while (m & 0x400) == 0 {
                    m <<= 1;
                    e -= 1;
                }
                sign | (e << 23) | ((m & 0x3FF) << 13)
            }
        }
        0x1F => sign | (0xFF << 23) | (u32::from(mant) << 13),
        _ => sign | ((u32::from(exp) + 112) << 23) | (u32::from(mant) << 13),
    };
    f32::from_bits(f32_bits)
}

/// MTBUF `op vdata, off, s[srsrc:srsrc+3], 0 format:format` (at byte 0).
const fn mtbuf(op: u8, vdata: u8, srsrc: u8, format: u8) -> [u32; 2] {
    [
        0x3A << 26 | (format as u32 & 0x7F) << 19 | (op as u32 & 7) << 16,
        0x80 << 24
            | (op as u32 >> 3 & 1) << 21
            | (srsrc as u32 >> 2 & 0x1F) << 16
            | (vdata as u32) << 8,
    ]
}

/// Stores `v1` to the buffer, then reads it back into `v0` with the tbuffer
/// load `op` in buffer format `format`.
fn tbuffer_load_of_v1(op: u8, format: u8) -> HostShader {
    let mut body = buffer_store_dword(1, 8, 0).to_vec();
    body.extend(mtbuf(op, 0, 8, format));
    build_valu_cs(&body)
}

#[test]
fn tbuffer_d16_loads_pack_two_channels_per_dword() {
    // tbuffer_load_format_d16_x and tbuffer_load_format_d16_xy.
    let (x, xy) = (8, 9);
    // (opcode, format, dword stored, dword loaded)
    let cases: &[(u8, u8, u32, u32)] = &[
        // 16_16_FLOAT: the f16 pair (1.0, 2.0) as it was stored.
        (xy, 29, 0x4000_3C00, 0x4000_3C00),
        // 16_16_UINT: the 16-bit integers as they were stored.
        (xy, 27, 0xBEEF_1234, 0xBEEF_1234),
        // 16_16_UNORM: 0.0 and 1.0, as f16.
        (xy, 23, 0xFFFF_0000, 0x3C00_0000),
        // 16_FLOAT, one channel: the high half is 0.
        (x, 13, 0x4000_3C00, 0x0000_3C00),
    ];
    for &(op, format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(op, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got, want,
                "{backend}: d16 opcode {op} format {format} of {stored:#010x} = {got:#010x}, want {want:#010x}",
            );
        });
    }
}

#[test]
fn normalized_buffer_formats_load_their_ends_exactly() {
    // tbuffer_load_format_x.
    let x = 0;
    // (format, dword stored, value loaded)
    let cases: &[(u8, u32, f32)] = &[
        (1, 0xFF, 1.0),   // 8_UNORM
        (7, 0xFFFF, 1.0), // 16_UNORM
        (2, 0x7F, 1.0),   // 8_SNORM: the top, the one above the lowest, the lowest
        (2, 0x81, -1.0),
        (2, 0x80, -1.0),
        (8, 0x7FFF, 1.0), // 16_SNORM
        (8, 0x8001, -1.0),
        (8, 0x8000, -1.0),
    ];
    for &(format, stored, want) in cases {
        let shader = tbuffer_load_of_v1(x, format);
        run_cases(&shader, &[[stored, 0, 0, 0]], |backend, _, _, got| {
            assert_eq!(
                got,
                want.to_bits(),
                "{backend}: format {format} of {stored:#x} = {got:#010x}, want {want}",
            );
        });
    }
}

/// Codes of the normalized formats sampled below: every 8-bit code; for
/// 16 bits a stride, the RX 6700 XT's division misses (257, 261, ...) and
/// the ends.
fn normalized_codes(bits: u32) -> Vec<u32> {
    if bits <= 10 {
        return (0..1 << bits).collect();
    }
    let edges = [1, 2, 3, 255, 256, 257, 261, 265, 269, 511, 512, 1000, 0x7FFF, 0x8000, 0x8001, 0xFFFE];
    (0..=0xFFFF).step_by(257).chain(edges).collect()
}

/// Every normalized code loads as the float nearest its exact quotient,
/// as the console's buffer unit converts it, not as the host divides (the
/// RX 6700 XT gave 125 of the 8-bit codes one ulp high and 16-bit codes
/// such as 257 one ulp low): unsigned and signed, 8, 10 (the x channel of
/// 2_10_10_10) and 16 bits, the signed formats' two lowest codes -1.0.
#[test]
fn normalized_buffer_formats_load_correctly_rounded_quotients() {
    let x = 0;
    // (format, bits, signed): 8_UNORM, 16_UNORM, 8_SNORM, 16_SNORM,
    // 2_10_10_10_UNORM and _SNORM.
    for (format, bits, signed) in [(1u8, 8u32, false), (7, 16, false), (2, 8, true), (8, 16, true), (50, 10, false), (51, 10, true)] {
        let shader = tbuffer_load_of_v1(x, format);
        let cases: Vec<[u32; 4]> = normalized_codes(bits).iter().map(|&a| [a, 0, 0, 0]).collect();
        run_cases(&shader, &cases, |backend, _, [a, ..], got| {
            let want = if signed {
                let top = (1i32 << (bits - 1)) - 1;
                let s = ((a << (32 - bits)) as i32) >> (32 - bits);
                (s.max(-top) as f32 / top as f32).to_bits()
            } else {
                (a as f32 / ((1u32 << bits) - 1) as f32).to_bits()
            };
            let sign = if signed { "signed" } else { "unsigned" };
            assert_eq!(got, want, "{backend}: {bits}-bit {sign} normalized {a:#x} loads as {got:#010x}, want {want:#010x}");
        });
    }
}

/// A title truncates `x * (2^n - 1)` back to the code it stored (the
/// material index a mesh program reads from a vertex colour's alpha, as
/// `v_mul_legacy_f32` by 255.0 then `v_cvt_u32_f32`): every code comes back
/// as itself, on hardware and WARP, which a quotient one ulp low breaks.
#[test]
fn normalized_codes_truncate_back_to_themselves() {
    let x = 0;
    for (format, bits) in [(1u8, 8u32), (7, 16)] {
        let scale = (((1u32 << bits) - 1) as f32).to_bits();
        let mut body = buffer_store_dword(1, 8, 0).to_vec();
        body.extend(mtbuf(x, 0, 8, format));
        // v_mul_legacy_f32 v0, literal, v0; v_cvt_u32_f32 v0, v0.
        body.extend([0x07 << 25 | 0xFF, scale]);
        body.push(vop1(0x07, 0, vgpr(0)));
        let shader = build_valu_cs(&body);
        let cases: Vec<[u32; 4]> = normalized_codes(bits).iter().map(|&a| [a, 0, 0, 0]).collect();
        run_cases(&shader, &cases, |backend, _, [a, ..], got| {
            assert_eq!(got, a, "{backend}: {bits}-bit code {a} comes back as {got}");
        });
    }
}

#[test]
fn packed_add_selects_halves_of_inline_constants() {
    // Add the inline constant 1/(2*pi) to v1 (zero) to read back its two halves.
    // The same GFX10 source field (248) has a different bit pattern for a
    // packed floating-point and a packed integer instruction.
    let cases = [
        // v_pk_add_f16 v0, 0x3118, v1
        ([0xcc0f4000, 0x1a0202f8], 0x0000_3118),
        // v_pk_add_f16 v0, 0x3118, v1 op_sel_hi:[0,1] (use low twice)
        ([0xcc0f4000, 0x120202f8], 0x3118_3118),
        // v_pk_add_u16 v0, 0x3e22f983, v1
        ([0xcc0a4000, 0x1a0202f8], 0x3e22_f983),
    ];
    for (words, expected) in cases {
        let shader = build_valu_cs(&words);
        run_cases(&shader, &[[0; 4]], |backend, _, _, got| {
            assert_eq!(got, expected, "{backend}: {words:08x?}");
        });
    }
}

/// The float `v_cvt_f32_f64` gives for `x`: Rust's rounding (to nearest
/// even), with results below the smallest normal float flushed to signed
/// zero as the console's default float mode does.
fn cvt_f32_f64_reference(x: f64) -> u32 {
    let r = x as f32;
    if r.is_subnormal() || (r == 0.0 && x != 0.0) {
        r.to_bits() & 0x8000_0000
    } else {
        r.to_bits()
    }
}

#[test]
#[ignore = "expects results below the smallest normal float flushed to zero; the IEEE lowering (v_cvt_f32_f64_matches_ieee754) is kept until the console's mode is checked"]
fn v_cvt_f32_f64_rounds_to_nearest_even() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x0f)
    let shader = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    let tie_even = 1.0f64 + f64::from(f32::EPSILON) / 2.0; // ties to 1.0
    let tie_odd = 1.0f64 + 1.5 * f64::from(f32::EPSILON); // ties up to 1 + 2 ulp
    let values = [
        0.0,
        -0.0,
        1.0,
        -2.5,
        std::f64::consts::PI,
        1.0e-3,
        123_456_789.123,
        tie_even,
        tie_odd,
        1.0 + f64::from(f32::EPSILON) * 0.75,
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_000_1, // rounds to infinity
        1.0e300,
        -1.0e300,
        1.0e-40, // a float denormal: flushed
        -1.0e-300,
        f64::from(f32::MIN_POSITIVE),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&shader, &inputs, |backend, i, _, got| {
        let x = values[i];
        if x.is_nan() {
            assert!(f32::from_bits(got).is_nan(), "{backend}: cvt({x}) = {got:#010x}, want NaN");
            return;
        }
        let want = cvt_f32_f64_reference(x);
        assert_eq!(got, want, "{backend}: cvt({x:e}) = {got:#010x}, want {want:#010x}");
    });
}

/// The lane `v_permlane16_b32` (`cross` false) or `v_permlanex16_b32` reads
/// for `lane`, with selector dwords `s1` (lanes 0-7 of a row) and `s2`.
fn permlane_source(lane: u32, s1: u32, s2: u32, cross: bool) -> u32 {
    let sel = u64::from(s1) | u64::from(s2) << 32;
    let pick = (sel >> (4 * (lane & 15)) & 15) as u32;
    let row = if cross {
        (lane & !31) | ((lane & 16) ^ 16)
    } else {
        lane & !15
    };
    row | pick
}

#[test]
fn v_permlane16_and_x16_read_the_lanes_their_selectors_pick() {
    // 32 lanes, each with its lane number in v0; v1 = permute(v0, s0, s1);
    // lane `read` of v1 into s12, then into v0 for the store.
    let selectors: [(u32, u32); 3] = [
        (0x7654_3210, 0xfedc_ba98), // identity within the row
        (0x89ab_cdef, 0x0123_4567), // reversed
        (0x1111_1111, 0xf0f0_f0f0), // repeats
    ];
    for (op, cross) in [(0x377u16, false), (0x378, true)] {
        for read in [0u8, 5, 15, 16, 22, 31] {
            let mut body = Vec::new();
            body.extend_from_slice(&vop3(op, 1, vgpr(0), sgpr(0), sgpr(1)));
            // v_readlane_b32 s12, v1, read (inline constant 128 + read)
            body.extend_from_slice(&vop3(0x360, 12, vgpr(1), 128 + u16::from(read), 0));
            body.push(vop1(0x001, 0, sgpr(12))); // v_mov_b32 v0, s12
            let shader = build_cs(&body, 32, 0);
            let inputs: Vec<[u32; 4]> = selectors.iter().map(|&(a, b)| [a, b, 0, 0]).collect();
            // Cross-lane code declares WaveSize(32), which WARP's pipeline
            // creation refuses: hardware only.
            run_cases_on(&shader, &inputs, &[("hardware", false)], |backend, i, inp, got| {
                let want = permlane_source(u32::from(read), inp[0], inp[1], cross);
                assert_eq!(
                    got, want,
                    "{backend}: op {op:#x} selectors {:08x?}, lane {read}",
                    selectors[i]
                );
            });
        }
    }
}

#[test]
fn v_med3_f32_matches_rdna2_with_nan() {
    // v_med3_f32 v0, v1, v2, v3 (VOP3 opcode 0x157)
    let shader = build_valu_cs(&vop3(0x157, 0, vgpr(1), vgpr(2), vgpr(3)));
    let nan = f32::NAN.to_bits();
    let f = f32::to_bits;

    let cases: &[([u32; 4], Option<f32>)] = &[
        // Normal medians (all orderings).
        ([f(1.0), f(2.0), f(3.0), 0], Some(2.0)),
        ([f(3.0), f(1.0), f(2.0), 0], Some(2.0)),
        ([f(-5.0), f(0.0), f(5.0), 0], Some(0.0)),
        ([f(1.0), f(1.0), f(2.0), 0], Some(1.0)),
        // One NaN: RDNA2 returns min3 of the non-NaN operands.
        ([f(1.0), f(2.0), nan, 0], Some(1.0)),
        ([nan, f(1.0), f(0.0), 0], Some(0.0)),
        ([nan, f(0.0), f(1.0), 0], Some(0.0)),
        ([f(2.0), nan, f(1.0), 0], Some(1.0)),
        ([f(-3.0), nan, f(4.0), 0], Some(-3.0)),
        // Two NaNs: returns the sole non-NaN operand.
        ([nan, f(5.0), nan, 0], Some(5.0)),
        ([nan, nan, f(-2.0), 0], Some(-2.0)),
        ([f(7.0), nan, nan, 0], Some(7.0)),
        // Three NaNs: returns NaN.
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert_eq!(
                got_f,
                want,
                "{backend}: v_med3_f32({:?}, {:?}, {:?}) = {got_f} ({got:#010x}), want {want}",
                f32::from_bits(inp[0]),
                f32::from_bits(inp[1]),
                f32::from_bits(inp[2]),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_med3_f32(NaN, NaN, NaN) = {got_f} ({got:#010x}), want NaN",
            ),
        }
    });
}

/// RDNA2 orders -0 below +0 in its maximums and minimums; DXIL's FMax and
/// FMin compare and select, which can give either zero.
#[test]
fn max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_0000_u32);
    // Each case has a -0 and a +0 among its first two operands.
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f32", 0x110, pz),
        ("v_min_f32", 0x10F, nz),
        ("v_max3_f32", 0x154, pz),
        ("v_min3_f32", 0x151, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    }
}

/// The f16 forms go through the same `max_min` helper, after widening to f32.
#[test]
fn f16_max_and_min_order_negative_zero_below_positive_zero() {
    let (pz, nz) = (0_u32, 0x8000_u32);
    let cases = [
        [nz, pz, nz, 0],
        [pz, nz, nz, 0],
        [nz, pz, pz, 0],
        [pz, nz, pz, 0],
    ];
    let ops = [
        ("v_max_f16", 0x139, pz),
        ("v_min_f16", 0x13A, nz),
        ("v_max3_f16", 0x354, pz),
        ("v_min3_f16", 0x351, nz),
    ];
    for (name, op, want) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            // The f16 result is the low half of v0.
            assert_eq!(got & 0xFFFF, want, "{backend}: {name} {inp:x?}");
        });
    }
}

#[test]
fn v_min3_max3_med3_integer_forms_match_rdna2() {
    // v_{min3,max3,med3}_{i32,u32} v0, v1, v2, v3 (VOP3 0x152..0x159).
    type Pick = fn([u32; 3]) -> u32;
    let ops: [(&str, u16, Pick); 6] = [
        ("v_min3_i32", 0x152, |t| {
            t.into_iter().min_by_key(|&x| x as i32).unwrap()
        }),
        ("v_min3_u32", 0x153, |t| t.into_iter().min().unwrap()),
        ("v_max3_i32", 0x155, |t| {
            t.into_iter().max_by_key(|&x| x as i32).unwrap()
        }),
        ("v_max3_u32", 0x156, |t| t.into_iter().max().unwrap()),
        ("v_med3_i32", 0x158, |mut t| {
            t.sort_by_key(|&x| x as i32);
            t[1]
        }),
        ("v_med3_u32", 0x159, |mut t| {
            t.sort_unstable();
            t[1]
        }),
    ];
    // Every ordering of three values, with the sign bit set on some so that
    // the signed and unsigned answers differ.
    let values = [0u32, 1, 0x7FFF_FFFF, 0x8000_0000, 0xFFFF_FFFF];
    let mut cases = Vec::new();
    for a in values {
        for b in values {
            for c in values {
                cases.push([a, b, c, 0]);
            }
        }
    }
    for (name, op, pick) in ops {
        let shader = build_valu_cs(&vop3(op, 0, vgpr(1), vgpr(2), vgpr(3)));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = pick([inp[0], inp[1], inp[2]]);
            assert_eq!(
                got, want,
                "{backend}: {name}({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                inp[0], inp[1], inp[2],
            );
        });
    }
}

#[test]
fn v_med3_f16_matches_rdna2_with_nan() {
    // v_med3_f16 v0, v1, v2, v3 (VOP3 opcode 0x357)
    let shader = build_valu_cs(&vop3(0x357, 0, vgpr(1), vgpr(2), vgpr(3)));
    let h0 = 0x0000u32; // 0.0
    let h1 = 0x3C00u32; // 1.0
    let h2 = 0x4000u32; // 2.0
    let h3 = 0x4200u32; // 3.0
    let h5 = 0x4500u32; // 5.0
    let hn2 = 0xC000u32; // -2.0
    let nan = 0x7E00u32;

    let cases: &[([u32; 4], Option<u16>)] = &[
        ([h1, h2, h3, 0], Some(h2 as u16)),
        ([hn2, h0, h3, 0], Some(h0 as u16)),
        ([h1, h2, nan, 0], Some(h1 as u16)),
        ([nan, h1, h0, 0], Some(h0 as u16)),
        ([nan, h0, h1, 0], Some(h0 as u16)),
        ([h2, nan, h1, 0], Some(h1 as u16)),
        ([nan, h5, nan, 0], Some(h5 as u16)),
        ([nan, nan, nan, 0], None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_med3_f16({:#06x}, {:#06x}, {:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0], inp[1], inp[2],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_med3_f16(NaN, NaN, NaN) = {got_h:#06x}, want NaN",
            ),
        }
    });
}

/// `v_cmp_lg_*` is "less or greater": ordered and not equal, so false when
/// either operand is a NaN (RDNA2 ISA; the same on gfx1036 hardware).
#[test]
fn v_cmp_lg_is_false_for_nan() {
    // v_cmp_lg_{f32,f16}_e64 s20, v1, v2 (VOPC 0x005 / 0x0cd in VOP3);
    // v_cndmask_b32_e64 v0, 0, 1, s20 (VOP3 0x101)
    let nan = f32::NAN.to_bits();
    let (inf, ninf) = (f32::INFINITY.to_bits(), f32::NEG_INFINITY.to_bits());
    let f = f32::to_bits;
    let f32_cases: &[([u32; 4], u32)] = &[
        ([f(1.0), f(2.0), 0, 0], 1),
        ([f(2.0), f(1.0), 0, 0], 1),
        ([f(1.0), f(1.0), 0, 0], 0),
        ([f(0.0), f(-0.0), 0, 0], 0),
        ([inf, ninf, 0, 0], 1),
        ([inf, inf, 0, 0], 0),
        ([nan, f(1.0), 0, 0], 0),
        ([f(1.0), nan, 0, 0], 0),
        ([nan, nan, 0, 0], 0),
        ([0x7F80_0001, f(0.0), 0, 0], 0), // a signalling NaN
    ];
    let f16_cases: &[([u32; 4], u32)] = &[
        ([0x3C00, 0x4000, 0, 0], 1),
        ([0x4000, 0x3C00, 0, 0], 1),
        ([0x3C00, 0x3C00, 0, 0], 0),
        ([0x0000, 0x8000, 0, 0], 0),
        ([0x7C00, 0xFC00, 0, 0], 1),
        ([0x7E00, 0x3C00, 0, 0], 0),
        ([0x3C00, 0x7E00, 0, 0], 0),
        ([0x7E00, 0x7E00, 0, 0], 0),
        ([0x7C01, 0x0000, 0, 0], 0),           // a signalling NaN
        ([0xABCD_3C00, 0x1234_7E00, 0, 0], 0), // only the low halves are read
    ];
    for (op, what, cases) in [
        (0x005u16, "v_cmp_lg_f32", f32_cases),
        (0x0cd, "v_cmp_lg_f16", f16_cases),
    ] {
        let mut body = vop3(op, 20, vgpr(1), vgpr(2), 0x80).to_vec();
        body.extend_from_slice(&vop3(0x101, 0, 0x80, 0x81, sgpr(20)));
        let shader = build_valu_cs(&body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(
                got, want,
                "{backend}: {what}({:#010x}, {:#010x}) = {got}, want {want}",
                inp[0], inp[1],
            );
        });
    }
}

#[test]
fn v_cvt_off_f32_i4_uses_the_signed_low_nibble() {
    // v_cvt_off_f32_i4 v0, v1 (VOP1 opcode 0x00e)
    let shader = build_valu_cs(&[vop1(0x00e, 0, vgpr(1))]);
    let nibble = |n: u32| (((n as i32) << 28) >> 28) as f32 / 16.0;

    let mut cases: Vec<u32> = (0..16).collect();
    // Bits above the low nibble are ignored.
    cases.extend([0xABCD_EF05, 0xFFFF_FFF8, 0x1234_5670, 0x8000_000F, 0x0000_00F7]);
    let inputs: Vec<[u32; 4]> = cases.iter().map(|&x| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let want = nibble(cases[i]).to_bits();
        assert_eq!(
            got, want,
            "{backend}: v_cvt_off_f32_i4({:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
            inp[0],
            f32::from_bits(got),
        );
    });
    // Spot-check the reference itself: 0x5 -> 0.3125, 0x8 -> -0.5, 0xF -> -0.0625.
    assert_eq!(nibble(0xABCD_EF05), 0.3125);
    assert_eq!(nibble(8), -0.5);
    assert_eq!(nibble(0xF), -0.0625);
}

#[test]
fn v_fract_f32_never_returns_one() {
    // v_fract_f32 v0, v1 (VOP1 opcode 0x020)
    let shader = build_valu_cs(&[vop1(0x020, 0, vgpr(1))]);
    let f = f32::to_bits;
    let below_one = 0x3F7F_FFFFu32;

    let cases: &[(u32, Option<u32>)] = &[
        // Tiny negatives: must clamp strictly below 1.0 (not 0x3f800000).
        (f(-1e-10), Some(below_one)),
        (f(-1e-8), Some(below_one)),
        (f(-2.0f32.powi(-25)), Some(below_one)),
        (f(-2.0f32.powi(-24)), Some(below_one)),
        // Normal values.
        (f(0.0), Some(f(0.0))),
        (f(-0.0), Some(f(0.0))),
        (f(0.25), Some(f(0.25))),
        (f(3.75), Some(f(0.75))),
        (f(-0.25), Some(f(0.75))),
        (f(-3.25), Some(f(0.75))),
        (f(1.0), Some(f(0.0))),
        (f(-1.0), Some(f(0.0))),
        // Non-finite inputs yield NaN.
        (f(f32::NAN), None),
        (f(f32::INFINITY), None),
        (f(f32::NEG_INFINITY), None),
    ];

    let inputs: Vec<[u32; 4]> = cases.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        match expect {
            Some(want) => assert_eq!(
                got,
                want,
                "{backend}: v_fract_f32({:?} / {:#010x}) = {got:#010x} ({:?}), want {want:#010x}",
                f32::from_bits(inp[0]),
                inp[0],
                f32::from_bits(got),
            ),
            None => assert!(
                f32::from_bits(got).is_nan(),
                "{backend}: v_fract_f32({:?}) = {got:#010x}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });
}

#[test]
fn v_fract_f16_never_returns_one() {
    // v_fract_f16 v0, v1 (VOP1 opcode 0x05f)
    let shader = build_valu_cs(&[vop1(0x05f, 0, vgpr(1))]);
    let below_one = 0x3BFFu16; // 1.0 - 2^-11

    assert_eq!(f16_to_f32(0x9400), -2.0f32.powi(-10));
    assert_eq!(f16_to_f32(0x9000), -2.0f32.powi(-11));
    assert_eq!(f16_to_f32(0x8C00), -2.0f32.powi(-12));

    let cases: &[(u16, Option<u16>)] = &[
        // Negatives in (-2^-11, 0): must clamp to 0x3BFF, never 0x3C00 (1.0).
        (0x8C00, Some(below_one)), // -2^-12
        (0x8400, Some(below_one)), // -2^-14
        (0x8001, Some(below_one)), // -2^-24 (smallest negative f16 subnormal)
        (0x9000, Some(below_one)), // -2^-11 (exact 1 - 2^-11 = 0x3BFF)
        (0x9400, Some(0x3BFE)),    // -2^-10 (exact 1 - 2^-10 = 0x3BFE)
        // Normal values.
        (0x0000, Some(0x0000)), // 0.0 -> 0.0
        (0x3400, Some(0x3400)), // 0.25 -> 0.25
        (0xB400, Some(0x3A00)), // -0.25 -> 0.75
        (0x4300, Some(0x3800)), // 3.5 -> 0.5
        (0x3C00, Some(0x0000)), // 1.0 -> 0.0
        // Non-finite inputs yield NaN.
        (0x7E00, None), // NaN
        (0x7C00, None), // +inf
        (0xFC00, None), // -inf
    ];

    let inputs: Vec<[u32; 4]> = cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&shader, &inputs, |backend, i, inp, got| {
        let expect = cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => assert_eq!(
                got_h, want,
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want {want:#06x}",
                inp[0],
            ),
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_fract_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// RDNA2's flt16_to_snorm16 / flt16_to_unorm16: clamp, scale by 32767 or
/// 65535, round the exact product to nearest even; NaN gives 0.
fn cvt_norm16_reference(bits: u16, signed: bool) -> u16 {
    let x = f16_to_f32(bits);
    if x.is_nan() {
        return 0;
    }
    let (lo, scale) = if signed {
        (-1.0, 32767.0)
    } else {
        (0.0, 65535.0)
    };
    // f64 holds the product of an f16 and a 16-bit scale exactly.
    let p = f64::from(x.clamp(lo, 1.0)) * scale;
    p.round_ties_even() as i32 as u16
}

#[test]
fn v_cvt_norm_i16_and_u16_f16_round_the_exact_product() {
    // Every 251st f16 pattern, plus the values where an f32 product would
    // round before the rounding to an integer, ties, and the edges.
    let mut bits: Vec<u16> = (0..=u16::MAX).step_by(251).collect();
    bits.extend([
        0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x0400, 0x3800, 0xB800, 0x3801, 0xB801, 0x37FF,
        0x3BFF, 0xBBFF, 0x3C00, 0xBC00, 0x3C01, 0x7BFF, 0x7C00, 0xFC00, 0x7E00, 0xFE00, 0x7C01,
        0x3A9B, 0xBA9B, 0x2E66, 0x1400,
    ]);
    for (op, signed, what) in [
        (0x063u16, true, "v_cvt_norm_i16_f16"),
        (0x064, false, "v_cvt_norm_u16_f16"),
    ] {
        // v_cvt_norm_*16_f16 v0, v1 (VOP1); v0's high half starts at 0.
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = bits.iter().map(|&b| [u32::from(b), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = cvt_norm16_reference(bits[i], signed);
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                bits[i],
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f32_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x035, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x036, 0, vgpr(1))]);
    let f = f32::to_bits;
    let half_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;

    let sin_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(0.0)),
        (0.25, Some(1.0)),
        (255.0, Some(0.0)),
        (255.25, Some(1.0)),
        (255.5, Some(0.0)),
        (-255.25, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_sin_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_sin_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    let cos_cases: &[(f32, Option<f32>)] = &[
        (0.0, Some(1.0)),
        (0.25, Some(0.0)),
        (255.0, Some(1.0)),
        (255.25, Some(0.0)),
        (255.5, Some(-1.0)),
        (-255.5, Some(-1.0)),
        (10_000.125, Some(half_sqrt2)),
        (f32::NAN, None),
        (f32::INFINITY, None),
        (f32::NEG_INFINITY, None),
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases.iter().map(|&(x, _)| [f(x), 0, 0, 0]).collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_f = f32::from_bits(got);
        match expect {
            Some(want) => assert!(
                (got_f - want).abs() < 1e-5,
                "{backend}: v_cos_f32({}) = {got_f}, want {want} (err = {})",
                f32::from_bits(inp[0]),
                (got_f - want).abs(),
            ),
            None => assert!(
                got_f.is_nan(),
                "{backend}: v_cos_f32({}) = {got_f}, want NaN",
                f32::from_bits(inp[0]),
            ),
        }
    });

    // Main's `trig_turns` preserves sin(-0.0) = -0.0 and produces exact +0.0 at
    // half-turns (sin) and quarter-turns (cos).
    let exact_sin: &[(u32, u32)] = &[
        (0x8000_0000, 0x8000_0000), // sin(-0.0) = -0.0
        (f(255.0), 0),              // sin(255.0) = +0.0
        (f(255.5), 0),              // sin(255.5) = +0.0
    ];
    let exact_sin_in: Vec<[u32; 4]> = exact_sin.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&sin_shader, &exact_sin_in, |backend, i, inp, got| {
        let want = exact_sin[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_sin_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
    let exact_cos: &[(u32, u32)] = &[
        (f(0.25), 0),   // cos(0.25) = +0.0
        (f(255.25), 0), // cos(255.25) = +0.0
    ];
    let exact_cos_in: Vec<[u32; 4]> = exact_cos.iter().map(|&(x, _)| [x, 0, 0, 0]).collect();
    run_cases(&cos_shader, &exact_cos_in, |backend, i, inp, got| {
        let want = exact_cos[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cos_f32({:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0],
        );
    });
}

#[test]
fn v_cvt_u16_and_i16_f16_saturate() {
    // v_mov_b32 v0, 0; v_cvt_u16_f16 / v_cvt_i16_f16 v0, v1 (VOP1 0x052 / 0x053).
    let cases: &[(u16, u16, u16)] = &[
        // (f16 input, u16 result, i16 result)
        (0x0000, 0, 0),          // 0
        (0x8000, 0, 0),          // -0
        (0x3E00, 1, 1),          // 1.5 -> 1 (toward zero)
        (0xBE00, 0, 0xFFFF),     // -1.5 -> 0 / -1
        (0x57D0, 125, 125),      // 125.0
        (0x7800, 32768, 32767),  // 32768 saturates i16
        (0x77FF, 32752, 32752),  // 32752, the largest f16 below 32768
        (0xF800, 0, 0x8000),     // -32768 fits i16
        (0xF801, 0, 0x8000),     // -32800 saturates i16
        (0x7BFF, 65504, 32767),  // max finite f16
        (0xFBFF, 0, 0x8000),     // -65504
        (0x7C00, 0xFFFF, 32767), // +inf
        (0xFC00, 0, 0x8000),     // -inf
        (0x7E00, 0, 0),          // NaN
        (0xFE00, 0, 0),          // -NaN
        (0x0001, 0, 0),          // smallest subnormal
    ];
    for (op, what, pick) in [
        (0x052u16, "v_cvt_u16_f16", 0usize),
        (0x053, "v_cvt_i16_f16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_cvt_f16_u16_and_i16_widen_the_integer_first() {
    // v_mov_b32 v0, 0; v_cvt_f16_u16 / v_cvt_f16_i16 v0, v1 (VOP1 0x050 / 0x051).
    // Only integers f16 holds exactly: how the others round is the narrowing's
    // business, not this conversion's. `None`: the result depends on rounding.
    let cases: &[(u16, Option<u16>, Option<u16>)] = &[
        // (u16 / i16 input bits, f16 of the unsigned value, f16 of the signed value)
        (0x0000, Some(0x0000), Some(0x0000)), // 0
        (0x0001, Some(0x3C00), Some(0x3C00)), // 1.0
        (0x07FF, Some(0x67FF), Some(0x67FF)), // 2047, the last integer below 2048 that f16 holds
        (0x8000, Some(0x7800), Some(0xF800)), // 32768 / -32768
        (0xFC00, Some(0x7BE0), Some(0xE400)), // 64512 / -1024
        (0xFFFE, None, Some(0xC000)),         // 65534 does not fit f16 / -2
        (0xFFFF, None, Some(0xBC00)),         // 65535 does not fit f16 / -1
    ];
    for (op, what, pick) in [
        (0x050u16, "v_cvt_f16_u16", 0usize),
        (0x051, "v_cvt_f16_i16", 1),
    ] {
        let shader = build_valu_cs(&[vop1(0x001, 0, 128), vop1(op, 0, vgpr(1))]);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| [u32::from(c.0), 0, 0, 0]).collect();
        run_cases(&shader, &inputs, |backend, i, _, got| {
            let want = if pick == 0 { cases[i].1 } else { cases[i].2 };
            let Some(want) = want else { return };
            assert_eq!(
                got,
                u32::from(want),
                "{backend}: {what}({:#06x}) = {got:#010x}, want {want:#06x}",
                cases[i].0,
            );
        });
    }
}

#[test]
fn v_sin_and_v_cos_f16_reduce_large_turns() {
    let sin_shader = build_valu_cs(&[vop1(0x060, 0, vgpr(1))]);
    let cos_shader = build_valu_cs(&[vop1(0x061, 0, vgpr(1))]);

    // Sanity-check the f16 bit encodings of large turns.
    assert_eq!(f16_to_f32(0x5BF8), 255.0);
    assert_eq!(f16_to_f32(0x5BFA), 255.25);
    assert_eq!(f16_to_f32(0x5BFC), 255.5);
    assert_eq!(f16_to_f32(0x7BFF), 65504.0);

    let sin_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(0.0)),  // 255.0
        (0x5BFA, Some(1.0)),  // 255.25
        (0x5BFC, Some(0.0)),  // 255.5
        (0xDBFA, Some(-1.0)), // -255.25
        (0x7BFF, Some(0.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let sin_inputs: Vec<[u32; 4]> = sin_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&sin_shader, &sin_inputs, |backend, i, inp, got| {
        let expect = sin_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_sin_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_sin_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });

    let cos_cases: &[(u16, Option<f32>)] = &[
        (0x5BF8, Some(1.0)),  // 255.0
        (0x5BFA, Some(0.0)),  // 255.25
        (0x5BFC, Some(-1.0)), // 255.5
        (0x7BFF, Some(1.0)),  // 65504.0 (largest finite f16, exact integer)
        (0x7E00, None),       // NaN
        (0x7C00, None),       // +inf
        (0xFC00, None),       // -inf
    ];
    let cos_inputs: Vec<[u32; 4]> = cos_cases
        .iter()
        .map(|&(x, _)| [u32::from(x), 0, 0, 0])
        .collect();
    run_cases(&cos_shader, &cos_inputs, |backend, i, inp, got| {
        let expect = cos_cases[i].1;
        let got_h = got as u16;
        match expect {
            Some(want) => {
                let got_f = f16_to_f32(got_h);
                assert!(
                    (got_f - want).abs() < 1e-3,
                    "{backend}: v_cos_f16({:#06x} = {}) = {got_f} ({got_h:#06x}), want {want}",
                    inp[0],
                    f16_to_f32(inp[0] as u16),
                );
            }
            None => assert!(
                is_f16_nan(got_h),
                "{backend}: v_cos_f16({:#06x}) = {got_h:#06x}, want NaN",
                inp[0],
            ),
        }
    });
}

/// A 32-bit literal in a signed 64-bit operand is sign-extended (an
/// unsigned one's is zero-extended): -100 compared with `v[1:2]`, and
/// shifted by `s_ashr_i64`.
#[test]
fn signed_64_bit_operands_sign_extend_their_literal() {
    const MINUS_100: u32 = -100_i32 as u32;
    let check = |name: &str, body: &[u32], cases: &[([u32; 4], u32)]| {
        let shader = build_valu_cs(body);
        let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
        run_cases(&shader, &inputs, |backend, i, inp, got| {
            let want = cases[i].1;
            assert_eq!(got, want, "{backend}: {name} {inp:x?}");
        });
    };
    // v_cmp_*_i64 s6, v[1:2], -100 (VOP3, literal src1), then
    // v_cndmask_b32 v0, 0, 1, s6.
    let compare = |op: u16| {
        let [a, b] = vop3(op, 6, vgpr(1), 255, 0);
        let [c, d] = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        [a, b, MINUS_100, c, d]
    };
    // An i64 as the two input dwords of `v[1:2]`.
    let pair = |n: i64| [n as u32, (n >> 32) as u32, 0, 0];
    let eq = [(pair(-100), 1), ([MINUS_100, 0, 0, 0], 0), (pair(0), 0)];
    check("v_cmp_eq_i64", &compare(0x0A2), &eq);
    let lt = [(pair(-200), 1), (pair(-50), 0), (pair(0), 0)];
    check("v_cmp_lt_i64", &compare(0x0A1), &lt);
    // s_ashr_i64 s[6:7], -100, 4; v_mov_b32 v0, s7: the high dword.
    let ashr = [sop2(0x23, 6, 255, 0x84), MINUS_100, vop1(0x001, 0, sgpr(7))];
    check("s_ashr_i64", &ashr, &[([0; 4], u32::MAX)]);
}

#[test]
fn bfe_signed_and_unsigned_match_rdna2() {
    // v_bfe_i32 v0, v1, v2, v3 (VOP3 opcode 0x149)
    let v_signed = build_valu_cs(&vop3(0x149, 0, vgpr(1), vgpr(2), vgpr(3)));
    // v_bfe_u32 v0, v1, v2, v3 (VOP3 opcode 0x148)
    let v_unsigned = build_valu_cs(&vop3(0x148, 0, vgpr(1), vgpr(2), vgpr(3)));
    // s_bfe_i32 s6, s0, s1 (SOP2 opcode 0x28: s1 = off[4:0] | width[22:16]); v_mov_b32 v0, s6
    let s_signed = build_valu_cs(&[sop2(0x28, 6, 0, 1), vop1(0x001, 0, sgpr(6))]);

    let signed_cases: &[([u32; 4], u32)] = &[
        // Field crosses bit 31: extends from bit 31.
        ([0xd2f9_0159, 28, 16, 0], 0xffff_fffd),
        ([0x8000_0000, 31, 4, 0], 0xffff_ffff),
        ([0x7000_0000, 28, 8, 0], 0x0000_0007),
        // Normal fields inside 32 bits.
        ([0x0000_0080, 4, 4, 0], 0xffff_fff8),
        ([0x0000_0070, 4, 4, 0], 0x0000_0007),
        // Edge widths: 0 yields 0; v_bfe masks width to 5 bits (32 -> 0),
        // while 31 extracts all remaining bits from offset 4.
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0xf800_0000),
    ];
    let v_in: Vec<[u32; 4]> = signed_cases.iter().map(|c| c.0).collect();
    run_cases(&v_signed, &v_in, |backend, i, inp, got| {
        let want = signed_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_i32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
    // v_bfe_i32 width is 5-bit (src2[4:0]), so width=32 wraps to 0.
    run_cases(
        &v_signed,
        &[[0x8000_0000, 4, 32, 0]],
        |backend, _, _, got| {
            assert_eq!(got, 0, "{backend}: v_bfe_i32 width 32 should mask to 0");
        },
    );

    // s_bfe_i32 width is 7-bit (src1[22:16]), so width=32 keeps all 32 bits.
    let mut s_cases = signed_cases.to_vec();
    s_cases.push(([0x8000_0000, 4, 32, 0], 0xf800_0000));
    let s_in: Vec<[u32; 4]> = s_cases
        .iter()
        .map(|&(inp, _)| [inp[0], (inp[1] & 31) | ((inp[2] & 0x7F) << 16), 0, 0])
        .collect();
    run_cases(&s_signed, &s_in, |backend, i, inp, got| {
        let want = s_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: s_bfe_i32({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1],
        );
    });

    let unsigned_cases: &[([u32; 4], u32)] = &[
        ([0xd2f9_0159, 28, 16, 0], 0x0000_000d),
        ([0x8000_0000, 31, 4, 0], 0x0000_0001),
        ([0xffff_ffff, 4, 0, 0], 0),
        ([0x8000_0000, 4, 31, 0], 0x0800_0000),
    ];
    let u_in: Vec<[u32; 4]> = unsigned_cases.iter().map(|c| c.0).collect();
    run_cases(&v_unsigned, &u_in, |backend, i, inp, got| {
        let want = unsigned_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_bfe_u32({:#010x}, {}, {}) = {got:#010x}, want {want:#010x}",
            inp[0], inp[1], inp[2],
        );
    });
}

#[test]
fn f16_omod_applies_before_clamp() {
    // VOP3 v_add_f16 (0x132 = 0x100 + VOP2 0x32): v0 = (v1 + v2) * omod [clamp]
    let mul2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 1, false));
    let mul4 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 2, false));
    let div2 = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, false));
    let div2_clamp = build_valu_cs(&vop3_mod(0x132, 0, vgpr(1), vgpr(2), 0, 0, 0, 3, true));

    // f16 constants: 0.25=0x3400, 0.5=0x3800, 0.75=0x3A00, 1.0=0x3C00, 1.5=0x3E00, 2.0=0x4000, -0.5=0xB800
    run_cases(
        &mul2,
        &[[0x3400, 0x3800, 0, 0]], // (0.25 + 0.5) * 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 mul:2");
        },
    );
    run_cases(
        &mul4,
        &[[0x3400, 0x3400, 0, 0]], // (0.25 + 0.25) * 4 = 2.0 (0x4000)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x4000, "{backend}: v_add_f16 mul:4");
        },
    );
    run_cases(
        &div2,
        &[[0x3E00, 0x3E00, 0, 0]], // (1.5 + 1.5) / 2 = 1.5 (0x3E00)
        |backend, _, _, got| {
            assert_eq!(got as u16, 0x3E00, "{backend}: v_add_f16 div:2");
        },
    );
    // Ordering check: (0.75 + 0.75) = 1.5; div:2 before clamp gives 0.75 (0x3A00),
    // whereas clamp before div:2 would give 0.5 (0x3800).
    let clamp_cases: &[(u16, u16, u16)] = &[
        (0x3A00, 0x3A00, 0x3A00), // (0.75 + 0.75) / 2 = 0.75 -> 0.75
        (0x3E00, 0x3E00, 0x3C00), // (1.5 + 1.5) / 2 = 1.5 -> clamped to 1.0
        (0xB800, 0xB800, 0x0000), // (-0.5 + -0.5) / 2 = -0.5 -> clamped to 0.0
    ];
    let inputs: Vec<[u32; 4]> = clamp_cases
        .iter()
        .map(|&(a, b, _)| [u32::from(a), u32::from(b), 0, 0])
        .collect();
    run_cases(&div2_clamp, &inputs, |backend, i, _, got| {
        let want = clamp_cases[i].2;
        assert_eq!(
            got as u16, want,
            "{backend}: v_add_f16 div:2 clamp case {i}: got {:#06x}, want {want:#06x}",
            got as u16,
        );
    });
}

#[test]
fn cvt_16bit_int_from_f16_saturates_and_honors_modifiers() {
    // VOP3 v_cvt_u16_f16 = 0x1D2 (0x180 + VOP1 0x52)
    // VOP3 v_cvt_i16_f16 = 0x1D3 (0x180 + VOP1 0x53)
    let cvt_u16 = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_u16_abs = build_valu_cs(&vop3_mod(0x1D2, 0, vgpr(1), 0, 0, 1, 0, 0, false));
    let cvt_i16 = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 0, 0, false));
    let cvt_i16_neg = build_valu_cs(&vop3_mod(0x1D3, 0, vgpr(1), 0, 0, 0, 1, 0, false));

    let u16_cases: &[(u16, u16)] = &[
        (0xBC00, 0),      // -1.0 -> 0
        (0xFC00, 0),      // -inf -> 0
        (0x7E00, 0),      // NaN -> 0
        (0x57BC, 123),    // 123.75 -> 123
        (0x7BFF, 65504),  // 65504.0 (max finite f16) -> 65504
        (0x7C00, 0xFFFF), // +inf -> 65535 (saturated)
    ];
    let u16_in: Vec<[u32; 4]> = u16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_u16, &u16_in, |backend, i, inp, got| {
        let want = u16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_u16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_u16_abs,
        &[[0xD7BC, 0, 0, 0]], // |-123.75| -> 123
        |backend, _, _, got| {
            assert_eq!(got as u16, 123, "{backend}: v_cvt_u16_f16 |src0|");
        },
    );

    let i16_cases: &[(u16, u16)] = &[
        (0xFC00, 0x8000),       // -inf -> -32768
        (0xF8E2, 0x8000),       // -40000.0 -> -32768 (saturated, not wrapped)
        (0xD7BC, (-123i16) as u16), // -123.75 -> -123
        (0x7E00, 0),            // NaN -> 0
        (0x57BC, 123),          // 123.75 -> 123
        (0x78E2, 0x7FFF),       // +40000.0 -> 32767 (saturated, not wrapped)
        (0x7C00, 0x7FFF),       // +inf -> 32767
    ];
    let i16_in: Vec<[u32; 4]> = i16_cases.iter().map(|&(x, _)| [u32::from(x), 0, 0, 0]).collect();
    run_cases(&cvt_i16, &i16_in, |backend, i, inp, got| {
        let want = i16_cases[i].1;
        assert_eq!(
            got as u16, want,
            "{backend}: v_cvt_i16_f16({:#06x}) = {:#06x}, want {want:#06x}",
            inp[0], got as u16,
        );
    });
    run_cases(
        &cvt_i16_neg,
        &[[0x57BC, 0, 0, 0], [0x78E2, 0, 0, 0]], // -(123.75) -> -123; -(40000.0) -> -32768
        |backend, i, _, got| {
            let want = [(-123i16) as u16, 0x8000][i];
            assert_eq!(got as u16, want, "{backend}: v_cvt_i16_f16 -src0 case {i}");
        },
    );
}

const S_BARRIER: u32 = 0xBF8A_0000;

/// VOP2 `op vdst, src0, vsrc1`.
const fn vop2(op: u8, vdst: u8, src0: u16, vsrc1: u8) -> u32 {
    (op as u32 & 0x3F) << 25 | (vdst as u32) << 17 | (vsrc1 as u32) << 9 | (src0 as u32 & 0x1FF)
}

/// DS `op vdst, addr, data0` at offset 0.
const fn ds(op: u8, vdst: u8, addr: u8, data0: u8) -> [u32; 2] {
    [
        0x36 << 26 | (op as u32) << 18,
        (vdst as u32) << 24 | (data0 as u32) << 8 | addr as u32,
    ]
}

const DS_WRITE_B32: u8 = 0x0D;
const DS_MIN_F32: u8 = 0x12;
const DS_MAX_F32: u8 = 0x13;
const DS_MIN_RTN_F32: u8 = 0x32;
const DS_MAX_RTN_F32: u8 = 0x33;
const DS_READ_B32: u8 = 0x36;

/// What RDNA2 writes for `ds_max_f32` / `ds_min_f32` over a dword holding
/// `old` with `data` (bits), as measured on gfx1036 with f32 denormals kept:
/// an sNaN, data first, then memory, is written quieted; otherwise a NaN
/// loses to a number (two quiet NaNs give the data); numbers compare as
/// floats with -0 below +0.
fn lds_min_max_ref(old: u32, data: u32, max: bool) -> u32 {
    let nan = |x: u32| x & 0x7FFF_FFFF > 0x7F80_0000;
    let snan = |x: u32| nan(x) && x & 0x40_0000 == 0;
    let key = |x: u32| {
        let b = x as i32;
        if b < 0 { b ^ 0x7FFF_FFFF } else { b }
    };
    if snan(data) {
        return data | 0x40_0000;
    }
    if snan(old) {
        return old | 0x40_0000;
    }
    match (nan(data), nan(old)) {
        (true, true) | (false, true) => return data,
        (true, false) => return old,
        (false, false) => {}
    }
    let take = if max {
        key(data) > key(old)
    } else {
        key(data) < key(old)
    };
    if take { data } else { old }
}

#[test]
fn ds_min_max_f32_order_floats_and_return_the_old_value() {
    let values = [
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.5,
        -2.5,
        1e-30,
        -1e-30,
        f32::from_bits(1), // denormal
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::MAX,
        f32::MIN,
    ];
    // Quiet and signalling NaNs of both signs, in memory and as data.
    let nans = [
        0x7FC0_0000u32,
        0xFFC0_0000,
        0x7F80_0001,
        0xFFA0_0000,
        0x7FBF_FFFF,
    ];
    let bits: Vec<u32> = values.iter().map(|v| v.to_bits()).chain(nans).collect();
    let mut cases: Vec<[u32; 4]> = Vec::new();
    for &old in &bits {
        for &data in &bits {
            cases.push([old, data, 0, 0]);
        }
    }
    for (rtn, plain, max, what) in [
        (DS_MAX_RTN_F32, DS_MAX_F32, true, "ds_max"),
        (DS_MIN_RTN_F32, DS_MIN_F32, false, "ds_min"),
    ] {
        // v5 = 0; LDS[v5] = v1; the atomic with v2; v0 = the old value
        // (rtn) or the dword afterwards (read back).
        let mut returns_old = vec![vop1(0x001, 5, 0x80)];
        returns_old.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_old.extend(ds(rtn, 0, 5, 2));
        let mut returns_new = vec![vop1(0x001, 5, 0x80)];
        returns_new.extend(ds(DS_WRITE_B32, 0, 5, 1));
        returns_new.extend(ds(plain, 0, 5, 2));
        returns_new.extend(ds(DS_READ_B32, 0, 5, 0));
        for (body, old_result) in [(returns_old, true), (returns_new, false)] {
            // One thread with LDS declares WaveSize(32), which WARP's
            // pipeline creation refuses: hardware only (the 64-thread test
            // below runs on WARP too).
            let shader = build_cs(&body, 1, 512);
            run_cases_on(
                &shader,
                &cases,
                &[("hardware", false)],
                |backend, _, inp, got| {
                    let (old, data) = (f32::from_bits(inp[0]), f32::from_bits(inp[1]));
                    let want = if old_result {
                        inp[0]
                    } else {
                        lds_min_max_ref(inp[0], inp[1], max)
                    };
                    assert_eq!(
                        got,
                        want,
                        "{backend}: {what}_f32 on {old:e} [{:#010x}] with {data:e} [{:#010x}] gave {got:#010x}, want {want:#010x} ({})",
                        inp[0],
                        inp[1],
                        if old_result {
                            "old value"
                        } else {
                            "dword after"
                        },
                    );
                },
            );
        }
    }
}

#[test]
fn ds_min_max_f32_are_atomic_across_waves() {
    // 64 threads (two host waves of 32 or one of 64): LDS[0] = v1, then
    // every thread applies (thread - 32) as a float; v0 = LDS[0].
    for (op, max, init, want) in [
        (DS_MAX_F32, true, f32::NEG_INFINITY, 31.0f32),
        (DS_MAX_F32, true, 100.0, 100.0),
        (DS_MIN_F32, false, f32::INFINITY, -32.0),
        (DS_MIN_F32, false, -0.0, -32.0),
    ] {
        let mut body = vec![vop1(0x001, 5, 0x80)];
        body.extend(ds(DS_WRITE_B32, 0, 5, 1));
        body.push(S_BARRIER);
        body.push(vop2(0x26, 6, vgpr(0), 3)); // v_sub_nc_u32 v6, v0, v3
        body.push(vop1(0x005, 6, vgpr(6))); // v_cvt_f32_i32
        body.extend(ds(op, 0, 5, 6));
        body.push(S_BARRIER);
        body.extend(ds(DS_READ_B32, 0, 5, 0));
        let shader = build_cs(&body, 64, 512);
        run_cases(
            &shader,
            &[[init.to_bits(), 0, 32, 0]],
            |backend, _, _, got| {
                assert_eq!(
                    f32::from_bits(got),
                    want,
                    "{backend}: {} from {init} over 64 threads",
                    if max { "max" } else { "min" },
                );
            },
        );
    }
}

#[test]
fn v_cmp_class_f16_matches_rdna2() {
    // v_cmp_class_f16_e64 s6, v1, v2 (VOP3 opcode 0x08f); v_cndmask_b32_e64 v0, 0, 1, s6
    let cmp_lo = build_valu_cs(&[
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[0],
        vop3(0x08f, 6, vgpr(1), vgpr(2), 0)[1],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);
    // Same with op_sel:[1,0] (high half of v1) + neg on src0 (bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x08f, 6, vgpr(1), vgpr(2), 0);
    w0 |= 1 << 11; // opsel[0] = 1 (high half)
    w1 |= 1 << 29; // neg[0] = 1
    let cmp_hi_neg = build_valu_cs(&[
        w0,
        w1,
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[0],
        vop3(0x101, 0, 0x80, 0x81, sgpr(6))[1],
    ]);

    // All 10 IEEE-754 binary16 classes: bit k corresponds to class k.
    let classes: &[(u32, u32)] = &[
        (0x7C01, 0), // sNaN (+sNaN, quiet bit 0x0200 clear)
        (0xFDFF, 0), // -sNaN (sign ignored for NaNs)
        (0x7E00, 1), // +qNaN
        (0xFE01, 1), // -qNaN
        (0xFC00, 2), // -inf
        (0xBC00, 3), // -normal (-1.0)
        (0x8001, 4), // -denormal
        (0x8000, 5), // -0.0
        (0x0000, 6), // +0.0
        (0x0001, 7), // +denormal
        (0x3C00, 8), // +normal (+1.0)
        (0x7C00, 9), // +inf
    ];
    let mut cases = Vec::new();
    for &(bits, cls) in classes {
        // Exact class bit -> true; complement of class bit across all 10 bits -> false.
        cases.push(([bits, 1u32 << cls, 0, 0], 1u32));
        cases.push(([bits, 0x3FF & !(1u32 << cls), 0, 0], 0u32));
    }
    let inputs: Vec<[u32; 4]> = cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_lo, &inputs, |backend, i, inp, got| {
        let want = cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16({:#06x}, {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });

    // High half with neg: high half 0x3C00 (+1.0) negated -> -normal (class 3);
    // high half 0x7C01 (+sNaN) negated -> still sNaN (class 0, not quieted).
    let hi_cases: &[([u32; 4], u32)] = &[
        ([0x3C00_0000, 1 << 3, 0, 0], 1),
        ([0x3C00_0000, 1 << 8, 0, 0], 0),
        ([0x7C01_FFFF, 1 << 0, 0, 0], 1),
        ([0x7C01_FFFF, 1 << 1, 0, 0], 0),
    ];
    let hi_in: Vec<[u32; 4]> = hi_cases.iter().map(|c| c.0).collect();
    run_cases(&cmp_hi_neg, &hi_in, |backend, i, inp, got| {
        let want = hi_cases[i].1;
        assert_eq!(
            got, want,
            "{backend}: v_cmp_class_f16(-hi({:#010x}), {:#05x}) = {got}, want {want}",
            inp[0], inp[1],
        );
    });
}

#[test]
fn v_cvt_f32_f64_matches_ieee754() {
    // v_cvt_f32_f64 v0, v[1:2] (VOP1 opcode 0x00f -> VOP3 opcode 0x18f).
    let plain = build_valu_cs(&[vop1(0x00f, 0, vgpr(1))]);
    // Same with abs + neg on src0 (VOP3: abs[0] = bit 8 of w0, neg[0] = bit 29 of w1).
    let [mut w0, mut w1] = vop3(0x18f, 0, vgpr(1), 0, 0);
    w0 |= 1 << 8;
    w1 |= 1 << 29;
    let neg_abs = build_valu_cs(&[w0, w1]);

    let values: &[f64] = &[
        0.0,
        -0.0,
        1.0,
        -1.0,
        3.5,
        -123456.75,
        // Tie-to-even: even mantissa stays, odd mantissa rounds up.
        f64::from_bits(0x3FF0_0000_1000_0000), // 1.0 + 0.5 ulp(f32) -> 1.0
        f64::from_bits(0x3FF0_0000_3000_0000), // (1.0 + 1 ulp) + 0.5 ulp -> 1.0 + 2 ulp
        f64::from_bits(0x3FF0_0000_1000_0001), // > 0.5 ulp -> rounds up
        // Largest finite f32 and overflow to +Inf / -Inf.
        f64::from(f32::MAX),
        f64::from(f32::MAX) * 1.000_001,
        -f64::from(f32::MAX) * 1.000_001,
        1e300,
        // Subnormal range (exp64 in [874, 896]), including carry into smallest normal.
        f64::from(f32::MIN_POSITIVE),
        f64::from(f32::MIN_POSITIVE) * 0.5,
        f64::from(f32::MIN_POSITIVE) * 0.75,
        f64::from(f32::from_bits(0x007F_FFFF)),
        f64::from_bits(0x380F_FFFF_F000_0000), // max subnormal + 0.5 ulp -> min normal
        // exp64 == 873: exact 2^-150 ties to 0; strictly greater rounds to min subnormal (1).
        f64::from_bits(0x3690_0000_0000_0000),
        f64::from_bits(0x3690_0000_0000_0001),
        f64::from_bits(0xB690_0000_0000_0001),
        // Deep underflow -> +/-0.0.
        1e-200,
        -1e-200,
        // Infinities and NaNs.
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7FF8_0000_0000_0000), // +qNaN
        f64::from_bits(0x7FF0_0000_0000_0001), // +sNaN -> quieted
    ];
    let inputs: Vec<[u32; 4]> = values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&plain, &inputs, |backend, i, _, got| {
        let x = values[i];
        let want = (x as f32).to_bits();
        if x.is_nan() {
            assert!(
                f32::from_bits(got).is_nan() && (got & 0x0040_0000) != 0,
                "{backend}: v_cvt_f32_f64({:#018x}) = {got:#010x}, expected qNaN",
                x.to_bits(),
            );
        } else {
            assert_eq!(
                got, want,
                "{backend}: v_cvt_f32_f64({x} / {:#018x}) = {got:#010x}, want {want:#010x}",
                x.to_bits(),
            );
        }
    });

    // -|x| via VOP3 abs + neg modifiers.
    let mod_values: &[f64] = &[2.5, -2.5, 0.0, f64::INFINITY];
    let mod_in: Vec<[u32; 4]> = mod_values
        .iter()
        .map(|&x| {
            let b = x.to_bits();
            [b as u32, (b >> 32) as u32, 0, 0]
        })
        .collect();
    run_cases(&neg_abs, &mod_in, |backend, i, _, got| {
        let x = mod_values[i];
        let want = (-(x.abs()) as f32).to_bits();
        assert_eq!(
            got, want,
            "{backend}: v_cvt_f32_f64(-|{x}|) = {got:#010x}, want {want:#010x}",
        );
    });
}

#[test]
fn whole_wave_f64_sum_and_cvt_f32_f64() {
    let s_or_saveexec_b32 =
        |sdst: u8, src0: u8| 0xBE80_0000 | (sdst as u32) << 16 | 0x3D << 8 | src0 as u32;
    let v_mov_dpp8 = |vdst: u8, vsrc: u8, sel: u32| [vop1(0x001, vdst, 0xE9), vsrc as u32 | (sel << 8)];
    let v_mov_row_mirror = |vdst: u8, vsrc: u8| {
        [
            vop1(0x001, vdst, 0xFA),
            vsrc as u32 | (0x140 << 8) | (1 << 19) | (0xF << 24) | (0xF << 28),
        ]
    };

    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80));
    words.extend_from_slice(&s_load_dwordx4(8, 4));
    // Per-lane f64 input in v[6:7] = (s0, s1).
    words.push(vop1(0x001, 6, sgpr(0)));
    words.push(vop1(0x001, 7, sgpr(1)));
    // Whole-wave f64 sum over Wave32 with s_mov_b32 copy before EXEC restore.
    words.push(s_or_saveexec_b32(12, 0xC1));
    words.extend_from_slice(&vop3(0x101, 2, 0x80, vgpr(6), sgpr(12)));
    words.extend_from_slice(&vop3(0x101, 3, 0x80, vgpr(7), sgpr(12)));
    for sel in [14570689u32, 9658386, 2340] {
        words.extend_from_slice(&v_mov_dpp8(4, 2, sel));
        words.extend_from_slice(&v_mov_dpp8(5, 3, sel));
        words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0)); // v_add_f64
    }
    words.extend_from_slice(&v_mov_row_mirror(4, 2));
    words.extend_from_slice(&v_mov_row_mirror(5, 3));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x378, 4, vgpr(2), 0x80, 0x80)); // v_permlanex16_b32
    words.extend_from_slice(&vop3(0x378, 5, vgpr(3), 0x80, 0x80));
    words.extend_from_slice(&vop3(0x164, 2, vgpr(4), vgpr(2), 0));
    words.extend_from_slice(&vop3(0x360, 14, vgpr(2), 0x80, 0)); // v_readlane_b32 s14, v2, 0
    words.extend_from_slice(&vop3(0x360, 15, vgpr(3), 0x80, 0)); // v_readlane_b32 s15, v3, 0
    words.push(s_mov_b32(16, 14));
    words.push(s_mov_b32(17, 15));
    words.push(s_mov_b32(126, 12)); // restore exec_lo
    words.push(vop1(0x00f, 0, sgpr(16))); // v_cvt_f32_f64 v0, s[16:17]
    words.extend_from_slice(&buffer_store_dword(0, 8, 0));
    words.push(S_ENDPGM);

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 12,
        layout: vec![
            RootParam {
                root: RootType::Constants,
                table_offset_dwords: 0,
                sgprs: 4,
                base_register: 0,
                user_data: 0,
                slot: 0,
            },
            RootParam {
                root: RootType::Table,
                table_offset_dwords: 0,
                sgprs: 1,
                base_register: 0,
                user_data: 4,
                slot: 1,
            },
        ],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 4,
            user_data: 4,
            slot: 1,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1,
            threads: [4, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");

    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: Vec::new(),
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }

    let cases: &[(f64, f32)] = &[(1.25, 5.0), (-0.75, -3.0)];
    let _one = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // WARP does not support fixed WaveSize(32) on SM 6.6.
            Err(e) if warp => {
                eprintln!("{label}: compute_pipeline skipped ({e})");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        for &(x, want) in cases {
            let b = x.to_bits();
            let inputs = [b as u32, (b >> 32) as u32, 0, 0];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                ud[u.index as usize] = match (u.root, u.slot) {
                    (RootType::Constants, 0) => inputs[u.dword as usize],
                    (RootType::Table, 1) => 0,
                    other => panic!("unexpected user data {other:?}"),
                };
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(1)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
            assert_eq!(
                out[0][0],
                want.to_bits(),
                "{label}: wave f64 sum({x}) = {:#010x}, want {:#010x}",
                out[0][0],
                want.to_bits(),
            );
        }
    }
}

/// Whether the hardware adapter is an AMD RDNA card's: what the translator
/// writes for one ([`super::rdna`]) holds there.
fn rdna_card() -> bool {
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::*;
    let Ok(host) = Host::new(false) else { return false };
    let mut waves = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: plain queries.
    unsafe {
        let amd = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
            .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter1>(host.device.GetAdapterLuid()))
            .and_then(|a| a.GetDesc1())
            .is_ok_and(|d| d.VendorId == 0x1002);
        amd && host
            .device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut waves).cast(),
                size_of_val(&waves) as u32,
            )
            .is_ok()
            && waves.WaveLaneCountMin == 32
            && waves.WaveLaneCountMax == 64
    }
}

/// The forms written for the console's own kind of hardware give, on such a
/// card, what the ones spelled out for every host give; a legacy product's
/// zero may have the other sign.
#[test]
fn rdna_forms_give_what_the_spelled_out_ones_give() {
    if !rdna_card() {
        eprintln!("no RDNA card; skipped");
        return;
    }
    const ONE: u16 = 242;
    const MINUS_ONE: u16 = 243;
    let (a, b, c) = (vgpr(1), vgpr(2), vgpr(3));
    let saturated = vop3_mod(0x110, 5, vgpr(1), vgpr(1), 0, 0, 0, 0, true);
    let two = |x: [u32; 2], y: [u32; 2]| [x, y].concat();
    let bodies: [(&str, Vec<u32>, bool); 14] = [
        ("max", vop3(0x110, 0, a, b, 0).to_vec(), true),
        ("min", vop3(0x10F, 0, a, b, 0).to_vec(), true),
        ("max3", vop3(0x154, 0, a, b, c).to_vec(), true),
        ("min3", vop3(0x151, 0, a, b, c).to_vec(), true),
        ("med3 0 1", vop3(0x157, 0, a, 128, ONE).to_vec(), true),
        ("med3 1 0", vop3(0x157, 0, ONE, a, 128).to_vec(), true),
        ("med3 1 -1", vop3(0x157, 0, ONE, MINUS_ONE, a).to_vec(), true),
        ("fract", vec![vop1(0x020, 0, a)], true),
        ("mul_legacy", vop3(0x107, 0, a, b, 0).to_vec(), false),
        ("mul_legacy by itself", vop3(0x107, 0, a, a, 0).to_vec(), false),
        ("mul_legacy by a constant", vop3(0x107, 0, a, MINUS_ONE, 0).to_vec(), false),
        ("mul_legacy of a saturated", two(saturated, vop3(0x107, 0, vgpr(5), b, 0)), false),
        ("mad_legacy", vop3(0x140, 0, a, b, c).to_vec(), false),
        ("max of a mul_legacy", two(vop3(0x107, 5, a, b, 0), vop3(0x110, 0, vgpr(5), c, 0)), false),
    ];
    let numbers = [0.0f32, -0.0, 1.0, -1.0, 0.3, -2.5e-7, 3e38, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    let mut cases = Vec::new();
    for (i, x) in numbers.iter().enumerate() {
        for (j, y) in numbers.iter().enumerate() {
            let z = numbers[(3 * i + j) % numbers.len()];
            cases.push([x.to_bits(), y.to_bits(), z.to_bits(), 0]);
        }
    }
    for (name, body, exact) in bodies {
        let mut got: Vec<Vec<u32>> = Vec::new();
        let mut written = Vec::new();
        for bits in [0, super::rdna::ALL] {
            lower::TEST_RDNA.set(Some(bits));
            let shader = build_valu_cs(&body);
            lower::TEST_RDNA.set(None);
            written.push(shader.dxil.clone());
            let mut all = Vec::new();
            run_cases_on(&shader, &cases, &[("hardware", false)], |_, _, _, r| all.push(r));
            got.push(all);
        }
        assert_ne!(written[0], written[1], "{name}: written alike");
        for (case, (&was, &is)) in cases.iter().zip(got[0].iter().zip(&got[1])) {
            let (x, y) = (f32::from_bits(was), f32::from_bits(is));
            let same = was == is || x.is_nan() && y.is_nan() || !exact && x == 0.0 && y == 0.0;
            assert!(same, "{name} of {:?}: {x} ({was:#x}), now {y} ({is:#x})", case.map(f32::from_bits));
        }
    }
}

/// MUBUF `op vdata, vaddr, s[srsrc:srsrc+3], 0 idxen`.
const fn mubuf_idxen(op: u8, vdata: u8, vaddr: u8, srsrc: u8) -> [u32; 2] {
    [
        0x38 << 26 | (op as u32) << 18 | 1 << 13,
        0x80 << 24 | ((srsrc as u32 >> 2) & 0x1F) << 16 | (vdata as u32) << 8 | vaddr as u32,
    ]
}

/// SMEM `s_load_dwordx4 s[dst:dst+3], s[base:base+1], byte_offset` (soffset = null).
const fn s_load_dwordx4_off(dst: u8, base: u8, byte_offset: u32) -> [u32; 2] {
    [
        0x3D << 26 | 0x02 << 18 | (dst as u32) << 6 | (base as u32 >> 1),
        (125 << 25) | (byte_offset & 0x1F_FFFF),
    ]
}

/// Builds a compute shader whose descriptors live in a `RootType::Table` at
/// `s4` (like `compute_graph_ops` and `ped_standard_head_cs`).
fn build_table_typed_cs(body: &[u32], used: Vec<UsedResource>) -> HostShader {
    let mut words = Vec::new();
    words.push(s_mov_b32(5, 0x80)); // s5 = 0
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 4,
            slot: 0,
        }],
        used,
        io: StageIo::Cs(ComputeIo {
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [4, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res =
        resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    })
    .expect("lower");
    let shader = HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    shader
}

/// `compute_graph_ops` (facial lip-sync & wrapped submesh solver) binds
/// `R16G16B16A16_UINT` / `R16_UINT` UAVs via `RootType::Table` (which on NVIDIA
/// are snapshotted into dynamic descriptor slots) and reads/writes FP16 bit
/// patterns with `buffer_load_format_d16_xyzw` and `buffer_store_format_d16_xyzw`.
/// Verify bit-exact round-trips for UINT, SINT, and FLOAT views (including
/// non-zero table offsets matching dynamic descriptor snapshots) on hardware
/// GPU and WARP.
#[test]
fn typed_d16_table_uav_roundtrip_uint_sint_and_float() {
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R16G16B16A16_FLOAT, DXGI_FORMAT_R16G16B16A16_SINT,
        DXGI_FORMAT_R16G16B16A16_UINT,
    };
    let _serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut body = Vec::new();
    // s[8:11] = table[0] (src typed UAV), s[12:15] = table[1] (dst typed UAV),
    // s[16:19] = table[2] (raw UAV dump of loaded v[2:3]).
    body.extend_from_slice(&s_load_dwordx4_off(8, 4, 0));
    body.extend_from_slice(&s_load_dwordx4_off(12, 4, 32));
    body.extend_from_slice(&s_load_dwordx4_off(16, 4, 64));
    // buffer_load_format_d16_xyzw v[2:3], v0, s[8:11], 0 idxen (op 0x83)
    body.extend_from_slice(&mubuf_idxen(0x83, 2, 0, 8));
    // buffer_store_format_d16_xyzw v[2:3], v0, s[12:15], 0 idxen (op 0x87)
    body.extend_from_slice(&mubuf_idxen(0x87, 2, 0, 12));
    // Also dump thread 0's loaded v[2:3] into raw UAV (table[2]) to verify the load itself.
    // v4 = v0 << 3 (byte offset = thread * 8)
    body.push(0x3408_0083); // v_lshlrev_b32 v4, 3, v0
    // buffer_store_dwordx2 v[2:3], v4, s[16:19], 0 offen (op 0x1D, offen bit 12)
    body.extend_from_slice(&[
        0x38 << 26 | 0x1D << 18 | 1 << 12,
        0x80 << 24 | (16u32 >> 2) << 16 | (2u32 << 8) | 4u32,
    ]);
    let used = (0..3u32)
        .map(|k| UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: k * 8,
            lower_bound: k,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 0,
        })
        .collect();
    let shader = build_table_typed_cs(&body, used);
    assert!(
        shader.root_srvs.contains(&super::SLOT_TABLE_STRIDES),
        "table typed d16 shader should request SLOT_TABLE_STRIDES"
    );
    assert!(
        !shader.root_srvs.contains(&super::SLOT_STRIDES),
        "table-only shader must not trigger the bindless SLOT_STRIDES sweep"
    );

    // 4 elements x 4 u16 channels = 8 u32 dwords.
    // Includes FP16 subnormal (0x0001), -0.0 (0x8000), 1.0 (0x3C00), -1.0 (0xBC00),
    // NaN (0x7E00), max u16 (0xFFFF), and arbitrary patterns.
    let uint_bits: [u32; 8] = [
        0x3C00_0001, 0x7E00_BC00,
        0xFFFF_8000, 0xABCD_1234,
        0x0002_0000, 0x7BFF_FBFF,
        0xDEAD_BEEF, 0xCAFE_BABE,
    ];
    let sint_bits: [u32; 8] = [
        0xFFFF_8000, 0x7FFF_0001,
        0xC000_0000, 0x8001_7FFE,
        0xFFFF_FFFF, 0x0000_0000,
        0xFEDC_BA98, 0x7654_3210,
    ];
    // Normal finite FP16 values that round-trip cleanly through f32.
    let float_bits: [u32; 8] = [
        0x4000_3C00, 0x3800_BC00, // 1.0, 2.0, -1.0, 0.5
        0xC200_3400, 0x4500_0000, // 0.25, -3.0, 0.0, 5.0
        0x3E00_BE00, 0x4400_C400, // -1.5, 1.5, -4.0, 4.0
        0x3C00_0000, 0x4200_C000, // 0.0, 1.0, -2.0, 3.0
    ];
    let zeros = [0u32; 8];

    let cases = [
        ("R16G16B16A16_UINT @0", DXGI_FORMAT_R16G16B16A16_UINT, 8 | super::VIEW_INTEGER, &uint_bits, 0u32),
        ("R16G16B16A16_UINT @3 (dynamic offset)", DXGI_FORMAT_R16G16B16A16_UINT, 8 | super::VIEW_INTEGER, &uint_bits, 3u32),
        ("R16G16B16A16_SINT @0", DXGI_FORMAT_R16G16B16A16_SINT, 8 | super::VIEW_INTEGER | super::VIEW_SIGNED, &sint_bits, 0u32),
        ("R16G16B16A16_FLOAT @0", DXGI_FORMAT_R16G16B16A16_FLOAT, 8, &float_bits, 0u32),
    ];

    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{backend}: compute_pipeline failed: {e}"));
        for &(label, format, stride_word, input, base_slot) in &cases {
            let mut views = Vec::new();
            let mut strides = Vec::new();
            for _ in 0..base_slot {
                views.push(View::RawUav(1));
                strides.push(0u32);
            }
            let uav_base = base_slot as usize;
            views.push(View::TypedUav { data: input, format, elem_bytes: 8 });
            strides.push(stride_word);
            views.push(View::TypedUav { data: &zeros, format, elem_bytes: 8 });
            strides.push(stride_word);
            views.push(View::RawUav(8));
            strides.push(1);

            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if u.root == RootType::Table && u.slot == 0 {
                    ud[u.index as usize] = base_slot * 32;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![
                            (0, Root::Constants(ud)),
                            (1, Root::View { srv: true, data: strides }),
                        ],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{backend} {label}: dispatch failed: {e}"));
            assert_eq!(
                out[uav_base + 2],
                input,
                "{backend} {label}: buffer_load_format_d16_xyzw mismatch"
            );
            assert_eq!(
                out[uav_base + 1],
                input,
                "{backend} {label}: buffer_store_format_d16_xyzw mismatch"
            );
        }
    }
}

/// Non-`d16` typed loads (`buffer_load_format_x`) are used by
/// `CS_BlendshapeTranslationsHSSGraphOutputExtract` (reading `HSSGraphOut`
/// `R16_UINT`) and `CS_PedStandardSkinning_*` (reading bone indices). Verify
/// exact integer bits for `R16_UINT`, `R16_SINT`, and `R32_UINT`.
#[test]
fn typed_non_d16_table_load_preserves_integer_bits() {
    use windows::Win32::Graphics::Dxgi::Common::{
        DXGI_FORMAT_R16_SINT, DXGI_FORMAT_R16_UINT, DXGI_FORMAT_R32_UINT,
    };
    let _serial = ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut body = Vec::new();
    // s[8:11] = table[0] (src typed SRV), s[12:15] = table[1] (raw UAV out)
    body.extend_from_slice(&s_load_dwordx4_off(8, 4, 0));
    body.extend_from_slice(&s_load_dwordx4_off(12, 4, 32));
    // buffer_load_format_x v2, v0, s[8:11], 0 idxen (op 0x00)
    body.extend_from_slice(&mubuf_idxen(0x00, 2, 0, 8));
    // v4 = v0 << 2 (byte offset = thread * 4)
    body.push(0x3408_0082); // v_lshlrev_b32 v4, 2, v0
    // buffer_store_dword v2, v4, s[12:15], 0 offen (op 0x1C, offen bit 12)
    body.extend_from_slice(&[
        0x38 << 26 | 0x1C << 18 | 1 << 12,
        0x80 << 24 | (12u32 >> 2) << 16 | (2u32 << 8) | 4u32,
    ]);
    let used = vec![
        UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 0,
        },
        UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 8,
            lower_bound: 1,
            stride: 4,
            dwords: 4,
            user_data: 4,
            slot: 0,
        },
    ];
    let shader = build_table_typed_cs(&body, used);
    // Non-d16 typed loads stay untouched (no SLOT_TABLE_STRIDES root SRV, no
    // CFG branches) so non-d16 pixel/mesh/particle/skinning shaders keep
    // identical DXIL and pipeline cache keys while still preserving raw bits.
    assert!(
        !shader.root_srvs.contains(&super::SLOT_TABLE_STRIDES),
        "non-d16 typed load should not add SLOT_TABLE_STRIDES"
    );

    let r16_data = [0x3C00_0001u32, 0xFFFF_BC00u32];
    let r32_data = [0xDEAD_BEEFu32, 0x0000_0001, 0x8000_0000, 0x7FFF_FFFF];
    let cases: [(&str, _, u32, &[u32], [u32; 4]); 3] = [
        (
            "R16_UINT",
            DXGI_FORMAT_R16_UINT,
            2,
            &r16_data,
            [0x0001, 0x3C00, 0xBC00, 0xFFFF],
        ),
        (
            "R16_SINT",
            DXGI_FORMAT_R16_SINT,
            2,
            &r16_data,
            [0x0001, 0x3C00, 0xFFFF_BC00, 0xFFFF_FFFF],
        ),
        (
            "R32_UINT",
            DXGI_FORMAT_R32_UINT,
            4,
            &r32_data,
            r32_data,
        ),
    ];

    for (backend, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else { continue };
        let (root, pso) = host
            .compute_pipeline(&shader)
            .unwrap_or_else(|e| panic!("{backend}: compute_pipeline failed: {e}"));
        for &(label, format, elem_bytes, input, want) in &cases {
            let views = [
                View::Buffer {
                    data: input,
                    stride: super::host::Stride::Format(format, elem_bytes),
                },
                View::RawUav(4),
            ];
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if u.root == RootType::Table && u.slot == 0 {
                    ud[u.index as usize] = 0;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &views,
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("{backend} {label}: dispatch failed: {e}"));
            assert_eq!(out[0], want, "{backend} {label}: buffer_load_format_x mismatch");
        }
    }
}

/// `victory_formats::bcpack::zz`, the function the test below runs as console code.
fn zz_reference(v: u8, bits: u32) -> u8 {
    let mask = ((1u32 << bits) - 1) as u8;
    ((v >> 1) ^ if v & 1 != 0 { mask } else { 0 }) & mask
}

/// BCPack's zigzag decode written as console code matches the formats crate's function on
/// the GPU: every symbol at every width, widths past a byte included.
#[test]
fn bcpack_zigzag_decode_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the symbol, v2 = its width in bits; v5 = the width's mask, as a byte.
    body.extend(vop3(0x11A, 5, vgpr(2), 129, 128)); // v_lshlrev_b32 v5, v2, 1
    body.extend(vop3(0x125, 5, 193, vgpr(5), 128)); // v_add_nc_u32 v5, -1, v5
    body.extend(vop3(0x148, 5, vgpr(5), 128, 128 + 8)); // v_bfe_u32 v5, v5, 0, 8
    body.extend(vop3(0x116, 6, 129, vgpr(1), 128)); // v_lshrrev_b32 v6, 1, v1
    body.extend(vop3(0x148, 7, vgpr(1), 128, 129)); // v_bfe_u32 v7, v1, 0, 1
    body.extend(vop3(0x0C5, 106, 128, vgpr(7), 128)); // v_cmp_ne_u32 vcc, 0, v7
    body.extend(vop3(0x101, 8, 128, vgpr(5), vcc)); // v_cndmask_b32 v8, 0, v5, vcc
    body.extend(vop3(0x11D, 9, vgpr(6), vgpr(8), 128)); // v_xor_b32 v9, v6, v8
    body.extend(vop3(0x11B, 0, vgpr(9), vgpr(5), 128)); // v_and_b32 v0, v9, v5
    let shader = build_valu_cs(&body);
    let mut cases = Vec::new();
    let mut want = Vec::new();
    for bits in (0u32..=9).chain([16, 31]) {
        for v in 0u32..256 {
            cases.push([v, bits, 0, 0]);
            want.push(u32::from(zz_reference(v as u8, bits)));
        }
    }
    run_cases(&shader, &cases, |backend, i, inputs, got| {
        assert_eq!(got, want[i], "{backend}: zz{inputs:?}");
    });
}

/// `victory_formats::bcpack::add_index`, the function the test below runs as console code.
fn add_index_reference(cur: i16, bits: u8, refv: i16, ref_bits: u8) -> i16 {
    let up = |v: i16, b: u8| {
        if b == 0 || b > 16 {
            0
        } else {
            (v as u16) << (16 - b)
        }
    };
    let sum = up(cur, bits).wrapping_add(up(refv, ref_bits)) as i16;
    if bits == 0 { 0 } else { sum >> (16 - bits) }
}

fn add_index_body() -> Vec<u32> {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = cur, v2 = bits, v3 = refv, v4 = ref_bits. v6 = cur left-aligned to 16 bits.
    body.extend(vop3(0x126, 5, 128 + 16, vgpr(2), 128)); // v_sub_nc_u32 v5, 16, v2
    body.extend(vop3(0x11A, 6, vgpr(5), vgpr(1), 128)); // v_lshlrev_b32 v6, v5, v1
    body.extend(vop3(0x148, 6, vgpr(6), 128, 128 + 16)); // v_bfe_u32 v6, v6, 0, 16
    body.extend(vop3(0x125, 7, 193, vgpr(2), 128)); // v_add_nc_u32 v7, -1, v2
    body.extend(vop3(0x0C4, 106, 128 + 16, vgpr(7), 128)); // v_cmp_gt_u32 vcc, 16, v7
    body.extend(vop3(0x101, 6, 128, vgpr(6), vcc)); // v_cndmask_b32 v6, 0, v6, vcc
    // v9 = refv left-aligned the same way.
    body.extend(vop3(0x126, 8, 128 + 16, vgpr(4), 128));
    body.extend(vop3(0x11A, 9, vgpr(8), vgpr(3), 128));
    body.extend(vop3(0x148, 9, vgpr(9), 128, 128 + 16));
    body.extend(vop3(0x125, 10, 193, vgpr(4), 128));
    body.extend(vop3(0x0C4, 106, 128 + 16, vgpr(10), 128));
    body.extend(vop3(0x101, 9, 128, vgpr(9), vcc));
    // The 16-bit sum, signed, back down to `bits` bits; nothing at a width of 0.
    body.extend(vop3(0x125, 11, vgpr(6), vgpr(9), 128)); // v_add_nc_u32 v11, v6, v9
    body.extend(vop3(0x149, 11, vgpr(11), 128, 128 + 16)); // v_bfe_i32 v11, v11, 0, 16
    body.extend(vop3(0x118, 0, vgpr(5), vgpr(11), 128)); // v_ashrrev_i32 v0, v5, v11
    body.extend(vop3(0x0C2, 106, 128, vgpr(2), 128)); // v_cmp_eq_u32 vcc, 0, v2
    body.extend(vop3(0x101, 0, vgpr(0), 128, vcc)); // v_cndmask_b32 v0, v0, 0, vcc
    body
}

/// BCPack's index residual plus its reference, written as console code, matches the formats
/// crate's function on the GPU at every width, reference widths out of range included.
#[test]
fn bcpack_index_residual_sum_matches_the_formats_function() {
    let shader = build_valu_cs(&add_index_body());
    let values: [i16; 6] = [0, 1, -1, 0x7FFF, -0x8000, -0x1235];
    let refs: [i16; 3] = [0, 0x2A5, -7];
    let mut cases = Vec::new();
    let mut want = Vec::new();
    for bits in 0u8..=16 {
        for ref_bits in [0u8, 1, 4, 8, 15, 16, 17] {
            for cur in values {
                for refv in refs {
                    cases.push([
                        cur as i32 as u32,
                        u32::from(bits),
                        refv as i32 as u32,
                        u32::from(ref_bits),
                    ]);
                    want.push(add_index_reference(cur, bits, refv, ref_bits) as i32 as u32);
                }
            }
        }
    }
    run_cases(&shader, &cases, |backend, i, inputs, got| {
        assert_eq!(got, want[i], "{backend}: add_index{inputs:?}");
    });
}

/// `victory_formats::bcpack::nibbles`, the function the test below runs as console code.
fn nibbles_reference(b: u8, odd_first: bool) -> [i16; 2] {
    let half = |v: u8| (0..4).fold(0u8, |acc, k| acc | (v >> (2 * k) & 1) << k);
    let z = |u: u8| i16::from(u >> 1) ^ -i16::from(u & 1);
    let (even, odd) = (z(half(b)), z(half(b >> 1)));
    if odd_first { [odd, even] } else { [even, odd] }
}

fn nibbles_body() -> Vec<u32> {
    let vcc = sgpr(106);
    // v1 = the byte, v2 = whether the odd bits come first. v5 and v6 gather the even and
    // the odd bits.
    let mut body = vec![vop1(0x001, 5, 128), vop1(0x001, 6, 128)]; // v_mov_b32 v5, 0; v6, 0
    for k in 0u16..4 {
        for (half, bit) in [(5, 2 * k), (6, 2 * k + 1)] {
            body.extend(vop3(0x148, 7, vgpr(1), 128 + bit, 129)); // v_bfe_u32 v7, v1, bit, 1
            body.extend(vop3(0x36F, half, vgpr(7), 128 + k, vgpr(half))); // v_lshl_or_b32
        }
    }
    // Zigzag to signed, kept as 16 bits: v8 from the even half, v11 from the odd one.
    for (half, out) in [(5, 8), (6, 11)] {
        body.extend(vop3(0x116, 9, 129, vgpr(half), 128)); // v_lshrrev_b32 v9, 1, half
        body.extend(vop3(0x148, 10, vgpr(half), 128, 129)); // v_bfe_u32 v10, half, 0, 1
        body.extend(vop3(0x126, 10, 128, vgpr(10), 128)); // v_sub_nc_u32 v10, 0, v10
        body.extend(vop3(0x11D, out, vgpr(9), vgpr(10), 128)); // v_xor_b32 out, v9, v10
        body.extend(vop3(0x148, out, vgpr(out), 128, 128 + 16)); // v_bfe_u32 out, out, 0, 16
    }
    // The pair in the order asked for, the second in the result's high half.
    body.extend(vop3(0x0C5, 106, 128, vgpr(2), 128)); // v_cmp_ne_u32 vcc, 0, v2
    body.extend(vop3(0x101, 14, vgpr(8), vgpr(11), vcc)); // v_cndmask_b32 v14, v8, v11, vcc
    body.extend(vop3(0x101, 15, vgpr(11), vgpr(8), vcc)); // v_cndmask_b32 v15, v11, v8, vcc
    body.extend(vop3(0x36F, 0, vgpr(15), 128 + 16, vgpr(14))); // v_lshl_or_b32 v0, v15, 16, v14
    body
}

/// BCPack's split of a byte into two signed nibbles, written as console code, matches the
/// formats crate's function on the GPU for every byte in both orders.
#[test]
fn bcpack_nibble_split_matches_the_formats_function() {
    let shader = build_valu_cs(&nibbles_body());
    let mut cases = Vec::new();
    let mut want = Vec::new();
    for odd_first in [false, true] {
        for b in 0u32..256 {
            cases.push([b, u32::from(odd_first), 0, 0]);
            let [first, second] = nibbles_reference(b as u8, odd_first);
            want.push(u32::from(second as u16) << 16 | u32::from(first as u16));
        }
    }
    run_cases(&shader, &cases, |backend, i, inputs, got| {
        assert_eq!(got, want[i], "{backend}: nibbles{inputs:?}");
    });
}

/// `victory_formats::bcpack::bc7_precision` with its table and `pbits`, the function the
/// test below runs as console code.
fn bc7_precision_reference(ctrl: u16) -> [u8; 4] {
    const BC7_BITS: [[u8; 4]; 8] = [
        [4, 4, 4, 0],
        [6, 6, 6, 0],
        [5, 5, 5, 0],
        [7, 7, 7, 0],
        [5, 5, 5, 6],
        [7, 7, 7, 8],
        [7, 7, 7, 7],
        [5, 5, 5, 5],
    ];
    fn pbits(mode: u16) -> bool {
        matches!(mode, 0 | 1 | 3 | 6 | 7)
    }
    let (mode, rot) = (ctrl & 7, usize::from(ctrl >> 3 & 3));
    let mut p = BC7_BITS[usize::from(mode)];
    if rot != 0 {
        p.swap(3, rot - 1);
    } else if pbits(mode) {
        p[0] += 1;
    }
    p
}

fn bc7_precision_body() -> Vec<u32> {
    let vcc = sgpr(106);
    // The table's rows as little-endian dwords, and the modes with a p-bit as a bit set.
    const ROWS: [u32; 8] = [
        0x0004_0404,
        0x0006_0606,
        0x0005_0505,
        0x0007_0707,
        0x0605_0505,
        0x0807_0707,
        0x0707_0707,
        0x0505_0505,
    ];
    let mut body = Vec::new();
    // v1 = the control word; v2 = its mode, v3 = its rotation, v5 = the mode's row.
    body.extend(vop3(0x148, 2, vgpr(1), 128, 128 + 3)); // v_bfe_u32 v2, v1, 0, 3
    body.extend(vop3(0x148, 3, vgpr(1), 128 + 3, 128 + 2)); // v_bfe_u32 v3, v1, 3, 2
    body.extend([vop1(0x001, 5, 255), ROWS[0]]); // v_mov_b32 v5, row 0
    for (mode, row) in ROWS.iter().enumerate().skip(1) {
        body.extend([vop1(0x001, 6, 255), *row]); // v_mov_b32 v6, row
        body.extend(vop3(0x0C2, 106, 128 + mode as u16, vgpr(2), 128)); // v_cmp_eq_u32 vcc, mode, v2
        body.extend(vop3(0x101, 5, vgpr(5), vgpr(6), vcc)); // v_cndmask_b32 v5, v5, v6, vcc
    }
    // No rotation: red gains the p-bit. v7 = the row with it.
    body.extend([vop1(0x001, 7, 255), 0b1100_1011]); // v_mov_b32 v7, modes 0, 1, 3, 6, 7
    body.extend(vop3(0x116, 7, vgpr(2), vgpr(7), 128)); // v_lshrrev_b32 v7, v2, v7
    body.extend(vop3(0x148, 7, vgpr(7), 128, 129)); // v_bfe_u32 v7, v7, 0, 1
    body.extend(vop3(0x125, 7, vgpr(5), vgpr(7), 128)); // v_add_nc_u32 v7, v5, v7
    // A rotation swaps alpha with channel `rot - 1`: v8 = that channel's shift, v9 = the
    // two bytes' difference, v10 = the row with them exchanged.
    body.extend(vop3(0x125, 8, 193, vgpr(3), 128)); // v_add_nc_u32 v8, -1, v3
    body.extend(vop3(0x11A, 8, 128 + 3, vgpr(8), 128)); // v_lshlrev_b32 v8, 3, v8
    body.extend(vop3(0x116, 9, vgpr(8), vgpr(5), 128)); // v_lshrrev_b32 v9, v8, v5
    body.extend(vop3(0x148, 9, vgpr(9), 128, 128 + 8)); // v_bfe_u32 v9, v9, 0, 8
    body.extend(vop3(0x148, 10, vgpr(5), 128 + 24, 128 + 8)); // v_bfe_u32 v10, v5, 24, 8
    body.extend(vop3(0x11D, 9, vgpr(9), vgpr(10), 128)); // v_xor_b32 v9, v9, v10
    body.extend(vop3(0x11A, 10, vgpr(8), vgpr(9), 128)); // v_lshlrev_b32 v10, v8, v9
    body.extend(vop3(0x11D, 10, vgpr(5), vgpr(10), 128)); // v_xor_b32 v10, v5, v10
    body.extend(vop3(0x11A, 9, 128 + 24, vgpr(9), 128)); // v_lshlrev_b32 v9, 24, v9
    body.extend(vop3(0x11D, 10, vgpr(10), vgpr(9), 128)); // v_xor_b32 v10, v10, v9
    body.extend(vop3(0x0C5, 106, 128, vgpr(3), 128)); // v_cmp_ne_u32 vcc, 0, v3
    body.extend(vop3(0x101, 0, vgpr(7), vgpr(10), vcc)); // v_cndmask_b32 v0, v7, v10, vcc
    body
}

/// BC7's channel precisions for a control word, written as console code, match the
/// formats crate's function on the GPU for every mode and rotation.
#[test]
fn bcpack_bc7_precisions_match_the_formats_function() {
    let shader = build_valu_cs(&bc7_precision_body());
    // Every mode and rotation, alone and under each of the control word's higher bits.
    let cases: Vec<[u32; 4]> = (0u32..32)
        .flat_map(|low| [0u32, 0x20, 0x8000, 0xFFE0].map(|high| [high | low, 0, 0, 0]))
        .collect();
    run_cases(&shader, &cases, |backend, _, inputs, got| {
        let want = u32::from_le_bytes(bc7_precision_reference(inputs[0] as u16));
        assert_eq!(got, want, "{backend}: bc7_precision({:#x})", inputs[0]);
    });
}

/// `victory_formats::bcpack::bc6_index_bits`, the function the test below runs as console
/// code.
fn bc6_index_bits_reference(mode: u8) -> u8 {
    match mode {
        1..=10 => 3,
        11..=14 => 4,
        _ => 0,
    }
}

/// BC6H's index width for a mode, written as console code, matches the formats crate's
/// function on the GPU for every byte.
#[test]
fn bcpack_bc6_index_width_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the mode. Two range tests as unsigned compares of the mode less the range's start.
    body.extend(vop3(0x125, 5, 193, vgpr(1), 128)); // v_add_nc_u32 v5, -1, v1
    body.extend(vop3(0x0C4, 106, 128 + 10, vgpr(5), 128)); // v_cmp_gt_u32 vcc, 10, v5
    body.extend(vop3(0x101, 6, 128, 128 + 3, vcc)); // v_cndmask_b32 v6, 0, 3, vcc
    body.extend(vop3(0x126, 5, vgpr(1), 128 + 11, 128)); // v_sub_nc_u32 v5, v1, 11
    body.extend(vop3(0x0C4, 106, 128 + 4, vgpr(5), 128)); // v_cmp_gt_u32 vcc, 4, v5
    body.extend(vop3(0x101, 7, 128, 128 + 4, vcc)); // v_cndmask_b32 v7, 0, 4, vcc
    body.extend(vop3(0x11C, 0, vgpr(6), vgpr(7), 128)); // v_or_b32 v0, v6, v7
    let shader = build_valu_cs(&body);
    let cases: Vec<[u32; 4]> = (0u32..256).map(|mode| [mode, 0, 0, 0]).collect();
    run_cases(&shader, &cases, |backend, _, inputs, got| {
        let want = u32::from(bc6_index_bits_reference(inputs[0] as u8));
        assert_eq!(got, want, "{backend}: bc6_index_bits({})", inputs[0]);
    });
}

/// `victory_formats::bcpack::pairs`, the function the test below runs as console code.
fn pairs_reference(rank: &[u8; 64], s: &[u8], ix: &mut [u8]) {
    for (k, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym & 63)];
        ix[2 * k] = v & 7;
        ix[2 * k + 1] = v >> 4 & 7;
    }
}

/// BCPack's 3-bit index residuals, one symbol per two pixels, written as console code,
/// match the formats crate's function on the GPU: a rank table that tells the two pixels
/// apart, read through every symbol, the symbols' two unused bits included.
#[test]
fn bcpack_pixel_pair_residuals_match_the_formats_function() {
    // No entry gives its two pixels the same residual, and the pair is never symmetric.
    // Alternate the unused bits so extracting four bits instead of three is observable.
    let rank: [u8; 64] = std::array::from_fn(|i| {
        let unused = [0, 0x08, 0x80, 0x88][i / 16];
        ((i % 8) | (((i * 3 + 1) % 8) << 4) | unused) as u8
    });
    assert!(rank.iter().all(|v| v & 7 != v >> 4 & 7));
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the symbol; v2 = its table index, v3 = the index's dword, v5 = that dword.
    body.extend(vop3(0x148, 2, vgpr(1), 128, 128 + 6)); // v_bfe_u32 v2, v1, 0, 6
    body.extend(vop3(0x116, 3, 128 + 2, vgpr(2), 128)); // v_lshrrev_b32 v3, 2, v2
    for (d, row) in rank.as_chunks::<4>().0.iter().enumerate() {
        let reg = if d == 0 { 5 } else { 6 };
        body.extend([vop1(0x001, reg, 255), u32::from_le_bytes(*row)]); // v_mov_b32 reg, row
        if d != 0 {
            body.extend(vop3(0x0C2, 106, 128 + d as u16, vgpr(3), 128)); // v_cmp_eq_u32 vcc, d, v3
            body.extend(vop3(0x101, 5, vgpr(5), vgpr(6), vcc)); // v_cndmask_b32 v5, v5, v6, vcc
        }
    }
    // The entry's byte, then its two residuals, the second pixel's in the next byte.
    body.extend(vop3(0x148, 7, vgpr(2), 128, 128 + 2)); // v_bfe_u32 v7, v2, 0, 2
    body.extend(vop3(0x11A, 7, 128 + 3, vgpr(7), 128)); // v_lshlrev_b32 v7, 3, v7
    body.extend(vop3(0x116, 8, vgpr(7), vgpr(5), 128)); // v_lshrrev_b32 v8, v7, v5
    body.extend(vop3(0x148, 9, vgpr(8), 128, 128 + 3)); // v_bfe_u32 v9, v8, 0, 3
    body.extend(vop3(0x148, 10, vgpr(8), 128 + 4, 128 + 3)); // v_bfe_u32 v10, v8, 4, 3
    body.extend(vop3(0x36F, 0, vgpr(10), 128 + 8, vgpr(9))); // v_lshl_or_b32 v0, v10, 8, v9
    let shader = build_valu_cs(&body);
    let symbols: Vec<u8> = (0..=255).collect();
    let mut ix = vec![0u8; 2 * symbols.len()];
    pairs_reference(&rank, &symbols, &mut ix);
    let cases: Vec<[u32; 4]> = symbols.iter().map(|&s| [u32::from(s), 0, 0, 0]).collect();
    run_cases(&shader, &cases, |backend, k, inputs, got| {
        let want = u32::from(ix[2 * k]) | u32::from(ix[2 * k + 1]) << 8;
        assert_eq!(got, want, "{backend}: pairs, symbol {}", inputs[0]);
    });
}

/// `victory_formats::bcpack::quads`, the function the test below runs as console code.
fn quads_reference(rank: &[u8; 256], s: &[u8], ix: &mut [u8]) {
    for (q, &sym) in s.iter().enumerate() {
        let v = rank[usize::from(sym)];
        let base = (q >> 1) * 8 + (q & 1) * 2;
        for (k, p) in [0, 1, 4, 5].into_iter().enumerate() {
            ix[base + p] = v >> (2 * k) & 3;
        }
    }
}

/// BCPack's 2-bit index residuals, one symbol per 2x2 quad, written as console code, match
/// the formats crate's function on the GPU: each quad's four residuals and where the quad
/// starts in the block row, for a rank table and a symbol order that are both permutations.
#[test]
fn bcpack_quad_residuals_match_the_formats_function() {
    let rank: [u8; 256] = std::array::from_fn(|i| (i * 167 + 13) as u8);
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the symbol, v2 = the quad's number; v3 = the symbol's dword in the table, v5 =
    // that dword.
    body.extend(vop3(0x148, 1, vgpr(1), 128, 128 + 8)); // v_bfe_u32 v1, v1, 0, 8
    body.extend(vop3(0x116, 3, 128 + 2, vgpr(1), 128)); // v_lshrrev_b32 v3, 2, v1
    for (d, row) in rank.as_chunks::<4>().0.iter().enumerate() {
        let reg = if d == 0 { 5 } else { 6 };
        body.extend([vop1(0x001, reg, 255), u32::from_le_bytes(*row)]); // v_mov_b32 reg, row
        if d != 0 {
            body.extend(vop3(0x0C2, 106, 128 + d as u16, vgpr(3), 128)); // v_cmp_eq_u32 vcc, d, v3
            body.extend(vop3(0x101, 5, vgpr(5), vgpr(6), vcc)); // v_cndmask_b32 v5, v5, v6, vcc
        }
    }
    // The entry's byte, then its four residuals, one to a nibble of the result.
    body.extend(vop3(0x148, 7, vgpr(1), 128, 128 + 2)); // v_bfe_u32 v7, v1, 0, 2
    body.extend(vop3(0x11A, 7, 128 + 3, vgpr(7), 128)); // v_lshlrev_b32 v7, 3, v7
    body.extend(vop3(0x116, 8, vgpr(7), vgpr(5), 128)); // v_lshrrev_b32 v8, v7, v5
    body.extend(vop3(0x148, 0, vgpr(8), 128, 128 + 2)); // v_bfe_u32 v0, v8, 0, 2
    for k in 1u16..4 {
        body.extend(vop3(0x148, 9, vgpr(8), 128 + 2 * k, 128 + 2)); // v_bfe_u32 v9, v8, 2k, 2
        body.extend(vop3(0x36F, 0, vgpr(9), 128 + 4 * k, vgpr(0))); // v_lshl_or_b32 v0, v9, 4k, v0
    }
    // Where the quad starts: eight pixels per pair of quads, two for the odd one. It goes
    // in the result's high half.
    body.extend(vop3(0x116, 10, 129, vgpr(2), 128)); // v_lshrrev_b32 v10, 1, v2
    body.extend(vop3(0x148, 11, vgpr(2), 128, 129)); // v_bfe_u32 v11, v2, 0, 1
    body.extend(vop3(0x11A, 11, 129, vgpr(11), 128)); // v_lshlrev_b32 v11, 1, v11
    body.extend(vop3(0x36F, 10, vgpr(10), 128 + 3, vgpr(11))); // v_lshl_or_b32 v10, v10, 3, v11
    body.extend(vop3(0x36F, 0, vgpr(10), 128 + 16, vgpr(0))); // v_lshl_or_b32 v0, v10, 16, v0
    let shader = build_valu_cs(&body);
    let symbols: Vec<u8> = (0..256).map(|k| (k * 91 + 7) as u8).collect();
    let mut ix = vec![0xFFu8; 4 * symbols.len()];
    quads_reference(&rank, &symbols, &mut ix);
    let cases: Vec<[u32; 4]> = symbols
        .iter()
        .enumerate()
        .map(|(q, &s)| [u32::from(s), q as u32, 0, 0])
        .collect();
    // Each backend must read every cell the function wrote, each once.
    let mut read = std::collections::BTreeMap::new();
    run_cases(&shader, &cases, |backend, q, _, got| {
        let base = (got >> 16) as usize;
        let seen: &mut Vec<bool> = read
            .entry(backend.to_string())
            .or_insert(vec![false; ix.len()]);
        for (k, p) in [0, 1, 4, 5].into_iter().enumerate() {
            let cell = base + p;
            assert!(
                cell < ix.len() && !seen[cell],
                "{backend}: quad {q} lands on cell {cell}"
            );
            seen[cell] = true;
            assert_eq!(
                got >> (4 * k) & 15,
                u32::from(ix[cell]),
                "{backend}: quad {q}, pixel {k}"
            );
        }
    });
    for (backend, seen) in read {
        assert!(seen.iter().all(|&s| s), "{backend}: cells never read");
    }
}

/// `victory_formats::bcpack::pick` with its history length, the function the test below
/// runs as console code.
fn pick_reference<'a, T>(
    i: usize,
    r: usize,
    history: &'a [T],
    dictionary: &'a [T],
    zero: &'a T,
) -> &'a T {
    const HISTORY: usize = 256;
    match r {
        0 => zero,
        r if r <= i => &history[(i - r) % HISTORY],
        r => &dictionary[r - i - 1],
    }
}

/// BCPack's choice of a block's reference, written as console code, matches the formats
/// crate's function on the GPU: none, a block of the history ring, or a dictionary entry
/// before the first block, on both sides of each boundary.
#[test]
fn bcpack_reference_choice_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the block, v2 = its reference. The result names the source in its high half
    // (1 the history, 2 the dictionary) and the entry in its low half.
    body.extend(vop3(0x126, 5, vgpr(1), vgpr(2), 128)); // v_sub_nc_u32 v5, v1, v2
    body.extend(vop3(0x148, 5, vgpr(5), 128, 128 + 8)); // v_bfe_u32 v5, v5, 0, 8
    body.extend(vop3(0x36F, 5, 129, 128 + 16, vgpr(5))); // v_lshl_or_b32 v5, 1, 16, v5
    body.extend(vop3(0x126, 6, vgpr(2), vgpr(1), 128)); // v_sub_nc_u32 v6, v2, v1
    body.extend(vop3(0x125, 6, 193, vgpr(6), 128)); // v_add_nc_u32 v6, -1, v6
    body.extend(vop3(0x36F, 6, 130, 128 + 16, vgpr(6))); // v_lshl_or_b32 v6, 2, 16, v6
    body.extend(vop3(0x0C3, 106, vgpr(2), vgpr(1), 128)); // v_cmp_le_u32 vcc, v2, v1
    body.extend(vop3(0x101, 7, vgpr(6), vgpr(5), vcc)); // v_cndmask_b32 v7, v6, v5, vcc
    body.extend(vop3(0x0C2, 106, 128, vgpr(2), 128)); // v_cmp_eq_u32 vcc, 0, v2
    body.extend(vop3(0x101, 0, vgpr(7), 128, vcc)); // v_cndmask_b32 v0, v7, 0, vcc
    let shader = build_valu_cs(&body);
    // Entries that say where they are, so the function's choice reads as the result does.
    let history: Vec<u32> = (0..256).map(|k| 1 << 16 | k).collect();
    let dictionary: Vec<u32> = (0..255).map(|k| 2 << 16 | k).collect();
    // Blocks and references on both sides of `r == 0`, `r == i` and the ring's length.
    let cases: Vec<[u32; 4]> = [0u32, 1, 2, 254, 255, 256, 1000]
        .into_iter()
        .flat_map(|i| [0u32, 1, 2, 3, 128, 254, 255].map(|r| [i, r, 0, 0]))
        .collect();
    run_cases(&shader, &cases, |backend, _, [i, r, ..], got| {
        let want = *pick_reference(i as usize, r as usize, &history, &dictionary, &0);
        assert_eq!(got, want, "{backend}: pick(block {i}, reference {r})");
    });
}

/// `victory_formats::xma::log2`, the function the test below runs as console code.
fn xma_log2_reference(v: usize) -> u32 {
    usize::BITS - 1 - v.max(1).leading_zeros()
}

/// XMA's integer base-2 logarithm, written as console code, matches the formats crate's
/// function on the GPU: zero, every power of two and both of its neighbours.
#[test]
fn xma_integer_log2_matches_the_formats_function() {
    let mut body = Vec::new();
    // v1 = the value; zero counts as one.
    body.extend(vop3(0x114, 5, vgpr(1), 129, 128)); // v_max_u32 v5, v1, 1
    body.push(vop1(0x039, 6, vgpr(5))); // v_ffbh_u32 v6, v5
    body.extend(vop3(0x126, 0, 128 + 31, vgpr(6), 128)); // v_sub_nc_u32 v0, 31, v6
    let shader = build_valu_cs(&body);
    let mut values = vec![0u32, u32::MAX];
    for bit in 0..32 {
        let p = 1u32 << bit;
        values.extend([p.wrapping_sub(1), p, p.wrapping_add(1)]);
    }
    let cases: Vec<[u32; 4]> = values.into_iter().map(|v| [v, 0, 0, 0]).collect();
    run_cases(&shader, &cases, |backend, _, inputs, got| {
        let want = xma_log2_reference(inputs[0] as usize);
        assert_eq!(got, want, "{backend}: log2({:#x})", inputs[0]);
    });
}

/// `victory_formats::xma::to_i16`, the function the test below runs as console code.
fn xma_to_i16_reference(v: f32) -> i16 {
    (v * 32768.0).round().clamp(-32768.0, 32767.0) as i16
}

/// XMA's sample conversion from float to 16 bits, written as console code, matches the
/// formats crate's function on the GPU: ties round away from zero, both ends saturate.
#[test]
fn xma_sample_to_i16_matches_the_formats_function() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = the sample. v5 = it scaled, v6 = that truncated, v7 = the fraction cut off.
    body.extend([vop1(0x001, 5, 255), 32768f32.to_bits()]); // v_mov_b32 v5, 32768.0
    body.extend(vop3(0x108, 5, vgpr(1), vgpr(5), 128)); // v_mul_f32 v5, v1, v5
    body.push(vop1(0x021, 6, vgpr(5))); // v_trunc_f32 v6, v5
    body.extend(vop3(0x104, 7, vgpr(5), vgpr(6), 128)); // v_sub_f32 v7, v5, v6
    // A fraction of a half or more moves one further from zero.
    body.extend(vop3(0x103, 8, vgpr(6), 242, 128)); // v_add_f32 v8, v6, 1.0
    body.extend(vop3(0x006, 106, vgpr(7), 240, 128)); // v_cmp_ge_f32 vcc, v7, 0.5
    body.extend(vop3(0x101, 9, vgpr(6), vgpr(8), vcc)); // v_cndmask_b32 v9, v6, v8, vcc
    body.extend(vop3(0x103, 8, vgpr(6), 243, 128)); // v_add_f32 v8, v6, -1.0
    body.extend(vop3(0x003, 106, vgpr(7), 241, 128)); // v_cmp_le_f32 vcc, v7, -0.5
    body.extend(vop3(0x101, 9, vgpr(9), vgpr(8), vcc)); // v_cndmask_b32 v9, v9, v8, vcc
    // Saturate, then convert.
    body.extend([vop1(0x001, 10, 255), (-32768f32).to_bits()]); // v_mov_b32 v10, -32768.0
    body.extend(vop3(0x110, 9, vgpr(9), vgpr(10), 128)); // v_max_f32 v9, v9, v10
    body.extend([vop1(0x001, 10, 255), 32767f32.to_bits()]); // v_mov_b32 v10, 32767.0
    body.extend(vop3(0x10F, 9, vgpr(9), vgpr(10), 128)); // v_min_f32 v9, v9, v10
    body.push(vop1(0x008, 0, vgpr(9))); // v_cvt_i32_f32 v0, v9
    // Not a number converts to zero: `v_max_f32` alone would have saturated it.
    body.extend(vop3(0x008, 106, vgpr(1), vgpr(1), 128)); // v_cmp_u_f32 vcc, v1, v1
    body.extend(vop3(0x101, 0, vgpr(0), 128, vcc)); // v_cndmask_b32 v0, v0, 0, vcc
    let shader = build_valu_cs(&body);
    let step = 1.0 / 32768.0;
    let mut samples = vec![
        0.0f32,
        -0.0,
        1.0,
        -1.0,
        2.0,
        -2.0,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    // Ties and their neighbours on both sides of zero, and just inside each end.
    for k in [0.5f32, 1.5, 2.5, 3.5, 32766.5, 32767.5] {
        for x in [
            k,
            f32::from_bits(k.to_bits() - 1),
            f32::from_bits(k.to_bits() + 1),
        ] {
            samples.extend([x * step, -x * step]);
        }
    }
    samples.extend((0..64).map(|i| (i as f32 * 0.618_034).sin() * 1.1));
    let cases: Vec<[u32; 4]> = samples.iter().map(|s| [s.to_bits(), 0, 0, 0]).collect();
    let mut wrong = Vec::new();
    run_cases(&shader, &cases, |backend, _, inputs, got| {
        let v = f32::from_bits(inputs[0]);
        let want = xma_to_i16_reference(v) as i32 as u32;
        if got != want {
            wrong.push(format!(
                "{backend}: to_i16({v:e}) = {} want {}",
                got as i32, want as i32
            ));
        }
    });
    assert!(wrong.is_empty(), "{wrong:#?}");
}

/// `victory_formats::bcpack::interleave`, the function the test below runs as console code.
fn interleave_reference(a: u8, b: u8) -> u8 {
    let spread = |v: u8| (0..4).fold(0, |acc, k| acc | (v >> k & 1) << (2 * k));
    spread(a) | spread(b) << 1
}

/// BCPack's interleave of two 4-bit symbols, written as console code, matches the formats
/// crate's function on the GPU: every pair of symbols, with and without bits above the
/// nibble that the function ignores.
#[test]
fn bcpack_symbol_interleave_matches_the_formats_function() {
    // v1 = the first symbol, on the even bits; v2 = the second, on the odd ones.
    let mut body = vec![vop1(0x001, 0, 128)]; // v_mov_b32 v0, 0
    for bit in 0u16..4 {
        for (symbol, position) in [(1, 2 * bit), (2, 2 * bit + 1)] {
            body.extend(vop3(0x148, 3, vgpr(symbol), 128 + bit, 129)); // v_bfe_u32 v3, symbol, bit, 1
            body.extend(vop3(0x36F, 0, vgpr(3), 128 + position, vgpr(0))); // v_lshl_or_b32 v0, v3, position, v0
        }
    }
    let shader = build_valu_cs(&body);
    let mut cases: Vec<[u32; 4]> = (0u32..256)
        .map(|pair| [pair & 15, pair >> 4, 0, 0])
        .collect();
    cases.extend((0u32..16).map(|a| [a | 0xF0, (15 - a) | 0xA0, 0, 0]));
    run_cases(&shader, &cases, |backend, _, [a, b, ..], got| {
        let want = u32::from(interleave_reference(a as u8, b as u8));
        assert_eq!(got, want, "{backend}: interleave({a:#x}, {b:#x})");
    });
}

/// `victory_formats::texture::dim`, one of the two functions the test below runs as
/// console code.
fn texture_dim_reference(texels: u32, mip: u32, block: u32) -> u32 {
    (texels >> mip).max(1).div_ceil(block)
}

/// `victory_formats::texture::padded_dim`, the other one.
fn texture_padded_dim_reference(texels: u32, mip: u32, block: u32) -> u32 {
    texels.max(1).div_ceil(block).div_ceil(1 << mip)
}

/// A mip's size in elements, as data sees it and as placement pads it, written as console
/// code, matches the formats crate's two functions on the GPU for texel and compressed
/// block sizes. The block is a power of two, so each division is a shift.
#[test]
fn texture_mip_sizes_match_the_formats_functions() {
    let vcc = sgpr(106);
    let mut body = Vec::new();
    // v1 = texels, v2 = the mip, v3 = the block, v4 = which size. v5 = the block's shift,
    // v6 = the block less one.
    body.push(vop1(0x03A, 5, vgpr(3))); // v_ffbl_b32 v5, v3
    body.extend(vop3(0x125, 6, 193, vgpr(3), 128)); // v_add_nc_u32 v6, -1, v3
    // As data sees it: halve rounding down, at least one, round up to whole elements.
    body.extend(vop3(0x116, 7, vgpr(2), vgpr(1), 128)); // v_lshrrev_b32 v7, v2, v1
    body.extend(vop3(0x114, 7, vgpr(7), 129, 128)); // v_max_u32 v7, v7, 1
    body.extend(vop3(0x125, 7, vgpr(7), vgpr(6), 128)); // v_add_nc_u32 v7, v7, v6
    body.extend(vop3(0x116, 7, vgpr(5), vgpr(7), 128)); // v_lshrrev_b32 v7, v5, v7
    // As placement pads it: mip 0's elements, halved rounding up.
    body.extend(vop3(0x114, 8, vgpr(1), 129, 128)); // v_max_u32 v8, v1, 1
    body.extend(vop3(0x125, 8, vgpr(8), vgpr(6), 128)); // v_add_nc_u32 v8, v8, v6
    body.extend(vop3(0x116, 8, vgpr(5), vgpr(8), 128)); // v_lshrrev_b32 v8, v5, v8
    body.extend(vop3(0x11A, 9, vgpr(2), 129, 128)); // v_lshlrev_b32 v9, v2, 1
    body.extend(vop3(0x125, 9, 193, vgpr(9), 128)); // v_add_nc_u32 v9, -1, v9
    body.extend(vop3(0x125, 8, vgpr(8), vgpr(9), 128)); // v_add_nc_u32 v8, v8, v9
    body.extend(vop3(0x116, 8, vgpr(2), vgpr(8), 128)); // v_lshrrev_b32 v8, v2, v8
    body.extend(vop3(0x0C5, 106, 128, vgpr(4), 128)); // v_cmp_ne_u32 vcc, 0, v4
    body.extend(vop3(0x101, 0, vgpr(7), vgpr(8), vcc)); // v_cndmask_b32 v0, v7, v8, vcc
    let shader = build_valu_cs(&body);
    let mut cases = Vec::new();
    for texels in [0u32, 1, 2, 3, 5, 8, 9, 100, 255, 257, 4096, 16383] {
        for mip in [0u32, 1, 2, 3, 5, 13] {
            for block in [1u32, 4] {
                cases.extend([[texels, mip, block, 0], [texels, mip, block, 1]]);
            }
        }
    }
    // The two sizes must differ somewhere, or the test could not tell them apart.
    assert!(cases.iter().any(|&[t, m, b, _]| {
        texture_dim_reference(t, m, b) != texture_padded_dim_reference(t, m, b)
    }));
    run_cases(
        &shader,
        &cases,
        |backend, _, [texels, mip, block, padded], got| {
            let want = if padded == 0 {
                texture_dim_reference(texels, mip, block)
            } else {
                texture_padded_dim_reference(texels, mip, block)
            };
            assert_eq!(
                got, want,
                "{backend}: {texels} texels, mip {mip}, block {block}, padded {padded}"
            );
        },
    );
}
