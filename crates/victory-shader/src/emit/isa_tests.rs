//! Vector compares and carry arithmetic against the RDNA2 instruction set
//! document's definitions: every float condition on edge values (NaNs,
//! infinities, signed zeros), the signed and unsigned integer conditions,
//! `v_cmp_class_f32`'s ten classes, and the add and subtract forms that
//! produce or take a carry. Each runs as a one-thread compute shader (the
//! instruction, then `v_cndmask_b32` where the result is a mask bit) on the
//! hardware adapter and WARP, over all pairs of the edge values.
//!
//! `v_cmp_f_f32` and `v_cmp_tru_f32` (always false, always true) are left
//! out: lowered as a real compare of their operands they have been seen to
//! remove the WARP device, which fails every later GPU test of the process.

use std::cmp::Ordering::{Equal, Greater, Less};

use super::HostShader;
use super::valu_tests::{build_valu_cs, run_cases, sgpr, vgpr, vop3};

/// Operands that sit on the edges of the float conditions.
const FLOATS: &[u32] = &[
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x3f80_0000, // 1
    0xbf80_0000, // -1
    0x3f00_0000, // 0.5
    0x4000_0000, // 2
    0x501c_f4a1, // 1e10
    0xd01c_f4a1, // -1e10
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // quiet NaN
    0xffc0_0001, // negative quiet NaN
    0x7f80_0001, // signalling NaN
    0x7f7f_ffff, // largest finite
    0xff7f_ffff, // most negative finite
    0x0080_0000, // smallest normal
];

/// Integers on the edges of the signed and unsigned conditions.
const INTEGERS: &[u32] = &[
    0,
    1,
    2,
    0x7fff_fffe,
    0x7fff_ffff,
    0x8000_0000,
    0x8000_0001,
    0xffff_fffe,
    0xffff_ffff,
];

/// `v_cmp_<op> s6, v1, v2` then `v_cndmask_b32 v0, 0, 1, s6`.
fn compare(opcode: u16) -> HostShader {
    let [c0, c1] = vop3(opcode, 6, vgpr(1), vgpr(2), 0);
    let [m0, m1] = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
    build_valu_cs(&[c0, c1, m0, m1])
}

fn pairs(values: &[u32]) -> Vec<[u32; 4]> {
    values
        .iter()
        .flat_map(|&a| values.iter().map(move |&b| [a, b, 0, 0]))
        .collect()
}

#[test]
fn float_compares_match_the_rdna2_conditions() {
    // Ordered conditions are false when either operand is a NaN, their
    // negations ("n...") true; `lg` is "less or greater".
    #[allow(clippy::type_complexity)]
    let conditions: &[(&str, u16, fn(f32, f32) -> bool)] = &[
        ("lt", 0x01, |a, b| a.partial_cmp(&b) == Some(Less)),
        ("eq", 0x02, |a, b| a.partial_cmp(&b) == Some(Equal)),
        ("le", 0x03, |a, b| {
            matches!(a.partial_cmp(&b), Some(Less | Equal))
        }),
        ("gt", 0x04, |a, b| a.partial_cmp(&b) == Some(Greater)),
        ("lg", 0x05, |a, b| {
            matches!(a.partial_cmp(&b), Some(Less | Greater))
        }),
        ("ge", 0x06, |a, b| {
            matches!(a.partial_cmp(&b), Some(Greater | Equal))
        }),
        ("o", 0x07, |a, b| a.partial_cmp(&b).is_some()),
        ("u", 0x08, |a, b| a.partial_cmp(&b).is_none()),
        ("nge", 0x09, |a, b| {
            !matches!(a.partial_cmp(&b), Some(Greater | Equal))
        }),
        ("nlg", 0x0a, |a, b| {
            !matches!(a.partial_cmp(&b), Some(Less | Greater))
        }),
        ("ngt", 0x0b, |a, b| a.partial_cmp(&b) != Some(Greater)),
        ("nle", 0x0c, |a, b| {
            !matches!(a.partial_cmp(&b), Some(Less | Equal))
        }),
        ("neq", 0x0d, |a, b| a.partial_cmp(&b) != Some(Equal)),
        ("nlt", 0x0e, |a, b| a.partial_cmp(&b) != Some(Less)),
    ];
    let cases = pairs(FLOATS);
    let mut failures = Vec::new();
    for &(name, opcode, reference) in conditions {
        let shader = compare(opcode);
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = u32::from(reference(f32::from_bits(inp[0]), f32::from_bits(inp[1])));
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: v_cmp_{name}_f32({:#010x}, {:#010x}) = {got}, want {want}",
                    inp[0], inp[1]
                ));
            }
        });
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn integer_compares_match_the_rdna2_conditions() {
    #[allow(clippy::type_complexity)]
    let conditions: &[(&str, u16, fn(u32, u32) -> bool)] = &[
        ("lt_i32", 0x81, |a, b| (a as i32) < (b as i32)),
        ("eq_i32", 0x82, |a, b| a == b),
        ("le_i32", 0x83, |a, b| (a as i32) <= (b as i32)),
        ("gt_i32", 0x84, |a, b| (a as i32) > (b as i32)),
        ("ne_i32", 0x85, |a, b| a != b),
        ("ge_i32", 0x86, |a, b| (a as i32) >= (b as i32)),
        ("lt_u32", 0xc1, |a, b| a < b),
        ("eq_u32", 0xc2, |a, b| a == b),
        ("le_u32", 0xc3, |a, b| a <= b),
        ("gt_u32", 0xc4, |a, b| a > b),
        ("ne_u32", 0xc5, |a, b| a != b),
        ("ge_u32", 0xc6, |a, b| a >= b),
    ];
    let cases = pairs(INTEGERS);
    let mut failures = Vec::new();
    for &(name, opcode, reference) in conditions {
        let shader = compare(opcode);
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = u32::from(reference(inp[0], inp[1]));
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: v_cmp_{name}({:#x}, {:#x}) = {got}, want {want}",
                    inp[0], inp[1]
                ));
            }
        });
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn class_f32_tests_the_ten_classes() {
    // Bit k of the mask asks for class k: 0 signalling NaN, 1 quiet NaN,
    // 2 -inf, 3 -normal, 4 -denormal, 5 -0, 6 +0, 7 +denormal, 8 +normal,
    // 9 +inf. The sign of a NaN does not matter.
    fn class(bits: u32) -> u32 {
        let (negative, exponent, fraction) = (bits >> 31 != 0, bits >> 23 & 0xff, bits & 0x7f_ffff);
        match (exponent, fraction, negative) {
            (0xff, f, _) if f != 0 => u32::from(f >> 22 != 0),
            (0xff, _, true) => 2,
            (0xff, _, false) => 9,
            (0, 0, true) => 5,
            (0, 0, false) => 6,
            (0, _, true) => 4,
            (0, _, false) => 7,
            (_, _, true) => 3,
            (_, _, false) => 8,
        }
    }
    let operands = [
        0x7f80_0001u32,
        0xff80_0001,
        0x7fc0_0000,
        0xffc0_0001,
        0xff80_0000,
        0xbf80_0000,
        0x8000_0001,
        0x807f_ffff,
        0x8000_0000,
        0x0000_0000,
        0x0000_0001,
        0x007f_ffff,
        0x3f80_0000,
        0x7f7f_ffff,
        0x7f80_0000,
    ];
    let masks: Vec<u32> = (0..10)
        .map(|k| 1 << k)
        .chain([0, 0x3ff, 0x60, 0x201, 0x3])
        .collect();
    let cases: Vec<[u32; 4]> = operands
        .iter()
        .flat_map(|&a| masks.iter().map(move |&m| [a, m, 0, 0]))
        .collect();
    let shader = compare(0x88);
    let mut failures = Vec::new();
    run_cases(&shader, &cases, |backend, _, inp, got| {
        let want = inp[1] >> class(inp[0]) & 1;
        if got != want && failures.len() < 40 {
            failures.push(format!(
                "{backend}: v_cmp_class_f32({:#010x}, {:#x}) = {got}, want {want}",
                inp[0], inp[1]
            ));
        }
    });
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// VOP3B `op vdst, sdst, src0, src1, src2` (the carry-out forms): the scalar
/// destination sits in bits 14:8 of the first dword.
fn vop3b(op: u16, vdst: u8, sdst: u8, src0: u16, src1: u16, src2: u16) -> [u32; 2] {
    let [w0, w1] = vop3(op, vdst, src0, src1, src2);
    [w0 | u32::from(sdst) << 8, w1]
}

#[test]
fn carry_arithmetic_matches_the_rdna2_definitions() {
    // (name, VOP3 opcode, takes a carry in, reference: the result and the
    // carry or borrow out from a, b and the carry in).
    #[allow(clippy::type_complexity)]
    let operations: &[(&str, u16, bool, fn(u32, u32, u32) -> (u32, bool))] = &[
        ("v_add_co_u32", 0x30f, false, |a, b, _| a.overflowing_add(b)),
        ("v_sub_co_u32", 0x310, false, |a, b, _| a.overflowing_sub(b)),
        ("v_subrev_co_u32", 0x319, false, |a, b, _| {
            b.overflowing_sub(a)
        }),
        ("v_add_co_ci_u32", 0x128, true, |a, b, c| {
            let sum = u64::from(a) + u64::from(b) + u64::from(c);
            (sum as u32, sum >> 32 != 0)
        }),
        ("v_sub_co_ci_u32", 0x129, true, |a, b, c| {
            (
                a.wrapping_sub(b).wrapping_sub(c),
                u64::from(a) < u64::from(b) + u64::from(c),
            )
        }),
        ("v_subrev_co_ci_u32", 0x12a, true, |a, b, c| {
            (
                b.wrapping_sub(a).wrapping_sub(c),
                u64::from(b) < u64::from(a) + u64::from(c),
            )
        }),
    ];
    let mut cases = Vec::new();
    for &a in INTEGERS {
        for &b in INTEGERS {
            for carry in [0, 1] {
                cases.push([a, b, carry, 0]);
            }
        }
    }
    let mut failures = Vec::new();
    for &(name, opcode, takes_carry, reference) in operations {
        // The carry in is a mask the translator tracks: v_cmp_ne_u32 s7,
        // v3, 0 sets lane 0's bit when the third input is not zero.
        let src2 = if takes_carry { sgpr(7) } else { 0 };
        let [c0, c1] = vop3(0xc5, 7, vgpr(3), 0x80, 0);
        let [i0, i1] = vop3b(opcode, 0, 6, vgpr(1), vgpr(2), src2);
        let [m0, m1] = vop3(0x101, 0, 0x80, 0x81, sgpr(6));
        let result = build_valu_cs(&[c0, c1, i0, i1]);
        let carry = build_valu_cs(&[c0, c1, i0, i1, m0, m1]);
        run_cases(&result, &cases, |backend, _, inp, got| {
            let (want, _) = reference(inp[0], inp[1], inp[2]);
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#x}, {:#x}, carry {}) = {got:#x}, want {want:#x}",
                    inp[0], inp[1], inp[2]
                ));
            }
        });
        run_cases(&carry, &cases, |backend, _, inp, got| {
            let want = u32::from(reference(inp[0], inp[1], inp[2]).1);
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#x}, {:#x}, carry {}) carry out = {got}, want {want}",
                    inp[0], inp[1], inp[2]
                ));
            }
        });
    }
    assert!(
        failures.is_empty(),
        "{}",
        failures.join(
            "
"
        )
    );
}

/// Float operands for the conversions and rounding: halves and quarters
/// either side of the integers, the 32-bit integer limits, infinities and
/// NaNs.
const CONVERSION_FLOATS: &[u32] = &[
    0x0000_0000, // 0
    0x8000_0000, // -0
    0x3e80_0000, // 0.25
    0xbe80_0000, // -0.25
    0x3f00_0000, // 0.5
    0xbf00_0000, // -0.5
    0x3f80_0000, // 1
    0xbf80_0000, // -1
    0x3fc0_0000, // 1.5
    0xbfc0_0000, // -1.5
    0x4020_0000, // 2.5
    0xc020_0000, // -2.5
    0x40a0_0000, // 5
    0x42f6_e979, // 123.456
    0xc2f6_e979, // -123.456
    0x4b00_0001, // 8388609 (2^23 + 1)
    0x4effffff,  // 2147483520 (largest f32 below 2^31)
    0x4f00_0000, // 2^31
    0xcf00_0000, // -2^31
    0x4f80_0000, // 2^32
    0x4f7f_ffff, // just below 2^32
    0x7f7f_ffff, // largest finite
    0xff7f_ffff, // most negative finite
    0x7f80_0000, // +inf
    0xff80_0000, // -inf
    0x7fc0_0000, // quiet NaN
    0x7f80_0001, // signalling NaN
];

/// Dwords for the integer-to-float conversions.
const CONVERSION_INTEGERS: &[u32] = &[
    0,
    1,
    2,
    0xff,
    0x100,
    0x00ff_ffff,
    0x0100_0000,
    0x0100_0001,
    0x7fff_ffff,
    0x8000_0000,
    0x8000_0001,
    0xffff_ff00,
    0xffff_ffff,
    0xdead_beef,
    0x1234_5678,
];

/// Same bits, or both NaN (a NaN's payload is not defined by the operation).
fn same_float(a: u32, b: u32) -> bool {
    a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

#[test]
fn conversions_and_rounding_match_the_rdna2_definitions() {
    // (name, VOP3 opcode, reference of one float operand: bits out).
    #[allow(clippy::type_complexity)]
    let from_float: &[(&str, u16, fn(f32) -> u32)] = &[
        // Saturating, NaN to 0.
        ("v_cvt_u32_f32", 0x187, |x| x as u32),
        ("v_cvt_i32_f32", 0x188, |x| x as i32 as u32),
        ("v_cvt_flr_i32_f32", 0x18d, |x| x.floor() as i32 as u32),
        ("v_cvt_rpi_i32_f32", 0x18c, |x| {
            (x + 0.5).floor() as i32 as u32
        }),
        ("v_trunc_f32", 0x1a1, |x| x.trunc().to_bits()),
        ("v_ceil_f32", 0x1a2, |x| x.ceil().to_bits()),
        ("v_rndne_f32", 0x1a3, |x| x.round_ties_even().to_bits()),
        ("v_floor_f32", 0x1a4, |x| x.floor().to_bits()),
    ];
    let cases: Vec<[u32; 4]> = CONVERSION_FLOATS.iter().map(|&x| [x, 0, 0, 0]).collect();
    let mut failures = Vec::new();
    for &(name, opcode, reference) in from_float {
        let shader = build_valu_cs(&vop3(opcode, 0, vgpr(1), 0, 0));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = reference(f32::from_bits(inp[0]));
            let float_result = name.ends_with("_f32") && !name.starts_with("v_cvt");
            let same = if float_result {
                same_float(got, want)
            } else {
                got == want
            };
            if !same && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#010x}) = {got:#010x}, want {want:#010x}",
                    inp[0]
                ));
            }
        });
    }
    // Integer to float: rounded to nearest even.
    #[allow(clippy::type_complexity)]
    let to_float: &[(&str, u16, fn(u32) -> u32)] = &[
        ("v_cvt_f32_i32", 0x185, |x| (x as i32 as f32).to_bits()),
        ("v_cvt_f32_u32", 0x186, |x| (x as f32).to_bits()),
        ("v_cvt_f32_ubyte0", 0x191, |x| ((x & 0xff) as f32).to_bits()),
        ("v_cvt_f32_ubyte1", 0x192, |x| {
            ((x >> 8 & 0xff) as f32).to_bits()
        }),
        ("v_cvt_f32_ubyte2", 0x193, |x| {
            ((x >> 16 & 0xff) as f32).to_bits()
        }),
        ("v_cvt_f32_ubyte3", 0x194, |x| ((x >> 24) as f32).to_bits()),
    ];
    let cases: Vec<[u32; 4]> = CONVERSION_INTEGERS.iter().map(|&x| [x, 0, 0, 0]).collect();
    for &(name, opcode, reference) in to_float {
        let shader = build_valu_cs(&vop3(opcode, 0, vgpr(1), 0, 0));
        run_cases(&shader, &cases, |backend, _, inp, got| {
            let want = reference(inp[0]);
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#010x}) = {got:#010x}, want {want:#010x}",
                    inp[0]
                ));
            }
        });
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn float_arithmetic_with_nan_and_infinity_matches_the_rdna2_definitions() {
    // Minimum and maximum: a NaN operand gives the other one; two NaNs a NaN.
    // Zeros of both signs are left out (the order of -0 and +0 is not
    // defined the same way everywhere).
    let numbers = [
        0x3f80_0000u32,
        0xbf80_0000,
        0x4000_0000,
        0x7f80_0000,
        0xff80_0000,
        0x7fc0_0000,
        0x7f7f_ffff,
    ];
    let pairs: Vec<[u32; 4]> = numbers
        .iter()
        .flat_map(|&a| numbers.iter().map(move |&b| [a, b, 0, 0]))
        .collect();
    fn pick(a: f32, b: f32, first: fn(f32, f32) -> f32) -> f32 {
        if a.is_nan() {
            b
        } else if b.is_nan() {
            a
        } else {
            first(a, b)
        }
    }
    #[allow(clippy::type_complexity)]
    let two: &[(&str, u16, fn(f32, f32) -> f32)] = &[
        ("v_min_f32", 0x10f, |a, b| pick(a, b, f32::min)),
        ("v_max_f32", 0x110, |a, b| pick(a, b, f32::max)),
        ("v_add_f32", 0x103, |a, b| a + b),
        ("v_sub_f32", 0x104, |a, b| a - b),
        ("v_subrev_f32", 0x105, |a, b| b - a),
        ("v_mul_f32", 0x108, |a, b| a * b),
    ];
    let mut failures = Vec::new();
    for &(name, opcode, reference) in two {
        let shader = build_valu_cs(&vop3(opcode, 0, vgpr(1), vgpr(2), 0));
        run_cases(&shader, &pairs, |backend, _, inp, got| {
            let want = reference(f32::from_bits(inp[0]), f32::from_bits(inp[1])).to_bits();
            if !same_float(got, want) && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                    inp[0], inp[1]
                ));
            }
        });
    }
    assert!(
        failures.is_empty(),
        "{}",
        failures.join(
            "
"
        )
    );
}

/// VOP3P `op vdst, src0, src1, src2` with the source half selectors of the
/// instruction: bit k of `op_sel` takes the high half of source k for the
/// low result, bit k of `op_sel_hi` the high half of source k for the high
/// result (both 0 or both 1 for a source that is read whole).
fn vop3p(op: u16, vdst: u8, srcs: [u16; 3], op_sel: u8, op_sel_hi: u8) -> [u32; 2] {
    [
        0x198 << 23
            | u32::from(op) << 16
            | u32::from(op_sel_hi >> 2 & 1) << 14
            | u32::from(op_sel & 7) << 11
            | u32::from(vdst),
        u32::from(srcs[0])
            | u32::from(srcs[1]) << 9
            | u32::from(srcs[2]) << 18
            | u32::from(op_sel_hi & 3) << 27,
    ]
}

#[test]
fn packed_integer_operations_match_the_rdna2_definitions() {
    // (name, VOP3P opcode, sources used, reference on 16-bit halves a, b, c
    // as u16 (the signed forms reinterpret them)).
    #[allow(clippy::type_complexity)]
    let operations: &[(&str, u16, fn(u16, u16, u16) -> u16)] = &[
        ("v_pk_mad_i16", 0x00, |a, b, c| {
            (a as i16).wrapping_mul(b as i16).wrapping_add(c as i16) as u16
        }),
        ("v_pk_mul_lo_u16", 0x01, |a, b, _| a.wrapping_mul(b)),
        ("v_pk_add_i16", 0x02, |a, b, _| a.wrapping_add(b)),
        ("v_pk_sub_i16", 0x03, |a, b, _| a.wrapping_sub(b)),
        // The shift count is the first operand's low four bits.
        ("v_pk_lshlrev_b16", 0x04, |a, b, _| b << (a & 15)),
        ("v_pk_lshrrev_b16", 0x05, |a, b, _| b >> (a & 15)),
        ("v_pk_ashrrev_i16", 0x06, |a, b, _| {
            ((b as i16) >> (a & 15)) as u16
        }),
        ("v_pk_max_i16", 0x07, |a, b, _| {
            (a as i16).max(b as i16) as u16
        }),
        ("v_pk_min_i16", 0x08, |a, b, _| {
            (a as i16).min(b as i16) as u16
        }),
        ("v_pk_mad_u16", 0x09, |a, b, c| {
            a.wrapping_mul(b).wrapping_add(c)
        }),
        ("v_pk_add_u16", 0x0a, |a, b, _| a.wrapping_add(b)),
        ("v_pk_sub_u16", 0x0b, |a, b, _| a.wrapping_sub(b)),
        ("v_pk_max_u16", 0x0c, |a, b, _| a.max(b)),
        ("v_pk_min_u16", 0x0d, |a, b, _| a.min(b)),
    ];
    let halves = [
        0u32, 1, 2, 0xf, 0x10, 0x7fff, 0x8000, 0x8001, 0xfffe, 0xffff,
    ];
    let mut cases = Vec::new();
    for (i, &a) in halves.iter().enumerate() {
        for (j, &b) in halves.iter().enumerate() {
            // Different halves in the two positions of every source.
            let word = |x: u32, y: u32| x | y << 16;
            let c = halves[(i + j) % halves.len()];
            cases.push([
                word(a, halves[(i + 3) % halves.len()]),
                word(b, halves[(j + 7) % halves.len()]),
                word(c, halves[(i + 2 * j) % halves.len()]),
                0,
            ]);
        }
    }
    // The half selectors: whole sources, then swapped halves, then the low
    // result from the high halves and the high result from the low ones.
    let selectors = [(0u8, 7u8), (7, 0), (5, 2), (2, 5)];
    let mut failures = Vec::new();
    for &(name, opcode, reference) in operations {
        for &(op_sel, op_sel_hi) in &selectors {
            let [w0, w1] = vop3p(opcode, 0, [vgpr(1), vgpr(2), vgpr(3)], op_sel, op_sel_hi);
            let shader = build_valu_cs(&[w0, w1]);
            run_cases(&shader, &cases, |backend, _, inp, got| {
                let half =
                    |source: usize, high: bool| (inp[source] >> (16 * u32::from(high))) as u16;
                let result = |high: bool| {
                    let sel = if high { op_sel_hi } else { op_sel };
                    let pick = |k: usize| half(k, sel >> k & 1 != 0);
                    u32::from(reference(pick(0), pick(1), pick(2)))
                };
                let want = result(true) << 16 | result(false);
                if got != want && failures.len() < 40 {
                    failures.push(format!(
                        "{backend}: {name} op_sel {op_sel:03b} op_sel_hi {op_sel_hi:03b} \
                         ({:#010x}, {:#010x}, {:#010x}) = {got:#010x}, want {want:#010x}",
                        inp[0], inp[1], inp[2]
                    ));
                }
            });
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
