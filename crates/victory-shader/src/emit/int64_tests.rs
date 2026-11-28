//! The 64-bit integer vector operations against the RDNA2 instruction set
//! document's definitions: the three 64-bit shifts and the 32 x 32 + 64
//! multiply-adds, each as a one-thread compute shader run twice (the low
//! dword of the result, then the high one) on the hardware adapter and WARP.

use super::valu_tests::{build_valu_cs, run_cases, vgpr, vop1, vop3};

/// Dwords for the halves of the 64-bit operands: around the sign and the
/// carry between the halves.
const DWORDS: &[u32] = &[
    0,
    1,
    2,
    0x7fff_ffff,
    0x8000_0000,
    0xffff_fffe,
    0xffff_ffff,
    0x1234_5678,
];

/// Runs `body` (leaving a 64-bit result in v[0:1]) and checks both dwords of
/// the result against `reference` over `cases`.
fn check(name: &str, body: &[u32], cases: &[[u32; 4]], reference: impl Fn([u32; 4]) -> u64) {
    let mut failures = Vec::new();
    let low = build_valu_cs(body);
    let mut high_body = body.to_vec();
    high_body.push(vop1(0x001, 0, vgpr(1))); // v_mov_b32 v0, v1
    let high = build_valu_cs(&high_body);
    for (half, shader) in [("low", &low), ("high", &high)] {
        run_cases(shader, cases, |backend, _, inp, got| {
            let want = reference(inp);
            let want = if half == "low" {
                want as u32
            } else {
                (want >> 32) as u32
            };
            if got != want && failures.len() < 40 {
                failures.push(format!(
                    "{backend}: {name}({:#x}, {:#x}, {:#x}, {:#x}) {half} dword = {got:#x}, \
                     want {want:#x}",
                    inp[0], inp[1], inp[2], inp[3]
                ));
            }
        });
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn shifts_of_64_bits_use_the_low_six_bits_of_the_count() {
    // v_{lshl,lshr,ashr}rev_b64 v[0:1], v1, v[2:3]: the count is the first
    // operand, the value the second.
    let mut cases = Vec::new();
    for count in [0u32, 1, 31, 32, 33, 63, 64, 65, 0xffff_ffc1] {
        for &lo in DWORDS {
            for &hi in DWORDS {
                cases.push([count, lo, hi, 0]);
            }
        }
    }
    let value = |c: [u32; 4]| u64::from(c[2]) << 32 | u64::from(c[1]);
    for (name, opcode, shift) in [
        (
            "v_lshlrev_b64",
            0x2ff,
            (|x: u64, n: u32| x << (n & 63)) as fn(u64, u32) -> u64,
        ),
        ("v_lshrrev_b64", 0x300, |x, n| x >> (n & 63)),
        ("v_ashrrev_i64", 0x301, |x, n| {
            ((x as i64) >> (n & 63)) as u64
        }),
    ] {
        let body = vop3(opcode, 0, vgpr(1), vgpr(2), 0);
        check(name, &body, &cases, |c| shift(value(c), c[0]));
    }
}

#[test]
fn multiply_add_of_32_by_32_plus_64_matches_the_definition() {
    // v_mad_{u64_u32,i64_i32} v[0:1], v1, v2, v[3:4]: a * b + c, the factors
    // 32 bits (unsigned or signed), the addend and the result 64.
    let mut cases = Vec::new();
    for &a in DWORDS {
        for &b in DWORDS {
            for (lo, hi) in [
                (0, 0),
                (0xffff_ffff, 0),
                (0xffff_ffff, 0xffff_ffff),
                (1, 0x8000_0000),
            ] {
                cases.push([a, b, lo, hi]);
            }
        }
    }
    let addend = |c: [u32; 4]| u64::from(c[3]) << 32 | u64::from(c[2]);
    let body = vop3(0x176, 0, vgpr(1), vgpr(2), vgpr(3));
    check("v_mad_u64_u32", &body, &cases, |c| {
        (u64::from(c[0]) * u64::from(c[1])).wrapping_add(addend(c))
    });
    let body = vop3(0x177, 0, vgpr(1), vgpr(2), vgpr(3));
    check("v_mad_i64_i32", &body, &cases, |c| {
        (i64::from(c[0] as i32) * i64::from(c[1] as i32)).wrapping_add(addend(c) as i64) as u64
    });
}
