//! Console primitive types the host lacks. A rectangle list gives three
//! corners per rectangle, the second the one between the other two, and
//! the hardware adds the fourth at `v0 + v2 - v1` (the title's
//! full-screen rectangles come as top left, bottom left, bottom right); on
//! the host each rectangle becomes two triangles, with the fourth vertex
//! made from the other three on the CPU.
//!
//! shortcut: the fourth vertex is computed from vertex inputs, not from
//! vertex shader outputs as the hardware does; that is the same for vertex
//! shaders linear in their inputs (blits, 2D quads). Inputs in formats
//! [`components`] does not know keep the first vertex's value, and so do
//! components past the end of the vertex (an element that reaches into the
//! next one, which the layout allows).
//!
//! A pipeline the title makes for rectangles completes them on the GPU
//! instead, in a geometry stage (`victory_shader::emit::rect_list_gs`):
//! from the vertex shader's outputs, whichever vertex the corner is, for
//! direct, indexed and indirect draws alike. This expansion is what is
//! left for a pipeline that could not get that stage.

/// Expands `rects` rectangles of `stride`-byte vertices from `src` into six
/// vertices each in `dst`: `(v0, v1, v2)` and `(v0, v2, v3)`, so both
/// triangles turn the same way. `elements` are (byte offset, format).
pub(crate) fn expand_rects(src: &[u8], dst: &mut [u8], stride: usize, elements: &[(u32, u32)]) {
    for (rect, out) in src
        .chunks_exact(3 * stride)
        .zip(dst.chunks_exact_mut(6 * stride))
    {
        let v = |i: usize| &rect[i * stride..(i + 1) * stride];
        for (slot, i) in [0, 1, 2, 0, 2].into_iter().enumerate() {
            out[slot * stride..(slot + 1) * stride].copy_from_slice(v(i));
        }
        let fourth = &mut out[5 * stride..];
        fourth.copy_from_slice(v(0));
        for &(offset, format) in elements {
            let at = offset as usize;
            let Some((n, kind)) = components(format) else {
                continue;
            };
            // Only the components inside the vertex: the rest belong to the
            // next one.
            let inside = stride.saturating_sub(at) / kind.bytes();
            for c in 0..n.min(inside) {
                let get = |i: usize| kind.read(&v(i)[at..], c);
                let value = get(0) + get(2) - get(1);
                kind.write(&mut fourth[at..], c, value);
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Kind {
    F32,
    F16,
    Unorm8,
    Unorm16,
}

/// Component count and kind of the vertex formats rectangles use.
fn components(format: u32) -> Option<(usize, Kind)> {
    Some(match format {
        2 => (4, Kind::F32),
        6 => (3, Kind::F32),
        16 => (2, Kind::F32),
        41 => (1, Kind::F32),
        10 => (4, Kind::F16),
        34 => (2, Kind::F16),
        54 => (1, Kind::F16),
        11 => (4, Kind::Unorm16),
        35 => (2, Kind::Unorm16),
        28 | 29 | 87 | 91 => (4, Kind::Unorm8),
        _ => return None,
    })
}

impl Kind {
    fn bytes(self) -> usize {
        match self {
            Kind::F32 => 4,
            Kind::F16 | Kind::Unorm16 => 2,
            Kind::Unorm8 => 1,
        }
    }

    fn read(self, at: &[u8], c: usize) -> f32 {
        let b = &at[c * self.bytes()..];
        match self {
            Kind::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Kind::F16 => half_to_f32(u16::from_le_bytes([b[0], b[1]])),
            Kind::Unorm16 => f32::from(u16::from_le_bytes([b[0], b[1]])) / 65535.0,
            Kind::Unorm8 => f32::from(b[0]) / 255.0,
        }
    }

    fn write(self, at: &mut [u8], c: usize, v: f32) {
        let n = self.bytes();
        let b = &mut at[c * n..(c + 1) * n];
        match self {
            Kind::F32 => b.copy_from_slice(&v.to_le_bytes()),
            Kind::F16 => b.copy_from_slice(&f32_to_half(v).to_le_bytes()),
            Kind::Unorm16 => {
                b.copy_from_slice(&((v.clamp(0.0, 1.0) * 65535.0).round() as u16).to_le_bytes())
            }
            Kind::Unorm8 => b[0] = (v.clamp(0.0, 1.0) * 255.0).round() as u8,
        }
    }
}

fn half_to_f32(h: u16) -> f32 {
    let sign = u32::from(h >> 15) << 31;
    let exp = u32::from((h >> 10) & 0x1f);
    let man = u32::from(h & 0x3ff);
    let bits = match (exp, man) {
        (0, 0) => sign,
        // Subnormal: scale the mantissa.
        (0, m) => {
            let v = m as f32 * 2f32.powi(-24);
            return if sign != 0 { -v } else { v };
        }
        (31, m) => sign | 0x7f80_0000 | m << 13,
        (e, m) => sign | (e + 112) << 23 | m << 13,
    };
    f32::from_bits(bits)
}

fn f32_to_half(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let raw_exp = (bits >> 23) & 0xff;
    let exp = raw_exp as i32 - 127 + 15;
    let man = bits & 0x7f_ffff;
    if raw_exp == 0xff {
        if man != 0 {
            return sign | 0x7e00 | ((man >> 13) & 0x03ff) as u16;
        }
        return sign | 0x7c00;
    }
    if exp >= 31 {
        return sign | 0x7c00; // too large: infinity
    }
    if exp <= 0 {
        if exp < -10 {
            return sign; // below half the smallest subnormal: zero
        }
        // Half subnormals are `m * 2^-24`. An f32 in this range is still
        // normal (exponent field >= 1), so its significand has the implicit
        // leading 1; shift it into the half mantissa, rounding to even.
        let sig = (1u32 << 23) | man;
        let shift = (14 - exp) as u32;
        let halfway = 1u32 << (shift - 1);
        let q = sig >> shift;
        let r = sig & ((1u32 << shift) - 1);
        let m = q + u32::from(r > halfway || (r == halfway && (q & 1) != 0));
        return sign | m as u16;
    }
    // Round to nearest, ties to even: a carry out of the mantissa is added
    // into the exponent (`2 - 2^-12` is 2.0); at most it reaches infinity,
    // 0x7c00.
    let half = (exp as u32) << 10;
    let rounded = (man + 0x0fff + ((man >> 13) & 1)) >> 13;
    sign | (half + rounded) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A rectangle of float2 positions and RGBA8 colors, given as top
    /// left, bottom left, bottom right: the fourth corner completes the
    /// parallelogram at the top right.
    #[test]
    fn rectangles_get_their_fourth_corner() {
        let vertex = |x: f32, y: f32, r: u8| {
            let mut v = Vec::new();
            v.extend(x.to_le_bytes());
            v.extend(y.to_le_bytes());
            v.extend([r, 0, 0, 255]);
            v
        };
        let src: Vec<u8> = [
            vertex(-1.0, 1.0, 0),
            vertex(-1.0, -1.0, 100),
            vertex(1.0, -1.0, 150),
        ]
        .concat();
        let mut dst = vec![0u8; 6 * 12];
        expand_rects(&src, &mut dst, 12, &[(0, 16), (8, 28)]);
        assert_eq!(dst[..36], src[..]);
        assert_eq!(dst[36..48], src[..12]);
        assert_eq!(dst[48..60], src[24..36]);
        assert_eq!(dst[60..], vertex(1.0, 1.0, 50)[..]);
        assert_eq!(f32_to_half(1.5), 0x3e00);
        assert_eq!(half_to_f32(0x3e00), 1.5);
        assert_eq!(half_to_f32(0xc000), -2.0);
    }

    /// A rounding that carries out of the mantissa raises the exponent
    /// (the carry used to be or-ed into an odd exponent, halving the value).
    #[test]
    fn halves_round_up_into_the_next_power_of_two() {
        assert_eq!(f32_to_half(2.0 - 2f32.powi(-12)), 0x4000);
        assert_eq!(f32_to_half(0.5 - 2f32.powi(-13)), 0x3800);
        assert_eq!(f32_to_half(65520.0), 0x7c00, "past the largest half");
        // An R16_FLOAT element at 1.0, 2^-12 and 1.0: the fourth corner is
        // 2 - 2^-12, which rounds to 2.0.
        let src: Vec<u8> = [0x3c00_u16, 0x0c00, 0x3c00]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let mut dst = vec![0u8; 6 * 2];
        expand_rects(&src, &mut dst, 2, &[(0, 54)]);
        assert_eq!(dst[10..], 0x4000_u16.to_le_bytes());
    }

    /// An element that reaches past the stride (float4 at 4 in 8-byte
    /// vertices, RGBA8 at the stride itself): the components inside get
    /// the fourth corner, the rest keep the first vertex's bytes. The old
    /// code indexed past the vertex and panicked.
    #[test]
    fn elements_past_the_stride_are_left_alone() {
        let vertex = |x: f32, y: f32| [x.to_le_bytes(), y.to_le_bytes()].concat();
        let src = [vertex(0.0, 2.0), vertex(1.0, 3.0), vertex(5.0, 7.0)].concat();
        let mut dst = vec![0u8; 6 * 8];
        expand_rects(&src, &mut dst, 8, &[(0, 16), (4, 2), (8, 28)]);
        assert_eq!(dst[..40], [&src[..], &src[..8], &src[16..]].concat()[..]);
        assert_eq!(dst[40..], vertex(4.0, 6.0)[..]);
    }

    /// `half_to_f32` reads subnormals, so an f16 rectangle output must be
    /// able to write them back instead of flushing every one to zero.
    #[test]
    fn half_subnormals_convert_in_both_directions() {
        assert_eq!(f32_to_half(2f32.powi(-24)), 0x0001);
        assert_eq!(f32_to_half(-2f32.powi(-24)), 0x8001);
        assert_eq!(f32_to_half(2f32.powi(-23)), 0x0002);
        assert_eq!(f32_to_half(1023.0 * 2f32.powi(-24)), 0x03ff);
        // Ties go to an even mantissa: 0/1 and 1/2.
        assert_eq!(f32_to_half(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_half(3.0 * 2f32.powi(-25)), 0x0002);
        // Below half the smallest subnormal, and normal values, are unchanged.
        assert_eq!(f32_to_half(2f32.powi(-26)), 0x0000);
        assert_eq!(f32_to_half(1.5), 0x3e00);
        assert_eq!(half_to_f32(0x0001), 2f32.powi(-24));
    }

    /// Issue #245: expanding `R16_FLOAT` rectangles with an exact halfway
    /// normal tie must round to even instead of always rounding up.
    #[test]
    fn r16_float_rect_normal_ties_round_to_even() {
        // 1.0, 0.0, 2^-11 -> fourth corner is 1.0 + 2^-11, halfway between
        // 0x3c00 (even) and 0x3c01 (odd).
        let src: Vec<u8> = [0x3c00_u16, 0x0000, 0x1000]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let mut dst = vec![0u8; 6 * 2];
        expand_rects(&src, &mut dst, 2, &[(0, 54)]);
        assert_eq!(u16::from_le_bytes([dst[10], dst[11]]), 0x3c00);
    }

    /// Issue #245: expanding `R16_FLOAT` rectangles with a NaN corner must
    /// preserve a quiet NaN instead of turning it into infinity (`0x7c00`).
    #[test]
    fn r16_float_rect_preserves_nan() {
        let src: Vec<u8> = [0x7e00_u16, 0x0000, 0x0000]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        let mut dst = vec![0u8; 6 * 2];
        expand_rects(&src, &mut dst, 2, &[(0, 54)]);
        let h = u16::from_le_bytes([dst[10], dst[11]]);
        assert!((h & 0x7fff) > 0x7c00, "expected quiet NaN, got {h:#06x}");
        assert_ne!(h & 0x0200, 0, "expected quiet NaN bit, got {h:#06x}");

        let neg_src: Vec<u8> = [0xfe00_u16, 0x0000, 0x0000]
            .iter()
            .flat_map(|h| h.to_le_bytes())
            .collect();
        expand_rects(&neg_src, &mut dst, 2, &[(0, 54)]);
        let neg_h = u16::from_le_bytes([dst[10], dst[11]]);
        assert_ne!(
            neg_h & 0x8000,
            0,
            "expected sign preserved, got {neg_h:#06x}"
        );
        assert!(
            (neg_h & 0x7fff) > 0x7c00,
            "expected negative quiet NaN, got {neg_h:#06x}"
        );
    }

    #[test]
    fn f32_to_half_ieee754_edge_cases() {
        // Normal ties-to-even (even stays, odd rounds up, non-ties follow nearest).
        assert_eq!(f32_to_half(1.0 + 2f32.powi(-11)), 0x3c00);
        assert_eq!(f32_to_half(1.0 + 2f32.powi(-10) + 2f32.powi(-11)), 0x3c02);
        assert_eq!(f32_to_half(1.0 + 2f32.powi(-11) + 2f32.powi(-23)), 0x3c01);
        assert_eq!(
            f32_to_half(1.0 + 2f32.powi(-10) + 2f32.powi(-11) - 2f32.powi(-23)),
            0x3c01
        );

        // Signed zeros.
        assert_eq!(f32_to_half(0.0), 0x0000);
        assert_eq!(f32_to_half(-0.0), 0x8000);

        // Signed infinities and finite overflow.
        assert_eq!(f32_to_half(f32::INFINITY), 0x7c00);
        assert_eq!(f32_to_half(f32::NEG_INFINITY), 0xfc00);
        assert_eq!(f32_to_half(65520.0), 0x7c00);
        assert_eq!(f32_to_half(-65520.0), 0xfc00);
        assert_eq!(f32_to_half(1e10), 0x7c00);
        assert_eq!(f32_to_half(-1e10), 0xfc00);

        // Quiet and low-payload NaNs keep sign and produce nonzero quiet NaN mantissa.
        for (bits, expect_sign) in [
            (0x7fc0_0000_u32, 0x0000_u16),
            (0xffc0_0000_u32, 0x8000_u16),
            (0x7f80_0001_u32, 0x0000_u16),
            (0xff80_0001_u32, 0x8000_u16),
        ] {
            let h = f32_to_half(f32::from_bits(bits));
            assert_eq!(h & 0x8000, expect_sign);
            assert!((h & 0x7fff) > 0x7c00, "got {h:#06x} for {bits:#010x}");
            assert_ne!(h & 0x0200, 0, "got {h:#06x} for {bits:#010x}");
        }
        assert_eq!(f32_to_half(f32::from_bits(0x7fc0_2000)), 0x7e01);

        // Subnormal round-to-even ties (positive, negative, and carry to smallest normal).
        assert_eq!(f32_to_half(2f32.powi(-25)), 0x0000);
        assert_eq!(f32_to_half(-2f32.powi(-25)), 0x8000);
        assert_eq!(f32_to_half(3.0 * 2f32.powi(-25)), 0x0002);
        assert_eq!(f32_to_half(-3.0 * 2f32.powi(-25)), 0x8002);
        assert_eq!(f32_to_half(2f32.powi(-14) - 2f32.powi(-25)), 0x0400);
    }
}
