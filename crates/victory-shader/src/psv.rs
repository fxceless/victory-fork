//! What a title container's pipeline state validation part (`PSV0`) says
//! about a geometry shader.
//!
//! The title's containers carry the standard `PSV0` part next to `ISG1` and
//! `OSG1` (runtime info version 3, 52 bytes, in the shipped containers
//! looked at). For a geometry shader it holds the stage limits that the
//! merged ES+GS stage's native stage data also encodes in a layout not read
//! yet ([`crate::native`]), in the documented form:
//!
//! ```text
//! +0x00  u32 runtime info size (0x18 version 0, 0x24 v1, 0x30 v2, 0x34 v3)
//! info   +0x00 u32 input primitive (DXIL InputPrimitive: 1 point, 2 line,
//!              3 triangle, 6 line with adjacency, 7 triangle with
//!              adjacency, 8..=39 patch of 1..=32 control points)
//!        +0x04 u32 output topology (DXIL PrimitiveTopology: 1 point list,
//!              3 line strip, 5 triangle strip)
//!        +0x08 u32 output stream mask (bit n: stream n)
//!        +0x0c u8  SV_Position written
//!        +0x10 u32 x2 minimum and maximum wave size
//! v1 on  +0x18 u8  shader stage (2 geometry), +0x19 u8 view id used,
//!        +0x1a u16 maximum output vertex count
//! ```
//!
//! The instance count (`[instance(n)]`) is not in `PSV0`: it is in the DXIL
//! entry metadata, which the title may strip, and in the native stage data.
//! The signature rows are the container's `ISG1`/`OSG1`, read with the other
//! stages' rows.

use crate::ShaderError::{self, Malformed};
use crate::dxbc::u32_at;

/// Runtime info from version 1 on: the first with the shader stage.
const INFO_V1: u32 = 0x24;
const INFO: usize = 4;
const STAGE: usize = INFO + 0x18;
const STAGE_GEOMETRY: u8 = 2;
const INPUT: usize = INFO;
const OUTPUT: usize = INFO + 0x04;
const STREAMS: usize = INFO + 0x08;
const POSITION: usize = INFO + 0x0c;
const MAX_VERTICES: usize = INFO + 0x1a;
/// D3D's `maxvertexcount` limit.
const MAX_VERTICES_LIMIT: u16 = 1024;
/// D3D's four geometry output streams.
const STREAM_MASK: u32 = 0xf;

/// A geometry shader's limits as its container's `PSV0` gives them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub input: InputPrimitive,
    pub output: OutputTopology,
    /// Bit n: the program emits to stream n.
    pub streams: u8,
    /// The program writes `SV_Position`.
    pub position: bool,
    /// Most vertices one invocation emits (`maxvertexcount`).
    pub max_vertices: u16,
}

/// The primitive one invocation reads.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputPrimitive {
    Point,
    Line,
    Triangle,
    LineAdjacency,
    TriangleAdjacency,
    /// A patch of this many control points (1..=32).
    Patch(u8),
}

impl InputPrimitive {
    /// The vertices one invocation reads.
    pub fn vertices(self) -> u32 {
        match self {
            Self::Point => 1,
            Self::Line => 2,
            Self::Triangle => 3,
            Self::LineAdjacency => 4,
            Self::TriangleAdjacency => 6,
            Self::Patch(n) => u32::from(n),
        }
    }

    fn from_raw(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::Point,
            2 => Self::Line,
            3 => Self::Triangle,
            6 => Self::LineAdjacency,
            7 => Self::TriangleAdjacency,
            8..=39 => Self::Patch((v - 7) as u8),
            _ => return None,
        })
    }
}

/// What the emitted vertices make.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputTopology {
    Points,
    LineStrips,
    TriangleStrips,
}

impl OutputTopology {
    fn from_raw(v: u32) -> Option<Self> {
        Some(match v {
            1 => Self::Points,
            3 => Self::LineStrips,
            5 => Self::TriangleStrips,
            _ => return None,
        })
    }
}

/// The geometry limits in a `PSV0` part's data; `None` when the part is of
/// another stage or too old to name its stage (runtime info version 0).
pub fn geometry(psv0: &[u8]) -> Result<Option<Geometry>, ShaderError> {
    let size = u32_at(psv0, 0)?;
    if size < INFO_V1 {
        return Ok(None);
    }
    if psv0.len() < INFO + size as usize {
        return Err(Malformed("PSV0 runtime info truncated"));
    }
    if psv0[STAGE] != STAGE_GEOMETRY {
        return Ok(None);
    }
    let input = InputPrimitive::from_raw(u32_at(psv0, INPUT)?)
        .ok_or(Malformed("PSV0 geometry input primitive unknown"))?;
    let output = OutputTopology::from_raw(u32_at(psv0, OUTPUT)?)
        .ok_or(Malformed("PSV0 geometry output topology unknown"))?;
    let streams = u32_at(psv0, STREAMS)?;
    if streams == 0 || streams & !STREAM_MASK != 0 {
        return Err(Malformed("PSV0 geometry stream mask out of range"));
    }
    let max_vertices = u16::from_le_bytes([psv0[MAX_VERTICES], psv0[MAX_VERTICES + 1]]);
    if !(1..=MAX_VERTICES_LIMIT).contains(&max_vertices) {
        return Err(Malformed("PSV0 geometry vertex maximum out of range"));
    }
    Ok(Some(Geometry {
        input,
        output,
        streams: streams as u8,
        position: psv0[POSITION] != 0,
        max_vertices,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `PSV0` part with runtime info of `size` bytes, for a geometry
    /// shader reading triangles and writing at most 3 vertices as a triangle
    /// strip with a position to stream 0, then `edit` applied to the info.
    fn part(size: u32, edit: impl FnOnce(&mut [u8])) -> Vec<u8> {
        let mut info = vec![0u8; size as usize];
        let mut put = |at: usize, v: u32| info[at..at + 4].copy_from_slice(&v.to_le_bytes());
        put(0x00, 3);
        put(0x04, 5);
        put(0x08, 1);
        put(0x14, u32::MAX);
        info[0x0c] = 1;
        if size >= INFO_V1 {
            info[0x18] = STAGE_GEOMETRY;
            info[0x1a..0x1c].copy_from_slice(&3u16.to_le_bytes());
        }
        edit(&mut info);
        let mut out = size.to_le_bytes().to_vec();
        out.extend(info);
        // No resources; the signature tables would follow.
        out.extend(0u32.to_le_bytes());
        out
    }

    fn with(edit: impl FnOnce(&mut [u8])) -> Result<Option<Geometry>, ShaderError> {
        geometry(&part(0x34, edit))
    }

    fn set_u32(info: &mut [u8], at: usize, v: u32) {
        info[at..at + 4].copy_from_slice(&v.to_le_bytes());
    }

    #[test]
    fn reads_the_title_runtime_info_version() {
        // Version 3 (52 bytes), as the title's containers carry it.
        assert_eq!(
            with(|_| {}).unwrap(),
            Some(Geometry {
                input: InputPrimitive::Triangle,
                output: OutputTopology::TriangleStrips,
                streams: 1,
                position: true,
                max_vertices: 3,
            })
        );
        // Versions 1 and 2 hold the same fields.
        for size in [0x24, 0x30] {
            assert_eq!(
                geometry(&part(size, |_| {})).unwrap(),
                with(|_| {}).unwrap()
            );
        }
    }

    #[test]
    fn input_primitive() {
        let cases = [
            (1, InputPrimitive::Point, 1),
            (2, InputPrimitive::Line, 2),
            (3, InputPrimitive::Triangle, 3),
            (6, InputPrimitive::LineAdjacency, 4),
            (7, InputPrimitive::TriangleAdjacency, 6),
            (8, InputPrimitive::Patch(1), 1),
            (39, InputPrimitive::Patch(32), 32),
        ];
        for (raw, input, vertices) in cases {
            let g = with(|i| set_u32(i, 0x00, raw)).unwrap().unwrap();
            assert_eq!(
                (g.input, g.input.vertices()),
                (input, vertices),
                "raw {raw}"
            );
        }
        for raw in [0, 4, 5, 40] {
            assert!(with(|i| set_u32(i, 0x00, raw)).is_err(), "raw {raw}");
        }
    }

    #[test]
    fn output_topology() {
        let cases = [
            (1, OutputTopology::Points),
            (3, OutputTopology::LineStrips),
            (5, OutputTopology::TriangleStrips),
        ];
        for (raw, output) in cases {
            let g = with(|i| set_u32(i, 0x04, raw)).unwrap().unwrap();
            assert_eq!(g.output, output, "raw {raw}");
        }
        // Lists are not a geometry shader's output.
        for raw in [0, 2, 4, 6] {
            assert!(with(|i| set_u32(i, 0x04, raw)).is_err(), "raw {raw}");
        }
    }

    #[test]
    fn stream_mask() {
        for mask in [1, 2, 0b1010, 0xf] {
            let g = with(|i| set_u32(i, 0x08, mask)).unwrap().unwrap();
            assert_eq!(u32::from(g.streams), mask);
        }
        for mask in [0, 0x10, 0x11] {
            assert!(with(|i| set_u32(i, 0x08, mask)).is_err(), "mask {mask:#x}");
        }
    }

    #[test]
    fn position_written() {
        assert!(!with(|i| i[0x0c] = 0).unwrap().unwrap().position);
        assert!(with(|i| i[0x0c] = 1).unwrap().unwrap().position);
    }

    #[test]
    fn vertex_maximum() {
        for n in [1u16, 12, 1024] {
            let g = with(|i| i[0x1a..0x1c].copy_from_slice(&n.to_le_bytes()));
            assert_eq!(g.unwrap().unwrap().max_vertices, n);
        }
        for n in [0u16, 1025] {
            let g = with(|i| i[0x1a..0x1c].copy_from_slice(&n.to_le_bytes()));
            assert!(g.is_err(), "{n}");
        }
    }

    #[test]
    fn other_stages_and_stageless_parts_are_not_geometry() {
        // Pixel and vertex, as the title's own containers tag them.
        for stage in [0, 1, 5, 13] {
            assert_eq!(with(|i| i[0x18] = stage).unwrap(), None, "stage {stage}");
        }
        // Version 0 names no stage.
        assert_eq!(geometry(&part(0x18, |_| {})).unwrap(), None);
    }

    #[test]
    fn truncated_parts_are_malformed() {
        assert!(geometry(&[]).is_err());
        assert!(geometry(&[0x34, 0, 0]).is_err());
        let p = part(0x34, |_| {});
        assert!(geometry(&p[..4 + 0x20]).is_err());
        // The runtime info alone is enough: what follows is not read.
        assert!(geometry(&p[..4 + 0x34]).unwrap().is_some());
        // A size past the data, however large, is an error, not a panic.
        let past = (p.len() - 4 + 1) as u32;
        for size in [past, u32::MAX] {
            let mut p = part(0x34, |_| {});
            p[..4].copy_from_slice(&size.to_le_bytes());
            assert!(geometry(&p).is_err(), "size {size:#x}");
        }
    }

    fn unhex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    /// The `PSV0` part DXC (Windows SDK 10.0.26100, `-T gs_6_0`) wrote for
    /// this program, signature tables and all:
    ///
    /// ```hlsl
    /// struct V { float4 pos : SV_Position; float2 uv : TEXCOORD0; };
    /// [maxvertexcount(3)]
    /// void main(triangle V v[3], inout TriangleStream<V> s)
    /// { s.Append(v[0]); s.RestartStrip(); }
    /// ```
    const DXC_TRIANGLES: &str = "\
        340000000300000005000000010000000100000000000000ffffffff0200030002020002\
        0200000000000000000000000000000013000000000000001800000000544558434f4f52\
        4400544558434f4f5244006d61696e000100000000000000100000000000000000000000\
        010044030304000001000000000000000101420003020000000000000000000001004403\
        030400000a00000000000000010142000302000001000000020000000400000008000000\
        10000000200000000000000000000000";

    /// As [`DXC_TRIANGLES`], with two point streams instead:
    /// `void main(triangle V v[3], inout PointStream<V> a,
    /// inout PointStream<V> b) { a.Append(v[0]); b.Append(v[0]); }`.
    const DXC_TWO_STREAMS: &str = "\
        340000000300000001000000030000000100000000000000ffffffff0200030002040002\
        020200000000000000000000000000001c000000000000002400000000544558434f4f52\
        4400544558434f4f524400544558434f4f5244006d61696e000000000100000000000000\
        100000000000000000000000010044030304000001000000000000000101420003020000\
        000000000000000001004403030400000a00000000000000010142000302000000000000\
        000000000100440303041000130000000000000001014200030210000100000002000000\
        040000000800000010000000200000000000000000000000010000000200000004000000\
        0800000010000000200000000000000000000000";

    #[test]
    fn reads_what_dxc_writes() {
        assert_eq!(
            geometry(&unhex(DXC_TRIANGLES)).unwrap(),
            Some(Geometry {
                input: InputPrimitive::Triangle,
                output: OutputTopology::TriangleStrips,
                streams: 1,
                position: true,
                max_vertices: 3,
            })
        );
        assert_eq!(
            geometry(&unhex(DXC_TWO_STREAMS)).unwrap(),
            Some(Geometry {
                input: InputPrimitive::Triangle,
                output: OutputTopology::Points,
                streams: 0b11,
                position: true,
                max_vertices: 3,
            })
        );
    }
}
