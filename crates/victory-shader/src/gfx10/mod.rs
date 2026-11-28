//! GFX10 machine code: instruction boundaries, encodings and opcodes.
//!
//! Instructions are found by following control flow from offset 0, not by a
//! linear sweep: compilers place data (constant tables) in the code region
//! after the end of the program. [`Instruction::fields`] decodes the operand
//! fields and [`Instruction::text`] prints them in AMD's SP3 syntax.
//!
//! Encodings by the top bits of the first dword:
//!
//! ```text
//! 0xxxxxxx  VOP2; 0111111 VOP1; 0111110 VOPC
//! 10        SOP2; 1011 SOPK; 101111101 SOP1; 101111110 SOPC; 101111111 SOPP
//! 110010 VINTRP  110011 VOP3P  110101 VOP3  110110 DS  110111 FLAT/GLOBAL/SCRATCH
//! 111000 MUBUF   111010 MTBUF  111100 MIMG  111101 SMEM  111110 EXP
//! ```

mod fields;
mod opcodes;
mod sp3;

pub use fields::{Cache, Dpp, Fields, NULL, Operand, Sdwa, Valu};

use std::collections::BTreeMap;

use crate::ShaderError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    Sop2,
    Sopk,
    Sop1,
    Sopc,
    Sopp,
    Smem,
    Vop2,
    Vop1,
    Vopc,
    Vop3,
    Vop3p,
    Vintrp,
    Ds,
    Mubuf,
    Mtbuf,
    Mimg,
    Exp,
    Flat,
    Global,
    Scratch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instruction {
    /// Byte offset in the program.
    pub offset: usize,
    pub encoding: Encoding,
    pub opcode: u16,
    /// Every dword of the instruction, including literals and extra dwords.
    pub words: Vec<u32>,
    /// [`name`] of `encoding` and `opcode`, resolved once at decode time.
    /// The translator asks about every instruction eight times; the table is
    /// a binary search.
    name: Option<&'static str>,
}

impl Instruction {
    /// An instruction of these words, its name resolved.
    pub fn new(offset: usize, encoding: Encoding, opcode: u16, words: Vec<u32>) -> Self {
        Self { offset, encoding, opcode, words, name: name(encoding, opcode) }
    }

    /// AMD's name for the opcode, if it is in our table.
    pub fn name(&self) -> Option<&'static str> {
        self.name
    }
}

/// Decodes every instruction reachable from offset 0, sorted by offset.
pub fn decode(code: &[u8]) -> Result<Vec<Instruction>, ShaderError> {
    if !code.len().is_multiple_of(4) {
        return Err(isa(code.len(), "code size is not a multiple of 4"));
    }
    let mut found = BTreeMap::new();
    let mut pending = vec![0usize];
    while let Some(mut offset) = pending.pop() {
        while !found.contains_key(&offset) {
            let inst = instruction(code, offset)?;
            let next = offset + 4 * inst.words.len();
            let flow = control(&inst, code.len(), |at| {
                found.range(..at).next_back().map(|(_, i)| i)
            })?;
            found.insert(offset, inst);
            match flow {
                Flow::Next => offset = next,
                Flow::Stop => break,
                Flow::Jump(target) => offset = target,
                Flow::Branch(target) => {
                    pending.push(target);
                    offset = next;
                }
            }
        }
    }
    let list: Vec<Instruction> = found.into_values().collect();
    if let Some(w) = list
        .windows(2)
        .find(|w| w[0].offset + 4 * w[0].words.len() > w[1].offset)
    {
        return Err(isa(w[1].offset, "a branch lands inside an instruction"));
    }
    Ok(list)
}

/// Where execution goes after an instruction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Flow {
    Next,
    /// End of the program, or an indirect jump that is not a far jump.
    Stop,
    Jump(usize),
    /// Conditional: to the target or the next instruction.
    Branch(usize),
}

/// Control flow after `inst`, resolving far jumps. `before(at)` returns the
/// decoded instruction with the highest offset below `at`, if any.
pub(crate) fn control<'a>(
    inst: &Instruction,
    len: usize,
    before: impl Fn(usize) -> Option<&'a Instruction>,
) -> Result<Flow, ShaderError> {
    Ok(match flow(inst, inst.offset + 4 * inst.words.len(), len)? {
        Flow::Stop if inst.name() == Some("s_setpc_b64") => {
            far_jump(before, inst, len)?.map_or(Flow::Stop, Flow::Jump)
        }
        flow => flow,
    })
}

fn flow(inst: &Instruction, next: usize, len: usize) -> Result<Flow, ShaderError> {
    let target = || {
        let delta = 4 * i64::from(inst.words[0] as u16 as i16);
        usize::try_from(next as i64 + delta)
            .ok()
            .filter(|&t| t < len)
            .ok_or(isa(inst.offset, "branch target outside the code"))
    };
    Ok(match (inst.encoding, inst.name().unwrap_or("")) {
        (
            Encoding::Sopp,
            "s_endpgm" | "s_endpgm_saved" | "s_endpgm_ordered_ps_done" | "s_code_end",
        ) => Flow::Stop,
        (Encoding::Sopp, "s_branch") => Flow::Jump(target()?),
        (Encoding::Sopp, name) if name.starts_with("s_cbranch_") => Flow::Branch(target()?),
        (Encoding::Sopk, "s_call_b64") => Flow::Branch(target()?),
        // Indirect jump; `decode` resolves the compiler's far-jump form.
        (Encoding::Sop1, "s_setpc_b64") => Flow::Stop,
        _ => Flow::Next,
    })
}

/// Resolves the far jump compilers emit beyond the reach of a branch offset:
/// `s_getpc_b64 s[n:n+1]`, `s_add_u32|s_sub_u32 sn, sn, literal`, a carry
/// into s(n+1), `s_setpc_b64 s[n:n+1]`, looking back over the straight-line
/// run just decoded. Any other `s_setpc_b64` (e.g. a handover to the next
/// merged stage through registers) gives `None`.
fn far_jump<'a>(
    before: impl Fn(usize) -> Option<&'a Instruction>,
    setpc: &Instruction,
    len: usize,
) -> Result<Option<usize>, ShaderError> {
    let n = setpc.words[0] & 0xFF;
    if n >= 127 {
        return Ok(None);
    }
    let mut lo: Option<(bool, u32)> = None;
    let mut hi: Option<(bool, u32)> = None;
    let mut at = setpc.offset;
    for _ in 0..4 {
        let Some(prev) = before(at) else {
            break;
        };
        if prev.offset + 4 * prev.words.len() != at {
            break;
        }
        // Never look back across a control-flow boundary into another block.
        if !matches!(flow(prev, at, len), Ok(Flow::Next))
            || matches!(prev.name(), Some("s_swappc_b64" | "s_rfe_b64"))
        {
            break;
        }
        let w = prev.words[0];
        let (sdst, ssrc0, ssrc1) = ((w >> 16) & 0x7F, w & 0xFF, (w >> 8) & 0xFF);
        let imm = |code: u32| -> Option<u32> {
            match code {
                128..=192 => Some(code - 128),
                193..=208 => Some((192i32 - code as i32) as u32),
                0xFF => prev.words.get(1).copied(),
                _ => None,
            }
        };
        match (prev.encoding, prev.name()) {
            (Encoding::Sop2, Some("s_addc_u32"))
                if hi.is_none() && lo.is_none() && sdst == n + 1 =>
            {
                let val = if ssrc0 == n + 1 {
                    imm(ssrc1)
                } else if ssrc1 == n + 1 {
                    imm(ssrc0)
                } else {
                    None
                };
                let Some(val) = val else { break };
                hi = Some((false, val));
            }
            (Encoding::Sop2, Some("s_subb_u32"))
                if hi.is_none() && lo.is_none() && sdst == n + 1 =>
            {
                let Some(val) = (ssrc0 == n + 1).then(|| imm(ssrc1)).flatten() else {
                    break;
                };
                hi = Some((true, val));
            }
            (Encoding::Sop2, Some("s_add_u32")) if lo.is_none() && sdst == n => {
                let val = if ssrc0 == n {
                    imm(ssrc1)
                } else if ssrc1 == n {
                    imm(ssrc0)
                } else {
                    None
                };
                let Some(val) = val else { break };
                lo = Some((false, val));
            }
            (Encoding::Sop2, Some("s_sub_u32")) if lo.is_none() && sdst == n => {
                let Some(val) = (ssrc0 == n).then(|| imm(ssrc1)).flatten() else {
                    break;
                };
                lo = Some((true, val));
            }
            (Encoding::Sop1, Some("s_getpc_b64")) if sdst == n => {
                let Some((is_sub, lo_val)) = lo else { break };
                let delta = match hi {
                    Some((hi_sub, hi_val)) if hi_sub == is_sub => {
                        let raw = ((u64::from(hi_val) << 32) | u64::from(lo_val)) as i64;
                        if is_sub {
                            raw.checked_neg()
                        } else {
                            Some(raw)
                        }
                    }
                    Some(_) => break,
                    None => {
                        let literal = i64::from(lo_val as i32);
                        Some(if is_sub { -literal } else { literal })
                    }
                };
                // s_getpc_b64 yields the address of the instruction after it.
                return delta
                    .and_then(|d| (prev.offset as i64).checked_add(4)?.checked_add(d))
                    .and_then(|t| usize::try_from(t).ok())
                    .filter(|&t| t < len && t.is_multiple_of(4))
                    .map(Some)
                    .ok_or(isa(setpc.offset, "far jump target outside the code"));
            }
            _ => {
                if writes_scalar_pair(prev, n as u8) {
                    break;
                }
            }
        }
        at = prev.offset;
    }
    Ok(None)
}

/// Whether `inst` writes either register in the scalar pair `s[n:n+1]`.
fn writes_scalar_pair(inst: &Instruction, n: u8) -> bool {
    let hits = |op: Operand| match op {
        Operand::Scalar { code, dwords } => {
            let end = code.saturating_add(dwords);
            code <= n + 1 && end > n
        }
        _ => false,
    };
    match inst.fields(true) {
        Fields::Sop { dst, .. } => dst.is_some_and(hits),
        Fields::Smem { data: Some(d), .. } => {
            let name = inst.name().unwrap_or("");
            !name.contains("_store_") && hits(d)
        }
        Fields::Valu(v) => v.dst.is_some_and(hits) || v.carry.is_some_and(hits),
        _ => false,
    }
}

fn instruction(code: &[u8], offset: usize) -> Result<Instruction, ShaderError> {
    let word = |i: usize| {
        code.get(offset + 4 * i..offset + 4 * i + 4)
            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
            .ok_or(isa(offset, "instruction runs past the end of the code"))
    };
    let w = word(0)?;
    let (encoding, opcode, len) = if w >> 31 == 0 {
        let src0 = w & 0x1FF;
        // Literal constant, SDWA, DPP16, DPP8, DPP8 with fetch-inactive.
        let extra = usize::from(matches!(src0, 0xFF | 0xF9 | 0xFA | 0xE9 | 0xEA));
        match w >> 25 {
            0x3F => (Encoding::Vop1, (w >> 9) & 0xFF, 1 + extra),
            0x3E => (Encoding::Vopc, (w >> 17) & 0xFF, 1 + extra),
            op => {
                let mk_ak = name(Encoding::Vop2, op as u16).is_some_and(|n| {
                    matches!(
                        n,
                        "v_madmk_f32"
                            | "v_madak_f32"
                            | "v_fmamk_f32"
                            | "v_fmaak_f32"
                            | "v_madmk_f16"
                            | "v_madak_f16"
                            | "v_fmamk_f16"
                            | "v_fmaak_f16"
                    )
                });
                (Encoding::Vop2, op, 1 + extra.max(usize::from(mk_ak)))
            }
        }
    } else if w >> 30 == 0b10 {
        let literal = |srcs: &[u32]| 1 + usize::from(srcs.contains(&0xFF));
        match w >> 23 {
            0x17D => {
                let op = (w >> 8) & 0xFF;
                let has_src0 = name(Encoding::Sop1, op as u16) != Some("s_getpc_b64");
                (
                    Encoding::Sop1,
                    op,
                    if has_src0 { literal(&[w & 0xFF]) } else { 1 },
                )
            }
            0x17E => (
                Encoding::Sopc,
                (w >> 16) & 0x7F,
                literal(&[w & 0xFF, (w >> 8) & 0xFF]),
            ),
            0x17F => (Encoding::Sopp, (w >> 16) & 0x7F, 1),
            _ if w >> 28 == 0b1011 => {
                let op = (w >> 23) & 0x1F;
                let imm32 = name(Encoding::Sopk, op as u16) == Some("s_setreg_imm32_b32");
                (Encoding::Sopk, op, 1 + usize::from(imm32))
            }
            _ => (
                Encoding::Sop2,
                (w >> 23) & 0x7F,
                literal(&[w & 0xFF, (w >> 8) & 0xFF]),
            ),
        }
    } else {
        let vop3 = |encoding: Encoding, op: u32, w1: u32| {
            let op_name = name(encoding, op as u16);
            let count = match op_name {
                Some(n) => fields::vop3_source_count(encoding, op as u16, n),
                None => 3,
            };
            // `v_interp_*_f16` stores {high, chan, attr} in src0, where 0xFF is
            // attr63.w rather than a 32-bit literal sentinel.
            let start = usize::from(
                encoding == Encoding::Vop3
                    && op_name.is_some_and(|n| n.starts_with("v_interp_")),
            );
            let has_lit = (start..count).any(|i| (w1 >> (9 * i)) & 0x1FF == 0xFF);
            2 + usize::from(has_lit)
        };
        match w >> 26 {
            0x32 => (Encoding::Vintrp, (w >> 16) & 0x3, 1),
            0x33 => {
                let op = (w >> 16) & 0x7F;
                (Encoding::Vop3p, op, vop3(Encoding::Vop3p, op, word(1)?))
            }
            0x35 => {
                let op = (w >> 16) & 0x3FF;
                (Encoding::Vop3, op, vop3(Encoding::Vop3, op, word(1)?))
            }
            0x36 => (Encoding::Ds, (w >> 18) & 0xFF, 2),
            0x37 => {
                let encoding = match (w >> 14) & 3 {
                    0 => Encoding::Flat,
                    1 => Encoding::Scratch,
                    2 => Encoding::Global,
                    _ => return Err(isa(offset, "reserved FLAT segment")),
                };
                (encoding, (w >> 18) & 0x7F, 2)
            }
            0x38 => (Encoding::Mubuf, (w >> 18) & 0x7F | (w >> 25 & 1) << 7, 2),
            0x3A => (
                Encoding::Mtbuf,
                (w >> 16) & 0x7 | (word(1)? >> 21 & 1) << 3,
                2,
            ),
            // Bits 2:1 count extra dwords of non-sequential address registers.
            0x3C => (
                Encoding::Mimg,
                (w >> 18) & 0x7F | (w & 1) << 7,
                2 + (w as usize >> 1 & 3),
            ),
            0x3D => (Encoding::Smem, (w >> 18) & 0xFF, 2),
            0x3E => (Encoding::Exp, 0, 2),
            _ => return Err(isa(offset, "unknown encoding")),
        }
    };
    let opcode = opcode as u16;
    Ok(Instruction {
        offset,
        encoding,
        opcode,
        words: (0..len).map(word).collect::<Result<_, _>>()?,
        name: name(encoding, opcode),
    })
}

fn name(encoding: Encoding, opcode: u16) -> Option<&'static str> {
    use opcodes as t;
    let (table, op) = match encoding {
        Encoding::Sop2 => (t::SOP2, opcode),
        Encoding::Sopk => (t::SOPK, opcode),
        Encoding::Sop1 => (t::SOP1, opcode),
        Encoding::Sopc => (t::SOPC, opcode),
        Encoding::Sopp => (t::SOPP, opcode),
        Encoding::Smem => (t::SMEM, opcode),
        Encoding::Vop2 => (t::VOP2, opcode),
        Encoding::Vop1 => (t::VOP1, opcode),
        Encoding::Vopc => (t::VOPC, opcode),
        // VOP3 also encodes the VOPC, VOP2 and VOP1 opcodes at fixed bases.
        Encoding::Vop3 => match opcode {
            0x000..0x100 => (t::VOPC, opcode),
            0x100..0x140 => (t::VOP2, opcode - 0x100),
            0x180..0x200 => (t::VOP1, opcode - 0x180),
            _ => (t::VOP3, opcode),
        },
        Encoding::Vop3p => (t::VOP3P, opcode),
        Encoding::Vintrp => (t::VINTRP, opcode),
        Encoding::Ds => (t::DS, opcode),
        Encoding::Mubuf => (t::MUBUF, opcode),
        Encoding::Mtbuf => (t::MTBUF, opcode),
        Encoding::Mimg => (t::MIMG, opcode),
        Encoding::Exp => return Some("exp"),
        Encoding::Flat => (t::FLAT, opcode),
        Encoding::Global => (t::GLOBAL, opcode),
        Encoding::Scratch => (t::SCRATCH, opcode),
    };
    table
        .binary_search_by_key(&op, |&(o, _)| o)
        .ok()
        .map(|i| table[i].1)
}

fn isa(offset: usize, reason: &'static str) -> ShaderError {
    ShaderError::Isa { offset, reason }
}

#[cfg(test)]
mod tests {
    use super::*;

    const S_ENDPGM: u32 = 0xBF81_0000;
    /// s_branch / s_cbranch_scc0 with a dword offset relative to the next instruction.
    const fn sopp(op: u32, simm16: i16) -> u32 {
        0xBF80_0000 | op << 16 | simm16 as u16 as u32
    }

    fn bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn summary(code: &[u32]) -> Vec<(usize, usize, &'static str)> {
        decode(&bytes(code))
            .unwrap()
            .iter()
            .map(|i| (i.offset, i.words.len(), i.name().unwrap_or("?")))
            .collect()
    }

    #[test]
    fn flat_offsets_ignore_bit_eleven() {
        let offset = |word0: u32| {
            let code = bytes(&[word0, 0x0000_0001, S_ENDPGM]);
            match decode(&code).unwrap()[0].fields(false) {
                Fields::Flat { offset, .. } => offset,
                fields => panic!("{fields:?}"),
            }
        };
        // flat_load_dword v0, v[1:2]: the FLAT offset is 11-bit and
        // unsigned, so bit 11 is ignored (0x800 -> 0, 0xfff -> 0x7ff).
        assert_eq!(offset(0xDC30_0800), 0);
        assert_eq!(offset(0xDC30_0FFF), 0x7FF);
    }

    #[test]
    fn tables_are_sorted() {
        for t in [
            opcodes::SOP2,
            opcodes::SOPK,
            opcodes::SOP1,
            opcodes::SOPC,
            opcodes::SOPP,
            opcodes::SMEM,
            opcodes::VOP2,
            opcodes::VOP1,
            opcodes::VOPC,
            opcodes::VOP3,
            opcodes::VOP3P,
            opcodes::VINTRP,
            opcodes::DS,
            opcodes::MUBUF,
            opcodes::MTBUF,
            opcodes::MIMG,
            opcodes::FLAT,
            opcodes::GLOBAL,
            opcodes::SCRATCH,
        ] {
            assert!(t.windows(2).all(|w| w[0].0 < w[1].0));
        }
    }

    #[test]
    fn lengths_and_names() {
        let code = [
            0xBE80_03FF,
            0x3F80_0000, // s_mov_b32 s0, literal
            0x0600_0000, // v_add_f32 v0, s0, v0
            0xD54B_0000,
            0x03FE_0000,
            0x4000_0000, // v_fma_f32 with a literal src2
            0xF080_0F0A,
            0,
            0, // image_sample (2D) with one extra address dword
            0xF800_080F,
            0, // exp
            S_ENDPGM,
        ];
        assert_eq!(
            summary(&code),
            [
                (0, 2, "s_mov_b32"),
                (8, 1, "v_add_f32"),
                (12, 3, "v_fma_f32"),
                (24, 3, "image_sample"),
                (36, 2, "exp"),
                (44, 1, "s_endpgm"),
            ]
        );
    }

    #[test]
    fn follows_branches_and_skips_data() {
        let code = [
            sopp(0x04, 1), // s_cbranch_scc0 -> 8
            sopp(0x02, 1), // s_branch -> 12
            S_ENDPGM,
            S_ENDPGM,
            S_ENDPGM, // data follows the program end
            0xFFFF_FFFF,
        ];
        let found: Vec<_> = summary(&code).iter().map(|i| i.0).collect();
        assert_eq!(found, [0, 4, 8, 12]);
    }

    #[test]
    fn follows_far_jumps() {
        let code = [
            0xBE88_1F00, // s_getpc_b64 s[8:9]
            0x8008_FF08,
            20,          // s_add_u32 s8, s8, 20  -> 4 + 20 = 24
            0x8209_8009, // s_addc_u32 s9, s9, 0
            0xBE80_2008, // s_setpc_b64 s[8:9]
            0xFFFF_FFFF, // not code
            S_ENDPGM,
        ];
        let found: Vec<_> = summary(&code).iter().map(|i| (i.0, i.2)).collect();
        assert_eq!(
            found,
            [
                (0, "s_getpc_b64"),
                (4, "s_add_u32"),
                (12, "s_addc_u32"),
                (16, "s_setpc_b64"),
                (24, "s_endpgm"),
            ]
        );
        // Through other registers (a handover to another stage): the path ends.
        assert_eq!(summary(&[0xBE80_2018, S_ENDPGM]).len(), 1);
    }

    #[test]
    fn follows_far_jumps_backward() {
        // The compiler's backward far jump adds a negative offset: the low
        // dword as a 32-bit literal, the high dword as -1 with the carry.
        let code = [
            sopp(0x02, 1), // s_branch -> 8, past the jump target
            S_ENDPGM,      // the far jump's target: only it reaches here
            0xBE88_1F00,   // s_getpc_b64 s[8:9]
            0x8008_FF08,
            0xFFFF_FFF8, // s_add_u32 s8, s8, -8 -> 12 - 8 = 4
            0x8209_C109, // s_addc_u32 s9, s9, -1
            0xBE80_2008, // s_setpc_b64 s[8:9]
        ];
        let found: Vec<_> = summary(&code).iter().map(|i| (i.0, i.2)).collect();
        assert_eq!(
            found,
            [
                (0, "s_branch"),
                (4, "s_endpgm"),
                (8, "s_getpc_b64"),
                (12, "s_add_u32"),
                (20, "s_addc_u32"),
                (24, "s_setpc_b64"),
            ]
        );
    }

    #[test]
    fn rejects_bad_code() {
        assert!(decode(&[0; 3]).is_err(), "size not a multiple of 4");
        assert!(
            decode(&bytes(&[sopp(0x02, 100)])).is_err(),
            "branch out of range"
        );
        assert!(
            decode(&bytes(&[0xBE80_03FF])).is_err(),
            "literal past the end"
        );
        assert!(
            decode(&bytes(&[0xE400_0000, 0])).is_err(),
            "unknown encoding"
        );
        // s_cbranch_scc0 into the literal of the s_mov_b32 at 4.
        assert!(decode(&bytes(&[sopp(0x04, 1), 0xBE80_03FF, S_ENDPGM, S_ENDPGM])).is_err());
    }

    #[test]
    fn vop3_interp_attr63_w_and_unused_slots_do_not_consume_literal() {
        // v_interp_p1ll_f16 (0x342) with src0 = 0xFF (attr63.w, high=0), src1 = v0 (256)
        let p1ll = [0xD400_0000 | (0x342 << 16), (256 << 9) | 0xFF];
        // v_interp_p2_f16 (0x35a) with src0 = 0xFF (attr63.w), src1 = v1, src2 = v2
        let p2 = [
            0xD400_0000 | (0x35A << 16),
            (258 << 18) | (257 << 9) | 0xFF,
        ];
        // 1-source VOP3 v_mov_b32 (0x181) with 0xFF in unused src1/src2
        let vop1_vop3 = [
            0xD400_0000 | (0x181 << 16),
            (0xFF << 18) | (0xFF << 9) | 256,
        ];
        // 2-source VOP3 v_add_f32 (0x103) with 0xFF in unused src2
        let vop2_vop3 = [
            0xD400_0000 | (0x103 << 16),
            (0xFF << 18) | (257 << 9) | 256,
        ];
        // 2-source VOP3P v_pk_add_f16 (0x00f) with 0xFF in unused src2
        let vop3p = [
            0xCC00_0000 | (0x00F << 16),
            (0xFF << 18) | (257 << 9) | 256,
        ];
        // s_getpc_b64 s[0:1] with 0xFF in its unused ssrc0 field
        let getpc = 0xBE80_1FFF;

        let code = [
            p1ll[0],
            p1ll[1],
            p2[0],
            p2[1],
            vop1_vop3[0],
            vop1_vop3[1],
            vop2_vop3[0],
            vop2_vop3[1],
            vop3p[0],
            vop3p[1],
            getpc,
            S_ENDPGM,
        ];
        assert_eq!(
            summary(&code),
            [
                (0, 2, "v_interp_p1ll_f16"),
                (8, 2, "v_interp_p2_f16"),
                (16, 2, "v_mov_b32"),
                (24, 2, "v_add_f32"),
                (32, 2, "v_pk_add_f16"),
                (40, 1, "s_getpc_b64"),
                (44, 1, "s_endpgm"),
            ]
        );
        let decoded = decode(&bytes(&code)).unwrap();
        let Fields::Valu(v) = decoded[0].fields(false) else {
            panic!("expected VALU");
        };
        assert_eq!(
            v.src[0],
            Operand::Attr {
                index: 63,
                chan: 3,
                high: false,
            }
        );
    }

    #[test]
    fn far_jump_respects_barriers_clobbers_alignment_and_inline_constants() {
        // Control-flow barrier (s_endpgm) between s_addc_u32 and s_setpc_b64:
        // s_setpc_b64 is reached via branch and must NOT look back across s_endpgm.
        let across_barrier = [
            sopp(0x04, 5), // 0: s_cbranch_scc0 -> 24 (s_setpc_b64)
            0xBE88_1F00,   // 4: s_getpc_b64 s[8:9]
            0x8008_FF08,   // 8: s_add_u32 s8, s8, 24
            24,            //    -> 8 + 24 = 32 (would land on trailing s_endpgm)
            0x8209_8009,   // 16: s_addc_u32 s9, s9, 0
            S_ENDPGM,      // 20: barrier
            0xBE80_2008,   // 24: s_setpc_b64 s[8:9] (handover)
            0xFFFF_FFFF,   // 28: data
            S_ENDPGM,      // 32: unreachable unless misresolved
        ];
        let offsets: Vec<_> = summary(&across_barrier).iter().map(|i| i.0).collect();
        assert!(!offsets.contains(&32), "crossed s_endpgm barrier: {offsets:?}");

        // Register clobber between s_addc_u32 and s_setpc_b64 stops far_jump.
        let clobbered = [
            0xBE88_1F00, // 0: s_getpc_b64 s[8:9]
            0x8008_FF08, // 4: s_add_u32 s8, s8, 24
            24,
            0x8209_8009, // 12: s_addc_u32 s9, s9, 0
            0xBE88_0300, // 16: s_mov_b32 s8, s0 (clobbers s8)
            0xBE80_2008, // 20: s_setpc_b64 s[8:9]
            0xFFFF_FFFF, // 24: data
            S_ENDPGM,    // 28: unreachable
        ];
        let offsets: Vec<_> = summary(&clobbered).iter().map(|i| i.0).collect();
        assert!(!offsets.contains(&28), "ignored s8 clobber: {offsets:?}");

        // Non-zero high carry word (1 << 32 + 20) must not be ignored as +20.
        let bad_high = [
            0xBE88_1F00, // 0: s_getpc_b64 s[8:9]
            0x8008_FF08, // 4: s_add_u32 s8, s8, 20
            20,
            0x8209_8109, // 12: s_addc_u32 s9, s9, 1 (inline const 1 = 129 = 0x81)
            0xBE80_2008, // 16: s_setpc_b64 s[8:9]
            0xFFFF_FFFF, // 20
            S_ENDPGM,    // 24
        ];
        assert!(decode(&bytes(&bad_high)).is_err());

        // Misaligned target (4 + 22 = 26) must be rejected.
        let misaligned = [
            0xBE88_1F00, // 0: s_getpc_b64 s[8:9]
            0x8008_FF08, // 4: s_add_u32 s8, s8, 22
            22,
            0x8209_8009, // 12: s_addc_u32 s9, s9, 0
            0xBE80_2008, // 16: s_setpc_b64 s[8:9]
            0xFFFF_FFFF, // 20
            S_ENDPGM,    // 24
            S_ENDPGM,    // 28
        ];
        assert!(decode(&bytes(&misaligned)).is_err());

        // Commutative s_add_u32 with inline constant 16 (code 144 = 0x90):
        // s_getpc_b64 at 0 -> PC=4, 4 + 16 = 20.
        let commutative_inline = [
            0xBE88_1F00, // 0: s_getpc_b64 s[8:9]
            0x8008_0890, // 4: s_add_u32 s8, 16, s8 (ssrc0=0x90=16, ssrc1=8)
            0x8209_0980, // 8: s_addc_u32 s9, 0, s9 (ssrc0=0x80=0, ssrc1=9)
            0xBE80_2008, // 12: s_setpc_b64 s[8:9]
            0xFFFF_FFFF, // 16: data
            S_ENDPGM,    // 20: target
        ];
        let found: Vec<_> = summary(&commutative_inline)
            .iter()
            .map(|i| (i.0, i.2))
            .collect();
        assert_eq!(
            found,
            [
                (0, "s_getpc_b64"),
                (4, "s_add_u32"),
                (8, "s_addc_u32"),
                (12, "s_setpc_b64"),
                (20, "s_endpgm"),
            ]
        );

        // Backward s_sub_u32 + s_subb_u32 with inline constant 8 (code 136 = 0x88):
        // s_getpc_b64 at 8 -> PC=12, 12 - 8 = 4.
        let backward_sub = [
            sopp(0x02, 1), // 0: s_branch -> 8
            S_ENDPGM,      // 4: far jump target
            0xBE88_1F00,   // 8: s_getpc_b64 s[8:9]
            0x8088_8808,   // 12: s_sub_u32 s8, s8, 8 (op 1 << 23 = 0x8080_0000)
            0x8289_8009,   // 16: s_subb_u32 s9, s9, 0 (op 5 << 23 = 0x8280_0000)
            0xBE80_2008,   // 20: s_setpc_b64 s[8:9]
        ];
        let found: Vec<_> = summary(&backward_sub).iter().map(|i| (i.0, i.2)).collect();
        assert_eq!(
            found,
            [
                (0, "s_branch"),
                (4, "s_endpgm"),
                (8, "s_getpc_b64"),
                (12, "s_sub_u32"),
                (16, "s_subb_u32"),
                (20, "s_setpc_b64"),
            ]
        );
    }
}
