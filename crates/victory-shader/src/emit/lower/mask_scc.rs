//! `mask_scc`, moved out of `lower.rs` (from PR #258) so that file stays under the
//! 512 KiB asset-policy limit (docs/asset-policy.md). The code is unchanged.

use super::*;

impl<'a> Lower<'a> {
    /// SCC of a lane-mask operation read as data (`s_cselect` on it, say):
    /// whether the mask has a lane set, from its ballot where it is written.
    pub(super) fn mask_scc(&mut self, scc: Value) -> Result<V, Why> {
        let p = self.x.p;
        let &b = self.def_block.get(&scc).ok_or(LANE_MASK)?;
        let inst: &Inst = p.blocks[b]
            .insts
            .iter()
            .find(|i| i.write(Reg::Scc) == Some(scc))
            .ok_or(LANE_MASK)?;
        let Fields::Sop {
            dst: Some(Operand::Scalar { code, .. }),
            ..
        } = inst.fields
        else {
            return Err(LANE_MASK);
        };
        if inst.write(Reg::S(EXEC)).is_some() {
            return Err(LANE_MASK);
        }
        let lo = inst.write(Reg::S(code)).ok_or(LANE_MASK)?;
        let x = self.mask_dword(lo)?;
        let any = match inst.write(Reg::S(code + 1)) {
            Some(hi) if self.wave() == 64 => {
                let y = self.mask_dword(hi)?;
                let at = if b == self.cur_ir {
                    self.cur
                } else {
                    self.tail[b].ok_or("a lane mask written after its use")?
                };
                self.s.m.bin(at, Bin::Or, x, y)
            }
            _ => x,
        };
        let at = if b == self.cur_ir {
            self.cur
        } else {
            self.tail[b].ok_or("a lane mask written after its use")?
        };
        let zero = self.s.m.i32(0);
        Ok(self.s.m.cmp(at, pred::NE, any, zero))
    }
}
