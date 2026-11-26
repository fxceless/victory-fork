//! The merged ES+GS stage (a geometry shader), step 1 of its support.
//!
//! The console links two programs into one hardware stage when it creates a
//! pipeline with a geometry shader: the ES program, which sits in the vertex
//! container next to the ordinary VS, and the GS program, which is all the
//! geometry container holds. Both run on the same wave, so they share one
//! SGPR file: eight system SGPRs first, then the user data of the root
//! signature ([`HwStage::user_sgpr`]).
//!
//! What this step knows, from the title's programs as the issue on this
//! (fork issue 138) writes it down (static analysis, not a run):
//!
//! * The GS half loads from a pointer in user data 0 and 1 (s8 and s9) that
//!   is no root parameter: the console runtime's own table. [`RUNTIME_TABLE`]
//!   names it; resource tracking takes the pair as that table when the merged
//!   root layout (both halves', [`pair_es_gs`]) puts nothing there
//!   ([`crate::resource::track_merged`]; a geometry program tracked alone cannot
//!   know the ES half leaves them free), and [`crate::io::Input::RuntimeTable`]
//!   names the two dwords.
//! * A root parameter is the same in both halves: [`pair_es_gs`] merges the
//!   two layouts and refuses programs that disagree about a user-data dword.
//!
//! What it does not do yet: translate the stage. The ES half hands its
//! records over through LDS, the GS half reads them from there and writes
//! its output vertices there too, with `s_sendmsg` GS_EMIT, GS_CUT and
//! GS_DONE; none of that is modeled. Emission refuses a geometry program with
//! [`crate::emit::GS_EMIT`].

use crate::ShaderError::{self, Malformed};
use crate::native::{HwStage, NativeProgram, RootParam, RootType};

/// The first user-data dword of the runtime table pointer: it takes dwords 0
/// (low half) and 1 (high half) of the merged stage's user data.
pub const RUNTIME_TABLE: u8 = 0;

/// An ES program and the GS program it is linked with.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MergedGs {
    pub es: NativeProgram,
    pub gs: NativeProgram,
    /// The root parameters of the merged stage: those of either half, by user
    /// data (a parameter both halves list is in it once).
    pub layout: Vec<RootParam>,
}

impl MergedGs {
    /// The GS program as the merged stage runs it: the code of the GS half
    /// with the root layout of both halves, which is what its user SGPRs
    /// hold.
    pub fn program(&self) -> NativeProgram {
        NativeProgram {
            layout: self.layout.clone(),
            ..self.gs.clone()
        }
    }
}

/// Pairs an ES program with the GS program of the same pipeline.
///
/// Fails when the programs are not an ES and a GS, run at different wave
/// sizes, or disagree about what a user-data dword holds (the two halves
/// share the SGPRs, so a dword has one meaning).
pub fn pair_es_gs(es: NativeProgram, gs: NativeProgram) -> Result<MergedGs, ShaderError> {
    if es.stage != HwStage::Es {
        return Err(Malformed("the vertex half of a merged stage is not an ES program"));
    }
    if gs.stage != HwStage::Gs {
        return Err(Malformed("the geometry half of a merged stage is not a GS program"));
    }
    if es.wave64 != gs.wave64 {
        return Err(Malformed("ES and GS programs run at different wave sizes"));
    }
    let mut layout = es.layout.clone();
    for g in &gs.layout {
        let overlaps = |e: &RootParam| {
            let end = |p: &RootParam| u32::from(p.user_data) + p.sgprs;
            u32::from(e.user_data) < end(g) && u32::from(g.user_data) < end(e)
        };
        // The same table pointer listed by both halves with other ranges (the
        // offset and the base register of a range differ) is one root
        // parameter read through two ranges: both are kept.
        let same_pointer = |e: &RootParam| {
            g.root == RootType::Table
                && (e.root, e.user_data, e.sgprs, e.slot) == (g.root, g.user_data, g.sgprs, g.slot)
        };
        let over: Vec<RootParam> = layout.iter().filter(|e| overlaps(e)).copied().collect();
        if over.contains(g) && over.iter().all(|e| e == g || same_pointer(e)) {
            continue;
        }
        if !over.iter().all(same_pointer) {
            return Err(Malformed("ES and GS programs disagree about a user-data dword"));
        }
        layout.push(*g);
    }
    layout.sort_by_key(|p| (p.user_data, p.slot));
    Ok(MergedGs { es, gs, layout })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native::StageIo;

    fn program(stage: HwStage, layout: Vec<RootParam>) -> NativeProgram {
        NativeProgram {
            stage,
            wave64: true,
            vgprs: 8,
            layout,
            used: vec![],
            io: if stage == HwStage::Gs {
                StageIo::Gs
            } else {
                StageIo::None
            },
            code: vec![],
        }
    }

    fn param(root: RootType, user_data: u8, sgprs: u32, slot: u8) -> RootParam {
        RootParam {
            root,
            table_offset_dwords: 0,
            sgprs,
            base_register: 0,
            user_data,
            slot,
        }
    }

    #[test]
    fn a_parameter_both_halves_list_is_in_the_layout_once() {
        let shared = param(RootType::Constants, 2, 4, 0);
        let es = program(HwStage::Es, vec![shared, param(RootType::Table, 6, 1, 1)]);
        let gs = program(HwStage::Gs, vec![shared, param(RootType::Descriptor, 8, 2, 2)]);
        let merged = pair_es_gs(es, gs).unwrap();
        let at: Vec<_> = merged.layout.iter().map(|p| p.user_data).collect();
        assert_eq!(at, [2, 6, 8]);
        // The merged program is the GS code over the merged layout.
        let p = merged.program();
        assert_eq!((p.stage, p.layout.len()), (HwStage::Gs, 3));
    }

    #[test]
    fn the_same_table_pointer_with_other_ranges_in_each_half_is_one_parameter() {
        // One descriptor table (user data 4, root slot 2): the ES half reads
        // a range of it, the GS half another.
        let range = |offset, base| RootParam {
            table_offset_dwords: offset,
            base_register: base,
            ..param(RootType::Table, 4, 1, 2)
        };
        let es = program(HwStage::Es, vec![range(0, 0)]);
        let gs = program(HwStage::Gs, vec![range(8, 3)]);
        let merged = pair_es_gs(es, gs).unwrap();
        assert_eq!(merged.layout, [range(0, 0), range(8, 3)]);
        // A range both list is in it once.
        let es = program(HwStage::Es, vec![range(0, 0)]);
        let gs = program(HwStage::Gs, vec![range(0, 0), range(8, 3)]);
        assert_eq!(pair_es_gs(es, gs).unwrap().layout, [range(0, 0), range(8, 3)]);
        // Another root parameter on that dword is still a conflict.
        let es = program(HwStage::Es, vec![range(0, 0)]);
        let gs = program(HwStage::Gs, vec![RootParam { slot: 3, ..range(8, 3) }]);
        assert!(pair_es_gs(es, gs).is_err());
        // So is a table where the other half has constants.
        let es = program(HwStage::Es, vec![range(0, 0)]);
        let gs = program(HwStage::Gs, vec![param(RootType::Constants, 4, 1, 2)]);
        assert!(pair_es_gs(es, gs).is_err());
    }

    #[test]
    fn halves_that_disagree_about_a_dword_are_refused() {
        // The same dword, a table in one half and constants in the other.
        let es = program(HwStage::Es, vec![param(RootType::Table, 3, 1, 1)]);
        let gs = program(HwStage::Gs, vec![param(RootType::Constants, 2, 4, 0)]);
        assert!(pair_es_gs(es, gs).is_err());
        // Equal start but not the same parameter.
        let es = program(HwStage::Es, vec![param(RootType::Table, 3, 1, 1)]);
        let gs = program(HwStage::Gs, vec![param(RootType::Table, 3, 1, 2)]);
        assert!(pair_es_gs(es, gs).is_err());
    }

    #[test]
    fn only_an_es_and_a_gs_at_the_same_wave_size_pair() {
        let es = || program(HwStage::Es, vec![]);
        let gs = || program(HwStage::Gs, vec![]);
        assert!(pair_es_gs(es(), gs()).is_ok());
        assert!(pair_es_gs(gs(), es()).is_err(), "swapped");
        assert!(pair_es_gs(program(HwStage::Vs, vec![]), gs()).is_err());
        assert!(pair_es_gs(es(), program(HwStage::Vs, vec![])).is_err());
        let mut narrow = gs();
        narrow.wave64 = false;
        assert!(pair_es_gs(es(), narrow).is_err());
    }
}
