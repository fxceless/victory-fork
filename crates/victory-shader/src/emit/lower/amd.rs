//! Changes to the translation that keep AMD's shader compiler from failing on the title's
//! programs: amdxc64 8.18.10.0514 (Radeon RX 9060 XT / 9070 XT, driver 32.0.32029.2010)
//! dereferences null, or ends the process, creating some of their pipelines as otherwise
//! translated. Found by creating every program's pipeline (2,684 compute; 23,279 others with a
//! stand-in partner stage) in a process of its own.
//!
//! The programs whose buffer size queries crash it are left to #1213 (a non-uniform handle
//! per query). Nine others build once one clean-up pass runs; it runs for them whatever
//! `VICTORY_SHADER_OPT` says (`passes`, by container digest).
//!
//! Kept to this file and a one-line hook where each change acts, so it carries to forks.

use std::cell::Cell;

use super::super::bitcode::opt::{FACTS, FOLD};

thread_local! {
    /// The clean-up passes of the program being lowered on this thread.
    static PASSES: Cell<u32> = const { Cell::new(0) };
}

/// (container digest, clean-up passes run whatever `VICTORY_SHADER_OPT` says).
const FIXES: [(u128, u32); 9] = [
    // indirect_diffuse_lighting:CS_ScreenProbes_ActivateRays_d60186e6_Wrapped: with no
    // clean-up the process ends (0xFFFFFFFF); folding constants alone is enough.
    (0x5212_83e5_0211_e301_e7fa_ed9e_0299_dc22, FOLD),
    // The glints pixel programs build once branches tell their conditions.
    // PS_MIRROR_PBR_MATERIAL_CLEARCOAT_GLINTS
    (0xe416_8a83_c5c6_ff23_73e3_4325_35ad_7c9d, FACTS),
    // PS_MIRROR_PBR_MATERIAL_GLINTS
    (0x3024_35a1_3c4c_b152_2a2c_e785_8fab_7e34, FACTS),
    // PS_MIRROR_PBR_MATERIAL_TERRAIN_GLINTS
    (0x60f0_c383_b6ba_41b5_895d_4a40_39e7_a781, FACTS),
    // PS_MIRROR_PBR_MATERIAL_TERRAIN_GLINTS_TRANSLUCENCY
    (0xb867_ed98_e402_b5f8_8844_35ad_5d48_5b36, FACTS),
    // PS_STUDIOSHOT_PBR_MATERIAL_CLEARCOAT_GLINTS_LIGHTING
    (0x8e32_45bb_242a_2e59_17c1_d2ce_d51c_7fc9, FACTS),
    // PS_STUDIOSHOT_PBR_MATERIAL_GLINTS_LIGHTING
    (0x5bbf_8eed_9a4f_5133_8b8d_2e36_cd9f_4b66, FACTS),
    // PS_STUDIOSHOT_PBR_MATERIAL_TERRAIN_GLINTS_LIGHTING
    (0x3c03_fca2_b4cc_b79b_fb87_eb51_81de_8a70, FACTS),
    // PS_STUDIOSHOT_PBR_MATERIAL_TERRAIN_GLINTS_TRANSLUCENCY_LIGHTING
    (0x3df1_012d_c24c_6daf_af0b_c2ce_8fc7_8b7b, FACTS),
];

/// Before lowering the program whose container digest is `digest` on this thread.
pub(in crate::emit) fn begin(digest: [u8; 16]) {
    let d = u128::from_be_bytes(digest);
    PASSES.set(FIXES.iter().find(|f| f.0 == d).map_or(0, |f| f.1));
}

/// Clean-up passes the program being lowered gets whatever `VICTORY_SHADER_OPT` says.
pub(in crate::emit) fn passes() -> u32 {
    PASSES.get()
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_the_listed_programs_get_passes() {
        super::begin([0; 16]);
        assert_eq!(super::passes(), 0);
        super::begin(0x5212_83e5_0211_e301_e7fa_ed9e_0299_dc22u128.to_be_bytes());
        assert_eq!(super::passes(), super::FOLD);
        super::begin(0x3024_35a1_3c4c_b152_2a2c_e785_8fab_7e34u128.to_be_bytes());
        assert_eq!(super::passes(), super::FACTS);
    }
}
