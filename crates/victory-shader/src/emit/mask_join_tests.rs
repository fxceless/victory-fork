//! A lane mask carried round a loop and read as a number where not all of
//! its lanes run: the loop header's join of the mask (from before the
//! loop) and a number (computed from it in the loop) is a join of numbers,
//! the mask's edge its ballot where it is written (`Lower::fill_phis`).
//! The shape of the title's wave-wide ray traversals (shadow masks,
//! distant shadows), whose lane masks go round the traversal loop and
//! through its stack in memory. Lowered, validated by `dxil.dll` when
//! present, and run on the host GPU and WARP.

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

/// Per lane: v2 = the sum over four iterations of the mask (lanes 0-15 at
/// first, shifted right by one each iteration) as a number, added only in
/// lanes 0-7; stored at the lane's dword.
const PROGRAM: [u32; 20] = [
    0x7D88_0090, // v_cmp_gt_u32 vcc_lo, 16, v0
    0xBE82_036A, // s_mov_b32 s2, vcc_lo
    0xBE83_0380, // s_mov_b32 s3, 0
    0x7E04_0280, // v_mov_b32 v2, 0
    // 0x10: the loop
    0x7D88_0088, // v_cmp_gt_u32 vcc_lo, 8, v0
    0xBE85_3C6A, // s_and_saveexec_b32 s5, vcc_lo
    0x7E02_0202, // v_mov_b32 v1, s2
    0x4A04_0302, // v_add_nc_u32 v2, v2, v1
    0xBEFE_0305, // s_mov_b32 exec_lo, s5
    0x9002_8102, // s_lshr_b32 s2, s2, 1
    0x8003_8103, // s_add_u32 s3, s3, 1
    0xBF0A_8403, // s_cmp_lt_u32 s3, 4
    0xBF85_FFF7, // s_cbranch_scc1 0x10
    0x3406_0082, // v_lshlrev_b32 v3, 2, v0
    0xBE81_0380, // s_mov_b32 s1, 0
    0xF408_0200,
    0xFA00_0000, // s_load_dwordx4 s[8:11], s[0:1], 0x0
    0xE070_1000,
    0x8002_0203, // buffer_store_dword v2, v3, s[8:11], 0 offen
    0xBF81_0000, // s_endpgm
];

fn build(words: &[u32]) -> Result<HostShader, crate::ShaderError> {
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![RootParam {
            root: RootType::Table,
            table_offset_dwords: 0,
            sgprs: 1,
            base_register: 0,
            user_data: 0,
            slot: 0,
        }],
        used: vec![UsedResource {
            kind: ResourceType::Uav,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 1,
            threads: [32, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
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
    })?;
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
        v.validate(&mut shader.dxil.clone()).expect("DXIL validation failed");
    }
    Ok(shader)
}

#[test]
fn a_mask_carried_round_a_loop_is_read_as_the_number_it_is() {
    let shader = build(&PROGRAM).expect("lowers");
    let sum: u32 = (0..4).map(|k| 0xffffu32 >> k).sum();
    let want: Vec<u32> = (0..32).map(|l| if l < 8 { sum } else { 0 }).collect();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    let mut ran = 0;
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = match host.compute_pipeline(&shader) {
            Ok(p) => p,
            // WARP has no 32-lane waves for the program to declare.
            Err(e) if warp => {
                eprintln!("{label}: no pipeline ({e}); skipped");
                continue;
            }
            Err(e) => panic!("{label}: compute_pipeline failed: {e}"),
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(32)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        assert_eq!(out[0], want, "{label}");
        ran += 1;
    }
    assert!(ran > 0, "no SM 6.6 device available");
}
