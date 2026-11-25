//! A console wave64 compute program on a host of 32-lane waves: a wave reduction inside a
//! per-lane branch, with that branch taken wave-wide ([`crate::structure::structure_wide`])
//! so that the two host waves can exchange their halves inside it. Lowered, validated by
//! `dxil.dll` when present, and run on the host GPU when it takes waves of 32.

use std::collections::HashSet;

use super::host::{Dispatch, Host, Root, View};
use super::{HostShader, lower, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{ShaderError, io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

/// 64 threads in one console wave. Each thread t holds the bit `1 << (t >> 1)`; the threads
/// below s0 (a root constant) OR theirs over the wave inside `if (t < s0) { ... }` and store
/// the result to dword 0 of the raw UAV.
/// `skip`: the branch past the body when no thread is below s0: `s_cbranch_execz`, or
/// `s_cbranch_scc0` on the SCC `s_and_saveexec` sets ("some lane is left").
fn program(skip: u32) -> NativeProgram {
    let mut words = vec![
        0x7E0A_0281, // v_mov_b32 v5, 1
        0x2C0C_0081, // v_lshrrev_b32 v6, 1, v0
        0x340E_0B06, // v_lshlrev_b32 v7, v6, v5
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        skip | 23,   // the branch to the restore of EXEC
    ];
    words.extend_from_slice(&super::wave::WAVE_OR_V7_TO_S7);
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ]);
    native(&words)
}

/// A wave64 compute program of 64 threads (one console wave; the thread index in v0) with 4
/// root constants in s0..s3 and a raw UAV's descriptor in s[8:11], then `body`.
fn native(body: &[u32]) -> NativeProgram {
    native_of(body, true, 64)
}

/// The same for a console of 32-lane waves when `wave64` is not set, with `threads` threads.
fn native_of(body: &[u32], wave64: bool, threads: u32) -> NativeProgram {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0x3D << 26 | 0x02 << 18 | 8 << 6 | 2,
        125 << 25, // s_load_dwordx4 s[8:11], s[4:5], 0
    ];
    words.extend_from_slice(body);
    words.push(S_ENDPGM);
    NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 16,
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
            rsrc2: 5 << 1, // 5 user SGPRs: s0..s4
            threads: [threads, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    }
}

/// `program()` for a host of 32-lane waves, with the branches closing `wide` taken wave-wide.
fn lowered(native: &NativeProgram, p: &ir::Program, wide: &HashSet<usize>) -> Result<HostShader, ShaderError> {
    lowered_on(native, p, wide, 32)
}

/// [`lowered`] for a host whose waves have at most `host_wave` lanes.
fn lowered_on(
    native: &NativeProgram,
    p: &ir::Program,
    wide: &HashSet<usize>,
    host_wave: u32,
) -> Result<HostShader, ShaderError> {
    let st = structure::structure_wide(p, wide).expect("structure");
    let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(p, native).expect("io::raise");
    let was = lower::TEST_HOST_WAVE.replace(Some(host_wave));
    let lowered = lower::lower(&lower::Parts {
        native,
        p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_HOST_WAVE.set(was);
    let lowered = lowered?;
    Ok(HostShader {
        stage: HwStage::Cs,
        dxil: lowered.dxil,
        user_data: lowered.user_data,
        samplers: lowered.samplers,
        static_samplers: Vec::new(),
        root_cbvs: lowered.root_cbvs,
        root_srvs: lowered.root_srvs,
        root_uavs: lowered.root_uavs,
        root_indexed_views: res.root_indexed_views,
    })
}

#[test]
fn a_sum_of_halves_over_the_wave_is_the_consoles_tree_bit_for_bit() {
    // thread t holds the 16-bit float with bits 0x3C00 + 37 t (1.0 and up, with odd
    // mantissas, so that additions round); all of them store the wave's sum
    let mut words = vec![
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    // (the wave OR's words with v_add_f16 for v_or_b32: opcode 50 for 28)
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_add_f16").count(), 6);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // the console's sum: lanes 2k and 2k+1, then pairs, ... each addition rounded to 16 bits
    // (to nearest, ties to even); values here are normal numbers
    let wide = |h: u16| f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13);
    let narrow = |f: f32| {
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let terms: Vec<u16> = (0..64u16).map(|t| 0x3C00 + 37 * t).collect();
    let mut tree = terms.clone();
    while tree.len() > 1 {
        tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
    }
    let in_a_row = terms.iter().copied().reduce(add).unwrap();
    assert_ne!(tree[0], in_a_row, "these terms tell the tree from another order");
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
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
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    // (the high half is lane 0's term's: zero)
    assert_eq!(out[0][0], u32::from(tree[0]), "tree {:#x}, in a row {in_a_row:#x}", tree[0]);
    eprintln!("ran on the hardware device: the sum {:#x} of the console's tree", tree[0]);
}

/// The console's 16-bit wave sum where the lanes at or past s0 have left the program (a
/// thread past the end of the data): on the console they add +0, and after each step of the
/// tree they hold their run's partial sum all the same. On a host of 64-lane waves (AMD) the
/// lanes that are here read each partner run's sum from a lane of it that is here.
#[test]
fn a_sum_of_halves_where_lanes_have_left_is_the_consoles_tree_bit_for_bit() {
    let mut words = vec![
        0x7DA8_0000, // v_cmpx_gt_u32 exec, s0, v0: the lanes below s0 stay
        0, // s_cbranch_execz to the end (filled in below)
        0x340C_0085, // v_lshlrev_b32 v6, 5, v0
        0x340E_0082, // v_lshlrev_b32 v7, 2, v0
        0x4A0E_0F06, // v_add_nc_u32 v7, v6, v7
        0x4A0E_0F00, // v_add_nc_u32 v7, v0, v7
        0x4A0E_0EFF,
        0x0000_3C00, // v_add_nc_u32 v7, 0x3c00, v7
    ];
    words.extend(super::wave::WAVE_OR_V7_TO_S7.iter().map(|&w| if w >> 25 == 28 { w & 0x01FF_FFFF | 50 << 25 } else { w }));
    words.extend_from_slice(&[
        0x7E00_0207, // v_mov_b32 v0, s7
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ]);
    // past every word after it, to the program's end
    words[1] = 0xBF88_0000 | (words.len() as u32 - 2);
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered_on(&native, &p, &HashSet::new(), 64).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let wide = |h: u16| {
        if h == 0 {
            0.0
        } else {
            f32::from_bits((u32::from(h >> 10) + 112) << 23 | u32::from(h & 0x3FF) << 13)
        }
    };
    let narrow = |f: f32| {
        if f == 0.0 {
            return 0;
        }
        let b = f.to_bits();
        let h = ((b >> 23) - 112) << 10 | (b >> 13 & 0x3FF);
        let rest = b & 0x1FFF;
        (h + u32::from(rest > 0x1000 || rest == 0x1000 && h & 1 == 1)) as u16
    };
    let add = |a: u16, b: u16| narrow(wide(a) + wide(b));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let mut ran = false;
    for stay in [64u32, 63, 40, 33, 32, 17, 5, 1] {
        let terms: Vec<u16> = (0..64u32)
            .map(|t| if t < stay { (0x3C00 + 37 * t) as u16 } else { 0 })
            .collect();
        let mut tree = terms.clone();
        while tree.len() > 1 {
            tree = tree.chunks(2).map(|c| add(c[0], c[1])).collect();
        }
        let Some(got) = on_the_card(&shader, [stay, 0]) else {
            return;
        };
        ran = true;
        // (the high half is lane 0's term's: zero)
        assert_eq!(got, u32::from(tree[0]), "{stay} lanes stay: got {got:#x}, the console's tree {:#x}", tree[0]);
    }
    if ran {
        eprintln!("ran on the hardware device: the console's tree with lanes gone, in every case");
    }
}

/// One `v_add_f16` whose exact result lies between two halves: 1 + 3/1024 (0x3C03) plus 2
/// (0x4000) is 3 + 1.5 steps of that range, which the console rounds up to 0x4202. The
/// host's `f32tof16` may give 0x4201 (toward zero); with `VICTORY_F16_NEAREST=1` the
/// translation gives the console's whatever the host does.
#[test]
fn a_16_bit_sum_rounds_to_nearest_when_asked_to() {
    let words = [
        0x7E0E_02FF,
        0x0000_3C03, // v_mov_b32 v7, 0x3c03
        0x7E0C_02FF,
        0x0000_4000, // v_mov_b32 v6, 0x4000
        0x6400_0D07, // v_add_f16 v0, v7, v6
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    assert!(p.blocks.iter().flat_map(|b| &b.insts).any(|i| i.name() == "v_add_f16"));
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for nearest in [false, true] {
        let was = lower::TEST_F16_NEAREST.replace(Some(nearest));
        let shader = lowered(&native, &p, &HashSet::new());
        lower::TEST_F16_NEAREST.set(was);
        let shader = shader.expect("lower");
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
        let Ok(host) = Host::new(false) else {
            eprintln!("no SM 6.6 hardware device; not run");
            return;
        };
        let Ok((root, pso)) = host.compute_pipeline(&shader) else {
            eprintln!("the hardware device does not take this program; not run");
            return;
        };
        let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
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
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let got = out[0][0] & 0xFFFF;
        eprintln!("v_add_f16 0x3c03 + 0x4000 on the hardware device, nearest {nearest}: {got:#x}");
        if nearest {
            assert_eq!(got, 0x4202);
        } else {
            assert!(got == 0x4201 || got == 0x4202, "{got:#x}");
        }
    }
}

#[test]
fn a_wave_or_inside_a_lane_branch_needs_the_branch_wave_wide() {
    for (skip, name) in [(0xBF88_0000, "s_cbranch_execz"), (0xBF84_0000, "s_cbranch_scc0")] {
        a_wave_or_inside(skip, name);
    }
}

fn a_wave_or_inside(skip: u32, name: &str) {
    let native = program(skip);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == name))
        .expect("the branch");
    // per lane, the two host waves cannot meet inside the branch
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a wave reduction where not every lane runs"),
        "{:?}",
        per_lane.err()
    );
    // ... and the lowering names the block, around which that branch is the one to widen
    let needs = lower::take_needs();
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &needs), Some(vec![branch]));
    // (taken, past the body, when every lane's bit is clear)
    assert_eq!(st.wave_any[branch], Some(false));
    // wave-wide, every lane runs the reduction (EXEC stays a guard on what it does there)
    let wide = HashSet::from([branch]);
    let st = structure::structure_wide(&p, &wide).expect("structure");
    let inside = match p.blocks[branch].exit {
        ir::Exit::Branch { not_taken, .. } => not_taken,
        _ => unreachable!(),
    };
    assert!(st.guards[inside].iter().any(Option::is_some));
    let shader = lowered(&native, &p, &wide).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // On a GPU that takes waves of 32 (two host waves for the console's one): the OR is of
    // both halves, whichever half's lanes take the branch.
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (threads below s0 take the branch; thread t gives bit t >> 1)
    for (below, expected) in [(64u32, u32::MAX), (40, 0x000F_FFFF), (33, 0x0001_FFFF), (10, 0x1F), (1, 1)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = below;
            }
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
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        assert_eq!(out[0][0], expected, "{name}: {below} threads in the branch");
    }
    eprintln!("ran on the hardware device ({name}): the OR of both halves in every case");
}

/// Runs `shader` on the hardware device with the root constants `s0`, `s1` and gives dword 0
/// of the UAV; None (and a line) without a device that takes it.
fn on_the_card(shader: &HostShader, constants: [u32; 2]) -> Option<u32> {
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return None;
    };
    let Ok((root, pso)) = host.compute_pipeline(shader) else {
        eprintln!("the hardware device does not take this program; not run");
        return None;
    };
    let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for u in &shader.user_data {
        if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
            ud[u.index as usize] = constants[d as usize];
        }
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
        .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
    Some(out[0][0])
}

/// A 32-lane console wave. A register holds a lane mask on one path (the lanes below s0)
/// and a number on the other, and is read as a number after the join, as the trace programs
/// keep their lanes' masks and their node numbers on one stack
/// (`deferred_tiled:CS_LaserLightShadowRaytracing`, s46 at 0x104, 0x75c and 0x188). The
/// mask arrives as the ballot of its lanes.
#[test]
fn a_lane_mask_joined_with_a_number_is_read_as_the_number_it_is() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE94_036A, // s_mov_b32 s20, vcc_lo
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0xBF85_0001, // s_cbranch_scc1 over the next
        0xBE94_0385, // s_mov_b32 s20, 5
        0x7E00_0214, // v_mov_b32 v0, s20
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    for (below, other, expected) in [(32u32, 0u32, u32::MAX), (10, 0, 0x3FF), (0, 0, 0), (10, 1, 5)] {
        let Some(got) = on_the_card(&shader, [below, other]) else { return };
        assert_eq!(got, expected, "{below} lanes in the mask, s1 = {other}");
    }
    eprintln!("ran on the hardware device: the mask's bits, or the number");
}

/// A 32-lane console wave. `if (t < s0) { ... SCC = (s1 == 7) }` and a branch on SCC after
/// the join: the skip brings the SCC of `s_and_saveexec` ("some lane is left") along
/// (`rtdirectionalshadows:CS_RTShadowTrace32HQ_TiledFarOnly`, 0x360-0x39c). On the console
/// the skip is taken by the wave, so with some lane inside every lane sees the inner test.
/// A lane that skipped on its own would see "some lane is left" instead: refused per lane,
/// right with the branch taken wave-wide. Lane 31 stores what it saw.
#[test]
fn scc_of_a_mask_operation_joined_with_a_scalar_test_needs_the_skip_wave_wide() {
    let words = [
        0x7D88_0000, // v_cmp_gt_u32 vcc_lo, s0, v0
        0xBE8D_3C6A, // s_and_saveexec_b32 s13, vcc_lo
        0xBF88_0001, // s_cbranch_execz over the next
        0xBF06_8701, // s_cmp_eq_u32 s1, 7
        0xBEFE_030D, // s_mov_b32 exec_lo, s13
        0xBF85_0002, // s_cbranch_scc1 to "seven"
        0x7E0A_0281, // v_mov_b32 v5, 1
        0xBF82_0001, // s_branch over the next
        0x7E0A_0282, // seven: v_mov_b32 v5, 2
        0x7DA4_009F, // v_cmpx_eq_u32 exec_lo, 31, v0
        0x38 << 26 | 0x1C << 18,
        0x80 << 24 | 2 << 16 | 5 << 8, // buffer_store_dword v5, off, s[8:11], 0
    ];
    let native = native_of(&words, false, 32);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let branch = (0..p.blocks.len())
        .find(|&b| p.blocks[b].insts.last().is_some_and(|i| i.name() == "s_cbranch_execz"))
        .expect("the skip");
    lower::take_needs();
    let per_lane = lowered(&native, &p, &HashSet::new());
    assert!(
        matches!(&per_lane, Err(ShaderError::Unsupported { what, .. }) if *what == "a lane mask read as data"),
        "{:?}",
        per_lane.err()
    );
    let st = structure::structure(&p).expect("structure");
    assert_eq!(lower::branches_around(&st, &lower::take_needs()), Some(vec![branch]));
    let shader = lowered(&native, &p, &HashSet::from([branch])).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    // (lanes below s0 inside, s1) -> what lane 31 stores: 2 if SCC was set at the branch
    for (below, s1, expected) in [(0u32, 7u32, 1u32), (10, 7, 2), (10, 3, 1), (32, 3, 1), (32, 7, 2)] {
        let Some(got) = on_the_card(&shader, [below, s1]) else { return };
        assert_eq!(got, expected, "{below} lanes inside, s1 = {s1}");
    }
    eprintln!("ran on the hardware device: the inner test's SCC for every lane once some lane is inside");
}

/// The ray-tracing structure builder's scan across the console waves of a group (its three
/// prefix-sum programs): lane 63 of each wave puts the wave's total in LDS, then after the
/// barrier every lane walks the totals of the waves below its own in a loop whose lanes leave
/// at different turns, carrying the running sum through `v_readfirstlane` of a value read
/// from LDS at an address every lane holds alike. Here each lane leaves after `t & 3` turns
/// (lanes of one host wave leave at different turns) and the totals are 8 w + 5.
/// `late_write`: one more LDS write after the barrier, in the stretch the loop reads in.
fn scan_program(late_write: bool) -> NativeProgram {
    let mut words = vec![
        0x2C1C_0086, // v_lshrrev_b32 v14, 6, v0 (the wave)
        0x3618_0083, // v_and_b32 v12, 3, v0 (turns)
        0x3414_1C83, // v_lshlrev_b32 v10, 3, v14
        0x4A14_1485, // v_add_nc_u32 v10, 5, v10
        0xD765_000D,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v13, -1, 0
        0xD766_000D,
        0x0002_1AC1, // v_mbcnt_hi_u32_b32 v13, -1, v13
        0xBE94_047E, // s_mov_b64 s[20:21], exec
        0x7DA4_1ABF, // v_cmpx_eq_u32 exec, 63, v13
        0x341A_1C82, // v_lshlrev_b32 v13, 2, v14
        0xD834_0000,
        0x0000_0A0D, // ds_write_b32 v13, v10
        0xBEFE_0414, // s_mov_b64 exec, s[20:21]
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xBF8A_0000, // s_barrier
    ];
    if late_write {
        words.extend_from_slice(&[
            0x7E1E_02FF,
            0x0000_0100, // v_mov_b32 v15, 0x100
            0xD834_0000,
            0x0000_0A0F, // ds_write_b32 v15, v10
        ]);
    }
    words.extend_from_slice(&[
        // the title's loop, word for word
        0x7E1A_0280, // v_mov_b32 v13, 0
        0xBE80_047E, // s_mov_b64 s[0:1], exec
        0x7DAA_1880, // v_cmpx_ne_u32 exec, 0, v12
        0xBF88_0013, // s_cbranch_execz (the restore)
        0xBE82_047E, // s_mov_b64 s[2:3], exec
        0xBE86_047E, // s_mov_b64 s[6:7], exec
        0xBE92_0380, // s_mov_b32 s18, 0
        0xBE91_0380, // s_mov_b32 s17, 0
        0xBF82_0002, // s_branch (the header)
        0x87FE_0610, // s_and_b64 exec, s[16:17], s[6:7]
        0x7E22_050D, // v_readfirstlane_b32 s17, v13
        0x8F10_8212, // s_lshl_b32 s16, s18, 2 (the header)
        0x8112_8112, // s_add_i32 s18, s18, 1
        0x7E1A_0210, // v_mov_b32 v13, s16
        0xD8D8_0000,
        0x0D00_000D, // ds_read_b32 v13, v13
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0x4A1A_1A11, // v_add_nc_u32 v13, s17, v13
        0xBE90_047E, // s_mov_b64 s[16:17], exec
        0x7DA4_1812, // v_cmpx_eq_u32 exec, s18, v12
        0xBF80_0000, // s_nop 0
        0x8A86_7E06, // s_andn2_b64 s[6:7], s[6:7], exec
        0xBF85_FFF2, // s_cbranch_scc1 (the back edge)
        0xBEFE_0400, // s_mov_b64 exec, s[0:1] (the restore)
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8002_0D03, // buffer_store_dword v13, v3, s[8:11], 0 offen
    ]);
    let mut native = native_of(&words, true, 256);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    native
}

#[test]
fn a_first_lane_read_of_a_value_read_alike_from_lds_where_lanes_have_left_a_loop() {
    let native = scan_program(false);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    // with an LDS write in the stretch the loop reads in, the read is not taken as alike
    let late = scan_program(true);
    let q = ir::build(&late.code, late.wave64, late.vgprs).expect("ir::build");
    assert!(
        matches!(lowered(&late, &q, &HashSet::new()), Err(ShaderError::Unsupported { what, .. }) if what == "a first-lane read where not every lane runs"),
        "an LDS read with a write in its stretch is taken as alike"
    );
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    let ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
    for _ in 0..8 {
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud.clone()))],
                    views: &[View::RawUav(256)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let want: Vec<u32> = (0..256u32).map(|t| (0..t & 3).map(|w| 8 * w + 5).sum()).collect();
        assert_eq!(out[0], want);
    }
    eprintln!("ran on the hardware device: every lane has the sum of the totals below its turn count");
}

/// A max or a min over the wave picked by a scalar branch, each arm its own whole-wave
/// section, and EXEC restored once where the arms meet, after the first-lane read of the
/// result (the water transmittance program's depth range, `ir::join_wave_sections`). Thread
/// t holds (t ^ 21) + 7; with s0 zero every thread stores the max over the wave at its dword, else
/// the min.
#[test]
fn wave_reductions_in_both_arms_of_a_scalar_branch_meet_at_one_restore() {
    let words = [
        0x3A08_0095, // v_xor_b32 v4, 21, v0
        0x7E08_0D04, // v_cvt_f32_u32 v4, v4
        0x0608_08FF,
        0x40E0_0000, // v_add_f32 v4, 7.0, v4
        0xBF06_8000, // s_cmp_eq_u32 s0, 0
        0x85C2_80C1, // s_cselect_b64 s[66:67], -1, 0
        // the title's words from here (the result in s12 for its s8)
        0x8AEA_427E, // s_andn2_b64 vcc, exec, s[66:67]
        0xBF87_0014, // s_cbranch_vccnz (the min)
        0xBE84_25C1, // s_or_saveexec_b64 s[4:5], -1
        0x7E04_02FF,
        0xFF80_0000, // v_mov_b32 v2, -inf
        0xD501_0002,
        0x0012_0902, // v_cndmask_b32 v2, v2, v4, s[4:5]
        0x2004_04E9,
        0xDE54_C102, // v_max_f32 v2, v2, v2 dpp8:[1,0,3,2,5,4,7,6]
        0x2004_04E9,
        0x9360_1202, // v_max_f32 v2, v2, v2 dpp8:[2,2,0,0,6,6,4,4]
        0x2004_04E9,
        0x0009_2402, // v_max_f32 v2, v2, v2 dpp8:[4,4,4,4,0,0,0,0]
        0x2004_04FA,
        0xFF09_4002, // v_max_f32 v2, v2, v2 row_mirror bound_ctrl:1
        0xD778_0003,
        0x0201_0102, // v_permlanex16_b32 v3, v2, 0, 0
        0x2004_0503, // v_max_f32 v2, v3, v2
        0xD760_000C,
        0x0001_4102, // v_readlane_b32 s12, v2, 32
        0x2004_040C, // v_max_f32 v2, s12, v2
        0xBF82_0013, // s_branch (the join)
        0xBE84_25C1, // s_or_saveexec_b64 s[4:5], -1 (the min)
        0x7E04_02FF,
        0x7F80_0000, // v_mov_b32 v2, +inf
        0xD501_0002,
        0x0012_0902, // v_cndmask_b32 v2, v2, v4, s[4:5]
        0x1E04_04E9,
        0xDE54_C102, // v_min_f32 v2, v2, v2 dpp8:[1,0,3,2,5,4,7,6]
        0x1E04_04E9,
        0x9360_1202, // v_min_f32 v2, v2, v2 dpp8:[2,2,0,0,6,6,4,4]
        0x1E04_04E9,
        0x0009_2402, // v_min_f32 v2, v2, v2 dpp8:[4,4,4,4,0,0,0,0]
        0x1E04_04FA,
        0xFF09_4002, // v_min_f32 v2, v2, v2 row_mirror bound_ctrl:1
        0xD778_0003,
        0x0201_0102, // v_permlanex16_b32 v3, v2, 0, 0
        0x1E04_0503, // v_min_f32 v2, v3, v2
        0xD760_000C,
        0x0001_4102, // v_readlane_b32 s12, v2, 32
        0x1E04_040C, // v_min_f32 v2, s12, v2
        0xD760_000C,
        0x0001_0102, // v_readlane_b32 s12, v2, 0 (the join)
        0xBEFE_0404, // s_mov_b64 exec, s[4:5]
        0x7E0A_020C, // v_mov_b32 v5, s12
        0x3406_0082, // v_lshlrev_b32 v3, 2, v0
        0xE070_1000,
        0x8002_0503, // buffer_store_dword v5, v3, s[8:11], 0 offen
    ];
    let native = native(&words);
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    // each arm ends its own section: the join's read of the first lane and restore are in
    // both (two reads each), and the join's head is gone
    assert_eq!(p.blocks.iter().flat_map(|b| &b.insts).filter(|i| i.name() == "v_readlane_b32").count(), 4);
    let restores = |b: &ir::Block| b.insts.iter().filter(|i| i.name() == "s_mov_b64" && i.write(ir::Reg::S(ir::EXEC)).is_some()).count();
    assert_eq!(p.blocks.iter().filter(|b| b.insts.iter().any(|i| i.name() == "v_permlanex16_b32") && restores(b) == 1).count(), 2);
    let shader = lowered(&native, &p, &HashSet::new()).expect("lower");
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    for (s0, result) in [(0u32, 70.0f32), (1, 7.0)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        ud[0] = s0;
        let out = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(64)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
        let want = vec![result.to_bits(); 64];
        assert_eq!(out[0], want, "s0 {s0}");
    }
    eprintln!("ran on the hardware device: the max and the min of the arms");
}

/// 64 threads in one console wave (512 bytes of LDS) sharing an LDS dword inside a per-lane
/// branch, as the console compiler leaves it without barriers for a group of one wave
/// (`sr_tracing:CS_Tiling_Generate_Opaque`, `GpuParticleUpdate:CS_Visibility`): thread 0 zeroes
/// dword 0; inside `if (t < s0)` the thread `t == s1` writes `s1 + 7` there and every thread
/// of the branch reads it back and stores it at its dword of the UAV; the branch also sets
/// s20, which every thread stores at dword 64 + t after it (1 when some thread took the
/// branch, as the console's scalar code runs for the whole wave). `in_loop` runs the branch
/// twice in a loop on a scalar count.
fn shared_in_a_branch(in_loop: bool) -> NativeProgram {
    let mut words = vec![
        0x7E02_0280, // v_mov_b32 v1, 0
        0x3408_0082, // v_lshlrev_b32 v4, 2, v0
        0xBE94_0380, // s_mov_b32 s20, 0
        0x7D84_0080, // v_cmp_eq_u32 vcc, 0, v0
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xD834_0000,
        0x0000_0101, // ds_write_b32 v1, v1
        0xBEFE_040C, // s_mov_b64 exec, s[12:13]
    ];
    if in_loop {
        words.push(0xBE95_0380); // s_mov_b32 s21, 0
    }
    words.extend_from_slice(&[
        0x7D88_0000, // v_cmp_gt_u32 vcc, s0, v0 (the loop's head)
        0xBE8C_246A, // s_and_saveexec_b64 s[12:13], vcc
        0xBF88_000C, // s_cbranch_execz (the restore)
        0xBE94_0381, // s_mov_b32 s20, 1
        0x7D84_0001, // v_cmp_eq_u32 vcc, s1, v0
        0xBE8E_246A, // s_and_saveexec_b64 s[14:15], vcc
        0x4A04_0087, // v_add_nc_u32 v2, 7, v0
        0xD834_0000,
        0x0000_0201, // ds_write_b32 v1, v2
        0xBEFE_040E, // s_mov_b64 exec, s[14:15]
        0xD8D8_0000,
        0x0300_0001, // ds_read_b32 v3, v1
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xE070_1000,
        0x8002_0304, // buffer_store_dword v3, v4, s[8:11], 0 offen
        0xBEFE_040C, // s_mov_b64 exec, s[12:13] (the restore)
    ]);
    if in_loop {
        words.extend_from_slice(&[
            0x8115_8115, // s_add_i32 s21, s21, 1
            0xBF0A_8215, // s_cmp_lt_u32 s21, 2
            0xBF85_FFED, // s_cbranch_scc1 (the loop's head)
        ]);
    }
    words.extend_from_slice(&[
        0x7E0A_0214, // v_mov_b32 v5, s20
        0xE070_1100,
        0x8002_0504, // buffer_store_dword v5, v4, s[8:11], 0 offen offset:256
    ]);
    let mut native = native_of(&words, true, 64);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // 512 bytes of LDS
    }
    native
}

/// The program, its form for a host of 32-lane waves, and what `f` makes of it.
fn on_a_32_lane_host<R>(native: &NativeProgram, f: impl FnOnce(&crate::dxbc::Container, &ir::Program) -> R) -> R {
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let c = crate::dxbc::Container { digest: [0; 16], version: (1, 0), parts: Vec::new() };
    let was = lower::TEST_HOST_WAVE.replace(Some(32));
    let r = f(&c, &p);
    lower::TEST_HOST_WAVE.set(was);
    r
}

#[test]
fn lds_shared_inside_a_lane_branch_runs_with_the_branch_wave_wide() {
    let native = shared_in_a_branch(false);
    let shader = on_a_32_lane_host(&native, |c, p| {
        let st = structure::structure(p).expect("structure");
        let io = io::raise(p, &native).expect("io::raise");
        let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
        // as it stands, a write and a read in one lane branch keep the console's 64 lanes
        let (_, wave) = super::emit_program(c, &native, p, &st, &io, &res).expect("emit_program");
        assert_eq!(wave, Some(64));
        // ... and the form with that branch taken wave-wide, which has it in no loop, does not
        let (shader, wave) = super::widen(c, &native, p, false, super::Wide::Acyclic).expect("widen");
        assert_eq!(wave, Some(32));
        // which is the form the translation picks
        let picked = super::emit(c, &native, p, &st, &io, &res).expect("emit");
        assert_eq!(picked.dxil, shader.dxil);
        shader
    });
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        let mut copy = shader.dxil.clone();
        v.validate(&mut copy).expect("DXIL validation failed");
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    let (root, pso) = match host.compute_pipeline(&shader) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
            return;
        }
    };
    // (s0 threads take the branch; thread s1 writes): across the halves both ways, within
    // one half, a writer outside the branch, no thread in it
    for (below, writer) in [(64u32, 40u32), (64, 5), (64, 63), (64, 0), (40, 33), (40, 50), (20, 10), (33, 32), (0, 3)] {
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, d @ 0..=1) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = [below, writer][d as usize];
            }
        }
        for _ in 0..4 {
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud.clone()))],
                        views: &[View::RawUav(128)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
            let shared = if writer < below { writer + 7 } else { 0 };
            let want: Vec<u32> = (0..128u32)
                .map(|d| match d {
                    t @ 0..64 if t < below => shared,
                    0..64 => 0,
                    _ => u32::from(below > 0),
                })
                .collect();
            assert_eq!(out[0], want, "{below} threads in the branch, thread {writer} writing");
        }
    }
    eprintln!("ran on the hardware device: every thread of the branch read what the other half wrote");
}

#[test]
fn a_wave_wide_branch_in_a_loop_is_taken_only_when_asked_for() {
    let native = shared_in_a_branch(true);
    on_a_32_lane_host(&native, |c, p| {
        let acyclic = super::widen(c, &native, p, false, super::Wide::Acyclic);
        assert!(
            matches!(&acyclic, Err(ShaderError::Unsupported { what, .. }) if *what == super::IN_LOOP),
            "{:?}",
            acyclic.map(|(_, wave)| wave)
        );
        let any = super::widen(c, &native, p, false, super::Wide::Any).map(|(_, wave)| wave);
        assert!(matches!(any, Ok(Some(32))), "{any:?}");
    });
}

/// 64 threads in one console wave; the threads below s0 (a root constant) go on, the others
/// leave. Those left reserve one slot each, as the title's append passes do
/// (`indirect_light_probes:CS_Primary_RegisterAllocationRequests`): the wave's first lane adds
/// the wave's count to dword 0 of the raw UAV, and each lane stores its thread number plus one
/// at dword 1 + (what that returned + the lanes before it).
fn append_program() -> NativeProgram {
    let body = [
        0xBE8C_047E, // s_mov_b64 s[12:13], exec
        0xD765_0001,
        0x0001_000C, // v_mbcnt_lo_u32_b32 v1, s12, 0
        0xD766_0001,
        0x0002_020D, // v_mbcnt_hi_u32_b32 v1, s13, v1
        0xBE8E_100C, // s_bcnt1_i32_b64 s14, s[12:13]
        0xBE8F_140C, // s_ff1_i32_b64 s15, s[12:13]
        0xD765_0002,
        0x0001_00C1, // v_mbcnt_lo_u32_b32 v2, -1, 0
        0xD766_0002,
        0x0002_04C1, // v_mbcnt_hi_u32_b32 v2, -1, v2
        0x7E06_020E, // v_mov_b32 v3, s14
        0xBE90_047E, // s_mov_b64 s[16:17], exec
        0x7DA4_040F, // v_cmpx_eq_u32 exec, s15, v2
        0x38 << 26 | 0x32 << 18 | 1 << 14,
        0x80 << 24 | 2 << 16 | 3 << 8, // buffer_atomic_add v3, off, s[8:11], 0 glc
        0xBEFE_0410, // s_mov_b64 exec, s[16:17]
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x7E24_0503, // v_readfirstlane_b32 s18, v3
        0x4A08_0212, // v_add_nc_u32 v4, s18, v1
        0x3408_0882, // v_lshlrev_b32 v4, 2, v4
        0x4A0A_0081, // v_add_nc_u32 v5, 1, v0
        0x38 << 26 | 0x1C << 18 | 1 << 12 | 4,
        0x80 << 24 | 2 << 16 | 5 << 8 | 4, // buffer_store_dword v5, v4, s[8:11], 0 offen offset:4
    ];
    let mut words = vec![
        0x7DA8_0000,                     // v_cmpx_gt_u32 exec, s0, v0
        0xBF88_0000 | body.len() as u32, // s_cbranch_execz (the end)
    ];
    words.extend_from_slice(&body);
    native(&words)
}

#[test]
fn an_append_with_an_early_out_takes_each_host_wave_on_its_own_when_asked_first() {
    let native = append_program();
    let names: Vec<String> = crate::gfx10::decode(&native.code)
        .expect("decode")
        .iter()
        .map(|i| i.text(true))
        .collect();
    for (at, want) in [(12, "buffer_atomic_add v3"), (19, "buffer_store_dword v5, v4"), (20, "s_endpgm")] {
        assert!(names.get(at + 2).is_some_and(|n| n.starts_with(want)), "{at}: {names:?}");
    }
    let (off, on, local) = on_a_32_lane_host(&native, |c, p| {
        let st = structure::structure(p).expect("structure");
        let io = io::raise(p, &native).expect("io::raise");
        let res = resource::track(p, &native.layout, native.stage, Some(&st)).expect("resource::track");
        // as it stands, the idiom lies where not every lane runs
        let as_is = super::emit_program(c, &native, p, &st, &io, &res).map(|(_, wave)| wave);
        assert!(
            matches!(&as_is, Err(ShaderError::Unsupported { what, .. }) if lower::needs_every_lane(what)),
            "{as_is:?}"
        );
        let pick = |first: bool| {
            let was = super::TEST_LOCAL_FIRST.replace(Some(first));
            let r = super::emit(c, &native, p, &st, &io, &res).expect("emit");
            super::TEST_LOCAL_FIRST.set(was);
            r
        };
        // the form without the early-out, each host wave its own wave
        let q = ir::without_early_outs(p).expect("an early-out to leave out");
        let qst = structure::structure(&q).expect("structure");
        let qio = io::raise(&q, &native).expect("io::raise");
        let qres = resource::track(&q, &native.layout, native.stage, Some(&qst)).expect("resource::track");
        let (local, wave) = lower::with_local_waves(|| super::emit_program(c, &native, &q, &qst, &qio, &qres))
            .expect("local waves");
        assert_eq!(wave, Some(32));
        (pick(false), pick(true), local)
    });
    assert_eq!(on.dxil, local.dxil, "the switch picks the local form");
    assert_ne!(off.dxil, on.dxil, "without it another form is picked");
    for shader in [&off, &on] {
        if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
            let mut copy = shader.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
    }
    let _one = super::valu_tests::ONE_AT_A_TIME.lock().unwrap_or_else(|e| e.into_inner());
    let Ok(host) = Host::new(false) else {
        eprintln!("no SM 6.6 hardware device; not run");
        return;
    };
    for (form, shader) in [("exchanging", &off), ("local", &on)] {
        let (root, pso) = match host.compute_pipeline(shader) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("the hardware device does not take this program (waves of 32?): {e}; not run");
                return;
            }
        };
        for below in [64u32, 40, 33, 32, 20, 1, 0] {
            let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
            for u in &shader.user_data {
                if let (RootType::Constants, 0, 0) = (u.root, u.slot, u.dword) {
                    ud[u.index as usize] = below;
                }
            }
            let out = host
                .dispatch(
                    &root,
                    &pso,
                    &Dispatch {
                        roots: vec![(0, Root::Constants(ud))],
                        views: &[View::RawUav(72)],
                        samplers: &[],
                        groups: [1, 1, 1],
                    },
                )
                .unwrap_or_else(|e| panic!("dispatch failed: {e}"));
            // a slot for each thread that went on, each thread's number once
            let mut slots = out[0][1..=below as usize].to_vec();
            slots.sort_unstable();
            assert_eq!(out[0][0], below, "{form}, {below} threads: the count");
            assert_eq!(slots, (1..=below).collect::<Vec<_>>(), "{form}, {below} threads: the slots");
            assert!(out[0][below as usize + 1..].iter().all(|&d| d == 0), "{form}, {below} threads: past the slots");
        }
    }
    eprintln!("ran on the hardware device: both forms reserve one slot per thread");
}
