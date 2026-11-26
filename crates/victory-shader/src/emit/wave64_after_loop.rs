// ISA encoders from fork PR #993 and loop fixture from darkworldhm 08fdf0b.
use super::*;

#[test]
fn lane_loop_exit_includes_repeated_postdominators() {
    use crate::structure::{Atom, LaneExit};
    let native = super::program(0xBF88_0000);
    let ir = crate::spill::build(&native).unwrap();
    let mut st = structure::structure(&ir).unwrap();
    let when = st
        .exits
        .iter()
        .find_map(|exit| match exit {
            LaneExit::Branch { when, .. }
                if st
                    .bools
                    .support(*when)
                    .iter()
                    .any(|a| !matches!(a, Atom::Uniform(_))) =>
            {
                Some(*when)
            }
            _ => None,
        })
        .expect("the fixture supplies a per-lane condition");
    st.wide.clear();
    st.exits = vec![
        LaneExit::Jump(1),
        LaneExit::Jump(2),
        LaneExit::Branch {
            when,
            taken: 1,
            not_taken: 3,
        },
        LaneExit::End,
    ];
    st.wave_any = vec![None, None, Some(true), None];
    assert_eq!(lower::branches_around(&st, &[2], false), Some(vec![]));
    assert_eq!(lower::branches_around(&st, &[2], true), Some(vec![2]));
    assert_eq!(lower::branches_around(&st, &[3], true), Some(vec![]));
}

const EXEC: u32 = 126;
const LIT: u32 = 255;

const fn v(n: u32) -> u32 {
    256 + n
}

const fn k(n: u32) -> u32 {
    128 + n
}

const fn sop1(op: u32, sdst: u32, src0: u32) -> u32 {
    0xBE80_0000 | sdst << 16 | op << 8 | src0
}

const fn sop2(op: u32, sdst: u32, src0: u32, src1: u32) -> u32 {
    0x8000_0000 | op << 23 | sdst << 16 | src1 << 8 | src0
}

/// A branch by `imm` dwords from the next instruction (or `s_barrier`, `imm` 0).
const fn sopp(op: u32, imm: i16) -> u32 {
    0xBF80_0000 | op << 16 | imm as u16 as u32
}

const fn vop1(op: u32, vdst: u32, src0: u32) -> u32 {
    0x7E00_0000 | vdst << 17 | op << 9 | src0
}

const fn vop2(op: u32, vdst: u32, src0: u32, vsrc1: u32) -> u32 {
    op << 25 | vdst << 17 | vsrc1 << 9 | src0
}

const fn vopc(op: u32, src0: u32, vsrc1: u32) -> u32 {
    0x7C00_0000 | op << 17 | vsrc1 << 9 | src0
}

const fn vop3(op: u32, dst: u32, src0: u32, src1: u32) -> [u32; 2] {
    [0xD400_0000 | op << 16 | dst, src0 | src1 << 9]
}

/// `buffer_store_dword v<data>, v<addr>, s[8:11], 0 offen offset:<offset>`
const fn store(data: u32, addr: u32, offset: u32) -> [u32; 2] {
    [0xE070_1000 | offset, 0x8002_0000 | data << 8 | addr]
}

/// `v<regs>` stored as each thread's dwords of the UAV, the thread's index (v0) times 16 on.
fn stores(regs: [u32; 4]) -> Vec<u32> {
    let mut w = vec![vop2(0x1A, 15, k(4), 0)]; // v_lshlrev_b32 v15, 4, v0
    for (n, r) in regs.into_iter().enumerate() {
        w.extend(store(r, 15, 4 * n as u32));
    }
    w
}

/// One console wave of 64 threads. Lane t goes round a loop `(t & s0) + 1` times, adding
/// `t & s1` to its sum each round; the loop is the compiler's usual one for lanes that leave
/// apart (a mask of the lanes still going round, the wave's branch back while some lane is
/// left). After it, over the lanes whose `t & s1` is not 0: their count (`s_bcnt1_i32_b64`),
/// the first of them (`v_readfirstlane_b32` of the thread index) and each one's number among
/// them (`v_mbcnt_lo/hi`). Every thread stores its sum, its number (0x55 where it has none),
/// the count and the first.
fn program() -> NativeProgram {
    let mut w = vec![
        vop2(0x1B, 4, 0, 0), // v_and_b32 v4, s0, v0: the lane's last round
        vop2(0x1B, 5, 1, 0), // v_and_b32 v5, s1, v0: what the lane adds a round
        vop1(0x01, 7, k(0)), // v_mov_b32 v7, 0
        vop1(0x01, 8, LIT),  // v_mov_b32 v8, 0x55
        0x55,
        sop1(0x03, 6, k(0)),      // s_mov_b32 s6, 0: the round
        sop1(0x04, 2, EXEC),      // s_mov_b64 s[2:3], exec
        sop1(0x04, 12, EXEC),     // s_mov_b64 s[12:13], exec: the lanes still going round
        sop1(0x04, 14, EXEC),     // s_mov_b64 s[14:15], exec
        sop2(0x0F, EXEC, 14, 12), // loop: s_and_b64 exec, s[14:15], s[12:13]
        vop2(0x25, 7, v(5), 7),   // v_add_nc_u32 v7, v5, v7
        sop1(0x04, EXEC, 14),     // s_mov_b64 exec, s[14:15]
        sop2(0x02, 6, 6, k(1)),   // s_add_i32 s6, s6, 1
    ];
    w.extend(vop3(0xC4, 14, 6, v(4))); // v_cmp_gt_u32 s[14:15], s6, v4: the lanes that are done
    w.extend([
        sop1(0x24, 14, 14),       // s_and_saveexec_b64 s[14:15], s[14:15]
        sop2(0x15, 12, 12, EXEC), // s_andn2_b64 s[12:13], s[12:13], exec
        sopp(0x05, -9),           // s_cbranch_scc1 loop
        sop1(0x04, EXEC, 2),      // s_mov_b64 exec, s[2:3]
        vopc(0xD5, k(0), 5),      // v_cmpx_ne_u32 exec, 0, v5
        sop1(0x10, 7, EXEC),      // s_bcnt1_i32_b64 s7, exec
        vop1(0x02, 16, v(0)),     // v_readfirstlane_b32 s16, v0
    ]);
    w.extend(vop3(0x365, 8, EXEC, k(0))); // v_mbcnt_lo_u32_b32 v8, exec_lo, 0
    w.extend(vop3(0x366, 8, EXEC + 1, v(8))); // v_mbcnt_hi_u32_b32 v8, exec_hi, v8
    w.extend([
        sop1(0x04, EXEC, 2), // s_mov_b64 exec, s[2:3]
        vop1(0x01, 9, 7),    // v_mov_b32 v9, s7
        vop1(0x01, 10, 16),  // v_mov_b32 v10, s16
    ]);
    w.extend(stores([7, 8, 9, 10]));
    native(&w)
}

/// What the console's 64 lanes store with the root constants `s0` and `s1`.
fn console(s0: u32, s1: u32) -> Vec<u32> {
    let counted: Vec<u32> = (0..64).filter(|t| t & s1 != 0).collect();
    (0..64u32)
        .flat_map(|t| {
            let number = if t & s1 != 0 {
                counted.iter().filter(|&&u| u < t).count() as u32
            } else {
                0x55
            };
            [
                (t & s1) * ((t & s0) + 1),
                number,
                counted.len() as u32,
                counted[0],
            ]
        })
        .collect()
}

#[test]
#[ignore = "requires a physical SM 6.6 GPU with 32-lane waves and dxil.dll"]
fn nvidia_after_divergent_loop_matches_cpu() {
    let cases = [(0, 1), (3, 1), (7, 5), (24, 63), (7, 32), (7, 16), (63, 9)];
    execute_wave32(&program(), &cases, console);
}

#[test]
#[ignore = "requires a physical SM 6.6 GPU with 32-lane waves and dxil.dll"]
fn nvidia_lds_between_wave_halves_matches_cpu() {
    let words = [
        vop2(0x1a, 1, k(2), 0), // v1 = byte address of this lane
        vop2(0x25, 2, k(1), 0), // v2 = thread index + 1
        0xd834_0000,
        0x0000_0201,             // ds_write_b32 v1, v2
        vop2(0x1d, 3, k(32), 0), // v3 = partner lane
        vop2(0x1a, 3, k(2), 3),
        0xd8d8_0000,
        0x0400_0003, // ds_read_b32 v4, v3, no explicit s_barrier
    ];
    let mut body = words.to_vec();
    body.extend(vop3(0x365, 5, 193, k(0))); // v_mbcnt_lo
    body.extend(vop3(0x366, 5, 193, v(5))); // v_mbcnt_hi
    body.extend(stores([4, 5, 2, 0]));
    let mut native = native(&body);
    if let StageIo::Cs(cs) = &mut native.io {
        cs.rsrc2 |= 1 << 15; // one 512-byte LDS unit
    }
    execute_wave32(&native, &[(0, 0)], |_, _| {
        (0..64)
            .flat_map(|lane| [(lane ^ 32) + 1, lane, lane + 1, lane])
            .collect()
    });
}

fn execute_wave32(
    native: &NativeProgram,
    cases: &[(u32, u32)],
    expected: impl Fn(u32, u32) -> Vec<u32>,
) {
    let _one = crate::emit::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    use windows::Win32::Graphics::Direct3D12::*;
    let host = Host::new(false).expect("physical SM 6.6 hardware adapter required");
    let mut caps = D3D12_FEATURE_DATA_D3D12_OPTIONS1::default();
    // SAFETY: the query writes the options structure of the supplied size.
    unsafe {
        host.device
            .CheckFeatureSupport(
                D3D12_FEATURE_D3D12_OPTIONS1,
                (&raw mut caps).cast(),
                size_of_val(&caps) as u32,
            )
            .unwrap();
    }
    let lanes = caps.WaveLaneCountMax;
    assert_eq!(lanes, 32, "this test exercises the NVIDIA path");
    let p = crate::spill::build(native).expect("build native shader");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resources");
    let io = io::raise(&p, native).expect("I/O");
    let bytes = crate::dxbc::build(&[]);
    let c = crate::dxbc::Container::parse(&bytes).unwrap();
    let was = lower::TEST_HOST_WAVE.replace(Some(lanes));
    let result = crate::emit::emit(&c, native, &p, &st, &io, &res);
    lower::TEST_HOST_WAVE.set(was);
    let mut shader = result.expect("production NVIDIA lowering");
    validator::Validator::load(validator::Validator::default_path().as_deref())
        .expect("dxil.dll required")
        .validate(&mut shader.dxil)
        .expect("DXIL validation");
    let (root, pso) = host
        .compute_pipeline(&shader)
        .expect("RTX compute pipeline");
    for &(s0, s1) in cases {
        let mut ud = vec![0; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            if let (RootType::Constants, 0, d @ 0..=3) = (u.root, u.slot, u.dword) {
                ud[u.index as usize] = [s0, s1, 0, 0][d as usize];
            }
        }
        let got = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    roots: vec![(0, Root::Constants(ud))],
                    views: &[View::RawUav(256)],
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .expect("dispatch");
        assert_eq!(got[0], expected(s0, s1), "s0={s0}, s1={s1}");
    }
    host.check_hardware_validation();
    eprintln!(
        "32-lane hardware: validated {}-byte DXIL, {} CPU comparisons passed",
        shader.dxil.len(),
        cases.len()
    );
}
