//! End-to-end tests of buffers a program picks from the whole view heap at
//! an index it computes (`RootType::ResourceHeap`): synthetic GFX10
//! programs lowered to DXIL, validated by `dxil.dll` when present, and run
//! on the host GPU and WARP with the host's table of strides by slot
//! ([`SLOT_STRIDES`]) bound, as it is when the title runs. What the table
//! says the picked slot holds decides how it is read.

use super::host::{Dispatch, Host, Root, Stride, View};
use super::{HostShader, SLOT_STRIDES, SLOT_STRIDES_SKIP, lower, shader, validator};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const SKIP: u32 = SLOT_STRIDES_SKIP;

/// The heap slot the tests' host heap keeps null, as a host heap keeps its
/// last one: the view-heap limit the programs are translated with. The
/// three slots below it hold the null views of a buffer SRV, a buffer UAV
/// and a constant buffer, as a host heap's do.
const NULL_SLOT: u32 = 6;

/// What the three slots below [`NULL_SLOT`] hold.
#[derive(Clone, Copy)]
enum Below {
    /// The null views a host heap keeps there.
    Null,
    /// Views that show which one a program reached: [`CBV_MARK`], a raw
    /// UAV of one dword (the second UAV of the heap) and [`SRV_MARK`].
    Marked,
}

/// The data of the buffer SRV one slot below [`NULL_SLOT`] ([`Below::Marked`]).
const SRV_MARK: [u32; 4] = [0xB0F0_0001, 0xB0F0_0002, 0xB0F0_0003, 0xB0F0_0004];

/// The data of the constant buffer three slots below [`NULL_SLOT`] ([`Below::Marked`]).
const CBV_MARK: [u32; 4] = [0xC0B0_0001, 0xC0B0_0002, 0xC0B0_0003, 0xC0B0_0004];

/// The picked slot's descriptor: `s_lshl_b32 s8, s0, 5` (the slot's byte
/// offset, records of 32 bytes), `s_mov_b32 s7, 0`, `s_load_dwordx4
/// s[12:15], s[6:7], s8` (the view heap at s6).
const PICK: [u32; 4] = [0x8F08_8500, 0xBE87_0380, 0xF408_0303, 0x1000_0000];

/// v0 to dword 0 of the raw UAV at entry 0 of the table at s4: `s_mov_b32
/// s5, 0`, `s_load_dwordx4 s[8:11], s[4:5], 0`, `buffer_store_dword v0,
/// off, s[8:11], 0`, `s_endpgm`.
const STORE: [u32; 6] = [
    0xBE85_0380,
    0xF408_0202,
    0xFA00_0000,
    0xE070_0000,
    0x8002_0000,
    0xBF81_0000,
];

/// The picked slot's first dword through the scalar cache:
/// `s_buffer_load_dword s16, s[12:15], 0`, `v_mov_b32 v0, s16`.
const SCALAR: [u32; 3] = [0xF420_0406, 0xFA00_0000, 0x7E00_0210];

/// The picked slot's first four dwords through the scalar cache, the last
/// kept: `s_buffer_load_dwordx4 s[16:19], s[12:15], 0`, `v_mov_b32 v0, s19`.
const SCALAR4: [u32; 3] = [0xF428_0406, 0xFA00_0000, 0x7E00_0213];

/// Element s1 of the picked slot per lane: `v_mov_b32 v1, s1`,
/// `buffer_load_dword v0, v1, s[12:15], 0 idxen`.
const VECTOR: [u32; 3] = [0x7E02_0201, 0xE030_2000, 0x8003_0001];

/// s1 stored at dword 0 of the picked slot, and the end: `v_mov_b32 v0,
/// s1`, `buffer_store_dword v0, off, s[12:15], 0`, `s_endpgm`.
const STORE_PICKED: [u32; 4] = [0x7E00_0201, 0xE070_0000, 0x8003_0000, 0xBF81_0000];

/// [`PICK`], `body` and [`STORE`].
fn read(body: &[u32]) -> Vec<u32> {
    [&PICK[..], body, &STORE].concat()
}

/// Lowers `words` as a wave32 compute program of one thread: root
/// constants at s0..s3 (s0 the slot to pick, s1 an element or a value),
/// the table of the result at s4 and the view heap at s6, translated for a
/// host heap whose null slot is [`NULL_SLOT`].
fn build(words: &[u32]) -> HostShader {
    let param = |root, user_data, sgprs, slot| RootParam {
        root,
        table_offset_dwords: 0,
        sgprs,
        base_register: 0,
        user_data,
        slot,
    };
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 8,
        layout: vec![
            param(RootType::Constants, 0, 4, 0),
            param(RootType::Table, 4, 1, 1),
            param(RootType::ResourceHeap, 6, 1, 2),
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
            rsrc2: 7 << 1, // user SGPRs s0..s6
            threads: [1, 1, 1],
        }),
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("track");
    let io = io::raise(&p, &native).expect("io::raise");
    let was = shader::TEST_VIEW_LIMIT.replace(Some(NULL_SLOT));
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
    });
    shader::TEST_VIEW_LIMIT.set(was);
    let lowered = lowered.expect("lower");
    assert_eq!(
        lowered.root_srvs,
        [SLOT_STRIDES],
        "the stride table is the one root view"
    );
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
        half_views: lowered.half_views,
    };
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        v.validate(&mut shader.dxil.clone())
            .expect("DXIL validation failed");
    } else {
        eprintln!("no dxil.dll; validation skipped");
    }
    shader
}

/// Runs `shader` with s0 `pick` and s1 `s1` over a heap of the result's
/// raw UAV, a raw buffer, a constant buffer, the three slots `below` says
/// and the null view, with `strides` as the host's table for the first
/// three slots (the others say they hold no buffer view), on the hardware
/// GPU and on WARP (each skipped without a device). Returns dword 0 of
/// each UAV of the heap, in slot order, after each.
fn run(
    shader: &HostShader,
    strides: [u32; 3],
    pick: u32,
    s1: u32,
    below: Below,
) -> Vec<(&'static str, Vec<u32>)> {
    let buffer = [0x1234_5678, 0x9ABC_DEF0, 0x0BAD_F00D, 0x7777_7777];
    let constants = [0xCAFE_0001, 0xCAFE_0002, 0xCAFE_0003, 0xCAFE_0004];
    let [cbv, uav, srv] = match below {
        Below::Null => [View::NullCbv, View::NullUav, View::NullBuffer],
        Below::Marked => [
            View::Cbv(&CBV_MARK),
            View::RawUav(1),
            View::Buffer {
                data: &SRV_MARK,
                stride: Stride::Raw,
            },
        ],
    };
    let views = [
        View::RawUav(1),
        View::Buffer {
            data: &buffer,
            stride: Stride::Raw,
        },
        View::Cbv(&constants),
        cbv,
        uav,
        srv,
        View::Null,
    ];
    assert_eq!(views.len() as u32, NULL_SLOT + 1);
    let mut table = vec![SKIP; views.len()];
    table[..3].copy_from_slice(&strides);
    let mut out = Vec::new();
    let _one = super::valu_tests::ONE_AT_A_TIME
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    for (label, warp) in [("hardware", false), ("warp", true)] {
        let Ok(host) = Host::new(warp) else {
            eprintln!("no SM 6.6 {label} device; skipped");
            continue;
        };
        let (root, pso) = host
            .compute_pipeline(shader)
            .unwrap_or_else(|e| panic!("{label}: compute_pipeline failed: {e}"));
        let mut ud = vec![0u32; shader.user_data_dwords().max(1) as usize];
        for u in &shader.user_data {
            ud[u.index as usize] = match (u.root, u.dword) {
                (RootType::Constants, 0) => pick,
                (RootType::Constants, 1) => s1,
                // The result's table starts at heap index 0.
                (RootType::Table, _) => 0,
                other => panic!("unexpected user data {other:?}"),
            };
        }
        let words = host
            .dispatch(
                &root,
                &pso,
                &Dispatch {
                    // The root views follow the user data.
                    roots: vec![
                        (0, Root::Constants(ud)),
                        (
                            1,
                            Root::View {
                                srv: true,
                                data: table.clone(),
                            },
                        ),
                    ],
                    views: &views,
                    samplers: &[],
                    groups: [1, 1, 1],
                },
            )
            .unwrap_or_else(|e| panic!("{label}: dispatch failed: {e}"));
        out.push((label, words.iter().map(|w| w[0]).collect()));
    }
    assert!(!out.is_empty(), "no SM 6.6 device");
    out
}

/// A scalar read where the table says the slot holds no buffer view reads
/// nothing at all, not even the null views below the null slot.
#[test]
fn a_scalar_read_through_a_slot_marked_skip_gives_zero() {
    let shader = build(&read(&SCALAR));
    for (label, got) in run(&shader, [SKIP, SKIP, 0], 1, 0, Below::Marked) {
        assert_eq!(got[0], 0, "{label}: {:#x}", got[0]);
    }
}

/// The same for a load of several dwords (the culls read their records so),
/// however the translator joins the dwords' reads.
#[test]
fn a_wide_scalar_read_through_a_slot_marked_skip_gives_zero() {
    let shader = build(&read(&SCALAR4));
    for (label, got) in run(&shader, [SKIP, SKIP, 0], 1, 0, Below::Marked) {
        assert_eq!(got[0], 0, "{label}: {:#x}", got[0]);
    }
}

#[test]
fn a_wide_scalar_read_through_a_buffer_slot_gives_its_fourth_dword() {
    let shader = build(&read(&SCALAR4));
    for (label, got) in run(&shader, [SKIP, 1, 0], 1, 0, Below::Null) {
        assert_eq!(got[0], 0x7777_7777, "{label}: {:#x}", got[0]);
    }
}

#[test]
fn a_scalar_read_through_a_buffer_slot_gives_its_data() {
    let shader = build(&read(&SCALAR));
    for (label, got) in run(&shader, [SKIP, 1, 0], 1, 0, Below::Null) {
        assert_eq!(got[0], 0x1234_5678, "{label}: {:#x}", got[0]);
    }
}

#[test]
fn a_scalar_read_through_a_constant_buffer_slot_gives_its_data() {
    let shader = build(&read(&SCALAR));
    for (label, got) in run(&shader, [SKIP, 1, 0], 2, 0, Below::Null) {
        assert_eq!(got[0], 0xCAFE_0001, "{label}: {:#x}", got[0]);
    }
}

/// The null buffer view a host heap keeps below its null slot reads zero.
#[test]
fn a_vector_read_through_a_slot_marked_skip_gives_zero() {
    let shader = build(&read(&VECTOR));
    for (label, got) in run(&shader, [SKIP, SKIP, 0], 1, 0, Below::Null) {
        assert_eq!(got[0], 0, "{label}: {:#x}", got[0]);
    }
}

/// A buffer read where the table says the slot holds no buffer view reads
/// the buffer SRV one slot below the null slot, not the null slot's 2D
/// view: a view of the access's own class.
#[test]
fn a_vector_read_through_a_slot_marked_skip_reads_the_buffer_view_below_the_null_slot() {
    let shader = build(&read(&VECTOR));
    for (label, got) in run(&shader, [SKIP, SKIP, 0], 1, 0, Below::Marked) {
        assert_eq!(got[0], SRV_MARK[0], "{label}: {:#x}", got[0]);
    }
}

#[test]
fn a_vector_read_through_a_buffer_slot_reads_its_element() {
    let shader = build(&read(&VECTOR));
    for (label, got) in run(&shader, [SKIP, 4, 0], 1, 1, Below::Null) {
        assert_eq!(got[0], 0x9ABC_DEF0, "{label}: {:#x}", got[0]);
    }
}

/// A write through a slot marked SKIP goes to the null UAV below the null
/// slot, where it is dropped.
#[test]
fn a_store_through_a_slot_marked_skip_is_dropped() {
    let shader = build(&[&PICK[..], &STORE_PICKED].concat());
    // Slot 0, the result's raw UAV, picked.
    for (label, got) in run(&shader, [SKIP, SKIP, 0], 0, 0x5EED_0001, Below::Null) {
        assert_eq!(got, [0], "{label}: {got:x?}");
    }
}

/// The write lands in the UAV two slots below the null slot, not in the
/// picked slot's view nor the null slot's 2D view.
#[test]
fn a_store_through_a_slot_marked_skip_reaches_the_unordered_view_below_the_null_slot() {
    let shader = build(&[&PICK[..], &STORE_PICKED].concat());
    for (label, got) in run(&shader, [SKIP, SKIP, 0], 0, 0x5EED_0001, Below::Marked) {
        assert_eq!(got, [0, 0x5EED_0001], "{label}: {got:x?}");
    }
}

#[test]
fn a_store_through_a_buffer_slot_lands() {
    let shader = build(&[&PICK[..], &STORE_PICKED].concat());
    for (label, got) in run(&shader, [1, SKIP, 0], 0, 0x5EED_0001, Below::Null) {
        assert_eq!(got, [0x5EED_0001], "{label}: {got:x?}");
    }
}
