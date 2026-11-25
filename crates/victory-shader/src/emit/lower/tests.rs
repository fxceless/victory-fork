use super::{Lowered, Numeric, Parts, buffer_format, half_shift, lower};
use crate::native::{
    ComputeIo, HwStage, NativeProgram, ResourceType, RootParam, RootType, StageIo, UsedResource,
};
use crate::{io, ir, resource, structure};

const S_ENDPGM: u32 = 0xBF81_0000;

#[test]
fn packed_pixel_inputs_find_their_second_row() {
    use super::super::container::SigElement;
    use super::half_input_partners;
    use crate::native::{PsInput, Usage};
    let entry = |row, row2, pack2| PsInput {
        usage: Usage::Generic,
        row,
        mask: 0xF,
        attr: row,
        flat: false,
        fp16: true,
        pack2,
        row2,
    };
    let element = |row| SigElement {
        name: "TEXCOORD".into(),
        index: row,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col: 0,
        cols: 4,
        usage: 0,
        half: true,
    };
    // Rows 0 and 3 pack and leave their second row out; 5 names its
    // own (6). Unnamed 16-bit rows: 1, 4, 6, 7.
    let io = StageIo::Ps {
        inputs: vec![entry(0, 0, true), entry(3, 3, true), entry(5, 6, true)],
        input_addr: 0,
        db_shader_control: 0,
        patches: vec![],
    };
    let sig: Vec<_> = [0, 1, 3, 4, 5, 6, 7].map(element).into();
    let p = half_input_partners(&io, &sig);
    assert_eq!(p.get(&0), Some(&1));
    assert_eq!(p.get(&3), Some(&4));
    assert_eq!(p.get(&5), None);
}

/// The packed pedestrian layouts of the title: an element
/// behind another one in its register follows in the first parameter
/// with a dword left, and is not a half of its register's row.
#[test]
fn packed_elements_behind_another_take_the_first_free_dword() {
    use super::super::container::SigElement;
    use super::packed_half;
    let element = |index, row, col, cols| SigElement {
        name: "TEXCOORD".into(),
        index,
        kind: 0,
        comp: 9,
        interp: 2,
        row: Some(row),
        col,
        cols,
        usage: 0,
        half: true,
    };
    // MS_VS_PedTransformPreSkinD_Packed 005fe88ec1bdc0c7 and its pixel
    // shader 93667197d3d05e1c: TEXCOORD6 is one value at register 4
    // column 3; parameter 3 (rows 6 and 7) carries it in dword 3.
    let sig = vec![
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
        element(6, 4, 3, 1),
        element(3, 5, 0, 4),
        element(5, 6, 0, 4),
        element(7, 7, 0, 2),
        element(9, 9, 0, 3),
        element(11, 10, 0, 3),
        element(15, 12, 0, 4),
    ];
    let packs = [
        (2, Some(3)),
        (4, Some(5)),
        (6, Some(7)),
        (9, Some(10)),
        (12, None),
    ];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    // Parameter 2: TEXCOORD4's three values, a zero half, TEXCOORD3.
    assert_eq!(at(4, 2), Some((4, 2)));
    assert_eq!(at(4, 3), None);
    assert_eq!(at(4, 4), Some((5, 0)));
    assert_eq!(at(4, 7), Some((5, 3)));
    // Parameter 3: TEXCOORD5, TEXCOORD7, then TEXCOORD6 alone.
    assert_eq!(at(6, 3), Some((6, 3)));
    assert_eq!(at(6, 5), Some((7, 1)));
    assert_eq!(at(6, 6), Some((4, 3)));
    assert_eq!(at(6, 7), None);
    // Parameters without such an element are as before.
    assert_eq!(at(9, 3), None);
    assert_eq!(at(9, 4), Some((10, 0)));
    assert_eq!(at(12, 3), Some((12, 3)));
    assert_eq!(at(12, 4), None);
    // The common layout (29805d23a495cdf8): TEXCOORD7 is two values at
    // register 1 columns 2-3; parameter 0 is TEXCOORD0, TEXCOORD1,
    // TEXCOORD7.
    let sig = vec![
        element(0, 1, 0, 2),
        element(7, 1, 2, 2),
        element(1, 2, 0, 4),
        element(2, 3, 0, 4),
        element(4, 4, 0, 3),
    ];
    let packs = [(3, Some(4)), (1, Some(2))];
    let at = |a, h| packed_half(&sig, &packs, a, h);
    assert_eq!(at(1, 1), Some((1, 1)));
    assert_eq!(at(1, 2), Some((2, 0)));
    assert_eq!(at(1, 5), Some((2, 3)));
    assert_eq!(at(1, 6), Some((1, 2)));
    assert_eq!(at(1, 7), Some((1, 3)));
    assert_eq!(at(3, 6), Some((4, 2)));
    assert_eq!(at(3, 7), None);
    // A row no element holds gives nothing, as before.
    assert_eq!(packed_half(&sig, &[(8, None)], 8, 0), None);
}

/// Whether a synthetic compute program translates with these input registers.
fn lowers(words: &[u32], rsrc2: u32) -> Result<(), crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, false, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .map(|_| ())
}

/// Lowers a compute program with the root parameters `layout` (in user
/// SGPRs from s0 on) and the used resources `used`.
fn lower_with(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    lower_wave(words, layout, used, false)
}

/// [`lower_with`] in wave64 if `wave64`.
fn lower_wave(
    words: &[u32],
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
    wave64: bool,
) -> Result<Lowered, crate::ShaderError> {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let sgprs: u32 = layout.iter().map(|p| p.sgprs).sum();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64,
        vgprs: 256,
        layout,
        used,
        io: StageIo::Cs(ComputeIo {
            // The layout's user SGPRs (at least two), 512 bytes of LDS.
            rsrc2: 1 << 15 | sgprs.max(2) << 1,
            threads: [1, 1, 1],
        }),
        code,
    };
    let p = crate::spill::build(&native).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
}

/// Root parameter `slot` in user SGPRs from `user_data` on.
fn root(root: RootType, user_data: u8, slot: u8) -> RootParam {
    RootParam {
        root,
        table_offset_dwords: 0,
        sgprs: if root == RootType::Descriptor { 2 } else { 1 },
        base_register: 0,
        user_data,
        slot,
    }
}

/// Lowers a one-thread compute program that copies four root constants
/// (s0..s3) into v1..v4, runs `body`, and stores v0 to a raw buffer from a
/// descriptor table, so the vector instructions in `body` have an effect.
fn lowers_valu(body: &[u32]) -> Result<(), crate::ShaderError> {
    lower_valu(body).map(|_| ())
}

/// [`lowers_valu`]'s program, lowered.
fn lower_valu(body: &[u32]) -> Result<Lowered, crate::ShaderError> {
    let mut words = vec![
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202, // s_load_dwordx4 s[8:11], s[4:5], 0
        0xFA00_0000,
    ];
    // v_mov_b32 v1..v4, s0..s3
    words.extend((0..4u32).map(|i| 0x7E00_0200 | (i + 1) << 17 | i));
    words.extend_from_slice(body);
    words.extend_from_slice(&[
        0xE070_0000, // buffer_store_dword v0, off, s[8:11], 0
        0x8002_0000,
        S_ENDPGM,
    ]);
    let constants = RootParam {
        sgprs: 4,
        ..root(RootType::Constants, 0, 0)
    };
    let buffer = UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 4,
        slot: 1,
    };
    lower_with(
        &words,
        vec![constants, root(RootType::Table, 4, 1)],
        vec![buffer],
    )
}

/// Whether the lowered program declares wave operations (`SFI0` 0x4000), which the clean-up
/// drops with the last one it keeps.
fn has_wave_ops(l: &Lowered) -> bool {
    let container = crate::Container::parse(&l.dxil).unwrap();
    let features = container.part(crate::FourCC::SFI0).unwrap();
    u64::from_le_bytes(features.data.try_into().unwrap()) & 0x4000 != 0
}

#[test]
fn scalar_min_and_max_set_scc_from_the_compare() {
    // `op s12, 0, 1` or `op s12, 0, -1`, then `s_cselect_b32 s14` between the constant 2
    // (on the SCC the RDNA2 ISA gives: whether the first operand was chosen) and a wave
    // read (on the other), stored. With constant operands the right SCC leaves no wave
    // operation; `result != 0` would pick the read in every case here.
    let first = 0x850E_0D82; // s_cselect_b32 s14, 2, s13
    let second = 0x850E_820D; // s_cselect_b32 s14, s13, 2
    for (op, select) in [
        (0x838C_8180u32, first), // s_min_u32 s12, 0, 1: 0, SCC 1
        (0x848C_8180, second),   // s_max_u32 s12, 0, 1: 1, SCC 0
        (0x830C_C180, second),   // s_min_i32 s12, 0, -1: -1, SCC 0
        (0x840C_C180, first),    // s_max_i32 s12, 0, -1: 0, SCC 1
    ] {
        let body = [
            0x7E1A_0501, // v_readfirstlane_b32 s13, v1
            op,
            select,
            0x7E00_020E, // v_mov_b32 v0, s14
        ];
        let l = lower_valu(&body).unwrap_or_else(|e| panic!("{op:#x}: {e}"));
        assert!(!has_wave_ops(&l), "{op:#x}: the select took the wave read");
    }
}

#[test]
fn always_false_and_always_true_compares_are_constants() {
    // `v_cmp_{f,tru}_{f32,f16} vcc, v1, v2`, then `v_cndmask_b32 v0` between v2 (where the
    // constant mask points) and a wave read in v5. A host `fcmp false`/`fcmp true` stays a
    // compare (the WARP device is removed by one), so both sides would stay.
    let wave = [
        0x7E1A_0501, // v_readfirstlane_b32 s13, v1
        0x7E0A_020D, // v_mov_b32 v5, s13
    ];
    for (cmp, select) in [
        (0x7C00_0501u32, 0x0200_0B02u32), // v_cmp_f_f32; v_cndmask_b32 v0, v2, v5, vcc
        (0x7C1E_0501, 0x0200_0505),       // v_cmp_tru_f32; v_cndmask_b32 v0, v5, v2, vcc
        (0x7D90_0501, 0x0200_0B02),       // v_cmp_f_f16
        (0x7DDE_0501, 0x0200_0505),       // v_cmp_tru_f16
    ] {
        let l = lower_valu(&[wave[0], wave[1], cmp, select])
            .unwrap_or_else(|e| panic!("{cmp:#x}: {e}"));
        assert!(!has_wave_ops(&l), "{cmp:#x}: the compare is not a constant");
    }
}

#[test]
fn the_24_bit_multiplies_lower_both_halves() {
    // VOP2 `op v0, v1, v2`: v_mul_i32_i24 and v_mul_u32_u24 (the low
    // dword) and v_mul_hi_i32_i24 and v_mul_hi_u32_u24 (the high one).
    for op in [0x09u32, 0x0A, 0x0B, 0x0C] {
        let word = op << 25 | 2 << 9 | 0x101;
        lowers_valu(&[word]).unwrap_or_else(|e| panic!("VOP2 {op:#x}: {e}"));
    }
}

/// A register array indexed through M0 (v8-v11) below VGPRs that hold descriptor words
/// (v12-v15), which have no host value: clamped (`v_med3_u32`, `v_and_b32`), the moves
/// reach only the array and the program lowers; unclamped, the read could reach the
/// descriptor words and the program is refused.
#[test]
fn relative_moves_reach_only_what_their_clamped_index_can() {
    let body = |second_index: u32| {
        let mut w = Vec::new();
        for k in 0..4u32 {
            w.push(0x7E00_0200 | (8 + k) << 17 | (138 + k)); // v_mov_b32 v8+k, 10+k
            w.push(0x7E00_0200 | (12 + k) << 17 | (8 + k)); // v_mov_b32 v12+k, s8+k
        }
        w.extend([
            0xD559_0005, // v_med3_u32 v5, v1, 0, 3
            0x020D_0101,
            0x7E18_0505,  // v_readfirstlane_b32 s12, v5
            0xBEFC_030C,  // s_mov_b32 m0, s12
            0x7E10_8503,  // v_movreld_b32 v8, v3: v[8 + m0] = v3
            second_index, // v6 = the second index
            0x7E18_0506,  // v_readfirstlane_b32 s12, v6
            0xBEFC_030C,  // s_mov_b32 m0, s12
            0x7E00_8708,  // v_movrels_b32 v0, v8: v0 = v[8 + m0]
        ]);
        w
    };
    // v_and_b32 v6, 3, v2
    lowers_valu(&body(0x360C_0483)).unwrap();
    // v_mov_b32 v6, v2
    let refused = lowers_valu(&body(0x7E0C_0302));
    assert!(
        matches!(
            refused,
            Err(crate::ShaderError::Unsupported {
                what: "a descriptor's address or format read as data",
                ..
            })
        ),
        "{refused:?}"
    );
}

#[test]
fn buffer_cache_invalidations_lower_as_no_ops() {
    // MUBUF opcode 0x071/0x072 with no data operand: the title asks the
    // host to invalidate its cached view of a buffer.
    assert!(lowers(&[0xE1C4_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl0_inv");
    assert!(lowers(&[0xE1C8_0000, 0, S_ENDPGM], 0).is_ok(), "buffer_gl1_inv");
}

#[test]
fn sums_of_absolute_differences_lower() {
    // VOP3 `op v0, v1, v2, v3`: v_sad_u8, v_sad_hi_u8, v_sad_u16,
    // v_sad_u32 and v_msad_u8.
    for op in [0x15Au32, 0x15B, 0x15C, 0x15D, 0x171] {
        let words = [0xD400_0000 | op << 16, 0x101 | 0x102 << 9 | 0x103 << 18];
        lowers_valu(&words).unwrap_or_else(|e| panic!("VOP3 {op:#x}: {e}"));
    }
}

#[test]
fn a_scalar_carry_chain_reaches_the_select() {
    // The high dword of a 64-bit add or subtract, then a select on the
    // carry (borrow) the high half leaves in SCC, and a branch on it
    // (only code with an effect needs its operands).
    let add = [
        0xBE80_03C1, // s_mov_b32 s0, -1
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8000_8100, // s_add_u32 s0, s0, 1
        0x8201_8001, // s_addc_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];
    let sub = [
        0xBE80_0380, // s_mov_b32 s0, 0
        0xBE81_0380, // s_mov_b32 s1, 0
        0x8080_8100, // s_sub_u32 s0, s0, 1
        0x8281_8001, // s_subb_u32 s1, s1, 0
        0x8502_0001, // s_cselect_b32 s2, s1, s0
        0xBF07_8002, // s_cmp_lg_u32 s2, 0
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE83_0380, // s_mov_b32 s3, 0
        S_ENDPGM,
    ];

    for words in [add, sub] {
        lowers(&words, 0).unwrap();
    }
}

/// A console wave64 group on 32-lane host waves that do not exchange (the DPP move stays
/// within each half, so the waves run as halves): a constant lane mask in `v_cndmask_b32`
/// has no console lane numbers to pick its bits by there (both host waves would take the
/// low dword), so it is refused.
#[test]
fn a_constant_select_mask_on_host_halves_is_refused() {
    let inline = [
        0x7E02_02FAu32,
        0xFF01_1100, // v_mov_b32_dpp v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf
        0xD501_0002,
        0x0205_0301, // v_cndmask_b32 v2, v1, 1, 1
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    let held = [
        0xBE80_03FFu32,
        0x5555_5555, // s_mov_b32 s0, 0x55555555
        0xBE81_0380, // s_mov_b32 s1, 0
        0x7E02_02FA,
        0xFF01_1100, // v_mov_b32_dpp v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf
        0xD501_0002,
        0x0001_0301, // v_cndmask_b32 v2, v1, 1, s[0:1]
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    // A compare combined with a constant, or with a pair written apart: masks of their own
    // (`Structured::masks`), read through `atom`. And a constant pair in `Structured::other`.
    let with_constant = [
        0x7D8A_0080u32, // v_cmp_ne_u32 vcc, 0, v0
        0x8780_816A,    // s_and_b64 s[0:1], vcc, 1
        0x7E02_02FA,
        0xFF01_1100, // v_mov_b32_dpp v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf
        0xD501_0002,
        0x0001_0301, // v_cndmask_b32 v2, v1, 1, s[0:1]
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    let with_pair = [
        0xBE82_03FFu32,
        0x5555_5555, // s_mov_b32 s2, 0x55555555
        0xBE83_0380, // s_mov_b32 s3, 0
        0x7D8A_0080, // v_cmp_ne_u32 vcc, 0, v0
        0x8780_026A, // s_and_b64 s[0:1], vcc, s[2:3]
        0x7E02_02FA,
        0xFF01_1100, // v_mov_b32_dpp v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf
        0xD501_0002,
        0x0001_0301, // v_cndmask_b32 v2, v1, 1, s[0:1]
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    let constant_pair = [
        0xBE80_0481u32, // s_mov_b64 s[0:1], 1
        0x7E02_02FA,
        0xFF01_1100, // v_mov_b32_dpp v1, v0 row_shr:1 row_mask:0xf bank_mask:0xf
        0xD501_0002,
        0x0001_0301, // v_cndmask_b32 v2, v1, 1, s[0:1]
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    for words in [
        &inline[..],
        &held[..],
        &with_constant[..],
        &with_pair[..],
        &constant_pair[..],
    ] {
        a_constant_select_mask_on_host_halves(words);
    }
}

fn a_constant_select_mask_on_host_halves(words: &[u32]) {
    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: true,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15 | 2 << 1,
            threads: [64, 1, 1],
        }),
        code,
    };
    let p = ir::build(&native.code, true, 256).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();
    let io = io::raise(&p, &native).unwrap();
    let was = super::TEST_HOST_WAVE.replace(Some(32));
    let lowered = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    });
    super::TEST_HOST_WAVE.set(was);
    let Err(e) = lowered else {
        panic!("a constant select mask on host halves was translated");
    };
    assert!(e.to_string().contains("lane mask in a half-wave"), "{e}");
}

#[test]
fn v_cndmask_accepts_uniform_and_constant_masks() {
    // Uniform select (`s_cselect_b32`) and inline constant masks in `v_cndmask_b32`
    // (resolved in `Structured::other` or directly as constants).
    let words = [
        0xBF06_8000u32, // s_cmp_eq_u32 s0, 0
        0x8500_80C1,    // s_cselect_b32 s0, -1, 0
        0xD501_0001,
        0x0001_0280, // v_cndmask_b32 v1, 0, 1, s0
        0xD501_0002,
        0x0201_0301, // v_cndmask_b32 v2, v1, 1, 0
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    // (TGID_X_EN: s0 starts as the group id.)
    lowers(&words, 0x8081).unwrap();
    // Another constant names lanes: refused where host lanes are not the console's,
    // inline or in a register.
    let mut inline = words;
    inline[5] = 0x0205_0301; // v_cndmask_b32 v2, v1, 1, 1
    let e = lowers(&inline, 0x8081).unwrap_err();
    assert!(e.to_string().contains("a constant lane mask"), "{e}");
    let held = [
        0xBE80_03FFu32,
        0x5555_5555, // s_mov_b32 s0, 0x55555555
        0xD501_0002,
        0x0001_0280, // v_cndmask_b32 v2, 0, 1, s0
        0xD834_0000,
        0x0000_0200, // ds_write_b32 v0, v2
        S_ENDPGM,
    ];
    let e = lowers(&held, 0x8001).unwrap_err();
    assert!(e.to_string().contains("a constant lane mask"), "{e}");
}

fn perm_features(selector: u32) -> u64 {
    // Synthetic compute shader:
    // v0 is the thread ID and can supply a dynamic selector.
    // Store the result in LDS to keep the permutation live.
    let words = [
        0x7E02_02FFu32,
        0x817F_34AB,
        0x7E04_02FF,
        0xFE01_56CD,
        0xD744_0003,
        257 | (258 << 9) | (selector << 18),
        0x3408_0082,
        0xD834_0000,
        0x0000_0304,
        S_ENDPGM,
    ];

    let code: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();

    let native = NativeProgram {
        stage: HwStage::Cs,
        wave64: false,
        vgprs: 256,
        layout: vec![],
        used: vec![],
        io: StageIo::Cs(ComputeIo {
            rsrc2: 1 << 15,
            threads: [64, 1, 1],
        }),
        code,
    };

    let p = ir::build(&native.code, false, 256).unwrap();

    assert!(
        p.blocks
            .iter()
            .flat_map(|b| &b.insts)
            .any(|i| i.name() == "v_perm_b32")
    );

    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).unwrap();

    let io = io::raise(&p, &native).unwrap();

    let emitted = lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![],
        outputs: vec![],
        prims: vec![],
        mesh_stride: 0,
    })
    .unwrap();

    let container = crate::Container::parse(&emitted.dxil).unwrap();

    let features = container.part(crate::FourCC::SFI0).unwrap();

    u64::from_le_bytes(features.data.try_into().unwrap())
}

#[test]
fn perm_constant_selector_needs_no_int64() {
    assert_eq!(perm_features(128) & 0x8000, 0);
}

#[test]
fn perm_dynamic_selector_needs_no_int64() {
    assert_eq!(
        perm_features(256) & 0x8000,
        0,
        "dynamic v_perm_b32 must not require INT64"
    );
}

#[test]
fn a_descriptor_selected_by_scc_picks_its_heap_index() {
    // Two buffer descriptors loaded from the table at s0, one picked by
    // a root constant (s1) with s_cselect_b64 per half, and loaded
    // through by index and stored to LDS so the load is needed.
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF408_0202,
        0xFA00_0000, // s_load_dwordx4 s[8:11], s[4:5], null
        0xF408_0302,
        0xFA00_0010, // s_load_dwordx4 s[12:15], s[4:5], 0x10
        0xBF06_8001, // s_cmp_eq_u32 s1, 0
        0x8590_0C08, // s_cselect_b64 s[16:17], s[8:9], s[12:13]
        0x8592_0E0A, // s_cselect_b64 s[18:19], s[10:11], s[14:15]
        0xE030_2000,
        0x8004_0100, // buffer_load_dword v1, v0, s[16:19], 0 idxen
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        S_ENDPGM,
    ];
    let layout = || vec![root(RootType::Table, 0, 0), root(RootType::Constants, 1, 1)];
    // Both are structured buffers of dwords in the table's range.
    let used = || {
        vec![UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 4,
            dwords: 4,
            user_data: 0,
            slot: 0,
        }]
    };
    lower_with(&words, layout(), used()).unwrap();
}

/// Tables of 2D textures at s0 and samplers at s1.
fn texture_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let range = |kind, user_data, slot| UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: if kind == ResourceType::Sampler { 4 } else { 8 },
        user_data,
        slot,
    };
    (
        vec![root(RootType::Table, 0, 0), root(RootType::Table, 1, 1)],
        vec![
            range(ResourceType::Srv, 0, 0),
            range(ResourceType::Sampler, 1, 1),
        ],
    )
}

#[test]
fn waterfall_loops_over_per_lane_descriptors_sample_with_each_lanes_own() {
    use crate::resource::tests::{
        WATERFALL_FETCHED_WAVE32, WATERFALL_FETCHED_WAVE64, WATERFALL_PICKED_WAVE32,
    };
    for (words, wave64) in [
        (WATERFALL_FETCHED_WAVE32, false),
        (WATERFALL_FETCHED_WAVE64, true),
        (WATERFALL_PICKED_WAVE32, false),
    ] {
        let (layout, used) = texture_tables();
        let l = lower_wave(words, layout, used, wave64).unwrap();
        // The loop's reads of the first lane became each lane's own
        // dwords: no wave operation is left.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

/// [`texture_tables`] with a raw buffer UAV at dword 16 of the first.
pub(crate) fn texture_and_uav_tables() -> (Vec<RootParam>, Vec<UsedResource>) {
    let (layout, mut used) = texture_tables();
    used.push(UsedResource {
        kind: ResourceType::Uav,
        root: RootType::Table,
        table_offset_dwords: 16,
        lower_bound: 0,
        stride: 0,
        dwords: 4,
        user_data: 0,
        slot: 0,
    });
    (layout, used)
}

#[test]
fn waterfall_loops_over_descriptors_joined_per_lane_sample_with_each_lanes_own() {
    use crate::resource::tests::{WATERFALL_JOINED_MIXED_WAVE32, WATERFALL_JOINED_WAVE32};
    for words in [WATERFALL_JOINED_WAVE32, WATERFALL_JOINED_MIXED_WAVE32] {
        let (layout, used) = texture_and_uav_tables();
        let l = lower_wave(words, layout, used, false).unwrap();
        // One image handle, from the join's per-lane heap index.
        assert_eq!(l.nonuniform_handles, 1);
        assert!(!l.wave_ops);
    }
}

/// The census of a gather at a level of red at (v0, v0) and level v0
/// through the immediate sampler whose word 0 is `w0` (word 0's low 16
/// bits; then levels 0-12, linear filters, a transparent black border).
fn gather_at_level_census(w0: u16) -> [u32; crate::emit::census::KINDS.len()] {
    let words = [
        0xBE84_0300, // s_mov_b32 s4, s0
        0xBE85_0380, // s_mov_b32 s5, 0
        0xF40C_0302,
        0xFA00_0000,                 // s_load_dwordx8 s[12:19], s[4:5], 0x0
        0xB008_0000 | u32::from(w0), // s_movk_i32 s8, w0
        0xBE89_03FF,
        0x00C0_0000, // s_mov_b32 s9, 0xc00000
        0xBE8A_03FF,
        0x1850_0000, // s_mov_b32 s10, 0x18500000
        0xBE8B_0380, // s_mov_b32 s11, 0
        0x7E02_0D00, // v_cvt_f32_u32 v1, v0
        0x7E04_0301, // v_mov_b32 v2, v1
        0x7E06_0D00, // v_cvt_f32_u32 v3, v0
        0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
        0xF110_0108,
        0x0043_0401, // image_gather4_l v[4:7], v[1:3], s[12:19], s[8:11] dmask:0x1 dim:2D
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0x0608_0B04, // v_add_f32 v4, v4, v5
        0x060C_0F06, // v_add_f32 v6, v6, v7
        0x0608_0D04, // v_add_f32 v4, v4, v6
        0xD834_0000,
        0x0000_0400, // ds_write_b32 v0, v4
        S_ENDPGM,
    ];
    let (layout, used) = texture_tables();
    crate::emit::census::TEST_CENSUS.set(true);
    let l = lower_wave(&words, layout, used, false).unwrap();
    crate::emit::census::take(&l.dxil).expect("counted").counts
}

#[test]
fn gathers_at_a_level_read_a_black_border_outside() {
    let kind = |name| {
        crate::emit::census::KINDS
            .iter()
            .position(|k| *k == name)
            .unwrap()
    };
    // Clamped to a border in x, y and z (as the title's culling sampler),
    // or to the last texel.
    let (border, edge) = (gather_at_level_census(0x1B6), gather_at_level_census(0x92));
    // Four loads at the level either way, each texel bounded in x and y (a
    // maximum and a minimum) before its conversion to an integer.
    let reads = |c: &[u32]| (c[kind("texture_read")], c[kind("math_call")]);
    assert_eq!(reads(&border).0, 4);
    assert_eq!(reads(&border), reads(&edge));
    // Only the edge sampler's upper bound is the last texel (size - 1, one
    // subtraction per axis); the border's is one past it, the size, which
    // the host loads as 0, the border's colour.
    assert_eq!(edge[kind("arith")], border[kind("arith")] + 2);
}

#[test]
fn cube_samples_with_gradients_take_the_log2_level_and_its_clamp() {
    // `sample`: image_sample_d(_cl) v[12:15], v[1:8], s[8:15], s[16:19]
    // dmask:0xf dim:CUBE, its result stored to LDS. Face gradients
    // (0.5, 0) and (0, 1), face coordinates 0.5, 0.5 on face 0, clamp 2.
    let level = |sample: u32| {
        let words = [
            0xBE84_0300, // s_mov_b32 s4, s0
            0xBE85_0380, // s_mov_b32 s5, 0
            0xBE86_0301, // s_mov_b32 s6, s1
            0xBE87_0380, // s_mov_b32 s7, 0
            0xF40C_0202,
            0xFA00_0000, // s_load_dwordx8 s[8:15], s[4:5], 0
            0xF408_0403,
            0xFA00_0000, // s_load_dwordx4 s[16:19], s[6:7], 0
            0x7E02_02F0, // v_mov_b32 v1, 0.5
            0x7E04_0280, // v_mov_b32 v2, 0
            0x7E06_0280, // v_mov_b32 v3, 0
            0x7E08_02F2, // v_mov_b32 v4, 1.0
            0x7E0A_02F0, // v_mov_b32 v5, 0.5
            0x7E0C_02F0, // v_mov_b32 v6, 0.5
            0x7E0E_0280, // v_mov_b32 v7, 0
            0x7E10_02F4, // v_mov_b32 v8, 2.0
            0xBF8C_C07F, // s_waitcnt lgkmcnt(0)
            sample,
            0x0082_0C01,
            0xBF8C_3F70, // s_waitcnt vmcnt(0)
            0xD834_0000,
            0x0000_0C00, // ds_write_b32 v0, v12
            S_ENDPGM,
        ];
        let (layout, used) = texture_tables();
        lower_with(&words, layout, used).unwrap();
        let calls = super::super::shader::WRITTEN.with(|w| {
            w.borrow()
                .as_ref()
                .unwrap()
                .call_args("dx.op.sampleLevel.f32")
        });
        assert_eq!(calls.len(), 1);
        calls[0][10].clone()
    };
    // Log (23) is base 2 already: the level is log2 of the longer
    // gradient in texels, with nothing after it.
    let plain = level(0xF088_0F18);
    assert!(plain.starts_with("unary(23, "), "{plain}");
    // `_cl`: FMax (35) with the clamp after the address of the face.
    let clamped = level(0xF08C_0F18);
    assert!(clamped.starts_with("binary(35, unary(23, "), "{clamped}");
    assert!(clamped.ends_with(", 2)"), "{clamped}");
}

#[test]
fn scalar_bit_tests_lower() {
    // Each tests one bit of a constant into SCC and branches on it (only
    // code with an effect needs its operands).
    let tail = [
        0xBF85_0001, // s_cbranch_scc1 -> s_endpgm
        0xBE84_0380, // s_mov_b32 s4, 0
        S_ENDPGM,
    ];
    let bit_tests: [&[u32]; 4] = [
        &[0xBE81_0387, 0xBF0C_8301], // s_mov_b32 s1, 7; s_bitcmp0_b32 s1, 3
        &[0xBE81_0387, 0xBF0D_8301], // s_mov_b32 s1, 7; s_bitcmp1_b32 s1, 3
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0E_8300, // s_bitcmp0_b64 s[0:1], 3
        ],
        &[
            0xBE80_0387, // s_mov_b32 s0, 7
            0xBE81_0380, // s_mov_b32 s1, 0
            0xBF0F_8300, // s_bitcmp1_b64 s[0:1], 3
        ],
    ];
    for head in bit_tests {
        let words: Vec<u32> = head.iter().chain(&tail).copied().collect();
        lowers(&words, 0).unwrap();
    }
}

#[test]
#[ignore = "ours marks joins read through joins only with VICTORY_STRICT_JOINS (40 programs stop otherwise)"]
fn a_missing_value_stays_missing_through_two_live_joins() {
    let mut words = [
        0xF490_0180,
        0,           // s_memtime s[6:7]
        0xBF06_8000, // s_cmp_eq_u32 s0, 0 (s0 = group id X)
        0xBF85_0001, // skip the first overwrite
        0xBE86_0380, // s_mov_b32 s6, 0
        0xBF06_8101, // s_cmp_eq_u32 s1, 1 (s1 = group id Y)
        0xBF85_0001, // skip the second overwrite
        0xBE86_0381, // s_mov_b32 s6, 1
        0xBF07_8006, // s_cmp_lg_u32 s6, 0
        0xBF85_0001, // use the result to branch
        0xBE83_0380,
        S_ENDPGM,
    ];
    let group_ids_xy = 3 << 7;
    // Compare to a direct read so future timestamp support works in both cases.
    let refusal = |code: &[u32]| match lowers(code, group_ids_xy) {
        Ok(()) => None,
        Err(crate::ShaderError::Unsupported { what, .. }) => Some(what),
        Err(error) => panic!("{error}"),
    };
    let direct = [&words[..2], &words[8..]].concat();
    assert_eq!(refusal(&words), refusal(&direct));
    words[8] = 0xBF07_8000; // read the group id instead; the timestamp is dead
    lowers(&words, group_ids_xy).expect("dead timestamp");
}

#[test]
fn buffer_formats_by_number() {
    // The numbers the disassembler names 8_8_8_8_UNORM, 16_16_SNORM,
    // 2_10_10_10_UNORM, 16_FLOAT, 32_32_32_32_FLOAT.
    assert_eq!(buffer_format(56), Some((&[8, 8, 8, 8][..], Numeric::Unorm)));
    assert_eq!(buffer_format(24), Some((&[16, 16][..], Numeric::Snorm)));
    assert_eq!(
        buffer_format(50),
        Some((&[10, 10, 10, 2][..], Numeric::Unorm))
    );
    assert_eq!(buffer_format(13), Some((&[16][..], Numeric::Float)));
    assert_eq!(
        buffer_format(77),
        Some((&[32, 32, 32, 32][..], Numeric::Float))
    );
    // Invalid, past the table.
    assert_eq!(buffer_format(0), None);
    assert_eq!(buffer_format(78), None);
}

#[test]
fn packed_buffer_formats_by_number() {
    // LLVM's `UfmtGFX10`: 10_11_11 is 30-36, 11_11_10 37-43,
    // 10_10_10_2 44-49. Channels from the lowest bits: R11G11B10_FLOAT
    // is 10_11_11 (Mesa), 10_10_10_2 has the 2-bit channel first.
    assert_eq!(buffer_format(30), Some((&[11, 11, 10][..], Numeric::Unorm)));
    assert_eq!(buffer_format(36), Some((&[11, 11, 10][..], Numeric::Float)));
    assert_eq!(buffer_format(37), Some((&[10, 11, 11][..], Numeric::Unorm)));
    assert_eq!(buffer_format(43), Some((&[10, 11, 11][..], Numeric::Float)));
    assert_eq!(
        buffer_format(44),
        Some((&[2, 10, 10, 10][..], Numeric::Unorm))
    );
    assert_eq!(
        buffer_format(45),
        Some((&[2, 10, 10, 10][..], Numeric::Snorm))
    );
    assert_eq!(
        buffer_format(46),
        Some((&[2, 10, 10, 10][..], Numeric::Uscaled))
    );
    assert_eq!(
        buffer_format(47),
        Some((&[2, 10, 10, 10][..], Numeric::Sscaled))
    );
    assert_eq!(
        buffer_format(48),
        Some((&[2, 10, 10, 10][..], Numeric::Uint))
    );
    assert_eq!(
        buffer_format(49),
        Some((&[2, 10, 10, 10][..], Numeric::Sint))
    );
}

/// A half's value (sign, 5-bit exponent biased 15, 10-bit mantissa), as
/// `legacyF16ToF32` widens it.
fn half(h: u32) -> f32 {
    let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 1023);
    let v = match e {
        0 => m as f32 * 2f32.powi(-24),
        31 if m == 0 => f32::INFINITY,
        31 => f32::NAN,
        _ => (1024 + m) as f32 * 2f32.powi(e as i32 - 25),
    };
    if s == 1 { -v } else { v }
}

#[test]
fn small_floats_widen_as_halves() {
    // Unsigned 11- and 10-bit floats: a 5-bit exponent biased 15 over a
    // 6- or 5-bit mantissa, no sign; values from the D3D/GL spec.
    let f11 = |x: u32| half(x << half_shift(11));
    let f10 = |x: u32| half(x << half_shift(10));
    assert_eq!(f11(0), 0.0);
    assert_eq!(f11(0x001), 2f32.powi(-20)); // smallest denormal
    assert_eq!(f11(0x03F), 63.0 * 2f32.powi(-20)); // largest denormal
    assert_eq!(f11(0x040), 2f32.powi(-14)); // smallest normal
    assert_eq!(f11(0x3C0), 1.0);
    assert_eq!(f11(0x3E0), 1.5);
    assert_eq!(f11(0x7BF), 65024.0); // largest
    assert_eq!(f11(0x7C0), f32::INFINITY);
    assert!(f11(0x7C1).is_nan());
    assert_eq!(f10(0x001), 2f32.powi(-19));
    assert_eq!(f10(0x020), 2f32.powi(-14));
    assert_eq!(f10(0x1E0), 1.0);
    assert_eq!(f10(0x3DF), 64512.0);
    assert_eq!(f10(0x3E0), f32::INFINITY);
    assert!(f10(0x3FF).is_nan());
    // A 16-bit float is a half already, sign included.
    assert_eq!(half(0xBC00 << half_shift(16)), -1.0);
}

/// Lowers a pixel program with the input VGPRs `input_addr` enables, one
/// float4 TEXCOORD0 attribute (attr 0, row 0), two float4 targets, and the
/// root parameters `layout` with the resources `used`.
fn lower_pixel(
    words: &[u32],
    input_addr: u32,
    layout: Vec<RootParam>,
    used: Vec<UsedResource>,
) -> Result<Lowered, crate::ShaderError> {
    use super::super::container::SigElement;
    use crate::native::{PsInput, Usage};
    let native = NativeProgram {
        stage: HwStage::Ps,
        wave64: false,
        vgprs: 16,
        layout,
        used,
        io: StageIo::Ps {
            inputs: vec![PsInput {
                usage: Usage::Generic,
                row: 0,
                mask: 0xF,
                attr: 0,
                flat: false,
                fp16: false,
                pack2: false,
                row2: 0,
            }],
            input_addr,
            db_shader_control: 0,
            patches: vec![],
        },
        code: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    };
    let element = |name: &str, kind, row, interp, usage| SigElement {
        name: name.into(),
        index: row,
        kind,
        comp: 9,
        interp,
        row: Some(row),
        col: 0,
        cols: 4,
        usage,
        half: false,
    };
    let p = ir::build(&native.patched_code(), false, native.vgprs).unwrap();
    let st = structure::structure(&p).unwrap();
    let res = resource::track(&p, &native.layout, native.stage, Some(&st))?;
    let io = io::raise(&p, &native).unwrap();
    lower(&Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: vec![element("TEXCOORD", 0, 0, 2, 0)],
        outputs: vec![
            element("SV_Target", 16, 0, 0, 15),
            element("SV_Target", 16, 1, 0, 15),
        ],
        prims: vec![],
        mesh_stride: 0,
    })
}

/// An alpha test (`v_cmpx` on EXEC) before two targets: the lanes the
/// first export leaves out are the ones the `done` export discards, so it
/// stores in every lane. With EXEC restored before the `done` export
/// those lanes live on, and the export stays refused.
#[test]
fn an_export_before_the_discarding_done_export_stores_unguarded() {
    let program = |restore: bool| {
        let mut words = vec![
            0xC808_0000, // v_interp_p1_f32 v2, v0, attr0.x
            0xC809_0001, // v_interp_p2_f32 v2, v1, attr0.x
            0xBE8C_037E, // s_mov_b32 s12, exec_lo
            0x7C22_0480, // v_cmpx_lt_f32 0, v2
            0xF800_100F,
            0x0202_0202, // exp mrt0 v2, v2, v2, v2 vm
        ];
        if restore {
            words.push(0xBEFE_030C); // s_mov_b32 exec_lo, s12
        }
        words.extend([
            0xF800_181F,
            0x0202_0202, // exp mrt1 v2, v2, v2, v2 done vm
            S_ENDPGM,
        ]);
        lower_pixel(&words, 1 << 1, vec![], vec![]).map(|_| ())
    };
    program(false).unwrap();
    let e = program(true).unwrap_err().to_string();
    assert!(e.contains("a conditional export"), "{e}");
}

/// A level query on an array texture reads no slice: here its register
/// holds the ancillary input, which has no DXIL form.
#[test]
fn a_level_query_skips_the_array_slice() {
    let words = [
        0xBE81_0380, // s_mov_b32 s1, 0
        0xF40C_0000,
        0xFA00_0000, // s_load_dwordx8 s[0:7], s[0:1], null
        0xBE88_0380, // s_mov_b32 s8, 0
        0xBE89_0380, // s_mov_b32 s9, 0
        0xBE8A_0380, // s_mov_b32 s10, 0
        0xBE8B_0380, // s_mov_b32 s11, 0
        0xC80C_0000, // v_interp_p1_f32 v3, v0, attr0.x
        0xC80D_0001, // v_interp_p2_f32 v3, v1, attr0.x
        0xC810_0100, // v_interp_p1_f32 v4, v0, attr0.y
        0xC811_0101, // v_interp_p2_f32 v4, v1, attr0.y
        0x7E0A_0302, // v_mov_b32 v5, v2 (the ancillary input)
        0xBF8C_0070, // s_waitcnt vmcnt(0) lgkmcnt(0)
        0xF180_0328,
        0x0040_0603, // image_get_lod v[6:7], v[3:5], s[0:7], s[8:11] dmask:0x3 dim:2D_ARRAY
        0xBF8C_3F70, // s_waitcnt vmcnt(0)
        0xF800_180F,
        0x0706_0706, // exp mrt0 v6, v7, v6, v7 done vm
        S_ENDPGM,
    ];
    let texture = UsedResource {
        kind: ResourceType::Srv,
        root: RootType::Table,
        table_offset_dwords: 0,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot: 0,
    };
    let table = RootParam {
        sgprs: 1,
        ..root(RootType::Table, 0, 0)
    };
    lower_pixel(&words, 1 << 1 | 1 << 13, vec![table], vec![texture]).unwrap();
}

/// The complement of a ballot (its bits for the lanes that skip the branch set) counted inside
/// that branch: no ballot there holds those lanes, so the count is `s_not` applied to the
/// compare's ballot, as the console computes it (and `s_orn2` against EXEC). Waves of 32 and
/// of 64.
#[test]
fn a_complemented_ballot_counts_where_not_every_lane_runs() {
    let wave32 = [
        0x7D8A_0080u32, // v_cmp_ne_u32 vcc_lo, 0, v0
        0xBE80_3C6A,    // s_and_saveexec_b32 s0, vcc_lo
        0xBF88_0006,    // s_cbranch_execz -> s_or_b32
        0x7D82_0085,    // v_cmp_lt_u32 vcc_lo, 5, v0
        0xBE81_076A,    // s_not_b32 s1, vcc_lo
        0xBE82_0F01,    // s_bcnt1_i32_b32 s2, s1
        0x7E02_0202,    // v_mov_b32 v1, s2
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        0x887E_007E, // s_or_b32 exec_lo, exec_lo, s0
        S_ENDPGM,
    ];
    let wave64 = [
        0x7D8A_0080u32, // v_cmp_ne_u32 vcc, 0, v0
        0xBE80_246A,    // s_and_saveexec_b64 s[0:1], vcc
        0xBF88_0006,    // s_cbranch_execz -> s_or_b64
        0x7D82_0085,    // v_cmp_lt_u32 vcc, 5, v0
        0xBE82_086A,    // s_not_b64 s[2:3], vcc
        0xBE84_1002,    // s_bcnt1_i32_b64 s4, s[2:3]
        0x7E02_0204,    // v_mov_b32 v1, s4
        0xD834_0000,
        0x0000_0100, // ds_write_b32 v0, v1
        0x88FE_007E, // s_or_b64 exec, exec, s[0:1]
        S_ENDPGM,
    ];
    let mut orn2 = wave32;
    orn2[4] = 0x8B01_6A7E; // s_orn2_b32 s1, exec_lo, vcc_lo
    for (words, wave64) in [
        (&wave32[..], false),
        (&orn2[..], false),
        (&wave64[..], true),
    ] {
        let l = lower_wave(words, vec![], vec![], wave64).unwrap();
        assert!(l.wave_ops);
    }
}
