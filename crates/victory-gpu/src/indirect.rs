//! Indirect execution's arguments, bounded and compacted before the GPU reads
//! them. The title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by
//! compute shaders of ours (HLSL compiled by the system's compiler; 5.1
//! shaders sharing one root signature inside), leaving the title's argument
//! and count buffers untouched.
//!
//! The scratch buffer is left readable as indirect arguments after a clamp
//! and stays so until the next clamp into it, which moves it straight back
//! to unordered access; it goes back to the common state once, when the
//! list closes. The executions that read it record no barrier of their
//! own, so draws that read records clamped together (`crate::draw`'s
//! windows) follow each other with nothing between them. Before, each
//! execution was followed by a move of the scratch buffer back to the
//! common state, which drivers may treat as a full wait.
//!
//! When a count buffer is given, threads at or past `min(max, count)` do
//! not charge the list's budget or read uninitialized tail slots. Each list
//! has a budget (2^27 vertex instances or threads, groups counted as 64
//! threads, mesh groups as 128): active records past it run empty,
//! saturating without 32-bit wraparound so a list of thousands of garbage
//! records cannot run for seconds either.
//!
//! For non-windowed multi-record calls (`records > 1`), empty/zero-cost and
//! budget-zeroed records are compacted out in exact submission order using
//! hierarchical `groupshared` prefix sums (no cross-workgroup spin loops), and
//! the compacted count is written inside the scratch buffer for
//! `ExecuteIndirect`. Windowed calls (`bound_ahead`) keep their in-slot layout
//! so later draws can index cached windows directly.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::{PCSTR, s};

use crate::host::{HostList, host, transition};

const TILE: u32 = 256;

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=8, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint c0;
    uint c1;
    uint c2;
    uint c3;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing.
    uint c4;
    uint c5;
    uint c6;
    uint c7;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// The vertices or indices an instance from which a draw record runs empty.
static const uint MOST_DRAW = 1u << 23;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
static const uint TILE = 256u;

// Each group charges the budget once: the costs of its records go here,
// one thread charges their sum, saturating as each record would in lane
// order, and every thread then reads whether its own record fitted. A
// compare and swap per record, 64+ threads racing for one word, cost a
// round for each record of a window, and in memory the CPU maps (where the
// budget was, and is with VICTORY_INDIRECT_CARD_BUDGET=0) every atomic
// crosses the bus on a card of its own.
groupshared uint s_costs[TILE];
groupshared uint s_fits[TILE];
groupshared uint s_scan[TILE];
groupshared uint s_largest_asked;
groupshared uint s_all_asked;

uint effective_count(uint records, uint flags, uint cnt_off) {
    uint eff = records;
    if ((flags & 1u) != 0u) {
        eff = min(eff, count_buf.Load(cnt_off));
    }
    return eff;
}

void clamp_header(
    uint kind,
    uint src_at,
    bool ahead,
    out uint2 draw_counts,
    out uint3 groups,
    out uint cost,
    out uint asked
) {
    draw_counts = uint2(0u, 0u);
    groups = uint3(0u, 0u, 0u);
    cost = 0u;
    asked = 0u;
    if (kind <= 1u) {
        // Draws: vertex or index count, then instances.
        uint2 want = src_args.Load2(src_at);
        if (ahead && (want.x > (1u << 24) || want.y > (1u << 24))) {
            want = uint2(0u, 0u);
        }
        uint count = min(want.x, 1u << 24);
        uint instances = min(want.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        // A record that asks for MOST_DRAW vertices or indices an instance
        // or more is no draw of the title's (no mesh of a console title has
        // 2.8 million triangles): it runs empty and charges nothing.
        if (count >= MOST_DRAW) {
            count = 0u;
            instances = 0u;
            cost = 0u;
        }
        draw_counts = uint2(count, instances);
    } else if (kind == 2u) {
        uint3 g = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0u && g.x * g.y > GROUPS / g.z) {
            g = uint3(0u, 0u, 0u);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is a
        // slot the title has not filled (the title leaves all
        // ones there): empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0u, 0u, 0u);
        }
        asked = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
}

bool charge_group(
    uint tid,
    uint group_n,
    bool active,
    uint kind,
    uint cost,
    uint asked,
    bool track_diag
) {
    if (tid == 0u) {
        s_largest_asked = 0u;
        s_all_asked = 0u;
    }
    s_costs[tid] = active ? cost : 0u;
    GroupMemoryBarrierWithGroupSync();
    if (track_diag && active && kind > 2u) {
        uint was;
        InterlockedMax(s_largest_asked, asked, was);
        InterlockedAdd(s_all_asked, min(asked, 1u << 24), was);
    }
    // Only the group's own active records are walked; groups past a count
    // buffer's count have `group_n == 0` and do not touch `budget`.
    if (tid == 0u && group_n > 0u) {
        uint before;
        budget.InterlockedOr(0, 0u, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0u; j < group_n; j++) {
                uint c = s_costs[j];
                if (c > 0u && next < BUDGET) {
                    next = (BUDGET - next <= c) ? BUDGET : (next + c);
                }
            }
            if (next == before) break;
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        uint at = before;
        for (uint k = 0u; k < group_n; k++) {
            uint c = s_costs[k];
            uint fit = 1u;
            if (c > 0u) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at <= c) ? BUDGET : (at + c);
            }
            s_fits[k] = fit;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (track_diag && tid == 0u && kind > 2u && group_n > 0u) {
        uint was;
        budget.InterlockedMax(4, s_largest_asked, was);
        budget.InterlockedAdd(8, s_all_asked, was);
    }
    return active && s_fits[tid] != 0u;
}

void store_record(uint kind, uint stride, uint dst_at, uint src_at, uint2 draw_counts, uint3 groups) {
    if (kind <= 1u) {
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8u; off < stride; off += 4u) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        dst_args.Store3(dst_at, groups);
        for (uint off = 12u; off < stride; off += 4u) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}

uint group_inclusive_scan(uint tid, uint val) {
    s_scan[tid] = val;
    GroupMemoryBarrierWithGroupSync();
    [unroll]
    for (uint s = 1u; s < TILE; s *= 2u) {
        uint add = (tid >= s) ? s_scan[tid - s] : 0u;
        GroupMemoryBarrierWithGroupSync();
        s_scan[tid] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    return s_scan[tid];
}

[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint gid = group_id.y * 65535u + group_id.x;
    uint tid = gtid.x;
    uint idx = gid * 64u + tid;
    uint eff = effective_count(c2, c4, c5);
    bool active = idx < eff;
    uint src_at = c3 + idx * c1;
    uint dst_at = idx * c1;
    bool ahead = (c4 & 2u) != 0u;
    uint2 draw_counts = uint2(0u, 0u);
    uint3 groups = uint3(0u, 0u, 0u);
    uint cost = 0u;
    uint asked = 0u;
    if (active) {
        clamp_header(c0, src_at, ahead, draw_counts, groups, cost, asked);
    }
    uint first = gid * 64u;
    uint group_n = eff > first ? min(64u, eff - first) : 0u;
    bool fits = charge_group(tid, group_n, active, c0, cost, asked, true);
    if (!active) return;
    if (!fits) {
        draw_counts = uint2(0u, 0u);
        groups = uint3(0u, 0u, 0u);
    }
    store_record(c0, c1, dst_at, src_at, draw_counts, groups);
}

[RootSignature(RS)]
[numthreads(TILE, 1, 1)]
void compact_single(uint3 gtid : SV_GroupThreadID) {
    uint tid = gtid.x;
    uint eff = effective_count(c2, c4, c5);
    bool active = tid < eff;
    uint src_at = c3 + tid * c1;
    bool ahead = (c4 & 2u) != 0u;
    uint2 draw_counts = uint2(0u, 0u);
    uint3 groups = uint3(0u, 0u, 0u);
    uint cost = 0u;
    uint asked = 0u;
    if (active) {
        clamp_header(c0, src_at, ahead, draw_counts, groups, cost, asked);
    }
    uint group_n = min(TILE, eff);
    bool fits = charge_group(tid, group_n, active, c0, cost, asked, true);
    uint keep = (active && cost > 0u && fits) ? 1u : 0u;
    uint inclusive = group_inclusive_scan(tid, keep);
    if (tid == 0u) {
        dst_args.Store(c6, s_scan[TILE - 1u]);
    }
    if (keep != 0u) {
        store_record(c0, c1, (inclusive - 1u) * c1, src_at, draw_counts, groups);
    }
}

[RootSignature(RS)]
[numthreads(TILE, 1, 1)]
void compact_pass1(uint3 group_id : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint gid = group_id.y * 65535u + group_id.x;
    uint tiles = (c2 + TILE - 1u) / TILE;
    if (gid >= tiles) return;
    uint tid = gtid.x;
    uint idx = gid * TILE + tid;
    uint eff = effective_count(c2, c4, c5);
    bool active = idx < eff;
    bool ahead = (c4 & 2u) != 0u;
    uint2 draw_counts = uint2(0u, 0u);
    uint3 groups = uint3(0u, 0u, 0u);
    uint cost = 0u;
    uint asked = 0u;
    if (active) {
        clamp_header(c0, c3 + idx * c1, ahead, draw_counts, groups, cost, asked);
    }
    uint first = gid * TILE;
    uint group_n = eff > first ? min(TILE, eff - first) : 0u;
    bool fits = charge_group(tid, group_n, active, c0, cost, asked, true);
    uint keep = (active && cost > 0u && fits) ? 1u : 0u;
    uint inclusive = group_inclusive_scan(tid, keep);
    if (idx < c2) {
        uint local_idx = inclusive - keep;
        dst_args.Store(c6 + idx * 4u, (keep != 0u) ? (0x80000000u | local_idx) : 0u);
    }
    if (tid == TILE - 1u) {
        dst_args.Store(c7 + gid * 4u, s_scan[TILE - 1u]);
    }
}

[RootSignature(RS)]
[numthreads(TILE, 1, 1)]
void scan_tiles(uint3 group_id : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint gid = group_id.y * 65535u + group_id.x;
    uint groups = (c1 + TILE - 1u) / TILE;
    if (gid >= groups) return;
    uint tid = gtid.x;
    uint idx = gid * TILE + tid;
    uint val = (idx < c1) ? dst_args.Load(c0 + idx * 4u) : 0u;
    uint inclusive = group_inclusive_scan(tid, val);
    if (idx < c1) {
        dst_args.Store(c0 + idx * 4u, inclusive - val);
    }
    if (tid == TILE - 1u) {
        dst_args.Store(c2 + gid * 4u, s_scan[TILE - 1u]);
    }
}

[RootSignature(RS)]
[numthreads(TILE, 1, 1)]
void add_offsets(uint3 group_id : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint gid = group_id.y * 65535u + group_id.x;
    uint idx = gid * TILE + gtid.x;
    if (idx >= c1) return;
    uint add = dst_args.Load(c2 + gid * 4u);
    if (add != 0u) {
        uint at = c0 + idx * 4u;
        dst_args.Store(at, dst_args.Load(at) + add);
    }
}

[RootSignature(RS)]
[numthreads(TILE, 1, 1)]
void compact_scatter(uint3 group_id : SV_GroupID, uint3 gtid : SV_GroupThreadID) {
    uint gid = group_id.y * 65535u + group_id.x;
    uint idx = gid * TILE + gtid.x;
    if (idx >= c2) return;
    uint m = dst_args.Load(c6 + idx * 4u);
    if ((m & 0x80000000u) == 0u) return;
    uint dst_idx = dst_args.Load(c7 + gid * 4u) + (m & 0x7FFFFFFFu);
    uint src_at = c3 + idx * c1;
    bool ahead = (c4 & 2u) != 0u;
    uint2 draw_counts;
    uint3 groups;
    uint cost;
    uint asked;
    clamp_header(c0, src_at, ahead, draw_counts, groups, cost, asked);
    store_record(c0, c1, dst_idx * c1, src_at, draw_counts, groups);
}
"#;

/// Clamped (and, when compacting, compacted) indirect arguments ready in a
/// device-local scratch buffer in `INDIRECT_ARGUMENT`.
pub(crate) struct Bounded {
    pub(crate) buffer: ID3D12Resource,
    /// Byte offset of the 32-bit compacted command count inside `buffer` when
    /// multi-record compaction ran; `None` when records stay in-slot.
    pub(crate) count: Option<u64>,
}

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    compact_single: Option<ID3D12PipelineState>,
    compact_pass1: Option<ID3D12PipelineState>,
    scan_tiles: Option<ID3D12PipelineState>,
    add_offsets: Option<ID3D12PipelineState>,
    compact_scatter: Option<ID3D12PipelineState>,
}

// SAFETY: D3D12 objects are free-threaded.
unsafe impl Send for Clamp {}
// SAFETY: as above.
unsafe impl Sync for Clamp {}

fn clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| match make() {
            Ok(c) => Some(c),
            Err(e) => {
                tracing::warn!(error = %e, "no argument clamp: indirect execution dropped");
                None
            }
        })
        .as_ref()
}

fn make() -> windows::core::Result<Clamp> {
    let device = &host().ok_or_else(windows::core::Error::empty)?.device;
    make_for(device)
}

fn compile_hlsl(source: &str, name: PCSTR, entry: PCSTR) -> windows::core::Result<ID3DBlob> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            name,
            None,
            None,
            entry,
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            Some(&mut errors),
        )
    };
    if let Err(e) = compiled {
        if let Some(errors) = errors {
            // SAFETY: the blob's own bytes.
            let text = unsafe {
                std::slice::from_raw_parts(
                    errors.GetBufferPointer().cast::<u8>(),
                    errors.GetBufferSize(),
                )
            };
            tracing::warn!("argument clamp: {}", String::from_utf8_lossy(text));
        }
        return Err(e);
    }
    code.ok_or_else(windows::core::Error::empty)
}

fn make_compute_pso(
    device: &ID3D12Device,
    root: &ID3D12RootSignature,
    blob: &ID3DBlob,
) -> windows::core::Result<ID3D12PipelineState> {
    // SAFETY: `blob` owns valid DXBC bytes and `root` stays alive through the call.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })
    }
}

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    let mut clamp = make_from(device, SOURCE)?;
    let compile_entry = |entry: PCSTR| -> windows::core::Result<ID3D12PipelineState> {
        make_compute_pso(device, &clamp.root, &compile_hlsl(SOURCE, s!("clamp"), entry)?)
    };
    clamp.compact_single = Some(compile_entry(s!("compact_single"))?);
    clamp.compact_pass1 = Some(compile_entry(s!("compact_pass1"))?);
    clamp.scan_tiles = Some(compile_entry(s!("scan_tiles"))?);
    clamp.add_offsets = Some(compile_entry(s!("add_offsets"))?);
    clamp.compact_scatter = Some(compile_entry(s!("compact_scatter"))?);
    Ok(clamp)
}

/// The clamp compiled from `source` (ours, or in the benchmark the one it
/// replaced). Single-entry sources (`CHECK_SOURCE`, `bench::OLD_SOURCE`) only
/// expose `main`.
fn make_from(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
    let code = compile_hlsl(source, s!("clamp"), s!("main"))?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    let root: ID3D12RootSignature = unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        device.CreateRootSignature(0, bytes)?
    };
    let pso = make_compute_pso(device, &root, &code)?;
    Ok(Clamp {
        root,
        pso,
        compact_single: None,
        compact_pass1: None,
        scan_tiles: None,
        add_offsets: None,
        compact_scatter: None,
    })
}

/// Minimum record byte stride for a supported PC indirect argument `kind`.
fn min_stride(kind: u32) -> Option<u32> {
    match kind {
        0 => Some(16),
        1 => Some(20),
        2 | 10 => Some(12),
        _ => None,
    }
}

/// Validates record alignment and returns the scratch byte size `stride * records`.
fn validate_layout(
    (kind, stride, records): (u32, u32, u32),
    src_offset: u64,
    count_offset: Option<u64>,
) -> Option<u64> {
    let min = min_stride(kind)?;
    if stride < min || !stride.is_multiple_of(4) || !src_offset.is_multiple_of(4) {
        return None;
    }
    if count_offset.is_some_and(|off| !off.is_multiple_of(4)) {
        return None;
    }
    let bytes = u64::from(stride).checked_mul(u64::from(records))?;
    (bytes <= u64::from(u32::MAX)).then_some(bytes)
}

/// Scratch layout inside `dst` for clamping (and optional multi-record
/// compaction). Falls back to in-slot layout if the compaction auxiliary
/// region would overflow 32-bit `ByteAddressBuffer` indexing.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ScratchLayout {
    total_bytes: u64,
    count_offset: Option<u64>,
    meta_offset: u64,
    tile_offset: u64,
    levels: Vec<(u32, u32)>,
}

fn scratch_layout(args_bytes: u64, records: u32, ahead: bool) -> ScratchLayout {
    let in_slot = ScratchLayout {
        total_bytes: args_bytes,
        count_offset: None,
        meta_offset: 0,
        tile_offset: 0,
        levels: Vec::new(),
    };
    if ahead || records <= 1 {
        return in_slot;
    }
    let try_compact = (|| -> Option<ScratchLayout> {
        let count_offset = args_bytes;
        if records <= TILE {
            let total_bytes = count_offset.checked_add(4)?;
            if total_bytes > u64::from(u32::MAX) {
                return None;
            }
            return Some(ScratchLayout {
                total_bytes,
                count_offset: Some(count_offset),
                meta_offset: 0,
                tile_offset: 0,
                levels: Vec::new(),
            });
        }
        let meta_offset = count_offset.checked_add(4)?;
        let tile_offset = meta_offset.checked_add(u64::from(records).checked_mul(4)?)?;
        let mut levels = Vec::new();
        let mut off = tile_offset;
        let mut n = records.div_ceil(TILE);
        loop {
            let level_u32 = u32::try_from(off).ok()?;
            levels.push((level_u32, n));
            let bytes = u64::from(n).checked_mul(4)?;
            off = off.checked_add(bytes)?;
            if n <= TILE {
                break;
            }
            n = n.div_ceil(TILE);
        }
        if off > u64::from(u32::MAX) {
            return None;
        }
        Some(ScratchLayout {
            total_bytes: off,
            count_offset: Some(count_offset),
            meta_offset,
            tile_offset,
            levels,
        })
    })();
    try_compact.unwrap_or(in_slot)
}

/// Records, in `list`, the non-destructive clamp (and, for `records > 1`,
/// order-preserving compaction of empty/zero-cost and budget-zeroed records)
/// of up to `records` records of `kind` (PC argument numbering) `stride`
/// bytes apart from `src` (and bounded by the 32-bit count in `count` when
/// given) into a device-local scratch buffer, at offset `0`. The scratch
/// buffer is left in `INDIRECT_ARGUMENT` until the next clamp into it:
/// executions read it with no barrier of their own. Replaces the list's
/// compute root signature and pipeline (callers bind theirs after). `None` if
/// the clamp or alignment is not available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<Bounded> {
    let args_bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let layout = scratch_layout(args_bytes, spec.2, false);
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(layout.total_bytes)?);
    clamp_into(list, c, &dst, budget, src, count, spec, false, &layout).then_some(Bounded {
        buffer: dst,
        count: layout.count_offset,
    })
}

/// [`bound`] for a window of records clamped ahead of the draws that will
/// read them (`crate::draw`'s windows; no count buffer). The records past
/// the ones the title filled are whatever the buffer holds, and each
/// record clamped charges the list's budget when it is clamped: a slot of
/// all ones, clamped to the most a mesh dispatch runs (65,535 groups),
/// charged a sixteenth of the budget, so sixteen of them emptied every
/// draw of the list after them (run 93: "the largest group count asked for
/// 4294967295", 43 to 153 such records a frame). Here a record that asks
/// for more than can run is left empty and charges nothing; a record
/// clamped alone is clamped and charged as before.
///
/// Windowed records keep their exact in-slot offsets without compaction so
/// subsequent cached draws can index `Window::scratch` directly.
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let args_bytes = validate_layout(spec, src.1, None)?;
    let layout = scratch_layout(args_bytes, spec.2, true);
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(layout.total_bytes)?);
    clamp_into(list, c, &dst, budget, src, None, spec, true, &layout).then_some(dst)
}

/// [`record_clamp_with`] into `list`'s scratch buffer `dst`, from the state
/// the list left it in; afterwards the list knows it is readable as
/// indirect arguments.
#[allow(clippy::too_many_arguments)]
fn clamp_into(
    list: &HostList,
    c: &Clamp,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
    layout: &ScratchLayout,
) -> bool {
    let before = list.indirect_state(dst);
    let recorded =
        record_clamp_with(c, &list.list, (dst, before), budget, src, count, spec, ahead, layout);
    if recorded {
        list.indirect_readable(dst);
    }
    recorded
}

/// Records the clamp into `dst`, which is in state `before` (common, or
/// indirect arguments after an earlier clamp into it), and leaves `dst` in
/// `INDIRECT_ARGUMENT`. `ahead`: the records are a window clamped ahead of
/// their draws ([`bound_ahead`]). False, with nothing recorded, if the
/// layout or the addresses do not fit the clamp.
#[allow(clippy::too_many_arguments)]
fn record_clamp_as(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
    ahead: bool,
) -> bool {
    let Some(args_bytes) = validate_layout(spec, src.1, count.map(|(_, off)| off)) else {
        return false;
    };
    let layout = if c.compact_single.is_some() {
        scratch_layout(args_bytes, spec.2, ahead)
    } else {
        ScratchLayout {
            total_bytes: args_bytes,
            count_offset: None,
            meta_offset: 0,
            tile_offset: 0,
            levels: Vec::new(),
        }
    };
    record_clamp_with(c, l, dst, budget, src, count, spec, ahead, &layout)
}

fn dispatch_groups(l: &ID3D12GraphicsCommandList, groups: u32) {
    let gx = groups.min(65_535);
    let gy = groups.div_ceil(65_535);
    // SAFETY: caller ensures an open command list and bound compute state.
    unsafe { l.Dispatch(gx, gy, 1) };
}

/// [`record_clamp_as`] using a precomputed [`ScratchLayout`].
#[allow(clippy::too_many_arguments)]
fn record_clamp_with(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    ahead: bool,
    layout: &ScratchLayout,
) -> bool {
    let Some(bytes) = validate_layout(
        (kind, stride, records),
        src_offset,
        count.map(|(_, off)| off),
    ) else {
        return false;
    };
    // SAFETY: live D3D12 buffer resources.
    let (src_base, dst_gpu) = unsafe { (src.GetGPUVirtualAddress(), dst.GetGPUVirtualAddress()) };
    if !src_base.is_multiple_of(4) || !dst_gpu.is_multiple_of(4) || !budget.is_multiple_of(4) {
        return false;
    }
    let (src_gpu, args_offset) = match src_offset
        .checked_add(bytes)
        .filter(|&end| end <= u64::from(u32::MAX))
    {
        Some(_) => (src_base, src_offset as u32),
        None => {
            let Some(rebased) = src_base.checked_add(src_offset) else {
                return false;
            };
            (rebased, 0)
        }
    };
    let (count_gpu, has_count, count_offset) = match count {
        Some((buf, off)) => {
            // SAFETY: a live D3D12 buffer resource.
            let base = unsafe { buf.GetGPUVirtualAddress() };
            if !base.is_multiple_of(4) {
                return false;
            }
            match off.checked_add(4).filter(|&end| end <= u64::from(u32::MAX)) {
                Some(_) => (base, 1u32, off as u32),
                None => {
                    let Some(rebased) = base.checked_add(off) else {
                        return false;
                    };
                    (rebased, 1u32, 0u32)
                }
            }
        }
        None => (src_gpu, 0u32, 0u32),
    };
    let flags = has_count | (u32::from(ahead) << 1);
    let compact_psos = match (
        &c.compact_single,
        &c.compact_pass1,
        &c.scan_tiles,
        &c.add_offsets,
        &c.compact_scatter,
    ) {
        (Some(s), Some(p1), Some(st), Some(ao), Some(sc)) => Some((s, p1, st, ao, sc)),
        _ => None,
    };
    let use_compact = layout.count_offset.filter(|_| compact_psos.is_some());
    // SAFETY: an open list; our root signature and pipeline; live source,
    // count, scratch destination, and budget buffers. `src` and `count` stay
    // in `COMMON` / `GENERIC_READ` via implicit buffer promotion while `dst`
    // transitions `before -> UNORDERED_ACCESS -> INDIRECT_ARGUMENT`. Moving
    // it out of `INDIRECT_ARGUMENT` orders this clamp's writes after every
    // earlier execution's reads of the scratch buffer; nothing else writes it.
    // Diagnostics: counted; `VICTORY_BARRIER_TEST=indirect` leaves the two
    // waits for every earlier shader write out (the state changes stay).
    #[cfg(feature = "diagnostics")]
    crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
    #[cfg(feature = "diagnostics")]
    let waits = crate::copystat::barrier_test() & 2 == 0;
    #[cfg(not(feature = "diagnostics"))]
    let waits = true;
    let first = [
        uav_barrier(),
        transition(dst, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
    ];
    let second = [
        uav_barrier(),
        transition(
            dst,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
        ),
    ];
    let skip = usize::from(!waits);
    let constants_count = if compact_psos.is_some() { 8 } else { 6 };
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        l.ResourceBarrier(&first[skip..]);
        if records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            match use_compact {
                None => {
                    let constants = [kind, stride, records, args_offset, flags, count_offset, 0, 0];
                    l.SetPipelineState(&c.pso);
                    l.SetComputeRoot32BitConstants(0, constants_count, constants.as_ptr().cast(), 0);
                    dispatch_groups(l, records.div_ceil(64));
                }
                Some(dst_count_off) if records <= TILE => {
                    let (compact_single, ..) = compact_psos.unwrap();
                    let constants = [
                        kind,
                        stride,
                        records,
                        args_offset,
                        flags,
                        count_offset,
                        dst_count_off as u32,
                        0,
                    ];
                    l.SetPipelineState(compact_single);
                    l.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    l.Dispatch(1, 1, 1);
                }
                Some(dst_count_off) => {
                    let (_, compact_pass1, scan_tiles, add_offsets, compact_scatter) =
                        compact_psos.unwrap();
                    let meta_off = layout.meta_offset as u32;
                    let tile_off = layout.tile_offset as u32;
                    let tiles = records.div_ceil(TILE);
                    let pass1_constants = [
                        kind,
                        stride,
                        records,
                        args_offset,
                        flags,
                        count_offset,
                        meta_off,
                        tile_off,
                    ];
                    l.SetPipelineState(compact_pass1);
                    l.SetComputeRoot32BitConstants(0, 8, pass1_constants.as_ptr().cast(), 0);
                    dispatch_groups(l, tiles);
                    l.ResourceBarrier(&[uav_barrier()]);

                    let last = layout.levels.len() - 1;
                    l.SetPipelineState(scan_tiles);
                    for (i, &(in_off, n)) in layout.levels.iter().enumerate() {
                        let out_off = if i == last {
                            dst_count_off as u32
                        } else {
                            layout.levels[i + 1].0
                        };
                        let scan_constants = [in_off, n, out_off, 0, 0, 0, 0, 0];
                        l.SetComputeRoot32BitConstants(0, 8, scan_constants.as_ptr().cast(), 0);
                        dispatch_groups(l, n.div_ceil(TILE));
                        l.ResourceBarrier(&[uav_barrier()]);
                    }

                    if last > 0 {
                        l.SetPipelineState(add_offsets);
                        for i in (0..last).rev() {
                            let (target_off, n) = layout.levels[i];
                            let block_off = layout.levels[i + 1].0;
                            let add_constants = [target_off, n, block_off, 0, 0, 0, 0, 0];
                            l.SetComputeRoot32BitConstants(0, 8, add_constants.as_ptr().cast(), 0);
                            dispatch_groups(l, n.div_ceil(TILE));
                            l.ResourceBarrier(&[uav_barrier()]);
                        }
                    }

                    let scatter_constants = [
                        kind,
                        stride,
                        records,
                        args_offset,
                        flags,
                        count_offset,
                        meta_off,
                        tile_off,
                    ];
                    l.SetPipelineState(compact_scatter);
                    l.SetComputeRoot32BitConstants(0, 8, scatter_constants.as_ptr().cast(), 0);
                    dispatch_groups(l, tiles);
                }
            }
        }
        l.ResourceBarrier(&second[skip..]);
    }
    true
}

/// Whether every draw served from a window is checked against a fresh
/// clamp of its record (`VICTORY_INDIRECT_WINDOW_CHECK=1`, with
/// `VICTORY_INDIRECT_WINDOW=1`; off by default). A window clamps its
/// records once, ahead of their draws; if anything wrote a record between
/// the window's clamp and its draw, the draw reads the old one. The check
/// clamps the record again just before the draw (into a buffer of its own,
/// charging a budget of its own) and compares the two on the card: the
/// counts are logged ("window check"), and the first record that differs in
/// each run of a list is named. For telling whether windows can be trusted
/// in a scene before they are on by default; it costs a clamp and a compare
/// per window draw.
pub(crate) fn window_check() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_INDIRECT_WINDOW_CHECK").is_some_and(|v| v == "1")
    })
}

/// The compare of [`window_check`]: `words` dwords of the record a window
/// clamped (`t0`) against the same record clamped just now (`t1`). Counts
/// the checks and the records that differ in `u0`, keeps the first record
/// that differs there (eight dwords of each), and zeroes the fresh clamp's
/// own budget (`u1`) for the next check.
const CHECK_SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
cbuffer Check : register(b0) {
    uint words;
};
ByteAddressBuffer window_record : register(t0);
ByteAddressBuffer fresh_record : register(t1);
RWByteAddressBuffer results : register(u0);
RWByteAddressBuffer fresh_budget : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    bool same = true;
    for (uint i = 0; i < words; i++) {
        if (window_record.Load(4 * i) != fresh_record.Load(4 * i)) {
            same = false;
        }
    }
    uint was;
    results.InterlockedAdd(4, 1, was);
    if (!same) {
        uint before;
        results.InterlockedAdd(0, 1, before);
        if (before == 0) {
            for (uint k = 0; k < min(words, 8u); k++) {
                results.Store(8 + 4 * k, window_record.Load(4 * k));
                results.Store(40 + 4 * k, fresh_record.Load(4 * k));
            }
        }
    }
    fresh_budget.Store3(0, uint3(0, 0, 0));
}
"#;

fn compare() -> Option<&'static Clamp> {
    static COMPARE: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    COMPARE
        .get_or_init(|| {
            let device = &host()?.device;
            match make_from(device, CHECK_SOURCE) {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no window check");
                    None
                }
            }
        })
        .as_ref()
}

/// A list's buffers for [`window_check`]: the fresh clamps' records and
/// budget (the card's memory), and the results (memory the CPU maps:
/// records that differed, checks, then the first differing record as the
/// window had it and as the fresh clamp made it, eight dwords each).
pub(crate) struct WindowCheck {
    args: ID3D12Resource,
    budget: ID3D12Resource,
    results: ID3D12Resource,
    data: usize,
}

// SAFETY: D3D12 objects are free-threaded; the mapping is read only at
// submission, when the list's last run has finished.
unsafe impl Send for WindowCheck {}
// SAFETY: as above.
unsafe impl Sync for WindowCheck {}

impl WindowCheck {
    pub(crate) fn new() -> Option<WindowCheck> {
        let host = host()?;
        let (results, data) = host.shared_buffer(256)?;
        // SAFETY: the mapping holds 256 bytes; nothing runs yet.
        unsafe { std::ptr::write_bytes(data, 0, 256) };
        Some(WindowCheck {
            args: host.uav_buffer(64 << 10)?,
            budget: host.uav_buffer(256)?,
            results,
            data: data as usize,
        })
    }

    /// What the list's last run found, zeroed for the next: checks, records
    /// that differed, and the first that did (as the window had it, as the
    /// fresh clamp made it). Only between runs of the list.
    pub(crate) fn take(&self) -> (u32, u32, [u32; 8], [u32; 8]) {
        let words = self.data as *mut u32;
        // SAFETY: the list's own mapped words; its last run has finished.
        unsafe {
            let at = |k: usize| words.add(k).read_volatile();
            let window = std::array::from_fn(|k| at(2 + k));
            let fresh = std::array::from_fn(|k| at(10 + k));
            let found = (at(1), at(0), window, fresh);
            std::ptr::write_bytes(words, 0, 18);
            found
        }
    }
}

/// Checks and differing records seen by [`window_check`] so far.
static CHECKED: [std::sync::atomic::AtomicU64; 2] =
    [const { std::sync::atomic::AtomicU64::new(0) }; 2];

/// [`window_check`] for a draw served from a window: the record of `kind`
/// at `src`, `step` bytes long, which the window clamped into `scratch` at
/// `at`, is clamped again into the list's check buffer and compared. Runs
/// compute work of ours, so the caller binds its pipeline again after it.
/// False if nothing was recorded.
pub(crate) fn check_hit(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    (kind, step): (u32, u32),
    scratch: &ID3D12Resource,
    at: u64,
) -> bool {
    const CHECK_USE: usize = usize::MAX - 0x78;
    let (Some(c), Some(cmp), Some(check)) = (clamp(), compare(), list.window_check()) else {
        return false;
    };
    if list.first_use(CHECK_USE) {
        // At this recording's submission: what the list's last run found.
        let data = check.data;
        let words = WindowCheck {
            args: check.args.clone(),
            budget: check.budget.clone(),
            results: check.results.clone(),
            data,
        };
        list.defer_host(move || {
            let (checked, differed, window, fresh) = words.take();
            if checked == 0 {
                return;
            }
            use std::sync::atomic::Ordering::Relaxed;
            let total = CHECKED[0].fetch_add(u64::from(checked), Relaxed) + u64::from(checked);
            let bad = CHECKED[1].fetch_add(u64::from(differed), Relaxed) + u64::from(differed);
            if differed > 0 {
                tracing::warn!(
                    differed,
                    checked,
                    ?window,
                    ?fresh,
                    "window check: a draw served from a window read a record that \
                     differs from a fresh clamp of it"
                );
            }
            if total / 10_000 != (total - u64::from(checked)) / 10_000 {
                tracing::info!(checked = total, differed = bad, "window check");
            }
        });
    }
    // SAFETY: live buffers.
    let (budget, results, fresh, window) = unsafe {
        (
            check.budget.GetGPUVirtualAddress(),
            check.results.GetGPUVirtualAddress(),
            check.args.GetGPUVirtualAddress(),
            scratch.GetGPUVirtualAddress() + at,
        )
    };
    // The record again, by the window's rules, into the check's buffer
    // (common before and after every check).
    let common = D3D12_RESOURCE_STATE_COMMON;
    let spec = (kind, step, 1);
    if !record_clamp_as(c, &list.list, (&check.args, common), budget, src, None, spec, true) {
        return false;
    }
    let args = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
    let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
    let words = step / 4;
    // SAFETY: an open list; the compare's own root signature and pipeline;
    // the scratch buffer is readable as arguments (and stays so), the
    // check's buffer was left so by the clamp above.
    unsafe {
        list.list.ResourceBarrier(&[
            transition(scratch, args, D3D12_RESOURCE_STATES(args.0 | read.0)),
            transition(&check.args, args, read),
        ]);
        list.list.SetComputeRootSignature(&cmp.root);
        list.list.SetPipelineState(&cmp.pso);
        list.list.SetComputeRoot32BitConstants(0, 1, (&raw const words).cast(), 0);
        list.list.SetComputeRootShaderResourceView(1, window);
        list.list.SetComputeRootShaderResourceView(2, fresh);
        list.list.SetComputeRootUnorderedAccessView(3, results);
        list.list.SetComputeRootUnorderedAccessView(4, budget);
        list.list.Dispatch(1, 1, 1);
        list.list.ResourceBarrier(&[
            transition(scratch, D3D12_RESOURCE_STATES(args.0 | read.0), args),
            transition(&check.args, read, common),
        ]);
    }
    true
}

fn uav_barrier() -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: std::mem::ManuallyDrop::new(None),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Foundation::HANDLE;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::Interface;

    const WORKER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer hits : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint prev;
    hits.InterlockedAdd(0, 1, prev);
}
"#;

    struct WarpHarness {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        info_queue: Option<ID3D12InfoQueue>,
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // SAFETY: plain DXGI WARP query and D3D12 test harness setup.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let info_queue = device.cast::<ID3D12InfoQueue>().ok();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_for(&device).expect("clamp shader must compile on WARP");
                let worker = compile_cs(&device, WORKER_CS).expect("worker shader must compile");
                Some(Self {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    info_queue,
                    clamp,
                    worker,
                })
            }
        }

        /// CPU-mapped L0 write-back buffer matching `Shadow`'s heap type.
        fn shared_buffer(&self, size: u64) -> (ID3D12Resource, *mut u8, u64) {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(4),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            let mut data = std::ptr::null_mut();
            // SAFETY: plain buffer creation and persistent CPU mapping.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
                let buffer = buffer.unwrap();
                buffer.Map(0, None, Some(&mut data)).unwrap();
                let gpu = buffer.GetGPUVirtualAddress();
                (buffer, data.cast(), gpu)
            }
        }

        /// Device-local `DEFAULT`-heap UAV scratch buffer.
        fn default_uav(&self, size: u64) -> ID3D12Resource {
            let heap = D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: size.max(64 << 10),
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut buffer: Option<ID3D12Resource> = None;
            // SAFETY: plain device-local buffer creation.
            unsafe {
                self.device
                    .CreateCommittedResource(
                        &heap,
                        D3D12_HEAP_FLAG_NONE,
                        &desc,
                        D3D12_RESOURCE_STATE_COMMON,
                        None,
                        &mut buffer,
                    )
                    .unwrap();
            }
            buffer.unwrap()
        }

        fn signature(&self, kind: u32, stride: u32) -> ID3D12CommandSignature {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: stride,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: valid single-argument command signature description.
            unsafe {
                self.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        }

        fn submit_and_wait(&mut self) {
            self.submitted += 1;
            // SAFETY: closing, executing, waiting on fence, and resetting our list.
            unsafe {
                self.list.Close().unwrap();
                let cmd: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(cmd)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence
                    .SetEventOnCompletion(self.submitted, self.event)
                    .unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
            if let Some(queue) = &self.info_queue {
                // SAFETY: `queue` is a live `ID3D12InfoQueue` on our test device.
                let errors = unsafe {
                    let count = queue.GetNumStoredMessagesAllowedByRetrievalFilter();
                    let mut found = Vec::new();
                    for idx in 0..count {
                        let mut len = 0usize;
                        if queue.GetMessage(idx, None, &mut len).is_ok() && len > 0 {
                            let mut buf = vec![0u64; len.div_ceil(8)];
                            let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                            if queue.GetMessage(idx, Some(msg), &mut len).is_ok()
                                && matches!(
                                    (*msg).Severity,
                                    D3D12_MESSAGE_SEVERITY_CORRUPTION
                                        | D3D12_MESSAGE_SEVERITY_ERROR
                                )
                            {
                                let text = std::slice::from_raw_parts(
                                    (*msg).pDescription,
                                    (*msg).DescriptionByteLength,
                                );
                                found.push(
                                    String::from_utf8_lossy(text)
                                        .trim_end_matches('\0')
                                        .to_owned(),
                                );
                            }
                        }
                    }
                    queue.ClearStoredMessages();
                    found
                };
                assert!(
                    errors.is_empty(),
                    "D3D12 debug layer reported error(s): {errors:?}"
                );
            }
        }

        fn copy_indirect(
            &self,
            dst: &ID3D12Resource,
            dst_off: u64,
            src: &ID3D12Resource,
            src_off: u64,
            bytes: u64,
            restore: Option<D3D12_RESOURCE_STATES>,
        ) {
            // SAFETY: open list; `src` is in `INDIRECT_ARGUMENT`, `dst` is a shared readback buffer.
            unsafe {
                self.list.ResourceBarrier(&[transition(
                    src,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                self.list.CopyBufferRegion(dst, dst_off, src, src_off, bytes);
                if let Some(after) = restore {
                    self.list.ResourceBarrier(&[transition(
                        src,
                        D3D12_RESOURCE_STATE_COPY_SOURCE,
                        after,
                    )]);
                }
            }
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let code = compile_hlsl(source, s!("worker"), s!("main"))?;
        // SAFETY: `code` owns valid DXBC bytes with an embedded root signature.
        unsafe {
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = make_compute_pso(device, &root, &code)?;
            Ok(Clamp {
                root,
                pso,
                compact_single: None,
                compact_pass1: None,
                scan_tiles: None,
                add_offsets: None,
                compact_scatter: None,
            })
        }
    }

    /// The compare of copies in the card's memory (`crate::host`), on the
    /// software adapter. Told 1 it flags the pieces whose bytes are not
    /// what was hashed last and leaves the hashes: a piece a shader changed
    /// and a later one changed back then looks untouched, which is right
    /// only if the piece was fetched after both. Told 2 (submissions sent
    /// ahead with the relay, whose pieces are fetched flight by flight) it
    /// stores the hash it made: each compare flags what changed since the
    /// compare before it, also a change back.
    #[test]
    fn warp_card_compare_stores_the_hash_it_made_when_told() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, crate::host::CARD_COMPARE).expect("the card compare must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = warp.shared_buffer(64);
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mapping holds `size` bytes.
        unsafe { std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size) };
        // One pass over the three pieces; what it left in `differs` (7s
        // where it stored nothing).
        let run = |warp: &mut WarpHarness, mode: u32| -> [u32; 3] {
            // SAFETY: our open list, live buffers, the compare's own root
            // signature; the mappings hold what is read and written.
            unsafe {
                for piece in 0..PIECES as usize {
                    differs_ptr.cast::<u32>().add(piece).write_volatile(7);
                }
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, 0, size as u32, PIECES, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(PIECES.div_ceil(64), 1, 1);
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            [0, 1, 2].map(|piece| unsafe { differs_ptr.cast::<u32>().add(piece).read_volatile() })
        };
        let at = PIECE + 1234;
        // SAFETY (each): inside the mapping.
        let set = |value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // 0: the hashes are stored, nothing is flagged.
        assert_eq!(run(&mut warp, 0), [7, 7, 7]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // A change in the second piece: flagged, and with 1 flagged again
        // (the hash stays the old one).
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        assert_eq!(run(&mut warp, 1), [0, 1, 0]);
        // Changed back: with 1 the piece looks untouched.
        set(content[at]);
        assert_eq!(run(&mut warp, 1), [0, 0, 0]);
        // With 2 the change is flagged once, and the hash is the new one.
        set(content[at] ^ 0x40);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
        // Changed back: flagged, where 1 saw nothing.
        set(content[at]);
        assert_eq!(run(&mut warp, 2), [0, 1, 0]);
        assert_eq!(run(&mut warp, 2), [0, 0, 0]);
    }

    /// The fetch inside flights (`crate::host::CARD_FETCH`), on the software
    /// adapter, in one list behind the compare as `Host::card_before`
    /// records it: each flagged piece takes a ticket and, while it is
    /// within the slots of the lowest ticket in the air, its bytes go to
    /// its slot; an unflagged piece's entry says so; a ticket past the
    /// slots is written but copies nothing.
    #[test]
    fn warp_card_fetch_copies_flagged_pieces_to_their_slots() {
        use crate::host::{CARD_COMPARE, CARD_FETCH, INFLIGHT_NONE, INFLIGHT_SLOT};
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let compare = compile_cs(&warp.device, CARD_COMPARE).expect("the card compare must compile on WARP");
        let fetch = compile_cs(&warp.device, CARD_FETCH).expect("the card fetch must compile on WARP");
        const PIECE: usize = 64 << 10;
        const PIECES: u32 = 3;
        // The copy starts 40 bytes into its piece 0: piece 0 is short, the
        // others straddle title pieces and start off a multiple of 16.
        const SKEW: u32 = 40;
        let size = PIECE * PIECES as usize;
        let (_now, now_ptr, now_gpu) = warp.shared_buffer(size as u64);
        let (_hashes, _, hashes_gpu) = warp.shared_buffer(16 * u64::from(PIECES + 1));
        let (_differs, _, differs_gpu) = warp.shared_buffer(64);
        let (_table, table_ptr, table_gpu) = warp.shared_buffer(4 + 64);
        let slots = 1u32;
        let (_fetched, fetched_ptr, fetched_gpu) = warp.shared_buffer(INFLIGHT_SLOT * u64::from(slots));
        let content: Vec<u8> = (0..size).map(|i| (i * 31 + i / 251) as u8).collect();
        // SAFETY: the mappings hold what is written.
        unsafe {
            std::ptr::copy_nonoverlapping(content.as_ptr(), now_ptr, size);
            table_ptr.cast::<u32>().write_volatile(0);
        }
        let pieces = PIECES + 1;
        let run = |warp: &mut WarpHarness, mode: u32, landed: u32| -> Vec<u32> {
            // SAFETY: our open list, live buffers, the shaders' own root
            // signatures; the mappings hold what is read and written.
            unsafe {
                warp.list.SetComputeRootSignature(&compare.root);
                warp.list.SetPipelineState(&compare.pso);
                let constants = [0u32, SKEW, size as u32, pieces, 0, mode];
                warp.list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                warp.list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                warp.list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                warp.list.Dispatch(1, 1, 1);
                if mode == 2 {
                    warp.list.ResourceBarrier(&[uav_barrier()]);
                    warp.list.SetComputeRootSignature(&fetch.root);
                    warp.list.SetPipelineState(&fetch.pso);
                    // Two groups for four pieces: the loop over pieces.
                    let constants = [0u32, SKEW, size as u32, pieces, 0, landed, slots, 2];
                    warp.list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                    warp.list.SetComputeRootShaderResourceView(1, now_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(2, differs_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(3, table_gpu);
                    warp.list.SetComputeRootUnorderedAccessView(4, fetched_gpu);
                    warp.list.Dispatch(2, 1, 1);
                }
            }
            warp.submit_and_wait();
            // SAFETY: as above.
            (0..=pieces as usize).map(|i| unsafe { table_ptr.cast::<u32>().add(i).read_volatile() }).collect()
        };
        // SAFETY (each): inside the mapping of the copy.
        let set = |at: usize, value: u8| unsafe { now_ptr.add(at).write_volatile(value) };
        // SAFETY (each): the slot's mapping holds `INFLIGHT_SLOT` bytes.
        let slot = |len: usize| unsafe { std::slice::from_raw_parts(fetched_ptr, len).to_vec() };
        run(&mut warp, 0, 0);
        // Piece 2 of the copy (title piece 2: bytes 2 * 64 KiB - 40 on), its
        // slot from the multiple of 16 below to the one above its end.
        let from = 2 * PIECE - SKEW as usize;
        let widened = from & !15;
        assert_ne!(widened, from);
        set(from + 100, content[from + 100] ^ 0x40);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table, [1, INFLIGHT_NONE, INFLIGHT_NONE, 0, INFLIGHT_NONE]);
        let mut wanted = content[widened..widened + INFLIGHT_SLOT as usize].to_vec();
        wanted[from + 100 - widened] ^= 0x40;
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted, "piece 2 in slot 0");
        // Two pieces flagged with one slot, ticket 0 not landed: tickets 1
        // and 2 get none and the slot keeps what it held.
        set(5, content[5] ^ 1);
        set(PIECE + 7, content[PIECE + 7] ^ 1);
        let table = run(&mut warp, 2, 0);
        assert_eq!(table[0], 3);
        let mut tickets = [table[1], table[2]];
        tickets.sort_unstable();
        assert_eq!(tickets, [1, 2]);
        assert_eq!(&table[3..], [INFLIGHT_NONE, INFLIGHT_NONE]);
        assert_eq!(slot(INFLIGHT_SLOT as usize), wanted);
        // Every ticket up to 2 landed: the next one, 3, gets slot 0. Piece
        // 0 starts at the copy's start.
        set(9, content[9] ^ 2);
        let table = run(&mut warp, 2, 3);
        assert_eq!(table, [4, 3, INFLIGHT_NONE, INFLIGHT_NONE, INFLIGHT_NONE]);
        let mut piece0 = content[..PIECE - SKEW as usize].to_vec();
        piece0[5] ^= 1;
        piece0[9] ^= 2;
        assert_eq!(slot(PIECE - SKEW as usize), piece0);
    }

    /// The clamp compiles with the system's compiler and the host takes it.
    #[test]
    fn argument_clamp_builds() {
        if super::host().is_none() {
            return; // no host GPU here
        }
        assert!(super::clamp().is_some());
    }

    #[test]
    fn rejects_unaligned_or_undersized_layouts() {
        assert_eq!(validate_layout((0, 16, 4), 0, None), Some(64));
        assert_eq!(validate_layout((1, 20, 4), 64, Some(128)), Some(80));
        assert_eq!(validate_layout((2, 12, 0), 0, Some(0)), Some(0));
        assert_eq!(validate_layout((10, 16, 2), 8, None), Some(32));
        // Undersized strides for each kind.
        assert_eq!(validate_layout((0, 12, 1), 0, None), None);
        assert_eq!(validate_layout((1, 16, 1), 0, None), None);
        assert_eq!(validate_layout((2, 8, 1), 0, None), None);
        assert_eq!(validate_layout((10, 8, 1), 0, None), None);
        // Unknown kind.
        assert_eq!(validate_layout((3, 16, 1), 0, None), None);
        // Unaligned stride, source offset, or count offset.
        assert_eq!(validate_layout((0, 18, 1), 0, None), None);
        assert_eq!(validate_layout((0, 16, 1), 2, None), None);
        assert_eq!(validate_layout((0, 16, 1), 0, Some(6)), None);
        // Total byte size exceeding 32-bit ByteAddressBuffer indexing.
        assert_eq!(validate_layout((1, 20, u32::MAX), 0, None), None);
    }

    #[test]
    fn scratch_layout_handles_in_slot_single_multi_tile_and_overflow() {
        let in_slot = scratch_layout(1024, 64, true);
        assert_eq!(in_slot.total_bytes, 1024);
        assert_eq!(in_slot.count_offset, None);

        let one = scratch_layout(16, 1, false);
        assert_eq!(one.total_bytes, 16);
        assert_eq!(one.count_offset, None);

        let single = scratch_layout(16 * 256, 256, false);
        assert_eq!(single.count_offset, Some(16 * 256));
        assert_eq!(single.total_bytes, 16 * 256 + 4);
        assert!(single.levels.is_empty());

        let one_level = scratch_layout(16 * 1000, 1000, false);
        assert_eq!(one_level.count_offset, Some(16_000));
        assert_eq!(one_level.meta_offset, 16_004);
        assert_eq!(one_level.tile_offset, 20_004);
        assert_eq!(one_level.levels, vec![(20_004, 4)]);
        assert_eq!(one_level.total_bytes, 20_020);

        let two_level = scratch_layout(16 * 70_000, 70_000, false);
        assert_eq!(two_level.levels.len(), 2);
        assert_eq!(two_level.levels[0].1, 274);
        assert_eq!(two_level.levels[1].1, 2);

        // If auxiliary buffers would push total size past 4 GiB, fall back cleanly.
        let near_cap = u64::from(u32::MAX) - 16;
        let fallback = scratch_layout(near_cap, 1024, false);
        assert_eq!(fallback.total_bytes, near_cap);
        assert_eq!(fallback.count_offset, None);
    }

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied and compacted
    ///     into `dst_args` at offset `0`, with compacted count written at `count_off`.
    #[test]
    fn warp_count_buffer_ignores_garbage_tail_and_preserves_source() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 24; // 6 dwords: 3 dispatch group counts + 3 extra payload dwords
        const MAX: u32 = 4096;
        const ARGS_OFFSET: usize = 64; // non-zero offset with sentinel prefix
        const COUNT_OFFSET_1: usize = 128;
        const COUNT_OFFSET_2: usize = 192;

        let layout = scratch_layout(u64::from(STRIDE) * u64::from(MAX), MAX, false);
        let count_off = layout.count_offset.unwrap();
        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8 + 16) as u64);
        let dst_buf = warp.default_uav(layout.total_bytes);
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Fill the entire source buffer with non-zero sentinel bytes, then
        // populate 3 valid records at `ARGS_OFFSET` and 4,093 huge garbage
        // records in `3..MAX` that would exhaust `BUDGET` ~2,000x over if read.
        let mut initial_src = vec![0xA5u8; src_bytes];
        let write_record = |buf: &mut [u8], idx: usize, words: [u32; 6]| {
            let base = ARGS_OFFSET + idx * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                buf[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        };
        write_record(
            &mut initial_src,
            0,
            [3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666],
        );
        write_record(
            &mut initial_src,
            1,
            [5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC],
        );
        // Record 2 exceeds 65,535 in X -> clamped to (65535, 1, 1).
        write_record(
            &mut initial_src,
            2,
            [100_000, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567],
        );
        for slot in 3..MAX as usize {
            // Each garbage slot would clamp to 65535 * 16 * 1 = 1,048,560 groups
            // (`cost = 67,107,840 ~= BUDGET / 2`), exhausting `BUDGET` immediately!
            write_record(
                &mut initial_src,
                slot,
                [0xFFFF_FFFF, 16, 1, 0xDEAD_BEEF, 0xBAD0_C0DE, slot as u32],
            );
        }

        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8 + 16) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records and compacted count out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list.
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_1 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list
                .ExecuteIndirect(&sig, MAX, &dst_buf, 0, Some(&dst_buf), count_off);
        }

        // Snapshot `dst_buf` after Call #1 (leave `dst_buf` in `INDIRECT_ARGUMENT`).
        warp.copy_indirect(
            &readback_buf,
            0,
            &dst_buf,
            0,
            u64::from(STRIDE * 5),
            Some(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        );
        warp.copy_indirect(
            &readback_buf,
            u64::from(STRIDE * 8),
            &dst_buf,
            count_off,
            4,
            Some(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        );

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). The scratch buffer goes straight from
        // `INDIRECT_ARGUMENT` back to `UNORDERED_ACCESS`.
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
            budget_gpu,
            (&src_buf, ARGS_OFFSET as u64),
            Some((&count_buf, COUNT_OFFSET_2 as u64)),
            (KIND_DISPATCH, STRIDE, MAX),
            false,
        ));
        // SAFETY: open command list, valid worker PSO/root and buffers.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list
                .ExecuteIndirect(&sig, MAX, &dst_buf, 0, Some(&dst_buf), count_off);
        }
        warp.copy_indirect(
            &readback_buf,
            u64::from(STRIDE * 8) + 4,
            &dst_buf,
            count_off,
            4,
            None,
        );

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words, count_1, count_2) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            let c1 = readback_ptr.add((STRIDE * 8) as usize).cast::<u32>().read_volatile();
            let c2 = readback_ptr.add((STRIDE * 8) as usize + 4).cast::<u32>().read_volatile();
            (hits, spent, after_src, after_count, words, c1, c2)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
        assert_eq!(count_1, 3);
        assert_eq!(count_2, 2);
        assert_eq!(
            spent,
            expected_groups * 64,
            "budget must reflect only active records below countBuffer"
        );
        assert_eq!(
            after_src, initial_src,
            "src_args must never be mutated in-place"
        );
        assert_eq!(
            after_count, initial_count,
            "count_buf must never be mutated"
        );

        // Check `dst_args` records 0..3: clamped group counts + preserved extra stride words.
        assert_eq!(
            &clamped_words[0..6],
            &[3, 2, 1, 0x1111_2222, 0x3333_4444, 0x5555_6666]
        );
        assert_eq!(
            &clamped_words[6..12],
            &[5, 1, 1, 0x7777_8888, 0x9999_AAAA, 0xBBBB_CCCC]
        );
        assert_eq!(
            &clamped_words[12..18],
            &[65_535, 1, 1, 0xDDDD_EEEE, 0xFFFF_0001, 0x0123_4567]
        );
        // Inactive slot 3 was never copied from `src_args`.
        assert_ne!(clamped_words[21], 0xDEAD_BEEF);
    }

    /// A window of mesh records clamped ahead: the filled ones stay in-slot and
    /// are charged, the slots of all ones come out empty and charge nothing;
    /// clamped directly as a non-windowed multi-record batch (`ahead = false`),
    /// empty records are compacted out while non-empty slots clamp and charge.
    #[test]
    fn warp_window_leaves_unfilled_slots_empty_and_uncharged() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 32;
        const RECORDS: u32 = 16;
        let filled = [5u32, 0, 519];
        let total = (STRIDE * RECORDS) as usize;
        let mut src = vec![0xffu8; total];
        for (i, groups) in filled.into_iter().enumerate() {
            let words = [groups, 1, 1, 0xAB00 + i as u32];
            for (w, word) in words.into_iter().enumerate() {
                let at = i * STRIDE as usize + w * 4;
                src[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        for ahead in [true, false] {
            let layout = scratch_layout(total as u64, RECORDS, ahead);
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(layout.total_bytes);
            let dst_buf = warp.default_uav(layout.total_bytes);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, layout.total_bytes as usize);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_MESH, STRIDE, RECORDS),
                ahead,
            ));
            warp.copy_indirect(&readback_buf, 0, &dst_buf, 0, layout.total_bytes, None);
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, mesh_all, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    (budget_ptr as *const u32).add(2).read_volatile(),
                    std::slice::from_raw_parts(
                        readback_ptr.cast::<u32>().cast_const(),
                        layout.total_bytes as usize / 4,
                    )
                    .to_vec(),
                )
            };
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for (i, groups) in filled.into_iter().enumerate() {
                    let rec = &out[i * 8..i * 8 + 4];
                    assert_eq!(rec, [groups, 1, 1, 0xAB00 + i as u32], "filled record {i}");
                }
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
                assert_eq!(mesh_all, 5 + 519);
            } else {
                // Record 1 (0 groups) is compacted out: 5, 519, then 13x 65,535.
                let count_off = layout.count_offset.unwrap();
                assert_eq!(out[count_off as usize / 4], 15);
                assert_eq!(&out[0..4], [5, 1, 1, 0xAB00]);
                assert_eq!(&out[8..12], [519, 1, 1, 0xAB02]);
                for slot in 2..15usize {
                    assert_eq!(&out[slot * 8..slot * 8 + 4], [65_535, 1, 1, u32::MAX]);
                }
                assert_eq!(spent, charged_filled + 13 * 65535 * 128, "clamped one by one, each slot is charged the most");
                assert_eq!(largest, u32::MAX);
            }
        }
    }

    #[test]
    fn warp_budget_saturates_without_wraparound_and_preserves_draw_fields() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_INDEXED: u32 = 1;
        const STRIDE: u32 = 24; // 5 indexed-draw words + 1 trailing custom word
        const RECORDS: u32 = 128; // 2 waves of 64: old InterlockedAdd wrapped 2^32 at 64!
        const BUDGET_CAP: u32 = 1 << 27;

        let total_bytes = (STRIDE * RECORDS) as usize;
        let layout = scratch_layout(total_bytes as u64, RECORDS, false);
        let count_off = layout.count_offset.unwrap();
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(layout.total_bytes);
        let dst_buf = warp.default_uav(layout.total_bytes);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            // The first record is no draw (2^24 indices >= MOST_DRAW): it runs
            // empty, charges nothing, and is compacted out.
            let words: [u32; 6] = [
                if i == 0 { 1 << 24 } else { 1 << 22 }, // IndexCount
                if i == 0 { 1 } else { u32::MAX }, // InstanceCount -> clamps to 16 (cost = 1 << 26)
                1000 + i as u32,             // StartIndexLocation
                (-200i32 + i as i32) as u32, // BaseVertexLocation
                5000 + i as u32,             // StartInstanceLocation
                0xCAFE_0000 | (i as u32),    // Extra stride word
            ];
            let base = i * (STRIDE as usize);
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }

        // SAFETY: mapped buffers have sufficient capacity.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, total_bytes);
            (budget_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, layout.total_bytes as usize);
        }

        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_INDEXED, STRIDE, RECORDS),
            false,
        ));

        warp.copy_indirect(&readback_buf, 0, &dst_buf, 0, layout.total_bytes, None);
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                layout.total_bytes as usize / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        // Record 0 (MOST_DRAW) and records 3..128 (budget-zeroed) are compacted
        // out, leaving only records 1 and 2 in exact submission order.
        assert_eq!(out_words[count_off as usize / 4], 2);
        for (slot, orig) in [(0usize, 1u32), (1usize, 2u32)] {
            let rec = &out_words[slot * 6..(slot + 1) * 6];
            assert_eq!((rec[0], rec[1]), (1 << 22, 16));
            assert_eq!(rec[2], 1000 + orig);
            assert_eq!(rec[3], (-200i32 + orig as i32) as u32);
            assert_eq!(rec[4], 5000 + orig);
            assert_eq!(rec[5], 0xCAFE_0000 | orig);
        }
    }

    #[test]
    fn warp_compaction_preserves_order_and_executes_multi_tile() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 16; // 3 dispatch words + 1 unique order tag dword
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // Exercises:
        // - 250 records: single-tile (`compact_single`)
        // - 1_000 records: multi-tile 1-level scan (`compact_pass1` + `scan_tiles` + `compact_scatter`)
        // - 70_000 records: multi-tile 2-level scan (`> 256 * 256`, exercises `add_offsets` too)
        for &records in &[250u32, 1_000u32, 70_000u32] {
            let args_bytes = u64::from(STRIDE) * u64::from(records);
            let layout = scratch_layout(args_bytes, records, false);
            let count_off = layout.count_offset.unwrap();
            let (src_buf, src_ptr, _) = warp.shared_buffer(args_bytes);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(args_bytes + 4);
            let dst_buf = warp.default_uav(layout.total_bytes);

            let mut initial_src = vec![0u8; args_bytes as usize];
            let mut expected_kept: Vec<(u32, u32, u32, u32)> = Vec::new();
            let mut expected_hits = 0u32;

            for i in 0..records {
                // Mix empty patterns around tile boundaries (255, 256, 257, 511, 512, 513)
                // and sparse active records so every tile has varied density.
                let pattern = match i % 7 {
                    0 => (0, 1, 1),
                    1 => (2, 0, 1),
                    2 => (2, 1, 0),
                    3 if i % 14 == 3 => (1, 1, 1),
                    4 if i == 256 || i == 512 || i == 65_537 => (3, 1, 1),
                    5 if i == 513 => (70_000, 1, 1), // clamped to 65,535
                    _ if i % 19 == 0 => (2, 1, 1),
                    _ => (0, 0, 0),
                };
                let tag = 0x1000_0000u32 + i;
                let base = (i * STRIDE) as usize;
                for (w, val) in [pattern.0, pattern.1, pattern.2, tag].into_iter().enumerate() {
                    initial_src[base + w * 4..base + (w + 1) * 4]
                        .copy_from_slice(&val.to_le_bytes());
                }
                let clamped = (pattern.0.min(65_535), pattern.1.min(65_535), pattern.2.min(65_535));
                if clamped.0 > 0 && clamped.1 > 0 && clamped.2 > 0 {
                    expected_hits += clamped.0 * clamped.1 * clamped.2;
                    expected_kept.push((clamped.0, clamped.1, clamped.2, tag));
                }
            }

            // SAFETY: mapped buffers with exact sizes above.
            unsafe {
                std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, args_bytes as usize);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                (hits_ptr as *mut u32).write_volatile(0);
                std::ptr::write_bytes(readback_ptr, 0xCC, (args_bytes + 4) as usize);
            }

            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_DISPATCH, STRIDE, records),
                false,
            ));
            // SAFETY: open list, valid worker PSO/root, compacted count inside `dst_buf`.
            unsafe {
                warp.list.SetComputeRootSignature(&warp.worker.root);
                warp.list.SetPipelineState(&warp.worker.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                warp.list.ExecuteIndirect(
                    &sig,
                    records,
                    &dst_buf,
                    0,
                    Some(&dst_buf),
                    count_off,
                );
            }
            warp.copy_indirect(&readback_buf, 0, &dst_buf, 0, args_bytes + 4, None);
            warp.submit_and_wait();

            // SAFETY: GPU work completed.
            let (hits, spent, after_src, out_words) = unsafe {
                let hits = (hits_ptr as *const u32).read_volatile();
                let spent = (budget_ptr as *const u32).read_volatile();
                let after_src =
                    std::slice::from_raw_parts(src_ptr.cast_const(), args_bytes as usize).to_vec();
                let out_words = std::slice::from_raw_parts(
                    readback_ptr.cast::<u32>().cast_const(),
                    (args_bytes as usize + 4) / 4,
                )
                .to_vec();
                (hits, spent, after_src, out_words)
            };

            let compacted_count = out_words[count_off as usize / 4];
            assert_eq!(
                compacted_count,
                expected_kept.len() as u32,
                "records={records}: compacted count mismatch"
            );
            assert_eq!(hits, expected_hits, "records={records}: worker hits mismatch");
            assert_eq!(spent, expected_hits * 64, "records={records}: budget mismatch");
            assert_eq!(after_src, initial_src, "records={records}: src mutated");

            for (slot, &(gx, gy, gz, tag)) in expected_kept.iter().enumerate() {
                let rec = &out_words[slot * 4..(slot + 1) * 4];
                assert_eq!(
                    rec,
                    &[gx, gy, gz, tag],
                    "records={records}: compacted slot {slot} out of order or corrupted"
                );
            }
        }

        // Also verify multi-tile order-preserving compaction for DRAW (`kind = 0`)
        // and MESH (`kind = 10`, including mesh diagnostics words 4 and 8).
        for &(kind, stride, records) in &[(0u32, 20u32, 320u32), (10u32, 16u32, 320u32)] {
            let args_bytes = u64::from(stride * records);
            let layout = scratch_layout(args_bytes, records, false);
            let count_off = layout.count_offset.unwrap();
            let (src_buf, src_ptr, _) = warp.shared_buffer(args_bytes);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(args_bytes + 4);
            let dst_buf = warp.default_uav(layout.total_bytes);
            let dwords = (stride / 4) as usize;

            let mut expected: Vec<Vec<u32>> = Vec::new();
            let mut expected_spent = 0u32;
            let mut mesh_max = 0u32;
            let mut mesh_sum = 0u32;
            // SAFETY: mapped buffers with exact sizes.
            unsafe {
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, (args_bytes + 4) as usize);
                for i in 0..records {
                    let rec: Vec<u32> = if kind == 0 {
                        let (verts, inst) = match i % 4 {
                            0 => (0, 2),
                            1 => (6, 0),
                            _ => (3 + (i % 7), 1 + (i % 3)),
                        };
                        if verts > 0 && inst > 0 {
                            expected_spent += verts * inst;
                            expected.push(vec![verts, inst, i, i * 2, 0xD000_0000 | i]);
                        }
                        vec![verts, inst, i, i * 2, 0xD000_0000 | i]
                    } else {
                        let gx = if i % 3 == 0 { 0 } else if i == 257 { 70_000 } else { 1 + (i % 9) };
                        mesh_max = mesh_max.max(gx);
                        mesh_sum += gx.min(1 << 24);
                        let cgx = gx.min(65_535);
                        if cgx > 0 {
                            expected_spent += cgx * 128;
                            expected.push(vec![cgx, 1, 1, 0xE000_0000 | i]);
                        }
                        vec![gx, 4, 2, 0xE000_0000 | i]
                    };
                    let base = src_ptr.add((i * stride) as usize).cast::<u32>();
                    for (w, &val) in rec.iter().enumerate() {
                        base.add(w).write_volatile(val);
                    }
                }
            }

            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (kind, stride, records),
                false,
            ));
            warp.copy_indirect(&readback_buf, 0, &dst_buf, 0, args_bytes + 4, None);
            warp.submit_and_wait();

            // SAFETY: GPU completed.
            let (spent, diag_max, diag_sum, out_words) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    (budget_ptr as *const u32).add(2).read_volatile(),
                    std::slice::from_raw_parts(
                        readback_ptr.cast::<u32>().cast_const(),
                        (args_bytes as usize + 4) / 4,
                    )
                    .to_vec(),
                )
            };
            assert_eq!(out_words[count_off as usize / 4], expected.len() as u32);
            assert_eq!(spent, expected_spent);
            if kind == 10 {
                assert_eq!(diag_max, mesh_max);
                assert_eq!(diag_sum, mesh_sum);
            }
            for (slot, exp) in expected.iter().enumerate() {
                assert_eq!(
                    &out_words[slot * dwords..(slot + 1) * dwords],
                    exp.as_slice(),
                    "kind={kind}: compacted slot {slot} mismatch"
                );
            }
        }
    }

    #[test]
    fn warp_compaction_zero_records_all_empty_and_zero_count() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const KIND_DRAW: u32 = 0;
        const KIND_INDEXED: u32 = 1;
        const KIND_MESH: u32 = 10;
        const STRIDE: u32 = 20;
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        // 1. All-empty records across all 4 indirect kinds (`records = 300`, multi-tile).
        for (kind, words) in [
            (KIND_DRAW, [0u32, 4, 10, 20, 0xAAAA]),
            (KIND_DRAW, [4u32, 0, 10, 20, 0xBBBB]),
            (KIND_INDEXED, [0u32, 3, 1, 2, 3]),
            (KIND_INDEXED, [5u32, 0, 1, 2, 3]),
            (KIND_DISPATCH, [0u32, 2, 3, 0, 0]),
            (KIND_DISPATCH, [4u32, 0, 3, 0, 0]),
            (KIND_DISPATCH, [4u32, 2, 0, 0, 0]),
            (KIND_MESH, [0u32, 1, 1, 0, 0]),
            (KIND_MESH, [4u32, 0, 1, 0, 0]),
        ] {
            const RECORDS: u32 = 300;
            let args_bytes = u64::from(STRIDE * RECORDS);
            let layout = scratch_layout(args_bytes, RECORDS, false);
            let count_off = layout.count_offset.unwrap();
            let (src_buf, src_ptr, _) = warp.shared_buffer(args_bytes);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(16);
            let dst_buf = warp.default_uav(layout.total_bytes);

            // SAFETY: mapped buffers with exact sizes.
            unsafe {
                for i in 0..RECORDS as usize {
                    let dst_words = src_ptr.add(i * STRIDE as usize).cast::<u32>();
                    for (w, val) in words.into_iter().enumerate() {
                        dst_words.add(w).write_volatile(val);
                    }
                }
                std::ptr::write_bytes(budget_ptr, 0, 256);
                (hits_ptr as *mut u32).write_volatile(0);
                (readback_ptr as *mut u32).write_volatile(0xDEAD_BEEF);
            }

            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (kind, STRIDE, RECORDS),
                false,
            ));
            if kind == KIND_DISPATCH {
                // SAFETY: open list, valid worker PSO/root and compacted count buffer.
                unsafe {
                    warp.list.SetComputeRootSignature(&warp.worker.root);
                    warp.list.SetPipelineState(&warp.worker.pso);
                    warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                    warp.list.ExecuteIndirect(
                        &sig,
                        RECORDS,
                        &dst_buf,
                        0,
                        Some(&dst_buf),
                        count_off,
                    );
                }
            }
            warp.copy_indirect(&readback_buf, 0, &dst_buf, count_off, 4, None);
            warp.submit_and_wait();

            // SAFETY: GPU completed.
            let (compacted, spent, hits) = unsafe {
                (
                    (readback_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).read_volatile(),
                    (hits_ptr as *const u32).read_volatile(),
                )
            };
            assert_eq!(compacted, 0, "all-empty batch for kind {kind} must compact to 0");
            assert_eq!(spent, 0);
            assert_eq!(hits, 0);
        }

        // 2. Count buffer = 0 with huge garbage in `src`, and count buffer = 260
        // crossing the 256-record tile boundary with sparse empty slots.
        const RECORDS: u32 = 300;
        let args_bytes = u64::from(STRIDE * RECORDS);
        let layout = scratch_layout(args_bytes, RECORDS, false);
        let count_off = layout.count_offset.unwrap();
        let (src_buf, src_ptr, _) = warp.shared_buffer(args_bytes);
        let (count_buf, count_ptr, _) = warp.shared_buffer(64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(args_bytes + 4);
        let dst_buf = warp.default_uav(layout.total_bytes);

        let mut initial_src = vec![0u8; args_bytes as usize];
        let mut expected_260 = Vec::new();
        let mut expected_hits_260 = 0u32;
        for i in 0..RECORDS {
            let words = if i < 260 {
                if i % 5 == 0 {
                    [2, 1, 1, 0x2600_0000 | i, 0xCAFE_0000 | i]
                } else {
                    [0, 1, 1, 0xDEAD_0000 | i, 0]
                }
            } else {
                // Tail slots past count = 260 hold huge garbage that would blow BUDGET.
                [u32::MAX, 64, 1, 0xBAD0_0000 | i, 0]
            };
            if i < 260 && i % 5 == 0 {
                expected_260.push(words);
                expected_hits_260 += 2;
            }
            let base = (i * STRIDE) as usize;
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }
        // SAFETY: mapped buffers.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, args_bytes as usize);
            (count_ptr as *mut u32).write_volatile(0);
            (count_ptr as *mut u32).add(1).write_volatile(260);
            std::ptr::write_bytes(budget_ptr, 0, 256);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (args_bytes + 4) as usize);
        }

        // First call: count = 0 -> compacted count = 0, hits = 0.
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            Some((&count_buf, 0)),
            (KIND_DISPATCH, STRIDE, RECORDS),
            false,
        ));
        // SAFETY: open list.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list
                .ExecuteIndirect(&sig, RECORDS, &dst_buf, 0, Some(&dst_buf), count_off);
        }
        warp.copy_indirect(
            &readback_buf,
            args_bytes,
            &dst_buf,
            count_off,
            4,
            Some(D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
        );
        warp.submit_and_wait();

        // SAFETY: GPU completed.
        let (compacted_0, spent_0, hits_0) = unsafe {
            (
                readback_ptr.add(args_bytes as usize).cast::<u32>().read_volatile(),
                (budget_ptr as *const u32).read_volatile(),
                (hits_ptr as *const u32).read_volatile(),
            )
        };
        assert_eq!((compacted_0, spent_0, hits_0), (0, 0, 0));

        // Second call (after reset, `dst_buf` starts in `COMMON`): count = 260 -> 52 kept records.
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            Some((&count_buf, 4)),
            (KIND_DISPATCH, STRIDE, RECORDS),
            false,
        ));
        // SAFETY: open list.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list
                .ExecuteIndirect(&sig, RECORDS, &dst_buf, 0, Some(&dst_buf), count_off);
        }
        warp.copy_indirect(&readback_buf, 0, &dst_buf, 0, args_bytes + 4, None);
        warp.submit_and_wait();

        // SAFETY: GPU completed.
        let (hits_260, spent_260, out_words) = unsafe {
            (
                (hits_ptr as *const u32).read_volatile(),
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(
                    readback_ptr.cast::<u32>().cast_const(),
                    (args_bytes as usize + 4) / 4,
                )
                .to_vec(),
            )
        };
        assert_eq!(out_words[count_off as usize / 4], expected_260.len() as u32);
        assert_eq!(hits_260, expected_hits_260);
        assert_eq!(spent_260, expected_hits_260 * 64);
        for (slot, expected) in expected_260.iter().enumerate() {
            assert_eq!(&out_words[slot * 5..(slot + 1) * 5], expected);
        }

        // 3. `records = 0` and `records = 1` do not compact and keep `INDIRECT_ARGUMENT` transitions valid.
        assert_eq!(scratch_layout(u64::from(STRIDE), 1, false).count_offset, None);
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_DISPATCH, STRIDE, 0),
            false,
        ));
        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_DISPATCH, STRIDE, 1),
            false,
        ));
        warp.submit_and_wait();
    }

    #[test]
    fn warp_compaction_budget_exhaustion_multi_tile() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 16;
        const RECORDS: u32 = 600; // 3 tiles
        const BUDGET_CAP: u32 = 1 << 27;

        let args_bytes = u64::from(STRIDE * RECORDS);
        let layout = scratch_layout(args_bytes, RECORDS, false);
        let count_off = layout.count_offset.unwrap();
        let (src_buf, src_ptr, _) = warp.shared_buffer(args_bytes);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(args_bytes + 4);
        let dst_buf = warp.default_uav(layout.total_bytes);
        let sig = warp.signature(KIND_DISPATCH, STRIDE);

        let mut initial_src = vec![0u8; args_bytes as usize];
        for i in 0..RECORDS {
            let words = if i % 3 == 0 {
                [0u32, 1, 1, 0xE000_0000 | i]
            } else {
                [u32::MAX, 16, 1, 0xA000_0000 | i]
            };
            let base = (i * STRIDE) as usize;
            for (w, val) in words.into_iter().enumerate() {
                initial_src[base + w * 4..base + (w + 1) * 4].copy_from_slice(&val.to_le_bytes());
            }
        }
        // SAFETY: mapped buffers.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, args_bytes as usize);
            std::ptr::write_bytes(budget_ptr, 0, 256);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (args_bytes + 4) as usize);
        }

        assert!(record_clamp_as(
            &warp.clamp,
            &warp.list,
            (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
            budget_gpu,
            (&src_buf, 0),
            None,
            (KIND_DISPATCH, STRIDE, RECORDS),
            false,
        ));
        // SAFETY: open list.
        unsafe {
            warp.list.SetComputeRootSignature(&warp.worker.root);
            warp.list.SetPipelineState(&warp.worker.pso);
            warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            warp.list
                .ExecuteIndirect(&sig, RECORDS, &dst_buf, 0, Some(&dst_buf), count_off);
        }
        warp.copy_indirect(&readback_buf, 0, &dst_buf, 0, args_bytes + 4, None);
        warp.submit_and_wait();

        // SAFETY: GPU completed.
        let (hits, spent, out_words) = unsafe {
            (
                (hits_ptr as *const u32).read_volatile(),
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(
                    readback_ptr.cast::<u32>().cast_const(),
                    (args_bytes as usize + 4) / 4,
                )
                .to_vec(),
            )
        };
        let compacted = out_words[count_off as usize / 4];
        assert_eq!(compacted, 2);
        assert_eq!(hits, 2 * 65_535 * 16);
        assert_eq!(spent, BUDGET_CAP);
        let tag0 = out_words[3] & 0x0FFF_FFFF;
        let tag1 = out_words[7] & 0x0FFF_FFFF;
        assert!(tag0 < tag1 && tag1 < RECORDS);
        assert_ne!(tag0 % 3, 0);
        assert_ne!(tag1 % 3, 0);
        assert_eq!(&out_words[0..3], [65_535, 16, 1]);
        assert_eq!(&out_words[4..7], [65_535, 16, 1]);
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, `bound_ahead`, the
    /// scratch buffer's tracked state, `Host::execute`, and `HostList::reset`
    /// end-to-end on the host device.
    #[test]
    fn host_list_reuses_and_grows_scratch_across_submissions() {
        let Some(host) = super::host() else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let Some((src_buf, src_ptr)) = host.shared_buffer(256 << 10) else {
            return;
        };
        let Some((count_buf, count_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped host buffers.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0x7F, 256 << 10);
            (count_ptr as *mut u32).write_volatile(2);
        }

        let args = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
        // Single-record calls stay in-slot (`count == None`).
        let b0 = bound(&list, (&src_buf, 0), None, (0, 16, 1)).unwrap();
        assert_eq!(b0.count, None);
        assert_eq!(list.indirect_state(&b0.buffer), args);

        // Multi-record calls compact (`count == Some(...)`) and fit inside the
        // initial 64 KiB scratch buffer, reusing the exact same `ID3D12Resource`,
        // which stays readable as arguments between them.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64)).unwrap();
        assert!(b1.count.is_some());
        assert_eq!(b1.buffer.as_raw(), b0.buffer.as_raw());
        assert_eq!(
            list.indirect_state(&b1.buffer),
            args,
            "a clamp leaves the scratch readable as arguments"
        );
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128)).unwrap();
        assert!(b2.count.is_some());
        assert_eq!(
            b1.buffer.as_raw(),
            b2.buffer.as_raw(),
            "scratch buffer within capacity must be reused"
        );
        assert_eq!(list.indirect_state(&b2.buffer), args);

        // Windowed calls (`bound_ahead`) keep in-slot layout and reuse scratch.
        let ahead = bound_ahead(&list, (&src_buf, 0), (10, 32, 16)).unwrap();
        assert_eq!(ahead.as_raw(), b1.buffer.as_raw());
        assert_eq!(list.indirect_state(&ahead), args);

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset. The new buffer starts out common.
        let layout3 = scratch_layout(5000 * 20, 5000, false);
        let grown = list.indirect_scratch(layout3.total_bytes).unwrap();
        assert_eq!(
            list.indirect_state(&grown),
            D3D12_RESOURCE_STATE_COMMON,
            "a new scratch buffer is common"
        );
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000)).unwrap();
        assert_eq!(b3.buffer.as_raw(), grown.as_raw());
        assert_ne!(
            b3.buffer.as_raw(),
            b1.buffer.as_raw(),
            "larger request must grow scratch buffer"
        );
        assert_eq!(list.indirect_state(&b3.buffer), args);
        assert_eq!(
            list.indirect_state(&b1.buffer),
            D3D12_RESOURCE_STATE_COMMON,
            "the replaced buffer went back to common"
        );

        list.close();
        // With the debug layer on (`VICTORY_D3D12_DEBUG=1`, diagnostics):
        // every barrier recorded named the state the buffer was in. Read
        // before the submission, which moves the messages to the log.
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();

        // After reset, requests up to the grown high-water mark reuse `b3` with
        // zero new allocations; the buffer decayed to common when the
        // submission ran.
        assert_eq!(
            list.indirect_state(&b3.buffer),
            D3D12_RESOURCE_STATE_COMMON,
            "a reset list starts with a common scratch buffer"
        );
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096)).unwrap();
        assert_eq!(
            b4.buffer.as_raw(),
            b3.buffer.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        assert_eq!(list.indirect_state(&b4.buffer), args);
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
    }

    #[test]
    fn host_indirect_compaction_executes_cleanly_under_debug_layer() {
        let Some(host) = super::host() else {
            return;
        };
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        const STRIDE: u32 = 16;
        const RECORDS: u32 = 300;
        let Some((src_buf, src_ptr)) = host.shared_buffer(u64::from(STRIDE * RECORDS)) else {
            return;
        };
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(64) else {
            return;
        };
        let Ok(worker) = compile_cs(&host.device, WORKER_CS) else {
            return;
        };
        let arg = D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        };
        let sig_desc = D3D12_COMMAND_SIGNATURE_DESC {
            ByteStride: STRIDE,
            NumArgumentDescs: 1,
            pArgumentDescs: &arg,
            NodeMask: 0,
        };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: plain single-argument command signature creation on `host.device`.
        unsafe {
            host.device
                .CreateCommandSignature(&sig_desc, None::<&ID3D12RootSignature>, &mut sig)
                .unwrap();
        }
        let sig = sig.unwrap();

        let mut expected_hits = 0u32;
        // SAFETY: mapped host buffers.
        unsafe {
            (hits_ptr as *mut u32).write_volatile(0);
            for i in 0..RECORDS {
                let words = if i % 4 == 0 {
                    expected_hits += 3;
                    [3u32, 1, 1, i]
                } else {
                    [0u32, 1, 1, i]
                };
                let base = src_ptr.add((i * STRIDE) as usize).cast::<u32>();
                for (w, val) in words.into_iter().enumerate() {
                    base.add(w).write_volatile(val);
                }
            }
        }

        let bounded = bound(&list, (&src_buf, 0), None, (2, STRIDE, RECORDS)).unwrap();
        let count_off = bounded.count.expect("multi-record bound must compact");
        assert_eq!(
            list.indirect_state(&bounded.buffer),
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT
        );
        // SAFETY: open host list, valid worker PSO/root, compacted scratch buffer in INDIRECT_ARGUMENT.
        unsafe {
            list.list.SetComputeRootSignature(&worker.root);
            list.list.SetPipelineState(&worker.pso);
            list.list
                .SetComputeRootUnorderedAccessView(0, hits_buf.GetGPUVirtualAddress());
            list.list.ExecuteIndirect(
                &sig,
                RECORDS,
                &bounded.buffer,
                0,
                Some(&bounded.buffer),
                count_off,
            );
        }
        list.close();
        assert_eq!(
            host.log_messages(),
            None,
            "host D3D12 debug layer must stay clean"
        );
        host.execute(&[&list]);
        list.reset();

        // SAFETY: GPU execution finished before `host.execute` returned.
        let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
        assert_eq!(hits, expected_hits);
    }

    /// A group charges its records in lane order: with 300 left of the
    /// budget, the first draw (100) fits, the second (250) does not and
    /// takes the rest, and the ones after it run empty (both in-slot and
    /// compacted).
    #[test]
    fn warp_a_group_charges_its_records_in_lane_order() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DRAW: u32 = 0;
        const STRIDE: u32 = 16;
        const BUDGET_CAP: u32 = 1 << 27;
        let counts = [100u32, 250, 50, 100];
        let total = counts.len() * STRIDE as usize;
        for ahead in [true, false] {
            let layout = scratch_layout(total as u64, counts.len() as u32, ahead);
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(layout.total_bytes);
            let dst_buf = warp.default_uav(layout.total_bytes);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                for (i, &count) in counts.iter().enumerate() {
                    let words = [count, 1, 7, 9];
                    std::ptr::copy_nonoverlapping(
                        words.as_ptr().cast::<u8>(),
                        src_ptr.add(i * STRIDE as usize),
                        16,
                    );
                }
                std::ptr::write_bytes(budget_ptr, 0, 256);
                (budget_ptr as *mut u32).write_volatile(BUDGET_CAP - 300);
            }
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, D3D12_RESOURCE_STATE_COMMON),
                budget_gpu,
                (&src_buf, 0),
                None,
                (KIND_DRAW, STRIDE, counts.len() as u32),
                ahead,
            ));
            warp.copy_indirect(&readback_buf, 0, &dst_buf, 0, layout.total_bytes, None);
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    std::slice::from_raw_parts(
                        readback_ptr.cast::<u32>().cast_const(),
                        layout.total_bytes as usize / 4,
                    )
                    .to_vec(),
                )
            };
            assert_eq!(out[0..4], [100, 1, 7, 9], "the first record fits (ahead={ahead})");
            if ahead {
                for i in 1..counts.len() {
                    assert_eq!(out[i * 4..i * 4 + 4], [0, 0, 7, 9], "record {i} runs empty");
                }
            } else {
                let count_off = layout.count_offset.unwrap();
                assert_eq!(out[count_off as usize / 4], 1, "zeroed records are compacted out");
            }
            assert_eq!(spent, BUDGET_CAP, "the record that did not fit took the rest");
        }
    }

    /// The list's budget starts empty at each submission, wherever it
    /// lives (the card's copy is zeroed in the list): a recording whose
    /// clamps spend all of it is followed, after the reset, by one whose
    /// small record still runs.
    #[test]
    fn host_list_budget_starts_empty_at_each_submission() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let _ = host.log_messages();
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(KIND_DISPATCH as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: STRIDE,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: a valid single-argument command signature description.
            unsafe {
                host.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        };
        // Three records of 65,535 x 16 groups: about half the budget each.
        let Some((big, big_ptr)) = host.shared_buffer(3 * u64::from(STRIDE)) else {
            return;
        };
        let Some((small, small_ptr)) = host.shared_buffer(u64::from(STRIDE)) else {
            return;
        };
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet.
        unsafe {
            for i in 0..3 {
                let words = [65535u32, 16, 1];
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    big_ptr.add(i * STRIDE as usize),
                    12,
                );
            }
            std::ptr::copy_nonoverlapping([1u32, 1, 1].as_ptr().cast::<u8>(), small_ptr, 12);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for round in 0..2 {
            // Spends the budget (clamped, not executed), then one group.
            bound(&list, (&big, 0), None, (KIND_DISPATCH, STRIDE, 3)).unwrap();
            let scratch = bound(&list, (&small, 0), None, (KIND_DISPATCH, STRIDE, 1)).unwrap();
            // SAFETY: an open list; the worker's root signature and
            // pipeline; buffers the test holds until the list has run.
            unsafe {
                list.list.SetComputeRootSignature(&worker.root);
                list.list.SetPipelineState(&worker.pso);
                list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                list.list.ExecuteIndirect(&signature, 1, &scratch.buffer, 0, None, 0);
            }
            list.close();
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            // The small record comes after the big ones spent the budget:
            // it runs empty, in every round alike (none carries over).
            assert_eq!(hits, 0, "round {round}: the spent budget empties the record after it");
        }
        // A recording with only the small record: the budget is empty
        // again, so it runs.
        let scratch = bound(&list, (&small, 0), None, (KIND_DISPATCH, STRIDE, 1)).unwrap();
        // SAFETY: as above.
        unsafe {
            list.list.SetComputeRootSignature(&worker.root);
            list.list.SetPipelineState(&worker.pso);
            list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
            list.list.ExecuteIndirect(&signature, 1, &scratch.buffer, 0, None, 0);
        }
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
        // SAFETY: the list has run; a mapped buffer.
        let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
        assert_eq!(hits, 1, "the budget starts empty at the next submission");
    }

    /// The window check: a record nothing wrote since its window clamped
    /// it compares equal; one written on the card after the window's clamp
    /// (an untracked writer) is found, and named as the window had it and
    /// as it is now.
    #[test]
    fn host_list_window_check_finds_a_record_written_under_its_window() {
        let Some(host) = super::host() else {
            return;
        };
        let Ok(list) = host.new_list() else {
            return;
        };
        let _ = host.log_messages();
        const KIND_DISPATCH: u32 = 2;
        const STEP: u32 = 32;
        let records = [[3u32, 1, 1], [5, 1, 1], [7, 1, 1], [9, 1, 1]];
        let total = records.len() * STEP as usize;
        let (Some((src, src_ptr)), Some((patch, patch_ptr))) =
            (host.shared_buffer(total as u64), host.shared_buffer(64))
        else {
            return;
        };
        // SAFETY: mapped buffers of these sizes; nothing runs yet.
        unsafe {
            std::ptr::write_bytes(src_ptr, 0, total);
            for (i, words) in records.iter().enumerate() {
                std::ptr::copy_nonoverlapping(
                    words.as_ptr().cast::<u8>(),
                    src_ptr.add(i * STEP as usize),
                    12,
                );
            }
            std::ptr::copy_nonoverlapping([11u32, 2, 1].as_ptr().cast::<u8>(), patch_ptr, 12);
        }
        let spec = (KIND_DISPATCH, STEP, records.len() as u32);
        let scratch = bound_ahead(&list, (&src, 0), spec).unwrap();
        // Record 0: nothing wrote it.
        assert!(check_hit(&list, (&src, 0), (KIND_DISPATCH, STEP), &scratch, 0));
        // Record 1 is written on the card after the window's clamp.
        let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        // SAFETY: an open list; `src` was read by the clamp (promoted to a
        // shader read), `patch` is common; both live until the list has run.
        unsafe {
            list.list.ResourceBarrier(&[transition(&src, read, D3D12_RESOURCE_STATE_COPY_DEST)]);
            list.list.CopyBufferRegion(&src, u64::from(STEP), &patch, 0, 12);
            list.list.ResourceBarrier(&[transition(&src, D3D12_RESOURCE_STATE_COPY_DEST, read)]);
        }
        let at = u64::from(STEP);
        assert!(check_hit(&list, (&src, at), (KIND_DISPATCH, STEP), &scratch, at));
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        let (checked, differed, window, fresh) = list.window_check().unwrap().take();
        list.reset();
        assert_eq!((checked, differed), (2, 1), "two checks, the written record differs");
        assert_eq!(window[..3], [5, 1, 1], "as the window clamped it");
        assert_eq!(fresh[..3], [11, 2, 1], "as it is now");
    }

    /// Records the dispatch records `groups` (three words each, `step`
    /// bytes apart, the rest of each record ones) into a fresh mapped
    /// buffer of `warp`'s device.
    fn dispatch_records(warp: &WarpHarness, groups: &[u32], step: u32) -> ID3D12Resource {
        let total = groups.len() * step as usize;
        let (buffer, ptr, _) = warp.shared_buffer(total as u64);
        let mut bytes = vec![0xffu8; total];
        for (i, &g) in groups.iter().enumerate() {
            for (w, word) in [g, 1, 1].into_iter().enumerate() {
                let at = i * step as usize + w * 4;
                bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
            }
        }
        // SAFETY: the mapping holds `total` bytes.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, total) };
        buffer
    }

    /// Two windows in one list, as `crate::draw` records them: each clamps
    /// its records ahead, then one execution per record reads the scratch
    /// buffer with no barrier between them; the second clamp moves the
    /// scratch buffer straight from arguments back to unordered access.
    /// Every record runs once, with the group count it asked for.
    #[test]
    fn warp_window_executions_read_their_records_with_no_barrier_between() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        const KIND_DISPATCH: u32 = 2;
        // The title names 12 bytes and keeps its records 32 apart.
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let sources: Vec<ID3D12Resource> =
            windows.iter().map(|g| dispatch_records(&warp, g, STEP)).collect();
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let dst_buf = warp.default_uav(u64::from(STEP) * 4);
        let sig = warp.signature(KIND_DISPATCH, STRIDE);
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::write_bytes(budget_ptr, 0, 256);
            std::ptr::write_bytes(hits_ptr, 0, 256);
        }
        let mut state = D3D12_RESOURCE_STATE_COMMON;
        for (groups, src) in windows.iter().zip(&sources) {
            assert!(record_clamp_as(
                &warp.clamp,
                &warp.list,
                (&dst_buf, state),
                budget_gpu,
                (src, 0),
                None,
                (KIND_DISPATCH, STEP, groups.len() as u32),
                true,
            ));
            state = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT;
            // SAFETY: an open list; the worker's root signature and
            // pipeline; the scratch buffer is readable as arguments.
            unsafe {
                warp.list.SetComputeRootSignature(&warp.worker.root);
                warp.list.SetPipelineState(&warp.worker.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                for i in 0..groups.len() as u64 {
                    warp.list.ExecuteIndirect(&sig, 1, &dst_buf, i * u64::from(STEP), None, 0);
                }
            }
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (hits, spent) = unsafe {
            (
                (hits_ptr as *const u32).read_volatile(),
                (budget_ptr as *const u32).read_volatile(),
            )
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        assert_eq!(hits, expected, "every record of both windows runs once");
        assert_eq!(spent, expected * 64, "and is charged what it asked for");
    }

    /// The windows of `warp_window_executions_read_their_records_with_no_barrier_between`
    /// through a host list (`bound_ahead`, the list's tracked scratch state,
    /// `Host::execute`), twice: on the first recording the scratch buffer
    /// starts common, on the second (after the reset) it starts common
    /// again. With the debug layer on (`VICTORY_D3D12_DEBUG=1` in a
    /// diagnostics build) no barrier names a state the buffer is not in.
    #[test]
    fn host_list_window_executions_need_no_barrier_of_their_own() {
        let Some(host) = super::host() else {
            return;
        };
        // What earlier tests left in the debug layer's queue, if it is on.
        let _ = host.log_messages();
        let Ok(list) = host.new_list() else {
            return;
        };
        let worker = compile_cs(&host.device, WORKER_CS).expect("worker shader must compile");
        const KIND_DISPATCH: u32 = 2;
        const STRIDE: u32 = 12;
        const STEP: u32 = 32;
        let signature = {
            let arg = D3D12_INDIRECT_ARGUMENT_DESC {
                Type: D3D12_INDIRECT_ARGUMENT_TYPE(KIND_DISPATCH as i32),
                ..Default::default()
            };
            let desc = D3D12_COMMAND_SIGNATURE_DESC {
                ByteStride: STRIDE,
                NumArgumentDescs: 1,
                pArgumentDescs: &arg,
                NodeMask: 0,
            };
            let mut sig: Option<ID3D12CommandSignature> = None;
            // SAFETY: a valid single-argument command signature description.
            unsafe {
                host.device
                    .CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig)
                    .unwrap();
            }
            sig.unwrap()
        };
        let windows: [&[u32]; 2] = [&[1, 2, 3, 4], &[10, 11, 12, 13]];
        let mut sources = Vec::new();
        for groups in windows {
            let total = groups.len() * STEP as usize;
            let Some((buffer, ptr)) = host.shared_buffer(total as u64) else {
                return;
            };
            let mut bytes = vec![0xffu8; total];
            for (i, &g) in groups.iter().enumerate() {
                for (w, word) in [g, 1, 1].into_iter().enumerate() {
                    let at = i * STEP as usize + w * 4;
                    bytes[at..at + 4].copy_from_slice(&word.to_le_bytes());
                }
            }
            // SAFETY: the mapping holds `total` bytes.
            unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr, total) };
            sources.push(buffer);
        }
        let Some((hits_buf, hits_ptr)) = host.shared_buffer(256) else {
            return;
        };
        let expected: u32 = windows.iter().flat_map(|g| g.iter()).sum();
        // SAFETY: a live buffer.
        let hits_gpu = unsafe { hits_buf.GetGPUVirtualAddress() };
        for _ in 0..2 {
            // SAFETY: the mapping holds 256 bytes; nothing runs.
            unsafe { std::ptr::write_bytes(hits_ptr, 0, 256) };
            for (groups, src) in windows.iter().zip(&sources) {
                let spec = (KIND_DISPATCH, STEP, groups.len() as u32);
                let scratch = bound_ahead(&list, (src, 0), spec).unwrap();
                assert_eq!(list.indirect_state(&scratch), D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT);
                // SAFETY: an open list; the worker's root signature and
                // pipeline; buffers the test holds until the list has run.
                unsafe {
                    list.list.SetComputeRootSignature(&worker.root);
                    list.list.SetPipelineState(&worker.pso);
                    list.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                    for i in 0..groups.len() as u64 {
                        let at = i * u64::from(STEP);
                        list.list.ExecuteIndirect(&signature, 1, &scratch, at, None, 0);
                    }
                }
            }
            list.close();
            // With the debug layer on: nothing it objects to was recorded.
            // Read before the submission, which moves the messages to the log.
            assert_eq!(host.log_messages(), None, "the debug layer found an error");
            host.execute(&[&list]);
            list.reset();
            // SAFETY: the list has run; a mapped buffer.
            let hits = unsafe { (hits_ptr as *const u32).read_volatile() };
            assert_eq!(hits, expected, "every record of both windows runs once");
        }
    }
}

/// The benchmark of the clamp and its barriers (ignored by default).
#[cfg(test)]
mod bench;

/// What one small draw costs this card, bare and with one thing changed
/// between every two draws: a measurement, not a check (it asserts only
/// that the card answered). The timing run of the lit scene (run 187) has
/// some 11,000 small commands a frame at 5 to 16 microseconds each, and no
/// log says whether that is what we put around a command or what the card
/// takes for one. Each variant here is 10,000 one-triangle draws into a
/// 2560x1440 target, timed on the card by one pair of timestamps around
/// the whole batch (not one per command, which would time the stamps).
///
/// Run by name, no game needed:
/// `cargo test -p victory-gpu --release -- --ignored bare_command_cost --nocapture`
#[cfg(test)]
mod bare_cost {
    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use crate::host::{compile, transition};

    /// Draws in a batch.
    const DRAWS: u32 = 10_000;
    const WIDTH: u32 = 2560;
    const HEIGHT: u32 = 1440;

    /// One small triangle in the middle of the target (about 180 pixels),
    /// and two pixel programs that read one texel of a texture through a
    /// table, so that the view heap set is in use.
    const DRAW: &str = r#"
#define RS "DescriptorTable(SRV(t0))"
Texture2D<float4> image : register(t0);
[RootSignature(RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(RS)]
float4 first(float4 at : SV_Position) : SV_Target {
    return image.Load(int3(0, 0, 0));
}
[RootSignature(RS)]
float4 second(float4 at : SV_Position) : SV_Target {
    return 1.0 - image.Load(int3(0, 0, 0));
}
"#;

    /// A compute pass of next to nothing, as our clamp of an indirect
    /// record is: one group of one thread that writes one word.
    const WORK: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void work() {
    words.Store(0, words.Load(0) + 1);
}
"#;

    /// What changes between every two draws of a batch.
    #[derive(Clone, Copy, PartialEq)]
    enum Between {
        /// Nothing.
        Nothing,
        /// The pipeline: two alternate.
        Pipeline,
        /// A second texture goes from target to shader read and back.
        BarrierPair,
        /// The depth texture bound for testing goes from write to read or
        /// back.
        DepthMove,
        /// The pair of descriptor heaps: two pairs alternate.
        Heaps,
        /// The colour target: two alternate.
        Targets,
        /// A compute pass of ours of one group, and the pipeline back.
        Compute,
        /// The same between two waits for every earlier shader write, as
        /// the clamp records them.
        ComputeWithWaits,
    }

    struct Rig {
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        root: ID3D12RootSignature,
        pipelines: [ID3D12PipelineState; 2],
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        words: ID3D12Resource,
        other: ID3D12Resource,
        depth: ID3D12Resource,
        /// Kept alive: the views below name them.
        _kept: Vec<ID3D12Resource>,
        _target_heaps: [ID3D12DescriptorHeap; 2],
        heap_pairs: [[Option<ID3D12DescriptorHeap>; 2]; 2],
        tables: [D3D12_GPU_DESCRIPTOR_HANDLE; 2],
        targets: [D3D12_CPU_DESCRIPTOR_HANDLE; 2],
        depth_written: D3D12_CPU_DESCRIPTOR_HANDLE,
        depth_tested: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
    }

    fn texture(
        device: &ID3D12Device,
        (width, height): (u32, u32),
        format: DXGI_FORMAT,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_DEFAULT,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: u64::from(width),
            Height: height,
            DepthOrArraySize: 1,
            MipLevels: 1,
            Format: format,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain texture creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn buffer(
        device: &ID3D12Device,
        size: u64,
        kind: D3D12_HEAP_TYPE,
        flags: D3D12_RESOURCE_FLAGS,
        state: D3D12_RESOURCE_STATES,
    ) -> ID3D12Resource {
        let heap = D3D12_HEAP_PROPERTIES {
            Type: kind,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            Flags: flags,
            ..Default::default()
        };
        let mut out: Option<ID3D12Resource> = None;
        // SAFETY: plain buffer creation.
        unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
                .unwrap();
        }
        out.unwrap()
    }

    fn heap(device: &ID3D12Device, kind: D3D12_DESCRIPTOR_HEAP_TYPE, count: u32, visible: bool) -> ID3D12DescriptorHeap {
        let flags = if visible {
            D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE
        } else {
            D3D12_DESCRIPTOR_HEAP_FLAG_NONE
        };
        // SAFETY: plain heap creation.
        unsafe {
            device
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: kind,
                    NumDescriptors: count,
                    Flags: flags,
                    NodeMask: 0,
                })
                .unwrap()
        }
    }

    fn pipeline(device: &ID3D12Device, root: &ID3D12RootSignature, vs: &[u8], ps: &[u8]) -> ID3D12PipelineState {
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let mut blend = D3D12_BLEND_DESC::default();
        blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
        let mut formats = [DXGI_FORMAT(0); 8];
        formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
        // SAFETY: the description points at live data and never releases
        // its pointer to the root signature.
        unsafe {
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(root),
                VS: code(vs),
                PS: code(ps),
                BlendState: blend,
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    DepthClipEnable: true.into(),
                    ..Default::default()
                },
                // Depth is tested and never written, so the texture may be
                // in the write state or in the read state under one view.
                DepthStencilState: D3D12_DEPTH_STENCIL_DESC {
                    DepthEnable: true.into(),
                    DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ZERO,
                    DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                    ..Default::default()
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: formats,
                DSVFormat: DXGI_FORMAT_D32_FLOAT,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            };
            device.CreateGraphicsPipelineState(&desc).unwrap()
        }
    }

    impl Rig {
        /// The default adapter, as the layer takes it; `None` without one
        /// or on a software adapter.
        fn new() -> Option<Self> {
            let vs = compile(DRAW, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = [
                compile(DRAW, s!("first"), s!("ps_5_1")).ok()?,
                compile(DRAW, s!("second"), s!("ps_5_1")).ok()?,
            ];
            let cs = compile(WORK, s!("work"), s!("cs_5_1")).ok()?;
            // SAFETY: plain object creation on a device of our own; the
            // views are made in heaps that live as long as the rig.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|factory| factory.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
                    .ok()?
                    .cast::<IDXGIAdapter1>()
                    .ok()?
                    .GetDesc1()
                    .ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    return None;
                }
                let card = String::from_utf16_lossy(&desc.Description)
                    .trim_end_matches('\0')
                    .to_owned();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                        Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                        ..Default::default()
                    })
                    .ok()?;
                let allocator: ID3D12CommandAllocator =
                    device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                    .ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;

                let root: ID3D12RootSignature = device.CreateRootSignature(0, &vs[..]).ok()?;
                let pipelines = [
                    pipeline(&device, &root, &vs, &ps[0]),
                    pipeline(&device, &root, &vs, &ps[1]),
                ];
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs[..]).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE {
                            pShaderBytecode: cs.as_ptr().cast(),
                            BytecodeLength: cs.len(),
                        },
                        ..Default::default()
                    })
                    .ok()?;

                let size = (WIDTH, HEIGHT);
                let colour = |state| texture(&device, size, DXGI_FORMAT_R8G8B8A8_UNORM, D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, state);
                let target = [
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                    colour(D3D12_RESOURCE_STATE_RENDER_TARGET),
                ];
                let other = colour(D3D12_RESOURCE_STATE_RENDER_TARGET);
                let depth = texture(
                    &device,
                    size,
                    DXGI_FORMAT_D32_FLOAT,
                    D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                    D3D12_RESOURCE_STATE_DEPTH_WRITE,
                );
                let image = texture(
                    &device,
                    (4, 4),
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                );
                let words = buffer(
                    &device,
                    64 << 10,
                    D3D12_HEAP_TYPE_DEFAULT,
                    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                );
                let back = buffer(
                    &device,
                    16,
                    D3D12_HEAP_TYPE_READBACK,
                    D3D12_RESOURCE_FLAG_NONE,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                );
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(
                        &D3D12_QUERY_HEAP_DESC {
                            Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                            Count: 2,
                            NodeMask: 0,
                        },
                        &mut stamps,
                    )
                    .ok()?;
                let stamps = stamps?;

                // Target and depth views.
                let rtvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, 2, false);
                let rtv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV) as usize;
                let mut targets = [rtvs.GetCPUDescriptorHandleForHeapStart(); 2];
                targets[1].ptr += rtv_step;
                device.CreateRenderTargetView(&target[0], None, targets[0]);
                device.CreateRenderTargetView(&target[1], None, targets[1]);
                let dsvs = heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_DSV, 2, false);
                let dsv_step = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_DSV) as usize;
                let depth_written = dsvs.GetCPUDescriptorHandleForHeapStart();
                let mut depth_tested = depth_written;
                depth_tested.ptr += dsv_step;
                device.CreateDepthStencilView(&depth, None, depth_written);
                let tested = D3D12_DEPTH_STENCIL_VIEW_DESC {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    ViewDimension: D3D12_DSV_DIMENSION_TEXTURE2D,
                    Flags: D3D12_DSV_FLAG_READ_ONLY_DEPTH,
                    ..Default::default()
                };
                device.CreateDepthStencilView(&depth, Some(&tested), depth_tested);

                // Two pairs of shader-visible heaps: views (one view of the
                // small texture each) and samplers (none used).
                let view_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV, 1, true),
                ];
                let sampler_heaps = [
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                    heap(&device, D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER, 1, true),
                ];
                for views in &view_heaps {
                    device.CreateShaderResourceView(&image, None, views.GetCPUDescriptorHandleForHeapStart());
                }
                let tables = [
                    view_heaps[0].GetGPUDescriptorHandleForHeapStart(),
                    view_heaps[1].GetGPUDescriptorHandleForHeapStart(),
                ];
                let heap_pairs = [
                    [Some(view_heaps[0].clone()), Some(sampler_heaps[0].clone())],
                    [Some(view_heaps[1].clone()), Some(sampler_heaps[1].clone())],
                ];

                // The first list: both targets and the depth texture cleared.
                list.ClearRenderTargetView(targets[0], &[0.0; 4], None);
                list.ClearRenderTargetView(targets[1], &[0.0; 4], None);
                list.ClearDepthStencilView(depth_written, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                let [first_target, second_target] = target;
                let mut rig = Rig {
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    root,
                    pipelines,
                    work_root,
                    work,
                    words,
                    other,
                    depth,
                    _kept: vec![first_target, second_target, image],
                    _target_heaps: [rtvs, dsvs],
                    heap_pairs,
                    tables,
                    targets,
                    depth_written,
                    depth_tested,
                    stamps,
                    back,
                    card,
                };
                rig.finish();
                Some(rig)
            }
        }

        /// Closes the open list, runs it, waits for the card and opens the
        /// list again.
        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // only after the card finished everything recorded from it.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }

        /// What every list starts with: a new list has nothing set.
        fn begin(&self) {
            let l = &self.list;
            // SAFETY: an open list and live objects and views.
            unsafe {
                l.SetDescriptorHeaps(&self.heap_pairs[0]);
                l.SetGraphicsRootSignature(&self.root);
                l.SetPipelineState(&self.pipelines[0]);
                l.SetGraphicsRootDescriptorTable(0, self.tables[0]);
                l.OMSetRenderTargets(1, Some(&self.targets[0]), false, Some(&self.depth_tested));
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: WIDTH as f32,
                    Height: HEIGHT as f32,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right: WIDTH as i32,
                    bottom: HEIGHT as i32,
                }]);
                // A triangle list.
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
            }
        }

        /// One batch of [`DRAWS`] draws in `lists` lists run one by one,
        /// with `between` recorded before every draw: the microseconds a
        /// draw took on the card, and the milliseconds recording took.
        fn batch(&mut self, between: Between, lists: u32) -> (f64, f64) {
            let (target, read, written, tested) = (
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_DEPTH_READ,
            );
            let started = std::time::Instant::now();
            let mut drawn = 0u32;
            for part in 0..lists {
                let l = &self.list;
                // SAFETY: an open list; our query heap; live objects.
                unsafe {
                    if part == 0 {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                    }
                }
                self.begin();
                // SAFETY: an open list with everything set; live objects.
                // Every state moved is moved back within the batch (the
                // draws of a list are an even number).
                unsafe {
                    for _ in 0..DRAWS / lists {
                        let odd = (drawn % 2) as usize;
                        match between {
                            Between::Nothing => {}
                            Between::Pipeline => l.SetPipelineState(&self.pipelines[odd]),
                            Between::BarrierPair => {
                                l.ResourceBarrier(&[transition(&self.other, target, read)]);
                                l.ResourceBarrier(&[transition(&self.other, read, target)]);
                            }
                            Between::DepthMove => {
                                let (from, to) = if odd == 0 { (written, tested) } else { (tested, written) };
                                l.ResourceBarrier(&[transition(&self.depth, from, to)]);
                            }
                            Between::Heaps => {
                                l.SetDescriptorHeaps(&self.heap_pairs[odd]);
                                l.SetGraphicsRootDescriptorTable(0, self.tables[odd]);
                            }
                            Between::Targets => {
                                l.OMSetRenderTargets(1, Some(&self.targets[odd]), false, Some(&self.depth_tested));
                            }
                            Between::Compute | Between::ComputeWithWaits => {
                                let waits = between == Between::ComputeWithWaits;
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetComputeRootSignature(&self.work_root);
                                l.SetPipelineState(&self.work);
                                l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                                l.Dispatch(1, 1, 1);
                                if waits {
                                    l.ResourceBarrier(&[super::uav_barrier()]);
                                }
                                l.SetPipelineState(&self.pipelines[0]);
                            }
                        }
                        l.DrawInstanced(3, 1, 0, 0);
                        drawn += 1;
                    }
                    if part + 1 < lists {
                        // The next list of the batch: this one is sent on
                        // its own and the list is opened again at once
                        // (the allocator is not reset before the card is
                        // done with all of them).
                        l.Close().unwrap();
                        let run: ID3D12CommandList = l.cast().unwrap();
                        self.queue.ExecuteCommandLists(&[Some(run)]);
                        l.Reset(&self.allocator, None).unwrap();
                    } else {
                        l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
                    }
                }
            }
            let recording = started.elapsed().as_secs_f64() * 1e3;
            self.finish();
            let mut data = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            // SAFETY: a read-back buffer the finished list resolved two
            // timestamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&range), Some(&mut data)).unwrap();
                let stamps = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                stamps[1].saturating_sub(stamps[0])
            };
            // SAFETY: a plain query of our queue.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            (ticks as f64 * 1e6 / frequency as f64 / f64::from(drawn.max(1)), recording)
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn bare_command_cost() {
        let Some(mut rig) = Rig::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        eprintln!("bare command cost on {}: {DRAWS} one-triangle draws into {WIDTH}x{HEIGHT}, three batches each", rig.card);
        // The first batch pays for whatever the driver does at first use.
        rig.batch(Between::Nothing, 1);
        let variants = [
            ("(a) nothing changes between draws", Between::Nothing, 1),
            ("(b) another pipeline every draw", Between::Pipeline, 1),
            ("(c) a barrier pair on a second texture every draw", Between::BarrierPair, 1),
            ("(d) the tested depth texture moved between write and read every draw", Between::DepthMove, 1),
            ("(e) another pair of descriptor heaps every draw", Between::Heaps, 1),
            ("(f) another colour target every draw", Between::Targets, 1),
            ("(g) nothing changes, in 100 lists run one by one", Between::Nothing, 100),
            ("(h) a compute pass of ours of one group before every draw", Between::Compute, 1),
            ("(i) the same pass between two waits for earlier shader writes", Between::ComputeWithWaits, 1),
        ];
        let mut least = Vec::new();
        for (name, between, lists) in variants {
            let mut runs: Vec<(f64, f64)> = (0..3).map(|_| rig.batch(between, lists)).collect();
            runs.sort_by(|a, b| a.0.total_cmp(&b.0));
            eprintln!(
                "{name}: {:.2} microseconds a draw on the card (the three: {:.2}, {:.2}, {:.2}); recording took {:.1} ms",
                runs[0].0, runs[0].0, runs[1].0, runs[2].0, runs[0].1
            );
            least.push(runs[0].0);
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}
