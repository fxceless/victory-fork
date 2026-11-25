//! Indirect execution's arguments, bounded before the GPU reads them. The
//! title's own compute passes write most of them; a pass we drop or
//! translate wrongly leaves whatever the memory held, and a draw of four
//! billion vertices or a dispatch past the host's limits runs until the
//! host resets the display driver. So active records are copied and clamped
//! on the GPU into a device-local scratch buffer, just before execution, by a
//! compute shader of ours (HLSL compiled by the system's compiler; a 5.1
//! shader with its root signature inside), leaving the title's argument and
//! count buffers untouched.
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
//! When a count buffer is given, threads at or past `min(max, count)` return
//! immediately without charging the list's budget or reading uninitialized
//! tail slots. Each list has a budget (2^27 vertex instances or threads,
//! groups counted as 64 threads, mesh groups as 128): active records past it
//! run empty, saturating without 32-bit wraparound so a list of thousands of
//! garbage records cannot run for seconds either.
//!
//! shortcut: the bounds (2^24 vertices, 2^26 vertex instances, 65,535
//! groups per dimension and 2^20 per dispatch, one-dimensional mesh
//! dispatches, the budget) cut legitimate work beyond them; raise them if
//! the title needs more.

use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::core::s;

use crate::host::{HostList, host, transition};

const SOURCE: &str = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), SRV(t1), UAV(u0), UAV(u1)"
#define BATCH_RS "SRV(t0), SRV(t1), SRV(t2), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) {
    uint rec_kind;
    uint rec_stride;
    uint rec_records;
    uint rec_args_offset;
    // Bit 0: a count buffer bounds the records. Bit 1: the records are a
    // window clamped ahead of their draws: one that asks for more than can
    // run is left empty and costs nothing.
    uint rec_has_count;
    uint rec_count_offset;
};
ByteAddressBuffer src_args : register(t0);
ByteAddressBuffer count_buf : register(t1);
// A run's jobs (`batch`): word 0 their number, then from byte 16 eight
// words each: kind, stride, records, offset of the first in `src_args`,
// flags (bit 0: bounded by the count at the next word's offset in
// `count_buf`), that offset, the offset of the first clamped in
// `dst_args`, nothing.
ByteAddressBuffer jobs : register(t2);
RWByteAddressBuffer dst_args : register(u0);
// The list's work so far (vertex instances, threads): past the budget,
// records run empty.
RWByteAddressBuffer budget : register(u1);
static const uint BUDGET = 1u << 27;
// The vertices or indices an instance from which a draw record runs empty.
static const uint MOST_DRAW = 1u << 23;
// Groups of one dispatch; more run empty.
static const uint GROUPS = 1u << 20;
// The record of `kind` at byte `src_at` of `src_args`, `stride` bytes
// long, clamped and charged to the budget, to byte `dst_at` of `dst_args`
// (a record at a time: the runs of `batch`).
void clamp_record(uint kind, uint stride, uint src_at, uint dst_at, bool ahead) {
    uint cost;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 asked = src_args.Load2(src_at);
        if (ahead && (asked.x > (1u << 24) || asked.y > (1u << 24))) {
            asked = uint2(0, 0);
        }
        uint count = min(asked.x, 1u << 24);
        uint instances = min(asked.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        // As in `main`: MOST_DRAW vertices or indices an instance or more
        // is no draw of the title's; it runs empty and charges nothing.
        if (count >= MOST_DRAW) {
            count = 0;
            instances = 0;
            cost = 0;
        }
        draw_counts = uint2(count, instances);
    } else if (kind == 2) {
        uint3 g = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
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
            g = uint3(0, 0, 0);
        }
        // Diagnostics, in the budget buffer's next words: the largest
        // group count a mesh record asked for, and the groups asked in all.
        uint was;
        budget.InterlockedMax(4, g.x, was);
        budget.InterlockedAdd(8, min(g.x, 1u << 24), was);
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    bool over = false;
    if (cost > 0) {
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = (BUDGET - before <= cost) ? BUDGET : (before + cost);
            uint prev;
            budget.InterlockedCompareExchange(0, before, next, prev);
            if (prev == before) break;
            before = prev;
        }
        over = before >= BUDGET || BUDGET - before < cost;
    }
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
// Each group charges the budget once: the costs of its records go here,
// one thread charges their sum, saturating as each record would in lane
// order, and every thread then reads whether its own record fitted. A
// compare and swap per record, 64 threads racing for one word, cost a
// round for each record of a window, and in memory the CPU maps (where the
// budget was, and is with VICTORY_INDIRECT_CARD_BUDGET=0) every atomic
// crosses the bus on a card of its own.
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint largest_asked;
groupshared uint all_asked;
// One indirect call's records, a thread each.
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID, uint gi : SV_GroupIndex) {
    uint kind = rec_kind;
    uint stride = rec_stride;
    uint eff_count = rec_records;
    bool ahead = (rec_has_count & 2u) != 0;
    if ((rec_has_count & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(rec_count_offset));
    }
    // Threads past the records take part in the group's charge (cost 0)
    // and store nothing.
    bool active = id.x < eff_count;
    uint src_at = rec_args_offset + id.x * stride;
    uint dst_at = id.x * stride;
    uint cost = 0;
    uint asked = 0;
    uint2 draw_counts = uint2(0, 0);
    uint3 groups = uint3(0, 0, 0);
    if (gi == 0) {
        largest_asked = 0;
        all_asked = 0;
    }
    if (active && kind <= 1) {
        // Draws: vertex or index count, then instances.
        uint2 want = src_args.Load2(src_at);
        if (ahead && (want.x > (1u << 24) || want.y > (1u << 24))) {
            want = uint2(0, 0);
        }
        uint count = min(want.x, 1u << 24);
        uint instances = min(want.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        // A record that asks for MOST_DRAW vertices or indices an instance
        // or more is no draw of the title's (no mesh of a console title has
        // 2.8 million triangles): it runs empty and charges nothing. Run
        // 206's window check found one list with 90 draws of 2^24 vertices
        // and one instance each. Clamped one draw at a time, the list's
        // budget lets eight of them run (2^27 in all) and empties the other
        // 82; served from a window, whose records are charged once however
        // often they are drawn, all 90 ran: 1.5 thousand million vertices
        // in one list, with windows on in the runs that lost the device
        // (201, 204).
        if (count >= MOST_DRAW) {
            count = 0;
            instances = 0;
            cost = 0;
        }
        draw_counts = uint2(count, instances);
    } else if (active && kind == 2) {
        uint3 g = min(src_args.Load3(src_at), uint3(65535u, 65535u, 65535u));
        // x * y fits 32 bits; the whole count may not.
        if (g.z != 0 && g.x * g.y > GROUPS / g.z) {
            g = uint3(0, 0, 0);
        }
        cost = g.x * g.y * g.z * 64u;
        groups = g;
    } else if (active) {
        // Mesh dispatches: the console's are one-dimensional.
        uint3 g = src_args.Load3(src_at);
        // Ahead of its draw, a record over the most a dispatch runs is a
        // slot the title has not filled (the title leaves all
        // ones there): empty, and not counted below either.
        if (ahead && g.x > 65535u) {
            g = uint3(0, 0, 0);
        }
        asked = g.x;
        g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
        cost = g.x * g.y * g.z * 128u;
        groups = g;
    }
    costs[gi] = cost;
    GroupMemoryBarrierWithGroupSync();
    if (active && kind > 2) {
        // Diagnostics, in the budget buffer's next words (below): the
        // largest group count a mesh record asked for, and the groups
        // asked in all.
        uint was;
        InterlockedMax(largest_asked, asked, was);
        InterlockedAdd(all_asked, min(asked, 1u << 24), was);
    }
    // The group's own records: none for a group past a count buffer's
    // count (an execution may ask for thousands and count a few), which
    // touches the budget not at all, as its threads did not before.
    uint first = id.x - gi;
    uint n = eff_count > first ? min(64u, eff_count - first) : 0u;
    if (gi == 0 && n > 0) {
        // What the records charge, from `before` on, in lane order: a
        // record that does not fit whole runs empty and takes the rest.
        // Only the group's own records are walked (one, clamped alone).
        uint before;
        budget.InterlockedOr(0, 0, before);
        [allow_uav_condition]
        while (before < BUDGET) {
            uint next = before;
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                if (c > 0 && next < BUDGET) {
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
        for (uint k = 0; k < n; k++) {
            uint c = costs[k];
            uint fit = 1;
            if (c > 0) {
                fit = (at < BUDGET && BUDGET - at >= c) ? 1u : 0u;
                at = (at >= BUDGET || BUDGET - at <= c) ? BUDGET : (at + c);
            }
            fits[k] = fit;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (gi == 0 && kind > 2 && n > 0) {
        uint was;
        budget.InterlockedMax(4, largest_asked, was);
        budget.InterlockedAdd(8, all_asked, was);
    }
    if (!active) {
        return;
    }
    bool over = fits[gi] == 0;
    if (kind <= 1) {
        if (over) {
            draw_counts = uint2(0, 0);
        }
        dst_args.Store2(dst_at, draw_counts);
        for (uint off = 8; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    } else {
        if (over) {
            groups = uint3(0, 0, 0);
        }
        dst_args.Store3(dst_at, groups);
        for (uint off = 12; off < stride; off += 4) {
            dst_args.Store(dst_at + off, src_args.Load(src_at + off));
        }
    }
}
// A run's indirect calls (`Run`): a group each, its records a thread each.
// Groups past the run's jobs do nothing.
[RootSignature(BATCH_RS)]
[numthreads(64, 1, 1)]
void batch(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint job = group.x;
    if (job >= jobs.Load(0)) return;
    uint4 a = jobs.Load4(16 + job * 32);
    uint4 b = jobs.Load4(32 + job * 32);
    uint eff_count = a.z;
    if ((b.x & 1u) != 0) {
        eff_count = min(eff_count, count_buf.Load(b.y));
    }
    for (uint r = lane; r < eff_count; r += 64) {
        clamp_record(a.x, a.y, a.w + r * a.y, b.z + r * a.y, false);
    }
}
"#;

struct Clamp {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
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

fn make_for(device: &ID3D12Device) -> windows::core::Result<Clamp> {
    make_entry(device, s!("main"))
}

/// The clamp of a run's records ([`Run`]), made once.
fn batch_clamp() -> Option<&'static Clamp> {
    static CLAMP: std::sync::OnceLock<Option<Clamp>> = std::sync::OnceLock::new();
    CLAMP
        .get_or_init(|| {
            let made = host().ok_or_else(windows::core::Error::empty).and_then(|h| make_entry(&h.device, s!("batch")));
            match made {
                Ok(c) => Some(c),
                Err(e) => {
                    tracing::warn!(error = %e, "no clamp of runs of indirect records: each is clamped alone");
                    None
                }
            }
        })
        .as_ref()
}

/// The clamp's entry `entry` ("main", or "batch" for runs).
fn make_entry(device: &ID3D12Device, entry: windows::core::PCSTR) -> windows::core::Result<Clamp> {
    make_source_entry(device, SOURCE, entry)
}

/// The clamp compiled from `source` (ours, or in the benchmark the one it
/// replaced), entry "main".
fn make_from(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
    make_source_entry(device, source, s!("main"))
}

fn make_source_entry(device: &ID3D12Device, source: &str, entry: windows::core::PCSTR) -> windows::core::Result<Clamp> {
    let mut code: Option<ID3DBlob> = None;
    let mut errors: Option<ID3DBlob> = None;
    // SAFETY: the source and names live through the call; out pointers ours.
    let compiled = unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            s!("clamp"),
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
    let code = code.ok_or_else(windows::core::Error::empty)?;
    // SAFETY: the blob's own bytes; the root signature is inside the shader.
    unsafe {
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
        let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
            pRootSignature: std::mem::transmute_copy(&root),
            CS: D3D12_SHADER_BYTECODE {
                pShaderBytecode: bytes.as_ptr().cast(),
                BytecodeLength: bytes.len(),
            },
            ..Default::default()
        })?;
        Ok(Clamp { root, pso })
    }
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

/// Records, in `list`, the non-destructive clamp of up to `records` records of
/// `kind` (PC argument numbering) `stride` bytes apart from `src` (and bounded
/// by the 32-bit count in `count` when given) into a device-local scratch
/// buffer, at offset `0`. The scratch buffer is left in `INDIRECT_ARGUMENT`
/// until the next clamp into it: executions read it with no barrier of
/// their own. Replaces the list's compute root signature and pipeline
/// (callers bind theirs after). `None` if the clamp or alignment is not
/// available.
pub(crate) fn bound(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, count.map(|(_, off)| off))?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    clamp_into(list, c, &dst, budget, src, count, spec, false).then_some(dst)
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
pub(crate) fn bound_ahead(
    list: &HostList,
    src: (&ID3D12Resource, u64),
    spec: (u32, u32, u32),
) -> Option<ID3D12Resource> {
    let bytes = validate_layout(spec, src.1, None)?;
    let (c, budget, dst) = (clamp()?, list.budget()?, list.indirect_scratch(bytes)?);
    clamp_into(list, c, &dst, budget, src, None, spec, true).then_some(dst)
}

/// [`record_clamp_as`] into `list`'s scratch buffer `dst`, from the state
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
) -> bool {
    let before = list.indirect_state(dst);
    let recorded = record_clamp_as(c, &list.list, (dst, before), budget, src, count, spec, ahead);
    if recorded {
        list.indirect_readable(dst);
        clamp_fenced(list);
    }
    recorded
}

/// A clamp pass was recorded into `list` ([`record_clamp_as`]): two calls,
/// the second a wait for every shader write, its own included (for
/// `VICTORY_BARRIER_LAZY_UAV` and the list's barrier counts).
fn clamp_fenced(list: &HostList) {
    #[cfg(feature = "diagnostics")]
    if crate::copystat::barrier_test() & 2 != 0 {
        return;
    }
    list.fenced_by_pass(2);
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
    (dst, before): (&ID3D12Resource, D3D12_RESOURCE_STATES),
    budget: u64,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    ahead: bool,
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
    let constants = [kind, stride, records, args_offset, has_count | (u32::from(ahead) << 1), count_offset];
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
    unsafe {
        // Earlier shader writes (the title's argument and count passes) first.
        crate::barstat::record(l, crate::barstat::Why::Clamp, &first[skip..]);
        if records > 0 {
            l.SetComputeRootSignature(&c.root);
            l.SetPipelineState(&c.pso);
            l.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, src_gpu);
            l.SetComputeRootShaderResourceView(2, count_gpu);
            l.SetComputeRootUnorderedAccessView(3, dst_gpu);
            l.SetComputeRootUnorderedAccessView(4, budget);
            l.Dispatch(records.div_ceil(64), 1, 1);
        }
        crate::barstat::record(l, crate::barstat::Why::Clamp, &second[skip..]);
    }
    true
}

/// Runs of indirect calls clamped in one pass (`VICTORY_INDIRECT_BATCH`):
/// 0 off (the default: each call's records are clamped alone, a compute
/// pass between two waits for every earlier shader write, then a third
/// wait as the scratch buffer goes back), 1 or 2 on.
///
/// What a call clamped alone costs the card: 7-12 us on an RTX 5070 Ti
/// for a draw of 0.3 us (`victory-shader`'s `mesh_bench_tests`; the
/// indirect mesh draws no longer pay it with `VICTORY_MESH_AMPLIFY=1`, the
/// indirect draws, compute dispatches and mesh draws of pipelines without
/// the amplification stage still do). With a run, the first call of a
/// run records one wait for every earlier shader write, one pass that
/// clamps the records of all the run's calls (a group a call, from a job
/// table the list's later calls fill in before it is submitted) into a
/// region of the list's run buffer of its own, and one wait; the calls of
/// the run read their records there, with nothing of ours between them.
///
/// The records must be final when the pass runs, where the run starts:
/// - 1: a run ends at anything that may write a buffer's copy on the card
///   before a later call of it reads its records: a dispatch (a direct
///   one before, an indirect one after its own call), a draw, mesh draw or
///   indirect call whose shaders store to buffers (after its own call), a
///   title copy recorded on the card, another host list or the list's
///   reset. The reasoning of `VICTORY_MESH_AMPLIFY`'s one wait a run.
///   Indirect dispatches gain little: each ends its run.
/// - 2: a run ends only at the title's own barriers and pipeline syncs
///   (and a title copy on the card, another host list, the reset): the
///   title orders the passes that write its arguments before the calls
///   that read them with those, as the console's command processor needs,
///   so a dispatch between two indirect calls with no barrier between them
///   does not write the second's records. Indirect dispatches between two
///   barriers are clamped in one pass.
///
/// - 3: a run ends only at what may write a buffer's copy on the card:
///   a dispatch or draw whose shaders can store to buffer memory
///   (`crate::pipeline::Translated::stores`: buffer, global, scalar and
///   flat stores and atomics, or code not all known; after its own call
///   if it is indirect), a title copy on the card, another host list or
///   the reset. Indirect records lie in buffers: a dispatch that writes
///   only images (textures) cannot change them, so it no longer ends the
///   run as with 1. Where 1 would have begun a run only because of such a
///   dispatch, one wait for every shader write is recorded before the
///   next call of the run instead of a wait, the clamp pass and a wait:
///   the calls are ordered against the work before them as with 1 (a
///   call after a dispatch waits for it, as the run 1 began there did),
///   only the pass is saved. A dispatch that stores is clamped alone or
///   ends the run it joined, as with 1.
/// - 4: as 3, without that wait: the calls of a run are ordered by the
///   title's own barriers only (as with 2), but their records stay final.
///
/// Why 2 garbled the picture (run n38) and 3 cannot: 2 ends a run only at
/// the title's barriers, so a pass between two calls of a run that stores
/// to the arguments' buffer (a dispatch building the next call's group
/// count) with nothing of the title's between them leaves the later call
/// with the records clamped before that pass wrote them
/// (`warp_storing_pass_inside_a_run_leaves_the_later_call_stale`). And
/// every call 1 clamps alone waited for all the work before it, which
/// none of 2's calls after a run's first does.
///
/// What the title's processor writes into argument buffers reaches their
/// copies when the list is submitted, before it runs, as before (each call
/// still refreshes its records where it is recorded).
pub(crate) fn batch_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("VICTORY_INDIRECT_BATCH").ok().as_deref().map(str::trim) {
        Some("1") => 1,
        Some("2") => 2,
        Some("3") => 3,
        Some("4") => 4,
        _ => 0,
    })
}

/// Where the runs of a [`batch_mode`] end, and what orders their calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RunRules {
    /// Runs at all (0: each call clamped alone).
    pub(crate) on: bool,
    /// Every dispatch ends the run (1), not only one whose shaders store.
    pub(crate) every_dispatch: bool,
    /// A dispatch or draw whose shaders store to buffers ends the run.
    pub(crate) stores: bool,
    /// The title's barriers and pipeline syncs end the run (2).
    pub(crate) title_barriers: bool,
    /// A wait for every shader write before a call joins a run after a
    /// dispatch was recorded since the run's pass or the last such wait (3).
    pub(crate) order: bool,
}

impl RunRules {
    pub(crate) const fn of(mode: u8) -> RunRules {
        let on = mode > 0;
        RunRules {
            on,
            every_dispatch: mode == 1,
            stores: matches!(mode, 1 | 3 | 4),
            title_barriers: mode == 2,
            order: mode == 3,
        }
    }

    /// A direct dispatch whose shaders store (`stores`) ends the run.
    pub(crate) fn dispatch_ends(self, stores: bool) -> bool {
        self.on && (self.every_dispatch || (self.stores && stores))
    }

    /// An indirect call of `kind` (PC numbering) whose shaders store
    /// (`stores`) ends its run after its own call: it goes in a run only
    /// if one is open to take it.
    pub(crate) fn call_ends(self, kind: u32, stores: bool) -> bool {
        const DISPATCH: u32 = 2;
        self.on && ((self.every_dispatch && kind == DISPATCH) || (self.stores && stores))
    }
}

/// [`RunRules`] of [`batch_mode`].
pub(crate) fn run_rules() -> RunRules {
    RunRules::of(batch_mode())
}

/// The most calls one run clamps (groups of its pass).
pub(crate) const RUN_JOBS: u32 = 64;
/// A job table's bytes: the count of jobs, then eight words a job.
const JOBS_BYTES: u64 = 16 + RUN_JOBS as u64 * 32;
/// The least room a run takes in the run buffer.
const RUN_ROOM: u64 = 4096;
/// The most bytes one call's records may take to go in a run.
const JOB_MOST_BYTES: u64 = 64 << 10;
/// The run buffer's first size, and its most.
const RUN_BUFFER_FIRST: u64 = 64 << 10;
const RUN_BUFFER_MOST: u64 = 64 << 20;

/// A host list's run buffer: device-local memory the runs of the list
/// clamp their records into, a region each, given out from the start
/// after every reset (`HostList::batch_args`).
#[derive(Default)]
pub(crate) struct BatchArgs {
    buffer: Option<ID3D12Resource>,
    cap: u64,
    used: u64,
    /// A run was clamped into the buffer since the reset: it is in the
    /// indirect-argument state (buffers go back to the common one when the
    /// list has run).
    clamped: bool,
    /// Changed at every reset and when the list's own work may write a
    /// buffer's copy outside what runs see ([`BatchArgs::end_runs`]): a
    /// run of another number is over. Numbers are never given twice (a
    /// list made where a dropped one was is not taken for it); 0: none
    /// given yet.
    epoch: u64,
}

/// The next number of a run buffer's state ([`BatchArgs::epoch`]).
static NEXT_EPOCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_epoch() -> u64 {
    NEXT_EPOCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

impl BatchArgs {
    /// The run buffer's bytes, 0 if none was made.
    pub(crate) fn kept(&self) -> u64 {
        if self.buffer.is_some() { self.cap } else { 0 }
    }

    /// The list was reset: the buffer is the list's again from its start.
    pub(crate) fn rewind(&mut self) {
        self.used = 0;
        self.clamped = false;
        self.epoch = next_epoch();
    }

    /// Something of ours writes a buffer's copy in the list (a title copy
    /// recorded on the card): no run of the list goes on past it.
    pub(crate) fn end_runs(&mut self) {
        self.epoch = next_epoch();
    }

    /// The number of the buffer's state now.
    fn epoch(&mut self) -> u64 {
        if self.epoch == 0 {
            self.epoch = next_epoch();
        }
        self.epoch
    }
}

/// A run of indirect calls whose records one pass clamped (or will: the
/// pass reads the job table when the list runs) ([`batch_mode`]).
pub(crate) struct Run {
    /// The host list (its address) and its run buffer's number then.
    list: usize,
    epoch: u64,
    /// Where the pass reads records and counts (the buffers' starts; no
    /// count buffer: 0).
    src: u64,
    count: u64,
    /// The job table: its CPU address, and the jobs in it.
    jobs: usize,
    n: u32,
    /// The run buffer, the run's region there and the bytes of it taken.
    scratch: ID3D12Resource,
    base: u64,
    room: u64,
    used: u64,
}

/// Indirect calls since the last report line, by how they were clamped:
/// alone, draws (kinds 0 and 1), dispatches, mesh draws, and of all
/// those the ones with a count buffer; in runs, the same three kinds; runs
/// begun (clamp passes); calls a run could not take with runs on.
/// With 3 and 4 also: waits recorded before a call joined a run instead
/// of a new run ([`RunRules::order`]), and direct dispatches that did not
/// end a run as they cannot store to buffers.
pub(crate) static CALLS: [std::sync::atomic::AtomicU64; 11] = [const { std::sync::atomic::AtomicU64::new(0) }; 11];
const ALONE: usize = 0;
const IN_RUNS: usize = 4;
const RUNS: usize = 7;
const RUN_REFUSED: usize = 8;
const ORDER_WAITS: usize = 9;
const RUN_KEPT: usize = 10;

/// A wait before a call that joined a run ([`RunRules::order`]).
pub(crate) fn note_order_wait() {
    CALLS[ORDER_WAITS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// A dispatch inside a run that does not end it (its shaders store to no
/// buffer, [`RunRules::dispatch_ends`]).
pub(crate) fn note_run_kept() {
    CALLS[RUN_KEPT].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

/// An indirect call of `kind` (PC numbering) clamped alone or in a run.
pub(crate) fn note_call(kind: u32, in_run: bool, counted: bool) {
    use std::sync::atomic::Ordering::Relaxed;
    let class = match kind {
        0 | 1 => 0,
        2 => 1,
        _ => 2,
    };
    CALLS[if in_run { IN_RUNS } else { ALONE } + class].fetch_add(1, Relaxed);
    if counted && !in_run {
        CALLS[ALONE + 3].fetch_add(1, Relaxed);
    }
}

/// The report line of [`CALLS`] per frame, taken.
pub(crate) fn calls_line(frames: f64) -> String {
    let c = CALLS.each_ref().map(|c| c.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames.max(1.0));
    format!(
        "indirect calls per frame clamped alone (a compute pass between waits each, 7-12 us of the card's a call): {:.1} draws, {:.1} dispatches, {:.1} mesh draws ({:.1} of them with a count buffer); in runs (VICTORY_INDIRECT_BATCH={}): {:.1} draws, {:.1} dispatches, {:.1} mesh draws in {:.1} passes, {:.1} calls a run could not take; {:.1} dispatches that store to no buffer left a run open, {:.1} waits before calls that joined a run after one",
        c[0], c[1], c[2], c[3], batch_mode(), c[4], c[5], c[6], c[7], c[8], c[10], c[9]
    )
}

/// The records of an indirect call (`kind`, `stride` and up to `records`
/// of them, PC numbering, from `src`, bounded by the count in `count` if
/// given) clamped in a run ([`batch_mode`]): `run` goes on if it can take
/// the call, else a new run is begun in its place. The run buffer and the
/// offset there of the call's clamped records (in the indirect-argument
/// state until the list has run), and whether a run was begun (its pass
/// replaced the list's compute root signature and pipeline). `None`: the
/// call cannot go in a run and `run` is over; clamp it alone.
///
/// `last`: the call ends its run (its shaders may write records): it
/// goes in `run` only if that takes it, a run of one call alone costs the
/// card what a call clamped alone does (3.6-4.2 us against 3.6,
/// `run_cost`), so it is clamped alone then.
pub(crate) fn batched(
    list: &HostList,
    run: &mut Option<Run>,
    (src, src_offset): (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    (kind, stride, records): (u32, u32, u32),
    last: bool,
) -> Option<(ID3D12Resource, u64, bool)> {
    let fits = || -> Option<(u64, u64, u64, u32, Option<u32>)> {
        let bytes = validate_layout((kind, stride, records), src_offset, count.map(|(_, off)| off))?;
        if bytes > JOB_MOST_BYTES || src_offset.checked_add(bytes)? > u64::from(u32::MAX) {
            return None;
        }
        // SAFETY: live buffers.
        let src_gpu = unsafe { src.GetGPUVirtualAddress() };
        let (count_gpu, count_at) = match count {
            Some((buf, off)) => {
                if off.checked_add(4)? > u64::from(u32::MAX) {
                    return None;
                }
                // SAFETY: as above.
                (unsafe { buf.GetGPUVirtualAddress() }, Some(off as u32))
            }
            None => (0, None),
        };
        (src_gpu.is_multiple_of(4) && count_gpu.is_multiple_of(4)).then_some((bytes, src_gpu, count_gpu, src_offset as u32, count_at))
    };
    let Some((bytes, src_gpu, count_gpu, args_at, count_at)) = fits() else {
        end_run(list, run.take());
        CALLS[RUN_REFUSED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        return None;
    };
    let here = std::ptr::from_ref(list) as usize;
    let epoch = list.batch_args().epoch();
    let takes = run.as_ref().is_some_and(|r| {
        r.list == here
            && r.epoch == epoch
            && r.src == src_gpu
            && (count_at.is_none() || r.count == count_gpu)
            && r.n < RUN_JOBS
            && r.used + bytes <= r.room
    });
    let began = !takes;
    if began && last {
        end_run(list, run.take());
        return None;
    }
    if began {
        end_run(list, run.take());
        *run = begin_run(list, src_gpu, count.map(|(b, _)| b), bytes);
        if run.is_none() {
            CALLS[RUN_REFUSED].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return None;
        }
    }
    let r = run.as_mut()?;
    let at = r.used;
    let job = [kind, stride, records, args_at, u32::from(count_at.is_some()), count_at.unwrap_or(0), at as u32, 0];
    // SAFETY: the job table holds `RUN_JOBS` jobs and `r.n` is below that
    // (a run that has them all takes no more); the list is not submitted
    // while it is recorded.
    unsafe { write_job(r.jobs as *mut u8, r.n, job) };
    r.n += 1;
    r.used = (at + bytes).next_multiple_of(16);
    Some((r.scratch.clone(), r.base + at, began))
}

/// `job` as job `n` of the table at `jobs`, and the table's count after it.
///
/// # Safety
/// `jobs` is a mapped job table of [`JOBS_BYTES`], `n < RUN_JOBS`.
unsafe fn write_job(jobs: *mut u8, n: u32, job: [u32; 8]) {
    // SAFETY: inside the table, per the caller.
    unsafe {
        let words = jobs.cast::<u32>();
        for (i, w) in job.into_iter().enumerate() {
            words.add(4 + n as usize * 8 + i).write_volatile(w);
        }
        words.write_volatile(n + 1);
    }
}

/// A run's start in `list`: its region of the run buffer (at least
/// `bytes`), its job table (empty), and the wait, the clamp pass and the
/// wait recorded. `None` without the clamp, the budget or the memory.
fn begin_run(list: &HostList, src_gpu: u64, count: Option<&ID3D12Resource>, bytes: u64) -> Option<Run> {
    let c = batch_clamp()?;
    let budget = list.budget()?;
    let (jobs_gpu, jobs) = list.scratch(JOBS_BYTES)?;
    let room = bytes.max(RUN_ROOM).next_multiple_of(256);
    let (scratch, base, before, epoch, old) = {
        let mut b = list.batch_args();
        let mut old = None;
        if b.buffer.is_none() || b.used + room > b.cap {
            let cap = (b.cap * 2).max(RUN_BUFFER_FIRST).max(room).next_multiple_of(64 << 10);
            if cap > RUN_BUFFER_MOST {
                return None;
            }
            let buffer = host()?.uav_buffer(cap)?;
            crate::host::name(&buffer, || format!("indirect runs {cap:#x}"));
            old = b.buffer.replace(buffer);
            (b.cap, b.used, b.clamped) = (cap, 0, false);
        }
        let base = b.used;
        b.used += room;
        let before = if b.clamped { D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT } else { D3D12_RESOURCE_STATE_COMMON };
        b.clamped = true;
        (b.buffer.clone()?, base, before, b.epoch(), old)
    };
    if let Some(old) = old {
        // Earlier runs' calls still read it.
        list.keep(&old);
    }
    // SAFETY: the list's own mapped scratch memory of `JOBS_BYTES`.
    unsafe { jobs.cast::<u32>().write_volatile(0) };
    // SAFETY: live buffers.
    let (count_gpu, scratch_gpu) = unsafe { (count.map_or(0, |c| c.GetGPUVirtualAddress()), scratch.GetGPUVirtualAddress()) };
    let count_bound = if count.is_some() { count_gpu } else { src_gpu };
    record_run(c, &list.list, (&scratch, before, scratch_gpu + base), (src_gpu, count_bound, jobs_gpu), budget);
    // It waits for every shader write before its pass and after.
    list.fenced_by_pass(2);
    CALLS[RUNS].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Some(Run { list: std::ptr::from_ref(list) as usize, epoch, src: src_gpu, count: count_gpu, jobs: jobs as usize, n: 0, scratch, base, room, used: 0 })
}

/// Records a run's pass in `l`: a wait for every earlier shader write
/// (the title's argument passes) with the run buffer from `before` to
/// unordered access, the clamp of the jobs (read from `jobs` when the
/// list runs) into the region at `region`, and a wait with the buffer to
/// indirect arguments.
fn record_run(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    (scratch, before, region): (&ID3D12Resource, D3D12_RESOURCE_STATES, u64),
    (src, count, jobs): (u64, u64, u64),
    budget: u64,
) {
    // SAFETY: an open list; our root signature and pipeline; live buffers
    // (the list holds the run buffer and the job table until its reset,
    // the host copies through its refreshes).
    unsafe {
        crate::barstat::record(
            l,
            crate::barstat::Why::Run,
            &[uav_barrier(), transition(scratch, before, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)],
        );
        l.SetComputeRootSignature(&c.root);
        l.SetPipelineState(&c.pso);
        l.SetComputeRootShaderResourceView(0, src);
        l.SetComputeRootShaderResourceView(1, count);
        l.SetComputeRootShaderResourceView(2, jobs);
        l.SetComputeRootUnorderedAccessView(3, region);
        l.SetComputeRootUnorderedAccessView(4, budget);
        l.Dispatch(RUN_JOBS, 1, 1);
        crate::barstat::record(
            l,
            crate::barstat::Why::Run,
            &[
                uav_barrier(),
                transition(scratch, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT),
            ],
        );
    }
}

/// `run` is over: the room it did not take goes back to the run buffer
/// if nothing was given out after it.
pub(crate) fn end_run(list: &HostList, run: Option<Run>) {
    let Some(run) = run else {
        return;
    };
    if run.list != std::ptr::from_ref(list) as usize {
        return;
    }
    let mut b = list.batch_args();
    if b.epoch == run.epoch && b.buffer.as_ref() == Some(&run.scratch) && b.used == run.base + run.room {
        b.used = run.base + run.used.next_multiple_of(256);
    }
}

/// A wait in `list` for every earlier shader write (the amplified mesh
/// draws' one wait, `draw.rs`).
pub(crate) fn wait_for_writes(list: &HostList) {
    // Owed instead with `VICTORY_BARRIER_LAZY_UAV` if the list wrote
    // nothing since its last such wait (`crate::barstat`).
    list.wait_for_writes(crate::barstat::Why::AmplifyWait, &[]);
}

/// After indirect execution (or if preparation failed after [`bound`]):
/// returns the scratch buffer to the common state for the next call.
pub(crate) fn done(list: &HostList, scratch: &ID3D12Resource) {
    list.barrier(
        crate::barstat::Why::ClampDone,
        &[transition(scratch, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, D3D12_RESOURCE_STATE_COMMON)],
    );
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

/// For this fork's tests: a clamp into `dst` from the common state, as
/// the clamp was recorded before upstream left the scratch buffer readable
/// (`record_clamp_as` with `before` common).
#[cfg(test)]
fn record_clamp(
    c: &Clamp,
    l: &ID3D12GraphicsCommandList,
    dst: &ID3D12Resource,
    budget: u64,
    src: (&ID3D12Resource, u64),
    count: Option<(&ID3D12Resource, u64)>,
    spec: (u32, u32, u32),
) -> bool {
    record_clamp_as(c, l, (dst, D3D12_RESOURCE_STATE_COMMON), budget, src, count, spec, false)
}

/// For this fork's tests: the scratch buffer back to the common state after
/// its execution (what `done` did before upstream dropped it).
#[cfg(test)]
fn done_on(list: &ID3D12GraphicsCommandList, scratch: &ID3D12Resource) {
    // SAFETY: an open list and a live scratch buffer in INDIRECT_ARGUMENT.
    unsafe {
        list.ResourceBarrier(&[transition(
            scratch,
            D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            D3D12_RESOURCE_STATE_COMMON,
        )])
    };
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
        clamp: Clamp,
        worker: Clamp,
    }

    impl WarpHarness {
        fn new() -> Option<Self> {
            // SAFETY: plain DXGI WARP query.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let adapter: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
                let desc = adapter.cast::<IDXGIAdapter1>().ok()?.GetDesc1().ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                    return None;
                }
                Self::on(&adapter)
            }
        }

        /// The same harness on the first hardware adapter (for the timings
        /// of tests run by hand), if there is one.
        fn hardware() -> Option<Self> {
            // SAFETY: plain DXGI adapter queries.
            unsafe {
                let factory: IDXGIFactory4 =
                    CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).ok()?;
                let mut i = 0;
                loop {
                    let adapter: IDXGIAdapter1 = factory.EnumAdapters1(i).ok()?;
                    let desc = adapter.GetDesc1().ok()?;
                    if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 == 0 {
                        return Self::on(&adapter.cast().ok()?);
                    }
                    i += 1;
                }
            }
        }

        fn on(adapter: &IDXGIAdapter) -> Option<Self> {
            // SAFETY: plain D3D12 test harness setup.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
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
        }
    }

    fn compile_cs(device: &ID3D12Device, source: &str) -> windows::core::Result<Clamp> {
        let mut code: Option<ID3DBlob> = None;
        // SAFETY: source and entry names outlive `D3DCompile`.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                s!("worker"),
                None,
                None,
                s!("main"),
                s!("cs_5_1"),
                0,
                0,
                &mut code,
                None,
            )?;
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
            let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes)?;
            let pso = device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })?;
            Ok(Clamp { root, pso })
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

    /// Timings on the hardware adapter, by hand (`cargo test -p victory-gpu
    /// --lib card_fetch_timings -- --ignored --nocapture`): the fetch inside
    /// flights ([`crate::host::CARD_FETCH`]) of 1,024 pieces (64 MiB) into
    /// the PC's memory, as upstream did, and into the card's own memory;
    /// a copy list of the same bytes into the PC's memory (the landing
    /// thread's fetch); and the compare over the same pieces.
    #[test]
    #[ignore]
    fn card_fetch_timings() {
        use crate::host::{CARD_COMPARE, CARD_FETCH, INFLIGHT_SLOT};
        let Some(mut hw) = WarpHarness::hardware() else {
            return;
        };
        let compare = compile_cs(&hw.device, CARD_COMPARE).unwrap();
        let fetch = compile_cs(&hw.device, CARD_FETCH).unwrap();
        const PIECES: u32 = 1024;
        let size = u64::from(PIECES) << 16;
        let now = hw.default_uav(size);
        let hashes = hw.default_uav(16 * u64::from(PIECES));
        let (_differs, differs_ptr, differs_gpu) = hw.shared_buffer(4 * u64::from(PIECES));
        let (_table, table_ptr, table_gpu) = hw.shared_buffer(4 + 4 * u64::from(PIECES));
        let slots = PIECES;
        let (pc, _, pc_gpu) = hw.shared_buffer(INFLIGHT_SLOT * u64::from(slots));
        let card = hw.default_uav(INFLIGHT_SLOT * u64::from(slots));
        // SAFETY: plain queries of live buffers.
        let (now_gpu, card_gpu, hashes_gpu) =
            unsafe { (now.GetGPUVirtualAddress(), card.GetGPUVirtualAddress(), hashes.GetGPUVirtualAddress()) };
        let time = |hw: &mut WarpHarness, flagged: u32, record: &dyn Fn(&ID3D12GraphicsCommandList)| -> f64 {
            let mut best = f64::MAX;
            for _ in 0..6 {
                // SAFETY: the mappings hold `PIECES` dwords (and the counter).
                unsafe {
                    for i in 0..PIECES as usize {
                        differs_ptr.cast::<u32>().add(i).write_volatile(u32::from((i as u32) < flagged));
                    }
                    table_ptr.cast::<u32>().write_volatile(0);
                }
                record(&hw.list);
                let started = std::time::Instant::now();
                hw.submit_and_wait();
                best = best.min(started.elapsed().as_secs_f64() * 1e3);
            }
            best
        };
        let fetch_into = |list: &ID3D12GraphicsCommandList, into: u64, pieces: u32| {
            // SAFETY: our open list, live buffers, the fetch's own root signature.
            unsafe {
                list.SetComputeRootSignature(&fetch.root);
                list.SetPipelineState(&fetch.pso);
                let constants = [0u32, 0, size as u32, pieces, 0, 0, slots, pieces];
                list.SetComputeRoot32BitConstants(0, 8, constants.as_ptr().cast(), 0);
                list.SetComputeRootShaderResourceView(1, now_gpu);
                list.SetComputeRootUnorderedAccessView(2, differs_gpu);
                list.SetComputeRootUnorderedAccessView(3, table_gpu);
                list.SetComputeRootUnorderedAccessView(4, into);
                list.Dispatch(pieces, 1, 1);
            }
        };
        let empty = time(&mut hw, 0, &|_| {});
        let pc_all = time(&mut hw, PIECES, &|l| fetch_into(l, pc_gpu, PIECES));
        let card_all = time(&mut hw, PIECES, &|l| fetch_into(l, card_gpu, PIECES));
        let pc_few = time(&mut hw, 40, &|l| fetch_into(l, pc_gpu, 500));
        let card_few = time(&mut hw, 40, &|l| fetch_into(l, card_gpu, 500));
        let pc_mid = time(&mut hw, 200, &|l| fetch_into(l, pc_gpu, PIECES));
        for flagged in [300u32, 400, 500, 600, 800] {
            let ms = time(&mut hw, flagged, &|l| fetch_into(l, pc_gpu, PIECES));
            let card_ms = time(&mut hw, flagged, &|l| fetch_into(l, card_gpu, PIECES));
            println!("{flagged} flagged pieces into the PC's memory {ms:.2} ms, into the card's {card_ms:.2} ms");
        }
        // SAFETY: our open list and live buffers.
        let copy_all = time(&mut hw, 0, &|l| unsafe { l.CopyBufferRegion(&pc, 0, &now, 0, size) });
        let copy_card = time(&mut hw, 0, &|l| unsafe { l.CopyBufferRegion(&pc, 0, &card, 0, 40 << 16) });
        let compare = &compare;
        let compare_at = |pieces: u32| {
            move |list: &ID3D12GraphicsCommandList| {
                // SAFETY: our open list, live buffers, the compare's own root signature.
                unsafe {
                    list.SetComputeRootSignature(&compare.root);
                    list.SetPipelineState(&compare.pso);
                    let constants = [0u32, 0, size as u32, pieces, 0, 2];
                    list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
                    list.SetComputeRootShaderResourceView(1, now_gpu);
                    list.SetComputeRootUnorderedAccessView(2, hashes_gpu);
                    list.SetComputeRootUnorderedAccessView(3, differs_gpu);
                    list.Dispatch(pieces.div_ceil(64), 1, 1);
                }
            }
        };
        let compare_all = time(&mut hw, 0, &compare_at(PIECES));
        let compare_few = time(&mut hw, 0, &compare_at(40));
        println!(
            "empty list {empty:.2} ms; the fetch in flight of 1024 flagged pieces (64 MiB): into the PC's memory {pc_all:.2} ms, \
             into the card's {card_all:.2} ms; 40 flagged of 500: PC {pc_few:.2} ms, card {card_few:.2} ms; 200 of 1024 into the PC's {pc_mid:.2} ms; a copy list of 64 MiB \
             card -> PC {copy_all:.2} ms, of 40 pieces {copy_card:.2} ms; the compare of 1024 pieces {compare_all:.2} ms, of 40 \
             {compare_few:.2} ms"
        );
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

    /// WARP D3D12 execution test proving:
    /// (a) `countBuffer` with a small count (`3`) and `4,093` garbage tail slots
    ///     (`max = 4096`) does NOT blow `BUDGET` or zero valid dispatches in
    ///     this or subsequent `ExecuteIndirect` calls in the same list,
    /// (b) `src_args` (and `count_buf`) are NOT mutated in-place, and
    /// (c) full `stride` bytes (`stride = 24` > `12`) are copied for active
    ///     records into `dst_args` at offset `0`.
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

        let src_bytes = ARGS_OFFSET + (STRIDE as usize) * (MAX as usize) + 64;
        let (src_buf, src_ptr, _) = warp.shared_buffer(src_bytes as u64);
        let (count_buf, count_ptr, _) = warp.shared_buffer(512);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (_hits_buf, hits_ptr, hits_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer((STRIDE * 8) as u64);
        let dst_buf = warp.default_uav(u64::from(STRIDE) * u64::from(MAX));
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

        // Second indirect call's source records (at offset 0 of a second buffer,
        // or later in `count_buf`): put 2 active records + garbage tail in a
        // second range to run back-to-back in the SAME command list.
        let mut initial_count = vec![0x5Au8; 512];
        initial_count[COUNT_OFFSET_1..COUNT_OFFSET_1 + 4].copy_from_slice(&3u32.to_le_bytes());
        initial_count[COUNT_OFFSET_2..COUNT_OFFSET_2 + 4].copy_from_slice(&2u32.to_le_bytes());

        // SAFETY: buffers are mapped with exact capacities above.
        unsafe {
            std::ptr::copy_nonoverlapping(initial_src.as_ptr(), src_ptr, src_bytes);
            std::ptr::copy_nonoverlapping(initial_count.as_ptr(), count_ptr, 512);
            (budget_ptr as *mut u32).write_volatile(0);
            (hits_ptr as *mut u32).write_volatile(0);
            std::ptr::write_bytes(readback_ptr, 0, (STRIDE * 8) as usize);
        }

        // Record Call #1 (`eff_count = 3` out of `MAX = 4096`), copy the first
        // 5 records of `dst_buf` out for inspection, then record Call #2
        // (`eff_count = 2` out of `MAX = 4096`) reusing `dst_buf` in the same list!
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
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_1 as u64,
            );
        }

        // Snapshot `dst_buf` after Call #1 (copy 5 * STRIDE bytes, then leave
        // `dst_buf` readable as arguments again, as the clamp left it).
        // SAFETY: open command list; the clamp left `dst_buf` in `INDIRECT_ARGUMENT`.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, u64::from(STRIDE * 5));
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
            )]);
        }

        // Call #2 in the SAME command list (`eff_count = 2` -> records 0 and 1:
        // 6 + 5 = 11 groups). Would run empty if Call #1's tail blew `BUDGET`!
        // The scratch buffer goes straight from arguments back to unordered
        // access.
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
            warp.list.ExecuteIndirect(
                &sig,
                MAX,
                &dst_buf,
                0,
                Some(&count_buf),
                COUNT_OFFSET_2 as u64,
            );
        }

        warp.submit_and_wait();

        // SAFETY: GPU execution finished; read mapped host results.
        let (hits, spent, after_src, after_count, clamped_words) = unsafe {
            let hits = (hits_ptr as *const u32).read_volatile();
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), src_bytes).to_vec();
            let after_count = std::slice::from_raw_parts(count_ptr.cast_const(), 512).to_vec();
            let words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                (STRIDE as usize / 4) * 5,
            )
            .to_vec();
            (hits, spent, after_src, after_count, words)
        };

        // Call #1: (3*2*1) + (5*1*1) + (65535*1*1) = 65,546 groups.
        // Call #2: (3*2*1) + (5*1*1) = 11 groups.
        let expected_groups = (6 + 5 + 65_535) + (6 + 5);
        assert_eq!(
            hits, expected_groups,
            "valid indirect dispatches must all execute"
        );
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

    /// WARP D3D12 test proving `BUDGET` saturates without 32-bit unsigned
    /// wraparound across >64 huge indexed-draw records (`count = None`), while
    /// preserving `StartIndexLocation`, `BaseVertexLocation`,
    /// `StartInstanceLocation`, and custom stride padding, and leaving
    /// `src_args` untouched.
    /// A window of mesh records clamped ahead: the filled ones come out as
    /// they are and are charged, the slots of all ones come out empty and
    /// charge nothing; clamped as single records the same slots are clamped
    /// to the most and spend the budget.
    /// A run's pass (`record_run`, the shader's `batch`) clamps each job's
    /// records as the pass of one call (`main`) clamps them: the same
    /// bytes at each job's place and the same budget words, with the jobs
    /// written into the table after the pass was recorded (as a list's
    /// later calls write them) and groups past the jobs doing nothing.
    #[test]
    fn warp_run_clamps_each_job_as_alone() {
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let batch = make_entry(&warp.device, s!("batch")).expect("the run's clamp must compile on WARP");
        const SRC: usize = 4096;
        let mut src = vec![0u8; SRC];
        let mut put = |at: usize, words: &[u32]| {
            for (i, w) in words.iter().enumerate() {
                src[at + i * 4..at + i * 4 + 4].copy_from_slice(&w.to_le_bytes());
            }
        };
        put(0, &[3, 2, 7, 9]);
        put(64, &[70_000, 2, 1]);
        put(128, &[5, 1, 1, 0xAB, u32::MAX, u32::MAX, u32::MAX, 0xAC, 519, 1, 1, 0xAD]);
        for i in 0..100u32 {
            put(1024 + i as usize * 20, &[i + 1, 2, i, 0, 7]);
        }
        // (kind, stride, records, first record's offset, count's offset);
        // the count at 8 is 70 of the 100.
        let jobs: [(u32, u32, u32, u32, Option<u32>); 4] =
            [(0, 16, 1, 0, None), (2, 12, 1, 64, None), (10, 16, 3, 128, None), (1, 20, 100, 1024, Some(8))];
        let mut at = 0u64;
        let places: Vec<u64> = jobs
            .iter()
            .map(|&(_, stride, records, _, _)| {
                let here = at;
                at = (at + u64::from(stride * records)).next_multiple_of(16);
                here
            })
            .collect();
        let (src_buf, src_ptr, src_gpu) = warp.shared_buffer(SRC as u64);
        let (count_buf, count_ptr, count_gpu) = warp.shared_buffer(64);
        let (_jobs_buf, jobs_ptr, jobs_gpu) = warp.shared_buffer(JOBS_BYTES);
        let (_alone_budget, alone_budget_ptr, alone_budget_gpu) = warp.shared_buffer(256);
        let (_run_budget, run_budget_ptr, run_budget_gpu) = warp.shared_buffer(256);
        let run_dst = warp.default_uav(8192);
        let (run_back, run_back_ptr, _) = warp.shared_buffer(8192);
        let alone: Vec<_> = jobs.iter().map(|_| (warp.default_uav(4096), warp.shared_buffer(4096))).collect();
        // SAFETY: mapped buffers of these sizes.
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, SRC);
            std::ptr::write_bytes(count_ptr, 0, 64);
            count_ptr.add(8).cast::<u32>().write(70);
            std::ptr::write_bytes(jobs_ptr, 0xCD, JOBS_BYTES as usize);
            jobs_ptr.cast::<u32>().write(0);
            std::ptr::write_bytes(alone_budget_ptr, 0, 256);
            std::ptr::write_bytes(run_budget_ptr, 0, 256);
        }
        // Each call alone, as today.
        for (&(kind, stride, records, first, count), (dst, _)) in jobs.iter().zip(&alone) {
            assert!(record_clamp(
                &warp.clamp,
                &warp.list,
                dst,
                alone_budget_gpu,
                (&src_buf, u64::from(first)),
                count.map(|off| (&count_buf, u64::from(off))),
                (kind, stride, records),
            ));
            done_on(&warp.list, dst);
        }
        // The run: its pass first, its jobs written after.
        // SAFETY: a live buffer.
        let run_gpu = unsafe { run_dst.GetGPUVirtualAddress() };
        record_run(
            &batch,
            &warp.list,
            (&run_dst, D3D12_RESOURCE_STATE_COMMON, run_gpu),
            (src_gpu, count_gpu, jobs_gpu),
            run_budget_gpu,
        );
        for (n, (&(kind, stride, records, first, count), &place)) in jobs.iter().zip(&places).enumerate() {
            let job = [kind, stride, records, first, u32::from(count.is_some()), count.unwrap_or(0), place as u32, 0];
            // SAFETY: a mapped job table; the list is not submitted yet.
            unsafe { write_job(jobs_ptr, n as u32, job) };
        }
        // SAFETY: an open list; the buffers are copied for the check.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &run_dst,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&run_back, 0, &run_dst, 0, 8192);
            for (dst, (back, _, _)) in &alone {
                warp.list.ResourceBarrier(&[transition(dst, D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                warp.list.CopyBufferRegion(back, 0, dst, 0, 4096);
            }
        }
        warp.submit_and_wait();
        let words = |p: *mut u8, n: usize| -> Vec<u32> {
            // SAFETY: the work is done; a mapped buffer of at least `n` words.
            unsafe { std::slice::from_raw_parts(p.cast::<u32>().cast_const(), n).to_vec() }
        };
        let run_out = words(run_back_ptr, 2048);
        for (i, (&(_, stride, records, _, count), (_, (_, back_ptr, _)))) in jobs.iter().zip(&alone).enumerate() {
            let ran = if count.is_some() { records.min(70) } else { records };
            let n = (stride * ran / 4) as usize;
            let from = (places[i] / 4) as usize;
            assert_eq!(&run_out[from..from + n], &words(*back_ptr, n)[..], "job {i} clamped as alone");
        }
        assert_eq!(&run_out[0..4], &[3, 2, 7, 9], "the draw as asked");
        assert_eq!(&run_out[4..7], &[65_535, 2, 1], "the dispatch cut to 65,535 in x");
        assert_eq!(&run_out[12..16], &[65_535, 1, 1, 0xAC], "a slot of all ones clamped and charged, as alone");
        assert_eq!(words(run_budget_ptr, 5), words(alone_budget_ptr, 5), "the same budget and notes");
        assert_eq!(words(run_budget_ptr, 1)[0], 6 + 65_535 * 2 * 64 + (5 + 65_535 + 519) * 128 + 70 * 71);
        // The run's jobs are not run again by a second pass reading a table of none.
        drop(src_buf);
    }

    #[test]
    fn run_rules_by_mode() {
        let none = RunRules::of(0);
        assert!(!none.on && !none.dispatch_ends(true) && !none.call_ends(2, true));
        let one = RunRules::of(1);
        assert!(one.dispatch_ends(false) && one.call_ends(2, false) && !one.call_ends(0, false) && one.call_ends(10, true));
        assert!(!one.title_barriers && !one.order);
        let two = RunRules::of(2);
        assert!(!two.dispatch_ends(true) && !two.call_ends(2, true) && two.title_barriers && !two.order);
        for mode in [3, 4] {
            let r = RunRules::of(mode);
            assert!(!r.dispatch_ends(false) && r.dispatch_ends(true), "mode {mode}: only a dispatch that stores ends the run");
            assert!(!r.call_ends(2, false) && r.call_ends(2, true) && r.call_ends(0, true) && !r.call_ends(10, false));
            assert!(!r.title_barriers);
            assert_eq!(r.order, mode == 3, "mode {mode}: waits where 1 began a run");
        }
    }

    /// `VICTORY_INDIRECT_BATCH=2`'s fault (run n38): a pass that stores
    /// the next call's group count inside a run with no barrier of the
    /// title's: the run clamped that call's record before the pass wrote
    /// it, so the call runs with the old count (here none of its groups).
    /// The modes whose runs end at a dispatch that stores (1, 3, 4) clamp
    /// it in a new run after the pass and run all five.
    #[test]
    fn warp_storing_pass_inside_a_run_leaves_the_later_call_stale() {
        const WRITER_CS: &str = r#"
#define RS "UAV(u0)"
RWByteAddressBuffer args : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    args.Store3(16, uint3(5, 1, 1));
}
"#;
        let Some(mut warp) = WarpHarness::new() else {
            return;
        };
        let batch = make_entry(&warp.device, s!("batch")).expect("the run's clamp must compile on WARP");
        let writer = compile_cs(&warp.device, WRITER_CS).expect("writer shader must compile");
        let sig = warp.signature(2, 12);
        for mode in 1..=4u8 {
            let rules = RunRules::of(mode);
            let (src_buf, src_ptr, src_gpu) = warp.shared_buffer(256);
            let (_budget, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (_hits, hits_ptr, hits_gpu) = warp.shared_buffer(256);
            let (_first_jobs, first_ptr, first_gpu) = warp.shared_buffer(JOBS_BYTES);
            let (_next_jobs, next_ptr, next_gpu) = warp.shared_buffer(JOBS_BYTES);
            let run_dst = warp.default_uav(8192);
            // SAFETY: a live buffer.
            let run_gpu = unsafe { run_dst.GetGPUVirtualAddress() };
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::write_bytes(src_ptr, 0, 256);
                // The first call's record; the second's (at 16) is what
                // the title's pass writes, none yet.
                src_ptr.cast::<[u32; 3]>().write([1, 1, 1]);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(hits_ptr, 0, 256);
                first_ptr.cast::<u32>().write(0);
                next_ptr.cast::<u32>().write(0);
            }
            let worker = |warp: &WarpHarness, at: u64| {
                // SAFETY: an open list; the worker's own root signature,
                // pipeline and buffers.
                unsafe {
                    warp.list.SetComputeRootSignature(&warp.worker.root);
                    warp.list.SetPipelineState(&warp.worker.pso);
                    warp.list.SetComputeRootUnorderedAccessView(0, hits_gpu);
                    warp.list.ExecuteIndirect(&sig, 1, &run_dst, at, None, 0);
                }
            };
            // The run begins at the first call.
            record_run(&batch, &warp.list, (&run_dst, D3D12_RESOURCE_STATE_COMMON, run_gpu), (src_gpu, src_gpu, first_gpu), budget_gpu);
            // SAFETY: a mapped job table; the list is not submitted yet.
            unsafe { write_job(first_ptr, 0, [2, 12, 1, 0, 0, 0, 0, 0]) };
            worker(&warp, 0);
            // The title's pass that writes the second call's record, with
            // no barrier after it.
            // SAFETY: an open list; the writer's root signature and
            // pipeline; a live buffer.
            unsafe {
                warp.list.SetComputeRootSignature(&writer.root);
                warp.list.SetPipelineState(&writer.pso);
                warp.list.SetComputeRootUnorderedAccessView(0, src_gpu);
                warp.list.Dispatch(1, 1, 1);
            }
            let at = if rules.dispatch_ends(true) {
                // A new run: its pass waits for the writer, then clamps.
                record_run(
                    &batch,
                    &warp.list,
                    (&run_dst, D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT, run_gpu + 4096),
                    (src_gpu, src_gpu, next_gpu),
                    budget_gpu,
                );
                // SAFETY: as above.
                unsafe { write_job(next_ptr, 0, [2, 12, 1, 16, 0, 0, 0, 0]) };
                4096
            } else {
                // SAFETY: as above.
                unsafe { write_job(first_ptr, 1, [2, 12, 1, 16, 0, 0, 16, 0]) };
                16
            };
            worker(&warp, at);
            warp.submit_and_wait();
            // SAFETY: the work is done; a mapped buffer.
            let hits = unsafe { hits_ptr.cast::<u32>().read_volatile() };
            if mode == 2 {
                assert_eq!(hits, 1, "mode 2: the second call ran with the record clamped before the pass wrote it");
            } else {
                assert_eq!(hits, 6, "mode {mode}: the second call ran the five groups the pass asked for");
            }
            drop(src_buf);
        }
    }

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
            let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
            let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
            let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
            let dst_buf = warp.default_uav(total as u64);
            // SAFETY: mapped buffers of these sizes.
            unsafe {
                std::ptr::copy_nonoverlapping(src.as_ptr(), src_ptr, total);
                std::ptr::write_bytes(budget_ptr, 0, 256);
                std::ptr::write_bytes(readback_ptr, 0, total);
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
            // SAFETY: an open list; the scratch buffer, which the clamp left
            // readable as arguments, is copied for the check.
            unsafe {
                warp.list.ResourceBarrier(&[transition(
                    &dst_buf,
                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    D3D12_RESOURCE_STATE_COPY_SOURCE,
                )]);
                warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
            }
            warp.submit_and_wait();
            // SAFETY: the work is done; mapped buffers.
            let (spent, largest, out) = unsafe {
                (
                    (budget_ptr as *const u32).read_volatile(),
                    (budget_ptr as *const u32).add(1).read_volatile(),
                    std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4).to_vec(),
                )
            };
            for (i, groups) in filled.into_iter().enumerate() {
                let rec = &out[i * 8..i * 8 + 4];
                assert_eq!(rec, [groups, u32::from(groups > 0).max(1).min(1), 1, 0xAB00 + i as u32], "filled record {i}, ahead {ahead}");
            }
            let charged_filled = (5 + 519) * 128;
            if ahead {
                for i in filled.len()..RECORDS as usize {
                    assert_eq!(out[i * 8..i * 8 + 3], [0, 0, 0], "unfilled slot {i} must come out empty");
                    assert_eq!(out[i * 8 + 3], u32::MAX, "the words after the record are copied as they are");
                }
                assert_eq!(spent, charged_filled, "unfilled slots charge nothing");
                assert_eq!(largest, 519, "and are not counted as asked for");
            } else {
                // Thirteen slots at 65,535 groups: sixteen would be all of it.
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
        let (src_buf, src_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total_bytes as u64);
        let dst_buf = warp.default_uav(total_bytes as u64);

        let mut initial_src = vec![0u8; total_bytes];
        for i in 0..RECORDS as usize {
            // The first record is no draw (2^24 indices): it runs empty
            // and charges nothing.
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
            std::ptr::write_bytes(readback_ptr, 0, total_bytes);
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

        // SAFETY: open command list; copy clamped scratch buffer (left
        // readable as arguments) for verification.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list
                .CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total_bytes as u64);
        }
        warp.submit_and_wait();

        // SAFETY: GPU work completed.
        let (spent, after_src, out_words) = unsafe {
            let spent = (budget_ptr as *const u32).read_volatile();
            let after_src = std::slice::from_raw_parts(src_ptr.cast_const(), total_bytes).to_vec();
            let out_words = std::slice::from_raw_parts(
                readback_ptr.cast::<u32>().cast_const(),
                total_bytes / 4,
            )
            .to_vec();
            (spent, after_src, out_words)
        };

        assert_eq!(
            spent, BUDGET_CAP,
            "budget must saturate at BUDGET without wrapping"
        );
        assert_eq!(after_src, initial_src, "src_args must remain untouched");

        let mut kept = 0usize;
        let mut zeroed = 0usize;
        for i in 0..RECORDS as usize {
            let rec = &out_words[i * 6..(i + 1) * 6];
            if rec[0] == (1 << 22) && rec[1] == 16 {
                assert!(i == 1 || i == 2, "the first two real draws fit the budget");
                kept += 1;
            } else {
                assert_eq!((rec[0], rec[1]), (0, 0), "excess draw must be zeroed");
                zeroed += 1;
            }
            assert_eq!(rec[2], 1000 + i as u32);
            assert_eq!(rec[3], (-200i32 + i as i32) as u32);
            assert_eq!(rec[4], 5000 + i as u32);
            assert_eq!(rec[5], 0xCAFE_0000 | (i as u32));
        }
        assert_eq!((kept, zeroed), (2, (RECORDS as usize) - 2));
    }

    /// Exercises `HostList::indirect_scratch`, `bound`, the scratch buffer's
    /// tracked state, `Host::execute`, and `HostList::reset` end-to-end on
    /// the host device.
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
        // First two calls fit inside the initial 64 KiB scratch buffer and
        // must reuse the exact same `ID3D12Resource`, which stays readable
        // as arguments between them.
        let b1 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (0, 16, 64)).unwrap();
        assert_eq!(
            list.indirect_state(&b1),
            args,
            "a clamp leaves the scratch readable as arguments"
        );
        let b2 = bound(&list, (&src_buf, 64), Some((&count_buf, 0)), (1, 20, 128)).unwrap();
        assert_eq!(
            b1.as_raw(),
            b2.as_raw(),
            "scratch buffer within capacity must be reused"
        );
        assert_eq!(list.indirect_state(&b2), args);

        // A call needing >64 KiB grows the scratch buffer mid-list while keeping
        // `b1` alive until reset. The new buffer starts out common.
        let grown = list.indirect_scratch(5000 * 20).unwrap();
        assert_eq!(
            list.indirect_state(&grown),
            D3D12_RESOURCE_STATE_COMMON,
            "a new scratch buffer is common"
        );
        let b3 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 5000)).unwrap();
        assert_eq!(b3.as_raw(), grown.as_raw());
        assert_ne!(
            b3.as_raw(),
            b1.as_raw(),
            "larger request must grow scratch buffer"
        );
        assert_eq!(list.indirect_state(&b3), args);
        assert_eq!(
            list.indirect_state(&b1),
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
            list.indirect_state(&b3),
            D3D12_RESOURCE_STATE_COMMON,
            "a reset list starts with a common scratch buffer"
        );
        let b4 = bound(&list, (&src_buf, 0), Some((&count_buf, 0)), (1, 20, 4096)).unwrap();
        assert_eq!(
            b4.as_raw(),
            b3.as_raw(),
            "grown scratch buffer must persist across reset"
        );
        assert_eq!(list.indirect_state(&b4), args);
        list.close();
        assert_eq!(host.log_messages(), None, "the debug layer found an error");
        host.execute(&[&list]);
        list.reset();
    }

    /// A group charges its records in lane order: with 300 left of the
    /// budget, the first draw (100) fits, the second (250) does not and
    /// takes the rest, and the ones after it run empty. (Racing for the
    /// word record by record, which of them fitted was up to the GPU.)
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
        let (src_buf, src_ptr, _) = warp.shared_buffer(total as u64);
        let (_budget_buf, budget_ptr, budget_gpu) = warp.shared_buffer(256);
        let (readback_buf, readback_ptr, _) = warp.shared_buffer(total as u64);
        let dst_buf = warp.default_uav(total as u64);
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
            false,
        ));
        // SAFETY: an open list; the clamped records are copied for the check.
        unsafe {
            warp.list.ResourceBarrier(&[transition(
                &dst_buf,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            warp.list.CopyBufferRegion(&readback_buf, 0, &dst_buf, 0, total as u64);
        }
        warp.submit_and_wait();
        // SAFETY: the work is done; mapped buffers.
        let (spent, out) = unsafe {
            (
                (budget_ptr as *const u32).read_volatile(),
                std::slice::from_raw_parts(readback_ptr.cast::<u32>().cast_const(), total / 4)
                    .to_vec(),
            )
        };
        assert_eq!(out[0..4], [100, 1, 7, 9], "the first record fits");
        for i in 1..counts.len() {
            assert_eq!(out[i * 4..i * 4 + 4], [0, 0, 7, 9], "record {i} runs empty");
        }
        assert_eq!(spent, BUDGET_CAP, "the record that did not fit took the rest");
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
                list.list.ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
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
            list.list.ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
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

/// What an indirect call costs the card by how its records are clamped
/// (`VICTORY_INDIRECT_BATCH`): calls of one record each, indirect
/// dispatches of one group of 64 threads that store, and indirect
/// one-triangle draws, timed by one pair of timestamps around each batch.
/// A measurement, not a check (it asserts only that the card answered).
///
/// Run by name, no game needed:
/// `cargo test -p victory-gpu --release --lib -- --ignored indirect_run_cost --nocapture --test-threads 1`
#[cfg(test)]
mod run_cost {
    use windows::Win32::Foundation::{HANDLE, RECT};
    use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY};
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::Common::*;
    use windows::Win32::Graphics::Dxgi::*;
    use windows::Win32::System::Threading::{CreateEventW, INFINITE, WaitForSingleObject};
    use windows::core::{Interface, s};

    use super::{Clamp, JOBS_BYTES, done_on, make_entry, record_clamp, record_run, uav_barrier, write_job};
    use crate::host::compile;

    /// Calls in a batch.
    const CALLS: u32 = 2048;

    /// The title's stand-ins: a pass of one group that stores (as most
    /// indirect dispatches do), and a draw of one small triangle.
    const SHADERS: &str = r#"
#define CS_RS "UAV(u0)"
#define DRAW_RS "RootFlags(0)"
RWByteAddressBuffer words : register(u0);
[RootSignature(CS_RS)]
[numthreads(64, 1, 1)]
void work(uint3 id : SV_DispatchThreadID) {
    uint was;
    words.InterlockedAdd((id.x & 63) * 4, 1, was);
}
[RootSignature(DRAW_RS)]
float4 corner(uint i : SV_VertexID) : SV_Position {
    float2 p = float2((i << 1) & 2, i & 2);
    return float4(p * 0.02 - 0.01, 0.5, 1.0);
}
[RootSignature(DRAW_RS)]
float4 paint(float4 at : SV_Position) : SV_Target {
    return float4(1, 0, 0, 1);
}
"#;

    /// How the calls of a batch get their records.
    #[derive(Clone, Copy, PartialEq, Debug)]
    enum How {
        /// Straight from the title's buffer: no clamp (the floor).
        Straight,
        /// Each clamped alone, as today: a pass between two waits, the
        /// scratch buffer back to common after the call.
        Alone,
        /// In runs of `n` calls: a wait, one pass, a wait at the start of
        /// each run.
        Runs(u32),
        /// As `Runs`, with a wait before each call after a run's first
        /// (`VICTORY_INDIRECT_BATCH=3` after dispatches).
        Ordered(u32),
    }

    struct Bench {
        device: ID3D12Device,
        queue: ID3D12CommandQueue,
        allocator: ID3D12CommandAllocator,
        list: ID3D12GraphicsCommandList,
        fence: ID3D12Fence,
        submitted: u64,
        event: HANDLE,
        clamp: Clamp,
        batch: Clamp,
        work_root: ID3D12RootSignature,
        work: ID3D12PipelineState,
        draw_root: ID3D12RootSignature,
        draw: ID3D12PipelineState,
        dispatch_sig: ID3D12CommandSignature,
        draw_sig: ID3D12CommandSignature,
        /// Records of both kinds (dispatches at 0, draws after), in
        /// CPU-written memory in the common state, as the title's copies.
        args: ID3D12Resource,
        words: ID3D12Resource,
        budget: ID3D12Resource,
        alone_scratch: ID3D12Resource,
        run_scratch: ID3D12Resource,
        jobs: (ID3D12Resource, *mut u8),
        _target: ID3D12Resource,
        _rtvs: ID3D12DescriptorHeap,
        rtv: D3D12_CPU_DESCRIPTOR_HANDLE,
        stamps: ID3D12QueryHeap,
        back: ID3D12Resource,
        card: String,
    }

    const DRAWS_AT: u64 = 12 * CALLS as u64;

    fn buffer(device: &ID3D12Device, heap: D3D12_HEAP_PROPERTIES, size: u64, flags: D3D12_RESOURCE_FLAGS, state: D3D12_RESOURCE_STATES) -> ID3D12Resource {
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
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: a plain buffer description.
        unsafe { device.CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut r).unwrap() };
        r.unwrap()
    }

    fn heap(kind: D3D12_HEAP_TYPE) -> D3D12_HEAP_PROPERTIES {
        D3D12_HEAP_PROPERTIES { Type: kind, ..Default::default() }
    }

    /// CPU-written memory in the card's own memory (a GPU upload heap, as
    /// the list's scratch memory is with `VICTORY_GPU_UPLOAD_COPIES=1`),
    /// else in the PC's.
    fn card_written(device: &ID3D12Device, size: u64) -> (ID3D12Resource, *mut u8, bool) {
        let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: size,
            Height: 1,
            DepthOrArraySize: 1,
            MipLevels: 1,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
            ..Default::default()
        };
        let mut r: Option<ID3D12Resource> = None;
        let mut data = std::ptr::null_mut();
        // SAFETY: a plain buffer description; mapped for the bench's life.
        let made = unsafe {
            device
                .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_GENERIC_READ, None, &mut r)
                .and_then(|()| r.as_ref().ok_or_else(windows::core::Error::empty)?.Map(0, None, Some(&mut data)))
        };
        match (made, r) {
            (Ok(()), Some(b)) => (b, data.cast(), true),
            _ => {
                let (b, p) = written(device, size);
                (b, p, false)
            }
        }
    }

    /// CPU-written memory read by the card, in the common state (as the
    /// title's buffers' copies in the PC's memory).
    fn written(device: &ID3D12Device, size: u64) -> (ID3D12Resource, *mut u8) {
        let props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        };
        let b = buffer(device, props, size, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COMMON);
        let mut data = std::ptr::null_mut();
        // SAFETY: a CPU-visible buffer mapped for the bench's life.
        unsafe { b.Map(0, None, Some(&mut data)).unwrap() };
        (b, data.cast())
    }

    fn signature(device: &ID3D12Device, kind: D3D12_INDIRECT_ARGUMENT_TYPE, stride: u32) -> ID3D12CommandSignature {
        let arg = D3D12_INDIRECT_ARGUMENT_DESC { Type: kind, ..Default::default() };
        let desc = D3D12_COMMAND_SIGNATURE_DESC { ByteStride: stride, NumArgumentDescs: 1, pArgumentDescs: &arg, NodeMask: 0 };
        let mut sig: Option<ID3D12CommandSignature> = None;
        // SAFETY: one argument, no root arguments.
        unsafe { device.CreateCommandSignature(&desc, None::<&ID3D12RootSignature>, &mut sig).unwrap() };
        sig.unwrap()
    }

    impl Bench {
        fn new() -> Option<Self> {
            let cs = compile(SHADERS, s!("work"), s!("cs_5_1")).ok()?;
            let vs = compile(SHADERS, s!("corner"), s!("vs_5_1")).ok()?;
            let ps = compile(SHADERS, s!("paint"), s!("ps_5_1")).ok()?;
            // SAFETY: plain object creation on a device of our own.
            unsafe {
                let mut device: Option<ID3D12Device> = None;
                D3D12CreateDevice(None, D3D_FEATURE_LEVEL_12_0, &mut device).ok()?;
                let device = device?;
                let desc = CreateDXGIFactory2::<IDXGIFactory4>(DXGI_CREATE_FACTORY_FLAGS(0))
                    .and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter3>(device.GetAdapterLuid()))
                    .ok()?
                    .cast::<IDXGIAdapter1>()
                    .ok()?
                    .GetDesc1()
                    .ok()?;
                if desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0 {
                    return None;
                }
                let card = String::from_utf16_lossy(&desc.Description).trim_end_matches('\0').to_owned();
                let queue: ID3D12CommandQueue = device
                    .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() })
                    .ok()?;
                let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).ok()?;
                let list: ID3D12GraphicsCommandList = device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None).ok()?;
                let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).ok()?;
                let event = CreateEventW(None, false, false, None).ok()?;
                let clamp = make_entry(&device, s!("main")).ok()?;
                let batch = make_entry(&device, s!("batch")).ok()?;
                let work_root: ID3D12RootSignature = device.CreateRootSignature(0, &cs).ok()?;
                let work: ID3D12PipelineState = device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&work_root),
                        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr().cast(), BytecodeLength: cs.len() },
                        ..Default::default()
                    })
                    .ok()?;
                let draw_root: ID3D12RootSignature = device.CreateRootSignature(0, &vs).ok()?;
                let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
                formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
                let mut blend = D3D12_BLEND_DESC::default();
                blend.RenderTarget[0].RenderTargetWriteMask = 0xf;
                let draw: ID3D12PipelineState = device
                    .CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&draw_root),
                        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
                        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
                        BlendState: blend,
                        SampleMask: u32::MAX,
                        RasterizerState: D3D12_RASTERIZER_DESC {
                            FillMode: D3D12_FILL_MODE_SOLID,
                            CullMode: D3D12_CULL_MODE_NONE,
                            DepthClipEnable: true.into(),
                            ..Default::default()
                        },
                        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                        NumRenderTargets: 1,
                        RTVFormats: formats,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        ..Default::default()
                    })
                    .ok()?;
                let dispatch_sig = signature(&device, D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH, 12);
                let draw_sig = signature(&device, D3D12_INDIRECT_ARGUMENT_TYPE_DRAW, 16);
                let (args, args_ptr) = written(&device, DRAWS_AT + 16 * u64::from(CALLS));
                let words = args_ptr.cast::<u32>();
                for i in 0..CALLS as usize {
                    for (w, v) in [1u32, 1, 1].into_iter().enumerate() {
                        words.add(i * 3 + w).write(v);
                    }
                    for (w, v) in [3u32, 1, 0, 0].into_iter().enumerate() {
                        words.add(DRAWS_AT as usize / 4 + i * 4 + w).write(v);
                    }
                }
                let uav = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS;
                let words = buffer(&device, heap(D3D12_HEAP_TYPE_DEFAULT), 64 << 10, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                let budget = buffer(&device, heap(D3D12_HEAP_TYPE_DEFAULT), 64 << 10, uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
                let alone_scratch = buffer(&device, heap(D3D12_HEAP_TYPE_DEFAULT), 64 << 10, uav, D3D12_RESOURCE_STATE_COMMON);
                let run_scratch = buffer(&device, heap(D3D12_HEAP_TYPE_DEFAULT), 4096 * u64::from(CALLS), uav, D3D12_RESOURCE_STATE_COMMON);
                let (jobs_buf, jobs_ptr, jobs_on_card) = card_written(&device, JOBS_BYTES.next_multiple_of(256) * u64::from(CALLS));
                let jobs = (jobs_buf, jobs_ptr);
                let card = format!("{card} (job tables in the {} memory)", if jobs_on_card { "card's" } else { "PC's" });
                let target = {
                    let mut r: Option<ID3D12Resource> = None;
                    device
                        .CreateCommittedResource(
                            &heap(D3D12_HEAP_TYPE_DEFAULT),
                            D3D12_HEAP_FLAG_NONE,
                            &D3D12_RESOURCE_DESC {
                                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                                Width: 256,
                                Height: 256,
                                DepthOrArraySize: 1,
                                MipLevels: 1,
                                Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                                ..Default::default()
                            },
                            D3D12_RESOURCE_STATE_RENDER_TARGET,
                            None,
                            &mut r,
                        )
                        .ok()?;
                    r?
                };
                let rtvs: ID3D12DescriptorHeap = device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC { Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV, NumDescriptors: 1, ..Default::default() })
                    .ok()?;
                let rtv = rtvs.GetCPUDescriptorHandleForHeapStart();
                device.CreateRenderTargetView(&target, None, rtv);
                let mut stamps: Option<ID3D12QueryHeap> = None;
                device
                    .CreateQueryHeap(&D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 2, NodeMask: 0 }, &mut stamps)
                    .ok()?;
                let back = buffer(&device, heap(D3D12_HEAP_TYPE_READBACK), 16, D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
                Some(Bench {
                    device,
                    queue,
                    allocator,
                    list,
                    fence,
                    submitted: 0,
                    event,
                    clamp,
                    batch,
                    work_root,
                    work,
                    draw_root,
                    draw,
                    dispatch_sig,
                    draw_sig,
                    args,
                    words,
                    budget,
                    alone_scratch,
                    run_scratch,
                    jobs,
                    _target: target,
                    _rtvs: rtvs,
                    rtv,
                    stamps: stamps?,
                    back,
                    card,
                })
            }
        }

        /// Binds what the calls of `draws` (else dispatches) need.
        fn bind(&self, draws: bool) {
            let l = &self.list;
            // SAFETY: an open list; live objects.
            unsafe {
                if draws {
                    l.SetGraphicsRootSignature(&self.draw_root);
                    l.SetPipelineState(&self.draw);
                    l.OMSetRenderTargets(1, Some(&self.rtv), false, None);
                    l.RSSetViewports(&[D3D12_VIEWPORT { TopLeftX: 0.0, TopLeftY: 0.0, Width: 256.0, Height: 256.0, MinDepth: 0.0, MaxDepth: 1.0 }]);
                    l.RSSetScissorRects(&[RECT { left: 0, top: 0, right: 256, bottom: 256 }]);
                    l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(4));
                } else {
                    l.SetComputeRootSignature(&self.work_root);
                    l.SetPipelineState(&self.work);
                    l.SetComputeRootUnorderedAccessView(0, self.words.GetGPUVirtualAddress());
                }
            }
        }

        /// One batch of [`CALLS`] calls: dispatches, or draws; the title's
        /// wait for every shader write before each call when `barriers`.
        /// The card's microseconds a call.
        fn batch(&mut self, draws: bool, how: How, barriers: bool) -> f64 {
            let (kind, stride, first, sig) =
                if draws { (0u32, 16u32, DRAWS_AT, self.draw_sig.clone()) } else { (2, 12, 0, self.dispatch_sig.clone()) };
            let l = self.list.clone();
            // SAFETY: live buffers.
            let (args_gpu, budget, run_gpu, jobs_gpu) = unsafe {
                (self.args.GetGPUVirtualAddress(), self.budget.GetGPUVirtualAddress(), self.run_scratch.GetGPUVirtualAddress(), self.jobs.0.GetGPUVirtualAddress())
            };
            let table = JOBS_BYTES.next_multiple_of(256);
            let mut run: Option<(u64, u32)> = None;
            let mut runs = 0u64;
            let mut clamped = false;
            // SAFETY: an open list; live objects; every state moved is
            // moved back by the end of the list (or decays: buffers).
            unsafe {
                l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                self.bind(draws);
                for i in 0..CALLS {
                    let at = first + u64::from(i * stride);
                    if barriers {
                        l.ResourceBarrier(&[uav_barrier()]);
                    }
                    let (from, offset) = match how {
                        How::Straight => (self.args.clone(), at),
                        How::Alone => {
                            assert!(record_clamp(&self.clamp, &l, &self.alone_scratch, budget, (&self.args, at), None, (kind, stride, 1)));
                            self.bind(draws);
                            (self.alone_scratch.clone(), 0)
                        }
                        How::Runs(n) | How::Ordered(n) => {
                            let begins = run.is_none_or(|(_, jobs)| jobs >= n);
                            if !begins && matches!(how, How::Ordered(_)) {
                                l.ResourceBarrier(&[uav_barrier()]);
                            }
                            if begins {
                                let region = runs * 4096;
                                let before = if clamped { D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT } else { D3D12_RESOURCE_STATE_COMMON };
                                let tables = self.jobs.1.add((runs * table) as usize);
                                tables.cast::<u32>().write_volatile(0);
                                record_run(&self.batch, &l, (&self.run_scratch, before, run_gpu + region), (args_gpu, args_gpu, jobs_gpu + runs * table), budget);
                                clamped = true;
                                run = Some((region, 0));
                                runs += 1;
                                self.bind(draws);
                            }
                            let (region, jobs) = run.as_mut().unwrap();
                            let dst = u64::from(*jobs * 16);
                            let tables = self.jobs.1.add(((runs - 1) * table) as usize);
                            write_job(tables, *jobs, [kind, stride, 1, at as u32, 0, 0, dst as u32, 0]);
                            *jobs += 1;
                            (self.run_scratch.clone(), *region + dst)
                        }
                    };
                    l.ExecuteIndirect(&sig, 1, &from, offset, None, 0);
                    if how == How::Alone {
                        done_on(&l, &self.alone_scratch);
                    }
                }
                l.EndQuery(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                l.ResolveQueryData(&self.stamps, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &self.back, 0);
            }
            self.finish();
            let mut data = std::ptr::null_mut();
            // SAFETY: the read-back buffer the finished list resolved two stamps into.
            let ticks = unsafe {
                self.back.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 16 }), Some(&mut data)).unwrap();
                let s = std::slice::from_raw_parts(data.cast::<u64>(), 2).to_vec();
                self.back.Unmap(0, None);
                s[1].saturating_sub(s[0])
            };
            // SAFETY: a plain query.
            let frequency = unsafe { self.queue.GetTimestampFrequency() }.unwrap_or(1).max(1);
            ticks as f64 * 1e6 / frequency as f64 / f64::from(CALLS)
        }

        fn finish(&mut self) {
            self.submitted += 1;
            // SAFETY: our list, queue and fence; the allocator is reset
            // after the card finished.
            unsafe {
                self.list.Close().unwrap();
                let run: ID3D12CommandList = self.list.cast().unwrap();
                self.queue.ExecuteCommandLists(&[Some(run)]);
                self.queue.Signal(&self.fence, self.submitted).unwrap();
                self.fence.SetEventOnCompletion(self.submitted, self.event).unwrap();
                WaitForSingleObject(self.event, INFINITE);
                if let Err(e) = self.device.GetDeviceRemovedReason() {
                    panic!("the card was lost: {e}");
                }
                self.allocator.Reset().unwrap();
                self.list.Reset(&self.allocator, None).unwrap();
            }
        }
    }

    #[test]
    #[ignore = "a measurement on the card: run it by name with --ignored --nocapture"]
    fn indirect_run_cost() {
        let Some(mut bench) = Bench::new() else {
            eprintln!("no hardware adapter, or our shaders did not compile; skipped");
            return;
        };
        eprintln!("indirect call cost on {}: {CALLS} calls of one record a batch, the least of five", bench.card);
        bench.batch(false, How::Straight, false);
        bench.batch(true, How::Straight, false);
        let mut least = Vec::new();
        for (draws, what) in [(false, "dispatches of one group that stores"), (true, "draws of one small triangle")] {
            for barriers in [false, true] {
                for how in [How::Straight, How::Alone, How::Runs(1), How::Runs(8), How::Runs(64), How::Ordered(8), How::Ordered(64)] {
                    let mut us: Vec<f64> = (0..5).map(|_| bench.batch(draws, how, barriers)).collect();
                    us.sort_by(f64::total_cmp);
                    eprintln!(
                        "{what}, {}: {how:?}: {:.2} us a call (median {:.2}, most {:.2})",
                        if barriers { "the title's wait before each" } else { "no wait of the title's" },
                        us[0], us[2], us[4]
                    );
                    least.push(us[0]);
                }
            }
        }
        assert!(least.iter().all(|us| us.is_finite() && *us > 0.0), "the card gave no time for a batch");
    }
}
