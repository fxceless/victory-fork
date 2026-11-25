//! Clamps for an actual prefix of indirect draws, before its host list.
//!
//! Only single-record, count-free draws whose shaders cannot store may join.
//! A write, copy, barrier, query or other execution seals the prefix forever
//! until reset. No list is cut and no future record is guessed. Each requested
//! record gets its own scratch slot. Independent preparation dispatches share
//! one compute binding with no barriers between them; one final group charges
//! the original list budget in request order. The prelude runs again on every
//! execution of the closed list. All resources and the allocator live through
//! that list's last fence, as the list itself does.

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::{Clamp, make_from, uav_barrier};
use crate::host::{HostList, host, transition};

/// Bounded storage per active list: 64 KiB of arguments and 4 KiB of metadata
/// (D3D12 may allocate a full 64 KiB for the latter). Later calls clamp inline.
pub(super) const CAPACITY: u32 = 256;
const SLOT: u32 = 256;

pub(crate) fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_INDIRECT_BATCH").is_some_and(|v| v == "1")
            // Serial per-command timing measures a different schedule.
            && !std::env::var_os("VICTORY_GPU_TIMING").is_some_and(|v| v == "1")
    }) && !crate::drvwatch::tripped()
}

pub(crate) fn accepts(kind: u32, max: u32, counted: bool, stores: bool) -> bool {
    matches!(kind, 0 | 1 | 10) && max == 1 && !counted && !stores
}

// Exactly the ordinary clamp's bounds, with no budget mutation. The
// differential GPU test checks counts and padding against that shader.
const PREPARE: &str = r#"
#define RS "RootConstants(num32BitConstants=4, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Record : register(b0) { uint kind; uint stride; uint dst_at; uint meta_at; };
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer meta : register(u1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint cost = 0;
    uint asked = 0;
    uint payload;
    if (kind <= 1) {
        uint2 want = src.Load2(0);
        uint count = min(want.x, 1u << 24);
        uint instances = min(want.y, (1u << 26) / max(count, 1u));
        cost = count * instances;
        if (count >= (1u << 23)) { count = 0; instances = 0; cost = 0; }
        dst.Store2(dst_at, uint2(count, instances));
        payload = 8;
    } else {
        uint3 g = src.Load3(0);
        if (kind == 2) {
            g = min(g, uint3(65535u, 65535u, 65535u));
            if (g.z != 0 && g.x * g.y > (1u << 20) / g.z) g = uint3(0, 0, 0);
            cost = g.x * g.y * g.z * 64u;
        } else {
            asked = g.x;
            g = uint3(min(g.x, 65535u), min(g.y, 1u), min(g.z, 1u));
            cost = g.x * g.y * g.z * 128u;
        }
        dst.Store3(dst_at, g);
        payload = 12;
    }
    for (uint off = payload; off < stride; off += 4) dst.Store(dst_at + off, src.Load(off));
    meta.Store4(meta_at, uint4(cost, dst_at, kind, asked));
}
"#;

// Only this group touches the budget. Preparation writes disjoint slots;
// all consumers follow this group. Zero-cost records fit even after budget
// exhaustion, exactly as they do in the ordinary clamp. Overflowing records
// run empty and saturate, rather than letting a later smaller record fit.
const CHARGE: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), UAV(u0), UAV(u1), UAV(u2)"
cbuffer Batch : register(b0) { uint records; };
RWByteAddressBuffer dst : register(u0);
RWByteAddressBuffer meta : register(u1);
RWByteAddressBuffer budget : register(u2);
groupshared uint costs[64];
groupshared uint fits[64];
groupshared uint at;
groupshared uint largest;
groupshared uint all;
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint t : SV_GroupIndex) {
    const uint BUDGET = 1u << 27;
    if (t == 0) { at = budget.Load(0); largest = budget.Load(4); all = budget.Load(8); }
    GroupMemoryBarrierWithGroupSync();
    for (uint base = 0; base < records; base += 64) {
        uint n = min(64u, records - base);
        uint4 m = uint4(0, 0, 0, 0);
        if (t < n) m = meta.Load4((base + t) * 16);
        costs[t] = m.x;
        GroupMemoryBarrierWithGroupSync();
        if (t == 0) {
            for (uint j = 0; j < n; j++) {
                uint c = costs[j];
                fits[j] = (c == 0 || (at < BUDGET && BUDGET - at >= c)) ? 1u : 0u;
                if (c > 0) at = (at >= BUDGET || BUDGET - at <= c) ? BUDGET : at + c;
                uint4 note = meta.Load4((base + j) * 16);
                if (note.z > 2) {
                    largest = max(largest, note.w);
                    all += min(note.w, 1u << 24);
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if (t < n && fits[t] == 0) {
            if (m.z <= 1) dst.Store2(m.y, uint2(0, 0));
            else dst.Store3(m.y, uint3(0, 0, 0));
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0) budget.Store3(0, uint3(at, largest, all));
}
"#;

fn shaders() -> Option<&'static (Clamp, Clamp)> {
    static SHADERS: std::sync::OnceLock<Option<(Clamp, Clamp)>> = std::sync::OnceLock::new();
    SHADERS
        .get_or_init(|| {
            let device = &host()?.device;
            match make_from(device, PREPARE).and_then(|p| make_from(device, CHARGE).map(|c| (p, c)))
            {
                Ok(pair) => Some(pair),
                Err(e) => {
                    tracing::warn!(error = %e, "no indirect batch shaders: calls clamp inline");
                    None
                }
            }
        })
        .as_ref()
}

struct Storage {
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    args: ID3D12Resource,
    meta: ID3D12Resource,
    budget: u64,
    meter: crate::cardmeter::Span,
}

impl Storage {
    fn new() -> Option<Self> {
        let host = host()?;
        let args = host.uav_buffer(u64::from(CAPACITY * SLOT))?;
        let meta = host.uav_buffer(u64::from(CAPACITY * 16))?;
        // SAFETY: ordinary D3D12 object creation; the empty list is closed
        // so each recording begins through the same reset path.
        let (allocator, list) = unsafe {
            let allocator: ID3D12CommandAllocator = host
                .device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .ok()?;
            let list: ID3D12GraphicsCommandList = host
                .device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .ok()?;
            list.Close().ok()?;
            (allocator, list)
        };
        Some(Self {
            allocator,
            list,
            args,
            meta,
            budget: 0,
            meter: crate::cardmeter::Span::NONE,
        })
    }

    fn begin(&mut self, owner: &HostList) -> Option<()> {
        // SAFETY: the owner is reset only after its last submission finished;
        // this allocator and these buffers have the same lifetime.
        unsafe {
            self.allocator.Reset().ok()?;
            self.list.Reset(&self.allocator, None).ok()?;
        }
        self.meter = crate::cardmeter::begin(&self.list, crate::cardmeter::Kind::Lists);
        self.budget = owner.budget_on(&self.list)?;
        Some(())
    }

    fn start_prepares(&self, prepare: &Clamp) {
        // SAFETY: an open list and our buffers, common at every recording's
        // start. The global barrier orders earlier title-list producers.
        unsafe {
            self.list.ResourceBarrier(&[
                uav_barrier(),
                transition(
                    &self.args,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
                transition(
                    &self.meta,
                    D3D12_RESOURCE_STATE_COMMON,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
            ]);
            self.list.SetComputeRootSignature(&prepare.root);
            self.list.SetPipelineState(&prepare.pso);
            self.list
                .SetComputeRootUnorderedAccessView(2, self.args.GetGPUVirtualAddress());
            self.list
                .SetComputeRootUnorderedAccessView(3, self.meta.GetGPUVirtualAddress());
        }
    }

    fn prepare(&self, index: u32, src_gpu: u64, kind: u32, stride: u32) {
        let constants = [kind, stride, index * SLOT, index * 16];
        // SAFETY: an open prelude with the preparation pipeline; unique
        // argument and metadata slots. No dispatch touches the budget.
        unsafe {
            self.list
                .SetComputeRoot32BitConstants(0, 4, constants.as_ptr().cast(), 0);
            self.list.SetComputeRootShaderResourceView(1, src_gpu);
            self.list.Dispatch(1, 1, 1);
        }
    }
}

struct First {
    src: ID3D12Resource,
    offset: u64,
    kind: u32,
    stride: u32,
    gpu: u64,
}

/// One prefix per native list. Once blocked it cannot restart, because a
/// second prefix would move clamps before the intervening producer/query.
#[derive(Default)]
pub(crate) struct Batch {
    blocked: bool,
    open: bool,
    count: u32,
    common: bool,
    storage: Option<Storage>,
    first: Option<First>,
}

// SAFETY: D3D12 objects are free-threaded; the owning HostList serializes
// recording with its batch mutex and waits for all executions before reset.
unsafe impl Send for Batch {}

impl Batch {
    pub(crate) fn append(
        &mut self,
        owner: &HostList,
        src: &ID3D12Resource,
        src_offset: u64,
        kind: u32,
        stride: u32,
    ) -> Option<(ID3D12Resource, u64)> {
        if self.blocked
            || self.count == CAPACITY
            || stride > SLOT
            || !accepts(kind, 1, false, false)
        {
            return None;
        }
        super::validate_layout((kind, stride, 1), src_offset, None)?;
        // SAFETY: a live D3D12 buffer. Check the complete stride, including
        // padding; rebase the root SRV so offsets past 4 GiB do not truncate.
        let src_gpu = unsafe {
            let desc = src.GetDesc();
            if desc.Dimension != D3D12_RESOURCE_DIMENSION_BUFFER
                || src_offset.checked_add(u64::from(stride))? > desc.Width
            {
                return None;
            }
            src.GetGPUVirtualAddress().checked_add(src_offset)?
        };
        if src_gpu == 0 || !src_gpu.is_multiple_of(4) {
            return None;
        }
        let (prepare, _) = shaders()?;
        super::clamp()?;
        if self.storage.is_none() {
            self.storage = Some(Storage::new()?);
        }
        let storage = self.storage.as_mut()?;
        if !self.open {
            if storage.begin(owner).is_none() {
                // SAFETY: the failed start was not submitted. Leave the
                // allocator closed before any later reset or fallback.
                let _ = unsafe { storage.list.Close() };
                self.blocked = true;
                return None;
            }
            self.open = true;
        }
        let dst_at = self.count * SLOT;
        owner.keep(src);
        if self.count == 0 {
            // Delay the first prepare until a second call joins. A prefix
            // of one then uses the ordinary one-dispatch clamp in its prelude.
            self.first = Some(First {
                src: src.clone(),
                offset: src_offset,
                kind,
                stride,
                gpu: src_gpu,
            });
        } else {
            if self.count == 1 {
                storage.start_prepares(prepare);
                let first = self.first.take().expect("the first prepare was deferred");
                storage.prepare(0, first.gpu, first.kind, first.stride);
            }
            storage.prepare(self.count, src_gpu, kind, stride);
        }
        self.count += 1;
        #[cfg(feature = "diagnostics")]
        crate::copystat::add(crate::copystat::C::IndirectBounded, 1);
        Some((storage.args.clone(), u64::from(dst_at)))
    }

    /// Close the preparation prefix. Keep scratch readable through the
    /// consumer list's actual end: a preparation helper can block a prefix
    /// before the most recently appended draw has recorded its execution.
    pub(crate) fn seal(&mut self) {
        self.blocked = true;
        if !self.open {
            return;
        }
        let storage = self.storage.as_ref().expect("an open batch has storage");
        let (_, charge) = shaders().expect("an open batch has compiled shaders");
        if self.count == 1 {
            let first = self.first.take().expect("the single prepare was deferred");
            let clamp = super::clamp().expect("the ordinary clamp was compiled before opening");
            assert!(
                super::record_clamp_as(
                    clamp,
                    &storage.list,
                    (&storage.args, D3D12_RESOURCE_STATE_COMMON),
                    storage.budget,
                    (&first.src, first.offset),
                    None,
                    (first.kind, first.stride, 1),
                    false,
                ),
                "a validated single-record batch must clamp"
            );
        } else {
            // SAFETY: the two lists are open. One UAV barrier orders all disjoint
            // preparations before the ordered charge. Scratch transitions order
            // its final writes before consumers and its last reads before reset.
            unsafe {
                storage.list.ResourceBarrier(&[uav_barrier()]);
                storage.list.SetComputeRootSignature(&charge.root);
                storage.list.SetPipelineState(&charge.pso);
                storage
                    .list
                    .SetComputeRoot32BitConstants(0, 1, (&raw const self.count).cast(), 0);
                storage
                    .list
                    .SetComputeRootUnorderedAccessView(1, storage.args.GetGPUVirtualAddress());
                storage
                    .list
                    .SetComputeRootUnorderedAccessView(2, storage.meta.GetGPUVirtualAddress());
                storage
                    .list
                    .SetComputeRootUnorderedAccessView(3, storage.budget);
                storage.list.Dispatch(1, 1, 1);
                storage.list.ResourceBarrier(&[
                    transition(
                        &storage.args,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                    ),
                    transition(
                        &storage.meta,
                        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        D3D12_RESOURCE_STATE_COMMON,
                    ),
                ]);
            }
        }
        crate::cardmeter::end(&storage.list, storage.meter);
        // SAFETY: all preparation commands are recorded. Consumer commands
        // keep the scratch readable, including a pending draw's execution.
        unsafe {
            if let Err(e) = storage.list.Close() {
                crate::host::fail(format_args!("an indirect batch did not close ({e})"));
            }
        }
        self.open = false;
    }

    pub(crate) fn finish_owner(&mut self, owner: &HostList) {
        self.seal();
        if self.count == 0 || self.common {
            return;
        }
        let storage = self.storage.as_ref().expect("a batch has storage");
        // SAFETY: the owner's last draw is recorded. Every replay ends in
        // COMMON before the next execution's preparation writes this buffer.
        unsafe {
            owner.list.ResourceBarrier(&[transition(
                &storage.args,
                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                D3D12_RESOURCE_STATE_COMMON,
            )])
        };
        self.common = true;
    }

    pub(crate) fn prelude(&self) -> Option<ID3D12CommandList> {
        if self.count == 0 {
            return None;
        }
        assert!(
            !self.open,
            "an indirect batch must be sealed before submission"
        );
        self.storage.as_ref()?.list.cast().ok()
    }

    pub(crate) fn reset(&mut self) {
        if self.open {
            // SAFETY: an abandoned recording was never submitted. Its owner
            // resets both native lists; no encoded scratch transition ran.
            if let Some(storage) = &self.storage {
                let _ = unsafe { storage.list.Close() };
            }
        }
        self.blocked = false;
        self.open = false;
        self.count = 0;
        self.common = false;
        self.first = None;
    }
}

#[cfg(test)]
mod tests;
