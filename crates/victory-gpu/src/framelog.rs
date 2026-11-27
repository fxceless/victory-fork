//! Cumulative stage counters for structured benchmark reports. The normal
//! interval meters and timeline remain independent of these opt-in totals.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::{Direct3D12::*, Dxgi::Common::*};
use windows::core::Interface;

use crate::bench::FrameLogTotals;
use crate::host::Host;

#[derive(Clone, Copy)]
#[repr(usize)]
pub(crate) enum Stage {
    Prepare,
    CardWait,
    CardIdle,
    Copy,
    CopyBack,
    Remap,
    Refresh,
    PipelineWait,
    GuardFault,
    CardSync,
    CardCheck,
    CardLists,
}

static TOTAL: [AtomicU64; Stage::CardLists as usize + 1] =
    [const { AtomicU64::new(0) }; Stage::CardLists as usize + 1];

pub(crate) fn add(stage: Stage, value: u64) {
    if crate::bench::active() {
        TOTAL[stage as usize].fetch_add(value, Relaxed);
    }
}

// A snapshot must see both sides of an interval reset, or neither.
static INTERVALS: Mutex<()> = Mutex::new(());

pub(crate) fn reset_interval(counter: &AtomicU64, stage: Option<Stage>) -> u64 {
    if let Some(stage) = stage.filter(|_| crate::bench::active()) {
        let _guard = INTERVALS.lock().unwrap_or_else(|e| e.into_inner());
        let value = counter.swap(0, Relaxed);
        TOTAL[stage as usize].fetch_add(value, Relaxed);
        value
    } else {
        counter.swap(0, Relaxed)
    }
}

pub(crate) fn snapshot_totals() -> FrameLogTotals {
    use Stage::*;
    let _guard = INTERVALS.lock().unwrap_or_else(|e| e.into_inner());
    let get = |stage: Stage| {
        TOTAL[stage as usize].load(Relaxed) + crate::host::bench_interval(stage)
    };
    FrameLogTotals {
        prepare_us: get(Prepare),
        card_wait_us: get(CardWait),
        card_idle_us: get(CardIdle),
        copy_us: get(Copy),
        copyback_us: get(CopyBack),
        remap_us: get(Remap),
        refresh_us: get(Refresh),
        pipeline_wait_us: get(PipelineWait),
        guard_faults: get(GuardFault),
        card_sync_us: get(CardSync),
        card_check_us: get(CardCheck),
        card_lists_us: get(CardLists),
    }
}

/// A submission's timestamp pairs, returned to the pool only after its fence
/// completes. The sync and compare pairs are recorded in their own lists;
/// two tiny marker lists bracket the actual submission lists.
pub(crate) struct GpuTiming {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
    markers: [ID3D12CommandList; 2],
    _allocator: ID3D12CommandAllocator,
    frequency: u64,
    used: [bool; 3],
}

// SAFETY: free-threaded D3D12 objects; timing slots move between the submit
// and landing threads and the pool owns only slots with no GPU work in flight.
unsafe impl Send for GpuTiming {}

static FREE: Mutex<Vec<GpuTiming>> = Mutex::new(Vec::new());

impl GpuTiming {
    pub(crate) fn take(host: &Host) -> Option<Self> {
        if !crate::bench::active() {
            return None;
        }
        if let Some(slot) = FREE.lock().unwrap_or_else(|e| e.into_inner()).pop() {
            return Some(slot);
        }
        match Self::new(host) {
            Ok(slot) => Some(slot),
            Err(error) => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| tracing::warn!(%error, "benchmark GPU timestamps unavailable"));
                None
            }
        }
    }

    fn new(host: &Host) -> windows::core::Result<Self> {
        // SAFETY: plain creation on the direct queue's device. The readback
        // buffer has room for six 64-bit timestamps at aligned offsets.
        unsafe {
            let frequency = host.queue().GetTimestampFrequency()?;
            let mut heap: Option<ID3D12QueryHeap> = None;
            host.device.CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: 6, NodeMask: 0 },
                &mut heap,
            )?;
            let heap = heap.ok_or_else(windows::core::Error::empty)?;
            let props = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: 48,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                ..Default::default()
            };
            let mut back: Option<ID3D12Resource> = None;
            host.device.CreateCommittedResource(
                &props, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back,
            )?;
            let back = back.ok_or_else(windows::core::Error::empty)?;
            let allocator: ID3D12CommandAllocator =
                host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)?;
            let list = |end: bool| -> windows::core::Result<ID3D12CommandList> {
                let list: ID3D12GraphicsCommandList = host.device.CreateCommandList(
                    0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None,
                )?;
                list.EndQuery(&heap, D3D12_QUERY_TYPE_TIMESTAMP, if end { 5 } else { 4 });
                if end {
                    list.ResolveQueryData(&heap, D3D12_QUERY_TYPE_TIMESTAMP, 4, 2, &back, 32);
                }
                list.Close()?;
                list.cast()
            };
            Ok(Self {
                markers: [list(false)?, list(true)?],
                heap,
                back,
                _allocator: allocator,
                frequency: frequency.max(1),
                used: [false; 3],
            })
        }
    }

    pub(crate) fn mark_preparation(&mut self, list: &ID3D12GraphicsCommandList, compare: bool, end: bool) {
        let pair = usize::from(compare);
        let start = 2 * pair as u32;
        // SAFETY: an open list on the same direct queue, with a private pair
        // of queries and an aligned range in the slot's readback buffer.
        unsafe {
            list.EndQuery(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, start + u32::from(end));
            if end {
                list.ResolveQueryData(&self.heap, D3D12_QUERY_TYPE_TIMESTAMP, start, 2, &self.back, u64::from(start) * 8);
                self.used[pair] = true;
            }
        }
    }

    pub(crate) fn bracket_lists(&mut self) -> [ID3D12CommandList; 2] {
        self.used[2] = true;
        self.markers.clone()
    }

    pub(crate) fn finish(mut self) {
        let mut data = std::ptr::null_mut();
        // SAFETY: the calling submission's fence has completed. Read only
        // pairs used by that submission, and do not write to the mapping.
        unsafe {
            if self.back.Map(0, None, Some(&mut data)).is_ok() {
                let ticks = data.cast::<u64>();
                if !ticks.is_null() {
                    let stages = [Stage::CardSync, Stage::CardCheck, Stage::CardLists];
                    for (i, stage) in stages.into_iter().enumerate() {
                        if self.used[i] {
                            let begin = ticks.add(2 * i).read();
                            let end = ticks.add(2 * i + 1).read();
                            let us = u128::from(end.saturating_sub(begin)) * 1_000_000
                                / u128::from(self.frequency);
                            add(stage, us as u64);
                        }
                    }
                }
                self.back.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            }
        }
        self.used = [false; 3];
        FREE.lock().unwrap_or_else(|e| e.into_inner()).push(self);
    }
}
