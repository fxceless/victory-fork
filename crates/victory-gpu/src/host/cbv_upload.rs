//! Exact conventional CBV reuse within one recording, never across lists.
use super::*;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CBV_REUSE_UPLOAD").is_some_and(|v| v == "1"))
}

#[derive(Default)]
pub(super) struct Cache {
    entries: std::collections::HashMap<(u64, u32), u64>,
    scanned: usize,
}

impl Cache {
    fn lookup(&mut self, copies: &[Step], src: u64, size: u32) -> Option<u64> {
        if self.scanned > copies.len()
            || copies[self.scanned..]
                .iter()
                .any(|s| !matches!(s, Step::Fill(..)))
        {
            self.entries.clear();
        }
        self.scanned = copies.len();
        self.entries.get(&(src, size)).copied()
    }

    fn remember(&mut self, src: u64, size: u32, gpu: u64, steps: usize) {
        if self.entries.len() >= 4096 {
            self.entries.clear();
        }
        self.entries.insert((src, size), gpu);
        self.scanned = steps;
    }
}

fn eligible(src: u64, size: u32) -> bool {
    src != 0
        && size != 0
        && size <= 65_536
        && size.is_multiple_of(256)
        && src.checked_add(u64::from(size)).is_some()
}

impl HostList {
    /// Preserve the original scratch path for malformed or unaligned sizes.
    /// Card fills are still handled by the caller before this function.
    pub(crate) fn table_cbv_upload(&self, src: u64, size: u32) -> Option<u64> {
        let reuse = enabled() && eligible(src, size);
        if reuse {
            let mut d = self.deferred();
            let d = &mut *d;
            if let Some(gpu) = d.cbv_uploads.lookup(&d.copies, src, size) {
                return Some(gpu);
            }
        }
        let (gpu, to) = self.scratch(u64::from(size))?;
        self.defer_fill(to, src as *const u8, size as usize);
        if reuse {
            let mut d = self.deferred();
            let steps = d.copies.len();
            d.cbv_uploads.remember(src, size, gpu, steps);
        }
        Some(gpu)
    }
}

#[test]
fn exact_table_cbv_cache_stops_at_non_fill_steps() {
    let mut c = Cache::default();
    let mut steps = vec![Step::Fill(0x2000, 0x1000, 256)];
    c.remember(0x1000, 256, 0x3000, steps.len());
    steps.push(Step::Fill(0x4000, 0x5000, 256));
    assert_eq!(c.lookup(&steps, 0x1000, 256), Some(0x3000));
    assert_eq!(c.lookup(&steps, 0x1000, 512), None);
    assert_eq!(c.lookup(&steps, 0x1001, 256), None);
    steps.push(Step::Copy(0x1000, 0x2000, 256));
    assert_eq!(c.lookup(&steps, 0x1000, 256), None);
    c.remember(0x1000, 256, 0x3000, steps.len());
    steps.push(Step::Run(Box::new(|| {}), None));
    assert_eq!(c.lookup(&steps, 0x1000, 256), None);
    c.remember(0x1000, 256, 0x3000, steps.len());
    steps.push(Step::Host(Box::new(|| {})));
    assert_eq!(c.lookup(&steps, 0x1000, 256), None);
    c.remember(0x1000, 256, 0x3000, steps.len());
    assert_eq!(c.lookup(&[], 0x1000, 256), None);
}

#[test]
fn table_cbv_reuse_only_accepts_bounded_aligned_sizes() {
    assert!(eligible(0x1000, 256));
    assert!(eligible(0x1000, 65_536));
    assert!(!eligible(0, 256));
    assert!(!eligible(0x1000, 0));
    assert!(!eligible(0x1000, 4));
    assert!(!eligible(0x1000, 65_792));
    assert!(!eligible(u64::MAX - 127, 256));
}

#[test]
#[ignore = "isolated real D3D12 adapter; run in separate processes with upload reuse off/on"]
fn hardware_exact_table_cbv_upload_replay() {
    use crate::descriptor::host::HostHeap;
    use windows::Win32::Graphics::Dxgi::{
        CreateDXGIFactory2, DXGI_ADAPTER_FLAG_SOFTWARE, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1,
        IDXGIFactory4,
    };
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };
    const CODE: &str = r#"
cbuffer Values : register(b0) { uint4 values[16]; };
cbuffer Index : register(b1) { uint dst; uint at; };
RWByteAddressBuffer result : register(u0);
[RootSignature("DescriptorTable(CBV(b0)),RootConstants(num32BitConstants=2,b1),UAV(u0)")]
[numthreads(1,1,1)]
void main() { result.Store4(dst * 16, values[at]); }
"#;
    let host = host().expect("D3D12 adapter");
    let reuse = enabled();
    // SAFETY: synthetic title allocation, descriptor heap, list and output live
    // through both fenced executions, with no external title process involved.
    unsafe {
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        let desc = adapter.GetDesc1().unwrap();
        assert_eq!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);
        eprintln!(
            "CBV adapter: {}",
            String::from_utf16_lossy(&desc.Description).trim_end_matches(char::from(0))
        );
        let code = compile(CODE, windows::core::s!("main"), windows::core::s!("cs_5_1")).unwrap();
        let root: ID3D12RootSignature = host.device.CreateRootSignature(0, &code).unwrap();
        let pso: ID3D12PipelineState = host
            .device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: code.as_ptr().cast(),
                    BytecodeLength: code.len(),
                },
                ..Default::default()
            })
            .unwrap();
        let heap = HostHeap::new(false, 3).unwrap();
        let first = heap.dynamic(3).unwrap();
        let title = VirtualAlloc(
            std::ptr::null(),
            crate::watch::PAGE,
            MEM_RESERVE | MEM_COMMIT,
            PAGE_READWRITE,
        )
        .cast::<u8>();
        assert!(!title.is_null());
        let src = title.add(16);
        let list = host.new_list().unwrap();
        let a = list.table_cbv_upload(src as u64, 256).unwrap();
        let b = list.table_cbv_upload(src as u64, 256).unwrap();
        assert_eq!(a == b, reuse);
        let at = src as usize;
        let changed = [101u32, 102, 103, 104];
        list.defer_at(at, 16, move || {
            std::ptr::copy_nonoverlapping(changed.as_ptr(), at as *mut u32, 4);
        });
        let c = list.table_cbv_upload(src as u64, 256).unwrap();
        assert_ne!(a, c);
        assert_ne!(b, c);
        for (i, gpu) in [a, b, c].into_iter().enumerate() {
            heap.constant_at(first + i as u32, gpu, 256);
        }
        let fills = list
            .deferred()
            .copies
            .iter()
            .filter(|s| matches!(s, Step::Fill(..)))
            .count();
        assert_eq!(fills, if reuse { 2 } else { 3 });
        let out = host.uav_buffer(64).unwrap();
        let read = host.read_back_buffer(64).unwrap();
        let mut mapped = std::ptr::null_mut();
        read.Map(0, None, Some(&mut mapped)).unwrap();
        list.list.SetComputeRootSignature(&root);
        list.list.SetPipelineState(&pso);
        list.list.SetDescriptorHeaps(&[Some(heap.heap.clone())]);
        list.list
            .SetComputeRootUnorderedAccessView(2, out.GetGPUVirtualAddress());
        list.list.ResourceBarrier(&[transition(
            &out,
            D3D12_RESOURCE_STATE_COMMON,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        )]);
        for (i, (slot, index)) in [(0, 0), (1, 0), (2, 0), (2, 15)].into_iter().enumerate() {
            list.list
                .SetComputeRootDescriptorTable(0, heap.gpu_handle(first + slot));
            list.list.SetComputeRoot32BitConstant(1, i as u32, 0);
            list.list.SetComputeRoot32BitConstant(1, index, 1);
            list.list.Dispatch(1, 1, 1);
        }
        list.list.ResourceBarrier(&[transition(
            &out,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        list.list.CopyBufferRegion(&read, 0, &out, 0, 64);
        list.list.ResourceBarrier(&[transition(
            &out,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_COMMON,
        )]);
        list.close();
        for round in 1u32..=2 {
            let start = [round, round + 1, round + 2, round + 3];
            let tail = [round + 10, round + 11, round + 12, round + 13];
            std::ptr::copy_nonoverlapping(start.as_ptr(), src.cast::<u32>(), 4);
            std::ptr::copy_nonoverlapping(tail.as_ptr(), src.add(240).cast::<u32>(), 4);
            host.execute(&[&list]);
            assert_eq!(
                std::slice::from_raw_parts(mapped.cast::<[u32; 4]>(), 4),
                &[start, start, changed, tail]
            );
            assert!(host.device.GetDeviceRemovedReason().is_ok());
        }
        read.Unmap(0, None);
        drop(list);
        heap.release(heap.id(), first, 3);
        assert_ne!(VirtualFree(title.cast(), 0, MEM_RELEASE), 0);
        eprintln!(
            "exact table CBV reuse={reuse}, fills={fills}, callback barrier, final vector and two replays verified"
        );
    }
}
