#![cfg(windows)]
#![allow(dead_code, clippy::all)]

use std::ffi::c_void;
use std::hint::black_box;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;
use std::time::Instant;
use windows::Win32::Foundation::{HANDLE, RECT, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST};
use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
use windows::Win32::Graphics::Direct3D::ID3DBlob;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, PCWSTR, s};
use windows_sys::Win32::System::Memory::{
    MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
};

struct Probe {
    device: ID3D12Device,
    adapter: IDXGIAdapter3,
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    submitted: u64,
    event: HANDLE,
    query_heap: ID3D12QueryHeap,
    query_readback: ID3D12Resource,
    freq: u64,
    messages: Option<ID3D12InfoQueue>,
}

impl Probe {
    fn new(with_debug: bool) -> Self {
        unsafe {
            if with_debug {
                let mut debug: Option<ID3D12Debug> = None;
                if D3D12GetDebugInterface(&mut debug).is_ok()
                    && let Some(d) = debug
                {
                    d.EnableDebugLayer();
                }
            }
            let factory: IDXGIFactory4 =
                CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).expect("dxgi factory");
            let mut chosen: Option<IDXGIAdapter1> = None;
            let mut idx = 0u32;
            while let Ok(ad) = factory.EnumAdapters1(idx) {
                let desc = ad.GetDesc1().expect("desc1");
                if (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32) == 0 {
                    let name = String::from_utf16_lossy(
                        &desc.Description[..desc
                            .Description
                            .iter()
                            .position(|&c| c == 0)
                            .unwrap_or(desc.Description.len())],
                    );
                    if chosen.is_none() || desc.VendorId == 0x8086 {
                        if chosen.is_none() || name.contains("Arc") {
                            chosen = Some(ad);
                        }
                    }
                }
                idx += 1;
            }
            let adapter1 = chosen.expect("hardware adapter");
            let adapter: IDXGIAdapter3 = adapter1.cast().expect("adapter3");
            let mut device: Option<ID3D12Device> = None;
            D3D12CreateDevice(&adapter1, D3D_FEATURE_LEVEL_12_0, &mut device).expect("device");
            let device = device.expect("device some");
            let messages = if with_debug {
                device.cast::<ID3D12InfoQueue>().ok()
            } else {
                None
            };
            let queue: ID3D12CommandQueue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .expect("queue");
            let freq = queue.GetTimestampFrequency().expect("timestamp freq");
            let allocator: ID3D12CommandAllocator = device
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .expect("allocator");
            let list: ID3D12GraphicsCommandList = device
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &allocator, None)
                .expect("list");
            let fence: ID3D12Fence = device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .expect("fence");
            let event = CreateEventW(None, false, false, None).expect("event");
            let mut query_heap: Option<ID3D12QueryHeap> = None;
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 64,
                        NodeMask: 0,
                    },
                    &mut query_heap,
                )
                .expect("query heap");
            let query_heap = query_heap.expect("query heap some");
            let query_readback = make_buffer(
                &device,
                D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_READBACK,
                    ..Default::default()
                },
                512,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            Self {
                device,
                adapter,
                queue,
                allocator,
                list,
                fence,
                submitted: 0,
                event,
                query_heap,
                query_readback,
                freq,
                messages,
            }
        }
    }

    fn drain_messages(&self, tag: &str) {
        let Some(q) = &self.messages else { return };
        unsafe {
            let n = q.GetNumStoredMessages();
            for i in 0..n {
                let mut sz = 0usize;
                if q.GetMessage(i, None, &mut sz).is_ok() && sz > 0 {
                    let mut buf = vec![0u64; sz.div_ceil(8)];
                    let msg = buf.as_mut_ptr().cast::<D3D12_MESSAGE>();
                    if q.GetMessage(i, Some(msg), &mut sz).is_ok() {
                        let m = &*msg;
                        let txt = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                        let s = String::from_utf8_lossy(txt).trim_end_matches('\0').to_owned();
                        println!("  [D3D12 {tag} sev={} id={}] {s}", m.Severity.0, m.ID.0);
                    }
                }
            }
            q.ClearStoredMessages();
        }
    }

    fn run(&mut self, f: impl FnOnce(&ID3D12GraphicsCommandList)) {
        unsafe {
            f(&self.list);
            self.list.Close().expect("close");
            let cl: ID3D12CommandList = self.list.cast().unwrap();
            self.queue.ExecuteCommandLists(&[Some(cl)]);
            self.submitted += 1;
            self.queue.Signal(&self.fence, self.submitted).expect("signal");
            self.fence
                .SetEventOnCompletion(self.submitted, self.event)
                .expect("event");
            let w = WaitForSingleObject(self.event, 15_000);
            assert_eq!(w, WAIT_OBJECT_0, "gpu wait timeout");
            self.allocator.Reset().expect("reset alloc");
            self.list.Reset(&self.allocator, None).expect("reset list");
        }
    }

    fn run_timed_us(&mut self, f: impl FnOnce(&ID3D12GraphicsCommandList)) -> f64 {
        unsafe {
            let qh = self.query_heap.clone();
            let qr = self.query_readback.clone();
            self.run(|list| {
                list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                f(list);
                list.EndQuery(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                list.ResolveQueryData(&qh, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &qr, 0);
            });
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let range = D3D12_RANGE { Begin: 0, End: 16 };
            self.query_readback
                .Map(0, Some(&range), Some(&mut ptr))
                .expect("map ts");
            let ts = std::slice::from_raw_parts(ptr.cast::<u64>(), 2).to_vec();
            self.query_readback.Unmap(0, None);
            let ticks = ts[1].saturating_sub(ts[0]);
            (ticks as f64) * 1_000_000.0 / (self.freq as f64)
        }
    }
}

fn make_buffer(
    device: &ID3D12Device,
    heap: D3D12_HEAP_PROPERTIES,
    size: u64,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size.max(1),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: flags,
        ..Default::default()
    };
    let mut out: Option<ID3D12Resource> = None;
    unsafe {
        device
            .CreateCommittedResource(&heap, D3D12_HEAP_FLAG_NONE, &desc, state, None, &mut out)
            .expect("create buffer");
    }
    out.unwrap()
}

fn map_ptr(res: &ID3D12Resource) -> *mut u8 {
    let mut p: *mut c_void = std::ptr::null_mut();
    unsafe {
        res.Map(0, None, Some(&mut p)).expect("map");
    }
    p.cast()
}

fn transition(
    res: &ID3D12Resource,
    before: D3D12_RESOURCE_STATES,
    after: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                pResource: unsafe { std::mem::transmute_copy(res) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: before,
                StateAfter: after,
            }),
        },
    }
}

fn uav_barrier(res: Option<&ID3D12Resource>) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                pResource: match res {
                    Some(r) => unsafe { std::mem::transmute_copy(r) },
                    None => std::mem::ManuallyDrop::new(None),
                },
            }),
        },
    }
}

fn compile_fxc(src: &str, entry: &windows::core::PCSTR, target: &windows::core::PCSTR) -> Vec<u8> {
    let mut blob: Option<ID3DBlob> = None;
    let mut err: Option<ID3DBlob> = None;
    let res = unsafe {
        D3DCompile(
            src.as_ptr().cast(),
            src.len(),
            s!("probe.hlsl"),
            None,
            None,
            *entry,
            *target,
            0,
            0,
            &mut blob,
            Some(&mut err),
        )
    };
    if let Err(e) = res {
        let msg = err
            .map(|b| unsafe {
                let s = std::slice::from_raw_parts(b.GetBufferPointer().cast::<u8>(), b.GetBufferSize());
                String::from_utf8_lossy(s).into_owned()
            })
            .unwrap_or_default();
        panic!("D3DCompile failed: {e}: {msg}");
    }
    let blob = blob.unwrap();
    unsafe {
        std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize()).to_vec()
    }
}

fn compile_dxc(src: &str, entry: &str, target: &str) -> Result<Vec<u8>, String> {
    let tmp = std::env::temp_dir().join(format!("arc_probe_{}_{}.hlsl", entry, target));
    let out = std::env::temp_dir().join(format!("arc_probe_{}_{}.dxil", entry, target));
    std::fs::write(&tmp, src).map_err(|e| e.to_string())?;
    let dxc_paths = [
        PathBuf::from(r"C:\VulkanSDK\1.4.363.0\Bin\dxc.exe"),
        PathBuf::from("dxc.exe"),
    ];
    let dxc = dxc_paths
        .iter()
        .find(|p| p.exists() || p.as_os_str() == "dxc.exe")
        .unwrap();
    let res = Command::new(dxc)
        .args([
            "-T",
            target,
            "-E",
            entry,
            "-O3",
            "-Fo",
            out.to_str().unwrap(),
            tmp.to_str().unwrap(),
        ])
        .output()
        .map_err(|e| format!("spawn dxc: {e}"))?;
    let _ = std::fs::remove_file(&tmp);
    if !res.status.success() {
        let _ = std::fs::remove_file(&out);
        return Err(String::from_utf8_lossy(&res.stderr).into_owned());
    }
    let bytes = std::fs::read(&out).map_err(|e| e.to_string())?;
    let _ = std::fs::remove_file(&out);
    Ok(bytes)
}

fn run_tile_experiment(p: &mut Probe, format: DXGI_FORMAT, map_packed: bool, explicit_null_lvl0: bool, via_default_src: bool) {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 1024,
        Height: 1024,
        DepthOrArraySize: 1,
        MipLevels: 11,
        Format: format,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
        ..Default::default()
    };
    let mut reserved: Option<ID3D12Resource> = None;
    unsafe {
        p.device
            .CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut reserved)
            .expect("reserved");
    }
    let tex = reserved.unwrap();

    let mut total_tiles = 0u32;
    let mut packed = D3D12_PACKED_MIP_INFO::default();
    let mut shape = D3D12_TILE_SHAPE::default();
    let mut count = 11u32;
    let mut tilings = [D3D12_SUBRESOURCE_TILING::default(); 11];
    unsafe {
        p.device.GetResourceTiling(
            &tex,
            Some(&mut total_tiles),
            Some(&mut packed),
            Some(&mut shape),
            Some(&mut count),
            0,
            tilings.as_mut_ptr(),
        );
    }

    let _heap_keep: Option<ID3D12Heap> = if map_packed {
        let heap_desc = D3D12_HEAP_DESC {
            SizeInBytes: u64::from(packed.NumTilesForPackedMips.max(1)) * 65536,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                ..Default::default()
            },
            Alignment: 65536,
            Flags: D3D12_HEAP_FLAG_DENY_BUFFERS | D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES,
        };
        let mut heap: Option<ID3D12Heap> = None;
        unsafe { p.device.CreateHeap(&heap_desc, &mut heap).expect("tile heap") };
        let heap = heap.unwrap();
        let at = D3D12_TILED_RESOURCE_COORDINATE {
            X: 0,
            Y: 0,
            Z: 0,
            Subresource: u32::from(packed.NumStandardMips),
        };
        let sz = D3D12_TILE_REGION_SIZE {
            NumTiles: packed.NumTilesForPackedMips,
            ..Default::default()
        };
        let flag = D3D12_TILE_RANGE_FLAG_NONE;
        let start = 0u32;
        let cnt = packed.NumTilesForPackedMips;
        unsafe {
            p.queue.UpdateTileMappings(
                &tex,
                1,
                Some(&at),
                Some(&sz),
                Some(&heap),
                1,
                Some(&flag),
                Some(&start),
                Some(&cnt),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
        Some(heap)
    } else {
        None
    };

    if explicit_null_lvl0 {
        let at = D3D12_TILED_RESOURCE_COORDINATE { X: 0, Y: 0, Z: 0, Subresource: 0 };
        let sz = D3D12_TILE_REGION_SIZE {
            NumTiles: tilings[0].WidthInTiles * u32::from(tilings[0].HeightInTiles),
            ..Default::default()
        };
        let flag = D3D12_TILE_RANGE_FLAG_NULL;
        let cnt = sz.NumTiles;
        unsafe {
            p.queue.UpdateTileMappings(
                &tex,
                1,
                Some(&at),
                Some(&sz),
                None,
                1,
                Some(&flag),
                None,
                Some(&cnt),
                D3D12_TILE_MAPPING_FLAG_NONE,
            );
        }
    }

    let mut footprints = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 11];
    let mut rows = [0u32; 11];
    let mut wide = [0u64; 11];
    let mut total = 0u64;
    unsafe {
        p.device.GetCopyableFootprints(
            &desc,
            0,
            11,
            0,
            Some(footprints.as_mut_ptr()),
            Some(rows.as_mut_ptr()),
            Some(wide.as_mut_ptr()),
            Some(&mut total),
        );
    }
    let upload = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
        total,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let up_ptr = map_ptr(&upload);
    unsafe {
        for lvl in 0..11 {
            let len = rows[lvl] as usize * footprints[lvl].Footprint.RowPitch as usize;
            std::slice::from_raw_parts_mut(up_ptr.add(footprints[lvl].Offset as usize), len).fill((lvl + 1) as u8);
        }
        upload.Unmap(0, None);
    }
    let rb = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        total,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );

    let default_src = via_default_src.then(|| {
        let sdesc = D3D12_RESOURCE_DESC {
            Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
            ..desc
        };
        let mut r: Option<ID3D12Resource> = None;
        unsafe {
            p.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &sdesc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut r,
                )
                .unwrap();
        }
        r.unwrap()
    });

    let sub = |res: &ID3D12Resource, idx: u32| D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(res) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: idx },
    };
    let placed = |buf: &ID3D12Resource, fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
        pResource: unsafe { std::mem::transmute_copy(buf) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
    };

    p.run(|l| unsafe {
        if let Some(ref dsrc) = default_src {
            for lvl in 0..11 {
                l.CopyTextureRegion(&sub(dsrc, lvl as u32), 0, 0, 0, &placed(&upload, footprints[lvl]), None);
            }
            l.ResourceBarrier(&[transition(dsrc, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            for lvl in 0..11 {
                l.CopyTextureRegion(&sub(&tex, lvl as u32), 0, 0, 0, &sub(dsrc, lvl as u32), None);
            }
        } else {
            for lvl in 0..11 {
                l.CopyTextureRegion(&sub(&tex, lvl as u32), 0, 0, 0, &placed(&upload, footprints[lvl]), None);
            }
        }
        l.ResourceBarrier(&[transition(&tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        for lvl in 0..11 {
            l.CopyTextureRegion(&placed(&rb, footprints[lvl]), 0, 0, 0, &sub(&tex, lvl as u32), None);
        }
        l.ResourceBarrier(&[transition(&tex, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_COPY_DEST)]);
    });
    p.drain_messages("tile_exp");

    let ptr = map_ptr(&rb);
    print!("  fmt={} map_packed={map_packed} exp_null0={explicit_null_lvl0} def_src={via_default_src}: ", format.0);
    unsafe {
        for lvl in 0..5 {
            let b0 = *ptr.add(footprints[lvl].Offset as usize);
            let mid = (rows[lvl] as usize / 2) * footprints[lvl].Footprint.RowPitch as usize + ((wide[lvl] as usize / 2) & !15);
            let bmid = *ptr.add(footprints[lvl].Offset as usize + mid);
            print!("m{lvl}=({b0},{bmid}) ");
        }
        println!();
        rb.Unmap(0, None);
    }

    // Also test shader SampleLevel/Load on unmapped mip 0 & 1 and mapped mip 3!
    let hlsl = r#"
Texture2D<float4> tex : register(t0);
SamplerState samp : register(s0);
RWByteAddressBuffer out_buf : register(u0);
[RootSignature("DescriptorTable(SRV(t0)), StaticSampler(s0, filter=FILTER_MIN_MAG_MIP_POINT), UAV(u0)")]
[numthreads(1,1,1)]
void main() {
    uint st0, st1, st2, st3;
    float4 l0 = tex.Load(int3(0, 0, 0), int2(0, 0), st0);
    float4 s0 = tex.SampleLevel(samp, float2(0.5, 0.5), 0.0, int2(0, 0), st1);
    float4 l1 = tex.Load(int3(0, 0, 1), int2(0, 0), st2);
    float4 l3 = tex.Load(int3(0, 0, 3), int2(0, 0), st3);
    out_buf.Store4(0, uint4(
        (uint)(l0.r * 255.0 + 0.5), CheckAccessFullyMapped(st0) ? 1u : 0u,
        (uint)(s0.r * 255.0 + 0.5), CheckAccessFullyMapped(st1) ? 1u : 0u
    ));
    out_buf.Store4(16, uint4(
        (uint)(l1.r * 255.0 + 0.5), CheckAccessFullyMapped(st2) ? 1u : 0u,
        (uint)(l3.r * 255.0 + 0.5), CheckAccessFullyMapped(st3) ? 1u : 0u
    ));
}
"#;
    let cs = compile_fxc(hlsl, &s!("main"), &s!("cs_5_1"));
    unsafe {
        let root: ID3D12RootSignature = p.device.CreateRootSignature(0, &cs).unwrap();
        let pso: ID3D12PipelineState = p.device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr().cast(), BytecodeLength: cs.len() },
                ..Default::default()
            })
            .unwrap();
        let heap: ID3D12DescriptorHeap = p.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })
            .unwrap();
        p.device.CreateShaderResourceView(&tex, None, heap.GetCPUDescriptorHandleForHeapStart());
        let out_uav = make_buffer(
            &p.device,
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
            256,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let out_rb = make_buffer(
            &p.device,
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
            256,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        p.run(|l| {
            l.ResourceBarrier(&[transition(&tex, D3D12_RESOURCE_STATE_COPY_DEST, D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)]);
            l.SetDescriptorHeaps(&[Some(heap.clone())]);
            l.SetComputeRootSignature(&root);
            l.SetPipelineState(&pso);
            l.SetComputeRootDescriptorTable(0, heap.GetGPUDescriptorHandleForHeapStart());
            l.SetComputeRootUnorderedAccessView(1, out_uav.GetGPUVirtualAddress());
            l.Dispatch(1, 1, 1);
            l.ResourceBarrier(&[transition(&out_uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            l.CopyBufferRegion(&out_rb, 0, &out_uav, 0, 32);
        });
        let ptr = map_ptr(&out_rb).cast::<u32>();
        let r = std::slice::from_raw_parts(ptr, 8).to_vec();
        out_rb.Unmap(0, None);
        println!(
            "    shader: load0={} (map={}) samp0={} (map={}) load1={} (map={}) load3={} (map={})",
            r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]
        );
    }
}

#[test]
#[ignore]
fn probe_2_tiles_unmapped_behavior() {
    for dbg in [true, false] {
        println!("\n=== probe_2_tiles_unmapped_behavior (debug_layer={dbg}) ===");
        let mut p = Probe::new(dbg);
        // 1. RGBA8 without vs with packed mapped
        run_tile_experiment(&mut p, DXGI_FORMAT_R8G8B8A8_UNORM, false, false, false);
        run_tile_experiment(&mut p, DXGI_FORMAT_R8G8B8A8_UNORM, true, false, false);
        // 2. BC7 (exact levels_on_demand format 98!) without vs with packed mapped vs explicit NULL vs DEFAULT src
        run_tile_experiment(&mut p, DXGI_FORMAT_BC7_UNORM, false, false, false);
        run_tile_experiment(&mut p, DXGI_FORMAT_BC7_UNORM, true, false, false);
        run_tile_experiment(&mut p, DXGI_FORMAT_BC7_UNORM, true, true, false);
        run_tile_experiment(&mut p, DXGI_FORMAT_BC7_UNORM, true, false, true);
        // 3. BC3 (format 77) with packed mapped
        run_tile_experiment(&mut p, DXGI_FORMAT_BC3_UNORM, true, false, false);
    }
}

#[test]
#[ignore]
fn probe_3_memory_heap_bandwidth() {
    println!("\n=== probe_3_memory_heap_bandwidth (64 MiB) ===");
    let mut p = Probe::new(false);
    const SZ: usize = 64 << 20;
    let default_a = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        SZ as u64,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let default_b = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        SZ as u64,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let upload = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
        SZ as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let readback = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        SZ as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );
    let shared_wb = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        },
        SZ as u64,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );

    let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
    let has_gpu_upload = unsafe {
        p.device
            .CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, (&raw mut o16).cast(), size_of_val(&o16) as u32)
            .is_ok()
            && o16.GPUUploadHeapSupported.as_bool()
    };
    let gpu_upload = has_gpu_upload.then(|| {
        make_buffer(
            &p.device,
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() },
            SZ as u64,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_COMMON,
        )
    });

    let valloc_ptr = unsafe { VirtualAlloc(std::ptr::null(), SZ, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE) } as *mut u8;
    let inplace_buf = if !valloc_ptr.is_null() {
        unsafe { std::ptr::write_bytes(valloc_ptr, 0x3a, SZ) };
        let opened = unsafe {
            p.device.cast::<ID3D12Device3>().map(|d3| d3.OpenExistingHeapFromAddress::<ID3D12Heap>(valloc_ptr.cast()))
        };
        println!("  OpenExistingHeapFromAddress result: {:?}", opened.as_ref().map(|r| r.as_ref().map(|_| "Ok").map_err(|e| format!("{e}"))));
        if let Ok(Ok(h)) = opened {
            let desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                Width: SZ as u64,
                Height: 1,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                ..Default::default()
            };
            let mut r: Option<ID3D12Resource> = None;
            let ok = unsafe {
                p.device.CreatePlacedResource(&h, 0, &desc, D3D12_RESOURCE_STATE_COMMON, None, &mut r)
            };
            println!("  CreatePlacedResource(inplace, UAV) result: {:?}", ok);
            if ok.is_ok() { r.map(|res| (h, res)) } else { None }
        } else {
            None
        }
    } else {
        None
    };

    let up_ptr = map_ptr(&upload);
    let rb_ptr = map_ptr(&readback);
    let swb_ptr = map_ptr(&shared_wb);
    let gup_ptr = gpu_upload.as_ref().map(map_ptr);
    let ram = vec![0x55u8; SZ];
    let gbps = |ms: f64, bytes: usize| (bytes as f64 / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);

    // Prevent dead-store/load elimination using explicit black_box checksum
    let sum64 = |ptr: *const u8, len: usize| -> u64 {
        let words = unsafe { std::slice::from_raw_parts(ptr.cast::<u64>(), len / 8) };
        let mut s = 0u64;
        for &w in words {
            s = s.wrapping_add(w);
        }
        black_box(s)
    };

    for (name, ptr) in [
        ("RAM vec", Some(ram.as_ptr() as *mut u8)),
        ("UPLOAD (WC)", Some(up_ptr)),
        ("SHARED_WB_L0", Some(swb_ptr)),
        ("GPU_UPLOAD", gup_ptr),
        ("INPLACE_VALLOC", inplace_buf.as_ref().map(|_| valloc_ptr)),
    ] {
        let Some(ptr) = ptr else { continue };
        if name != "RAM vec" {
            unsafe { std::ptr::copy_nonoverlapping(ram.as_ptr(), ptr, SZ) };
        }
        let t = Instant::now();
        for i in 0..5usize {
            let src = if name == "RAM vec" { black_box(ptr) } else { ram.as_ptr() };
            let dst = if name == "RAM vec" { ram.as_ptr() as *mut u8 } else { ptr };
            if name != "RAM vec" {
                unsafe { std::ptr::copy_nonoverlapping(src, dst, SZ) };
                black_box(unsafe { dst.add((i * 4096) % SZ).read_volatile() });
            }
        }
        if name == "RAM vec" {
            let mut ram2 = vec![0u8; SZ];
            let t2 = Instant::now();
            for _ in 0..5 {
                unsafe { std::ptr::copy_nonoverlapping(ram.as_ptr(), ram2.as_mut_ptr(), SZ) };
                black_box(ram2[0]);
            }
            let ms = t2.elapsed().as_secs_f64() * 1000.0 / 5.0;
            println!("  CPU write 64 MiB -> {name:16}: {ms:7.3} ms ({:6.2} GB/s)", gbps(ms, SZ));
        } else {
            let ms = t.elapsed().as_secs_f64() * 1000.0 / 5.0;
            println!("  CPU write 64 MiB -> {name:16}: {ms:7.3} ms ({:6.2} GB/s)", gbps(ms, SZ));
        }
    }

    // Seed GPU->CPU buffers first so reads aren't zero-page shortcuts!
    p.run(|l| unsafe {
        l.CopyBufferRegion(&readback, 0, &default_a, 0, SZ as u64);
        l.CopyBufferRegion(&shared_wb, 0, &default_a, 0, SZ as u64);
    });
    for (name, ptr) in [
        ("RAM vec", Some(ram.as_ptr())),
        ("READBACK", Some(rb_ptr as *const u8)),
        ("SHARED_WB_L0", Some(swb_ptr as *const u8)),
        ("UPLOAD (WC)", Some(up_ptr as *const u8)),
        ("GPU_UPLOAD", gup_ptr.map(|x| x as *const u8)),
        ("INPLACE_VALLOC", inplace_buf.as_ref().map(|_| valloc_ptr as *const u8)),
    ] {
        let Some(ptr) = ptr else { continue };
        black_box(sum64(ptr, SZ));
        let t = Instant::now();
        for _ in 0..5 {
            black_box(sum64(ptr, SZ));
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / 5.0;
        println!("  CPU read  64 MiB <- {name:16}: {ms:7.3} ms ({:6.2} GB/s)", gbps(ms, SZ));
    }

    // GPU CopyBufferRegion 64 MiB
    let copy_cases: Vec<(&str, &ID3D12Resource, &ID3D12Resource)> = {
        let mut v = vec![
            ("UPLOAD -> DEFAULT", &upload, &default_a),
            ("SHARED_WB_L0 -> DEFAULT", &shared_wb, &default_a),
            ("DEFAULT -> READBACK", &default_a, &readback),
            ("DEFAULT -> SHARED_WB_L0", &default_a, &shared_wb),
            ("DEFAULT -> DEFAULT", &default_a, &default_b),
        ];
        if let Some((_, ref r)) = inplace_buf {
            v.push(("INPLACE -> DEFAULT", r, &default_a));
            v.push(("DEFAULT -> INPLACE", &default_a, r));
        }
        if let Some(ref g) = gpu_upload {
            v.push(("GPU_UPLOAD -> DEFAULT", g, &default_a));
        }
        v
    };
    for (label, src, dst) in copy_cases {
        p.run_timed_us(|l| unsafe { l.CopyBufferRegion(dst, 0, src, 0, SZ as u64) });
        let mut sum_us = 0.0;
        for _ in 0..5 {
            sum_us += p.run_timed_us(|l| unsafe { l.CopyBufferRegion(dst, 0, src, 0, SZ as u64) });
        }
        let ms = (sum_us / 5.0) / 1000.0;
        println!("  GPU CopyBufferRegion 64 MiB {label:25}: {ms:7.3} ms ({:6.2} GB/s)", gbps(ms, SZ));
    }

    // End-to-end: GPU writes 64 MiB -> CopyBufferRegion to READBACK vs SHARED_WB_L0 -> CPU sum64!
    for (label, dst, ptr) in [
        ("READBACK", &readback, rb_ptr as *const u8),
        ("SHARED_WB_L0", &shared_wb, swb_ptr as *const u8),
    ] {
        p.run(|l| unsafe { l.CopyBufferRegion(dst, 0, &default_a, 0, SZ as u64) });
        black_box(sum64(ptr, SZ));
        let t = Instant::now();
        for _ in 0..5 {
            p.run(|l| unsafe { l.CopyBufferRegion(dst, 0, &default_a, 0, SZ as u64) });
            black_box(sum64(ptr, SZ));
        }
        let ms = t.elapsed().as_secs_f64() * 1000.0 / 5.0;
        println!("  End-to-end GPU copy + CPU read 64 MiB ({label:12}): {ms:7.3} ms ({:6.2} GB/s)", gbps(ms, SZ));
    }

    // Scattered 128 x 64 KiB fetch (8 MiB), simulating relay_card_back
    const PIECES: usize = 128;
    const PIECE_SZ: u64 = 65536;
    for (label, dst) in [("READBACK", &readback), ("SHARED_WB_L0", &shared_wb)] {
        p.run_timed_us(|l| unsafe {
            for i in 0..PIECES {
                l.CopyBufferRegion(dst, i as u64 * PIECE_SZ, &default_a, (i as u64 * 7 % 512) * PIECE_SZ, PIECE_SZ);
            }
        });
        let mut sum_us = 0.0;
        for _ in 0..10 {
            sum_us += p.run_timed_us(|l| unsafe {
                for i in 0..PIECES {
                    l.CopyBufferRegion(dst, i as u64 * PIECE_SZ, &default_a, (i as u64 * 7 % 512) * PIECE_SZ, PIECE_SZ);
                }
            });
        }
        let ms = (sum_us / 10.0) / 1000.0;
        println!("  GPU 128x64KiB scattered copy DEFAULT -> {label:12}: {ms:7.3} ms ({:6.2} GB/s)", gbps(ms, PIECES * PIECE_SZ as usize));
    }

    let cs_src = r#"
        ByteAddressBuffer src : register(t0);
        RWByteAddressBuffer dst : register(u0);
        [RootSignature("SRV(t0), UAV(u0)")]
        [numthreads(64, 1, 1)]
        void main(uint3 id : SV_DispatchThreadID) {
            uint base = id.x * 64;
            [unroll]
            for (uint k = 0; k < 4; ++k) {
                uint4 v = src.Load4(base + k * 16);
                dst.Store4(base + k * 16, v ^ uint4(1, 2, 3, 4));
            }
        }
    "#;
    let cs = compile_fxc(cs_src, &s!("main"), &s!("cs_5_1"));
    let root: ID3D12RootSignature = unsafe { p.device.CreateRootSignature(0, &cs).unwrap() };
    let pso: ID3D12PipelineState = unsafe {
        p.device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr().cast(), BytecodeLength: cs.len() },
                ..Default::default()
            })
            .unwrap()
    };
    let groups = ((SZ / 64) / 64) as u32;
    let mut cs_cases: Vec<(&str, &ID3D12Resource, &ID3D12Resource)> = vec![
        ("DEFAULT -> DEFAULT", &default_a, &default_b),
        ("SHARED_WB_L0 -> DEFAULT", &shared_wb, &default_b),
        ("DEFAULT -> SHARED_WB_L0", &default_a, &shared_wb),
        ("SHARED_WB_L0 -> SHARED_WB_L0", &shared_wb, &shared_wb),
        ("UPLOAD -> DEFAULT", &upload, &default_b),
    ];
    if let Some((_, ref r)) = inplace_buf {
        cs_cases.push(("INPLACE -> DEFAULT", r, &default_b));
        cs_cases.push(("DEFAULT -> INPLACE", &default_a, r));
    }
    if let Some(ref g) = gpu_upload {
        cs_cases.push(("GPU_UPLOAD -> DEFAULT", g, &default_b));
    }
    for (label, src, dst) in cs_cases {
        let same = std::ptr::eq(src as *const _, dst as *const _);
        let run_one = |p: &mut Probe| {
            p.run_timed_us(|l| unsafe {
                l.SetComputeRootSignature(&root);
                l.SetPipelineState(&pso);
                l.SetComputeRootShaderResourceView(0, src.GetGPUVirtualAddress());
                l.SetComputeRootUnorderedAccessView(1, dst.GetGPUVirtualAddress());
                l.Dispatch(groups, 1, 1);
                if same {
                    l.ResourceBarrier(&[uav_barrier(Some(dst))]);
                }
            })
        };
        run_one(&mut p);
        let mut sum_us = 0.0;
        for _ in 0..5 {
            sum_us += run_one(&mut p);
        }
        let ms = (sum_us / 5.0) / 1000.0;
        println!("  CS 64 MiB R+W {label:30}: {ms:7.3} ms ({:6.2} GB/s payload)", gbps(ms, SZ));
    }

    drop(inplace_buf);
    if !valloc_ptr.is_null() {
        unsafe { VirtualFree(valloc_ptr.cast(), 0, MEM_RELEASE) };
    }
}

#[test]
#[ignore]
fn probe_4_wavesize_and_psos() {
    println!("\n=== probe_4_wavesize_and_psos ===");
    let mut p = Probe::new(false);

    let make_cs_hlsl = |ws_attr: &str, lds_mode: bool| {
        if lds_mode {
            format!(
                r#"
cbuffer C : register(b0) {{ uint iters; uint pad0; uint pad1; uint pad2; }};
RWByteAddressBuffer out_buf : register(u0);
groupshared uint lds[64];
[RootSignature("RootConstants(num32BitConstants=4, b0), UAV(u0)")]
{ws_attr}
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex, uint3 dtid : SV_DispatchThreadID) {{
    uint lanes = WaveGetLaneCount();
    uint acc = dtid.x + 1;
    uint x = asuint(1.0001f + float(dtid.x & 15) * 0.001f);
    for (uint i = 0; i < iters; ++i) {{
        float f = asfloat(x);
        f = mad(f, 1.00001f, 0.00003f);
        x = asuint(f);
        acc += WaveActiveSum(x & 0xff);
        lds[gidx] = acc;
        GroupMemoryBarrierWithGroupSync();
        acc ^= lds[(gidx + 17) & 63];
        GroupMemoryBarrierWithGroupSync();
    }}
    if (dtid.x == 0) out_buf.Store(0, lanes);
    out_buf.Store(16 + (dtid.x & 1023) * 4, acc);
}}
"#
            )
        } else {
            format!(
                r#"
cbuffer C : register(b0) {{ uint iters; uint pad0; uint pad1; uint pad2; }};
RWByteAddressBuffer out_buf : register(u0);
[RootSignature("RootConstants(num32BitConstants=4, b0), UAV(u0)")]
{ws_attr}
[numthreads(64, 1, 1)]
void main(uint3 gid : SV_GroupID, uint gidx : SV_GroupIndex, uint3 dtid : SV_DispatchThreadID) {{
    uint lanes = WaveGetLaneCount();
    float a = 1.0001f + float(dtid.x & 15) * 0.001f;
    float b = 0.9999f;
    uint w = dtid.x;
    for (uint i = 0; i < iters; ++i) {{
        a = mad(a, 1.00001f, 0.00003f);
        b = mad(b, 0.99999f, a * 0.0001f);
        w += WaveActiveSum((uint)(a * 100.0f) & 0xff) ^ WaveReadLaneAt(w, (gidx + 3) & (lanes - 1));
    }}
    if (dtid.x == 0) out_buf.Store(0, lanes);
    out_buf.Store(16 + (dtid.x & 1023) * 4, w + asuint(b));
}}
"#
            )
        }
    };

    let uav = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        65536,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    let rb = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        256,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );

    for (mode_name, lds_mode) in [("LDS+Wave (4096g x 64t x 128i)", true), ("ALU+WaveShuffle (4096g x 64t x 128i)", false)] {
        println!("--- Compute Shader {mode_name} ---");
        for (label, target, attr) in [
            ("cs_6_0 omitted", "cs_6_0", ""),
            ("cs_6_6 omitted", "cs_6_6", ""),
            ("cs_6_6 WaveSize(8)", "cs_6_6", "[WaveSize(8)]"),
            ("cs_6_6 WaveSize(16)", "cs_6_6", "[WaveSize(16)]"),
            ("cs_6_6 WaveSize(32)", "cs_6_6", "[WaveSize(32)]"),
        ] {
            let hlsl = make_cs_hlsl(attr, lds_mode);
            let dxil = match compile_dxc(&hlsl, "main", target) {
                Ok(b) => b,
                Err(e) => {
                    println!("  {label:22}: DXC error: {e}");
                    continue;
                }
            };
            let Ok(root) = (unsafe { p.device.CreateRootSignature::<ID3D12RootSignature>(0, &dxil) }) else {
                println!("  {label:22}: CreateRootSignature failed");
                continue;
            };
            let pso: Result<ID3D12PipelineState, _> = unsafe {
                p.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
                    ..Default::default()
                })
            };
            let pso = match pso {
                Ok(p) => p,
                Err(e) => {
                    println!("  {label:22}: CreateComputePipelineState FAILED: {e}");
                    continue;
                }
            };
            let run_cs = |p: &mut Probe, iters: u32| {
                p.run_timed_us(|l| unsafe {
                    l.SetComputeRootSignature(&root);
                    l.SetPipelineState(&pso);
                    let c = [iters, 0, 0, 0];
                    l.SetComputeRoot32BitConstants(0, 4, c.as_ptr().cast(), 0);
                    l.SetComputeRootUnorderedAccessView(1, uav.GetGPUVirtualAddress());
                    l.Dispatch(4096, 1, 1);
                })
            };
            run_cs(&mut p, 128);
            p.run(|l| unsafe {
                l.ResourceBarrier(&[transition(&uav, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
                l.CopyBufferRegion(&rb, 0, &uav, 0, 16);
                l.ResourceBarrier(&[transition(&uav, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)]);
            });
            let lanes = unsafe {
                let ptr = map_ptr(&rb).cast::<u32>();
                let v = *ptr;
                rb.Unmap(0, None);
                v
            };
            let mut sum_us = 0.0;
            for _ in 0..8 {
                sum_us += run_cs(&mut p, 128);
            }
            println!(
                "  {label:22}: WaveGetLaneCount={lanes:2}, time={:7.1} us (dxil={} B)",
                sum_us / 8.0,
                dxil.len()
            );
        }
    }

    println!("--- Pixel Shader WaveSize Probe (2560x1440 fullscreen, 64 ALU/wave ops) ---");
    let vs_hlsl = r#"
        float4 main(uint vid : SV_VertexID) : SV_Position {
            float2 p = float2((vid << 1) & 2, vid & 2);
            return float4(p * 2.0 - 1.0, 0.0, 1.0);
        }
    "#;
    let vs_dxil = compile_dxc(vs_hlsl, "main", "vs_6_0").expect("vs_6_0");
    let rt_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 2560,
        Height: 1440,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut rt: Option<ID3D12Resource> = None;
    unsafe {
        p.device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &rt_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut rt,
            )
            .unwrap();
    }
    let rt = rt.unwrap();
    let rtv_heap: ID3D12DescriptorHeap = unsafe {
        p.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .unwrap()
    };
    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    unsafe { p.device.CreateRenderTargetView(&rt, None, rtv) };
    let tex_readback = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() },
        2560 * 16,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );

    let make_ps_hlsl = || {
        r#"
cbuffer C : register(b0) { uint iters; uint p0; uint p1; uint p2; };
[RootSignature("RootConstants(num32BitConstants=4, b0)")]
float4 main(float4 pos : SV_Position) : SV_Target {
    float lanes = (float)WaveGetLaneCount();
    float v = pos.x * 0.001f + pos.y * 0.002f;
    float sum = 0.0f;
    for (uint i = 0; i < iters; ++i) {
        v = mad(v, 1.0001f, 0.00017f);
        sum += WaveActiveSum(ddx_fine(v) + ddy_fine(v));
    }
    return float4(lanes, sum, v, 1.0f);
}
"#
    };

    for (label, target) in [
        ("ps_6_0 omitted", "ps_6_0"),
        ("ps_6_6 omitted", "ps_6_6"),
    ] {
        let hlsl = make_ps_hlsl();
        let dxil = compile_dxc(hlsl, "main", target).expect("ps");
        let root: ID3D12RootSignature = unsafe { p.device.CreateRootSignature(0, &dxil).unwrap() };
        let pso: ID3D12PipelineState = unsafe {
            p.device
                .CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&root),
                    VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs_dxil.as_ptr().cast(), BytecodeLength: vs_dxil.len() },
                    PS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr().cast(), BytecodeLength: dxil.len() },
                    BlendState: D3D12_BLEND_DESC {
                        RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: 0xf, ..Default::default() }; 8],
                        ..Default::default()
                    },
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: [
                        DXGI_FORMAT_R32G32B32A32_FLOAT,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                    ],
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                })
                .unwrap()
        };
        let run_ps = |p: &mut Probe, iters: u32| {
            p.run_timed_us(|l| unsafe {
                l.SetGraphicsRootSignature(&root);
                l.SetPipelineState(&pso);
                let c = [iters, 0, 0, 0];
                l.SetGraphicsRoot32BitConstants(0, 4, c.as_ptr().cast(), 0);
                l.OMSetRenderTargets(1, Some(&rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0,
                    TopLeftY: 0.0,
                    Width: 2560.0,
                    Height: 1440.0,
                    MinDepth: 0.0,
                    MaxDepth: 1.0,
                }]);
                l.RSSetScissorRects(&[RECT { left: 0, top: 0, right: 2560, bottom: 1440 }]);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                l.DrawInstanced(3, 1, 0, 0);
            })
        };
        run_ps(&mut p, 64);
        p.run(|l| unsafe {
            l.ResourceBarrier(&[transition(&rt, D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&tex_readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
                            Width: 1,
                            Height: 1,
                            Depth: 1,
                            RowPitch: 256,
                        },
                    },
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&rt),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
            };
            let bx = D3D12_BOX { left: 0, top: 0, front: 0, right: 1, bottom: 1, back: 1 };
            l.CopyTextureRegion(&dst, 0, 0, 0, &src, Some(&bx));
            l.ResourceBarrier(&[transition(&rt, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_RENDER_TARGET)]);
        });
        let lanes = unsafe {
            let ptr = map_ptr(&tex_readback).cast::<f32>();
            let v = *ptr;
            tex_readback.Unmap(0, None);
            v
        };
        let mut sum_us = 0.0;
        for _ in 0..8 {
            sum_us += run_ps(&mut p, 64);
        }
        println!("  {label:22}: WaveGetLaneCount={lanes:2.0}, time={:7.1} us (dxil={} B)", sum_us / 8.0, dxil.len());
    }

    println!("--- PSO Compile & PipelineLibrary1 Threading Probe (16 complex PS+VS PSOs) ---");
    let mut ps_variants = Vec::new();
    for k in 0..16u32 {
        let src = format!(
            r#"
cbuffer C : register(b0) {{ uint iters; uint p0; uint p1; uint p2; }};
[RootSignature("RootConstants(num32BitConstants=4, b0)")]
float4 main(float4 pos : SV_Position) : SV_Target {{
    float x = pos.x * 0.001f + {k}.0f;
    float y = pos.y * 0.002f + {k}.5f;
    [loop]
    for (uint i = 0; i < iters + 16; ++i) {{
        x = sin(x * 1.1f + y) + cos(y * 0.7f - x);
        y = mad(x, 0.99f, y * 0.5f + float({k}) * 0.01f);
    }}
    return float4(x, y, float({k}), 1.0f);
}}
"#
        );
        ps_variants.push(compile_dxc(&src, "main", "ps_6_0").expect("variant"));
    }
    let make_desc = |rt_sig: &ID3D12RootSignature, ps: &[u8]| D3D12_GRAPHICS_PIPELINE_STATE_DESC {
        pRootSignature: unsafe { std::mem::transmute_copy(rt_sig) },
        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs_dxil.as_ptr().cast(), BytecodeLength: vs_dxil.len() },
        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
        BlendState: D3D12_BLEND_DESC {
            RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: 0xf, ..Default::default() }; 8],
            ..Default::default()
        },
        SampleMask: u32::MAX,
        RasterizerState: D3D12_RASTERIZER_DESC {
            FillMode: D3D12_FILL_MODE_SOLID,
            CullMode: D3D12_CULL_MODE_NONE,
            ..Default::default()
        },
        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
        NumRenderTargets: 1,
        RTVFormats: [
            DXGI_FORMAT_R32G32B32A32_FLOAT,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
            DXGI_FORMAT_UNKNOWN,
        ],
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        ..Default::default()
    };

    let dev1 = Probe::new(false).device;
    let root1: ID3D12RootSignature = unsafe { dev1.CreateRootSignature(0, &ps_variants[0]).unwrap() };
    let t1 = Instant::now();
    let mut psos1: Vec<ID3D12PipelineState> = Vec::new();
    for ps in &ps_variants {
        let d = make_desc(&root1, ps);
        psos1.push(unsafe { dev1.CreateGraphicsPipelineState(&d).unwrap() });
    }
    let ms1 = t1.elapsed().as_secs_f64() * 1000.0;

    let dev8 = Arc::new(Probe::new(false).device);
    let root8: Arc<ID3D12RootSignature> = Arc::new(unsafe { dev8.CreateRootSignature(0, &ps_variants[0]).unwrap() });
    let variants = Arc::new(ps_variants);
    let t8 = Instant::now();
    let mut handles = Vec::new();
    for tid in 0..8usize {
        let dev = dev8.clone();
        let rt = root8.clone();
        let vars = variants.clone();
        let vs = vs_dxil.clone();
        handles.push(std::thread::spawn(move || {
            for k in (tid * 2)..(tid * 2 + 2) {
                let d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                    pRootSignature: unsafe { std::mem::transmute_copy(&*rt) },
                    VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
                    PS: D3D12_SHADER_BYTECODE { pShaderBytecode: vars[k].as_ptr().cast(), BytecodeLength: vars[k].len() },
                    BlendState: D3D12_BLEND_DESC {
                        RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: 0xf, ..Default::default() }; 8],
                        ..Default::default()
                    },
                    SampleMask: u32::MAX,
                    RasterizerState: D3D12_RASTERIZER_DESC {
                        FillMode: D3D12_FILL_MODE_SOLID,
                        CullMode: D3D12_CULL_MODE_NONE,
                        ..Default::default()
                    },
                    PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                    NumRenderTargets: 1,
                    RTVFormats: [
                        DXGI_FORMAT_R32G32B32A32_FLOAT,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                        DXGI_FORMAT_UNKNOWN,
                    ],
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    ..Default::default()
                };
                unsafe { dev.CreateGraphicsPipelineState::<ID3D12PipelineState>(&d).unwrap() };
            }
        }));
    }
    for h in handles {
        h.join().unwrap();
    }
    let ms8 = t8.elapsed().as_secs_f64() * 1000.0;
    println!("  CreateGraphicsPipelineState (16 PSOs): 1-thread={ms1:.2} ms, 8-thread={ms8:.2} ms");

    if let Ok(dev1_1) = dev1.cast::<ID3D12Device1>() {
        let lib: ID3D12PipelineLibrary1 = unsafe { dev1_1.CreatePipelineLibrary(&[]).unwrap() };
        let t_store = Instant::now();
        for (i, pso) in psos1.iter().enumerate() {
            let name: Vec<u16> = format!("pso_{i}\0").encode_utf16().collect();
            unsafe { lib.StorePipeline(PCWSTR(name.as_ptr()), pso).unwrap() };
        }
        let store_ms = t_store.elapsed().as_secs_f64() * 1000.0;
        let sz = unsafe { lib.GetSerializedSize() };
        let mut blob = vec![0u8; sz];
        let t_ser = Instant::now();
        unsafe { lib.Serialize(&mut blob).unwrap() };
        let ser_ms = t_ser.elapsed().as_secs_f64() * 1000.0;

        let dev_load = Arc::new(Probe::new(false).device);
        let root_load: Arc<ID3D12RootSignature> = Arc::new(unsafe { dev_load.CreateRootSignature(0, &variants[0]).unwrap() });
        let dev_load1: ID3D12Device1 = dev_load.cast().unwrap();
        let lib2: Arc<ID3D12PipelineLibrary1> = Arc::new(unsafe { dev_load1.CreatePipelineLibrary(&blob).unwrap() });
        let t_l1 = Instant::now();
        for i in 0..16usize {
            let name: Vec<u16> = format!("pso_{i}\0").encode_utf16().collect();
            let d = make_desc(&root_load, &variants[i]);
            unsafe { lib2.LoadGraphicsPipeline::<_, ID3D12PipelineState>(PCWSTR(name.as_ptr()), &d).unwrap() };
        }
        let load1_ms = t_l1.elapsed().as_secs_f64() * 1000.0;

        let lib3: Arc<ID3D12PipelineLibrary1> = Arc::new(unsafe { dev_load1.CreatePipelineLibrary(&blob).unwrap() });
        let t_l8 = Instant::now();
        let mut hs = Vec::new();
        for tid in 0..8usize {
            let l = lib3.clone();
            let rt = root_load.clone();
            let vars = variants.clone();
            let vs = vs_dxil.clone();
            hs.push(std::thread::spawn(move || {
                for i in (tid * 2)..(tid * 2 + 2) {
                    let name: Vec<u16> = format!("pso_{i}\0").encode_utf16().collect();
                    let d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                        pRootSignature: unsafe { std::mem::transmute_copy(&*rt) },
                        VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs.as_ptr().cast(), BytecodeLength: vs.len() },
                        PS: D3D12_SHADER_BYTECODE { pShaderBytecode: vars[i].as_ptr().cast(), BytecodeLength: vars[i].len() },
                        BlendState: D3D12_BLEND_DESC {
                            RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: 0xf, ..Default::default() }; 8],
                            ..Default::default()
                        },
                        SampleMask: u32::MAX,
                        RasterizerState: D3D12_RASTERIZER_DESC {
                            FillMode: D3D12_FILL_MODE_SOLID,
                            CullMode: D3D12_CULL_MODE_NONE,
                            ..Default::default()
                        },
                        PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                        NumRenderTargets: 1,
                        RTVFormats: [
                            DXGI_FORMAT_R32G32B32A32_FLOAT,
                            DXGI_FORMAT_UNKNOWN,
                            DXGI_FORMAT_UNKNOWN,
                            DXGI_FORMAT_UNKNOWN,
                            DXGI_FORMAT_UNKNOWN,
                            DXGI_FORMAT_UNKNOWN,
                            DXGI_FORMAT_UNKNOWN,
                            DXGI_FORMAT_UNKNOWN,
                        ],
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        ..Default::default()
                    };
                    unsafe { l.LoadGraphicsPipeline::<_, ID3D12PipelineState>(PCWSTR(name.as_ptr()), &d).unwrap() };
                }
            }));
        }
        for h in hs {
            h.join().unwrap();
        }
        let load8_ms = t_l8.elapsed().as_secs_f64() * 1000.0;
        println!(
            "  PipelineLibrary1 (16 PSOs, {sz} B): store={store_ms:.2}ms ser={ser_ms:.2}ms load-1t={load1_ms:.2}ms load-8t={load8_ms:.2}ms"
        );
    }
}

#[test]
#[ignore]
fn probe_5_deep_dive() {
    use std::sync::Barrier;

    println!("\n=== probe_5_deep_dive ===");

    // 1. Check whether CopyTextureRegion into unmapped Reserved textures allocates VRAM and how long it takes.
    {
        let mut p = Probe::new(false);
        let mem = |p: &Probe| -> (u64, u64) {
            let mut l = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            let mut nl = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
            unsafe {
                p.adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut l).unwrap();
                p.adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &mut nl).unwrap();
            }
            (l.CurrentUsage, nl.CurrentUsage)
        };
        let (l0, nl0) = mem(&p);

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 4096,
            Height: 4096,
            DepthOrArraySize: 1,
            MipLevels: 13,
            Format: DXGI_FORMAT_BC7_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Layout: D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
            ..Default::default()
        };
        let mut footprints = [D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default(); 13];
        let mut rows = [0u32; 13];
        let mut wide = [0u64; 13];
        let mut total = 0u64;
        unsafe {
            p.device.GetCopyableFootprints(
                &desc,
                0,
                13,
                0,
                Some(footprints.as_mut_ptr()),
                Some(rows.as_mut_ptr()),
                Some(wide.as_mut_ptr()),
                Some(&mut total),
            );
        }
        let upload = make_buffer(
            &p.device,
            D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
            total,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        let up_ptr = map_ptr(&upload);
        unsafe { std::ptr::write_bytes(up_ptr, 0x33, total as usize); upload.Unmap(0, None); }

        let mut reserveds = Vec::new();
        for _ in 0..16 {
            let mut r: Option<ID3D12Resource> = None;
            unsafe { p.device.CreateReservedResource(&desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut r).unwrap() };
            reserveds.push(r.unwrap());
        }
        let (l1, nl1) = mem(&p);

        let mut total_all_us = 0.0;
        let mut total_packed_only_us = 0.0;
        for (idx, tex) in reserveds.iter().enumerate() {
            let sub = |s: u32| D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { std::mem::transmute_copy(tex) },
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: s },
            };
            let placed = |fp: D3D12_PLACED_SUBRESOURCE_FOOTPRINT| D3D12_TEXTURE_COPY_LOCATION {
                pResource: unsafe { std::mem::transmute_copy(&upload) },
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
            };
            let us_all = p.run_timed_us(|l| unsafe {
                for m in 0..13 {
                    l.CopyTextureRegion(&sub(m as u32), 0, 0, 0, &placed(footprints[m]), None);
                }
            });
            let us_packed = p.run_timed_us(|l| unsafe {
                for m in 5..13 {
                    l.CopyTextureRegion(&sub(m as u32), 0, 0, 0, &placed(footprints[m]), None);
                }
            });
            if idx > 0 {
                total_all_us += us_all;
                total_packed_only_us += us_packed;
            }
        }
        let (l2, nl2) = mem(&p);
        println!("--- Unmapped Reserved CopyTextureRegion (16 x 4096x4096 BC7, total={total} B each) ---");
        println!(
            "  VRAM LOCAL: base={}MB -> reserved={}MB -> after CopyTextureRegion(unmapped)={}MB; NON_LOCAL: {}->{}->{}MB",
            l0 >> 20, l1 >> 20, l2 >> 20, nl0 >> 20, nl1 >> 20, nl2 >> 20
        );
        println!(
            "  CopyTextureRegion avg/tex: all 13 mips (mips 0..4 unmapped!) = {:.1} us, packed mips 5..12 only = {:.1} us ({:.1}x)",
            total_all_us / 15.0,
            total_packed_only_us / 15.0,
            (total_all_us / 15.0) / (total_packed_only_us / 15.0).max(0.01)
        );
    }

    // 2. PipelineLibrary1 LoadGraphicsPipeline & CreateGraphicsPipelineState scaling WITHOUT thread::spawn overhead!
    {
        let p = Probe::new(false);
        let vs_hlsl = r#"
            float4 main(uint vid : SV_VertexID) : SV_Position {
                float2 p = float2((vid << 1) & 2, vid & 2);
                return float4(p * 2.0 - 1.0, 0.0, 1.0);
            }
        "#;
        let vs_dxil = compile_dxc(vs_hlsl, "main", "vs_6_0").expect("vs_6_0");
        let mut ps_variants = Vec::new();
        for k in 0..32u32 {
            let src = format!(
                r#"
cbuffer C : register(b0) {{ uint iters; uint p0; uint p1; uint p2; }};
[RootSignature("RootConstants(num32BitConstants=4, b0)")]
float4 main(float4 pos : SV_Position) : SV_Target {{
    float x = pos.x * 0.001f + {k}.0f;
    float y = pos.y * 0.002f + {k}.5f;
    [loop]
    for (uint i = 0; i < iters + 16; ++i) {{
        x = sin(x * 1.1f + y) + cos(y * 0.7f - x);
        y = mad(x, 0.99f, y * 0.5f + float({k}) * 0.01f);
    }}
    return float4(x, y, float({k}), 1.0f);
}}
"#
            );
            ps_variants.push(compile_dxc(&src, "main", "ps_6_0").expect("variant"));
        }
        let dev1_1: ID3D12Device1 = p.device.cast().unwrap();
        let lib: ID3D12PipelineLibrary1 = unsafe { dev1_1.CreatePipelineLibrary(&[]).unwrap() };
        let root: ID3D12RootSignature = unsafe { p.device.CreateRootSignature(0, &ps_variants[0]).unwrap() };
        let make_desc = |rt_sig: &ID3D12RootSignature, ps: &[u8]| D3D12_GRAPHICS_PIPELINE_STATE_DESC {
            pRootSignature: unsafe { std::mem::transmute_copy(rt_sig) },
            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs_dxil.as_ptr().cast(), BytecodeLength: vs_dxil.len() },
            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps.as_ptr().cast(), BytecodeLength: ps.len() },
            BlendState: D3D12_BLEND_DESC {
                RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: 0xf, ..Default::default() }; 8],
                ..Default::default()
            },
            SampleMask: u32::MAX,
            RasterizerState: D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                ..Default::default()
            },
            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
            NumRenderTargets: 1,
            RTVFormats: [
                DXGI_FORMAT_R32G32B32A32_FLOAT,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
                DXGI_FORMAT_UNKNOWN,
            ],
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        for (i, ps) in ps_variants.iter().enumerate() {
            let d = make_desc(&root, ps);
            let pso: ID3D12PipelineState = unsafe { p.device.CreateGraphicsPipelineState(&d).unwrap() };
            let name: Vec<u16> = format!("pso_{i}\0").encode_utf16().collect();
            unsafe { lib.StorePipeline(PCWSTR(name.as_ptr()), &pso).unwrap() };
        }
        let sz = unsafe { lib.GetSerializedSize() };
        let mut blob = vec![0u8; sz];
        unsafe { lib.Serialize(&mut blob).unwrap() };

        println!("--- PipelineLibrary1::LoadGraphicsPipeline (32 PSOs, pre-spawned threads) ---");
        let vars = Arc::new(ps_variants);
        let vs = Arc::new(vs_dxil);
        let blob = Arc::new(blob);
        for threads in [1usize, 2, 4, 8] {
            let mut sum_ms = 0.0;
            for _ in 0..3 {
                let d_new = Arc::new(Probe::new(false).device);
                let rt_new = Arc::new(unsafe { d_new.CreateRootSignature::<ID3D12RootSignature>(0, &vars[0]).unwrap() });
                let d_new1: ID3D12Device1 = d_new.cast().unwrap();
                let lib_new: Arc<ID3D12PipelineLibrary1> = Arc::new(unsafe { d_new1.CreatePipelineLibrary(&blob).unwrap() });
                let bar = Arc::new(Barrier::new(threads + 1));
                let mut hs = Vec::new();
                for tid in 0..threads {
                    let b = bar.clone();
                    let l = lib_new.clone();
                    let rt = rt_new.clone();
                    let v = vars.clone();
                    let vsh = vs.clone();
                    hs.push(std::thread::spawn(move || {
                        b.wait();
                        for i in (tid..32).step_by(threads) {
                            let name: Vec<u16> = format!("pso_{i}\0").encode_utf16().collect();
                            let d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                                pRootSignature: unsafe { std::mem::transmute_copy(&*rt) },
                                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vsh.as_ptr().cast(), BytecodeLength: vsh.len() },
                                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: v[i].as_ptr().cast(), BytecodeLength: v[i].len() },
                                BlendState: D3D12_BLEND_DESC {
                                    RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: 0xf, ..Default::default() }; 8],
                                    ..Default::default()
                                },
                                SampleMask: u32::MAX,
                                RasterizerState: D3D12_RASTERIZER_DESC {
                                    FillMode: D3D12_FILL_MODE_SOLID,
                                    CullMode: D3D12_CULL_MODE_NONE,
                                    ..Default::default()
                                },
                                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                                NumRenderTargets: 1,
                                RTVFormats: [
                                    DXGI_FORMAT_R32G32B32A32_FLOAT,
                                    DXGI_FORMAT_UNKNOWN,
                                    DXGI_FORMAT_UNKNOWN,
                                    DXGI_FORMAT_UNKNOWN,
                                    DXGI_FORMAT_UNKNOWN,
                                    DXGI_FORMAT_UNKNOWN,
                                    DXGI_FORMAT_UNKNOWN,
                                    DXGI_FORMAT_UNKNOWN,
                                ],
                                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                                ..Default::default()
                            };
                            unsafe { l.LoadGraphicsPipeline::<_, ID3D12PipelineState>(PCWSTR(name.as_ptr()), &d).unwrap() };
                        }
                    }));
                }
                bar.wait();
                let t = Instant::now();
                for h in hs {
                    h.join().unwrap();
                }
                sum_ms += t.elapsed().as_secs_f64() * 1000.0;
            }
            println!("  {threads} thread(s): avg={:.2} ms ({:.2} us/PSO)", sum_ms / 3.0, (sum_ms / 3.0) * 1000.0 / 32.0);
        }

        // Also check CreateGraphicsPipelineState (fresh device) 1 vs 2 vs 4 vs 8 threads (barrier-synced!)
        println!("--- CreateGraphicsPipelineState (32 PSOs, pre-spawned threads) ---");
        for threads in [1usize, 2, 4, 8] {
            let d_new = Arc::new(Probe::new(false).device);
            let rt_new = Arc::new(unsafe { d_new.CreateRootSignature::<ID3D12RootSignature>(0, &vars[0]).unwrap() });
            let bar = Arc::new(Barrier::new(threads + 1));
            let mut hs = Vec::new();
            for tid in 0..threads {
                let b = bar.clone();
                let dev = d_new.clone();
                let rt = rt_new.clone();
                let v = vars.clone();
                let vsh = vs.clone();
                hs.push(std::thread::spawn(move || {
                    b.wait();
                    for i in (tid..32).step_by(threads) {
                        let d = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                            pRootSignature: unsafe { std::mem::transmute_copy(&*rt) },
                            VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vsh.as_ptr().cast(), BytecodeLength: vsh.len() },
                            PS: D3D12_SHADER_BYTECODE { pShaderBytecode: v[i].as_ptr().cast(), BytecodeLength: v[i].len() },
                            BlendState: D3D12_BLEND_DESC {
                                RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: 0xf, ..Default::default() }; 8],
                                ..Default::default()
                            },
                            SampleMask: u32::MAX,
                            RasterizerState: D3D12_RASTERIZER_DESC {
                                FillMode: D3D12_FILL_MODE_SOLID,
                                CullMode: D3D12_CULL_MODE_NONE,
                                ..Default::default()
                            },
                            PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                            NumRenderTargets: 1,
                            RTVFormats: [
                                DXGI_FORMAT_R32G32B32A32_FLOAT,
                                DXGI_FORMAT_UNKNOWN,
                                DXGI_FORMAT_UNKNOWN,
                                DXGI_FORMAT_UNKNOWN,
                                DXGI_FORMAT_UNKNOWN,
                                DXGI_FORMAT_UNKNOWN,
                                DXGI_FORMAT_UNKNOWN,
                                DXGI_FORMAT_UNKNOWN,
                            ],
                            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                            ..Default::default()
                        };
                        unsafe { dev.CreateGraphicsPipelineState::<ID3D12PipelineState>(&d).unwrap() };
                    }
                }));
            }
            bar.wait();
            let t = Instant::now();
            for h in hs {
                h.join().unwrap();
            }
            let ms = t.elapsed().as_secs_f64() * 1000.0;
            println!("  {threads} thread(s): {ms:.2} ms ({:.2} ms/PSO)", ms / 32.0);
        }
    }

    // 3. End-to-end Scratch / Texture Upload benchmark: UPLOAD vs GPU_UPLOAD (with Debug Layer check too!)
    {
        println!("--- UPLOAD vs GPU_UPLOAD Realistic End-to-End (Scratch CBV/SRV + Texture2D upload, debug_layer=true) ---");
        let mut p = Probe::new(true);
        let mut o16 = D3D12_FEATURE_DATA_D3D12_OPTIONS16::default();
        let ok16 = unsafe {
            p.device.CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS16, (&raw mut o16).cast(), size_of_val(&o16) as u32).is_ok()
                && o16.GPUUploadHeapSupported.as_bool()
        };
        println!("  GPUUploadHeapSupported = {ok16}");
        if ok16 {
            // (a) 256 KiB scratch buffer: CPU write + 100 GPU CS reads (simulating CBV/vertex/strides read across draws!)
            const SCRATCH_SZ: usize = 256 << 10;
            let up_scratch = make_buffer(
                &p.device,
                D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                SCRATCH_SZ as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let gup_scratch = make_buffer(
                &p.device,
                D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() },
                SCRATCH_SZ as u64,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let dst_uav = make_buffer(
                &p.device,
                D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                256,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            );
            let cs_src = r#"
                ByteAddressBuffer src : register(t0);
                RWByteAddressBuffer dst : register(u0);
                [RootSignature("SRV(t0), UAV(u0)")]
                [numthreads(64, 1, 1)]
                void main(uint3 id : SV_DispatchThreadID) {
                    uint acc = 0;
                    for (uint off = id.x * 16; off < 262144; off += 64 * 64 * 16) {
                        uint4 v = src.Load4(off);
                        acc ^= v.x + v.y + v.z + v.w;
                    }
                    if (id.x == 0) dst.Store(0, acc);
                }
            "#;
            let cs = compile_fxc(cs_src, &s!("main"), &s!("cs_5_1"));
            let root: ID3D12RootSignature = unsafe { p.device.CreateRootSignature(0, &cs).unwrap() };
            let pso: ID3D12PipelineState = unsafe {
                p.device
                    .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                        pRootSignature: std::mem::transmute_copy(&root),
                        CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cs.as_ptr().cast(), BytecodeLength: cs.len() },
                        ..Default::default()
                    })
                    .unwrap()
            };
            let up_p = map_ptr(&up_scratch);
            let gup_p = map_ptr(&gup_scratch);
            let src_data = vec![0x7au8; SCRATCH_SZ];
            for (name, buf, ptr) in [("UPLOAD", &up_scratch, up_p), ("GPU_UPLOAD", &gup_scratch, gup_p)] {
                let mut sum_total_us = 0.0;
                let mut sum_gpu_us = 0.0;
                for _ in 0..20 {
                    let t = Instant::now();
                    unsafe { std::ptr::copy_nonoverlapping(src_data.as_ptr(), ptr, SCRATCH_SZ) };
                    let gpu_us = p.run_timed_us(|l| unsafe {
                        l.SetComputeRootSignature(&root);
                        l.SetPipelineState(&pso);
                        l.SetComputeRootUnorderedAccessView(1, dst_uav.GetGPUVirtualAddress());
                        for _ in 0..50 {
                            l.SetComputeRootShaderResourceView(0, buf.GetGPUVirtualAddress());
                            l.Dispatch(64, 1, 1);
                        }
                    });
                    sum_total_us += t.elapsed().as_secs_f64() * 1_000_000.0;
                    sum_gpu_us += gpu_us;
                }
                p.drain_messages(name);
                println!("  256KiB scratch CPU write + 50 GPU dispatches ({name:10}): cpu+gpu={:.1} us, gpu_only={:.1} us", sum_total_us / 20.0, sum_gpu_us / 20.0);
            }

            // (b) 16 MiB texture upload (2048x2048 BC7): CPU write + CopyTextureRegion
            let tdesc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 2048,
                Height: 2048,
                DepthOrArraySize: 1,
                MipLevels: 1,
                Format: DXGI_FORMAT_BC7_UNORM,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
                ..Default::default()
            };
            let mut tex_def: Option<ID3D12Resource> = None;
            unsafe {
                p.device.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                    D3D12_HEAP_FLAG_NONE,
                    &tdesc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut tex_def,
                ).unwrap();
            }
            let tex_def = tex_def.unwrap();
            let mut fp = D3D12_PLACED_SUBRESOURCE_FOOTPRINT::default();
            let mut total = 0u64;
            unsafe { p.device.GetCopyableFootprints(&tdesc, 0, 1, 0, Some(&mut fp), None, None, Some(&mut total)) };
            let up_tex = make_buffer(
                &p.device,
                D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
                total,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            let gup_tex = make_buffer(
                &p.device,
                D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() },
                total,
                D3D12_RESOURCE_FLAG_NONE,
                D3D12_RESOURCE_STATE_COMMON,
            );
            let utp = map_ptr(&up_tex);
            let gutp = map_ptr(&gup_tex);
            let tex_src = vec![0x42u8; total as usize];
            for (name, buf, ptr) in [("UPLOAD", &up_tex, utp), ("GPU_UPLOAD", &gup_tex, gutp)] {
                let mut sum_total_us = 0.0;
                let mut sum_gpu_us = 0.0;
                for _ in 0..10 {
                    let t = Instant::now();
                    unsafe { std::ptr::copy_nonoverlapping(tex_src.as_ptr(), ptr, total as usize) };
                    let gpu_us = p.run_timed_us(|l| unsafe {
                        let dst = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(&tex_def),
                            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
                        };
                        let src = D3D12_TEXTURE_COPY_LOCATION {
                            pResource: std::mem::transmute_copy(buf),
                            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { PlacedFootprint: fp },
                        };
                        l.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
                    });
                    sum_total_us += t.elapsed().as_secs_f64() * 1_000_000.0;
                    sum_gpu_us += gpu_us;
                }
                p.drain_messages(name);
                println!("  4 MiB Tex2D CPU write + CopyTextureRegion ({name:10}): cpu+gpu={:.1} us, gpu_only={:.1} us", sum_total_us / 10.0, sum_gpu_us / 10.0);
            }
        }
    }
}

#[test]
#[ignore]
fn probe_6_graphics_gpu_upload_and_card_compare() {
    println!("\n=== probe_6_graphics_gpu_upload_and_card_compare ===");
    let mut p = Probe::new(true);

    // 1. Graphics pipeline reading Vertex Buffer + Root CBV + SRV from UPLOAD vs GPU_UPLOAD (COMMON state, debug_layer=true!)
    let vs_src = r#"
        struct VSIn { float4 pos : POSITION; float4 col : COLOR; };
        struct VSOut { float4 pos : SV_Position; float4 col : COLOR; };
        cbuffer C : register(b0) { float4 scale; float4 bias; };
        ByteAddressBuffer buf : register(t0);
        [RootSignature("RootFlags(ALLOW_INPUT_ASSEMBLER_INPUT_LAYOUT), CBV(b0), SRV(t0)")]
        VSOut main(VSIn v, uint vid : SV_VertexID) {
            VSOut o;
            o.pos = float4(v.pos.xy * scale.xy + bias.xy, 0.5f, 1.0f);
            o.col = v.col + asfloat(buf.Load4((vid & 1023) * 16));
            return o;
        }
    "#;
    let ps_src = r#"
        struct PSIn { float4 pos : SV_Position; float4 col : COLOR; };
        float4 main(PSIn i) : SV_Target { return i.col; }
    "#;
    let vs_dxil = compile_dxc(vs_src, "main", "vs_6_0").expect("vs");
    let ps_dxil = compile_dxc(ps_src, "main", "ps_6_0").expect("ps");
    let root: ID3D12RootSignature = unsafe { p.device.CreateRootSignature(0, &vs_dxil).unwrap() };
    let elems = [
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("POSITION"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
        D3D12_INPUT_ELEMENT_DESC {
            SemanticName: s!("COLOR"),
            SemanticIndex: 0,
            Format: DXGI_FORMAT_R32G32B32A32_FLOAT,
            InputSlot: 0,
            AlignedByteOffset: 16,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        },
    ];
    let pso: ID3D12PipelineState = unsafe {
        let res = p.device
            .CreateGraphicsPipelineState(&D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: D3D12_SHADER_BYTECODE { pShaderBytecode: vs_dxil.as_ptr().cast(), BytecodeLength: vs_dxil.len() },
                PS: D3D12_SHADER_BYTECODE { pShaderBytecode: ps_dxil.as_ptr().cast(), BytecodeLength: ps_dxil.len() },
                BlendState: D3D12_BLEND_DESC {
                    RenderTarget: [D3D12_RENDER_TARGET_BLEND_DESC { RenderTargetWriteMask: 0xf, ..Default::default() }; 8],
                    ..Default::default()
                },
                SampleMask: u32::MAX,
                RasterizerState: D3D12_RASTERIZER_DESC {
                    FillMode: D3D12_FILL_MODE_SOLID,
                    CullMode: D3D12_CULL_MODE_NONE,
                    ..Default::default()
                },
                InputLayout: D3D12_INPUT_LAYOUT_DESC {
                    pInputElementDescs: elems.as_ptr(),
                    NumElements: elems.len() as u32,
                },
                PrimitiveTopologyType: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
                NumRenderTargets: 1,
                RTVFormats: [
                    DXGI_FORMAT_R8G8B8A8_UNORM,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                    DXGI_FORMAT_UNKNOWN,
                ],
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                ..Default::default()
            });
        p.drain_messages("create_gfx_pso");
        res.unwrap()
    };
    let rt_desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
        Width: 512,
        Height: 512,
        DepthOrArraySize: 1,
        MipLevels: 1,
        Format: DXGI_FORMAT_R8G8B8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_UNKNOWN,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
        ..Default::default()
    };
    let mut rt: Option<ID3D12Resource> = None;
    unsafe {
        p.device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                D3D12_HEAP_FLAG_NONE,
                &rt_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut rt,
            )
            .unwrap();
    }
    let rt = rt.unwrap();
    let rtv_heap: ID3D12DescriptorHeap = unsafe {
        p.device
            .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: 1,
                ..Default::default()
            })
            .unwrap()
    };
    let rtv = unsafe { rtv_heap.GetCPUDescriptorHandleForHeapStart() };
    unsafe { p.device.CreateRenderTargetView(&rt, None, rtv) };

    const BUF_SZ: usize = 4 << 20;
    let up_b = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_UPLOAD, ..Default::default() },
        BUF_SZ as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let gup_b = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_GPU_UPLOAD, ..Default::default() },
        BUF_SZ as u64,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let up_ptr = map_ptr(&up_b);
    let gup_ptr = map_ptr(&gup_b);
    let mut verts = vec![0.0f32; BUF_SZ / 4];
    for i in 0..(BUF_SZ / 16) {
        verts[i * 4 + 0] = 0.001;
        verts[i * 4 + 1] = 0.001;
        verts[i * 4 + 2] = 0.5;
        verts[i * 4 + 3] = 1.0;
    }
    let src_bytes: &[u8] = unsafe { std::slice::from_raw_parts(verts.as_ptr().cast(), BUF_SZ) };
    for (label, buf, ptr) in [("UPLOAD", &up_b, up_ptr), ("GPU_UPLOAD", &gup_b, gup_ptr)] {
        let mut cpu_gpu = 0.0;
        let mut gpu_only = 0.0;
        for _ in 0..10 {
            let t = Instant::now();
            unsafe { std::ptr::copy_nonoverlapping(src_bytes.as_ptr(), ptr, BUF_SZ) };
            let us = p.run_timed_us(|l| unsafe {
                let va = buf.GetGPUVirtualAddress();
                l.SetGraphicsRootSignature(&root);
                l.SetPipelineState(&pso);
                l.OMSetRenderTargets(1, Some(&rtv), false, None);
                l.RSSetViewports(&[D3D12_VIEWPORT {
                    TopLeftX: 0.0, TopLeftY: 0.0, Width: 512.0, Height: 512.0, MinDepth: 0.0, MaxDepth: 1.0
                }]);
                l.RSSetScissorRects(&[RECT { left: 0, top: 0, right: 512, bottom: 512 }]);
                l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
                let vbv = D3D12_VERTEX_BUFFER_VIEW {
                    BufferLocation: va + 1024,
                    SizeInBytes: (BUF_SZ - 1024) as u32,
                    StrideInBytes: 32,
                };
                l.IASetVertexBuffers(0, Some(&[vbv]));
                l.SetGraphicsRootConstantBufferView(0, va);
                l.SetGraphicsRootShaderResourceView(1, va + 256);
                for _ in 0..20 {
                    l.DrawInstanced(120_000, 1, 0, 0);
                }
            });
            cpu_gpu += t.elapsed().as_secs_f64() * 1_000_000.0;
            gpu_only += us;
        }
        p.drain_messages(label);
        println!("  Graphics 4MiB VB+CBV+SRV x20 draws ({label:10}): cpu+gpu={:.1} us, gpu_only={:.1} us", cpu_gpu / 10.0, gpu_only / 10.0);
    }

    // 2. Benchmark CARD_COMPARE HLSL on Arc A770 (64 MiB = 1024 pieces)
    let card_compare_cs = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first; uint skew; uint size; uint pieces; uint out_at; uint compare;
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pieces) return;
    uint piece = first + id.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [loop]
    for (uint at = lo; at < hi; at += 16u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (compare != 0u) {
        differs.Store(out_at + id.x * 4u, any(h != hashes.Load4(piece * 16u)) ? 1u : 0u);
        if (compare == 2u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
"#;
    let cc = compile_fxc(card_compare_cs, &s!("main"), &s!("cs_5_1"));
    let cc_root: ID3D12RootSignature = unsafe { p.device.CreateRootSignature(0, &cc).unwrap() };
    let cc_pso: ID3D12PipelineState = unsafe {
        p.device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&cc_root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: cc.as_ptr().cast(), BytecodeLength: cc.len() },
                ..Default::default()
            })
            .unwrap()
    };
    const CMP_SZ: u64 = 64 << 20;
    let cmp_src = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        CMP_SZ,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let cmp_hashes = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        1024 * 16,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let cmp_differs = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        },
        1024 * 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let run_cc = |p: &mut Probe, pieces: u32| {
        p.run_timed_us(|l| unsafe {
            l.SetComputeRootSignature(&cc_root);
            l.SetPipelineState(&cc_pso);
            let c = [0u32, 0, CMP_SZ as u32, pieces, 0, 2];
            l.SetComputeRoot32BitConstants(0, 6, c.as_ptr().cast(), 0);
            l.SetComputeRootShaderResourceView(1, cmp_src.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(2, cmp_hashes.GetGPUVirtualAddress());
            l.SetComputeRootUnorderedAccessView(3, cmp_differs.GetGPUVirtualAddress());
            l.Dispatch(pieces.div_ceil(64), 1, 1);
        })
    };
    run_cc(&mut p, 1024);
    for pieces in [1u32, 64, 1024] {
        let mut s = 0.0;
        for _ in 0..8 {
            s += run_cc(&mut p, pieces);
        }
        println!("  CARD_COMPARE ({pieces:4} pieces = {:5.1} MiB): {:.1} us", f64::from(pieces) * 64.0 / 1024.0, s / 8.0);
    }
}

#[test]
#[ignore]
fn probe_7_coalesced_card_compare() {
    println!("\n=== probe_7_coalesced_card_compare ===");
    let mut p = Probe::new(true);

    let orig_cs = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first; uint skew; uint size; uint pieces; uint out_at; uint compare;
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pieces) return;
    uint piece = first + id.x;
    uint lo = piece * 65536u;
    lo = lo > skew ? lo - skew : 0u;
    uint hi = min((piece + 1u) * 65536u - skew, size);
    lo = lo & ~15u;
    hi = (hi + 15u) & ~15u;
    uint4 h = uint4(0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u, 0x27D4EB2Fu);
    [loop]
    for (uint at = lo; at < hi; at += 16u) {
        h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
        h = h.yzwx ^ (h >> 15);
    }
    if (compare != 0u) {
        differs.Store(out_at + id.x * 4u, any(h != hashes.Load4(piece * 16u)) ? 1u : 0u);
        if (compare == 2u) hashes.Store4(piece * 16u, h);
    } else {
        hashes.Store4(piece * 16u, h);
    }
}
"#;

    // 1 threadgroup (64 threads) per piece: coalesced 16-byte loads across 64 threads, shared reduction.
    let fast_cs = r#"
#define RS "RootConstants(num32BitConstants=6, b0), SRV(t0), UAV(u0), UAV(u1)"
cbuffer Span : register(b0) {
    uint first; uint skew; uint size; uint pieces; uint out_at; uint compare;
};
ByteAddressBuffer now : register(t0);
RWByteAddressBuffer hashes : register(u0);
RWByteAddressBuffer differs : register(u1);
groupshared uint4 lane_h[64];
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint3 g : SV_GroupID, uint t : SV_GroupIndex) {
    for (uint i = g.x; i < pieces; i += 65535u) {
        uint piece = first + i;
        uint lo = piece * 65536u;
        lo = lo > skew ? lo - skew : 0u;
        uint hi = min((piece + 1u) * 65536u - skew, size);
        lo = lo & ~15u;
        hi = (hi + 15u) & ~15u;
        uint4 h = uint4(0x9E3779B9u ^ t, 0x85EBCA6Bu + t * 17u, 0xC2B2AE35u, 0x27D4EB2Fu);
        [loop]
        for (uint at = lo + t * 16u; at < hi; at += 1024u) {
            h = (h ^ now.Load4(at)) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
            h = h.yzwx ^ (h >> 15);
        }
        lane_h[t] = h;
        GroupMemoryBarrierWithGroupSync();
        if (t < 32u) {
            uint4 b = lane_h[t + 32u];
            lane_h[t] = (lane_h[t] ^ b.yzwx) * uint4(0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du, 0x27D4EB2Fu);
        }
        GroupMemoryBarrierWithGroupSync();
        if (t < 16u) {
            uint4 b = lane_h[t + 16u];
            lane_h[t] = (lane_h[t] ^ b.zwxy) * uint4(0x01000193u, 0x9E3779B9u, 0x85EBCA6Bu, 0xC2B2AE35u);
        }
        GroupMemoryBarrierWithGroupSync();
        if (t == 0u) {
            uint4 acc = lane_h[0];
            [unroll]
            for (uint k = 1u; k < 16u; ++k) {
                acc = (acc ^ lane_h[k]) * uint4(0x01000193u, 0x9E3779B1u, 0x85EBCA77u, 0xC2B2AE3Du);
                acc = acc.yzwx ^ (acc >> 15);
            }
            if (compare != 0u) {
                differs.Store(out_at + i * 4u, any(acc != hashes.Load4(piece * 16u)) ? 1u : 0u);
                if (compare == 2u) hashes.Store4(piece * 16u, acc);
            } else {
                hashes.Store4(piece * 16u, acc);
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
"#;

    let compile = |src: &str| {
        let bin = compile_fxc(src, &s!("main"), &s!("cs_5_1"));
        let root: ID3D12RootSignature = unsafe { p.device.CreateRootSignature(0, &bin).unwrap() };
        let pso: ID3D12PipelineState = unsafe {
            p.device.CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: bin.as_ptr().cast(), BytecodeLength: bin.len() },
                ..Default::default()
            }).unwrap()
        };
        (root, pso)
    };
    let (orig_root, orig_pso) = compile(orig_cs);
    let (fast_root, fast_pso) = compile(fast_cs);

    const CMP_SZ: u64 = 64 << 20;
    let cmp_src = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        CMP_SZ,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let cmp_hashes = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
        1024 * 16,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );
    let cmp_differs = make_buffer(
        &p.device,
        D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_CUSTOM,
            CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
            MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
            ..Default::default()
        },
        1024 * 4,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_COMMON,
    );

    for (name, root, pso, per_group) in [
        ("orig (1 thr/piece)", &orig_root, &orig_pso, false),
        ("fast (64 thr/piece)", &fast_root, &fast_pso, true),
    ] {
        let dispatch = |p: &mut Probe, pieces: u32| {
            p.run_timed_us(|l| unsafe {
                l.SetComputeRootSignature(root);
                l.SetPipelineState(pso);
                let c = [0u32, 0, CMP_SZ as u32, pieces, 0, 2];
                l.SetComputeRoot32BitConstants(0, 6, c.as_ptr().cast(), 0);
                l.SetComputeRootShaderResourceView(1, cmp_src.GetGPUVirtualAddress());
                l.SetComputeRootUnorderedAccessView(2, cmp_hashes.GetGPUVirtualAddress());
                l.SetComputeRootUnorderedAccessView(3, cmp_differs.GetGPUVirtualAddress());
                let g = if per_group { pieces.clamp(1, 65535) } else { pieces.div_ceil(64) };
                l.Dispatch(g, 1, 1);
            })
        };
        dispatch(&mut p, 1024);
        for pieces in [1u32, 8, 64, 1024] {
            let mut sum = 0.0;
            for _ in 0..8 {
                sum += dispatch(&mut p, pieces);
            }
            println!(
                "  {name:20} ({pieces:4} pieces = {:5.2} MiB): {:7.1} us",
                f64::from(pieces) * 64.0 / 1024.0,
                sum / 8.0
            );
        }
    }
    p.drain_messages("probe_7");
}
