//! Card time of mesh draws, to tell what a translated console vertex program run as a mesh
//! shader pays per draw: a list of many small `DispatchMesh` calls between two timestamps,
//! the time per call printed. Ignored unless asked for (they need a card and `dxc.exe`;
//! `VICTORY_TEST_DXC` names it, default the Windows SDK's):
//! `cargo test -p victory-shader --lib mesh_bench -- --ignored --nocapture --test-threads 1`,
//! `mesh_bench_cascade_shadows_draw` with `VICTORY_TEST_MESH_CONTAINER` = a container file of
//! the title's `MS_VS_CascadeShadows_draw` (db875f50a166bd91).
//!
//! What they showed on an RTX 5070 Ti (2026-11-25): the translated cascade-shadow mesh shader
//! costs 0.3 us a draw of 1 to 16 meshlets of 128 vertices (cold caches, vertex data in the
//! card's or the PC's memory alike) and 4 us for 256; draws overlap on the card, a timestamp
//! after each adds 0.2 us. Any barrier after a draw (a UAV barrier, or a transition of an
//! unrelated buffer) drains the card: 4-6 us a draw; `victory-gpu`'s clamp of an indirect
//! record before each draw (a compute pass between two such waits) 7-12 us a draw, the
//! 12-16 us the title's profile shows per mesh draw. Clamped 64 records at a time, 0.3 us;
//! read and clamped by an amplification shader in front (`amplify.rs`), 0.4-0.6 us, with
//! the same depth image as a direct dispatch.

use windows::Win32::Graphics::Direct3D::*;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::Interface;

use super::host::Host;

fn dxc() -> String {
    std::env::var("VICTORY_TEST_DXC").unwrap_or_else(|_| {
        r"C:\Program Files (x86)\Windows Kits\10\bin\10.0.26100.0\x64\dxc.exe".to_owned()
    })
}

/// `source` compiled for `profile` with `defines`.
fn compile(source: &str, profile: &str, defines: &[String]) -> Vec<u8> {
    let dir = std::env::temp_dir().join(format!("victory-mesh-bench-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
    let k = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let src = dir.join(format!("s{k}.hlsl"));
    let out = dir.join(format!("s{k}.dxil"));
    std::fs::write(&src, source).unwrap();
    let mut cmd = std::process::Command::new(dxc());
    cmd.arg("-T")
        .arg(profile)
        .arg("-E")
        .arg("main")
        .arg("-O3")
        .arg("-Fo")
        .arg(&out)
        .arg(&src);
    for d in defines {
        cmd.arg("-D").arg(d);
    }
    let o = cmd.output().expect("dxc");
    assert!(
        o.status.success(),
        "dxc: {}",
        String::from_utf8_lossy(&o.stderr)
    );
    std::fs::read(&out).unwrap()
}

fn err(e: windows::core::Error) -> String {
    e.message().to_string()
}

fn buffer(
    dev: &ID3D12Device,
    heap: D3D12_HEAP_TYPE,
    size: u64,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: size,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut r: Option<ID3D12Resource> = None;
    // SAFETY: valid descriptions and out pointer.
    unsafe {
        dev.CreateCommittedResource(
            &D3D12_HEAP_PROPERTIES {
                Type: heap,
                ..Default::default()
            },
            D3D12_HEAP_FLAG_NONE,
            &desc,
            state,
            None,
            &mut r,
        )
        .unwrap();
    }
    r.unwrap()
}

/// A default-heap buffer holding `words`.
fn upload(
    dev: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    keep: &mut Vec<ID3D12Resource>,
    words: &[u32],
) -> ID3D12Resource {
    let size = (words.len() * 4).max(256) as u64;
    let up = buffer(
        dev,
        D3D12_HEAP_TYPE_UPLOAD,
        size,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let gpu = buffer(
        dev,
        D3D12_HEAP_TYPE_DEFAULT,
        size,
        D3D12_RESOURCE_STATE_COMMON,
    );
    // SAFETY: mapped memory of `size` bytes; the copy is recorded on an open list.
    unsafe {
        let mut p = std::ptr::null_mut();
        up.Map(0, None, Some(&mut p)).unwrap();
        std::ptr::copy_nonoverlapping(words.as_ptr(), p.cast::<u32>(), words.len());
        up.Unmap(0, None);
        list.CopyBufferRegion(&gpu, 0, &up, 0, (words.len() * 4) as u64);
        list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    pResource: std::mem::transmute_copy(&gpu),
                    StateBefore: D3D12_RESOURCE_STATE_COPY_DEST,
                    StateAfter: D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                }),
            },
            ..Default::default()
        }]);
    }
    keep.push(up);
    gpu
}

/// What each draw of a run binds: root parameter 0 constants (the draw's index is patched
/// into dword `draw_dword` when set), then root SRVs by parameter.
pub(crate) struct Run<'a> {
    pub ms: &'a [u8],
    pub root: &'a ID3D12RootSignature,
    pub constants: Vec<u32>,
    pub draw_dword: Option<usize>,
    pub root_srvs: Vec<(u32, u64)>,
    /// When not empty: draw d takes constants and root SRVs `objects[d % len]` instead.
    pub objects: Vec<(Vec<u32>, Vec<(u32, u64)>)>,
    pub groups: u32,
    pub draws: u32,
    /// Two pipelines (this mesh shader with two pixel shaders) taken in turn.
    pub alternate: bool,
    /// A timestamp after every draw (as a profile of runs a draw long writes).
    pub stamp_each: bool,
    /// What is recorded after every draw.
    pub between: Between,
    /// A buffer the draws read, in the common state (for `Between::CopyRead`).
    pub read: Option<&'a ID3D12Resource>,
    /// How each draw gets its group count.
    pub launch: Launch,
    /// With `Launch::Amplified`: the first of the three root parameters of the
    /// amplification shader (SRV `t0`, one constant `b0`, UAV `u0`, in space 7).
    pub amplify_params: Option<u32>,
}

/// How a draw of a run is launched.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Launch {
    /// `DispatchMesh(groups, 1, 1)`.
    Direct,
    /// `ExecuteIndirect` of one record (`groups, 1, 1`), as the title's draws are.
    Indirect,
    /// The host's clamp before each: a wait for every shader write and the scratch buffer
    /// to unordered access, a compute dispatch copying the record clamped, the same wait and
    /// the scratch buffer to indirect arguments, the indirect draw from it, the scratch
    /// buffer back to common (`victory-gpu`'s `indirect::bound` and `done`).
    Clamped,
    /// The clamp of `n` records at a time ahead of their draws (`VICTORY_INDIRECT_WINDOW`).
    Window(u32),
    /// `DispatchMesh(1, 1, 1)` of a pipeline with the host's amplification shader in front
    /// (`super::amplification_shader`), which reads the record (a root SRV at it), clamps it
    /// and launches its groups: no compute pass, no wait.
    Amplified,
}

/// The clamp's stand-in: copies records `[first, first + n)` of 12 bytes, the group count
/// cut to 65,535.
const CLAMP_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), SRV(t0), UAV(u0)"
cbuffer C : register(b0) { uint first; uint n; };
ByteAddressBuffer src : register(t0);
RWByteAddressBuffer dst : register(u0);
[RootSignature(RS)]
[numthreads(64, 1, 1)]
void main(uint i : SV_DispatchThreadID) {
    if (i >= n) return;
    uint3 r = src.Load3((first + i) * 12);
    dst.Store3(i * 12, uint3(min(r.x, 65535u), r.y, r.z));
}
"#;

/// A host command recorded after every draw of a run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Between {
    Nothing,
    /// A UAV barrier on all resources.
    UavAll,
    /// An unrelated buffer to the copy destination state and back.
    Transition,
    /// 256 bytes copied into an unrelated buffer between those two transitions (a refresh
    /// of a host copy).
    Copy,
    /// The same 256 bytes copied into the buffer the draws read (between transitions).
    CopyRead,
    /// `SetDescriptorHeaps` with the same heap.
    Heaps,
    /// The root signature set again, and the root arguments with it.
    RootSignature,
    /// The targets set again.
    Targets,
}

const PS_A: &str = "float4 main(float4 p : SV_Position) : SV_Target { return float4(1,0,0,1); }";
const PS_B: &str = "float4 main(float4 p : SV_Position) : SV_Target { return float4(0,1,0,1); }";

pub(crate) fn mesh_pso_with(
    dev: &ID3D12Device,
    root: &ID3D12RootSignature,
    amplify: Option<&[u8]>,
    ms: &[u8],
    ps: &[u8],
) -> Result<ID3D12PipelineState, String> {
    #[repr(C, align(8))]
    struct Sub<T>(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, T);
    #[repr(C)]
    struct Stream {
        root: Sub<*mut std::ffi::c_void>,
        amplify: Sub<D3D12_SHADER_BYTECODE>,
        ms: Sub<D3D12_SHADER_BYTECODE>,
        ps: Sub<D3D12_SHADER_BYTECODE>,
        raster: Sub<D3D12_RASTERIZER_DESC>,
        depth: Sub<D3D12_DEPTH_STENCIL_DESC>,
        targets: Sub<D3D12_RT_FORMAT_ARRAY>,
        dsv: Sub<DXGI_FORMAT>,
        samples: Sub<DXGI_SAMPLE_DESC>,
    }
    let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
        pShaderBytecode: b.as_ptr().cast(),
        BytecodeLength: b.len(),
    };
    let mut targets = D3D12_RT_FORMAT_ARRAY::default();
    targets.RTFormats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
    targets.NumRenderTargets = 1;
    let mut stream = Stream {
        root: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE,
            root.as_raw(),
        ),
        amplify: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
            amplify.map_or(D3D12_SHADER_BYTECODE::default(), code),
        ),
        ms: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, code(ms)),
        ps: Sub(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, code(ps)),
        raster: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER,
            D3D12_RASTERIZER_DESC {
                FillMode: D3D12_FILL_MODE_SOLID,
                CullMode: D3D12_CULL_MODE_NONE,
                DepthClipEnable: true.into(),
                ..Default::default()
            },
        ),
        depth: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL,
            D3D12_DEPTH_STENCIL_DESC {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                ..Default::default()
            },
        ),
        targets: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS,
            targets,
        ),
        dsv: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT,
            DXGI_FORMAT_D32_FLOAT,
        ),
        samples: Sub(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_DESC,
            DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            },
        ),
    };
    let desc = D3D12_PIPELINE_STATE_STREAM_DESC {
        SizeInBytes: size_of::<Stream>(),
        pPipelineStateSubobjectStream: (&raw mut stream).cast(),
    };
    let device: ID3D12Device2 = dev.cast().map_err(err)?;
    // SAFETY: the stream points at live data.
    unsafe { device.CreatePipelineState(&desc) }.map_err(|e| format!("{}{}", err(e), messages(dev)))
}

/// The debug layer's stored messages (`VICTORY_D3D12_DEBUG`), emptied.
fn messages(dev: &ID3D12Device) -> String {
    let mut text = String::new();
    let Ok(queue) = dev.cast::<ID3D12InfoQueue>() else {
        return text;
    };
    // SAFETY: each message is read into a buffer of the size the queue asks for.
    unsafe {
        for i in 0..queue.GetNumStoredMessages() {
            let mut len = 0usize;
            let _ = queue.GetMessage(i, None, &mut len);
            let mut buffer = vec![0u64; len.div_ceil(8)];
            let message = buffer.as_mut_ptr().cast::<D3D12_MESSAGE>();
            if queue.GetMessage(i, Some(message), &mut len).is_ok() {
                let m = &*message;
                let bytes = std::slice::from_raw_parts(
                    m.pDescription,
                    m.DescriptionByteLength.saturating_sub(1),
                );
                text.push_str(" | ");
                text.push_str(&String::from_utf8_lossy(bytes));
            }
        }
        queue.ClearStoredMessages();
    }
    text
}

/// The bench: a heap of views (made by `views` on the open list), then each run of
/// `runs` recorded between two timestamps; returns microseconds per draw of each run.
pub(crate) struct Bench {
    host: Host,
    queue: ID3D12CommandQueue,
    pub heap: ID3D12DescriptorHeap,
    pub keep: Vec<ID3D12Resource>,
    view_size: usize,
    pub views: u32,
    /// The colour target, kept while its view is bound.
    _rt: ID3D12Resource,
    ds: ID3D12Resource,
    rtv: ID3D12DescriptorHeap,
    dsv: ID3D12DescriptorHeap,
    ps: [Vec<u8>; 2],
}

pub(crate) const SIZE: u32 = 2048;

impl Bench {
    pub fn new() -> Option<Self> {
        let host = match Host::new(false) {
            Ok(h) => h,
            Err(e) => {
                eprintln!("no card: {e}");
                return None;
            }
        };
        let dev = host.device.clone();
        // SAFETY: plain object creation.
        unsafe {
            let queue: ID3D12CommandQueue = dev
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: D3D12_COMMAND_LIST_TYPE_DIRECT,
                    ..Default::default()
                })
                .unwrap();
            let heap: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                    NumDescriptors: 4096,
                    Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                    ..Default::default()
                })
                .unwrap();
            let view_size = dev
                .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV)
                as usize;
            let tex = |format, flags, state, clear: D3D12_CLEAR_VALUE| {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: u64::from(SIZE),
                        Height: SIZE,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: format,
                        SampleDesc: DXGI_SAMPLE_DESC {
                            Count: 1,
                            Quality: 0,
                        },
                        Flags: flags,
                        ..Default::default()
                    },
                    state,
                    Some(&clear),
                    &mut r,
                )
                .unwrap();
                r.unwrap()
            };
            let rt = tex(
                DXGI_FORMAT_R8G8B8A8_UNORM,
                D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                D3D12_CLEAR_VALUE {
                    Format: DXGI_FORMAT_R8G8B8A8_UNORM,
                    Anonymous: D3D12_CLEAR_VALUE_0 { Color: [0.0; 4] },
                },
            );
            let ds = tex(
                DXGI_FORMAT_D32_FLOAT,
                D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_CLEAR_VALUE {
                    Format: DXGI_FORMAT_D32_FLOAT,
                    Anonymous: D3D12_CLEAR_VALUE_0 {
                        DepthStencil: D3D12_DEPTH_STENCIL_VALUE {
                            Depth: 1.0,
                            Stencil: 0,
                        },
                    },
                },
            );
            let rtv: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            let dsv: ID3D12DescriptorHeap = dev
                .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                    Type: D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
                    NumDescriptors: 1,
                    ..Default::default()
                })
                .unwrap();
            dev.CreateRenderTargetView(&rt, None, rtv.GetCPUDescriptorHandleForHeapStart());
            dev.CreateDepthStencilView(&ds, None, dsv.GetCPUDescriptorHandleForHeapStart());
            let ps = [compile(PS_A, "ps_6_6", &[]), compile(PS_B, "ps_6_6", &[])];
            Some(Bench {
                host,
                queue,
                heap,
                keep: Vec::new(),
                view_size,
                views: 0,
                _rt: rt,
                ds,
                rtv,
                dsv,
                ps,
            })
        }
    }

    pub fn device(&self) -> &ID3D12Device {
        &self.host.device
    }

    /// Records `f` on a list and waits for it.
    pub fn once(&mut self, f: impl FnOnce(&mut Self, &ID3D12GraphicsCommandList)) {
        let dev = self.host.device.clone();
        // SAFETY: plain D3D12 calls; resources stay in `keep`.
        unsafe {
            let alloc: ID3D12CommandAllocator = dev
                .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                .unwrap();
            let list: ID3D12GraphicsCommandList = dev
                .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, None)
                .unwrap();
            f(self, &list);
            list.Close().unwrap();
            self.queue
                .ExecuteCommandLists(&[Some(list.cast().unwrap())]);
            self.wait();
        }
    }

    fn wait(&self) {
        // SAFETY: a fence on our queue.
        unsafe {
            let fence: ID3D12Fence = self
                .host
                .device
                .CreateFence(0, D3D12_FENCE_FLAG_NONE)
                .unwrap();
            self.queue.Signal(&fence, 1).unwrap();
            while fence.GetCompletedValue() < 1 {
                if let Err(e) = self.host.device.GetDeviceRemovedReason() {
                    panic!("device removed: {}", err(e));
                }
                std::thread::yield_now();
            }
        }
    }

    /// A buffer of `words` in the card's memory, and with `view`, a raw SRV of it at the
    /// next heap index (returned).
    pub fn data(
        &mut self,
        list: &ID3D12GraphicsCommandList,
        words: &[u32],
        view: bool,
    ) -> (ID3D12Resource, u32) {
        let mut keep = std::mem::take(&mut self.keep);
        let b = upload(&self.host.device, list, &mut keep, words);
        self.keep = keep;
        self.view_of(b, words.len() as u32, view)
    }

    /// The same in an upload heap: the PC's memory, read by the card over PCIe.
    pub fn data_in_pc(&mut self, words: &[u32], view: bool) -> (ID3D12Resource, u32) {
        let b = buffer(
            &self.host.device,
            D3D12_HEAP_TYPE_UPLOAD,
            (words.len() * 4).max(256) as u64,
            D3D12_RESOURCE_STATE_GENERIC_READ,
        );
        // SAFETY: mapped memory of the buffer's size.
        unsafe {
            let mut p = std::ptr::null_mut();
            b.Map(0, None, Some(&mut p)).unwrap();
            std::ptr::copy_nonoverlapping(words.as_ptr(), p.cast::<u32>(), words.len());
            b.Unmap(0, None);
        }
        self.view_of(b, words.len() as u32, view)
    }

    fn view_of(&mut self, b: ID3D12Resource, words: u32, view: bool) -> (ID3D12Resource, u32) {
        let index = self.views;
        if view {
            self.raw_view(&b, words);
        }
        self.keep.push(b.clone());
        (b, index)
    }

    pub fn raw_view(&mut self, b: &ID3D12Resource, words: u32) -> u32 {
        let index = self.views;
        self.views += 1;
        // SAFETY: a heap slot inside the heap.
        unsafe {
            let mut at = self.heap.GetCPUDescriptorHandleForHeapStart();
            at.ptr += index as usize * self.view_size;
            self.host.device.CreateShaderResourceView(
                b,
                Some(&D3D12_SHADER_RESOURCE_VIEW_DESC {
                    Format: DXGI_FORMAT_R32_TYPELESS,
                    ViewDimension: D3D12_SRV_DIMENSION_BUFFER,
                    Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                    Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                        Buffer: D3D12_BUFFER_SRV {
                            FirstElement: 0,
                            NumElements: words,
                            StructureByteStride: 0,
                            Flags: D3D12_BUFFER_SRV_FLAG_RAW,
                        },
                    },
                }),
                at,
            );
        }
        index
    }

    pub fn cbv(&mut self, b: &ID3D12Resource, bytes: u32) -> u32 {
        let index = self.views;
        self.views += 1;
        // SAFETY: a heap slot inside the heap.
        unsafe {
            let mut at = self.heap.GetCPUDescriptorHandleForHeapStart();
            at.ptr += index as usize * self.view_size;
            self.host.device.CreateConstantBufferView(
                Some(&D3D12_CONSTANT_BUFFER_VIEW_DESC {
                    BufferLocation: b.GetGPUVirtualAddress(),
                    SizeInBytes: bytes,
                }),
                at,
            );
        }
        index
    }

    /// Microseconds of card time per draw of `run` (the median of five lists).
    pub fn time(&mut self, run: &Run) -> f64 {
        let dev = self.host.device.clone();
        let amplify = (run.launch == Launch::Amplified).then(super::amplification_shader);
        let psos = [
            mesh_pso_with(&dev, run.root, amplify.as_deref(), run.ms, &self.ps[0]).unwrap(),
            mesh_pso_with(&dev, run.root, amplify.as_deref(), run.ms, &self.ps[1]).unwrap(),
        ];
        let mut times = Vec::new();
        // SAFETY: plain D3D12 calls with live objects.
        unsafe {
            let freq = self.queue.GetTimestampFrequency().unwrap() as f64;
            let qheap: ID3D12QueryHeap = {
                let mut q: Option<ID3D12QueryHeap> = None;
                dev.CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                        Count: 2 + 4096,
                        NodeMask: 0,
                    },
                    &mut q,
                )
                .unwrap();
                q.unwrap()
            };
            let readback = buffer(
                &dev,
                D3D12_HEAP_TYPE_READBACK,
                16,
                D3D12_RESOURCE_STATE_COPY_DEST,
            );
            let spare = buffer(
                &dev,
                D3D12_HEAP_TYPE_DEFAULT,
                65536,
                D3D12_RESOURCE_STATE_COMMON,
            );
            // indirect records (groups, 1, 1), one a draw, and the clamp's scratch buffer
            let records: Vec<u32> = (0..run.draws).flat_map(|_| [run.groups, 1, 1]).collect();
            let args = buffer(
                &dev,
                D3D12_HEAP_TYPE_UPLOAD,
                (records.len() * 4) as u64,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            {
                let mut p = std::ptr::null_mut();
                args.Map(0, None, Some(&mut p)).unwrap();
                std::ptr::copy_nonoverlapping(records.as_ptr(), p.cast::<u32>(), records.len());
                args.Unmap(0, None);
            }
            let scratch = {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 65536,
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
                    },
                    D3D12_RESOURCE_STATE_COMMON,
                    None,
                    &mut r,
                )
                .unwrap();
                r.unwrap()
            };
            // the budget of the amplification shader: zeros, in unordered access throughout
            let budget = {
                let mut r: Option<ID3D12Resource> = None;
                dev.CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                        Width: 256,
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
                    },
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut r,
                )
                .unwrap();
                r.unwrap()
            };
            let signature: ID3D12CommandSignature = {
                let argument = D3D12_INDIRECT_ARGUMENT_DESC {
                    Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH_MESH,
                    ..Default::default()
                };
                let mut sig: Option<ID3D12CommandSignature> = None;
                dev.CreateCommandSignature(
                    &D3D12_COMMAND_SIGNATURE_DESC {
                        ByteStride: 12,
                        NumArgumentDescs: 1,
                        pArgumentDescs: &argument,
                        NodeMask: 0,
                    },
                    None,
                    &mut sig,
                )
                .unwrap();
                sig.unwrap()
            };
            let clamp_code = compile(CLAMP_CS, "cs_6_6", &[]);
            let clamp_root: ID3D12RootSignature = dev.CreateRootSignature(0, &clamp_code).unwrap();
            let clamp_pso: ID3D12PipelineState = dev
                .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::transmute_copy(&clamp_root),
                    CS: D3D12_SHADER_BYTECODE {
                        pShaderBytecode: clamp_code.as_ptr().cast(),
                        BytecodeLength: clamp_code.len(),
                    },
                    ..Default::default()
                })
                .unwrap();
            let spare_src = buffer(
                &dev,
                D3D12_HEAP_TYPE_UPLOAD,
                65536,
                D3D12_RESOURCE_STATE_GENERIC_READ,
            );
            for _ in 0..6 {
                let alloc: ID3D12CommandAllocator = dev
                    .CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT)
                    .unwrap();
                let list: ID3D12GraphicsCommandList = dev
                    .CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &alloc, &psos[0])
                    .unwrap();
                let list6: ID3D12GraphicsCommandList6 = list.cast().unwrap();
                list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
                list.SetGraphicsRootSignature(run.root);
                let rtv = self.rtv.GetCPUDescriptorHandleForHeapStart();
                let dsv = self.dsv.GetCPUDescriptorHandleForHeapStart();
                list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
                list.ClearDepthStencilView(dsv, D3D12_CLEAR_FLAG_DEPTH, 1.0, 0, None);
                list.RSSetViewports(&[D3D12_VIEWPORT {
                    Width: SIZE as f32,
                    Height: SIZE as f32,
                    MaxDepth: 1.0,
                    ..Default::default()
                }]);
                list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                    right: SIZE as i32,
                    bottom: SIZE as i32,
                    ..Default::default()
                }]);
                for &(param, va) in &run.root_srvs {
                    list.SetGraphicsRootShaderResourceView(param, va);
                }
                list.EndQuery(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                let mut c = run.constants.clone();
                for d in 0..run.draws {
                    if run.alternate {
                        list.SetPipelineState(&psos[(d & 1) as usize]);
                    }
                    if let Some(k) = run.draw_dword {
                        c[k] = d;
                    }
                    if let Some((oc, os)) = run.objects.get(d as usize % run.objects.len().max(1)) {
                        c.clone_from(oc);
                        for &(param, va) in os {
                            list.SetGraphicsRootShaderResourceView(param, va);
                        }
                    }
                    list.SetGraphicsRoot32BitConstants(0, c.len() as u32, c.as_ptr().cast(), 0);
                    let moved = |r: &ID3D12Resource, before, after| D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            Transition: std::mem::ManuallyDrop::new(
                                D3D12_RESOURCE_TRANSITION_BARRIER {
                                    pResource: std::mem::transmute_copy(r),
                                    StateBefore: before,
                                    StateAfter: after,
                                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                },
                            ),
                        },
                        ..Default::default()
                    };
                    let uav_all = D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                pResource: std::mem::ManuallyDrop::new(None),
                            }),
                        },
                        ..Default::default()
                    };
                    // the clamp of records [first, first + n) into the scratch buffer
                    let clamp = |first: u32, n: u32| {
                        list.ResourceBarrier(&[
                            uav_all.clone(),
                            moved(
                                &scratch,
                                D3D12_RESOURCE_STATE_COMMON,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                            ),
                        ]);
                        list.SetComputeRootSignature(&clamp_root);
                        list.SetPipelineState(&clamp_pso);
                        list.SetComputeRoot32BitConstants(0, 2, [first, n].as_ptr().cast(), 0);
                        list.SetComputeRootShaderResourceView(1, args.GetGPUVirtualAddress());
                        list.SetComputeRootUnorderedAccessView(2, scratch.GetGPUVirtualAddress());
                        list.Dispatch(n.div_ceil(64), 1, 1);
                        list.ResourceBarrier(&[
                            uav_all.clone(),
                            moved(
                                &scratch,
                                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                            ),
                        ]);
                        // the graphics pipeline again (the clamp replaced the compute one only,
                        // but the host binds its pipeline after)
                        list.SetPipelineState(
                            &psos[if run.alternate { (d & 1) as usize } else { 0 }],
                        );
                    };
                    match run.launch {
                        Launch::Direct => list6.DispatchMesh(run.groups, 1, 1),
                        Launch::Amplified => {
                            let at = run
                                .amplify_params
                                .expect("the root parameters of the amplification shader");
                            let launch = [super::AMPLIFY_RECORDS, 0, 12, 0];
                            list.SetGraphicsRoot32BitConstants(at, 4, launch.as_ptr().cast(), 0);
                            list.SetGraphicsRootShaderResourceView(
                                at + 1,
                                args.GetGPUVirtualAddress() + u64::from(d) * 12,
                            );
                            list.SetGraphicsRootShaderResourceView(
                                at + 2,
                                args.GetGPUVirtualAddress(),
                            );
                            list.SetGraphicsRootUnorderedAccessView(
                                at + 3,
                                budget.GetGPUVirtualAddress(),
                            );
                            list6.DispatchMesh(1, 1, 1);
                        }
                        Launch::Indirect => {
                            list.ExecuteIndirect(&signature, 1, &args, u64::from(d) * 12, None, 0)
                        }
                        Launch::Clamped => {
                            clamp(d, 1);
                            list.ExecuteIndirect(&signature, 1, &scratch, 0, None, 0);
                            list.ResourceBarrier(&[moved(
                                &scratch,
                                D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                                D3D12_RESOURCE_STATE_COMMON,
                            )]);
                        }
                        Launch::Window(n) => {
                            if d % n == 0 {
                                if d > 0 {
                                    list.ResourceBarrier(&[moved(
                                        &scratch,
                                        D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                                        D3D12_RESOURCE_STATE_COMMON,
                                    )]);
                                }
                                clamp(d, n.min(run.draws - d));
                            }
                            list.ExecuteIndirect(
                                &signature,
                                1,
                                &scratch,
                                u64::from(d % n) * 12,
                                None,
                                0,
                            );
                            if d + 1 == run.draws {
                                list.ResourceBarrier(&[moved(
                                    &scratch,
                                    D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
                                    D3D12_RESOURCE_STATE_COMMON,
                                )]);
                            }
                        }
                    }
                    if run.stamp_each {
                        list.EndQuery(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 2 + (d % 4096));
                    }
                    let moved = |r: &ID3D12Resource, before, after| D3D12_RESOURCE_BARRIER {
                        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                        Anonymous: D3D12_RESOURCE_BARRIER_0 {
                            Transition: std::mem::ManuallyDrop::new(
                                D3D12_RESOURCE_TRANSITION_BARRIER {
                                    pResource: std::mem::transmute_copy(r),
                                    StateBefore: before,
                                    StateAfter: after,
                                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                                },
                            ),
                        },
                        ..Default::default()
                    };
                    let (common, dest) =
                        (D3D12_RESOURCE_STATE_COMMON, D3D12_RESOURCE_STATE_COPY_DEST);
                    match run.between {
                        Between::Nothing => {}
                        Between::UavAll => list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                            Type: D3D12_RESOURCE_BARRIER_TYPE_UAV,
                            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                                UAV: std::mem::ManuallyDrop::new(D3D12_RESOURCE_UAV_BARRIER {
                                    pResource: std::mem::ManuallyDrop::new(None),
                                }),
                            },
                            ..Default::default()
                        }]),
                        Between::Transition => {
                            list.ResourceBarrier(&[moved(&spare, common, dest)]);
                            list.ResourceBarrier(&[moved(&spare, dest, common)]);
                        }
                        Between::Copy | Between::CopyRead => {
                            let to = if run.between == Between::Copy {
                                &spare
                            } else {
                                run.read.unwrap_or(&spare)
                            };
                            list.ResourceBarrier(&[moved(to, common, dest)]);
                            list.CopyBufferRegion(to, 0, &spare_src, 0, 256);
                            list.ResourceBarrier(&[moved(to, dest, common)]);
                        }
                        Between::Heaps => list.SetDescriptorHeaps(&[Some(self.heap.clone())]),
                        Between::RootSignature => {
                            list.SetGraphicsRootSignature(run.root);
                            for &(param, va) in &run.root_srvs {
                                list.SetGraphicsRootShaderResourceView(param, va);
                            }
                        }
                        Between::Targets => {
                            list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv))
                        }
                    }
                }
                list.EndQuery(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                list.ResolveQueryData(&qheap, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &readback, 0);
                list.Close().unwrap();
                self.queue
                    .ExecuteCommandLists(&[Some(list.cast().unwrap())]);
                self.wait();
                let mut p = std::ptr::null_mut();
                readback.Map(0, None, Some(&mut p)).unwrap();
                let t = std::slice::from_raw_parts(p.cast::<u64>(), 2);
                times.push((t[1] - t[0]) as f64 / freq * 1e6 / f64::from(run.draws));
                readback.Unmap(0, None);
            }
        }
        times.remove(0);
        times.sort_by(f64::total_cmp);
        times[times.len() / 2]
    }

    /// The root signature the translator's shaders take (`Host::root_signature_for`) with the parameters of the host's amplification shader after
    /// (4 constants at `b0`, SRVs `t0` and `t1` and UAV `u0` in space 1, amplification stage
    /// only), as `victory-gpu` makes them; also the index of the first of them.
    pub fn host_root_amplified(
        &self,
        ms: &super::HostShader,
        ps: &super::HostShader,
    ) -> (ID3D12RootSignature, u32) {
        let stages = [
            (ms, D3D12_SHADER_VISIBILITY_MESH),
            (ps, D3D12_SHADER_VISIBILITY_PIXEL),
        ];
        let view = |kind, register| D3D12_ROOT_PARAMETER1 {
            ParameterType: kind,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                    ShaderRegister: register,
                    RegisterSpace: 1,
                    Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        };
        let extra = [
            D3D12_ROOT_PARAMETER1 {
                ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
                Anonymous: D3D12_ROOT_PARAMETER1_0 {
                    Constants: D3D12_ROOT_CONSTANTS {
                        ShaderRegister: 0,
                        RegisterSpace: 0,
                        Num32BitValues: super::AMPLIFY_CONSTANTS,
                    },
                },
                ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
            },
            view(D3D12_ROOT_PARAMETER_TYPE_SRV, 0),
            view(D3D12_ROOT_PARAMETER_TYPE_SRV, 1),
            view(D3D12_ROOT_PARAMETER_TYPE_UAV, 0),
        ];
        let views = ms.root_cbvs.len() + ms.root_srvs.len() + ms.root_uavs.len();
        let first = (stages.len() + views) as u32;
        (
            self.host.root_signature_with(&stages, &extra).unwrap(),
            first,
        )
    }

    /// Prints how many pixels of the depth target the last run wrote (a check that its
    /// triangles were drawn at all).
    pub fn check_target(&mut self) -> u64 {
        let dev = self.host.device.clone();
        let size = u64::from(SIZE) * u64::from(SIZE) * 4;
        let readback = buffer(
            &dev,
            D3D12_HEAP_TYPE_READBACK,
            size,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        let ds = self.ds.clone();
        let transition = |before, after| D3D12_RESOURCE_BARRIER {
            Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
            Anonymous: D3D12_RESOURCE_BARRIER_0 {
                Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                    // SAFETY: a borrowed interface pointer the barrier does not release.
                    pResource: unsafe { std::mem::transmute_copy(&ds) },
                    StateBefore: before,
                    StateAfter: after,
                    Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                }),
            },
            ..Default::default()
        };
        // SAFETY: recorded on an open list; the resources outlive the wait.
        self.once(|_, list| unsafe {
            list.ResourceBarrier(&[transition(
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )]);
            let dst = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&readback),
                Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                        Offset: 0,
                        Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                            Format: DXGI_FORMAT_R32_TYPELESS,
                            Width: SIZE,
                            Height: SIZE,
                            Depth: 1,
                            RowPitch: SIZE * 4,
                        },
                    },
                },
            };
            let src = D3D12_TEXTURE_COPY_LOCATION {
                pResource: std::mem::transmute_copy(&ds),
                Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
                Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                    SubresourceIndex: 0,
                },
            };
            list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
            list.ResourceBarrier(&[transition(
                D3D12_RESOURCE_STATE_COPY_SOURCE,
                D3D12_RESOURCE_STATE_DEPTH_WRITE,
            )]);
        });
        // SAFETY: a mapped readback buffer of `size` bytes.
        unsafe {
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let depth = std::slice::from_raw_parts(p.cast::<f32>(), (size / 4) as usize);
            let drawn = depth.iter().filter(|&&d| d < 1.0).count();
            use std::hash::{Hash, Hasher};
            let mut h = std::collections::hash_map::DefaultHasher::new();
            for d in depth {
                d.to_bits().hash(&mut h);
            }
            readback.Unmap(0, None);
            println!("  depth target: {drawn} pixels drawn");
            h.finish()
        }
    }

    /// A root signature of 16 root constants at `b0` and directly indexed heaps.
    pub fn simple_root(&self) -> ID3D12RootSignature {
        let params = [D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: 16,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_ALL,
        }];
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: 1,
                    pParameters: params.as_ptr(),
                    Flags: D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED,
                    ..Default::default()
                },
            },
        };
        // SAFETY: the description points at live arrays.
        unsafe {
            let mut blob = None;
            D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
            let blob: ID3DBlob = blob.unwrap();
            let bytes = std::slice::from_raw_parts(
                blob.GetBufferPointer().cast::<u8>(),
                blob.GetBufferSize(),
            );
            self.host.device.CreateRootSignature(0, bytes).unwrap()
        }
    }
}

/// A mesh shader of 128 threads, 128 vertices and primitives, whose vertex position takes
/// `HOPS` dependent loads from a large buffer (a chain through random words), and `LOADS`
/// independent ones; `UNIFORM` makes every lane of a group follow the same chain. Its
/// triangles are a pixel each.
const MS: &str = r#"
cbuffer C : register(b0) { uint draw; uint mask; uint verts; uint prims; uint view; };
[outputtopology("triangle")]
[numthreads(128, 1, 1)]
void main(uint t : SV_GroupThreadID, uint g : SV_GroupID,
          out vertices float4 pos[MAXV] : SV_Position, out indices uint3 tri[MAXP]) {
    SetMeshOutputCounts(verts, prims);
    ByteAddressBuffer b = ResourceDescriptorHeap[view];
#if UNIFORM
    uint a = ((draw * 7919u + g * 104729u) * 64u) & mask;
#else
    uint a = ((draw * 7919u + g * 104729u) * 64u + t * 4u) & mask;
#endif
    [unroll] for (uint i = 0; i < HOPS; i++) { a = b.Load(a) & mask; }
    float s = 0;
    [unroll] for (uint j = 0; j < LOADS; j++) { s += asfloat(b.Load((a + j * 4096u) & mask) & 0x3fffffffu); }
    float x = (float(t % 16u) + float(g % 64u) * 16.0) / 512.0 - 1.0 + s * 1e-30;
    float y = (float(t / 16u) + float(draw % 128u) * 8.0) / 512.0 - 1.0 + float(a & 1u) * 1e-30;
    if (t < verts) pos[t] = float4(x, y, 0.5, 1);
    if (t < prims) tri[t] = uint3(t, min(t + 1u, verts - 1u), min(t + 16u, verts - 1u));
}
"#;

fn ms(hops: u32, loads: u32, uniform: bool, maxv: u32, maxp: u32) -> Vec<u8> {
    compile(
        MS,
        "ms_6_6",
        &[
            format!("HOPS={hops}"),
            format!("LOADS={loads}"),
            format!("UNIFORM={}", u32::from(uniform)),
            format!("MAXV={maxv}"),
            format!("MAXP={maxp}"),
        ],
    )
}

#[test]
#[ignore = "needs a card and dxc"]
fn mesh_bench_card_basics() {
    let Some(mut b) = Bench::new() else { return };
    // 256 MiB of words, each the byte offset of a random word: chains miss the caches.
    let words = 64 << 20;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let data: Vec<u32> = (0..words)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 20) as u32 % words) * 4
        })
        .collect();
    b.once(|b, list| {
        b.data(list, &data, true);
    });
    let root = b.simple_root();
    let mask = (words * 4 - 1) & !3;
    let draws = 400;
    println!("per-draw card time, {draws} draws a list (us/draw):");
    for (hops, loads, uniform) in [
        (0, 0, false),
        (1, 0, false),
        (2, 0, false),
        (4, 0, false),
        (8, 0, false),
        (4, 0, true),
        (2, 24, false),
        (2, 24, true),
    ] {
        for &(maxv, maxp, v, p) in &[
            (128, 128, 128, 128),
            (128, 128, 64, 64),
            (64, 64, 64, 64),
            (256, 256, 128, 128),
        ] {
            let code = ms(hops, loads, uniform, maxv, maxp);
            let mut line = format!(
                "hops {hops} loads {loads:2} uniform {} max {maxv}/{maxp} counts {v}/{p}:",
                u8::from(uniform)
            );
            for groups in [1, 3, 16, 128] {
                for alternate in [false, true] {
                    let run = Run {
                        ms: &code,
                        root: &root,
                        constants: vec![0, mask, v, p, 0],
                        draw_dword: Some(0),
                        root_srvs: Vec::new(),
                        objects: Vec::new(),
                        groups,
                        draws,
                        alternate,
                        stamp_each: false,
                        between: Between::Nothing,
                        read: None,
                        launch: Launch::Direct,
                        amplify_params: None,
                    };
                    let us = b.time(&run);
                    line.push_str(&format!(
                        " g{groups}{}={us:.2}",
                        if alternate { "alt" } else { "" }
                    ));
                }
            }
            println!("{line}");
        }
    }
}

/// What a timestamp after every draw does to the time per draw, and what reading the
/// vertex data from the PC's memory (over PCIe) does.
#[test]
#[ignore = "needs a card and dxc"]
fn mesh_bench_stamps_and_pc_memory() {
    let Some(mut b) = Bench::new() else { return };
    let words = 16 << 20;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let data: Vec<u32> = (0..words)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 20) as u32 % words) * 4
        })
        .collect();
    let mut card = 0;
    b.once(|b, list| {
        card = b.data(list, &data, true).1;
    });
    let pc = b.data_in_pc(&data, true).1;
    let root = b.simple_root();
    let mask = (words * 4 - 1) & !3;
    let draws = 400;
    println!(
        "per-draw card time, {draws} draws a list (us/draw); stamp = a timestamp after each draw:"
    );
    for (where_, view) in [("card", card), ("pc", pc)] {
        for (hops, loads, uniform) in [
            (0, 0, false),
            (2, 0, false),
            (4, 0, false),
            (8, 0, false),
            (4, 0, true),
            (2, 24, true),
        ] {
            let code = ms(hops, loads, uniform, 128, 128);
            let mut line = format!(
                "{where_:4} hops {hops} loads {loads:2} uniform {}:",
                u8::from(uniform)
            );
            for groups in [1, 3, 16, 128] {
                for (stamp_each, barrier_each) in [(false, false), (true, false), (true, true)] {
                    let between = if barrier_each {
                        Between::UavAll
                    } else {
                        Between::Nothing
                    };
                    let run = Run {
                        ms: &code,
                        root: &root,
                        constants: vec![0, mask, 128, 128, view],
                        draw_dword: Some(0),
                        root_srvs: Vec::new(),
                        objects: Vec::new(),
                        groups,
                        draws,
                        alternate: false,
                        stamp_each,
                        between,
                        read: None,
                        launch: Launch::Direct,
                        amplify_params: None,
                    };
                    let us = b.time(&run);
                    let tag = match (stamp_each, barrier_each) {
                        (false, _) => "",
                        (true, false) => "stamp",
                        (true, true) => "drain",
                    };
                    line.push_str(&format!(" g{groups}{tag}={us:.2}"));
                }
            }
            println!("{line}");
        }
    }
}

/// The title's `MS_VS_CascadeShadows_draw` (`VICTORY_TEST_MESH_CONTAINER`, a container file of
/// it; db875f50a166bd91) as translated, on meshlet data made for it: 256 meshlets of `nv`
/// vertices and `np` triangles of a lattice, each draw `groups` meshlets.
#[test]
#[ignore = "needs a card and VICTORY_TEST_MESH_CONTAINER"]
fn mesh_bench_cascade_shadows_draw() {
    let Ok(path) = std::env::var("VICTORY_TEST_MESH_CONTAINER") else {
        eprintln!("VICTORY_TEST_MESH_CONTAINER unset; nothing to do");
        return;
    };
    super::set_host_max_wave(32);
    let bytes = std::fs::read(&path).unwrap();
    let shader = super::translate(&bytes).unwrap();
    // the same with a payload, for the amplification shader in front (`VICTORY_MESH_AMPLIFY`)
    // SAFETY: tests that read the switch run on one thread (`--test-threads 1`).
    unsafe { std::env::set_var("VICTORY_MESH_AMPLIFY", "1") };
    let with_payload = super::translate(&bytes).unwrap();
    unsafe { std::env::remove_var("VICTORY_MESH_AMPLIFY") };
    {
        let Some(b) = Bench::new() else { return };
        let ps = super::host::stub_ps();
        let (root, _) = b.host_root_amplified(&shader, &ps);
        let amplify = super::amplification_shader();
        for (what, ms, amp) in [
            ("payload 0, no amplification", &shader.dxil, None),
            ("payload 4, no amplification", &with_payload.dxil, None),
            (
                "payload 4, amplified",
                &with_payload.dxil,
                Some(&amplify[..]),
            ),
            ("payload 0, amplified", &shader.dxil, Some(&amplify[..])),
        ] {
            let made = mesh_pso_with(b.device(), &root, amp, ms, &b.ps[0]);
            println!(
                "pipeline with {what}: {}",
                made.as_ref().map_or_else(
                    |e| e.chars().take(300).collect::<String>(),
                    |_| "made".into()
                )
            );
        }
    }
    println!(
        "user data {} dwords, root srvs {:?}, cbvs {:?}, indexed {}",
        shader.user_data_dwords(),
        shader.root_srvs,
        shader.root_cbvs,
        shader.root_indexed_views.len()
    );
    let Some(mut b) = Bench::new() else { return };
    let ps = super::host::stub_ps();
    let (root, amplify_at) = b.host_root_amplified(&shader, &ps);
    for (nv, np) in [(128u32, 128u32), (64, 124), (64, 64)] {
        let meshlets = 256u32;
        // The lattice: meshlet m is a 16 x (nv / 16) patch of it.
        let rows = nv / 16;
        let mut verts = Vec::new();
        for m in 0..meshlets {
            let (mx, my) = (m % 16, m / 16);
            for v in 0..nv {
                let (x, y) = (mx * 16 + v % 16, my * rows + v / 16);
                // a vertex every 2 pixels of a 2048 target over [-1, 1]
                verts.push(((x * 2) << 12, (y * 2) << 12));
            }
        }
        // meshlet data: count, 0, 16-byte records from byte 12, then per meshlet 128
        // triangle dwords and 128 halfword vertex indices
        let base = 3 + meshlets * 4;
        let mut md = vec![0u32; (base + meshlets * 192 + 64) as usize];
        md[0] = meshlets;
        for m in 0..meshlets {
            let poff = m * 192;
            let voff = (m * 192 + 128) * 2;
            let a = (nv - 1)
                | ((np - 1) << 7)
                | (((voff >> 16) & 1) << 14)
                | (((poff >> 16) & 1) << 15);
            md[(3 + m * 4) as usize] = a;
            md[(3 + m * 4 + 1) as usize] = ((poff & 0xffff) << 16) | (voff & 0xffff);
            for p in 0..np {
                let q = p / 2;
                let (qx, qy) = (q % 15, (q / 15) % rows.saturating_sub(1).max(1));
                let v0 = qy * 16 + qx;
                let (i0, i1, i2) = if p % 2 == 0 {
                    (v0, v0 + 1, v0 + 16)
                } else {
                    (v0 + 1, v0 + 17, v0 + 16)
                };
                let (i0, i1, i2) = (i0.min(nv - 1), i1.min(nv - 1), i2.min(nv - 1));
                md[(base + poff + p) as usize] = i0 | (i1 << 7) | (i2 << 14);
            }
            for v in 0..nv {
                // halfword `voff + v` of the area
                let at = (voff + v) as usize;
                let word = base as usize + at / 2;
                let index = m * nv + v;
                md[word] |= (index & 0xffff) << (16 * (at % 2));
            }
        }
        // vertex data: a header, then 8 bytes a vertex from byte 128
        let mut vd = vec![0u32; 32 + verts.len() * 2];
        vd[0] = 64 | (128 << 7);
        vd[24] = (-1.0f32).to_bits();
        vd[25] = (-1.0f32).to_bits();
        vd[28] = (2.0f32 / 16_777_216.0).to_bits();
        vd[29] = (2.0f32 / 16_777_216.0).to_bits();
        for (i, &(x, y)) in verts.iter().enumerate() {
            vd[32 + i * 2] = (x << 8) | (y & 255);
            vd[32 + i * 2 + 1] = (y >> 8) & 0xffff;
        }
        let mut object = vec![0u32; 64];
        // the program takes the rows as z, y, x: x to x, y to y, z to z
        object[8] = 1.0f32.to_bits();
        object[5] = 1.0f32.to_bits();
        object[2] = 1.0f32.to_bits();
        let mut camera = vec![0u32; 16384];
        for (reg, row) in [
            (22, [1.0f32, 0.0, 0.0, 0.0]),
            (23, [0.0, 1.0, 0.0, 0.0]),
            (24, [0.0; 4]),
            (25, [0.0, 0.0, 0.5, 1.0]),
        ] {
            for (k, f) in row.iter().enumerate() {
                camera[reg * 4 + k] = f.to_bits();
            }
        }
        let strides = vec![0u32; 1 << 16];
        let mut sbuf = None;
        b.once(|b, list| sbuf = Some(b.data(list, &strides, false).0));
        let sbuf = sbuf.unwrap();
        let sva = unsafe { sbuf.GetGPUVirtualAddress() };
        // `copies` objects of their own buffers (256 of them are past the card's L2), in
        // the card's memory or the PC's
        let objects_of =
            |b: &mut Bench, copies: usize, pc: bool| -> Vec<(Vec<u32>, Vec<(u32, u64)>)> {
                let mut out = Vec::new();
                for _ in 0..copies {
                    let (mut h, mut vbuf) = (0, None);
                    if pc {
                        h = b.data_in_pc(&md, true).1;
                        b.data_in_pc(&object, true);
                        let (cb, _) = b.data_in_pc(&camera, false);
                        b.cbv(&cb, 65536);
                        vbuf = Some(b.data_in_pc(&vd, false).0);
                    } else {
                        b.once(|b, list| {
                            h = b.data(list, &md, true).1;
                            b.data(list, &object, true);
                            let (cb, _) = b.data(list, &camera, false);
                            b.cbv(&cb, 65536);
                            vbuf = Some(b.data(list, &vd, false).0);
                        });
                    }
                    let vbuf = vbuf.unwrap();
                    let mut constants = vec![0u32; shader.user_data_dwords().max(1) as usize];
                    constants[9] = h * 32;
                    // root parameters: one constants parameter per stage, then views by slot
                    let mut views: Vec<u8> = shader.root_srvs.clone();
                    views.sort_unstable();
                    // SAFETY: live buffers.
                    let vva = unsafe { vbuf.GetGPUVirtualAddress() };
                    let srvs = views
                        .iter()
                        .enumerate()
                        .map(|(i, &slot)| (2 + i as u32, if slot == 2 { vva } else { sva }))
                        .collect();
                    out.push((constants, srvs));
                }
                out
            };
        let mut line = format!("translated cascade draw, meshlets of {nv}/{np}:");
        let mut hashes = Vec::new();
        for (label, copies, pc, barrier_each, launch) in [
            ("one object", 1, false, false, Launch::Direct),
            ("256 objects", 256, false, false, Launch::Direct),
            ("256 objects in PC memory", 256, true, false, Launch::Direct),
            (
                "one object, a barrier after each draw",
                1,
                false,
                true,
                Launch::Direct,
            ),
            (
                "256 objects, a barrier after each draw",
                256,
                false,
                true,
                Launch::Direct,
            ),
            (
                "256 objects in PC memory, a barrier after each draw",
                256,
                true,
                true,
                Launch::Direct,
            ),
            (
                "256 objects, indirect, no clamp",
                256,
                false,
                false,
                Launch::Indirect,
            ),
            (
                "256 objects, indirect, clamped before each (the host now)",
                256,
                false,
                false,
                Launch::Clamped,
            ),
            (
                "256 objects in PC memory, indirect, clamped before each",
                256,
                true,
                false,
                Launch::Clamped,
            ),
            (
                "256 objects, indirect, clamped 16 at a time",
                256,
                false,
                false,
                Launch::Window(16),
            ),
            (
                "256 objects, indirect, clamped 64 at a time",
                256,
                false,
                false,
                Launch::Window(64),
            ),
            (
                "256 objects, clamped by an amplification shader",
                256,
                false,
                false,
                Launch::Amplified,
            ),
            (
                "256 objects in PC memory, clamped by an amplification shader",
                256,
                true,
                false,
                Launch::Amplified,
            ),
        ] {
            if b.views + 4 * copies as u32 > 4000 {
                b.views = 0;
                b.keep.clear();
            }
            let objects = objects_of(&mut b, copies, pc);
            line.push_str(&format!(
                "
  {label}:"
            ));
            for groups in [1, 3, 16, 128, 256] {
                let run = Run {
                    ms: if launch == Launch::Amplified {
                        &with_payload.dxil
                    } else {
                        &shader.dxil
                    },
                    root: &root,
                    constants: objects[0].0.clone(),
                    draw_dword: None,
                    root_srvs: objects[0].1.clone(),
                    objects: objects.clone(),
                    groups,
                    draws: 400,
                    alternate: false,
                    stamp_each: true,
                    between: if barrier_each {
                        Between::UavAll
                    } else {
                        Between::Nothing
                    },
                    read: None,
                    launch,
                    amplify_params: Some(amplify_at),
                };
                let us = b.time(&run);
                line.push_str(&format!(" g{groups}={us:.2}"));
                // what the last list drew, by launch (the depth target is cleared at each
                // list's start): every launch draws the same
                if groups == 16 {
                    hashes.push((label, b.check_target()));
                }
            }
        }
        let first = hashes[0].1;
        for (label, h) in &hashes {
            assert_eq!(
                *h, first,
                "{label}: another depth image than a direct dispatch's"
            );
        }
        line.push_str(&format!(
            "\n  depth image the same under all {} launches",
            hashes.len()
        ));
        println!("{line} (us/draw)");
    }
}

/// What each host command recorded after every draw costs the card per draw (the draws
/// read random words, so a draw's latency shows where the card drains).
#[test]
#[ignore = "needs a card and dxc"]
fn mesh_bench_commands_between_draws() {
    let Some(mut b) = Bench::new() else { return };
    let words = 16 << 20;
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let data: Vec<u32> = (0..words)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 20) as u32 % words) * 4
        })
        .collect();
    let mut buf = None;
    b.once(|b, list| buf = Some(b.data(list, &data, true)));
    let (buf, view) = buf.unwrap();
    let root = b.simple_root();
    let mask = (words * 4 - 1) & !3;
    let code = ms(2, 0, false, 128, 128);
    println!("hops 2, 400 draws, a timestamp after each; us/draw by what follows each draw:");
    for between in [
        Between::Nothing,
        Between::UavAll,
        Between::Transition,
        Between::Copy,
        Between::CopyRead,
        Between::Heaps,
        Between::RootSignature,
        Between::Targets,
    ] {
        let mut line = format!("{between:?}:");
        for groups in [1, 3, 16] {
            let run = Run {
                ms: &code,
                root: &root,
                constants: vec![0, mask, 128, 128, view],
                draw_dword: Some(0),
                root_srvs: Vec::new(),
                objects: Vec::new(),
                groups,
                draws: 400,
                alternate: false,
                stamp_each: true,
                between,
                read: Some(&buf),
                launch: Launch::Direct,
                amplify_params: None,
            };
            let us = b.time(&run);
            line.push_str(&format!(" g{groups}={us:.2}"));
        }
        println!("{line}");
    }
}

/// The host's amplification shader launches what `victory-gpu`'s clamp would leave in a
/// record: the groups a direct count or a record asks for, at most 65,535, none for records
/// at or past a count, none once the list's budget (2^27 threads, a mesh group as 128) is
/// spent. A mesh shader of one thread counts the groups that ran.
#[test]
#[ignore = "needs a card and dxc"]
fn mesh_bench_amplification_launches_what_the_clamp_would() {
    const COUNTER: &str = r#"
struct Payload { uint unused; };
RWByteAddressBuffer hits : register(u0, space2);
[outputtopology("triangle")]
[numthreads(1, 1, 1)]
void main(uint g : SV_GroupID, in payload Payload p, out vertices float4 v[1] : SV_Position, out indices uint3 t[1]) {
    SetMeshOutputCounts(1, 1);
    v[0] = float4(2, 2, 0, 1);
    t[0] = uint3(0, 0, 0);
    uint was;
    hits.InterlockedAdd(0, 1, was);
}
"#;
    let Some(mut b) = Bench::new() else { return };
    let dev = b.device().clone();
    let ms = compile(COUNTER, "ms_6_6", &[]);
    let amplify = super::amplification_shader();
    let view = |kind, register, space, vis| D3D12_ROOT_PARAMETER1 {
        ParameterType: kind,
        Anonymous: D3D12_ROOT_PARAMETER1_0 {
            Descriptor: D3D12_ROOT_DESCRIPTOR1 {
                ShaderRegister: register,
                RegisterSpace: space,
                Flags: D3D12_ROOT_DESCRIPTOR_FLAG_NONE,
            },
        },
        ShaderVisibility: vis,
    };
    let params = [
        D3D12_ROOT_PARAMETER1 {
            ParameterType: D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS,
            Anonymous: D3D12_ROOT_PARAMETER1_0 {
                Constants: D3D12_ROOT_CONSTANTS {
                    ShaderRegister: 0,
                    RegisterSpace: 0,
                    Num32BitValues: super::AMPLIFY_CONSTANTS,
                },
            },
            ShaderVisibility: D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        },
        view(
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            0,
            1,
            D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        ),
        view(
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            1,
            1,
            D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        ),
        view(
            D3D12_ROOT_PARAMETER_TYPE_UAV,
            0,
            1,
            D3D12_SHADER_VISIBILITY_AMPLIFICATION,
        ),
        view(
            D3D12_ROOT_PARAMETER_TYPE_UAV,
            0,
            2,
            D3D12_SHADER_VISIBILITY_MESH,
        ),
    ];
    // SAFETY: the description points at live arrays.
    let root: ID3D12RootSignature = unsafe {
        let desc = D3D12_VERSIONED_ROOT_SIGNATURE_DESC {
            Version: D3D_ROOT_SIGNATURE_VERSION_1_1,
            Anonymous: D3D12_VERSIONED_ROOT_SIGNATURE_DESC_0 {
                Desc_1_1: D3D12_ROOT_SIGNATURE_DESC1 {
                    NumParameters: params.len() as u32,
                    pParameters: params.as_ptr(),
                    ..Default::default()
                },
            },
        };
        let mut blob = None;
        D3D12SerializeVersionedRootSignature(&desc, &mut blob, None).unwrap();
        let blob: ID3DBlob = blob.unwrap();
        let bytes =
            std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        dev.CreateRootSignature(0, bytes).unwrap()
    };
    let pso = mesh_pso_with(&dev, &root, Some(&amplify), &ms, &b.ps[0]).unwrap();
    let uav = |size: u64| {
        let mut r: Option<ID3D12Resource> = None;
        // SAFETY: a plain buffer.
        unsafe {
            dev.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: size,
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
                },
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut r,
            )
            .unwrap();
        }
        r.unwrap()
    };
    // (what, flags, direct, records as (x, y, z), count, AS groups) -> mesh groups expected
    let cases: Vec<(&str, u32, u32, Vec<[u32; 3]>, u32, u32, u64)> = vec![
        ("direct 7", 0, 7, vec![], 0, 1, 7),
        (
            "direct 70,000 (cut to 65,535)",
            0,
            70_000,
            vec![],
            0,
            1,
            65_535,
        ),
        (
            "a record of 5",
            super::AMPLIFY_RECORDS,
            0,
            vec![[5, 1, 1]],
            0,
            1,
            5,
        ),
        (
            "a record of 70,000 x 3 x 2 (cut to 65,535 x 1 x 1)",
            super::AMPLIFY_RECORDS,
            0,
            vec![[70_000, 3, 2]],
            0,
            1,
            65_535,
        ),
        (
            "a record of 9 x 0 (empty)",
            super::AMPLIFY_RECORDS,
            0,
            vec![[9, 0, 1]],
            0,
            1,
            0,
        ),
        (
            "three records",
            super::AMPLIFY_RECORDS,
            0,
            vec![[1, 1, 1], [2, 1, 1], [3, 1, 1]],
            0,
            3,
            6,
        ),
        (
            "three records, a count of 2",
            super::AMPLIFY_RECORDS | super::AMPLIFY_COUNTED,
            0,
            vec![[1, 1, 1], [2, 1, 1], [3, 1, 1]],
            2,
            3,
            3,
        ),
        // 2^27 threads are 1,048,576 groups: 16 records of 65,535 fit, the 17th not
        (
            "17 records of 65,535 (the budget runs out)",
            super::AMPLIFY_RECORDS,
            0,
            vec![[65_535, 1, 1]; 17],
            0,
            17,
            16 * 65_535,
        ),
    ];
    for (what, flags, direct, records, count, groups, expected) in cases {
        let words: Vec<u32> = records.iter().flatten().copied().chain([0; 3]).collect();
        let (mut args, mut counts) = (None, None);
        b.once(|b, list| {
            args = Some(b.data(list, &words, false).0);
            counts = Some(b.data(list, &[count, 0, 0, 0], false).0);
        });
        let (args, counts) = (args.unwrap(), counts.unwrap());
        let (budget, hits) = (uav(256), uav(256));
        let readback = buffer(
            &dev,
            D3D12_HEAP_TYPE_READBACK,
            256,
            D3D12_RESOURCE_STATE_COPY_DEST,
        );
        // SAFETY: live heaps.
        let (rtv, dsv) = unsafe {
            (
                b.rtv.GetCPUDescriptorHandleForHeapStart(),
                b.dsv.GetCPUDescriptorHandleForHeapStart(),
            )
        };
        // SAFETY: recorded on an open list; the buffers outlive the wait.
        b.once(|_, list| unsafe {
            let list6: ID3D12GraphicsCommandList6 = list.cast().unwrap();
            list.SetGraphicsRootSignature(&root);
            list.SetPipelineState(&pso);
            list.OMSetRenderTargets(1, Some(&rtv), false, Some(&dsv));
            list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: SIZE as f32,
                Height: SIZE as f32,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                right: SIZE as i32,
                bottom: SIZE as i32,
                ..Default::default()
            }]);
            let launch = [flags, direct, 12, 0];
            list.SetGraphicsRoot32BitConstants(0, 4, launch.as_ptr().cast(), 0);
            list.SetGraphicsRootShaderResourceView(1, args.GetGPUVirtualAddress());
            list.SetGraphicsRootShaderResourceView(2, counts.GetGPUVirtualAddress());
            list.SetGraphicsRootUnorderedAccessView(3, budget.GetGPUVirtualAddress());
            list.SetGraphicsRootUnorderedAccessView(4, hits.GetGPUVirtualAddress());
            list6.DispatchMesh(groups, 1, 1);
            list.ResourceBarrier(&[D3D12_RESOURCE_BARRIER {
                Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
                Anonymous: D3D12_RESOURCE_BARRIER_0 {
                    Transition: std::mem::ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                        pResource: std::mem::transmute_copy(&hits),
                        StateBefore: D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                        StateAfter: D3D12_RESOURCE_STATE_COPY_SOURCE,
                        Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                    }),
                },
                ..Default::default()
            }]);
            list.CopyBufferRegion(&readback, 0, &hits, 0, 4);
        });
        // SAFETY: a mapped readback buffer of 256 bytes.
        let ran = unsafe {
            let mut p = std::ptr::null_mut();
            readback.Map(0, None, Some(&mut p)).unwrap();
            let n = p.cast::<u32>().read();
            readback.Unmap(0, None);
            u64::from(n)
        };
        println!("{what}: {ran} mesh groups ran");
        assert_eq!(ran, expected, "{what}");
    }
}

/// The title's mesh programs of `VICTORY_TEST_MESH_CONTAINERS` (`;`-separated container
/// files) translated for the amplification stage in front (`VICTORY_MESH_AMPLIFY=1`) and
/// without: `dxc -dumpbin` of both may differ only in the payload size of the mesh
/// shader's metadata (0 against `MESH_PAYLOAD`). The mesh groups read `groupId` alone for
/// their place in the draw, which the stage's `DispatchMesh(n)` numbers 0..n as a direct
/// `DispatchMesh(n)` does (the launch tests above, bit for bit on the card): what such a
/// program draws cannot depend on the stage. Checked for the programs that went missing
/// in amplified runs (2026-11-25): 4ef35022ff26c5bb (MS_VS_TransformAlphaPrepass, the
/// window's wire mesh) and 6937838f6f0a9992 (MS_VS_TransformD).
#[test]
#[ignore = "needs VICTORY_TEST_MESH_CONTAINERS and dxc"]
fn mesh_title_programs_amplified_differ_only_in_payload() {
    let Ok(list) = std::env::var("VICTORY_TEST_MESH_CONTAINERS") else {
        eprintln!("VICTORY_TEST_MESH_CONTAINERS unset; nothing to do");
        return;
    };
    super::set_host_max_wave(32);
    let dir = std::env::temp_dir().join(format!("victory-mesh-payload-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let dump = |name: &str, dxil: &[u8]| -> Vec<String> {
        let at = dir.join(name);
        std::fs::write(&at, dxil).unwrap();
        let o = std::process::Command::new(dxc()).arg("-dumpbin").arg(&at).output().expect("dxc");
        assert!(o.status.success(), "dxc -dumpbin: {}", String::from_utf8_lossy(&o.stderr));
        String::from_utf8_lossy(&o.stdout).lines().map(str::to_owned).collect()
    };
    for path in list.split(';').map(str::trim).filter(|p| !p.is_empty()) {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let plain = super::translate(&bytes).unwrap();
        // SAFETY: tests that read the switch run on one thread (`--test-threads 1`).
        unsafe { std::env::set_var("VICTORY_MESH_AMPLIFY", "1") };
        let amplified = super::translate(&bytes);
        unsafe { std::env::remove_var("VICTORY_MESH_AMPLIFY") };
        let amplified = amplified.unwrap();
        assert_eq!(plain.stage, crate::HwStage::Ms, "{path}: not a mesh program");
        assert_eq!(
            (&plain.user_data, &plain.root_srvs, &plain.root_cbvs, &plain.root_uavs),
            (&amplified.user_data, &amplified.root_srvs, &amplified.root_cbvs, &amplified.root_uavs),
            "{path}: other bindings"
        );
        let (a, b) = (dump("plain.dxil", &plain.dxil), dump("amplified.dxil", &amplified.dxil));
        assert_eq!(a.len(), b.len(), "{path}: listings of other lengths");
        let differ: Vec<(&String, &String)> = a.iter().zip(&b).filter(|(x, y)| x != y).collect();
        let payload = format!(", i32 {}}}", super::MESH_PAYLOAD);
        for (x, y) in &differ {
            assert!(
                x.strip_suffix(", i32 0}").is_some_and(|x| y.strip_suffix(payload.as_str()) == Some(x)),
                "{path}: the translations differ in more than the payload:\n  {x}\n  {y}"
            );
        }
        assert_eq!(differ.len(), 1, "{path}: the payload size not found in the mesh metadata");
        let groups = a.iter().filter(|l| l.contains("@dx.op.groupId.i32(i32 94")).count();
        println!(
            "{path}: the same program but its payload ({} vs {}); {groups} reads of the group id",
            differ[0].0.trim(),
            differ[0].1.trim()
        );
    }
}
