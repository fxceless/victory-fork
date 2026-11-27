//! The console's second display plane, blended over the first at present.
//!
//! With the Series S profile the title presents two planes: the game at
//! 1920x1080 (plane 0) and its interface at 3840x2160 (plane 1), each
//! placed over the whole screen. The console's display blends the second
//! over the first; showing only the first leaves out everything the title
//! draws on the interface plane, which at boot is everything: the logos,
//! the account picker and the loading screens were black for over a minute
//! (with the Series X profile the title draws them into its one plane).
//!
//! The first plane reaches the back buffer as with one plane: copied, bit
//! for bit (`Host::present_now`). The pass then draws one triangle over
//! it: the interface plane, filtered to the back buffer's size and blended
//! by the pipeline with premultiplied alpha, as display planes take it
//! (`VICTORY_PLANE_ALPHA=straight`: straight alpha instead). Where the
//! interface plane is clear the back buffer keeps the first plane's bits.
//!
//! The pass has no sampler: its root signature is one table of one view,
//! and the interface plane is filtered in the shader from four texels read
//! as they are. The first form of this module had a static sampler (linear,
//! clamped, the root signature language's own values for the rest), and
//! with it the title's picture went wrong on `main` of 2026-11-27 (1f5d17a;
//! RTX 3090 Ti, driver 616.92, Series S profile): in the prologue's cell
//! the main character's hair came out bright yellow and the wall beside her
//! lit in blocks, in every run with it (20 of 20) and in none without (11
//! of 11). The root signature only had to be made: the picture was the
//! same with the pass never recorded, and with nothing made but that root
//! signature. What does it is one more static sampler unlike any of the
//! title's own (23 different ones in its first 2,048 root signatures),
//! made before them: the title's linear clamped sampler with another last
//! level, another maximum anisotropy or another border colour does it; the
//! same sampler with another comparison function, which a filter that does
//! not compare makes no use of, does not. Why the order in which different
//! static samplers are first made changes what the title's pipelines
//! sample with is not known yet; nothing of the layer's own should bring a
//! static sampler until it is.

use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D::{D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

use crate::host::{compile, transition};

const SOURCE: &str = r#"
Texture2D<float4> overlay : register(t0);
float4 corner(uint i : SV_VertexID, out float2 uv : TEXCOORD0) : SV_Position {
    float2 p = float2(float(i & 1) * 4.0 - 1.0, float(i >> 1) * 4.0 - 1.0);
    uv = float2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return float4(p, 0.0, 1.0);
}
// The linear filter by hand, from four texels read as they are (clamped at
// the edges): this pass has no sampler (see the module's text).
float4 blend(float4 position : SV_Position, float2 uv : TEXCOORD0) : SV_Target {
    uint w, h;
    overlay.GetDimensions(w, h);
    float2 at = uv * float2(w, h) - 0.5;
    float2 low = floor(at);
    float2 f = at - low;
    int2 a = int2(low);
    int2 last = int2(w, h) - 1;
    float4 t00 = overlay.Load(int3(clamp(a, 0, last), 0));
    float4 t10 = overlay.Load(int3(clamp(a + int2(1, 0), 0, last), 0));
    float4 t01 = overlay.Load(int3(clamp(a + int2(0, 1), 0, last), 0));
    float4 t11 = overlay.Load(int3(clamp(a + int2(1, 1), 0, last), 0));
    return lerp(lerp(t00, t10, f.x), lerp(t01, t11, f.x), f.y);
}
"#;

/// Views of the interface plane (and target views) used in turn, one per
/// present: more than the presents that can be in the air at once.
const RING: u32 = 16;
const PIXEL_SHADER_RESOURCE: u32 = 0x80;

pub(crate) struct Compose {
    format: DXGI_FORMAT,
    root: ID3D12RootSignature,
    pipeline: ID3D12PipelineState,
    views: ID3D12DescriptorHeap,
    targets: ID3D12DescriptorHeap,
    view_step: u32,
    target_step: u32,
    next: u32,
}

/// Whether two planes are blended (`VICTORY_PLANES=0`: only the first is
/// shown, as before).
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLANES").is_none_or(|v| v != "0"))
}

fn straight() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLANE_ALPHA").is_some_and(|v| v == "straight"))
}

/// The view format of a plane's texture (typeless ones read as UNORM).
fn view_format(format: DXGI_FORMAT) -> DXGI_FORMAT {
    match format {
        DXGI_FORMAT_R8G8B8A8_TYPELESS => DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_TYPELESS => DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_R10G10B10A2_TYPELESS => DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_TYPELESS => DXGI_FORMAT_R16G16B16A16_FLOAT,
        f => f,
    }
}

/// The pass's root signature: one table of one view for the pixel shader,
/// and no sampler. It is serialized here from its description and not
/// written inside the pixel shader: a compiler that returns a shader
/// without the root signature written in its source (Wine's, on which
/// making the root signature from the shader failed: Filip722 on pull
/// request #1081) then makes no difference.
fn root_signature(device: &ID3D12Device) -> windows::core::Result<ID3D12RootSignature> {
    let ranges = [D3D12_DESCRIPTOR_RANGE {
        RangeType: D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        NumDescriptors: 1,
        BaseShaderRegister: 0,
        RegisterSpace: 0,
        OffsetInDescriptorsFromTableStart: 0,
    }];
    let parameters = [D3D12_ROOT_PARAMETER {
        ParameterType: D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE,
        Anonymous: D3D12_ROOT_PARAMETER_0 {
            DescriptorTable: D3D12_ROOT_DESCRIPTOR_TABLE {
                NumDescriptorRanges: ranges.len() as u32,
                pDescriptorRanges: ranges.as_ptr(),
            },
        },
        ShaderVisibility: D3D12_SHADER_VISIBILITY_PIXEL,
    }];
    let desc = D3D12_ROOT_SIGNATURE_DESC {
        NumParameters: parameters.len() as u32,
        pParameters: parameters.as_ptr(),
        NumStaticSamplers: 0,
        pStaticSamplers: std::ptr::null(),
        Flags: D3D12_ROOT_SIGNATURE_FLAG_NONE,
    };
    let mut blob: Option<ID3DBlob> = None;
    // SAFETY: the description points at the live arrays above; the blob
    // owns its bytes.
    unsafe {
        D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &mut blob, None)?;
        let blob = blob.ok_or_else(windows::core::Error::empty)?;
        let bytes = std::slice::from_raw_parts(blob.GetBufferPointer().cast::<u8>(), blob.GetBufferSize());
        device.CreateRootSignature(0, bytes)
    }
}

impl Compose {
    /// The pass for back buffers of `format`.
    pub(crate) fn new(device: &ID3D12Device, format: DXGI_FORMAT) -> windows::core::Result<Compose> {
        use windows::core::s;
        let vs = compile(SOURCE, s!("corner"), s!("vs_5_1"))?;
        let ps = compile(SOURCE, s!("blend"), s!("ps_5_1"))?;
        let code = |bytes: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: bytes.as_ptr().cast(),
            BytecodeLength: bytes.len(),
        };
        let root = root_signature(device)?;
        // SAFETY: the description points at live data and never releases
        // its pointer.
        unsafe {
            let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
            formats[0] = format;
            // The second plane over what the back buffer holds: with
            // premultiplied alpha, or straight (`straight`).
            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0] = D3D12_RENDER_TARGET_BLEND_DESC {
                BlendEnable: true.into(),
                LogicOpEnable: false.into(),
                SrcBlend: if straight() { D3D12_BLEND_SRC_ALPHA } else { D3D12_BLEND_ONE },
                DestBlend: D3D12_BLEND_INV_SRC_ALPHA,
                BlendOp: D3D12_BLEND_OP_ADD,
                SrcBlendAlpha: D3D12_BLEND_ONE,
                DestBlendAlpha: D3D12_BLEND_INV_SRC_ALPHA,
                BlendOpAlpha: D3D12_BLEND_OP_ADD,
                LogicOp: D3D12_LOGIC_OP_NOOP,
                RenderTargetWriteMask: D3D12_COLOR_WRITE_ENABLE_ALL.0 as u8,
            };
            let desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                VS: code(&vs[..]),
                PS: code(&ps[..]),
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
                SampleDesc: DXGI_SAMPLE_DESC {
                    Count: 1,
                    Quality: 0,
                },
                ..Default::default()
            };
            let pipeline: ID3D12PipelineState = device.CreateGraphicsPipelineState(&desc)?;
            let views: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: RING,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            let targets: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
                NumDescriptors: RING,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                NodeMask: 0,
            })?;
            tracing::info!(format = format.0, "present planes: the second is blended over the first");
            Ok(Compose {
                format,
                root,
                pipeline,
                view_step: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
                target_step: device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV),
                views,
                targets,
                next: 0,
            })
        }
    }

    pub(crate) fn format(&self) -> DXGI_FORMAT {
        self.format
    }

    /// Records into `list` the pass that blends `interface` (in state
    /// `interface_state`, which it is left in) over what `back` holds (in
    /// the present state, which it is left in): the first plane, copied
    /// there by the caller.
    ///
    /// # Safety
    /// Live resources; `list` open; `back` of this pass's format and
    /// `width` x `height`.
    pub(crate) unsafe fn record(
        &mut self,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        back: &ID3D12Resource,
        (width, height): (u32, u32),
        interface: &ID3D12Resource,
        interface_state: u32,
    ) {
        let slot = self.next % RING;
        self.next = self.next.wrapping_add(1);
        // SAFETY: (all below) descriptor writes into our heaps, then
        // commands into an open list, on live resources.
        unsafe {
            let view = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: view_format(interface.GetDesc().Format),
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV {
                        MostDetailedMip: 0,
                        MipLevels: 1,
                        PlaneSlice: 0,
                        ResourceMinLODClamp: 0.0,
                    },
                },
            };
            let at = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.views.GetCPUDescriptorHandleForHeapStart().ptr + (slot * self.view_step) as usize,
            };
            device.CreateShaderResourceView(interface, Some(&view), at);
            let target = D3D12_CPU_DESCRIPTOR_HANDLE {
                ptr: self.targets.GetCPUDescriptorHandleForHeapStart().ptr + (slot * self.target_step) as usize,
            };
            device.CreateRenderTargetView(back, None, target);
            // What moves for the pass, and back after it.
            let mut moves: Vec<(&ID3D12Resource, D3D12_RESOURCE_STATES, D3D12_RESOURCE_STATES)> = Vec::new();
            if interface_state & PIXEL_SHADER_RESOURCE == 0 {
                moves.push((
                    interface,
                    D3D12_RESOURCE_STATES(interface_state as i32),
                    D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
                ));
            }
            moves.push((back, D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET));
            let before: Vec<_> = moves.iter().map(|&(r, a, b)| transition(r, a, b)).collect();
            list.ResourceBarrier(&before);
            list.SetGraphicsRootSignature(&self.root);
            list.SetDescriptorHeaps(&[Some(self.views.clone())]);
            let gpu = self.views.GetGPUDescriptorHandleForHeapStart().ptr + u64::from(slot * self.view_step);
            list.SetGraphicsRootDescriptorTable(0, D3D12_GPU_DESCRIPTOR_HANDLE { ptr: gpu });
            list.SetPipelineState(&self.pipeline);
            list.OMSetRenderTargets(1, Some(&target), false, None);
            list.RSSetViewports(&[D3D12_VIEWPORT {
                TopLeftX: 0.0,
                TopLeftY: 0.0,
                Width: width as f32,
                Height: height as f32,
                MinDepth: 0.0,
                MaxDepth: 1.0,
            }]);
            list.RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                left: 0,
                top: 0,
                right: width as i32,
                bottom: height as i32,
            }]);
            list.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            list.DrawInstanced(3, 1, 0, 0);
            let after: Vec<_> = moves.iter().map(|&(r, a, b)| transition(r, b, a)).collect();
            list.ResourceBarrier(&after);
        }
    }
}

#[cfg(test)]
mod readback;
