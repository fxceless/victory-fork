use super::tests::{compile, dxc};
use super::*;
use crate::pipeline::make::{Pending, now, ready};
use std::path::Path;
use victory_shader::emit::HostShader;

fn stage(dxc: &Path, dir: &Path, stage: &'static str, profile: &str, source: &str) -> Arc<Shader> {
    let dxil = compile(dxc, dir, source, profile);
    Arc::new(Shader {
        stage,
        hash: blake3::hash(&dxil).to_hex()[..16].into(),
        container: Vec::new(),
        translated: Some(Translated {
            host: HostShader {
                stage: if stage == "vertex" {
                    victory_shader::HwStage::Vs
                } else {
                    victory_shader::HwStage::Ps
                },
                dxil,
                user_data: Vec::new(),
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs: Vec::new(),
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                root_indexed_views: Vec::new(),
                half_views: Vec::new(),
            },
            used: Vec::new(),
            stores: false,
        })
        .into(),
        walks: Default::default(),
    })
}

fn drain(queue: &ID3D12InfoQueue) -> Vec<i32> {
    let mut errors = Vec::new();
    // SAFETY: each message is retrieved into aligned storage of the reported size.
    unsafe {
        for i in 0..queue.GetNumStoredMessages() {
            let mut bytes = 0;
            queue.GetMessage(i, None, &mut bytes).unwrap();
            let mut storage = vec![0u64; bytes.div_ceil(8)];
            let ptr = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
            queue.GetMessage(i, Some(ptr), &mut bytes).unwrap();
            let m = &*ptr;
            if m.Severity.0 <= D3D12_MESSAGE_SEVERITY_WARNING.0 {
                let text = std::slice::from_raw_parts(m.pDescription, m.DescriptionByteLength);
                eprintln!(
                    "D3D12 {:?} #{}: {}",
                    m.Severity,
                    m.ID.0,
                    String::from_utf8_lossy(text).trim_end_matches('\0')
                );
                if m.Severity != D3D12_MESSAGE_SEVERITY_WARNING {
                    errors.push(m.ID.0);
                }
            }
        }
        assert_eq!(queue.GetNumMessagesDiscardedByMessageCountLimit(), 0);
        queue.ClearStoredMessages();
    }
    errors
}

#[test]
#[ignore = "requires hardware D3D12, diagnostics, and PC DXC"]
fn nvidia_r11_target_variant_reproduces_and_fixes_613() {
    use windows::core::Interface;
    let mut debug: Option<ID3D12Debug1> = None;
    // SAFETY: validation is configured before this isolated test creates its device.
    unsafe {
        D3D12GetDebugInterface(&mut debug).unwrap();
        let debug = debug.unwrap();
        debug.EnableDebugLayer();
        debug.SetEnableGPUBasedValidation(true);
    }
    let host = host().expect("D3D12 hardware required");
    let device = &host.device;
    // SAFETY: looking up the adapter of the live host device.
    unsafe {
        use windows::Win32::Graphics::Dxgi::*;
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory.EnumAdapterByLuid(device.GetAdapterLuid()).unwrap();
        let desc = adapter.GetDesc1().unwrap();
        assert_eq!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32, 0);
        eprintln!(
            "hardware: {}",
            String::from_utf16_lossy(&desc.Description).trim_end_matches('\0')
        );
    }
    let info = device.cast().expect("D3D12 info queue");
    let dxc = dxc().expect("VICTORY_PC_DXC must point at dxc.exe");
    let dir = tempfile::tempdir().unwrap();
    let shaders = vec![
        stage(
            &dxc,
            dir.path(),
            "vertex",
            "vs_6_5",
            "float4 main(uint v : SV_VertexID) : SV_Position {
            return float4(v == 2 ? 3 : -1, v == 1 ? 3 : -1, 0, 1); }",
        ),
        stage(
            &dxc,
            dir.path(),
            "pixel",
            "ps_6_5",
            "float4 main() : SV_Target { return float4(1, 0.5, 0.25, 1); }",
        ),
    ];
    let mut g = GraphicsState::default();
    g.depth.DepthEnable = false.into();
    g.rasterizer.CullMode = D3D12_CULL_MODE_NONE;
    g.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    g.targets.NumRenderTargets = 1;
    g.targets.RTFormats[0] = DXGI_FORMAT_R10G10B10A2_UNORM;
    let samples = g.samples;
    let pending = Arc::new(Pending::new(shaders, Some(Box::new(g))));
    assert!(
        Pending::for_r11(
            &pending,
            DXGI_SAMPLE_DESC {
                Count: 2,
                Quality: 0
            }
        )
        .is_none()
    );
    let variant = Pending::for_r11(&pending, samples).unwrap();
    assert!(Arc::ptr_eq(
        variant,
        Pending::for_r11(&pending, samples).unwrap()
    ));
    assert!(now(&pending) && now(variant));
    let (original, fixed) = (ready(&pending).unwrap(), ready(variant).unwrap());
    assert_ne!(original.pso.as_raw(), fixed.pso.as_raw());
    assert!(drain(&info).is_empty(), "pipeline creation errors");
    let back = host.read_back_buffer(256).unwrap();

    // SAFETY: one render-target resource and live heaps; command lists are
    // either discarded before submission or fenced before mapping readback.
    unsafe {
        let mut color: Option<ID3D12Resource> = None;
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: 1,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: DXGI_FORMAT_R11G11B10_FLOAT,
                    SampleDesc: samples,
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                    ..Default::default()
                },
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut color,
            )
            .unwrap();
        let color = color.unwrap();
        let rtv = host.rtv(Some(&color), None);
        let heaps = [
            D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
            D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
        ]
        .map(|ty| {
            Some(
                device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: ty,
                        NumDescriptors: 1,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                        NodeMask: 0,
                    })
                    .unwrap(),
            )
        });
        let draw = |p: &HostPipeline| {
            let list = host.new_list().unwrap();
            list.list.SetDescriptorHeaps(&heaps);
            list.list.SetGraphicsRootSignature(&p.root);
            list.list.SetPipelineState(&p.pso);
            list.list.OMSetRenderTargets(1, Some(&rtv), false, None);
            list.list.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 1.0,
                Height: 1.0,
                MaxDepth: 1.0,
                ..Default::default()
            }]);
            list.list
                .RSSetScissorRects(&[windows::Win32::Foundation::RECT {
                    left: 0,
                    top: 0,
                    right: 1,
                    bottom: 1,
                }]);
            list.list.IASetPrimitiveTopology(
                windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
            );
            list.list.DrawInstanced(3, 1, 0, 0);
            list
        };
        let invalid = draw(original);
        assert_eq!(drain(&info), [613], "reproduce run 289 at recording");
        drop(invalid);

        let list = draw(fixed);
        list.list.ResourceBarrier(&[crate::host::transition(
            &color,
            D3D12_RESOURCE_STATE_RENDER_TARGET,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        )]);
        let src = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&color),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                SubresourceIndex: 0,
            },
        };
        let dst = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&back),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                        Format: DXGI_FORMAT_R11G11B10_FLOAT,
                        Width: 1,
                        Height: 1,
                        Depth: 1,
                        RowPitch: 256,
                    },
                },
            },
        };
        list.list.CopyTextureRegion(&dst, 0, 0, 0, &src, None);
        list.list.Close().unwrap();
        assert!(drain(&info).is_empty(), "variant recording errors");
        let queue = host.queue();
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        queue.ExecuteCommandLists(&[Some(list.list.cast().unwrap())]);
        queue.Signal(&fence, 1).unwrap();
        fence.SetEventOnCompletion(1, Default::default()).unwrap();
        assert_eq!(fence.GetCompletedValue(), 1);
        let mut ptr = std::ptr::null_mut();
        back.Map(0, Some(&D3D12_RANGE { Begin: 0, End: 4 }), Some(&mut ptr))
            .unwrap();
        let pixel = ptr.cast::<u32>().read();
        back.Unmap(0, Some(&D3D12_RANGE::default()));
        assert_eq!(pixel, (15 << 6) | (14 << (11 + 6)) | (13 << (22 + 5)));
        assert!(drain(&info).is_empty(), "variant execution errors");
        eprintln!("variant executed with GPU validation: packed R11G11B10 pixel {pixel:#010x}");
    }
}
