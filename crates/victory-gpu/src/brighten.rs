//! `VICTORY_BRIGHTEN=<gamma>[,<gain>]` (for example `0.45` or `0.4,1.2`): the
//! presented image runs through a gamma curve and a gain before it reaches
//! the window, so the parts the host does not light yet (the title's
//! ray-traced bounce light and shadows) can be seen. A view aid only: what
//! is exactly black stays black. Off unless set.

use std::sync::OnceLock;

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// `brighten.hlsl`, built with `dxc -T cs_6_0 -E main` (root signature
/// inside).
static CS: &[u8] = include_bytes!("brighten.cso");

/// The curve, if `VICTORY_BRIGHTEN` asks for one: (gamma, gain).
pub(crate) fn settings() -> Option<(f32, f32)> {
    static V: OnceLock<Option<(f32, f32)>> = OnceLock::new();
    *V.get_or_init(|| {
        let v = std::env::var("VICTORY_BRIGHTEN").ok()?;
        let mut parts = v.split(',').map(|s| s.trim().parse::<f32>());
        let gamma = parts.next()?.ok().filter(|g| *g > 0.0 && *g <= 4.0)?;
        let gain = match parts.next() {
            Some(Ok(g)) if g > 0.0 && g <= 8.0 => g,
            _ => 1.0,
        };
        tracing::info!(gamma, gain, "VICTORY_BRIGHTEN: the presented image is brightened");
        Some((gamma, gain))
    })
}

/// Descriptor pairs (the image's view, the target's) in the heap, used in
/// turn. Presents are sent ahead without a wait (`crate::host`'s
/// `late_free_present`), so the views a present's dispatch reads must not
/// be written over until it has run: with one pair, the next present wrote
/// its own image's view over the one a dispatch still queued was to read
/// (that image, at that point of the card's timeline, possibly in another
/// state or not drawn yet). This many presents are never in the air.
const RING: usize = 64;

/// The pipeline and the texture the curve writes, made on first use.
pub(crate) struct Brighten {
    root: ID3D12RootSignature,
    pso: ID3D12PipelineState,
    heap: ID3D12DescriptorHeap,
    /// The pair the next record uses ([`RING`]).
    next: usize,
    /// (texture in UNORDERED_ACCESS between frames, width, height, format)
    target: Option<(ID3D12Resource, u32, u32, DXGI_FORMAT)>,
}

impl Brighten {
    pub(crate) fn new(device: &ID3D12Device) -> windows::core::Result<Self> {
        // SAFETY: plain object creation from a validated, signed blob.
        unsafe {
            let root: ID3D12RootSignature = device.CreateRootSignature(0, CS)?;
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(root.clone())),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: CS.as_ptr().cast(),
                    BytecodeLength: CS.len(),
                },
                ..Default::default()
            };
            let pso = device.CreateComputePipelineState(&desc);
            drop(std::mem::ManuallyDrop::into_inner(desc.pRootSignature));
            let pso: ID3D12PipelineState = pso?;
            let heap: ID3D12DescriptorHeap = device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 2 * RING as u32,
                Flags: D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
                NodeMask: 0,
            })?;
            Ok(Brighten { root, pso, heap, next: 0, target: None })
        }
    }

    /// Records `texture` (in `state`, read as `view`) through the curve into
    /// the brightened texture (of `format`, the swap chain's) and leaves it in
    /// COPY_SOURCE, `texture` back in `state`; the caller copies from it and
    /// then calls [`Self::after_copy`]. `None`: nothing recorded.
    ///
    /// # Safety
    /// `list` is open; `texture` is a live 2D texture in `state`.
    pub(crate) unsafe fn record(
        &mut self,
        device: &ID3D12Device,
        list: &ID3D12GraphicsCommandList,
        texture: &ID3D12Resource,
        state: D3D12_RESOURCE_STATES,
        view: DXGI_FORMAT,
        format: DXGI_FORMAT,
        (width, height): (u32, u32),
        (gamma, gain): (f32, f32),
    ) -> Option<ID3D12Resource> {
        // SAFETY: see the function's contract; our own heap and texture.
        unsafe {
            if !matches!(&self.target, Some((_, w, h, f)) if (*w, *h, *f) == (width, height, format)) {
                let heap = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() };
                let desc = D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                    Width: u64::from(width),
                    Height: height,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    Format: format,
                    SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                    Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                    ..Default::default()
                };
                let mut made: Option<ID3D12Resource> = None;
                if let Err(e) = device.CreateCommittedResource(
                    &heap,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                    None,
                    &mut made,
                ) {
                    tracing::warn!(error = %e, width, height, format = format.0, "VICTORY_BRIGHTEN: no target texture");
                    return None;
                }
                self.target = Some((made?, width, height, format));
            }
            let target = self.target.as_ref()?.0.clone();
            let size = device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) as usize;
            let pair = self.next % RING;
            self.next = self.next.wrapping_add(1);
            let mut cpu = self.heap.GetCPUDescriptorHandleForHeapStart();
            cpu.ptr += 2 * pair * size;
            let mut gpu = self.heap.GetGPUDescriptorHandleForHeapStart();
            gpu.ptr += (2 * pair * size) as u64;
            let srv = D3D12_SHADER_RESOURCE_VIEW_DESC {
                Format: view,
                ViewDimension: D3D12_SRV_DIMENSION_TEXTURE2D,
                Shader4ComponentMapping: D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING,
                Anonymous: D3D12_SHADER_RESOURCE_VIEW_DESC_0 {
                    Texture2D: D3D12_TEX2D_SRV { MipLevels: 1, ..Default::default() },
                },
            };
            device.CreateShaderResourceView(texture, Some(&srv), cpu);
            let uav = D3D12_UNORDERED_ACCESS_VIEW_DESC {
                Format: format,
                ViewDimension: D3D12_UAV_DIMENSION_TEXTURE2D,
                Anonymous: D3D12_UNORDERED_ACCESS_VIEW_DESC_0 { Texture2D: D3D12_TEX2D_UAV::default() },
            };
            device.CreateUnorderedAccessView(
                &target,
                None,
                Some(&uav),
                D3D12_CPU_DESCRIPTOR_HANDLE { ptr: cpu.ptr + size },
            );
            let read = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
            if state != read {
                crate::barstat::record(&list, crate::barstat::Why::Present, &[crate::host::transition(texture, state, read)]);
            }
            list.SetDescriptorHeaps(&[Some(self.heap.clone())]);
            list.SetComputeRootSignature(&self.root);
            list.SetPipelineState(&self.pso);
            list.SetComputeRoot32BitConstant(0, gamma.to_bits(), 0);
            list.SetComputeRoot32BitConstant(0, gain.to_bits(), 1);
            list.SetComputeRootDescriptorTable(1, gpu);
            list.Dispatch(width.div_ceil(8), height.div_ceil(8), 1);
            let mut back = vec![crate::host::transition(
                &target,
                D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                D3D12_RESOURCE_STATE_COPY_SOURCE,
            )];
            if state != read {
                back.push(crate::host::transition(texture, read, state));
            }
            crate::barstat::record(&list, crate::barstat::Why::Present, &back);
            Some(target)
        }
    }

    /// After the copy from the texture [`Self::record`] returned: back to
    /// UNORDERED_ACCESS for the next frame.
    ///
    /// # Safety
    /// `list` is open; `target` is the texture `record` returned, in COPY_SOURCE.
    pub(crate) unsafe fn after_copy(list: &ID3D12GraphicsCommandList, target: &ID3D12Resource) {
        crate::barstat::record(
            list,
            crate::barstat::Why::Present,
            &[crate::host::transition(target, D3D12_RESOURCE_STATE_COPY_SOURCE, D3D12_RESOURCE_STATE_UNORDERED_ACCESS)],
        );
    }
}
