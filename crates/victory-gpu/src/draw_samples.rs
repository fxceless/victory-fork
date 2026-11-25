//! Per-draw samples passing the GPU depth/stencil tests on selected diagnostic frames.
use super::*;
use std::io::Write;
use std::sync::atomic::AtomicU64;

const PER_BLOCK: usize = 256;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("VICTORY_DRAW_SAMPLES").is_some()
            && crate::capture::folder().is_some_and(|path| match victory_core::ensure_outside_repo(
                path,
            ) {
                Ok(()) => true,
                Err(e) => {
                    tracing::warn!(%e, "draw sample capture directory rejected");
                    false
                }
            })
    })
}

struct Block {
    heap: ID3D12QueryHeap,
    back: ID3D12Resource,
}
impl Block {
    fn new(device: &ID3D12Device) -> Option<Self> {
        let mut heap = None;
        let mut back = None;
        let props = D3D12_HEAP_PROPERTIES {
            Type: D3D12_HEAP_TYPE_READBACK,
            ..Default::default()
        };
        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
            Width: (PER_BLOCK * 8) as u64,
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
        // SAFETY: private query slots and a readback buffer that stays in COPY_DEST.
        unsafe {
            device
                .CreateQueryHeap(
                    &D3D12_QUERY_HEAP_DESC {
                        Type: D3D12_QUERY_HEAP_TYPE_OCCLUSION,
                        Count: PER_BLOCK as u32,
                        NodeMask: 0,
                    },
                    &mut heap,
                )
                .ok()?;
            device
                .CreateCommittedResource(
                    &props,
                    D3D12_HEAP_FLAG_NONE,
                    &desc,
                    D3D12_RESOURCE_STATE_COPY_DEST,
                    None,
                    &mut back,
                )
                .ok()?;
        }
        Some(Self {
            heap: heap?,
            back: back?,
        })
    }
    fn values(&self, n: usize) -> Option<Vec<u64>> {
        let range = D3D12_RANGE {
            Begin: 0,
            End: n * 8,
        };
        let mut ptr = std::ptr::null_mut();
        // SAFETY: called after the submission fence, for exactly the initialized slots.
        unsafe {
            self.back.Map(0, Some(&range), Some(&mut ptr)).ok()?;
            let values = std::slice::from_raw_parts(ptr.cast::<u64>(), n).to_vec();
            self.back.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
            Some(values)
        }
    }
}

pub(crate) struct SampleGuard<'a> {
    list: &'a ID3D12GraphicsCommandList,
    heap: ID3D12QueryHeap,
    slot: u32,
    id: u64,
}
impl Drop for SampleGuard<'_> {
    fn drop(&mut self) {
        // SAFETY: the guard belongs to the scope of one graphics draw on this direct list.
        unsafe {
            self.list
                .EndQuery(&self.heap, D3D12_QUERY_TYPE_OCCLUSION, self.slot)
        };
    }
}

#[derive(Default)]
pub(super) struct Samples {
    blocks: Vec<Block>,
    keys: Vec<(u64, u64)>,
}
impl Samples {
    fn begin<'a>(
        &mut self,
        list: &'a ID3D12GraphicsCommandList,
        device: &ID3D12Device,
        frame: u64,
    ) -> Option<SampleGuard<'a>> {
        static SERIAL: AtomicU64 = AtomicU64::new(1);
        let index = self.keys.len();
        if index / PER_BLOCK == self.blocks.len() {
            self.blocks.push(Block::new(device)?);
        }
        let heap = self.blocks[index / PER_BLOCK].heap.clone();
        let slot = (index % PER_BLOCK) as u32;
        let id = SERIAL.fetch_add(1, Ordering::Relaxed);
        self.keys.push((frame, id));
        // SAFETY: the slot belongs to this recording of this command list.
        unsafe { list.BeginQuery(&heap, D3D12_QUERY_TYPE_OCCLUSION, slot) };
        Some(SampleGuard {
            list,
            heap,
            slot,
            id,
        })
    }
    pub(super) fn resolve(&self, list: &ID3D12GraphicsCommandList) {
        for (block, keys) in self.blocks.iter().zip(self.keys.chunks(PER_BLOCK)) {
            // SAFETY: only the completed queries in this recording are resolved.
            unsafe {
                list.ResolveQueryData(
                    &block.heap,
                    D3D12_QUERY_TYPE_OCCLUSION,
                    0,
                    keys.len() as u32,
                    &block.back,
                    0,
                )
            };
        }
    }
    pub(super) fn reset(&mut self) {
        self.keys.clear();
    }
    fn save(&self, folder: &std::path::Path, submission: u64, execution: usize) {
        if self.keys.is_empty() {
            return;
        }
        // The host serializes queue submissions. Keep different recording frames distinct.
        let mut reports = std::collections::BTreeMap::<u64, String>::new();
        for (block, keys) in self.blocks.iter().zip(self.keys.chunks(PER_BLOCK)) {
            let Some(values) = block.values(keys.len()) else {
                tracing::warn!("draw sample readback failed");
                continue;
            };
            for (&(frame, id), n) in keys.iter().zip(values) {
                use std::fmt::Write;
                let _ = writeln!(
                    reports.entry(frame).or_default(),
                    "{id}\t{submission}\t{execution}\t{n}"
                );
            }
        }
        for (frame, text) in reports {
            let path = folder.join(format!("frame-{frame}-draw-samples.tsv"));
            if let Err(e) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .and_then(|mut file| file.write_all(text.as_bytes()))
            {
                tracing::warn!(%e,"draw sample report write failed");
            }
        }
    }
}

/// A repeated list resolves into the same private buffer. Only its last execution
/// in an unseparated group can be attributed after the group fence.
pub(super) fn save_group(runs: &[&HostList], submission: u64) {
    let mut seen = std::collections::HashSet::new();
    for (ordinal, list) in runs.iter().copied().enumerate().rev() {
        if seen.insert(std::ptr::from_ref(list)) {
            list.save_draw_samples(submission, ordinal);
        }
    }
}

impl HostList {
    pub(super) fn has_draw_samples(&self) -> bool {
        !self
            .samples
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .keys
            .is_empty()
    }
    pub(crate) fn sampled_draw(
        &self,
        graphics: bool,
        what: impl FnOnce() -> String,
    ) -> Option<SampleGuard<'_>> {
        let sample = if graphics && enabled() {
            let frame = crate::queue::presented() + 1;
            if crate::capture::listed(frame) {
                host().and_then(|h| {
                    self.samples
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .begin(&self.list, &h.device, frame)
                })
            } else {
                None
            }
        } else {
            None
        };
        self.note_op(|| {
            let mut line = what();
            if let Some(s) = &sample {
                use std::fmt::Write;
                let _ = write!(line, " draw-query={}", s.id);
            }
            line
        });
        sample
    }
    pub(super) fn save_draw_samples(&self, submission: u64, execution: usize) {
        let Some(folder) = crate::capture::folder() else {
            return;
        };
        self.samples
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .save(folder, submission, execution);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST;
    use windows::core::s;

    #[test]
    fn gpu_reports_empty_scissored_and_full_draws() -> windows::core::Result<()> {
        let Some(host) = host() else {
            eprintln!("D3D12 adapter unavailable; skipping hardware coverage test");
            return Ok(());
        };
        let device = &host.device;
        let source = r#"
[RootSignature("RootFlags(0)")]
float4 corner(uint id : SV_VertexID) : SV_Position {
  return float4(float(id&1)*4-1,float(id>>1)*4-1,0.5,1);
}
float4 color() : SV_Target { return float4(1,1,1,1); }
"#;
        let vs = crate::host::compile(source, s!("corner"), s!("vs_5_1"))?;
        let ps = crate::host::compile(source, s!("color"), s!("ps_5_1"))?;
        let code = |b: &[u8]| D3D12_SHADER_BYTECODE {
            pShaderBytecode: b.as_ptr().cast(),
            BytecodeLength: b.len(),
        };
        // SAFETY: all resource descriptions and bytecode live through creation and execution.
        unsafe {
            let sig: ID3D12RootSignature = device.CreateRootSignature(0, &vs)?;
            let mut blend = D3D12_BLEND_DESC::default();
            blend.RenderTarget[0].RenderTargetWriteMask = 15;
            let mut formats = [DXGI_FORMAT_UNKNOWN; 8];
            formats[0] = DXGI_FORMAT_R8G8B8A8_UNORM;
            let sample = DXGI_SAMPLE_DESC {
                Count: 1,
                Quality: 0,
            };
            let mut desc = D3D12_GRAPHICS_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::ManuallyDrop::new(Some(sig.clone())),
                VS: code(&vs),
                PS: code(&ps),
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
                SampleDesc: sample,
                ..Default::default()
            };
            let pso: windows::core::Result<ID3D12PipelineState> =
                device.CreateGraphicsPipelineState(&desc);
            std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
            let pso = pso?;
            let rt_desc = D3D12_RESOURCE_DESC {
                Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                Width: 8,
                Height: 8,
                DepthOrArraySize: 1,
                MipLevels: 1,
                SampleDesc: sample,
                Format: formats[0],
                Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                ..Default::default()
            };
            let mut rt: Option<ID3D12Resource> = None;
            device.CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: D3D12_HEAP_TYPE_DEFAULT,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &rt_desc,
                D3D12_RESOURCE_STATE_RENDER_TARGET,
                None,
                &mut rt,
            )?;
            let rt = rt.unwrap();
            let target = host.rtv(Some(&rt), None);
            let list = host.new_list()?;
            let l = &list.list;
            l.SetGraphicsRootSignature(&sig);
            l.SetPipelineState(&pso);
            l.IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);
            l.OMSetRenderTargets(1, Some(&target), false, None);
            l.RSSetViewports(&[D3D12_VIEWPORT {
                Width: 8.,
                Height: 8.,
                MaxDepth: 1.,
                ..Default::default()
            }]);
            let mut samples = Samples::default();
            drop(samples.begin(l, device, 1).unwrap());
            for (right, bottom) in [(3, 4), (8, 8)] {
                l.RSSetScissorRects(&[RECT {
                    left: 0,
                    top: 0,
                    right,
                    bottom,
                }]);
                let guard = samples.begin(l, device, 1).unwrap();
                l.DrawInstanced(3, 1, 0, 0);
                drop(guard);
            }
            // Cross the heap boundary and exercise more than one recording frame.
            for _ in 3..=PER_BLOCK {
                drop(samples.begin(l, device, 2).unwrap());
            }
            samples.resolve(l);
            list.close();
            host.execute(&[&list]);
            assert_eq!(samples.blocks[0].values(3).unwrap(), [0, 12, 64]);
            let output = tempfile::tempdir().unwrap();
            samples.save(output.path(), 7, 2);
            let read_rows = |frame| {
                let file = output
                    .path()
                    .join(format!("frame-{frame}-draw-samples.tsv"));
                std::fs::read_to_string(file)
                    .unwrap()
                    .lines()
                    .map(|line| {
                        let numbers: Vec<u64> =
                            line.split('\t').map(|part| part.parse().unwrap()).collect();
                        <[u64; 4]>::try_from(numbers).unwrap()
                    })
                    .collect::<Vec<_>>()
            };
            let expected: Vec<_> = samples.keys[..3]
                .iter()
                .zip([0, 12, 64])
                .map(|(&(_, id), count)| [id, 7, 2, count])
                .collect();
            assert_eq!(read_rows(1), expected);
            let extra = read_rows(2);
            assert_eq!(extra.len(), PER_BLOCK - 2);
            for (row, &(_, id)) in extra.iter().zip(&samples.keys[3..]) {
                assert_eq!(*row, [id, 7, 2, 0]);
            }
            samples.reset();
            assert!(samples.keys.is_empty());
        }
        Ok(())
    }
}
