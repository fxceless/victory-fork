#![cfg(windows)]
#![deny(unsafe_op_in_unsafe_fn, unused_must_use)]

use std::mem::ManuallyDrop;
use std::process::Command;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D::{D3D_FEATURE_LEVEL_12_0, Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::Win32::System::Threading::{CreateEventW, WaitForSingleObject};
use windows::core::{Interface, s};

const MODE_ENV: &str = "VICTORY_D3D12_BENCH_WORKER";
const ADAPTER_ENV: &str = "VICTORY_D3D12_BENCH_ADAPTER";
const DISPATCHES: u32 = 512;
const OUTPUT_BYTES: u64 = DISPATCHES as u64 * 4;
const WARMUP: u32 = 8;
const REPEATS: u32 = 31;
const SHADER: &str = r#"
#define RS "RootConstants(num32BitConstants=2, b0), UAV(u0)"
cbuffer Args : register(b0) { uint index; uint salt; }
RWByteAddressBuffer output : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() { output.Store(4 * index, index ^ salt); }
"#;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Off,
    Breadcrumbs,
    Dred,
    Debug,
    Gbv,
}

use Mode::{Breadcrumbs, Debug, Dred, Gbv, Off};
const ROUNDS: [[Mode; 5]; 3] = [
    [Off, Breadcrumbs, Dred, Debug, Gbv],
    [Gbv, Debug, Off, Dred, Breadcrumbs],
    [Dred, Breadcrumbs, Gbv, Debug, Off],
];

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Off => "off",
            Breadcrumbs => "breadcrumbs",
            Dred => "dred",
            Debug => "debug",
            Gbv => "gbv",
        }
    }
}

#[test]
#[ignore = "hardware benchmark; see the fixture README"]
fn d3d12_diagnostics_overhead() {
    let adapter = std::env::var(ADAPTER_ENV).unwrap_or_else(|_| "hardware".into());
    let selector = adapter.to_ascii_lowercase();
    if let Ok(worker) = std::env::var(MODE_ENV) {
        let (round, mode) = worker.split_once(':').expect("worker round:mode");
        let round: usize = round.parse().expect("worker round");
        let mode = *ROUNDS[round]
            .iter()
            .find(|candidate| candidate.name() == mode)
            .expect("worker mode");
        run_configuration(&selector, round, mode);
        return;
    }

    println!(
        "sample,adapter,round,mode,pair,order,timestamps,record_us,submit_us,completion_us,gpu_ticks,gpu_us"
    );
    let executable = std::env::current_exe().expect("benchmark executable");
    // The diagnostics switches apply before device creation. Use a fresh
    // process per mode and round, with at most one list in flight.
    for (round, modes) in ROUNDS.iter().enumerate() {
        for mode in modes {
            let status = Command::new(&executable)
                .args([
                    "--ignored",
                    "--nocapture",
                    "--exact",
                    "d3d12_diagnostics_overhead",
                ])
                .env(MODE_ENV, format!("{round}:{}", mode.name()))
                .status()
                .expect("start D3D12 benchmark worker");
            assert!(
                status.success(),
                "round {} {} failed",
                round + 1,
                mode.name()
            );
        }
    }
}

fn device(selector: &str, mode: Mode) -> (ID3D12Device, DXGI_ADAPTER_DESC1) {
    fn enable(on: bool) -> D3D12_DRED_ENABLEMENT {
        if on {
            D3D12_DRED_ENABLEMENT_FORCED_ON
        } else {
            D3D12_DRED_ENABLEMENT_FORCED_OFF
        }
    }

    // SAFETY: select an actual DXGI adapter and configure the public
    // diagnostic interfaces before creating the only device in this process.
    unsafe {
        let mut settings: Option<ID3D12DeviceRemovedExtendedDataSettings> = None;
        D3D12GetDebugInterface(&mut settings).expect("DRED settings");
        let settings = settings.unwrap();
        settings.SetAutoBreadcrumbsEnablement(enable(matches!(mode, Breadcrumbs | Dred)));
        settings.SetPageFaultEnablement(enable(mode == Dred));
        if matches!(mode, Debug | Gbv) {
            let mut layer: Option<ID3D12Debug1> = None;
            D3D12GetDebugInterface(&mut layer).expect("D3D12 Graphics Tools debug layer");
            let layer = layer.unwrap();
            layer.EnableDebugLayer();
            layer.SetEnableGPUBasedValidation(mode == Gbv);
        }

        let factory: IDXGIFactory6 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let hardware_id = match selector {
            "hardware" | "warp" => None,
            _ => {
                let (vendor, device) = selector.split_once(':').expect("adapter as vendor:device");
                Some((
                    u32::from_str_radix(vendor, 16).expect("hex vendor ID"),
                    u32::from_str_radix(device, 16).expect("hex device ID"),
                ))
            }
        };
        let is_software =
            |desc: &DXGI_ADAPTER_DESC1| desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE.0 as u32 != 0;
        let adapter: IDXGIAdapter1 = if selector == "warp" {
            factory.EnumWarpAdapter().expect("WARP adapter")
        } else {
            (0..)
                .map_while(|index| {
                    factory
                        .EnumAdapterByGpuPreference::<IDXGIAdapter1>(
                            index,
                            DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                        )
                        .ok()
                })
                .find(|adapter| {
                    let desc = adapter.GetDesc1().expect("DXGI adapter description");
                    !is_software(&desc)
                        && hardware_id.is_none_or(|id| id == (desc.VendorId, desc.DeviceId))
                })
                .expect("requested hardware adapter")
        };
        let desc = adapter.GetDesc1().unwrap();
        assert_eq!(is_software(&desc), selector == "warp", "DXGI software flag");
        let mut device: Option<ID3D12Device> = None;
        D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device).expect("D3D12 device");
        let device = device.unwrap();
        let luid = device.GetAdapterLuid();
        assert_eq!(
            (luid.HighPart, luid.LowPart),
            (desc.AdapterLuid.HighPart, desc.AdapterLuid.LowPart)
        );
        (device, desc)
    }
}

fn make_pipeline(device: &ID3D12Device) -> (ID3D12RootSignature, ID3D12PipelineState) {
    // SAFETY: the source is static and the compiled blob outlives both
    // creation calls. The borrowed root in the descriptor isn't dropped.
    unsafe {
        let mut code: Option<ID3DBlob> = None;
        D3DCompile(
            SHADER.as_ptr().cast(),
            SHADER.len(),
            s!("synthetic"),
            None,
            None,
            s!("main"),
            s!("cs_5_1"),
            0,
            0,
            &mut code,
            None,
        )
        .expect("compile synthetic compute shader");
        let code = code.unwrap();
        let bytes =
            std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize());
        let root: ID3D12RootSignature = device.CreateRootSignature(0, bytes).unwrap();
        let pso = device
            .CreateComputePipelineState(&D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&root),
                CS: D3D12_SHADER_BYTECODE {
                    pShaderBytecode: bytes.as_ptr().cast(),
                    BytecodeLength: bytes.len(),
                },
                ..Default::default()
            })
            .unwrap();
        (root, pso)
    }
}

fn buffer(
    device: &ID3D12Device,
    bytes: u64,
    heap: D3D12_HEAP_TYPE,
    flags: D3D12_RESOURCE_FLAGS,
    state: D3D12_RESOURCE_STATES,
) -> ID3D12Resource {
    let mut resource = None;
    // SAFETY: descriptions live for the call and describe a buffer in the supplied heap.
    unsafe {
        device
            .CreateCommittedResource(
                &D3D12_HEAP_PROPERTIES {
                    Type: heap,
                    CreationNodeMask: 1,
                    VisibleNodeMask: 1,
                    ..Default::default()
                },
                D3D12_HEAP_FLAG_NONE,
                &D3D12_RESOURCE_DESC {
                    Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
                    Width: bytes,
                    Height: 1,
                    DepthOrArraySize: 1,
                    MipLevels: 1,
                    SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC {
                        Count: 1,
                        Quality: 0,
                    },
                    Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
                    Flags: flags,
                    ..Default::default()
                },
                state,
                None,
                &mut resource,
            )
            .unwrap();
    }
    resource.unwrap()
}

fn transition(
    resource: &ID3D12Resource,
    from: D3D12_RESOURCE_STATES,
    to: D3D12_RESOURCE_STATES,
) -> D3D12_RESOURCE_BARRIER {
    D3D12_RESOURCE_BARRIER {
        Type: D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
        Flags: D3D12_RESOURCE_BARRIER_FLAG_NONE,
        Anonymous: D3D12_RESOURCE_BARRIER_0 {
            Transition: ManuallyDrop::new(D3D12_RESOURCE_TRANSITION_BARRIER {
                // SAFETY: the caller keeps the resource alive across recording
                // and execution. The barrier never releases the borrowed pointer.
                pResource: unsafe { std::mem::transmute_copy(resource) },
                Subresource: D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                StateBefore: from,
                StateAfter: to,
            }),
        },
    }
}

struct Event(HANDLE);

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: this event is owned by this wrapper.
        unsafe { CloseHandle(self.0) }.expect("close fence event");
    }
}

fn micros(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000_000.0
}

fn assert_validation_clean(info: &ID3D12InfoQueue) {
    // SAFETY: GetMessage fills an aligned buffer of the size it reports.
    // Its description points into this buffer and is terminated by NUL.
    unsafe {
        assert_eq!(info.GetNumMessagesDiscardedByMessageCountLimit(), 0);
        for index in 0..info.GetNumStoredMessagesAllowedByRetrievalFilter() {
            let mut bytes = 0;
            info.GetMessage(index, None, &mut bytes).unwrap();
            let mut storage = vec![0usize; bytes.div_ceil(std::mem::size_of::<usize>())];
            let message = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();
            info.GetMessage(index, Some(message), &mut bytes).unwrap();
            let message = &*message;
            if matches!(
                message.Severity,
                D3D12_MESSAGE_SEVERITY_CORRUPTION | D3D12_MESSAGE_SEVERITY_ERROR
            ) {
                let text = std::ffi::CStr::from_ptr(message.pDescription.cast());
                panic!(
                    "D3D12 validation error {:?}: {}",
                    message.ID,
                    text.to_string_lossy()
                );
            }
        }
    }
}

fn run_configuration(selector: &str, round: usize, mode: Mode) {
    let (device, desc) = device(selector, mode);
    let (root, pipeline) = make_pipeline(&device);
    let adapter_id = format!("{:04x}:{:04x}", desc.VendorId, desc.DeviceId);
    let adapter_name = String::from_utf16_lossy(&desc.Description);
    let uav = buffer(
        &device,
        OUTPUT_BYTES,
        D3D12_HEAP_TYPE_DEFAULT,
        D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
    );
    let readback = buffer(
        &device,
        OUTPUT_BYTES + 16,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    );

    // SAFETY: objects stay alive until after the last fence. An allocator is
    // reset and the readback is accessed only after all submitted work completes.
    unsafe {
        let kind = D3D12_COMMAND_LIST_TYPE_DIRECT;
        let queue: ID3D12CommandQueue = device
            .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                Type: kind,
                ..Default::default()
            })
            .unwrap();
        let allocator: ID3D12CommandAllocator = device.CreateCommandAllocator(kind).unwrap();
        let list: ID3D12GraphicsCommandList =
            device.CreateCommandList(0, kind, &allocator, None).unwrap();
        list.Close().unwrap();
        let lists = [Some(list.cast::<ID3D12CommandList>().unwrap())];
        let fence: ID3D12Fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
        let event = Event(CreateEventW(None, false, false, None).unwrap());
        let mut queries: Option<ID3D12QueryHeap> = None;
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut queries,
            )
            .unwrap();
        let queries = queries.unwrap();
        let frequency = queue.GetTimestampFrequency().unwrap();
        assert!(frequency > 0, "queue timestamp frequency");
        let info = device.cast::<ID3D12InfoQueue>().ok();
        assert_eq!(
            info.is_some(),
            matches!(mode, Debug | Gbv),
            "debug layer state"
        );
        if let Some(info) = &info {
            let mut severities = [
                D3D12_MESSAGE_SEVERITY_CORRUPTION,
                D3D12_MESSAGE_SEVERITY_ERROR,
            ];
            info.AddStorageFilterEntries(&D3D12_INFO_QUEUE_FILTER {
                AllowList: D3D12_INFO_QUEUE_FILTER_DESC {
                    NumSeverities: severities.len() as u32,
                    pSeverityList: severities.as_mut_ptr(),
                    ..Default::default()
                },
                ..Default::default()
            })
            .unwrap();
        }
        let to_copy = transition(
            &uav,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
        );
        let to_uav = transition(
            &uav,
            D3D12_RESOURCE_STATE_COPY_SOURCE,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        );
        let address = uav.GetGPUVirtualAddress();
        let mut sequence = 0;
        let mut rows = Vec::with_capacity((2 * REPEATS) as usize);
        for pair in 0..WARMUP + REPEATS {
            let queries_first = !(pair as usize + round).is_multiple_of(2);
            for (order, timed) in [queries_first, !queries_first].into_iter().enumerate() {
                sequence += 1;
                let salt = sequence as u32 * 65536;
                let start = Instant::now();
                allocator.Reset().unwrap();
                list.Reset(&allocator, &pipeline).unwrap();
                list.SetComputeRootSignature(&root);
                list.SetComputeRootUnorderedAccessView(1, address);
                list.SetComputeRoot32BitConstant(0, salt, 1);
                if timed {
                    list.EndQuery(&queries, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                }
                // Writes are disjoint, so dispatches need no intervening UAV barrier.
                for index in 0..DISPATCHES {
                    list.SetComputeRoot32BitConstant(0, index, 0);
                    list.Dispatch(1, 1, 1);
                }
                if timed {
                    list.EndQuery(&queries, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                    list.ResolveQueryData(
                        &queries,
                        D3D12_QUERY_TYPE_TIMESTAMP,
                        0,
                        2,
                        &readback,
                        OUTPUT_BYTES,
                    );
                }
                list.ResourceBarrier(std::slice::from_ref(&to_copy));
                list.CopyBufferRegion(&readback, 0, &uav, 0, OUTPUT_BYTES);
                list.ResourceBarrier(std::slice::from_ref(&to_uav));
                list.Close().unwrap();
                let recorded = Instant::now();
                queue.ExecuteCommandLists(&lists);
                queue.Signal(&fence, sequence).unwrap();
                let submitted = Instant::now();
                fence.SetEventOnCompletion(sequence, event.0).unwrap();
                let wait = WaitForSingleObject(event.0, 30_000);
                let completed = Instant::now();
                device.GetDeviceRemovedReason().expect("D3D12 device lost");
                assert_eq!(wait, WAIT_OBJECT_0, "fence wait");
                assert!(fence.GetCompletedValue() >= sequence, "incomplete fence");

                let mut mapped = std::ptr::null_mut();
                let end = OUTPUT_BYTES as usize + if timed { 16 } else { 0 };
                readback
                    .Map(
                        0,
                        Some(&D3D12_RANGE { Begin: 0, End: end }),
                        Some(&mut mapped),
                    )
                    .unwrap();
                let values = std::slice::from_raw_parts(mapped.cast::<u32>(), DISPATCHES as usize);
                let correct = values
                    .iter()
                    .enumerate()
                    .all(|(i, &value)| value == i as u32 ^ salt);
                let ticks = timed.then(|| {
                    let ends = mapped
                        .cast::<u8>()
                        .add(OUTPUT_BYTES as usize)
                        .cast::<[u64; 2]>()
                        .read();
                    ends[1].checked_sub(ends[0]).filter(|&value| value != 0)
                });
                readback.Unmap(0, Some(&D3D12_RANGE { Begin: 0, End: 0 }));
                assert!(correct, "compute output mismatch in batch {sequence}");
                let ticks = ticks.map(|ticks| ticks.expect("invalid GPU timestamps"));
                if pair >= WARMUP {
                    rows.push((
                        pair - WARMUP + 1,
                        order + 1,
                        recorded - start,
                        submitted - recorded,
                        completed - start,
                        ticks,
                    ));
                }
            }
        }
        if let Some(info) = &info {
            assert_validation_clean(info);
        }
        println!(
            "adapter,{adapter_id},luid={:08x}:{:08x},subsys={:08x},flags={:#x},vram_bytes={},queue_hz={frequency},round={},mode={},dispatches={DISPATCHES},warmup_pairs={WARMUP},measured_pairs={REPEATS},name={}",
            desc.AdapterLuid.HighPart,
            desc.AdapterLuid.LowPart,
            desc.SubSysId,
            desc.Flags,
            desc.DedicatedVideoMemory,
            round + 1,
            mode.name(),
            adapter_name.trim_end_matches('\0'),
        );
        for (pair, order, record, submit, completion, ticks) in rows {
            let (raw, gpu) = ticks.map_or_else(
                || (String::new(), String::new()),
                |ticks| {
                    (
                        ticks.to_string(),
                        format!("{:.3}", ticks as f64 * 1_000_000.0 / frequency as f64),
                    )
                },
            );
            println!(
                "sample,{adapter_id},{},{},{pair},{order},{},{:.3},{:.3},{:.3},{raw},{gpu}",
                round + 1,
                mode.name(),
                u8::from(ticks.is_some()),
                micros(record),
                micros(submit),
                micros(completion),
            );
        }
    }
}
