//! Hardware-only timings for the card compare shaders over synthetic data.
//! Run with `cargo test --release --locked -p victory-gpu bench_card_compare_hardware -- --ignored --nocapture --test-threads=1`.

use super::{Clamp, bare_cost::buffer, tests::compile_cs, transition, uav_barrier};
use crate::host::{CARD_COMPARE, CARD_COMPARE_WIDE};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_12_0;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::*;
use windows::core::Interface;

const PIECE: u32 = 64 << 10;
const MAX_PIECES: u32 = 4096;
const WARMUPS: usize = 4;
const SAMPLES: usize = 10;

fn hardware_device() -> (ID3D12Device, String) {
    // SAFETY: only explicitly enumerated non-software adapters are passed to D3D12.
    unsafe {
        let factory: IDXGIFactory6 =
            CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).expect("create DXGI factory");
        for index in 0.. {
            let adapter: IDXGIAdapter1 = match factory
                .EnumAdapterByGpuPreference(index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE)
            {
                Ok(adapter) => adapter,
                Err(err) if err.code() == DXGI_ERROR_NOT_FOUND => break,
                Err(err) => panic!("enumerating hardware adapters: {err}"),
            };
            let desc = adapter.GetDesc1().expect("describe DXGI adapter");
            if desc.Flags & (DXGI_ADAPTER_FLAG_SOFTWARE.0 | DXGI_ADAPTER_FLAG_REMOTE.0) as u32 != 0
            {
                continue;
            }
            let name = String::from_utf16_lossy(&desc.Description);
            let name = name.trim_end_matches('\0');
            let mut device = None;
            match D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_12_0, &mut device) {
                Ok(()) => return (device.expect("D3D12 hardware device"), name.to_owned()),
                Err(err) => eprintln!("D3D12 hardware adapter {name} unavailable: {err}"),
            }
        }
    }
    panic!("no hardware D3D12 adapter with feature level 12_0; hardware benchmark unavailable");
}

struct Queue {
    queue: ID3D12CommandQueue,
    allocator: ID3D12CommandAllocator,
    list: ID3D12GraphicsCommandList,
    fence: ID3D12Fence,
    submitted: u64,
}

impl Queue {
    fn new(device: &ID3D12Device) -> Self {
        let kind = D3D12_COMMAND_LIST_TYPE_DIRECT;
        // SAFETY: a matching direct queue, allocator and list on the same device.
        unsafe {
            let queue = device
                .CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC {
                    Type: kind,
                    ..Default::default()
                })
                .unwrap();
            let allocator = device.CreateCommandAllocator(kind).unwrap();
            let list = device.CreateCommandList(0, kind, &allocator, None).unwrap();
            let fence = device.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            Self {
                queue,
                allocator,
                list,
                fence,
                submitted: 0,
            }
        }
    }

    fn submit_and_wait(&mut self) {
        self.submitted += 1;
        // SAFETY: all recorded resources stay live through the wait. A null
        // event makes SetEventOnCompletion wait before the allocator is reset.
        unsafe {
            self.list.Close().unwrap();
            self.queue
                .ExecuteCommandLists(&[Some(self.list.cast().unwrap())]);
            self.queue.Signal(&self.fence, self.submitted).unwrap();
            self.fence
                .SetEventOnCompletion(self.submitted, HANDLE::default())
                .unwrap();
            assert_eq!(
                self.fence.GetCompletedValue(),
                self.submitted,
                "GPU fence failed"
            );
            self.allocator.Reset().unwrap();
            self.list.Reset(&self.allocator, None).unwrap();
        }
        // DEFAULT buffers decay to COMMON after each separate submission.
    }

    fn upload(&mut self, device: &ID3D12Device, dst: &ID3D12Resource, offset: u64, bytes: &[u8]) {
        let src = upload_buffer(device, bytes);
        // SAFETY: the caller supplies an in-bounds destination range. The
        // COMMON destination promotes to COPY_DEST for this submission.
        unsafe {
            self.list
                .CopyBufferRegion(dst, offset, &src, 0, bytes.len() as u64)
        };
        self.submit_and_wait();
    }
}

fn default_buffer(
    device: &ID3D12Device,
    bytes: u32,
    flags: D3D12_RESOURCE_FLAGS,
) -> ID3D12Resource {
    buffer(
        device,
        bytes.into(),
        D3D12_HEAP_TYPE_DEFAULT,
        flags,
        D3D12_RESOURCE_STATE_COMMON,
    )
}

fn upload_buffer(device: &ID3D12Device, bytes: &[u8]) -> ID3D12Resource {
    let resource = buffer(
        device,
        bytes.len() as u64,
        D3D12_HEAP_TYPE_UPLOAD,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_GENERIC_READ,
    );
    let mut ptr = std::ptr::null_mut();
    // SAFETY: the upload mapping is at least bytes.len() bytes and is only written here.
    unsafe {
        resource
            .Map(0, Some(&D3D12_RANGE::default()), Some(&mut ptr))
            .unwrap();
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.cast(), bytes.len());
        resource.Unmap(
            0,
            Some(&D3D12_RANGE {
                Begin: 0,
                End: bytes.len(),
            }),
        );
    }
    resource
}

fn readback_buffer(device: &ID3D12Device, bytes: u64) -> ID3D12Resource {
    buffer(
        device,
        bytes,
        D3D12_HEAP_TYPE_READBACK,
        D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATE_COPY_DEST,
    )
}

fn readback<const N: usize>(resource: &ID3D12Resource) -> [u8; N] {
    let mut ptr = std::ptr::null_mut();
    // SAFETY: the callers fence a copy or query resolve of N bytes into this buffer.
    unsafe {
        resource
            .Map(0, Some(&D3D12_RANGE { Begin: 0, End: N }), Some(&mut ptr))
            .unwrap();
        let bytes = ptr.cast::<[u8; N]>().read();
        resource.Unmap(0, Some(&D3D12_RANGE::default()));
        bytes
    }
}

fn synthetic(bytes: u32) -> Vec<u8> {
    assert_eq!(bytes % 8, 0);
    let mut state = 0xd1b5_4a32_d192_ed03_u64;
    let mut data = Vec::with_capacity(bytes as usize);
    for _ in 0..bytes / 8 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        data.extend_from_slice(&state.to_le_bytes());
    }
    data
}

struct Shader {
    name: &'static str,
    pipeline: Clamp,
    groups: fn(u32) -> u32,
}

struct CardBuffers {
    input: ID3D12Resource,
    hashes: ID3D12Resource,
    flags: ID3D12Resource,
}

impl CardBuffers {
    fn new(device: &ID3D12Device, input: &ID3D12Resource, hashes: u32, flags: u32) -> Self {
        Self {
            input: input.clone(),
            hashes: default_buffer(
                device,
                hashes * 16,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ),
            flags: default_buffer(
                device,
                flags * 4,
                D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            ),
        }
    }

    fn bind(&self, list: &ID3D12GraphicsCommandList, shader: &Shader, constants: [u32; 6]) {
        // SAFETY: these addresses and six constants match both shaders' root
        // signatures. Callers keep the resources live and bound ranges in buffer.
        unsafe {
            list.SetComputeRootSignature(&shader.pipeline.root);
            list.SetPipelineState(&shader.pipeline.pso);
            list.SetComputeRoot32BitConstants(0, 6, constants.as_ptr().cast(), 0);
            list.SetComputeRootShaderResourceView(1, self.input.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(2, self.hashes.GetGPUVirtualAddress());
            list.SetComputeRootUnorderedAccessView(3, self.flags.GetGPUVirtualAddress());
        }
    }
}

fn check_correctness(queue: &mut Queue, device: &ID3D12Device, shader: &Shader) {
    const SKEW: u32 = 19;
    const FIRST: u32 = 1;
    const START: u32 = PIECE - SKEW;
    const BOUNDARY: u32 = 2 * PIECE - SKEW;
    const SIZE: u32 = BOUNDARY + 23;
    const PADDED: u32 = 3 * PIECE;
    const OUT_AT: u32 = 8;
    const GUARD: u32 = 7;
    const FLAG_BYTES: usize = 24;

    let input = default_buffer(device, PADDED, D3D12_RESOURCE_FLAG_NONE);
    let buffers = CardBuffers::new(device, &input, 3, 6);
    let content = synthetic(PADDED);
    queue.upload(device, &input, 0, &content);
    queue.upload(device, &buffers.hashes, 0, &[0; 3 * 16]);
    let seed = upload_buffer(device, [GUARD.to_le_bytes(); 6].as_flattened());
    let back = readback_buffer(device, FLAG_BYTES as u64);
    let run = |queue: &mut Queue, size, pieces, mode, expected: [u32; 2], case: &str| {
        let list = &queue.list;
        // SAFETY: input and hashes promote from COMMON to SRV and UAV, respectively.
        // The flags are seeded, transitioned to UAV, and transitioned back to a copy
        // state to order the shader writes before readback. All resources stay live.
        unsafe {
            let flags = &buffers.flags;
            list.CopyBufferRegion(flags, 0, &seed, 0, FLAG_BYTES as u64);
            let uav = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
            list.ResourceBarrier(&[transition(flags, D3D12_RESOURCE_STATE_COPY_DEST, uav)]);
            buffers.bind(list, shader, [FIRST, SKEW, size, pieces, OUT_AT, mode]);
            list.Dispatch((shader.groups)(pieces), 1, 1);
            list.ResourceBarrier(&[transition(flags, uav, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
            list.CopyBufferRegion(&back, 0, flags, 0, FLAG_BYTES as u64);
        }
        queue.submit_and_wait();
        let bytes = readback::<FLAG_BYTES>(&back);
        let flags = std::array::from_fn(|i| u32::from_le_bytes(bytes.as_chunks::<4>().0[i]));
        assert_eq!(
            flags,
            [GUARD, GUARD, expected[0], expected[1], GUARD, GUARD],
            "{} {case}, size {size:#x}, compare={mode}",
            shader.name
        );
    };
    let write_byte = |queue: &mut Queue, at: u32, xor: u8| {
        queue.upload(device, &input, at.into(), &[content[at as usize] ^ xor]);
    };

    run(queue, SIZE, 2, 0, [GUARD; 2], "hash initialization");
    run(queue, SIZE, 2, 1, [0, 0], "unchanged input");
    for (at, expected) in [
        ((START & !15) - 1, [0, 0]), // before the first selected piece's coverage
        (START & !15, [1, 0]),       // leading alignment padding
        ((BOUNDARY & !15) - 1, [1, 0]),
        (BOUNDARY - 1, [1, 1]), // the two pieces share the aligned boundary block
        (BOUNDARY, [1, 1]),
        (BOUNDARY.next_multiple_of(16), [0, 1]),
        (SIZE - 1, [0, 1]),
        (SIZE, [0, 1]), // trailing padding in the final 16-byte load
        (SIZE.next_multiple_of(16), [0, 0]),
    ] {
        write_byte(queue, at, 0x5a);
        run(queue, SIZE, 2, 1, expected, &format!("byte {at:#x}"));
        run(queue, SIZE, 2, 1, expected, "compare-only repeated");
        write_byte(queue, at, 0);
        run(queue, SIZE, 2, 1, [0, 0], "restored after compare-only");
    }
    write_byte(queue, SIZE - 1, 0x5a);
    run(queue, SIZE, 2, 2, [0, 1], "compare-and-store change");
    run(queue, SIZE, 2, 2, [0, 0], "changed hash was stored");
    write_byte(queue, SIZE - 1, 0);
    run(queue, SIZE, 2, 2, [0, 1], "compare-and-store restore");
    run(queue, SIZE, 2, 2, [0, 0], "restored hash was stored");

    let short = START + 2;
    run(
        queue,
        short,
        1,
        0,
        [GUARD; 2],
        "initialize two-byte logical span",
    );
    run(
        queue,
        short,
        1,
        1,
        [0, GUARD],
        "unchanged two-byte logical span",
    );
    write_byte(queue, short, 0x5a);
    run(
        queue,
        short,
        1,
        1,
        [1, GUARD],
        "two-byte span trailing padding",
    );
    write_byte(queue, short, 0);
    run(queue, short, 1, 1, [0, GUARD], "two-byte span restored");
    println!("{} hardware correctness passed", shader.name);
}

fn check_debug_messages(device: &ID3D12Device) {
    let queue: ID3D12InfoQueue = device.cast().unwrap();
    let mut errors = 0;
    let mut warnings = 0;

    // SAFETY: message storage is sized and aligned as required by D3D12.
    unsafe {
        for i in 0..queue.GetNumStoredMessagesAllowedByRetrievalFilter() {
            let mut size = 0;
            queue.GetMessage(i, None, &mut size).unwrap();

            let mut storage = vec![0u64; size.div_ceil(8)];
            let ptr = storage.as_mut_ptr().cast::<D3D12_MESSAGE>();

            queue.GetMessage(i, Some(ptr), &mut size).unwrap();

            let message = &*ptr;
            if message.Severity.0 <= D3D12_MESSAGE_SEVERITY_WARNING.0 {
                let bytes =
                    std::slice::from_raw_parts(message.pDescription, message.DescriptionByteLength);
                eprintln!("D3D12 {}: {}", message.ID.0, String::from_utf8_lossy(bytes));

                if message.Severity.0 <= D3D12_MESSAGE_SEVERITY_ERROR.0 {
                    errors += 1;
                } else {
                    warnings += 1;
                }
            }
        }

        assert_eq!(queue.GetNumMessagesDiscardedByMessageCountLimit(), 0);
    }

    println!("D3D12 validation: {errors} errors, {warnings} warnings");
    assert_eq!(errors, 0);
}

#[test]
#[ignore = "requires a hardware D3D12 adapter; run by name with --ignored --nocapture"]
fn bench_card_compare_hardware() {
    let debug = std::env::var_os("VICTORY_CARD_COMPARE_DEBUG").is_some();

    if debug {
        // SAFETY: enabled before creating the D3D12 device.
        unsafe {
            let mut layer: Option<ID3D12Debug> = None;
            D3D12GetDebugInterface(&mut layer).expect("D3D12 debug layer unavailable");
            layer.unwrap().EnableDebugLayer();
        }
    }

    let (device, card) = hardware_device();
    println!("D3D12 hardware adapter: {card}");
    let shaders = [
        (
            "CARD_COMPARE",
            CARD_COMPARE,
            (|p: u32| p.div_ceil(64)) as fn(u32) -> u32,
        ),
        ("CARD_COMPARE_WIDE", CARD_COMPARE_WIDE, |p| p),
    ]
    .map(|(name, source, groups)| Shader {
        name,
        pipeline: compile_cs(&device, source).unwrap_or_else(|err| panic!("compile {name}: {err}")),
        groups,
    });
    let mut queue = Queue::new(&device);
    for shader in &shaders {
        check_correctness(&mut queue, &device, shader);
    }

    let input = default_buffer(&device, MAX_PIECES * PIECE, D3D12_RESOURCE_FLAG_NONE);
    queue.upload(&device, &input, 0, &synthetic(MAX_PIECES * PIECE));
    let buffers = std::array::from_fn::<_, 2, _>(|_| {
        CardBuffers::new(&device, &input, MAX_PIECES, MAX_PIECES)
    });
    for buffers in &buffers {
        queue.upload(&device, &buffers.flags, 0, &[0; MAX_PIECES as usize * 4]);
    }
    let back = readback_buffer(&device, 16);
    // SAFETY: the direct queue supports timestamps; our heap holds two of them.
    let (frequency, queries): (u64, ID3D12QueryHeap) = unsafe {
        let frequency = queue
            .queue
            .GetTimestampFrequency()
            .expect("GPU timestamp frequency");
        let mut queries = None;
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC {
                    Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP,
                    Count: 2,
                    NodeMask: 0,
                },
                &mut queries,
            )
            .expect("GPU timestamp query heap");
        (frequency, queries.unwrap())
    };
    assert_ne!(frequency, 0, "GPU timestamp frequency is zero");
    println!(
        "Timestamp frequency {frequency} Hz; {WARMUPS} warm-ups and {SAMPLES} samples per shader, piece count and mode; alternating shader order"
    );
    println!(
        "GPU interval includes dispatches and one UAV barrier between passes within each batch. Repeated passes use the same input and addresses. Allocation, uploads, query resolve and CPU fence waits are outside the interval."
    );
    println!(
        "Effective logical GiB/s = pieces × 64 KiB / median pass time (not physical memory bandwidth)."
    );
    println!(
        "pieces mode               batch | CARD_COMPARE us/pass (min..max)   GiB/s | CARD_COMPARE_WIDE us/pass (min..max)   GiB/s | WIDE speedup"
    );

    for pieces in [1, 4, 16, 64, 256, 1024, MAX_PIECES] {
        let bytes = pieces * PIECE;
        let passes = (256 / pieces).clamp(1, 64);
        for (mode, label) in [(0, "init"), (1, "compare"), (2, "compare-and-store")] {
            let mut samples = [[0.0_f64; SAMPLES]; 2];
            for iteration in 0..WARMUPS + SAMPLES {
                for algorithm in [iteration % 2, 1 - iteration % 2] {
                    let shader = &shaders[algorithm];
                    let list = &queue.list;
                    buffers[algorithm].bind(list, shader, [0, 0, bytes, pieces, 0, mode]);
                    // SAFETY: all three DEFAULT buffers promote from COMMON to their
                    // shader states. Mode 0 initializes each shader's hashes for this
                    // piece count before modes 1 and 2 run. Intra-batch UAV barriers
                    // order writes to the same addresses before the next dispatch.
                    unsafe {
                        list.EndQuery(&queries, D3D12_QUERY_TYPE_TIMESTAMP, 0);
                        for pass in 0..passes {
                            if pass != 0 {
                                list.ResourceBarrier(&[uav_barrier()]);
                            }
                            list.Dispatch((shader.groups)(pieces), 1, 1);
                        }
                        list.EndQuery(&queries, D3D12_QUERY_TYPE_TIMESTAMP, 1);
                        list.ResolveQueryData(&queries, D3D12_QUERY_TYPE_TIMESTAMP, 0, 2, &back, 0);
                    }
                    queue.submit_and_wait();
                    let stamps = readback::<16>(&back);
                    let [start, end] = stamps.as_chunks::<8>().0 else {
                        unreachable!()
                    };
                    let (start, end) = (u64::from_le_bytes(*start), u64::from_le_bytes(*end));
                    assert!(
                        end > start,
                        "invalid timestamps for {}, {pieces} pieces, mode {mode}",
                        shader.name
                    );
                    if iteration >= WARMUPS {
                        samples[algorithm][iteration - WARMUPS] =
                            (end - start) as f64 * 1e6 / frequency as f64 / f64::from(passes);
                    }
                }
            }
            let [(old, old_min, old_max), (wide, wide_min, wide_max)] = samples.map(|mut times| {
                times.sort_by(f64::total_cmp);
                (
                    (times[(SAMPLES - 1) / 2] + times[SAMPLES / 2]) / 2.0,
                    times[0],
                    times[SAMPLES - 1],
                )
            });
            let logical_gib_s = |us| f64::from(bytes) * 1e6 / (us * f64::from(1_u32 << 30));
            println!(
                "{pieces:>6} {label:<17} {passes:>5} | {old:>9.3} ({old_min:.3}..{old_max:.3}) {:>8.2} | {wide:>9.3} ({wide_min:.3}..{wide_max:.3}) {:>8.2} | {:>7.2}x",
                logical_gib_s(old),
                logical_gib_s(wide),
                old / wide
            );
        }
    }
    if debug {
        check_debug_messages(&device);
    }
}
