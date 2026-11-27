//! The draw sequences of the title's list calls, timed as the title makes
//! them: through the list's method table, with a host device, title
//! buffers in write-watched memory and a shader-visible title view heap
//! (ported from the list bench of KainanP/VIctory `perf/list-call-bench`).
//!
//! `table_bench` binds a descriptor table with every draw: a constant
//! buffer view and five views of textures the title placed in memory, at
//! a place of its own in the title's heap, as the world's draws bind them
//! (#1379: 18-21 us a bind in the tables, about half the tables made
//! having the contents of one made before). Run with
//! `cargo test -p victory-gpu --release --lib list_bench -- --ignored --nocapture --test-threads=1`.
//!
//! `TABLE_BENCH_REPEAT=<0..100>`: the share of the tables that repeat the
//! contents of an earlier one (50 by default); `TABLE_BENCH_THREADS=1,5`.

use super::*;
use crate::object::{S_OK, slot};
use std::time::{Duration, Instant};
use victory_shader::emit::{HostShader, UserData};
use victory_shader::{HwStage, ResourceType, RootType, UsedResource};
use windows::Win32::Graphics::Direct3D::{Fxc::D3DCompile, ID3DBlob};
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::core::{GUID, Interface, PCSTR, s};

/// Draws recorded into one list before it is closed and reset.
const PER_LIST: usize = 2000;

fn compile(source: &str, target: PCSTR) -> Vec<u8> {
    let mut code: Option<ID3DBlob> = None;
    // SAFETY: the arguments are live and `code` is the output pointer.
    unsafe {
        D3DCompile(
            source.as_ptr().cast(),
            source.len(),
            PCSTR::null(),
            None,
            None,
            s!("main"),
            target,
            0,
            0,
            &mut code,
            None,
        )
        .unwrap();
    }
    let code = code.unwrap();
    // SAFETY: the blob owns these bytes.
    unsafe {
        std::slice::from_raw_parts(code.GetBufferPointer().cast::<u8>(), code.GetBufferSize())
    }
    .to_vec()
}

/// A shader whose translation is `bytes` (host code that reads nothing, so
/// any root signature takes it) with the bindings a title shader has: what
/// the draw path does for it on the CPU is the same.
fn preset(
    stage: &'static str,
    hw: HwStage,
    bytes: &[u8],
    user_data: Vec<UserData>,
    root_cbvs: Vec<u8>,
    used: Vec<UsedResource>,
) -> Arc<Shader> {
    let shader = shader(stage, bytes);
    shader.translated.get_or_init(|| {
        Some(Translated {
            host: HostShader {
                stage: hw,
                dxil: bytes.to_vec(),
                user_data,
                samplers: Vec::new(),
                static_samplers: Vec::new(),
                root_cbvs,
                root_srvs: Vec::new(),
                root_uavs: Vec::new(),
                root_indexed_views: Vec::new(),
                half_views: Vec::new(),
            },
            used,
            stores: false,
            threads: [0; 3],
        })
    });
    shader
}

fn table_entry(kind: ResourceType, slot: u8, k: u32) -> UsedResource {
    UsedResource {
        kind,
        root: RootType::Table,
        table_offset_dwords: 8 * k,
        lower_bound: 0,
        stride: 0,
        dwords: 8,
        user_data: 0,
        slot,
    }
}

fn user(index: u32, slot: u8, root: RootType, dword: u32) -> UserData {
    UserData {
        index,
        slot,
        root,
        dword,
    }
}

fn made(pso: This) -> &'static host::HostPipeline {
    let started = Instant::now();
    loop {
        if let Some(p) = host_pipeline(pso) {
            return p;
        }
        assert!(
            started.elapsed() < Duration::from_secs(60),
            "pipeline not made"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[repr(C, align(8))]
struct Depth1(u32, D3D12_DEPTH_STENCIL_DESC1);

fn graphics_pipeline(vs: &Arc<Shader>, ps: &Arc<Shader>) -> This {
    #[repr(C)]
    struct Stream {
        vs_kind: u64,
        vs: Bytecode,
        ps_kind: u64,
        ps: Bytecode,
        depth: Depth1,
        topology: [u32; 2],
        dsv: [u32; 2],
    }
    let stream = Stream {
        vs_kind: 1,
        vs: Bytecode {
            code: vs.container.as_ptr(),
            size: vs.container.len(),
        },
        ps_kind: 2,
        ps: Bytecode {
            code: ps.container.as_ptr(),
            size: ps.container.len(),
        },
        depth: Depth1(
            21,
            D3D12_DEPTH_STENCIL_DESC1 {
                DepthEnable: true.into(),
                DepthWriteMask: D3D12_DEPTH_WRITE_MASK_ALL,
                DepthFunc: D3D12_COMPARISON_FUNC_LESS_EQUAL,
                ..Default::default()
            },
        ),
        topology: [14, D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0 as u32],
        dsv: [16, DXGI_FORMAT_D32_FLOAT.0 as u32],
    };
    let desc = StreamDesc {
        size: size_of_val(&stream),
        stream: (&raw const stream).cast(),
    };
    let mut pso = std::ptr::null();
    assert_eq!(
        create_stream(std::ptr::null(), &desc, &PIPELINE_IIDS[0], &mut pso),
        S_OK
    );
    made(pso);
    pso
}

fn iid(g: GUID) -> Iid {
    guid(g.data1, g.data2, g.data3, g.data4)
}

const RESOURCE_IID: GUID = GUID::from_values(
    0x696442be,
    0xa72e,
    0x4059,
    [0xbc, 0x79, 0x5b, 0x5c, 0x98, 0x04, 0x0f, 0xad],
);

/// A title buffer of `bytes` at `address`.
fn buffer_at(address: u64, bytes: u64) -> This {
    let desc: [u32; 14] = [
        1,
        0,
        0,
        0,
        bytes as u32,
        (bytes >> 32) as u32,
        1,
        1 | 1 << 16,
        0,
        1,
        0,
        1,
        0,
        0,
    ];
    let mut out = std::ptr::null();
    let made = crate::resource::create_at_address(
        std::ptr::null(),
        address,
        desc.as_ptr().cast(),
        0,
        std::ptr::null(),
        &iid(RESOURCE_IID),
        &mut out,
    );
    assert_eq!(made, S_OK);
    out
}

/// Write-watched title memory of `pages` 64 KiB pages (leaked: title memory
/// outlives a test).
fn pool(pages: usize) -> u64 {
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc,
    };
    const MEM_WRITE_WATCH: u32 = 0x20_0000;
    // SAFETY: a fresh allocation.
    let p = unsafe {
        VirtualAlloc(
            std::ptr::null(),
            pages * crate::watch::PAGE,
            MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH,
            PAGE_READWRITE,
        )
    } as u64;
    assert_ne!(p, 0);
    p
}

/// Title memory of `bytes`, 64 KiB aligned, not write-watched.
fn title_memory(bytes: usize) -> u64 {
    let layout = std::alloc::Layout::from_size_align(bytes, 1 << 16).unwrap();
    // SAFETY: a non-zero size; leaked, as title memory outlives a test.
    unsafe { std::alloc::alloc_zeroed(layout) as u64 }
}

/// A 64x64 RGBA8 texture the title placed at `address` (resource
/// description), and the view of all of it.
const PLACED_TEXTURE: [u32; 14] = [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0, 0, 0];

fn placed_view(address: u64) -> [u32; 16] {
    let mut desc = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    (desc[10], desc[11]) = (address as u32, (address >> 32) as u32);
    desc
}

/// Views in a table: one constant buffer view, then five textures.
const TABLE_VIEWS: u32 = 6;
/// Tables a thread's list binds, one a draw, in order.
const TABLES: usize = PER_LIST;
/// Distinct contents the repeating tables take theirs from.
const MATERIALS: usize = 64;
/// Bytes of a table's constants.
const CONSTANTS: u32 = 256;

/// What every timed list shares: the title's objects.
struct Scene {
    heap: This,
    heap_start: u64,
    /// Constant ring (write-watched) and index buffer.
    ring: u64,
    index: u64,
    draw: [This; 2],
    /// The pipeline whose pixel shader reads a table of [`TABLE_VIEWS`].
    tabled: This,
    /// The title's tables, [`TABLE_VIEWS`] slots each from slot 64 of the
    /// heap: whether each repeats the contents of an earlier one.
    repeats: Vec<bool>,
    /// The texture memory the tables' views show, by table and view.
    #[allow(dead_code)]
    shown: Vec<[u64; 5]>,
    #[allow(dead_code)]
    constants_of: Vec<u64>,
}

// SAFETY: the title objects are free-threaded, as the title uses them.
unsafe impl Sync for Scene {}

const RING_PAGES: usize = 16;

impl Scene {
    fn new(repeat_percent: usize) -> Scene {
        let null = std::ptr::null();
        let vs_code = compile(
            "float4 main(uint id : SV_VertexID) : SV_Position { return float4(id, 0, 0, 1); }",
            s!("vs_5_1"),
        );
        let ps_code = |k: u32| compile(&format!("void main() {{ float x = {k}; }}"), s!("ps_5_1"));
        // Vertex shader: frame and object constants as root views, four
        // root constants. Pixel shader: six textures through a table, frame
        // constants. As the title's (title root slots 0-3).
        let vs = preset(
            "vertex",
            HwStage::Vs,
            &vs_code,
            (0..4).map(|d| user(d, 3, RootType::Constants, d)).collect(),
            vec![0, 1],
            Vec::new(),
        );
        let ps: Vec<_> = (0..2)
            .map(|k| {
                preset(
                    "pixel",
                    HwStage::Ps,
                    &ps_code(k),
                    vec![user(0, 2, RootType::Table, 0)],
                    vec![0],
                    (0..6)
                        .map(|t| table_entry(ResourceType::Srv, 2, t))
                        .collect(),
                )
            })
            .collect();
        // The world's material shader: its constants and five textures
        // through one table (root slot 2).
        // (Code of its own: a container is shared by its bytes, and with
        // the translation of the first shader of those bytes.)
        let tabled_code = compile("void main(float4 at : SV_Position) {}", s!("ps_5_1"));
        assert!(ps.iter().all(|p| p.container != tabled_code), "the material shader's container is its own");
        let tabled_ps = preset(
            "pixel",
            HwStage::Ps,
            &tabled_code,
            vec![user(0, 2, RootType::Table, 0)],
            vec![0],
            std::iter::once(table_entry(ResourceType::Cbv, 2, 0))
                .chain((1..TABLE_VIEWS).map(|t| table_entry(ResourceType::Srv, 2, t)))
                .collect(),
        );
        let draw = [
            graphics_pipeline(&vs, &ps[0]),
            graphics_pipeline(&vs, &ps[1]),
        ];
        let tabled = graphics_pipeline(&vs, &tabled_ps);

        let ring = pool(RING_PAGES);
        buffer_at(ring, (RING_PAGES * crate::watch::PAGE) as u64);
        let index = pool(1);
        buffer_at(index, crate::watch::PAGE as u64);

        // Textures placed in memory: one set for the materials, one for
        // the tables that repeat nothing (5 each).
        let textures = 5 * (MATERIALS + TABLES);
        let memory = title_memory(textures << 14);
        let texture = |k: usize| memory + ((k as u64) << 14);

        let slots = 64 + TABLES as u32 * TABLE_VIEWS;
        let heap_desc: [u32; 8] = [0, slots.next_power_of_two(), 1, 0, 0, 0, 0, 0];
        let mut heap = null;
        let heap_iid = iid(ID3D12DescriptorHeap::IID);
        assert_eq!(
            crate::descriptor::create_heap(null, heap_desc.as_ptr().cast(), &heap_iid, &mut heap),
            S_OK
        );
        let heap_start = crate::descriptor::heap(heap).start;
        // Slots 0-7: textures for the plain draws' table.
        for k in 0..8u64 {
            crate::descriptor::create_placed_srv(
                null,
                &PLACED_TEXTURE,
                &placed_view(texture(k as usize)),
                heap_start + 32 * k,
            );
        }
        // The tables: every `100 / repeat` one has a material's contents
        // (its constants and textures); the others constants and textures
        // of their own. The constants of material m sit at ring + m * 256,
        // a table's own past them.
        let mut repeats = Vec::with_capacity(TABLES);
        let mut shown = Vec::with_capacity(TABLES);
        let mut constants_of = Vec::with_capacity(TABLES);
        let mut debt = 0;
        for t in 0..TABLES {
            debt += repeat_percent;
            let repeat = debt >= 100;
            if repeat {
                debt -= 100;
            }
            let (constants, first_texture) = if repeat {
                let m = t % MATERIALS;
                (ring + (m as u64) * u64::from(CONSTANTS), 5 * m)
            } else {
                (
                    ring + ((MATERIALS + t) as u64) * u64::from(CONSTANTS),
                    5 * (MATERIALS + t),
                )
            };
            let at = heap_start + 32 * (64 + t as u64 * u64::from(TABLE_VIEWS));
            crate::descriptor::create_cbv(
                null,
                &[constants as u32, (constants >> 32) as u32, CONSTANTS, 0],
                at,
            );
            let mut views = [0; 5];
            for (v, view) in views.iter_mut().enumerate() {
                *view = texture(first_texture + v);
                crate::descriptor::create_placed_srv(
                    null,
                    &PLACED_TEXTURE,
                    &placed_view(*view),
                    at + 32 * (1 + v as u64),
                );
            }
            repeats.push(repeat);
            shown.push(views);
            constants_of.push(constants);
        }
        Scene {
            heap,
            heap_start,
            ring,
            index,
            draw,
            tabled,
            repeats,
            shown,
            constants_of,
        }
    }

    /// The title's handle of table `t`.
    fn table(&self, t: usize) -> u64 {
        self.heap_start + 32 * (64 + (t % TABLES) as u64 * u64::from(TABLE_VIEWS))
    }
}

/// A title list and its allocator.
struct TitleList {
    list: This,
    allocator: This,
    newer: bool,
}

type Release = extern "system" fn(This) -> u32;

impl TitleList {
    fn new(pipeline: This) -> TitleList {
        let null = std::ptr::null();
        let (mut allocator, mut list) = (null, null);
        assert_eq!(
            crate::list::create_allocator(
                null,
                0,
                &iid(ID3D12CommandAllocator::IID),
                &mut allocator
            ),
            S_OK
        );
        assert_eq!(
            crate::list::create_list(
                null,
                0,
                0,
                allocator,
                pipeline,
                &iid(ID3D12GraphicsCommandList::IID),
                &mut list
            ),
            S_OK
        );
        TitleList {
            list,
            allocator,
            newer: crate::device::newer(),
        }
    }

    fn at<F: Copy>(&self, older: usize) -> F {
        let offset = if self.newer && older >= 0x1d8 {
            older + 0x10
        } else {
            older
        };
        slot::<F>(self.list, offset)
    }

    fn close(&self) {
        assert_eq!(
            slot::<extern "system" fn(This) -> HRESULT>(self.list, 0x48)(self.list),
            S_OK
        );
    }

    fn reset(&self, pipeline: This) {
        assert_eq!(
            slot::<extern "system" fn(This, This, This) -> HRESULT>(self.list, 0x50)(
                self.list,
                self.allocator,
                pipeline
            ),
            S_OK
        );
    }
}

impl Drop for TitleList {
    fn drop(&mut self) {
        slot::<Release>(self.list, 0x10)(self.list);
        slot::<Release>(self.allocator, 0x10)(self.allocator);
    }
}

#[repr(C)]
struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// The title's method types, by the older layout's offsets.
type Heaps = extern "system" fn(This, u32, *const This);
type SetPipeline = extern "system" fn(This, This);
type Table = extern "system" fn(This, u32, u64);
type View = extern "system" fn(This, u32, u64);
type Constants = extern "system" fn(This, u32, u32, *const u32, u32);
type IndexBuffer = extern "system" fn(This, *const IndexView);
type Viewports = extern "system" fn(This, u32, *const D3D12_VIEWPORT);
type Topology = extern "system" fn(This, u32);
type DrawIndexed = extern "system" fn(This, u32, u32, u32, i32, u32);

/// A title list set up for draws with `pipeline`: heaps, pipeline,
/// topology, viewport, tables and root views as a pass sets them.
fn graphics_list(scene: &Scene, pipeline: This) -> TitleList {
    let l = TitleList::new(pipeline);
    set_up_graphics(&l, scene, pipeline);
    l
}

fn set_up_graphics(l: &TitleList, scene: &Scene, pipeline: This) {
    let list = l.list;
    l.at::<Heaps>(0x100)(list, 1, &scene.heap);
    l.at::<SetPipeline>(0xc8)(list, pipeline);
    l.at::<Topology>(0xa0)(list, 4);
    l.at::<Viewports>(0xa8)(
        list,
        1,
        &D3D12_VIEWPORT {
            Width: 1920.0,
            Height: 1080.0,
            MaxDepth: 1.0,
            ..Default::default()
        },
    );
    l.at::<Table>(0x120)(list, 2, scene.heap_start);
    l.at::<View>(0x150)(list, 0, scene.ring);
    l.at::<View>(0x150)(list, 1, scene.ring + 0x100);
    l.at::<Constants>(0x140)(list, 3, 4, [1u32, 2, 3, 4].as_ptr(), 0);
    l.at::<IndexBuffer>(0x178)(
        list,
        &IndexView {
            address: scene.index,
            size: 72,
            format: DXGI_FORMAT_R16_UINT.0 as u32,
        },
    );
}

/// One draw as the title records it in a pass: object constants (a ring
/// position of their own), four root constants, the index buffer, then the
/// draw; with `tables`, the draw's material table first.
fn draw_sequence(l: &TitleList, scene: &Scene, list: This, i: usize, tables: bool) {
    if tables {
        l.at::<Table>(0x120)(list, 2, scene.table(i));
    }
    let ring_bytes = (RING_PAGES * crate::watch::PAGE) as u64;
    l.at::<View>(0x150)(
        list,
        1,
        scene.ring + 0x100 + (i as u64 * 0x100) % (ring_bytes - 0x200),
    );
    l.at::<Constants>(0x140)(list, 3, 4, [i as u32, 2, 3, 4].as_ptr(), 0);
    l.at::<IndexBuffer>(0x178)(
        list,
        &IndexView {
            address: scene.index,
            size: 72,
            format: DXGI_FORMAT_R16_UINT.0 as u32,
        },
    );
    l.at::<DrawIndexed>(0x68)(list, 36, 1, 0, 0, 0);
}

/// Nanoseconds a draw over `draws` draws, the list closed (timed: work a
/// list leaves to its close counts), reset and set up again every
/// [`PER_LIST`] (not timed).
fn time(l: &TitleList, scene: &Scene, draws: usize, pipeline: This, tables: bool) -> f64 {
    let mut total = Duration::ZERO;
    let mut done = 0;
    while done < draws {
        let n = PER_LIST.min(draws - done);
        let started = Instant::now();
        for i in done..done + n {
            draw_sequence(l, scene, l.list, i, tables);
        }
        l.close();
        total += started.elapsed();
        done += n;
        l.reset(pipeline);
        set_up_graphics(l, scene, pipeline);
    }
    total.as_nanos() as f64 / draws as f64
}

/// The fastest of `runs` timings (the machine's noise only adds).
fn best(runs: usize, mut f: impl FnMut() -> f64) -> f64 {
    (0..runs).map(|_| f()).fold(f64::INFINITY, f64::min)
}

/// Waits until draws with `pipeline` run on the host (its host pipeline
/// for the list's targets is made on another thread; draws before are
/// dropped).
fn ready(scene: &Scene, pipeline: This) {
    let started = Instant::now();
    let l = graphics_list(scene, pipeline);
    loop {
        let drawn = crate::draw::drawn();
        draw_sequence(&l, scene, l.list, 0, pipeline == scene.tabled);
        if crate::draw::drawn() > drawn {
            break;
        }
        assert!(started.elapsed() < Duration::from_secs(60), "draws never ran");
        std::thread::sleep(Duration::from_millis(10));
    }
    l.close();
}

/// The draw sequence on `threads` threads at once, each recording its own
/// list of draws, as the world's render threads do: ns a draw per thread.
fn on_threads(scene: &Scene, threads: usize, runs: usize, draws: usize, tables: bool) -> f64 {
    let pipeline = if tables { scene.tabled } else { scene.draw[0] } as usize;
    ready(scene, pipeline as This);
    let drawn = crate::draw::drawn();
    let started = std::sync::Barrier::new(threads);
    let per_thread: Vec<f64> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                s.spawn(|| {
                    let pipeline = pipeline as This;
                    let l = graphics_list(scene, pipeline);
                    started.wait();
                    best(runs, || time(&l, scene, draws, pipeline, tables))
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    assert!(crate::draw::drawn() - drawn >= (threads * runs * draws) as u64, "every draw ran on the host");
    per_thread.iter().sum::<f64>() / threads as f64
}

fn adapter_name() -> String {
    let host = crate::host::host().expect("a host device");
    // SAFETY: plain adapter queries on a live device.
    let adapter = unsafe {
        use windows::Win32::Graphics::Dxgi::{
            CreateDXGIFactory2, DXGI_CREATE_FACTORY_FLAGS, IDXGIAdapter1, IDXGIFactory4,
        };
        let factory: IDXGIFactory4 = CreateDXGIFactory2(DXGI_CREATE_FACTORY_FLAGS(0)).unwrap();
        let adapter: IDXGIAdapter1 = factory
            .EnumAdapterByLuid(host.device.GetAdapterLuid())
            .unwrap();
        adapter.GetDesc1().unwrap()
    };
    String::from_utf16_lossy(&adapter.Description)
        .trim_end_matches('\0')
        .to_owned()
}

fn threads_asked() -> Vec<usize> {
    std::env::var("TABLE_BENCH_THREADS")
        .ok()
        .map(|v| v.split(',').filter_map(|n| n.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![1, 5])
}

/// The draw sequence of the VIctory list bench (root views, constants,
/// index buffer, draw; no table changes) on one and five threads.
#[test]
#[ignore = "timing; run alone on a quiet machine with --release --ignored --nocapture"]
fn list_bench() {
    if crate::host::host().is_none() {
        eprintln!("list_bench skipped: no host D3D12 device");
        return;
    }
    let scene = Scene::new(50);
    eprintln!("list_bench on {}", adapter_name());
    for threads in threads_asked() {
        let ns = on_threads(&scene, threads, 5, 20_000, false);
        eprintln!("draw sequence on {threads} thread(s) at once: {ns:.0} ns a draw (per thread)");
    }
}

/// The draw sequence with a material table bound by every draw (a
/// constant view and five placed textures, at a place of its own in the
/// title's heap; `TABLE_BENCH_REPEAT` percent of them with the contents of
/// an earlier table), on one and five threads.
#[test]
#[ignore = "timing; run alone on a quiet machine with --release --ignored --nocapture"]
fn table_bench() {
    if crate::host::host().is_none() {
        eprintln!("table_bench skipped: no host D3D12 device");
        return;
    }
    let repeat = std::env::var("TABLE_BENCH_REPEAT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(50);
    let scene = Scene::new(repeat);
    let repeating = scene.repeats.iter().filter(|r| **r).count();
    eprintln!(
        "table_bench on {} ({} tables of {TABLE_VIEWS} views a list, {repeating} repeating the contents of an earlier one)",
        adapter_name(),
        TABLES
    );
    let drawn = crate::draw::drawn();
    let runs = 5;
    let draws = 4 * PER_LIST;
    // Warm: every texture's host copy made and uploaded once.
    on_threads(&scene, 1, 1, PER_LIST, true);
    for threads in threads_asked() {
        let plain = on_threads(&scene, threads, runs, draws, false);
        let tabled = on_threads(&scene, threads, runs, draws, true);
        eprintln!(
            "TABLEBENCH threads={threads} repeat={repeat} plain_ns={plain:.0} tabled_ns={tabled:.0} table_ns={:.0}",
            tabled - plain
        );
    }
    assert!(crate::draw::drawn() > drawn, "draws ran on the host");
}

/// What making a table afresh is made of: each piece timed alone, warm,
/// on one thread (ns a call).
#[test]
#[ignore = "timing; run alone on a quiet machine with --release --ignored --nocapture"]
fn table_parts() {
    if crate::host::host().is_none() {
        eprintln!("table_parts skipped: no host D3D12 device");
        return;
    }
    let scene = Scene::new(50);
    on_threads(&scene, 1, 1, PER_LIST, true);
    let l = graphics_list(&scene, scene.tabled);
    let list = crate::list::host_list(l.list).unwrap();
    let heap = crate::descriptor::heap(scene.heap);
    let host = heap.host().unwrap();
    let handle = scene.table(1) + 32;
    let view = crate::descriptor::view(handle).unwrap();
    let title = heap.index(handle).unwrap();
    let n = 100_000;
    let each = |name: &str, f: &mut dyn FnMut()| {
        let ns = best(5, || {
            let started = Instant::now();
            for _ in 0..n {
                f();
            }
            started.elapsed().as_nanos() as f64 / n as f64
        });
        eprintln!("  part {name:<52} {ns:>7.1} ns");
    };
    each("descriptor::view (placed texture)", &mut || {
        std::hint::black_box(crate::descriptor::view(std::hint::black_box(handle)));
    });
    each("descriptor::host::placed_texture", &mut || {
        std::hint::black_box(crate::descriptor::host::placed_texture(&view));
    });
    let first = host.dynamic(8).unwrap();
    each("HostHeap::make_at (placed texture)", &mut || {
        host.make_at(first, &view)
    });
    each("HostHeap::make_null", &mut || host.make_null(first + 1));
    each("HostHeap::constant_at", &mut || {
        host.constant_at(first + 2, 0, 256)
    });
    each("HostHeap::mirror, title slot made", &mut || {
        std::hint::black_box(host.mirror(title, handle, true));
    });
    each("HostHeap::texture_slot_tabled", &mut || {
        host.texture_slot_tabled(title, true)
    });
    host.release(host.id(), first, 8);
    each("HostHeap::dynamic(6) + release", &mut || {
        let at = host.dynamic(6).unwrap();
        host.release(host.id(), at, 6);
    });
    each("HostList::scratch(256) + defer_fill", &mut || {
        let (_, to) = list.scratch(256).unwrap();
        list.defer_fill(to, scene.ring as *const u8, 256);
    });
}

/// The tables made afresh for draws show the views their title slots
/// hold: each texture slot's descriptor is the one the host makes of the
/// view, and the slot holds that view's host texture, in a first list and
/// in the lists after it.
#[test]
fn tables_show_their_views() {
    if crate::host::host().is_none() {
        return; // no host GPU
    }
    let scene = Scene::new(50);
    let heap = crate::descriptor::heap(scene.heap).host().unwrap();
    let tables = 64;
    ready(&scene, scene.tabled);
    for pass in 0..3 {
        let l = graphics_list(&scene, scene.tabled);
        for t in 0..tables {
            draw_sequence(&l, &scene, l.list, t, true);
        }
        let made: Vec<(u32, u32)> = crate::list::dynamic_ranges(l.list).into_iter().filter(|r| r.1 == TABLE_VIEWS).collect();
        assert_eq!(made.len(), tables, "a table made for each draw");
        for (t, &(first, _)) in made.iter().enumerate() {
            for v in 1..TABLE_VIEWS {
                let view = crate::descriptor::view(scene.table(t) + 32 * u64::from(v)).unwrap();
                let Some(crate::descriptor::View::Shader(texture, _)) = crate::descriptor::host::placed_texture(&view) else {
                    panic!("a placed texture view");
                };
                assert_eq!(
                    heap.slot_bytes(first + v),
                    crate::descriptor::host::made_bytes(&view).unwrap(),
                    "table {t} view {v} (list {pass}): the descriptor of its view"
                );
                assert_eq!(
                    heap.kept_at(first + v).as_ref(),
                    crate::resource::host_resource(texture as This),
                    "table {t} view {v} (list {pass}): holds its host texture"
                );
            }
        }
        l.close();
    }
}
