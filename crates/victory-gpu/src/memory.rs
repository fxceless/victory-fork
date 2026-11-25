//! Title memory by address, for translated shaders that read and write
//! through raw 40-bit addresses (`victory_shader::emit::SLOT_MEMORY`: the
//! console's ray-tracing structure builder and the programs that walk its
//! structures; the structures themselves lie in plain title memory, in no
//! buffer object).
//!
//! The copies are the ones every other access uses ([`crate::shadow`]: a
//! title buffer's copy, or the 4 MiB window of memory outside buffers), so
//! a root view and a raw address reach the same host bytes. This module
//! keeps what a shader needs to find them:
//!
//! - the **directory**: per 64 KiB page of title address space the number
//!   of the copy that holds the page (0: none) and the page's byte offset
//!   in it; one buffer for all lists, entries only ever go from empty to
//!   filled;
//! - per view heap a run of slots with a raw unordered access view of each
//!   numbered copy (slot 0 of the run: a null view, for pages without a
//!   copy);
//! - the **faults**: the page numbers shaders found empty, read back here
//!   once a frame, so the copies exist from the next frame on. A pointer a
//!   shader takes out of memory cannot be known before it runs; the
//!   addresses a dispatch is given as root views are made ready before it.
//!
//! **A page that is not there when a shader touches it is a lost access**:
//! the read gives 0, the store is dropped, and the fault note only helps
//! the next frame. For the structure builder that is the same as a dropped
//! stage, so before a dispatch the host makes ready everything it can see
//! the dispatch name: the whole copy around each root view, around each
//! pair of root constants that is a title address and around each root
//! constant that is one in 256-byte units (the builder passes structures
//! all three ways), and the 4 MiB after each (a structure may cross a
//! window's end).
//! shortcut: what a stage reaches only through pointers it reads out of
//! memory (InitHeaders' table of structure addresses, LoadInstances'
//! instance records) is not made ready unless something else named it;
//! the fault count in the log says whether that happens. The whole answer
//! is the extent of the title's allocation around an address, which the
//! memory module of the layer knows and this one does not ask yet.
//!
//! shortcut: a copy that is numbered is kept for the rest of the run, and
//! there are at most [`COPIES`] of them (4 MiB windows mostly: 4 GiB at
//! most). Every dispatch of such a shader brings all numbered copies up to
//! date and, if it can store, declares all of them written (the bytes are
//! compared on the CPU afterwards, as for any unordered access view): a
//! dirty-page mask written by the shader would bound both.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use victory_shader::emit::{MEMORY_FAULTS, MEMORY_PAGES};
use windows::Win32::Graphics::Direct3D12::*;

use crate::descriptor::host::HostHeap;
use crate::host::{Host, HostList, host};
use crate::shadow::Shadow;

/// Copies that can be numbered (the run of heap slots is one longer). A
/// listed frame of the opening cutscene names about 50 windows.
const COPIES: u32 = 1023;
/// A window of memory outside buffers (`shadow.rs`).
const CELL: u64 = 4 << 20;
/// A directory page.
const PAGE: u64 = 1 << 16;

/// Newly published directory entries to copy into the GPU directory buffer in
/// queue order inside a list's prelude (`HostList::prelude`).
pub(crate) struct DirectoryPatch {
    pub(crate) directory: ID3D12Resource,
    pub(crate) upload: ID3D12Resource,
    /// `(dst_offset, src_offset, byte_len)` regions in `directory` / `upload`.
    pub(crate) regions: Vec<(u64, u64, u64)>,
}

struct Memory {
    /// Device-local GPU directory buffer and its GPU virtual address.
    directory: (ID3D12Resource, u64),
    /// Authoritative CPU copy of the directory (`MEMORY_PAGES` pairs of
    /// `(copy_number, offset)` dwords). Only updated on the CPU until a
    /// submission publishes deltas in GPU queue order.
    cpu_dir: Vec<u32>,
    /// Directory page indices written into `cpu_dir` since the last queue-
    /// ordered GPU publication.
    pending: Vec<u32>,
    faults: (ID3D12Resource, *mut u32, u64),
    /// Copy `n` is `copies[n - 1]`.
    copies: Vec<Arc<Shadow>>,
    /// Per host view heap (by address): the first slot of its run, and how
    /// many copies have their view there.
    heaps: HashMap<usize, (u32, u32)>,
    /// The frame the faults were last read in.
    frame: u64,
    /// Pages asked for this frame that are not the title's memory.
    refused: std::collections::HashSet<u64>,
}

// SAFETY: the fault buffer mapping is owned by `Memory`; all access is under
// the mutex.
unsafe impl Send for Memory {}

static MEMORY: Mutex<Option<Memory>> = Mutex::new(None);

/// What a draw binds.
pub(crate) struct Ready {
    pub(crate) directory: u64,
    pub(crate) faults: u64,
    /// The view heap slot of the null view; copy `n`'s view is `n` further.
    pub(crate) base: u32,
}

impl Memory {
    fn new() -> Option<Memory> {
        let host = host()?;
        // Committed with `D3D12_HEAP_FLAG_NONE`, so the host zeroes it on creation.
        let dir_bytes = u64::from(MEMORY_PAGES) * 8;
        let dir_buf = host.uav_buffer(dir_bytes)?;
        crate::host::name(&dir_buf, || "title memory by address: directory".to_owned());
        // SAFETY: a live buffer.
        let dir_gpu = unsafe { dir_buf.GetGPUVirtualAddress() };

        let fault_bytes = u64::from(MEMORY_FAULTS) * 4;
        let (fault_buf, fault_data) = host.shared_buffer(fault_bytes)?;
        crate::host::name(&fault_buf, || "title memory by address: faults".to_owned());
        // SAFETY: a fresh mapping of `fault_bytes`.
        unsafe { std::ptr::write_bytes(fault_data, 0, fault_bytes as usize) };
        // SAFETY: a live buffer.
        let fault_gpu = unsafe { fault_buf.GetGPUVirtualAddress() };

        tracing::info!("title memory by address: a shader reads or writes through raw addresses");
        Some(Memory {
            directory: (dir_buf, dir_gpu),
            cpu_dir: vec![0; MEMORY_PAGES as usize * 2],
            pending: Vec::new(),
            faults: (fault_buf, fault_data.cast::<u32>(), fault_gpu),
            copies: Vec::new(),
            heaps: HashMap::new(),
            frame: u64::MAX,
            refused: Default::default(),
        })
    }

    /// Publishes `(copy_num, offset)` for `page` into the CPU directory and
    /// queues the page for the next GPU queue-ordered directory delta.
    fn publish_page(&mut self, page: u64, copy_num: u32, offset: u32) {
        let idx = page as usize * 2;
        if self.cpu_dir[idx] == 0 {
            self.cpu_dir[idx] = copy_num;
            self.cpu_dir[idx + 1] = offset;
            self.pending.push(page as u32);
        }
    }

    /// Gives the page of `address` its copy, if it has none and the memory
    /// is the title's.
    fn map(&mut self, address: u64) {
        let page = address / PAGE;
        // The last entry stays empty: addresses past the directory read it.
        if page >= u64::from(MEMORY_PAGES) - 1 {
            return;
        }
        if self.cpu_dir[page as usize * 2] != 0 {
            return;
        }
        let start = page * PAGE;
        // A shader's pointer, or a constant taken for an address, may be
        // anything: only memory the title has.
        if self.refused.contains(&page) {
            return;
        }
        if !crate::texture::committed(start as *const u8, PAGE as usize) {
            self.refused.insert(page);
            return;
        }
        // The copy a root view at that address would get; a buffer that
        // ends inside the page is passed over for what holds all of it.
        let whole = |s: &Arc<Shadow>| s.offset_in(start).is_some_and(|_| s.left(start) >= PAGE);
        let Some(shadow) = crate::shadow::find(start, 1)
            .filter(whole)
            .or_else(|| crate::shadow::find(start, PAGE).filter(whole))
        else {
            return;
        };
        // SAFETY: a live buffer.
        let flags = unsafe { shadow.buffer.GetDesc() }.Flags;
        if flags.0 & D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 == 0 {
            return;
        }
        let n = match self.copies.iter().position(|c| Arc::ptr_eq(c, &shadow)) {
            Some(n) => n,
            None if self.copies.len() < COPIES as usize => {
                self.copies.push(shadow.clone());
                let n = self.copies.len() - 1;
                // Every page the copy holds whole, not only the one asked
                // for: a structure is more than 64 KiB.
                // shortcut: a page of a title buffer inside this one that
                // has a copy of its own is given this copy too.
                let (first, len) = shadow.span();
                let mut at = first.next_multiple_of(PAGE);
                while at + PAGE <= first + len && at / PAGE < u64::from(MEMORY_PAGES) - 1 {
                    self.publish_page(at / PAGE, n as u32 + 1, (at - first) as u32);
                    at += PAGE;
                }
                n
            }
            None => {
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    tracing::warn!(COPIES, "title memory by address: no number left for another copy; its pages read zeros")
                });
                return;
            }
        };
        let offset = shadow.offset_in(start).unwrap_or(0);
        self.publish_page(page, n as u32 + 1, offset as u32);
    }

    /// The pages shaders found empty since the last frame.
    fn take_faults(&mut self) {
        let mut pages = Vec::new();
        for i in 1..MEMORY_FAULTS as usize {
            // SAFETY: inside the faults' mapping; the GPU may be writing,
            // a number missed now is written again next frame.
            let at = unsafe { self.faults.1.add(i) };
            // SAFETY: as above.
            let page = unsafe { at.read_volatile() };
            if page != 0 {
                pages.push(page);
                // SAFETY: as above.
                unsafe { at.write_volatile(0) };
            }
        }
        if !pages.is_empty() {
            // Each is an access that read 0 or was dropped.
            static SAID: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
            if SAID.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 50 {
                tracing::warn!(
                    pages = pages.len(),
                    first = format_args!("{:#x}", u64::from(pages[0]) * PAGE),
                    "title memory by address: shaders touched pages that had no copy (those accesses read 0 or were dropped); the copies are made now"
                );
            }
        }
        for page in pages {
            self.map(u64::from(page) * PAGE);
        }
    }

    /// What a dispatch names: the copy around `address` and the 4 MiB
    /// after it.
    fn name(&mut self, address: u64) {
        self.map(address);
        self.map((address & !(CELL - 1)) + CELL);
    }

    /// Ensures every numbered copy in `self.copies` has its raw UAV installed
    /// in `heap`'s reserved dynamic run, returning the run's base slot.
    fn install_copies(&mut self, heap: &HostHeap) -> Option<u32> {
        let key = std::ptr::from_ref(heap) as usize;
        let (base, made) = match self.heaps.get(&key) {
            Some(&run) => run,
            None => {
                // Held for the heap's life: not given back with a list's slots.
                let base = heap.dynamic(COPIES + 1)?;
                for i in 0..=COPIES {
                    heap.null_raw_uav_at(base + i);
                }
                (base, 0)
            }
        };
        for (n, shadow) in self.copies.iter().enumerate().skip(made as usize) {
            let (_, len) = shadow.span();
            heap.raw_uav_at(base + 1 + n as u32, &shadow.buffer, 0, len);
        }
        self.heaps.insert(key, (base, self.copies.len() as u32));
        Some(base)
    }
}

/// Ensures `list` has `Shadow::refresh` (and `Shadow::gpu_writes` when any
/// draw on `list` writes through raw addresses) recorded for all copies in
/// `copies` not yet covered on `list`.
fn reconcile_list(copies: &[Arc<Shadow>], list: &HostList) {
    let (prev_reads, prev_writes, writes) = list.memory_prepared();
    for shadow in copies.iter().skip(prev_reads) {
        let (start, len) = shadow.span();
        shadow.refresh(list, start, len);
    }
    let mut new_writes = prev_writes;
    if writes {
        for shadow in copies.iter().skip(prev_writes) {
            let (start, len) = shadow.span();
            shadow.gpu_writes(list, start, len);
        }
        new_writes = copies.len();
    }
    list.set_memory_prepared(copies.len(), new_writes);
}

/// Before a draw whose shader reads title memory by address: what its
/// root arguments name gets its copies (`addresses`: its root views;
/// `constants`: its root constants, see the module), every numbered copy
/// has its view in `heap`'s run and is brought up to date when `list` is
/// submitted (and, with `writes`, copied back after). None if the host
/// cannot hold it; the draw is then dropped.
pub(crate) fn ready(
    list: &HostList,
    heap: &Arc<HostHeap>,
    addresses: &[u64],
    constants: &[u32],
    writes: bool,
) -> Option<Ready> {
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Memory::new();
    }
    let m = guard.as_mut()?;
    let frame = crate::frame::current();
    if m.frame != frame {
        m.frame = frame;
        m.refused.clear();
        m.take_faults();
    }
    for &address in addresses {
        m.name(address);
    }
    // A title address in two constants (low dword, then bits 32 to 39), or
    // in one in 256-byte units. A constant that only looks like one names
    // memory nothing reads; one that is not the title's memory is refused.
    for (i, &c) in constants.iter().enumerate() {
        if let Some(&high) = constants.get(i + 1)
            && (1..=0xFF).contains(&high)
            && c >= 0x1_0000
        {
            m.name(u64::from(c) | u64::from(high) << 32);
        }
        if c >= 0x0100_0000 {
            m.name(u64::from(c) << 8);
        }
    }
    let base = m.install_copies(heap)?;
    list.record_memory(heap.clone(), writes);
    reconcile_list(&m.copies, list);
    const KEEP: usize = usize::MAX - 0x6d;
    if list.first_use(KEEP) {
        list.keep(&m.directory.0);
        list.keep(&m.faults.0);
    }
    Some(Ready {
        directory: m.directory.1,
        faults: m.faults.2,
        base,
    })
}

/// Reconciles every memory-using command list in `runs` with all numbered copies
/// published so far (updating their view heaps and `refresh`/`gpu_writes`
/// coverage) and attaches any pending CPU-side directory delta to the first
/// memory-using list's prelude so it becomes visible on the GPU in queue order.
pub(crate) fn prepare_submission(host: &Host, runs: &[&HostList]) {
    let Some(first_list) = runs.iter().copied().find(|l| l.uses_memory()) else {
        return;
    };
    let mut guard = MEMORY.lock().unwrap_or_else(|e| e.into_inner());
    let Some(m) = guard.as_mut() else {
        return;
    };
    for &list in runs {
        let Some(heaps) = list.memory_heaps() else {
            continue;
        };
        for heap in heaps {
            let _ = m.install_copies(&heap);
        }
        reconcile_list(&m.copies, list);
    }
    if m.pending.is_empty() {
        return;
    }
    let mut pages = std::mem::take(&mut m.pending);
    pages.sort_unstable();
    pages.dedup();
    let total_bytes = (pages.len() * 8) as u64;
    let Some((upload, ptr)) = first_list.upload(host, total_bytes) else {
        m.pending = pages;
        return;
    };
    let out = ptr.cast::<u32>();
    let mut regions = Vec::new();
    let mut i = 0;
    while i < pages.len() {
        let start_page = pages[i];
        let mut end_idx = i + 1;
        while end_idx < pages.len() && pages[end_idx] == pages[end_idx - 1] + 1 {
            end_idx += 1;
        }
        let src_offset = (i * 8) as u64;
        for (k, &page) in pages[i..end_idx].iter().enumerate() {
            let dir_idx = page as usize * 2;
            let dst = (i + k) * 2;
            // SAFETY: `upload` is mapped for `pages.len() * 8` bytes.
            unsafe {
                out.add(dst).write(m.cpu_dir[dir_idx]);
                out.add(dst + 1).write(m.cpu_dir[dir_idx + 1]);
            }
        }
        regions.push((
            u64::from(start_page) * 8,
            src_offset,
            ((end_idx - i) * 8) as u64,
        ));
        i = end_idx;
    }
    first_list.set_memory_dir_patch(DirectoryPatch {
        directory: m.directory.0.clone(),
        upload,
        regions,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::Fxc::D3DCompile;
    use windows::Win32::Graphics::Direct3D::ID3DBlob;
    use windows_sys::Win32::System::Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE, VirtualAlloc, VirtualFree,
    };

    const PROBE_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), SRV(t0), UAV(u0), DescriptorTable(UAV(u0, space=1, numDescriptors=8))"
cbuffer Page : register(b0) {
    uint page;
};
ByteAddressBuffer directory : register(t0);
RWByteAddressBuffer report : register(u0);
RWByteAddressBuffer copies[8] : register(u0, space1);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint2 entry = directory.Load2(page * 8);
    report.Store(0, entry.x);
    report.Store(4, entry.y);
    if (entry.x != 0) {
        uint val = copies[entry.x].Load(entry.y);
        report.Store(8, val);
        copies[entry.x].Store(entry.y, val ^ 0x11111111u);
    }
}
"#;

    const DIR_ONLY_CS: &str = r#"
#define RS "RootConstants(num32BitConstants=1, b0), SRV(t0), UAV(u0)"
cbuffer Page : register(b0) {
    uint page;
};
ByteAddressBuffer directory : register(t0);
RWByteAddressBuffer report : register(u0);
[RootSignature(RS)]
[numthreads(1, 1, 1)]
void main() {
    uint2 entry = directory.Load2(page * 8);
    report.Store(0, entry.x);
    report.Store(4, entry.y);
}
"#;

    fn compile_cs(
        device: &ID3D12Device,
        source: &str,
    ) -> (ID3D12RootSignature, ID3D12PipelineState) {
        let mut code: Option<ID3DBlob> = None;
        let mut errors: Option<ID3DBlob> = None;
        // SAFETY: source is static ASCII HLSL.
        unsafe {
            D3DCompile(
                source.as_ptr().cast(),
                source.len(),
                windows::core::s!("memory_test"),
                None,
                None,
                windows::core::s!("main"),
                windows::core::s!("cs_5_1"),
                0,
                0,
                &mut code,
                Some(&mut errors),
            )
            .unwrap_or_else(|e| {
                let msg = errors
                    .map(|b| {
                        let bytes = std::slice::from_raw_parts(
                            b.GetBufferPointer().cast::<u8>(),
                            b.GetBufferSize(),
                        );
                        String::from_utf8_lossy(bytes).into_owned()
                    })
                    .unwrap_or_default();
                panic!("D3DCompile failed: {e}: {msg}");
            });
            let code = code.unwrap();
            let bytes = std::slice::from_raw_parts(
                code.GetBufferPointer().cast::<u8>(),
                code.GetBufferSize(),
            );
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

    struct TitlePage(*mut u8, usize);

    impl TitlePage {
        fn new(id: usize) -> Self {
            // `MEMORY_PAGES` covers title addresses below 128 GiB (`0x20_0000_0000`).
            // Plain `VirtualAlloc(NULL, ...)` on 64-bit Windows typically lands above 2 TiB,
            // so probe for a free 64 KiB page inside the title-addressable range.
            let mut ptr = std::ptr::null_mut();
            let limit = u64::from(MEMORY_PAGES - 1) * PAGE;
            let mut cand = 0x10_0000_0000u64;
            while cand + PAGE <= limit {
                // SAFETY: probing an explicit aligned address with MEM_RESERVE | MEM_COMMIT.
                let got = unsafe {
                    VirtualAlloc(
                        cand as *const _,
                        PAGE as usize,
                        MEM_COMMIT | MEM_RESERVE,
                        PAGE_READWRITE,
                    )
                };
                if !got.is_null() {
                    ptr = got;
                    break;
                }
                cand += 1 << 20;
            }
            assert!(!ptr.is_null(), "failed to allocate TitlePage below 128 GiB");
            assert_eq!((ptr as usize) % (PAGE as usize), 0);
            crate::shadow::register(id, ptr as u64, PAGE);
            Self(ptr.cast(), id)
        }

        fn addr(&self) -> u64 {
            self.0 as u64
        }

        fn write_u32(&self, value: u32) {
            // SAFETY: within the 64 KiB allocation.
            unsafe { self.0.cast::<u32>().write_volatile(value) };
        }

        fn read_u32(&self) -> u32 {
            // SAFETY: within the 64 KiB allocation.
            unsafe { self.0.cast::<u32>().read_volatile() }
        }
    }

    impl Drop for TitlePage {
        fn drop(&mut self) {
            crate::shadow::unregister(self.1, self.0 as u64);
            // SAFETY: allocated by VirtualAlloc above.
            unsafe { VirtualFree(self.0.cast(), 0, MEM_RELEASE) };
        }
    }

    #[test]
    fn closed_list_reconciles_later_copies_and_publishes_directory_in_queue_order() {
        let Some(host) = crate::host::host() else {
            return;
        };
        let _ = host.log_messages();
        let h1 = Arc::new(HostHeap::new(false, 64).expect("host view heap 1"));
        let h2 = Arc::new(HostHeap::new(false, 64).expect("host view heap 2"));
        let p1 = TitlePage::new(0x9370_0001);
        let p2 = TitlePage::new(0x9370_0002);
        let p3 = TitlePage::new(0x9370_0003);
        p1.write_u32(0x1234_5678);
        p2.write_u32(0xDEAD_BEEF);
        p3.write_u32(0xCAFE_BABE);

        let (probe_root, probe_pso) = compile_cs(&host.device, PROBE_CS);
        let (dir_root, dir_pso) = compile_cs(&host.device, DIR_ONLY_CS);
        let (report_buf, report_ptr) = host.shared_buffer(64).expect("report buffer");
        // SAFETY: live mapped buffer and device.
        let (report_gpu, step) = unsafe {
            std::ptr::write_bytes(report_ptr, 0, 64);
            (
                report_buf.GetGPUVirtualAddress(),
                host.device
                    .GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV),
            )
        };

        // List A records on heap H1 when only p1's copy exists, and closes.
        let list_a = host.new_list().expect("list_a");
        let ready_a = ready(&list_a, &h1, &[p1.addr()], &[], true).expect("ready_a");
        // SAFETY: live heap, root sig, PSO, and buffers kept alive through execute().
        unsafe {
            let h1_gpu = h1.heap.GetGPUDescriptorHandleForHeapStart();
            let table = D3D12_GPU_DESCRIPTOR_HANDLE {
                ptr: h1_gpu.ptr + u64::from(ready_a.base) * u64::from(step),
            };
            list_a.list.SetDescriptorHeaps(&[Some(h1.heap.clone())]);
            list_a.list.SetComputeRootSignature(&probe_root);
            list_a.list.SetPipelineState(&probe_pso);
            list_a
                .list
                .SetComputeRoot32BitConstant(0, (p2.addr() / PAGE) as u32, 0);
            list_a
                .list
                .SetComputeRootShaderResourceView(1, ready_a.directory);
            list_a.list.SetComputeRootUnorderedAccessView(2, report_gpu);
            list_a.list.SetComputeRootDescriptorTable(3, table);
            list_a.list.Dispatch(1, 1, 1);
        }
        list_a.close();

        // List B records on heap H2 and numbers p2's copy, without submitting.
        let list_b = host.new_list().expect("list_b");
        let _ready_b = ready(&list_b, &h2, &[p2.addr()], &[], true).expect("ready_b");

        // Submit ONLY List A (which closed before p2 was numbered on H2).
        host.execute(&[&list_a]);
        list_a.reset();

        // SAFETY: GPU execution finished; reading mapped report buffer.
        let (seen_copy, seen_val) = unsafe {
            let r = report_ptr.cast::<u32>();
            (r.read_volatile(), r.add(2).read_volatile())
        };
        assert!(
            seen_copy != 0,
            "p2 directory entry must be published before List A runs"
        );
        assert_eq!(
            seen_val, 0xDEAD_BEEF,
            "List A must refresh p2 and install its UAV in H1"
        );
        assert_eq!(
            p2.read_u32(),
            0xDEAD_BEEF ^ 0x1111_1111,
            "List A must copy back shader writes to p2"
        );
        assert_eq!(
            host.log_messages(),
            None,
            "D3D12 debug layer found an error"
        );

        // Now test queue-ordered isolation: List C numbers p3 without submitting,
        // and an independent GPU read of the live directory must NOT see p3 yet.
        let list_c = host.new_list().expect("list_c");
        let ready_c = ready(&list_c, &h2, &[p3.addr()], &[], false).expect("ready_c");
        let probe_list = host.new_list().expect("probe_list");
        // SAFETY: reads the GPU directory resource without going through memory::ready.
        unsafe {
            std::ptr::write_bytes(report_ptr, 0, 64);
            probe_list.list.SetComputeRootSignature(&dir_root);
            probe_list.list.SetPipelineState(&dir_pso);
            probe_list
                .list
                .SetComputeRoot32BitConstant(0, (p3.addr() / PAGE) as u32, 0);
            probe_list
                .list
                .SetComputeRootShaderResourceView(1, ready_c.directory);
            probe_list
                .list
                .SetComputeRootUnorderedAccessView(2, report_gpu);
            probe_list.list.Dispatch(1, 1, 1);
        }
        probe_list.close();
        host.execute(&[&probe_list]);
        probe_list.reset();

        // SAFETY: GPU execution finished; reading mapped report buffer.
        let before_submit_copy = unsafe { report_ptr.cast::<u32>().read_volatile() };
        assert_eq!(
            before_submit_copy, 0,
            "unsubmitted p3 mapping must not be visible in the GPU directory yet"
        );

        // Once List C itself is submitted, its prelude publishes p3 to the GPU directory.
        list_c.close();
        host.execute(&[&list_c]);
        list_c.reset();

        let probe_after = host.new_list().expect("probe_after");
        // SAFETY: reads the GPU directory after List C executed.
        unsafe {
            std::ptr::write_bytes(report_ptr, 0, 64);
            probe_after.list.SetComputeRootSignature(&dir_root);
            probe_after.list.SetPipelineState(&dir_pso);
            probe_after
                .list
                .SetComputeRoot32BitConstant(0, (p3.addr() / PAGE) as u32, 0);
            probe_after
                .list
                .SetComputeRootShaderResourceView(1, ready_c.directory);
            probe_after
                .list
                .SetComputeRootUnorderedAccessView(2, report_gpu);
            probe_after.list.Dispatch(1, 1, 1);
        }
        probe_after.close();
        host.execute(&[&probe_after]);
        probe_after.reset();

        // SAFETY: GPU execution finished.
        let after_submit_copy = unsafe { report_ptr.cast::<u32>().read_volatile() };
        assert_ne!(
            after_submit_copy, 0,
            "p3 must be published in GPU queue order when List C submits"
        );
        list_b.close();
        list_b.reset();
        assert_eq!(
            host.log_messages(),
            None,
            "D3D12 debug layer found an error"
        );
    }
}
