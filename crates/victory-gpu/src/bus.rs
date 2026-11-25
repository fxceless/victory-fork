//! What the card reads from the PC's memory over the bus, and the buffers
//! of ours that the CPU writes and shaders read moved into the card's
//! memory (`VICTORY_GPU_UPLOAD`).
//!
//! Run n12 against n11: 2,536 MB of windows outside buffers moved from the
//! PC's memory to the card's (`VICTORY_CARD_WINDOWS_MB=4096`) took the
//! card's time from 122.8 to 104.7 ms a frame. Shaders that read the PC's
//! memory wait for the bus on every line they fetch, and a draw of a few
//! hundred vertices costs 13-35 us that way. This module counts, per frame,
//! the bytes draws bind by where they lie ([`line`]), so that what is left
//! in the PC's memory has a number.
//!
//! What it counts is what lists ask for: a host copy's range each time a
//! list asks it to be brought up to date (`Shadow::refresh`: vertex and
//! index buffers, root views, tables' views, indirect arguments, title
//! buffer copies), and the bytes of the lists' scratch memory handed out
//! (constants filled from title memory, rectangles' corners). Shaders may
//! read less (a root view is asked for to its copy's end) or read the same
//! bytes many times (four shadow cascades); the count is what was bound.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

/// Where the bytes a draw binds lie.
pub(crate) mod kind {
    /// Host copies in the card's memory (title buffers, windows).
    pub const CARD: usize = 0;
    /// Host copies of title buffers in the PC's memory (shared buffers,
    /// write-back pages the card reads through the bus), and buffer views'
    /// copies of their own bytes.
    pub const PC_BUFFER: usize = 1;
    /// Host copies of windows outside buffers in the PC's memory.
    pub const PC_WINDOW: usize = 2;
    /// Title memory in place (`VICTORY_INPLACE`): the PC's memory.
    pub const IN_PLACE: usize = 3;
    /// Lists' scratch in the upload heap (the PC's memory).
    pub const SCRATCH_PC: usize = 4;
    /// Lists' scratch in the card's memory ([`super::gpu_upload_asked`]).
    pub const SCRATCH_CARD: usize = 5;
    /// Host copies in a GPU upload heap (`crate::shadow::gpu_upload_copies`):
    /// the card's memory.
    pub const CARD_BAR: usize = 6;
    pub const ALL: usize = 7;
}

/// Since the last frame line, per [`kind`]: bytes bound and the times.
static BYTES: [AtomicU64; kind::ALL] = [const { AtomicU64::new(0) }; kind::ALL];
static CALLS: [AtomicU64; kind::ALL] = [const { AtomicU64::new(0) }; kind::ALL];

/// Since the last frame line: draws and dispatches that bound a heap's
/// table of element strides (shaders that pick buffers from the view heap
/// by an index they compute read it), with the table in the PC's memory
/// and in the card's.
static STRIDES: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the start: buffers of ours made in the card's memory the CPU
/// writes ([`gpu_upload_asked`]) and their bytes; buffers asked for there
/// and made in the upload heap after all.
pub(crate) static MADE: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

/// Of [`MADE`], alive now: scratch chunks and their bytes.
pub(crate) static ALIVE: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Whether the buffers of ours that the CPU writes and shaders only read,
/// and that every draw of a kind reads again, are made in the card's own
/// memory, mapped for the CPU through the card's window on the bus (a GPU
/// upload heap: resizable BAR), instead of the upload heap in the PC's
/// memory: `VICTORY_GPU_UPLOAD=1` (off by default; read when a buffer is
/// made, so it takes a fresh start). Where the card or the runtime has no
/// such heap the upload heap is used as before, said once.
///
/// What it moves:
/// - the lists' scratch chunks (`crate::host` `take_chunk`: 4 MiB, and 256
///   KiB for a list's first): the constant views of tables filled from
///   title memory at submission (13,211 fills a frame in run n12's
///   gameplay), root constant views not 256-byte aligned in their copy,
///   rectangles' corners; and with `VICTORY_CARD_NO_PC_COPY=1` the bytes
///   going up to copies in the card's memory;
/// - the view heap's table of element strides (8 MiB, one dword per slot),
///   which every shader that picks buffers from the heap by a computed
///   index reads for each buffer it picks.
///
/// What the CPU does with them is unchanged: it writes them, through a
/// write-combined mapping as before, and never reads them back. The card's
/// rule for upload memory holds for this heap too: what the CPU wrote
/// before a submission is what the submission's shaders read.
///
/// What it costs: the chunks' bytes in the card's memory (170 MB alive in
/// run n12, of which 88 MB kept spare), taken from the textures' room; the
/// CPU's writes cross the bus as before.
///
/// shortcut: on this PC (RTX 5070 Ti, resizable BAR on) the driver puts
/// the upload heap in the card's memory already: [`tests::probe_heaps`]
/// saw 256 MiB, 64 x 4 MiB and 256 x 256 KiB of upload buffers all counted
/// in the card's own segment, none in the PC's. So here the switch only
/// says so explicitly; it matters where a driver keeps upload memory in
/// the PC's. What the card does read through the bus are the shared
/// buffers (custom heaps, write-back, pool L0): host copies with no card
/// side, and card sides' CPU halves (`crate::shadow::gpu_upload_copies`).
pub(crate) fn gpu_upload_asked() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_GPU_UPLOAD").is_some_and(|v| v == "1"))
}

/// `bytes` of memory of `kind` bound by a list.
pub(crate) fn bound(kind: usize, bytes: u64) {
    BYTES[kind].fetch_add(bytes, Relaxed);
    CALLS[kind].fetch_add(1, Relaxed);
}

/// A draw or dispatch bound a table of element strides, in the card's
/// memory or not.
pub(crate) fn strides_bound(card: bool) {
    STRIDES[usize::from(card)].fetch_add(1, Relaxed);
}

/// The counts of [`line`] for `frames` frames, taken: per [`kind`] (bytes,
/// times) per frame, then the strides tables bound in the PC's memory and
/// in the card's per frame.
fn take(frames: f64) -> ([(f64, f64); kind::ALL], [f64; 2]) {
    let frames = frames.max(1.0);
    let by = std::array::from_fn(|k| {
        (BYTES[k].swap(0, Relaxed) as f64 / frames, CALLS[k].swap(0, Relaxed) as f64 / frames)
    });
    let strides = [0, 1].map(|i| STRIDES[i].swap(0, Relaxed) as f64 / frames);
    (by, strides)
}

/// With the frame line: what lists bound per frame by where it lies.
pub(crate) fn line(frames: f64) {
    let (by, strides) = take(frames);
    let mb = |k: usize| by[k].0 / 1e6;
    let n = |k: usize| by[k].1;
    let pc = mb(kind::PC_BUFFER) + mb(kind::PC_WINDOW) + mb(kind::IN_PLACE) + mb(kind::SCRATCH_PC);
    let card = mb(kind::CARD) + mb(kind::CARD_BAR) + mb(kind::SCRATCH_CARD);
    tracing::info!(
        "what lists bound per frame by where it lies (bytes asked for, a root view to its copy's end): in the PC's memory \
         {pc:.1} MB [title buffers' copies {:.1} MB in {:.0}, windows outside buffers {:.1} MB in {:.0}, title memory in place \
         {:.1} MB in {:.0}, scratch in the upload heap {:.2} MB in {:.0}, strides tables {:.0}x]; in the card's memory \
         {card:.1} MB [copies {:.1} MB in {:.0}, copies in a GPU upload heap {:.1} MB in {:.0} (VICTORY_GPU_UPLOAD_COPIES={}), \
         scratch {:.2} MB in {:.0} (VICTORY_GPU_UPLOAD: {}), strides tables {:.0}x]",
        mb(kind::PC_BUFFER),
        n(kind::PC_BUFFER),
        mb(kind::PC_WINDOW),
        n(kind::PC_WINDOW),
        mb(kind::IN_PLACE),
        n(kind::IN_PLACE),
        mb(kind::SCRATCH_PC),
        n(kind::SCRATCH_PC),
        strides[0],
        mb(kind::CARD),
        n(kind::CARD),
        mb(kind::CARD_BAR),
        n(kind::CARD_BAR),
        crate::shadow::gpu_upload_copies().0,
        mb(kind::SCRATCH_CARD),
        n(kind::SCRATCH_CARD),
        if gpu_upload_asked() { "on" } else { "off" },
        strides[1],
    );
}

/// For the memory line: the buffers of ours in the card's memory that the
/// CPU writes ([`gpu_upload_asked`]).
pub(crate) fn memory_text() -> String {
    let mb = |v: u64| v as f64 / (1u64 << 20) as f64;
    let [made, made_bytes, refused] = MADE.each_ref().map(|m| m.load(Relaxed));
    let [chunks, chunk_bytes] = ALIVE.each_ref().map(|a| a.load(Relaxed));
    let [copies, copy_bytes, written] = crate::shadow::bar_copies();
    format!(
        "buffers in the card's memory the CPU writes (GPU upload heaps): {made} made {:.0} MB since the start, {refused} made \
         elsewhere instead; host copies there (VICTORY_GPU_UPLOAD_COPIES={}) {copies} alive {:.0} MB, {:.1} MB of their \
         ranges declared written by shaders since the start (read back through the bus); scratch chunks there \
         (VICTORY_GPU_UPLOAD: {}) {chunks} alive {:.0} MB",
        mb(made_bytes),
        crate::shadow::gpu_upload_copies().0,
        mb(copy_bytes),
        mb(written),
        if gpu_upload_asked() { "on" } else { "off" },
        mb(chunk_bytes),
    )
}

#[cfg(test)]
mod tests {
    /// The line's counts are per frame and taken.
    #[test]
    fn counts_are_per_frame_and_taken() {
        super::bound(super::kind::PC_WINDOW, 4_000_000);
        super::bound(super::kind::PC_WINDOW, 2_000_000);
        super::strides_bound(true);
        let (by, strides) = super::take(2.0);
        assert!(by[super::kind::PC_WINDOW].0 >= 3_000_000.0);
        assert!(by[super::kind::PC_WINDOW].1 >= 1.0);
        assert!(strides[1] >= 0.5);
    }

    /// What a buffer costs the system's commit, this process's commit and
    /// the card's budget, by heap: the card's own (a copy's card side),
    /// shared with the PC (a copy's CPU side), the upload heap and a GPU
    /// upload heap. Asks the real card (256 MiB each, one at a time):
    /// `cargo test -p victory-gpu --lib probe_heaps -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn probe_heaps() {
        let Some(host) = crate::host::host() else {
            return;
        };
        const SIZE: u64 = 256 << 20;
        type Seen = (i64, [u64; 3], [u64; 4]);
        let seen = || -> Seen {
            (
                crate::memline::system_commit().map_or(0, |(used, _)| used) as i64,
                crate::memline::process_memory(),
                host.video_memory().unwrap_or_default(),
            )
        };
        let mb = |a: u64, b: u64| (a as i64 - b as i64) >> 20;
        let say = |what: &str, before: Seen| {
            let now = seen();
            println!(
                "{what}: system commit {:+} MB, ours private {:+} MB, ours shared commit {:+} MB; the card's own in use {:+} MB,                  the PC's shared with it in use {:+} MB",
                (now.0 - before.0) >> 20,
                mb(now.1[0], before.1[0]),
                mb(now.1[1], before.1[1]),
                mb(now.2[1], before.2[1]),
                mb(now.2[3], before.2[3]),
            );
        };
        let before = seen();
        let own = host.uav_buffer(SIZE);
        say("default heap", before);
        drop(own);
        let before = seen();
        let shared = host.shared_buffer(SIZE);
        say("custom write-back L0", before);
        drop(shared);
        let before = seen();
        let upload = host.upload_buffer(SIZE);
        say("upload heap", before);
        drop(upload);
        // The sizes lists' scratch takes: where does the driver put them?
        for (what, size, n) in [("upload heap, 64 x 4 MiB", 4u64 << 20, 64), ("upload heap, 256 x 256 KiB", 256 << 10, 256)] {
            let before = seen();
            let many: Vec<_> = (0..n).map(|_| host.upload_buffer(size)).collect();
            say(what, before);
            drop(many);
        }
        // SAFETY: set before anything reads it, on this test's own thread.
        unsafe { std::env::set_var("VICTORY_GPU_UPLOAD", "1") };
        let before = seen();
        let written = host.cpu_written_buffer(SIZE);
        say(&format!("GPU upload heap (in the card's memory: {:?})", written.as_ref().map(|w| w.2)), before);
        if let Some((_, data, _)) = &written {
            let started = std::time::Instant::now();
            // SAFETY: the buffer's mapping, written only.
            unsafe { std::ptr::write_bytes(*data, 1, 64 << 20) };
            println!("64 MiB written through its mapping in {:.1} ms", started.elapsed().as_secs_f64() * 1e3);
        }
        drop(written);
        let before = seen();
        let shared = host.card_shared_buffer(SIZE);
        say(&format!("GPU upload heap, unordered access (made: {})", shared.is_some()), before);
        if let Some((_, data)) = &shared {
            let started = std::time::Instant::now();
            // SAFETY: the buffer's mapping.
            unsafe { std::ptr::write_bytes(*data, 1, 64 << 20) };
            let wrote = started.elapsed().as_secs_f64() * 1e3;
            let mut back = vec![0u8; 4 << 20];
            let started = std::time::Instant::now();
            // SAFETY: inside the mapping.
            unsafe { std::ptr::copy_nonoverlapping(*data, back.as_mut_ptr(), back.len()) };
            let read = started.elapsed().as_secs_f64() * 1e3;
            assert!(back.iter().all(|&b| b == 1));
            println!("64 MiB written in {wrote:.1} ms, 4 MiB read back in {read:.1} ms");
        }
    }

    /// What host lists cost the process's private bytes on this card's
    /// driver: lists made and left closed, then each recording `copies`
    /// small buffer copies, then reset and recording one (does an
    /// allocator keep what its biggest recording took?), then let go; with
    /// the private allocations of 1 GiB the walk of a run shows (3 of them,
    /// 2 GB committed, owner unknown) counted at each step:
    /// `cargo test -p victory-gpu --lib probe_list_memory -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn probe_list_memory() {
        use windows::Win32::Graphics::Direct3D12::*;
        use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEM_FREE, MEM_PRIVATE, MEMORY_BASIC_INFORMATION, VirtualQuery};
        let Some(host) = crate::host::host() else {
            return;
        };
        // Private allocations of 1 GiB and their committed bytes.
        let giants = || -> (usize, u64) {
            let mut whole = std::collections::HashMap::<usize, (usize, u64)>::new();
            let mut at = 0usize;
            loop {
                // SAFETY: plain data filled in by the query.
                let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                // SAFETY: a valid out pointer of the right size.
                if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
                    break;
                }
                if m.State != MEM_FREE && m.Type == MEM_PRIVATE {
                    let e = whole.entry(m.AllocationBase as usize).or_default();
                    e.0 += m.RegionSize;
                    if m.State == MEM_COMMIT {
                        e.1 += m.RegionSize as u64;
                    }
                }
                let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
                if next <= at {
                    break;
                }
                at = next;
            }
            let big: Vec<_> = whole.values().filter(|e| e.0 == 1 << 30).collect();
            (big.len(), big.iter().map(|e| e.1).sum())
        };
        let private = || crate::memline::process_memory()[0] as i64;
        let say = |what: &str, before: i64| {
            let (n, committed) = giants();
            println!(
                "{what}: private {:+} MB; private allocations of 1 GiB: {n}, {} MB committed",
                (private() - before) >> 20,
                committed >> 20
            );
        };
        const LISTS: usize = 256;
        let start = private();
        say("before", start);
        let src = host.uav_buffer(1 << 20).expect("a buffer");
        let dst = host.uav_buffer(1 << 20).expect("a buffer");
        // SAFETY: plain object creation and recording on live objects.
        unsafe {
            let made: Vec<(ID3D12CommandAllocator, ID3D12GraphicsCommandList)> = (0..LISTS)
                .map(|_| {
                    let a: ID3D12CommandAllocator = host.device.CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT).unwrap();
                    let l: ID3D12GraphicsCommandList =
                        host.device.CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, &a, None).unwrap();
                    l.Close().unwrap();
                    (a, l)
                })
                .collect();
            say(&format!("{LISTS} allocators and lists made, closed"), start);
            for copies in [1_000u64, 10_000] {
                let before = private();
                for (a, l) in &made {
                    a.Reset().unwrap();
                    l.Reset(a, None).unwrap();
                    for i in 0..copies {
                        l.CopyBufferRegion(&dst, (i % 4096) * 256, &src, (i % 4096) * 256, 256);
                    }
                    l.Close().unwrap();
                }
                say(&format!("each recorded {copies} copies (from the step before)"), before);
            }
            let before = private();
            for (a, l) in &made {
                a.Reset().unwrap();
                l.Reset(a, None).unwrap();
                l.CopyBufferRegion(&dst, 0, &src, 0, 256);
                l.Close().unwrap();
            }
            say("each reset and recorded one copy (from the step before)", before);
            drop(made);
        }
        say("let go (from the start)", start);
        // Where the card's own memory shows in the address space: the
        // write-combined private bytes before and after a buffer of 256 MiB
        // in a default heap.
        let combined = || -> u64 {
            use windows_sys::Win32::System::Memory::PAGE_WRITECOMBINE;
            let mut bytes = 0u64;
            let mut at = 0usize;
            loop {
                // SAFETY: as above.
                let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
                // SAFETY: as above.
                if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
                    break;
                }
                if m.State == MEM_COMMIT && m.Type == MEM_PRIVATE && m.Protect & PAGE_WRITECOMBINE != 0 {
                    bytes += m.RegionSize as u64;
                }
                let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
                if next <= at {
                    break;
                }
                at = next;
            }
            bytes
        };
        let (before, wc) = (private(), combined());
        let own = host.uav_buffer(256 << 20);
        println!(
            "a default heap buffer of 256 MiB (made: {}): private {:+} MB, write-combined private {:+} MB",
            own.is_some(),
            (private() - before) >> 20,
            (combined() as i64 - wc as i64) >> 20
        );
    }
}
