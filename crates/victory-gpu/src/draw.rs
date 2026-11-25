//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// Whether the title's recording calls are timed on its recording threads
/// (`VICTORY_RECORD_TIMES=1`; off by default), for the frame-rate line's
/// "further:" part ([`record_line`]): every list call that goes through a
/// list's draw state, the draws (mesh dispatches too) and dispatches
/// (indirect ones too), and of them the binding ([`State::bind`]), its
/// descriptor tables ([`State::tables`]) and the vertex and index buffers.
///
/// Why: some 4,700 draws a frame are recorded on the title's five or six
/// recording threads, beside the submitting thread; whether their cost
/// stands in the frame's path (the title's own time between submissions)
/// or idles beside it, nothing measured in release builds.
fn record_times() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RECORD_TIMES").is_some_and(|v| v == "1"))
}

/// The parts of [`RECORD`]: counts, and nanoseconds summed over threads.
pub(crate) mod rec {
    pub const CALLS: usize = 0;
    pub const CALLS_NS: usize = 1;
    pub const DRAWS: usize = 2;
    pub const DRAWS_NS: usize = 3;
    pub const DISPATCHES: usize = 4;
    pub const DISPATCHES_NS: usize = 5;
    pub const BIND_NS: usize = 6;
    pub const TABLES_NS: usize = 7;
    pub const INPUTS_NS: usize = 8;
    /// Waiting for a list's draw state lock (another thread holds it).
    pub const LOCK_NS: usize = 9;
    pub const ALL: usize = 10;
}

/// [`record_times`]'s sums since the last frame-rate line.
static RECORD: [AtomicU64; rec::ALL] = [const { AtomicU64::new(0) }; rec::ALL];

thread_local! {
    /// This thread's sums not yet in [`RECORD`], and how many adds since
    /// they went there (a few atomics per draw on six threads would share
    /// their cache lines).
    static RECORD_HERE: std::cell::Cell<([u64; rec::ALL], u32)> = const { std::cell::Cell::new(([0; rec::ALL], 0)) };
}

fn record_add(i: usize, n: u64) {
    RECORD_HERE.with(|here| {
        let (mut v, mut adds) = here.get();
        v[i] += n;
        adds += 1;
        if adds >= 256 {
            for (sum, n) in RECORD.iter().zip(v.iter_mut()) {
                if *n != 0 {
                    sum.fetch_add(std::mem::take(n), Ordering::Relaxed);
                }
            }
            adds = 0;
        }
        here.set((v, adds));
    });
}

/// A timed stretch of [`record_times`]: its time goes to `ns` (and one to
/// `count`, if any) when it is dropped.
pub(crate) struct RecordTimed(Option<(std::time::Instant, usize, Option<usize>)>);

impl RecordTimed {
    pub(crate) fn new(ns: usize, count: Option<usize>) -> Self {
        Self(record_times().then(|| (std::time::Instant::now(), ns, count)))
    }
}

impl Drop for RecordTimed {
    fn drop(&mut self) {
        if let Some((started, ns, count)) = self.0 {
            record_add(ns, started.elapsed().as_nanos() as u64);
            if let Some(count) = count {
                record_add(count, 1);
            }
        }
    }
}

/// With [`record_times`]: the time a list call waited for its draw state.
pub(crate) fn record_lock(started: Option<std::time::Instant>) {
    if let Some(started) = started {
        record_add(rec::LOCK_NS, started.elapsed().as_nanos() as u64);
    }
}

/// When a list call starts, with [`record_times`].
pub(crate) fn record_start() -> Option<std::time::Instant> {
    record_times().then(std::time::Instant::now)
}

/// For the frame-rate line's "further:" part.
pub(crate) fn record_line(frames: f64) -> String {
    if !record_times() || frames <= 0.0 {
        return "recording (VICTORY_RECORD_TIMES=1 times it): off".to_owned();
    }
    let v: Vec<u64> = RECORD.iter().map(|n| n.swap(0, Ordering::Relaxed)).collect();
    let ms = |i: usize| v[i] as f64 / 1e6 / frames;
    let n = |i: usize| v[i] as f64 / frames;
    let us_each = |ns: usize, count: usize| if v[count] == 0 { 0.0 } else { v[ns] as f64 / 1e3 / v[count] as f64 };
    format!(
        "recording on the title's threads (VICTORY_RECORD_TIMES=1, all threads summed): {:.0} list calls {:.1} ms \
         (waiting for a list's state {:.1}); {:.0} draws {:.1} ms ({:.2} us each), {:.0} dispatches {:.1} ms ({:.2} us \
         each); of draws and dispatches: binding {:.1} ms (tables {:.1}), vertex and index buffers {:.1}",
        n(rec::CALLS),
        ms(rec::CALLS_NS),
        ms(rec::LOCK_NS),
        n(rec::DRAWS),
        ms(rec::DRAWS_NS),
        us_each(rec::DRAWS_NS, rec::DRAWS),
        n(rec::DISPATCHES),
        ms(rec::DISPATCHES_NS),
        us_each(rec::DISPATCHES_NS, rec::DISPATCHES),
        ms(rec::BIND_NS),
        ms(rec::TABLES_NS),
        ms(rec::INPUTS_NS),
    )
}

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// While `State::early_textures` runs the texture pass: the
    /// milliseconds and the bytes that run may take, in place of the
    /// pass's own per frame.
    static PASS_BUDGET: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
}

/// The milliseconds of every frame the pass that uploads textures picked
/// by index takes before a shader that picks by index has asked for it
/// (`VICTORY_EARLY_TEXTURES_MS`, 12; 0: never, as before), and the
/// milliseconds it may take just before a draw of the title's decoration
/// bake (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special).
///
/// The pass ran only from the first world frame on, where it met some
/// 24,000 slots at once and needed 60-80 frames for them. The title bakes
/// its characters' skin and decorations into textures of its own while it
/// loads, 159-1,165 frames before that (runs 156-174), with a pixel shader
/// that picks its sources by index: whatever was not on the card then was
/// baked as nothing, once, for the run. On the console a texture is there
/// when its bytes are.
fn early_budgets() -> (u64, u64) {
    static MS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (ms("VICTORY_EARLY_TEXTURES_MS", 12), ms("VICTORY_BAKE_TEXTURES_MS", 1500))
    })
}

/// The pixel shaders of the title's decoration bake (the title:
/// `PS_BakeDecorations_*`), by the first 16 hex digits of their hash as the
/// frame listings print it. They draw into a 256x256 corner of 2048x2048
/// targets and name no texture in a table.
const BAKE_PIXEL_SHADERS: [&str; 4] =
    ["c0a7fe77b1862e10", "0d1575a396dfe513", "b7cfd8dda91ebb6b", "280967ad158cee0b"];

/// Since the start: early runs of the texture pass, and those of them
/// made for a bake draw.
pub(crate) static EARLY_TEXTURE_PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours and no barrier (the scratch
/// buffer stays readable as arguments until the next clamp), until one of
/// these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and the title names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
///
/// Why windows may hand a draw a stale record (two device losses
/// upstream, one a page fault): a record in a window is refreshed and
/// clamped where the window starts, and a later draw that reads it there
/// refreshes nothing. Between the two, the list may change it in ways that
/// end no window: a title copy into the buffer (copied at submission, in
/// the list's order, after the window's refresh: the host copy
/// keeps the bytes from before), a copy of title memory
/// (`CopyMemory`), a title copy recorded on the card
/// (`VICTORY_TABLE_GPU_COPY=2`). A record from before (last frame's, its
/// first index or instance far from this frame's data) is in range for the
/// clamp, and a shader that fetches through root views by those numbers
/// reads outside its buffer. The runs of `VICTORY_INDIRECT_BATCH` clamp
/// only calls the title made, each refreshed where it is recorded; with
/// them on, windows are off.
///
/// Off for the rest of the run from the first error the card's driver
/// reports to the system (`crate::drvwatch`): every loss of the card in a
/// run with windows was announced that way seconds ahead.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_some_and(|v| v == "1")) && !crate::drvwatch::tripped()
}

/// The waits around indirect mesh draws launched by the amplification stage
/// (`VICTORY_MESH_AMPLIFY=1`), `VICTORY_MESH_AMPLIFY_WAIT`:
/// - `1` (default): one wait where a run of them starts (for the title's
///   argument passes) and one where it ends, before the next dispatch or draw
///   that may write ([`State::amplified_reads_done`]);
/// - `0`: no wait where a run ends (as 2b2a36d);
/// - `2`: a wait before every such draw as well (to tell an ordering fault
///   from anything else; the clamp pass it replaced drained the card there).
///
/// The console's command processor reads an indirect record when it reaches
/// the draw, in order with what follows, so the title may write a record's
/// buffer (its next culling pass) right after the draw with no wait; the
/// clamp pass copied each record before its draw and drained the card behind
/// it. The amplification stage reads the record while the draw runs: without
/// the wait at the end of a run, a dispatch after the draws may rewrite the
/// records (or their count) under draws still reading them.
pub(crate) fn amplify_wait() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| {
        std::env::var("VICTORY_MESH_AMPLIFY_WAIT").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(1).min(2)
    })
}

/// Whether a record that is not on the window's walk opens a window of its
/// own at once, with the step the title was last seen to walk that
/// argument buffer by (`VICTORY_WINDOW_JUMP=1`; `alt`: twenty seconds
/// with, twenty without, in turn, each change said in the log). Off by
/// default. Needs [`windows`].
///
/// Run 243 counted, per frame, 2,370 indirect commands read from a window
/// and 2,370 clamped by a pass of ours; 960 of those passes were for
/// records "off the window's walk": the title draws a few records one
/// after the other and then jumps to another place in the same argument
/// buffer, some 480 times a frame. The record after a jump was clamped
/// alone (its step is not known from one record), and the one after it
/// opened a window of sixteen: two passes a jump, each one a trip of the
/// card from drawing to compute and back (9 microseconds in the game).
///
/// With the switch the step of the last window that draws did read from
/// is remembered with its argument buffer, kind and stride, and a record
/// that would be clamped alone on that buffer opens sixteen records at
/// that step instead: what the next record would have opened anyway, one
/// record earlier, so no more of the list's budget is spent than before.
/// If no draw reads from a window opened that way before the next pass,
/// the step is forgotten until a walk shows it again: a title that draws
/// lone records all over a buffer gets what it had.
fn jump_windows() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_WINDOW_JUMP").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "a record off the window's walk opens a window at the remembered step from now (VICTORY_WINDOW_JUMP=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// The records of a window opened at a remembered step ([`jump_windows`]):
/// what the record after a lone one opens.
const WINDOW_JUMP: u32 = 16;

/// What a record that would be clamped alone opens instead, if the step
/// its argument buffer is walked by is remembered: that step and how many
/// records (never past the buffer, of which `left` bytes are left from
/// the record on). None if nothing is remembered for this buffer, kind
/// and stride, or no second record fits.
fn jump_window(remembered: Option<(u64, u32, u32, u32)>, (copy, kind, stride): (u64, u32, u32), left: u64) -> Option<(u32, u32)> {
    let (c, k, s, step) = remembered?;
    if (c, k, s) != (copy, kind, stride) || step == 0 {
        return None;
    }
    let fit = (left / u64::from(step)).min(u64::from(WINDOW_JUMP)) as u32;
    (fit > 1).then_some((step, fit))
}

/// Diagnostics of [`windows`]: indirect commands since the last line, by
/// what became of their clamp. In order: read from a window (no pass of
/// ours); then clamped by a pass of ours because: it is a dispatch; it
/// asks for more than one record; it has a count buffer; its shaders can
/// store to buffers; it is the list's first; there was no window on its
/// argument buffer (another buffer, kind or stride, or the window was
/// ended); its record does not continue the window's walk; it is past the
/// window's end (the next window: as meant). And the records those passes
/// clamped in all; the passes that opened a window at a remembered step
/// ([`jump_windows`]), and those of them no draw read from.
#[cfg(feature = "diagnostics")]
pub(crate) static WINDOW_WHY: [std::sync::atomic::AtomicU64; 12] = [const { std::sync::atomic::AtomicU64::new(0) }; 12];

/// The line of [`WINDOW_WHY`], per frame over `frames` frames.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_line(frames: f64) {
    let n: Vec<f64> = WINDOW_WHY.iter().map(|n| n.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames.max(1.0)).collect();
    tracing::info!(
        "indirect commands per frame by their clamp: {:.0} read from a window (no pass of ours); a pass of ours for {:.0}: {:.0} dispatches, {:.0} of more than one record, {:.0} with a count \
         buffer, {:.0} whose shaders can store to buffers, {:.0} first on their list, {:.0} with no window on their argument buffer, {:.0} off the window's walk, {:.0} past the window's end; \
         {:.0} records clamped by those passes; {:.0} of the passes opened a window at a remembered step (VICTORY_WINDOW_JUMP=1), {:.0} of those windows were read by no draw",
        n[0],
        n[1..9].iter().sum::<f64>(),
        n[1],
        n[2],
        n[3],
        n[4],
        n[5],
        n[6],
        n[7],
        n[8],
        n[9],
        n[10],
        n[11],
    );
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    targets_set: bool,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    depth_bounds: Option<[f32; 2]>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// Indirect mesh draws launched by their pipeline's amplification stage
    /// (`VICTORY_MESH_AMPLIFY=1`) since the list last waited for every
    /// earlier shader write, with nothing between them that writes: the next
    /// such draw needs no wait of its own ([`State::amplified_indirect`]).
    amplified_run: bool,
    /// Indirect mesh draws launched by the amplification stage were recorded
    /// since the list last waited for every shader write: their stage may
    /// still read their records ([`State::amplified_reads_done`]).
    amplified_reads: bool,
    /// The run of indirect calls whose records one pass of ours clamps
    /// (`VICTORY_INDIRECT_BATCH`, `crate::indirect::batched`).
    batch: Option<crate::indirect::Run>,
    /// A dispatch was recorded since the list last waited for every shader
    /// write before an indirect call (a clamp pass, a run's pass, a wait
    /// of [`crate::indirect::RunRules::order`] or the title's barrier): the
    /// next call that joins a run waits for it first (`VICTORY_INDIRECT_BATCH=3`).
    dispatched: bool,
    /// With [`jump_windows`]: the argument buffer (its copy's number), kind
    /// and stride of the last window of several records that a draw read
    /// from, and that window's step.
    walk_step: Option<(u64, u32, u32, u32)>,
    /// Whether [`State::window`] was opened at the remembered step, and
    /// whether a draw has read from it since.
    window_jumped: bool,
    window_read: bool,
    /// Diagnostics: the argument buffer and the offset of the indirect
    /// command before this one on the list.
    #[cfg(feature = "diagnostics")]
    last_indirect: (u64, u64),
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// The dynamic slots the list holds: title heap, first, count; given
    /// back when the list has run, is reset or released.
    dynamic: Vec<(usize, u32, u32)>,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, first, n);
        }
    }
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        for (heap, first, n) in self.dynamic.drain(..) {
            crate::descriptor::release_dynamic(heap, first, n);
        }
        self.fresh_tables.clear();
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.tables_gone = true;
    }

    /// The run of indirect calls clamped together, if any, is over
    /// ([`crate::indirect::batch_mode`]).
    pub(crate) fn end_batch(&mut self, list: &HostList) {
        crate::indirect::end_run(list, self.batch.take());
    }

    /// With runs ending at anything that may write (`VICTORY_INDIRECT_BATCH`
    /// 1, 3, 4): the bound pipeline's shaders can store to buffers, so the
    /// run ends.
    fn end_batch_if_stores(&mut self, list: &HostList) {
        if self.batch.is_some() && crate::indirect::run_rules().stores && self.pipeline_stores() {
            self.end_batch(list);
        }
    }

    /// Whether the bound pipeline's shaders can store to buffer memory
    /// (one without a host pipeline runs nothing, so writes nothing).
    fn pipeline_stores(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This)
            .is_some_and(|p| p.stages.iter().any(|stage| stage.translated().stores))
    }

    /// The title's barrier, barrier group or pipeline sync: with runs
    /// ending at those (`VICTORY_INDIRECT_BATCH=2`), the run ends.
    /// `waits`: a wait for every shader write is recorded for it (a
    /// barrier group, `crate::list`'s `barrier_groups`), so the dispatches
    /// before it need no wait of ours (3).
    pub(crate) fn title_barrier(&mut self, list: &HostList, waits: bool) {
        let rules = crate::indirect::run_rules();
        if rules.title_barriers {
            self.end_batch(list);
        }
        if rules.order && waits {
            self.dispatched = false;
        }
    }
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. The title's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// Of those passes, for the frame line: nanoseconds walking the slots
/// (`mirror` and `buffer_copy` per slot), building the spans and deferring
/// their refreshes, and in [`State::heap_textures`]; slots walked; passes
/// that walked every slot, and their nanoseconds walking.
pub(crate) static HEAP_BUFFER_SPLIT: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
/// Of the walk, for the frame line: nanoseconds checking slots against what
/// was resolved for them before ([`slot_cache`]), in `mirror`, and in
/// `buffer_copy` (the host copy's lookup); slots kept from before, slots
/// resolved, and of those the ones that had something kept that failed the
/// check; slots that took 100 us or more to resolve (a lock another thread
/// held, or a host copy made) and their nanoseconds; and nanoseconds of the
/// spans' part building them (the rest is deferring the refreshes).
pub(crate) static HEAP_WALK_SPLIT: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// Whether the pass over the heap's buffer views keeps what it resolved
/// for a slot (host copy and range) from pass to pass, and looks again
/// only where that may have changed: `VICTORY_HEAP_SLOT_CACHE=1` (read
/// once), off by default. Without it every slot the pass walks (the ones
/// written since the last pass, and the rescan's: a sixteenth of some
/// 83,000 a pass with `VICTORY_HEAP_ROLLING_RESCAN=1`) is resolved anew:
/// `mirror` (two locks, one of them the global table of views) and the
/// host copy's lookup (the global table of buffers, or of windows), 1 to
/// 6 us a slot, 5-51 ms a pass. With it a slot keeps its resolution while
/// - the record in the title's slot is the one it was resolved from,
/// - that record still names a view (a table entry's resources were not
///   released),
/// - and the host slot is still made from that record and still views the
///   host copy kept for it;
///
/// checked for all walked slots under one lock of each table. The host slot
/// is what shaders read, and it is made again only when the record changes,
/// so the copy it views is the one to bring up to date; a slot that fails
/// any check is resolved as before. As a further net every slot is
/// resolved anew once every 256 passes ([`deep_rescan`]) whatever its
/// checks say. Differs from resolving anew only where the lookup would now
/// find another copy than the host slot views (counted as views elsewhere
/// in diagnostics builds): that copy is refreshed then, not the slot's.
fn slot_cache() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_SLOT_CACHE").is_some_and(|v| v == "1"))
}

/// Whether the heap pass number `pass` resolves slot `index` anew whatever
/// was kept for it ([`slot_cache`]): once every 256 passes for each slot,
/// both when a sixteenth is rescanned each pass (slot `index` is walked
/// at passes with `pass % 16 == index % 16`) and when all slots are walked
/// every 16th pass.
fn deep_rescan(index: u32, pass: u64) -> bool {
    u64::from(index / 16 % 16) == pass / 16 % 16
}

/// A hasher for the heap pass's maps of slot numbers and of copies (by
/// address), each hashed tens of thousands of times a pass: a rotate, xor
/// and multiply per word (rustc's FxHasher), with the result rotated so
/// that the low bits a table indexes by are not the low bits of an aligned
/// address times an odd number (always zero). Any order of the maps'
/// entries is as good as another: the spans are minima and maxima.
#[derive(Default, Clone, Copy)]
struct SlotHasher(u64);

impl std::hash::Hasher for SlotHasher {
    fn finish(&self) -> u64 {
        self.0.rotate_left(26)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0xf135_7aea_2e62_a9c5);
    }
}

type SlotMap<K, V> = std::collections::HashMap<K, V, std::hash::BuildHasherDefault<SlotHasher>>;

/// What the heap pass resolved for a buffer view's slot: the host copy and
/// title range it reads, the record words it was resolved from (`None`:
/// not one of ours, never kept), and the pass that last resolved or
/// checked it.
struct Resolved {
    shadow: std::sync::Arc<crate::shadow::Shadow>,
    from: u64,
    to: u64,
    words: Option<[u32; 2]>,
    pass: u64,
    /// An unordered access view (shaders may write through it).
    uav: bool,
}

/// Whether what shaders write through buffer views they pick from the heap
/// by index is declared as written (`VICTORY_PICKED_UAV_WRITES=1`, read
/// once; off by default): in every list that binds a stage that can store
/// through such a view (`crate::pipeline::Translated::picked_stores`),
/// every title range behind an unordered access buffer view of the heap
/// (as the last heap pass found them, [`PICKED_UAVS`]) is declared
/// written, once per list, as a table's unordered access view is
/// (`readable`). The relay's reads then wait for those flights, and their
/// fetches bring those writes back, instead of the writes staying in the
/// host copy. Without it they are only counted ([`PICKED_STATS`]).
/// shortcut: the whole set, not the slots the shader's computed index
/// reaches (nothing tells which); and the set of the frame's heap pass,
/// so views written into the heap since are not in it.
pub(crate) fn picked_uav_writes() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var_os("VICTORY_PICKED_UAV_WRITES").is_some_and(|v| v == "1");
        if on {
            tracing::warn!(
                "VICTORY_PICKED_UAV_WRITES=1: every list running a shader that can store through a buffer view it picks \
                 from the heap declares every unordered access buffer view of the heap written"
            );
        }
        on
    })
}

/// The title ranges behind the heap's unordered access buffer views, as
/// the last heap pass that measured them found them: per host copy, the
/// views' ranges joined where they overlap or touch (title memory in place
/// left out: nothing to declare there).
static PICKED_UAVS: std::sync::Mutex<Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>> =
    std::sync::Mutex::new(Vec::new());

/// Of the heap's unordered access buffer views, as the last heap pass that
/// measured them found them (kept, not reset by the frame line): views,
/// host copies, bytes of their joined ranges ([`PICKED_UAVS`]), bytes of
/// the span from the first to the last byte per copy, and of the joined
/// bytes those in copies in the card's memory, in GPU upload heaps
/// (`VICTORY_GPU_UPLOAD_COPIES`), and in title memory in place (nothing to
/// declare); passes that measured them (since the start); joined ranges.
pub(crate) static PICKED_SET: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// Since the last frame line: draws and dispatches binding a stage that
/// can store through a buffer view picked by index (`picked_stores`),
/// lists they ran in, ranges and bytes declared for them
/// ([`picked_uav_writes`]), draws and dispatches of stages that pick
/// buffer views by index but store through none, and lists that declared
/// nothing because the set was over `VICTORY_PICKED_UAV_WRITES_MB`.
pub(crate) static PICKED_STATS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The key [`picked_uav_use`] notes in a list (`HostList::first_use`) that
/// binds a stage that can store through buffer views it picks from the
/// heap: such a list's flight may write copies where it declared nothing
/// (`crate::cpuwait::queue2_picked`).
pub(crate) const PICKED_STORE_KEY: usize = usize::MAX - 0x5f;

/// A stage of the draw being bound that picks views from the heap by
/// index: counted, and with `picked` (it can store through them) and
/// [`picked_uav_writes`], the heap's unordered access buffer views declared
/// written in `list` (once per list).
fn picked_uav_use(list: &HostList, picked: bool, writes: bool) {
    if !picked {
        PICKED_STATS[4].fetch_add(1, Ordering::Relaxed);
        return;
    }
    PICKED_STATS[0].fetch_add(1, Ordering::Relaxed);
    // One key per list, as the heap pass has.
    const ONCE: usize = PICKED_STORE_KEY;
    if !list.first_use(ONCE) {
        return;
    }
    PICKED_STATS[1].fetch_add(1, Ordering::Relaxed);
    if !picked_uav_writes() || !writes {
        return;
    }
    // `VICTORY_PICKED_UAV_WRITES_MB=<n>`: a set of more than n MB is not
    // declared at all (counted; 0 or unset: no limit), a way out should
    // the set be too large to compare or fetch per flight.
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_PICKED_UAV_WRITES_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .filter(|&mb| mb > 0)
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let set = PICKED_UAVS.lock().unwrap_or_else(|e| e.into_inner()).clone();
    let bytes: u64 = set.iter().map(|(_, from, to)| to - from).sum();
    if bytes > limit {
        PICKED_STATS[5].fetch_add(1, Ordering::Relaxed);
        return;
    }
    for (shadow, from, to) in &set {
        shadow.gpu_writes(list, *from, to - from);
    }
    PICKED_STATS[2].fetch_add(set.len() as u64, Ordering::Relaxed);
    PICKED_STATS[3].fetch_add(bytes, Ordering::Relaxed);
}

/// Whether what a program writes through views it picks from the heap is
/// declared as written, view by view, from the dispatch's own arguments
/// (`VICTORY_PICKED_EXACT=1`, read once; off by default: the views are
/// still worked out and their memory noted, for the counters).
///
/// The 37 programs that store through picked views (`bvh_skinning`:
/// skinned vertex and velocity streams, cable, rope, tree animation, and
/// `CSUnpackIB*`: index buffers) declare nothing, so their writes stay on
/// the card: no flight holds those ranges (refreshes and merges put CPU
/// bytes over them while later flights write them), no fetch brings them
/// back (title memory, and with it a card copy's CPU side, keeps bytes from
/// whenever a declared range last happened to cover their piece), and
/// title buffer copies and fills run on the CPU read those bytes. How much
/// of that happens depends on when flights land: the garbage that comes
/// and goes with timing (bottom bars, blocks by the camera, steps on the
/// skinned shoulder, whose velocity stream these programs write).
///
/// Their heap index is simple (`victory_shader::picked`): for
/// `CSUnpackIB*` a root constant times 32; for the others a 24-bit field
/// (byte 60) of each job record `[c0, c0 + c1)` of a structured buffer in
/// the dispatch's table (`t117`, `c0`/`c1` root constants), times 32. With
/// the switch each view so found is made ready as a table's unordered
/// access view is ([`readable`]: refreshed, and declared written), once
/// per list, in place of the whole heap's set (`VICTORY_PICKED_UAV_WRITES`):
/// the relay, the refreshes, the fetches and the copy waits then treat
/// those writes as every declared write. A program whose slots the
/// analysis could not follow keeps the old way.
/// shortcut: the job records are read from title memory when the dispatch
/// is recorded (the title writes them before; a job buffer written by the
/// card would be read stale).
pub(crate) fn picked_exact() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PICKED_EXACT").is_some_and(|v| v == "1"))
}

/// Of [`picked_exact`], since the last frame line: draws and dispatches
/// whose picked slots were worked out, ones the analysis could not follow,
/// job records read, slots found (per dispatch, joined), slots whose view
/// is no buffer view, views declared written (new in their list), their
/// bytes, and dispatches whose records ran past [`PICKED_RECORDS_MOST`].
pub(crate) static PICKED_EXACT: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
mod exact_stat {
    pub const FOLLOWED: usize = 0;
    pub const NOT_FOLLOWED: usize = 1;
    pub const RECORDS: usize = 2;
    pub const SLOTS: usize = 3;
    pub const NO_BUFFER: usize = 4;
    pub const DECLARED: usize = 5;
    pub const DECLARED_BYTES: usize = 6;
    pub const CAPPED: usize = 7;
}

/// Job records read for one dispatch at most.
const PICKED_RECORDS_MOST: u64 = 1 << 14;

/// A draw binds `[address, address + size)` of `shadow` as a vertex or
/// index buffer: its bytes that the picked-view programs wrote in another
/// copy are counted (`crate::shadow::skinned::DRAWN_ELSEWHERE`).
fn drawn_skinned(shadow: &crate::shadow::Shadow, address: u64, size: u32) {
    let n = crate::shadow::skinned_elsewhere(address, address + u64::from(size), shadow);
    if n > 0 {
        crate::shadow::SKINNED_OVER[crate::shadow::skinned::DRAWN_ELSEWHERE].fetch_add(n, Ordering::Relaxed);
    }
}

/// The frame line of [`picked_exact`] and of what the host did with the
/// memory those programs write (`crate::shadow::SKINNED_OVER`).
pub(crate) fn picked_exact_line(frames: f64) {
    use exact_stat::*;
    let n: [f64; 8] = std::array::from_fn(|i| PICKED_EXACT[i].swap(0, Ordering::Relaxed) as f64 / frames);
    let o: [f64; 6] = std::array::from_fn(|i| crate::shadow::SKINNED_OVER[i].swap(0, Ordering::Relaxed) as f64 / frames);
    let r = crate::shadow::SKINNED_READERS.each_ref().map(|n| n.load(Ordering::Relaxed));
    let (ranges, bytes) = crate::shadow::skinned_known();
    tracing::info!(
        "views picked from the heap, slot by slot (VICTORY_PICKED_EXACT={}), per frame: {:.0} draws and dispatches \
         followed ({:.0} not followed, {:.0} past the record cap), {:.0} job records read, {:.0} slots, {:.0} of them no \
         buffer view; {:.0} views {:.2} MB declared written. Memory those programs write ({ranges} ranges {:.1} MB known), \
         per frame: {:.3} MB sent up over it from copies' CPU sides, {:.3} MB refreshed over it in copies without a card \
         side, {:.1} KB of it read from title memory by title copies on the CPU, {:.1} KB by fills, {:.3} MB merged into \
         title memory from fetches, {:.3} MB of it bound as vertex or index buffers in another copy than the one written; \
         heap views that read it at the last measure: {}, {} of them showing another copy ({:.2} MB)",
        u8::from(picked_exact()),
        n[FOLLOWED],
        n[NOT_FOLLOWED],
        n[CAPPED],
        n[RECORDS],
        n[SLOTS],
        n[NO_BUFFER],
        n[DECLARED],
        n[DECLARED_BYTES] / 1e6,
        bytes as f64 / 1e6,
        o[crate::shadow::skinned::SENT] / 1e6,
        o[crate::shadow::skinned::PC] / 1e6,
        o[crate::shadow::skinned::COPY_READ] / 1024.0,
        o[crate::shadow::skinned::FILL_READ] / 1024.0,
        o[crate::shadow::skinned::MERGED] / 1e6,
        o[crate::shadow::skinned::DRAWN_ELSEWHERE] / 1e6,
        r[0],
        r[1],
        r[2] as f64 / 1e6,
    );
}

/// Joins the heap pass's unordered access ranges (copy, from, to) per copy
/// and keeps them as [`PICKED_UAVS`] and [`PICKED_SET`].
fn picked_uav_set(mut uavs: Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)>) {
    let views = uavs.len() as u64;
    uavs.sort_unstable_by_key(|(s, from, _)| (std::sync::Arc::as_ptr(s) as usize, *from));
    let mut joined: Vec<(std::sync::Arc<crate::shadow::Shadow>, u64, u64)> = Vec::new();
    // From the first to the last byte, per copy.
    let mut spanned = 0u64;
    let mut first = 0u64;
    for (shadow, from, to) in uavs {
        match joined.last_mut() {
            Some(last) if std::sync::Arc::ptr_eq(&last.0, &shadow) => {
                if from <= last.2 {
                    last.2 = last.2.max(to);
                } else {
                    joined.push((shadow, from, to));
                }
            }
            _ => {
                if let Some(last) = joined.last() {
                    spanned += last.2 - first;
                }
                first = from;
                joined.push((shadow, from, to));
            }
        }
    }
    if let Some(last) = joined.last() {
        spanned += last.2 - first;
    }
    let (mut bytes, mut card, mut bar, mut placed, mut copies) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut previous: Option<usize> = None;
    for (shadow, from, to) in &joined {
        let len = to - from;
        bytes += len;
        if shadow.is_placed() {
            placed += len;
        } else if shadow.carded() {
            card += len;
        } else if shadow.bar_mapped() {
            bar += len;
        }
        let key = std::sync::Arc::as_ptr(shadow) as usize;
        if previous != Some(key) {
            copies += 1;
            previous = Some(key);
        }
    }
    joined.retain(|(shadow, _, _)| !shadow.is_placed());
    for (i, n) in [views, copies, bytes, spanned, card, bar, placed].into_iter().enumerate() {
        PICKED_SET[i].store(n, Ordering::Relaxed);
    }
    PICKED_SET[7].fetch_add(1, Ordering::Relaxed);
    PICKED_SET[8].store(joined.len() as u64, Ordering::Relaxed);
    // Kept only to be declared: the set holds its copies alive.
    if picked_uav_writes() {
        *PICKED_UAVS.lock().unwrap_or_else(|e| e.into_inner()) = joined;
    }
}
/// Of [`State::heap_textures`], for the frame line: nanoseconds taking the
/// queue (remapped ranges, due slots, census), asking what of placed
/// textures is mapped, making host slots and textures (`mirror`), staging
/// and recording uploads (`readable`), and checking what was staged and
/// handing slots back; slots looked at; passes stopped by the time budget,
/// and by the byte budget.
pub(crate) static HEAP_TEXTURE_SPLIT: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
/// Of the texture side's queue and looks, for the frame line: nanoseconds
/// taking the remapped ranges, taking the due slots and the census, and
/// letting go of untaken worker items; slots left at the workers' cap
/// without a look, and worker items taken without a look
/// (`VICTORY_CHEAP_LOOKS`).
pub(crate) static HEAP_LOOK_SPLIT: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Whether the pass over textures picked by index, with texture workers
/// (`VICTORY_ASYNC_TEXTURES`), skips asking what is mapped of a slot when
/// the answer would not be used now: `VICTORY_CHEAP_LOOKS=1` (read once),
/// off by default. A slot no job can be queued for (the workers' cap) is
/// left due without the look: before, it was asked (some 35 us a slot,
/// 475 slots a frame in the opening cutscene) and left due all the same,
/// unless nothing or no more of it was mapped, when it waited 1, 2, 4, ...
/// frames instead, or was made null if streamed out; with the switch that
/// waits until the cap lets a job be queued for it. A slot whose texture a
/// worker made ready is uploaded with what the worker found mapped (it
/// asked just before it read), without asking again; what was mapped or
/// unmapped since is seen at the slot's next look, as is what is mapped
/// after the upload now.
fn cheap_looks() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_CHEAP_LOOKS").is_some_and(|v| v == "1"))
}

/// The time of a pass split into phases: [`Phases::to`] adds the time
/// since the last change to the phase that ran, and starts `next`.
struct Phases<'a> {
    counters: &'a [AtomicU64],
    phase: usize,
    since: std::time::Instant,
}

impl<'a> Phases<'a> {
    fn new(counters: &'a [AtomicU64], phase: usize) -> Self {
        Phases { counters, phase, since: std::time::Instant::now() }
    }

    fn to(&mut self, next: usize) {
        let now = std::time::Instant::now();
        self.counters[self.phase].fetch_add((now - self.since).as_nanos() as u64, Ordering::Relaxed);
        (self.phase, self.since) = (next, now);
    }
}
/// Table entries shaders picked by a root constant in tables made from
/// the title's own slots (not fresh), since the last frame line.
pub(crate) static TABLE_INDEXED: AtomicU64 = AtomicU64::new(0);

/// Whether a draw makes the table entries its shaders pick by a root
/// constant in tables made from the title's own slots
/// (`VICTORY_TABLE_INDEXED=1`; off by default: they are made only when the
/// title writes them, by its view calls or the heap sync).
pub(crate) fn table_indexed() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_TABLE_INDEXED").is_some_and(|v| v == "1"))
}

/// Of [`HEAP_BUFFER_MICROS`], the microseconds of the pass over textures
/// picked by index that runs inside it (`heap_textures`), and the buffer
/// slots the pass over buffer views looked at; since the last frame line.
pub(crate) static HEAP_PASS_PARTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);

/// Frames after a whole upload of a texture picked by index at which its
/// memory is looked at again, each counted from the look before.
const RECHECK_FRAMES: [u64; 5] = [4, 12, 40, 120, 400];

/// Looks at uploaded textures picked by index that found the memory as it
/// was, and that found it changed (uploaded again), since the start.
pub(crate) static HEAP_TEXTURES_RECHECKED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Uploads of textures already on the card made while it had no room for
/// new ones, since the start.
pub(crate) static HEAP_TEXTURES_REUPLOADED_FULL: AtomicU64 = AtomicU64::new(0);

/// Whether a pass over the textures picked by index on a full card holds
/// back the slots that need a new host copy (`VICTORY_FULL_CARD_PASS=0`:
/// no, as before: they stay due, each pass looks at the same ones again
/// and stops at the 257th). Held back, such a slot is looked at again
/// after about four seconds or as soon as the card has room, and the pass
/// goes on to the slots behind it: the ones whose host copy exists and
/// only wants bringing up to date, and the ones whose memory went (letting
/// their host copies go is what makes room). What is uploaded while the
/// card has room is the same either way.
fn full_card_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FULL_CARD_PASS").is_none_or(|v| v != "0"))
}

/// Whether a texture that has its host copy is still brought up to date
/// while the card has no room for new ones (`VICTORY_REUPLOAD_WHEN_FULL=0`:
/// no).
fn reupload_when_full() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REUPLOAD_WHEN_FULL").is_none_or(|v| v != "0"))
}

/// Whether textures picked by index are looked at again after their upload
/// (`VICTORY_HEAP_RECHECK=0`: no).
fn heap_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK").is_none_or(|v| v != "0"))
}
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: 1,452 of the 2,021 mesh vertex
/// streams of a cutscene frame reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some(gpu);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize, crate::host::FILL_ROOT);
    Some(gpu)
}

/// With [`crate::host::table_on_card`]: the host address a constant view of
/// a descriptor table (`size` bytes of title memory at `address`) is bound
/// at in its buffer's host copy, as an aligned root view is ([`root_view`]),
/// instead of a copy in the list's upload memory filled from title memory
/// at submission. The fill reads on the CPU what an earlier flight's
/// shaders may have written (GPU-driven constants), and a submission with
/// such a fill settles the flights first (`Host::submit`, "title memory
/// read"); the copy holds what they wrote, and the card reads it when it
/// runs the draw, after them. `None` (the fill, as before) unless the
/// range lies in one title buffer whose copy no other copy overlaps
/// (`crate::shadow::sole_buffer_copy`), in the card's memory at `1`, the
/// view's host address is 256-byte aligned and its rounded size lies in
/// the copy.
fn table_constant_on_card(list: &HostList, address: u64, size: u32) -> Option<u64> {
    use crate::host::table_card as t;
    let mode = crate::host::table_on_card();
    if mode == 0 || size == 0 {
        return None;
    }
    let len = u64::from(size);
    let shadow = match crate::shadow::sole_buffer_copy(address, len) {
        Ok(shadow) => shadow,
        Err(crate::shadow::NotSole::NoBuffer) => {
            t::count(t::NO_BUFFER);
            return None;
        }
        Err(crate::shadow::NotSole::OtherCopy) => {
            t::count(t::OTHER_COPY);
            return None;
        }
    };
    if mode == 1 && !shadow.carded() {
        t::count(t::NOT_CARDED);
        return None;
    }
    let gpu = shadow.gpu(address);
    if gpu % 256 != 0 || shadow.left(address) < len.next_multiple_of(256) {
        t::count(t::UNALIGNED);
        return None;
    }
    shadow.refresh(list, address, len);
    list.note_table_on_card(address as usize, size as usize);
    t::count(t::BOUND);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// A shader takes the address of a buffer view from its table and the
/// host finds none (it reads 0): logged once per shader, slot and dword,
/// with what the view is (the first 64 such). The ray-tracing programs
/// take their structures' addresses this way; one that reads 0 walks
/// nothing (its first node read finds no title memory there) and every
/// ray it traces misses.
fn no_address(pipeline: This, slot: u8, dword: u32, view: Option<&Option<View>>) {
    static SEEN: std::sync::Mutex<Vec<(usize, u8, u32)>> = std::sync::Mutex::new(Vec::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let key = (pipeline as usize, slot, dword);
    if seen.len() >= 64 || seen.contains(&key) {
        return;
    }
    seen.push(key);
    drop(seen);
    let why = match view {
        Some(view) => crate::descriptor::host::why_no_address(view.as_ref()),
        None => "the root argument is not a descriptor table".to_owned(),
    };
    tracing::warn!(
        shaders = %crate::pipeline::hashes(pipeline),
        slot,
        dword,
        why,
        "a shader takes the address of a buffer view the host has no buffer for: it reads 0"
    );
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        self.targets.clear();
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            crate::queue::note_written(record.resource);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            rtvs.push(host.rtv(Some(texture), record.desc().as_ref()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            self.depth_state = depth_state(texture, record.depth_desc().as_ref());
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, record.depth_desc().as_ref()))
        });
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some([min, max]);
        crate::draw::depth_bounds(list, min, max);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    pub(crate) fn replay(&mut self, list: &HostList) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            self.set_targets(list, &handles, self.depth_handle);
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
        if let Some([min, max]) = self.depth_bounds {
            crate::draw::depth_bounds(list, min, max);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{USER_MEMORY_BASE, USER_VIEW_ADDRESS, USER_VIEW_BIAS};
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        // The title address of the buffer view at that dword of the table.
        let (address, view) = match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => {
                let view = crate::descriptor::view(table + u64::from(dword) * 4);
                let address = view
                    .as_ref()
                    .and_then(crate::descriptor::host::buffer_range)
                    .map_or(0, |range| range.0);
                (address, Some(view))
            }
            _ => (0, None),
        };
        if address == 0 {
            once!(slot, dword, "a shader takes the address of a buffer view that is not there: it reads 0");
            no_address(self.pipeline as This, slot, dword, view.as_ref());
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    fn memory_ready(&self, list: &HostList, compute: bool, writes: bool) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, heap, &addresses, &constants, writes)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        let inputs = RecordTimed::new(rec::INPUTS_NS, None);
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                continue;
            };
            drawn_skinned(&shadow, v.address, v.size);
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                return false;
            };
            drawn_skinned(&shadow, v.address, v.size);
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        drop(inputs);
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let _timed = RecordTimed::new(rec::BIND_NS, None);
        let l = &list.list;
        // Textures picked by index, before anything asks for them.
        self.early_textures(list, p, compute);
        if crate::host::walls_trace() {
            self.walls_census(p, compute);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            list.keep(&p.pso);
            list.keep(&p.root);
        }
        list.prof_bound(p.pso.as_raw() as usize, &p.name);
        list.bound_writes(p.stages.iter().any(|stage| stage.translated().writes));
        for stage in &p.stages {
            let t = stage.translated();
            STORES.set(t.stores);
            let made = {
                let _timed = RecordTimed::new(rec::TABLES_NS, None);
                self.tables(list, &t.used, &t.host.root_indexed_views, compute)
            };
            STORES.set(true);
            if made.iter().any(|m| m.1 == u32::MAX) {
                once!("a draw whose descriptor table found no free slots is dropped");
                return false;
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
            // Buffer views it may write that it picks from the heap by
            // index: no table or root view names them.
            if t.picked_stores || t.host.root_srvs.contains(&victory_shader::emit::SLOT_STRIDES) {
                // Slot by slot ([`picked_exact`]): declared here, and then
                // not as the heap's whole set.
                let exact = t.picked_stores && self.picked_slots(list, t, compute, writes && picked_exact());
                picked_uav_use(list, t.picked_stores, writes && !exact);
            }
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
                let Some(ready) = self.memory_ready(list, compute, writes) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    return false;
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                // On the stack: a root signature holds 64 words at most
                // (this was an allocation per stage of every draw).
                let mut on_stack = [0u32; 64];
                let mut on_heap = Vec::new();
                let values: &mut [u32] = if dwords as usize <= on_stack.len() {
                    &mut on_stack[..dwords as usize]
                } else {
                    on_heap.resize(dwords as usize, 0);
                    &mut on_heap
                };
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory {
                            // SAFETY: an open list; the directory lives as
                            // long as the process and the list holds it.
                            unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, m.directory);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory);
                                }
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list) {
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => once!("a shader picks buffers from a view heap that is not set"),
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory {
                        // SAFETY: an open list; the buffer lives as long as
                        // the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, m.faults);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                            }
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            s.refresh(list, a, len);
            if writes {
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    return false;
                };
                self.dynamic.push((key.0, index, 1));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.refresh(list, *a, s.left(*a));
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        crate::bus::strides_bound(host.strides_on_card());
        // One key per list, as textures have: no title resource is there.
        const ONCE: usize = usize::MAX - 0x5e;
        if !list.first_use(ONCE) {
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran up to 51 times a
        // frame over 81,000 views and 4,100 host copies, 18 s of every
        // frame (2026-11-24, run 11).
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late.
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) == frame {
            return Some(gpu);
        }
        let started = std::time::Instant::now();
        // Per host heap: the host copy and range each slot's buffer view
        // reads, kept from pass to pass. Only the slots written since the
        // last pass are looked at again (83,000 slots a pass otherwise, 0.1 s
        // at best); all of them every 16 passes, because a view of a
        // released resource goes stale without a write, and a slot written
        // before its memory had a host copy is only made then.
        type Ranges = SlotMap<u32, Resolved>;
        static RANGES: std::sync::Mutex<Option<std::collections::HashMap<usize, (u64, Ranges)>>> =
            std::sync::Mutex::new(None);
        let mut kept = RANGES.lock().unwrap_or_else(|e| e.into_inner());
        let (passes, ranges) = kept
            .get_or_insert_with(Default::default)
            .entry(std::ptr::from_ref(host) as usize)
            .or_insert_with(|| (0, Ranges::default()));
        let changed = host.buffer_changes();
        // `VICTORY_HEAP_ROLLING_RESCAN=1`: a sixteenth of the slots (by
        // slot number) is walked again at every pass, with the changed
        // ones, instead of all of them at every sixteenth pass. Each slot is
        // still looked at again within 16 passes; the walk of all 83,000
        // (0.1 s or more) no longer falls on one frame. Off by default.
        static ROLLING: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let rolling = *ROLLING.get_or_init(|| std::env::var_os("VICTORY_HEAP_ROLLING_RESCAN").is_some_and(|v| v == "1"));
        // The first pass walks them all either way.
        let full = if rolling { *passes == 0 } else { *passes % 16 == 0 };
        let pass = *passes;
        let turn = (pass % 16) as u32;
        // The slots whose kept resolution the walk replaces: what it does
        // not keep or resolve again of them goes after it (a slot that
        // holds no buffer view any more is not walked).
        let rescanned = |index: u32| full || (rolling && index % 16 == turn);
        let slots = if rolling && !full {
            let mut slots: Vec<u32> = host.buffer_slots_where(|i| i % 16 == turn);
            slots.extend(changed.into_iter().filter(|&i| i % 16 != turn));
            slots
        } else if full {
            host.buffer_slots()
        } else {
            changed
        };
        *passes += 1;
        HEAP_BUFFER_SPLIT[3].fetch_add(slots.len() as u64, Ordering::Relaxed);
        if full {
            HEAP_BUFFER_SPLIT[4].fetch_add(1, Ordering::Relaxed);
        }
        let walk = std::time::Instant::now();
        // With `slot_cache`: the walked slots whose kept resolution still
        // holds (see there), checked under one lock of each table.
        let mut still = vec![false; slots.len()];
        if slot_cache() {
            let mut asked: Vec<(usize, u32, [u32; 2])> = Vec::new();
            for (n, &index) in slots.iter().enumerate() {
                if deep_rescan(index, pass) {
                    continue;
                }
                let Some(words) = ranges.get(&index).and_then(|r| r.words) else {
                    continue;
                };
                if heap.handle(index).and_then(crate::descriptor::record_words) == Some(words) {
                    asked.push((n, index, words));
                }
            }
            let live = crate::descriptor::records_live(asked.iter().map(|a| a.2));
            asked = asked.into_iter().zip(live).filter_map(|(a, live)| live.then_some(a)).collect();
            let shown: Vec<(u32, u64, &ID3D12Resource)> = asked
                .iter()
                .map(|&(_, index, [i, generation])| {
                    (index, u64::from(generation) << 32 | u64::from(i), &ranges[&index].shadow.buffer)
                })
                .collect();
            let made = host.still_made(&shown);
            drop(shown);
            for (&(n, index, _), made) in asked.iter().zip(made) {
                if made && let Some(r) = ranges.get_mut(&index) {
                    r.pass = pass;
                    still[n] = true;
                }
            }
            let hits = still.iter().filter(|&&s| s).count() as u64;
            HEAP_WALK_SPLIT[0].fetch_add(walk.elapsed().as_nanos() as u64, Ordering::Relaxed);
            HEAP_WALK_SPLIT[3].fetch_add(hits, Ordering::Relaxed);
        }
        // Whether the device was there before the pass: a view that removes
        // it is named below.
        static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut alive = !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
        // Slots resolved, of them ones whose kept resolution failed its
        // checks, slow ones and their ns, ns in `mirror` and in the lookup.
        let (mut resolved, mut failed, mut slow, mut slow_ns, mut mirror_ns, mut copy_ns) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
        HEAP_PASS_PARTS[1].fetch_add(slots.len() as u64, Ordering::Relaxed);
        for (n, &index) in slots.iter().enumerate() {
            if still[n] {
                continue;
            }
            resolved += 1;
            if ranges.remove(&index).is_some_and(|r| r.words.is_some()) && slot_cache() && !deep_rescan(index, pass) {
                failed += 1;
            }
            let Some(handle) = heap.handle(index) else {
                continue;
            };
            // Read before `mirror` reads them: a record written in between
            // fails the next check, as the host slot is made from that one.
            let words = crate::descriptor::record_words(handle);
            let mirroring = std::time::Instant::now();
            let view = host.mirror(index, handle, true);
            let mirrored = std::time::Instant::now();
            let spent = (mirrored - mirroring).as_nanos() as u64;
            mirror_ns += spent;
            let Some(view) = view else {
                if spent >= 100_000 {
                    (slow, slow_ns) = (slow + 1, slow_ns + spent);
                }
                continue;
            };
            let uav = view_kind(&view) == Some(ResourceType::Uav);
            // Every 64 views while the device lives (a cheap query).
            if alive
                && resolved % 64 == 63
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                alive = false;
                GONE.store(true, Ordering::Relaxed);
                let last: Vec<String> = slots[n.saturating_sub(63)..=n]
                    .iter()
                    .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                    .map(|s| s.chars().take(200).collect())
                    .collect();
                tracing::error!(why, slot = index, "the host device went while buffer views of the heap were made; one of: {last:#?}");
                crate::host::log_removal_context();
            }
            // The copy the slot shows (`buffer_copy`), not the one the
            // view's range alone would find.
            let copy = crate::descriptor::host::buffer_copy(&view);
            let looked = (std::time::Instant::now() - mirrored).as_nanos() as u64;
            copy_ns += looked;
            if spent + looked >= 100_000 {
                (slow, slow_ns) = (slow + 1, slow_ns + spent + looked);
            }
            let Some((shadow, address, len)) = copy else {
                continue;
            };
            #[cfg(feature = "diagnostics")]
            if host.shows(index, &shadow.buffer) == Some(false) {
                HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            }
            ranges.insert(index, Resolved { shadow, from: address, to: address + len, words, pass, uav });
        }
        for (i, n) in [(1, mirror_ns), (2, copy_ns), (4, resolved), (5, failed), (6, slow), (7, slow_ns)] {
            HEAP_WALK_SPLIT[i].fetch_add(n, Ordering::Relaxed);
        }
        let walked = walk.elapsed().as_nanos() as u64;
        HEAP_BUFFER_SPLIT[0].fetch_add(walked, Ordering::Relaxed);
        if full {
            HEAP_BUFFER_SPLIT[5].fetch_add(walked, Ordering::Relaxed);
        }
        let spanning = std::time::Instant::now();
        // Per host copy, the span its picked views cover; the rescanned
        // slots' resolutions the walk did not keep or make again go.
        let mut spans: SlotMap<usize, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)> = SlotMap::default();
        // The unordered access ones' ranges ([`picked_uav_set`]): at every
        // pass with `VICTORY_PICKED_UAV_WRITES=1`, else at every 16th (for
        // the counts alone).
        let measure = picked_uav_writes() || pass % 16 == 0;
        let mut uavs = Vec::new();
        // With the measure: where draws find the bytes of each such view
        // (`crate::shadow::PICKED_ELSEWHERE`).
        let mut elsewhere = [0u64; 6];
        // With the measure: views that only read memory the picked-view
        // programs write (`crate::shadow::SKINNED_READERS`).
        let mut readers = [0u64; 3];
        ranges.retain(|&index, r| {
            if rescanned(index) && r.pass != pass {
                return false;
            }
            if measure && !r.uav && crate::shadow::skinned_bytes(r.from, r.to) > 0 {
                readers[0] += 1;
                let other = crate::shadow::skinned_elsewhere(r.from, r.to, &r.shadow);
                if other > 0 {
                    readers[1] += 1;
                    readers[2] += other;
                }
            }
            if r.uav {
                // Its copy's picked span ([`crate::shadow::picked_guard`]).
                r.shadow.mark_picked_uav(r.from, r.to - r.from);
                if measure {
                    uavs.push((r.shadow.clone(), r.from, r.to));
                    elsewhere[5] += 1;
                    match crate::shadow::peek(r.from, r.to - r.from) {
                        Some(found) if std::sync::Arc::ptr_eq(&found, &r.shadow) => {}
                        Some(_) => {
                            elsewhere[0] += 1;
                            elsewhere[1] += r.to - r.from;
                            if r.shadow.is_window() {
                                elsewhere[2] += 1;
                            } else if crate::shadow::is_own_bytes(&r.shadow) {
                                elsewhere[3] += 1;
                            }
                        }
                        None => elsewhere[4] += 1,
                    }
                }
            }
            let key = std::sync::Arc::as_ptr(&r.shadow) as usize;
            let span = spans.entry(key).or_insert_with(|| (r.shadow.clone(), r.from, r.to));
            span.1 = span.1.min(r.from);
            span.2 = span.2.max(r.to);
            true
        });
        let views = ranges.len() as u32;
        drop(kept);
        if measure {
            picked_uav_set(uavs);
            for (i, n) in elsewhere.into_iter().enumerate() {
                crate::shadow::PICKED_ELSEWHERE[i].store(n, Ordering::Relaxed);
            }
            for (i, n) in readers.into_iter().enumerate() {
                crate::shadow::SKINNED_READERS[i].store(n, Ordering::Relaxed);
            }
        }
        HEAP_WALK_SPLIT[8].fetch_add(spanning.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
        // Memory that is not write-watched is copied whole by every refresh.
        // It was once refreshed only every 120 frames (unmapped pages read
        // one by one cost 14 GB a frame in run 11, before they were
        // remembered): streamed geometry then drew from stale buffers, and
        // the garbage ended runs (the card stopped answering 14-90 frames
        // into the opening cutscene; with every frame it ran on). The pass
        // costs some 30 ms a frame this way.
        static WHOLE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> = std::sync::Mutex::new(None);
        let mut whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
        let whole = whole.get_or_insert_with(Default::default);
        for (shadow, from, to) in spans.values() {
            if shadow.unwatched() {
                // By the copy's number: a copy made where one let go was
                // (`shadow::let_go_idle`) has never been refreshed.
                let key = shadow.id() as usize;
                // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
                static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let every = *EVERY.get_or_init(|| {
                    std::env::var("VICTORY_UNWATCHED_FRAMES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                });
                match whole.get(&key) {
                    Some(&last) if frame < last + every => continue,
                    _ => {
                        whole.insert(key, frame);
                    }
                }
            }
            shadow.refresh(list, *from, to - from);
        }
        HEAP_BUFFER_SPLIT[1].fetch_add(spanning.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let texturing = std::time::Instant::now();
        self.heap_textures(list, heap, host);
        HEAP_BUFFER_SPLIT[2].fetch_add(texturing.elapsed().as_nanos() as u64, Ordering::Relaxed);
        HEAP_PASS_PARTS[0].fetch_add(texturing.elapsed().as_micros() as u64, Ordering::Relaxed);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        // Which thread it ran on, for the frame's path.
        {
            use crate::host::{PATH, path};
            let place = if crate::host::on_submitting_thread() { path::HEAP_SUBMITTER } else { path::HEAP_OTHER };
            PATH[place].fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        }
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        // The first pass and every growth of the set by half.
        let seen = LOGGED.load(Ordering::Relaxed);
        if u64::from(views) > seen + seen / 2 {
            LOGGED.store(u64::from(views), Ordering::Relaxed);
            tracing::info!(
                slots = slots.len(),
                views,
                copies = spans.len(),
                mb = bytes >> 20,
                ms = started.elapsed().as_millis(),
                "buffer views of the heap made ready for shaders that pick them by index"
            );
        }
        Some(gpu)
    }

    /// Runs the pass that uploads textures picked by index
    /// ([`State::heap_textures`]) where nothing has asked for it yet
    /// ([`early_budgets`]): once a frame for a few milliseconds, at the
    /// frame's first draw with a view heap set, and with a larger budget
    /// of its own before every draw whose pixel shader is one of the
    /// title's bake programs. The uploads are recorded into `list` ahead
    /// of the draw being bound, as the world's pass records them.
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        let (early_ms, bake_ms) = early_budgets();
        if compute || (early_ms == 0 && bake_ms == 0) {
            return;
        }
        let bake = bake_ms > 0
            && p.stages.iter().any(|s| BAKE_PIXEL_SHADERS.contains(&s.shader.hash.as_str()));
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        let first = FRAME.swap(frame, Ordering::Relaxed) != frame;
        if first {
            BAKED_MS.store(0, Ordering::Relaxed);
        }
        let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
        let budget = if bake && left > 0 {
            // What a bake may read must be there now: up to 2 GiB.
            (left, 2 << 30)
        } else if first && early_ms > 0 {
            (early_ms, 64 << 20)
        } else {
            return;
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        let bake = bake && left > 0;
        let n = EARLY_TEXTURE_PASSES[usize::from(bake)].fetch_add(1, Ordering::Relaxed) + 1;
        if bake && n.is_power_of_two() {
            let [slots, due, ..] = HEAP_TEXTURES_QUEUE.each_ref().map(|c| c.load(Ordering::Relaxed));
            tracing::info!(
                frame,
                bakes = n,
                queued_at_the_last_pass = slots,
                due_at_the_last_pass = due,
                uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                "a draw of the title's decoration bake: textures picked by index are uploaded first (VICTORY_BAKE_TEXTURES_MS=0: not)"
            );
        }
        let began = std::time::Instant::now();
        PASS_BUDGET.set(Some(budget));
        self.heap_textures(list, heap, host);
        PASS_BUDGET.set(None);
        if bake {
            BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
    }

    /// The queue's side of the card accounting line
    /// (`crate::resource::card_accounting`, which says what the line
    /// holds): the slots that are due, by what their texture has on the
    /// card. Off unless `VICTORY_CARD_ACCOUNTING=1`; then at the first pass
    /// that finds the card without room, and every five seconds after it.
    /// Nothing is made, uploaded or counted as used here. For 18,000 due
    /// slots: 18,000 looks at a title slot and at the texture it views, a
    /// lock each, some milliseconds of the thread that records the frame.
    fn card_accounting(
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        frame: u64,
        room: bool,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ACCOUNTING").is_some_and(|v| v == "1")) {
            return;
        }
        // When the next line is due (0: none yet, the card has had room).
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms().max(1);
        let next = NEXT.load(Ordering::Relaxed);
        // (With levels on demand the card may never be without room: the
        // lines then start with the first texture made that way.)
        let by_level = crate::tiles::STATS[4].load(Ordering::Relaxed) > 0;
        let due_now = if next == 0 { !room || by_level } else { now >= next };
        if !due_now || NEXT.compare_exchange(next, now + 5000, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        // Due slots; of them uploaded before; of them viewing nothing the
        // title fills; and the textures (each once) whose host copy is
        // alive, with their bytes on the card.
        let mut due = [0u64; 5];
        // Textures (each once) that never had a host copy, and ones whose
        // host copy was let go: where their memory is, and its bytes.
        let (mut never, mut gone) = (Vec::new(), Vec::new());
        let mut seen = std::collections::BTreeSet::new();
        for (index, uploaded) in host.texture_slots_all_due(frame) {
            due[0] += 1;
            due[1] += u64::from(uploaded);
            let state = match heap.handle(index).and_then(crate::descriptor::view) {
                Some(View::Placed(false, resource, desc)) if resource[0] != 1 => {
                    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                    match crate::resource::placed_known(address, &resource) {
                        Some(texture) => crate::resource::copy_state(texture),
                        // No title texture was made for it: the pass never
                        // got as far as making one.
                        None => Some((0, crate::resource::placed_bytes(&resource).unwrap_or(0), address)),
                    }
                }
                Some(View::Shader(object, _)) if crate::resource::title_fills(object as This) => {
                    crate::resource::copy_state(object as This)
                }
                _ => None,
            };
            let Some((state, bytes, memory)) = state else {
                due[2] += 1;
                continue;
            };
            if !seen.insert(memory) {
                continue;
            }
            match state {
                2 => {
                    due[3] += 1;
                    due[4] += bytes;
                }
                1 => gone.push((memory, bytes)),
                _ => never.push((memory, bytes)),
            }
        }
        crate::resource::card_accounting(room, due, never, gone);
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    fn heap_textures(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return;
        }
        // Bytes of texture memory staged per frame: 192 MB, or
        // `VICTORY_TEXTURE_STAGE_MB` (read once; `VICTORY_HEAP_TEXTURE_MB`,
        // our earlier name for it, when that is unset). More drains a backlog
        // of tens of thousands of textures (a scene's first minutes) in fewer
        // frames; less is more frames a second.
        static PER_FRAME_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let per_frame: u64 = *PER_FRAME_MB.get_or_init(|| {
            ["VICTORY_TEXTURE_STAGE_MB", "VICTORY_HEAP_TEXTURE_MB"]
                .iter()
                .find_map(|name| std::env::var(name).ok().and_then(|v| v.trim().parse().ok()))
                .unwrap_or(192)
        }) << 20;
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        let mut room = crate::resource::room();
        // Bytes uploaded by this pass since the card was last asked
        // whether it has room: asked again every 32 MiB (a pass may upload
        // 192 MB and more, and one asking a pass let the queue run some
        // 400 MB past its line while the world loaded: run 193).
        let mut since_asked = 0u64;
        if !room {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        let mut refused = 0;
        // For the frame line: where the pass's time goes
        // ([`HEAP_TEXTURE_SPLIT`]); the queue first.
        let mut phases = Phases::new(&HEAP_TEXTURE_SPLIT, 0);
        // Room again: the slots held back while the card was full are due
        // now (`full_card_pass`).
        if room {
            host.texture_slots_room(frame);
        }
        // Slots whose texture's memory the runtime mapped, unmapped or
        // wrote since the last pass are due again.
        let taking = std::time::Instant::now();
        host.texture_slots_remapped(frame);
        let remapped = taking.elapsed();
        // Levels on demand: the slots of textures whose first level with
        // memory changed since the last pass are made again from it.
        host.texture_views_remade();
        // And with room, the textures whose upload left a held level
        // without the card's memory are uploaded again.
        if room {
            host.texture_slots_waiting(frame);
        }
        let (due, census) = host.texture_slots_due(frame, 4096);
        HEAP_LOOK_SPLIT[0].fetch_add(remapped.as_nanos() as u64, Ordering::Relaxed);
        HEAP_LOOK_SPLIT[1].fetch_add((taking.elapsed() - remapped).as_nanos() as u64, Ordering::Relaxed);
        HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
        for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
            count.store(now, Ordering::Relaxed);
        }
        // Where the card's memory is, and what the due slots would need
        // (`VICTORY_CARD_ACCOUNTING=1`; a line when the card first has no
        // room, and every five seconds after it).
        Self::card_accounting(heap, host, frame, room);
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        let took = now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1);
        let most = (4000 / took).clamp(16, 256);
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging 192 MB of textures took
        // 1.6-2.2 s of each of the world's first twenty frames (run 20).
        // The rest waits for the next frames, as it does past the bytes.
        // 250 ms, or `VICTORY_TEXTURE_STAGE_MS` (read once;
        // `VICTORY_HEAP_TEXTURE_MS`, our earlier name for it, when that is
        // unset): the opening cutscene's backlog of thousands of due slots
        // kept the pass at its budget, most of a 380-500 ms frame (fork4: the
        // heap pass 265 ms of it). Less is more frames a second, with
        // textures streamed in later.
        static SPAN_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let span = std::time::Duration::from_millis(*SPAN_MS.get_or_init(|| {
            ["VICTORY_TEXTURE_STAGE_MS", "VICTORY_HEAP_TEXTURE_MS"]
                .iter()
                .find_map(|name| std::env::var(name).ok().and_then(|v| v.trim().parse().ok()))
                .unwrap_or(250)
        }));
        // A backlog (a scene's first minutes: 22,000 slots due, 2-3 minutes of black
        // surfaces at 192 MB a frame) gets three times the bytes and the time until it
        // drains; `VICTORY_TEXTURE_CATCHUP=0` keeps the plain budget.
        static CATCHUP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let catchup = *CATCHUP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_CATCHUP").is_none_or(|v| v != "0"));
        let (per_frame, span) = if catchup && due.len() >= 1024 { (per_frame * 3, span * 3) } else { (per_frame, span) };
        // A run made early (`State::early_textures`) has a budget of its
        // own: its bytes on top of what the frame has spent so far.
        // (Not `most`: that is the longest wait between looks, used below.)
        let (span, bytes_at_most) = match PASS_BUDGET.get() {
            Some((ms, bytes)) => {
                (std::time::Duration::from_millis(ms), SPENT.load(Ordering::Relaxed).saturating_add(bytes))
            }
            None => (span, per_frame),
        };
        // Placed textures made ready on worker threads
        // (`VICTORY_ASYNC_TEXTURES`, `crate::prepare`): what is left here
        // is the slot's view, the copy and its barriers.
        let ahead = crate::prepare::workers() > 0;
        if ahead {
            let sweeping = std::time::Instant::now();
            crate::prepare::sweep();
            HEAP_LOOK_SPLIT[2].fetch_add(sweeping.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
        let cheap = ahead && cheap_looks();
        let began = std::time::Instant::now();
        for (index, write) in due {
            // What of the slot before took the time since the last phase.
            phases.to(1);
            let spent = SPENT.load(Ordering::Relaxed) >= bytes_at_most;
            if spent || began.elapsed() > span {
                HEAP_TEXTURE_SPLIT[if spent { 7 } else { 6 }].fetch_add(1, Ordering::Relaxed);
                break;
            }
            HEAP_TEXTURE_SPLIT[5].fetch_add(1, Ordering::Relaxed);
            // A worker has it: nothing to look at until it is done (the
            // slot stays due). Asking what is mapped of every one of them
            // at every pass would cost what the workers save.
            if ahead && crate::prepare::busy(index) {
                continue;
            }
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            // A slot uploaded whole is looked at again a few times
            // (`RECHECK_FRAMES`): the title writes a texture's view, and
            // may fill the texture's memory itself only afterwards, with
            // no descriptor write and nothing from the runtime to tell of
            // it. The host copy then kept what the memory held at the
            // first look for the rest of the run. The look reads 48 KiB;
            // only a changed texture is uploaded again.
            // `VICTORY_HEAP_RECHECK=0`: uploaded once, as before.
            // Of its memory, only the bytes before the next texture placed in
            // it (`descriptor::host::texture_own_bytes`): the title keeps only a
            // texture's small levels and places the next textures where its
            // large ones would be, so its fingerprint's middle and end
            // windows saw the neighbours stream in, and it was uploaded again
            // with their bytes as its own large levels (runs n7-n14: the most
            // re-uploaded slots are 4, 8 and 16 KiB apart, 344-684 KiB each;
            // the blocks on the near surfaces at the bottom of the picture).
            if let Some((at, len, print, looks, uploaded)) = host.texture_slot_checked(index) {
                if crate::texture::fingerprint(at, crate::descriptor::host::texture_own_bytes(at, len)) == print {
                    HEAP_TEXTURES_RECHECKED[0].fetch_add(1, Ordering::Relaxed);
                    match RECHECK_FRAMES.get(usize::from(looks) + 1) {
                        Some(wait) => {
                            host.texture_slot_recheck(index, write, frame + wait, (at, len), print, looks + 1, uploaded)
                        }
                        None => host.texture_slot_done(index, write, None),
                    }
                    continue;
                }
                let n = HEAP_TEXTURES_RECHECKED[1].fetch_add(1, Ordering::Relaxed);
                if n < 200 || n.is_multiple_of(1000) {
                    tracing::warn!(
                        slot = index,
                        address = format_args!("{at:#x}"),
                        bytes = len,
                        frames_after_upload = frame.saturating_sub(uploaded),
                        looks,
                        changed = n + 1,
                        "a texture picked by index changed after its upload with nothing to tell of it: uploaded again"
                    );
                }
            }
            let mut partly = None;
            let looked = heap.handle(index).and_then(crate::descriptor::view);
            // A placed texture's memory: address and bytes.
            let mut placed_at = None;
            if let Some(View::Placed(false, resource, desc)) = &looked {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let len = crate::resource::placed_bytes(resource).unwrap_or(1);
                placed_at = Some((address, len));
                // `VICTORY_CHEAP_LOOKS`: what a worker would do with the
                // slot first; the look only when it is used.
                let ahead_of_look = if cheap && resource[0] != 1 {
                    crate::prepare::ahead_of_look(index, len)
                } else {
                    crate::prepare::Ahead::Look
                };
                let mapped = match ahead_of_look {
                    crate::prepare::Ahead::Capped => {
                        HEAP_LOOK_SPLIT[3].fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                    // Uploaded with what the worker found mapped
                    // (`Prepared::mapped`, below).
                    crate::prepare::Ahead::Ready => {
                        HEAP_LOOK_SPLIT[4].fetch_add(1, Ordering::Relaxed);
                        len
                    }
                    crate::prepare::Ahead::Look => crate::texture::mapped_bytes(address as *const u8, len as usize),
                };
                if mapped == 0 {
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                if mapped < len {
                    if mapped <= host.texture_slot_mapped(index) {
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some(&View::Shader(object, _)) = looked.as_ref()
                && crate::resource::buffer_bytes(object as This).is_none()
                && !crate::resource::title_fills(object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            // Not so a slot already made from this very view: its host
            // texture exists, and uploading it again takes none of the
            // card's memory. It was refused all the same, so a texture the
            // title filled after its first upload kept its empty host copy
            // from the moment the card reached the line, which is the first
            // world frame in eight launches of nine: the same slots were
            // found changed again at every pass and never uploaded (run
            // 179: 33,000 such findings), and the main character's face
            // had its colour only in the launches that entered the world
            // under the line. `VICTORY_REUPLOAD_WHEN_FULL=0`: refused, as
            // before.
            if !room {
                let has_its_texture = reupload_when_full()
                    && heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle));
                // Under `VICTORY_PRESSED_LEVELS_KB`: a texture made level by
                // level is made all the same, its small levels alone given
                // memory (`crate::tiles::pressed_cap`), not left a null view.
                let low_levels = !has_its_texture
                    && looked
                        .as_ref()
                        .and_then(|view| match view {
                            View::Shader(object, _) => Some(*object),
                            _ => match crate::descriptor::host::placed_texture(view) {
                                Some(View::Shader(texture, _)) => Some(texture),
                                _ => None,
                            },
                        })
                        .is_some_and(|texture| crate::resource::low_levels_fit(texture as This));
                if low_levels {
                    crate::memline::SLOTS_PRESSED[1].fetch_add(1, Ordering::Relaxed);
                } else if !has_its_texture {
                    refused += 1;
                    crate::memline::SLOTS_PRESSED[0].fetch_add(1, Ordering::Relaxed);
                    if full_card_pass() {
                        // Held back for the frames of about four seconds
                        // (`most`), or until the card has room; the pass
                        // goes on, within a count that bounds one pass.
                        host.texture_slot_held(index, write, frame + most);
                        if refused > 1024 {
                            break;
                        }
                    } else if refused > 256 {
                        break;
                    }
                    continue;
                }
                let n = if low_levels { 0 } else { HEAP_TEXTURES_REUPLOADED_FULL.fetch_add(1, Ordering::Relaxed) + 1 };
                if n == 1 || n.is_multiple_of(500) && n > 0 {
                    tracing::info!(
                        so_far = n,
                        slot = index,
                        "a texture already on the card is brought up to date although the card has no room for new ones"
                    );
                }
            }
            // A placed texture: made ready by a worker if one has done it,
            // else handed to one, the slot left due. One a worker could
            // not make is made here, as before.
            // Not so a refresh on a full card (`reupload_when_full`): a
            // worker makes nothing while the card has no room
            // (`crate::prepare`), so the job came back "later" at every pass
            // and the slot was queued again, never uploaded (runs n7-n14:
            // some 640 jobs queued a frame, none made ready; 400,000
            // "brought up to date" counted for none done). It is done here,
            // as upstream does it: its host texture exists.
            let mut prepared = None;
            if ahead
                && room
                && let Some((address, len)) = placed_at
                && let Some(View::Shader(texture, _)) =
                    looked.as_ref().and_then(crate::descriptor::host::placed_texture)
            {
                match crate::prepare::look(index, texture, address, len) {
                    crate::prepare::Look::Ready(item) => prepared = Some(item),
                    crate::prepare::Look::Inline => {}
                    crate::prepare::Look::Wait => continue,
                }
            }
            phases.to(2);
            let view = heap.handle(index).and_then(|handle| host.mirror(index, handle, true));
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            let title_fills = crate::resource::host_resource(resource).is_some_and(crate::texture::title_written);
            if !title_fills {
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            phases.to(4);
            let tabled = crate::texture::is_staged(resource);
            // The worker's staging buffer goes in the table for the upload
            // below to find, detiled already; not for a texture tables read
            // (left to their draws, as before), nor one the slot no longer
            // shows. It was read before the upload's submission, whose
            // check reads again what the title wrote since; uploaded with
            // what was mapped when the worker asked, it is looked at again
            // as a texture not all there (more may be mapped now).
            let mut adopted = false;
            if let Some(item) = prepared.take()
                && !tabled
                && item.resource == resource as usize
                && crate::texture::adopt(resource, item.staging.clone())
            {
                adopted = true;
                if let Some((_, len)) = placed_at
                    && item.mapped < len
                {
                    partly = Some(partly.map_or(item.mapped, |m: u64| m.min(item.mapped)));
                }
            }
            phases.to(3);
            if let Some(view) = &view {
                readable(list, sampled.kind, view);
            }
            phases.to(4);
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                crate::host::log_removal_context();
                return;
            }
            let (bytes, whole) = crate::texture::staged(resource).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            let memory = if again.is_none() && heap_recheck() {
                crate::texture::staged_memory(resource)
            } else {
                None
            };
            crate::texture::unstage(resource);
            if adopted {
                crate::prepare::recorded();
            }
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            since_asked += bytes;
            if room && since_asked >= 32 << 20 {
                since_asked = 0;
                room = crate::resource::room();
            }
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            match memory {
                // Whole and uploaded: looked at again (see above).
                Some((at, len)) => {
                    let len = crate::descriptor::host::texture_own_bytes(at, len);
                    let print = crate::texture::fingerprint(at, len);
                    host.texture_slot_recheck(index, write, frame + RECHECK_FRAMES[0], (at, len), print, 0, frame);
                }
                None => host.texture_slot_done(index, write, again),
            }
        }
        phases.to(0);
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        let _timed = RecordTimed::new(rec::DISPATCHES_NS, Some(rec::DISPATCHES));
        // It may write the arguments a window holds clamped.
        self.window = None;
        self.amplified_run = false;
        self.amplified_reads_done(list);
        // A dispatch that may write records ends the run (1: every one).
        let rules = crate::indirect::run_rules();
        if rules.on {
            if rules.dispatch_ends(self.pipeline_stores()) {
                self.end_batch(list);
            } else if self.batch.is_some() {
                crate::indirect::note_run_kept();
            }
            self.dispatched = true;
        }
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // A program that walks ray-tracing structures nothing has built.
        if crate::pipeline::walk_held(self.pipeline as This) {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_why("a walk of ray-tracing structures that are not built (VICTORY_BVH_HOLD=1)".to_owned());
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        if !self.bind(list, p, true) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        list.prof_draw(false);
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(x, y, z) };
        list.prof_draw(true);
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The title's scissor and viewport at this draw (the first of
            // each): the title draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute && !list_slots().is_empty() {
            text.push_str(&self.slots_text());
        }
        text
    }

    /// For the frame listing, after a draw whose shader `VICTORY_LIST_SLOTS`
    /// names: the heap slots it will pick textures from by index, and what
    /// the title and the host hold there (one `~slot` line each), with the
    /// words they were read from (`~raw` lines, to decode offline what is
    /// not decoded here).
    ///
    /// Where the slots are, for the title's character shaders
    /// (`PS_DeferredPedTexturedNC`, from their native code): in the small
    /// buffers and constants of the draw's own table (every word of those
    /// that names a texture view gets a line); in the material record, at
    /// the byte offset the draw's constants (table 1, dword 56) hold at
    /// 0x1c, of the buffer at table 0, dword 160; and in the record at the
    /// byte offset the second root constant holds of the buffer at table 0,
    /// dword 688. A word that is a small number by chance and names a
    /// texture gets a line too.
    #[cfg(feature = "diagnostics")]
    fn slots_text(&self) -> String {
        use std::fmt::Write;
        const MATERIALS: u32 = 160;
        const GLOBAL: u32 = 688;
        const DRAW_CONSTANTS: u32 = 56;
        let wanted = list_slots();
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return String::new();
        };
        let Some(stage) = p.stages.iter().find(|s| wanted.iter().any(|w| s.shader.hash.starts_with(w.as_str())))
        else {
            return String::new();
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return String::new();
        };
        let t = stage.translated();
        let mut text = String::new();
        let entry = |slot: u8, dwords: u32| -> Option<View> {
            let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(slot)) else {
                return None;
            };
            crate::descriptor::view(table + u64::from(dwords) * 4)
        };
        let mut material = None;
        for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind != ResourceType::Sampler) {
            let label = format!("s{}+{}", u.slot, u.table_offset_dwords);
            let Some(view) = entry(u.slot, u.table_offset_dwords) else {
                let _ = write!(text, "\n    ~table {label} no view");
                continue;
            };
            let range = match &view {
                View::Constant(address, size) => Some((*address, u64::from(*size))),
                other => crate::descriptor::host::buffer_range(other),
            };
            match range {
                // A small buffer or constants: its words.
                Some((address, len)) if len <= 2048 => {
                    Self::slot_words(&mut text, heap, &label, address, len as usize);
                    if u.slot == 1 && u.table_offset_dwords == DRAW_CONSTANTS && len >= 0x20 {
                        let mut word = [0u8; 4];
                        if crate::host::read_into((address + 0x1c) as *const u8, &mut word) {
                            material = Some(u32::from_le_bytes(word));
                        }
                    }
                }
                Some(_) => {}
                // A texture read through the table: the title slot it sits in.
                None => {
                    let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(u.slot)) else {
                        continue;
                    };
                    match heap.index(table + u64::from(u.table_offset_dwords) * 4) {
                        Some(index) => text.push_str(&Self::slot_line(heap, &format!("table:{label}"), 0, index)),
                        None => {
                            let _ = write!(text, "\n    ~table {label} title=[{}]", view_state(Some(&view)));
                        }
                    }
                }
            }
        }
        let uses = |slot: u8, dwords: u32| {
            t.used.iter().any(|u| u.root == RootType::Table && u.slot == slot && u.table_offset_dwords == dwords)
        };
        if let Some(at) = material
            && uses(0, MATERIALS)
            && let Some((base, len)) = entry(0, MATERIALS).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~material record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "M", base + at, 64);
            }
            if at + 1936 + 512 <= len {
                Self::slot_words(&mut text, heap, "M+1936", base + at + 1936, 512);
            }
        }
        if uses(0, GLOBAL)
            && let Some(slot) = t.used.iter().find(|u| u.root == RootType::Constants).map(|u| u.slot)
            && let Some(Root::Constants(words)) = self.roots_ref(false).get(usize::from(slot))
            && let Some(&at) = words.get(1)
            && let Some((base, len)) = entry(0, GLOBAL).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~root constants {words:08x?}: record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "G", base + at, 64);
            }
        }
        text
    }

    /// Up to 512 bytes of title memory at `address` as hex words (a `~raw`
    /// line), and a `~slot` line for every word of them that names a
    /// texture view in `heap`.
    #[cfg(feature = "diagnostics")]
    fn slot_words(text: &mut String, heap: &crate::descriptor::Heap, label: &str, address: u64, len: usize) {
        use std::fmt::Write;
        let mut bytes = vec![0u8; len.min(512) & !3];
        if bytes.is_empty() || !crate::host::read_into(address as *const u8, &mut bytes) {
            let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}: unreadable");
            return;
        }
        let words: Vec<u32> =
            bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}:");
        for w in &words {
            let _ = write!(text, " {w:08x}");
        }
        for (i, &index) in words.iter().enumerate() {
            let names = index != 0
                && heap
                    .handle(index)
                    .and_then(crate::descriptor::view)
                    .as_ref()
                    .is_some_and(crate::descriptor::host::is_texture_view);
            if names {
                text.push_str(&Self::slot_line(heap, label, i * 4, index));
            }
        }
    }

    /// One `~slot` line: where the slot number was read (`from`, byte
    /// `at`), what the title's heap holds in slot `index` with the state
    /// of that texture ([`view_state`]), what the host slot was made from,
    /// and what the queue of textures picked by index knows of it.
    #[cfg(feature = "diagnostics")]
    fn slot_line(heap: &crate::descriptor::Heap, from: &str, at: usize, index: u32) -> String {
        let title = heap.handle(index).and_then(crate::descriptor::view);
        let host: String = heap
            .host()
            .and_then(|h| h.describe_slot(index))
            .unwrap_or_else(|| "none".into())
            .chars()
            .take(90)
            .collect();
        let queue = match heap.host() {
            Some(h) => h.texture_slot_state(index, crate::frame::current()),
            None => "no host heap".into(),
        };
        format!(
            "\n    ~slot from={from}+{at:#x} slot={index} title=[{}] host=[{host}] queue=[{queue}]",
            view_state(title.as_ref())
        )
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        let _timed = RecordTimed::new(rec::DRAWS_NS, Some(rec::DRAWS));
        self.window = None;
        self.end_batch_if_stores(list);
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let ran = mesh_test() != 1
            && mesh_kept(self.pipeline)
            && self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    // A pipeline with the amplification stage: it launches the
                    // groups (the count in its constants).
                    let amplify = crate::pipeline::host_pipeline(self.pipeline as This).and_then(|p| {
                        if p.stages.iter().any(|stage| stage.translated().stores) {
                            self.amplified_run = false;
                            if std::mem::take(&mut self.amplified_reads) && amplify_wait() >= 1 {
                                crate::indirect::wait_for_writes(list);
                                AMPLIFIED_WAITS.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        p.amplify
                    });
                    if let Some(at) = amplify {
                        let budget = list.budget().unwrap_or(0);
                        let launch = [0, x, 0, 0];
                        // SAFETY: an open list; the root signature has these
                        // parameters; the budget buffer is the list's.
                        unsafe {
                            list.list.SetGraphicsRoot32BitConstants(at, 4, launch.as_ptr().cast(), 0);
                            list.list.SetGraphicsRootShaderResourceView(at + 1, budget);
                            list.list.SetGraphicsRootShaderResourceView(at + 2, budget);
                            list.list.SetGraphicsRootUnorderedAccessView(at + 3, budget);
                        }
                    }
                    self.mesh_scissor(list, true);
                    list.prof_draw(false);
                    // SAFETY: an open list with everything set.
                    unsafe { l6.DispatchMesh(if amplify.is_some() { 1 } else { x }, 1, 1) };
                    list.prof_draw(true);
                    self.mesh_scissor(list, false);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// The walls trace (`crate::host::walls_trace`): one log line per
    /// frame with the draws of its main pass (four or more colour targets)
    /// by scissor rectangle, and how often each of the title's culling
    /// programs was dispatched. The title clips what it draws
    /// through a window or door to that opening's rectangle while it holds
    /// the camera to be in a room (run 182, frame 1529: 1,386 draws under
    /// `531 0 1789 1337`); in a launch without the room's walls the same
    /// shot has no such rectangle (run 187, frame 1414). The line shows
    /// the frame in which that changes.
    fn walls_census(&self, p: &HostPipeline, compute: bool) {
        const CULL: [(&str, &str); 9] = [
            ("55679a0547649d0b", "tree"),
            ("89d08e561897d4e5", "tree_imposter"),
            ("9209777e8face9dd", "grass"),
            ("17af15c4015cb4b6", "prop"),
            ("fdd56d97a806bcca", "prop_nocull"),
            ("42f717f1b4d18e94", "simple"),
            ("1c4b3c0547e3e6fa", "slod"),
            ("9146291b38ea9ce1", "meshlet_args"),
            ("42683eebec081520", "depth_pyramid_cpu_occlusion"),
        ];
        type Census = (u64, std::collections::BTreeMap<[i32; 4], u32>, [u32; 9]);
        static CENSUS: std::sync::Mutex<Census> =
            std::sync::Mutex::new((u64::MAX, std::collections::BTreeMap::new(), [0; 9]));
        let frame = crate::frame::current();
        let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
        if census.0 != frame {
            let draws: u32 = census.1.values().sum();
            if draws > 0 {
                let mut rects: Vec<([i32; 4], u32)> = census.1.iter().map(|(r, n)| (*r, *n)).collect();
                rects.sort_unstable_by(|a, b| b.1.cmp(&a.1));
                let scissors: Vec<String> = rects
                    .iter()
                    .take(5)
                    .map(|(r, n)| format!("{} {} {} {} x{n}", r[0], r[1], r[2], r[3]))
                    .collect();
                let counts = census.2;
                let culling: Vec<String> = CULL
                    .iter()
                    .zip(counts)
                    .filter(|(_, n)| *n > 0)
                    .map(|(c, n)| format!("{}={n}", c.1))
                    .collect();
                tracing::info!(
                    frame = census.0,
                    main_pass_draws = draws,
                    rectangles = rects.len(),
                    scissors = ?scissors,
                    culling = ?culling,
                    "walls trace: the frame's main pass by scissor rectangle, and its culling dispatches"
                );
            }
            census.0 = frame;
            census.1.clear();
            census.2 = [0; 9];
        }
        if compute {
            for (i, c) in CULL.iter().enumerate() {
                if p.stages.iter().any(|s| s.shader.hash.as_str() == c.0) {
                    census.2[i] += 1;
                }
            }
        } else if self.targets.iter().filter(|&&t| t != 0).count() >= 4 {
            let r = self.scissors.first().map_or([0; 4], |r| [r.left, r.top, r.right, r.bottom]);
            *census.1.entry(r).or_insert(0) += 1;
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            }
        }
    }

    /// The heap slots a stage that stores through views it picks from the
    /// heap reaches with the bound arguments (`t.picks`, [`picked_exact`]):
    /// their buffer views' memory is noted (`crate::shadow::note_skinned`),
    /// and with `declare` each view is made ready and declared written in
    /// `list`, once per list. Whether the slots were worked out (and with
    /// `declare`, declared).
    fn picked_slots(&self, list: &HostList, t: &crate::pipeline::Translated, compute: bool, declare: bool) -> bool {
        use crate::pipeline::PickFrom;
        use exact_stat::*;
        let Some(picks) = &t.picks else {
            PICKED_EXACT[NOT_FOLLOWED].fetch_add(1, Ordering::Relaxed);
            return false;
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return false;
        };
        let constant = |slot: u8, dword: u32| self.user_value(compute, slot, RootType::Constants, dword);
        let mut offsets: Vec<u32> = Vec::new();
        for pick in picks {
            let at = |x: u32| pick.offset.wrapping_add((x & pick.mask) << pick.shift);
            match pick.source {
                PickFrom::Fixed => offsets.push(pick.offset),
                PickFrom::Constant(slot, dword) => offsets.push(at(constant(slot, dword))),
                PickFrom::Records { table, entry, field, first, count } => {
                    let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(table)) else {
                        return false;
                    };
                    let Some(view) = crate::descriptor::view(table + u64::from(entry)) else {
                        return false;
                    };
                    let (Some((address, len)), Some(stride)) =
                        (crate::descriptor::host::buffer_range(&view), crate::descriptor::host::buffer_element(&view))
                    else {
                        return false;
                    };
                    if u64::from(field) + 4 > stride {
                        return false;
                    }
                    let records = len / stride;
                    let first = first.map_or(0, |(s, d)| u64::from(constant(s, d)));
                    let left = records.saturating_sub(first);
                    let count = count.map_or(left, |(s, d)| u64::from(constant(s, d)).min(left));
                    if count > PICKED_RECORDS_MOST {
                        PICKED_EXACT[CAPPED].fetch_add(1, Ordering::Relaxed);
                        return false;
                    }
                    PICKED_EXACT[RECORDS].fetch_add(count, Ordering::Relaxed);
                    for i in first..first + count {
                        let mut word = [0u8; 4];
                        let at_record = address + i * stride + u64::from(field);
                        if !crate::host::read_into(at_record as *const u8, &mut word) {
                            return false;
                        }
                        offsets.push(at(u32::from_le_bytes(word)));
                    }
                }
            }
        }
        offsets.sort_unstable();
        offsets.dedup();
        PICKED_EXACT[FOLLOWED].fetch_add(1, Ordering::Relaxed);
        PICKED_EXACT[SLOTS].fetch_add(offsets.len() as u64, Ordering::Relaxed);
        for offset in offsets {
            let handle = heap.start + u64::from(offset);
            let Some(index) = heap.index(handle) else {
                PICKED_EXACT[NO_BUFFER].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            let view = if declare {
                heap.host().and_then(|host| host.mirror(index, handle, true))
            } else {
                crate::descriptor::view(handle)
            };
            let Some(view) = view else {
                PICKED_EXACT[NO_BUFFER].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            let Some((address, len)) = crate::descriptor::host::buffer_range(&view).filter(|r| r.1 > 0) else {
                PICKED_EXACT[NO_BUFFER].fetch_add(1, Ordering::Relaxed);
                continue;
            };
            let uav = view_kind(&view) == Some(ResourceType::Uav);
            // Once per list for the same memory (keys apart from the
            // texture pass's: bit 62 set).
            let key = ((address ^ len.rotate_left(40)) as usize) | (1 << 62);
            if declare && uav && list.first_use(key) {
                readable(list, ResourceType::Uav, &view);
                PICKED_EXACT[DECLARED].fetch_add(1, Ordering::Relaxed);
                PICKED_EXACT[DECLARED_BYTES].fetch_add(len, Ordering::Relaxed);
            }
            // The copy the slot shows, as the heap sync found it for the
            // slot (the same question; the copy is there already).
            let copy = crate::descriptor::host::buffer_copy(&view).map(|c| c.0);
            crate::shadow::note_skinned(address, len, copy.map_or(u64::MAX, |c| c.id()));
        }
        declare
    }

    /// A table entry the shader picks by a root constant (`index`) in a
    /// table made from the title's own slots: made as [`State::table_entry`]
    /// makes the entries the used-resource list names. Without it the slot
    /// is made only by the pass over records copied in as bytes, which
    /// leaves tables' slots to their draws (`VICTORY_HEAP_SYNC_TABLES=0`).
    fn table_indexed_entry(&self, list: &HostList, index: &RootIndexedView, compute: bool) {
        let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(index.table_slot)) else {
            return;
        };
        let constant = self.user_value(compute, index.constant_slot, RootType::Constants, index.constant_dword);
        // The shader adds i32 byte offsets and shifts by 5.
        let k = index.byte_offset.wrapping_add(constant << index.shift) / 32;
        let handle = table + u64::from(k) * 32;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        if heap.samplers() {
            return;
        }
        let (Some(host), Some(slot)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        if let Some(view) = host.mirror(slot, handle, true) {
            if let Some(kind) = view_kind(&view) {
                readable(list, kind, &view);
            }
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(slot, true);
            }
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        compute: bool,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        for index in indexed.iter().filter(|index| !fresh.contains(&index.table_slot)) {
            TABLE_INDEXED.fetch_add(1, Ordering::Relaxed);
            if table_indexed() {
                self.table_indexed_entry(list, index, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                continue;
            }
            // Before the records are read ([`HostHeap::drawn_fresh`]).
            let seq = crate::descriptor::host::sync_seq();
            // The table and what its entries hold (record words), and each
            // entry's record key ([`HostHeap::drawn_fresh`]).
            let mut read = Vec::with_capacity(n as usize);
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    let words = unsafe { (at as *const [u32; 3]).read_unaligned() };
                    words.hash(&mut h);
                    read.push(crate::descriptor::host::words_key(words));
                }
                (table, n, h.finish())
            };
            if let Some(&first) = self.fresh_tables.get(&key) {
                if let Some(title) = heap.index(table) {
                    for k in 0..n {
                        host.drawn_fresh(title + k, seq, read[k as usize]);
                    }
                }
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view);
                    }
                }
                for k in selected {
                    if entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        readable(list, kind, &view);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            let Some(first) = host.dynamic(n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                continue;
            };
            self.fresh_tables.insert(key, first);
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, first, n));
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                if let Some(title) = title_first {
                    host.drawn_fresh(title + k, seq, read[k as usize]);
                }
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        if let Some(gpu) = table_constant_on_card(list, address, size) {
                            host.constant_at(index, gpu, size);
                            continue;
                        }
                        // Copied on the card instead of filled on the CPU
                        // (`VICTORY_TABLE_GPU_COPY`).
                        if let Some(gpu) = list.card_constant(address, size) {
                            host.constant_at(index, gpu, size);
                            continue;
                        }
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize, crate::host::FILL_TABLE);
                        host.constant_at(index, gpu, size);
                    }
                    Some(view) => {
                        host.make_at(index, &view);
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view);
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror(title + k, table + u64::from(k) * 32, true);
                            }
                        }
                    }
                    None => {
                        host.make_null(index);
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        let _timed = RecordTimed::new(rec::DRAWS_NS, Some(rec::DRAWS));
        self.window = None;
        self.amplified_run = false;
        self.amplified_reads_before_stores(list);
        self.end_batch_if_stores(list);
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && !self.completes_rects();
        let ran = ready && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            list.prof_draw(false);
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            list.prof_draw(true);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // The vertices are read in place, at submission: no change
                // of mapping under the read (see
                // `crate::watch::title_memory_reading`).
                let _mapped = crate::watch::title_memory_reading();
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        let _timed = RecordTimed::new(rec::DRAWS_NS, Some(rec::DRAWS));
        self.window = None;
        self.amplified_run = false;
        self.amplified_reads_before_stores(list);
        self.end_batch_if_stores(list);
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
        }
        let ran = !lost && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            list.prof_draw(false);
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            list.prof_draw(true);
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        let _timed = RecordTimed::new(rec::DISPATCHES_NS, Some(rec::DISPATCHES));
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        // A program that walks ray-tracing structures nothing has built:
        // dropped before the argument clamp, like a skipped one.
        if crate::pipeline::walk_held(self.pipeline as This) {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_why("a walk of ray-tracing structures that are not built (VICTORY_BVH_HOLD=1)".to_owned());
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        // Indirect rectangles (the title's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        let Some(signature) = matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
            .then(|| host_signature(kind, stride))
            .flatten()
        else {
            why("no host signature");
            return false;
        };
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // A mesh pipeline with the amplification stage reads and clamps its
        // records itself: no pass of ours, no wait between the draws.
        if kind == MESH
            && let Some(at) = p.amplify
        {
            let count = counter.as_ref().map(|(s, at)| (s, *at));
            self.end_batch_if_stores(list);
            return match self.amplified_indirect(list, p, at, (stride, max), (&args, arguments), count) {
                Ok(()) => true,
                Err(what) => {
                    why(what);
                    false
                }
            };
        }
        // What follows may write (our clamp, a dispatch, a draw that stores).
        self.amplified_run = false;
        self.amplified_reads_done(list);
        let stores = p.stages.iter().any(|stage| stage.translated().stores);
        let offset = args.offset(arguments);
        // A run of calls whose records one pass clamps
        // (`crate::indirect::batch_mode`); one it cannot take is clamped
        // alone below.
        let in_run = if crate::indirect::batch_mode() > 0 {
            self.window = None;
            args.refresh(list, arguments, len);
            if let Some((s, at)) = &counter {
                s.refresh(list, *at, 4);
            }
            let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
            // With runs ending at any write, a call whose shaders store
            // (1: and any dispatch) ends its run.
            let rules = crate::indirect::run_rules();
            let last = rules.call_ends(kind, stores);
            let joined = crate::indirect::batched(list, &mut self.batch, (&args.buffer, offset), count_arg, (kind, stride, max), last).map(
                |(scratch, at, began)| {
                    if began {
                        // Its pass bound a compute pipeline of ours.
                        self.host_pipeline = 0;
                        self.host_roots[1] = 0;
                    } else if rules.order && self.dispatched {
                        // Where 1 would have begun a run (a dispatch since
                        // the last wait), a wait without the pass.
                        list.wait_for_writes(crate::barstat::Why::RunOrder, &[]);
                        crate::indirect::note_order_wait();
                    }
                    (scratch, at)
                },
            );
            // Clamped in a run or (below) alone: either way the work before
            // has been waited for.
            self.dispatched = false;
            joined
        } else {
            None
        };
        let batched = in_run.is_some();
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let windowed = windows()
            && crate::indirect::batch_mode() == 0
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        let first_on_list = list.first_use(WINDOW_USE);
        if !windowed || first_on_list {
            self.window = None;
        }
        // Diagnostics: why this command gets a pass of ours, if it does.
        #[cfg(feature = "diagnostics")]
        let why_not = if !windows() {
            usize::MAX
        } else if kind == DISPATCH {
            1
        } else if max != 1 {
            2
        } else if count.is_some() {
            3
        } else if !windowed {
            4
        } else if first_on_list {
            5
        } else {
            match &self.window {
                None => 6,
                Some(w) if !(w.copy == args.id() && w.kind == kind && w.stride == stride) => 6,
                Some(w) if offset >= w.base && (offset - w.base) % u64::from(w.step) == 0 => 8,
                Some(_) => 7,
            }
        };
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = self
            .window
            .as_ref()
            .filter(|w| same(w) && offset >= w.base)
            .filter(|w| (offset - w.base) % u64::from(w.step) == 0 && (offset - w.base) / u64::from(w.step) < u64::from(w.records))
            .map(|w| (w.scratch.clone(), offset - w.base, w.step));
        let hit = match hit {
            Some((scratch, at, step)) => {
                // Diagnostics of windows: the record clamped again and
                // compared (`crate::indirect::window_check`).
                if crate::indirect::window_check()
                    && crate::indirect::check_hit(
                        list,
                        (&args.buffer, offset),
                        (kind, step),
                        &scratch,
                        at,
                    )
                {
                    self.host_pipeline = 0;
                    self.host_roots[1] = 0;
                }
                #[cfg(feature = "diagnostics")]
                WINDOW_WHY[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if !self.window_read {
                    self.window_read = true;
                    if let Some(w) = self.window.as_ref().filter(|w| w.records > 1) {
                        self.walk_step = Some((w.copy, w.kind, w.stride, w.step));
                    }
                }
                Some((scratch, at))
            }
            None => None,
        };
        let (bounded, at) = match in_run.or(hit) {
            Some(hit) => hit,
            None => {
                // A window opened at the remembered step that no draw read
                // from: the title does not walk here by that step.
                if self.window_jumped && !self.window_read {
                    self.walk_step = None;
                    #[cfg(feature = "diagnostics")]
                    WINDOW_WHY[11].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                self.window_jumped = false;
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                // A record that would be clamped alone, on a buffer whose
                // step is remembered: the window the next record would
                // open, opened here ([`jump_windows`]).
                let (step, records) = match (windowed && records == 1 && jump_windows())
                    .then(|| jump_window(self.walk_step, (args.id(), kind, stride), args.left(arguments)))
                    .flatten()
                {
                    Some(jumped) => {
                        self.window_jumped = true;
                        #[cfg(feature = "diagnostics")]
                        WINDOW_WHY[10].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        jumped
                    }
                    None => (step, records),
                };
                self.window_read = false;
                args.refresh(list, arguments, len.max(u64::from(step) * u64::from(records)));
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                // Bounded first: the clamp binds a compute pipeline of its
                // own and copies clamped active records into a scratch
                // buffer at offset 0.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                let bounded = if windowed && records > 1 {
                    crate::indirect::bound_ahead(list, (&args.buffer, offset), (kind, step, records))
                } else {
                    crate::indirect::bound(list, (&args.buffer, offset), count_arg, (kind, step, records))
                };
                let Some(bounded) = bounded else {
                    self.window = None;
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                #[cfg(feature = "diagnostics")]
                if why_not < WINDOW_WHY.len() {
                    WINDOW_WHY[why_not].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    WINDOW_WHY[9].fetch_add(u64::from(records), std::sync::atomic::Ordering::Relaxed);
                    // One in 256 of the draws that miss their window, said:
                    // where the record lies against the window and against
                    // the record drawn before it.
                    if (6..=8).contains(&why_not) {
                        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) % 256 == 0 {
                            let (last_copy, last_offset) = self.last_indirect;
                            tracing::info!(
                                "a window miss (one in 256): cause {why_not}, kind {kind}, stride {stride}, buffer {} offset {offset:#x} ({} left); the command before it on the list: buffer {last_copy} offset \
                                 {last_offset:#x} (this one is {} bytes on); the window it missed: {}; clamped now: {records} records at step {step}",
                                args.id(),
                                args.left(arguments),
                                offset as i64 - last_offset as i64,
                                match &self.window {
                                    Some(w) => format!("buffer {} kind {} stride {} base {:#x} step {} records {} (this record is {} bytes from its base)", w.copy, w.kind, w.stride, w.base, w.step, w.records, offset as i64 - w.base as i64),
                                    None => "none".to_string(),
                                }
                            );
                        }
                    }
                }
                self.host_pipeline = 0;
                self.host_roots[1] = 0;
                self.window = windowed.then(|| Window {
                    copy: args.id(),
                    kind,
                    stride,
                    step,
                    base: offset,
                    records,
                    scratch: bounded.clone(),
                });
                (bounded, 0)
            }
        };
        #[cfg(feature = "diagnostics")]
        {
            self.last_indirect = (args.id(), offset);
        }
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            why("prepare failed");
            return false;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        if kind == MESH {
            self.mesh_scissor(list, true);
        }
        list.prof_draw(false);
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &bounded,
                at,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        list.prof_draw(true);
        if kind == MESH {
            self.mesh_scissor(list, false);
        }
        // The scratch buffer stays readable as arguments: the next draw of
        // the window reads it with no barrier, and the next clamp moves it
        // back (`crate::indirect`). A run's buffer is left so too.
        crate::indirect::note_call(kind, batched, count.is_some());
        // Its shaders may write records later calls of the run read.
        let rules = crate::indirect::run_rules();
        if rules.call_ends(kind, stores) {
            self.end_batch(list);
        }
        if kind == DISPATCH && rules.on {
            self.dispatched = true;
            if self.batch.is_some() {
                crate::indirect::note_run_kept();
            }
        }
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

impl State {
    /// Before a dispatch or a draw that may write: if amplified indirect mesh
    /// draws were recorded since the last wait for every shader write, a wait
    /// for them, so that what follows cannot rewrite the records (or the
    /// count) their stage may still read ([`amplify_wait`]).
    fn amplified_reads_done(&mut self, list: &HostList) {
        if std::mem::take(&mut self.amplified_reads) && amplify_wait() >= 1 {
            crate::indirect::wait_for_writes(list);
            AMPLIFIED_WAITS.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The host list is closed (the title closes its list, or it is cut at a
    /// wait or a value write): amplified draws it holds are waited for at its
    /// end. Lists of one submission may overlap on the card, and the next
    /// one's first dispatch may rewrite records those draws still read.
    pub(crate) fn host_list_closing(&mut self, list: &HostList) {
        self.amplified_run = false;
        self.amplified_reads_done(list);
    }

    /// [`State::amplified_reads_done`] before a draw, if its shaders store.
    fn amplified_reads_before_stores(&mut self, list: &HostList) {
        if self.amplified_reads
            && crate::pipeline::host_pipeline(self.pipeline as This)
                .is_some_and(|p| p.stages.iter().any(|stage| stage.translated().stores))
        {
            self.amplified_reads_done(list);
        }
    }

    /// [`State::execute_indirect`] of mesh records by a pipeline with the
    /// host's amplification stage (`VICTORY_MESH_AMPLIFY=1`,
    /// `victory_shader::emit::amplification_shader`): `DispatchMesh(max)`
    /// of it, group `i` reading record `i` (`stride` bytes apart, at title
    /// address `arguments` of the host copy `args`), bounded by the count at
    /// `count` if given, clamping it as `indirect::bound` would and launching
    /// its mesh groups. In place of that clamp's compute pass between two
    /// waits for every earlier shader write (the card drained before and
    /// after every record: 7-12 us a draw on an RTX 5070 Ti for a draw of
    /// 0.3 us), one such wait is recorded where a run of these draws starts
    /// (the title's argument passes are done by then); the draws of a run
    /// write nothing the next reads.
    fn amplified_indirect(
        &mut self,
        list: &HostList,
        p: &crate::pipeline::host::HostPipeline,
        at: u32,
        (stride, max): (u32, u32),
        (args, arguments): (&std::sync::Arc<crate::shadow::Shadow>, u64),
        count: Option<(&std::sync::Arc<crate::shadow::Shadow>, u64)>,
    ) -> Result<(), &'static str> {
        use victory_shader::emit::{AMPLIFY_COUNTED, AMPLIFY_RECORDS};
        /// The key of the list's first amplified draw ([`HostList::first_use`]).
        const AMPLIFIED_USE: usize = usize::MAX - 0x78;
        if max > 65_535 {
            return Err("more records than one amplified launch runs");
        }
        if stride < 12 || !stride.is_multiple_of(4) {
            return Err("mesh records not 4-byte aligned");
        }
        self.window = None;
        let len = u64::from(stride) * u64::from(max.max(1));
        args.refresh(list, arguments, len);
        if let Some((s, place)) = count {
            s.refresh(list, place, 4);
        }
        if !self.prepare_for(list, false, true) {
            return Err("prepare failed");
        }
        let budget = list.budget().ok_or("no budget for the clamp")?;
        // SAFETY: live host copies.
        let records = unsafe { args.buffer.GetGPUVirtualAddress() } + args.offset(arguments);
        let (counted, flags) = match count {
            // SAFETY: as above.
            Some((s, place)) => (unsafe { s.buffer.GetGPUVirtualAddress() } + s.offset(place), AMPLIFY_RECORDS | AMPLIFY_COUNTED),
            None => (records, AMPLIFY_RECORDS),
        };
        if !records.is_multiple_of(4) || !counted.is_multiple_of(4) {
            return Err("mesh records not 4-byte aligned");
        }
        // Earlier shader writes (the title's argument passes) first, once a run.
        let first = list.first_use(AMPLIFIED_USE);
        let stores = p.stages.iter().any(|stage| stage.translated().stores);
        // A draw that stores may write what earlier draws of the run still
        // read: their reads first, as the clamp pass's wait did.
        let before_stores = stores && self.amplified_reads && amplify_wait() >= 1;
        if !self.amplified_run || first || before_stores || amplify_wait() == 2 {
            crate::indirect::wait_for_writes(list);
        }
        self.amplified_run = !stores;
        self.amplified_reads = true;
        let Ok(l6) = list.list.cast::<ID3D12GraphicsCommandList6>() else {
            return Err("no mesh dispatch on the host list");
        };
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("amplified indirect max {max} at {arguments:#x}"), false));
        let launch = [flags, 0, stride, 0];
        // SAFETY: an open list with everything set; the root signature has the
        // stage's parameters at `at`; the host copies and the budget buffer are
        // kept alive by the list.
        unsafe {
            list.list.SetGraphicsRoot32BitConstants(at, 4, launch.as_ptr().cast(), 0);
            list.list.SetGraphicsRootShaderResourceView(at + 1, records);
            list.list.SetGraphicsRootShaderResourceView(at + 2, counted);
            list.list.SetGraphicsRootUnorderedAccessView(at + 3, budget);
        }
        self.mesh_scissor(list, true);
        list.prof_draw(false);
        // SAFETY: as above.
        unsafe { l6.DispatchMesh(max, 1, 1) };
        list.prof_draw(true);
        self.mesh_scissor(list, false);
        DRAWN.fetch_add(1, Ordering::Relaxed);
        AMPLIFIED.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }
}

/// Indirect mesh draws launched by the amplification stage ([`State::amplified_indirect`]).
pub(crate) static AMPLIFIED: AtomicU64 = AtomicU64::new(0);
/// Waits recorded where a run of them ends ([`State::amplified_reads_done`]).
pub(crate) static AMPLIFIED_WAITS: AtomicU64 = AtomicU64::new(0);

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source));
            if !aliased
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// The pixel shaders whose draws get their slots listed
/// ([`State::slots_text`]): `VICTORY_LIST_SLOTS`, hash prefixes separated
/// by commas. None by default.
#[cfg(feature = "diagnostics")]
fn list_slots() -> &'static [String] {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_LIST_SLOTS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

/// A fingerprint of the 4 KiB of title memory at `at` (FNV-1a, 64 bits; of
/// 4 KiB of zeros it is always the same value, which `slots.py` knows), or
/// "unreadable".
#[cfg(feature = "diagnostics")]
fn print_4k(at: u64) -> String {
    let mut bytes = [0u8; 4096];
    if at == 0 || !crate::host::read_into(at as *const u8, &mut bytes) {
        return "unreadable".into();
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The `bytes` of title memory at `at`: how many are mapped, and the
/// fingerprints of the first and the last 4 KiB (a texture's smallest
/// levels lie first, its largest last).
#[cfg(feature = "diagnostics")]
fn memory_state(at: u64, bytes: u64) -> String {
    let mapped = crate::texture::mapped_bytes(at as *const u8, bytes as usize);
    let last = at + bytes.saturating_sub(4096);
    format!("memory={at:#x} mapped={mapped}/{bytes} first4k={} last4k={}", print_4k(at), print_4k(last))
}

/// Title texture `this` for the frame listing: its size, whether it has a
/// host copy (not made here), its staging buffer's state, whether the host
/// copied into it, and its memory ([`memory_state`]).
#[cfg(feature = "diagnostics")]
fn texture_state(this: This) -> String {
    let Some((at, desc)) = crate::resource::texture_memory(this) else {
        return "not a texture".into();
    };
    // SAFETY: 56 bytes in the description's layout.
    let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
    let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
    let staged = match crate::texture::staged_state(this) {
        Some((uploaded, dirty)) => format!("staged=yes uploaded={uploaded} dirty={dirty}"),
        None => "staged=no".into(),
    };
    format!(
        "{}x{} f{} levels={} host_copy={} {staged} host_written={} title_fills={} {}",
        u64::from(words[4]) | u64::from(words[5]) << 32,
        words[6],
        words[8],
        words[7] >> 16,
        crate::resource::host_made(this).is_some(),
        crate::texture::host_wrote(this as usize),
        crate::resource::title_fills(this),
        memory_state(at as u64, bytes)
    )
}

/// A view of the title's heap for the frame listing: a texture with its
/// state ([`texture_state`]; a placed one that has no resource yet with
/// its memory alone), anything else in a few words.
#[cfg(feature = "diagnostics")]
fn view_state(view: Option<&View>) -> String {
    match view {
        None => "no view".into(),
        Some(View::Placed(false, r, d)) if r[0] != 1 => {
            let address = u64::from(d[10]) | u64::from(d[11]) << 32;
            let state = match crate::resource::texture_at_made(address, r) {
                Some(this) => texture_state(this),
                None => format!(
                    "{}x{} f{} levels={} resource=none {}",
                    u64::from(r[4]) | u64::from(r[5]) << 32,
                    r[6],
                    r[8],
                    r[7] >> 16,
                    memory_state(address, crate::resource::placed_bytes(r).unwrap_or(0))
                ),
            };
            // (Last, so that readers of the older lines still find
            // theirs: the level the view starts at, and how many it shows.)
            format!("placed texture view_format={} {state} view_levels={}+{}", d[0], d[4], d[5] as i32)
        }
        Some(View::Shader(r, d)) if crate::resource::texture_memory(*r as This).is_some() => {
            format!(
                "texture object {r:#x} view_format={} {} view_levels={}",
                d.as_ref().map_or(0, |d| d[0]),
                texture_state(*r as This),
                d.as_ref().map_or("all".to_owned(), |d| format!("{}+{}", d[4], d[5] as i32))
            )
        }
        Some(v) => format!("{v:?}").chars().take(80).collect(),
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => format!(
                "uav resource {r:#x} buffer {:?}",
                crate::descriptor::host::buffer_range(v)
            ),
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

pub(crate) fn depth_bounds(list: &HostList, min: f32, max: f32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("DEPTH BOUNDS {min}..{max}")
    });
    if crate::host::host().is_some_and(|h| h.depth_bounds())
        && let Ok(l1) = list.list.cast::<ID3D12GraphicsCommandList1>()
    {
        // SAFETY: an open list on a device that supports depth bounds.
        unsafe { l1.OMSetDepthBounds(min, max) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every slot is resolved anew exactly once in 256 passes, whether a
    /// sixteenth is rescanned each pass or all of them every 16th pass.
    #[test]
    fn every_slot_is_resolved_anew_once_in_256_passes() {
        for index in 0..4096u32 {
            let rolling = (0..256u64)
                .filter(|&pass| pass % 16 == u64::from(index % 16) && deep_rescan(index, pass))
                .count();
            assert_eq!(rolling, 1, "slot {index}, a sixteenth a pass");
            let full = (0..256u64)
                .filter(|&pass| pass % 16 == 0 && deep_rescan(index, pass))
                .count();
            assert_eq!(full, 1, "slot {index}, all every 16th pass");
        }
    }

    /// The pass's maps hold slot numbers and aligned addresses apart.
    #[test]
    fn slot_maps_keep_their_keys_apart() {
        let mut slots: SlotMap<u32, u32> = SlotMap::default();
        let mut copies: SlotMap<usize, usize> = SlotMap::default();
        for i in 0..100_000u32 {
            slots.insert(i, i);
            copies.insert(0x1_0000_0000 + i as usize * 256, i as usize);
        }
        assert_eq!((slots.len(), copies.len()), (100_000, 100_000));
        assert!((0..100_000u32).all(|i| slots[&i] == i && copies[&(0x1_0000_0000 + i as usize * 256)] == i as usize));
    }

    #[test]
    fn a_lone_record_opens_a_window_at_the_remembered_step() {
        // (a mesh dispatch is kind 10, a draw kind 0)
        let remembered = Some((7u64, 10u32, 12u32, 32u32));
        // the same buffer, kind and stride: sixteen records at the title's step
        assert_eq!(jump_window(remembered, (7, 10, 12), 1 << 20), Some((32, 16)));
        // never past the buffer: 100 bytes left hold three records 32 apart
        assert_eq!(jump_window(remembered, (7, 10, 12), 100), Some((32, 3)));
        // one record fits: nothing to open
        assert_eq!(jump_window(remembered, (7, 10, 12), 40), None);
        // another buffer, kind or stride, or nothing remembered: alone, as before
        assert_eq!(jump_window(remembered, (8, 10, 12), 1 << 20), None);
        assert_eq!(jump_window(remembered, (7, 0, 12), 1 << 20), None);
        assert_eq!(jump_window(remembered, (7, 10, 16), 1 << 20), None);
        assert_eq!(jump_window(None, (7, 10, 12), 1 << 20), None);
    }
}
