//! Title buffer copies whose source flights in the air write
//! (`VICTORY_CARD_READ_COPY=1`; off by default; `2` adds the constant
//! buffer reads below, `3` the sources flights wrote lately).
//!
//! Why: a flight declares the ranges its shaders may store to, and the title
//! copies out of such a range every frame (in the prologue cell, 2 KiB at a
//! time from one source into two buffers, a few bytes of the source changed
//! under the flight 202 times of 212 checks).
//! The CPU copies at submission, so it settled the flights first to read the
//! bytes the shaders wrote: 8 to 10 ms of the submitting thread a frame, the
//! "title memory read" settle. `fill_ahead` refuses these copies for good
//! after the first one that took stale bytes.
//!
//! What it does: the copy is also recorded on the card, from the source's
//! card copy to the destination's, behind the flights (the queue runs them
//! first, and the copy's barrier waits for their stores), and the
//! destination is declared written, so that the copy-back puts the bytes into
//! title memory once the list has run, as it does for any range shaders
//! write. The title sees them when its list is done, as on the console.
//! At submission the CPU leaves the copy to the card unless something else in
//! the submission needs the bytes in title memory first ([`places`]).
//!
//! The something else can be a fill of upload memory: a constant buffer the
//! console reads at 16 bytes (not 256) is copied into upload memory at
//! submission, and that fill reads the destination on the CPU. With `2` the
//! list copies such a constant buffer on the card instead, from the card's
//! copy of the destination into a buffer of its own
//! ([`HostList::card_constant`]), so that no step on the CPU reads the
//! destination and the whole chain stays on the card.
//!
//! A copy kept on the CPU also waits, whatever its source does: the card's
//! copy declares the destination written, so the next frame's CPU copy into
//! it finds the previous frame's flights writing it and settles them first
//! (`blocked_by`). With `2` a copy no flight writes at submission is left to
//! the card too, unless a later step pins it.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::{HostList, Step, host, line, touches, transition};

/// The level of the switch: 0 off, 1 card copies, 2 also constant buffers
/// read from the card, 3 also copies of sources flights wrote lately.
fn level() -> u8 {
    static LEVEL: OnceLock<u8> = OnceLock::new();
    *LEVEL.get_or_init(
        || match std::env::var("VICTORY_CARD_READ_COPY").as_deref() {
            Ok("1") => 1,
            Ok("2") => 2,
            Ok("3") => 3,
            _ => 0,
        },
    )
}

/// Whether the card does these copies (`VICTORY_CARD_READ_COPY=1` to `3`).
pub(super) fn on() -> bool {
    level() >= 1
}

/// Whether the whole chain stays on the card (`VICTORY_CARD_READ_COPY=2` or
/// `3`):
/// constant buffers that read a card copy's destination are copied on the
/// card, and a copy is left to the card whether or not a flight writes its
/// source at submission ([`places`]).
fn chain() -> bool {
    level() >= 2
}

/// Whether a source flights wrote lately counts as written
/// (`VICTORY_CARD_READ_COPY=3`).
fn lately() -> bool {
    level() >= 3
}

/// Since the last frame line: copies recorded on the card; copies a
/// submission left to the card; copies it kept on the CPU; copies recorded
/// on the card only because their source was written lately; copies that
/// had to stay copies by the CPU for want of a host copy.
static COUNT: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

/// Since the last frame line: the copies kept on the CPU, by [`Pin`].
static PINNED: [AtomicU64; PINS] = [const { AtomicU64::new(0) }; PINS];

/// Since the last frame line: constant buffers copied on the card
/// ([`HostList::card_constant`]) and their bytes.
static CONSTANTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// What a step of a submission does with title memory on the CPU, as far as
/// the choice of a copy's place cares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Access {
    /// A copy by the CPU: destination, source, bytes.
    Copy(usize, usize, usize),
    /// A [`Step::CardCopy`]: destination, source, bytes.
    Card(usize, usize, usize),
    /// A fill of upload memory from title memory (address, bytes).
    Fill(usize, usize),
    /// A texture's staging buffer read from title memory (address, bytes).
    Texture(usize, usize),
    /// Work that may read and write title memory anywhere.
    Anywhere,
    Other,
}

impl Access {
    fn of(step: &Step) -> Access {
        match step {
            &Step::Copy(dst, src, len) => Access::Copy(dst, src, len),
            &Step::CardCopy(dst, src, len, _) => Access::Card(dst, src, len),
            &Step::Fill(_, src, len) => Access::Fill(src, len),
            Step::Texture(staging) => {
                let (at, len) = staging.memory();
                Access::Texture(at, len)
            }
            Step::Run(..) => Access::Anywhere,
            Step::Refresh(..) | Step::GpuWrite(..) | Step::Host(_) => Access::Other,
        }
    }
}

/// Why a [`Step::CardCopy`] is done on the CPU at submission.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Pin {
    /// No flight in the air writes the source any more: the copy waits for
    /// nothing, and this is no pin.
    Settled,
    /// Work of ours that may go anywhere, before or after.
    Anywhere,
    /// An earlier step writes the source on the CPU.
    SourceWritten,
    /// A later copy reads or writes the destination.
    LaterCopy,
    /// A later fill of upload memory reads the destination.
    LaterFill,
    /// A later texture reads the destination.
    LaterTexture,
    /// A later card copy kept on the CPU reads or writes the destination.
    LaterCard,
}

/// How many [`Pin`]s there are.
const PINS: usize = 7;

/// Where a step of a submission is done.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Place {
    /// On the CPU, as the step it is: not a card copy.
    Cpu,
    /// A card copy left to the card.
    Card,
    /// A card copy kept on the CPU, for this reason, by the step at this
    /// index.
    Kept(Pin, usize),
}

impl Place {
    /// Whether the CPU does the step.
    fn on_cpu(&self) -> bool {
        !matches!(self, Place::Card)
    }
}

/// A range of title memory a step of a submission reads or writes on the
/// CPU, and what that makes of a card copy it meets.
struct Reach {
    from: usize,
    to: usize,
    by: usize,
    pin: Pin,
}

impl Reach {
    fn new(at: usize, len: usize, by: usize, pin: Pin) -> Reach {
        Reach {
            from: at,
            to: at + len,
            by,
            pin,
        }
    }
}

/// The first step, in order, whose reach meets `[from, to)`, with its pin.
fn nearest(reaches: &[Reach], from: usize, to: usize) -> Option<(Pin, usize)> {
    reaches
        .iter()
        .filter(|r| r.from < to && from < r.to)
        .map(|r| (r.by, r.pin))
        .min_by_key(|&(by, _)| by)
        .map(|(by, pin)| (pin, by))
}

/// For each step of a submission, in order: where it is done. Only a
/// [`Access::Card`] step can be left to the card, and it is when flights in
/// `written` (in order, joined) write its source, or whatever `all` says
/// that, and nothing else of the submission depends on title memory holding
/// the copy or its source before the card has run:
/// - a later step reads or writes the destination on the CPU (a fill of
///   upload memory that takes the copied bytes, a copy of them onward, a
///   second copy into the same bytes, a texture, work of ours that goes
///   anywhere);
/// - an earlier step writes the source on the CPU, so that the card's copy
///   of it lacks those bytes, or goes anywhere.
///
/// A copy whose source no flight writes costs the CPU nothing to make, but
/// the card's copy declared its destination written, and a CPU copy into a
/// range flights in the air may write waits for them: with `all` those
/// copies are left to the card as well.
///
/// A card copy kept on the CPU is still also done on the card, which is
/// why the steps around it need not look at its own place.
pub(super) fn places(steps: &[Access], written: &[(usize, usize)], all: bool) -> Vec<Place> {
    let early = early_hazards(steps);
    let mut places = vec![Place::Cpu; steps.len()];
    let mut later: Vec<Reach> = Vec::new();
    let mut later_anywhere: Option<usize> = None;
    for (i, step) in steps.iter().enumerate().rev() {
        match *step {
            Access::Copy(dst, src, len) => {
                later.push(Reach::new(dst, len, i, Pin::LaterCopy));
                later.push(Reach::new(src, len, i, Pin::LaterCopy));
            }
            Access::Card(dst, src, len) => {
                let pinned = later_anywhere
                    .map(|by| (Pin::Anywhere, by))
                    .or(early[i])
                    .or_else(|| nearest(&later, dst, dst + len));
                places[i] = match pinned {
                    _ if !all && !touches(written, src, src + len) => Place::Kept(Pin::Settled, i),
                    Some((pin, by)) => Place::Kept(pin, by),
                    None => Place::Card,
                };
                if places[i].on_cpu() {
                    later.push(Reach::new(dst, len, i, Pin::LaterCard));
                    later.push(Reach::new(src, len, i, Pin::LaterCard));
                }
            }
            Access::Fill(at, len) => later.push(Reach::new(at, len, i, Pin::LaterFill)),
            Access::Texture(at, len) => later.push(Reach::new(at, len, i, Pin::LaterTexture)),
            Access::Anywhere => later_anywhere = Some(i),
            Access::Other => {}
        }
    }
    places
}

/// For each step: what, before a [`Access::Card`] step, writes its source
/// on the CPU or goes anywhere, and the index of that step.
fn early_hazards(steps: &[Access]) -> Vec<Option<(Pin, usize)>> {
    let mut early = vec![None; steps.len()];
    let mut wrote: Vec<Reach> = Vec::new();
    let mut anywhere: Option<usize> = None;
    for (i, step) in steps.iter().enumerate() {
        match *step {
            Access::Copy(dst, _, len) => wrote.push(Reach::new(dst, len, i, Pin::SourceWritten)),
            Access::Card(_, src, len) => {
                early[i] = anywhere
                    .map(|by| (Pin::Anywhere, by))
                    .or_else(|| nearest(&wrote, src, src + len));
            }
            Access::Anywhere => {
                anywhere.get_or_insert(i);
            }
            Access::Fill(..) | Access::Texture(..) | Access::Other => {}
        }
    }
    early
}

/// Before a submission is prepared: sets the place of each of its card
/// copies ([`Step::CardCopy`]) from what flights in the air write now. A
/// copy that is not set stays on the CPU, as a plain title copy.
pub(super) fn decide(runs: &[&HostList]) {
    let any = || {
        runs.iter().any(|list| {
            list.deferred()
                .copies
                .iter()
                .any(|s| matches!(s, Step::CardCopy(..)))
        })
    };
    if !on() || !any() {
        return;
    }
    let written = line().written.clone();
    let steps: Vec<Access> = runs
        .iter()
        .flat_map(|list| {
            list.deferred()
                .copies
                .iter()
                .map(Access::of)
                .collect::<Vec<_>>()
        })
        .collect();
    let places = places(&steps, &written, chain());
    let mut at = 0;
    for list in runs {
        for step in &list.deferred().copies {
            if let Step::CardCopy(.., flag) = step {
                flag.store(places[at].on_cpu(), Relaxed);
                count(&steps, at, places[at]);
            }
            at += 1;
        }
    }
}

/// Counts the place of the card copy at step `at` for the frame line, and
/// says the first few pins in diagnostics builds.
fn count(steps: &[Access], at: usize, place: Place) {
    match place {
        Place::Kept(pin, by) => {
            COUNT[2].fetch_add(1, Relaxed);
            PINNED[pin as usize].fetch_add(1, Relaxed);
            #[cfg(feature = "diagnostics")]
            say_pin(steps, at, pin, by);
            #[cfg(not(feature = "diagnostics"))]
            let _ = (steps, at, by);
        }
        Place::Card => {
            COUNT[1].fetch_add(1, Relaxed);
        }
        Place::Cpu => {}
    }
}

/// Diagnostics: the first 32 copies kept on the CPU for a reason that is
/// a pin, with the step that pinned them.
#[cfg(feature = "diagnostics")]
fn say_pin(steps: &[Access], at: usize, pin: Pin, by: usize) {
    static SAID: AtomicU64 = AtomicU64::new(0);
    if pin == Pin::Settled || SAID.fetch_add(1, Relaxed) >= 32 {
        return;
    }
    tracing::info!(
        "a title buffer copy done on the card stays on the CPU: {:?}; the copy is step {at} ({:?}), the step that pins it is step {by} ({:?}), of {} steps; frame {}",
        pin,
        steps[at],
        steps[by],
        steps.len(),
        crate::frame::current()
    );
}

/// Whether flights in the air may write `[address, address + len)`.
fn flights_write(address: usize, len: usize) -> bool {
    touches(&line().written, address, address + len)
}

/// The destinations of the card copies recorded lately (start, end, the
/// frame), the newest last: constant buffers that read them are copied on
/// the card ([`HostList::card_constant`]).
static RECENT: Mutex<Vec<(usize, usize, u64)>> = Mutex::new(Vec::new());

/// How many destinations [`RECENT`] holds.
const RECENT_MOST: usize = 64;

/// A destination of `[from, to)` copied in frame `frame` is noted in
/// `recent`, in place of an earlier note of the same range.
fn note_recent(recent: &mut Vec<(usize, usize, u64)>, from: usize, to: usize, frame: u64) {
    recent.retain(|r| (r.0, r.1) != (from, to));
    if recent.len() == RECENT_MOST {
        recent.remove(0);
    }
    recent.push((from, to, frame));
}

/// Whether `recent` holds a destination that meets `[from, to)`, noted in
/// this frame or the one before.
fn recent_meets(recent: &[(usize, usize, u64)], from: usize, to: usize, frame: u64) -> bool {
    recent
        .iter()
        .any(|&(a, b, f)| f + 1 >= frame && a < to && from < b)
}

/// The ranges flights were sent to write lately (start, end, the frame
/// they were last declared in), for [`written_lately`].
static SEEN: Mutex<Vec<(usize, usize, u64)>> = Mutex::new(Vec::new());

/// How many ranges [`SEEN`] holds.
const SEEN_MOST: usize = 256;

/// How many frames a declared range counts as written after its last
/// declaration. The title declares the ranges of its big buffers in every
/// frame it draws; a few frames bridge a landing that empties the line.
const SEEN_FRAMES: u64 = 4;

/// `ranges`, declared written in frame `frame`, are noted in `seen`: a range
/// held already has its frame renewed; with `seen` full the one declared
/// longest ago makes room.
fn note_seen(seen: &mut Vec<(usize, usize, u64)>, ranges: &[(usize, usize)], frame: u64) {
    for &(from, to) in ranges.iter().filter(|r| r.0 < r.1) {
        if let Some(held) = seen.iter_mut().find(|s| (s.0, s.1) == (from, to)) {
            held.2 = frame;
            continue;
        }
        if seen.len() == SEEN_MOST {
            let oldest = (0..seen.len()).min_by_key(|&i| seen[i].2).unwrap_or(0);
            seen.swap_remove(oldest);
        }
        seen.push((from, to, frame));
    }
}

/// Whether `seen` holds a range that meets `[from, to)`, declared in `frame`
/// or at most `within` frames before it.
fn seen_meets(
    seen: &[(usize, usize, u64)],
    from: usize,
    to: usize,
    frame: u64,
    within: u64,
) -> bool {
    seen.iter()
        .any(|&(a, b, f)| f + within >= frame && a < to && from < b)
}

/// With `VICTORY_CARD_READ_COPY=3`: a submission was sent whose shaders may
/// write what `sent` says ([`Declared`](super::Declared) ranges and texture
/// read backs).
pub(super) fn note_sent(sent: &super::Sent) {
    if !lately() {
        return;
    }
    let mut ranges: Vec<(usize, usize)> = sent
        .back
        .iter()
        .flat_map(|(_, r)| r.iter().copied())
        .chain(sent.texture_back_ranges.iter().copied())
        .collect();
    super::join_ranges(&mut ranges, []);
    note_seen(
        &mut SEEN.lock().unwrap_or_else(|e| e.into_inner()),
        &ranges,
        crate::frame::current(),
    );
}

/// With `VICTORY_CARD_READ_COPY=3`: whether flights were sent lately that
/// may write `[address, address + len)`. A copy out of such a range is
/// taken to be a copy out of what flights write, even when none is in the
/// air as it is recorded.
fn written_lately(address: usize, len: usize) -> bool {
    lately()
        && seen_meets(
            &SEEN.lock().unwrap_or_else(|e| e.into_inner()),
            address,
            address + len,
            crate::frame::current(),
            SEEN_FRAMES,
        )
}

/// A barrier group of global barriers; `barriers` must outlive its use.
fn global_group(barriers: &[D3D12_GLOBAL_BARRIER]) -> D3D12_BARRIER_GROUP {
    D3D12_BARRIER_GROUP {
        Type: D3D12_BARRIER_TYPE_GLOBAL,
        NumBarriers: barriers.len() as u32,
        Anonymous: D3D12_BARRIER_GROUP_0 {
            pGlobalBarriers: barriers.as_ptr(),
        },
    }
}

/// The most bytes of a constant buffer the host reads
/// (`constant_view_size`).
const CONSTANT_MOST: u64 = 1 << 16;

impl HostList {
    /// A title buffer copy of `len` bytes from `src` to `dst` (title
    /// addresses) recorded on the card, in the list, between two barriers:
    /// the card's copy of the source (brought up to date from title memory
    /// first, since the shaders that wrote it ran in earlier lists and the
    /// title may have written bytes beside what they wrote) to the card's
    /// copy of the destination, whose range is then declared written, so
    /// that it reaches title memory when the list has run. At submission,
    /// where the CPU does a title's copy, the shaders have not run yet: the
    /// copy took what the source held before them. False if either side has
    /// no host copy; the caller then copies at submission.
    fn copy_on_card(&self, dst: u64, src: u64, len: u64) -> bool {
        let (Some(from), Some(to)) = (crate::shadow::find(src, len), crate::shadow::find(dst, len))
        else {
            return false;
        };
        let Ok(l7) = self.list.cast::<ID3D12GraphicsCommandList7>() else {
            return false;
        };
        from.refresh(self, src, len);
        to.refresh(self, dst, len);
        to.gpu_writes(self, dst, len);
        let barrier = |before_sync, after_sync, before, after| D3D12_GLOBAL_BARRIER {
            SyncBefore: before_sync,
            SyncAfter: after_sync,
            AccessBefore: before,
            AccessAfter: after,
        };
        let before = [barrier(
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,
            D3D12_BARRIER_ACCESS_COPY_SOURCE | D3D12_BARRIER_ACCESS_COPY_DEST,
        )];
        let after = [barrier(
            D3D12_BARRIER_SYNC_COPY,
            D3D12_BARRIER_SYNC_ALL,
            D3D12_BARRIER_ACCESS_COPY_DEST,
            D3D12_BARRIER_ACCESS_UNORDERED_ACCESS
                | D3D12_BARRIER_ACCESS_SHADER_RESOURCE
                | D3D12_BARRIER_ACCESS_COPY_SOURCE,
        )];
        // SAFETY: an open list; both buffers are live host copies the list
        // keeps until it is reset.
        unsafe {
            l7.Barrier(&[global_group(&before)]);
            self.list.CopyBufferRegion(
                &to.buffer,
                to.offset(dst),
                &from.buffer,
                from.offset(src),
                len,
            );
            l7.Barrier(&[global_group(&after)]);
        }
        self.keep(&to.buffer);
        self.keep(&from.buffer);
        true
    }

    /// A title buffer copy of `len` bytes from `src` to `dst` (title
    /// addresses), recorded on the card as well as for the CPU, if flights in
    /// the air write the source (with `3`, or wrote it lately): true when it
    /// was. False with the switch off, with no flight writing the source,
    /// or if either side has no host copy; the caller then records the copy
    /// as before.
    pub(crate) fn copy_after_flights(&self, dst: *mut u8, src: *const u8, len: usize) -> bool {
        if !on() {
            return false;
        }
        let in_air = flights_write(src as usize, len);
        if !in_air && !written_lately(src as usize, len) {
            return false;
        }
        if !self.copy_on_card(dst as u64, src as u64, len as u64) {
            COUNT[4].fetch_add(1, Relaxed);
            return false;
        }
        if !in_air {
            COUNT[3].fetch_add(1, Relaxed);
        }
        COUNT[0].fetch_add(1, Relaxed);
        note_recent(
            &mut RECENT.lock().unwrap_or_else(|e| e.into_inner()),
            dst as usize,
            dst as usize + len,
            crate::frame::current(),
        );
        let mut deferred = self.deferred();
        deferred.copies.push(Step::CardCopy(
            dst as usize,
            src as usize,
            len,
            std::sync::atomic::AtomicBool::new(true),
        ));
        deferred.title_written();
        if super::list_reuse_set() {
            deferred.filled.clear();
        }
        true
    }

    /// With `VICTORY_CARD_READ_COPY=2`: the host GPU address of `len` bytes
    /// (at most 64 KiB) of a constant buffer at title address `address`
    /// that a card copy wrote lately, copied on the card into a buffer of
    /// its own, 256-byte aligned, which the list keeps until it is reset.
    /// The CPU would fill upload memory with them at submission, a step that
    /// reads the card copy's destination in title memory and so keeps that
    /// copy on the CPU. None for any other range, with no host copy of it,
    /// or if the buffer cannot be made; the caller then fills upload memory
    /// as before. Asked again for the same bytes, it answers with the same
    /// buffer until a title copy intervenes.
    pub(crate) fn card_constant(&self, address: u64, len: u64) -> Option<u64> {
        let len = len.min(CONSTANT_MOST);
        let (from, to) = (address as usize, (address + len) as usize);
        if !chain()
            || !recent_meets(
                &RECENT.lock().unwrap_or_else(|e| e.into_inner()),
                from,
                to,
                crate::frame::current(),
            )
        {
            return None;
        }
        let key = (from, len as usize);
        let found = self.deferred().carded.get(&key).copied();
        if found.is_some() {
            return found;
        }
        let l7 = self.list.cast::<ID3D12GraphicsCommandList7>().ok()?;
        let shadow = crate::shadow::find(address, len)?;
        let buffer = host()?.uav_buffer(len.next_multiple_of(256))?;
        // The bytes beside the copy's destination come from title memory, as
        // the fill would have read them; the destination's own are the card's.
        shadow.refresh(self, address, len);
        let before = [D3D12_GLOBAL_BARRIER {
            SyncBefore: D3D12_BARRIER_SYNC_ALL,
            SyncAfter: D3D12_BARRIER_SYNC_COPY,
            AccessBefore: D3D12_BARRIER_ACCESS_UNORDERED_ACCESS | D3D12_BARRIER_ACCESS_COPY_DEST,
            AccessAfter: D3D12_BARRIER_ACCESS_COPY_SOURCE,
        }];
        let readable = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER
            | D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE
            | D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE;
        // SAFETY: an open list; both buffers are kept until it is reset. The
        // new buffer's own transitions order its copy before the reads that
        // follow, so no barrier is needed after the copy.
        unsafe {
            l7.Barrier(&[global_group(&before)]);
            self.list.ResourceBarrier(&[transition(
                &buffer,
                D3D12_RESOURCE_STATE_COMMON,
                D3D12_RESOURCE_STATE_COPY_DEST,
            )]);
            self.list
                .CopyBufferRegion(&buffer, 0, &shadow.buffer, shadow.offset(address), len);
            self.list.ResourceBarrier(&[transition(
                &buffer,
                D3D12_RESOURCE_STATE_COPY_DEST,
                readable,
            )]);
        }
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        self.keep(&buffer);
        self.keep(&shadow.buffer);
        self.deferred().carded.insert(key, gpu);
        CONSTANTS[0].fetch_add(1, Relaxed);
        CONSTANTS[1].fetch_add(len, Relaxed);
        Some(gpu)
    }
}

/// With the frame line.
pub(super) fn line_per_frame(frames: f64) {
    let [recorded, carded, kept, by_history, no_host_copy] =
        [0, 1, 2, 3, 4].map(|i| COUNT[i].swap(0, Relaxed));
    let pins = PINNED
        .each_ref()
        .map(|n| n.swap(0, Relaxed) as f64 / frames);
    let [constants, bytes] = [0, 1].map(|i| CONSTANTS[i].swap(0, Relaxed));
    if recorded == 0 && carded == 0 && kept == 0 && constants == 0 && no_host_copy == 0 {
        return;
    }
    tracing::info!(
        "title buffer copies from memory flights write, done on the card (VICTORY_CARD_READ_COPY={}): {:.1} per frame recorded ({:.1} only because flights wrote the source lately, {:.1} refused for want of a host copy); \
         {:.1} left to the card at submission, {:.1} kept on the CPU (no flight writing the source any more {:.1}, work of ours {:.1}, \
         an earlier write of the source {:.1}, a later copy {:.1}, a later fill of upload memory {:.1}, a later texture {:.1}, a later card copy {:.1}); \
         {:.1} constant buffers a frame read from the card's copy ({:.1} KB)",
        level(),
        recorded as f64 / frames,
        by_history as f64 / frames,
        no_host_copy as f64 / frames,
        carded as f64 / frames,
        kept as f64 / frames,
        pins[Pin::Settled as usize],
        pins[Pin::Anywhere as usize],
        pins[Pin::SourceWritten as usize],
        pins[Pin::LaterCopy as usize],
        pins[Pin::LaterFill as usize],
        pins[Pin::LaterTexture as usize],
        pins[Pin::LaterCard as usize],
        constants as f64 / frames,
        bytes as f64 / frames / 1024.0
    );
}

#[cfg(test)]
mod tests {
    use super::Access::{Anywhere, Card, Copy, Fill, Other, Texture};
    use super::*;

    const WRITTEN: [(usize, usize); 1] = [(0x1000, 0x2000)];
    const SRC: usize = 0x1100;
    const DST: usize = 0x9000;

    fn card() -> Access {
        Card(DST, SRC, 0x800)
    }

    fn on_cpu(steps: &[Access], written: &[(usize, usize)]) -> Vec<bool> {
        places(steps, written, false)
            .iter()
            .map(Place::on_cpu)
            .collect()
    }

    fn pin_of(steps: &[Access], written: &[(usize, usize)]) -> Place {
        places(steps, written, false)[0]
    }

    #[test]
    fn a_copy_from_what_flights_write_is_left_to_the_card() {
        assert_eq!(on_cpu(&[card()], &WRITTEN), [false]);
    }

    #[test]
    fn a_source_no_flight_writes_is_copied_by_the_cpu() {
        assert_eq!(on_cpu(&[Card(DST, 0x3000, 0x800)], &WRITTEN), [true]);
        assert_eq!(on_cpu(&[Card(DST, 0x0800, 0x800)], &WRITTEN), [true]);
        assert_eq!(on_cpu(&[card()], &[]), [true]);
    }

    #[test]
    fn a_later_cpu_read_of_the_destination_keeps_the_copy_on_the_cpu() {
        assert_eq!(
            on_cpu(&[card(), Fill(DST + 0x7ff, 16)], &WRITTEN),
            [true, true]
        );
        assert_eq!(
            on_cpu(&[card(), Copy(0x5000, DST, 16)], &WRITTEN),
            [true, true]
        );
    }

    #[test]
    fn a_later_cpu_write_into_the_destination_keeps_the_copy_on_the_cpu() {
        assert_eq!(
            on_cpu(&[card(), Copy(DST + 8, 0x7000, 16)], &WRITTEN),
            [true, true]
        );
    }

    #[test]
    fn reads_beside_the_destination_leave_the_copy_to_the_card() {
        let beside = [
            card(),
            Fill(DST + 0x800, 16),
            Fill(DST - 16, 16),
            Copy(0x5000, 0x6000, 16),
        ];
        assert_eq!(on_cpu(&beside, &WRITTEN), [false, true, true, true]);
    }

    #[test]
    fn work_that_goes_anywhere_keeps_the_copy_on_the_cpu_before_or_after() {
        assert_eq!(on_cpu(&[card(), Anywhere], &WRITTEN), [true, true]);
        assert_eq!(on_cpu(&[Anywhere, card()], &WRITTEN), [true, true]);
    }

    #[test]
    fn an_earlier_cpu_write_into_the_source_keeps_the_copy_on_the_cpu() {
        assert_eq!(
            on_cpu(&[Copy(SRC + 4, 0x7000, 8), card()], &WRITTEN),
            [true, true]
        );
        assert_eq!(
            on_cpu(&[Copy(0x7000, SRC, 8), card()], &WRITTEN),
            [true, false]
        );
    }

    #[test]
    fn steps_that_touch_no_title_memory_change_nothing() {
        assert_eq!(
            on_cpu(&[Other, card(), Other], &WRITTEN),
            [true, false, true]
        );
    }

    #[test]
    fn a_second_card_copy_left_to_the_card_does_not_pin_the_first() {
        let second = Card(0xa000, DST + 8, 0x40);
        let written = [(0x1000, 0x2000), (DST, DST + 0x800)];
        assert_eq!(on_cpu(&[card(), second], &written), [false, false]);
    }

    #[test]
    fn a_second_card_copy_kept_on_the_cpu_pins_the_first() {
        let second = Card(0xa000, DST + 8, 0x40);
        assert_eq!(on_cpu(&[card(), second], &WRITTEN), [true, true]);
    }

    #[test]
    fn each_copy_is_chosen_for_itself() {
        let other = Card(0xb000, 0x4000, 0x800);
        assert_eq!(
            on_cpu(&[card(), other, card()], &WRITTEN),
            [false, true, false]
        );
    }

    #[test]
    fn a_copy_nothing_pins_is_left_to_the_card_and_says_so() {
        assert_eq!(
            places(&[Other, card()], &WRITTEN, false),
            [Place::Cpu, Place::Card]
        );
    }

    #[test]
    fn a_later_fill_of_the_destination_is_named_as_the_pin() {
        assert_eq!(
            pin_of(&[card(), Other, Fill(DST + 0x10, 16)], &WRITTEN),
            Place::Kept(Pin::LaterFill, 2)
        );
    }

    #[test]
    fn a_later_texture_of_the_destination_is_named_as_the_pin() {
        assert_eq!(
            pin_of(&[card(), Texture(DST, 0x100)], &WRITTEN),
            Place::Kept(Pin::LaterTexture, 1)
        );
    }

    #[test]
    fn a_later_copy_of_the_destination_is_named_as_the_pin() {
        assert_eq!(
            pin_of(&[card(), Copy(0x5000, DST, 16)], &WRITTEN),
            Place::Kept(Pin::LaterCopy, 1)
        );
    }

    #[test]
    fn a_later_card_copy_kept_on_the_cpu_is_named_as_the_pin() {
        let second = Card(0xa000, DST + 8, 0x40);
        assert_eq!(
            pin_of(&[card(), second], &WRITTEN),
            Place::Kept(Pin::LaterCard, 1)
        );
    }

    #[test]
    fn the_pin_is_the_first_step_after_the_copy_that_meets_the_destination() {
        let steps = [card(), Fill(0x7000, 16), Fill(DST, 16), Texture(DST, 16)];
        assert_eq!(pin_of(&steps, &WRITTEN), Place::Kept(Pin::LaterFill, 2));
    }

    #[test]
    fn work_that_goes_anywhere_is_named_as_the_pin_before_or_after() {
        assert_eq!(
            pin_of(&[card(), Anywhere], &WRITTEN),
            Place::Kept(Pin::Anywhere, 1)
        );
        assert_eq!(
            places(&[Anywhere, card()], &WRITTEN, false)[1],
            Place::Kept(Pin::Anywhere, 0)
        );
    }

    #[test]
    fn an_earlier_write_of_the_source_is_named_as_the_pin() {
        assert_eq!(
            places(&[Other, Copy(SRC, 0x7000, 8), card()], &WRITTEN, false)[2],
            Place::Kept(Pin::SourceWritten, 1)
        );
    }

    #[test]
    fn a_source_no_flight_writes_is_named_as_settled() {
        assert_eq!(
            pin_of(&[card(), Fill(DST, 16)], &[]),
            Place::Kept(Pin::Settled, 0)
        );
    }

    #[test]
    fn with_all_a_copy_no_flight_writes_is_left_to_the_card_too() {
        let steps = [card(), Fill(0x7000, 16)];
        assert_eq!(places(&steps, &[], true), [Place::Card, Place::Cpu]);
        assert_eq!(places(&steps, &[], false)[0], Place::Kept(Pin::Settled, 0));
        assert_eq!(
            places(&steps, &WRITTEN, true),
            places(&steps, &WRITTEN, false)
        );
    }

    #[test]
    fn with_all_a_pinned_copy_stays_on_the_cpu() {
        let steps = [card(), Fill(DST, 16)];
        assert_eq!(places(&steps, &[], true)[0], Place::Kept(Pin::LaterFill, 1));
        assert_eq!(
            places(&[Anywhere, card()], &[], true)[1],
            Place::Kept(Pin::Anywhere, 0)
        );
        assert_eq!(
            places(&[Copy(SRC, 0x7000, 8), card()], &[], true)[1],
            Place::Kept(Pin::SourceWritten, 0)
        );
    }

    #[test]
    fn a_destination_is_recent_in_its_frame_and_the_next_only() {
        let mut recent = Vec::new();
        note_recent(&mut recent, DST, DST + 0x800, 10);
        assert!(recent_meets(&recent, DST + 0x10, DST + 0x20, 10));
        assert!(recent_meets(&recent, DST + 0x7ff, DST + 0x900, 11));
        assert!(!recent_meets(&recent, DST + 0x10, DST + 0x20, 12));
    }

    #[test]
    fn a_range_beside_the_recent_destinations_is_not_recent() {
        let mut recent = Vec::new();
        note_recent(&mut recent, DST, DST + 0x800, 10);
        assert!(!recent_meets(&recent, DST + 0x800, DST + 0x900, 10));
        assert!(!recent_meets(&recent, DST - 0x100, DST, 10));
        assert!(!recent_meets(&[], DST, DST + 1, 10));
    }

    #[test]
    fn noting_a_destination_again_renews_it_and_the_oldest_makes_room() {
        let mut recent = Vec::new();
        note_recent(&mut recent, DST, DST + 0x800, 10);
        note_recent(&mut recent, DST, DST + 0x800, 20);
        assert_eq!(recent, [(DST, DST + 0x800, 20)]);
        for k in 0..RECENT_MOST {
            note_recent(
                &mut recent,
                0x100000 + k * 0x1000,
                0x100800 + k * 0x1000,
                20,
            );
        }
        assert_eq!(recent.len(), RECENT_MOST);
        assert!(!recent_meets(&recent, DST, DST + 1, 20));
    }

    #[test]
    fn a_range_declared_lately_is_seen_for_a_few_frames() {
        let mut seen = Vec::new();
        note_seen(&mut seen, &[(0x1000, 0x9000)], 10);
        assert!(seen_meets(&seen, 0x8000, 0x8100, 10, SEEN_FRAMES));
        assert!(seen_meets(
            &seen,
            0x8000,
            0x8100,
            10 + SEEN_FRAMES,
            SEEN_FRAMES
        ));
        assert!(!seen_meets(
            &seen,
            0x8000,
            0x8100,
            11 + SEEN_FRAMES,
            SEEN_FRAMES
        ));
    }

    #[test]
    fn a_range_seen_beside_the_copy_does_not_count() {
        let mut seen = Vec::new();
        note_seen(&mut seen, &[(0x1000, 0x2000)], 10);
        assert!(!seen_meets(&seen, 0x2000, 0x2100, 10, SEEN_FRAMES));
        assert!(!seen_meets(&seen, 0x0f00, 0x1000, 10, SEEN_FRAMES));
        assert!(seen_meets(&seen, 0x0f00, 0x1001, 10, SEEN_FRAMES));
    }

    #[test]
    fn a_range_declared_again_renews_its_frame_and_is_kept_once() {
        let mut seen = Vec::new();
        note_seen(&mut seen, &[(0x1000, 0x2000)], 10);
        note_seen(&mut seen, &[(0x1000, 0x2000)], 20);
        assert_eq!(seen, [(0x1000, 0x2000, 20)]);
        assert!(!seen_meets(&seen, 0x1000, 0x1100, 30, SEEN_FRAMES));
    }

    #[test]
    fn an_empty_range_is_not_noted() {
        let mut seen = Vec::new();
        note_seen(&mut seen, &[(0x1000, 0x1000), (0x3000, 0x2000)], 10);
        assert!(seen.is_empty());
    }

    #[test]
    fn a_full_table_lets_the_range_declared_longest_ago_go() {
        let mut seen = Vec::new();
        for i in 0..SEEN_MOST {
            note_seen(&mut seen, &[(i * 0x100, i * 0x100 + 0x10)], 10 + i as u64);
        }
        note_seen(&mut seen, &[(0x10_0000, 0x10_0010)], 1000);
        assert_eq!(seen.len(), SEEN_MOST);
        assert!(!seen.iter().any(|s| s.0 == 0));
        assert!(seen.iter().any(|s| s.0 == 0x10_0000));
        assert!(seen.iter().any(|s| s.0 == 0x100));
    }
}
