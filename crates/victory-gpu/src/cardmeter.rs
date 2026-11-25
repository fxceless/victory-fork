//! What the card's time goes to, by kind of list, in every run.
//!
//! The card's clock (`crate::host`) says how long the card had work of
//! ours, not which work. The timing run (`VICTORY_GPU_TIMING`) names every
//! title command but sends each list on its own, which changes the frame,
//! and it never timed the lists the layer adds for itself: the one that
//! brings copies in the card's memory up to date and hashes their pieces
//! again, and the one that compares pieces after a submission's lists.
//! Run 187's table left 64 ms a frame of card time unexplained beside
//! them (40 ms by the count outside timing runs); with draw windows off
//! the card is the frame's limit (run 225: 85 to 98 % busy), so that sum
//! decides what is worth changing.
//!
//! Here a list gets one timestamp where it begins and one where it ends,
//! both resolved into one read-back buffer by the list itself. The card
//! writes them when it gets there. A pair is read after its submission's
//! fence completes and it has been pending for a second and a half.
//! Its length goes to the frame log's totals by kind. So the frame
//! log's card columns are of the frames of a moment ago: right for the
//! usual frame, not for placing one long frame.
//!
//! On by default in diagnostics builds; `VICTORY_CARD_METER=0` or `1` says
//! otherwise. Two queries and a resolve a list.

use std::collections::BTreeSet;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;

/// What a timed list is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// Ours: changed bytes go up to copies in the card's memory and their
    /// pieces are hashed again (`Host::card_before`).
    Sync,
    /// Ours: the compare of the pieces a submission's shaders may have
    /// written, and the fetch of the flagged ones inside the flight.
    Check,
    /// Every other host list: the title's own, our present copies.
    Lists,
}

/// Pairs of timestamps in the ring.
const PAIRS: u32 = 8192;
/// Age a submission before reporting it, even if it has finished already.
const AFTER: Duration = Duration::from_millis(1500);

/// A timestamp pair that can still be used by a recorded command list.
pub(crate) struct Span(u32);

impl Span {
    pub(crate) fn pair(&self) -> u32 {
        self.0
    }
}

impl Drop for Span {
    fn drop(&mut self) {
        let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(Some(m)) = meter.as_mut() {
            m.slots.release(self.0);
        }
    }
}

struct Slot {
    kind: Kind,
    owned: bool,
    /// The latest submission that can write the pair and when it was sent.
    submitted: Option<(u64, Instant)>,
}

struct Slots {
    free: Vec<u32>,
    entries: Vec<Option<Slot>>,
    pending: BTreeSet<(u64, u32)>,
}

impl Slots {
    fn new(count: u32) -> Self {
        Self {
            free: (0..count).rev().collect(),
            entries: (0..count).map(|_| None).collect(),
            pending: BTreeSet::new(),
        }
    }

    fn reserve(&mut self, kind: Kind) -> Option<u32> {
        let pair = self.free.pop()?;
        self.entries[pair as usize] = Some(Slot { kind, owned: true, submitted: None });
        Some(pair)
    }

    fn release(&mut self, pair: u32) {
        let slot = self.entries[pair as usize].as_mut().expect("owned timestamp pair");
        slot.owned = false;
        if slot.submitted.is_none() {
            self.give_back(pair);
        }
    }

    fn give_back(&mut self, pair: u32) {
        self.entries[pair as usize] = None;
        self.free.push(pair);
    }

    fn submit(&mut self, pair: u32, fence: u64, at: Instant) {
        let slot = self.entries[pair as usize].as_mut().expect("submitted timestamp pair");
        if let Some((prior, _)) = slot.submitted.replace((fence, at)) {
            self.pending.remove(&(prior, pair));
        }
        self.pending.insert((fence, pair));
    }

    fn take_ready(&mut self, completed: u64, now: Instant) -> Option<(u32, Kind)> {
        let &(fence, pair) = self.pending.first()?;
        let slot = self.entries[pair as usize].as_mut().expect("pending timestamp pair");
        let (_, sent) = slot.submitted.expect("a timestamp pair's submission");
        if fence > completed || now.saturating_duration_since(sent) < AFTER {
            return None;
        }
        self.pending.pop_first();
        slot.submitted = None;
        let kind = slot.kind;
        if !slot.owned {
            self.give_back(pair);
        }
        Some((pair, kind))
    }
}

struct Meter {
    heap: ID3D12QueryHeap,
    fence: ID3D12Fence,
    back: ID3D12Resource,
    /// The read-back buffer's bytes, mapped for good.
    mapped: usize,
    /// Ticks a second of the queue's clock.
    frequency: u64,
    slots: Slots,
}

// SAFETY: host objects (free-threaded) and a mapping only read, behind a lock.
unsafe impl Send for Meter {}

static METER: Mutex<Option<Option<Meter>>> = Mutex::new(None);

/// Whether lists are timed.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var_os("VICTORY_CARD_METER") {
        Some(v) => v != "0",
        None => cfg!(feature = "diagnostics"),
    })
}

fn make() -> Option<Meter> {
    let host = crate::host::host()?;
    let device = host.device();
    let mut heap: Option<ID3D12QueryHeap> = None;
    let mut back: Option<ID3D12Resource> = None;
    let readback = D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_READBACK, ..Default::default() };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(PAIRS) * 16,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    // SAFETY: plain object creation; the buffer is mapped once and stays so.
    unsafe {
        device
            .CreateQueryHeap(
                &D3D12_QUERY_HEAP_DESC { Type: D3D12_QUERY_HEAP_TYPE_TIMESTAMP, Count: PAIRS * 2, NodeMask: 0 },
                &mut heap,
            )
            .ok()?;
        device
            .CreateCommittedResource(&readback, D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_COPY_DEST, None, &mut back)
            .ok()?;
        let (heap, back) = (heap?, back?);
        let mut mapped = std::ptr::null_mut();
        back.Map(0, None, Some(&mut mapped)).ok()?;
        let frequency = host.timestamp_frequency();
        if mapped.is_null() || frequency == 0 {
            return None;
        }
        tracing::info!(
            "the card's time is measured by kind of list, a timestamp where each list begins and ends (VICTORY_CARD_METER=0: not)"
        );
        Some(Meter {
            heap, fence: host.completion_fence().clone(), back, mapped: mapped as usize, frequency,
            slots: Slots::new(PAIRS),
        })
    }
}

/// A list of `kind` begins here: the first timestamp of a pair.
pub(crate) fn begin(list: &ID3D12GraphicsCommandList, kind: Kind) -> Option<Span> {
    if !on() {
        return None;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let m = meter.get_or_insert_with(make).as_mut()?;
    let pair = m.slots.reserve(kind)?;
    // SAFETY: an open list and our query heap.
    unsafe { list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, pair * 2) };
    Some(Span(pair))
}

/// The list ends here: the pair's second timestamp, and both go to the
/// read-back buffer when the card has run the list.
pub(crate) fn end(list: &ID3D12GraphicsCommandList, span: Option<&Span>) {
    let Some(span) = span else {
        return;
    };
    let meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_ref() else {
        return;
    };
    // SAFETY: an open list, our heap and our read-back buffer, the pair's
    // own sixteen bytes of it.
    unsafe {
        list.EndQuery(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, span.0 * 2 + 1);
        list.ResolveQueryData(&m.heap, D3D12_QUERY_TYPE_TIMESTAMP, span.0 * 2, 2, &m.back, u64::from(span.0) * 16);
    }
}

/// Registers the fence before the direct queue can write a reused pair.
/// The caller keeps every span alive until submission.
pub(crate) fn submitted(fence: u64, pairs: &[u32]) {
    if pairs.is_empty() {
        return;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(Some(m)) = meter.as_mut() {
        let now = Instant::now();
        for &pair in pairs {
            m.slots.submit(pair, fence, now);
        }
    }
}

/// Microseconds between two timestamps of a clock of `frequency` ticks a
/// second; 0 if the second is not after the first (a pair the card has not
/// written, or a list that never ran).
fn micros(begin: u64, end: u64, frequency: u64) -> u64 {
    if end <= begin || frequency == 0 {
        return 0;
    }
    ((end - begin) as u128 * 1_000_000 / frequency as u128).min(10_000_000) as u64
}

/// Adds the completed, aged pairs to the frame log. Called once a frame.
pub(crate) fn collect() {
    if !on() {
        return;
    }
    let mut meter = METER.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(m)) = meter.as_mut() else {
        return;
    };
    // SAFETY: this is the direct queue's completion fence.
    let completed = unsafe { m.fence.GetCompletedValue() };
    if completed == u64::MAX {
        return; // the device was removed
    }
    collect_ready(m, completed, Instant::now());
}

fn collect_ready(m: &mut Meter, completed: u64, now: Instant) {
    while let Some((pair, kind)) = m.slots.take_ready(completed, now) {
        // SAFETY: the fence covers both writes to the pair's readback bytes.
        // The meter lock keeps a new submission from writing them meanwhile.
        let (begin, end) = unsafe {
            let p = (m.mapped as *const u64).add(pair as usize * 2);
            (p.read_volatile(), p.add(1).read_volatile())
        };
        let us = micros(begin, end, m.frequency);
        use crate::framelog::{T, add};
        match kind {
            Kind::Sync => {
                add(T::CardSyncUs, us);
                add(T::CardSyncLists, 1);
            }
            Kind::Check => {
                add(T::CardCheckUs, us);
                add(T::CardCheckLists, 1);
            }
            Kind::Lists => {
                add(T::CardListsUs, us);
                add(T::CardLists, 1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_unsubmitted_list_reserves_its_pair() {
        let mut slots = Slots::new(2);
        let older = slots.reserve(Kind::Lists).unwrap();
        let newer = slots.reserve(Kind::Sync).unwrap();
        assert_ne!(older, newer);
        assert!(slots.reserve(Kind::Check).is_none());
        assert_eq!(slots.take_ready(10, Instant::now() + AFTER), None);
        assert!(slots.reserve(Kind::Check).is_none());
        slots.release(newer);
        assert_eq!(slots.reserve(Kind::Check), Some(newer));
    }

    #[test]
    fn dropping_a_submitted_list_keeps_its_pair_until_the_fence_and_sample() {
        let mut slots = Slots::new(1);
        let pair = slots.reserve(Kind::Check).unwrap();
        let now = Instant::now();
        slots.submit(pair, 9, now);
        slots.release(pair);

        assert_eq!(slots.take_ready(9, now), None);
        assert_eq!(slots.take_ready(8, now + AFTER), None);
        assert!(slots.reserve(Kind::Lists).is_none());
        assert_eq!(slots.take_ready(9, now + AFTER), Some((pair, Kind::Check)));
        assert_eq!(slots.reserve(Kind::Lists), Some(pair));
    }

    #[test]
    fn a_resubmitted_pair_waits_for_its_latest_write() {
        let mut slots = Slots::new(2);
        let repeated = slots.reserve(Kind::Lists).unwrap();
        let between = slots.reserve(Kind::Sync).unwrap();
        let now = Instant::now();
        slots.submit(repeated, 3, now);
        slots.submit(between, 4, now);
        slots.submit(repeated, 5, now);
        assert_eq!(slots.take_ready(3, now + AFTER), None);
        assert_eq!(slots.take_ready(4, now + AFTER), Some((between, Kind::Sync)));
        assert_eq!(slots.take_ready(4, now + AFTER), None);
        assert_eq!(slots.take_ready(5, now + AFTER), Some((repeated, Kind::Lists)));
        assert_eq!(slots.take_ready(5, now + AFTER), None);
        assert!(slots.reserve(Kind::Check).is_none());
    }

    #[test]
    fn a_host_submission_resolves_its_owned_pair() {
        if !on() {
            return;
        }
        let Some(host) = crate::host::host() else {
            assert!(std::env::var_os("VICTORY_D3D12_WARP").is_none(), "a WARP host is required");
            return;
        };
        let list = host.new_list().unwrap();
        list.close();
        list.reset();
        list.close();
        let pair = list.meter_pair().expect("the closed list owns its timestamp pair");
        host.execute(&[&list]);

        let mut meter = METER.lock().unwrap();
        let m = meter.as_mut().unwrap().as_mut().unwrap();
        let (signal, at) = m.slots.entries[pair as usize].as_ref().unwrap().submitted.unwrap();
        // SAFETY: the fence belongs to the queue that executed the list.
        let completed = unsafe { m.fence.GetCompletedValue() };
        assert!(completed != u64::MAX && completed >= signal);
        // SAFETY: the queue completed the submission that resolved the pair.
        let (begin, end) = unsafe {
            let p = (m.mapped as *const u64).add(pair as usize * 2);
            (p.read_volatile(), p.add(1).read_volatile())
        };
        assert!(end > 0 && end >= begin, "invalid timestamps {begin}, {end}");
        collect_ready(m, completed, at + AFTER);
        let slot = m.slots.entries[pair as usize].as_ref().unwrap();
        assert_eq!(slot.submitted, None);
        assert!(slot.owned);
    }

    #[cfg(feature = "diagnostics")]
    #[test]
    fn a_grouped_empty_submission_signals_the_direct_queue() {
        if std::env::var_os("VICTORY_GPU_TIMING").is_none() {
            return;
        }
        let Some(host) = crate::host::host() else {
            assert!(std::env::var_os("VICTORY_D3D12_WARP").is_none(), "a WARP host is required");
            return;
        };
        // SAFETY: the direct queue's live fence; the submission waits for
        // its own signal before returning.
        let before = unsafe { host.completion_fence().GetCompletedValue() };
        host.execute(&[]);
        let after = unsafe { host.completion_fence().GetCompletedValue() };
        assert!(
            after != u64::MAX && after > before,
            "an empty timed submission needs a new completion signal"
        );
    }

    #[test]
    fn a_pairs_length_is_in_microseconds_of_the_queues_clock() {
        assert_eq!(micros(1_000, 11_000, 10_000_000), 1_000);
        assert_eq!(micros(0, 25_000_000, 25_000_000), 1_000_000);
        // Not written, or the wrong way round: nothing.
        assert_eq!(micros(5, 5, 1_000), 0);
        assert_eq!(micros(9, 5, 1_000), 0);
        assert_eq!(micros(1, 2, 0), 0);
        // A stale second stamp from the ring's last round cannot add hours.
        assert_eq!(micros(1, u64::MAX, 1), 10_000_000);
    }
}
