//! Small textures in shared heaps: host copies of title textures that the
//! card may place at 4 KiB instead of 64 KiB (`VICTORY_SMALL_TEXTURES=1`;
//! off by default).
//!
//! A host copy is a committed texture (`resource.rs` `host_texture_as`),
//! and a committed resource takes its own heap, a multiple of 64 KiB. A
//! texture the card lays out in 4 KiB pages (one whose 4 KiB-aligned size
//! is 64 KiB or less: `GetResourceAllocationInfo` answers 4,096 as its
//! alignment) takes 64 KiB that way whatever its size: a 64x64 BC1 texture
//! with its levels is under 3 KiB. Here such a texture is a placed one in a
//! heap of ours of [`HEAP_BYTES`], at a run of 4 KiB pages of its own.
//!
//! The rules:
//! - Only textures the title fills (no target, depth or unordered access:
//!   their contents are written whole by their upload, which copies every
//!   level from the staging buffer, before anything reads them), never
//!   ones made level by level (`crate::tiles`).
//! - A run goes back to its heap only when the texture is gone: the host
//!   copy was dropped and no other reference to the texture is left
//!   (slots and lists hold the texture's COM object). Until then the run
//!   waits, holding one reference of its own ([`Placed::release`],
//!   [`collect`] at each sweep): a stale view must not read a later
//!   texture placed in the same pages, and a heap does not go while a
//!   texture in it lives.
//! - A heap with every page free goes back to the card, unless it is the
//!   pool's only one.
//!
//! What it saves is counted on the memory line, beside what the committed
//! textures that could be placed waste (counted with the switch off too).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;

/// The card's small page: 4 KiB.
pub(crate) const PAGE: u64 = 4 << 10;

/// A heap of the pool: 16 MiB, 4,096 pages.
const HEAP_BYTES: u64 = 16 << 20;
const HEAP_PAGES: u32 = (HEAP_BYTES / PAGE) as u32;

/// Now: placed textures alive, their bytes, the bytes of the pool's heaps,
/// the 64 KiB padding the placed ones do not take; committed textures that
/// could have been placed, and their padding; since the start: heaps made,
/// heaps given back, runs that waited for a texture's last reference.
pub(crate) static STATS: [AtomicU64; 9] = [const { AtomicU64::new(0) }; 9];

/// [`STATS`]' places.
pub(crate) mod stat {
    pub(crate) const PLACED: usize = 0;
    pub(crate) const PLACED_BYTES: usize = 1;
    pub(crate) const HEAPS_BYTES: usize = 2;
    pub(crate) const SAVED: usize = 3;
    pub(crate) const COULD: usize = 4;
    pub(crate) const COULD_PADDING: usize = 5;
    pub(crate) const HEAPS_MADE: usize = 6;
    pub(crate) const HEAPS_GONE: usize = 7;
    pub(crate) const WAITED: usize = 8;
}

/// Whether small textures are placed (`VICTORY_SMALL_TEXTURES=1`).
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SMALL_TEXTURES").is_some_and(|v| v == "1"))
}

/// One heap of the pool and its free runs of pages (first page, pages), in
/// order, no two touching.
struct Heap {
    heap: ID3D12Heap,
    free: Vec<(u32, u32)>,
}

// SAFETY: a heap of the device, named only in device calls, which any
// thread may make; its free runs are plain numbers behind `POOL`'s lock.
unsafe impl Send for Heap {}

/// The pool's heaps by place; a place is empty after its heap went back.
static POOL: Mutex<Vec<Option<Heap>>> = Mutex::new(Vec::new());

/// Runs whose texture still had other references when its host copy was
/// dropped: given back by [`collect`] once only ours is left.
static WAITING: Mutex<Vec<Waiting>> = Mutex::new(Vec::new());

struct Waiting {
    texture: ID3D12Resource,
    place: usize,
    first: u32,
    pages: u32,
}

// SAFETY: the texture is only asked for its reference count.
unsafe impl Send for Waiting {}

/// A run of pages of the pool's heap at `place`, taken by one texture.
/// It holds no reference to the texture (the sweep counts those): the
/// host copy hands the texture to [`Placed::release`] when it is dropped.
pub(crate) struct Placed {
    place: usize,
    first: u32,
    pages: u32,
    /// The bytes the texture would have taken committed.
    committed: u64,
    /// Given back already ([`Placed::release`]).
    released: bool,
}

impl Placed {
    /// The bytes it holds of the heap.
    pub(crate) fn bytes(&self) -> u64 {
        u64::from(self.pages) * PAGE
    }

    fn uncount(&self) {
        let bytes = self.bytes();
        STATS[stat::PLACED].fetch_sub(1, Ordering::Relaxed);
        STATS[stat::PLACED_BYTES].fetch_sub(bytes, Ordering::Relaxed);
        STATS[stat::SAVED].fetch_sub(self.committed.saturating_sub(bytes), Ordering::Relaxed);
    }

    /// The host copy of `texture`, placed in this run, is dropped: the run
    /// goes back once the reference kept here is the texture's last
    /// ([`collect`], at the next sweep), so no texture is placed in pages
    /// whose texture is alive and no heap goes while a texture in it does.
    pub(crate) fn release(mut self, texture: &ID3D12Resource) {
        self.released = true;
        self.uncount();
        // The caller's reference and ours: others are slots' and lists'.
        if crate::host::references(texture) > 2 {
            STATS[stat::WAITED].fetch_add(1, Ordering::Relaxed);
        }
        WAITING.lock().unwrap_or_else(|e| e.into_inner()).push(Waiting {
            texture: texture.clone(),
            place: self.place,
            first: self.first,
            pages: self.pages,
        });
    }
}

impl Drop for Placed {
    /// Dropped without [`Placed::release`]: no texture is left in the run.
    fn drop(&mut self) {
        if !self.released {
            self.uncount();
            give_back(self.place, self.first, self.pages);
        }
    }
}

/// Gives back the runs whose texture has no reference left but ours.
/// Called before a submission starts, with the sweeps.
pub(crate) fn collect() {
    let done: Vec<Waiting> = {
        let mut waiting = WAITING.lock().unwrap_or_else(|e| e.into_inner());
        if waiting.is_empty() {
            return;
        }
        let (done, keep) =
            std::mem::take(&mut *waiting).into_iter().partition(|w| crate::host::references(&w.texture) == 1);
        *waiting = keep;
        done
    };
    for w in done {
        let (place, first, pages) = (w.place, w.first, w.pages);
        drop(w);
        give_back(place, first, pages);
    }
}

/// The bytes a committed `desc` takes, and those it takes placed at 4 KiB
/// if the card allows that for it (`None` otherwise).
pub(crate) fn sizes(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC) -> (u64, Option<u64>) {
    let mut d = *desc;
    d.Alignment = 0;
    // SAFETY: plain queries on a live device with a whole description.
    let committed = unsafe { device.GetResourceAllocationInfo(0, &[d]) }.SizeInBytes;
    const RT_DS: i32 = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 | D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL.0;
    // A texture the card may place at 4 KiB is one whose most detailed
    // level takes 64 KiB or less: committed, a few 64 KiB pages at most.
    // Bigger ones are not asked (a debug layer reports a 4 KiB alignment
    // refused).
    if committed > 256 << 10
        || d.Flags.0 & RT_DS != 0
        || d.Dimension == D3D12_RESOURCE_DIMENSION_BUFFER
        || d.SampleDesc.Count > 1
    {
        return (committed, None);
    }
    d.Alignment = PAGE;
    // SAFETY: as above.
    let small = unsafe { device.GetResourceAllocationInfo(0, &[d]) };
    let fits = small.Alignment == PAGE && small.SizeInBytes != u64::MAX && small.SizeInBytes < committed;
    (committed, fits.then_some(small.SizeInBytes))
}

/// `pages` pages of one heap, the lowest free run that holds them: the
/// heap, its place and the first page. A new heap is made when none has
/// such a run.
fn take(device: &ID3D12Device, pages: u32) -> Option<(ID3D12Heap, usize, u32)> {
    if pages == 0 || pages > HEAP_PAGES {
        return None;
    }
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let found = pool.iter().enumerate().find_map(|(place, h)| {
        let h = h.as_ref()?;
        let run = h.free.iter().position(|run| run.1 >= pages)?;
        Some((place, run))
    });
    let (place, run) = match found {
        Some(found) => found,
        None => {
            let desc = D3D12_HEAP_DESC {
                SizeInBytes: HEAP_BYTES,
                Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() },
                Alignment: 0,
                Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            };
            let mut made: Option<ID3D12Heap> = None;
            // SAFETY: plain creation; the out pointer is ours.
            if let Err(e) = unsafe { device.CreateHeap::<ID3D12Heap>(&raw const desc, &raw mut made) } {
                tracing::warn!(error = %e, "no heap for small textures: they are made committed");
                return None;
            }
            let fresh = Heap { heap: made?, free: vec![(0, HEAP_PAGES)] };
            STATS[stat::HEAPS_BYTES].fetch_add(HEAP_BYTES, Ordering::Relaxed);
            STATS[stat::HEAPS_MADE].fetch_add(1, Ordering::Relaxed);
            let place = match pool.iter().position(Option::is_none) {
                Some(place) => {
                    pool[place] = Some(fresh);
                    place
                }
                None => {
                    pool.push(Some(fresh));
                    pool.len() - 1
                }
            };
            (place, 0)
        }
    };
    let h = pool[place].as_mut()?;
    let (first, free) = h.free[run];
    if free == pages {
        h.free.remove(run);
    } else {
        h.free[run] = (first + pages, free - pages);
    }
    Some((h.heap.clone(), place, first))
}

/// Gives the run back to the heap at `place`; runs that touch become one.
/// A heap with every page free goes back to the card unless it is the
/// pool's only one.
fn give_back(place: usize, first: u32, pages: u32) {
    let mut pool = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(Some(h)) = pool.get_mut(place) else {
        return;
    };
    let at = h.free.partition_point(|run| run.0 < first);
    h.free.insert(at, (first, pages));
    // Join with the next, then with the one before.
    if at + 1 < h.free.len() && h.free[at].0 + h.free[at].1 == h.free[at + 1].0 {
        h.free[at].1 += h.free[at + 1].1;
        h.free.remove(at + 1);
    }
    if at > 0 && h.free[at - 1].0 + h.free[at - 1].1 == h.free[at].0 {
        h.free[at - 1].1 += h.free[at].1;
        h.free.remove(at);
    }
    let all_free = h.free.len() == 1 && h.free[0] == (0, HEAP_PAGES);
    if all_free && pool.iter().flatten().count() > 1 {
        pool[place] = None;
        STATS[stat::HEAPS_BYTES].fetch_sub(HEAP_BYTES, Ordering::Relaxed);
        STATS[stat::HEAPS_GONE].fetch_add(1, Ordering::Relaxed);
    }
}

/// A texture of `desc` placed at 4 KiB in a heap of the pool, in `state`,
/// if the card allows it for `desc` and a run can be had: the texture and
/// its run. `None`: the caller makes it committed.
pub(crate) fn make(
    device: &ID3D12Device,
    desc: &D3D12_RESOURCE_DESC,
    state: D3D12_RESOURCE_STATES,
) -> Option<(ID3D12Resource, Placed)> {
    let (committed, Some(small)) = sizes(device, desc) else {
        return None;
    };
    let pages = u32::try_from(small.div_ceil(PAGE)).ok()?;
    let (heap, place, first) = take(device, pages)?;
    let mut d = *desc;
    d.Alignment = PAGE;
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: a heap of the device that allows such textures, an offset in
    // a run of ours of the size the device asked for, 4 KiB aligned; the out
    // pointer is ours. A texture the title fills has no clear value.
    let made = unsafe {
        device.CreatePlacedResource(&heap, u64::from(first) * PAGE, &raw const d, state, None, &mut texture)
    };
    let texture = match (made, texture) {
        (Ok(()), Some(texture)) => texture,
        (made, _) => {
            if let Err(e) = made {
                tracing::warn!(error = %e, ?d, "no small texture placed: made committed");
            }
            give_back(place, first, pages);
            return None;
        }
    };
    let bytes = u64::from(pages) * PAGE;
    STATS[stat::PLACED].fetch_add(1, Ordering::Relaxed);
    STATS[stat::PLACED_BYTES].fetch_add(bytes, Ordering::Relaxed);
    STATS[stat::SAVED].fetch_add(committed.saturating_sub(bytes), Ordering::Relaxed);
    let placed = Placed { place, first, pages, committed, released: false };
    Some((texture, placed))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pool's runs on the software adapter: taken lowest first, given
    /// back and joined, a heap made when none has room, one emptied heap
    /// given back while another is left; and a texture the device allows
    /// at 4 KiB is placed at a smaller size than committed.
    #[test]
    fn small_textures_take_and_give_back_runs() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::Common::*;
        let Ok(adapter) = crate::host::warp_adapter() else {
            return;
        };
        let mut device: Option<ID3D12Device> = None;
        // SAFETY: plain creation on the software adapter.
        if unsafe { D3D12CreateDevice(&adapter, D3D_FEATURE_LEVEL_11_0, &mut device) }.is_err() {
            return;
        }
        let device = device.expect("a device");
        let (_, place, a) = take(&device, 3).expect("three pages");
        let (_, _, b) = take(&device, 5).expect("five pages");
        assert_eq!((a, b), (0, 3));
        give_back(place, a, 3);
        let (_, _, c) = take(&device, 2).expect("two pages in the first run");
        assert_eq!(c, 0);
        give_back(place, c, 2);
        give_back(place, b, 5);
        {
            let pool = POOL.lock().unwrap();
            assert_eq!(pool[place].as_ref().expect("the only heap stays").free, vec![(0, HEAP_PAGES)]);
        }
        // A whole heap's run makes a second heap, which goes back.
        let (_, first_place, whole) = take(&device, HEAP_PAGES).expect("a whole heap");
        let (_, second_place, other) = take(&device, 1).expect("a page of another heap");
        assert_ne!(first_place, second_place);
        give_back(second_place, other, 1);
        assert!(POOL.lock().unwrap()[second_place].is_none(), "an empty heap goes while another is left");
        give_back(first_place, whole, HEAP_PAGES);

        let desc = D3D12_RESOURCE_DESC {
            Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
            Width: 64,
            Height: 64,
            DepthOrArraySize: 1,
            MipLevels: 7,
            Format: DXGI_FORMAT_BC1_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            ..Default::default()
        };
        let (committed, small) = sizes(&device, &desc);
        if let Some(small) = small {
            assert!(small < committed && small % PAGE == 0);
            let (texture, placed) = make(&device, &desc, D3D12_RESOURCE_STATE_COPY_DEST).expect("placed");
            assert_eq!(placed.bytes(), small.div_ceil(PAGE) * PAGE);
            placed.release(&texture);
            collect();
            assert_eq!(WAITING.lock().unwrap().len(), 1, "the run waits while the caller holds the texture");
            drop(texture);
            collect();
            assert!(WAITING.lock().unwrap().is_empty(), "the run went back with the texture's last reference");
            let pool = POOL.lock().unwrap();
            assert!(pool.iter().flatten().all(|h| h.free == vec![(0, HEAP_PAGES)]), "every page free again");
        }
    }
}
