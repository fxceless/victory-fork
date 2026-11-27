//! Texture views made once and copied into the tables made afresh (#1379).
//!
//! Every table that holds a constant buffer view is made afresh in dynamic
//! slots for each draw list that binds it ([`crate::draw::State::tables`]),
//! and each of its texture views was made again with it: the placed
//! texture and its host copy looked up, the description checked, the view
//! created, 0.3 us a view on an RTX 4060 Laptop, most of a table's making.
//! The world's tables show the same textures list after list.
//!
//! Here a texture view is made once into a heap the shaders do not see,
//! keyed by its record (`(generation, index)`: the same record is the same
//! view), and copied from there into a table's slot (`CopyDescriptorsSimple`
//! of one slot, about 5 ns). What the view made from a record shows can
//! change while the record stays: the host copy of the texture (let go when
//! idle and made again, or another texture placed at the same memory) and
//! the first level shown of a texture whose levels arrive on demand
//! (`crate::tiles`). An entry is copied only while both are the ones it was
//! made from; otherwise it is made again. The host copy is told by the
//! address of its shared allocation, which the entry keeps from reuse with
//! a weak reference (not the texture: a copy the title no longer uses can
//! still be let go). Anything else (buffer views, unordered access, views
//! the host cannot make) is made as before.
//!
//! `VICTORY_TABLE_VIEW_CACHE=1` turns it on (off by default).

use std::collections::HashMap;
use std::sync::{Mutex, Weak};

use windows::Win32::Graphics::Direct3D12::*;
use windows::core::Interface;

use super::{View, make, placed_texture};
use crate::host::host;
use crate::resource::HostTexture;

/// Locks the entries are spread over, by record.
const SHARDS: usize = 64;
/// Slots of the heap each shard owns: 65,536 in all.
const PER_SHARD: u32 = 1024;

/// Whether tables copy their texture views from the cache.
pub(crate) fn on() -> bool {
    use std::sync::atomic::Ordering::Relaxed;
    match ON.load(Relaxed) {
        0 => {
            let on = std::env::var_os("VICTORY_TABLE_VIEW_CACHE").is_some_and(|v| v == "1");
            ON.store(if on { 2 } else { 1 }, Relaxed);
            on
        }
        state => state == 2,
    }
}

/// 0: not read yet; 1 off; 2 on.
static ON: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// Tests: turns the cache on or off for the tables made from now on.
#[cfg(test)]
pub(crate) fn set_on(on: bool) {
    ON.store(if on { 2 } else { 1 }, std::sync::atomic::Ordering::Relaxed);
}

/// What a slot of the cache was made from.
struct Entry {
    slot: u32,
    /// The title's texture object the view shows.
    title: usize,
    /// Its host copy, and the first level shown.
    copy: Weak<HostTexture>,
    first: Option<u32>,
}

#[derive(Default)]
struct Shard {
    entries: HashMap<u64, Entry>,
    /// The record each of the shard's slots holds (0: none), and the slot
    /// to take next when every slot holds one.
    owners: Vec<u64>,
    next: u32,
}

pub(crate) struct ViewCache {
    /// Held for its slots (`cpu`).
    _heap: ID3D12DescriptorHeap,
    cpu: usize,
    step: u32,
    shards: Vec<Mutex<Shard>>,
}

// SAFETY: the heap is only written through the device (free-threaded) and
// the slots of a shard only under its lock.
unsafe impl Send for ViewCache {}
// SAFETY: as above.
unsafe impl Sync for ViewCache {}

/// The cache, made on first use; none without a host device.
pub(crate) fn cache() -> Option<&'static ViewCache> {
    static CACHE: std::sync::OnceLock<Option<ViewCache>> = std::sync::OnceLock::new();
    CACHE
        .get_or_init(|| {
            let host = host()?;
            // SAFETY: plain heap creation and queries.
            unsafe {
                let heap: ID3D12DescriptorHeap = host
                    .device
                    .CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                        Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                        NumDescriptors: SHARDS as u32 * PER_SHARD,
                        Flags: D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
                        NodeMask: 0,
                    })
                    .ok()?;
                let cpu = heap.GetCPUDescriptorHandleForHeapStart().ptr;
                let step = host.device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
                Some(ViewCache {
                    _heap: heap,
                    cpu,
                    step,
                    shards: (0..SHARDS).map(|_| Mutex::new(Shard::default())).collect(),
                })
            }
        })
        .as_ref()
}

/// Of [`ViewCache::copy`]: what the slot was given.
pub(crate) struct Copied {
    /// The host texture the view shows, for the slot to hold.
    pub(crate) texture: ID3D12Resource,
    /// Whether the view was made (not found).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) made: bool,
}

impl ViewCache {
    fn cpu(&self, slot: u32) -> D3D12_CPU_DESCRIPTOR_HANDLE {
        D3D12_CPU_DESCRIPTOR_HANDLE { ptr: self.cpu + slot as usize * self.step as usize }
    }

    /// Writes the view of record `key` (`view`) to `to`, from the cache:
    /// made there first if it is not, or no longer shows what the record
    /// shows now. `None` for a view the cache does not keep (the caller
    /// makes it as before), or one the host cannot make.
    pub(crate) fn copy(&self, view: &View, key: u64, to: D3D12_CPU_DESCRIPTOR_HANDLE) -> Option<Copied> {
        let placed;
        let shown = match view {
            View::Shader(..) => view,
            View::Placed(false, ..) => {
                placed = placed_texture(view)?;
                &placed
            }
            _ => return None,
        };
        let &View::Shader(title, _) = shown else {
            return None;
        };
        let resource = title as crate::object::This;
        // A buffer view has no host copy: as before.
        let copy = crate::resource::host_copy(resource)?;
        let first = crate::resource::first_shown(resource);
        let device = &host()?.device;
        let mix = key.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let shard = &self.shards[(mix >> 58) as usize % SHARDS];
        let mut shard = shard.lock().unwrap_or_else(|e| e.into_inner());
        let base = ((mix >> 58) as usize % SHARDS) as u32 * PER_SHARD;
        let found = shard.entries.get(&key).filter(|e| {
            e.title == title && std::ptr::eq(e.copy.as_ptr(), std::sync::Arc::as_ptr(&copy)) && e.first == first
        });
        if let Some(entry) = found {
            // SAFETY: a slot of the cache, made from this record and these.
            unsafe { device.CopyDescriptorsSimple(1, to, self.cpu(entry.slot), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
            return Some(Copied { texture: copy.texture.clone(), made: false });
        }
        // Made (again): in the slot the record had, else a free one, else
        // the next in turn (its record made again when next asked for).
        let slot = match shard.entries.remove(&key) {
            Some(old) => old.slot,
            None => {
                if shard.owners.is_empty() {
                    shard.owners = vec![0; PER_SHARD as usize];
                }
                let at = shard.next;
                shard.next = (at + 1) % PER_SHARD;
                let owner = std::mem::replace(&mut shard.owners[at as usize], key);
                if owner != 0 {
                    shard.entries.remove(&owner);
                }
                base + at
            }
        };
        shard.owners[(slot - base) as usize] = key;
        // As `HostHeap::make_at` makes it; anything else than this host
        // copy's view (not made, or another resource) is not kept.
        let made = make(shown, self.cpu(slot)).flatten();
        let texture = made.filter(|t| t.as_raw() == copy.texture.as_raw())?;
        shard.entries.insert(
            key,
            Entry { slot, title, copy: std::sync::Arc::downgrade(&copy), first },
        );
        // SAFETY: as above.
        unsafe { device.CopyDescriptorsSimple(1, to, self.cpu(slot), D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV) };
        Some(Copied { texture, made: true })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A slot copied from the cache shows the host copy the texture has
    /// now: one let go and made again since the view was cached is shown,
    /// not the old (whose memory is gone). Fails if the cache serves an
    /// entry without comparing the host copy it was made from.
    #[test]
    fn a_view_is_made_again_for_a_new_host_copy() {
        let (Some(cache), Some(host)) = (cache(), host()) else {
            return; // no host GPU
        };
        // A 64x64 RGBA8 texture the title placed in memory of its own.
        let layout = std::alloc::Layout::from_size_align(1 << 16, 1 << 16).unwrap();
        // SAFETY: a non-zero size; leaked, as title memory outlives a test.
        let at = unsafe { std::alloc::alloc_zeroed(layout) } as u64;
        let mut desc = [28, 4, 0x1688, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        (desc[10], desc[11]) = (at as u32, (at >> 32) as u32);
        let view = View::Placed(false, [3, 0, 0, 0, 64, 0, 64, 1 | 1 << 16, 28, 1, 0, 0, 0, 0], desc);
        let title = [0u32; 8];
        let handle = title.as_ptr() as u64;
        crate::descriptor::record(handle, view);
        let (view, key) = crate::descriptor::view_and_key(handle).unwrap();
        // Where tables would get it: a heap of our own.
        // SAFETY: plain heap creation.
        let heap: ID3D12DescriptorHeap = unsafe {
            host.device.CreateDescriptorHeap(&D3D12_DESCRIPTOR_HEAP_DESC {
                Type: D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
                NumDescriptors: 1,
                ..Default::default()
            })
        }
        .unwrap();
        // SAFETY: a live heap.
        let to = unsafe { heap.GetCPUDescriptorHandleForHeapStart() };
        let texture = || {
            let Some(View::Shader(r, _)) = placed_texture(&view) else { unreachable!() };
            crate::resource::host_resource(r as crate::object::This).unwrap().clone()
        };
        let first = cache.copy(&view, key, to).expect("a texture view is cached");
        assert!(first.made);
        assert_eq!(first.texture, texture());
        let again = cache.copy(&view, key, to).unwrap();
        assert!(!again.made, "the same view is copied, not made again");
        assert_eq!(again.texture, first.texture);
        // The host copy goes (idle) and the next use makes another.
        let Some(View::Shader(r, _)) = placed_texture(&view) else { unreachable!() };
        crate::resource::let_go_host_copy(r as crate::object::This);
        let now = texture();
        assert_ne!(now, first.texture, "another host copy");
        let after = cache.copy(&view, key, to).unwrap();
        assert!(after.made, "a view of a host copy let go is made again");
        assert_eq!(after.texture, now);
        assert_eq!(super::super::made_bytes(&view), Some(slot_bytes(to)), "the slot shows the new host copy");
        // The cache does not hold the host copy it was made from: the
        // texture can still be let go when idle.
        drop((first, again, after, now));
        let copy = crate::resource::host_copy(r as crate::object::This).unwrap();
        assert_eq!(std::sync::Arc::strong_count(&copy), 2, "held by the texture and here only");
    }

    fn slot_bytes(at: D3D12_CPU_DESCRIPTOR_HANDLE) -> Vec<u8> {
        // SAFETY: a plain query, then a slot of a heap the shaders do not see.
        unsafe {
            let step = host().unwrap().device.GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
            std::slice::from_raw_parts(at.ptr as *const u8, step as usize).to_vec()
        }
    }
}
