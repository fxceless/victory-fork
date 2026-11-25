//! Experimental private backing for sampled-only, single-slice BC1 textures.
//! Existing views keep the original resource and subresource numbering.
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};
use windows::Win32::Graphics::Direct3D12::*;

/// Opt in to reserved backing for hidden, wholly uncommitted detailed mips.
pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SPARSE_BC1").is_some_and(|v| v == "1"))
}

pub(crate) struct Sparse {
    // Overall starting tile for each standard subresource, then the packed tail and/or end.
    starts: Vec<u32>,
    first: AtomicU32,
    refused: AtomicBool,
    change: Mutex<()>,
}
impl Sparse {
    /// On an unpublished resource or the host's direct queue, add mappings.
    /// Newly added tiles are never reused or removed while the resource lives.
    fn map(texture: &ID3D12Resource, first: u32, count: u32) -> windows::core::Result<()> {
        let host = crate::host::host().ok_or(windows::core::Error::empty())?;
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: u64::from(count) * 65536,
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_DEFAULT,
                CreationNodeMask: 1,
                VisibleNodeMask: 1,
                ..Default::default()
            },
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
            ..Default::default()
        };
        let mut heap: Option<ID3D12Heap> = None;
        unsafe {
            host.device.CreateHeap(&desc, &mut heap)?;
            let heap = heap.ok_or(windows::core::Error::empty())?;
            // The resource's COM reference count (including descriptor heaps)
            // governs these backing heaps' lifetime, even beyond the title handle.
            let key = windows::core::GUID::from_u128(
                0x0128e665_6b89_412d_a55e_12d744300000 + u128::from(first),
            );
            texture.SetPrivateDataInterface(&key, &heap)?;
            host.map_sparse(texture, first, &heap, count);
        }
        Ok(())
    }

    pub(crate) fn try_new(
        mut desc: D3D12_RESOURCE_DESC,
        state: D3D12_RESOURCE_STATES,
        limit: u32,
    ) -> Option<(ID3D12Resource, Sparse)> {
        if limit == 0 {
            return None;
        }
        let host = crate::host::host()?;
        static SUPPORTED: OnceLock<bool> = OnceLock::new();
        if !*SUPPORTED.get_or_init(|| {
            let mut opt = D3D12_FEATURE_DATA_D3D12_OPTIONS::default();
            unsafe {
                host.device
                    .CheckFeatureSupport(
                        D3D12_FEATURE_D3D12_OPTIONS,
                        (&raw mut opt).cast(),
                        size_of_val(&opt) as u32,
                    )
                    .is_ok()
                    && opt.TiledResourcesTier.0 >= 2
            }
        }) {
            return None;
        }
        let full_committed_bytes = unsafe {
            host.device
                .GetResourceAllocationInfo(0, &[desc])
                .SizeInBytes
        };
        if full_committed_bytes == 0 || full_committed_bytes == u64::MAX {
            return None;
        }
        desc.Layout = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE;
        let mut texture = None;
        unsafe {
            host.device
                .CreateReservedResource(&desc, state, None, &mut texture)
                .ok()?;
        }
        let texture: ID3D12Resource = texture?;
        let mut total = 0;
        let mut packed = D3D12_PACKED_MIP_INFO::default();
        let mut num = u32::from(desc.MipLevels);
        let mut locations = vec![D3D12_SUBRESOURCE_TILING::default(); num as usize];
        unsafe {
            host.device.GetResourceTiling(
                &texture,
                Some(&mut total),
                Some(&mut packed),
                None,
                Some(&mut num),
                0,
                locations.as_mut_ptr(),
            );
        }
        let standard = usize::from(packed.NumStandardMips);
        if standard > locations.len()
            || standard > num as usize
            || total == 0
            || standard + usize::from(packed.NumPackedMips) != usize::from(desc.MipLevels)
        {
            return None;
        }
        let mut starts: Vec<_> = locations[..standard]
            .iter()
            .map(|t| t.StartTileIndexInOverallResource)
            .collect();
        if packed.NumPackedMips > 0 {
            if packed
                .StartTileIndexInOverallResource
                .checked_add(packed.NumTilesForPackedMips)
                != Some(total)
            {
                return None;
            }
            starts.push(packed.StartTileIndexInOverallResource);
        }
        starts.push(total);
        if starts.first() != Some(&0) || starts.windows(2).any(|s| s[0] >= s[1]) {
            return None;
        }
        let first = limit
            .min(u32::from(packed.NumStandardMips))
            .min(u32::from(desc.MipLevels) - 1);
        if first == 0 {
            return None;
        }
        let skipped = starts[first as usize];
        // Never select a device/shape with larger capacity after full promotion,
        // even when its initial tail would be cheap.
        if u64::from(total) * 65536 > full_committed_bytes {
            return None;
        }
        Self::map(&texture, first, total - skipped).ok()?;
        Some((
            texture,
            Self {
                starts,
                first: AtomicU32::new(first),
                refused: AtomicBool::new(false),
                change: Mutex::new(()),
            },
        ))
    }

    pub(crate) fn first(&self) -> u32 {
        self.first.load(Relaxed)
    }

    pub(crate) fn promote(&self, texture: &ID3D12Resource, first: u32) -> bool {
        if first >= self.first() {
            return true;
        }
        let _guard = self.change.lock().unwrap_or_else(|e| e.into_inner());
        let previous = self.first();
        if first >= previous {
            return true;
        }
        if self.refused.load(Relaxed) {
            return false;
        }
        let count = self.starts[previous as usize] - self.starts[first as usize];
        #[cfg(test)]
        let result = if FAIL_PROMOTION.with(|fail| fail.replace(false)) {
            Err(windows::core::Error::from_hresult(
                windows::Win32::Foundation::E_OUTOFMEMORY,
            ))
        } else {
            Self::map(texture, first, count)
        };
        #[cfg(not(test))]
        let result = Self::map(texture, first, count);
        if let Err(e) = result {
            if let Some(host) = crate::host::host()
                && unsafe { host.device.GetDeviceRemovedReason() }.is_err()
            {
                crate::host::fail(format_args!(
                    "device lost during sparse texture promotion: {e}"
                ));
            }
            // Like a refused committed texture, do not retry an allocation
            // every draw. This generation can still serve all resident mips.
            // A replacement after normal retirement uses its widest known view.
            self.refused.store(true, Relaxed);
            tracing::warn!(error=%e, "cannot grow sparse texture; keeping its resident mip range");
            return false;
        }
        self.first.store(first, Relaxed);
        true
    }
}
#[cfg(test)]
thread_local! { pub(crate) static FAIL_PROMOTION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
