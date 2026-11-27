//! The blend pass on a host device, with no title: a back buffer that
//! holds a first plane, an interface plane with clear, half-covered and
//! opaque texels over it, and what comes out read back. After Filip722's
//! fixture for the pass's first form (pull request #1081, 8f14ae9).

use super::*;

/// Fifteen texels: where the interface plane is clear the back buffer
/// keeps the first plane's bits; where it is half covered (premultiplied)
/// the two are added as the display adds its planes; where it is opaque
/// it shows alone.
#[test]
fn the_interface_plane_is_blended_over_what_the_back_buffer_holds() {
    use crate::host::host;
    use windows::Win32::Foundation::RECT;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT_R8G8B8A8_UNORM as RGBA8;

    let Some(host) = host() else {
        return; // no host device
    };
    if straight() {
        return; // the texels below are for premultiplied alpha
    }
    let texture = |state| {
        let mut resource: Option<ID3D12Resource> = None;
        // SAFETY: plain creation of a texture with one RGBA8 subresource.
        unsafe {
            host.device
                .CreateCommittedResource(
                    &D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    D3D12_HEAP_FLAG_NONE,
                    &D3D12_RESOURCE_DESC {
                        Dimension: D3D12_RESOURCE_DIMENSION_TEXTURE2D,
                        Width: 5,
                        Height: 3,
                        DepthOrArraySize: 1,
                        MipLevels: 1,
                        Format: RGBA8,
                        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                        Flags: D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
                        ..Default::default()
                    },
                    state,
                    None,
                    &mut resource,
                )
                .expect("a small texture");
        }
        resource.expect("a small texture")
    };
    let sub = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: the copy location borrows this resource.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
    };
    let rows = |r: &ID3D12Resource| D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: the copy location borrows this resource.
        pResource: unsafe { std::mem::transmute_copy(r) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                Offset: 0,
                Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                    Format: RGBA8,
                    Width: 5,
                    Height: 3,
                    Depth: 1,
                    RowPitch: 256,
                },
            },
        },
    };
    let mut pass = Compose::new(&host.device, RGBA8).expect("the pass's shaders, root signature and pipeline");
    let target_state = D3D12_RESOURCE_STATE_RENDER_TARGET;
    let present = D3D12_RESOURCE_STATE_PRESENT;
    let interface = texture(target_state);
    let back = texture(target_state);
    let read = host.read_back_buffer(768).expect("a read-back buffer");
    let commands = host.new_list().expect("a host list");
    let list = &commands.list;
    // SAFETY: the views and resources live until the submission has run.
    unsafe {
        // The first plane, as the caller's copy leaves it in the back buffer.
        let back_view = host.rtv(Some(&back), None);
        list.ClearRenderTargetView(back_view, &[0.0, 0.0, 1.0, 1.0], None);
        list.ResourceBarrier(&[transition(&back, target_state, present)]);
        // The interface plane: two columns clear, one half covered
        // (premultiplied), two opaque.
        let interface_view = host.rtv(Some(&interface), None);
        let strip = |left, right| RECT { left, right, top: 0, bottom: 3 };
        list.ClearRenderTargetView(interface_view, &[0.0; 4], None);
        list.ClearRenderTargetView(interface_view, &[0.0, 128.0 / 255.0, 0.0, 128.0 / 255.0], Some(&[strip(2, 3)]));
        list.ClearRenderTargetView(interface_view, &[1.0, 0.0, 0.0, 1.0], Some(&[strip(3, 5)]));
        pass.record(&host.device, list, &back, (5, 3), &interface, target_state.0 as u32);
        list.ResourceBarrier(&[transition(&back, present, D3D12_RESOURCE_STATE_COPY_SOURCE)]);
        list.CopyTextureRegion(&rows(&read), 0, 0, 0, &sub(&back), None);
    }
    commands.close();
    host.execute(&[&commands]);
    // SAFETY: a plain query after the submission was waited for.
    unsafe { host.device.GetDeviceRemovedReason() }.expect("the device is still there");
    let mut texels = [[0u8; 20]; 3];
    // SAFETY: `execute` waited for the copy; each row holds five texels.
    unsafe {
        let mut at = std::ptr::null_mut();
        read.Map(0, None, Some(&mut at)).expect("the read-back buffer mapped");
        for (y, row) in texels.iter_mut().enumerate() {
            row.copy_from_slice(std::slice::from_raw_parts(at.cast::<u8>().add(y * 256), 20));
        }
        read.Unmap(0, None);
    }
    let expected = [
        [0, 0, 255, 255],
        [0, 0, 255, 255],
        [0, 128, 127, 255],
        [255, 0, 0, 255],
        [255, 0, 0, 255],
    ];
    for (y, row) in texels.iter().enumerate() {
        assert_eq!(&row[..], expected.as_flattened(), "row {y}");
    }
}
