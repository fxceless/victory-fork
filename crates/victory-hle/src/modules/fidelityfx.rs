//! `amd_fidelityfx_loader_xbox.dll`: the AMD FidelityFX API the title's
//! Series S build calls for its upscaler (FSR), answered here instead of
//! by the title's DLL.
//!
//! Why: the upscaler's programs come in a container our translator does
//! not read (no `DXIL` part: 20 compute programs of
//! `amd_fidelityfx_upscaler_xbox.dll`, all first bound when the world
//! starts, none of them in a Series X run), so its pass drew nothing and
//! the game's plane stayed black under the interface.
//!
//! What instead: the title is told to render at the display size (an
//! upscale ratio of 1, no camera jitter) if it asks (this title does not:
//! it renders at its own 1280x720 and asks only the jitter's phase count),
//! and the upscale dispatch becomes a bilinear stretch of the rendered
//! picture over the upscaler's output, recorded in the title's own list
//! (`victory_gpu::list::stretch_texture`).
//!
//! The descriptions follow the FidelityFX API's public headers (`ffx_api.h`,
//! `ffx_upscale.h`, MIT): each starts with its type and a pointer to the
//! next one. shortcut: the field offsets are the PC headers'; every
//! description is logged the first times its type is seen, and the
//! dispatch is checked against the textures it names before anything is
//! copied, so a different layout shows in the log instead of in the card.
//!
//! `VICTORY_FFX=0`: not answered here (the title's DLL runs, as before).
//! `VICTORY_FFX_RENDER=keep`: the render size the upscaler would answer
//! (the display size over the quality mode's ratio).
//! `VICTORY_FFX_STRETCH=0`: the picture is copied to the output's top left
//! instead of stretched over it.

#![allow(non_snake_case)]

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

use victory_gpu::object::Object;

use crate::{HleModule, hle_export};

/// `ffxReturnCode_t` values.
const OK: u32 = 0;
const ERROR_PARAMETER: u32 = 6;

/// Description types (`ffx_upscale.h`).
const CREATE_UPSCALE: u64 = 0x0001_0000;
const DISPATCH_UPSCALE: u64 = 0x0001_0001;
const QUERY_UPSCALE_RATIO: u64 = 0x0001_0002;
const QUERY_RENDER_RESOLUTION: u64 = 0x0001_0003;
const QUERY_JITTER_PHASE_COUNT: u64 = 0x0001_0004;
const QUERY_JITTER_OFFSET: u64 = 0x0001_0005;

/// Offsets in `ffxDispatchDescUpscale`: the command list, then seven
/// resources of 48 bytes (a pointer, a 32-byte description, a state),
/// colour first and output last, then two float pairs and the sizes.
const AT_LIST: usize = 16;
const AT_COLOR: usize = 24;
const AT_OUTPUT: usize = 24 + 6 * 48;
const AT_RENDER_SIZE: usize = 24 + 7 * 48 + 16;
const AT_UPSCALE_SIZE: usize = AT_RENDER_SIZE + 8;

/// The start of every description.
#[repr(C)]
struct Header {
    kind: u64,
    next: *const Header,
}

/// Whether the title renders at the display size (the default).
fn native() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| !std::env::var("VICTORY_FFX_RENDER").is_ok_and(|v| v.trim() == "keep"))
}

/// Whether the picture is stretched over the output (the default).
fn stretch() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FFX_STRETCH").is_none_or(|v| v != "0"))
}

/// The upscaler's ratio for a quality mode (native AA, quality,
/// balanced, performance, ultra performance).
fn ratio(mode: u32) -> f32 {
    match mode {
        0 => 1.0,
        1 => 1.5,
        2 => 1.7,
        3 => 2.0,
        _ => 3.0,
    }
}

/// The descriptions of a chain (at most 16) and their types.
fn chain(first: *const Header) -> Vec<(*const Header, u64)> {
    let mut out = Vec::new();
    let mut at = first;
    while !at.is_null() && out.len() < 16 {
        // SAFETY: the title passes a chain of descriptions.
        let header = unsafe { &*at };
        out.push((at, header.kind));
        at = header.next;
    }
    out
}

/// A field of a description, `offset` bytes in.
fn field<T: Copy>(at: *const Header, offset: usize) -> T {
    // SAFETY: the caller names a field of the description's type.
    unsafe { at.cast::<u8>().add(offset).cast::<T>().read_unaligned() }
}

/// Writes through a pointer the title passed, if it is not null.
fn put<T>(out: *mut T, value: T) {
    if !out.is_null() {
        // SAFETY: the title passes somewhere to write its answer.
        unsafe { out.write_unaligned(value) };
    }
}

/// Logs a description the first eight times its type is seen, with its
/// first words (shortcut: a shorter description is read past, in the
/// title's memory; diagnostics only).
fn note(call: &'static str, at: *const Header, kind: u64) {
    static SEEN: Mutex<BTreeMap<u64, u32>> = Mutex::new(BTreeMap::new());
    let mut seen = SEEN.lock().unwrap_or_else(|e| e.into_inner());
    let n = seen.entry(kind).or_insert(0);
    *n += 1;
    if *n > 8 {
        return;
    }
    let base = at.cast::<u64>();
    let words: Vec<String> = (0..12)
        // SAFETY: see above.
        .map(|i| format!("{:#x}", unsafe { base.add(i).read_unaligned() }))
        .collect();
    tracing::info!(call, kind = format_args!("{kind:#x}"), ?words, "FidelityFX description (answered by us)");
}

extern "system" fn ffxCreateContext(
    context: *mut *mut c_void,
    desc: *const Header,
    _allocation: *const c_void,
) -> u32 {
    if context.is_null() {
        return ERROR_PARAMETER;
    }
    for (at, kind) in chain(desc) {
        note("create", at, kind);
        if kind == CREATE_UPSCALE {
            tracing::info!(
                flags = format_args!("{:#x}", field::<u32>(at, 16)),
                max_render = ?(field::<u32>(at, 20), field::<u32>(at, 24)),
                max_upscale = ?(field::<u32>(at, 28), field::<u32>(at, 32)),
                native = native(),
                "FidelityFX upscaler context: answered by us, no upscaling"
            );
        }
    }
    put(context, Box::into_raw(Box::new(0u64)).cast::<c_void>());
    OK
}

extern "system" fn ffxDestroyContext(context: *mut *mut c_void, _allocation: *const c_void) -> u32 {
    if context.is_null() {
        return ERROR_PARAMETER;
    }
    // SAFETY: the title passes the context `ffxCreateContext` gave it.
    let ours = unsafe { *context };
    if !ours.is_null() {
        // SAFETY: made by `ffxCreateContext` above.
        drop(unsafe { Box::from_raw(ours.cast::<u64>()) });
        put(context, std::ptr::null_mut());
    }
    OK
}

extern "system" fn ffxQuery(_context: *mut *mut c_void, desc: *mut Header) -> u32 {
    static LOGGED: AtomicU64 = AtomicU64::new(0);
    for (at, kind) in chain(desc) {
        note("query", at, kind);
        match kind {
            QUERY_RENDER_RESOLUTION => {
                let (width, height, mode) =
                    (field::<u32>(at, 16), field::<u32>(at, 20), field::<u32>(at, 24));
                let (w, h) = if native() {
                    (width, height)
                } else {
                    let r = ratio(mode);
                    ((width as f32 / r) as u32, (height as f32 / r) as u32)
                };
                put(field::<*mut u32>(at, 32), w);
                put(field::<*mut u32>(at, 40), h);
                if LOGGED.fetch_add(1, Ordering::Relaxed) < 8 {
                    tracing::info!(display = ?(width, height), mode, render = ?(w, h), "FidelityFX render size asked");
                }
            }
            QUERY_UPSCALE_RATIO => {
                let r = if native() { 1.0 } else { ratio(field::<u32>(at, 16)) };
                put(field::<*mut f32>(at, 24), r);
            }
            QUERY_JITTER_PHASE_COUNT => {
                let (render, display) = (field::<u32>(at, 16), field::<u32>(at, 20));
                let phases = if native() || render == 0 {
                    1
                } else {
                    let r = display as f32 / render as f32;
                    ((8.0 * r * r) as i32).max(1)
                };
                put(field::<*mut i32>(at, 24), phases);
            }
            QUERY_JITTER_OFFSET => {
                put(field::<*mut f32>(at, 24), 0.0);
                put(field::<*mut f32>(at, 32), 0.0);
            }
            _ => {}
        }
    }
    OK
}

extern "system" fn ffxDispatch(_context: *mut *mut c_void, desc: *const Header) -> u32 {
    static SEEN: AtomicU64 = AtomicU64::new(0);
    for (at, kind) in chain(desc) {
        note("dispatch", at, kind);
        // Others (the reactive mask's) are left undone.
        if kind != DISPATCH_UPSCALE {
            continue;
        }
        let n = SEEN.fetch_add(1, Ordering::Relaxed);
        let list = field::<*const Object>(at, AT_LIST);
        let color = field::<*const Object>(at, AT_COLOR);
        let output = field::<*const Object>(at, AT_OUTPUT);
        let render = (field::<u32>(at, AT_RENDER_SIZE), field::<u32>(at, AT_RENDER_SIZE + 4));
        let upscale = (field::<u32>(at, AT_UPSCALE_SIZE), field::<u32>(at, AT_UPSCALE_SIZE + 4));
        let from = victory_gpu::resource::texture_shape(color);
        let to = victory_gpu::resource::texture_shape(output);
        let log = n < 4 || n.is_power_of_two();
        let (Some(from), Some(to)) = (from, to) else {
            if log {
                tracing::warn!(dispatch = n, ?render, ?upscale, ?from, ?to, "FidelityFX upscale: colour or output is not a texture of ours (layout?), nothing copied");
            }
            continue;
        };
        if list.is_null() {
            if log {
                tracing::warn!(dispatch = n, "FidelityFX upscale: no command list; nothing done");
            }
            continue;
        }
        // The rendered part of the colour texture (all of it if the size
        // in the description is not plausible).
        let plausible = |v: u32| (16..=16384).contains(&v);
        let (w, h) = if plausible(render.0) && plausible(render.1) {
            (render.0.min(from.0 as u32), render.1.min(from.1))
        } else {
            (from.0 as u32, from.1)
        };
        // Stretched over the output (`VICTORY_FFX_STRETCH=0`: copied to its
        // top left instead); copied as a last resort.
        let how = if stretch() && victory_gpu::list::stretch_texture(list, output, color, w, h) {
            "stretched"
        } else if victory_gpu::list::copy_texture_top_left(list, output, color, w.min(to.0 as u32), h.min(to.1)) {
            "copied to the top left"
        } else {
            "nothing (the host copies differ in format)"
        };
        if log {
            tracing::info!(dispatch = n, ?render, ?upscale, ?from, ?to, done = how, "FidelityFX upscale replaced");
        }
    }
    OK
}

pub(crate) fn module() -> HleModule {
    let on = std::env::var_os("VICTORY_FFX").is_none_or(|v| v != "0");
    HleModule {
        name: "fidelityfx",
        dlls: &["amd_fidelityfx_loader_xbox.dll"],
        exports: if on {
            vec![
                hle_export!(ffxCreateContext),
                hle_export!(ffxDestroyContext),
                hle_export!(ffxQuery),
                hle_export!(ffxDispatch),
            ]
        } else {
            Vec::new()
        },
    }
}
