//! GPU captures with PIX on Windows, taken from inside the run: with
//! `VICTORY_PIX=<folder>` (diagnostics builds) the newest installed PIX's
//! `WinPixGpuCapturer.dll` is loaded before the host device is made, and a
//! file named `capture` put in the folder (holding a frame count, 1 if
//! empty) saves the next frames as `<folder>\frame-<n>.wpix`. Open one in
//! PIX: every draw's shaders as DXIL, the views and samplers it read, the
//! render targets before and after, and a pixel's history (which draw
//! wrote it).
//!
//! The capturer goes through the device like a debug layer and slows every
//! frame; it is only loaded when asked for.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use windows::Win32::System::LibraryLoader::{
    GetProcAddress, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR, LoadLibraryExW,
};
use windows::core::{HSTRING, PCSTR, PCWSTR};

/// `CaptureNextFrame(file, frames)`, exported by the capturer.
type CaptureNextFrame = unsafe extern "system" fn(PCWSTR, u32) -> windows::core::HRESULT;
/// `SetHUDOptions(options)`.
type SetHudOptions = unsafe extern "system" fn(u32) -> windows::core::HRESULT;
/// `PIX_HUD_SHOW_ON_NO_WINDOWS`: PIX draws nothing over the title's
/// frames (they are captured as PNGs too).
const HUD_NONE: u32 = 4;

struct Pix {
    folder: PathBuf,
    capture: CaptureNextFrame,
}

static PIX: OnceLock<Option<Pix>> = OnceLock::new();

/// The capturer, if `VICTORY_PIX` asks for it. Call before the host device
/// is made: PIX only sees devices made after it is loaded.
pub(crate) fn load() {
    PIX.get_or_init(|| {
        let folder = PathBuf::from(std::env::var_os("VICTORY_PIX")?);
        let Some(dll) = newest_capturer() else {
            tracing::warn!("VICTORY_PIX: no WinPixGpuCapturer.dll under %ProgramFiles%\\Microsoft PIX (install PIX)");
            return None;
        };
        // SAFETY: loading a library by full path; its exports are looked
        // up by name and called with the documented signatures.
        unsafe {
            let flags = LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR;
            let module = match LoadLibraryExW(&HSTRING::from(dll.as_path()), None, flags) {
                Ok(m) => m,
                Err(e) => {
                    tracing::warn!(error = %e, dll = %dll.display(), "VICTORY_PIX: the capturer did not load");
                    return None;
                }
            };
            let capture = GetProcAddress(module, PCSTR(c"CaptureNextFrame".as_ptr().cast()))?;
            if let Some(hud) = GetProcAddress(module, PCSTR(c"SetHUDOptions".as_ptr().cast())) {
                let hud: SetHudOptions = std::mem::transmute(hud);
                let _ = hud(HUD_NONE);
            }
            let _ = std::fs::create_dir_all(&folder);
            tracing::info!(
                dll = %dll.display(),
                folder = %folder.display(),
                "PIX capturer loaded: a file named `capture` in the folder saves the next frame"
            );
            Some(Pix {
                folder,
                capture: std::mem::transmute::<unsafe extern "system" fn() -> isize, CaptureNextFrame>(
                    capture,
                ),
            })
        }
    });
}

/// Whether the capturer is loaded (the host device is PIX's then).
pub(crate) fn loaded() -> bool {
    matches!(PIX.get(), Some(Some(_)))
}

/// `%ProgramFiles%\Microsoft PIX\<newest version>\WinPixGpuCapturer.dll`
/// (versions are dated folder names: the greatest name is the newest).
fn newest_capturer() -> Option<PathBuf> {
    let root = Path::new(&std::env::var_os("ProgramFiles")?).join("Microsoft PIX");
    std::fs::read_dir(root)
        .ok()?
        .filter_map(Result::ok)
        .map(|e| e.path().join("WinPixGpuCapturer.dll"))
        .filter(|p| p.is_file())
        .max()
}

/// At the present of frame `frame`: a `capture` file in the folder asks
/// for the next frames.
pub(crate) fn at_present(frame: u64) {
    let Some(Some(pix)) = PIX.get() else { return };
    let ask = pix.folder.join("capture");
    let Ok(text) = std::fs::read_to_string(&ask) else {
        return;
    };
    if std::fs::remove_file(&ask).is_err() {
        return;
    }
    let frames = text.trim().parse::<u32>().unwrap_or(1).clamp(1, 8);
    let file = pix.folder.join(format!("frame-{}.wpix", frame + 1));
    // SAFETY: the export takes a NUL-terminated path and a frame count.
    let hr = unsafe { (pix.capture)(PCWSTR(HSTRING::from(file.as_path()).as_ptr()), frames) };
    if hr.is_ok() {
        tracing::info!(file = %file.display(), frames, "PIX capture of the next frames");
    } else {
        tracing::warn!(error = %windows::core::Error::from(hr), "PIX capture not started");
    }
}
