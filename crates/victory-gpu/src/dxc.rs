//! DirectXShaderCompiler as a fallback for `D3DCompile`.
//!
//! Wine's `d3dcompiler_47` is vkd3d-shader's compiler: it refuses `Interlocked*`
//! and stores on raw or structured buffers, and `GetDimensions` on a
//! `RWTexture2D`. The same HLSL compiles with DXC as `<stage>_6_0`, which
//! vkd3d-proton accepts. Nothing here runs when `dxcompiler.dll` is absent.

#![allow(dead_code)]

// ---- DXC fallback -------------------------------------------------------
//
// `D3DCompile` under Wine is vkd3d-shader's compiler: it refuses the indirect
// clamp's shape (`Interlocked*` and stores on a raw or structured buffer) and
// `GetDimensions` on a `RWTexture2D`. The same HLSL compiles with Microsoft's
// DXC as `*_6_0`, so a shader of ours falls back to it when `D3DCompile`
// fails. `VICTORY_DXC=1` forces the fallback, for testing.

#[repr(C)]
struct DxcBuffer {
    ptr: *const core::ffi::c_void,
    size: usize,
    encoding: u32,
}

const CLSID_DXC_COMPILER: windows::core::GUID =
    windows::core::GUID::from_u128(0x73e22d93_e6ce_47f3_b5bf_f0664f39c1b0);
const IID_DXC_COMPILER3: windows::core::GUID =
    windows::core::GUID::from_u128(0x228b4687_5a6a_4730_900c_9702b2203f54);
const IID_DXC_RESULT: windows::core::GUID =
    windows::core::GUID::from_u128(0x58346cda_dde7_4497_9461_6f87af5e0659);
const IID_DXC_BLOB: windows::core::GUID =
    windows::core::GUID::from_u128(0x8ba5fb08_5195_40e2_ac58_0d989c3a0102);
const DXC_OUT_OBJECT: u32 = 1;
const DXC_OUT_ERRORS: u32 = 2;

type DxcCreate = unsafe extern "system" fn(
    *const windows::core::GUID,
    *const windows::core::GUID,
    *mut *mut core::ffi::c_void,
) -> windows::core::HRESULT;

fn dxc_create_instance() -> Option<DxcCreate> {
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    static MODULE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let module = *MODULE.get_or_init(|| unsafe {
        LoadLibraryW(windows::core::w!("dxcompiler.dll"))
            .map(|m| m.0 as usize)
            .unwrap_or(0)
    });
    if module == 0 {
        return None;
    }
    let f = unsafe { GetProcAddress(HMODULE(module as *mut core::ffi::c_void), windows::core::s!("DxcCreateInstance")) }?;
    Some(unsafe { core::mem::transmute(f) })
}

fn wide(text: &str) -> Vec<u16> {
    text.encode_utf16().chain(core::iter::once(0)).collect()
}

/// Whether the fallback is forced (`VICTORY_DXC=1`).
fn dxc_forced() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_DXC").is_ok_and(|v| v == "1"))
}

/// `compile` through `dxcompiler.dll`: the same source as `<stage>_6_0`.
pub(crate) fn compile(
    source: &str,
    entry: windows::core::PCSTR,
    target: windows::core::PCSTR,
) -> windows::core::Result<Vec<u8>> {
    use windows::core::HRESULT;
    let stage = unsafe { core::ffi::CStr::from_ptr(target.as_ptr().cast()) }
        .to_string_lossy()
        .to_string();
    let profile = format!("{}_6_0", stage.split('_').next().unwrap_or("cs"));
    let entry = unsafe { core::ffi::CStr::from_ptr(entry.as_ptr().cast()) }
        .to_string_lossy()
        .to_string();
    let args: Vec<Vec<u16>> = vec![wide("-T"), wide(&profile), wide("-E"), wide(&entry)];
    let mut argv: Vec<*const u16> = args.iter().map(|a| a.as_ptr()).collect();
    unsafe {
        let create = dxc_create_instance().ok_or_else(windows::core::Error::empty)?;
        let mut compiler: *mut core::ffi::c_void = core::ptr::null_mut();
        create(&CLSID_DXC_COMPILER, &IID_DXC_COMPILER3, &mut compiler).ok()?;
        if compiler.is_null() {
            return Err(windows::core::Error::empty());
        }
        let vt = *(compiler as *const *const usize);
        let release: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32 =
            core::mem::transmute(*vt.add(2));
        let compile: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            *const DxcBuffer,
            *const *const u16,
            u32,
            *mut core::ffi::c_void,
            *const windows::core::GUID,
            *mut *mut core::ffi::c_void,
        ) -> HRESULT = core::mem::transmute(*vt.add(3));
        let buffer = DxcBuffer {
            ptr: source.as_ptr().cast(),
            size: source.len(),
            encoding: 0,
        };
        let mut result: *mut core::ffi::c_void = core::ptr::null_mut();
        let hr = compile(
            compiler,
            &buffer,
            argv.as_mut_ptr(),
            argv.len() as u32,
            core::ptr::null_mut(),
            &IID_DXC_RESULT,
            &mut result,
        );
        release(compiler);
        if hr.is_err() || result.is_null() {
            tracing::warn!(error = %hr, "DXC refused a shader of ours");
            return Err(windows::core::Error::from_hresult(hr));
        }
        let rvt = *(result as *const *const usize);
        let get_status: unsafe extern "system" fn(*mut core::ffi::c_void, *mut HRESULT) -> HRESULT =
            core::mem::transmute(*rvt.add(3));
        let get_output: unsafe extern "system" fn(
            *mut core::ffi::c_void,
            u32,
            *const windows::core::GUID,
            *mut *mut core::ffi::c_void,
            *mut *mut core::ffi::c_void,
        ) -> HRESULT = core::mem::transmute(*rvt.add(7));
        let mut status = HRESULT(0);
        let _ = get_status(result, &mut status);
        if status.is_err() {
            let mut errors: *mut core::ffi::c_void = core::ptr::null_mut();
            if get_output(result, DXC_OUT_ERRORS, &IID_DXC_BLOB, &mut errors, core::ptr::null_mut()).is_ok()
                && !errors.is_null()
            {
                let bvt = *(errors as *const *const usize);
                let ptr: unsafe extern "system" fn(*mut core::ffi::c_void) -> *const core::ffi::c_void =
                    core::mem::transmute(*bvt.add(3));
                let size: unsafe extern "system" fn(*mut core::ffi::c_void) -> usize =
                    core::mem::transmute(*bvt.add(4));
                let rel: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32 =
                    core::mem::transmute(*bvt.add(2));
                let text = std::slice::from_raw_parts(ptr(errors).cast::<u8>(), size(errors));
                tracing::warn!("a shader of ours (DXC): {}", String::from_utf8_lossy(text));
                rel(errors);
            }
            release(result);
            return Err(windows::core::Error::from_hresult(status));
        }
        let mut blob: *mut core::ffi::c_void = core::ptr::null_mut();
        let got = get_output(result, DXC_OUT_OBJECT, &IID_DXC_BLOB, &mut blob, core::ptr::null_mut());
        if got.is_err() || blob.is_null() {
            release(result);
            return Err(windows::core::Error::from_hresult(got));
        }
        let bvt = *(blob as *const *const usize);
        let ptr: unsafe extern "system" fn(*mut core::ffi::c_void) -> *const core::ffi::c_void =
            core::mem::transmute(*bvt.add(3));
        let size: unsafe extern "system" fn(*mut core::ffi::c_void) -> usize =
            core::mem::transmute(*bvt.add(4));
        let rel: unsafe extern "system" fn(*mut core::ffi::c_void) -> u32 =
            core::mem::transmute(*bvt.add(2));
        let bytes = std::slice::from_raw_parts(ptr(blob).cast::<u8>(), size(blob)).to_vec();
        rel(blob);
        release(result);
        tracing::info!(profile, "a shader of ours was built by DXC");
        Ok(bytes)
    }
}