//! LOCAL-ONLY (branch `local/win10-agility`, never pushed): the D3D12
//! Agility SDK exports.
//!
//! Windows 10's in-box `d3d12.dll` stops at shader model 6.5 and refuses root
//! signatures with the directly indexed heap flags (0x400, 0x800), which every
//! translated pipeline uses: "Unsupported bit-flag set (root signature flags
//! c00)". D3D12 reads these two symbols from the process executable; when they
//! name a newer runtime than the system's, it loads `D3D12Core.dll` from
//! `D3D12SDKPath`, relative to the executable.
//!
//! `build.rs` exports both. `D3D12Core.dll` (package `Microsoft.Direct3D.D3D12`
//! 1.620.2, whose minor version is the number below) must sit in `.\D3D12\`
//! beside `victory.exe`. Windows 11 or a missing folder keeps the in-box one.
#![allow(non_upper_case_globals)]

/// The runtime's SDK version: the minor number of the package (1.620.2).
#[unsafe(no_mangle)]
pub static D3D12SDKVersion: u32 = 620;

/// A C string pointer D3D12 reads once at device creation.
#[repr(transparent)]
pub struct SdkPath(*const std::ffi::c_char);

// SAFETY: points at a static, immutable C string.
unsafe impl Sync for SdkPath {}

/// Where `D3D12Core.dll` is, relative to the executable.
#[unsafe(no_mangle)]
pub static D3D12SDKPath: SdkPath = SdkPath(c".\\D3D12\\".as_ptr());
