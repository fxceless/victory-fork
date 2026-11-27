//! `victory doctor`: report whether this machine can run the title's world,
//! and what is missing. Needs no game data, so it can be run on any PC.

use std::ptr;

use anyhow::Result;
use victory_gpu::gpu_info;
use victory_shader::emit::validator::Validator;
use windows::Win32::System::Threading::{IsProcessorFeaturePresent, PF_AVX2_INSTRUCTIONS_AVAILABLE};

/// Pure verdict from the facts we gathered, so it can be unit-tested.
pub fn verdict(
    sm66: bool,
    mesh: bool,
    avx2: bool,
    ram_gb: f64,
    vram_gb: f64,
) -> &'static str {
    if !avx2 {
        return "NOT READY: CPU lacks AVX2 (title runs x86-64 code natively)";
    }
    if !sm66 {
        return "NOT READY: GPU lacks shader model 6.6";
    }
    if ram_gb < 24.0 {
        return "MENUS ONLY: world needs ~24 GB of working set (upgrade RAM)";
    }
    if vram_gb < 8.0 {
        return "MENUS ONLY: world needs >= 8 GB of VRAM";
    }
    if !mesh {
        return "MENUS ONLY: world needs mesh shader support";
    }
    "READY: menus + world possible (expect 2-4 fps on reference hardware)"
}

pub fn run() -> Result<()> {
    println!("victory doctor");
    println!();

    // OS
    let build = reg_string(
        r"SOFTWARE\Microsoft\Windows NT\CurrentVersion",
        "CurrentBuild",
    )
    .unwrap_or_else(|| "?".to_owned());
    let ubr = reg_u32(r"SOFTWARE\Microsoft\Windows NT\CurrentVersion", "UBR")
        .map(|n| n.to_string())
        .unwrap_or_else(|| "0".to_owned());
    println!("OS build     : {build}.{ubr}");

    // CPU
    let cpu = reg_string(
        r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
        "ProcessorNameString",
    )
    .unwrap_or_else(|| "unknown".to_owned());
    let avx2 = unsafe { IsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE).as_bool() };
    println!("CPU          : {}", cpu.trim());
    println!("AVX2         : {}", if avx2 { "yes" } else { "no" });

    // Memory
    let (total, avail, commit_total, commit_avail) = memory_status();
    println!(
        "RAM          : {:.1} GB total, {:.1} GB available",
        total as f64 / 1e9,
        avail as f64 / 1e9
    );
    println!(
        "Commit       : {:.1} / {:.1} GB",
        (commit_total as f64 - commit_avail as f64) / 1e9,
        commit_total as f64 / 1e9
    );

    // GPU (from the runtime device's adapter)
    let info = gpu_info();
    let name = info
        .as_ref()
        .map(|i| i.name.clone())
        .unwrap_or_else(|| "unknown".to_owned());
    let vram_gb = info
        .as_ref()
        .map(|i| i.vram_bytes as f64 / 1e9)
        .unwrap_or(0.0);
    let sm66 = info.as_ref().map(|i| i.shader_model_6_6).unwrap_or(false);
    let mesh = info.as_ref().map(|i| i.mesh_shaders).unwrap_or(false);
    println!("GPU          : {name}");
    println!("VRAM         : {vram_gb:.1} GB");
    println!("Shader model : {}", if sm66 { "6.6" } else { "< 6.6" });
    println!("Mesh shaders : {}", if mesh { "yes" } else { "no" });

    // dxc (informational only)
    let dxc_path = Validator::default_path();
    println!(
        "dxc.exe      : {}",
        dxc_path
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "not found (informational)".to_owned())
    );

    println!();
    println!(
        "{}",
        verdict(sm66, mesh, avx2, total as f64 / 1e9, vram_gb)
    );
    Ok(())
}

fn memory_status() -> (u64, u64, u64, u64) {
    use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    let mut ms = MEMORYSTATUSEX {
        dwLength: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: a valid, sized structure.
    if unsafe { GlobalMemoryStatusEx(&mut ms) }.is_ok() {
        (
            ms.ullTotalPhys,
            ms.ullAvailPhys,
            ms.ullTotalPageFile,
            ms.ullAvailPageFile,
        )
    } else {
        (0, 0, 0, 0)
    }
}

fn reg_string(key: &str, value: &str) -> Option<String> {
    use windows::Win32::System::Registry::*;
    use windows::core::PCWSTR;
    unsafe {
        let mut hkey = HKEY(ptr::null_mut());
        let key_w: Vec<u16> = key.encode_utf16().chain(std::iter::once(0)).collect();
        if RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            PCWSTR(key_w.as_ptr()),
            None,
            KEY_READ,
            &mut hkey,
        ) != windows::Win32::Foundation::ERROR_SUCCESS
        {
            return None;
        }
        let value_w: Vec<u16> = value.encode_utf16().chain(std::iter::once(0)).collect();
        let mut buf = vec![0u8; 512];
        let mut len = buf.len() as u32;
        let r = RegQueryValueExW(
            hkey,
            PCWSTR(value_w.as_ptr()),
            None,
            None,
            Some(buf.as_mut_ptr()),
            Some(&mut len),
        );
        let _ = RegCloseKey(hkey);
        if r != windows::Win32::Foundation::ERROR_SUCCESS {
            return None;
        }
        let u16s: Vec<u16> = buf[..len as usize]
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .collect();
        Some(String::from_utf16_lossy(&u16s).trim_end_matches('\0').to_owned())
    }
}

fn reg_u32(key: &str, value: &str) -> Option<u32> {
    use windows::Win32::System::Registry::*;
    use windows::core::PCWSTR;
    unsafe {
        let mut hkey = HKEY(ptr::null_mut());
        let key_w: Vec<u16> = key.encode_utf16().chain(std::iter::once(0)).collect();
        if RegOpenKeyExW(
            HKEY_LOCAL_MACHINE,
            PCWSTR(key_w.as_ptr()),
            None,
            KEY_READ,
            &mut hkey,
        ) != windows::Win32::Foundation::ERROR_SUCCESS
        {
            return None;
        }
        let value_w: Vec<u16> = value.encode_utf16().chain(std::iter::once(0)).collect();
        let mut v: u32 = 0;
        let mut len = std::mem::size_of::<u32>() as u32;
        let r = RegQueryValueExW(
            hkey,
            PCWSTR(value_w.as_ptr()),
            None,
            None,
            Some(&mut v as *mut u32 as *mut u8),
            Some(&mut len),
        );
        let _ = RegCloseKey(hkey);
        if r != windows::Win32::Foundation::ERROR_SUCCESS {
            None
        } else {
            Some(v)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verdict_reports_missing_avx2() {
        assert!(verdict(true, true, false, 32.0, 12.0).contains("NOT READY"));
        assert!(verdict(true, true, false, 32.0, 12.0).contains("AVX2"));
    }

    #[test]
    fn verdict_reports_missing_sm66() {
        assert!(verdict(false, true, true, 32.0, 12.0).contains("NOT READY"));
        assert!(verdict(false, true, true, 32.0, 12.0).contains("6.6"));
    }

    #[test]
    fn verdict_reports_low_ram() {
        assert!(verdict(true, true, true, 16.0, 12.0).contains("MENUS ONLY"));
        assert!(verdict(true, true, true, 16.0, 12.0).contains("RAM"));
    }

    #[test]
    fn verdict_reports_low_vram() {
        assert!(verdict(true, true, true, 32.0, 4.0).contains("VRAM"));
    }

    #[test]
    fn verdict_reports_missing_mesh() {
        assert!(verdict(true, false, true, 32.0, 12.0).contains("MENUS ONLY"));
        assert!(verdict(true, false, true, 32.0, 12.0).contains("mesh"));
    }

    #[test]
    fn verdict_reports_ready() {
        assert!(verdict(true, true, true, 32.0, 12.0,).starts_with("READY"));
    }
}
