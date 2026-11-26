//! The title's clocks, slowed (`VICTORY_TIME_SCALE=<0..1>`; off by default).
//!
//! The title measures a frame by its clocks and steps its physics by that.
//! Here a frame takes 300-500 ms where the console's takes 33, so every step
//! is ten times the console's; in the first mission the player's boat sinks
//! as soon as the cutscene hands over, with the water heights it asks the
//! card for arriving as they should (`VICTORY_WATCH_SHADER`). With a scale
//! of 0.1 the clocks advance a tenth of real time: the title runs in slow
//! motion and its steps are the console's again. A test of that cause, and
//! a way through scenes that need it.
//!
//! The clocks the title imports: `QueryPerformanceCounter`, the C++
//! runtime's `_Query_perf_counter` (`std::chrono::steady_clock`) and
//! `_Xtime_get_ticks` (`system_clock`), and `GetTickCount64`; each scaled
//! from the moment it is first read, so each stays monotonic and they agree
//! with one another. Frequencies are the host's.
//! shortcut: `rdtsc` in the title's own code, waits with timeouts and the
//! audio the layer plays keep real time; the wall-clock calls
//! (`GetSystemTimeAsFileTime`, `GetLocalTime`) are left alone.

use std::sync::OnceLock;

use crate::{HleModule, hle_export};

/// The scale, if one in (0, 1] is set.
fn scale() -> Option<f64> {
    static SCALE: OnceLock<Option<f64>> = OnceLock::new();
    *SCALE.get_or_init(|| {
        let scale = std::env::var("VICTORY_TIME_SCALE")
            .ok()
            .and_then(|v| v.trim().parse::<f64>().ok())
            .filter(|s| *s > 0.0 && *s <= 1.0);
        if let Some(s) = scale {
            tracing::info!(scale = s, "the title's clocks run at {s} of real time (VICTORY_TIME_SCALE)");
        }
        scale
    })
}

/// `now` on a clock first read at `start`, scaled from there.
fn scaled(start: &OnceLock<i64>, now: i64) -> i64 {
    let first = *start.get_or_init(|| now);
    let s = scale().unwrap_or(1.0);
    first + ((now - first) as f64 * s) as i64
}

fn performance_counter() -> i64 {
    static START: OnceLock<i64> = OnceLock::new();
    let mut now = 0;
    // SAFETY: a local out value.
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut now) };
    scaled(&START, now)
}

fn performance_frequency() -> i64 {
    let mut f = 0;
    // SAFETY: a local out value.
    unsafe { windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut f) };
    f
}

#[allow(non_snake_case)]
pub extern "system" fn QueryPerformanceCounter(count: *mut i64) -> i32 {
    if count.is_null() {
        return 0;
    }
    // SAFETY: the title's out pointer.
    unsafe { *count = performance_counter() };
    1
}

#[allow(non_snake_case)]
pub extern "system" fn QueryPerformanceFrequency(frequency: *mut i64) -> i32 {
    if frequency.is_null() {
        return 0;
    }
    // SAFETY: the title's out pointer.
    unsafe { *frequency = performance_frequency() };
    1
}

/// `std::chrono::steady_clock`'s ticks: the performance counter.
#[allow(non_snake_case)]
pub extern "C" fn _Query_perf_counter() -> i64 {
    performance_counter()
}

#[allow(non_snake_case)]
pub extern "C" fn _Query_perf_frequency() -> i64 {
    performance_frequency()
}

/// `std::chrono::system_clock`'s ticks: 100 ns units since 1970.
#[allow(non_snake_case)]
pub extern "C" fn _Xtime_get_ticks() -> i64 {
    static START: OnceLock<i64> = OnceLock::new();
    const EPOCH: i64 = 116_444_736_000_000_000; // 1601 to 1970, in 100 ns
    let mut ft = windows_sys::Win32::Foundation::FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    // SAFETY: a local out value.
    unsafe { windows_sys::Win32::System::SystemInformation::GetSystemTimePreciseAsFileTime(&mut ft) };
    let now = (i64::from(ft.dwHighDateTime) << 32 | i64::from(ft.dwLowDateTime)) - EPOCH;
    scaled(&START, now)
}

#[allow(non_snake_case)]
pub extern "system" fn GetTickCount64() -> u64 {
    static START: OnceLock<i64> = OnceLock::new();
    // SAFETY: no arguments.
    let now = unsafe { windows_sys::Win32::System::SystemInformation::GetTickCount64() } as i64;
    scaled(&START, now) as u64
}

pub(crate) fn module() -> HleModule {
    HleModule {
        name: "clock",
        dlls: &["api-ms-win-core-profile-*", "api-ms-win-core-sysinfo-*", "kernel32.dll", "msvcp140.dll"],
        // Without a scale the host's own functions are bound, as before.
        exports: if scale().is_some() {
            vec![
                hle_export!(QueryPerformanceCounter),
                hle_export!(QueryPerformanceFrequency),
                hle_export!(GetTickCount64),
                hle_export!(_Query_perf_counter),
                hle_export!(_Query_perf_frequency),
                hle_export!(_Xtime_get_ticks),
            ]
        } else {
            Vec::new()
        },
    }
}
