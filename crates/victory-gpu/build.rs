//! D3D12 Agility SDK exports for this crate's test executables
//! (`src/agility.rs`), as in `victory-shader`'s build script.
//!
//! D3D12 uses the runtime an executable names only if it is there: with
//! the exports but no `D3D12Core.dll`, device creation fails. So the tests
//! export them only when `cargo xtask agility` has put the runtime in this
//! profile's `D3D12\` folder; without it they use the in-box runtime.
//!
//! `cargo xtask agility` touches this file after installing (or removing)
//! the runtime; watching it here makes cargo rerun this script. Watching a
//! missing D3D12Core.dll would rebuild every time.

use std::path::PathBuf;

fn main() {
    println!("cargo::rerun-if-changed=build.rs");

    // OUT_DIR is <target>/<profile>/build/<package>-<hash>/out.
    let out = PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo sets OUT_DIR"));
    let Some(profile) = out.ancestors().nth(3) else {
        return;
    };
    let core = profile.join("D3D12").join("D3D12Core.dll");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc");
    if msvc && core.exists() {
        println!("cargo::rerun-if-changed={}", core.display());
        // Every executable this crate links is a test executable.
        println!("cargo::rustc-link-arg=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo::rustc-link-arg=/EXPORT:D3D12SDKPath,DATA");
    }
}
