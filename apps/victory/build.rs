// LOCAL-ONLY (branch local/win10-agility, never pushed): exports the two
// symbols D3D12 reads from the process executable to select an app-local
// D3D12 Agility SDK runtime. Without them, Windows 10's in-box d3d12.dll
// refuses every translated pipeline's root signature
// ("Unsupported bit-flag set (root signature flags c00)").
fn main() {
    if std::env::var("CARGO_CFG_TARGET_ENV").as_deref() == Ok("msvc") {
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKVersion,DATA");
        println!("cargo:rustc-link-arg-bins=/EXPORT:D3D12SDKPath,DATA");
    }
}
