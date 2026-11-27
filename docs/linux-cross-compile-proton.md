# Building and running on Linux (Proton / Wine)

> **Note:** Windows is the primary target of this repository. The opt-in switches
> and instructions below are strictly secondary — for contributors who are on
> Linux and want to test or work on the codebase without a Windows machine.
> They are off by default and do not change or affect Windows behavior.

## Cross-compiling `victory.exe` on Linux

Install the Rust toolchain pinned in `rust-toolchain.toml` with the Windows MSVC
target, plus [`cargo-xwin`](https://github.com/rust-cross/cargo-xwin) (requires
`clang` and `lld`):

```sh
rustup target add x86_64-pc-windows-msvc
cargo install --locked cargo-xwin
```

Build the diagnostics or release binary:

```sh
cargo xwin build --profile diag -p victory --features diagnostics --target x86_64-pc-windows-msvc
# or:
cargo xwin build --release -p victory --target x86_64-pc-windows-msvc
```

The resulting `victory.exe` is at `target/x86_64-pc-windows-msvc/<profile>/victory.exe`.

## Running under Steam Proton (vkd3d-proton)

Point `VICTORY_GAME_DIR`, `VICTORY_CACHE_DIR`, and `VICTORY_DATA_DIR` at paths
visible inside your Wine/Proton prefix (e.g. `Z:\...`), then run `victory.exe`
with Proton:

```sh
export VICTORY_GAME_DIR='Z:\path\to\game'
export VICTORY_CACHE_DIR='Z:\path\to\victory-cache'
export VICTORY_DATA_DIR='Z:\path\to\victory-data'

# Optional Proton/Linux helpers (all off by default, no effect on Windows):
export VICTORY_LOG_FILE='Z:\path\to\victory.log'  # capture logs when Proton's steam.exe swallows stderr
export VICTORY_SKIP_TITLE_INT3=1                  # skip non-fatal int3 breakpoints inside title images
export VKD3D_SHADER_CACHE_PATH=/path/to/vkd3d-cache
export MESA_VK_WSI_PRESENT_MODE=fifo              # keep 30 fps video/menu presentation in order on RADV/Xwayland
```

Then launch using `proton runinprefix` (with `STEAM_COMPAT_DATA_PATH` and
`STEAM_COMPAT_CLIENT_INSTALL_PATH` set) or Proton's `wine` binary directly:

```sh
export STEAM_COMPAT_DATA_PATH=/path/to/compatdata
export STEAM_COMPAT_CLIENT_INSTALL_PATH="$HOME/.local/share/Steam"
"$PROTON_DIR/proton" runinprefix victory.exe run 'Z:\path\to\game\<title.exe>' --execute
# or directly with Proton's wine:
# WINEPREFIX="$STEAM_COMPAT_DATA_PATH/pfx" "$PROTON_DIR/files/bin/wine64" victory.exe run 'Z:\path\to\game\<title.exe>' --execute
```

### Bubblewrap (`bwrap`) note

If you run Proton inside a `bwrap` sandbox with `--dev /dev`, make sure
`--dev-bind /dev/dri /dev/dri` comes **after** `--dev /dev`, otherwise the
fresh `/dev` mount hides `/dev/dri` and Vulkan/vkd3d-proton will fail to create
a device.
