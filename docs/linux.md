# Linux (experimental)

VIctory's runtime uses Windows APIs and Direct3D 12. On Linux, cross-compile
the Windows MSVC executable and run it with **GE-Proton via umu-run**.
Proton's vkd3d-proton translates D3D12 to Vulkan. Native Linux builds are
useful for platform-independent tools, but cannot execute the title.

This workflow is experimental: @MiguVT confirmed that it launches and loads
the cinematic on Linux, but rendering is mostly dark and performance is
unplayable. The fork's rendering and performance limitations in the README
still apply. Existing Wine-specific memory handling is included in this branch.

## Reported test: @MiguVT

Reported by [@MiguVT](https://github.com/MiguVT) on their PC. Outcome:
**launches, cinematic loads, not playable**.

| Observation | Result |
| --- | --- |
| Cinematic | Loads, but appears fully dark for most of the scene |
| Visible rendering | Sky occasionally visible; scene otherwise remains dark |
| Reported frame time | Approximately **6,275 ms/frame** (6.275 seconds per frame) |
| Equivalent frame rate | Approximately **0.16 FPS**, calculated from the reported frame time |
| Playability | **Not playable** |

This is a user-observed frame time, not an averaged benchmark. It does not
establish that the whole cinematic completes or that gameplay works.
The cause of the Linux rendering failure has not been diagnosed.

### Hardware and host software

CPU and GPU were supplied by the tester and confirmed on the development
host. The remaining host details were collected after the report; they are
not a capture of the game's launch environment.

| Component | Recorded specification |
| --- | --- |
| CPU | AMD Ryzen 9 5950X, 16 cores / 32 threads, x86-64, AVX2 |
| GPU | NVIDIA GeForce RTX 3090 |
| VRAM | 24 GiB (24,576 MiB reported by `nvidia-smi`) |
| System memory | 62 GiB usable RAM reported by `free -h` |
| Swap | 18 GiB total configured; not a measurement of game memory use |
| Distribution | NixOS 26.11 (Zokor) |
| OS build | `26.11.20261121.2786a2a` |
| Kernel | `7.3.4-zen2`, x86-64 |
| NVIDIA driver | `595.112.07` |
| Desktop session | Wayland; the game's actual window-system backend was not captured |
| Installed launcher | Nixpkgs `umu-launcher` 1.4.4 (`bwrap` package) |
| Source base | `4e17478b9c27bb530643e5b0e16a767cf4419e08`, with local Linux workflow additions |

### Runtime and reproduction limits

The documented launch path is the Windows x86-64 MSVC release executable,
run through umu and GE-Proton, with D3D12 translated to Vulkan by
vkd3d-proton. The Linux cross-build and launcher checks passed separately.

The exact Proton build selected for this game run, bundled Wine and
vkd3d-proton versions, launch command, environment switches, console profile,
resolution and shader-cache state were not captured. Several Proton builds
are installed on the host, so the installed list cannot identify which was
used. The exact tested binary revision also needs confirmation against the
source base above.

For a follow-up comparison, save those details alongside terminal output
and `crashes.log` if present. In particular, record an explicit `PROTONPATH`
instead of relying on the moving `GE-Proton` selector, and note whether the
shader cache is warm or cold. No Linux audio, input, saving, stability or
playable-world result is claimed by this report.

## Requirements

- x86-64 Linux, an AVX2-capable CPU, and enough RAM/swap (the fork reports
  around 24 GB of memory during gameplay on Windows).
- Current Vulkan drivers and a GPU supporting the D3D12 features used by
  VIctory, including shader model 6.6 and mesh shaders for the world.
- [umu-launcher](https://github.com/Open-Wine-Components/umu-launcher)
  (`umu-run`), installed using its distribution instructions.
- Rustup, the toolchain in `rust-toolchain.toml`, Clang, LLD and LLVM.
- [cargo-xwin](https://github.com/rust-cross/cargo-xwin):
  `cargo install --locked cargo-xwin` (unnecessary in the Nix shell).
- Your own title files, kept outside this repository.

umu downloads GE-Proton and the Steam runtime on first launch, so allow
network access and disk space. Steam itself does not need to be running.
cargo-xwin downloads Microsoft's CRT/SDK on the first build; consult its
documentation for the applicable licensing and cache settings.

## Nix / NixOS

From the repository root:

```sh
nix-shell
```

`shell.nix` delegates to `dev/shell.nix`, providing rustup, cargo-xwin, LLVM
tools, umu-launcher and ShellCheck. Rustup uses the repository's pinned Rust
version; it downloads that toolchain when first invoked. No flake is needed.
The shell uses your `<nixpkgs>` channel, so its package versions are not
pinned. Use a recent channel containing `cargo-xwin` and `umu-launcher`.
You can supply your own pinned package set through the `pkgs` argument.

On NixOS, configure graphics at the system level, then rebuild:

```nix
hardware.graphics = {
  enable = true;
  enable32Bit = true;
};
```

Select the appropriate GPU driver in your NixOS configuration as usual.
The development shell does not install system graphics drivers. Use the
Nixpkgs-packaged umu-launcher so its runtime integration is available.

## Build

```sh
./scripts/build-linux.sh
```

This runs `cargo xwin build --locked --release --target
x86_64-pc-windows-msvc -p victory`. It preserves native Cargo defaults for
other development work. Extra build arguments are passed through:

```sh
./scripts/build-linux.sh --features diagnostics
```

The resulting executable is
`target/x86_64-pc-windows-msvc/release/victory.exe`. A Windows-built
executable can also be selected with `VICTORY_EXE=/absolute/path/victory.exe`.

## Configure and launch

Copy `victory.example.toml` to `victory.toml`. Paths inside that file are
read by a **Windows process**: use Proton's `Z:` drive mapping for Linux
files, and TOML single quotes to keep backslashes literal:

```toml
[paths]
game_dir = 'Z:\home\alice\Games\title'
cache_dir = 'Z:\home\alice\.cache\victory'
data_dir = 'Z:\home\alice\.local\share\victory\data'
test_titles = []
```

Replace these paths with your own; keep game files, cache and saves outside
the repository. `VICTORY_CONFIG`, `VICTORY_GAME_DIR`, `VICTORY_CACHE_DIR`,
`VICTORY_DATA_DIR` and other path overrides also need Windows-format paths.
If you omit cache/data overrides, VIctory uses Windows local-app-data
directories inside the selected prefix.

```sh
./scripts/run-linux.sh config
./scripts/run-linux.sh run "/home/alice/Games/title/title.exe" --execute
```

The launcher converts the title path immediately after `run` from a Linux
path to `Z:\...`; relative paths resolve against your calling directory.
Already-Windows paths are passed through. Other commands' path arguments
must be supplied in Windows form, for example:

```sh
./scripts/run-linux.sh inspect 'Z:\home\alice\Games\title\title.exe'
```

The launcher uses the repository as its working directory so `victory.toml`
is found. It defaults to a dedicated prefix at
`${XDG_DATA_HOME:-$HOME/.local/share}/victory/prefix`, `GAMEID=umu-default`,
and `PROTONPATH=GE-Proton` (umu's automatically downloaded latest GE build).
All three can be overridden:

```sh
PROTONPATH="/home/alice/.local/share/Steam/compatibilitytools.d/GE-Proton-YOUR-VERSION" \
WINEPREFIX="/home/alice/Games/victory-prefix" \
  ./scripts/run-linux.sh run "/home/alice/Games/title/title.exe" --execute
```

Use an explicit Proton path when reproducing a bug so an automatic update
does not change the test environment. Custom prefixes must retain the `Z:`
mapping for Linux-path conversion to work.

## Terminal logs and troubleshooting

The launcher directly `exec`s umu-run, inheriting stdin, stdout and stderr.
It passes `victory.exe` as a Windows `Z:\...` path so GE-Proton uses its
direct Wine launch path instead of the `umu.exe` helper used for Unix paths,
which can open a separate console window. The prefix must retain its `Z:`
mapping. The Linux executable directory is still supplied to umu for runtime
mount discovery. No terminal emulator or `wineconsole` is started by the script.
umu may print `Executable not found: Z:\...` because its Linux-side check
does not resolve Windows paths; it then passes the path to Proton. This
warning alone is not a launch failure (the script checks the Linux file
beforehand). A `--version` smoke test with GE-Proton11-14 successfully
returned `victory 0.1.0` through the calling terminal's output.
Keep logs in the current terminal, or capture them with:

```sh
set -o pipefail
./scripts/run-linux.sh run "/home/alice/Games/title/title.exe" --execute \
  2>&1 | tee "$HOME/victory-linux.log"
```

- **Missing executable:** build first, or set `VICTORY_EXE` to an existing
  Windows binary. A custom Cargo target directory also needs this override.
- **Paths not found:** run `config`; check Windows path spelling, the `Z:`
  mapping, Linux filename case, and access to external drives.
- **D3D12/Vulkan initialization fails:** check host Vulkan drivers and GPU
  capabilities first. Do not force a D3D feature level to mask missing support.
- **umu startup fails:** retry with `UMU_LOG=debug` and consult umu's
  documentation. `PROTON_LOG=1` enables additional Proton file logging;
  it is optional, not required for normal terminal output.
- **Crashes:** check `crashes.log` under the configured data directory.
  Report the commit, Proton version, GPU/driver, distribution, command and
  relevant logs. Windows performance numbers are not Linux benchmarks.

For script checks, run `shellcheck scripts/*-linux.sh`. Cross-compiling
checks the Windows runtime; it does not establish game compatibility.
