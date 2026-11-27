# Running the title on Linux/Proton

Status, 2026-11-25: **experimental.** The world renders, but at 0.3–0.5 fps on
the host below and with everything in shadow black. The Windows path is the
supported one; this page records what a Linux/Proton run needs and what to
expect. Nothing here is supported by upstream VIctory.

Measured on Arch/CachyOS, Ryzen 7 3700X, 32 GB, RTX 3070 (8 GB, 32-lane
waves), Proton Experimental with its bundled vkd3d-proton, title build
`b99f876b74d4a77f` (Xbox Series profile).

No game files are in this repository. Everything is built from the user's own
dump, as [the asset policy](asset-policy.md) requires.

## What you need

| Piece | Requirement |
|---|---|
| Title files | Your own dump of the Xbox Series build, extracted to a directory. Read-only is fine. |
| Steam | Proton Experimental and Steam Linux Runtime 4.0, installed from Library → Tools. |
| GPU | Direct3D 12 with shader model 6.6 and mesh shaders. The world was measured on an RTX 3070. |
| Build tools | Rust (pinned by `rust-toolchain.toml`), the `x86_64-pc-windows-msvc` target, `cargo-xwin`, clang and lld. |
| Compiler DLLs | Microsoft DXC v1.9.2609, Windows x64 `dxcompiler.dll` and `dxil.dll`, beside `victory.exe`. The runtime loads them inside the game process; the Linux `dxc` binary is not enough. |

The DXC fallback and the watchdog hooks are open PRs; see
[Where the code lives](#where-the-code-lives). Without them the world stays
black and runs end at the title's watchdog.

## Build

```sh
rustup toolchain install
rustup target add x86_64-pc-windows-msvc
cargo install cargo-xwin
# a compiler and linker for the cross build
sudo pacman -S clang lld        # Arch
# sudo apt install clang lld    # Debian/Ubuntu

cargo xwin build --profile diag --features diagnostics \
    -p victory --target x86_64-pc-windows-msvc --locked
```

`cargo-xwin` downloads the Windows SDK and CRT on the first build. Set
`XWIN_CACHE_DIR` to keep that cache somewhere specific. Do not set
`CFLAGS=/FIstring.h` (a common `cargo-xwin` workaround): it breaks blake3's
assembly.

Put the compiler beside the executable:

```sh
# from https://github.com/microsoft/DirectXShaderCompiler/releases (v1.9.2609)
unzip dxc_2026_09_29.zip 'bin\x64\dxcompiler.dll' 'bin\x64\dxil.dll'
cp bin/x64/dxcompiler.dll bin/x64/dxil.dll \
   target/x86_64-pc-windows-msvc/diag/
```

`dxil.dll` is not optional: DXC loads it to validate and sign the DXIL.

## Configure

1. Copy `run.example.sh` to `run.sh`, set `repo`, `game`, `work` and `steam`
   at the top, and `chmod +x run.sh`.
2. Create the Proton prefix once:

   ```sh
   ./run.sh cmd /c exit
   ```

3. Write `$work/compatdata/pfx/drive_c/victory.toml`. Use `Z:\...` paths for
   Linux directories:

   ```toml
   [paths]
   game_dir  = 'Z:\path\to\the-title-dump'
   cache_dir = 'Z:\path\to\victory-data\cache'
   data_dir  = 'Z:\path\to\victory-data\data'
   test_titles = []
   ```

The wrapper sets `STEAM_COMPAT_DATA_PATH`, `STEAM_COMPAT_INSTALL_PATH`,
`STEAM_COMPAT_MOUNTS` (so the game and the build are visible in the prefix)
and `WINEDLLOVERRIDES='d3d12,d3d12core,dxgi=n'`, then runs the Windows
executable inside the runtime.

## Run

```sh
export VICTORY_LOG=info
export VICTORY_NO_WATCHDOG=1              # with the watchdog PR
export VICTORY_SHADER_LOOP_LIMIT=4096
export VKD3D_DISABLE_EXTENSIONS=VK_NV_raw_access_chains
# the pad-only account picker needs Enter presses
export VICTORY_KEYS=$(seq 20 5 300 | awk '{printf "%s:Enter,",$1}' | sed 's/,$//')

./run.sh ./target/x86_64-pc-windows-msvc/diag/victory.exe \
    run 'Z:\path\to\the-title-dump\<the title exe>' --execute
```

`VICTORY_KEYS` may need adapting: on a fresh profile Enter alone can loop on
the Legal Notices screen, where Down selects "Next".

### Environment sets

Picture first (this is what produced the visible world):

```sh
VICTORY_LOG=info VICTORY_NO_WATCHDOG=1 VICTORY_SHADER_LOOP_LIMIT=4096
VKD3D_DISABLE_EXTENSIONS=VK_NV_raw_access_chains
```

Then the fork's speed switches, off by default:

```sh
VICTORY_TEXTURE_NO_WAIT=1                 # the big one on this host
VICTORY_LATE_WAIT=3 VICTORY_RELAY=2 VICTORY_WIDE_BACK=1
VICTORY_CARD_BUFFERS_MIN_MB=1 VICTORY_CARD_BUFFERS_MB=1024
```

On RADV, `VICTORY_RELAY=2` has caused a GPU page fault and device loss; leave
it off there.

Evidence capture (the directory **must exist**, or every save logs
`Path not found`):

```sh
mkdir -p /path/to/capture
VICTORY_CAPTURE=/path/to/capture VICTORY_DUMP_TARGETS=1 \
VICTORY_CAPTURE_FRAMES='4100,4600,5200'
```

## Check it worked

```
argument clamp: the system compiler refused it (...); trying DXC
```

must be followed by **no** `no argument clamp: indirect execution dropped`
line, and the drop summary must lose `not bounded`.

```
frame presented ... frame=4096 draws=22144 drawn=22142 dispatches=4219 dispatched=4210
```

`drawn` close to `draws` is the good state. The frame levels line should show
content rather than zeros, e.g. `levels=[105,119,118,119,118,0,0,1,0,1,1,128]`.

## Troubleshooting

| Symptom | Cause | What to do |
|---|---|---|
| Black world; `indirect kind …: not bounded` every frame | Wine's `d3dcompiler_47` refuses the argument clamp | Use the DXC fallback (#20) with `dxcompiler.dll` and `dxil.dll` beside the executable |
| Process ends at `+0x23cf683` after 6–8 minutes, or during loading | The title's own GPU-progress watchdog | Build with the hooks (#364) and set `VICTORY_NO_WATCHDOG=1` |
| Log stops after `host pipeline: making shaders=["mesh …"]`, CPU stays busy, no frame | A pipeline worker overflowed its stack | Build with #28 (512 MiB worker stacks) |
| Dies in ~40 s at `+0x164a298` (read `0x2c`) after scripted Enter | An old save makes "Continue" the default | Move `data/7952987A/saves` aside, or use a fresh `data_dir` |
| Captures show loading art | Capture numbers are presented frames, which advance while loading; the world starts around 4,000 | Ask for 4,100 and later |
| Frames take longer than ~2 s for a long stretch | The title's watchdog eventually fires | Shorter frames matter more than any single pass; the pipeline library survives, so the next run starts warmer |
| Run is ~2.5× slower with `VICTORY_D3D12_DRED` | DRED cost in the world | Do not set it for normal play |
| The title's crash reporter tries to upload a minidump | It targets `submit.backtrace.io`; the runtime blocks it | Run offline if you do not want the attempt |

Do **not** patch the watchdog thread's entry point instead of the kill path.
That was tried: it stops the kills, but the title then never draws (0 shaders
translated, 0 frames), because that thread also does work the title needs at
startup.

The first run is cold: 500–1,300 pipelines are built (the fork loads about
960 from its library and builds the rest). Later runs reuse them.

## Diagnosing

- `VKD3D_DEBUG=warn` — why vkd3d-proton refuses a pipeline.
- `WINEDEBUG=+server` — which wineserver requests dominate; pipe it into a
  counter.
- `victory translate --list --validate` — offline translation report over the
  dump. Set `VICTORY_DXIL` to the prefix's `dxil.dll`.

## Speed

Where the time went on the measured host, from `VICTORY_LOG=info` frame lines:

```
18,684 ms/frame = 14,069 preparing (copies 8,635 ms / 156 MB, textures 4,242 ms)
                + 686 GPU + 3,773 copy-back
write-watch askings of 1 ms or more: 774 a frame, 8,595 ms a frame
texture refreshes: 4,242 ms, of them 3,810 ms waiting for title memory to hold still
```

`VICTORY_TEXTURE_NO_WAIT=1` targets the 3.8 s remap wait exactly; a texture may
arrive one frame late. With it and the relay switches:

```
texture refreshes   4,242 ms → 267–380 ms   (the remap wait is 0.0 ms)
frame time          10–18 s  → 1.4–8.8 s    (typical 0.15–0.4 fps)
what remains        title copies 1.2–2.5 s, copy-back up to 5.7 s
```

After the copy work (#363) the world runs at **0.3–0.5 fps**, 2.1–3.6 s per
frame, with copies at 1.1–2.1 s and the card about 7–14 % busy: the CPU is the
frame. The copy path is still the next target. `VICTORY_JOIN_COPIES=1` joins
adjacent title copies; `VICTORY_CARD_BUFFERS_MIN_MB` and
`VICTORY_CARD_BUFFERS_MB` move large copies to the card; `VICTORY_INPLACE`
avoids copies by using title memory in place (first stage only).

## Expected warnings

- `a pipeline creation method that is not implemented ... slot=0x1c8 /
  slot=0x218` — the title is asking for pipeline states it serialized offline
  at build time. Those payloads are native console GPU microcode, so a PC GPU
  cannot use them. VIctory answers `E_NOTIMPL`, the title falls back to
  building the pipeline from its shaders, and that fallback is the path that
  renders. A few thousand of these lines at world entry are normal.
- `a stage of the ray-tracing structure builder, which is off` — expected while
  `VICTORY_BVH_BUILDER` is unset; this is also why everything in shadow is
  black.

## Limitations

- Everything in shadow is black: the title's ray-traced bounce light and
  lamp/sun shadows do not run yet.
- The world is 0.3–0.5 fps on this host; the Windows host reports 2–4 fps.
- Four compute programs overflow even a 128 MiB worker stack
  (`55679a0547649d0b`, `afeea48bbb9913ad`, `e8c962358d862e3f`,
  `44a9625f01769f33`). They recurse without bound; the supervisor skips them
  per run.
- `VICTORY_INPLACE=2` only places a probe window: the title's buffers are view
  memory, not private committed memory, so they are refused.

## Approaches that did not work

Do not repeat these:

- **Upstream/`r12` branch**: menus and loading art only; the world stayed
  black because the branching, descriptor and shader fixes were missing.
- **Without DXC**: black world, every indirect execution dropped, watchdog
  after a few minutes.
- **A stand-in pixel shader on every draw** (`VICTORY_STAND_IN_ALL=1`): useful
  as a diagnostic (flat geometry, no text), not a fix.
- **The rectangle geometry stage** from upstream #1020: this fork already
  handles rectangle lists, so it was not needed.

## Optional: automatic restarts

`scripts/auto-run.sh` is a supervisor for unattended runs. It is not needed
for a normal run. It starts the game, watches the log's mtime, and when a run
goes quiet for 150 s it reads the last `host pipeline: making` line, adds those
shader hashes to `auto-runs/skips.txt`, and restarts with
`VICTORY_SKIP_PIPELINES` set from that file. Each run appends its best frame
and fps to `auto-runs/ledger.txt`; three runs without a new skip stop the loop.

Configure it with `REPO`, `EXE`, `GAME`, `WORK`, `RUNNER`, `MAX_RUNS`,
`STALL_SECS`, `LOOP_LIMIT` and `SAME_STALL_LIMIT`. It defaults to the `run.sh`
wrapper at the repository root.

## Where the code lives

| Piece | Where |
|---|---|
| DXC fallback for the host's HLSL | [#20](https://github.com/fxceless/victory-fork/pull/20) |
| Watchdog hooks (upstream #454) | [#364](https://github.com/fxceless/victory-fork/pull/364) |
| 512 MiB pipeline-worker stacks | [#28](https://github.com/fxceless/victory-fork/pull/28) |
| Copy and pipeline-library work | [#363](https://github.com/fxceless/victory-fork/pull/363) |
| No blending on integer render targets | [#51](https://github.com/fxceless/victory-fork/pull/51) |
| Wine writes without a wineserver request | [#256](https://github.com/fxceless/victory-fork/pull/256) |
| This page and wrapper (supervisor optional) | [#365](https://github.com/fxceless/victory-fork/pull/365) |

## References

- Upstream issues that describe the same walls: #841 (2560×1440 scene vs empty
  3840×2160), #850 (cutscene black with audio), #1020 (rectangle lists), #1107
  (WaveSize 64 refused on 32-lane devices), #1133 (descriptor heap cap).
- XCL issue #662 (the GPU frame-signal write never lands, then the "GPU Loc
  Sig" watchdog) and #1525 (wave-size refusals are a device-capability wall).
- TuxBox: `docs/PROJECTSCARLETT.md` for the DXC-beside-the-executable
  arrangement, and `runtime/d3d12/src/device/rectangle.rs` for the
  rectangle-list algorithm.
- Another launcher that works for a tester: Proton-CachyOS Latest through
  `umu-run`, with `WINEDLLOVERRIDES=d3d12,d3d12core,d3dcompiler_47=n` and
  vkd3d-proton built from master beside the executable (Proton's bundled copy
  crashes RADV when the world loads), plus Microsoft's `d3dcompiler_47.dll`.
  The measurements above use Proton Experimental with its bundled
  vkd3d-proton, `VKD3D_DISABLE_EXTENSIONS=VK_NV_raw_access_chains`, and
  Microsoft's `dxcompiler.dll` and `dxil.dll` beside `victory.exe`.
