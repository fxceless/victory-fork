# Local 16 GiB RAM / 8 GiB VRAM branch

Prepared 2026-11-27 on **victory-fork/main at
2eade68cade9b2f8cd2511cc97328e83f58508c3**. Updated 2026-11-28 to merge main at
`368967f8fea2a982cea134c5a051b2afe7d9fc8f` into `local/16gb-8gb`.
The source comparison below describes the original inspected heads. A full
Rust build and gameplay measurements of the updated branch are still required.

## Source comparison

This is a code-review recommendation, not a controlled performance ranking.

| Inspected source | Local head | Relevant evidence | Decision |
|---|---|---|---|
| VIctory/main | 63bcca4 | Portable runtime and PC FidelityFX bridge; its README describes an early, much slower renderer | Interoperability reference |
| victory-fork/main | 2eade68 | New texture-block admission and deferred retry, descriptor lifetime/backpressure work, target-memory reservations, ray-page budgeting, AMD wave detection, native title FSR path | Recommended base |
| victory-unified/main | 92dc878 | Useful wave, codec and depth fixes; explicitly tuned for RTX 3070 and 32 GB RAM | Reference; its launcher forces NVIDIA and is not a generic 16 GB preset |
| cgbj-fork wave1/upstream | 47e4b60 | Earlier renderer; our previous local branch adds smaller caches and presentation FSR 1 | Retained as a comparison branch |
| cgbj-fork wave2/upstream | 253ea94 | Same earlier renderer with residency controls and placement option | Retained as a comparison branch |

The recommended base already contains the BC7 rotation correction seen in
victory-unified. It is not reapplied as a new fix; this branch adds regression
coverage for the valid rotation modes and packed mode-6 channel ordering.
Its host GPU initialization detects AMD/RDNA waves; NVIDIA-focused features
are optional. This does not establish tested performance on the user's AMD GPU.

The original base had a new-block gate from `2eade68`. Main at `368967f` returns
to the run-403 renderer and removes that gate, its wait/retry helpers and the
dynamic descriptor-wait switch. This merge keeps main's allocator and shader/
water fixes. It adapts our supplemental texture cap to main's live WDDM usage,
free pool space and buffer/target reservations. It does not reintroduce the
removed gate or claim a hard bound on every new heap allocation. The launchers
no longer advertise the removed switches. This is not a wholesale merge of the
different renderers.

## Fixes applied here

- Explicit texture caps are now supplemental to the real WDDM budget and
  reservations. They cannot bypass reported-memory admission.
- The initial-texture floor no longer bypasses live budget checks.
  A known exhausted budget cannot be treated as spare
  capacity merely because few textures have been created.
- Texture usage and reservations share saturating arithmetic between admission
  and reclamation. The removed block-gate arithmetic and tests are not retained.
- Automatic <=16 GiB RAM detection and `VICTORY_MEMORY_PROFILE=16gb-8gb` select
  smaller speculative work before memory pressure has already developed.
- Small-profile texture passes have soft limits of 32 MiB, 16 MiB under
  pressure or substantial staging, and 8 MiB when critical. A single texture
  can cross the soft limit; live resources are not silently discarded.
- Speculative staging checks free physical RAM as well as commit headroom.
  Required material-bake work retains the existing behavior.
- Idle texture-pool storage is reduced from up to 4 GiB to 256 MiB on the
  forced profile; <=8 GiB WDDM budgets otherwise keep at most 512 MiB.
- Idle staging keeps at most two 64 MiB blocks in the small profile, one under
  pressure and none when critical. Idle CPU snapshots have a strict 64 MiB
  cache; oversized snapshots are not retained for reuse.
- Small-profile presentation waits for completed work after two ahead slots
  rather than continually allocating another command list.
- Launchers now forward Series S/X selection, reject missing files and runner
  failures, and restore the parent environment and current directory on exit,
  including on failure.

Host-memory admission retains the existing rules for mandatory work. Required
title allocations, active buffer mirrors, pending GPU work, Windows and the
driver are not subject to a hard 16 GiB total-process cap. These changes do not
establish that the game fits or runs well on 16 GiB / 8 GiB hardware.

## Upscaling

The recommended base uses **the title's temporal FidelityFX path**, controlled
by its existing `VICTORY_FSR_PATH` and mode-selection code. It has the title's
depth, motion and camera information. The low-memory launcher disables DLSS,
NVIDIA frame generation and ray tracing by default, selects Series S and asks
for the quality mode through that existing path.

The mode variable is named `VICTORY_DLSS_MODE` in this upstream runtime even
when selecting the title's own FSR path. The launcher uses it without enabling
DLSS. The existing executable hash and byte checks guard the optional scale
patch. Unsupported builds report that it was not applied and keep the stock
path; this is not a guarantee that every dump supports that patch.

The separate VIctory/main source also contains a **PC FidelityFX SDK bridge**.
That is a different implementation from this base's Streamline interposer and
native title FSR path. It was inspected, but not blindly substituted into these
interfaces. No DLL, neural model or game file from Watch Dogs 2 was copied or
loaded. The earlier cgbj local branches retain their optional presentation FSR 1;
it is not stacked on this base's temporal reconstruction.

## Build and run

```powershell
cargo build --release --locked -p victory
scripts\play-low-memory.ps1 -Exe 'D:\YourXboxDump\game_scarlett_master.exe'
# Keep the stock upscaler path for comparison:
scripts\play-low-memory.ps1 -Exe 'D:\YourXboxDump\game_scarlett_master.exe' -Upscaling stock
# Review without starting anything:
scripts\play-low-memory.ps1 -Exe 'D:\YourXboxDump\game_scarlett_master.exe' -ShowSettings
# Use a binary built elsewhere:
scripts\play-low-memory.ps1 -Exe 'D:\YourXboxDump\game_scarlett_master.exe' -Runner 'D:\Builds\victory.exe'
```

The executable path above is an example; no compatible Xbox title is configured
here. `-Upscaling balanced` / `performance` select the existing lower-resolution
paths. Output resolution remains the title/profile's; the launcher does not
pretend that stretching the final image reduces earlier rendering work.
`-RayTracing` opts back into ray tracing. `-More` supports explicit overrides.
Presets are starting points for measurement, not demonstrated optimal values.

## Validation

```powershell
tests\low-memory-launcher.ps1
scripts\check-local-fixes.ps1 -RequireRust
```

Observed: production launcher checks pass for all four scale modes and explicit
overrides. A synthetic runner verifies actual argument/settings forwarding,
environment restoration after success and exit code 7, current-directory
restoration, and rejection of a missing runner. PowerShell syntax and diff/text
checks are also performed. The stub never launches or contains a game.

Additional Rust regression tests cover cache/admission rules, cap intersection,
reservation arithmetic/overflow cases and BC7 precision/packing. **They have not
run locally**, because Rust/Cargo is missing in this execution environment. The pinned
toolchain remains unchanged. Full workspace Clippy, workspace tests and game
qualification are pending. The previous branches' six FSR 1 WARP readbacks do
not validate this different temporal renderer.

PortBridge independently executed the launcher and six standalone budget checks
against the earlier published head `4351fb4`. Its report does not validate this
merge's changed admission wiring or a full GPU/crate build. Those still require
the updated CI and gameplay checks.

To qualify the target, build with the pinned toolchain and run the same scene on
actual 16 GiB / 8 GiB hardware. Measure working set, private bytes, commit
headroom, local/non-local WDDM use, frame-time tails and texture arrival during a
sustained traversal; verify resize, retries and heavy scenes under the D3D12
debug layer. No measured FPS or total-memory claim is made by this branch.
