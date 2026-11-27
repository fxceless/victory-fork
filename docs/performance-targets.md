# Performance targets and reproducible measurements

These are engineering goals, **not achieved benchmark results**. No compatible
Xbox game is configured on this development host; Rust/Cargo is unavailable.
The Rust changes, release build, hardware performance and visual correctness
have not been verified by running the game.

The user's reference hardware was confirmed from Windows: Ryzen 5 7500X3D,
Radeon RX 9070 GRE, and about 31.1 GiB usable system RAM (32 GB installed).
The existing user hardware description identifies the GPU as the 12 GB model.
The user normally uses a 1080p monitor; **this source's existing patched output
mode is 1440p**, so results at that mode must not be labelled 1080p benchmarks.

| Tier | Intended hardware class | Minimum FPS goal | Target frame time | Default cap |
|---|---|---:|---:|---:|
| reference | User's hardware or similar | 90+ | 11.111 ms | 99 FPS |
| high | Faster hardware | 120+ | 8.333 ms | 132 FPS |
| mid | Slower hardware | 60+ | 16.667 ms | 66 FPS |
| entry | Lower-end hardware | 30+ | 33.333 ms | 33 FPS |

Hardware classes are not established by measurements or automatic detection.
Choose the tier being evaluated. The cap is 10% above the goal, allowing small
pacing overheads without making the cap itself prevent reaching the minimum.
A higher cap does not make the renderer finish within the target time.

## Graphics settings and "Ultra"

This is an Xbox compatibility layer: a PC "Ultra" preset is not exposed. A claim
of "Ultra at 90 FPS" cannot be made from a launcher label. Tests must report
the console model, output resolution, internal resolution/upscaler mode,
ray tracing, enabled effects, image quality and the actual scene tested.

The performance launcher defaults to Series X, the existing guarded 1440p
output patch and the title's FSR quality path. It keeps water lighting enabled
and clears the list of deliberately omitted shader passes. It disables DLSS
and frame generation so generated/displayed frames cannot satisfy a rendered
FPS goal. Ray tracing is an explicit option, and must be stated in results.
These settings preserve requested visual work; they are not a claim that every
shader translates, that no draw is dropped, or that an unsupported title build
successfully accepts the existing resolution/upscaler patches.

Use `-Upscaling stock` for the title's stock upscaling behavior. FSR quality,
balanced and performance change internal resolution. `-Console series-s
-Output stock` is a separate console-quality comparison, not the Ultra result.
No arbitrary render-target resizing or unguarded resolution patch was added.

## Stutter work and issues fixed

- Explicit `VICTORY_TEXTURE_STAGE_MB` ceilings previously bypassed RAM-pressure
  admission. They now intersect it, so a requested byte ceiling cannot force
  speculative texture work through critically low RAM/commit headroom.
- Performance profiles disable the 3x texture catch-up burst and cap ordinary
  texture-pass time to a small part of the frame budget (1/1/2/4 ms for
  high/reference/mid/entry). A single operation can overrun the soft deadline;
  required one-shot material bakes are deliberately retained.
- Known used pipelines can be warmed **before** starting the game, with bounded
  worker counts. A failed warm-up prevents launch and restores the parent's
  environment. First-time pipelines not yet recorded can still stall or drop
  draws; first traversal and warm traversal must be tested separately.
- Pipeline recording stays enabled so later starts know which pipelines were
  requested. Shared pipeline wait accounting and the existing dynamic
  descriptor wait fix remain enabled. Pipeline waits were not set to zero:
  making a frame faster by losing required passes would invalidate quality.
- Frame-clock and title simulation rate are set together. **All required timing
  patches must now succeed before execution**, including the simulation step;
  an unsupported/partial override stops with an error instead of running at an
  incorrect game speed. Code-protection restoration and instruction-cache
  flushing failures are also treated as patch failures.
- Failed or occluded Present calls no longer count as successful presented
  frames. Streamline's completion hooks still run on those outcomes.

Remaining bottlenecks include GPU shader execution, CPU command translation,
driver compilation, texture detiling/upload, mapped memory traffic and live
memory requirements. Texture throttling can delay texture arrival. Warm-up and
streaming budgets do not demonstrate the several-fold rendering improvement
that 90-120 FPS might require relative to upstream's reported frame times.

## Run a profile

```powershell
cargo build --release --locked -p victory
scripts\play-performance.ps1 -Exe 'D:\YourXboxDump\game_scarlett_master.exe' -Tier reference
# Other requested goals:
scripts\play-performance.ps1 -Exe 'D:\YourXboxDump\game_scarlett_master.exe' -Tier high
scripts\play-performance.ps1 -Exe 'D:\YourXboxDump\game_scarlett_master.exe' -Tier mid
scripts\play-performance.ps1 -Exe 'D:\YourXboxDump\game_scarlett_master.exe' -Tier entry
# Show the proposed settings without starting or writing anything:
scripts\play-performance.ps1 -Exe 'unused' -Tier reference -ShowSettings
```

The paths above are examples, not an installed game. Native FSR and timing
patches retain their existing supported-build checks. An explicit `-Runner`
can select a binary built elsewhere. `-SkipWarm` permits a separately labelled
cold-cache measurement. `-FrameCapFps` cannot be below the tier goal. `-More`
is for explicit overrides, which must be disclosed with results.

Known-pipeline warm-up does not discover every possible shader in an unvisited
world. First collect a representative route, exit, then repeat it with the same
build, driver, cache and settings. Driver/translator changes can require new
warm-up. Preserve per-device caches and check the logged adapter: this PC has
both integrated AMD graphics and the discrete RX 9070 GRE.

## Capture and analyze frame times

The performance launcher sets `VICTORY_FRAME_REPORT` outside the checkout, under
`%LOCALAPPDATA%\victory\benchmarks\<unique-run>\frames.csv`. The app adds its PID,
so a title relaunch writes a separate report rather than overwriting one.

CSV samples measure time between successful host presentation calls. They are
not GPU timestamps, scan-out measurements or proof of graphics quality. The
first presentation establishes the clock and does not fabricate an interval.
The render thread tries a bounded 1024-sample channel; a separate buffered
writer performs disk I/O. A full channel loses samples instead of blocking
rendering, and the report records loss and sequence information. An abrupt
termination can leave an incomplete report, which cannot qualify the target.

```powershell
scripts\analyze-frame-report.ps1 -Path 'C:\YourRun\frames-1234.csv' -TargetFps 90 -WarmupSeconds 30 -MinimumSeconds 120 -OutputJson 'C:\YourRun\summary.json'
```

Capture a specified active gameplay route; exclude startup, loading and menus
with an appropriate warm-up interval. Keep the game visible and avoid resizing
during the timed segment. A recreated presenter or occlusion can invalidate a
capture; test resizing separately for correctness. Do not let fast menus stand
in for gameplay.

The analyzer reports average FPS, nearest-rank p50/p95/p99/p99.9 frame times,
the reciprocal of the mean of the slowest 1% of intervals, worst frame, >50 ms
hitches and intervals over twice the target budget. It rejects qualification
when capture is incomplete, samples are missing, Present calls were rejected,
or frame generation was requested/detected. It checks the expected final sample
count as well as internal sequence gaps.

`MeetsTimingTarget` is **timing-only**: default qualification requires at least
60 measured seconds, average FPS at or above the goal, slowest-1% FPS at least
90% of the goal, p99 no more than 125% of the target frame budget, and no >50 ms
hitches. Set at least 120 seconds for a submitted result. It never certifies
"Ultra", image correctness, correct simulation speed or an untested hardware
class. Please include video and settings with each timing report.

## Validation status

PowerShell checks exercise the real launcher, all four targets, coupled timing
settings, full-pass switches, warm-up ordering/failure, environment restoration
and the report analyzer. Synthetic reports test known percentiles and 1% lows,
hitches hidden by averages, higher unmet targets, warm-up boundaries, missing
tail samples, sequence gaps, generated frames and incomplete captures.

Rust regressions cover a full nonblocking report queue, fractional-microsecond
timestamp consistency and partial timing-patch rejection. They have not run on
this host. CI also runs the launcher/analyzer regressions and a release build;
an unrun or pending CI job is not a passing result. `scripts\check-local-fixes.ps1
-RequireRust` runs the required workspace Clippy, tests and release build once
the pinned Rust toolchain and MSVC developer environment are available. No
hardware/Ultra FPS result is claimed by these changes.
