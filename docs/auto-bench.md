# Deterministic Traversal Benchmarking & Closed-Loop Optimization

This workflow lets you measure real-world traversal scenes reproducibly, compare
baseline vs. candidate builds with interleaved A/B runs, and optionally drive an
autonomous optimization loop with strict anti-reward-hacking gates.

All benchmark and input-script outputs **must live outside the VIctory
repository** (see [asset-policy.md](asset-policy.md)). Relative and in-repo
output paths are rejected at runtime.

---

## Quick start: record one route and replay it twice

**1. Record one traversal route** (captures pad/key changes keyed to
`world_frame` and exits cleanly once warmup + measured frames complete):

```powershell
$env:VICTORY_INPUT_RECORD      = 'D:\victory-bench\scripts\route.jsonl'
$env:VICTORY_BENCH_WARMUP      = '30'
$env:VICTORY_BENCH_EXIT_FRAMES = '180'
$env:VICTORY_BENCH_OUT         = 'D:\victory-bench\runs\record.json'
& 'D:\victory-bench\launch-title.cmd'
```

**2. Replay that route twice** (e.g. baseline vs. candidate, or two repeats to
verify stability):

```powershell
Remove-Item Env:VICTORY_INPUT_RECORD -ErrorAction SilentlyContinue
$env:VICTORY_INPUT_REPLAY      = 'D:\victory-bench\scripts\route.jsonl'
$env:VICTORY_BENCH_WARMUP      = '30'
$env:VICTORY_BENCH_EXIT_FRAMES = '180'

$env:VICTORY_BENCH_OUT = 'D:\victory-bench\runs\replay_1.json'
& 'D:\victory-bench\launch-base.cmd'

$env:VICTORY_BENCH_OUT = 'D:\victory-bench\runs\replay_2.json'
& 'D:\victory-bench\launch-cand.cmd'
```

Or run automated interleaved `base -> cand -> base` replays with all six
anti-reward-hacking gates via `xtask`:

```powershell
cargo xtask bench-loop --config D:\victory-bench\playlist.toml --mode ab --runs 2
```

---

## 1. Runtime Environment Variables (Opt-In)

### GPU Benchmark Telemetry (`victory-gpu::bench`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `VICTORY_BENCH_OUT` | unset | Absolute path outside the repo where structured JSON metrics are written atomically (`<path>.tmp.<pid>.<seq>` + rename). |
| `VICTORY_BENCH_WARMUP` | `30` | Number of initial world frames (`>= 500` draws latch) excluded from statistics. |
| `VICTORY_BENCH_EXIT_FRAMES` | unset | Flush the JSON report and exit cleanly with code `0` once `warmup + N` world frames complete. |
| `VICTORY_BENCH_CHECK_EVERY` | unset | Emit periodic sanity checkpoints every `K` measured world frames. |

`victory_gpu::bench::world_frame()` stays `0` through menus and loading screens
and latches to `1, 2, 3, ...` starting at the first presented frame with at
least `500` draws. On the queued presentation path, the display worker
advances the clock after it handles the frame. Use a Windows diagnostics
build for black-frame probes. The probes sample supported 8-bit surfaces;
for other formats, configure checkpoint PNGs to check visual parity.

### Deterministic Input Recording & Replay (`victory-hle::modules::gameinput_script`)

| Variable | Default | Purpose |
| --- | --- | --- |
| `VICTORY_INPUT_RECORD` | unset | Absolute path outside the repo (`.jsonl`) recording `PadState` and held virtual keys on every state change, tagged with `wframe` and `ms`. |
| `VICTORY_INPUT_REPLAY` | unset | Path to a `.jsonl` input script replayed deterministically by `world_frame`. Holds neutral input while `world_frame == 0` and reports its BLAKE3 digest in the `input_script_sha256` compatibility field. |

---

## 2. Recording a Traversal Input Script

Run your title once with `VICTORY_INPUT_RECORD` pointing outside the repo and
walk or drive a representative route:

```powershell
$env:VICTORY_INPUT_RECORD = 'D:\victory-bench\scripts\traversal_route.jsonl'
$env:VICTORY_BENCH_WARMUP = '30'
$env:VICTORY_BENCH_EXIT_FRAMES = '180'
$env:VICTORY_BENCH_OUT = 'D:\victory-bench\runs\record_check.json'
& 'D:\victory-bench\launch-title.cmd'
```

You can also author declarative world-frame range scripts by hand (or mix range
lines with recorded JSONL lines):

```json
{"from":0,"to":120,"ly":1.0,"rt":0.8}
{"from":121,"to":180,"lx":0.35,"ly":1.0,"rt":0.7,"buttons":4}
```

Because replay is keyed to `world_frame` (not wall-clock milliseconds), the
exact same traversal happens regardless of whether the build runs at 10 FPS or
60 FPS, and loading screens (`world_frame == 0`) are waited through
automatically.

### Delivery contract

- **Loading screens (`world_frame == 0`)**: replay holds neutral input and neither advances playback state nor consumes pending pulses.
- **Recorded traces (`wframe`)**: step-function holds — the latest recorded entry with `entry.wframe <= world_frame` stays active until the next recorded change.
- **Multi-frame declarative spans (`to > from`)**: active strictly while `from <= world_frame <= to`; spans never latch past `to` if all frames in `[from, to]` are skipped between polls.
- **One-frame declarative pulses (`from == to`, `from >= 1`)**: on the live stateful playback path (`ReplaySource` / `ReplayPlayback`), if no poll occurs during `world_frame == from`, an unpolled pulse in the zero-poll gap (`last_polled_wframe < from <= world_frame`) latches and delivers once on the first eligible polled world frame (`world_frame >= from`), staying consistent across repeated polls and pad/key reads within that single world frame before releasing permanently on the next world frame. Stateless point queries (`InputScript::sample`) evaluate exact `from <= world_frame <= to`.

---

## 3. Writing a Playlist Config

Create a playlist TOML file **outside the repo**, e.g.
`D:\victory-bench\playlist.toml`:

```toml
base_cmd = "D:/victory-bench/run-base.cmd"
runner_cmd = "D:/victory-bench/run-cand.cmd"
experiments = "D:/victory-bench/experiments.jsonl"
vram_cap_mb = 9200.0
ram_cap_mb = 18000.0

[[scenes]]
name = "dense_traversal"
setup_cmd = "D:/victory-bench/restore-save-dense.cmd"
input_script = "D:/victory-bench/scripts/traversal_route.jsonl"
warmup = 30
frames = 150
check_every = 15

[[scenes]]
name = "interior_lighting"
setup_cmd = "D:/victory-bench/restore-save-interior.cmd"
input_script = "D:/victory-bench/scripts/interior_walk.jsonl"
warmup = 30
frames = 120
check_every = 15
```

`xtask bench-loop` injects `VICTORY_BENCH_OUT`, `VICTORY_BENCH_WARMUP`,
`VICTORY_BENCH_EXIT_FRAMES`, `VICTORY_BENCH_CHECK_EVERY`, and
`VICTORY_INPUT_REPLAY` into every runner invocation (and also expands `{bench_out}`,
`{input_script}`, `{warmup}`, `{frames}`, `{role}`, `{scene}` placeholders if
present in `base_cmd`/`runner_cmd`).

---

## 4. Running Interleaved A/B Benchmarks

Compare baseline vs. candidate with interleaved `base -> cand -> base` runs (or
multiple rounds via `--runs N`, taking per-scene medians):

```powershell
cargo xtask bench-loop --config D:\victory-bench\playlist.toml --mode ab --runs 2
```

For pre-built binaries where you want to skip the workspace build/test/clippy
gate during an ad-hoc A/B check, pass `--skip-build-gate`.

---

## 5. Running an Autonomous Optimization Loop

Point `--mode loop` at any CLI agent that accepts a prompt on stdin and edits
the working tree in place:

```powershell
# Claude Code
cargo xtask bench-loop --config D:\victory-bench\playlist.toml --mode loop `
  --iterations 10 --agent-cmd "claude -p"

# Codex CLI
cargo xtask bench-loop --config D:\victory-bench\playlist.toml --mode loop `
  --iterations 10 --agent-cmd "codex exec -"

# Gemini CLI
cargo xtask bench-loop --config D:\victory-bench\playlist.toml --mode loop `
  --iterations 10 --agent-cmd "gemini -p"
```

Optional `--fast-cmd "<cmd>"` runs a fast offline microbenchmark or unit check
before committing to the expensive full scene run. Every iteration appends a
structured record (hypothesis, git SHA, diff stat, top `framelog` bottleneck
stages, per-gate verdicts, `keep`/`revert`) to `experiments.jsonl` and feeds
recent dead ends back into the next prompt so the agent does not repeat failed
attempts.

---

## 6. Structured Bench JSON Schema

Each run writes:

- `world_frames`, `warmup_frames`
- `frame_ms`: `avg`, `median`, `p95`, `p99`, `low_5pct_fps`, `low_1pct_fps`
- `draws`: `avg`, `min`, `max`
- `device_removed` (`bool`), `zero_draw_world_frames`, `black_frames`
- `memory`: `vram_used_peak_mb`, `vram_budget_mb`, `ram_private_commit_peak_mb`, `ram_working_set_peak_mb`
- `framelog` per-frame averages: `prepare_ms`, `card_wait_ms`, `card_idle_ms`, `copy_ms`, `copyback_ms`, `remap_ms`, `refresh_ms`, `pipeline_wait_ms`, `guard_faults`, `card_sync_ms`, `card_check_ms`, `card_lists_ms`
- `input_script_sha256`

---

## 7. Six Hard Anti-Reward-Hacking Gates

A candidate is kept **only** if every single gate passes:

1. **Build, Tests, Policy & Clippy** — workspace builds cleanly, `cargo test` passes, `cargo xtask policy` passes, and `cargo clippy --workspace --all-targets -- -D warnings -A deprecated` is clean.
2. **Stability & Watchdog** — exit code `0`, completion within `--timeout-secs`, valid JSON report, and `device_removed == false`.
3. **Draw-Count Parity** — candidate `draws.avg` stays within `±15%` (configurable via `--max-draw-drift-pct`) of baseline on every scene.
4. **Visual Sanity** — `zero_draw_world_frames == 0`, `black_frames == 0`, and (when checkpoint PNGs are configured) matching dimensions, non-black luminance, and mean luminance parity within `--max-luma-diff`.
5. **Warmup & Interleaved Median** — warmup frames excluded and full frame counts met across every baseline and candidate repeat (`min(warmup_frames) >= warmup`, `min(world_frames) >= max(1, frames)`, `frame_ms.median > 0`, matching `input_script_sha256`), evaluated over interleaved `base -> cand -> base` rounds using per-scene medians.
6. **Composite Gain & Non-Regression** — aggregate median `frame_ms` improves by at least `--min-gain-pct` (default `2.0%`), with non-regression on `low_1pct_fps`, `vram_used_peak_mb`, and `ram_private_commit_peak_mb` caps.
