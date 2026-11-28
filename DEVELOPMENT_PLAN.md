# Victory-fork — Development Plan

Source: `https://github.com/fxceless/victory-fork.git`, `main` as of **27 Nov 2026**.
This plan is derived from reading **all 279 files (~8.66 MB, ~220,800 lines)** in the
snapshot: every README, all 9 `docs/*.md`, workspace + per-crate `Cargo.toml`s,
`apps/victory` (12 files), all crates, `scripts/`, `xtask/`, CI, toolchain, tests
and fixtures. Nothing was skipped; the per-file inventory is summarised in §2.

> Scope note: this repository contains **no game files, code or assets** (see
> `docs/asset-policy.md` + §9). All work builds on the user's own dump of the
> title's Xbox Series build `v1.0.2.1` (`game_scarlett_master.exe`) outside the repo.

## 0. Where the fork stands (read first)

- **Base.** Since 26 Nov 2026 the tree is
  [VIctory-NVIDIA](https://github.com/trustno1-coder/VIctory-NVIDIA) at `959dcaf`
  (which itself sits on [foxhound](https://github.com/readmet3xt-lab/foxhound)
  `02fc18a` → [VIctory](https://github.com/Marvisak/VIctory) + ~300 measured runs
  across 10+ forks). `main` and `raytracing` branches are identical.
- **This fork's delta on top** (the only reason it exists):
  1. Ray-tracing resolve-pipeline queuing fix (`victory-gpu/src/pipeline/make.rs`)
     + 512 MB page budget instead of a 2,048-page cap (`memory.rs`) + ≥50 ms
     stall log line. Before: ~280 ms/frame while RT programs built.
  2. `wait_for_slots` stall fix (`victory-gpu/src/host.rs`, `VICTORY_DYNAMIC_WAIT=1`
     by default): busy scenes −13 ms/frame, −48 ms/frame with RT on.
  3. Launcher default `VICTORY_BVH_BUILDER=1` (title's own RT on). `play --set
     VICTORY_BVH_BUILDER=0` restores the ambient estimate.
  4. Wording de-branding ("the title"), `play-framegen.ps1`, `TITLE_*` env names.
- **Measured (RTX 4070, 1920×1080 scene):** ~11 fps at 5–9k draws/frame, ~8 fps at
  9–12k (vs ~18–14 / 12.5 without RT). First visit to a place stutters (driver
  builds RT programs; cached next start). Water lighting wrong on 32-lane cards.
- **NVIDIA path (RTX 5070 12 GB, Series X, 2560×1440, DLSS Quality):** 26–30 fps
  rendered in opening rooms, 2×–6× frame-gen on top (Streamline count). Menus /
  video / saves / F1 menu / F3 panel / open water / boat ride work.
- **Known-wrong:** RT slow + incomplete (§6); frame-gen needs VRAM headroom + focus
  + ~1 s; first build run stutters; `SAVE AND RESTART NOW` can orphan a process;
  heat-haze pass + decal-heap defrag left out; cutscene hitches; 30 fps title cap
  unless `-Fps 60` experiment lands.

## 1. Architecture (from `docs/architecture.md`)

```
user files (outside repo)              VIctory
game assets ───────────────► victory-pipeline + victory-game
                             (victory-formats, victory-shader)
                                        ▼
                             cache_dir: converted data
                                        ▼
game exe ─► victory-loader ──► title code (native x86-64)
            prepare/map/bind      │ imports
                                  ▼
                             victory-hle ── host Windows (shared APIs)
                                │ D3D12.X
                                ▼
                             victory-gpu ─► host D3D12
                                │ shaders at pipeline creation
                                ▼
                             victory-shader ◄─► cache_dir
```

- Shaders translate at pipeline creation from title-passed bytes (`victory translate`
  only warms the cache). Import resolution order: our `victory-hle::modules` →
  host pass-through per `HostPolicy` → title-shipped DLL → generated stub
  (`--unimplemented trap|zero`). `LoadLibrary*`/`GetProcAddress`/`QueryApiImpl`
  rules, virtual-drive path rewrite (`files`), TLS slot fix, crash reporter,
  `victory census` — see `docs/architecture.md` §§ Import resolution.
- Shader pipeline: `ir::build` (SSA, 1 IR / machine insn) → `structure` (EXEC→
  per-lane guards) → `resource` (descriptor + raw-address tracing, waterfall
  `v_readfirstlane_b32` handling) → `emit` (DXIL).

## 2. Complete file inventory (nothing omitted)

Counts from the snapshot (lines include blanks/comments):

| Area | Files | ~Lines | Contents |
|---|---|---|---|
| repo root | 17 | 11,310 | `README.md` (382), `Cargo.toml`/`Cargo.lock`, `LICENSE` (GPL-3.0-or-later, 675), `rust-toolchain.toml` (pinned), `rustfmt.toml`, `.cargo/config.toml`, `.gitignore`, `.gitattributes`, `.githooks/pre-commit`, `.github/workflows/ci.yml`, `.victory-root`, `victory.example.toml`, `Get-NVIDIA-DLSS.cmd`, `Play-VIctory.cmd`, `PLAY_FRAMEGEN.cmd` |
| `apps/victory` | 12 | 4,246 | `main.rs` (CLI), `run.rs` (682), `translate.rs` (964), `replay.rs` (508), `play.rs` (501), `census.rs` (335), `inspect.rs` (155), `shader.rs` (256), `warm.rs` (211), `native_1440p.rs` (215), `Cargo.toml`, `tests/census.rs` |
| `crates/victory-core` | 5 | 517 | `config.rs` (env→file→default, outside-repo guard), `policy.rs`, `logging.rs`, `lib.rs` |
| `crates/victory-pe` | 5 | 1,655 | `image.rs` (PE32+ parse), `implib.rs`, `layout.rs`, `lib.rs` |
| `crates/victory-loader` | 18 | 5,681 | `prepare.rs`, `windows.rs`, `threads.rs`, `tls.rs`, `thunks.rs`, `crash.rs` (module+rva + unwind → `crashes.log`), `profile.rs`, `publish.rs`, `resolve.rs`, `stacks.rs`, `frame_rate.rs`, `console_insn*.rs` (Xbox insn), `write_sites.rs`, `lib.rs`, tests `load_probe.rs`, `title_dlls.rs` |
| `crates/victory-hle` | 39 | 26,245 | `host.rs`, `runtime.rs`, `registry.rs`, `image_watch.rs`, `modules/mod.rs` + `kernel`, `files` (virtual drives), `xmem` (157 KB, largest), `xasync`, `xgame*`, `xgamesave`, `xgameui`, `xg`, `xgameruntime`, `xpackage`, `xstore`, `xsystem`, `xuser`, `xnetworking`, `net`, `rpcrt4`, `storage`, `lifecycle`, `media`, `speech` (silent stub), `dsp` (zeros), `apu`, `audio`+`audio/capture`, `dstorage` (97 KB), `display`, `bink`, `d3d12x`, `fidelityfx` (DLSS interpose), `gameinput`+`dualsense`, `streamline` (check path), tests `no_dll_planting.rs` |
| `crates/victory-gpu` (largest) | 105 | 111,926 | see §2.1 |
| `crates/victory-shader` | 42 | 50,432 | see §2.2 |
| `crates/victory-formats` | 8 | 5,580 | `texture.rs` (59 KB tiling/detiling), `bcpack.rs` (62 KB) + `bcpack-tables.bin` (78 KB exception), `xma.rs` (70 KB)+`tables.rs`, `lib.rs`, bench, `Cargo.toml` |
| `crates/victory-pipeline` | 3 | 282 | `cache.rs` (content-addressed cache), `lib.rs` (converter trait) |
| `crates/victory-game` | 3 | 261 | `sgd5.rs` (title formats/build ids/hooks), `lib.rs` |
| `docs` | 9 | ~1,459 | `README-foxhound.md`, `README-fxceless.md`, `architecture.md`, `asset-policy.md`, `bug-journal.md`, `dlss.md`, `findings.md`, `fork-ray-tracing.md`, `replay.md` |
| `scripts` | 7 | 667 | `play-foxhound.ps1` (151, launcher canonical), `play-framegen.ps1` (147), `play.ps1` (39), `play-nvidia.ps1` (alias, 4), `get-nvidia-sdks.ps1` (137), `fullscreen.ps1` (37), `faultbook.py` (152) |
| `xtask` | 3 | 375 | `main.rs`, `policy.rs` (300, allowlist enforcer — see §9) |
| `tests/fixtures/hle-probe` | 4 | 148 | stand-in "title" (`lib.rs`, `build.rs`, `Cargo.toml`) for loader tests |
| GPU tests/examples | 5 | ~3,300 | `tests/replay.rs` (1,860), `tests/replay_big.rs` (399), `examples/capture_probe.rs`, `examples/lut3d_probe.rs`, `build.rs` |

### 2.1 `victory-gpu` — every module (105 files)

Core objects: `lib.rs`, `device.rs`, `dxgi.rs`, `resource.rs` (190 KB), `descriptor.rs`
(84 KB) + `descriptor/host.rs` (313 KB), `root.rs`, `signature.rs`, `sampler.rs`,
`query.rs`, `fence.rs`, `object.rs`, `console.rs`, `cores.rs`, `switch.rs`, `frame.rs`,
`external.rs`. Lists/queues: `list.rs` (98 KB) + `list/thin.rs`, `queue.rs` +
`queue/behind.rs`, `ab.rs`. Submission/host: `host.rs` (530 KB — the hot path) +
`host/{aside,awake,card_steps,font,indirects,menu,panel,place,send,spans,stamps,
torn,traffic,two_queues,upcheck}.rs`, `draw.rs` (274 KB), `compose.rs`,
`streamline.rs` + `streamline/{abi,plan}.rs`, `captur`e paths.
Memory/textures: `memory.rs` (71 KB — title-by-address), `shadow.rs` (267 KB —
write-watch/guards), `watch.rs` (234 KB), `inplace.rs`, `tiles.rs` (73 KB),
`texture.rs` (124 KB) + `texture/ahead.rs`, `texback.rs`, `residency.rs`,
`pool.rs`, `heap_marks.rs`, `safecopy.rs`, `upload.rs`, `copystat.rs` (diag),
`land.rs` + `land/tests.rs`, `port.rs`, `picked.rs`, `primitive.rs`.
Pipelines/shaders: `pipeline.rs` + `pipeline/{host,known,library,make}.rs`
(`make.rs` carries this fork's RT-queue fix), `shader_cache.rs`, `indirect.rs`
(189 KB) + `indirect/{batch,bench,merge}.rs`. Capture/replay/diag:
`capture.rs`, `replay/{mod,record,snap,wire,states,pso,hook,edit,gpu}.rs`,
`replay/play.rs` (140 KB) + `play/{diverge,look,merge,pixel,upcopies}.rs`,
`timeline.rs` + `timeline/{hooks,record,report}.rs`, `book.rs`, `faultbook.rs`,
`popwatch.rs`, `tdr.rs`, `ab.rs`, `ambient.rs` (36 KB — light estimate),
`titlewrite.rs` + `titlewrite/decode.rs`. `build.rs` wires Streamline stubs.

### 2.2 `victory-shader` — every module (42 files)

`lib.rs`, `ir.rs` (80 KB), `resource.rs` (166 KB — descriptor + raw-address
tracing), `structure.rs` (71 KB), `native.rs` (46 KB), `io.rs` (34 KB),
`spill.rs`, `bvh.rs` (30 KB — `image_bvh*_intersect_ray` reference), `dxbc.rs`,
`dxil.rs`, `emit/mod.rs` (138 KB) + `emit/{lower.rs` (612 KB — the translator
core), `host.rs`, `shader.rs`, `bitcode.rs` + `bitcode/{census,opt}.rs`,
`container.rs`, `validator.rs`, `sampler.rs`, `rect.rs`, `intersect.rs`,
`threading.rs`, `wave.rs`, `own_bits.rs`, `filter3d.rs`, `sp3`/`gfx10/{mod,
fields,opcodes,sp3}.rs` + all `*_tests.rs` (`diff`, `valu` 90 KB, `wave64`,
`waterfall`, `permlane`, `probe`, `cube_lod`, `filter3d`, `size`, `intersect`),
`benches/guard_runs.rs`, `build.rs`.

### 2.3 CLI, scripts, entry points

`victory` commands (`apps/victory/src/main.rs`): `inspect`, `census`,
`lib-exports`, `shader-info`, `translate` (`--validate --pipelines --list --jobs
--warm`), `shaders`, `warm`, `play` (bare double-click = play), `get-dlss`,
`run --execute --unimplemented trap|zero --strict --console series-x|s`,
`replay` (diag only), `config`, `upscaler-check [--device --present]`.
Launchers: `scripts/play-foxhound.ps1` (canonical flags: `-FrameGen/-FrameGenX`,
`-Dlss`, `-Fps`, `-SeriesS`, `-Res`, `-Build diag`, `-More`), `play-framegen.ps1`
(de-branded), `play-nvidia.ps1` (alias), `play.ps1`, `Get-NVIDIA-DLSS.cmd` /
`get-nvidia-sdks.ps1` (70 MB NVIDIA fetch, git-ignored `nvidia-sdk/`),
`fullscreen.ps1`, `faultbook.py`, `Play-VIctory.cmd`, `PLAY_FRAMEGEN.cmd`.
`victory.example.toml` → `victory.toml` (all paths must be outside repo).

### 2.4 Environment switches (all 430 `VICTORY_*` read)

Full list is in the repo (grep `VICTORY_[A-Z0-9_]+`); families that matter:

- Paths/logging: `GAME_DIR/CACHE_DIR/DATA_DIR/TEST_TITLES/CONFIG/LOG/LOG_FILE/
  SKIP_CPU_CHECK/WINDOW_MAX`.
- Console/GPU model: `HOST_WAVE/HOST_MAX_WAVE/HOST_RDNA/RDNA_FLOAT/CONSOLE_DXC/
  PC_DXC/DXIL/D3D12_{DEBUG,GBV,DRED,WARP,BACKGROUND}`.
- Submission: `THIN/ASYNC_SUBMIT/ASYNC_QUEUE/RELAY/LATE_WAIT/LATE_WAITS/
  LATE_WAIT_{LISTS,WRITES}/TWO_QUEUES/BURY_LAG/QUEUE_{BATCH,WAIT,MARKS}/
  JOIN_{COPIES,LISTS,PARTS}/STRICT_JOINS/SUBMIT_{DEPTH,FRAMES}/DYNAMIC_{SLOTS,WAIT}`.
- Memory: `VIEW_GUARD/SOFT_VIEWS/MAPPED_BY_{PAGE,VIEWS}/MAP_{LEAN,RUNS,THREADS}/
  GUARD_{ASK,HOLES,HOT,REST}/INPLACE{,_MB}/PORT_CONSTANTS/WC_NO_BACK{,_MIN_KB}/
  COPY_{ASK,BESIDE,MODE,OVER_TITLE}/MEMORY_{CARD_MB,DUMP,MAP}/CARD_{BUFFERS{,_MB,
  _MIN_KB,_MIN_MB},ROOM_FOR_{BUFFERS,TARGETS{,_SETTLE_MS}},SPANS,STEPS,WINDOWS_MB,
  HASH_LEN,REHASH_RUNS,ACCOUNTING,ALL,AWAKE,STEPS}/TEXTURE_* (budget/stage/sweep/
  keep/guard/hot/pool/once/same), POOL_SMALL/WINDOW_{KB,TAILS}/SECTION_WINDOW/
  PLACED_{FILLER,ORDER}/LENT_HEAP/UPLOAD_{LOW,AHEAD}/LEVELS_ON_DEMAND/
  HELD_LEVELS/LEVELS_MIN_KB/KEEP_{FREED,SHOWN,SNAPSHOTS{,_MB},VIEWS}/
  FULL_CARD_PASS/REUPLOAD_WHEN_FULL/READ_BACK/TEXTURE_BACK`.
- Shader/translate: `TRANSLATOR/SHADER_{CENSUS,LOOP_LIMIT,OPT,SAVE,SOURCE_HASH,
  SWITCHES}/EMIT_DUMP/VIEW_{LIMIT,BYTES,COPY,EXTENT,INLINE,THREADS}/RAW_VIEWS/
  WAVE_{WIDE,DEEP_SKIPS,LOCAL,OWN_BITS,WHY}/WATER_LIGHT/HELPER_{OWN,PRIORITY}/
  FIRSTLANE_OWN/FIRST_HALF/HOST_WAVE/RDNA_FLOAT`.
- Ray tracing / light: `BVH_{BUILDER,BOUNDS,FENCES,SIZES_WAIT,TRACE,QUARANTINE}/
  AMBIENT{,_AIR,_ONCE,_SHADE,_SPECULAR{,_ALPHA}}/ACTIVATE_RAYS/RT_RESOLVE`.
- DLSS/FrameGen: `DLSS{,_MODE,_PRESET,_JITTER,_MV,_REACTIVE}/FSR_PATH/FRAMEGEN{,
  _X,_CYCLE,_HUDLESS}/STREAMLINE_{DIR,LOG,PRESENT}/NATIVE_1440P/RENDER_1080P/
  DISPLAY_{MODE,PROFILE_TEST_IMAGE,REFRESH}/FULLSCREEN{,_RES}`.
- HLE/audio/input: `XMA_{DUMP,STREAMS,REFERENCE}/AUDIO_CAPTURE{,_OF}/DSTORAGE_*/
  SKIP_BINK/IME_ENGLISH/ONE_PAD/DUALSENSE/MOUSE_{CAPTURE,WARP}/KEYS{,_FILE}/
  MENU_* (F1 menu state), TITLE_FPS/FRAME_INTERVAL_{MS,US}/FRAMEGEN_X.
- Diag/replay: `RECORD{,_ALWAYS,_AT,_FRAMES}/CAPTURE{,_FRAMES,_WORLD{,_ALSO}}/
  DUMP_TARGETS/TIMELINE{,_AT,_FRAMES}/REPLAY_{GBV,PEEK}/PIPELINE_{BOOK,WAIT_MS,
  WAIT_SHARED,WORKERS,SLOW,LOWEST}/WARM_SECONDS/SHADER_SAVE/GPU_TIMING/
  BREADCRUMBS/CRASH_SAVE/STACKS/WATCH/POKE/KEYS/FAULT_BOOK{,_AT,_UNLOCKED}/
  UPCOPY_CHECK/CAPTURE_WORLD*/AB (live toggles).
- Every switch is documented where it is read; launcher `play-*.ps1` + `play.rs`
  `SETTINGS`/`KEPT` tables are the user-facing subset (F1 menu persists to
  `victory-settings.txt`).

## 3. Quality gates (non-negotiable)

1. **Asset policy is rule #1** (`docs/asset-policy.md`, `xtask/src/policy.rs`,
   `.githooks/pre-commit`, CI `policy` job): only small text files we wrote;
   ≤640 KiB/file; allowlisted extensions; no NUL bytes, no magic
   (`MZ/DXBC/DDS/RIFF/LLVM-bitcode/PDB/ar/zip/7z/cab/MSI`), no >512-char
   hex/base64 runs or `\x`/`0x` literal runs. One exception:
   `victory-formats/src/bcpack-tables.bin` (78 KB format tables, owner's
   2026-11-23 decision). Tests needing real files take env paths and skip when
   unset; `ensure_outside_repo` at runtime; `victory.toml` paths outside repo.
   Violation → `filter-repo --path <file> --invert-paths` + force-push + re-clone.
2. `cargo xtask install-hooks`, `cargo test --workspace` (no game data),
   `cargo build --release -p victory`, diag build for GPU work
   (`--profile diag --features diagnostics`), `cargo xtask policy`, rustfmt,
   workspace lints (`unsafe_op_in_unsafe_fn`/`unused_must_use` deny; 11
   `undocumented_unsafe_blocks` + 8 style allows are tech debt to burn down).
3. Shader changes: `victory translate --validate --pipelines` + `victory-shader`
   lib suite (281 passed/4 ignored baseline) + `victory-gpu` suite (335 passed/34
   ignored baseline) + targeted `--validate` on VFX/particle/eye shaders before
   claiming coverage.

## 4. Known gaps (from docs + `shortcut:`/`TODO` scan, 54 files)

- RT: BVH overlap (8/65 good in light-10), size-read races (`BVH_SIZES_WAIT`
  costs ~20 ms, still stale), 5 shadow-mask trace programs untranslated, lost
  accesses (read=0/drop store), charts in §fork-ray-tracing Known gaps; water
  light excluded on 32-lane (`WATER_LIGHT=1` re-enables); `CopyForBVHUpdate`
  13.3 ms + traces 24 ms in dorm.
- GPU: helper-lane wave reads (square holes/dashes on AMD), indirect dispatch
  cut at 65,535 groups, `0x470` ignored, bottom-of-pipe waits hold whole lists,
  aliased-resource state = last-use, upload-heap replay divergence, multisample /
  CPU-visible / >256 MB levels not captured.
- HLE: speech silent, DSP no conv/block (zeros), reverb silent, trigger rumble
  dropped, HDR not done, screenshots fail as unavailable, overlapped reads
  partial, net offline-by-design, XMA/Opus unverified vs title.
- Perf debt: `VICTORY_TRANSLATOR` 570 KB source; `lower.rs` 612 KB + `host.rs`
  530 KB + `draw.rs` 274 KB + `shadow.rs` 267 KB + `watch.rs` 234 KB are the
  review hotspots; 8 style lints + 11 missing SAFETY comments allowed and owed.

## 5. Roadmap

### Phase 0 — Hygiene (1–2 days, no behaviour change)
- `cargo xtask install-hooks` everywhere; CI green; record baselines
  (`translate --validate --pipelines` counts, both test suites, `census`).
- Arm `replay --compare/--diverge/--pixel/--upcopies/--flow` + `warm` as the
  standard repro loop; document capture folder layout per `docs/replay.md`.

### Phase 1 — Correctness first (RT + water + tests)
1. BVH size-race: fence the compacted-size slot the title polls (wait only that
   word, not the submission); add `BVH_TRACE`-driven test asserting neighbour
   structures intact after overlap runs.
2. Bound + quarantine on by default for one milestone; measure dorm `CopyForBVHUpdate`.
3. Water light on 32-lane: narrow the exclusion to the two failing programs
   behind `WATER_LIGHT=1`; keep darker/bluer water until probe tests pass.
4. Land the 5 missing shadow-mask trace programs (computed-index descriptor →
   per-slot address table, as `replay --slots` shows).
5. Close `todo!`/`unimplemented!` in title-reachable HLE paths to logged stubs.

### Phase 2 — Frame time (submit thread → card)
Order from `docs/findings.md` (prepare 48 ms → card 41 ms → 36k list calls):
1. Keep `VICTORY_DYNAMIC_WAIT=1` default; add `dynamic descriptor slots` line to
   F3 panel + assertion that `settle_for_fence` never holds the submit lock.
2. Split one submission over two threads; ask next submission while sending;
   `VIEW_*`/`WIDE_*`/`THIN_THREADS` autotune behind `WIDE_REFRESH` pattern.
3. Card copies of big title buffers on by default where `replay --buffers vram`
   wins; AVX2 title-bytes-under-shader-writes path as default.
4. `warm` + `PIPELINE_BOOK` into first-run UX (progress %, per-driver cache dir);
   cut >200 ms frames to <5% of a session.

### Phase 3 — Picture (DLSS/FG/ambient)
- Series X HUD-less colour copy: tag interface plane properly instead of
  shader-hash heuristic; add `presentCommon() was not observed` regression test
  behind `STREAMLINE_PRESENT=1`.
- `AMBIENT`/`AMBIENT_SPECULAR`/`AMBIENT_SHADE` auto-tune from `CacheScreenSpace
  Radiance` stats; keep `constant:` escape hatch; scope `AMBIENT_AIR` per-profile.
- HDR swap-chain + 64-lane-wave correctness behind explicit switches with
  `replay --compare` gates.

### Phase 4 — Cleanup + upstream
- Burn down the 8 `allow`ed clippy lints + 11 missing SAFETY comments; split
  `lower.rs`/`host.rs`/`draw.rs` review units; `cargo xtask policy` stays green.
- Rebase onto upstream `VIctory-NVIDIA`/`foxhound`/`VIctory` monthly; keep
  `TITLE_*` de-branding + `play-framegen.ps1` + RT queue/budget + dynamic-wait
  as the documented delta; never carry game data across the rebase.

## 6. Verification matrix (every change)

| Change type | Must run |
|---|---|
| Any Rust | `cargo test --workspace`, `cargo xtask policy`, fmt + clippy, `victory config` |
| Loader/HLE | `inspect/census`, `run` without `--execute`, `load_probe` + `hle-probe` fixture |
| Shader | `translate --validate --pipelines --list`, shader lib suite, `warm` |
| GPU/host | diag build + `replay --compare`, `--timing detailed`, `--pixel`, `--diverge`, `--upcopies`, F3 numbers |
| DLSS/FG | `upscaler-check [--device --present]`, 10-point checklist in `docs/dlss.md` § The first run |
| Perf claim | same save + same cache, ≥15 ≥1k-draw intervals, median + p95 + peak commit + VRAM, note scene mismatch |
| Commit | hook `policy` + CI; no `target/`, no `nvidia-sdk/`, no `victory.toml`, no logs/captures |

## 7. Risks

- Scene-mismatched benchmarks (past comparisons were directional only) → pin
  saves + caches + caption intervals per `bug-journal.md` method.
- Driver compiles per program path → `warm` dir discipline; don't move release
  binaries between folders mid-measurement.
- 12 GB VRAM + 32 GB RAM floor; commit headroom <2 GiB invalidates runs.
- RT pages re-made when evicted (by design) → watch `title memory by address`
  churn before raising the 512 MB budget.
- Any `state` outside `victory.toml`-configured dirs → policy violation; fix by
  history rewrite, not a follow-up commit.

## 8. GUI development — full control (F1 menu, F3 panel, FPS line, compositor, window, input)

This is the complete owner-side UI stack. There is no external UI toolkit:
everything is a GPU-resident overlay (`victory-gpu/src/host/{menu,panel,font}.rs`),
a two-plane compositor (`compose.rs`), window/input policy
(`victory-hle/src/modules/{lifecycle,gameinput,display}.rs` + `victory-gpu/src/dxgi.rs`),
launcher defaults (`apps/victory/src/play.rs`) and scripts. All of it works in
every build (menu/FPS line) or every build with `VICTORY_PANEL=1` (panel); nothing
requires game data and everything is testable headless except the final blit.

### 8.1 Surfaces and who owns them

| Surface | File | When | Where on screen | Source of truth |
|---|---|---|---|---|
| F1 settings menu | `host/menu.rs` (570 lines) | every build, `F1` toggles | centred black box, 60 cols × 20 rows + header/help/footer | `ITEMS` + `ACTIONS`, `values[]` vs `started[]` |
| F3 stats panel | `host/panel.rs` (455 lines) | every build when `VICTORY_PANEL=1` or F3/menu `VICTORY_PANEL=1` | top-left black box, 40 cols × 10 lines | `Counters` + `process_memory()` + `host.video_memory()` |
| FPS line | `host/menu.rs` `fps_line()` | every build unless `VICTORY_FPS_LINE=0`; hidden while panel shown | top-left single line | `METER_FPS_TENTHS` + `streamline::shown_per_frame()` |
| 5×7 font | `host/font.rs` (79 lines) | shared | `glyph(c)->[u8;7]`, `colours(fmt)` | const fn, unit-tested |
| Scene+interface composite | `compose.rs` (325 lines) | Series S two-plane present | fullscreen triangle, premultiplied `saturate(o.rgb+s.rgb*(1-saturate(o.a)))` | `Composer::record()` per presenter slot |
| Title window | `lifecycle.rs` `fit/hook/fullscreen/english_input/capture_mouse` | at `CreateWindowExA/W` | fitted Centred `WS_POPUP` optional | `VICTORY_WINDOW_MAX/FULLSCREEN{,_RES}/MOUSE_*/IME_ENGLISH` |
| Reported display | `dxgi.rs` `display_size/refresh_rates/modes` | at title DXGI queries | virtual 3840×2160 @60 Hz (+ optional 61–240 Hz) | `VICTORY_DISPLAY_MODE/REFRESH` |
| Input devices | `gameinput.rs` + `dualsense.rs` | always; 8 ms poll thread | 4 host pads + fallback + keyboard + mouse | `FALLBACK_BUTTONS`, `VICTORY_ONE_PAD/DUALSENSE` |

Presenter rule (read it before touching UI): the menu/panel are **copied onto the
swap-chain buffer behind the title's picture** via `CopyTextureRegion`, **never**
drawn by the title. The diag frame recorder explicitly excludes them
(`replay::record::unrecorded()` guard in both `draw()`s), so captures and
`replay.png` see the title only. Screenshots by the user *do* carry them — that
is how numbers ship in bug reports.

### 8.2 Full option catalog (GUI + window + input)

Launcher-facing (`VICTORY_MENU_*`, persisted to `%LOCALAPPDATA%\victory\victory-settings.txt`
via `VICTORY_MENU_FILE`; `play.rs::parse_menu/settings()`):

- `VICTORY_MENU_DLSS` = `quality|balanced|performance|dlaa|off` (default `quality`).
- `VICTORY_MENU_FRAMEGEN` = `0|2|3|4|5|6` (`0` = off; `1` from old files means `2`). Only honoured when Streamline dir found; else DLSS+FG keys are omitted entirely.
- `VICTORY_MENU_FPS` = `1000|120|60|30` → `VICTORY_FRAME_INTERVAL_US` (`1e6/fps`).
- `VICTORY_MENU_CONSOLE` = `series-x|series-s` (Series S skips `NATIVE_1440P/RENDER_1080P`).
- `VICTORY_MENU_RES` = `1440|1080to1440|4k` → `VICTORY_NATIVE_1440P=1` / `VICTORY_RENDER_1080P=1` / stock.
- `VICTORY_MENU_FILE/PLAY/ARGS/OPEN` = plumbing: settings path, launcher path, relaunch args, `=1` shows menu from first frame (for headless screenshots).

In-menu items (`ITEMS`, `host/menu.rs:59-88` — first value = launcher default):

- `VICTORY_FPS_LINE` `1|0` **LIVE** — top line on/off at once.
- `VICTORY_PANEL` `0|1` **LIVE** — F3 panel on/off at once.
- `VICTORY_MENU_FRAMEGEN` **LIVE only if run started framegen-ready** (`streamline::framegen_live()`; launcher sets `FRAMEGEN_X=0` ready-off whenever Streamline exists unless `--no-framegen-ready`).
- `VICTORY_FULLSCREEN` unset|`1` (restart) — `WS_POPUP`, no border, `FULLSCREEN_RES` or monitor size.
- `VICTORY_MOUSE_CAPTURE` unset|`1` (restart) — clip to client rect, hide cursor (balanced `ShowCursor` count), deltas to gameinput; `VICTORY_MOUSE_WARP=1` recentres every sample.
- `VICTORY_IME_ENGLISH` unset|`1` (restart) — force en-US layout (`0x04090409`) on window thread + `WM_SETFOCUS`.
- `VICTORY_DLSS_PRESET` `""|k|j|m|l` (restart) — DLSS model per mode.
- `VICTORY_AMBIENT` `0.8|1|0.5|""` , `VICTORY_AMBIENT_SPECULAR` `1|""`, `VICTORY_AMBIENT_AIR` `1.5|1|2|""` (restart; estimate off while `BVH_BUILDER=1`).
- `VICTORY_BVH_BUILDER` unset|`1` (restart) — title RT on/off.
- `VICTORY_WATER_LIGHT` `""|1` (restart) — off = no dark blocks; on = green + blocks on 32-lane.
- `VICTORY_LEAVE_OUT` `02384…|""` (restart) — heat-haze + decal-defrag skip list.
- `VICTORY_EARLY_DEPTH/WC_NO_BACK/SOFT_VIEWS/TEXTURE_NO_WAIT/PORT_CONSTANTS` `1|""` (restart).
- `VICTORY_RELAY/CARD_STEPS/INDIRECT_BATCH` `2|1|0` (restart).
- `VICTORY_CARD_BUFFERS_MIN_KB` `64|1024` (restart).
- Actions: `SAVE AND RESTART NOW` (save + `powershell Start-Sleep 3; & play …` in new console + `TerminateProcess(GetCurrentProcess)`, because `exit` + Streamline deadlocks), `SAVE FOR NEXT START`, `RESET ALL TO PLAY DEFAULTS` (deletes settings file), `CLOSE`.

Window/display/input env (outside menu, read once):

- `VICTORY_WINDOW_MAX=WxH` — shrink top-level window to fit work area, keep aspect, centre; render targets unchanged (swap chain stretches). Invalid → warn + ignore.
- `VICTORY_FULLSCREEN_RES=WxH` — with `FULLSCREEN=1`, that size instead of monitor size.
- `VICTORY_DISPLAY_MODE=WxH` (640–7680 × 360–4320, default 3840×2160) — size the console *reports* (a Series X on a 1080p TV lists 1080p); `VICTORY_DISPLAY_REFRESH=hz` (61–240) adds a second mode so the title unlocks 120 Hz intervals. Both `OnceLock`, logged once.
- `VICTORY_ONE_PAD=0` — expose each host pad separately instead of one merged always-there pad; `VICTORY_DUALSENSE` — DualSense path; keyboard stand-in always available: `WASD` left stick, `IJKL` right, arrows D-pad, `Enter/X/Y/Q/E/Z/C/F/H/Esc/Tab` = `A/B/X/Y/shoulders/triggers/clicks/Menu/View` (see `FALLBACK_BUTTONS` + README table); trigger rumble dropped (XInput has none).
- `VICTORY_MENU_OPEN=1` — open menu from frame 0 (screenshot/CI use).
- `VICTORY_AB` — live-toggle harness for any `switch::on` name in diag builds.

### 8.3 Controls — exact behaviour

- **Key thread:** `menu::keys()` spawns one `settings menu keys` thread at first `Menu::new()`; every 10 ms it checks `GetForegroundWindow()==ours` + `GetAsyncKeyState(VK_F1/UP/DOWN/LEFT/RIGHT/RETURN/ESCAPE)` edge-to-pushed and queues. 10 ms exists because a 33–50 ms frame would otherwise swallow taps. Panel F3 uses direct `GetAsyncKeyState` edge detect per `frame()` (no thread) plus the same foreground check.
- **While menu open:** `SETTINGS_MENU_OPEN=true`; `gameinput` zeroes mouse buttons and **discards motion deltas** (keeps virtual mouse connected at last position → no camera jump, no disconnect churn on close); title sees no keyboard. Pad still works — by design, so a controller-only user can close the menu.
- **Navigation:** `Up/Down` move `at` over `ITEMS.len()+ACTIONS.len()` (windowed scroll, `ROWS/2` lookahead); `Left/Right/Enter` on an item cycles `values[]` with `rem_euclid` wraparound (unknown env value shows itself uppercase at end); `Enter` on an action runs it; `F1` or `Esc` closes (reopening only via `F1`); `*` = changed-not-live, `*LIVE` = changed-live, `LIVE` = live-capable unchanged, footer explains unless `note` overrides (e.g. `NO SETTINGS FILE`, `COULD NOT SAVE`, `SAVED…`, `RESET…`).
- **Live apply:** `apply_live()` writes `std::env::{set,remove}_var` immediately (safe: those vars are re-read per use; env block + std locks held) then matches: `MENU_FRAMEGEN→streamline::set_framegen(n)` from next present, `FPS_LINE→fps_line_on()`, `PANEL→panel::shown()`. Everything else needs restart — the footer says so.

### 8.4 Rendering — pixel-exact spec to preserve

- **Font:** 5×7 bitmaps, `GLYPH=(5,7)`, `CELL=(6,9)` (1 px gap right + 2 px below), uppercase-normalised (`to_ascii_uppercase`), full `0-9A-Z./- ><*: _=()+ ,!?space` coverage; unknown → solid box. Tests assert every used char has art, fits `COLUMNS`, rows fit `GLYPH.0`, no two chars share art.
- **Geometry:** menu `COLUMNS=60`, `ROWS=20`, `LINES=ROWS+5`, `RIM=4`; panel `COLUMNS=40`, `LINES=10`, `RIM=3`. `size(scale)=((COLUMNS*CELL.0+2*RIM)*scale,(LINES*CELL.1+2*RIM)*scale)`. `scale=(height/540).max(1)` — both overlays scale with output height; all math in unscaled dots then expanded.
- **Raster:** `lit(lines,x,y)` → `painted(lines,scale,colours,pitch)`; `pitch=(width*bytes).next_multiple_of(256)`; `colours(fmt)` supports `R8G8B8A8/B8G8R8A8` (black `[0,0,0,255]`, white `[255;4]`), `R10G10B10A2` (`[0,0,0,0xC0]`/`[255;4]`), `R16G16B16A16_FLOAT` (halves 0/1); anything else → `None` → overlay silently skips that frame (`failed` only if `Buffers::new` returns `None`).
- **Buffers:** `UPLOADS=4` mapped upload buffers round-robined + one `DEFAULT`-heap card buffer; `write()` memcpys picture into `uploads[turn]`, sets `fresh`; `draw()` on `fresh` does `CopyResource(upload→card)` + barrier `COPY_DEST→COPY_SOURCE` (menu skips the first barrier via `used` flag: starts `COMMON`, promoted by first copy), then `CopyTextureRegion` with `PLACED_FOOTPRINT` (`RowPitch=pitch`, full `size(scale)` footprint) clipped to `(w,h)` and positioned centred (menu) or `(0,0)` (FPS line/panel), `D3D12_BOX{left:0,top:0,right:w,bottom:h}`. Panel rebuilds text at most every 500 ms (`EVERY`); `draw` itself only copies (cost tracked in `cost`, shown as `THIS PANEL n US A FRAME`).
- **Panel text (10 lines):** `FPS/F RAME/LONGEST`, `FRAME GEN nX SHOWN`, `DRAWS/DROPPED` (`draw::drawn/dropped_now`), `SUBMIT BUSY/WAITS FOR GPU`, `GPU BUSY/IDLE BEHIND US`, `RECORDING/COPY BACK` (diag) or `COPY BACK`, `MB UP/TEXTURES/BACK`, `RAM/IN MEMORY`, `VRAM/budget/SHARED`, `THIS PANEL cost`. Counters are per-frame deltas with wraparound-safe `since()`; `worst` = longest frame in window.
- **Compositor (Series S path + reference):** `Composer` compiles one `vs_5_1/ps_5_1` pair via `D3DCompile` at `Host` creation; per presenter slot owns a 2-descriptor shader-visible heap (grown on demand, indexed by completion fence so in-flight presents never alias); `record()` creates SRVs for both planes (format via `compose::view()` — typeless→display type, rejects multisample/depth/deny-SRV), sets one RTV, viewport+scissor = output, `DrawInstanced(3,1)`. Series X HUD-less FG copy reuses the same copy-region machinery in front of the known interface-composite draw (see `docs/dlss.md` § colour without interface).
- **Failure semantics:** `failed=true` stops all future draws (no spam); `colours()=None` skips silently; buffer recreation on `(format,scale)` change `mem::forget`s the old (card may still read it — size changes once or twice per run); `w/h` clamped with `min(width,height)`.

### 8.5 Window, display, input — full control notes

- **Creation hook:** `CreateWindowExA/W` → `fit()` → real create → `hook()` (store title proc in `victory.wndproc` prop, install `window_proc`). `fit()` leaves child windows and `CW_USEDEFAULT` alone; else scales to work area (`SPI_GETWORKAREA`) or `WINDOW_MAX`, centres, logs `width,height,to,limit`.
- **`window_proc`:** `WM_SETFOCUS/MOVE/SIZE/ACTIVATE/KILLFOCUS/NCDESTROY` maintain mouse capture; `WM_SYSCOMMAND/SC_CLOSE` → `end_title()` (console never sees close); `SC_KEYMENU` swallowed (Alt/F10 would otherwise enter an invisible modal menu loop and the title self-aborts ~30 s later); everything else forwarded to title proc; `NCDESTROY` removes the prop. Fullscreen/IME/capture applied only to top-level windows at hook time.
- **Fullscreen:** `SetWindowLongW(WS_POPUP)` + strip edge styles + `SetWindowPos(FRAMECHANGED|NOZORDER|NOACTIVATE)` to `FULLSCREEN_RES` or screen size. External alternative: `scripts/fullscreen.ps1` (DPI-aware, waits 180 s for `victory` window, borderless to monitor rect) — use when `FULLSCREEN=1` was not set at start.
- **Mouse:** `ClipCursor(client rect→screen)`, hide via balanced `ShowCursor` loop (`hidden` count restored on release); `captured_mouse_delta()` returns `None` unless foreground + captured; `MOUSE_WARP=1` recentres to `center` each sample and reports motion vs warp target.
- **Display lies (deliberate):** adapter reports `VIctory console GPU`, vendor AMD, `dedicated_video_memory=gpu_optimal`, `shared=standard` (`console::current().title_memory()`); exactly one adapter, one output; `EnumAdapters/Outputs(n>0)=NOT_FOUND` + nulls the out-pointer. Title paces by reported modes — change `DISPLAY_*` only if a title is observed rendering for the mode.
- **Keyboard-as-pad + text:** `gameinput` merges all host pads + fallback unless `ONE_PAD=0`; scan codes via `MapVirtualKeyW(MAPVK_VK_TO_VSC_EX)`; no code point 0 chars (shortcut); IME forced English only when asked.

### 8.6 Persistence format (edit by hand or by menu — same file)

`victory-settings.txt`: first line `# Saved by the F1 menu; …`, then `NAME=value` per line (`\r\n`), `NAME=` (empty) means unset, non-`VICTORY_` lines ignored on load, later duplicates win (`rev().find`). `save()` retains unedited lines, replaces edited vars, appends new ones, rewrites whole file. Precedence at next start: `play.rs SETTINGS` < menu file (non-`MENU_` vars unset-or-set) < `--set NAME=VALUE` < derived `MENU_*` snapshot. `KEPT` (`CONFIG/GAME_DIR/CACHE_DIR/DATA_DIR/STREAMLINE_DIR/LOG`) survive the menu restart wipe; every other `VICTORY_*` in the environment is cleared first.

### 8.7 How to extend the GUI (recipes)

1. **New menu row:** add `Item{var,label,values,live}` to `ITEMS` (keep first value = `play.rs` default; keep line ≤60 chars — test enforces), add switch to `play.rs SETTINGS` if it needs a default, handle live in `apply_live()` match if `live:true` (else restart note is automatic), extend `every_line_fits…`/`values_cycle…` expectations. If it is a launcher choice (`RES/FPS/DLSS/FG/CONSOLE`), mirror the `VICTORY_MENU_*` dance in `play.rs::settings()` + `ITEMS` + tests (`a_plain_start_is_the_launchers`, `what_the_menu_saved…`).
2. **New live toggle:** gate reads on `std::env::var` per use (never cache unless `OnceLock` + live-store like `fps_line_on/panel::shown`), add `AtomicBool` + setter in `apply_live`, document `LIVE` in label behaviour automatically.
3. **New overlay:** copy `panel.rs Buffers/painted/lit/size` skeleton; implement `frame()` (update ≤2 Hz, track `cost`) + `draw()` (copy-only, `unrecorded()` guard, `(format,scale)` recreate, clamp, `fresh` barrier dance); wire `Menu/Panel`-style fields into presenter (`host.rs` presenter struct) and call `frame()` per present, `draw()` in present's copy pass; add `VICTORY_PANEL`-style `switch::on` + F-key/menu hookup.
4. **New window/display/input flag:** read in `lifecycle.rs` via `enabled()` (once-per-window for window ops, per-message for focus ops), add to `ITEMS` if menu-exposed, add `fullscreen.ps1`-style fallback note if it can be applied externally.
5. **Perf budget:** keep `draw()` under ~100 µs (one `CopyResource` + one `CopyTextureRegion`); never allocate/format-compile per frame; never `D3DCompile` at runtime except `Composer::new`; keep text rebuild ≤2 Hz.

### 8.8 GUI test plan (no game data)

- `cargo test -p victory-gpu host::{menu,panel}` — line-fit/glyph-coverage/cycle/counter-wrap/generated-line tests (extend on every new row/line).
- `cargo test -p victory-hle lifecycle` — `WS_POPUP` iff `FULLSCREEN`, `WINDOW_MAX` parse, mouse-capture `None` when disabled, `fit_into` aspect cases.
- `cargo test -p victory-gpu compose` — diag GPU readback test (transparent/half/green-opaque compositing ±1 LSB).
- `cargo test -p victory -- play::tests` — launcher/menu precedence (plain start, no-Streamline, menu-over-options, `--set`).
- Manual: `VICTORY_MENU_OPEN=1` + `VICTORY_PANEL=1` screenshot at 720p/1080p/1440p/4k (scale 1/2/2/4) on each swap-chain format; F1 open/close, every row both directions, each action; `victory-settings.txt` diff; restart path with Streamline loaded.
- Diag only: confirm overlays absent from `replay.png`/captures, present in user screenshots.

### 8.9 GUI roadmap (do in this order)

1. Extract shared `host/overlay.rs` (`glyph/lit/painted/Buffers/size/scale`) — menu+panel duplicate ~120 lines today; keep pixel output bit-identical (screenshot diff test).
2. Add `VICTORY_DYNAMIC_WAIT` + `VICTORY_BVH_BUILDER` stall lines to F3 panel (the two fork deltas deserve first-class visibility) + `dynamic descriptor slots` pressure line (§5 Phase 2.1).
3. Menu search (`/` filters rows) once `ITEMS` passes ~35 rows; keep 10 ms key thread, add `BACKSPACE` handling.
4. Per-overlay `VICTORY_OVERLAY_SCALE=auto|1|2|4` override for 4k screenshots; HDR-format colours (currently only SDR + halves).
5. Proper Series X interface-plane tagging for FG HUD-less copy (replace shader-hash heuristic) — GUI-visible as stable prompts/map during camera turns (see `docs/dlss.md` §8 checklist).

## 9. Memory management and memory-leak solutions

No single allocator owns memory here. There are **five memory worlds** that must
agree, and every leak in `docs/bug-journal.md` was one world holding what another
had let go. Read this section with `pool.rs`, `residency.rs`, `shadow.rs`,
`watch.rs`, `memory.rs`, `texture.rs` + `texture/ahead.rs`, `tiles.rs`,
`resource.rs`, `copystat.rs`, `xmem.rs` and `host.rs` open.

### 9.1 The five worlds and who frees what

| # | World | Owner files | Unit | Free rule (nothing else frees) |
|---|---|---|---|---|
| 1 | Title address space (console view) | `victory-hle/src/modules/xmem.rs`, `victory-gpu/src/watch.rs` | 4 KiB pages, 2 MiB pools, 64 KiB views | Title `VirtualFree`/pool unregister; we only unmap views (`title_memory_unmapping/splitting/remapping`) and drop guards |
| 2 | Host copies of title buffers (CPU-readable shadow) | `shadow.rs` (`Shadow`), `watch.rs`, `land.rs`, `inplace.rs` | per-buffer `Shadow` + 4 MiB `window`s outside buffers | owners rule: last `Arc<Shadow>` + no list names it + idle past `let_go_idle`; windows also need `used` old |
| 3 | Title memory by address (RT arenas on card) | `memory.rs` (`Held`, `Packed`, directory + faults) | 64 KiB arena pages (`ARENA=64MiB`, `MOST=3GiB`, `WINDOWS=512`) | page given back when title memory gone, packed header unseen `SWEEP=128` frames, or fault-LRU at `FAULT_MB=512MiB` (`give_back`); list-named pages pinned for the run (known shortcut) |
| 4 | Texture + staging pools (host + card) | `pool.rs` (64 MiB `BLOCK`s), `texture.rs` (`Staging`), `tiles.rs`, `texback.rs`, `residency.rs` | pool pieces; per-texture `Staging` + detiled bytes | texture piece: last host-texture ref drops; staging piece: last recording list resets; empty blocks reclaimed to `SPARES` limit after `REST=2500ms` (`pool::reclaim`) |
| 5 | GPU objects + descriptor slots | `resource.rs`, `descriptor.rs` + `descriptor/host.rs`, `pipeline/library.rs`, `host.rs` (`Detached`) | resources, heap slots, table strides, dynamic slots (262k NVIDIA cap) | `Detached::release_tables()` on GPU completion + `settle_for_fence()` retry; views pool `views` freed with heap; library mapped not read (`LIBRARY_MAP`) |

Cross-cutting: `copystat.rs` (diag) walks the address space (~0.6 s, own thread)
and classifies commit into 10 `KINDS` (title heaps <1 TiB vs host >>2⁴⁰, page
pool, driver shader cache, process heap, 4 MiB windows, pipeline library file).
`host::ALIVE[]`, `shadow::live_span_bytes()`, `texture::live_staging_bytes()`,
`resource::KINDS[]`, `pool::memory()`, `KEPT_BYTES` feed the F3 panel and the
once-per-second `memory …` / `memory in blocks …` / `memory let go …` log lines.
Treat those lines as the leak detector: steady growth across identical caption
intervals = leak; sawtooth that returns = healthy pooling.

### 9.2 Budgets, reserves and admission gates (exact numbers)

- **Process-wide pressure** (`residency.rs::refresh`, 1 Hz): `PRESSURED` when
  `availPhys ≤3 GiB` or `availCommit ≤4 GiB`; clears only when `phys ≥5 GiB`
  **and** `commit ≥6 GiB` (hysteresis — do not "fix" flapping by narrowing it).
  `HEADROOM=min(phys,commit)` feeds F3/log. Commit sampled at most every 32 ms
  (`NEXT_COMMIT_SAMPLE`), including inside long sweeps.
- **Critical commit** (`critical_commit()`): `now < RETRY_UPLOAD_AT` (500 ms after
  any `E_OUTOFMEMORY`) **or** `availCommit <1536 MiB`. Critical path: staging
  growth → 0 spare blocks, prefetch → 16 MiB/frame, bake prefetch deferred below
  2 GiB, ordinary prefetch below 4 GiB, texture sweep interrupted and retryable.
- **Upload admission:** `admit_texture_pass(bake)` needs `commit ≥256 MiB` (bake)
  else `≥512 MiB`; `allow_staging_growth(bake)` needs `≥256 MiB` (bake) else
  `≥1024 MiB`. Growth = brand-new 64 MiB OS block; reuse of committed free ranges
  stays open whenever admission passes. `take_staging_deferred()` /
  `upload_refused()` thread-local + global flags propagate refusal to `draw.rs`
  (drops draws with reason, does not spin).
- **Prefetch ceiling** (`prefetch_limit`): critical → `min(bytes,16MiB)`;
  pressure or (32 GiB host **and** live staging ≥1 GiB) → `min(bytes,64MiB)`; else
  uncapped. Single texture may cross the cap; queue retains unserved slots.
- **Pool spares** (`pool::room`, 1 Hz): card <85 % budget **and** `keep_freed` **and**
  no pressure → keep 64 empty texture blocks (`SPARE_KEPT`, 4 GiB); else 2
  (`SPARE`). Staging: critical → 0, pressure → 1, else same limit. Staging blocks
  are PC memory — run crash-1 held 5.2 GB; cap is `STAGING_KEPT=8`.
- **Reuse delay** (`REST=2500 ms`): a freed piece is not reused for 2.5 s (nothing
  the card was told before then still runs; Windows resets cards slower than that
  over one submission). Bake 64 MiB growth gating scoped to speculative bindless
  only (`REST` in `pool.rs`) — one-shot ped/decoration bakes bypass via dedicated
  sync host list (`BAKE_UPLOAD`/`execute_sync_internal`, `prepare_bake_slot`).
- **RT arenas** (`memory.rs`): `FAULTS=128` pages/frame from fault notes (bursts
  of 2.5–3.9k notes in world-first frames self-terminate by frame 3);
  `FAULT_YOUNG=128` frames before a fault page is evictable (prevents churn when
  rays go anywhere); `FAULT_SOON=4` frames (re-want inside 4 = still read,
  submissions land ≤2 frames late); `YOUNG=4`, `AGAIN=8`, `SWEEP=128`,
  `SPARE=1024` free pages at list record. Report line `title memory by address:`
  prints pages held / given back / wanted again every 10 s — rising "wanted again"
  means raise `FAULT_MB`, rising faults/frame means `BVH_BOUNDS` is off or sizes
  race.
- **Shadow windows:** default 4 MiB however little read; `WINDOW_TAILS=1` keeps
  longer tail; idle windows let go by `let_go_idle` (age + unreferenced). Card-side
  big buffers (`CARD_BUFFERS_MIN_KB=64` default) keep shader-written ranges on the
  card; hashing on card (`CARD_HASH_LEN`, `CARD_REHASH_RUNS=1`) avoids CPU compares
  (550–710 MB → 25 MB in prologue).

### 9.3 Leak taxonomy — every pattern seen, where, and the fix

1. **Strided-table slots never populated → unbounded heap growth.**
   `descriptor/host.rs` (`SLOT_TABLE_STRIDES=252`, `heap_strides`): zero-sweep
   view-kind lookup + dynamic-slot stride fill on publish. Symptom: `draw.rs`
   `dropped draw due to exhausted dynamic descriptor slots`. Fix pattern:
   populate strides at `prepare_bake_slot`/`publish_uploaded_slot`, never leave
   `neutral()` published. Test: `descriptor` stride unit + `draw` settle test.
2. **Completed submissions pin dynamic slots across `bury_lag` (16).**
   `host.rs` (`relay_land`, `settle`): NVIDIA 262k slot pool exhausted; draws
   dropped. Fix: `Detached::release_tables()` immediately on GPU completion +
   `settle_for_fence()` retry in `draw.rs`; default `VICTORY_INDIRECT_REST=0` so one
   oversized indirect record zeroes nothing after it. Log: `dynamic descriptor
   slots:` line; assert `settle` never holds submit lock (`DYNAMIC_WAIT=1`).
3. **Staging admission before catch-up tripled budget under pressure.**
   Early prefetch added its own allowance on top of pressure cap. Fix:
   `residency::admit_texture_pass` + `prefetch_limit` as the single gate (16/64 MiB
   caps above); maintenance pass releases all eligible spare staging blocks under
   host pressure. Verify: `residency` unit tests (`bake_and_lookahead…`,
   `either_constraint…`, `prefetch_respects…`) + `pool::room` reclaim test.
4. **`E_OUTOFMEMORY` burst → Rust abort on 786 kB alloc.**
   64 MiB pool requests + 24 kB uploads failed; allocator retried in a tight loop
   (8 block creations in one burst). Fix: `RETRY_UPLOAD_AT = now+500 ms` backoff
   on growth (`allocation_error`), interrupt descriptor sweep on host OOM
   (retryable), commit check ≤32 ms cadence, defer ordinary prefetch <4 GiB / bake
   <2 GiB. Verify: replay 8 reported sizes through release pool with failing block
   allocator (bug-journal 15:56–15:59); expect 0 failed upload allocations, 0 Rust
   aborts.
5. **Aliased view map/unmap flood → 2–4 GiB bake-pass truncation.**
   Propagating aliases into `REMAPPED` + widened lookahead truncated frames
   1368–1369 (face-mask/shorts regression). Fix: **reverted** (`cap_alias_fix_rel`);
   rule: alias changes never widen bake lookahead without a `replay --compare`
   picture gate. Keep `REMAPPED` narrow.
6. **Bake slots overwritten with `neutral()` / silently dropped.**
   `descriptor/host.rs` + `draw.rs::flush_for_bake`: pending bake slots lost on
   defer/unmapped pages; false `address=0x0` unordered-SRV spam. Fix:
   `prepare_bake_slot`/`publish_uploaded_slot` ownership + `flush_for_bake`
   ordering; removed false warning. Symptom cleared: no `neutral()` over pending.
7. **Guarded-block accounting under storage writes.**
   10–50k guarded copies lay under `dstorage` pages per run. Fix:
   `VICTORY_PAGE_WRITES=guards|1|slots|all` opens guards where title maps pages;
   `VICTORY_VIEW_GUARD=0` disables. `watch.rs` + `shadow.rs::card_all` coordinate;
   kernel reads beside + copy-in (`COPY_BESIDE`) so re-guarded blocks don't fail.
8. **Texture sweep never releases (card-full from frame 1).**
   World frame 1 fills 85–99 % budget; 26k textures let go / 3.4k remade per
   5 min; 2.8 GB render targets pinned. Fixes stacked: `TEXTURE_POOL=1` (6 µs vs
   310 µs alloc), `LEVELS_ON_DEMAND=1` (2.2–3 GB unheld levels not allocated),
   `FULL_CARD_PASS` hold-back, `CARD_ROOM_FOR_{BUFFERS,TARGETS}`, `bake` upload
   paths, `SHOW_AFTER=1` (slot-after-upload ordering killed ~16-min device loss),
   `let_go` + sweep + `reclaim`. Monitor: `texture::let_go()`, `pool::memory()`,
   F3 `MB UP/TEXTURES/BACK` + `VRAM used/budget`.
9. **RT pages exhausted in ~1 min (2,048-page cap) → zeros forever.**
   Oldest fix in this fork: cap → 512 MB LRU (`FAULT_MB`), oldest evicted, re-made
   next frame on re-fault (one frame of zeros, counted). Follow-ups: `BVH_BOUNDS=1`
   (clamp wild stores; 2,500 → 90 missing pages/frame in dorm), `BVH_QUARANTINE=1`
   for first tests (stores never return), `BVH_SIZES_WAIT` (wait size slot only —
   still costs ~20 ms, still stale; see §5 Phase 1.1). Never raise `FAULT_MB`
   without checking card `budget−used` at run 406 levels (9,252/10,095 MB).
10. **Wave64→Wave32 transmission black squares.**
    `resource::Tracer::descriptor()/live()` dropped per-lane guards at join edges;
    `CS_AlphTransmissionPrefilterMip` output zeroed → `ptfx_PS_Transmission`
    black eyes/alpha. Fix in `resource.rs`, keep `VICTORY_WAVE_WIDE=0` (loop-widen
    TDR risk). Verify: `translate --validate` 43/43 graphics + 96/111 compute on
    154 VFX/particle/eye shaders, 0 regressions; `victory-shader` 281 passed.

### 9.4 Diagnostics — proving a leak or proving health

- **Always on:** F3 `RAM/IN MEMORY`, `VRAM used/budget/SHARED`, `MB UP/TEXTURES/
  BACK`, `THIS PANEL cost`; log `memory …` (commit by kind), `memory in blocks`
  (pool textures/staging MB), `memory let go` (windows/staging/snapshots),
  `dynamic descriptor slots:`, `title memory by address:` (10 s), `video memory
  … frame generation on` deltas. Screenshot these with the caption interval.
- **Diag builds:** `VICTORY_MEMORY_MAP=1` (address-space walk by kind; ~0.6 s own
  thread — note the 89 ms stall risk on giant regions), `VICTORY_CARD_ACCOUNTING`,
  `VICTORY_WALLS_TRACE`, `VICTORY_MERGE_RACE`, `VICTORY_UPCOPY_CHECK=1`
  (`replay --upcopies` offline equivalent), `VICTORY_COPystat` phases.
  `replay --check-written` asserts every changed word is restored;
  `--no-restore` exposes retention on cards too small for restore copies.
- **Leak repro recipe:** same save + same cache + fixed caption intervals (≥15 at
  ≥1k draws, bug-journal method); record `HEADROOM`, `pressured/critical` flips,
  `RETY_UPLOAD_AT` trips, `take_staging_deferred` counts, pool `SPARES`/reclaim
  events, `Detached` table holds, `Shadow::let_go` + `pool::reclaim` volumes.
  Growth across runs with identical intervals = leak; flat = pressure handling.
- **OOM triage order:** (1) `E_OUTOFMEMORY` site + Rust alloc site (past log had
  neither — add both before any other change); (2) commit headroom at trip (<2 GiB
  invalidates run); (3) staging vs texture vs shadow split (`memory in blocks`
  + `live_*_bytes`); (4) `critical_commit` active?; (5) bake vs speculative
  (`admit_texture_pass(bake)` path); (6) driver shader-cache/file-mapped growth
  (mapped, not private — not a leak).

### 9.5 Full switch reference for memory work (grouped)

- Gates: `TEXTURE_POOL`, `RIPE_BLOCKS`, `TEXTURE_NO_WAIT/CATCHUP/STAGE_{MB,MS}`,
  `LEVELS_ON_DEMAND`, `HELD_LEVELS`, `LEVELS_MIN_KB`, `FULL_CARD_PASS`,
  `CARD_ROOM_FOR_{BUFFERS,TARGETS{,_SETTLE_MS}}`, `KEEP_{FREED,SHOWN,SNAPSHOTS{,_MB},VIEWS}`,
  `POOL_SMALL`, `REUPLOAD_WHEN_FULL`, `READ_BACK`, `TEXTURE_BACK`, `TEXTURE_KEEP/
  GUARD/HOT/SAME/SWEEP/ONCE/AWAY_MS/BUDGET_MB/CATCHUP/FILL/ROOM_PERCENT/STAGE_*`.
- Watches/guards: `VIEW_GUARD`, `SOFT_VIEWS`, `MAPPED_BY_{PAGE,VIEWS}`,
  `MAP_{LEAN,RUNS,THREADS}`, `GUARD_{ASK,HOLES,HOT,REST}`, `INPLACE{,_MB}`,
  `PORT_CONSTANTS`, `LENT_HEAP`, `WC_NO_BACK{,_MIN_KB}`, windows `WINDOW_{KB,TAILS}`,
  `SECTION_WINDOW`, `PLACED_{FILLER,ORDER}`, `KEEP_*`, `COPY_{ASK,BESIDE,MODE,OVER_TITLE}`,
  `TEXTURE_GUARD`, `PAGE_WRITES`, `PAGE_PARTS`, `TAKE_PARTS`, `PAIRS`, `DIRECT_WRITES`,
  `CUT_{WAITS,WRITES}`, `EXACT_{BACK,NO_WAIT}`, `FRESH_OPEN`.
- RT: `MEMORY_CARD_MB/DUMP`, `BVH_{BUILDER,BOUNDS,FENCES,SIZES_WAIT,TRACE,QUARANTINE}`,
  `UPLOAD_LOW`, `LENT_HEAP`, `MEMORY_MAP`.
- Observability: `MEMORY_MAP`, `CARD_ACCOUNTING`, `WALLS_TRACE`, `MERGE_RACE`,
  `UPCOPY_CHECK`, `COPystat` phase, `FAULT_BOOK{,_AT,_UNLOCKED}`, `POP_WATCH`,
  `AB` toggles.
- Precedence reminder: launcher `SETTINGS` < menu file < `--set`; memory switches
  are restart-only (no `LIVE`); `VICTORY_MENU_OPEN/PANEL` do not change budgets.

### 9.6 Rules for touching memory code (agents)

1. **Owners rule first:** every allocation names its single free path
   (ref-drop, list-reset, frame-sweep, GPU-completion). No second free path, no
   `neutral()` overwrite of pending slots, no alias widening without picture gate.
2. **Time, not frames:** reuse delays in ms (`REST`), sweeps in frames (`SWEEP`)
   with ms comment; loading screens run 30 fps but submissions in air hit 18 —
   reason in frames is wrong, reason in ms + submissions.
3. **Pressure in, growth out:** any new speculative allocation checks
   `admit_texture_pass`/`allow_staging_growth`/`critical_commit` first and honours
   `take_staging_deferred`/`upload_refused`; new pools register in `pool::room`
   reclaim and `copystat` accounting from day one.
4. **Count everything:** new buffers update `ALIVE`/span/bytes counters, F3/log
   lines stay parseable, unit tests assert caps (`residency` trio,
   `pool::reclaim`, `memory::give_back` LRU, `Shadow::let_go`).
5. **Repro before fix:** `replay --compare/--upcopies/--check-written` + same-save
   caption medians; OOM fixes replay reported sizes through a failing allocator;
   alias/bake changes need VFX/particle/eye `translate --validate` + picture
  compare.
6. **Never:** grow `FAULT_MB/MOST/SPARE_KEPT/STAGING_KEPT/REST` defaults without
   card-budget math in the PR; cache inside repo; log absolute paths; add
   `undocumented_unsafe_blocks` (pool/shadow `SAFETY` comments stay complete).

### 9.7 Roadmap (memory order — do not skip)

1. `Detached::release_tables` audit for every new slot kind + `settle_for_fence`
   coverage in `draw.rs` (dynamic-slot drops are the cheapest frames to win back).
2. Fence only the compacted-size word the title polls (`BVH_SIZES_WAIT` scope
   shrink, §5 Phase 1.1) — measures in dorm `CopyForBVHUpdate` ms.
3. `pool::reclaim` + `Shadow::let_go_idle` telemetry into F3 (spare counts, let-go
   MB/s) so pressure vs leak reads off the screenshot.
4. Extract `residency` policy table (thresholds/caps/backoffs) into one tested
   module; bug-journal numbers become unit tests, not prose.
5. Split `shadow.rs`/`watch.rs` review units; burn down missing `SAFETY` comments;
   keep `cargo xtask policy` + suites green throughout.

## 10. Resolution management — render size, output size, upscaling, pacing

Resolution here is **four independent sizes** that must agree. Confusing any two
is the most common way to get a black screen, a stretched picture, wrong DLSS
ratios, or a HUD that drifts. The chain is: **title render (scene) → title output
(swap chain) → host window (fit/stretch) → reported display (what the title is
told)**. Frame pacing (`frame.rs`, `frame_rate.rs`) is the fifth axis: it decides
how many of those pictures exist per second.

### 10.1 The four sizes + pacing (mental model)

| Size | Decided by | Typical values in this fork | Where it is enforced |
|---|---|---|---|
| Scene (render) | `VICTORY_DLSS_MODE` scale × output, or `RENDER_1080P`, or stock temporal 2/3 | 2560×1440 (native 1440p), 1920×1080 (1080to1440), 1706×960 (Quality on 1440p), 3840×2160 (stock 4K), 1280×720 (Series S) | `native_1440p::prepare` (`TEMPORAL_SCALE` 0x28F1E7E) + `prepare_upscaler` (renderer 7 + `UPSCALER_SIZING` 0x28EF69A) |
| Output (swap chain) | `VICTORY_NATIVE_1440P` patch or stock title | 2560×1440 (patched), 3840×2160 (stock 4K), 1920×1080 (Series S) | `prepare` (`SDR_MODE` 0x615C100 + `SCANOUT_RESERVATION` 0x247AD4 + `UI_VIEWPORT_LOAD` 0x58048D4), `run.rs` sets `DISPLAY_MODE=2560x1440` after patch |
| Host window | `lifecycle::fit()` + `WINDOW_MAX` + `FULLSCREEN{,_RES}` | work-area fit, aspect-kept, centred; `WS_POPUP` optional | `CreateWindowExA/W` hook, `fullscreen.ps1` external fallback |
| Reported display | `dxgi::display_size/refresh_rates/modes` | 3840×2160@60 default; 2560×1440@60 after 1440p patch; +120 Hz with `DISPLAY_REFRESH` | `display_size()` (`DISPLAY_MODE`), `refresh_rates()` (`DISPLAY_REFRESH` 61–240) |
| Pace | `frame.rs` clock + `frame_rate.rs` loader patch | 30 fps title stock; `FRAME_INTERVAL_US` override (1000 = uncapped, 16667 = 60, 33333 = 30) | `set_interval`/`wait` (origin/vblank/flip), `-Fps` launcher flag, F1 `FRAME RATE CAP` |

Key invariant: **the title computes render size itself** as `trunc(output × scale)`.
`ffxQuery` answers cannot change it. `Mode::render_size([W,H])` mirrors the title:
`(W×scale) as u32` truncation — use it for every budget, log line and test, never
round.

### 10.2 Output profiles (what ships today)

- **Series X 1440p (default, launcher `-Res 1440`).** `prepare(image,false)`:
  `SDR_MODE` 16 bytes → `WIDTH/HEIGHT` LE (`2560/1440`); scanout pool recomputed
  from `TextureDesc{Tex2d,2560×1440,RGBA8,SW_64KB_D}` → `next_multiple(64KiB)×2 →
  next_multiple(2MiB)` written into `SCANOUT_RESERVATION`; `TEMPORAL_SCALE`
  `2/3→1.0` (must equal stock bytes before write); `UI_VIEWPORT_LOAD` redirected
  to read `SDR_MODE` (displacement patch) so Settings previews use 1440p. Returns
  `(2560,1440)`. Cache dir `images/native-1440p-sdr` (or `-upscaler-<mode>-sdr`
  when DLSS/FG/FSR active). `DISPLAY_MODE` forced to `2560x1440` for this run.
- **Series X 1080-to-1440 (`-Res 1080to1440`, `RENDER_1080P=1`).** Same output
  patch, but `TEMPORAL_SCALE = 1920/2560 = 0.75` and returns `(1920,1080)`.
  Ignored when an upscaler mode is active (`run.rs` prints `RENDER_1080P is left
  out: DLSS_MODE sets the scene's size`) — DLSS scale wins, never both.
- **Series X stock 4K (`-Res 4k`).** No `prepare()`; output stays 3840×2160,
  temporal scale stays 2/3 (scene 2560×1440, title's own temporal upscale). With
  `--dlss <mode>`, `prepare_upscaler(mode)` reroutes renderer 7 at that scale.
- **Series S (`-SeriesS` / `MENU_CONSOLE=series-s`).** No 1440p patch allowed
  (`ensure!(SeriesX)` — 1440p on Series S is a hard error, not a fallback).
  Stock path: 1280×720 inside, 1920×1080 out, FidelityFX renderer 7 natively,
  two-plane composite (`compose.rs`) for scene + interface.
- **Build gate:** every patch site checks `RETAIL_BUILD =
  b99f876b…e239e93` + exact stock bytes (`STOCK_*` consts). Patched-twice =
  refused (`prepare` twice errors — test asserts). Unknown build → no patch +
  console warning, title runs stock. Never fuzzy-match.

### 10.3 Upscaler path (renderer 7) and DLSS scale table

Stock Series X uses renderer 6 (title's own temporal pass) and never loads its
upscaler DLL. `prepare_upscaler(mode)` forces renderer 7 everywhere:

1. `RENDERER_SELECTION` 0x7E9421: `mov eax,6; cmovne` → `mov eax,7; nop×3`.
2. `UPSCALER_SIZING` 0x28EF69A: `movzx/test/jne` → `mov eax,<ffx_mode>; mov
   [r14+0x90],eax; jmp scaled` (lands exactly on `SCALED_LOAD` — test asserts
   `UPSCALER_SIZING+14+0x0A == SCALED_LOAD`).
3. Non-Quality modes: native branch's `1.0` load operand replaced by a literal of
   our scale; scaled branch loads it via recomputed displacement (shared
   `UPSCALER_SCALE` 0x615D95C const left untouched — other code reads it).
4. `TEMPORAL_SCALE` ← `mode.scale()` (stock Series X pairs them 2/3 each).

`Mode` (`streamline/plan.rs`): `Dlaa=1.0 (ffx 0)`, `Quality=2/3 (ffx 1)`,
`Balanced=0.58 (ffx 2)`, `Performance=0.5 (ffx 3)`, `UltraPerformance=1/3
(ffx 4)`. Render sizes (truncated, title-identical):

| Output | Dlaa | Quality | Balanced | Performance | UltraPerf |
|---|---|---|---|---|---|
| 3840×2160 | 3840×2160 | 2560×1440 | 2227×1252 | 1920×1080 | 1280×720 |
| 2560×1440 | 2560×1440 | 1706×960 | 1484×835 | 1280×720 | 853×480 |
| 1920×1080 (S) | 1920×1080 | 1280×720 | 1113×626 | 960×540 | 640×360 |

Launcher `play.rs`: `--dlss off|dlaa|quality|balanced|performance` (default
`quality`); `FRAMEGEN` without DLSS needs `Dlaa` + title FidelityFX (Series S
sizes only — at 1440p the context fails and the title faults at first upscale,
so `run.rs` requires both switches). `VICTORY_FSR_PATH=1` runs the same path
with the title's own FidelityFX (DLSS comparison baseline). `Asked::from_env`
defaults: no `DLSS_MODE` + FG-without-DLSS → `Dlaa`, else `Quality`; bad names
are hard errors. At dispatch, DLSS reads jitter/MV/near/far/VFOV/reset from the
dispatch + `DLSS_JITTER/MV` multipliers (default `1,1`; flip-x `-1,1` trial
documented in `docs/dlss.md` §8) and tags viewport 1, frame index = upscale count.

### 10.4 Per-resolution runtime corrections

- **Frontend viewport fix** (`draw.rs::native_frontend_viewport`): with
  `NATIVE_1440P=1`, the interface-composite draw (shaders `0602c238…` +
  `a78d088f…`, target = presented texture, blend premultiplied) expects a
  3840×2160 canvas in one constant (`480`-byte view at table+64, legacy matrix
  `[2/3840, -2/2160, …]`). The fix widens only the viewport to 3840×2160 while
  scissor + target stay 1440p, so the projection lands. Logs once
  (`the frontend's legacy 4K projection…`). Remove only with a 4K-vs-1440p
  `replay --compare` picture gate.
- **Series X HUD-less colour** (FG): copy in front of the same composite draw
  into our `back-buffer-size` textures (`streamline::before_target_draw`),
  tagged `kBufferTypeHUDLessColor` at present; other pipelines drawing into
  presented textures are log-named once (`this pipeline draws into the texture
  the title presents`). `VICTORY_FRAMEGEN_HUDLESS=0` disables (interface warps).
- **Compositor scaling** (`compose.rs`): fullscreen-triangle `SampleLevel(uv,0)`
  with linear clamp; `view()` accepts `R16G16B16F/R10G10B10A2/R8G8B8A8/B8G8R8A8`
  + typeless-displayable, rejects multisample/deny-SRV. Each presenter slot owns
  its 2-SRV heap (fence-selected, never aliased in flight).
- **Overlay scaling:** menu/panel `scale=(height/540).max(1)` — 1 at 720p, 2 at
  1080p/1440p, 4 at 4K. Pitch 256-aligned; buffers recreated on `(format,scale)`
  change. See §8.4.

### 10.5 Window, display-report and pacing controls

- **Window fit** (`lifecycle.rs`): `WINDOW_MAX=WxH` caps the work-area bound
  before aspect-kept shrink + centre; render targets unchanged (swap chain
  stretches). `FULLSCREEN=1` → `WS_POPUP` at `FULLSCREEN_RES` or screen size
  (applied only to top-level windows at hook time; `fullscreen.ps1` is the
  post-hoc equivalent). Log: `title window fitted to the screen (w,h → to)`.
- **Reported display** (`dxgi.rs`): `DISPLAY_MODE=WxH` (640–7680 × 360–4320,
  default 3840×2160) = "the TV plugged in"; `DISPLAY_REFRESH=hz` (61–240) adds a
  second 60+hz mode so the title unlocks 120 Hz intervals. Both `OnceLock`,
  info-logged once. Shortcuts recorded in code: not the host size by default;
  fixed modes not host rates — change only when a title is observed pacing by them.
- **Frame pacing** (`frame.rs` + `loader/frame_rate.rs`): title sets interval
  (`set_interval`: `length==0 → 16.667ms`), waits origin/vblank/flip (`wait`).
  Overrides: `FRAME_INTERVAL_MS` (0–1000, `0` = no wait) replaces title interval
  **and** patches loader frame rate; `FRAME_INTERVAL_US` (1000–1e6) replaces our
  clock only (16667 = 60 fps, 8333 = 120, 1000 = uncapped). Launcher `-Fps N` →
  `FRAME_INTERVAL_US=round(1e6/N)` (`1000` default = title's own 30). `LATE_WAITS=1`
  restores sleeping to next tick; default answers already-passed ticks at once
  (saved 16.8 ms per 730 ms frame in run 58). `TIMER_LOG=1` bands pacing every 5 s.
  Title's own 30 fps cap lifted by `frame_rate.rs` patch; `-Fps 60` is experimental
  because the layer itself costs ~33 ms/frame (see foxhound README).

### 10.6 Resolution switch reference (copy-paste safe)

```
Series X output:  VICTORY_NATIVE_1440P=1 | VICTORY_RENDER_1080P=1 (implies 1440p output, 1080p scene, ignored with DLSS) | neither (stock 4K)
Upscaler:         VICTORY_DLSS=1 + VICTORY_DLSS_MODE=dlaa|quality|balanced|performance|ultraperformance
                  VICTORY_DLSS_PRESET=default|j|k|l|m (deprecated e,f) | VICTORY_FSR_PATH=1 (title FidelityFX, no Streamline)
                  VICTORY_DLSS_JITTER=x,y VICTORY_DLSS_MV=x,y VICTORY_DLSS_REACTIVE=0 VICTORY_STREAMLINE_{DIR,LOG,PRESENT} VICTORY_FRAMEGEN{,_X,_CYCLE,_HUDLESS}
Window/display:   VICTORY_WINDOW_MAX=WxH VICTORY_FULLSCREEN=1 VICTORY_FULLSCREEN_RES=WxH
                  VICTORY_DISPLAY_MODE=WxH VICTORY_DISPLAY_REFRESH=hz VICTORY_DISPLAY_PROFILE_TEST_IMAGE=<exe for tests>
Pacing:           VICTORY_FRAME_INTERVAL_MS=ms VICTORY_FRAME_INTERVAL_US=us VICTORY_TITLE_FPS VICTORY_FPS_LINE VICTORY_LATE_WAITS VICTORY_TIMER_LOG
Menu mirrors:     VICTORY_MENU_{CONSOLE=series-x|series-s,RES=1440|1080to1440|4k,DLSS,FPS=1000|120|60|30,FRAMEGEN=0|2..6,DLSS files}
```

Precedence: `play.rs SETTINGS` < menu file (`VICTORY_MENU_*` decoded to switches)
< `--set NAME=VALUE`; `run.rs` derives `DISPLAY_MODE/NATIVE_1440P` after patching
(overwrites env for the run). `KEPT` vars survive menu-restart wipe.

### 10.7 How to change resolution behaviour (recipes)

1. **New output size (e.g. 1080p native):** add `prepare_1080p`-style fn beside
   `prepare()` with new `WIDTH/HEIGHT`, recompute scanout pool via
   `TextureDesc`, find new RVAs + stock bytes (refuse unknown builds), new cache
   dir `images/native-1080p-sdr`, new `-Res` value + `MENU_RES` entry + tests
   mirroring `patches_the_retail_image_for_1440p`. Never parameterise the 1440p
   consts in place — the stock-byte checks are per-profile.
2. **New DLSS scale:** extend `Mode` + `scale/ffx_mode/sl_mode/render_size` +
   `MODES` order (largest first) + `UPSCALER_SIZING` literal path + launcher
   `DLSS_MODES` + menu row + `docs/dlss.md` setup table. Keep truncation
   (`as u32`), never rounding.
3. **New viewport/projection exception:** follow `native_frontend_viewport`:
   gate on `NATIVE_1440P`, match exact viewport + shader hashes + constant bytes,
   log once via `FIRST.call_once`, return `Option<VIEWPORT>`; add ignored-unless-
   `DISPLAY_PROFILE_TEST_IMAGE` test + `replay --compare` 1440p-vs-4K gate.
4. **New pacing mode:** prefer `FRAME_INTERVAL_US` (clock only) over `MS`
   (also patches loader rate); keep `asked()`/`interval_override()` `OnceLock`
   + warn-on-invalid logging; update F1 `FRAME RATE CAP` values + `play.rs -Fps`
   range + `frame.rs` unit tests (`the_switch_reads_milliseconds`,
   `waits_end_on_ticks…`).

### 10.8 Resolution test plan (no game data + with title)

- Without title: `cargo test -p victory` (`native_1440p` tests skip unless
  `VICTORY_DISPLAY_PROFILE_TEST_IMAGE` points at the retail exe — never commit
  it); `frame.rs` tick/token tests; `streamline::plan` mode/scale/render_size
  asserts; `lifecycle` fit/window-limit tests; `dxgi` device→output walk test.
- With title (each output × each DLSS mode): `upscaler-check --device --present`,
  then the 11-step `docs/dlss.md § The first run` checklist (switch-on line,
  first tagged frame, 2N shown, no status-4/HDR, on/off at menus, HUD-less copies
  ≈ presents, MV ×1,1 single-edge, jitter sharp-in-1s, VRAM delta ~489 MB@1440p,
  no `presentCommon() was not observed` with `STREAMLINE_PRESENT=1`).
- Perf: same save + cache per (output, mode); record scene size from
  `DLSS: upscaling recorded` (`mvec_scale` = 1/render_w,h) + output size +
  FPS/frame-gen line; never compare across different scene sizes.
- Screenshots: `MENU_OPEN=1` + `PANEL=1` at 720p/1080p/1440p/4K outputs to verify
  overlay `scale` 1/2/2/4 and centred/top-left placement on every swap-chain format.

### 10.9 Resolution roadmap (do in this order)

1. Surface scene/output/window/reported sizes as four F3 lines (today only FPS +
   VRAM read off screenshots); log `display … scene … through … (mode)` already
   exists at startup — keep its wording stable for grep.
2. Per-driver `warm` cache dirs already exist (`images/native-1440p-upscaler-*`);
   extend to `warm` pipeline book per (output, mode) so first-run stutter is
   measured per resolution, not globally.
3. `VICTORY_OVERLAY_SCALE=auto|1|2|4` (see §8.9.4) before any 4K-default move —
   overlays at `scale=4` are unreadable in current font metrics on some formats.
4. Revisit `DISPLAY_MODE=host size by default` shortcut only with a title observed
   rendering for the reported mode + `replay --compare` across both reports.

## 11. Lighting management — ray tracing, estimate, water, haze, depth

Two exclusive paths produce indirect light; supporting passes decide whether
water, haze and glass read the right depth. **Never both at once:**
`ambient::on()` is `(AMBIENT set OR specular set) AND NOT builder_runs()`
(`victory-gpu/src/ambient.rs`). The launcher ships the title's ray tracing ON
(`VICTORY_BVH_BUILDER=1` in `play.rs SETTINGS`) with the estimate configured
but dormant (`AMBIENT=0.8`, `AIR=1.5`, `SPECULAR=1`); turning the builder off
wakes the estimate. Code default (env unset) is builder off + estimate off
(title's own output); `docs/README-foxhound.md`'s "BVH off" table predates the
launcher default — trust `play.rs` for what a run does.

### 11.1 The two paths (mental model)

| Path | What lights the room | Cost today | When to use it |
|---|---|---|---|
| Title ray tracing (`BVH_BUILDER=1`) | Real probes/bounce/reflections + lamp/sun shadows via title traversal over host-built structures | ~+40 ms/frame card time in the dorm (75/90 ms busy vs 34/64 off); first visits stutter while RT programs compile; ~280 ms/frame while resolves queue on a cold cache (§11.3) | Default launcher path; any "ground truth" picture gate |
| Ambient estimate (`BVH_BUILDER=0` + `AMBIENT`/`SPECULAR`) | Screen-space stand-in injected after the title's probe/miss programs (§11.4) | Near zero (7 small dispatches after existing programs) | Fast runs, cards where the builder misbehaves, before/after comparisons |

Without either, anything the sun does not hit directly is black (interiors,
shadowed faces, lamp-only rooms) and reflections fall back to sky light.

### 11.2 Title ray tracing (what it is + safety rules)

The title uses no ray-tracing API: ordinary compute shaders walk structures
through raw 40-bit title-memory addresses (per-thread stack in shared memory).
Static-world structures ship pre-built in title data (console-GPU native 64-byte
nodes: box nodes with four children in 16-bit or 32-bit float bounds, triangle
nodes, instance nodes with 3×4 matrix + root pointer) and are unpacked on-GPU
once by `DecompressHeader`/`Box32`/`Box16`/`LeafRehydration` (~3,000 in the
opening). Per-frame dynamics come from the title's `XboxBvhBuilder.dll`
(52 compute programs): this fork translates all 52 (host pipelines on RX 7900
XTX; title creates 53), routes buffer addresses via the spilled user-data table
+ root UAVs, and adds byte/16-bit buffer access + control-flow shapes. Only the
per-node test is hardware: `image_bvh64_intersect_ray` / `image_bvh_intersect_ray`
(slab tests + sorting network for boxes, watertight test for triangles),
emulated in DXIL and checked bit-for-bit on-GPU over 311 ray/node cases against
`victory-shader/src/bvh.rs`. 59 of 64 trace programs + the irradiance-cache
shading program translate; 5 shadow-mask trace programs do not (address-out-of-
heap-picked-descriptor — host would need an address table by slot; unobserved
in any run so far).

Host side (`pipeline.rs::Builder`, `memory.rs`): stages run **all or none**.
A builder dispatch runs only while every builder pipeline the title created has
a host pipeline — the first builder dispatch builds them all on the spot;
anything unmakeable or unbindable keeps the builder off (`off` count) or rests
it for the current + next 2 frames (`builder_rests`, `rests`). `memory.rs` keeps
stores on-card (only small title-given buffers come back), unpacks packed
structures whole, and titles buffers take stores only after being handed over
as UAVs. `lower.rs::Lower::memory` + `resource.rs` address tracking implement
loads/stores/atomic add/float min-max through the 64 KiB-page directory; an
access to a missing page is lost (read zero, store dropped).

| Switch | Effect |
|---|---|
| `VICTORY_BVH_BUILDER=1` | Builder stages may run (asks; all-or-none gate still applies) |
| `VICTORY_BVH_QUARANTINE=1` | Builder stores stay in host copies, never copied back (first-run safety; title reads back sizes/counters and gets none — not a way to run) |
| `VICTORY_BVH_BOUNDS=0` | Unbound builder accesses (default bounded: over-written size fields used to send copies through GBs of structures; bounded cut fault pages 2,500 → 90/frame) |
| `VICTORY_BVH_FENCES=0` | No fences around builder buffer atomics (beloooo's `72e748ea` also has them) |
| `VICTORY_BVH_SIZES_WAIT=1` | Wait for submissions that ask structure sizes (off: no help, slow — run light-15: still stale, +~20 ms/frame, open gap) |
| `VICTORY_BVH_TRACE=1` | Log sizes asked/when/copies (diagnostics) |
| `VICTORY_MEMORY_CARD_MB` (4096) / `VICTORY_MEMORY_DUMP=<dir>` (+`go` file) | Cap for addressable memory on-card / dump arenas out |

Known gaps (from `docs/fork-ray-tracing.md` + runs light-10/11/13/15, n19):
dynamic structures overlap (8/65 good in light-10, 48/275 in light-15 — title
reads a compacted size 1–2 frames late, our submissions land late, the copy
overwrites neighbours; `SIZES_WAIT` did not mend it); `CopyForBVHUpdate`
13.3 ms in 63 dispatches + traces 24 ms (`RaytraceVBuffer` 7.9, screen probes
6.4, irradiance cache 3.5, primary probes 2.3, shadows 1.6); captures made with
the builder on or off hang the card in replay (cause unknown — **do not try**).

### 11.3 Resolve queue (why cold RT runs are slow)

~1,000 per-material `CS_RaytracingResolve` programs each take seconds of driver
compile. `pipeline/make.rs`: without the builder they are **left out** unless
`VICTORY_RT_RESOLVE=1` (they would shade no hits while clogging the queue);
`CS_ScreenProbes_ActivateRays` is likewise left out unless
`VICTORY_ACTIVATE_RAYS=1` (with the builder it runs unless `=0`). With the
builder on, resolves are `no_wait`: their dispatches drop at once instead of
sleeping the translating threads inside the per-frame wait budget (run 403:
279 ms/frame while resolves were being made vs 54 ms after, same 3–4k draws;
`NOT_WAITED` counts dropped dispatches since start). Screen passes (no vertex
input, no depth) and confirmed first-scene/bake producers get workers + a longer
budget so lighting tables are not lost behind cold RT shaders at the world's
first frame (run textures-d2 had no lamps/diffuse to its end).

### 11.4 Ambient estimate (`ambient.rs`: roles, passes, switches)

Follows four title programs by container name (`role()`; unit-tested):
`CS_ScreenProbes_TemporalResolve_*` → `Resolve` (diffuse light),
`CS_CacheScreenSpaceRadiance_*` → `Radiance` (keeps the light visible surfaces
send out, 1280×720, frame before — the estimate's source),
`CS_AccumulateLocalLightInscatteringVolume_*` → `Air` (last writer of the
in-scattering volume), `CS_RayMiss_<8 hex>_` → `Miss` (reflections that found
nothing; explicitly NOT `CS_RayMiss_Volumetrics_*` nor `CS_ScreenProbe_RayMiss_*`).
Injection points in title memory (root slot, table dword): diffuse
`LIGHT=(1,176)` as R9G9B9E5-in-R32_UINT + `DEPTH=(0,464)`; radiance
`RADIANCE=(1,16)`; air volume `AIR=(1,32)`; miss `MISS_LIGHT=(1,104)`
R11G11B10 at half picture size + `MISS_DEPTH=(0,1000)` full-size 32-bit depth;
miss layer = first root constant (`MISS_OPAQUE=0`, `MISS_ALPHA=1`).

Seven host compute passes share one root signature
(`RootConstants(12) + SRV(t0) depth + UAV(u0) image + UAV(u1) cells + UAV(u2)
points`; constants: `level, constant, strength, radius, near=0.1, tan_y=0.45,
width, height, follow=0.04, most=4.0, air, once`; `cs_5_1`; UAV barriers
between): `keep` (8×4 threads: per 64×36 cell, mean of 25 taps of the radiance
image; lamps/glints clamped by `/max(1,luma/20)`; NaN/Inf → 0; with `ONCE` each
tap counted by the title's-own share),
`mean` (slow all-cell mean, `follow` lerp — off-screen light changes slowly so
turning into a dark corner is not instantly black),
`place` (cell world positions from depth),
`gather` (160×90 points: distance-weighted `c.w/(d²+r²)` directional means +
first-order harmonics per channel),
`raise` (per-pixel diffuse: normal from depth cross products, crease shade from
12 golden-angle taps within `radius` metres à la Belobug `cd9f78ce`, 4-point
depth-weighted gather with dither, directional `1.2×n` term clamped 0.2–2.5×
mean, lerped 0.2 toward slow mean, scaled by `level×openness`; title leftover
kept only up to `most×estimate` luma — sky outdoors survives, stale probe-memory
blocks do not; magenta garbage `g<0.5·min(r,b)` and blue `b>2.5g,b>r` filtered
to 0; with `ONCE` the pixel's title-share is written for the next keep),
`clear_air` (per-voxel: scatter capped at `air × lerp(slow-mean luma, seen luma,
depth fraction)` — windows may stay sky-bright),
`reflect` (per half-size texel: reflected direction from depth normal, 4-point
gather along it, `2.0×dir` term clamped 0.1–3× mean, 0.2 slow-mean lerp × share;
title sky kept by `sky` share, never brightened past the estimate).
`constant:<level>` mode skips gather and writes flat warm light
`(1.0,0.98,0.94)×level×openness` — Belobug's constant, no frame following.
Buffers: cells `(2×2304+1+25×2304)×16` B, points `160×90×48` B, made once via
`uav_buffer` (warn once on failure, then silently off). Dispatches: Radiance →
keep+mean (sets `KEPT`; later roles refuse until kept); Resolve →
place+gather+raise (skipped in constant mode); Air → clear_air; Miss →
place+gather+reflect. All dispatches run behind `replay::record::unrecorded()`
so captures/`replay.png` stay title-only.

| Switch (launcher default) | Parse / effect |
|---|---|
| `VICTORY_AMBIENT` (`0.8`) | Unset/`0`/`off` → title's own; number → that share of the estimate (1 = all); `constant:<level>` → flat light |
| `VICTORY_AMBIENT_SHADE` (`1,0.6`) | `strength,radius-metres`: crease darkening (0 strength or `px<1.5`/depth>400 m skips) |
| `VICTORY_AMBIENT_ONCE` (unset) | `=1`: estimate is one bounce of the title's own light — removes `level·a/(1−level·a)` feedback (at 0.8/0.6-albedo walls ≈1.9× over-bright, worse near pale walls/lamps) |
| `VICTORY_AMBIENT_AIR` (`1.5`) | Unset/`0` → title's (brown sunset veil indoors stays, run light-14); multiple → air-scatter cap (§11.4 `clear_air`) |
| `VICTORY_AMBIENT_SPECULAR` (`1`) | Unset/`0` → title sky; `<share>[,<sky 0–1>]` → that share of the along-reflection estimate, `sky` of the title sky kept (clamped) |
| `VICTORY_AMBIENT_SPECULAR_ALPHA` (unset) | `=1`: also run the alpha layer (glass/foam/wakes; default skipped — opaque depth behind transparency gives wrong reflect dirs and dark water patches, 2026-11-27 boat session) |

Logging: first 5 injections (`role, size, level, constant`); every 10 s a
reflections line (frames, dispatches, layers opaque/alpha/water/other, water +
not-ready counts); refusal reasons logged once per cause (`other tables`,
`another image`, stencil-plane redirect).

### 11.5 Draw integration (`draw.rs::ambient`)

Runs after the title dispatch, only if `ambient::on()`. Layer gate (Miss only):
water layers (anything but constant 0, and 1 unless `SPECULAR_ALPHA=1`) are noted
and left untouched. View lookup: exact `(slot, table-offset)` match in the
translated program's `used` + live root tables, with `placed_texture` resolve
and `ambient_tables` start cache; Miss falls back to scanning tables (first 2D
R11G11B10 UAV for the image; depth must be ~2× the image within 1 px, 32-bit
forms only) while Resolve falls back to a same-size `depth_like` SRV — both
refuse stencil-plane views (typeless `22/47` or plane-1 word-6 views: they make
every point far away with camera-facing normals; logged once, another depth
taken). Format/size/volume gates: Resolve needs R32U (fmt 42) + depth == image
size; Miss needs R11G11B10 (fmt 26) + depth ≈ 2× image (≥1×, ≤4×); Air needs a
3D target. After a successful injection the stale compute root/pipeline is
cleared (`host_pipeline=0`, `host_roots[1]=0`) so the next dispatch rebinds.
Per-miss-program table dumps (first 16 unseen names) name the image/depth
places when lookup fails — read them before adding a new fallback.

### 11.6 Water tiled light (`make.rs::water_light_left_out` + `emit`)

`water_shading:CS_TransmissionVolumetrics_Tiled[_LocalLights]` (hashes
`5dc3b25986622949`, `6f225b641c5b0708`) are 64-thread groups that only translate
in wave-wide form on 32-lane hosts (4 and 7 branches wave-wide; the underwater
`CS_TransmissionVolumetricsSubmerged` needs 29). In that form they leave whole
open-water tiles dark (around boats, shores, bridges) and cost ~15 ms/frame
(RTX 5070 runs r_water1/r_water3: 25 fps with, 30 without). Default on 32-lane
hosts: **left out** (water loses its own depth light — darker, bluer — but no
blocks); `VICTORY_WAVE_WIDE=1` stays off globally (runs 121–124 lost the card
on two moved programs) so this is a targeted exemption, not a mode change.
`VICTORY_WATER_LIGHT=1` runs them as before. Menu default OFF ("no dark
blocks"); turn ON only for a water-lighting picture gate with the fps cost
recorded.

### 11.7 Heat-haze + decal defrag (`VICTORY_LEAVE_OUT`)

Default `02384a60fda6f2d0,731a39171b568298` (menu: LEFT OUT vs RUN slower +
smears): `CS_DefragHeap_ComputeRelocations` (decal heap; groups of 128 hold two
console waves walking the heap apart; 11.85 ms/frame first gameplay area, 11.9 ms
run 184458; sibling `CS_CollectTriangles_Meshlets` groups-of-1024 never fits
wave32) and the heat-haze/in-scattering pass (per-pixel-distance effect pick;
smears blocks incl. water; with DLSS shows as smearing, in frame-gen as warped
silhouettes — `docs/dlss.md` Limits; needs glass-correct depth, §11.8). Added
entries must be 16-hex strings; `VICTORY_SKIP_PIPELINES` is the diagnostics-only
twin. Pipelines named here are never made (diag `missing()` reports the reason).

### 11.8 Early depth (`VICTORY_EARLY_DEPTH=1`, translator flag)

Pixel programs whose console code tests depth before running get D3D12's early
test (`ForceEarlyDepthStencil`, DXIL bit 3). Without it the host tests late, so
UAV writes + discards land for pixels depth would have rejected — the title's
glass pass (nearest depth/motion via 64-bit atomics into a texture, later
fullscreen write with depth test off) is the victim: glass behind hair took the
hair's slot and heat-haze read glass/world distance there. Part of translation:
both GPU and shader caches key on it. `queue.rs` 10 s line reports
`given/off/exports` counts. Launcher default on; turning off is a picture-gate
change (glass, haze, motion vectors all move).

### 11.9 Menu / launcher map

F1 rows (`host/menu.rs`, all `live:false` → restart): `LIGHT ESTIMATE`
(`AMBIENT` 0.8/1/0.5/OFF), `REFLECTION ESTIMATE` (`SPECULAR` ON_UNSET), `FOG CAP`
(`AIR` 1.5/1/2/NONE), `RAY TRACING (SLOW, BROKEN)` (`BVH_BUILDER` UNSET_ON),
`WATER LIGHT` (OFF default/ON), `HEAT HAZE AND DECAL DEFRAG` (LEFT OUT
default/RUN), `EARLY DEPTH TEST` (ON_UNSET). `play.rs SETTINGS` defaults §11.4
table + `BVH_BUILDER=1`, `EARLY_DEPTH=1`, `DYNAMIC_WAIT=1`; `--set` beats menu
file beats `SETTINGS`; `KEPT` untouched by lighting. Snapshot vars
`VICTORY_MENU_*` persist the choice to `victory-settings.txt`.

### 11.10 How to change lighting behaviour (recipes)

1. **Retune brightness:** change `SETTINGS` `AMBIENT` (and menu first value) —
   never both the share and `most=4.0`/`follow=0.04` consts in one step; record
   the same save + cache caption medians (§5) since feedback gain moves with
   albedo. Constant-mode rooms need only the `constant:` parse path.
2. **New title program variant:** extend `role()` name match + `LIGHT/DEPTH`
   consts if its tables moved + `usual_depth`/scan fallback if it binds
   elsewhere; add the `roles_by_name` unit case; gate any viewport-style
   exception on exact program + table + constant bytes with a `replay --compare`
   picture gate (same rule as §10.4).
3. **New estimate pass:** add the HLSL under the next `#ifdef`, a `PipelineState`
   in `Pass::make`, the `run()` arm in `after()`, and barrier discipline
   (UAV barrier before + after every dispatch); keep dispatches unrecorded.
4. **New left-out / water-like exemption:** hash-gate in `make.rs` beside
   `WATER_LIGHT` with the measured ms/frame + run ids in the comment; menu row
   + `SETTINGS` default + `docs/dlss.md` Limits entry if it smears motion.
5. **Early-depth change:** touch `emit::early_depth_of` + `lower.rs` flag site
   together (they are one feature); cache keys already include it — say so in
   the PR since warm caches invalidate.

### 11.11 Lighting test plan (no game data + with title)

- Without title: `ambient::tests` (`settings`, `specular_settings`,
  `roles_by_name`, settle-model `ONCE` math), `bvh` 311 GPU bit-exact cases,
  `emit` early-depth tests (`early_depth_is_given_where…`), `menu`/`play` tests
  (row defaults, `SETTINGS` precedence, `LEAVE_OUT` parse), `wave64_tests` for
  defrag/volumetric shapes.
- With title: builder OFF vs ON picture gate on the same save (interiors +
  lamp rooms + mirror/water reflections + sunset room for `AIR`); `SPECULAR`
  share ladder (0/0.6/1) + `sky` ladder; `AIR` ladder (unset/1/1.5/2); `ONCE`
  on/off over-bright check near pale walls; `WATER_LIGHT` off/on fps + dark-tile
  check around boats; `LEAVE_OUT=""` haze run vs default smear check
  (DLSS on: silhouettes; frame-gen: warps); `EARLY_DEPTH` off/on glass-behind-
  hair + haze-distance check. Never compare fps across different light modes —
  record per (builder, ambient, water, haze) tuple with ≥15 ≥1k-draw caption
  intervals.
- Diagnostics: `BVH_TRACE`/`MEMORY_DUMP` + `title memory by address` line for
  builder runs; `replay --compare` for any table/fallback change; never replay a
  builder capture (hangs the card).

### 11.12 Lighting roadmap (do in this order)

1. Resolve the stale-size overlap (§11.2): read-back fence or versioned
   size slots, gated by `replay --compare` on light-10/15-style saves — unlocks
   trusting the builder before any quality work.
2. Then the 5 missing shadow-mask programs (address-table-by-slot design) —
   lamp/sun shadows are the largest remaining "wrong" after indirect light.
3. Then water: exact-form (non-wave-wide) translation or tile-scoped fallback
   so `WATER_LIGHT=1` costs no 15 ms and dark tiles disappear — before any
   default flip.
4. Then `OVERLAY_SCALE`-style exposure control for the estimate (share ladder
   is too coarse across hours/rooms) + F1 `LIGHT ESTIMATE` live-apply (per-use
   env re-read + `apply_live` arm, §8 rule) — only after 1–3 stop moving the
   picture underneath it.

## 12. Ray-tracing management — builder, memory by address, traversal

§11 decides *which* path lights the room (title rays vs estimate). This section
owns the title-ray path itself: how its programs are detected, gated, fed
memory, emulated, queued and diagnosed. Spec detail:
`docs/fork-ray-tracing.md`, `victory-gpu/src/pipeline.rs` (`Builder`),
`memory.rs`, `draw.rs::dispatch/bind`, `victory-shader/src/{spill,bvh}.rs`,
`emit/{intersect,intersect_tests,lower,resource}.rs`. Launcher default is
builder ON (`play.rs SETTINGS VICTORY_BVH_BUILDER=1`); code default (env unset)
is OFF; `README-foxhound.md`'s switch table ("off") predates the launcher.

### 12.1 What the title does (and what we do about it)

No ray-tracing API: ordinary compute shaders traverse structures through raw
40-bit title-memory addresses (per-thread stack in group shared memory). Static
world structures ship pre-built in title data (console-GPU native 64-byte
nodes: box nodes with four children in 16- or 32-bit float bounds, triangle
nodes, instance nodes: 3×4 matrix + id/mask/flags + root pointer + bottom-level
address in 256-B units; top-level offset-13 instances). They ship **packed
("dehydrated")** and are unpacked in place once streamed in by
`DecompressHeader` / `Box32` / `Box16` / `LeafRehydration` (~3,000 in the
opening; header: 64 B, version 6 + marks `0x8000/0x4/0x2/0x100`, size, element
count, bounds, box/element/parent offsets). Per-frame dynamics come from the
title's `XboxBvhBuilder.dll` (52 compute programs; title creates 53 pipelines):
this fork translates all 52, routes buffer addresses via the spilled user-data
table + root UAVs, and added byte/16-bit buffer access + control-flow shapes.
Only the per-node test is hardware — `image_bvh64_intersect_ray` /
`image_bvh_intersect_ray` (box: four slab tests + sorting network, nearest
first when box-sort enabled; triangle: watertight Woop/Benthin/Wald with
numerator/denominator + I/J barycentric numerators) — emulated in DXIL and
checked bit-for-bit on-GPU over 311 ray/node cases against the Rust reference
(`bvh.rs`). 59 of 64 trace programs + the irradiance-cache shading program
translate; 5 shadow-mask trace programs do not (structure address out of a
heap-picked descriptor by computed index — host would need an address table by
slot; unobserved in any run). Traversal loops (the stack, instance transforms,
culling, `numerator/denominator` hit distance) stay title code; mapping to host
DXR is deliberately not attempted (layer would need geometry the title never
describes).

### 12.2 Detection (what counts as "the builder")

`pipeline::of_builder()`: exactly one compute shader whose container's DXIL
part yields native programs with a **spilled user-data table**
(`spill::is_spilled`: layout entries past the program's user SGPRs, first at
index 16, address in `s0` + `s1&0xff` bits 32–39; `spill::build` rewrites the
`s_load` table reads into entry values so tracking/lowering see plain
parameters — 118/118 such loads land on declared parameters where measured).
In the title exactly the builder's programs look like this. The flag lives on
the title pipeline (`builder` bool); `is_builder()` / `builder_quarantined()`
read it per dispatch/draw.

### 12.3 The all-or-none gate (`pipeline::Builder`)

Stages feed each other through memory: a stage running on a dropped stage's
unwritten output computes store offsets from garbage (run 98: 14/35 stages
translated → title died within a second on pointer-bit-pattern bounds). So:

- State: `pipelines` (every builder pipeline seen), `whole` (all have host
  pipelines), `off` (count at last failure), `rests` (no stage runs before this
  frame). First builder dispatch builds them all on the spot ("still being
  made" would be a dropped stage).
- `builder_runs()`: false unless `VICTORY_BVH_BUILDER=1` (`ALLOWED` OnceLock);
  false while `off == len` or `frame < rests`; fast true once `whole`; else
  collect missing host pipelines → warn (counts + hashes) and stay off, or
  info-log `the ray-tracing structure builder runs` and latch `whole`.
- `draw.rs::dispatch`: builder pipeline + gate closed → dropped (`DROPPED++`,
  diag `note_why` + `list_dropped_roots` with group counts + root constants /
  views / buffer extents). Bind failure on a builder stage → `builder_rests`
  (this frame + next 2; warn, first 20). `THREAD_DIMENSIONS`-counted dispatches
  are the builder's own (anything else logs once). Builder dispatches emit
  `capture::op "builder <name>"` in diagnostics builds.
- `draw.rs::bind` memory side: builder dispatches (or any shader with
  `stores_by_address` + writes) bind with `Use::Builds{back}`; quarantined
  pipelines skip the copy-back (§12.4).

### 12.4 Memory by address (`memory.rs`: directory, faults, arenas)

Structures lie in plain title memory between title objects, so granularity is
the 64 KiB page: in-buffer pages use that buffer's copy (`shadow`), others get
arena pages (64 MiB arenas, card copies via `shadow::by_address`). Per-frame
machinery: **directory** (per-page copy number + byte offset; CPU-written,
card copy refreshed over the bus), per-heap run of raw UAVs of numbered copies
(slot 0 = null view), **faults** (pages shaders found empty, read back once a
frame; walkers write the first half of the notes, builder the second).
Lost-access semantics: missing page → read 0 / dropped store; the note only
helps the *next* frame — for the builder that equals a dropped stage, hence
pre-dispatch `ready` fills: 256 KiB per out-of-buffer root view (4 MiB per
written UAV = structure being built), the page of each address-like root
constant (incl. 256-B-unit ones), first page of every structure named by list
upload memory (top-level build instances), and any packed structure whole
(same bytes re-streamed compare equal; card holds them unpacked — run light-3:
re-unpacking a quarter of structures wrote garbage over neighbours).
Copy-back: only what the builder stores in ≤256 KiB (`BACK`) title buffers it
was given (sizes/counters the title reads to shrink structures); structures and
scratch stay on-card (first whole-builder run wrote garbage over title
objects). Title-written pages upload only changed dwords since last taken.
Bounds: title-buffer pages accept builder stores only after a dispatch took
that buffer as a UAV; everything else lands in arenas. Packed structures die
when no header remains (`SWEEP` 128 frames; pages held for them alone are freed;
title-gone pages go at once).
Caps (see §9 for the leak-side rules): arenas most 3 GiB (`MOST`), windows 512,
packed-take 64 MiB, young 4 / spare 1024 / again 8; fault path `FAULTS=128` /
`FAULT_MB=512` (8,192 pages; runs 405/406 showed unbounded growth zeroing pages
all run — rays follow the player, so at the limit oldest-for-newest in
`give_back`), `FAULT_YOUNG=128` (new waits instead of churning in-use pages),
`FAULT_SOON=4` (given-back + wanted-again inside 4 frames = still read;
submissions land ≤2 frames late). First-two-world-frames bursts (3.7–3.9k then
2.5–2.9k notes, 4,095 slots) end by themselves; >half the lowest pages are not
title memory. Report (every 10 s, `title memory by address:`): pages/MB/arenas/
copies, per-frame read/build dispatches, fault/turned-down/brought-back rates,
looked-at pages, KB changed, packed count, ms, windows, packed in/out deltas,
per-source page origins, fault-held pages/MB vs 512 (walkers vs builder),
oldest, made/given-back/re-wanted/dead/adopted, non-title/low/over-limit/
roomless/builder-left rates, above-4 GiB wants. `VICTORY_MEMORY_DUMP=<dir>`
(+ `go` file) dumps arenas; header comment in `memory.rs` is normative.

### 12.5 Translator support (spill, addresses, intersection)

- `spill.rs`: `build` + `user_data` rewrite (loads left alone — non-constant
  dword offset, or dword in no layout entry — keep refusing as before).
- `resource.rs`: `Tracer::descriptor/live` track descriptors made from numbers
  + per-instruction `lanes.guards` on pruned edges (regression 2026-11-27:
  omitting guards zeroed the transmission buffer — same class of bug would eat
  builder addresses); every `image_bvh*` records where its descriptor came
  from; `stores_by_address` marks builder-like stores for the `Builds` path.
- `lower.rs::Lower::memory` + `intersect.rs`: raw-address loads/stores/atomics
  (add/sub, float min/max) through the `SLOT_MEMORY=254` directory
  (`MEMORY_PAGES` 2-dword entries, <128 GiB) + `MEMORY_FAULTS` notes (dword 0 =
  sink); ray test lowered to identical arithmetic in identical order to
  `bvh.rs` (box sorting network as the open-source Vulkan driver for this GPU
  family; triangle = JCGT-2013 watertight). Assumed, undocumented: console
  rounding, same-distance box order, edge-through-ray result (hit in both
  triangles), kinds 6/7 (triangle miss — title programs never read it).
- `intersect_tests.rs`: GPU runs of both instruction forms vs `bvh::intersect`
  over the 311 cases via `TEST_MEMORY` slot/base harness (same program reads
  scene memory without it); `native.rs` tests gate container parsing.
- Waves: title code is 64-lane; on 32-lane hosts only the wave-wide form
  translates some builder/trace shapes (group barriers in branches; runs
  121–124 lost the card on two moved programs) — `VICTORY_WAVE_WIDE=1` stays off
  globally; water/volumetric exemptions are targeted (§11.6), never a mode flip.

### 12.6 Queue interaction (`make.rs`: resolves, activation, budgets)

~1,000 per-material `CS_RaytracingResolve` programs cost seconds of driver
compile each: without the builder they are left out unless
`VICTORY_RT_RESOLVE=1` (they would shade no hits while clogging the cold queue);
`CS_ScreenProbes_ActivateRays` likewise unless `VICTORY_ACTIVATE_RAYS=1` (with
the builder it runs unless `=0`). With the builder on, resolves are `no_wait`:
dispatches drop immediately instead of sleeping translator threads inside the
per-frame wait budget (run 403: 279 ms/frame while resolves queued vs 54 ms
after, same 3–4k draws; `NOT_WAITED` counts drops since start). Screen passes
(no vertex input, no depth) + confirmed first-scene/bake producers get workers
+ longer budgets so lighting tables are not lost behind cold RT shaders at the
world's first frame; the first builder dispatch makes all its stages at once
rather than over frames. Default per-frame budget is 0.5 s of title frame;
over-budget unready pipelines drop the draw/dispatch like untranslated ones.

### 12.7 Switch reference (copy-paste safe)

```
Builder gate:   VICTORY_BVH_BUILDER=1 (asks; all-or-none still applies)
First-run:      VICTORY_BVH_QUARANTINE=1 (stores stay in host copies; title gets no sizes back — not a way to run)
Bounds/fences:  VICTORY_BVH_BOUNDS=0 (unbound; default bounded: 2500 -> 90 fault pages/frame) | VICTORY_BVH_FENCES=0 (no atomics fences)
Sizes:          VICTORY_BVH_SIZES_WAIT=1 (wait for size-asking submissions; usually still stale, ~+20 ms/frame)
Diag:           VICTORY_BVH_TRACE=1 (sizes asked/when/copies) | VICTORY_MEMORY_DUMP=<dir> (+go file) | VICTORY_MEMORY_CARD_MB (default 4096)
Queue:          VICTORY_RT_RESOLVE=1 (resolves without BVH) | VICTORY_ACTIVATE_RAYS=1/0 (rays without/with BVH)
Memory caps:    (see §9: FAULT_MB faults budget, pool/residency pressure, prefetch/spares)
```

Precedence: `play.rs SETTINGS` < menu file < `--set`; `run.rs` derives nothing
for RT (unlike `DISPLAY_MODE` for 1440p). Menu row `RAY TRACING (SLOW, BROKEN)`
(`UNSET_ON`, `live:false` → restart). Ambient estimate switches itself off
while this gate is open (§11.4) — setting both is not an error, the builder wins.

### 12.8 Diagnostics (what to read first)

Order: `builder_runs` info/warn lines (runs? which hashes missing?) →
`title memory by address` 10 s line (fault-held MB vs 512, made/given-back/
re-wanted, non-title share) → `BVH_TRACE` (who asked which size when, what the
slot held, what the copy overwrote) → diag `list_dropped_roots` + `builder
<name>` capture ops (which roots/buffers each dropped stage would have touched)
→ `MEMORY_DUMP` arena walk for the overlap pattern (neighbour-size writes).
Known signatures: dynamic-structure overlap (8/65 good light-10, 48/275
light-15: title reads compacted size 1–2 frames before our submission lands,
takes that-sized memory, copy overwrites neighbours); `CopyForBVHUpdate`
13.3 ms/63 dispatches + traces 24 ms (`RaytraceVBuffer` 7.9, screen probes 6.4,
irradiance 3.5, primary 2.3, shadows 1.6 — timing run light-13, title commands
84 ms, dorm 75/90 vs 34/64 ms); re-unpack garbage (light-3); fault storms at
world entry (405/406 bursts, self-limiting). Captures made with the builder on
*or* off hang the card in replay (fork-ray-tracing.md: cause unknown —
**do not try**); `replay --compare/--diverge/--wait-all/--wait-lists` still
applies to non-builder changes around it.

### 12.9 How to change ray-tracing behaviour (recipes)

1. **New builder program variant:** confirm `is_spilled` shape (index-16 table,
   ≥2 address registers); if it binds differently, extend `spill::user_data`
   + `resource.rs` address tracking together (never one without the other);
   add `native.rs`-level parse test + `replay --compare` gate on a builder save.
2. **New trace-program support:** extend `bvh.rs` reference + `intersect.rs`
   lowering in lockstep (dword-for-dword); add the ray/node case to the 311;
   never "fix" timing by lengthening waits — prove with
   `replay --wait-all/--wait-lists/--diverge`.
3. **Quarantine-first-run procedure:** `QUARANTINE=1` + `BVH_TRACE=1` +
   `MEMORY_DUMP`, run the new area, read sizes/counters mismatch before
   allowing copy-back; a stage storing out of place must stop the run (run 98),
   never widen `BOUNDS` past the fault-line check.
4. **Fault-cap tuning:** move `FAULT_MB` only with card-budget math (run-406
   calibration: 8,192 fault pages at 9,252/10,095 MB + 478 MB texture headroom)
   + report-line before/after (re-wanted-within-4 = still-read signal).
5. **Resolve/activation gates:** new skip in `make.rs::left_out` needs the
   measured cold-queue ms + run ids in the comment and a menu/`SETTINGS`/docs
   entry if user-facing.

### 12.10 Test plan (no game data + with title)

- Without title: `cargo test -p victory-shader` (bvh reference, intersect GPU
  bit-exact incl. `TEST_MEMORY` harness, spill rewrite incl. guard-pruning
  regression, `native_programs` parse/err cases, early-depth interplay);
  `cargo test -p victory-gpu` (pipeline keep/share, skip-hash parse,
  `spill::is_spilled` shape asserts, `memory::give_back`/residency/pool unit
  tests); `cargo test -p victory` (play `SETTINGS`/precedence incl. RT rows).
- With title, same save + cache per tuple: builder OFF vs ON picture gate
  (interiors/lamp rooms/reflections/shadow masks); `QUARANTINE=1` first-run
  shape check; `BOUNDS=0/1` fault-pages/frame; `SIZES_WAIT=0/1` frame-ms +
  overlap-goodness count; dorm busy-ms (target: 75/90 → down, never by
  lengthening waits); `RT_RESOLVE`/`ACTIVATE_RAYS` ladders on cold cache with
  `NOT_WAITED` + wait-budget ms recorded. Never compare fps across different
  (builder, resolve, ambient) tuples; ≥15 ≥1k-draw caption intervals each.
- Never replay a builder capture; `replay --compare` gates every table,
  fallback, lowering or lookahead change first.

### 12.11 Roadmap (do in this order)

1. Stale-size overlap (§12.8): versioned size slots or read-back fence for the
   compacted-size handshake; success = light-10/15-style saves go 48/275 → ~all
   good with no `SIZES_WAIT` ms cost. Unlocks trusting the builder at all.
2. The 5 missing shadow-mask programs (address-table-by-slot design) — lamp/sun
   shadows are the largest remaining "wrong" once indirect light lands.
3. Root-cause the builder-capture replay hang before any capture-based RT
   workflow is documented.
4. Report-driven fault-cap policy (auto `FAULT_MB` posture from re-wanted/dead
   rates) only after 1–3 stop moving the numbers underneath it.

## 13. Full test runs, verbose logging, variable logging, debugging

Every behaviour claim in §§8–12 is proved the same way: a scripted run with a
fixed env set, the log's contract lines quoted before/after, and (for GPU
changes) a `replay --compare` gate. This section is the run book: how logging
is built, what a full verbose run sets, which lines form the parseable
contract, what each `VICTORY_*` debug variable does, and which workflow each
failure class gets. Nothing here replaces `docs/replay.md` +
`docs/README-fxceless.md` "Debugging" — it indexes them into one procedure.

### 13.1 Logging architecture (`victory-core/src/logging.rs`)

One `tracing` setup for every binary, `init(default_level)` at startup:
filter = `VICTORY_LOG`, else `RUST_LOG`, else the caller's default
(EnvFilter syntax, e.g. `VICTORY_LOG=victory_loader=trace,info`). Build cap:
diagnostics → `TRACE`, plain release → `INFO` (a release build ignores
debug/trace whatever the filter asks). Sinks: stderr (targets on) plus
`VICTORY_LOG_FILE=<path>` appended without colours — `victory play` names one
per start; open retries 20× at 50 ms (virus scanners hold new files) then warns
once and continues window-only. Every thread's stacks / watches / crashes go to
the same stream — one file per run is the unit of evidence. Keep the log
outside the repo (asset policy + §3: no logs, captures, timings, absolute user
paths committed).

### 13.2 Switch semantics (`victory-gpu/src/switch.rs`)

`on(name)` = set AND not `"0"` (unset and `0`/empty-off are both off; any other
value is on — historical "exists means on" is gone, so a leftover `=0` from an
A/B comparison now means off). `toggled!(NAME)` (diagnostics): on-unless-`0`,
overridable mid-run by `VICTORY_AB` A/B phases. Every `VICTORY_*` below follows
`on()` unless its comment says otherwise (document the exception where read +
launcher table if user-facing, §4 rule). All debug switches are read through
`OnceLock`: mid-run `set` does not retarget a running process — restart.

### 13.3 The full verbose test run (protocol)

Diag build + warm cache + fixed env, same save, same cache dir every time:

```powershell
cargo build --locked --profile diag -p victory --features diagnostics
$env:VICTORY_LOG_FILE = '<outside-repo>\run-<date>.log'
$env:VICTORY_LOG = 'info'                 # debug/trace only for the subsystem under test
$env:VICTORY_PIPELINE_BOOK = '1'
victory translate --validate --pipelines --list --jobs <n>   # translator gate first
victory warm                                              # needs BOOK runs first
victory upscaler-check [--device] [--present]             # before touching DLSS/FG
scripts\play-foxhound.ps1 -Exe <exe> [-Build diag] [-FrameGen [-FrameGenX N]] [-Dlss ...] [-Fps ...] [-Res ...]
```

Quote in the PR: ≥15 caption intervals ≥1,000 draws/frame (median + p95 + peak
commit + VRAM), the 1 s meter block (§13.4) before/after, and any 10 s line from
the family under test. Perf runs additionally fix DLSS mode + resolution tuple
(never compare across scene sizes, §10.8) and light tuple (never across
(builder, ambient, water, haze), §11.11). State the scene mismatch explicitly
if the save/cache moved.

### 13.4 The contract lines (stable, parseable — keep them so)

- **Window caption** (`host.rs::Meter`, every presented frame): title caption +
  ` | VIctory {fps:.0} fps {ms:.1} ms/frame {draws:.0} draws/frame` (frame-gen:
  `fps (frame gen X fps)`). Feeds `menu.rs::fps_line` (`FPS {x:.1}` /
  `FRAME GEN {n}X SHOWN {y:.1} FPS`) and `panel.rs` first line
  (`FPS {x:.1} FRAME {ms:.1} MS LONGEST {w:.0}` + frame-gen tail).
- **Every second** (same `Meter::frame`): indirect-mesh group counts + empty
  budget records; heap merge line; shadow exact line; descriptor sync line;
  late/relay line; queue-behind line; streamline line (FG present/past-state/upscale
  ms/copy-HUD-less ms, §10/11); card-clock line (`nothing queued {idle} ms,
  ~{busy}% busy`); two-queues line; `spans` (diag); shadow back-copy line;
  submit-split line; fresh-open line (`VICTORY_FRESH_OPEN=0` behaviour); heap
  marks; `split` (diag); card hash/compare passes + write-combined leftovers;
  texture refresh waits (own/asking/watch/reading split); no-wait/exact
  refresh splits; title-buffer copy joins; heap-buffer picked passes + twin
  copies + planes + card copies; watch slow-asks (diag); pipeline-drop reasons
  (diag, top 40 `dropped {n}/frame without a host pipeline: {why}`).
- **Every 5 s** (`logged` gate): placed-view keeps; `tiles::status`; heap
  texture queue (queued/due/oldest/away/partly + start totals + stale/mapped
  ranges); texture room (`used − free-blocks + buffer-room + target-room`
  vs line + % of budget → `room` / `no room, new textures wait`); picked-pass
  timings; queue lock takes/waits; first-serve latencies; show-new-later count.
- **Power-of-two frames** (`queue.rs` present path): `frame presented (since
  the last line: draws and dispatches, and those run on the host)` + (≥256)
  `shaders from the cache and translated so far` + early-depth split
  (`given/off/exports`, §11.8). Only totals, queue-waits and relay copies are
  ever expected in replay `notes.txt`.
- **Every 10 s**: `memory*` family (`copystat::log_memory`: process/private,
  working, free/commit-left, copies, windows, snapshots, staging, textures,
  made-so-far upload/shared/card, placed-filler twins, holders, lists, views,
  own-allocs by size band, map walk every 6th line on its own thread);
  `title memory by address` (§12.4); ambient reflections line (§11.4);
  `TIMER_LOG=1` pacing bands (`frame.rs`); `GPU_TIMING` slow lists/commands.
- **Event lines** (warn, act on sight): `a black frame after a frame that was
  not` (+ per-draw `black frame {n}: {d}`, +6-frame `levels after…` fade check);
  `host GPU device removed after …` / `device removed:` + Streamline status-4/5/6/7
  + `slShutdown` note; `the ray-tracing structure builder runs/rests/stays off`
  (§12.8); `a stage … which is off`; `stand-in: … other tables / another image /
  stencil plane` (§11.5); `presentCommon() was not observed` (§10.8).

### 13.5 Variable-logging reference (what each knob records)

```
Core:   VICTORY_LOG=<target>=debug,...   VICTORY_LOG_FILE=<outside-repo path>
        VICTORY_DXIL=<dxil.dll> (test/offline validator only)
Loader: VICTORY_STACKS=20,60 (every thread's stacks at those seconds)
        VICTORY_WATCH=<rva>,... (log a title u32 on change)
        VICTORY_POKE=<rva>:<byte>,... (write a title byte 2x/s)
        VICTORY_KEYS=30:Enter,35:X (press keys at those seconds)
        VICTORY_CRASH_SAVE=<rva>:<len>,... (title memory to cache dir on crash; crashes.log in data dir)
Frames: VICTORY_FPS_LINE=0/1  VICTORY_TITLE_FPS  VICTORY_TIMER_LOG=1 (5 s pacing bands)
        VICTORY_CAPTURE=<dir> VICTORY_CAPTURE_FRAMES=<list> VICTORY_CAPTURE_WORLD[=_ALSO]
        VICTORY_DUMP_TARGETS=1 (render targets of listed frames)  VICTORY_AUDIO_CAPTURE=<wav>  VICTORY_XMA_DUMP=<dir>
Record: VICTORY_RECORD=<dir> VICTORY_RECORD_AT=<s|fN> VICTORY_RECORD_FRAMES=<n> VICTORY_RECORD_ALWAYS=1 (diag)
GPU:    VICTORY_GPU_TIMING=1 (per-list timing + slow commands)  VICTORY_CARD_AWAKE[=turns]  VICTORY_STREAMLINE_LOG=<dir>
        VICTORY_D3D12_DEBUG=1 (debug layer)  VICTORY_D3D12_GBV=1 (+validation)  VICTORY_D3D12_WARP=1 (software adapter)
        VICTORY_D3D12_DRED[=light|=faults] (allocations + in-flight commands / breadcrumbs / fault data)
        VICTORY_D3D12_BACKGROUND=0 (no runtime background shader work)
Gates:  VICTORY_PIPELINE_WAIT_MS=<ms> (default 500)  VICTORY_SKIP_PIPELINES=<hash>,... (diag; LEAVE_OUT is the ship twin)
        VICTORY_SHADER_LOOP_LIMIT=<n> (loops stop after n/lane, own cache keys; faster world at lower n = loop-bound)
Memory: VICTORY_MEMORY_MAP=0 (skip the map walk)  VICTORY_UPCOPY_CHECK=1 (live --upcopies)  VICTORY_MERGE_RACE=1
        VICTORY_WALLS_TRACE VICTORY_CARD_ACCOUNTING + texture-sweep census  VICTORY_EXACT_BACK  VICTORY_REFRESH_CHECK=1 (~100 ms/frame)
Pipes:  VICTORY_PIPELINE_BOOK=1 (write-down for `warm`)  VICTORY_BARRIER_TEST=groups|indirect|both (timing only)
```

Audio/video capture variables sit beside the frame ones; census/profile tools
below are commands, not env. Anything absent from this table is read where? —
`grep VICTORY_` + the comment at the read site is normative.

### 13.6 Debugging workflows (which tool for which symptom)

1. **Black / wrong picture:** `DUMP_TARGETS=1` + listed-frame capture → which
   target diverged; `replay <dir> --pixel X,Y [--of WxH]` names the first command
   per pipeline that moved the texel; `--targets-only` if UAV noise drowns it;
   `--images <dir>` for least/mean/max per written texture. Black-frame warns
   (§13.4) distinguish fade (levels line) from single drop (draw list names it).
2. **Slow frame:** `GPU_TIMING=1` line (slow lists + slowest commands by
   pipeline/DXIL hash) → `replay --timing detailed` median of several loops
   (first loop discarded) → `STREAMLINE_PRESENT=1` if `presentCommon() was not
   observed` appears → `SHADER_LOOP_LIMIT=<n>` ladder if a lower n jumps fps
   (loop-bound proof) → `BARRIER_TEST` / `--merge-barriers` / `--no-barriers`
   only as timing experiments (never as fixes).
3. **Leak / growth:** same-save 10 s `memory*` lines (§9 rules: ≥15 ≥1k-draw
   caption intervals; mapped ≠ private); `MEMORY_MAP=1` walk (biggest groups);
   texture census + `host copies the sweep never lets go`; address-memory line
   for fault-held growth; `UPCOPY_CHECK=1` / `replay --upcopies --check-written`
   for undeclared writer detection.
4. **Missing draws:** diag `dropped … without a host pipeline: {why}` top-40 +
   `left_out`/`SKIP_PIPELINES` membership + `PIPELINE_WAIT_MS` budget + dynamic
   slot waits; `replay --flow/--slots/--listing` for root/heap cause.
5. **Hang / device removed:** `D3D12_DRED` (+`=light` breadcrumbs, `=faults`
   fault data) + `D3D12_DEBUG` errors in the replay report; `STACKS` schedule
   for what the title waited on; `WATCH`/`POKE`/`KEYS` for hang repro; save
   other programs' work first (a diverging replay can take the device).
6. **Divergent replay:** `notes.txt` first (only totals/queue-waits/relay
   expected); `--wait-all/--lists` to rule out waits; `--diverge [--lists
   --commands --step]` for first-differing command; `--no-merge` to expose
   merge-masked diffs; `--retranslate` with this build's switches for
   translator-bisect; never replay a builder capture (hangs the card, §12.8).

### 13.7 Capture and replay files (`docs/replay.md`)

Record path: `RECORD` dir → `capture-<n>` (9 GB scale for title-size frames:
~8 GB card memory at ~2.3 GB/s + ~1 GB CPU writes; title stands still; 3–10 ms
overhead/frame while recording, one capture per process, `frame.bin` = done,
`failed.txt` = not). Replay path: `victory replay <dir> [--loops N] [--timing
detailed]` (timestamps on-card; per-submission + copies-as-CPU-work + gaps; N
loops median; `replay.png` vs presented). Pixel path (`--pixel`), runs-differ
path (`--diverge*`, atomics merge in `merge.rs`, `--no-merge` to disable),
introspection path (`--flow/--slots/--watch/--images/--dump/--until/--only/
--listing/--one-submit/--batch-clamps/--skip-clamps/--no-barriers/--buffers/
--pace/--awake/--breadcrumbs/--debug-layer`). Limits (recorded-but-reported):
reserved resources, bundles, ray tracing, render passes, enhanced barriers,
predication, stream output, `ClearUnorderedAccessView`, shading-rate, newer
creation methods; multisample/CPU-visible/>256 MB-level contents not kept;
aliasing = no contents; addresses not rewritten (indirect args reported, may be
wrong). Tests: `cargo test -p victory-gpu --features diagnostics`
(`tests/replay.rs` byte-compare; `VICTORY_TEST_D3D_DEBUG=1` adds the layer;
`replay_big.rs --ignored`).

### 13.8 Offline inspection (no title needed)

`victory translate --validate --pipelines --list --jobs N` (translator gate;
`shaders`/`warm` around it); `victory shader-info [paths] [--verbose]
[--disasm]`; `victory census [paths] [--missing] [--strict]`;
`victory inspect --imports/--exports`, `lib-exports`; `victory upscaler-check
[--device] [--present]`; `scripts/faultbook.py` for fault-book triage;
`cargo test --workspace` (no game data; title paths via env, skipped when
unset, asserts properties never snapshots). Results that change shader bytes
invalidate warm caches — say so in the PR (§4).

### 13.9 Hygiene (logs are evidence, not artefacts)

`VICTORY_LOG_FILE`, captures, replays, `*.log`, timings and `crashes.log`
copies never enter the repo (§6 asset policy; `.gitignore` + `policy` gate);
absolute user paths never in committed text; per-run filenames carry date +
switch tuple so before/after pairs cannot be mixed; `victory-settings.txt`
changes ride with the PR that needs them. Debug-only code paths stay behind
`cfg(feature = "diagnostics")` — a plain release build must not grow record,
replay, DRED or trace surfaces.

### 13.10 Logging roadmap (do in this order)

1. Contract-test the §13.4 lines (a test that fails when a wording drifts —
   the perf claims in reviews grep these).
2. One `--dump-log-contract` run mode emitting the meter + 10 s lines as JSON
   beside the text (plotting without regex).
3. Replay-report schema versioning (`notes.txt` + report fields) before any new
   `--flow/--pixel` output is added — downstream triage scripts break silently
   today.

## 14. Appendix — doc map


- Start: `README.md` → `docs/README-foxhound.md` → `docs/README-fxceless.md`.
- Run: `docs/dlss.md`, `scripts/play-foxhound.ps1`, `apps/victory/src/play.rs`.
- Understand: `docs/architecture.md`, `docs/findings.md`,
  `docs/fork-ray-tracing.md`, `docs/replay.md`, `docs/bug-journal.md`.
- Obey: `docs/asset-policy.md`, `xtask/src/policy.rs`, `LICENSE`.
