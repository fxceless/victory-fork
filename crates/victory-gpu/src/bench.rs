//! Deterministic traversal benchmark telemetry and clean-exit hooks.
//!
//! All collection and process-exit behavior is opt-in via environment
//! variables and disabled by default:
//! - `VICTORY_BENCH_OUT=<abs_path.json>`: write structured JSON metrics
//!   atomically outside the repository (in-repo paths are rejected).
//! - `VICTORY_BENCH_WARMUP=<W>`: world frames excluded from stats (default 30).
//! - `VICTORY_BENCH_EXIT_FRAMES=<N>`: flush metrics and exit cleanly (`0`)
//!   after `warmup + N` world frames.
//! - `VICTORY_BENCH_CHECK_EVERY=<K>`: emit periodic sanity heartbeats every
//!   `K` measured world frames (zero-draw and black-frame counts are tracked on
//!   every measured world frame whenever benchmarking is active).
//!
//! [`world_frame`] is always updated with minimal atomic overhead at each
//! present so input recording/replay can key events deterministically to world
//! frames regardless of frame rate.

use std::path::{Component, Path, PathBuf};
#[cfg(windows)]
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
#[cfg(windows)]
use std::{sync::OnceLock, time::Instant};

/// Marker file at the root of any VIctory source checkout.
const ROOT_MARKER: &str = ".victory-root";

/// Draw-call threshold in a single presented frame that latches world-frame
/// counting (matches the world-frame threshold in `capture::frame_drew`).
pub const WORLD_DRAW_THRESHOLD: u64 = 500;

/// Default number of initial world frames excluded as warmup.
pub const DEFAULT_WARMUP_FRAMES: u64 = 30;

/// Monotonic world-frame counter: `0` before the first frame with at least
/// [`WORLD_DRAW_THRESHOLD`] draws, then `1, 2, 3, ...` on every presented frame.
static WORLD_FRAME: AtomicU64 = AtomicU64::new(0);

/// Cumulative host draws observed at the previous present.
#[cfg(windows)]
static PREV_DRAWN: AtomicU64 = AtomicU64::new(0);

/// Whether the host GPU device reported removal during this run.
#[cfg(windows)]
static DEVICE_REMOVED: AtomicBool = AtomicBool::new(false);

/// Optional digest of the active replay input script (set by `victory-hle`).
static INPUT_SCRIPT_SHA256: Mutex<Option<String>> = Mutex::new(None);

/// Monotonic world frame count (`0` during menus/loading before the world
/// latches at `>= 500` draws in a frame, then `1, 2, 3, ...`).
pub fn world_frame() -> u64 {
    WORLD_FRAME.load(Ordering::Acquire)
}

/// Records the digest of the active input script for inclusion in the
/// benchmark JSON report.
pub fn set_input_script_sha256(digest: impl Into<String>) {
    let digest = digest.into();
    let mut guard = INPUT_SCRIPT_SHA256
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    *guard = (!digest.is_empty()).then_some(digest);
}

/// Hashes `bytes` with `blake3`, stores the 64-character hex digest for the
/// benchmark report, and returns it.
pub fn register_input_script_bytes(bytes: &[u8]) -> String {
    let hex = blake3::hash(bytes).to_hex().to_string();
    set_input_script_sha256(hex.clone());
    hex
}

/// Returns the currently registered input-script digest, if any.
pub fn input_script_sha256() -> Option<String> {
    INPUT_SCRIPT_SHA256
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
}

/// Source tree root this crate was compiled from (`crates/victory-gpu -> ../..`).
fn compiled_workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

/// Validates that `path` is absolute and lies strictly outside every VIctory
/// source checkout (neither beneath the compiled workspace root nor beneath any
/// ancestor containing `.victory-root`).
pub fn ensure_outside_repo(path: &Path) -> Result<PathBuf, String> {
    if !path.is_absolute() {
        return Err(format!(
            "refusing relative output path {}: use an absolute path outside the repository",
            path.display()
        ));
    }
    let resolved = resolve_path(path)?;
    if let Ok(root) = std::fs::canonicalize(compiled_workspace_root())
        && resolved.starts_with(&root)
    {
        return Err(format!(
            "refusing to write {} inside the VIctory repository ({})",
            path.display(),
            root.display()
        ));
    }
    for ancestor in resolved.ancestors() {
        if ancestor.join(ROOT_MARKER).is_file() {
            return Err(format!(
                "refusing to write {} inside a VIctory repository ({})",
                path.display(),
                ancestor.display()
            ));
        }
    }
    Ok(resolved)
}

/// Canonicalizes the longest existing prefix of `path` and lexically
/// normalizes any non-existing tail components.
fn resolve_path(path: &Path) -> Result<PathBuf, String> {
    let mut existing = path;
    let mut rest = Vec::new();
    loop {
        match std::fs::canonicalize(existing) {
            Ok(mut canon) => {
                for component in rest.iter().rev() {
                    canon.push(component);
                }
                return Ok(normalize_lexical(&canon));
            }
            Err(err) => match (existing.parent(), existing.file_name()) {
                (Some(parent), Some(name)) => {
                    rest.push(name.to_os_string());
                    existing = parent;
                }
                _ => {
                    return Err(format!("cannot resolve {}: {err}", path.display()));
                }
            },
        }
    }
}

fn normalize_lexical(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            c => out.push(c),
        }
    }
    out
}

/// Cumulative `framelog` counter snapshot in microseconds (except `guard_faults`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FrameLogTotals {
    pub prepare_us: u64,
    pub card_wait_us: u64,
    pub card_idle_us: u64,
    pub copy_us: u64,
    pub copyback_us: u64,
    pub remap_us: u64,
    pub refresh_us: u64,
    pub pipeline_wait_us: u64,
    pub guard_faults: u64,
    pub card_sync_us: u64,
    pub card_check_us: u64,
    pub card_lists_us: u64,
}

/// Per-frame averaged `framelog` stage costs (in milliseconds, except
/// `guard_faults` which is average faults per world frame).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameLogAverages {
    pub prepare_ms: f64,
    pub card_wait_ms: f64,
    pub card_idle_ms: f64,
    pub copy_ms: f64,
    pub copyback_ms: f64,
    pub remap_ms: f64,
    pub refresh_ms: f64,
    pub pipeline_wait_ms: f64,
    pub guard_faults: f64,
    pub card_sync_ms: f64,
    pub card_check_ms: f64,
    pub card_lists_ms: f64,
}

/// Instantaneous memory readings for a frame sample.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MemorySample {
    pub vram_used_mb: f64,
    pub vram_budget_mb: f64,
    pub ram_private_commit_mb: f64,
    pub ram_working_set_mb: f64,
}

/// Peak memory summary over the measured window.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MemorySummary {
    pub vram_used_peak_mb: f64,
    pub vram_budget_mb: f64,
    pub ram_private_commit_peak_mb: f64,
    pub ram_working_set_peak_mb: f64,
}

/// Frame-time distribution statistics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct FrameTimeStats {
    pub avg: f64,
    pub median: f64,
    pub p95: f64,
    pub p99: f64,
    pub low_5pct_fps: f64,
    pub low_1pct_fps: f64,
}

/// Draw-call distribution statistics over measured world frames.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DrawStats {
    pub avg: f64,
    pub min: u64,
    pub max: u64,
}

/// Complete structured benchmark report written to `VICTORY_BENCH_OUT`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct BenchReport {
    pub world_frames: u64,
    pub warmup_frames: u64,
    pub frame_ms: FrameTimeStats,
    pub draws: DrawStats,
    pub device_removed: bool,
    pub zero_draw_world_frames: u64,
    pub black_frames: u64,
    pub memory: MemorySummary,
    pub framelog: FrameLogAverages,
    pub input_script_sha256: Option<String>,
}

/// Computes `avg`, `median`, `p95`, `p99`, `low_5pct_fps`, and `low_1pct_fps`
/// from raw per-frame durations in milliseconds.
pub fn compute_frame_stats(samples: &[f64]) -> FrameTimeStats {
    let mut vals: Vec<f64> = samples
        .iter()
        .copied()
        .filter(|v| v.is_finite() && *v >= 0.0)
        .collect();
    if vals.is_empty() {
        return FrameTimeStats::default();
    }
    let n = vals.len();
    let avg = vals.iter().sum::<f64>() / n as f64;
    vals.sort_by(f64::total_cmp);

    let median = if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) * 0.5
    };
    let p95 = percentile_sorted(&vals, 0.95);
    let p99 = percentile_sorted(&vals, 0.99);
    let low_5pct_fps = tail_low_fps(&vals, 0.05);
    let low_1pct_fps = tail_low_fps(&vals, 0.01);

    FrameTimeStats {
        avg,
        median,
        p95,
        p99,
        low_5pct_fps,
        low_1pct_fps,
    }
}

fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let pos = p.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = pos - lo as f64;
    sorted[lo] + (sorted[hi] - sorted[lo]) * frac
}

/// Average FPS of the slowest `fraction` of frames (e.g. `0.01` for 1% low).
fn tail_low_fps(sorted_asc: &[f64], fraction: f64) -> f64 {
    if sorted_asc.is_empty() {
        return 0.0;
    }
    let k = ((sorted_asc.len() as f64 * fraction).ceil() as usize).clamp(1, sorted_asc.len());
    let tail = &sorted_asc[sorted_asc.len() - k..];
    let mean_ms = tail.iter().sum::<f64>() / tail.len() as f64;
    if mean_ms > 0.0 {
        1000.0 / mean_ms
    } else {
        0.0
    }
}

/// Pure-std JSON serialization for [`BenchReport`].
impl BenchReport {
    pub fn to_json(&self) -> String {
        let sha = match &self.input_script_sha256 {
            Some(s) => format!("\"{}\"", escape_json(s)),
            None => "null".to_owned(),
        };
        format!(
            concat!(
                "{{\n",
                "  \"world_frames\": {},\n",
                "  \"warmup_frames\": {},\n",
                "  \"frame_ms\": {{\n",
                "    \"avg\": {:.4},\n",
                "    \"median\": {:.4},\n",
                "    \"p95\": {:.4},\n",
                "    \"p99\": {:.4},\n",
                "    \"low_5pct_fps\": {:.4},\n",
                "    \"low_1pct_fps\": {:.4}\n",
                "  }},\n",
                "  \"draws\": {{\n",
                "    \"avg\": {:.4},\n",
                "    \"min\": {},\n",
                "    \"max\": {}\n",
                "  }},\n",
                "  \"device_removed\": {},\n",
                "  \"zero_draw_world_frames\": {},\n",
                "  \"black_frames\": {},\n",
                "  \"memory\": {{\n",
                "    \"vram_used_peak_mb\": {:.2},\n",
                "    \"vram_budget_mb\": {:.2},\n",
                "    \"ram_private_commit_peak_mb\": {:.2},\n",
                "    \"ram_working_set_peak_mb\": {:.2}\n",
                "  }},\n",
                "  \"framelog\": {{\n",
                "    \"prepare_ms\": {:.4},\n",
                "    \"card_wait_ms\": {:.4},\n",
                "    \"card_idle_ms\": {:.4},\n",
                "    \"copy_ms\": {:.4},\n",
                "    \"copyback_ms\": {:.4},\n",
                "    \"remap_ms\": {:.4},\n",
                "    \"refresh_ms\": {:.4},\n",
                "    \"pipeline_wait_ms\": {:.4},\n",
                "    \"guard_faults\": {:.4},\n",
                "    \"card_sync_ms\": {:.4},\n",
                "    \"card_check_ms\": {:.4},\n",
                "    \"card_lists_ms\": {:.4}\n",
                "  }},\n",
                "  \"input_script_sha256\": {}\n",
                "}}\n",
            ),
            self.world_frames,
            self.warmup_frames,
            self.frame_ms.avg,
            self.frame_ms.median,
            self.frame_ms.p95,
            self.frame_ms.p99,
            self.frame_ms.low_5pct_fps,
            self.frame_ms.low_1pct_fps,
            self.draws.avg,
            self.draws.min,
            self.draws.max,
            self.device_removed,
            self.zero_draw_world_frames,
            self.black_frames,
            self.memory.vram_used_peak_mb,
            self.memory.vram_budget_mb,
            self.memory.ram_private_commit_peak_mb,
            self.memory.ram_working_set_peak_mb,
            self.framelog.prepare_ms,
            self.framelog.card_wait_ms,
            self.framelog.card_idle_ms,
            self.framelog.copy_ms,
            self.framelog.copyback_ms,
            self.framelog.remap_ms,
            self.framelog.refresh_ms,
            self.framelog.pipeline_wait_ms,
            self.framelog.guard_faults,
            self.framelog.card_sync_ms,
            self.framelog.card_check_ms,
            self.framelog.card_lists_ms,
            sha,
        )
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

/// Sequence counter ensuring unique temp filenames for concurrent atomic writes.
static WRITE_SEQ: AtomicU64 = AtomicU64::new(0);

/// Atomically writes `report` JSON to `out_path` via `<out_path>.tmp.<pid>.<seq>` +
/// rename after verifying `out_path` is strictly outside the repository.
pub fn write_report_atomic(out_path: &Path, report: &BenchReport) -> Result<(), String> {
    let safe_path = ensure_outside_repo(out_path)?;
    if let Some(parent) = safe_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create parent dir {}: {e}", parent.display()))?;
    }
    let pid = std::process::id();
    let seq = WRITE_SEQ.fetch_add(1, Ordering::Relaxed);
    let file_name = safe_path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| format!("invalid output file name: {}", safe_path.display()))?;
    let tmp_path = safe_path.with_file_name(format!("{file_name}.tmp.{pid}.{seq}"));
    let body = report.to_json();
    use std::io::Write as _;
    let mut file = std::fs::File::create_new(&tmp_path)
        .map_err(|e| format!("create temp report {}: {e}", tmp_path.display()))?;
    let written = file.write_all(body.as_bytes());
    drop(file);
    if let Err(error) = written {
        let _ = std::fs::remove_file(&tmp_path);
        return Err(format!("write temp report {}: {error}", tmp_path.display()));
    }
    std::fs::rename(&tmp_path, &safe_path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp_path);
        format!(
            "rename temp report {} -> {}: {e}",
            tmp_path.display(),
            safe_path.display()
        )
    })
}

/// Configuration for a benchmark collector instance.
#[derive(Clone, Debug)]
pub struct BenchConfig {
    pub out_path: Option<PathBuf>,
    pub warmup_frames: u64,
    pub exit_frames: Option<u64>,
    pub check_every: Option<u64>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            out_path: None,
            warmup_frames: DEFAULT_WARMUP_FRAMES,
            exit_frames: None,
            check_every: None,
        }
    }
}

impl BenchConfig {
    pub fn is_active(&self) -> bool {
        self.out_path.is_some() || self.exit_frames.is_some()
    }

    #[cfg(windows)]
    fn from_env() -> Self {
        let out_path = std::env::var("VICTORY_BENCH_OUT")
            .ok()
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty())
            .and_then(|raw| {
                let p = PathBuf::from(&raw);
                match ensure_outside_repo(&p) {
                    Ok(valid) => Some(valid),
                    Err(msg) => {
                        tracing::error!("VICTORY_BENCH_OUT ignored: {msg}");
                        None
                    }
                }
            });
        let warmup_frames = std::env::var("VICTORY_BENCH_WARMUP")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .unwrap_or(DEFAULT_WARMUP_FRAMES);
        let exit_frames = std::env::var("VICTORY_BENCH_EXIT_FRAMES")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0);
        let check_every = std::env::var("VICTORY_BENCH_CHECK_EVERY")
            .ok()
            .and_then(|s| s.trim().parse::<u64>().ok())
            .filter(|&n| n > 0);
        Self {
            out_path,
            warmup_frames,
            exit_frames,
            check_every,
        }
    }
}

/// Deterministic state machine for world-frame latching, warmup exclusion, and
/// metric accumulation (instantiated globally for runtime use and directly in
/// unit tests).
pub struct BenchCollector {
    config: BenchConfig,
    world_frame: u64,
    prev_drawn: u64,
    warmup_seen: u64,
    frame_ms: Vec<f64>,
    draws: Vec<u64>,
    zero_draw_world_frames: u64,
    black_frames: u64,
    device_removed: bool,
    memory: MemorySummary,
    prev_totals: FrameLogTotals,
    measured_start_totals: Option<FrameLogTotals>,
    measured_end_totals: FrameLogTotals,
}

/// Outcome of recording one presented frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StepOutcome {
    pub world_frame: u64,
    pub measured_frames: u64,
    pub should_exit: bool,
}

impl BenchCollector {
    pub fn new(config: BenchConfig) -> Self {
        Self {
            config,
            world_frame: 0,
            prev_drawn: 0,
            warmup_seen: 0,
            frame_ms: Vec::new(),
            draws: Vec::new(),
            zero_draw_world_frames: 0,
            black_frames: 0,
            device_removed: false,
            memory: MemorySummary::default(),
            prev_totals: FrameLogTotals::default(),
            measured_start_totals: None,
            measured_end_totals: FrameLogTotals::default(),
        }
    }

    pub fn world_frame(&self) -> u64 {
        self.world_frame
    }

    pub fn note_device_removed(&mut self) {
        self.device_removed = true;
    }

    /// Advances the collector by one presented frame.
    pub fn record_present(
        &mut self,
        drawn_total: u64,
        dt_ms: Option<f64>,
        black_probe: Option<bool>,
        device_removed: bool,
        mem: MemorySample,
        totals: FrameLogTotals,
    ) -> StepOutcome {
        let frame_draws = drawn_total.saturating_sub(self.prev_drawn);
        self.prev_drawn = drawn_total;
        if device_removed {
            self.device_removed = true;
        }

        if self.world_frame == 0 {
            if frame_draws >= WORLD_DRAW_THRESHOLD {
                self.world_frame = 1;
            } else {
                self.prev_totals = totals;
                return StepOutcome {
                    world_frame: 0,
                    measured_frames: 0,
                    should_exit: false,
                };
            }
        } else {
            self.world_frame += 1;
        }

        if self.world_frame <= self.config.warmup_frames {
            self.warmup_seen = self.world_frame;
            self.prev_totals = totals;
            return StepOutcome {
                world_frame: self.world_frame,
                measured_frames: 0,
                should_exit: false,
            };
        }

        if self.measured_start_totals.is_none() {
            self.measured_start_totals = Some(self.prev_totals);
        }
        self.measured_end_totals = totals;
        self.prev_totals = totals;

        if let Some(ms) = dt_ms.filter(|v| v.is_finite() && *v >= 0.0) {
            self.frame_ms.push(ms);
        }
        self.draws.push(frame_draws);
        if frame_draws == 0 {
            self.zero_draw_world_frames += 1;
        }
        if black_probe == Some(true) {
            self.black_frames += 1;
        }

        if mem.vram_used_mb > self.memory.vram_used_peak_mb {
            self.memory.vram_used_peak_mb = mem.vram_used_mb;
        }
        if mem.vram_budget_mb > 0.0 {
            self.memory.vram_budget_mb = mem.vram_budget_mb;
        }
        if mem.ram_private_commit_mb > self.memory.ram_private_commit_peak_mb {
            self.memory.ram_private_commit_peak_mb = mem.ram_private_commit_mb;
        }
        if mem.ram_working_set_mb > self.memory.ram_working_set_peak_mb {
            self.memory.ram_working_set_peak_mb = mem.ram_working_set_mb;
        }

        let measured = self.draws.len() as u64;
        if let Some(every) = self.config.check_every
            && every > 0
            && measured.is_multiple_of(every)
        {
            tracing::info!(
                world_frame = self.world_frame,
                measured,
                frame_draws,
                zero_draw_world_frames = self.zero_draw_world_frames,
                black_frames = self.black_frames,
                vram_used_peak_mb = format_args!("{:.1}", self.memory.vram_used_peak_mb),
                "bench checkpoint"
            );
        }

        let should_exit = self
            .config
            .exit_frames
            .is_some_and(|target| measured >= target);
        StepOutcome {
            world_frame: self.world_frame,
            measured_frames: measured,
            should_exit,
        }
    }

    pub fn report(&self, input_script_sha256: Option<String>) -> BenchReport {
        let world_frames = self.draws.len() as u64;
        let draws = if self.draws.is_empty() {
            DrawStats::default()
        } else {
            let sum: u64 = self.draws.iter().copied().sum();
            let min = self.draws.iter().copied().min().unwrap_or(0);
            let max = self.draws.iter().copied().max().unwrap_or(0);
            DrawStats {
                avg: sum as f64 / self.draws.len() as f64,
                min,
                max,
            }
        };
        let framelog = match (self.measured_start_totals, world_frames) {
            (Some(start), n) if n > 0 => {
                let denom_ms = (n as f64) * 1000.0;
                let denom = n as f64;
                let end = self.measured_end_totals;
                let d = |a: u64, b: u64| a.saturating_sub(b) as f64;
                FrameLogAverages {
                    prepare_ms: d(end.prepare_us, start.prepare_us) / denom_ms,
                    card_wait_ms: d(end.card_wait_us, start.card_wait_us) / denom_ms,
                    card_idle_ms: d(end.card_idle_us, start.card_idle_us) / denom_ms,
                    copy_ms: d(end.copy_us, start.copy_us) / denom_ms,
                    copyback_ms: d(end.copyback_us, start.copyback_us) / denom_ms,
                    remap_ms: d(end.remap_us, start.remap_us) / denom_ms,
                    refresh_ms: d(end.refresh_us, start.refresh_us) / denom_ms,
                    pipeline_wait_ms: d(end.pipeline_wait_us, start.pipeline_wait_us) / denom_ms,
                    guard_faults: d(end.guard_faults, start.guard_faults) / denom,
                    card_sync_ms: d(end.card_sync_us, start.card_sync_us) / denom_ms,
                    card_check_ms: d(end.card_check_us, start.card_check_us) / denom_ms,
                    card_lists_ms: d(end.card_lists_us, start.card_lists_us) / denom_ms,
                }
            }
            _ => FrameLogAverages::default(),
        };

        BenchReport {
            world_frames,
            warmup_frames: self.warmup_seen,
            frame_ms: compute_frame_stats(&self.frame_ms),
            draws,
            device_removed: self.device_removed,
            zero_draw_world_frames: self.zero_draw_world_frames,
            black_frames: self.black_frames,
            memory: self.memory,
            framelog,
            input_script_sha256,
        }
    }
}

#[cfg(windows)]
struct RuntimeState {
    collector: BenchCollector,
    last_present: Option<Instant>,
    first_measured_frame: Option<u64>,
}

#[cfg(windows)]
static CONFIG: OnceLock<BenchConfig> = OnceLock::new();
#[cfg(windows)]
static RUNTIME: Mutex<Option<RuntimeState>> = Mutex::new(None);

#[cfg(windows)]
fn config() -> &'static BenchConfig {
    CONFIG.get_or_init(BenchConfig::from_env)
}

#[cfg(windows)]
pub(crate) fn active() -> bool {
    config().is_active()
}

#[cfg(windows)]
fn sample_memory() -> MemorySample {
    use windows_sys::Win32::System::ProcessStatus::{
        K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS, PROCESS_MEMORY_COUNTERS_EX,
    };
    use windows_sys::Win32::System::Threading::GetCurrentProcess;

    let mut out = MemorySample::default();
    if let Some(host) = crate::host::host()
        && let Some(m) = host.video_memory()
    {
        out.vram_budget_mb = m[0] as f64 / (1024.0 * 1024.0);
        out.vram_used_mb = m[1] as f64 / (1024.0 * 1024.0);
    }
    let mut c = PROCESS_MEMORY_COUNTERS_EX {
        cb: size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32,
        ..Default::default()
    };
    // SAFETY: our own process handle and a properly sized PROCESS_MEMORY_COUNTERS_EX
    // passed through its header pointer as documented for K32GetProcessMemoryInfo.
    let ok = unsafe {
        K32GetProcessMemoryInfo(
            GetCurrentProcess(),
            (&raw mut c).cast::<PROCESS_MEMORY_COUNTERS>(),
            c.cb,
        )
    };
    if ok != 0 {
        out.ram_private_commit_mb = c.PrivateUsage as f64 / (1024.0 * 1024.0);
        out.ram_working_set_mb = c.WorkingSetSize as f64 / (1024.0 * 1024.0);
    }
    out
}

#[cfg(windows)]
fn check_device_removed() -> bool {
    if DEVICE_REMOVED.load(Ordering::Acquire) {
        return true;
    }
    if crate::descriptor::host::device_removed().is_some() {
        DEVICE_REMOVED.store(true, Ordering::Release);
        return true;
    }
    false
}

/// Called from `host::fail` when D3D12 reports a device removal. Marks
/// `device_removed = true` and immediately flushes the benchmark JSON if
/// `VICTORY_BENCH_OUT` is active.
#[cfg(windows)]
pub(crate) fn on_device_removed() {
    DEVICE_REMOVED.store(true, Ordering::Release);
    let cfg = config();
    let Some(out_path) = cfg.out_path.as_ref() else {
        return;
    };
    let report = {
        let mut guard = RUNTIME.lock().unwrap_or_else(|e| e.into_inner());
        let state = guard.get_or_insert_with(|| RuntimeState {
            collector: BenchCollector::new(cfg.clone()),
            last_present: None,
            first_measured_frame: None,
        });
        state.collector.note_device_removed();
        state.collector.report(input_script_sha256())
    };
    if let Err(err) = write_report_atomic(out_path, &report) {
        tracing::error!("bench report flush on device_removed failed: {err}");
    }
}

#[cfg(windows)]
fn advance_world_frame(drawn_total: u64) {
    let prev = PREV_DRAWN.swap(drawn_total, Ordering::Relaxed);
    let enough_draws = drawn_total.saturating_sub(prev) >= WORLD_DRAW_THRESHOLD;
    let _ = WORLD_FRAME.fetch_update(Ordering::Release, Ordering::Relaxed, |frame| {
        (frame != 0 || enough_draws).then(|| frame.saturating_add(1))
    });
}

/// Called once on either the immediate or queued presentation path. When
/// benchmarking is disabled, only the input script clock advances.
#[cfg(windows)]
pub(crate) fn on_present(frame: u64, drawn_total: u64, black_probe: Option<bool>) {
    let cfg = config();
    if !cfg.is_active() {
        advance_world_frame(drawn_total);
        return;
    }

    // Serialize benchmark samples without holding RUNTIME across host calls,
    // which can invoke the fatal device-removal hook on this or another thread.
    static PRESENT: Mutex<()> = Mutex::new(());
    let _present = PRESENT.lock().unwrap_or_else(|e| e.into_inner());
    advance_world_frame(drawn_total);
    let now = Instant::now();
    let mem = sample_memory();
    let totals = crate::framelog::snapshot_totals();
    let removed = check_device_removed();

    let (outcome, first_measured_frame) = {
        let mut guard = RUNTIME.lock().unwrap_or_else(|e| e.into_inner());
        let state = guard.get_or_insert_with(|| RuntimeState {
            collector: BenchCollector::new(cfg.clone()),
            last_present: None,
            first_measured_frame: None,
        });
        let dt_ms = state.last_present.replace(now).map(|then| (now - then).as_secs_f64() * 1000.0);
        let outcome = state.collector.record_present(drawn_total, dt_ms, black_probe, removed, mem, totals);
        if outcome.measured_frames == 1 {
            state.first_measured_frame = Some(frame);
        }
        (outcome, state.first_measured_frame)
    };

    // At the end only, finish the outstanding readbacks without adding this
    // drain to the measured CPU timings or to the frame-time distribution.
    let completed = outcome.should_exit.then(|| {
        crate::host::settle_for_present();
        crate::framelog::snapshot_totals()
    });
    #[cfg(feature = "diagnostics")]
    let late_black = crate::host::host().map_or(0, |host| host.bench_black_frames(first_measured_frame, frame));
    #[cfg(not(feature = "diagnostics"))]
    let late_black = {
        let _ = first_measured_frame;
        0
    };
    let periodic_flush = cfg.out_path.is_some()
        && cfg.exit_frames.is_none()
        && outcome.measured_frames > 0
        && (outcome.measured_frames == 1 || outcome.measured_frames.is_multiple_of(60));
    let mut guard = RUNTIME.lock().unwrap_or_else(|e| e.into_inner());
    let state = guard.as_mut().expect("the present initialized the collector");
    state.collector.black_frames += late_black;
    if let Some(totals) = completed {
        state.collector.measured_end_totals.card_sync_us = totals.card_sync_us;
        state.collector.measured_end_totals.card_check_us = totals.card_check_us;
        state.collector.measured_end_totals.card_lists_us = totals.card_lists_us;
    }
    let removed = removed || DEVICE_REMOVED.load(Ordering::Acquire);
    if removed {
        state.collector.note_device_removed();
    }
    if let Some(out_path) = cfg.out_path.as_ref()
        && (outcome.should_exit || periodic_flush || removed)
    {
        let report = state.collector.report(input_script_sha256());
        if let Err(err) = write_report_atomic(out_path, &report) {
            tracing::error!("bench report write failed: {err}");
        } else if outcome.should_exit || periodic_flush {
            tracing::info!(
                path = %out_path.display(),
                world_frames = report.world_frames,
                median_ms = format_args!("{:.2}", report.frame_ms.median),
                "bench report flushed"
            );
        }
    }

    if outcome.should_exit {
        tracing::info!(
            world_frame = outcome.world_frame,
            measured_frames = outcome.measured_frames,
            "VICTORY_BENCH_EXIT_FRAMES reached; exiting cleanly"
        );
        std::process::exit(0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_frame_latches_at_500_draws_and_excludes_warmup() {
        let mut c = BenchCollector::new(BenchConfig {
            out_path: None,
            warmup_frames: 2,
            exit_frames: Some(3),
            check_every: None,
        });
        let mem = MemorySample {
            vram_used_mb: 1000.0,
            vram_budget_mb: 8000.0,
            ram_private_commit_mb: 2048.0,
            ram_working_set_mb: 1536.0,
        };
        let mut drawn = 0u64;
        // Pre-world loading frames (< 500 draws each) keep world_frame == 0.
        for _ in 0..4 {
            drawn += 120;
            let s = c.record_present(drawn, Some(33.0), Some(false), false, mem, FrameLogTotals::default());
            assert_eq!(s.world_frame, 0);
            assert_eq!(s.measured_frames, 0);
            assert!(!s.should_exit);
        }
        // Frame with 500 draws latches world_frame = 1 (warmup frame 1 of 2).
        drawn += 500;
        let s = c.record_present(drawn, Some(500.0), Some(true), false, mem, FrameLogTotals { prepare_us: 10_000, ..Default::default() });
        assert_eq!(s.world_frame, 1);
        assert_eq!(s.measured_frames, 0);

        // Warmup frame 2 of 2 (excluded from stats and black/zero counters).
        drawn += 600;
        let t_warm_end = FrameLogTotals {
            prepare_us: 25_000,
            card_wait_us: 40_000,
            guard_faults: 10,
            ..Default::default()
        };
        let s = c.record_present(drawn, Some(400.0), Some(true), false, mem, t_warm_end);
        assert_eq!(s.world_frame, 2);
        assert_eq!(s.measured_frames, 0);

        // Measured frames 1..=3.
        drawn += 1000;
        let s = c.record_present(drawn, Some(10.0), Some(false), false, mem, FrameLogTotals { prepare_us: 27_000, card_wait_us: 45_000, guard_faults: 13, ..Default::default() });
        assert_eq!((s.world_frame, s.measured_frames, s.should_exit), (3, 1, false));

        drawn += 1200;
        let s = c.record_present(drawn, Some(20.0), Some(false), false, MemorySample { vram_used_mb: 1500.0, ..mem }, FrameLogTotals { prepare_us: 30_000, card_wait_us: 52_000, guard_faults: 19, ..Default::default() });
        assert_eq!((s.world_frame, s.measured_frames, s.should_exit), (4, 2, false));

        // 0-draw world frame after latch counts as zero_draw_world_frames.
        let s = c.record_present(drawn, Some(30.0), Some(true), false, mem, FrameLogTotals { prepare_us: 34_000, card_wait_us: 55_000, guard_faults: 25, ..Default::default() });
        assert_eq!((s.world_frame, s.measured_frames, s.should_exit), (5, 3, true));

        let r = c.report(Some("abc123".to_owned()));
        assert_eq!(r.world_frames, 3);
        assert_eq!(r.warmup_frames, 2);
        assert!((r.frame_ms.avg - 20.0).abs() < 1e-6);
        assert!((r.frame_ms.median - 20.0).abs() < 1e-6);
        assert_eq!(r.draws.min, 0);
        assert_eq!(r.draws.max, 1200);
        assert_eq!(r.zero_draw_world_frames, 1);
        assert_eq!(r.black_frames, 1);
        assert!((r.memory.vram_used_peak_mb - 1500.0).abs() < 1e-6);
        // Deltas from t_warm_end: prepare = 9000 us / 3 frames = 3.0 ms/frame.
        assert!((r.framelog.prepare_ms - 3.0).abs() < 1e-6);
        assert!((r.framelog.card_wait_ms - 5.0).abs() < 1e-6);
        assert!((r.framelog.guard_faults - 5.0).abs() < 1e-6);
    }

    #[test]
    fn percentiles_and_low_fps_math_are_exact() {
        // 100 frames: 95 frames at 10 ms, 4 frames at 20 ms, 1 frame at 100 ms.
        let mut samples = vec![10.0; 95];
        samples.extend([20.0; 4]);
        samples.push(100.0);
        let s = compute_frame_stats(&samples);
        // avg = (950 + 80 + 100) / 100 = 11.3 ms
        assert!((s.avg - 11.3).abs() < 1e-6);
        assert!((s.median - 10.0).abs() < 1e-6);
        // 1% low = slowest 1 frame = 100 ms => 10.0 FPS
        assert!((s.low_1pct_fps - 10.0).abs() < 1e-6);
        // 5% low = slowest 5 frames = (4*20 + 100)/5 = 36 ms => 27.7778 FPS
        assert!((s.low_5pct_fps - (1000.0 / 36.0)).abs() < 1e-4);
        assert!(s.p95 >= 10.0 && s.p95 <= 20.0);
        assert!(s.p99 > 20.0 && s.p99 <= 100.0);
    }

    #[test]
    fn atomic_json_write_and_outside_repo_safety() {
        // Reject relative paths.
        assert!(ensure_outside_repo(Path::new("bench.json")).is_err());

        // Reject paths inside this repo checkout.
        let inside = compiled_workspace_root().join("bench-out.json");
        assert!(ensure_outside_repo(&inside).is_err());

        // Reject any clone marked by `.victory-root`.
        let fake_repo = tempfile::tempdir().unwrap();
        std::fs::write(fake_repo.path().join(ROOT_MARKER), "").unwrap();
        assert!(ensure_outside_repo(&fake_repo.path().join("sub").join("out.json")).is_err());

        // Accept path outside repo and write valid JSON atomically.
        let tmp = tempfile::tempdir().unwrap();
        let out = tmp.path().join("nested").join("metrics.json");
        let sha = register_input_script_bytes(b"{\"from\":0,\"to\":10,\"ly\":1.0}\n");
        assert_eq!(sha.len(), 64);

        let report = BenchReport {
            world_frames: 60,
            warmup_frames: 30,
            frame_ms: compute_frame_stats(&[16.0, 17.0, 18.0]),
            draws: DrawStats {
                avg: 1200.0,
                min: 1100,
                max: 1300,
            },
            device_removed: false,
            zero_draw_world_frames: 0,
            black_frames: 0,
            memory: MemorySummary {
                vram_used_peak_mb: 3500.25,
                vram_budget_mb: 10048.0,
                ram_private_commit_peak_mb: 6144.0,
                ram_working_set_peak_mb: 4096.0,
            },
            framelog: FrameLogAverages {
                prepare_ms: 1.5,
                card_wait_ms: 3.25,
                ..Default::default()
            },
            input_script_sha256: input_script_sha256(),
        };
        write_report_atomic(&out, &report).unwrap();
        let text = std::fs::read_to_string(&out).unwrap();
        assert!(text.contains("\"world_frames\": 60"));
        assert!(text.contains("\"warmup_frames\": 30"));
        assert!(text.contains("\"low_1pct_fps\":"));
        assert!(text.contains("\"vram_used_peak_mb\": 3500.25"));
        assert!(text.contains(&format!("\"input_script_sha256\": \"{sha}\"")));

        // No stray `.tmp.<pid>` left behind.
        let entries: Vec<_> = std::fs::read_dir(out.parent().unwrap())
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert_eq!(entries.len(), 1);
    }
}
