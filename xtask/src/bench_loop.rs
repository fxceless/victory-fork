//! Deterministic traversal benchmark runner & closed-loop optimization harness.
//!
//! Zero extra crate dependencies: uses only `std`, `anyhow`, and `clap`.
//! Implements:
//! - Pure-std TOML/KV playlist config parsing (`[[scenes]]`)
//! - Pure-std JSON parser for `victory-gpu::bench` reports
//! - Pure-std PNG (RGB/RGBA, RFC 1950/1951 inflate + row unfilter) luma parity
//! - Interleaved A/B benchmarking (`--mode ab`: `base -> cand -> base` or N-run median)
//! - Autonomous optimization loop (`--mode loop`) with pluggable `--agent-cmd`,
//!   optional `--fast-cmd` offline pre-filter, `experiments.jsonl` dead-end history,
//!   and 6 hard anti-reward-hacking gates.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Args as ClapArgs, ValueEnum};

const ROOT_MARKER: &str = ".victory-root";

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum Mode {
    /// Interleaved A/B comparison (`base -> cand -> base` or N-run median).
    Ab,
    /// Autonomous optimization loop with pluggable `--agent-cmd`.
    Loop,
}

#[derive(Clone, Debug, ClapArgs)]
pub struct Args {
    /// Path to playlist TOML/KV config file (kept outside the repository).
    #[arg(long)]
    pub config: PathBuf,

    /// Execution mode: `ab` (interleaved A/B) or `loop` (autonomous optimizer).
    #[arg(long, value_enum, default_value_t = Mode::Ab)]
    pub mode: Mode,

    /// Interleaved rounds per scene (`1` runs `base -> cand -> base`; `N` runs
    /// `N` interleaved `base -> cand` pairs plus a trailing `base`).
    #[arg(long, default_value_t = 1)]
    pub runs: usize,

    /// Minimum aggregate median `frame_ms` gain (%) required to keep a candidate.
    #[arg(long, default_value_t = 2.0)]
    pub min_gain_pct: f64,

    /// Maximum allowed `draws.avg` drift (%) vs baseline per scene.
    #[arg(long, default_value_t = 15.0)]
    pub max_draw_drift_pct: f64,

    /// Maximum allowed `low_1pct_fps` regression (%) vs baseline per scene.
    #[arg(long, default_value_t = 2.0)]
    pub max_low1_regress_pct: f64,

    /// Maximum allowed `vram_used_peak_mb` increase (MB) vs baseline per scene.
    #[arg(long, default_value_t = 128.0)]
    pub max_vram_regress_mb: f64,

    /// Maximum allowed `ram_private_commit_peak_mb` increase (MB) vs baseline per scene.
    #[arg(long, default_value_t = 256.0)]
    pub max_ram_regress_mb: f64,

    /// Optional hard peak VRAM ceiling (MB), overriding playlist `vram_cap_mb`.
    #[arg(long)]
    pub vram_cap_mb: Option<f64>,

    /// Optional hard peak private-commit RAM ceiling (MB), overriding playlist `ram_cap_mb`.
    #[arg(long)]
    pub ram_cap_mb: Option<f64>,

    /// Maximum allowed mean luma delta (0..255) for optional checkpoint PNG parity.
    #[arg(long, default_value_t = 12.0)]
    pub max_luma_diff: f64,

    /// Per-scene watchdog timeout in seconds.
    #[arg(long, default_value_t = 300)]
    pub timeout_secs: u64,

    /// Number of autonomous optimization iterations (`--mode loop`).
    #[arg(long, default_value_t = 5)]
    pub iterations: u32,

    /// Pluggable coding-agent command for `--mode loop` (prompt sent on stdin).
    #[arg(long)]
    pub agent_cmd: Option<String>,

    /// Optional fast offline pre-filter command run before full scene suite.
    #[arg(long)]
    pub fast_cmd: Option<String>,

    /// Path to `experiments.jsonl` log (kept outside the repository).
    #[arg(long)]
    pub experiments: Option<PathBuf>,

    /// Override playlist `base_cmd`.
    #[arg(long)]
    pub base_cmd: Option<String>,

    /// Override playlist `runner_cmd` / `cand_cmd`.
    #[arg(long)]
    pub cand_cmd: Option<String>,

    /// Scratch directory for run JSON outputs (defaults to system temp dir).
    #[arg(long)]
    pub work_dir: Option<PathBuf>,

    /// Skip Gate 1 (workspace build/test/policy/clippy) for prebuilt A/B runs.
    #[arg(long, default_value_t = false)]
    pub skip_build_gate: bool,
}

// ---------------------------------------------------------------------------
// Playlist TOML/KV Config
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SceneConfig {
    pub name: String,
    pub setup_cmd: Option<String>,
    pub input_script: PathBuf,
    pub warmup: u64,
    pub frames: u64,
    pub check_every: Option<u64>,
    pub checkpoint_png_base: Option<PathBuf>,
    pub checkpoint_png_cand: Option<PathBuf>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Playlist {
    pub base_cmd: Option<String>,
    pub runner_cmd: Option<String>,
    pub fast_cmd: Option<String>,
    pub experiments: Option<PathBuf>,
    pub repo_dir: Option<PathBuf>,
    pub vram_cap_mb: Option<f64>,
    pub ram_cap_mb: Option<f64>,
    pub build_gate_cmds: Vec<String>,
    pub scenes: Vec<SceneConfig>,
}

impl Playlist {
    pub fn load(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read playlist {}", path.display()))?;
        Self::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Self> {
        let mut pl = Playlist::default();
        let mut current_scene: Option<SceneConfig> = None;

        for (idx, raw) in text.lines().enumerate() {
            let lineno = idx + 1;
            let line = strip_toml_comment(raw).trim();
            if line.is_empty() {
                continue;
            }
            if line == "[[scenes]]" || line == "[scene]" {
                if let Some(s) = current_scene.take() {
                    pl.scenes.push(s);
                }
                current_scene = Some(SceneConfig {
                    warmup: 30,
                    frames: 120,
                    ..Default::default()
                });
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                bail!("playlist line {lineno}: expected `key = value`, got `{line}`");
            };
            let key = k.trim();
            let val = v.trim();
            if let Some(scene) = current_scene.as_mut() {
                match key {
                    "name" => scene.name = parse_toml_string(val, lineno)?,
                    "setup_cmd" => {
                        let s = parse_toml_string(val, lineno)?;
                        if !s.is_empty() {
                            scene.setup_cmd = Some(s);
                        }
                    }
                    "input_script" => {
                        scene.input_script = PathBuf::from(parse_toml_string(val, lineno)?);
                    }
                    "warmup" => {
                        scene.warmup = val
                            .parse::<u64>()
                            .with_context(|| format!("playlist line {lineno}: invalid warmup"))?;
                    }
                    "frames" => {
                        scene.frames = val
                            .parse::<u64>()
                            .with_context(|| format!("playlist line {lineno}: invalid frames"))?;
                    }
                    "check_every" => {
                        let n = val.parse::<u64>().with_context(|| {
                            format!("playlist line {lineno}: invalid check_every")
                        })?;
                        scene.check_every = (n > 0).then_some(n);
                    }
                    "checkpoint_png_base" => {
                        scene.checkpoint_png_base = Some(PathBuf::from(parse_toml_string(val, lineno)?));
                    }
                    "checkpoint_png_cand" => {
                        scene.checkpoint_png_cand = Some(PathBuf::from(parse_toml_string(val, lineno)?));
                    }
                    other => bail!("playlist line {lineno}: unknown scene key `{other}`"),
                }
            } else {
                match key {
                    "base_cmd" => pl.base_cmd = Some(parse_toml_string(val, lineno)?),
                    "runner_cmd" | "cand_cmd" => {
                        pl.runner_cmd = Some(parse_toml_string(val, lineno)?);
                    }
                    "fast_cmd" => pl.fast_cmd = Some(parse_toml_string(val, lineno)?),
                    "experiments" => {
                        pl.experiments = Some(PathBuf::from(parse_toml_string(val, lineno)?));
                    }
                    "repo_dir" => {
                        pl.repo_dir = Some(PathBuf::from(parse_toml_string(val, lineno)?));
                    }
                    "vram_cap_mb" => {
                        pl.vram_cap_mb = Some(val.parse::<f64>().with_context(|| {
                            format!("playlist line {lineno}: invalid vram_cap_mb")
                        })?);
                    }
                    "ram_cap_mb" => {
                        pl.ram_cap_mb = Some(val.parse::<f64>().with_context(|| {
                            format!("playlist line {lineno}: invalid ram_cap_mb")
                        })?);
                    }
                    "build_gate_cmds" => {
                        pl.build_gate_cmds = parse_toml_string_array(val, lineno)?;
                    }
                    other => bail!("playlist line {lineno}: unknown top-level key `{other}`"),
                }
            }
        }

        if let Some(s) = current_scene.take() {
            pl.scenes.push(s);
        }
        if pl.scenes.is_empty() {
            bail!("playlist must define at least one `[[scenes]]` entry");
        }
        for (i, s) in pl.scenes.iter().enumerate() {
            if s.name.is_empty() {
                bail!("playlist scene #{} is missing `name`", i + 1);
            }
            if s.name.contains(['/', '\\', ':']) || matches!(s.name.as_str(), "." | "..") {
                bail!("playlist scene `{}` must be a single file-name component", s.name);
            }
            if s.frames == 0 {
                bail!("playlist scene `{}` must have `frames > 0`", s.name);
            }
        }
        Ok(pl)
    }
}

fn strip_toml_comment(line: &str) -> &str {
    let mut in_double = false;
    let mut in_single = false;
    let mut escaped = false;
    for (i, ch) in line.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        match ch {
            '\\' if in_double => escaped = true,
            '"' if !in_single => in_double = !in_double,
            '\'' if !in_double => in_single = !in_single,
            '#' if !in_double && !in_single => return &line[..i],
            _ => {}
        }
    }
    line
}

fn parse_toml_string(val: &str, lineno: usize) -> Result<String> {
    if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
        let inner = &val[1..val.len() - 1];
        let mut out = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some(other) => {
                        out.push('\\');
                        out.push(other);
                    }
                    None => bail!("playlist line {lineno}: trailing backslash in string"),
                }
            } else {
                out.push(c);
            }
        }
        Ok(out)
    } else if val.starts_with('\'') && val.ends_with('\'') && val.len() >= 2 {
        Ok(val[1..val.len() - 1].to_owned())
    } else {
        Ok(val.to_owned())
    }
}

fn parse_toml_string_array(val: &str, lineno: usize) -> Result<Vec<String>> {
    let trimmed = val.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        bail!("playlist line {lineno}: expected array `[...]`");
    }
    let inner = &trimmed[1..trimmed.len() - 1];
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut in_quote: Option<char> = None;
    let mut escaped = false;
    for c in inner.chars() {
        if escaped {
            cur.push(c);
            escaped = false;
            continue;
        }
        if let Some(q) = in_quote {
            if c == '\\' && q == '"' {
                cur.push(c);
                escaped = true;
            } else if c == q {
                cur.push(c);
                in_quote = None;
            } else {
                cur.push(c);
            }
        } else if c == '"' || c == '\'' {
            cur.push(c);
            in_quote = Some(c);
        } else if c == ',' {
            let item = cur.trim();
            if !item.is_empty() {
                out.push(parse_toml_string(item, lineno)?);
            }
            cur.clear();
        } else {
            cur.push(c);
        }
    }
    let last = cur.trim();
    if !last.is_empty() {
        out.push(parse_toml_string(last, lineno)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Pure-std JSON Parser & BenchReport Schema
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<JsonValue>),
    Object(BTreeMap<String, JsonValue>),
}

impl JsonValue {
    fn as_object(&self) -> Result<&BTreeMap<String, JsonValue>> {
        match self {
            JsonValue::Object(m) => Ok(m),
            _ => bail!("expected JSON object"),
        }
    }

    fn as_f64(&self) -> Result<f64> {
        match self {
            JsonValue::Number(v) if v.is_finite() && *v >= 0.0 => Ok(*v),
            _ => bail!("expected a finite, non-negative JSON metric"),
        }
    }

    fn as_u64(&self) -> Result<u64> {
        let v = self.as_f64()?;
        if v < 0.0 || !v.is_finite() {
            bail!("expected non-negative integer, got {v}");
        }
        Ok(v.round() as u64)
    }

    fn as_bool(&self) -> Result<bool> {
        match self {
            JsonValue::Bool(b) => Ok(*b),
            _ => bail!("expected JSON bool"),
        }
    }
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn skip_ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue> {
        self.skip_ws();
        if self.pos >= self.bytes.len() {
            bail!("unexpected end of JSON");
        }
        match self.bytes[self.pos] {
            b'{' => self.parse_object(),
            b'[' => self.parse_array(),
            b'"' => self.parse_string().map(JsonValue::String),
            b't' => self.expect_literal(b"true", JsonValue::Bool(true)),
            b'f' => self.expect_literal(b"false", JsonValue::Bool(false)),
            b'n' => self.expect_literal(b"null", JsonValue::Null),
            b'-' | b'0'..=b'9' => self.parse_number(),
            c => bail!("unexpected JSON byte `{}` at {}", c as char, self.pos),
        }
    }

    fn expect_literal(&mut self, lit: &[u8], val: JsonValue) -> Result<JsonValue> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(val)
        } else {
            bail!("expected literal at byte {}", self.pos)
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue> {
        self.pos += 1; // '{'
        let mut map = BTreeMap::new();
        self.skip_ws();
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b'}' {
            self.pos += 1;
            return Ok(JsonValue::Object(map));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.pos >= self.bytes.len() || self.bytes[self.pos] != b':' {
                bail!("expected ':' after key `{key}`");
            }
            self.pos += 1;
            let val = self.parse_value()?;
            map.insert(key, val);
            self.skip_ws();
            if self.pos >= self.bytes.len() {
                bail!("unterminated JSON object");
            }
            match self.bytes[self.pos] {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    return Ok(JsonValue::Object(map));
                }
                c => bail!("expected ',' or '}}', got `{}`", c as char),
            }
        }
    }

    fn parse_array(&mut self) -> Result<JsonValue> {
        self.pos += 1; // '['
        let mut arr = Vec::new();
        self.skip_ws();
        if self.pos < self.bytes.len() && self.bytes[self.pos] == b']' {
            self.pos += 1;
            return Ok(JsonValue::Array(arr));
        }
        loop {
            arr.push(self.parse_value()?);
            self.skip_ws();
            if self.pos >= self.bytes.len() {
                bail!("unterminated JSON array");
            }
            match self.bytes[self.pos] {
                b',' => self.pos += 1,
                b']' => {
                    self.pos += 1;
                    return Ok(JsonValue::Array(arr));
                }
                c => bail!("expected ',' or ']', got `{}`", c as char),
            }
        }
    }

    fn parse_string(&mut self) -> Result<String> {
        if self.pos >= self.bytes.len() || self.bytes[self.pos] != b'"' {
            bail!("expected '\"' at byte {}", self.pos);
        }
        self.pos += 1;
        let mut out = String::new();
        while self.pos < self.bytes.len() {
            let b = self.bytes[self.pos];
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    if self.pos >= self.bytes.len() {
                        bail!("unterminated escape sequence");
                    }
                    let e = self.bytes[self.pos];
                    self.pos += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\x08'),
                        b'f' => out.push('\x0c'),
                        b'u' => {
                            if self.pos + 4 > self.bytes.len() {
                                bail!("short \\u escape");
                            }
                            let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])?;
                            self.pos += 4;
                            let code = u32::from_str_radix(hex, 16)?;
                            out.push(char::from_u32(code).unwrap_or('\u{FFFD}'));
                        }
                        other => out.push(other as char),
                    }
                }
                _ if b < 0x80 => out.push(b as char),
                _ => {
                    // Decode UTF-8 multibyte sequence starting at `self.pos - 1`.
                    let start = self.pos - 1;
                    let s = std::str::from_utf8(&self.bytes[start..])
                        .context("invalid UTF-8 in JSON string")?;
                    let ch = s.chars().next().ok_or_else(|| anyhow!("empty UTF-8"))?;
                    out.push(ch);
                    self.pos = start + ch.len_utf8();
                }
            }
        }
        bail!("unterminated JSON string")
    }

    fn parse_number(&mut self) -> Result<JsonValue> {
        let start = self.pos;
        while self.pos < self.bytes.len()
            && matches!(
                self.bytes[self.pos],
                b'-' | b'+' | b'0'..=b'9' | b'.' | b'e' | b'E'
            )
        {
            self.pos += 1;
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])?;
        let n = s
            .parse::<f64>()
            .with_context(|| format!("invalid JSON number `{s}`"))?;
        Ok(JsonValue::Number(n))
    }
}

pub fn parse_json(text: &str) -> Result<JsonValue> {
    let mut p = JsonParser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let val = p.parse_value()?;
    p.skip_ws();
    if p.pos != p.bytes.len() {
        bail!("trailing bytes in JSON at offset {}", p.pos);
    }
    Ok(val)
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BenchReport {
    pub world_frames: u64,
    pub warmup_frames: u64,
    pub frame_ms_avg: f64,
    pub frame_ms_median: f64,
    pub frame_ms_p95: f64,
    pub frame_ms_p99: f64,
    pub low_5pct_fps: f64,
    pub low_1pct_fps: f64,
    pub draws_avg: f64,
    pub draws_min: u64,
    pub draws_max: u64,
    pub device_removed: bool,
    pub zero_draw_world_frames: u64,
    pub black_frames: u64,
    pub vram_used_peak_mb: f64,
    pub vram_budget_mb: f64,
    pub ram_private_commit_peak_mb: f64,
    pub ram_working_set_peak_mb: f64,
    pub framelog: BTreeMap<String, f64>,
    pub input_script_sha256: Option<String>,
}

impl BenchReport {
    pub fn from_json_str(text: &str) -> Result<Self> {
        fn req<'a>(m: &'a BTreeMap<String, JsonValue>, k: &str) -> Result<&'a JsonValue> {
            m.get(k).ok_or_else(|| anyhow!("missing JSON field `{k}`"))
        }
        let root = parse_json(text)?;
        let obj = root.as_object()?;
        let frame_ms = req(obj, "frame_ms")?.as_object()?;
        let draws = req(obj, "draws")?.as_object()?;
        let memory = req(obj, "memory")?.as_object()?;
        let mut framelog = BTreeMap::new();
        if let Some(fl) = obj.get("framelog").and_then(|v| v.as_object().ok()) {
            for (k, v) in fl {
                if let Ok(n) = v.as_f64() {
                    framelog.insert(k.clone(), n);
                }
            }
        }
        let input_script_sha256 = match obj.get("input_script_sha256") {
            Some(JsonValue::String(s)) => Some(s.clone()),
            _ => None,
        };
        Ok(Self {
            world_frames: req(obj, "world_frames")?.as_u64()?,
            warmup_frames: req(obj, "warmup_frames")?.as_u64()?,
            frame_ms_avg: req(frame_ms, "avg")?.as_f64()?,
            frame_ms_median: req(frame_ms, "median")?.as_f64()?,
            frame_ms_p95: req(frame_ms, "p95")?.as_f64()?,
            frame_ms_p99: req(frame_ms, "p99")?.as_f64()?,
            low_5pct_fps: req(frame_ms, "low_5pct_fps")?.as_f64()?,
            low_1pct_fps: req(frame_ms, "low_1pct_fps")?.as_f64()?,
            draws_avg: req(draws, "avg")?.as_f64()?,
            draws_min: req(draws, "min")?.as_u64()?,
            draws_max: req(draws, "max")?.as_u64()?,
            device_removed: req(obj, "device_removed")?.as_bool()?,
            zero_draw_world_frames: req(obj, "zero_draw_world_frames")?.as_u64()?,
            black_frames: req(obj, "black_frames")?.as_u64()?,
            vram_used_peak_mb: req(memory, "vram_used_peak_mb")?.as_f64()?,
            vram_budget_mb: req(memory, "vram_budget_mb")?.as_f64()?,
            ram_private_commit_peak_mb: req(memory, "ram_private_commit_peak_mb")?.as_f64()?,
            ram_working_set_peak_mb: req(memory, "ram_working_set_peak_mb")?.as_f64()?,
            framelog,
            input_script_sha256,
        })
    }

    pub fn from_file(path: &Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("read bench report {}", path.display()))?;
        Self::from_json_str(&text)
    }

    /// Top `k` `framelog` timing stages (`*_ms`) sorted descending by average ms.
    pub fn top_bottlenecks(&self, k: usize) -> Vec<(String, f64)> {
        let mut stages: Vec<(String, f64)> = self
            .framelog
            .iter()
            .filter(|(name, _)| name.ends_with("_ms"))
            .map(|(n, &v)| (n.clone(), v))
            .collect();
        stages.sort_by(|a, b| b.1.total_cmp(&a.1));
        stages.truncate(k);
        stages
    }
}

fn median_f64(mut vals: Vec<f64>) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    vals.sort_by(f64::total_cmp);
    let n = vals.len();
    if n % 2 == 1 {
        vals[n / 2]
    } else {
        (vals[n / 2 - 1] + vals[n / 2]) * 0.5
    }
}

/// Aggregates multiple runs of a scene into a single conservative summary:
/// medians for timings/draws, minimums for `world_frames`/`warmup_frames`, and
/// worst-case (max/OR) for errors/peaks/sanity.
pub fn aggregate_runs(runs: &[BenchReport]) -> Result<BenchReport> {
    if runs.is_empty() {
        bail!("cannot aggregate zero bench runs");
    }
    let mut framelog = BTreeMap::new();
    for key in runs[0].framelog.keys() {
        let vals: Vec<f64> = runs
            .iter()
            .filter_map(|r| r.framelog.get(key).copied())
            .collect();
        framelog.insert(key.clone(), median_f64(vals));
    }
    Ok(BenchReport {
        world_frames: runs.iter().map(|r| r.world_frames).min().unwrap_or(0),
        warmup_frames: runs.iter().map(|r| r.warmup_frames).min().unwrap_or(0),
        frame_ms_avg: median_f64(runs.iter().map(|r| r.frame_ms_avg).collect()),
        frame_ms_median: median_f64(runs.iter().map(|r| r.frame_ms_median).collect()),
        frame_ms_p95: median_f64(runs.iter().map(|r| r.frame_ms_p95).collect()),
        frame_ms_p99: median_f64(runs.iter().map(|r| r.frame_ms_p99).collect()),
        low_5pct_fps: median_f64(runs.iter().map(|r| r.low_5pct_fps).collect()),
        low_1pct_fps: median_f64(runs.iter().map(|r| r.low_1pct_fps).collect()),
        draws_avg: median_f64(runs.iter().map(|r| r.draws_avg).collect()),
        draws_min: runs.iter().map(|r| r.draws_min).min().unwrap_or(0),
        draws_max: runs.iter().map(|r| r.draws_max).max().unwrap_or(0),
        device_removed: runs.iter().any(|r| r.device_removed),
        zero_draw_world_frames: runs.iter().map(|r| r.zero_draw_world_frames).max().unwrap_or(0),
        black_frames: runs.iter().map(|r| r.black_frames).max().unwrap_or(0),
        vram_used_peak_mb: runs
            .iter()
            .map(|r| r.vram_used_peak_mb)
            .fold(0.0, f64::max),
        vram_budget_mb: median_f64(runs.iter().map(|r| r.vram_budget_mb).collect()),
        ram_private_commit_peak_mb: runs
            .iter()
            .map(|r| r.ram_private_commit_peak_mb)
            .fold(0.0, f64::max),
        ram_working_set_peak_mb: runs
            .iter()
            .map(|r| r.ram_working_set_peak_mb)
            .fold(0.0, f64::max),
        framelog,
        input_script_sha256: runs[0].input_script_sha256.clone(),
    })
}

// ---------------------------------------------------------------------------
// Pure-std PNG Decoder (RGB/RGBA 8-bit, RFC 1950/1951 Inflate + Unfilter)
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct PngParity {
    pub width: u32,
    pub height: u32,
    pub base_mean_luma: f64,
    pub cand_mean_luma: f64,
    pub mean_abs_luma_diff: f64,
    pub cand_is_black: bool,
}

struct DecodedPng {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
}

/// Decodes a standard 8-bit non-interlaced RGB (`color_type=2`) or RGBA
/// (`color_type=6`) PNG into packed RGB bytes using pure `std`.
fn decode_png_rgb(bytes: &[u8]) -> Result<DecodedPng> {
    const SIG: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];
    if bytes.len() < 33 || bytes[..8] != SIG {
        bail!("invalid PNG signature");
    }
    let mut pos = 8usize;
    let mut width = 0u32;
    let mut height = 0u32;
    let mut bit_depth = 0u8;
    let mut color_type = 0u8;
    let mut idat = Vec::new();
    let mut saw_ihdr = false;

    while pos + 12 <= bytes.len() {
        let len = u32::from_be_bytes(bytes[pos..pos + 4].try_into()?) as usize;
        let kind = &bytes[pos + 4..pos + 8];
        let data_start = pos + 8;
        let data_end = data_start
            .checked_add(len)
            .ok_or_else(|| anyhow!("PNG chunk overflow"))?;
        if data_end + 4 > bytes.len() {
            bail!("truncated PNG chunk");
        }
        let data = &bytes[data_start..data_end];
        match kind {
            b"IHDR" => {
                if data.len() != 13 {
                    bail!("invalid IHDR length");
                }
                width = u32::from_be_bytes(data[0..4].try_into()?);
                height = u32::from_be_bytes(data[4..8].try_into()?);
                bit_depth = data[8];
                color_type = data[9];
                if data[12] != 0 {
                    bail!("interlaced PNG not supported");
                }
                saw_ihdr = true;
            }
            b"IDAT" => idat.extend_from_slice(data),
            b"IEND" => break,
            _ => {}
        }
        pos = data_end + 4;
    }

    if !saw_ihdr || width == 0 || height == 0 {
        bail!("PNG missing valid IHDR");
    }
    if bit_depth != 8 || !matches!(color_type, 2 | 6) {
        bail!("unsupported PNG format: bit_depth={bit_depth}, color_type={color_type}");
    }
    let bpp = if color_type == 6 { 4usize } else { 3usize };
    let raw = inflate_zlib(&idat)?;
    let row_bytes = (width as usize) * bpp;
    let expected = (height as usize) * (row_bytes + 1);
    if raw.len() < expected {
        bail!(
            "PNG inflated data too short: got {}, need {expected}",
            raw.len()
        );
    }

    let mut prev = vec![0u8; row_bytes];
    let mut cur = vec![0u8; row_bytes];
    let mut rgb = Vec::with_capacity((width as usize) * (height as usize) * 3);

    for y in 0..height as usize {
        let base = y * (row_bytes + 1);
        let filter = raw[base];
        let src = &raw[base + 1..base + 1 + row_bytes];
        for i in 0..row_bytes {
            let a = if i >= bpp { cur[i - bpp] } else { 0 };
            let b = prev[i];
            let c = if i >= bpp { prev[i - bpp] } else { 0 };
            let recon = match filter {
                0 => src[i],
                1 => src[i].wrapping_add(a),
                2 => src[i].wrapping_add(b),
                3 => src[i].wrapping_add(((u16::from(a) + u16::from(b)) / 2) as u8),
                4 => src[i].wrapping_add(paeth_predictor(a, b, c)),
                other => bail!("unsupported PNG filter type {other}"),
            };
            cur[i] = recon;
        }
        for px in cur.chunks_exact(bpp) {
            rgb.extend_from_slice(&px[..3]);
        }
        std::mem::swap(&mut prev, &mut cur);
    }

    Ok(DecodedPng { width, height, rgb })
}

fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let (a, b, c) = (i32::from(a), i32::from(b), i32::from(c));
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a as u8
    } else if pb <= pc {
        b as u8
    } else {
        c as u8
    }
}

/// Minimal RFC 1950 + RFC 1951 (DEFLATE) inflater supporting stored, fixed
/// Huffman, and dynamic Huffman blocks.
fn inflate_zlib(data: &[u8]) -> Result<Vec<u8>> {
    if data.len() < 6 {
        bail!("zlib stream too short");
    }
    let cmf = data[0];
    let flg = data[1];
    let header = u16::from_be_bytes([cmf, flg]);
    if (cmf & 0x0F) != 8 || !header.is_multiple_of(31) {
        bail!("invalid zlib header");
    }
    let start = if (flg & 0x20) != 0 { 6 } else { 2 };
    let mut reader = BitReader::new(&data[start..]);
    let mut out = Vec::new();

    loop {
        let bfinal = reader.read_bits(1)?;
        let btype = reader.read_bits(2)?;
        match btype {
            0 => {
                reader.align_byte();
                let len = reader.read_u16_le()? as usize;
                let nlen = reader.read_u16_le()? as usize;
                if (len ^ 0xFFFF) != nlen {
                    bail!("deflate stored block length mismatch");
                }
                let slice = reader.read_bytes(len)?;
                out.extend_from_slice(slice);
            }
            1 => {
                let (lit, dist) = fixed_huffman_tables();
                inflate_huffman_block(&mut reader, &lit, &dist, &mut out)?;
            }
            2 => {
                let (lit, dist) = read_dynamic_huffman_tables(&mut reader)?;
                inflate_huffman_block(&mut reader, &lit, &dist, &mut out)?;
            }
            _ => bail!("invalid deflate block type {btype}"),
        }
        if bfinal == 1 {
            break;
        }
    }
    Ok(out)
}

struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_buf: u32,
    bit_count: u8,
}

impl<'a> BitReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_buf: 0,
            bit_count: 0,
        }
    }

    fn read_bits(&mut self, n: u8) -> Result<u32> {
        while self.bit_count < n {
            if self.byte_pos >= self.data.len() {
                bail!("unexpected EOF in deflate stream");
            }
            self.bit_buf |= u32::from(self.data[self.byte_pos]) << self.bit_count;
            self.byte_pos += 1;
            self.bit_count += 8;
        }
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let val = self.bit_buf & mask;
        self.bit_buf >>= n;
        self.bit_count -= n;
        Ok(val)
    }

    fn align_byte(&mut self) {
        let drop = self.bit_count % 8;
        self.bit_buf >>= drop;
        self.bit_count -= drop;
    }

    fn read_u16_le(&mut self) -> Result<u16> {
        let lo = self.read_bits(8)? as u16;
        let hi = self.read_bits(8)? as u16;
        Ok(lo | (hi << 8))
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let mut rewind = 0usize;
        while self.bit_count >= 8 {
            self.bit_count -= 8;
            rewind += 1;
        }
        self.bit_buf = 0;
        self.byte_pos = self.byte_pos.saturating_sub(rewind);
        let end = self
            .byte_pos
            .checked_add(len)
            .ok_or_else(|| anyhow!("deflate slice overflow"))?;
        if end > self.data.len() {
            bail!("truncated deflate stored block");
        }
        let slice = &self.data[self.byte_pos..end];
        self.byte_pos = end;
        Ok(slice)
    }
}

struct HuffTable {
    counts: [u16; 16],
    symbols: Vec<u16>,
}

impl HuffTable {
    fn from_lengths(lengths: &[u8]) -> Result<Self> {
        let mut counts = [0u16; 16];
        for &len in lengths {
            if usize::from(len) >= counts.len() {
                bail!("invalid Huffman code length {len}");
            }
            if len > 0 {
                counts[usize::from(len)] += 1;
            }
        }
        let mut offsets = [0u16; 16];
        let mut total = 0u16;
        for bits in 1..16 {
            offsets[bits] = total;
            total += counts[bits];
        }
        let mut symbols = vec![0u16; usize::from(total)];
        for (sym, &len) in lengths.iter().enumerate() {
            if len > 0 {
                let idx = usize::from(offsets[usize::from(len)]);
                symbols[idx] = sym as u16;
                offsets[usize::from(len)] += 1;
            }
        }
        Ok(Self { counts, symbols })
    }

    fn decode(&self, reader: &mut BitReader<'_>) -> Result<u16> {
        let mut code = 0i32;
        let mut first = 0i32;
        let mut index = 0i32;
        for len in 1..16 {
            code |= reader.read_bits(1)? as i32;
            let count = i32::from(self.counts[len]);
            if code - count < first {
                return Ok(self.symbols[(index + (code - first)) as usize]);
            }
            index += count;
            first = (first + count) << 1;
            code <<= 1;
        }
        bail!("invalid Huffman code in deflate stream")
    }
}

fn fixed_huffman_tables() -> (HuffTable, HuffTable) {
    let mut lit = vec![0u8; 288];
    for v in &mut lit[0..=143] {
        *v = 8;
    }
    for v in &mut lit[144..=255] {
        *v = 9;
    }
    for v in &mut lit[256..=279] {
        *v = 7;
    }
    for v in &mut lit[280..=287] {
        *v = 8;
    }
    let dist = vec![5u8; 32];
    (
        HuffTable::from_lengths(&lit).unwrap(),
        HuffTable::from_lengths(&dist).unwrap(),
    )
}

fn read_dynamic_huffman_tables(reader: &mut BitReader<'_>) -> Result<(HuffTable, HuffTable)> {
    const ORDER: [usize; 19] = [
        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
    ];
    let hlit = reader.read_bits(5)? as usize + 257;
    let hdist = reader.read_bits(5)? as usize + 1;
    let hclen = reader.read_bits(4)? as usize + 4;
    let mut cl_lens = [0u8; 19];
    for &idx in ORDER.iter().take(hclen) {
        cl_lens[idx] = reader.read_bits(3)? as u8;
    }
    let cl_table = HuffTable::from_lengths(&cl_lens)?;
    let total = hlit + hdist;
    let mut lengths = Vec::with_capacity(total);
    while lengths.len() < total {
        let sym = cl_table.decode(reader)?;
        match sym {
            0..=15 => lengths.push(sym as u8),
            16 => {
                let prev = *lengths
                    .last()
                    .ok_or_else(|| anyhow!("repeat code with no prior length"))?;
                let rep = reader.read_bits(2)? as usize + 3;
                lengths.extend(std::iter::repeat_n(prev, rep));
            }
            17 => {
                let rep = reader.read_bits(3)? as usize + 3;
                lengths.extend(std::iter::repeat_n(0, rep));
            }
            18 => {
                let rep = reader.read_bits(7)? as usize + 11;
                lengths.extend(std::iter::repeat_n(0, rep));
            }
            _ => bail!("invalid code-length symbol {sym}"),
        }
    }
    lengths.truncate(total);
    Ok((
        HuffTable::from_lengths(&lengths[..hlit])?,
        HuffTable::from_lengths(&lengths[hlit..])?,
    ))
}

fn inflate_huffman_block(
    reader: &mut BitReader<'_>,
    lit: &HuffTable,
    dist: &HuffTable,
    out: &mut Vec<u8>,
) -> Result<()> {
    const LEN_BASE: [usize; 29] = [
        3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115,
        131, 163, 195, 227, 258,
    ];
    const LEN_EXTRA: [u8; 29] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
    ];
    const DIST_BASE: [usize; 30] = [
        1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
        2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
    ];
    const DIST_EXTRA: [u8; 30] = [
        0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12,
        13, 13,
    ];

    loop {
        let sym = lit.decode(reader)?;
        if sym < 256 {
            out.push(sym as u8);
        } else if sym == 256 {
            return Ok(());
        } else {
            let lidx = usize::from(sym - 257);
            if lidx >= LEN_BASE.len() {
                bail!("invalid deflate length symbol {sym}");
            }
            let length = LEN_BASE[lidx] + reader.read_bits(LEN_EXTRA[lidx])? as usize;
            let didx = usize::from(dist.decode(reader)?);
            if didx >= DIST_BASE.len() {
                bail!("invalid deflate distance symbol {didx}");
            }
            let distance = DIST_BASE[didx] + reader.read_bits(DIST_EXTRA[didx])? as usize;
            if distance == 0 || distance > out.len() {
                bail!("invalid deflate back-reference distance {distance}");
            }
            for _ in 0..length {
                let b = out[out.len() - distance];
                out.push(b);
            }
        }
    }
}

/// Compares baseline and candidate checkpoint PNGs for dimensions, non-black
/// luminance, and per-pixel mean absolute luma difference.
pub fn compare_checkpoint_pngs(base_path: &Path, cand_path: &Path) -> Result<PngParity> {
    let base_bytes = std::fs::read(base_path)
        .with_context(|| format!("read base checkpoint PNG {}", base_path.display()))?;
    let cand_bytes = std::fs::read(cand_path)
        .with_context(|| format!("read cand checkpoint PNG {}", cand_path.display()))?;
    let base = decode_png_rgb(&base_bytes)?;
    let cand = decode_png_rgb(&cand_bytes)?;
    if (base.width, base.height) != (cand.width, cand.height) {
        bail!(
            "checkpoint PNG dimension mismatch: base {}x{} vs cand {}x{}",
            base.width,
            base.height,
            cand.width,
            cand.height
        );
    }
    let pixels = (base.width as usize) * (base.height as usize);
    let mut base_sum = 0.0f64;
    let mut cand_sum = 0.0f64;
    let mut abs_diff_sum = 0.0f64;
    for (bp, cp) in base.rgb.chunks_exact(3).zip(cand.rgb.chunks_exact(3)) {
        let bl = 0.299 * f64::from(bp[0]) + 0.587 * f64::from(bp[1]) + 0.114 * f64::from(bp[2]);
        let cl = 0.299 * f64::from(cp[0]) + 0.587 * f64::from(cp[1]) + 0.114 * f64::from(cp[2]);
        base_sum += bl;
        cand_sum += cl;
        abs_diff_sum += (bl - cl).abs();
    }
    let n = pixels.max(1) as f64;
    let base_mean_luma = base_sum / n;
    let cand_mean_luma = cand_sum / n;
    Ok(PngParity {
        width: base.width,
        height: base.height,
        base_mean_luma,
        cand_mean_luma,
        mean_abs_luma_diff: abs_diff_sum / n,
        cand_is_black: cand_mean_luma < 2.0,
    })
}

// ---------------------------------------------------------------------------
// Process Runner with Watchdog
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CmdOutcome {
    Exited(i32),
    TimedOut,
}

fn build_shell_command(cmd_str: &str) -> Command {
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        let mut c = Command::new("cmd");
        // /S removes the outer pair and leaves quotes inside the command intact.
        c.args(["/D", "/S", "/C"]).raw_arg(format!("\"{cmd_str}\""));
        c
    }
    #[cfg(not(windows))]
    {
        let mut c = Command::new("sh");
        c.arg("-c").arg(cmd_str);
        c
    }
}

fn kill_child_tree(child: &mut std::process::Child) {
    #[cfg(windows)]
    {
        let _ = Command::new("taskkill")
            .args(["/F", "/T", "/PID", &child.id().to_string()])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }
    let _ = child.kill();
    let _ = child.wait();
}

fn spawn_shell_cmd(
    cmd_str: &str,
    cwd: Option<&Path>,
    envs: &[(&str, String)],
    stdin_bytes: Option<&[u8]>,
    timeout: Duration,
) -> Result<(CmdOutcome, String, String)> {
    use std::io::Read;
    let mut cmd = build_shell_command(cmd_str);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.stdin(if stdin_bytes.is_some() {
        Stdio::piped()
    } else {
        Stdio::null()
    });
    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn command `{cmd_str}`"))?;
    if let Some(input) = stdin_bytes
        && let Some(mut stdin) = child.stdin.take()
    {
        let _ = stdin.write_all(input);
    }
    let stdout_pipe = child.stdout.take();
    let stderr_pipe = child.stderr.take();
    let out_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut r) = stdout_pipe {
            let _ = r.read_to_end(&mut buf);
        }
        buf
    });
    let err_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(mut r) = stderr_pipe {
            let _ = r.read_to_end(&mut buf);
        }
        buf
    });

    let start = Instant::now();
    let outcome = loop {
        if let Some(status) = child.try_wait()? {
            break CmdOutcome::Exited(status.code().unwrap_or(-1));
        }
        if start.elapsed() >= timeout {
            kill_child_tree(&mut child);
            break CmdOutcome::TimedOut;
        }
        std::thread::sleep(Duration::from_millis(15));
    };
    let stdout = String::from_utf8_lossy(&out_handle.join().unwrap_or_default()).into_owned();
    let stderr = String::from_utf8_lossy(&err_handle.join().unwrap_or_default()).into_owned();
    Ok((outcome, stdout, stderr))
}

pub fn run_shell_with_timeout(
    cmd_str: &str,
    cwd: Option<&Path>,
    envs: &[(&str, String)],
    timeout: Duration,
) -> Result<CmdOutcome> {
    let mut cmd = build_shell_command(cmd_str);
    if let Some(dir) = cwd {
        cmd.current_dir(dir);
    }
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.stdin(Stdio::null());
    let mut child = cmd
        .spawn()
        .with_context(|| format!("spawn `{cmd_str}`"))?;
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(CmdOutcome::Exited(status.code().unwrap_or(-1)));
        }
        if start.elapsed() >= timeout {
            kill_child_tree(&mut child);
            return Ok(CmdOutcome::TimedOut);
        }
        std::thread::sleep(Duration::from_millis(15));
    }
}

fn expand_template(
    template: &str,
    scene: &SceneConfig,
    role: &str,
    bench_out: &Path,
    checkpoint_png: Option<&Path>,
) -> String {
    let mut s = template.to_owned();
    s = s.replace("{bench_out}", &bench_out.display().to_string());
    s = s.replace("{input_script}", &scene.input_script.display().to_string());
    s = s.replace("{warmup}", &scene.warmup.to_string());
    s = s.replace("{frames}", &scene.frames.to_string());
    s = s.replace("{role}", role);
    s = s.replace("{scene}", &scene.name);
    if let Some(png) = checkpoint_png {
        s = s.replace("{checkpoint_png}", &png.display().to_string());
    }
    s
}

#[derive(Clone, Debug)]
pub struct SceneRunResult {
    pub outcome: CmdOutcome,
    pub report: Option<BenchReport>,
    pub error: Option<String>,
}

fn run_scene_once(
    cmd_template: &str,
    scene: &SceneConfig,
    role: &str,
    run_index: usize,
    work_dir: &Path,
    timeout: Duration,
) -> SceneRunResult {
    if let Some(setup) = scene.setup_cmd.as_deref() {
        match run_shell_with_timeout(setup, None, &[], timeout) {
            Ok(CmdOutcome::Exited(0)) => {}
            Ok(CmdOutcome::Exited(code)) => {
                return SceneRunResult {
                    outcome: CmdOutcome::Exited(code),
                    report: None,
                    error: Some(format!("setup_cmd exited with code {code}")),
                };
            }
            Ok(CmdOutcome::TimedOut) => {
                return SceneRunResult {
                    outcome: CmdOutcome::TimedOut,
                    report: None,
                    error: Some("setup_cmd timed out".to_owned()),
                };
            }
            Err(e) => {
                return SceneRunResult {
                    outcome: CmdOutcome::Exited(-1),
                    report: None,
                    error: Some(format!("setup_cmd failed: {e}")),
                };
            }
        }
    }

    let out_file = work_dir.join(format!("{}_{}_{}.json", scene.name, role, run_index));
    let clear_output = ensure_outside_repo(&out_file).and_then(|()| {
        match std::fs::remove_file(&out_file) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(error).with_context(|| format!("remove stale report {}", out_file.display())),
        }
    });
    if let Err(error) = clear_output {
        return SceneRunResult {
            outcome: CmdOutcome::Exited(-1),
            report: None,
            error: Some(error.to_string()),
        };
    }
    let checkpoint = if role == "base" {
        scene.checkpoint_png_base.as_deref()
    } else {
        scene.checkpoint_png_cand.as_deref()
    };
    let expanded = expand_template(cmd_template, scene, role, &out_file, checkpoint);

    let mut envs = vec![
        ("VICTORY_BENCH_OUT", out_file.display().to_string()),
        ("VICTORY_BENCH_WARMUP", scene.warmup.to_string()),
        ("VICTORY_BENCH_EXIT_FRAMES", scene.frames.to_string()),
        (
            "VICTORY_INPUT_REPLAY",
            scene.input_script.display().to_string(),
        ),
        ("VICTORY_BENCH_ROLE", role.to_owned()),
        ("VICTORY_BENCH_SCENE", scene.name.clone()),
        ("VICTORY_BENCH_RUN", run_index.to_string()),
    ];
    if let Some(every) = scene.check_every {
        envs.push(("VICTORY_BENCH_CHECK_EVERY", every.to_string()));
    }
    if let Some(png) = checkpoint {
        envs.push(("VICTORY_BENCH_CHECKPOINT_PNG", png.display().to_string()));
    }

    match run_shell_with_timeout(&expanded, None, &envs, timeout) {
        Ok(CmdOutcome::TimedOut) => SceneRunResult {
            outcome: CmdOutcome::TimedOut,
            report: None,
            error: Some(format!(
                "scene `{}` ({role} #{run_index}) timed out after {}s",
                scene.name,
                timeout.as_secs()
            )),
        },
        Ok(CmdOutcome::Exited(code)) => {
            let report = BenchReport::from_file(&out_file).ok();
            let err = if code != 0 {
                Some(format!(
                    "scene `{}` ({role} #{run_index}) exited with code {code}",
                    scene.name
                ))
            } else if report.is_none() {
                Some(format!(
                    "scene `{}` ({role} #{run_index}) did not produce valid JSON at {}",
                    scene.name,
                    out_file.display()
                ))
            } else {
                None
            };
            SceneRunResult {
                outcome: CmdOutcome::Exited(code),
                report,
                error: err,
            }
        }
        Err(e) => SceneRunResult {
            outcome: CmdOutcome::Exited(-1),
            report: None,
            error: Some(format!("failed to launch scene `{}`: {e}", scene.name)),
        },
    }
}

// ---------------------------------------------------------------------------
// 6 Hard Anti-Reward-Hacking Gates
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct GateConfig {
    pub min_gain_pct: f64,
    pub max_draw_drift_pct: f64,
    pub max_low1_regress_pct: f64,
    pub max_vram_regress_mb: f64,
    pub max_ram_regress_mb: f64,
    pub vram_cap_mb: Option<f64>,
    pub ram_cap_mb: Option<f64>,
    pub max_luma_diff: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GateVerdict {
    pub gate: u8,
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct EvaluationSummary {
    pub gates: Vec<GateVerdict>,
    pub baseline_median_ms: f64,
    pub candidate_median_ms: f64,
    pub gain_pct: f64,
    pub keep: bool,
}

#[derive(Clone, Debug)]
pub struct SceneEvaluationInput {
    pub scene: SceneConfig,
    pub base_runs: Vec<SceneRunResult>,
    pub cand_runs: Vec<SceneRunResult>,
}

/// Evaluates Gates 1..=6 across all scenes. `gate1_ok` / `gate1_detail` come
/// from running the build/test/policy/clippy gate before scene execution.
pub fn evaluate_all_gates(
    gate1_ok: bool,
    gate1_detail: String,
    scenes: &[SceneEvaluationInput],
    cfg: &GateConfig,
) -> EvaluationSummary {
    let mut gates = Vec::with_capacity(6);

    // Gate 1: build + cargo test + cargo xtask policy + clippy
    gates.push(GateVerdict {
        gate: 1,
        name: "build_test_policy_clippy",
        passed: gate1_ok,
        detail: gate1_detail,
    });

    if !gate1_ok || scenes.is_empty() {
        let reason = if !gate1_ok {
            "skipped: Gate 1 failed"
        } else {
            "skipped: no scenes ran"
        };
        for (num, name) in [
            (2, "stability_watchdog_device"),
            (3, "draw_count_parity"),
            (4, "visual_sanity"),
            (5, "warmup_interleaved_median"),
            (6, "composite_gain_non_regression"),
        ] {
            gates.push(GateVerdict {
                gate: num,
                name,
                passed: false,
                detail: reason.to_owned(),
            });
        }
        return EvaluationSummary {
            gates,
            baseline_median_ms: 0.0,
            candidate_median_ms: 0.0,
            gain_pct: 0.0,
            keep: false,
        };
    }

    // Gate 2: exit code 0 + watchdog timeout + valid JSON + device_removed == false
    let mut g2_errs = Vec::new();
    let mut base_agg = Vec::new();
    let mut cand_agg = Vec::new();
    for item in scenes {
        let mut ok_base = Vec::new();
        for r in &item.base_runs {
            if let Some(err) = &r.error {
                g2_errs.push(format!("base: {err}"));
            } else if r.outcome != CmdOutcome::Exited(0) {
                g2_errs.push(format!("base scene `{}` outcome {:?}", item.scene.name, r.outcome));
            }
            if let Some(rep) = &r.report {
                if rep.device_removed {
                    g2_errs.push(format!("base scene `{}` reported device_removed", item.scene.name));
                }
                ok_base.push(rep.clone());
            }
        }
        let mut ok_cand = Vec::new();
        for r in &item.cand_runs {
            if let Some(err) = &r.error {
                g2_errs.push(format!("cand: {err}"));
            } else if r.outcome != CmdOutcome::Exited(0) {
                g2_errs.push(format!("cand scene `{}` outcome {:?}", item.scene.name, r.outcome));
            }
            if let Some(rep) = &r.report {
                if rep.device_removed {
                    g2_errs.push(format!("cand scene `{}` reported device_removed", item.scene.name));
                }
                ok_cand.push(rep.clone());
            }
        }
        if ok_base.is_empty() {
            g2_errs.push(format!("scene `{}` has no valid baseline report", item.scene.name));
        } else {
            base_agg.push((&item.scene, aggregate_runs(&ok_base).unwrap()));
        }
        if ok_cand.is_empty() {
            g2_errs.push(format!("scene `{}` has no valid candidate report", item.scene.name));
        } else {
            cand_agg.push((&item.scene, aggregate_runs(&ok_cand).unwrap()));
        }
    }
    gates.push(GateVerdict {
        gate: 2,
        name: "stability_watchdog_device",
        passed: g2_errs.is_empty(),
        detail: if g2_errs.is_empty() {
            "all runs exited 0 within timeout with device_removed=false".to_owned()
        } else {
            g2_errs.join("; ")
        },
    });

    if base_agg.len() != scenes.len() || cand_agg.len() != scenes.len() {
        for (num, name) in [
            (3, "draw_count_parity"),
            (4, "visual_sanity"),
            (5, "warmup_interleaved_median"),
            (6, "composite_gain_non_regression"),
        ] {
            gates.push(GateVerdict {
                gate: num,
                name,
                passed: false,
                detail: "skipped: missing baseline or candidate reports".to_owned(),
            });
        }
        return EvaluationSummary {
            gates,
            baseline_median_ms: 0.0,
            candidate_median_ms: 0.0,
            gain_pct: 0.0,
            keep: false,
        };
    }

    // Gate 3: avg_draws within ±max_draw_drift_pct of baseline
    let mut g3_errs = Vec::new();
    for ((scene, base), (_, cand)) in base_agg.iter().zip(cand_agg.iter()) {
        if base.draws_avg <= 0.0 {
            g3_errs.push(format!(
                "scene `{}` invalid baseline draws.avg {:.2} (must be > 0)",
                scene.name, base.draws_avg
            ));
        } else {
            let drift_pct = ((cand.draws_avg - base.draws_avg).abs() / base.draws_avg) * 100.0;
            if drift_pct > cfg.max_draw_drift_pct {
                g3_errs.push(format!(
                    "scene `{}` draws.avg drift {:.2}% (base {:.1}, cand {:.1}, limit {:.1}%)",
                    scene.name, drift_pct, base.draws_avg, cand.draws_avg, cfg.max_draw_drift_pct
                ));
            }
        }
    }
    gates.push(GateVerdict {
        gate: 3,
        name: "draw_count_parity",
        passed: g3_errs.is_empty(),
        detail: if g3_errs.is_empty() {
            format!("all scenes within +/-{:.1}% avg draws", cfg.max_draw_drift_pct)
        } else {
            g3_errs.join("; ")
        },
    });

    // Gate 4: visual sanity (zero_draw_world_frames == 0, black_frames == 0, optional PNG parity)
    let mut g4_errs = Vec::new();
    for (scene, cand) in &cand_agg {
        if cand.zero_draw_world_frames > 0 {
            g4_errs.push(format!(
                "scene `{}` had {} zero-draw world frames",
                scene.name, cand.zero_draw_world_frames
            ));
        }
        if cand.black_frames > 0 {
            g4_errs.push(format!(
                "scene `{}` had {} black frames",
                scene.name, cand.black_frames
            ));
        }
        if let (Some(base_png), Some(cand_png)) =
            (scene.checkpoint_png_base.as_deref(), scene.checkpoint_png_cand.as_deref())
        {
            match compare_checkpoint_pngs(base_png, cand_png) {
                Ok(p) => {
                    if p.cand_is_black {
                        g4_errs.push(format!(
                            "scene `{}` checkpoint PNG is black (luma {:.2})",
                            scene.name, p.cand_mean_luma
                        ));
                    } else if p.mean_abs_luma_diff > cfg.max_luma_diff {
                        g4_errs.push(format!(
                            "scene `{}` checkpoint PNG luma diff {:.2} > {:.2}",
                            scene.name, p.mean_abs_luma_diff, cfg.max_luma_diff
                        ));
                    }
                }
                Err(e) => g4_errs.push(format!("scene `{}` checkpoint PNG check failed: {e}", scene.name)),
            }
        }
    }
    gates.push(GateVerdict {
        gate: 4,
        name: "visual_sanity",
        passed: g4_errs.is_empty(),
        detail: if g4_errs.is_empty() {
            "zero_draw_world_frames=0, black_frames=0, checkpoint parity clean".to_owned()
        } else {
            g4_errs.join("; ")
        },
    });

    // Gate 5: warmup exclusion, input-script digest parity & minimum measured frames
    let mut g5_errs = Vec::new();
    for (idx, (scene, cand)) in cand_agg.iter().enumerate() {
        let base = &base_agg[idx].1;
        let min_frames = scene.frames.max(1);
        for (side, agg, runs) in [
            ("baseline", base, &scenes[idx].base_runs),
            ("candidate", cand, &scenes[idx].cand_runs),
        ] {
            if agg.warmup_frames < scene.warmup {
                g5_errs.push(format!(
                    "scene `{}` {side} warmup_frames {} < required {}",
                    scene.name, agg.warmup_frames, scene.warmup
                ));
            }
            if agg.world_frames < min_frames {
                g5_errs.push(format!(
                    "scene `{}` {side} world_frames {} < required {}",
                    scene.name, agg.world_frames, min_frames
                ));
            }
            if !(agg.frame_ms_median.is_finite() && agg.frame_ms_median > 0.0) {
                g5_errs.push(format!(
                    "scene `{}` {side} frame_ms.median {} must be > 0",
                    scene.name, agg.frame_ms_median
                ));
            } else if let Some(bad) = runs
                .iter()
                .filter_map(|r| r.report.as_ref())
                .find(|rep| !(rep.frame_ms_median.is_finite() && rep.frame_ms_median > 0.0))
            {
                g5_errs.push(format!(
                    "scene `{}` {side} repeat frame_ms.median {} must be > 0",
                    scene.name, bad.frame_ms_median
                ));
            }
        }
        if !scene.input_script.as_os_str().is_empty() {
            if base
                .input_script_sha256
                .as_deref()
                .is_none_or(str::is_empty)
            {
                g5_errs.push(format!(
                    "scene `{}` input_script configured but baseline reported null input_script_sha256",
                    scene.name
                ));
            }
            if cand
                .input_script_sha256
                .as_deref()
                .is_none_or(str::is_empty)
            {
                g5_errs.push(format!(
                    "scene `{}` input_script configured but candidate reported null input_script_sha256",
                    scene.name
                ));
            }
        }
        if base.input_script_sha256 != cand.input_script_sha256 {
            g5_errs.push(format!(
                "scene `{}` input_script_sha256 mismatch (base {:?}, cand {:?})",
                scene.name, base.input_script_sha256, cand.input_script_sha256
            ));
        }
        for (side, runs) in [
            ("baseline", &scenes[idx].base_runs),
            ("candidate", &scenes[idx].cand_runs),
        ] {
            for run in runs {
                if let Some(rep) = &run.report
                    && rep.input_script_sha256 != base.input_script_sha256
                {
                    g5_errs.push(format!(
                        "scene `{}` {side} repeat input_script_sha256 mismatch",
                        scene.name
                    ));
                    break;
                }
            }
        }
    }
    gates.push(GateVerdict {
        gate: 5,
        name: "warmup_interleaved_median",
        passed: g5_errs.is_empty(),
        detail: if g5_errs.is_empty() {
            "warmup excluded and full interleaved world-frame counts met".to_owned()
        } else {
            g5_errs.join("; ")
        },
    });

    // Gate 6: composite gain >= min_gain_pct + 1% low / VRAM / RAM non-regression
    let baseline_median_ms =
        base_agg.iter().map(|(_, r)| r.frame_ms_median).sum::<f64>() / base_agg.len() as f64;
    let candidate_median_ms =
        cand_agg.iter().map(|(_, r)| r.frame_ms_median).sum::<f64>() / cand_agg.len() as f64;
    let gain_pct = if baseline_median_ms > 0.0 {
        (baseline_median_ms - candidate_median_ms) / baseline_median_ms * 100.0
    } else {
        0.0
    };

    let mut g6_errs = Vec::new();
    if gain_pct < cfg.min_gain_pct {
        g6_errs.push(format!(
            "median frame_ms gain {:.2}% < min_gain_pct {:.2}% (base {:.2} ms, cand {:.2} ms)",
            gain_pct, cfg.min_gain_pct, baseline_median_ms, candidate_median_ms
        ));
    }
    for ((scene, base), (_, cand)) in base_agg.iter().zip(cand_agg.iter()) {
        if base.low_1pct_fps > 0.0 {
            let regress_pct = (base.low_1pct_fps - cand.low_1pct_fps) / base.low_1pct_fps * 100.0;
            if regress_pct > cfg.max_low1_regress_pct {
                g6_errs.push(format!(
                    "scene `{}` 1% low FPS regressed {:.2}% (base {:.2}, cand {:.2})",
                    scene.name, regress_pct, base.low_1pct_fps, cand.low_1pct_fps
                ));
            }
        }
        if let Some(cap) = cfg.vram_cap_mb
            && cand.vram_used_peak_mb > cap
        {
            g6_errs.push(format!(
                "scene `{}` VRAM peak {:.1} MB > cap {:.1} MB",
                scene.name, cand.vram_used_peak_mb, cap
            ));
        }
        if base.vram_budget_mb > 0.0
            && base.vram_used_peak_mb <= base.vram_budget_mb
            && cand.vram_used_peak_mb > base.vram_budget_mb
        {
            g6_errs.push(format!(
                "scene `{}` VRAM peak {:.1} MB exceeded budget {:.1} MB",
                scene.name, cand.vram_used_peak_mb, base.vram_budget_mb
            ));
        }
        let vram_delta = cand.vram_used_peak_mb - base.vram_used_peak_mb;
        if vram_delta > cfg.max_vram_regress_mb {
            g6_errs.push(format!(
                "scene `{}` VRAM peak regressed +{:.1} MB (limit +{:.1} MB)",
                scene.name, vram_delta, cfg.max_vram_regress_mb
            ));
        }
        if let Some(cap) = cfg.ram_cap_mb
            && cand.ram_private_commit_peak_mb > cap
        {
            g6_errs.push(format!(
                "scene `{}` RAM commit peak {:.1} MB > cap {:.1} MB",
                scene.name, cand.ram_private_commit_peak_mb, cap
            ));
        }
        let ram_delta = cand.ram_private_commit_peak_mb - base.ram_private_commit_peak_mb;
        if ram_delta > cfg.max_ram_regress_mb {
            g6_errs.push(format!(
                "scene `{}` RAM commit peak regressed +{:.1} MB (limit +{:.1} MB)",
                scene.name, ram_delta, cfg.max_ram_regress_mb
            ));
        }
    }
    gates.push(GateVerdict {
        gate: 6,
        name: "composite_gain_non_regression",
        passed: g6_errs.is_empty(),
        detail: if g6_errs.is_empty() {
            format!(
                "gain +{:.2}% (base {:.2} ms -> cand {:.2} ms), 1% low/VRAM/RAM non-regressed",
                gain_pct, baseline_median_ms, candidate_median_ms
            )
        } else {
            g6_errs.join("; ")
        },
    });

    let keep = gates.iter().all(|g| g.passed);
    EvaluationSummary {
        gates,
        baseline_median_ms,
        candidate_median_ms,
        gain_pct,
        keep,
    }
}

// ---------------------------------------------------------------------------
// Experiments JSONL Log & Outside-Repo Safety
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ExperimentEntry {
    pub timestamp_unix: u64,
    pub iteration: u32,
    pub hypothesis: String,
    pub git_sha: String,
    pub diff_stat: String,
    pub top_bottlenecks: Vec<(String, f64)>,
    pub baseline_median_ms: f64,
    pub candidate_median_ms: f64,
    pub gain_pct: f64,
    pub gates: Vec<GateVerdict>,
    pub decision: &'static str,
}

impl ExperimentEntry {
    pub fn to_jsonl(&self) -> String {
        let finite = |v: f64| if v.is_finite() { v } else { 0.0 };
        let bottlenecks: Vec<String> = self
            .top_bottlenecks
            .iter()
            .map(|(k, v)| format!("{{\"stage\":\"{}\",\"ms\":{:.3}}}", escape_json(k), finite(*v)))
            .collect();
        let gates: Vec<String> = self
            .gates
            .iter()
            .map(|g| {
                format!(
                    "{{\"gate\":{},\"name\":\"{}\",\"passed\":{},\"detail\":\"{}\"}}",
                    g.gate,
                    escape_json(g.name),
                    g.passed,
                    escape_json(&g.detail)
                )
            })
            .collect();
        format!(
            "{{\"timestamp_unix\":{},\"iteration\":{},\"hypothesis\":\"{}\",\"git_sha\":\"{}\",\"diff_stat\":\"{}\",\"top_bottlenecks\":[{}],\"baseline_median_ms\":{:.3},\"candidate_median_ms\":{:.3},\"gain_pct\":{:.3},\"gates\":[{}],\"decision\":\"{}\"}}",
            self.timestamp_unix,
            self.iteration,
            escape_json(&self.hypothesis),
            escape_json(&self.git_sha),
            escape_json(&self.diff_stat),
            bottlenecks.join(","),
            finite(self.baseline_median_ms),
            finite(self.candidate_median_ms),
            finite(self.gain_pct),
            gates.join(","),
            self.decision,
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

fn ensure_outside_repo(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        bail!("benchmark output path must be absolute: {}", path.display());
    }
    // Resolve symlinks and junctions in the existing prefix, then attach
    // the not-yet-created output components.
    let mut existing = path;
    let mut missing = Vec::new();
    let resolved = loop {
        match existing.canonicalize() {
            Ok(mut prefix) => {
                for part in missing.into_iter().rev() {
                    prefix.push(part);
                }
                break prefix;
            }
            Err(error) => {
                let (Some(parent), Some(name)) = (existing.parent(), existing.file_name()) else {
                    return Err(error).with_context(|| format!("resolve {}", path.display()));
                };
                if error.kind() != std::io::ErrorKind::NotFound {
                    return Err(error).with_context(|| format!("resolve {}", path.display()));
                }
                missing.push(name);
                existing = parent;
            }
        }
    };
    if let Ok(root) = crate::workspace_root().canonicalize()
        && resolved.starts_with(&root)
    {
        bail!("refusing to write benchmark output {} in the workspace", path.display());
    }
    for anc in resolved.ancestors() {
        if anc.join(ROOT_MARKER).is_file() {
            bail!(
                "refusing to write benchmark output {} inside VIctory repo ({})",
                path.display(),
                anc.display()
            );
        }
    }
    Ok(())
}

pub fn append_experiment(path: &Path, entry: &ExperimentEntry) -> Result<()> {
    ensure_outside_repo(path)?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut f = OpenOptions::new().create(true).append(true).open(path)?;
    writeln!(f, "{}", entry.to_jsonl())?;
    f.flush()?;
    Ok(())
}

pub fn load_recent_experiments_summary(path: &Path, limit: usize) -> String {
    let Ok(text) = std::fs::read_to_string(path) else {
        return String::new();
    };
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    let start = lines.len().saturating_sub(limit);
    lines[start..].join("\n")
}

// ---------------------------------------------------------------------------
// Orchestration: Interleaved A/B & Autonomous Loop
// ---------------------------------------------------------------------------

fn default_build_gate_cmds() -> Vec<String> {
    vec![
        "cargo build --locked".to_owned(),
        "cargo test --locked".to_owned(),
        "cargo run --locked -p xtask -- policy".to_owned(),
        "cargo clippy --workspace --all-targets --locked -- -D warnings -A deprecated".to_owned(),
    ]
}

fn run_build_gate(cmds: &[String], repo_dir: &Path, timeout: Duration) -> (bool, String) {
    for cmd in cmds {
        match run_shell_with_timeout(cmd, Some(repo_dir), &[], timeout) {
            Ok(CmdOutcome::Exited(0)) => {}
            Ok(CmdOutcome::Exited(code)) => {
                return (false, format!("gate command `{cmd}` exited {code}"));
            }
            Ok(CmdOutcome::TimedOut) => {
                return (false, format!("gate command `{cmd}` timed out"));
            }
            Err(e) => {
                return (false, format!("gate command `{cmd}` failed: {e}"));
            }
        }
    }
    (true, format!("{} build/test/policy/clippy checks passed", cmds.len()))
}

fn run_interleaved_scenes(
    playlist: &Playlist,
    base_cmd: &str,
    cand_cmd: &str,
    rounds: usize,
    work_dir: &Path,
    timeout: Duration,
) -> Vec<SceneEvaluationInput> {
    let rounds = rounds.max(1);
    let mut results = Vec::with_capacity(playlist.scenes.len());
    for scene in &playlist.scenes {
        let mut base_runs = Vec::new();
        let mut cand_runs = Vec::new();
        // Interleaved schedule: base -> cand -> base (-> cand -> base ...)
        for r in 0..rounds {
            base_runs.push(run_scene_once(base_cmd, scene, "base", r * 2, work_dir, timeout));
            cand_runs.push(run_scene_once(cand_cmd, scene, "cand", r, work_dir, timeout));
        }
        base_runs.push(run_scene_once(
            base_cmd,
            scene,
            "base",
            rounds * 2,
            work_dir,
            timeout,
        ));
        results.push(SceneEvaluationInput {
            scene: scene.clone(),
            base_runs,
            cand_runs,
        });
    }
    results
}

fn git_stdout(repo_dir: &Path, args: &[&str]) -> String {
    Command::new("git")
        .args(args)
        .current_dir(repo_dir)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}

fn git_untracked_files(repo_dir: &Path) -> std::collections::BTreeSet<String> {
    git_stdout(
        repo_dir,
        &[
            "-c",
            "core.quotepath=false",
            "ls-files",
            "--others",
            "--exclude-standard",
        ],
    )
    .lines()
    .map(str::trim)
    .filter(|l| !l.is_empty())
    .map(ToOwned::to_owned)
    .collect()
}

fn git_commit_kept(repo_dir: &Path, iter: u32, hypothesis: &str) -> Result<String> {
    let dirty = git_stdout(repo_dir, &["status", "--porcelain"]);
    if dirty.is_empty() {
        return Ok(git_stdout(repo_dir, &["rev-parse", "--short", "HEAD"]));
    }
    let add = Command::new("git")
        .args(["add", "-A"])
        .current_dir(repo_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("git add -A")?;
    if !add.success() {
        bail!("git add -A failed while committing kept iteration #{iter}");
    }
    let msg = format!("bench-loop iter #{iter}: {hypothesis}");
    let commit = Command::new("git")
        .args([
            "-c",
            "user.name=victory-bench",
            "-c",
            "user.email=bench@localhost",
            "commit",
            "-m",
            &msg,
        ])
        .current_dir(repo_dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .context("git commit")?;
    if !commit.success() {
        bail!("git commit failed while saving kept iteration #{iter}");
    }
    Ok(git_stdout(repo_dir, &["rev-parse", "--short", "HEAD"]))
}

fn git_revert_worktree(
    repo_dir: &Path,
    pre_sha: &str,
    pre_untracked: &std::collections::BTreeSet<String>,
) {
    if !pre_sha.is_empty() {
        let _ = Command::new("git")
            .args(["reset", "--hard", pre_sha])
            .current_dir(repo_dir)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        for rel in git_untracked_files(repo_dir) {
            if !pre_untracked.contains(&rel) {
                let p = repo_dir.join(&rel);
                if p.is_dir() {
                    let _ = std::fs::remove_dir_all(&p);
                } else {
                    let _ = std::fs::remove_file(&p);
                }
            }
        }
    }
}

fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

fn print_summary(label: &str, eval: &EvaluationSummary) {
    println!(
        "=== bench-loop ({label}): {} | base={:.2}ms cand={:.2}ms gain={:+.2}% ===",
        if eval.keep { "KEEP" } else { "REJECT" },
        eval.baseline_median_ms,
        eval.candidate_median_ms,
        eval.gain_pct
    );
    for g in &eval.gates {
        println!(
            "  Gate {} [{}] {:4}: {}",
            g.gate,
            g.name,
            if g.passed { "PASS" } else { "FAIL" },
            g.detail
        );
    }
}

pub fn run(args: Args) -> Result<()> {
    let mode = args.mode;
    let eval = run_with_summary(args)?;
    if mode == Mode::Ab && !eval.keep {
        bail!("bench-loop rejected candidate (see gate verdicts above)");
    }
    Ok(())
}

pub fn run_with_summary(args: Args) -> Result<EvaluationSummary> {
    let playlist = Playlist::load(&args.config)?;
    let repo_dir = playlist.repo_dir.clone().unwrap_or_else(crate::workspace_root);
    let work_dir = args
        .work_dir
        .clone()
        .unwrap_or_else(|| std::env::temp_dir().join("victory-bench-loop"));
    ensure_outside_repo(&work_dir)?;
    std::fs::create_dir_all(&work_dir)?;
    for scene in &playlist.scenes {
        for path in [&scene.checkpoint_png_base, &scene.checkpoint_png_cand].into_iter().flatten() {
            ensure_outside_repo(path)?;
        }
    }

    let experiments_path = args.experiments.clone().or(playlist.experiments.clone());
    if let Some(exp) = experiments_path.as_deref() {
        ensure_outside_repo(exp)?;
    }

    let gate_cfg = GateConfig {
        min_gain_pct: args.min_gain_pct,
        max_draw_drift_pct: args.max_draw_drift_pct,
        max_low1_regress_pct: args.max_low1_regress_pct,
        max_vram_regress_mb: args.max_vram_regress_mb,
        max_ram_regress_mb: args.max_ram_regress_mb,
        vram_cap_mb: args.vram_cap_mb.or(playlist.vram_cap_mb),
        ram_cap_mb: args.ram_cap_mb.or(playlist.ram_cap_mb),
        max_luma_diff: args.max_luma_diff,
    };

    let build_cmds = if playlist.build_gate_cmds.is_empty() {
        default_build_gate_cmds()
    } else {
        playlist.build_gate_cmds.clone()
    };
    let timeout = Duration::from_secs(args.timeout_secs);
    let cand_cmd = args
        .cand_cmd
        .clone()
        .or(playlist.runner_cmd.clone())
        .ok_or_else(|| anyhow!("missing `runner_cmd` in playlist or `--cand-cmd`"))?;
    let base_cmd = args
        .base_cmd
        .clone()
        .or(playlist.base_cmd.clone())
        .unwrap_or_else(|| cand_cmd.clone());

    match args.mode {
        Mode::Ab => {
            let (g1_ok, g1_detail) = if args.skip_build_gate {
                (true, "skipped via --skip-build-gate".to_owned())
            } else {
                run_build_gate(&build_cmds, &repo_dir, timeout)
            };
            let scene_results = if g1_ok {
                run_interleaved_scenes(&playlist, &base_cmd, &cand_cmd, args.runs, &work_dir, timeout)
            } else {
                Vec::new()
            };
            let eval = evaluate_all_gates(g1_ok, g1_detail, &scene_results, &gate_cfg);
            print_summary("ab", &eval);

            if let Some(exp_path) = experiments_path.as_deref() {
                let bottlenecks = scene_results
                    .first()
                    .and_then(|s| s.base_runs.first())
                    .and_then(|r| r.report.as_ref())
                    .map(|r| r.top_bottlenecks(4))
                    .unwrap_or_default();
                let entry = ExperimentEntry {
                    timestamp_unix: unix_now(),
                    iteration: 1,
                    hypothesis: "ab comparison".to_owned(),
                    git_sha: git_stdout(&repo_dir, &["rev-parse", "--short", "HEAD"]),
                    diff_stat: git_stdout(&repo_dir, &["diff", "--shortstat"]),
                    top_bottlenecks: bottlenecks,
                    baseline_median_ms: eval.baseline_median_ms,
                    candidate_median_ms: eval.candidate_median_ms,
                    gain_pct: eval.gain_pct,
                    gates: eval.gates.clone(),
                    decision: if eval.keep { "keep" } else { "revert" },
                };
                append_experiment(exp_path, &entry)?;
            }
            Ok(eval)
        }
        Mode::Loop => {
            let agent_cmd = args
                .agent_cmd
                .as_deref()
                .ok_or_else(|| anyhow!("`--mode loop` requires `--agent-cmd`"))?;
            let fast_cmd = args.fast_cmd.clone().or(playlist.fast_cmd.clone());
            let exp_path = experiments_path
                .unwrap_or_else(|| work_dir.join("experiments.jsonl"));
            ensure_outside_repo(&exp_path)?;

            // Protect unrelated user WIP when operating inside a git repository.
            let initial_head = git_stdout(&repo_dir, &["rev-parse", "HEAD"]);
            if !initial_head.is_empty() {
                let dirty = git_stdout(&repo_dir, &["status", "--porcelain"]);
                if !dirty.is_empty() {
                    bail!(
                        "`--mode loop` requires a clean git worktree in {} before starting (commit or stash pending edits first)",
                        repo_dir.display()
                    );
                }
            }

            // Baseline measurement before loop iterations (fail fast if baseline is broken).
            let mut baseline_by_scene: Vec<Vec<SceneRunResult>> = Vec::with_capacity(playlist.scenes.len());
            for scene in &playlist.scenes {
                let mut runs = Vec::new();
                for r in 0..args.runs.max(1) {
                    let res = run_scene_once(&base_cmd, scene, "base", r, &work_dir, timeout);
                    if let Some(err) = &res.error {
                        bail!("initial baseline failed for scene `{}`: {err}", scene.name);
                    }
                    if res.outcome != CmdOutcome::Exited(0) {
                        bail!(
                            "initial baseline failed for scene `{}`: {:?}",
                            scene.name,
                            res.outcome
                        );
                    }
                    if res.report.as_ref().is_some_and(|rep| rep.device_removed) {
                        bail!(
                            "initial baseline reported device_removed for scene `{}`",
                            scene.name
                        );
                    }
                    runs.push(res);
                }
                baseline_by_scene.push(runs);
            }

            let mut last_eval = EvaluationSummary {
                gates: Vec::new(),
                baseline_median_ms: 0.0,
                candidate_median_ms: 0.0,
                gain_pct: 0.0,
                keep: false,
            };

            for iter in 1..=args.iterations {
                let pre_sha = git_stdout(&repo_dir, &["rev-parse", "HEAD"]);
                let pre_untracked = if pre_sha.is_empty() {
                    std::collections::BTreeSet::new()
                } else {
                    git_untracked_files(&repo_dir)
                };
                let bottlenecks = baseline_by_scene
                    .first()
                    .and_then(|v| v.first())
                    .and_then(|r| r.report.as_ref())
                    .map(|r| r.top_bottlenecks(4))
                    .unwrap_or_default();
                let history = load_recent_experiments_summary(&exp_path, 10);
                let prompt = build_agent_prompt(iter, &bottlenecks, &history, &gate_cfg);
                let prompt_file = work_dir.join(format!("prompt_iter_{iter}.txt"));
                {
                    let mut f = File::create(&prompt_file)?;
                    f.write_all(prompt.as_bytes())?;
                }

                let agent_envs = [
                    ("VICTORY_BENCH_ITER", iter.to_string()),
                    (
                        "VICTORY_BENCH_PROMPT_FILE",
                        prompt_file.display().to_string(),
                    ),
                    (
                        "VICTORY_BENCH_EXPERIMENTS_FILE",
                        exp_path.display().to_string(),
                    ),
                ];
                let (agent_outcome, agent_out, _) = spawn_shell_cmd(
                    agent_cmd,
                    Some(&repo_dir),
                    &agent_envs,
                    Some(prompt.as_bytes()),
                    timeout,
                )?;
                let hypothesis = agent_out
                    .lines()
                    .map(str::trim)
                    .find(|l| !l.is_empty())
                    .unwrap_or("agent iteration")
                    .chars()
                    .take(200)
                    .collect::<String>();

                let diff_stat = git_stdout(&repo_dir, &["diff", "--shortstat"]);

                let agent_err = match agent_outcome {
                    CmdOutcome::Exited(0) => None,
                    CmdOutcome::Exited(code) => Some(format!("agent command exited with code {code}")),
                    CmdOutcome::TimedOut => Some(format!(
                        "agent command timed out after {}s",
                        timeout.as_secs()
                    )),
                };

                // Optional fast offline pre-filter (skipped if agent failed).
                let mut prefilter_ok = agent_err.is_none();
                let mut prefilter_msg = agent_err.unwrap_or_default();
                if prefilter_ok && let Some(fc) = fast_cmd.as_deref() {
                    match run_shell_with_timeout(fc, Some(&repo_dir), &[], timeout)? {
                        CmdOutcome::Exited(0) => {}
                        CmdOutcome::Exited(c) => {
                            prefilter_ok = false;
                            prefilter_msg = format!("fast pre-filter exited {c}");
                        }
                        CmdOutcome::TimedOut => {
                            prefilter_ok = false;
                            prefilter_msg = "fast pre-filter timed out".to_owned();
                        }
                    }
                }

                let (g1_ok, g1_detail) = if !prefilter_ok {
                    (false, prefilter_msg)
                } else if args.skip_build_gate {
                    (true, "skipped via --skip-build-gate".to_owned())
                } else {
                    run_build_gate(&build_cmds, &repo_dir, timeout)
                };

                let scene_inputs = if g1_ok {
                    playlist
                        .scenes
                        .iter()
                        .enumerate()
                        .map(|(si, scene)| {
                            let mut cand_runs = Vec::new();
                            for r in 0..args.runs.max(1) {
                                cand_runs.push(run_scene_once(
                                    &cand_cmd,
                                    scene,
                                    "cand",
                                    (iter as usize) * 100 + r,
                                    &work_dir,
                                    timeout,
                                ));
                            }
                            SceneEvaluationInput {
                                scene: scene.clone(),
                                base_runs: baseline_by_scene[si].clone(),
                                cand_runs,
                            }
                        })
                        .collect()
                } else {
                    Vec::new()
                };

                let eval = evaluate_all_gates(g1_ok, g1_detail, &scene_inputs, &gate_cfg);
                print_summary(&format!("loop #{iter}"), &eval);

                let (decision, git_sha) = if eval.keep {
                    // Promote candidate runs to new baseline and commit kept edits
                    // so subsequent reverts only undo future failed iterations.
                    for (si, inp) in scene_inputs.iter().enumerate() {
                        baseline_by_scene[si] = inp.cand_runs.clone();
                    }
                    let sha = if !pre_sha.is_empty() {
                        git_commit_kept(&repo_dir, iter, &hypothesis)?
                    } else {
                        String::new()
                    };
                    ("keep", sha)
                } else {
                    git_revert_worktree(&repo_dir, &pre_sha, &pre_untracked);
                    (
                        "revert",
                        git_stdout(&repo_dir, &["rev-parse", "--short", "HEAD"]),
                    )
                };

                let entry = ExperimentEntry {
                    timestamp_unix: unix_now(),
                    iteration: iter,
                    hypothesis,
                    git_sha,
                    diff_stat,
                    top_bottlenecks: bottlenecks,
                    baseline_median_ms: eval.baseline_median_ms,
                    candidate_median_ms: eval.candidate_median_ms,
                    gain_pct: eval.gain_pct,
                    gates: eval.gates.clone(),
                    decision,
                };
                append_experiment(&exp_path, &entry)?;
                last_eval = eval;
            }
            Ok(last_eval)
        }
    }
}

fn build_agent_prompt(
    iter: u32,
    bottlenecks: &[(String, f64)],
    history: &str,
    cfg: &GateConfig,
) -> String {
    let mut s = format!(
        "Optimization iteration #{iter}.\nTarget: reduce median frame_ms by at least {:.2}% without skipping draws (+/-{:.1}%), without black/zero-draw frames, without device_removed, and without regressing 1% low FPS, VRAM, or RAM.\n",
        cfg.min_gain_pct, cfg.max_draw_drift_pct
    );
    if !bottlenecks.is_empty() {
        s.push_str("Top framelog bottleneck stages (ms/frame):\n");
        for (stage, ms) in bottlenecks {
            s.push_str(&format!("  - {stage}: {ms:.3} ms\n"));
        }
    }
    if !history.is_empty() {
        s.push_str("Recent experiment history (do not repeat reverted dead ends):\n");
        s.push_str(history);
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "victory-xtask-bench-{tag}-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[allow(clippy::too_many_arguments)]
    fn bench_json(
        world_frames: u64,
        warmup_frames: u64,
        median_ms: f64,
        low_1pct_fps: f64,
        draws_avg: f64,
        device_removed: bool,
        zero_draw: u64,
        black: u64,
        vram_peak: f64,
        ram_peak: f64,
    ) -> String {
        let p95 = median_ms * 1.1;
        let p99 = median_ms * 1.2;
        let low5 = low_1pct_fps * 1.1;
        format!(
            r#"{{
  "world_frames": {world_frames},
  "warmup_frames": {warmup_frames},
  "frame_ms": {{
    "avg": {median_ms:.3},
    "median": {median_ms:.3},
    "p95": {p95:.3},
    "p99": {p99:.3},
    "low_5pct_fps": {low5:.3},
    "low_1pct_fps": {low_1pct_fps:.3}
  }},
  "draws": {{ "avg": {draws_avg:.2}, "min": 900, "max": 1100 }},
  "device_removed": {device_removed},
  "zero_draw_world_frames": {zero_draw},
  "black_frames": {black},
  "memory": {{
    "vram_used_peak_mb": {vram_peak:.2},
    "vram_budget_mb": 8000.00,
    "ram_private_commit_peak_mb": {ram_peak:.2},
    "ram_working_set_peak_mb": 4000.00
  }},
  "framelog": {{
    "prepare_ms": 4.5,
    "card_wait_ms": 8.2,
    "card_idle_ms": 1.1,
    "copy_ms": 0.9,
    "copyback_ms": 0.2,
    "remap_ms": 0.3,
    "refresh_ms": 1.5,
    "pipeline_wait_ms": 0.1,
    "guard_faults": 42.0,
    "card_sync_ms": 1.2,
    "card_check_ms": 0.4,
    "card_lists_ms": 6.0
  }},
  "input_script_sha256": "0123456789abcdef"
}}
"#
        )
    }

    /// Writes a cross-platform mock runner script that copies `payload_file` to
    /// `$VICTORY_BENCH_OUT` (or `%VICTORY_BENCH_OUT%`) and exits with `exit_code`.
    fn write_mock_runner(dir: &Path, name: &str, payload: &str, exit_code: i32) -> String {
        let json_path = dir.join(format!("{name}_payload.json"));
        std::fs::write(&json_path, payload).unwrap();
        if cfg!(windows) {
            let script = dir.join(format!("{name}.cmd"));
            std::fs::write(
                &script,
                format!(
                    "@echo off\r\ncopy /Y \"{}\" \"%VICTORY_BENCH_OUT%\" >nul\r\nexit /b {exit_code}\r\n",
                    json_path.display()
                ),
            )
            .unwrap();
            format!("\"{}\"", script.display())
        } else {
            let script = dir.join(format!("{name}.sh"));
            std::fs::write(
                &script,
                format!(
                    "#!/bin/sh\ncp \"{}\" \"$VICTORY_BENCH_OUT\"\nexit {exit_code}\n",
                    json_path.display()
                ),
            )
            .unwrap();
            format!("sh \"{}\"", script.display())
        }
    }

    fn write_playlist(dir: &Path, base_cmd: &str, cand_cmd: &str, extra_scene: &str) -> PathBuf {
        let p = dir.join("playlist.toml");
        let body = format!(
            "base_cmd = '{}'\nrunner_cmd = '{}'\nbuild_gate_cmds = []\n\n[[scenes]]\nname = 'route_loop'\ninput_script = '{}'\nwarmup = 30\nframes = 120\n{extra_scene}\n",
            base_cmd,
            cand_cmd,
            dir.join("script.jsonl").display()
        );
        std::fs::write(&p, body).unwrap();
        p
    }

    fn base_args(config: PathBuf, work_dir: PathBuf, experiments: PathBuf) -> Args {
        Args {
            config,
            mode: Mode::Ab,
            runs: 1,
            min_gain_pct: 2.0,
            max_draw_drift_pct: 15.0,
            max_low1_regress_pct: 2.0,
            max_vram_regress_mb: 128.0,
            max_ram_regress_mb: 256.0,
            vram_cap_mb: None,
            ram_cap_mb: None,
            max_luma_diff: 12.0,
            timeout_secs: 30,
            iterations: 2,
            agent_cmd: None,
            fast_cmd: None,
            experiments: Some(experiments),
            base_cmd: None,
            cand_cmd: None,
            work_dir: Some(work_dir),
            skip_build_gate: true,
        }
    }

    /// Minimal valid 8-bit RGB PNG encoder (stored zlib block) for PNG parity tests.
    fn encode_stored_rgb_png(width: u32, height: u32, rgb: &[u8]) -> Vec<u8> {
        fn crc32(bytes: &[u8]) -> u32 {
            let mut crc = 0xFFFF_FFFFu32;
            for &b in bytes {
                crc ^= u32::from(b);
                for _ in 0..8 {
                    crc = (crc >> 1) ^ (0xEDB8_8320 & (0u32.wrapping_sub(crc & 1)));
                }
            }
            !crc
        }
        fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
            out.extend_from_slice(&(data.len() as u32).to_be_bytes());
            let start = out.len();
            out.extend_from_slice(kind);
            out.extend_from_slice(data);
            let crc = crc32(&out[start..]);
            out.extend_from_slice(&crc.to_be_bytes());
        }
        let mut raw = Vec::new();
        let row_len = (width as usize) * 3;
        for y in 0..height as usize {
            raw.push(0u8); // filter None
            raw.extend_from_slice(&rgb[y * row_len..(y + 1) * row_len]);
        }
        // zlib wrapper with single stored deflate block (raw.len() < 65535).
        let mut zlib = vec![0x78, 0x01, 0x01];
        let len = raw.len() as u16;
        zlib.extend_from_slice(&len.to_le_bytes());
        zlib.extend_from_slice(&(!len).to_le_bytes());
        zlib.extend_from_slice(&raw);
        let (mut s1, mut s2) = (1u32, 0u32);
        for &b in &raw {
            s1 = (s1 + u32::from(b)) % 65521;
            s2 = (s2 + s1) % 65521;
        }
        zlib.extend_from_slice(&((s2 << 16) | s1).to_be_bytes());

        let mut png = vec![137, 80, 78, 71, 13, 10, 26, 10];
        let mut ihdr = Vec::new();
        ihdr.extend_from_slice(&width.to_be_bytes());
        ihdr.extend_from_slice(&height.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        chunk(&mut png, b"IHDR", &ihdr);
        chunk(&mut png, b"IDAT", &zlib);
        chunk(&mut png, b"IEND", &[]);
        png
    }

    #[test]
    fn keep_on_real_win() {
        let dir = unique_tmp_dir("keep");
        let base_cmd = write_mock_runner(
            &dir,
            "base",
            &bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cand_cmd = write_mock_runner(
            &dir,
            "cand",
            &bench_json(120, 30, 17.0, 45.0, 1010.0, false, 0, 0, 4020.0, 6050.0),
            0,
        );
        let cfg = write_playlist(&dir, &base_cmd, &cand_cmd, "");
        let exp = dir.join("experiments.jsonl");
        let eval = run_with_summary(base_args(cfg, dir.join("work"), exp.clone())).unwrap();
        assert!(eval.keep, "{:#?}", eval.gates);
        assert!((eval.gain_pct - 15.0).abs() < 1e-3);
        let log = std::fs::read_to_string(&exp).unwrap();
        assert!(log.contains("\"decision\":\"keep\""));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_on_draw_skip() {
        let dir = unique_tmp_dir("drawskip");
        let base_cmd = write_mock_runner(
            &dir,
            "base",
            &bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        // Cheating candidate skips 30% of draws to look faster.
        let cand_cmd = write_mock_runner(
            &dir,
            "cand",
            &bench_json(120, 30, 12.0, 60.0, 700.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cfg = write_playlist(&dir, &base_cmd, &cand_cmd, "");
        let eval = run_with_summary(base_args(cfg, dir.join("work"), dir.join("exp.jsonl"))).unwrap();
        assert!(!eval.keep);
        assert!(eval.gates.iter().find(|g| g.gate == 2).unwrap().passed);
        let g3 = eval.gates.iter().find(|g| g.gate == 3).unwrap();
        assert!(!g3.passed);
        assert!(g3.detail.contains("draws.avg drift"), "{}", g3.detail);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_on_device_removed() {
        let dir = unique_tmp_dir("devrem");
        let base_cmd = write_mock_runner(
            &dir,
            "base",
            &bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cand_cmd = write_mock_runner(
            &dir,
            "cand",
            &bench_json(120, 30, 15.0, 50.0, 1000.0, true, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cfg = write_playlist(&dir, &base_cmd, &cand_cmd, "");
        let eval = run_with_summary(base_args(cfg, dir.join("work"), dir.join("exp.jsonl"))).unwrap();
        assert!(!eval.keep);
        let g2 = eval.gates.iter().find(|g| g.gate == 2).unwrap();
        assert!(!g2.passed);
        assert!(g2.detail.contains("device_removed"), "{}", g2.detail);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_on_black_frame_and_png_parity() {
        let dir = unique_tmp_dir("black");
        let base_png = dir.join("base.png");
        let cand_png = dir.join("cand.png");
        std::fs::write(&base_png, encode_stored_rgb_png(4, 4, &[120u8; 4 * 4 * 3])).unwrap();
        std::fs::write(&cand_png, encode_stored_rgb_png(4, 4, &[0u8; 4 * 4 * 3])).unwrap();

        let base_cmd = write_mock_runner(
            &dir,
            "base",
            &bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cand_cmd = write_mock_runner(
            &dir,
            "cand",
            &bench_json(120, 30, 15.0, 50.0, 1000.0, false, 0, 1, 4000.0, 6000.0),
            0,
        );
        let extra = format!(
            "checkpoint_png_base = '{}'\ncheckpoint_png_cand = '{}'\n",
            base_png.display(),
            cand_png.display()
        );
        let cfg = write_playlist(&dir, &base_cmd, &cand_cmd, &extra);
        let eval = run_with_summary(base_args(cfg, dir.join("work"), dir.join("exp.jsonl"))).unwrap();
        assert!(!eval.keep);
        let g4 = eval.gates.iter().find(|g| g.gate == 4).unwrap();
        assert!(!g4.passed);
        assert!(g4.detail.contains("black frames"));
        assert!(g4.detail.contains("checkpoint PNG is black"));

        // Verify matching bright PNGs pass parity cleanly.
        std::fs::write(&cand_png, encode_stored_rgb_png(4, 4, &[124u8; 4 * 4 * 3])).unwrap();
        let parity = compare_checkpoint_pngs(&base_png, &cand_png).unwrap();
        assert!(!parity.cand_is_black);
        assert!((parity.mean_abs_luma_diff - 4.0).abs() < 1e-3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_on_vram_breach() {
        let dir = unique_tmp_dir("vram");
        let base_cmd = write_mock_runner(
            &dir,
            "base",
            &bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        // Faster, but leaks +600 MB VRAM past regression limit and cap.
        let cand_cmd = write_mock_runner(
            &dir,
            "cand",
            &bench_json(120, 30, 16.0, 48.0, 1000.0, false, 0, 0, 4600.0, 6000.0),
            0,
        );
        let cfg = write_playlist(&dir, &base_cmd, &cand_cmd, "");
        let mut args = base_args(cfg, dir.join("work"), dir.join("exp.jsonl"));
        args.vram_cap_mb = Some(4500.0);
        let eval = run_with_summary(args).unwrap();
        assert!(!eval.keep);
        let g6 = eval.gates.iter().find(|g| g.gate == 6).unwrap();
        assert!(!g6.passed);
        assert!(g6.detail.contains("VRAM peak"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_on_noise_below_threshold() {
        let dir = unique_tmp_dir("noise");
        let base_cmd = write_mock_runner(
            &dir,
            "base",
            &bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        // Only 0.5% gain (19.9 ms vs 20.0 ms), below 2.0% threshold.
        let cand_cmd = write_mock_runner(
            &dir,
            "cand",
            &bench_json(120, 30, 19.9, 40.2, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cfg = write_playlist(&dir, &base_cmd, &cand_cmd, "");
        let eval = run_with_summary(base_args(cfg, dir.join("work"), dir.join("exp.jsonl"))).unwrap();
        assert!(!eval.keep);
        let g6 = eval.gates.iter().find(|g| g.gate == 6).unwrap();
        assert!(!g6.passed);
        assert!(g6.detail.contains("min_gain_pct"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loop_mode_records_experiments_jsonl_and_dead_end_history() {
        let dir = unique_tmp_dir("loop");
        let state_file = dir.join("iter_state.json");
        let base_json = bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0);
        let bad_json = bench_json(120, 30, 19.8, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0);
        let good_json = bench_json(120, 30, 16.0, 48.0, 1000.0, false, 0, 0, 4000.0, 6000.0);
        let bad_path = dir.join("bad.json");
        let good_path = dir.join("good.json");
        std::fs::write(&bad_path, &bad_json).unwrap();
        std::fs::write(&good_path, &good_json).unwrap();

        let base_cmd = write_mock_runner(&dir, "base", &base_json, 0);
        let cand_cmd = if cfg!(windows) {
            let s = dir.join("cand_dynamic.cmd");
            std::fs::write(
                &s,
                format!(
                    "@echo off\r\ncopy /Y \"{}\" \"%VICTORY_BENCH_OUT%\" >nul\r\nexit /b 0\r\n",
                    state_file.display()
                ),
            )
            .unwrap();
            format!("\"{}\"", s.display())
        } else {
            let s = dir.join("cand_dynamic.sh");
            std::fs::write(
                &s,
                format!(
                    "#!/bin/sh\ncp \"{}\" \"$VICTORY_BENCH_OUT\"\nexit 0\n",
                    state_file.display()
                ),
            )
            .unwrap();
            format!("sh \"{}\"", s.display())
        };

        // Mock agent: iter 1 writes noise payload (reverted); iter 2 checks prompt history
        // and writes real win payload (kept).
        let seen_prompt_2 = dir.join("seen_prompt_2.txt");
        let agent_cmd = if cfg!(windows) {
            let s = dir.join("mock_agent.cmd");
            std::fs::write(
                &s,
                format!(
                    "@echo off\r\nif \"%VICTORY_BENCH_ITER%\"==\"1\" (\r\n  copy /Y \"{}\" \"{}\" >nul\r\n  echo hypothesis: noisy micro tweak\r\n) else (\r\n  copy /Y \"%VICTORY_BENCH_PROMPT_FILE%\" \"{}\" >nul\r\n  copy /Y \"{}\" \"{}\" >nul\r\n  echo hypothesis: coalesce hot upload path\r\n)\r\nexit /b 0\r\n",
                    bad_path.display(),
                    state_file.display(),
                    seen_prompt_2.display(),
                    good_path.display(),
                    state_file.display(),
                ),
            )
            .unwrap();
            format!("\"{}\"", s.display())
        } else {
            let s = dir.join("mock_agent.sh");
            std::fs::write(
                &s,
                format!(
                    "#!/bin/sh\nif [ \"$VICTORY_BENCH_ITER\" = \"1\" ]; then\n  cp \"{}\" \"{}\"\n  echo \"hypothesis: noisy micro tweak\"\nelse\n  cp \"$VICTORY_BENCH_PROMPT_FILE\" \"{}\"\n  cp \"{}\" \"{}\"\n  echo \"hypothesis: coalesce hot upload path\"\nfi\n",
                    bad_path.display(),
                    state_file.display(),
                    seen_prompt_2.display(),
                    good_path.display(),
                    state_file.display(),
                ),
            )
            .unwrap();
            format!("sh \"{}\"", s.display())
        };

        let cfg = dir.join("playlist.toml");
        std::fs::write(
            &cfg,
            format!(
                "repo_dir = '{}'\nbase_cmd = '{}'\nrunner_cmd = '{}'\n\n[[scenes]]\nname = 'loop_scene'\ninput_script = '{}'\nwarmup = 30\nframes = 120\n",
                dir.display(),
                base_cmd,
                cand_cmd,
                dir.join("in.jsonl").display()
            ),
        )
        .unwrap();

        let exp = dir.join("experiments.jsonl");
        let mut args = base_args(cfg, dir.join("work"), exp.clone());
        args.mode = Mode::Loop;
        args.iterations = 2;
        args.agent_cmd = Some(agent_cmd);
        let eval = run_with_summary(args).unwrap();
        assert!(eval.keep, "{:#?}", eval.gates);

        let lines: Vec<String> = std::fs::read_to_string(&exp)
            .unwrap()
            .lines()
            .map(ToOwned::to_owned)
            .collect();
        assert_eq!(lines.len(), 2, "{lines:#?}");
        assert!(lines[0].contains("\"decision\":\"revert\""));
        assert!(lines[0].contains("noisy micro tweak"));
        assert!(lines[0].contains("\"card_wait_ms\""));
        assert!(lines[1].contains("\"decision\":\"keep\""));
        assert!(lines[1].contains("coalesce hot upload path"));

        // Iteration 2 prompt included iteration 1 dead-end history and bottleneck stages.
        let p2 = std::fs::read_to_string(&seen_prompt_2).unwrap();
        assert!(p2.contains("noisy micro tweak"), "{p2}");
        assert!(p2.contains("card_wait_ms"), "{p2}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_on_input_script_sha_mismatch() {
        let dir = unique_tmp_dir("sha");
        let base_cmd = write_mock_runner(
            &dir,
            "base",
            &bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cand_payload = bench_json(120, 30, 15.0, 50.0, 1000.0, false, 0, 0, 4000.0, 6000.0)
            .replace("\"0123456789abcdef\"", "null");
        let cand_cmd = write_mock_runner(&dir, "cand", &cand_payload, 0);
        let cfg = write_playlist(&dir, &base_cmd, &cand_cmd, "");
        let eval = run_with_summary(base_args(cfg, dir.join("work"), dir.join("exp.jsonl"))).unwrap();
        assert!(!eval.keep);
        let g5 = eval.gates.iter().find(|g| g.gate == 5).unwrap();
        assert!(!g5.passed);
        assert!(g5.detail.contains("input_script_sha256"), "{}", g5.detail);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_on_gate1_failure_marks_all_downstream_gates_and_writes_valid_json() {
        let dir = unique_tmp_dir("g1fail");
        let base_cmd = write_mock_runner(
            &dir,
            "base",
            &bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cand_cmd = write_mock_runner(
            &dir,
            "cand",
            &bench_json(120, 30, 15.0, 50.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cfg = dir.join("playlist.toml");
        let fail_cmd = if cfg!(windows) { "exit /b 1" } else { "exit 1" };
        std::fs::write(
            &cfg,
            format!(
                "repo_dir = '{}'\nbase_cmd = '{}'\nrunner_cmd = '{}'\nbuild_gate_cmds = ['{}']\n\n[[scenes]]\nname = 'g1_scene'\ninput_script = '{}'\nwarmup = 30\nframes = 120\n",
                dir.display(),
                base_cmd,
                cand_cmd,
                fail_cmd,
                dir.join("in.jsonl").display()
            ),
        )
        .unwrap();
        let exp = dir.join("experiments.jsonl");
        let mut args = base_args(cfg, dir.join("work"), exp.clone());
        args.skip_build_gate = false;
        let eval = run_with_summary(args).unwrap();
        assert!(!eval.keep);
        assert_eq!(eval.gates.len(), 6);
        assert!(eval.gates.iter().all(|g| !g.passed), "{:#?}", eval.gates);
        assert!(eval.baseline_median_ms.is_finite());
        assert!(eval.candidate_median_ms.is_finite());

        let line = std::fs::read_to_string(&exp).unwrap();
        assert!(!line.contains("NaN"), "{line}");
        let parsed = parse_json(line.trim()).expect("experiments.jsonl must be valid JSON");
        assert!(parsed.as_object().is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loop_mode_keeps_iter1_commit_across_iter2_revert_in_git_repo() {
        let dir = unique_tmp_dir("gitloop");
        let repo = dir.join("repo");
        std::fs::create_dir_all(&repo).unwrap();
        let run_git = |args: &[&str]| {
            let st = Command::new("git")
                .args(args)
                .current_dir(&repo)
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .unwrap();
            assert!(st.success(), "git {args:?} failed");
        };
        run_git(&["init"]);
        run_git(&["config", "user.name", "test"]);
        run_git(&["config", "user.email", "test@localhost"]);
        let tracked = repo.join("hot.txt");
        std::fs::write(&tracked, "v0\n").unwrap();
        run_git(&["add", "hot.txt"]);
        run_git(&["commit", "-m", "initial"]);

        let state_file = dir.join("state.json");
        let base_json = bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0);
        let win_json = bench_json(120, 30, 16.0, 48.0, 1000.0, false, 0, 0, 4000.0, 6000.0);
        let lose_json = bench_json(120, 30, 19.9, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0);
        let win_path = dir.join("win.json");
        let lose_path = dir.join("lose.json");
        std::fs::write(&win_path, &win_json).unwrap();
        std::fs::write(&lose_path, &lose_json).unwrap();

        let base_cmd = write_mock_runner(&dir, "base", &base_json, 0);
        let cand_cmd = if cfg!(windows) {
            let s = dir.join("cand.cmd");
            std::fs::write(
                &s,
                format!(
                    "@echo off\r\ncopy /Y \"{}\" \"%VICTORY_BENCH_OUT%\" >nul\r\nexit /b 0\r\n",
                    state_file.display()
                ),
            )
            .unwrap();
            format!("\"{}\"", s.display())
        } else {
            let s = dir.join("cand.sh");
            std::fs::write(
                &s,
                format!(
                    "#!/bin/sh\ncp \"{}\" \"$VICTORY_BENCH_OUT\"\nexit 0\n",
                    state_file.display()
                ),
            )
            .unwrap();
            format!("sh \"{}\"", s.display())
        };

        // Iter 1: edits hot.txt to "v1_kept" and wins.
        // Iter 2: edits hot.txt to "v2_reverted", creates junk.tmp, and loses.
        let agent_cmd = if cfg!(windows) {
            let s = dir.join("agent.cmd");
            std::fs::write(
                &s,
                format!(
                    "@echo off\r\nif \"%VICTORY_BENCH_ITER%\"==\"1\" (\r\n  echo v1_kept>hot.txt\r\n  copy /Y \"{}\" \"{}\" >nul\r\n  echo iter1 real win\r\n) else (\r\n  echo v2_reverted>hot.txt\r\n  echo junk>junk.tmp\r\n  copy /Y \"{}\" \"{}\" >nul\r\n  echo iter2 dead end\r\n)\r\nexit /b 0\r\n",
                    win_path.display(),
                    state_file.display(),
                    lose_path.display(),
                    state_file.display(),
                ),
            )
            .unwrap();
            format!("\"{}\"", s.display())
        } else {
            let s = dir.join("agent.sh");
            std::fs::write(
                &s,
                format!(
                    "#!/bin/sh\nif [ \"$VICTORY_BENCH_ITER\" = \"1\" ]; then\n  echo v1_kept > hot.txt\n  cp \"{}\" \"{}\"\n  echo iter1 real win\nelse\n  echo v2_reverted > hot.txt\n  echo junk > junk.tmp\n  cp \"{}\" \"{}\"\n  echo iter2 dead end\nfi\n",
                    win_path.display(),
                    state_file.display(),
                    lose_path.display(),
                    state_file.display(),
                ),
            )
            .unwrap();
            format!("sh \"{}\"", s.display())
        };

        let cfg = dir.join("playlist.toml");
        std::fs::write(
            &cfg,
            format!(
                "repo_dir = '{}'\nbase_cmd = '{}'\nrunner_cmd = '{}'\n\n[[scenes]]\nname = 'git_scene'\ninput_script = '{}'\nwarmup = 30\nframes = 120\n",
                repo.display(),
                base_cmd,
                cand_cmd,
                dir.join("in.jsonl").display()
            ),
        )
        .unwrap();

        let exp = dir.join("experiments.jsonl");
        let make_args = || {
            let mut a = base_args(cfg.clone(), dir.join("work"), exp.clone());
            a.mode = Mode::Loop;
            a.iterations = 2;
            a.agent_cmd = Some(agent_cmd.clone());
            a
        };

        // Dirty worktree must fail fast before iter 1.
        std::fs::write(&tracked, "dirty_wip\n").unwrap();
        assert!(run(make_args()).is_err());
        std::fs::write(&tracked, "v0\n").unwrap();

        // Clean start: iter 1 kept (committed), iter 2 reverted, `run()` succeeds!
        run(make_args()).unwrap();
        let final_content = std::fs::read_to_string(&tracked).unwrap();
        assert_eq!(final_content.trim(), "v1_kept");
        assert!(!repo.join("junk.tmp").exists(), "untracked file from reverted iter must be removed");
        assert!(
            git_stdout(&repo, &["status", "--porcelain"]).is_empty(),
            "repo should be clean after loop"
        );
        let lines: Vec<String> = std::fs::read_to_string(&exp)
            .unwrap()
            .lines()
            .map(ToOwned::to_owned)
            .collect();
        assert_eq!(lines.len(), 2);
        assert!(lines[0].contains("\"decision\":\"keep\""));
        assert!(lines[1].contains("\"decision\":\"revert\""));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn loop_mode_handles_agent_failure_without_running_scenes() {
        let dir = unique_tmp_dir("agentfail");
        let marker = dir.join("scene_ran.txt");
        let base_cmd = write_mock_runner(
            &dir,
            "base",
            &bench_json(120, 30, 20.0, 40.0, 1000.0, false, 0, 0, 4000.0, 6000.0),
            0,
        );
        let cand_cmd = if cfg!(windows) {
            format!("echo ran>\"{}\"", marker.display())
        } else {
            format!("echo ran > \"{}\"", marker.display())
        };
        let cfg = dir.join("playlist.toml");
        std::fs::write(
            &cfg,
            format!(
                "repo_dir = '{}'\nbase_cmd = '{}'\nrunner_cmd = '{}'\n\n[[scenes]]\nname = 'af_scene'\ninput_script = '{}'\nwarmup = 30\nframes = 120\n",
                dir.display(),
                base_cmd,
                cand_cmd,
                dir.join("in.jsonl").display()
            ),
        )
        .unwrap();
        let exp = dir.join("experiments.jsonl");
        let mut args = base_args(cfg, dir.join("work"), exp.clone());
        args.mode = Mode::Loop;
        args.iterations = 1;
        args.agent_cmd = Some(if cfg!(windows) {
            "exit /b 7".to_owned()
        } else {
            "exit 7".to_owned()
        });
        let eval = run_with_summary(args).unwrap();
        assert!(!eval.keep);
        assert!(!marker.exists(), "cand runner must not run when agent fails");
        assert!(eval.gates[0].detail.contains("agent command exited with code 7"));
        let raw = std::fs::read_to_string(&exp).unwrap();
        assert!(parse_json(raw.trim()).is_ok());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn reject_when_a_repeat_ends_before_requested_frame_count() {
        fn make_run(world_frames: u64, warmup_frames: u64, median_ms: f64, low1: f64) -> SceneRunResult {
            let json = bench_json(
                world_frames,
                warmup_frames,
                median_ms,
                low1,
                1000.0,
                false,
                0,
                0,
                4000.0,
                6000.0,
            );
            SceneRunResult {
                outcome: CmdOutcome::Exited(0),
                report: Some(BenchReport::from_json_str(&json).unwrap()),
                error: None,
            }
        }

        let scene = SceneConfig {
            name: "route_loop".to_owned(),
            input_script: PathBuf::from("D:/bench/script.jsonl"),
            warmup: 30,
            frames: 120,
            ..Default::default()
        };
        let gate_cfg = GateConfig {
            min_gain_pct: 2.0,
            max_draw_drift_pct: 15.0,
            max_low1_regress_pct: 2.0,
            max_vram_regress_mb: 128.0,
            max_ram_regress_mb: 256.0,
            vram_cap_mb: None,
            ram_cap_mb: None,
            max_luma_diff: 12.0,
        };

        let eval_repeats = |base_specs: [(u64, u64, f64); 3],
                            cand_specs: [(u64, u64, f64); 3],
                            scene_override: Option<SceneConfig>| {
            let input = SceneEvaluationInput {
                scene: scene_override.unwrap_or_else(|| scene.clone()),
                base_runs: base_specs
                    .into_iter()
                    .map(|(wf, wu, ms)| make_run(wf, wu, ms, 40.0))
                    .collect(),
                cand_runs: cand_specs
                    .into_iter()
                    .map(|(wf, wu, ms)| make_run(wf, wu, ms, 48.0))
                    .collect(),
            };
            evaluate_all_gates(true, "ok".to_owned(), &[input], &gate_cfg)
        };

        // Control: all repeats complete full warmup + measured frames with positive median.
        let full_base = [(120, 30, 20.0), (120, 30, 20.0), (120, 30, 20.0)];
        let full_cand = [(120, 30, 16.0), (120, 30, 16.0), (120, 30, 16.0)];
        let ok = eval_repeats(full_base, full_cand, None);
        assert!(ok.keep, "{:#?}", ok.gates);

        // Candidate short repeat [120, 1, 120] must fail Gate 5 (even though median is 120).
        let short_cand = eval_repeats(
            full_base,
            [(120, 30, 16.0), (1, 30, 16.0), (120, 30, 16.0)],
            None,
        );
        assert!(!short_cand.keep);
        assert!(short_cand.gates.iter().filter(|g| g.gate <= 4).all(|g| g.passed));
        let g5 = short_cand.gates.iter().find(|g| g.gate == 5).unwrap();
        assert!(!g5.passed);
        assert!(
            g5.detail.contains("candidate world_frames 1 < required 120"),
            "{}",
            g5.detail
        );

        // Baseline short repeat [120, 1, 120] must also fail Gate 5.
        let short_base = eval_repeats(
            [(120, 30, 20.0), (1, 30, 20.0), (120, 30, 20.0)],
            full_cand,
            None,
        );
        assert!(!short_base.keep);
        assert!(short_base.gates.iter().filter(|g| g.gate <= 4).all(|g| g.passed));
        let g5 = short_base.gates.iter().find(|g| g.gate == 5).unwrap();
        assert!(!g5.passed);
        assert!(
            g5.detail.contains("baseline world_frames 1 < required 120"),
            "{}",
            g5.detail
        );

        // Short warmup [30, 5, 30] on either side must fail Gate 5.
        let short_base_warmup = eval_repeats(
            [(120, 30, 20.0), (120, 5, 20.0), (120, 30, 20.0)],
            full_cand,
            None,
        );
        let g5 = short_base_warmup.gates.iter().find(|g| g.gate == 5).unwrap();
        assert!(!g5.passed);
        assert!(
            g5.detail.contains("baseline warmup_frames 5 < required 30"),
            "{}",
            g5.detail
        );
        let short_cand_warmup = eval_repeats(
            full_base,
            [(120, 30, 16.0), (120, 5, 16.0), (120, 30, 16.0)],
            None,
        );
        let g5 = short_cand_warmup.gates.iter().find(|g| g.gate == 5).unwrap();
        assert!(!g5.passed);
        assert!(
            g5.detail.contains("candidate warmup_frames 5 < required 30"),
            "{}",
            g5.detail
        );

        // Non-positive median_frame_ms on either baseline or candidate must fail Gate 5.
        let zero_base_ms = eval_repeats(
            [(120, 30, 0.0), (120, 30, 0.0), (120, 30, 0.0)],
            full_cand,
            None,
        );
        let g5 = zero_base_ms.gates.iter().find(|g| g.gate == 5).unwrap();
        assert!(!g5.passed);
        assert!(
            g5.detail.contains("baseline frame_ms.median 0 must be > 0"),
            "{}",
            g5.detail
        );
        let zero_cand_repeat_ms = eval_repeats(
            full_base,
            [(120, 30, 16.0), (120, 30, 0.0), (120, 30, 16.0)],
            None,
        );
        let g5 = zero_cand_repeat_ms.gates.iter().find(|g| g.gate == 5).unwrap();
        assert!(!g5.passed);
        assert!(
            g5.detail.contains("candidate repeat frame_ms.median 0 must be > 0"),
            "{}",
            g5.detail
        );

        // Even if scene.frames == 0, Gate 5 enforces at least 1 measured world frame.
        let mut zero_req_scene = scene.clone();
        zero_req_scene.frames = 0;
        let zero_frames_base = eval_repeats(
            [(0, 30, 20.0), (0, 30, 20.0), (0, 30, 20.0)],
            full_cand,
            Some(zero_req_scene),
        );
        let g5 = zero_frames_base.gates.iter().find(|g| g.gate == 5).unwrap();
        assert!(!g5.passed);
        assert!(
            g5.detail.contains("baseline world_frames 0 < required 1"),
            "{}",
            g5.detail
        );
    }

    #[test]
    fn reject_when_configured_input_script_reports_null_digest() {
        fn make_run_with_sha(ms: f64, low1: f64, sha: Option<&str>) -> SceneRunResult {
            let mut json = bench_json(120, 30, ms, low1, 1000.0, false, 0, 0, 4000.0, 6000.0);
            let replacement = match sha {
                Some(s) => format!("\"{s}\""),
                None => "null".to_owned(),
            };
            json = json.replace("\"0123456789abcdef\"", &replacement);
            SceneRunResult {
                outcome: CmdOutcome::Exited(0),
                report: Some(BenchReport::from_json_str(&json).unwrap()),
                error: None,
            }
        }

        let scene = SceneConfig {
            name: "route_loop".to_owned(),
            input_script: PathBuf::from("D:/bench/route.jsonl"),
            warmup: 30,
            frames: 120,
            ..Default::default()
        };
        let gate_cfg = GateConfig {
            min_gain_pct: 2.0,
            max_draw_drift_pct: 15.0,
            max_low1_regress_pct: 2.0,
            max_vram_regress_mb: 128.0,
            max_ram_regress_mb: 256.0,
            vram_cap_mb: None,
            ram_cap_mb: None,
            max_luma_diff: 12.0,
        };

        for (label, base_sha, cand_sha, expected_substr) in [
            (
                "baseline_null_only",
                None,
                Some("0123456789abcdef"),
                "baseline reported null input_script_sha256",
            ),
            (
                "candidate_null_only",
                Some("0123456789abcdef"),
                None,
                "candidate reported null input_script_sha256",
            ),
            (
                "both_null",
                None,
                None,
                "baseline reported null input_script_sha256",
            ),
        ] {
            let input = SceneEvaluationInput {
                scene: scene.clone(),
                base_runs: vec![make_run_with_sha(20.0, 40.0, base_sha)],
                cand_runs: vec![make_run_with_sha(16.0, 48.0, cand_sha)],
            };
            let eval = evaluate_all_gates(true, "ok".to_owned(), &[input], &gate_cfg);
            assert!(!eval.keep, "{label}: should reject");
            let g5 = eval.gates.iter().find(|g| g.gate == 5).unwrap();
            assert!(!g5.passed, "{label}: Gate 5 should fail");
            assert!(
                g5.detail.contains(expected_substr),
                "{label}: unexpected detail {}",
                g5.detail
            );
        }
    }

    #[test]
    fn watchdog_kills_hung_runner() {
        let dir = unique_tmp_dir("timeout");
        let cmd = if cfg!(windows) {
            "ping -n 10 127.0.0.1 >nul"
        } else {
            "sleep 10"
        };
        let out = run_shell_with_timeout(cmd, Some(&dir), &[], Duration::from_millis(250)).unwrap();
        assert_eq!(out, CmdOutcome::TimedOut);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
