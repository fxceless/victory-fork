//! Deterministic world-frame input recording and replay for `gameinput.dll`.
//!
//! Opt-in via environment variables (disabled with zero overhead when unset):
//! - `VICTORY_INPUT_RECORD=<abs_path.jsonl>`: wraps host input and records
//!   every state change of the active gamepad (`PadState`) and held keyboard
//!   virtual keys as JSONL tagged with `wframe` (`victory_gpu::bench::world_frame()`)
//!   and elapsed `ms`. In-repo and relative output paths are rejected.
//! - `VICTORY_INPUT_REPLAY=<abs_path.jsonl>`: drives player-0 gamepad and
//!   keyboard state deterministically by `world_frame`, immune to FPS jitter
//!   or speedups. Waits in neutral state through loading screens while
//!   `world_frame == 0`, supports both recorded JSONL traces and declarative
//!   range lines (`{"from":0,"to":120,"ly":1.0,"rt":0.8}`), and registers the
//!   script digest with `victory_gpu::bench::register_input_script_bytes`.
//!
//! ## Delivery contract
//! - While `world_frame == 0` (loading screens / menus), replay holds neutral
//!   input and neither advances playback state nor consumes pulses.
//! - Recorded state-change traces (`wframe`) behave as step-function holds:
//!   the latest entry with `entry.wframe <= world_frame` remains active until
//!   the next recorded change.
//! - Stateless point lookups (`InputScript::sample`) evaluate declarative
//!   ranges strictly at `from <= world_frame <= to`.
//! - Stateful live playback (`ReplayPlayback` / `ReplaySource`) guarantees
//!   once-only delivery for authored 1-frame pulses (`from == to`, `from >= 1`):
//!   if no poll occurs on world frame `from`, any unpolled pulse in the gap
//!   `last_polled_wframe < from <= world_frame` latches onto the first eligible
//!   polled world frame, stays consistent across repeated polls and pad/key
//!   reads during that single world frame, and transitions to consumed on the
//!   next world frame so it never repeats.
//! - Multi-frame declarative spans (`to > from`) never latch past `to`.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::gameinput::{KeyState, MouseState, PADS, PadState, Source, fallback, merged};

/// Wraps `source` according to `VICTORY_INPUT_REPLAY` / `VICTORY_INPUT_RECORD`.
/// Returns `source` untouched when neither environment variable is set.
pub(super) fn wrap_from_env(mut source: Box<dyn Source>, one_pad: bool) -> Box<dyn Source> {
    if let Ok(raw) = std::env::var("VICTORY_INPUT_REPLAY") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed);
            match ReplaySource::from_file(&path) {
                Ok(replay) => {
                    tracing::info!(path = %path.display(), "VICTORY_INPUT_REPLAY active");
                    source = Box::new(replay);
                }
                Err(err) => {
                    tracing::error!(path = %path.display(), "VICTORY_INPUT_REPLAY failed: {err}");
                }
            }
        }
    }

    if let Ok(raw) = std::env::var("VICTORY_INPUT_RECORD") {
        let trimmed = raw.trim();
        if !trimmed.is_empty() {
            let path = PathBuf::from(trimmed);
            match RecordingSource::new(source, &path, Box::new(victory_gpu::bench::world_frame)) {
                Ok(mut rec) => {
                    rec.one_pad = one_pad;
                    tracing::info!(path = %path.display(), "VICTORY_INPUT_RECORD active");
                    return Box::new(rec);
                }
                Err((orig, err)) => {
                    tracing::error!(path = %path.display(), "VICTORY_INPUT_RECORD ignored: {err}");
                    return orig;
                }
            }
        }
    }

    source
}

/// Bit-exact comparison key for `PadState` (handles `-0.0` vs `+0.0` and exact f32 bit patterns).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PadBits {
    buttons: u32,
    lt: u32,
    rt: u32,
    lx: u32,
    ly: u32,
    rx: u32,
    ry: u32,
}

impl From<PadState> for PadBits {
    fn from(p: PadState) -> Self {
        Self {
            buttons: p.buttons,
            lt: p.left_trigger.to_bits(),
            rt: p.right_trigger.to_bits(),
            lx: p.left_x.to_bits(),
            ly: p.left_y.to_bits(),
            rx: p.right_x.to_bits(),
            ry: p.right_y.to_bits(),
        }
    }
}

/// One recorded state-change line in a JSONL input trace.
#[derive(Clone, Debug, PartialEq)]
pub struct RecordedEvent {
    pub wframe: u64,
    pub ms: u64,
    pub pad: PadState,
    pub keys: Vec<u8>,
}

/// One declarative world-frame range rule (`from <= wframe <= to` while `wframe > 0`).
#[derive(Clone, Debug, PartialEq)]
pub struct RangeRule {
    pub from: u64,
    pub to: u64,
    pub buttons: Option<u32>,
    pub lt: Option<f32>,
    pub rt: Option<f32>,
    pub lx: Option<f32>,
    pub ly: Option<f32>,
    pub rx: Option<f32>,
    pub ry: Option<f32>,
    pub keys: Vec<u8>,
}

/// Parsed replay script supporting recorded events, declarative ranges, or both.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct InputScript {
    pub events: Vec<RecordedEvent>,
    pub ranges: Vec<RangeRule>,
}

impl InputScript {
    /// Parses a JSONL input script containing recorded events (`wframe`),
    /// declarative range lines (`from`/`to`), or a mix of both.
    pub fn parse(text: &str) -> Result<Self, String> {
        let mut script = Self::default();
        for (idx, raw_line) in text.lines().enumerate() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') || line.starts_with("//") {
                continue;
            }
            let obj = parse_flat_json_object(line)
                .map_err(|e| format!("line {}: {e}", idx + 1))?;
            if obj.contains_key("from") || obj.contains_key("to") {
                let from = get_u64(&obj, "from").ok_or_else(|| {
                    format!("line {}: range entry missing numeric 'from'", idx + 1)
                })?;
                let to = get_u64(&obj, "to").ok_or_else(|| {
                    format!("line {}: range entry missing numeric 'to'", idx + 1)
                })?;
                let buttons = get_u32(&obj, "buttons");
                let lt = get_f32(&obj, "lt").or_else(|| get_f32(&obj, "left_trigger"));
                let rt = get_f32(&obj, "rt").or_else(|| get_f32(&obj, "right_trigger"));
                let lx = get_f32(&obj, "lx").or_else(|| get_f32(&obj, "left_x"));
                let ly = get_f32(&obj, "ly").or_else(|| get_f32(&obj, "left_y"));
                let rx = get_f32(&obj, "rx").or_else(|| get_f32(&obj, "right_x"));
                let ry = get_f32(&obj, "ry").or_else(|| get_f32(&obj, "right_y"));
                let keys = get_u8_vec(&obj, "keys")
                    .map_err(|e| format!("line {}: {e}", idx + 1))?;
                script.ranges.push(RangeRule {
                    from,
                    to,
                    buttons,
                    lt,
                    rt,
                    lx,
                    ly,
                    rx,
                    ry,
                    keys,
                });
            } else {
                let wframe = get_u64(&obj, "wframe").ok_or_else(|| {
                    format!(
                        "line {}: entry must have either 'wframe' or 'from'/'to'",
                        idx + 1
                    )
                })?;
                let ms = get_u64(&obj, "ms").unwrap_or(0);
                let pad = PadState {
                    buttons: get_u32(&obj, "buttons").unwrap_or(0),
                    left_trigger: get_f32(&obj, "lt")
                        .or_else(|| get_f32(&obj, "left_trigger"))
                        .unwrap_or(0.0),
                    right_trigger: get_f32(&obj, "rt")
                        .or_else(|| get_f32(&obj, "right_trigger"))
                        .unwrap_or(0.0),
                    left_x: get_f32(&obj, "lx")
                        .or_else(|| get_f32(&obj, "left_x"))
                        .unwrap_or(0.0),
                    left_y: get_f32(&obj, "ly")
                        .or_else(|| get_f32(&obj, "left_y"))
                        .unwrap_or(0.0),
                    right_x: get_f32(&obj, "rx")
                        .or_else(|| get_f32(&obj, "right_x"))
                        .unwrap_or(0.0),
                    right_y: get_f32(&obj, "ry")
                        .or_else(|| get_f32(&obj, "right_y"))
                        .unwrap_or(0.0),
                };
                let keys = get_u8_vec(&obj, "keys")
                    .map_err(|e| format!("line {}: {e}", idx + 1))?;
                script.events.push(RecordedEvent {
                    wframe,
                    ms,
                    pad,
                    keys,
                });
            }
        }
        // Keep recorded events sorted stably by world frame so binary search /
        // scan always resolves the latest event at or before `wframe`.
        script.events.sort_by_key(|ev| ev.wframe);
        Ok(script)
    }

    fn sample_matching(
        &self,
        wframe: u64,
        mut range_active: impl FnMut(usize, &RangeRule) -> bool,
    ) -> (PadState, Vec<u8>) {
        if wframe == 0 {
            return (PadState::default(), Vec::new());
        }

        let mut pad = PadState::default();
        let mut keys = Vec::new();

        // Latest recorded event with `ev.wframe <= wframe`.
        let idx = self.events.partition_point(|ev| ev.wframe <= wframe);
        if idx > 0 {
            let ev = &self.events[idx - 1];
            pad = ev.pad;
            keys.clone_from(&ev.keys);
        }

        // Overlay matching declarative ranges in file order.
        for (rule_idx, rule) in self.ranges.iter().enumerate() {
            if range_active(rule_idx, rule) {
                if let Some(b) = rule.buttons {
                    pad.buttons |= b;
                }
                if let Some(v) = rule.lt {
                    pad.left_trigger = v;
                }
                if let Some(v) = rule.rt {
                    pad.right_trigger = v;
                }
                if let Some(v) = rule.lx {
                    pad.left_x = v;
                }
                if let Some(v) = rule.ly {
                    pad.left_y = v;
                }
                if let Some(v) = rule.rx {
                    pad.right_x = v;
                }
                if let Some(v) = rule.ry {
                    pad.right_y = v;
                }
                for &vk in &rule.keys {
                    if !keys.contains(&vk) {
                        keys.push(vk);
                    }
                }
            }
        }

        keys.sort_unstable();
        keys.dedup();
        (pad, keys)
    }

    /// Evaluates deterministic stateless `(PadState, held_virtual_keys)` at exact `wframe`.
    ///
    /// While `wframe == 0` (loading screens / menus before world latch),
    /// always returns neutral pad and empty keys so replay waits cleanly until
    /// world rendering begins. For stateful playback with zero-poll 1-frame
    /// pulse latching (`from == to`), use [`ReplayPlayback`] or `ReplaySource`.
    pub fn sample(&self, wframe: u64) -> (PadState, Vec<u8>) {
        self.sample_matching(wframe, |_, rule| rule.from <= wframe && wframe <= rule.to)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PulseState {
    /// Authored 1-frame pulse (`from == to`) has not yet been delivered.
    Pending,
    /// Latched and active for all polls on this single world frame.
    ActiveOn(u64),
    /// Already delivered on an earlier world frame; must never repeat.
    Consumed,
}

/// Stateful world-frame replay cursor over an [`InputScript`] that latches
/// unpolled 1-frame declarative pulses (`from == to`, `from >= 1`) across
/// zero-poll gaps so each pulse delivers once on the first eligible world frame
/// and never repeats. Multi-frame spans (`to > from`) never latch past `to`.
#[derive(Clone, Debug)]
pub struct ReplayPlayback {
    script: InputScript,
    last_polled_wframe: u64,
    pulse_states: Vec<PulseState>,
}

impl ReplayPlayback {
    pub fn new(script: InputScript) -> Self {
        let pulse_states = vec![PulseState::Pending; script.ranges.len()];
        Self {
            script,
            last_polled_wframe: 0,
            pulse_states,
        }
    }

    pub fn into_script(self) -> InputScript {
        self.script
    }

    fn advance_to(&mut self, wframe: u64) {
        if wframe == 0 || wframe <= self.last_polled_wframe {
            return;
        }
        let prev = self.last_polled_wframe;
        for (state, rule) in self.pulse_states.iter_mut().zip(&self.script.ranges) {
            if rule.from != rule.to {
                continue;
            }
            match *state {
                PulseState::ActiveOn(delivered_wf) if delivered_wf < wframe => {
                    *state = PulseState::Consumed;
                }
                PulseState::Pending if prev < rule.from && rule.from <= wframe => {
                    *state = PulseState::ActiveOn(wframe);
                }
                _ => {}
            }
        }
        self.last_polled_wframe = wframe;
    }

    /// Advances playback state to `wframe` and evaluates `(PadState, held_virtual_keys)`.
    pub fn sample(&mut self, wframe: u64) -> (PadState, Vec<u8>) {
        self.advance_to(wframe);
        let states = &self.pulse_states;
        self.script.sample_matching(wframe, |idx, rule| {
            if rule.from == rule.to {
                states[idx] == PulseState::ActiveOn(wframe)
            } else {
                rule.from <= wframe && wframe <= rule.to
            }
        })
    }
}

/// Formats one recorded state-change line as JSONL.
pub fn format_event_jsonl(wframe: u64, ms: u64, pad: &PadState, keys: &[u8]) -> String {
    let mut keys_str = String::new();
    for (i, &k) in keys.iter().enumerate() {
        if i > 0 {
            keys_str.push(',');
        }
        keys_str.push_str(&k.to_string());
    }
    format!(
        "{{\"wframe\":{wframe},\"ms\":{ms},\"buttons\":{},\"lt\":{},\"rt\":{},\"lx\":{},\"ly\":{},\"rx\":{},\"ry\":{},\"keys\":[{keys_str}]}}",
        pad.buttons,
        pad.left_trigger,
        pad.right_trigger,
        pad.left_x,
        pad.left_y,
        pad.right_x,
        pad.right_y,
    )
}

/// Input `Source` wrapper that records every state change to a JSONL file
/// outside the repository.
pub(super) struct RecordingSource {
    inner: Box<dyn Source>,
    writer: BufWriter<File>,
    start: Instant,
    world_frame_fn: Box<dyn Fn() -> u64 + Send>,
    one_pad: bool,
    pads: [Option<PadState>; PADS],
    last_recorded: Option<(PadBits, Vec<u8>)>,
}

impl RecordingSource {
    pub(super) fn new(
        inner: Box<dyn Source>,
        path: &Path,
        world_frame_fn: Box<dyn Fn() -> u64 + Send>,
    ) -> Result<Self, (Box<dyn Source>, String)> {
        let safe = match victory_gpu::bench::ensure_outside_repo(path) {
            Ok(p) => p,
            Err(e) => return Err((inner, e)),
        };
        if let Some(parent) = safe.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            return Err((
                inner,
                format!("create parent dir {}: {e}", parent.display()),
            ));
        }
        let file = match OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&safe)
        {
            Ok(f) => f,
            Err(e) => return Err((inner, format!("open {}: {e}", safe.display()))),
        };
        Ok(Self {
            inner,
            writer: BufWriter::new(file),
            start: Instant::now(),
            world_frame_fn,
            one_pad: false,
            pads: [None; PADS],
            last_recorded: None,
        })
    }

    fn maybe_record(&mut self, effective_pad: PadState, keys: &[KeyState]) {
        let mut vks: Vec<u8> = keys.iter().map(|k| k.virtual_key).collect();
        vks.sort_unstable();
        vks.dedup();
        let bits = PadBits::from(effective_pad);
        let unchanged = self
            .last_recorded
            .as_ref()
            .is_some_and(|(prev_bits, prev_keys)| *prev_bits == bits && *prev_keys == vks);
        if unchanged {
            return;
        }
        let wframe = (self.world_frame_fn)();
        let ms = self.start.elapsed().as_millis() as u64;
        let line = format_event_jsonl(wframe, ms, &effective_pad, &vks);
        if let Err(err) = writeln!(self.writer, "{line}").and_then(|()| self.writer.flush()) {
            tracing::warn!("input recording write failed: {err}");
        }
        self.last_recorded = Some((bits, vks));
    }
}

impl Source for RecordingSource {
    fn mouse(&mut self) -> Option<MouseState> {
        self.inner.mouse()
    }

    fn pad(&mut self, slot: usize) -> Option<PadState> {
        let pad = self.inner.pad(slot);
        self.pads[slot] = pad;
        pad
    }

    fn pad_connected(&mut self, slot: usize) -> bool {
        self.inner.pad_connected(slot)
    }

    fn keys(&mut self) -> Vec<KeyState> {
        let keys = self.inner.keys();
        let effective = if self.one_pad {
            self.pads.iter().flatten().copied().fold(fallback(&keys), merged)
        } else if self.pads.iter().any(Option::is_some) {
            self.pads[0].unwrap_or_default()
        } else {
            fallback(&keys)
        };
        self.maybe_record(effective, &keys);
        keys
    }

    fn rumble(&mut self, slot: usize, low: u16, high: u16) {
        self.inner.rumble(slot, low, high);
    }
}

impl Drop for RecordingSource {
    fn drop(&mut self) {
        let _ = self.writer.flush();
    }
}

/// Deterministic input `Source` driven by `world_frame`.
pub(super) struct ReplaySource {
    playback: ReplayPlayback,
    world_frame_fn: Box<dyn Fn() -> u64 + Send>,
    /// World frame latched during `pad(0)` so `keys()` in the same `Hub::poll`
    /// cycle samples the exact same frame as the gamepad.
    latched_wframe: Option<u64>,
}

impl ReplaySource {
    pub(super) fn from_file(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path)
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        let text = std::str::from_utf8(&bytes)
            .map_err(|e| format!("UTF-8 decode {}: {e}", path.display()))?;
        let script = InputScript::parse(text)?;
        victory_gpu::bench::register_input_script_bytes(&bytes);
        Ok(Self::with_clock(
            script,
            Box::new(victory_gpu::bench::world_frame),
        ))
    }

    pub(super) fn with_clock(
        script: InputScript,
        world_frame_fn: Box<dyn Fn() -> u64 + Send>,
    ) -> Self {
        Self {
            playback: ReplayPlayback::new(script),
            world_frame_fn,
            latched_wframe: None,
        }
    }
}

impl Source for ReplaySource {
    fn pad(&mut self, slot: usize) -> Option<PadState> {
        if slot != 0 {
            return None;
        }
        let wf = (self.world_frame_fn)();
        self.latched_wframe = Some(wf);
        Some(self.playback.sample(wf).0)
    }

    fn pad_connected(&mut self, slot: usize) -> bool {
        slot == 0
    }

    fn keys(&mut self) -> Vec<KeyState> {
        let wf = self
            .latched_wframe
            .take()
            .unwrap_or_else(|| (self.world_frame_fn)());
        let (_, vks) = self.playback.sample(wf);
        vks.into_iter()
            .map(|vk| KeyState {
                scan_code: vk_to_scan_code(vk),
                virtual_key: vk,
                ..Default::default()
            })
            .collect()
    }

    fn rumble(&mut self, _slot: usize, _low: u16, _high: u16) {}
}

#[cfg(windows)]
fn vk_to_scan_code(vk: u8) -> u32 {
    use windows_sys::Win32::UI::Input::KeyboardAndMouse::{
        MAPVK_VK_TO_VSC_EX, MapVirtualKeyW,
    };
    // SAFETY: plain query.
    unsafe { MapVirtualKeyW(u32::from(vk), MAPVK_VK_TO_VSC_EX) }
}

#[cfg(not(windows))]
fn vk_to_scan_code(_vk: u8) -> u32 {
    0
}

// ---------------------------------------------------------------------------
// Minimal pure-std JSONL object parser (no extra dependencies)
// ---------------------------------------------------------------------------

use std::collections::BTreeMap;

#[derive(Debug)]
enum FlatVal {
    Num(String),
    Arr(Vec<String>),
    Other,
}

fn parse_flat_json_object(line: &str) -> Result<BTreeMap<String, FlatVal>, String> {
    let bytes = line.as_bytes();
    let mut i = 0usize;
    let skip_ws = |i: &mut usize| {
        while *i < bytes.len() && bytes[*i].is_ascii_whitespace() {
            *i += 1;
        }
    };
    skip_ws(&mut i);
    if i >= bytes.len() || bytes[i] != b'{' {
        return Err("expected '{'".to_owned());
    }
    i += 1;
    let mut map = BTreeMap::new();
    loop {
        skip_ws(&mut i);
        if i >= bytes.len() {
            return Err("unterminated object".to_owned());
        }
        if bytes[i] == b'}' {
            break;
        }
        let key = parse_json_string(bytes, &mut i)?;
        skip_ws(&mut i);
        if i >= bytes.len() || bytes[i] != b':' {
            return Err(format!("expected ':' after key '{key}'"));
        }
        i += 1;
        skip_ws(&mut i);
        if i >= bytes.len() {
            return Err(format!("missing value for key '{key}'"));
        }
        let val = match bytes[i] {
            b'[' => {
                i += 1;
                let mut items = Vec::new();
                loop {
                    skip_ws(&mut i);
                    if i >= bytes.len() {
                        return Err("unterminated array".to_owned());
                    }
                    if bytes[i] == b']' {
                        i += 1;
                        break;
                    }
                    items.push(parse_number_token(bytes, &mut i)?);
                    skip_ws(&mut i);
                    if i < bytes.len() && bytes[i] == b',' {
                        i += 1;
                    } else if i < bytes.len() && bytes[i] == b']' {
                        i += 1;
                        break;
                    } else {
                        return Err("expected ',' or ']' in array".to_owned());
                    }
                }
                FlatVal::Arr(items)
            }
            b'"' => {
                let _ = parse_json_string(bytes, &mut i)?;
                FlatVal::Other
            }
            b'-' | b'0'..=b'9' => FlatVal::Num(parse_number_token(bytes, &mut i)?),
            _ => {
                // Consume simple literal (`true`/`false`/`null`).
                while i < bytes.len() && !matches!(bytes[i], b',' | b'}' | b' ' | b'\t') {
                    i += 1;
                }
                FlatVal::Other
            }
        };
        map.insert(key, val);
        skip_ws(&mut i);
        if i < bytes.len() && bytes[i] == b',' {
            i += 1;
        } else if i < bytes.len() && bytes[i] == b'}' {
            break;
        } else {
            return Err("expected ',' or '}'".to_owned());
        }
    }
    Ok(map)
}

fn parse_json_string(bytes: &[u8], i: &mut usize) -> Result<String, String> {
    if *i >= bytes.len() || bytes[*i] != b'"' {
        return Err("expected string".to_owned());
    }
    *i += 1;
    let mut out = String::new();
    while *i < bytes.len() {
        let b = bytes[*i];
        *i += 1;
        if b == b'"' {
            return Ok(out);
        }
        if b == b'\\' {
            if *i >= bytes.len() {
                return Err("unterminated escape".to_owned());
            }
            let esc = bytes[*i];
            *i += 1;
            match esc {
                b'"' => out.push('"'),
                b'\\' => out.push('\\'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                other => out.push(other as char),
            }
        } else if b < 0x80 {
            out.push(b as char);
        } else {
            let start = *i - 1;
            let s = std::str::from_utf8(&bytes[start..])
                .map_err(|_| "invalid UTF-8 in string".to_owned())?;
            let ch = s.chars().next().ok_or_else(|| "empty UTF-8".to_owned())?;
            out.push(ch);
            *i = start + ch.len_utf8();
        }
    }
    Err("unterminated string".to_owned())
}

fn parse_number_token(bytes: &[u8], i: &mut usize) -> Result<String, String> {
    let start = *i;
    while *i < bytes.len()
        && matches!(
            bytes[*i],
            b'-' | b'+' | b'0'..=b'9' | b'.' | b'e' | b'E'
        )
    {
        *i += 1;
    }
    if *i == start {
        return Err("expected number".to_owned());
    }
    Ok(String::from_utf8_lossy(&bytes[start..*i]).into_owned())
}

fn get_u64(map: &BTreeMap<String, FlatVal>, k: &str) -> Option<u64> {
    match map.get(k) {
        Some(FlatVal::Num(s)) => s.parse::<u64>().ok(),
        _ => None,
    }
}

fn get_u32(map: &BTreeMap<String, FlatVal>, k: &str) -> Option<u32> {
    match map.get(k) {
        Some(FlatVal::Num(s)) => s.parse::<u32>().ok(),
        _ => None,
    }
}

fn get_f32(map: &BTreeMap<String, FlatVal>, k: &str) -> Option<f32> {
    match map.get(k) {
        Some(FlatVal::Num(s)) => s.parse::<f32>().ok().filter(|v| v.is_finite()),
        _ => None,
    }
}

fn get_u8_vec(map: &BTreeMap<String, FlatVal>, k: &str) -> Result<Vec<u8>, String> {
    match map.get(k) {
        None => Ok(Vec::new()),
        Some(FlatVal::Arr(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                let v = item
                    .parse::<u8>()
                    .map_err(|_| format!("invalid virtual key '{item}' in '{k}'"))?;
                out.push(v);
            }
            Ok(out)
        }
        Some(_) => Err(format!("'{k}' must be an array of u8 virtual key codes")),
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering};

    use super::*;

    // The file-loading tests both exercise the process-wide report digest.
    static DIGEST_TEST: std::sync::Mutex<()> = std::sync::Mutex::new(());

    struct MockSource {
        pad0: Option<PadState>,
        keys: Vec<KeyState>,
    }

    impl Source for MockSource {
        fn pad(&mut self, slot: usize) -> Option<PadState> {
            if slot == 0 { self.pad0 } else { None }
        }
        fn keys(&mut self) -> Vec<KeyState> {
            self.keys.clone()
        }
        fn rumble(&mut self, _: usize, _: u16, _: u16) {}
    }

    #[test]
    fn record_save_replay_bit_exact_across_variable_frame_speeds() {
        let _digest = DIGEST_TEST.lock().unwrap();
        let tmp = tempfile::tempdir().unwrap();
        let script_path = tmp.path().join("traversal.jsonl");

        let clock = Arc::new(AtomicU64::new(0));
        let rec_clock = Arc::clone(&clock);

        let p1 = PadState {
            buttons: 0x0014,
            left_trigger: 0.125,
            right_trigger: 0.875,
            left_x: -0.33333334,
            left_y: 0.9999695,
            right_x: 0.25,
            right_y: -0.625,
        };
        let p2 = PadState {
            buttons: 0x0404,
            left_trigger: 0.0,
            right_trigger: 1.0,
            left_x: 0.5,
            left_y: 0.75,
            right_x: -0.125,
            right_y: 0.0,
        };

        {
            let mock = MockSource {
                pad0: Some(PadState::default()),
                keys: Vec::new(),
            };
            let mut rec = RecordingSource::new(
                Box::new(mock),
                &script_path,
                Box::new(move || rec_clock.load(Ordering::Relaxed)),
            )
            .ok()
            .unwrap();

            // World frame 1: first input change.
            clock.store(1, Ordering::Relaxed);
            // Drive inner state through a fresh helper by setting state via direct calls.
            // To keep RecordingSource black-box, re-create with a shared state mock below.
            let _ = rec.pad(0);
            let _ = rec.keys();
        }

        // Shared-state mock for multi-frame recording.
        struct SharedMock(Arc<std::sync::Mutex<(Option<PadState>, Vec<KeyState>)>>);
        impl Source for SharedMock {
            fn pad(&mut self, slot: usize) -> Option<PadState> {
                if slot == 0 {
                    self.0.lock().unwrap().0
                } else {
                    None
                }
            }
            fn keys(&mut self) -> Vec<KeyState> {
                self.0.lock().unwrap().1.clone()
            }
            fn rumble(&mut self, _: usize, _: u16, _: u16) {}
        }

        let shared = Arc::new(std::sync::Mutex::new((Some(PadState::default()), Vec::new())));
        let rec_clock2 = Arc::clone(&clock);
        {
            let mut rec = RecordingSource::new(
                Box::new(SharedMock(Arc::clone(&shared))),
                &script_path,
                Box::new(move || rec_clock2.load(Ordering::Relaxed)),
            )
            .ok()
            .unwrap();

            // Loading screen (wframe 0): neutral.
            clock.store(0, Ordering::Relaxed);
            let _ = rec.pad(0);
            let _ = rec.keys();

            // World frame 5: p1 + key 'W' (87).
            clock.store(5, Ordering::Relaxed);
            *shared.lock().unwrap() = (
                Some(p1),
                vec![KeyState {
                    virtual_key: b'W',
                    ..Default::default()
                }],
            );
            let _ = rec.pad(0);
            let _ = rec.keys();
            // Duplicate poll at same state does not emit duplicate JSONL line.
            let _ = rec.pad(0);
            let _ = rec.keys();

            // World frame 25: p2 + no keys.
            clock.store(25, Ordering::Relaxed);
            *shared.lock().unwrap() = (Some(p2), Vec::new());
            let _ = rec.pad(0);
            let _ = rec.keys();
        }

        let raw = std::fs::read_to_string(&script_path).unwrap();
        assert_eq!(raw.lines().count(), 3, "expected 3 state transitions:\n{raw}");

        // Reload via ReplaySource::from_file (also registers sha256 in victory-gpu::bench).
        let replay_from_file = ReplaySource::from_file(&script_path).unwrap();
        let expected_sha = blake3_hex_via_bench(raw.as_bytes());
        assert_eq!(
            victory_gpu::bench::input_script_sha256().as_deref(),
            Some(expected_sha.as_str())
        );

        // Replay at two completely different polling speeds (slow 10 FPS vs fast 120 FPS).
        let replay_clock = Arc::new(AtomicU64::new(0));
        let mut replay = ReplaySource::with_clock(
            replay_from_file.playback.into_script(),
            Box::new({
                let c = Arc::clone(&replay_clock);
                move || c.load(Ordering::Relaxed)
            }),
        );

        // Loading screen hold (wframe == 0).
        replay_clock.store(0, Ordering::Relaxed);
        assert_eq!(replay.pad(0), Some(PadState::default()));
        assert!(replay.keys().is_empty());

        // Slow speed: multiple polls while stuck on wframe 5 stay bit-exact on p1.
        replay_clock.store(5, Ordering::Relaxed);
        for _ in 0..5 {
            let got = replay.pad(0).unwrap();
            assert_eq!(PadBits::from(got), PadBits::from(p1));
            let k: Vec<u8> = replay.keys().into_iter().map(|x| x.virtual_key).collect();
            assert_eq!(k, vec![b'W']);
        }

        // Still p1 at wframe 24 (just before next event).
        replay_clock.store(24, Ordering::Relaxed);
        assert_eq!(PadBits::from(replay.pad(0).unwrap()), PadBits::from(p1));

        // Fast jump directly to wframe 40 (past 25) yields bit-exact p2 and empty keys.
        replay_clock.store(40, Ordering::Relaxed);
        assert_eq!(PadBits::from(replay.pad(0).unwrap()), PadBits::from(p2));
        assert!(replay.keys().is_empty());
        // Slots 1..3 remain disconnected during replay.
        assert_eq!(replay.pad(1), None);
    }

    fn blake3_hex_via_bench(bytes: &[u8]) -> String {
        victory_gpu::bench::register_input_script_bytes(bytes)
    }

    #[test]
    fn declarative_range_script_and_loading_screen_hold() {
        let text = r#"
# Drive forward for first 120 world frames, then steer right and hold A
{"from":0,"to":120,"ly":1.0,"rt":0.8}
{"from":60,"to":150,"lx":0.5,"buttons":4,"keys":[69]}
"#;
        let script = InputScript::parse(text).unwrap();

        // Loading screen (wframe == 0) always holds neutral even with from:0.
        assert_eq!(script.sample(0), (PadState::default(), Vec::new()));

        // World frame 1: first range active.
        let (p, k) = script.sample(1);
        assert_eq!(p.left_y, 1.0);
        assert_eq!(p.right_trigger, 0.8);
        assert_eq!(p.buttons, 0);
        assert!(k.is_empty());

        // World frame 90: both ranges overlap and combine cleanly.
        let (p, k) = script.sample(90);
        assert_eq!(p.left_y, 1.0);
        assert_eq!(p.right_trigger, 0.8);
        assert_eq!(p.left_x, 0.5);
        assert_eq!(p.buttons, 4);
        assert_eq!(k, vec![69]);

        // World frame 130: only second range active.
        let (p, k) = script.sample(130);
        assert_eq!(p.left_y, 0.0);
        assert_eq!(p.left_x, 0.5);
        assert_eq!(p.buttons, 4);
        assert_eq!(k, vec![69]);

        // After all ranges: neutral.
        assert_eq!(script.sample(200), (PadState::default(), Vec::new()));

        // ReplaySource latches wframe on pad(0) so keys() in the same poll
        // cycle stays synchronized even if the world_frame counter advances mid-poll.
        let tick = Arc::new(AtomicU64::new(60));
        let mut replay = ReplaySource::with_clock(
            script,
            Box::new({
                let t = Arc::clone(&tick);
                move || t.fetch_add(100, Ordering::Relaxed)
            }),
        );
        // pad(0) reads wframe=60 (active in both ranges -> button 4 + key 69),
        // then advances clock to 160 (outside ranges); keys() must still see wframe=60!
        let pad_at_60 = replay.pad(0).unwrap();
        assert_eq!(pad_at_60.buttons, 4);
        let keys_at_60: Vec<u8> = replay.keys().into_iter().map(|k| k.virtual_key).collect();
        assert_eq!(keys_at_60, vec![69]);
    }

    #[test]
    fn recording_rejects_in_repo_and_relative_paths() {
        let mock = || -> Box<dyn Source> {
            Box::new(MockSource {
                pad0: None,
                keys: Vec::new(),
            })
        };
        assert!(
            RecordingSource::new(mock(), Path::new("relative.jsonl"), Box::new(|| 0)).is_err()
        );
        let in_repo = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("forbidden.jsonl");
        assert!(RecordingSource::new(mock(), &in_repo, Box::new(|| 0)).is_err());
    }

    fn replay_with_clock(script: InputScript, clock: Arc<AtomicU64>) -> ReplaySource {
        ReplaySource::with_clock(script, Box::new(move || clock.load(Ordering::Relaxed)))
    }

    #[test]
    fn one_frame_pulse_latches_across_zero_poll_gap_and_delivers_once() {
        let script = InputScript::parse("{\"from\":2,\"to\":2,\"ly\":1.0,\"keys\":[87]}\n").unwrap();
        let clock = Arc::new(AtomicU64::new(0));
        let mut replay = replay_with_clock(script, Arc::clone(&clock));

        // Poll frames [1, 3, 4]: frame 2 has zero polls, so the 1-frame pulse
        // latches onto frame 3, stays consistent within frame 3, and clears on frame 4.
        clock.store(1, Ordering::Relaxed);
        assert_eq!(replay.pad(0), Some(PadState::default()));
        assert!(replay.keys().is_empty());

        clock.store(3, Ordering::Relaxed);
        assert_eq!(replay.pad(0).unwrap().left_y, 1.0);
        let k3: Vec<u8> = replay.keys().into_iter().map(|k| k.virtual_key).collect();
        assert_eq!(k3, vec![87]);

        // Repeated poll within the same delivery world frame keeps the latched pulse active.
        assert_eq!(replay.pad(0).unwrap().left_y, 1.0);
        let k3_again: Vec<u8> = replay.keys().into_iter().map(|k| k.virtual_key).collect();
        assert_eq!(k3_again, vec![87]);

        // Advancing past the delivery world frame releases the pulse permanently.
        clock.store(4, Ordering::Relaxed);
        assert_eq!(replay.pad(0), Some(PadState::default()));
        assert!(replay.keys().is_empty());

        clock.store(5, Ordering::Relaxed);
        assert_eq!(replay.pad(0), Some(PadState::default()));
        assert!(replay.keys().is_empty());

        // First world frame skipped after loading screen (`[0, 2, 3]` for `from=1, to=1`).
        let first_frame_pulse = InputScript::parse("{\"from\":1,\"to\":1,\"ly\":1.0}\n").unwrap();
        let clock2 = Arc::new(AtomicU64::new(0));
        let mut replay2 = replay_with_clock(first_frame_pulse, Arc::clone(&clock2));
        assert_eq!(replay2.pad(0), Some(PadState::default()));
        clock2.store(2, Ordering::Relaxed);
        assert_eq!(replay2.pad(0).unwrap().left_y, 1.0);
        clock2.store(3, Ordering::Relaxed);
        assert_eq!(replay2.pad(0), Some(PadState::default()));

        // Loading-screen pulse (`from=0, to=0`) never leaks into world frame 1.
        let zero_pulse = InputScript::parse("{\"from\":0,\"to\":0,\"ly\":1.0}\n").unwrap();
        let clock3 = Arc::new(AtomicU64::new(0));
        let mut replay3 = replay_with_clock(zero_pulse, Arc::clone(&clock3));
        assert_eq!(replay3.pad(0), Some(PadState::default()));
        clock3.store(1, Ordering::Relaxed);
        assert_eq!(replay3.pad(0), Some(PadState::default()));
    }

    #[test]
    fn one_frame_pulse_normal_poll_delivers_once_on_target_frame() {
        let script = InputScript::parse("{\"from\":2,\"to\":2,\"ly\":1.0,\"keys\":[87]}\n").unwrap();
        let clock = Arc::new(AtomicU64::new(0));
        let mut replay = replay_with_clock(script, Arc::clone(&clock));

        // Normal poll sequence [1, 2, 3]: delivers on frame 2, neutral on 1 & 3.
        clock.store(1, Ordering::Relaxed);
        assert_eq!(replay.pad(0), Some(PadState::default()));
        assert!(replay.keys().is_empty());

        clock.store(2, Ordering::Relaxed);
        assert_eq!(replay.pad(0).unwrap().left_y, 1.0);
        let k2: Vec<u8> = replay.keys().into_iter().map(|k| k.virtual_key).collect();
        assert_eq!(k2, vec![87]);
        assert_eq!(replay.pad(0).unwrap().left_y, 1.0);
        let k2_again: Vec<u8> = replay.keys().into_iter().map(|k| k.virtual_key).collect();
        assert_eq!(k2_again, vec![87]);

        clock.store(3, Ordering::Relaxed);
        assert_eq!(replay.pad(0), Some(PadState::default()));
        assert!(replay.keys().is_empty());
    }

    #[test]
    fn multi_frame_span_does_not_latch_past_to_across_zero_poll_gap() {
        let script = InputScript::parse("{\"from\":2,\"to\":3,\"ly\":1.0,\"keys\":[87]}\n").unwrap();
        let clock = Arc::new(AtomicU64::new(0));
        let mut replay = replay_with_clock(script, Arc::clone(&clock));

        // Multi-frame span (to > from) polled at [1, 4] must NOT latch past `to`.
        clock.store(1, Ordering::Relaxed);
        assert_eq!(replay.pad(0), Some(PadState::default()));
        assert!(replay.keys().is_empty());

        clock.store(4, Ordering::Relaxed);
        assert_eq!(replay.pad(0), Some(PadState::default()));
        assert!(replay.keys().is_empty());
    }

    #[test]
    fn invalid_replay_script_does_not_register_digest() {
        let _digest = DIGEST_TEST.lock().unwrap();
        let dir = tempfile::tempdir().unwrap();
        let bad_path = dir.path().join("invalid.jsonl");
        std::fs::write(&bad_path, "{not valid json\n").unwrap();

        victory_gpu::bench::set_input_script_sha256("");
        assert_eq!(victory_gpu::bench::input_script_sha256(), None);
        assert!(ReplaySource::from_file(&bad_path).is_err());
        assert_eq!(victory_gpu::bench::input_script_sha256(), None);

        let good_path = dir.path().join("valid.jsonl");
        let good_bytes = b"{\"from\":1,\"to\":2,\"ly\":1.0}\n";
        std::fs::write(&good_path, good_bytes).unwrap();
        assert!(ReplaySource::from_file(&good_path).is_ok());
        assert_eq!(
            victory_gpu::bench::input_script_sha256().as_deref(),
            Some(blake3_hex_via_bench(good_bytes).as_str())
        );
    }
}
