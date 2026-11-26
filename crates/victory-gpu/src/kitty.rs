//! KittyOverlay: a small panel of the frame's numbers, drawn into the top
//! left corner of every frame shown (like [`crate::fpsbox`], whose place it
//! takes while it is on).
//!
//! The panel says the frame rate and where the frame's time goes, so the
//! limit can be read off the picture: the PC's processors and memory, the
//! card's work and memory, and the parts of our own submitting thread
//! (preparing, copies up, copies back, waiting for the card).
//!
//! Like the frame-rate box it is drawn on the CPU with a 5 by 7 dot font
//! into a buffer the card copies from, once a second when the numbers
//! change; a buffer is never written again after it is made, so a present
//! still in flight reads what it was recorded with. The corners are
//! rounded by copying the top and bottom rows a little narrower.
//!
//! `VICTORY_KITTY_OVERLAY`: unset or `2` = the whole panel, `1` = only the
//! first six lines (frame rate, processors, memory), `0` = off (the
//! frame-rate box is drawn instead).

use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;

/// How much of the panel is drawn: 0 none, 1 the short one, 2 all.
pub(crate) fn level() -> u8 {
    static LEVEL: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *LEVEL.get_or_init(|| match std::env::var("VICTORY_KITTY_OVERLAY").as_deref() {
        Ok("0") => 0,
        Ok("1") => 1,
        _ => 2,
    })
}

/// The numbers the frame meter measured over its last second.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub(crate) struct Measured {
    pub fps: f64,
    pub ms: f64,
    /// The submitting thread preparing submissions, ms a frame.
    pub preparing: f64,
    /// Of that: copies of title memory up to the card (ms, MB a frame).
    pub copy_ms: f64,
    pub copy_mb: f64,
    /// Copies back into title memory (ms, MB a frame).
    pub back_ms: f64,
    pub back_mb: f64,
    /// The submitting thread waiting for the card, ms a frame.
    pub waiting: f64,
    /// The passes over the heap's views picked by index (textures staged
    /// for them among them; three times the budget while a backlog drains).
    pub heap_ms: f64,
    pub draws: f64,
    /// Draws a frame dropped for want of a host pipeline.
    pub dropped: f64,
    /// Pipelines being made (the shader cache warming up).
    pub making: f64,
}

/// The last [`Measured`] and a count that moves with every new one.
static MEASURED: Mutex<Option<Measured>> = Mutex::new(None);
static GENERATION: AtomicU64 = AtomicU64::new(0);
/// The card's busy share by our clock, in tenths of a percent;
/// `u32::MAX` while not known.
static CARD_BUSY: AtomicU32 = AtomicU32::new(u32::MAX);

/// The frame meter's numbers of the last second.
pub(crate) fn measured(m: Measured) {
    if level() == 0 {
        return;
    }
    *MEASURED.lock().unwrap_or_else(|e| e.into_inner()) = Some(m);
    GENERATION.fetch_add(1, Ordering::Relaxed);
}

/// The card's busy share (0 to 100) by the card clock's count.
pub(crate) fn card_busy(percent: f64) {
    if percent.is_finite() {
        CARD_BUSY.store((percent.clamp(0.0, 100.0) * 10.0).round() as u32, Ordering::Relaxed);
    }
}

/// The PC's processors, busy share of all of them since the last call
/// (0 to 100); none on the first call.
fn processors_busy() -> Option<f64> {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::Threading::GetSystemTimes;
    static LAST: Mutex<Option<[u64; 3]>> = Mutex::new(None);
    let (mut idle, mut kernel, mut user) = (FILETIME::default(), FILETIME::default(), FILETIME::default());
    // SAFETY: three out pointers to plain structures.
    if unsafe { GetSystemTimes(&mut idle, &mut kernel, &mut user) } == 0 {
        return None;
    }
    let ticks = |t: FILETIME| u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime);
    let now = [ticks(idle), ticks(kernel), ticks(user)];
    let last = LAST.lock().unwrap_or_else(|e| e.into_inner()).replace(now)?;
    // The kernel's time includes the idle time.
    let idle = now[0].saturating_sub(last[0]) as f64;
    let total = (now[1].saturating_sub(last[1]) + now[2].saturating_sub(last[2])) as f64;
    (total > 0.0).then(|| (100.0 * (1.0 - idle / total)).clamp(0.0, 100.0))
}

/// The PC's memory in use and in all, and this process's own commit, in bytes.
fn memory() -> Option<(u64, u64, u64)> {
    use windows_sys::Win32::System::ProcessStatus::{GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of::<MEMORYSTATUSEX>() as u32;
    // SAFETY: a structure that says its own size.
    if unsafe { GlobalMemoryStatusEx(&mut m) } == 0 {
        return None;
    }
    let mut p: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    p.cb = size_of::<PROCESS_MEMORY_COUNTERS_EX>() as u32;
    // SAFETY: this process's pseudo handle and a structure of the size given.
    let own = unsafe { GetProcessMemoryInfo(GetCurrentProcess(), (&raw mut p).cast(), p.cb) } != 0;
    Some((m.ullTotalPhys - m.ullAvailPhys, m.ullTotalPhys, if own { p.PrivateUsage as u64 } else { 0 }))
}

/// A colour, red, green and blue from 0 to 1 (as shown, no curve applied).
type Rgb = (f32, f32, f32);
const BACK: Rgb = (0.075, 0.082, 0.10);
const EDGE: Rgb = (0.25, 0.27, 0.31);
const LABEL: Rgb = (0.62, 0.65, 0.70);
const WHITE: Rgb = (0.92, 0.92, 0.92);
const GREEN: Rgb = (0.36, 0.86, 0.42);
const YELLOW: Rgb = (1.0, 0.86, 0.22);
const RED: Rgb = (1.0, 0.36, 0.36);

/// Green, yellow or red: `good` while `v` is at most `ok`, then yellow up to `bad`.
fn rising(v: f64, ok: f64, bad: f64) -> Rgb {
    if v <= ok {
        GREEN
    } else if v <= bad {
        YELLOW
    } else {
        RED
    }
}

/// The frame rate's colour: 30 and up green, 15 and up yellow.
fn rate_colour(fps: f64) -> Rgb {
    if fps >= 29.5 {
        GREEN
    } else if fps >= 15.0 {
        YELLOW
    } else {
        RED
    }
}

/// A line of the panel: a label, then pieces of text in their colours.
type Line = (&'static str, Vec<(String, Rgb)>);

/// What holds the frame back, by the numbers: the card, our submitting
/// thread, or neither (the title's own threads, or its waits).
fn limit(m: &Measured, card: Option<f64>) -> (&'static str, Rgb) {
    if card.is_some_and(|c| c >= 85.0) {
        return ("GPU", RED);
    }
    // Pipelines still being made and their draws dropped: the cache is
    // warming up (a new build or the first run); later frames say more.
    if m.making >= 20.0 && m.dropped >= 100.0 {
        return ("SHADER WARMUP", YELLOW);
    }
    if m.ms > 0.0 && m.heap_ms >= 0.3 * m.ms {
        return ("TEXTURE PASS", RED);
    }
    let ours = m.preparing + m.back_ms;
    if m.ms > 0.0 && ours >= 0.5 * m.ms {
        if m.copy_ms + m.back_ms >= 0.5 * ours {
            return ("COPIES", RED);
        }
        return ("PREPARE", RED);
    }
    ("TITLE CPU", YELLOW)
}

/// The panel's lines for `level`.
fn lines(level: u8, m: &Measured, card: Option<f64>, cpu: Option<f64>, mem: Option<(u64, u64, u64)>, video: Option<[u64; 4]>) -> Vec<Line> {
    let gb = |b: u64| b as f64 / (1u64 << 30) as f64;
    let mut out: Vec<Line> = Vec::new();
    let rate = rate_colour(m.fps);
    out.push(("FPS", vec![(format!("{:.1}", m.fps), rate), (format!("  {:.0} MS", m.ms), rate)]));
    out.push(("CPU", vec![cpu.map_or(("-".into(), LABEL), |c| (format!("{c:.0} %"), rising(c, 75.0, 90.0)))]));
    out.push(("GPU", vec![card.map_or(("-".into(), LABEL), |c| (format!("{c:.0} %"), rising(c, 80.0, 95.0)))]));
    out.push((
        "RAM",
        vec![mem.map_or(("-".into(), LABEL), |(used, all, _)| {
            let share = used as f64 / all.max(1) as f64;
            (format!("{:.1} / {:.0} GB", gb(used), gb(all)), rising(share, 0.80, 0.92))
        })],
    ));
    out.push((
        "VRAM",
        vec![video.map_or(("-".into(), LABEL), |v| {
            // [budget, in use] of the card's own memory come first.
            let (budget, used) = (v[0], v[1]);
            let share = used as f64 / budget.max(1) as f64;
            (format!("{:.1} / {:.1} GB", gb(used), gb(budget)), rising(share, 0.85, 0.97))
        })],
    ));
    out.push(("PROC", vec![mem.map_or(("-".into(), LABEL), |(_, _, own)| (format!("{:.1} GB", gb(own)), WHITE))]));
    if level < 2 {
        return out;
    }
    let part = |ms: f64| rising(if m.ms > 0.0 { ms / m.ms } else { 0.0 }, 0.25, 0.5);
    out.push(("PREP", vec![(format!("{:.1} MS", m.preparing), part(m.preparing))]));
    out.push((
        "COPY",
        vec![(format!("{:.1} MS", m.copy_ms), part(m.copy_ms)), (format!("  {:.0} MB", m.copy_mb), LABEL)],
    ));
    out.push((
        "BACK",
        vec![(format!("{:.1} MS", m.back_ms), part(m.back_ms)), (format!("  {:.0} MB", m.back_mb), LABEL)],
    ));
    out.push(("HEAP", vec![(format!("{:.1} MS", m.heap_ms), part(m.heap_ms))]));
    out.push(("WAIT", vec![(format!("{:.1} MS", m.waiting), WHITE)]));
    out.push((
        "DRAW",
        vec![
            (format!("{:.0}", m.draws), WHITE),
            (format!("  DROP {:.0}", m.dropped), if m.dropped >= 1.0 { YELLOW } else { LABEL }),
        ],
    ));
    out.push(("SHDR", vec![(format!("{:.0} BUILDING", m.making), if m.making >= 1.0 { YELLOW } else { LABEL })]));
    let (what, colour) = limit(m, card);
    out.push(("LIMIT", vec![(what.into(), colour)]));
    out
}

/// A glyph's seven rows, five dots each (bit 4 is the left one).
fn glyph(c: char) -> [u8; 7] {
    match c {
        '0'..='9' | '.' => {
            let mut rows = [0; 7];
            let fps = crate::fpsbox::glyph_rows(c);
            rows.copy_from_slice(&fps);
            rows
        }
        'A' => [0x0E, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'B' => [0x1E, 0x11, 0x11, 0x1E, 0x11, 0x11, 0x1E],
        'C' => [0x0E, 0x11, 0x10, 0x10, 0x10, 0x11, 0x0E],
        'D' => [0x1E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x1E],
        'E' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x1F],
        'F' => [0x1F, 0x10, 0x10, 0x1E, 0x10, 0x10, 0x10],
        'G' => [0x0E, 0x11, 0x10, 0x17, 0x11, 0x11, 0x0F],
        'H' => [0x11, 0x11, 0x11, 0x1F, 0x11, 0x11, 0x11],
        'I' => [0x0E, 0x04, 0x04, 0x04, 0x04, 0x04, 0x0E],
        'J' => [0x07, 0x02, 0x02, 0x02, 0x02, 0x12, 0x0C],
        'K' => [0x11, 0x12, 0x14, 0x18, 0x14, 0x12, 0x11],
        'L' => [0x10, 0x10, 0x10, 0x10, 0x10, 0x10, 0x1F],
        'M' => [0x11, 0x1B, 0x15, 0x15, 0x11, 0x11, 0x11],
        'N' => [0x11, 0x11, 0x19, 0x15, 0x13, 0x11, 0x11],
        'O' => [0x0E, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'P' => [0x1E, 0x11, 0x11, 0x1E, 0x10, 0x10, 0x10],
        'Q' => [0x0E, 0x11, 0x11, 0x11, 0x15, 0x12, 0x0D],
        'R' => [0x1E, 0x11, 0x11, 0x1E, 0x14, 0x12, 0x11],
        'S' => [0x0F, 0x10, 0x10, 0x0E, 0x01, 0x01, 0x1E],
        'T' => [0x1F, 0x04, 0x04, 0x04, 0x04, 0x04, 0x04],
        'U' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x11, 0x0E],
        'V' => [0x11, 0x11, 0x11, 0x11, 0x11, 0x0A, 0x04],
        'W' => [0x11, 0x11, 0x11, 0x15, 0x15, 0x15, 0x0A],
        'X' => [0x11, 0x11, 0x0A, 0x04, 0x0A, 0x11, 0x11],
        'Y' => [0x11, 0x11, 0x0A, 0x04, 0x04, 0x04, 0x04],
        'Z' => [0x1F, 0x01, 0x02, 0x04, 0x08, 0x10, 0x1F],
        '%' => [0x18, 0x19, 0x02, 0x04, 0x08, 0x13, 0x03],
        '/' => [0x01, 0x01, 0x02, 0x04, 0x08, 0x10, 0x10],
        '-' => [0, 0, 0, 0x1F, 0, 0, 0],
        ':' => [0, 0x0C, 0x0C, 0, 0x0C, 0x0C, 0],
        _ => [0; 7],
    }
}

/// Characters of the label column (the longest label and a gap).
const LABEL_COLUMNS: u32 = 6;

/// The panel as pixels: width, height, then a colour per pixel row by row,
/// and the corner radius in pixels.
fn bitmap(lines: &[Line], scale: u32) -> (u32, u32, Vec<Rgb>, u32) {
    let pad = 4 * scale;
    let advance = 6 * scale;
    let line_height = 10 * scale;
    let columns = lines
        .iter()
        .map(|(_, pieces)| LABEL_COLUMNS + pieces.iter().map(|(t, _)| t.chars().count() as u32).sum::<u32>())
        .max()
        .unwrap_or(0);
    let width = 2 * pad + columns * advance - scale;
    let height = 2 * pad + lines.len() as u32 * line_height - 3 * scale;
    let mut pixels = vec![BACK; (width * height) as usize];
    // A one-dot frame, a little lighter than the panel.
    for x in 0..width {
        for y in [0, height - 1] {
            pixels[(y * width + x) as usize] = EDGE;
        }
    }
    for y in 0..height {
        for x in [0, width - 1] {
            pixels[(y * width + x) as usize] = EDGE;
        }
    }
    let mut put = |text: &str, colour: Rgb, column: u32, row: u32| {
        for (i, c) in text.chars().enumerate() {
            for (dy, bits) in glyph(c).into_iter().enumerate() {
                for dx in 0..5u32 {
                    if bits >> (4 - dx) & 1 == 0 {
                        continue;
                    }
                    let x0 = pad + (column + i as u32) * advance + dx * scale;
                    let y0 = pad + row * line_height + dy as u32 * scale;
                    for y in y0..y0 + scale {
                        for x in x0..x0 + scale {
                            pixels[(y * width + x) as usize] = colour;
                        }
                    }
                }
            }
        }
    };
    for (row, (label, pieces)) in lines.iter().enumerate() {
        put(label, LABEL, 0, row as u32);
        let mut column = LABEL_COLUMNS;
        for (text, colour) in pieces {
            put(text, *colour, column, row as u32);
            column += text.chars().count() as u32;
        }
    }
    (width, height, pixels, 3 * scale)
}

/// How far row `y` of a panel `height` high is cut in at either side for
/// corners of `radius`.
fn inset(y: u32, height: u32, radius: u32) -> u32 {
    let from_edge = y.min(height - 1 - y);
    if from_edge >= radius {
        return 0;
    }
    let r = radius as f64;
    let dy = r - from_edge as f64 - 0.5;
    (r - (r * r - dy * dy).max(0.0).sqrt()).round() as u32
}

/// A panel the card copies from.
struct Made {
    buffer: ID3D12Resource,
    width: u32,
    height: u32,
    pitch: u32,
    radius: u32,
    format: DXGI_FORMAT,
    generation: u64,
    scale: u32,
}

// SAFETY: a free-threaded D3D12 object and plain numbers.
unsafe impl Send for Made {}

/// The panels made last, newest last; a few are kept so a present still in
/// flight has its buffer.
static MADE: Mutex<Vec<Made>> = Mutex::new(Vec::new());
const KEPT: usize = 8;

/// Copies the panel onto `back` (see [`crate::fpsbox::draw`] for the
/// arguments). False if the overlay is off, so the frame-rate box is drawn
/// instead.
pub(crate) fn draw(host: &crate::host::Host, list: &ID3D12GraphicsCommandList, back: &ID3D12Resource, (width, height, format): (u32, u32, DXGI_FORMAT)) -> bool {
    let level = level();
    if level == 0 {
        return false;
    }
    let Some(bytes) = crate::fpsbox::pixel_bytes(format) else {
        return true;
    };
    let generation = GENERATION.load(Ordering::Relaxed);
    let Some(measured) = *MEASURED.lock().unwrap_or_else(|e| e.into_inner()) else {
        return true;
    };
    let scale = (height / 480).clamp(2, 8);
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    let fresh = made.last().is_some_and(|m| m.generation == generation && m.format == format && m.scale == scale);
    if !fresh {
        let card = match CARD_BUSY.load(Ordering::Relaxed) {
            u32::MAX => None,
            tenths => Some(f64::from(tenths) / 10.0),
        };
        let lines = lines(level, &measured, card, processors_busy(), memory(), host.video_memory());
        let (w, h, pixels, radius) = bitmap(&lines, scale);
        let pitch = (w as usize * bytes).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT as usize);
        let Some((buffer, data)) = host.shared_buffer((pitch * h as usize) as u64) else {
            return true;
        };
        // SAFETY: the buffer's own mapped bytes, `pitch * h` of them.
        let out = unsafe { std::slice::from_raw_parts_mut(data, pitch * h as usize) };
        for y in 0..h as usize {
            for x in 0..w as usize {
                let at = y * pitch + x * bytes;
                crate::fpsbox::colour(format, pixels[y * w as usize + x], &mut out[at..at + bytes]);
            }
        }
        made.push(Made { buffer, width: w, height: h, pitch: pitch as u32, radius, format, generation, scale });
        if made.len() > KEPT {
            made.remove(0);
        }
    }
    let Some(m) = made.last() else {
        return true;
    };
    let margin = 4 * scale;
    if m.width + 2 * margin > width || m.height + 2 * margin > height {
        return true;
    }
    // SAFETY: an open list; the back buffer is a copy's destination in it
    // and every box lies inside it; our buffer holds `pitch * height` bytes
    // and is kept until [`KEPT`] newer panels were made.
    unsafe {
        let from = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(&m.buffer),
            Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
                PlacedFootprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
                    Offset: 0,
                    Footprint: D3D12_SUBRESOURCE_FOOTPRINT { Format: format, Width: m.width, Height: m.height, Depth: 1, RowPitch: m.pitch },
                },
            },
        };
        let to = D3D12_TEXTURE_COPY_LOCATION {
            pResource: std::mem::transmute_copy(back),
            Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
            Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 { SubresourceIndex: 0 },
        };
        // The rounded rows one by one, the straight middle in one copy.
        let mut y = 0;
        while y < m.height {
            let cut = inset(y, m.height, m.radius);
            let rows = if cut == 0 { (m.height - m.radius).saturating_sub(y).max(1) } else { 1 };
            let rect = D3D12_BOX { left: cut, top: y, front: 0, right: m.width - cut, bottom: y + rows, back: 1 };
            list.CopyTextureRegion(&to, margin + cut, margin + y, 0, &from, Some(&rect));
            y += rows;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Measured {
        Measured {
            fps: 16.5,
            ms: 61.0,
            preparing: 40.0,
            copy_ms: 15.0,
            copy_mb: 110.0,
            back_ms: 12.0,
            back_mb: 70.0,
            waiting: 5.0,
            heap_ms: 4.0,
            draws: 4700.0,
            dropped: 3.0,
            making: 0.0,
        }
    }

    #[test]
    fn every_character_the_panel_uses_has_a_glyph() {
        let lines = lines(2, &sample(), Some(66.0), Some(64.0), Some((30 << 30, 31 << 30, 20 << 30)), Some([10 << 30, 9 << 30, 0, 0]));
        assert_eq!(lines.len(), 14);
        for (label, pieces) in &lines {
            for c in label.chars().chain(pieces.iter().flat_map(|(t, _)| t.chars())) {
                assert!(c == ' ' || glyph(c).iter().any(|&r| r != 0), "{c:?} has no glyph");
            }
        }
        assert_eq!(super::lines(1, &sample(), None, None, None, None).len(), 6);
    }

    #[test]
    fn colours_follow_the_thresholds() {
        assert_eq!(rate_colour(30.0), GREEN);
        assert_eq!(rate_colour(16.5), YELLOW);
        assert_eq!(rate_colour(9.0), RED);
        assert_eq!(rising(0.95, 0.80, 0.92), RED);
        assert_eq!(rising(64.0, 75.0, 90.0), GREEN);
    }

    #[test]
    fn the_limit_names_the_largest_part() {
        assert_eq!(limit(&sample(), Some(90.0)).0, "GPU");
        let preparing = Measured { copy_ms: 5.0, back_ms: 5.0, ..sample() };
        assert_eq!(limit(&preparing, Some(40.0)).0, "PREPARE");
        let copies = Measured { copy_ms: 25.0, back_ms: 20.0, ..sample() };
        assert_eq!(limit(&copies, Some(40.0)).0, "COPIES");
        let title = Measured { preparing: 5.0, back_ms: 2.0, ..sample() };
        assert_eq!(limit(&title, Some(40.0)).0, "TITLE CPU");
        // Run 053618 in the world: 450 pipelines in the making, 1,100 to 2,700 draws dropped.
        let warming = Measured { making: 449.0, dropped: 1108.0, heap_ms: 1300.0, ms: 1193.0, ..sample() };
        assert_eq!(limit(&warming, Some(13.0)).0, "SHADER WARMUP");
        let staging = Measured { heap_ms: 755.0, ms: 1200.0, ..sample() };
        assert_eq!(limit(&staging, Some(13.0)).0, "TEXTURE PASS");
    }

    #[test]
    fn corners_are_cut_only_near_the_edges() {
        assert!(inset(0, 100, 9) > inset(4, 100, 9));
        assert_eq!(inset(9, 100, 9), 0);
        assert_eq!(inset(50, 100, 9), 0);
        assert_eq!(inset(99, 100, 9), inset(0, 100, 9));
    }

    #[test]
    fn the_panel_fits_its_text() {
        let lines = lines(2, &sample(), Some(66.0), Some(64.0), None, None);
        let (w, h, pixels, radius) = bitmap(&lines, 2);
        assert_eq!(pixels.len(), (w * h) as usize);
        assert_eq!(radius, 6);
        assert_eq!(pixels[0], EDGE);
        assert!(pixels.contains(&LABEL));
    }
}
