//! An on-screen panel of where the frame goes, drawn over the presented
//! frame's top left corner from the start: F3 hides or shows it
//! (`VICTORY_OVERLAY=0` starts with it hidden).
//!
//! It holds what the frame-rate line logs, once a second: frames per
//! second, time per frame and the last frames' spread; of a frame, the
//! submitting thread's preparation (title-memory copies, texture refreshes
//! and what they waited for), the GPU's time, the copy-back of what shaders
//! wrote; how busy the card was, how long it waited for us and we for it;
//! the processor time the process takes; draws, draws dropped and why,
//! pipelines being made, made and loaded, the driver's slow compiles;
//! textures picked by index and their queue; title buffer copies; process,
//! video and shared memory. The largest of preparing, GPU, copying back and
//! the heap passes in the title's threads is named the bottleneck. Below, a bar
//! per frame for the last frames' times.
//!
//! Each second's panel is also appended to a log, shown or not
//! (`VICTORY_OVERLAY_LOG=<file>`; by default `overlay.log` in the cache
//! folder; `0`: none).
//!
//! The text is drawn with GDI into a small bitmap once a second; each
//! present writes the panel into an upload buffer of its own and copies it
//! onto the back buffer, opaque (no pipeline or shader: it works whatever
//! the title draws, and costs about a millisecond of the present thread).

use std::collections::VecDeque;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use crate::host::private_bytes;
use windows::Win32::Foundation::COLORREF;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Gdi::*;
use windows::Win32::UI::Input::KeyboardAndMouse::{GetAsyncKeyState, VK_F3};
use windows::Win32::UI::WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId};
use windows::core::w;

/// What the panel shows, from the frame-rate line (per frame, over the
/// last second).
#[derive(Clone, Default)]
pub(crate) struct Stats {
    pub fps: f64,
    pub ms: f64,
    pub preparing: f64,
    pub copy_ms: f64,
    pub copy_mb: f64,
    pub texture_ms: f64,
    pub texture_mb: f64,
    /// Texture refreshes on all threads: ms in all, waiting for a texture
    /// another thread refreshes, asking the write watch, waiting for title
    /// memory to hold still.
    pub refresh: [f64; 4],
    pub gpu: f64,
    pub back_ms: f64,
    pub back_mb: f64,
    /// MB the CPU compared looking for shader writes.
    pub scanned_mb: f64,
    pub heap_pass_ms: f64,
    pub heap_passes: f64,
    /// Of the heap passes: ms making buffer views, refreshing their host
    /// copies, uploading textures picked by index; MB of the spans
    /// refreshed, and of them in copies not write-watched (read whole).
    pub heap_parts: [f64; 5],
    pub draws: f64,
    pub dropped: f64,
    pub indirect: f64,
    pub mesh_groups: f64,
    /// Percent of the frame the card had work of ours.
    pub card_busy: f64,
    /// Ms of a frame the card stood idle while the submitting thread worked.
    pub idle_behind: f64,
    /// Ms of a frame the submitting thread stood waiting for the card.
    pub stood: f64,
    /// Submissions sent ahead: flights, settles, ms settles waited.
    pub sent_ahead: [f64; 3],
    /// Title buffer copies recorded, and of them continuing the one before.
    pub copies: [f64; 2],
    /// Copies between depth textures and their twins, and planes drawn.
    pub twin_copies: f64,
    /// MB sent up to copies in the card's memory.
    pub card_sent_mb: f64,
    pub making: u64,
    /// The largest reasons draws were dropped (diagnostics builds).
    pub drops: Vec<(String, f64)>,
}

/// The panel's lines (colour, text) for the last second, and a count that
/// changes with them.
static LINES: Mutex<(u64, Vec<(u32, String)>)> = Mutex::new((0, Vec::new()));

/// The last second's figures: the panel's lines, logged.
pub(crate) fn update(stats: Stats) {
    let lines = lines(&stats);
    log(&lines);
    let mut shown = LINES.lock().unwrap_or_else(|e| e.into_inner());
    shown.0 += 1;
    shown.1 = lines;
}

/// Appends `lines` to the overlay's log, made afresh at the first call.
fn log(lines: &[(u32, String)]) {
    static FILE: Mutex<Option<Option<std::io::BufWriter<std::fs::File>>>> = Mutex::new(None);
    let mut file = FILE.lock().unwrap_or_else(|e| e.into_inner());
    let file = file.get_or_insert_with(|| {
        let path = match std::env::var_os("VICTORY_OVERLAY_LOG") {
            Some(v) if v == "0" => return None,
            Some(v) => PathBuf::from(v),
            None => crate::pipeline::library::dir()
                .and_then(Path::parent)
                .unwrap_or(Path::new("."))
                .join("overlay.log"),
        };
        match std::fs::File::create(&path) {
            Ok(f) => {
                tracing::info!(path = %path.display(), "overlay log");
                Some(std::io::BufWriter::new(f))
            }
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "no overlay log");
                None
            }
        }
    });
    let Some(file) = file else { return };
    for (_, text) in lines {
        let _ = writeln!(file, "{text}");
    }
    let _ = writeln!(file);
    let _ = file.flush();
}

/// Driver compiles of 2 s or more: how many, their milliseconds in all, the
/// slowest; and the last one's shaders.
pub(crate) static SLOW: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
pub(crate) static LAST_SLOW: Mutex<String> = Mutex::new(String::new());

/// Frame times, newest last.
const HISTORY: usize = 240;
static TIMES: Mutex<(Option<Instant>, VecDeque<f32>)> = Mutex::new((None, VecDeque::new()));

static SHOWN: AtomicBool = AtomicBool::new(false);
static KEY_DOWN: AtomicBool = AtomicBool::new(false);
/// The presented frame's width and height.
static SIZE: AtomicU64 = AtomicU64::new(0);

/// At each present of a `width` x `height` frame: the frame's time, and F3
/// (while one of our windows has the focus) shows or hides the panel.
/// Whether it is shown.
pub(crate) fn presented(width: u32, height: u32) -> bool {
    static STARTED: std::sync::Once = std::sync::Once::new();
    STARTED.call_once(|| {
        let on = std::env::var_os("VICTORY_OVERLAY").is_none_or(|v| v != "0");
        SHOWN.store(on, Ordering::Relaxed);
        tracing::info!(shown = on, "overlay (F3 shows or hides it)");
    });
    SIZE.store(
        u64::from(width) << 32 | u64::from(height),
        Ordering::Relaxed,
    );
    {
        let mut times = TIMES.lock().unwrap_or_else(|e| e.into_inner());
        let now = Instant::now();
        if let Some(last) = times.0 {
            if times.1.len() == HISTORY {
                times.1.pop_front();
            }
            times.1.push_back((now - last).as_secs_f32() * 1000.0);
        }
        times.0 = Some(now);
    }
    // SAFETY: plain queries.
    let down = unsafe {
        let mut process = 0;
        GetWindowThreadProcessId(GetForegroundWindow(), Some(&mut process));
        process == std::process::id() && GetAsyncKeyState(i32::from(VK_F3.0)) < 0
    };
    if down && !KEY_DOWN.swap(true, Ordering::Relaxed) {
        let shown = !SHOWN.fetch_xor(true, Ordering::Relaxed);
        tracing::info!(shown, "overlay (F3)");
    } else if !down {
        KEY_DOWN.store(false, Ordering::Relaxed);
    }
    SHOWN.load(Ordering::Relaxed)
}

/// The panel's pixels for one present: an upload buffer, mapped, laid out
/// as the back buffer's format; which text it holds.
pub(crate) struct Panel {
    upload: ID3D12Resource,
    data: *mut u8,
    footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT,
    text_version: u64,
}

// SAFETY: the mapping is only written by the present thread, under the
// presenter's lock.
unsafe impl Send for Panel {}

/// The text bitmap (BGRA, top-down), drawn once per [`update`].
struct Text {
    version: u64,
    width: u32,
    height: u32,
    line: u32,
    bgra: Vec<u8>,
}

static TEXT: Mutex<Option<Text>> = Mutex::new(None);

/// Colours, as GDI's 0x00BBGGRR.
const WHITE: u32 = 0x00E8_E8E8;
const GREY: u32 = 0x00A0_A0A0;
const RED: u32 = 0x004C_4CFF;
const YELLOW: u32 = 0x0040_D8FF;
const GREEN: u32 = 0x0060_E060;
const BACKGROUND: [u8; 4] = [18, 18, 18, 255];

/// The process's processor time since the last call, in percent of one
/// processor, and the seconds since the process started.
fn processor() -> (f64, f64) {
    use windows_sys::Win32::Foundation::FILETIME;
    use windows_sys::Win32::System::SystemInformation::GetSystemTimeAsFileTime;
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};
    static LAST: Mutex<(u64, u64)> = Mutex::new((0, 0));
    let ticks = |t: FILETIME| u64::from(t.dwHighDateTime) << 32 | u64::from(t.dwLowDateTime);
    let [mut created, mut exited, mut kernel, mut user, mut now] = [FILETIME {
        dwLowDateTime: 0,
        dwHighDateTime: 0,
    }; 5];
    // SAFETY: our own process and structures we own.
    unsafe {
        GetProcessTimes(
            GetCurrentProcess(),
            &mut created,
            &mut exited,
            &mut kernel,
            &mut user,
        );
        GetSystemTimeAsFileTime(&mut now);
    }
    let (busy, now) = (ticks(kernel) + ticks(user), ticks(now));
    let mut last = LAST.lock().unwrap_or_else(|e| e.into_inner());
    let percent = if last.1 != 0 && now > last.1 {
        100.0 * busy.saturating_sub(last.0) as f64 / (now - last.1) as f64
    } else {
        0.0
    };
    *last = (busy, now);
    // FILETIME counts 100 ns.
    (percent, now.saturating_sub(ticks(created)) as f64 / 1e7)
}

/// The panel's lines and their colours.
fn lines(s: &Stats) -> Vec<(u32, String)> {
    let mut out = Vec::new();
    let [count, total_ms, max_ms] = [0, 1, 2].map(|i| SLOW[i].load(Ordering::Relaxed));
    static CARD: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    let card = CARD.get_or_init(|| {
        crate::host::host()
            .and_then(|h| h.card_name())
            .unwrap_or_default()
    });
    let size = SIZE.load(Ordering::Relaxed);
    let (busy, up) = processor();
    out.push((
        WHITE,
        format!(
            "VIctory overlay (F3 hides)   {card}   {}x{}   up {up:.0} s   frame {}",
            size >> 32,
            size & 0xffff_ffff,
            crate::frame::current()
        ),
    ));
    let fps_colour = if s.fps >= 25.0 {
        GREEN
    } else if s.fps >= 10.0 {
        YELLOW
    } else {
        RED
    };
    out.push((
        fps_colour,
        format!("{:5.1} fps   {:7.1} ms/frame", s.fps, s.ms),
    ));
    {
        let mut times: Vec<f32> = TIMES
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .1
            .iter()
            .copied()
            .collect();
        if !times.is_empty() {
            times.sort_unstable_by(f32::total_cmp);
            let median = times[times.len() / 2];
            let p95 = times[times.len() * 95 / 100];
            let worst = times[times.len() - 1];
            let hitches = times
                .iter()
                .filter(|&&ms| ms > 2.0 * median.max(1.0))
                .count();
            out.push((
                if hitches > 0 { YELLOW } else { GREY },
                format!(
                    "last {} frames: median {median:.0} ms, 95% under {p95:.0} ms, worst {worst:.0} ms, {hitches} over twice the median; \
                     graph to {:.0} ms",
                    times.len(),
                    graph_top(median, worst)
                ),
            ));
        }
    }
    // What holds the frame up: the submitting thread's work (preparing,
    // waiting for the card, copying back) and the heap passes draws make
    // in the title's recording threads (the title cannot submit a list
    // before they end).
    let parts = [
        ("preparing (submitting thread)", s.preparing),
        ("GPU", s.gpu),
        ("copying back", s.back_ms),
        (
            "heap passes in the title's recording threads",
            s.heap_pass_ms,
        ),
    ];
    let (worst, worst_ms) = parts
        .iter()
        .copied()
        .fold(("", 0.0), |a, b| if b.1 > a.1 { b } else { a });
    let colour = |ms: f64| {
        if ms == worst_ms && ms > 0.0 {
            RED
        } else {
            WHITE
        }
    };
    out.push((
        colour(s.preparing),
        format!(
            "preparing    {:7.1} ms  copies {:.1} ms {:.0} MB, textures {:.1} ms {:.0} MB",
            s.preparing, s.copy_ms, s.copy_mb, s.texture_ms, s.texture_mb
        ),
    ));
    let [all, other_thread, asking, still] = s.refresh;
    out.push((
        GREY,
        format!(
            "  texture refreshes, all threads {all:.1} ms: waiting for another thread's {other_thread:.1}, \
             asking the write watch {asking:.1}, title memory to hold still {still:.1}"
        ),
    ));
    out.push((
        colour(s.gpu),
        format!(
            "GPU          {:7.1} ms  card busy {:.0}%, idle behind us {:.1} ms",
            s.gpu, s.card_busy, s.idle_behind
        ),
    ));
    out.push((
        colour(s.back_ms),
        format!(
            "copying back {:7.1} ms  {:.1} MB; shader writes looked for on the CPU in {:.0} MB",
            s.back_ms, s.back_mb, s.scanned_mb
        ),
    ));
    out.push((
        colour(s.heap_pass_ms),
        format!(
            "heap passes  {:7.1} ms  in {:.0} passes, the title's recording threads",
            s.heap_pass_ms, s.heap_passes
        ),
    ));
    let [views_ms, refresh_ms, textures_ms, span_mb, unwatched_mb] = s.heap_parts;
    out.push((
        GREY,
        format!(
            "  buffer views {views_ms:.1} ms, refreshing their copies {refresh_ms:.1} ms ({span_mb:.0} MB, {unwatched_mb:.0} MB not \
             write-watched), textures picked by index {textures_ms:.1} ms"
        ),
    ));
    let [flights, settles, settle_ms] = s.sent_ahead;
    out.push((
        GREY,
        format!(
            "waiting for the card {:.1} ms; sent ahead {flights:.0} in {settles:.1} settles ({settle_ms:.1} ms)",
            s.stood
        ),
    ));
    // The submitting thread's busy time against the frame's: what is left
    // is the title's (recording, its own work, our heap passes in it).
    let ours = s.preparing + s.gpu + s.back_ms;
    out.push((
        GREY,
        if ours < s.ms {
            format!(
                "submitting thread busy {ours:.0} of the frame's {:.0} ms; {:.0} ms waiting for the title to submit",
                s.ms,
                s.ms - ours
            )
        } else {
            format!("submitting thread busy all the frame's {:.0} ms ({ours:.0} ms of work, some of it overlapping)", s.ms)
        },
    ));
    if !worst.is_empty() {
        out.push((
            RED,
            format!(
                "bottleneck: {worst}, {worst_ms:.0} ms ({:.0}% of the frame's time)",
                100.0 * worst_ms / s.ms.max(1.0)
            ),
        ));
    }
    let processors = std::thread::available_parallelism().map_or(1, usize::from);
    out.push((
        if busy >= 90.0 * processors as f64 {
            RED
        } else {
            WHITE
        },
        format!("processor: {busy:.0}% of one ({processors} in all)"),
    ));
    let dropped_colour = if s.dropped >= 1.0 { YELLOW } else { WHITE };
    out.push((
        dropped_colour,
        format!(
            "draws {:.0}   dropped {:.0}   indirect {:.0}   mesh groups {:.0}",
            s.draws, s.dropped, s.indirect, s.mesh_groups
        ),
    ));
    for (why, n) in s.drops.iter().take(5) {
        let why: String = why.chars().take(72).collect();
        out.push((GREY, format!("  {n:6.1}  {why}")));
    }
    out.push((
        if s.making > 0 { YELLOW } else { WHITE },
        format!(
            "pipelines: {} being made, {} made, {} from the library; slow compiles {} ({:.0} s, slowest {:.1} s)",
            s.making,
            crate::pipeline::library::MADE.load(Ordering::Relaxed),
            crate::pipeline::library::LOADED.load(Ordering::Relaxed),
            count,
            total_ms as f64 / 1000.0,
            max_ms as f64 / 1000.0
        ),
    ));
    if count > 0 {
        let last = LAST_SLOW.lock().unwrap_or_else(|e| e.into_inner()).clone();
        let last: String = last.chars().take(72).collect();
        out.push((GREY, format!("  last slow: {last}")));
    }
    let waiting = crate::draw::HEAP_TEXTURES_WAITING.load(Ordering::Relaxed);
    out.push((
        if waiting > 0 { YELLOW } else { WHITE },
        format!(
            "textures by index: {} uploaded ({} MB), {} waiting, {} passed over unmapped",
            crate::draw::HEAP_TEXTURES.load(Ordering::Relaxed),
            crate::draw::HEAP_TEXTURE_BYTES.load(Ordering::Relaxed) >> 20,
            waiting,
            crate::draw::HEAP_TEXTURES_AWAY.load(Ordering::Relaxed)
        ),
    ));
    let [queued, due, away, partly, oldest] =
        [0, 1, 2, 3, 4].map(|i| crate::draw::HEAP_TEXTURES_QUEUE[i].load(Ordering::Relaxed));
    out.push((
        GREY,
        format!(
            "  queue: {queued} slots, {due} due (the oldest {oldest} frames), {away} waiting for their memory, {partly} uploaded in part"
        ),
    ));
    let [copies, joined] = s.copies;
    out.push((
        WHITE,
        format!(
            "title buffer copies {copies:.0} ({joined:.0} continue the one before); {:.1} depth twin copies",
            s.twin_copies
        ),
    ));
    if crate::shadow::cards() {
        out.push((
            GREY,
            format!(
                "  copies in the card's memory: {:.1} MB sent up",
                s.card_sent_mb
            ),
        ));
    }
    let gb = |b: u64| b as f64 / f64::from(1u32 << 30);
    let video = crate::host::host().and_then(|h| h.video_memory());
    out.push(match video {
        Some([budget, used, _, shared]) => (
            if used > budget || shared > 1 << 30 { RED } else { WHITE },
            format!(
                "memory: process {:.1} GB; video {:.1} of {:.1} GB, and {:.1} GB of the PC's shared",
                gb(private_bytes()),
                gb(used),
                gb(budget),
                gb(shared)
            ),
        ),
        None => (WHITE, format!("memory: process {:.1} GB", gb(private_bytes()))),
    });
    out
}

/// Draws `lines` with GDI: a bitmap `line` pixels a line, wide enough for
/// the longest.
fn draw_text(lines: &[(u32, String)], line: u32, version: u64) -> Option<Text> {
    // SAFETY: GDI objects made, selected, drawn into and freed here; the
    // bitmap's bits are read while it is alive.
    unsafe {
        let dc = CreateCompatibleDC(None);
        if dc.is_invalid() {
            return None;
        }
        let font = CreateFontW(
            -(line as i32 - 2),
            0,
            0,
            0,
            FW_NORMAL.0 as i32,
            0,
            0,
            0,
            DEFAULT_CHARSET,
            OUT_DEFAULT_PRECIS,
            CLIP_DEFAULT_PRECIS,
            ANTIALIASED_QUALITY,
            (FIXED_PITCH.0 | FF_MODERN.0) as u32,
            w!("Consolas"),
        );
        let old_font = SelectObject(dc, font.into());
        let mut size = windows::Win32::Foundation::SIZE::default();
        let widest = lines
            .iter()
            .map(|(_, t)| t.chars().count())
            .max()
            .unwrap_or(1);
        let _ = GetTextExtentPoint32W(dc, w!("M").as_wide(), &mut size);
        let pad = line / 2;
        let width = (size.cx.max(1) as u32 * widest as u32 + 2 * pad).next_multiple_of(4);
        let height = line * lines.len() as u32 + pad;
        let info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: width as i32,
                biHeight: -(height as i32),
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut bits = std::ptr::null_mut();
        let bitmap = CreateDIBSection(Some(dc), &info, DIB_RGB_COLORS, &mut bits, None, 0);
        let text = match bitmap {
            Ok(bitmap) if !bits.is_null() => {
                let old_bitmap = SelectObject(dc, bitmap.into());
                let len = (width * height * 4) as usize;
                let pixels = std::slice::from_raw_parts_mut(bits.cast::<u8>(), len);
                for p in pixels.chunks_exact_mut(4) {
                    p.copy_from_slice(&BACKGROUND);
                }
                SetBkMode(dc, TRANSPARENT);
                for (i, (colour, text)) in lines.iter().enumerate() {
                    SetTextColor(dc, COLORREF(*colour));
                    let wide: Vec<u16> = text.encode_utf16().collect();
                    let _ = TextOutW(dc, pad as i32, (pad / 2 + line * i as u32) as i32, &wide);
                }
                let _ = GdiFlush();
                let bgra = pixels.to_vec();
                SelectObject(dc, old_bitmap);
                let _ = DeleteObject(bitmap.into());
                Some(Text {
                    version,
                    width,
                    height,
                    line,
                    bgra,
                })
            }
            _ => None,
        };
        SelectObject(dc, old_font);
        let _ = DeleteObject(font.into());
        let _ = DeleteDC(dc);
        text
    }
}

/// Bytes per pixel of the back buffer formats [`crate::host`] presents.
fn bytes_per_pixel(format: DXGI_FORMAT) -> Option<u32> {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM | DXGI_FORMAT_B8G8R8A8_UNORM | DXGI_FORMAT_R10G10B10A2_UNORM => {
            Some(4)
        }
        DXGI_FORMAT_R16G16B16A16_FLOAT => Some(8),
        _ => None,
    }
}

/// An 8-bit channel as a half float (0..1).
fn half(c: u8) -> u16 {
    if c == 0 {
        return 0;
    }
    let f = f32::from(c) / 255.0;
    let bits = f.to_bits();
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = (bits >> 13) & 0x3ff;
    if exp <= 0 {
        0
    } else {
        ((exp as u32) << 10 | mantissa) as u16
    }
}

/// Writes one BGRA pixel at `to` in `format`.
fn put(format: DXGI_FORMAT, to: &mut [u8], [b, g, r, a]: [u8; 4]) {
    match format {
        DXGI_FORMAT_R8G8B8A8_UNORM => to[..4].copy_from_slice(&[r, g, b, a]),
        DXGI_FORMAT_B8G8R8A8_UNORM => to[..4].copy_from_slice(&[b, g, r, a]),
        DXGI_FORMAT_R10G10B10A2_UNORM => {
            let wide = |c: u8| u32::from(c) * 1023 / 255;
            let v = wide(r) | wide(g) << 10 | wide(b) << 20 | 3 << 30;
            to[..4].copy_from_slice(&v.to_le_bytes());
        }
        DXGI_FORMAT_R16G16B16A16_FLOAT => {
            for (k, c) in [r, g, b, a].into_iter().enumerate() {
                to[k * 2..k * 2 + 2].copy_from_slice(&half(c).to_le_bytes());
            }
        }
        _ => {}
    }
}

/// Records the panel's copy onto `back` (in the copy-destination state) in
/// `list`, using (and making, or remaking for a new size) `panel`. Nothing
/// if the panel is hidden or the format is one it cannot write.
pub(crate) fn record(
    device: &ID3D12Device,
    list: &ID3D12GraphicsCommandList,
    back: &ID3D12Resource,
    panel: &mut Option<Panel>,
    (width, height, format): (u32, u32, DXGI_FORMAT),
) {
    let Some(bpp) = bytes_per_pixel(format) else {
        return;
    };
    let (version, lines) = LINES.lock().unwrap_or_else(|e| e.into_inner()).clone();
    if version == 0 {
        return;
    }
    // The text: once per figures, a line a 64th of the frame's height.
    let line = (height / 64).clamp(12, 34);
    let mut text = TEXT.lock().unwrap_or_else(|e| e.into_inner());
    if text
        .as_ref()
        .is_none_or(|t| t.version != version || t.line != line)
    {
        let drawn = draw_text(&lines, line, version);
        if drawn.is_none() {
            tracing::warn!("overlay text not drawn");
            return;
        }
        *text = drawn;
    }
    let Some(text) = text.as_ref() else { return };
    let graph = line * 4;
    let (w, h) = (text.width.min(width), (text.height + graph).min(height));
    let pitch = (w * bpp).next_multiple_of(D3D12_TEXTURE_DATA_PITCH_ALIGNMENT);
    if panel.as_ref().is_none_or(|p| {
        let f = &p.footprint.Footprint;
        (f.Width, f.Height, f.Format) != (w, h, format)
    }) {
        *panel = make(device, w, h, pitch, format);
    }
    let Some(p) = panel.as_mut() else { return };
    // SAFETY: the buffer is mapped for its whole life and `pitch * h`
    // bytes long; no copy of it is in flight (a slot is reused once its
    // last present has run).
    let pixels = unsafe { std::slice::from_raw_parts_mut(p.data, (pitch * h) as usize) };
    let text_rows = text.height.min(h);
    if p.text_version != text.version {
        for y in 0..text_rows {
            let from = &text.bgra[(y * text.width * 4) as usize..][..(w * 4) as usize];
            let to = &mut pixels[(y * pitch) as usize..][..(w * bpp) as usize];
            for (src, dst) in from.chunks_exact(4).zip(to.chunks_exact_mut(bpp as usize)) {
                put(format, dst, [src[0], src[1], src[2], 255]);
            }
        }
        p.text_version = text.version;
    }
    // The graph: a bar per frame, newest at the right; the top is
    // [`graph_top`] (taller frames reach it in a lighter red); a line at
    // 33.3 ms (30 fps).
    let rows = h - text_rows;
    if rows > 0 {
        let times = TIMES.lock().unwrap_or_else(|e| e.into_inner()).1.clone();
        let mut sorted: Vec<f32> = times.iter().copied().collect();
        sorted.sort_unstable_by(f32::total_cmp);
        let top = sorted
            .last()
            .map_or(100.0, |&worst| graph_top(sorted[sorted.len() / 2], worst));
        let bar = w.div_ceil(HISTORY as u32).max(2);
        let at_30 = rows - ((33.3 / top) * rows as f32) as u32;
        for y in 0..rows {
            let row = &mut pixels[((text_rows + y) * pitch) as usize..][..(w * bpp) as usize];
            for x in 0..w {
                let from_right = ((w - 1 - x) / bar) as usize;
                let colour = match times
                    .len()
                    .checked_sub(1 + from_right)
                    .and_then(|i| times.get(i))
                {
                    Some(&ms) if y >= rows - ((ms.min(top) / top) * rows as f32) as u32 => {
                        if ms <= 34.0 {
                            [96, 224, 96, 255]
                        } else if ms <= 100.0 {
                            [64, 216, 255, 255]
                        } else if ms <= top {
                            [76, 76, 255, 255]
                        } else {
                            [160, 160, 255, 255]
                        }
                    }
                    _ if y == at_30 => [80, 80, 80, 255],
                    _ => BACKGROUND,
                };
                put(format, &mut row[(x * bpp) as usize..], colour);
            }
        }
    }
    let to = D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: borrowed for the call, as the copy locations are used.
        pResource: unsafe { std::mem::transmute_copy(back) },
        Type: D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            SubresourceIndex: 0,
        },
    };
    let from = D3D12_TEXTURE_COPY_LOCATION {
        // SAFETY: as above.
        pResource: unsafe { std::mem::transmute_copy(&p.upload) },
        Type: D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT,
        Anonymous: D3D12_TEXTURE_COPY_LOCATION_0 {
            PlacedFootprint: p.footprint,
        },
    };
    // SAFETY: a list being recorded, a back buffer in the copy-destination
    // state and an upload buffer laid out as the footprint says.
    unsafe { list.CopyTextureRegion(&to, 0, 0, 0, &from, None) };
    static FIRST: std::sync::Once = std::sync::Once::new();
    FIRST.call_once(|| tracing::info!(width = w, height = h, format = format.0, "overlay drawn"));
}

/// The frame-time graph's top: the slowest frame, but no more than four
/// times the median (one stall of seconds would flatten the rest), and at
/// least 100 ms.
fn graph_top(median: f32, worst: f32) -> f32 {
    worst.min(4.0 * median).max(100.0)
}

/// An upload buffer for a `w` x `h` panel of `format`, mapped.
fn make(device: &ID3D12Device, w: u32, h: u32, pitch: u32, format: DXGI_FORMAT) -> Option<Panel> {
    let heap = D3D12_HEAP_PROPERTIES {
        Type: D3D12_HEAP_TYPE_UPLOAD,
        ..Default::default()
    };
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: u64::from(pitch * h),
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        ..Default::default()
    };
    let mut upload: Option<ID3D12Resource> = None;
    let mut data = std::ptr::null_mut();
    // SAFETY: plain creation and mapping of an upload buffer.
    let made = unsafe {
        device
            .CreateCommittedResource(
                &heap,
                D3D12_HEAP_FLAG_NONE,
                &desc,
                D3D12_RESOURCE_STATE_GENERIC_READ,
                None,
                &mut upload,
            )
            .and_then(|()| {
                let u = upload.as_ref().ok_or_else(windows::core::Error::empty)?;
                u.Map(0, None, Some(&mut data))
            })
    };
    if let Err(e) = made {
        tracing::warn!(error = %e, "no overlay buffer");
        return None;
    }
    Some(Panel {
        upload: upload?,
        data: data.cast(),
        footprint: D3D12_PLACED_SUBRESOURCE_FOOTPRINT {
            Offset: 0,
            Footprint: D3D12_SUBRESOURCE_FOOTPRINT {
                Format: format,
                Width: w,
                Height: h,
                Depth: 1,
                RowPitch: pitch,
            },
        },
        text_version: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pixels reach the back buffer's layout: channels in its order, ten-bit
    /// channels scaled, half floats of 0..1.
    #[test]
    fn panel_pixels_take_the_back_buffers_layout() {
        let bgra = [10, 20, 255, 128];
        let mut px = [0u8; 8];
        put(DXGI_FORMAT_R8G8B8A8_UNORM, &mut px, bgra);
        assert_eq!(px[..4], [255, 20, 10, 128]);
        put(DXGI_FORMAT_B8G8R8A8_UNORM, &mut px, bgra);
        assert_eq!(px[..4], [10, 20, 255, 128]);
        put(DXGI_FORMAT_R10G10B10A2_UNORM, &mut px, [0, 0, 255, 255]);
        assert_eq!(
            u32::from_le_bytes(px[..4].try_into().unwrap()),
            1023 | 3 << 30
        );
        put(DXGI_FORMAT_R16G16B16A16_FLOAT, &mut px, [0, 0, 255, 255]);
        // 1.0 as a half is 0x3c00; blue 0.
        assert_eq!(px, [0x00, 0x3c, 0, 0, 0, 0, 0x00, 0x3c]);
        assert_eq!(half(0), 0);
        assert!((0x3800..0x3c00).contains(&half(128)), "{:#x}", half(128));
    }
}
