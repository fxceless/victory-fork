//! Samples of where the line's thread (`victory-submit`, the thread that
//! prepares and sends submissions with `VICTORY_ASYNC_SUBMIT=1`) is, taken by
//! a thread of its own (`VICTORY_SAMPLE_LINE=<microseconds between samples>`,
//! off by default; 2000 is a good start, 1000 with named threads: each
//! sample stops the thread for tens of microseconds).
//!
//! Each sample: suspend the line's thread, read its registers, copy the part
//! of its stack in use (at most [`COPY_MOST`]) into a buffer made at the
//! start, resume it. Between the suspend and the resume nothing allocates,
//! locks, logs or looks anything up: the suspended thread may hold the heap,
//! the loader's locks or the function tables' lock. The walk up the stack
//! (`RtlLookupFunctionEntry` + `RtlVirtualUnwind`) runs afterwards, on the
//! copy, its pointers into the stack moved onto the copy.
//!
//! With every frame-rate line (or every 10 s without one) the samples are
//! named (dbghelp, our PDB beside the program, the system's modules by their
//! exports) and logged in lines that begin `line samples #<n>`: the head, the
//! functions the thread was in (`self`, inlined functions counted as their
//! own, and `self-fn`, by the function the code lies in), those anywhere on
//! the stack (`incl`), whole stacks (`stack`: the innermost and five
//! callers), modules (`module`), system functions (`sys`) and a `sys sum` of
//! waits and memory calls. `VICTORY_SAMPLE_LINE_KEEP=1`: each report covers
//! everything since the start instead of the time since the one before.
//! A table's row reads `line samples #<n> <table> <rank>: <share>% <ms> ms/f
//! <samples> | <name>`, so one report is the lines of one `#<n>`, from its
//! `head` to its `end`.
//!
//! `VICTORY_SAMPLE_THREADS=<prefix>[,<prefix>...]` (with
//! `VICTORY_SAMPLE_LINE`, which gives the time between samples): at every
//! sample, every thread of the process whose name begins with one of the
//! prefixes is sampled too, one after the other (one suspended at a time,
//! into the same buffer). They are found by a snapshot of the process's
//! threads every second; their stacks' bounds come from their TEBs (its
//! `NT_TIB`: the stack's top and its committed bottom). Threads are pooled in
//! groups by their names without trailing numbers (`victory-records-3` is
//! `victory-records`), each reported in the same tables, lines that begin
//! `line samples #<n> [<group>]`; their `ms/f` are summed over the group's
//! threads (samples / samples taken of the group x ms per frame). When the
//! threads stood suspended more than a tenth of the time, groups are left
//! out of a sample in turn. Without `VICTORY_ASYNC_SUBMIT=1` (no line thread)
//! the groups alone are sampled, from the first frame-rate line on.

use std::collections::HashMap;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE, INVALID_HANDLE_VALUE, LocalFree, STILL_ACTIVE};
use windows_sys::Win32::System::Diagnostics::Debug::{
    CONTEXT, CONTEXT_CONTROL_AMD64, CONTEXT_INTEGER_AMD64, GetThreadContext, RtlLookupFunctionEntry,
    RtlVirtualUnwind, UNW_FLAG_NHANDLER,
};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{
    CreateToolhelp32Snapshot, TH32CS_SNAPTHREAD, THREADENTRY32, Thread32First, Thread32Next,
};
use windows_sys::Win32::System::Threading::{
    GetCurrentProcessId, GetCurrentThread, GetCurrentThreadId, GetCurrentThreadStackLimits, GetExitCodeThread,
    GetThreadDescription, OpenThread, ResumeThread, SetThreadPriority, SuspendThread, THREAD_GET_CONTEXT,
    THREAD_PRIORITY_HIGHEST, THREAD_QUERY_INFORMATION, THREAD_QUERY_LIMITED_INFORMATION, THREAD_SUSPEND_RESUME,
};

/// Frames kept of a stack, the innermost first.
const DEPTH: usize = 24;
/// Most of the stack copied per sample, from the stack pointer up.
const COPY_MOST: usize = 512 * 1024;
/// Words of the buffer past the most copied, for reads past a stack's top
/// by a broken frame.
const SLACK: usize = 8192;
/// Share of the time the sampled threads may stand suspended before groups
/// are left out of samples.
const BOUND: f64 = 0.10;
/// Most suspended time saved up for later samples (seconds).
const CREDIT_MOST: f64 = 0.010;
/// Rows of the tables of functions.
const ROWS: usize = 40;
/// Rows of the table of whole stacks.
const STACK_ROWS: usize = 25;
/// Frames of a whole stack: the innermost and its callers.
const STACK_FRAMES: usize = 9;
/// Longest name logged.
const NAME_MOST: usize = 900;

/// The line's thread: its id, its stack's bounds.
static ID: AtomicU32 = AtomicU32::new(0);
static LOW: AtomicUsize = AtomicUsize::new(0);
static HIGH: AtomicUsize = AtomicUsize::new(0);
/// A frame-rate line came: report, for this many frames (`f64` bits).
static DUE: AtomicBool = AtomicBool::new(false);
static FRAMES: AtomicU64 = AtomicU64::new(0);

/// `VICTORY_SAMPLE_LINE`: the time between samples, if asked for.
fn every() -> Option<Duration> {
    static EVERY: OnceLock<Option<Duration>> = OnceLock::new();
    *EVERY.get_or_init(|| {
        let us = std::env::var("VICTORY_SAMPLE_LINE").ok()?.trim().parse::<u64>().ok()?;
        (us > 0).then(|| Duration::from_micros(us.max(100)))
    })
}

/// `VICTORY_SAMPLE_THREADS`: the beginnings of the names of the threads
/// sampled besides the line's.
fn prefixes() -> &'static [String] {
    static PREFIXES: OnceLock<Vec<String>> = OnceLock::new();
    PREFIXES.get_or_init(|| {
        std::env::var("VICTORY_SAMPLE_THREADS")
            .map(|v| v.split(',').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect())
            .unwrap_or_default()
    })
}

/// The thread that takes the samples runs.
static RUNNING: AtomicBool = AtomicBool::new(false);

/// Starts the thread that takes the samples, once: whether it runs.
fn start(every: Duration) -> bool {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::Relaxed) {
        return RUNNING.load(Ordering::Acquire);
    }
    match std::thread::Builder::new().name("victory-line-samples".into()).spawn(move || run(every)) {
        Ok(_) => {
            RUNNING.store(true, Ordering::Release);
            true
        }
        Err(e) => {
            tracing::warn!(error = %e, "line samples: no thread to take them");
            false
        }
    }
}

/// The calling thread is the line's: it is sampled, if asked for (the first
/// thread to call this alone).
pub(crate) fn line_thread() {
    let Some(every) = every() else {
        return;
    };
    static CHOSEN: AtomicBool = AtomicBool::new(false);
    if CHOSEN.swap(true, Ordering::Relaxed) {
        return;
    }
    let (mut low, mut high) = (0usize, 0usize);
    // SAFETY: writes the calling thread's stack bounds.
    unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
    // SAFETY: plain query.
    let id = unsafe { GetCurrentThreadId() };
    LOW.store(low, Ordering::Relaxed);
    HIGH.store(high, Ordering::Relaxed);
    ID.store(id, Ordering::Release);
    if start(every) {
        tracing::info!(
            "line samples: the line's thread (id {id}, stack {} KB) sampled every {} us (VICTORY_SAMPLE_LINE)",
            (high - low) / 1024,
            every.as_micros()
        );
    }
}

/// With the frame-rate line: the samples since the last are reported. (The
/// first starts the sampling when only named threads are asked for.)
pub(crate) fn frame_line(frames: f64) {
    if !RUNNING.load(Ordering::Acquire) {
        if !prefixes().is_empty()
            && let Some(every) = every()
        {
            start(every);
        }
        return;
    }
    FRAMES.store(frames.to_bits(), Ordering::Relaxed);
    DUE.store(true, Ordering::Release);
}

/// A `CONTEXT` at the 16-byte alignment `GetThreadContext` and
/// `RtlVirtualUnwind` need (windows-sys declares it 8-byte aligned).
#[repr(C, align(16))]
struct Aligned(CONTEXT);

/// What was taken since the last report.
#[derive(Default)]
struct Taken {
    /// Stacks (zero after their end) and how often each was seen.
    stacks: HashMap<[u64; DEPTH], u64>,
    samples: u64,
    /// Not suspended or no registers read.
    failed: u64,
    /// Stacks deeper than [`COPY_MOST`].
    cut: u64,
    /// Time from before the suspend to after the resume.
    suspended: Duration,
    copied: u64,
    frames_walked: u64,
    /// Frame-rate lines' frames, and the time, covered.
    frames: f64,
    began: Option<Instant>,
}

impl Taken {
    fn new() -> Taken {
        Taken {
            began: Some(Instant::now()),
            ..Taken::default()
        }
    }

    /// Emptied for the next report, keeping the table's room.
    fn clear(&mut self) {
        self.stacks.clear();
        *self = Taken {
            stacks: std::mem::take(&mut self.stacks),
            ..Taken::new()
        };
    }
}

/// Threads of the same name but for trailing numbers, sampled together.
struct Group {
    name: String,
    taken: Taken,
    /// Samples at which the group's threads were taken, and those at which
    /// they were left out (the bound on the time suspended).
    ticks: u64,
    skipped: u64,
    /// Threads taken since the last report.
    seen: Vec<u32>,
}

/// A thread of a group.
struct Other {
    id: u32,
    handle: HANDLE,
    /// Its TEB.
    teb: usize,
    group: usize,
}

/// The group a thread of this name falls in: the name without trailing
/// numbers (and the `-`, `_`, `#` or space before them).
fn group_of(name: &str) -> &str {
    let bare = name.trim_end_matches(|c: char| c.is_ascii_digit());
    if bare.len() == name.len() {
        return name;
    }
    let bare = bare.trim_end_matches(['-', '_', '#', ' ']);
    if bare.is_empty() { name } else { bare }
}

type QueryThread = unsafe extern "system" fn(HANDLE, u32, *mut core::ffi::c_void, u32, *mut u32) -> i32;

/// ntdll's `NtQueryInformationThread`.
fn query_thread() -> Option<QueryThread> {
    use windows_sys::Win32::System::LibraryLoader::{GetModuleHandleW, GetProcAddress};
    type Raw = unsafe extern "system" fn() -> isize;
    let ntdll: Vec<u16> = "ntdll.dll".encode_utf16().chain([0]).collect();
    // SAFETY: the system's own module (always loaded) and function, by its
    // name and signature.
    unsafe {
        let module = GetModuleHandleW(ntdll.as_ptr());
        if module.is_null() {
            return None;
        }
        let f = GetProcAddress(module, c"NtQueryInformationThread".as_ptr().cast())?;
        Some(std::mem::transmute::<Raw, QueryThread>(f))
    }
}

/// The TEB of the thread opened as `handle` (0: not known).
fn teb_of(query: QueryThread, handle: HANDLE) -> usize {
    // `THREAD_BASIC_INFORMATION` (48 bytes): the exit status (and padding),
    // the TEB's address, the client id (2), the affinity, the priorities.
    let mut info = [0usize; 6];
    let mut len = 0u32;
    // SAFETY: `ThreadBasicInformation` (0) into a buffer of its size.
    let status = unsafe { query(handle, 0, info.as_mut_ptr().cast(), size_of_val(&info) as u32, &mut len) };
    if status < 0 { 0 } else { info[1] }
}

/// The name of the thread opened as `handle`.
fn description(handle: HANDLE) -> Option<String> {
    let mut text: windows_sys::core::PWSTR = std::ptr::null_mut();
    // SAFETY: the system allocates the text; it is read up to its end and
    // freed as the system says.
    unsafe {
        if GetThreadDescription(handle, &mut text) < 0 || text.is_null() {
            return None;
        }
        let mut len = 0;
        while *text.add(len) != 0 {
            len += 1;
        }
        let name = String::from_utf16_lossy(std::slice::from_raw_parts(text, len));
        LocalFree(text.cast());
        Some(name)
    }
}

/// Every thread of the process named as asked for joins `others` (those
/// ended leave it first); `rejected`: threads named otherwise, not looked at
/// again while they live; `skip`: the line's thread and this one.
fn scan(
    query: QueryThread,
    skip: [u32; 2],
    others: &mut Vec<Other>,
    groups: &mut Vec<Group>,
    rejected: &mut Vec<u32>,
) {
    others.retain(|o| {
        let mut code = 0u32;
        // SAFETY: a handle of ours.
        let alive = unsafe { GetExitCodeThread(o.handle, &mut code) } != 0 && code == STILL_ACTIVE as u32;
        if !alive {
            // SAFETY: ours, not used again.
            unsafe { CloseHandle(o.handle) };
        }
        alive
    });
    // SAFETY: plain queries; the snapshot's handle is closed below.
    let (pid, snapshot) = unsafe { (GetCurrentProcessId(), CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0)) };
    if snapshot.is_null() || snapshot == INVALID_HANDLE_VALUE {
        return;
    }
    let mut entry = THREADENTRY32 {
        dwSize: size_of::<THREADENTRY32>() as u32,
        ..THREADENTRY32::default()
    };
    let mut still: Vec<u32> = Vec::with_capacity(rejected.len());
    // SAFETY: the snapshot's handle; an entry of the size it says.
    let mut more = unsafe { Thread32First(snapshot, &mut entry) } != 0;
    while more {
        let id = entry.th32ThreadID;
        if entry.th32OwnerProcessID == pid && !skip.contains(&id) && !others.iter().any(|o| o.id == id) {
            if rejected.contains(&id) {
                still.push(id);
            } else {
                match open(query, id) {
                    Opened::Ours(handle, teb, name) => {
                        let group = group_of(&name);
                        let group = match groups.iter().position(|g| g.name == group) {
                            Some(g) => g,
                            None => {
                                let mut taken = Taken::new();
                                taken.stacks.reserve(4096);
                                groups.push(Group {
                                    name: group.to_string(),
                                    taken,
                                    ticks: 0,
                                    skipped: 0,
                                    seen: Vec::new(),
                                });
                                groups.len() - 1
                            }
                        };
                        others.push(Other { id, handle, teb, group });
                    }
                    Opened::Other => still.push(id),
                    // Not named yet, perhaps: looked at again next time.
                    Opened::NotNow => {}
                }
            }
        }
        // SAFETY: as above.
        more = unsafe { Thread32Next(snapshot, &mut entry) } != 0;
    }
    // SAFETY: ours, not used again.
    unsafe { CloseHandle(snapshot) };
    *rejected = still;
}

enum Opened {
    /// Named as asked for: its handle, TEB, name.
    Ours(HANDLE, usize, String),
    /// Named otherwise.
    Other,
    /// Not opened, not named or no TEB.
    NotNow,
}

fn open(query: QueryThread, id: u32) -> Opened {
    let access = THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION | THREAD_QUERY_LIMITED_INFORMATION;
    // SAFETY: a thread of our own process.
    let handle = unsafe { OpenThread(access, 0, id) };
    if handle.is_null() {
        return Opened::NotNow;
    }
    let close = || {
        // SAFETY: ours, not used again.
        unsafe { CloseHandle(handle) };
    };
    let name = match description(handle) {
        Some(name) if !name.is_empty() => name,
        _ => {
            close();
            return Opened::NotNow;
        }
    };
    if !prefixes().iter().any(|p| name.starts_with(p.as_str())) {
        close();
        return Opened::Other;
    }
    let teb = teb_of(query, handle);
    if teb == 0 {
        close();
        return Opened::NotNow;
    }
    Opened::Ours(handle, teb, name)
}

fn run(every: Duration) {
    // SAFETY: this thread's own priority (to be off the line's thread's
    // back soon once it is suspended).
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST) };
    // SAFETY: plain query.
    let me = unsafe { GetCurrentThreadId() };
    let query = if prefixes().is_empty() {
        None
    } else {
        let query = query_thread();
        match query {
            Some(_) => tracing::info!(
                "line samples: threads named {:?}... sampled too, every {} us, grouped by name without trailing numbers \
                 (VICTORY_SAMPLE_THREADS)",
                prefixes(),
                every.as_micros()
            ),
            None => tracing::warn!("line samples: no NtQueryInformationThread: named threads are not sampled"),
        }
        query
    };
    // The line's thread: its handle and stack bounds, once it is there.
    let mut line: Option<(HANDLE, usize, usize)> = None;
    let mut line_id = 0u32;
    // The stacks' mirror: the line's whole reserve, or the most copied of
    // another's, plus room for reads past its top by a broken frame, so the
    // walk never reads outside it. Threads are sampled one after the other
    // into it.
    let mut copy: Vec<u64> = if query.is_some() { vec![0; COPY_MOST / 8 + SLACK] } else { Vec::new() };
    let keep = std::env::var_os("VICTORY_SAMPLE_LINE_KEEP").is_some_and(|v| v == "1");
    let mut taken = Taken::new();
    taken.stacks.reserve(16384);
    let mut others: Vec<Other> = Vec::new();
    let mut groups: Vec<Group> = Vec::new();
    let mut rejected: Vec<u32> = Vec::new();
    let mut scanned: Option<Instant> = None;
    // Time the threads may still stand suspended (seconds), and where the
    // groups' turn begins.
    let mut credit = CREDIT_MOST;
    let mut credited = Instant::now();
    let mut turn = 0usize;
    // The groups' report period: since, frames.
    let mut period = (Instant::now(), 0.0f64);
    let mut names = Names::default();
    let mut number = 0u32;
    let mut reported = Instant::now();
    let mut next = Instant::now();
    loop {
        next += every;
        let now = Instant::now();
        if next > now {
            std::thread::sleep(next - now);
        } else if now - next > every * 8 {
            next = now;
        }
        if line_id == 0 {
            line_id = ID.load(Ordering::Acquire);
            if line_id != 0 {
                let (low, high) = (LOW.load(Ordering::Relaxed), HIGH.load(Ordering::Relaxed));
                // SAFETY: a thread of our own process.
                let thread = unsafe { OpenThread(THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT, 0, line_id) };
                if thread.is_null() || high <= low {
                    tracing::warn!("line samples: the line's thread could not be opened");
                    if query.is_none() {
                        return;
                    }
                } else {
                    copy.resize(copy.len().max((high - low) / 8 + SLACK), 0);
                    line = Some((thread, low, high));
                }
            }
        }
        let now = Instant::now();
        credit = (credit + (now - credited).as_secs_f64() * BOUND).min(CREDIT_MOST);
        credited = now;
        if let Some((thread, low, high)) = line {
            credit -= take(thread, Stack::Bounds { low, high }, &mut copy, &mut taken).as_secs_f64();
        }
        if let Some(query) = query {
            if scanned.is_none_or(|s| s.elapsed() >= Duration::from_secs(1)) {
                scan(query, [me, line_id], &mut others, &mut groups, &mut rejected);
                scanned = Some(Instant::now());
            }
            let n = groups.len();
            if n > 0 {
                turn = (turn + 1) % n;
            }
            for k in 0..n {
                let g = (turn + k) % n;
                if !others.iter().any(|o| o.group == g) {
                    continue;
                }
                let group = &mut groups[g];
                if credit <= 0.0 {
                    group.skipped += 1;
                    continue;
                }
                group.ticks += 1;
                for o in others.iter().filter(|o| o.group == g) {
                    credit -= take(o.handle, Stack::Teb(o.teb), &mut copy, &mut group.taken).as_secs_f64();
                    if !group.seen.contains(&o.id) {
                        group.seen.push(o.id);
                    }
                }
            }
        }
        let due = DUE.swap(false, Ordering::Acquire);
        if due || reported.elapsed() >= Duration::from_secs(10) {
            if due {
                let frames = f64::from_bits(FRAMES.load(Ordering::Relaxed));
                taken.frames += frames;
                period.1 += frames;
            }
            number += 1;
            names.refresh();
            if line.is_some() {
                report(number, &taken, &mut names, keep);
            }
            let mut order: Vec<usize> = (0..groups.len()).collect();
            order.sort_by(|&a, &b| groups[a].name.cmp(&groups[b].name));
            for g in order {
                let live = others.iter().filter(|o| o.group == g).count();
                report_group(number, &groups[g], live, period, &mut names, keep);
            }
            reported = Instant::now();
            if !keep {
                taken.clear();
                for group in &mut groups {
                    group.taken.clear();
                    group.ticks = 0;
                    group.skipped = 0;
                    group.seen.clear();
                }
                period = (Instant::now(), 0.0);
            }
            // Naming may have taken long (the PDB loads at the first).
            next = Instant::now();
        }
    }
}

/// Where a sampled thread's stack lies.
#[derive(Clone, Copy)]
enum Stack {
    /// The line's: its bounds, mirrored whole (`low` at the buffer's start).
    Bounds { low: usize, high: usize },
    /// Another's: its TEB, whose `NT_TIB` gives the stack's top and its
    /// committed bottom, read while it stands suspended; the copy goes at
    /// the buffer's start.
    Teb(usize),
}

/// One sample: the time the thread stood suspended.
fn take(thread: HANDLE, stack: Stack, copy: &mut [u64], taken: &mut Taken) -> Duration {
    let mut context = Aligned(CONTEXT::default());
    context.0.ContextFlags = CONTEXT_CONTROL_AMD64 | CONTEXT_INTEGER_AMD64;
    let room = copy.len() * 8;
    let base = copy.as_mut_ptr().cast::<u8>();
    let began = Instant::now();
    // SAFETY: the thread is another one of this process. Between the suspend
    // and the resume: one read of its registers, for another than the line's
    // two reads of its TEB (alive: a thread that ends cannot be suspended),
    // and one copy of its stack's used part (committed, as it is in use;
    // another's from its stack pointer up, not below the committed bottom,
    // so never a guard page) into our buffer made before, inside it as the
    // line's stack lies inside `low..high` and another's copy is at most
    // `COPY_MOST`, which the buffer holds with `SLACK` to spare; nothing
    // that allocates, locks or logs.
    let (got, copied, at, high) = unsafe {
        if SuspendThread(thread) == u32::MAX {
            (false, 0, 0, 0)
        } else {
            let got = GetThreadContext(thread, &mut context.0) != 0;
            let rsp = context.0.Rsp as usize;
            // The stack's bounds, the address mirrored at the buffer's start
            // (another's 16-byte aligned like the stack), the room past it.
            let (low, high, at, most) = match stack {
                Stack::Bounds { low, high } => (low, high, low, room),
                Stack::Teb(teb) => {
                    let tib = teb as *const usize;
                    let (top, bottom) = (tib.add(1).read_volatile(), tib.add(2).read_volatile());
                    (bottom, top, rsp & !15, room.saturating_sub(SLACK * 8))
                }
            };
            let mut copied = 0;
            if got && rsp >= low && rsp < high && rsp - at < most {
                copied = (high - rsp).min(COPY_MOST).min(most - (rsp - at));
                std::ptr::copy_nonoverlapping(rsp as *const u8, base.add(rsp - at), copied);
            }
            ResumeThread(thread);
            (got, copied, at, high)
        }
    };
    let spent = began.elapsed();
    taken.suspended += spent;
    if !got {
        taken.failed += 1;
        return spent;
    }
    let rsp = context.0.Rsp as usize;
    if copied > 0 && rsp + copied < high {
        taken.cut += 1;
    }
    let mut frames = [0u64; DEPTH];
    let n = walk(&mut context, at, copied, base, &mut frames);
    taken.samples += 1;
    taken.copied += copied as u64;
    taken.frames_walked += n as u64;
    *taken.stacks.entry(frames).or_insert(0) += 1;
    spent
}

/// The return addresses of the stack copied at `base` (mirroring the stack
/// from `low`): its pointers into itself and the registers that point into
/// it are moved onto the copy first. Frames: how many.
fn walk(context: &mut Aligned, low: usize, copied: usize, base: *mut u8, frames: &mut [u64; DEPTH]) -> usize {
    let c = &mut context.0;
    frames[0] = c.Rip;
    if copied == 0 {
        return 1;
    }
    let rsp0 = c.Rsp as usize;
    let delta = (base as u64).wrapping_sub(low as u64);
    let (from, to) = (rsp0 as u64, (rsp0 + copied) as u64);
    let moved = |v: u64| if v >= from && v < to { v.wrapping_add(delta) } else { v };
    let words = base.cast::<u64>();
    for i in (rsp0 - low).div_ceil(8)..(rsp0 - low + copied) / 8 {
        // SAFETY: inside the part just copied.
        unsafe {
            let w = words.add(i);
            w.write(moved(w.read()));
        }
    }
    c.Rsp = moved(c.Rsp);
    c.Rbp = moved(c.Rbp);
    c.Rbx = moved(c.Rbx);
    c.Rsi = moved(c.Rsi);
    c.Rdi = moved(c.Rdi);
    c.R12 = moved(c.R12);
    c.R13 = moved(c.R13);
    c.R14 = moved(c.R14);
    c.R15 = moved(c.R15);
    let lo = base as u64 + (rsp0 - low) as u64;
    let hi = lo + copied as u64;
    let mut n = 1;
    while n < DEPTH {
        let rsp = c.Rsp;
        if rsp < lo || rsp + 8 > hi {
            break;
        }
        let mut image = 0u64;
        // SAFETY: looks up the function table entry covering `Rip`, if any
        // (the line's thread runs on: this may wait for a lock it holds).
        let entry = unsafe { RtlLookupFunctionEntry(c.Rip, &mut image, std::ptr::null_mut()) };
        if entry.is_null() {
            // A leaf without unwind data has its return address on top;
            // anything further up without it is not to be trusted.
            if n > 1 {
                break;
            }
            // SAFETY: `rsp` is inside the copy.
            c.Rip = unsafe { (rsp as *const u64).read_unaligned() };
            c.Rsp = rsp + 8;
        } else {
            let (mut data, mut frame) = (std::ptr::null_mut(), 0u64);
            // SAFETY: `entry` belongs to `image`; the context is ours,
            // aligned, its stack the copy (the buffer runs on past the
            // stack's top, so a broken frame reads inside it).
            unsafe {
                RtlVirtualUnwind(
                    UNW_FLAG_NHANDLER,
                    image,
                    c.Rip,
                    entry,
                    c,
                    &mut data,
                    &mut frame,
                    std::ptr::null_mut(),
                )
            };
        }
        if c.Rip == 0 || c.Rsp <= rsp || c.Rsp > hi {
            break;
        }
        frames[n] = c.Rip;
        n += 1;
    }
    n
}

/// An address named.
struct Named {
    /// The module's file name.
    module: String,
    /// The system's (ntdll, win32u, kernelbase, kernel32).
    system: bool,
    /// Innermost first: the functions inlined there, then the function the
    /// code lies in (always there, last).
    names: Vec<String>,
}

/// Names of addresses, kept from report to report.
#[derive(Default)]
struct Names {
    symbols: Option<Option<Symbols>>,
    /// (address, whether the innermost frame) to its place in `named`.
    known: HashMap<(u64, bool), usize>,
    named: Vec<Named>,
}

impl Names {
    fn of(&mut self, address: u64, leaf: bool) -> usize {
        if let Some(&at) = self.known.get(&(address, leaf)) {
            return at;
        }
        let symbols = self.symbols.get_or_insert_with(Symbols::load).as_ref();
        let named = name(symbols, address, leaf);
        self.named.push(named);
        self.known.insert((address, leaf), self.named.len() - 1);
        self.named.len() - 1
    }

    /// Modules loaded since the last report are looked at too.
    fn refresh(&mut self) {
        if let Some(Some(s)) = &self.symbols
            && let Some(refresh) = s.refresh
        {
            // SAFETY: the symbol handler's own process handle.
            unsafe { refresh(s.process) };
        }
    }
}

type FromAddr = unsafe extern "system" fn(HANDLE, u64, *mut u64, *mut u8) -> i32;
type InlineCount = unsafe extern "system" fn(HANDLE, u64) -> u32;
type InlineQuery = unsafe extern "system" fn(HANDLE, u64, u32, u64, u64, *mut u32, *mut u32) -> i32;
type FromInline = unsafe extern "system" fn(HANDLE, u64, u32, *mut u64, *mut u8) -> i32;
type Refresh = unsafe extern "system" fn(HANDLE) -> i32;

/// The system's dbghelp, loaded at the first report.
struct Symbols {
    process: HANDLE,
    from_address: FromAddr,
    inline: Option<(InlineCount, InlineQuery, FromInline)>,
    refresh: Option<Refresh>,
}

/// `SYMFLAG_EXPORT`: named by the module's export table alone.
const EXPORT: u32 = 0x200;

impl Symbols {
    fn load() -> Option<Symbols> {
        use windows_sys::Win32::System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExA};
        use windows_sys::Win32::System::Threading::GetCurrentProcess;
        type Raw = unsafe extern "system" fn() -> isize;
        type Initialize = unsafe extern "system" fn(HANDLE, *const u16, i32) -> i32;
        type Options = unsafe extern "system" fn(u32) -> u32;
        // SAFETY: the system's own library and functions, by their names and
        // signatures.
        unsafe {
            let module = LoadLibraryExA(c"dbghelp.dll".as_ptr().cast(), std::ptr::null_mut(), LOAD_LIBRARY_SEARCH_SYSTEM32);
            if module.is_null() {
                tracing::warn!("line samples: no dbghelp.dll: addresses are not named");
                return None;
            }
            let get = |name: &std::ffi::CStr| GetProcAddress(module, name.as_ptr().cast());
            let options = std::mem::transmute::<Raw, Options>(get(c"SymSetOptions")?);
            let initialize = std::mem::transmute::<Raw, Initialize>(get(c"SymInitializeW")?);
            let from_address = std::mem::transmute::<Raw, FromAddr>(get(c"SymFromAddr")?);
            let inline = match (get(c"SymAddrIncludeInlineTrace"), get(c"SymQueryInlineTrace"), get(c"SymFromInlineContext")) {
                (Some(count), Some(query), Some(from)) => Some((
                    std::mem::transmute::<Raw, InlineCount>(count),
                    std::mem::transmute::<Raw, InlineQuery>(query),
                    std::mem::transmute::<Raw, FromInline>(from),
                )),
                _ => None,
            };
            let refresh = get(c"SymRefreshModuleList").map(|f| std::mem::transmute::<Raw, Refresh>(f));
            let process = GetCurrentProcess();
            // Undecorated names, loaded when asked for, no dialogs.
            options(0x2 | 0x4 | 0x200);
            // Our symbols lie beside the program. (It fails if the handler
            // was set up before, by a timeline or a backtrace: it is there.)
            let beside = std::env::current_exe().ok().and_then(|p| Some(p.parent()?.to_path_buf())).unwrap_or_default();
            let beside: Vec<u16> = std::os::windows::ffi::OsStrExt::encode_wide(beside.as_os_str()).chain([0]).collect();
            initialize(process, beside.as_ptr(), 1);
            Some(Symbols {
                process,
                from_address,
                inline,
                refresh,
            })
        }
    }

    /// The symbol at `address` (in inline frame `context`, if one): its
    /// name, how far past its start, its flags.
    fn symbol(&self, address: u64, context: Option<u32>) -> Option<(String, u64, u32)> {
        // `SYMBOL_INFO`: 88 bytes with the name's first character at 84.
        const NAME: usize = 84;
        const ROOM: usize = 1024;
        let mut info = [0u64; (NAME + ROOM) / 8 + 2];
        let at = info.as_mut_ptr().cast::<u8>();
        let mut off = 0u64;
        // SAFETY: a buffer of the structure's size and the name's, its size
        // fields set as asked.
        unsafe {
            at.cast::<u32>().write(88);
            at.add(80).cast::<u32>().write(ROOM as u32);
            let found = match (context, self.inline) {
                (Some(context), Some((_, _, from))) => from(self.process, address, context, &mut off, at),
                (Some(_), None) => 0,
                (None, _) => (self.from_address)(self.process, address, &mut off, at),
            };
            if found == 0 {
                return None;
            }
            let flags = at.add(40).cast::<u32>().read();
            let name = std::ffi::CStr::from_ptr(at.add(NAME).cast());
            Some((name.to_string_lossy().into_owned(), off, flags))
        }
    }

    /// The functions inlined at `address`, innermost first.
    fn inlined(&self, address: u64) -> Vec<String> {
        let Some((count, query, _)) = self.inline else {
            return Vec::new();
        };
        let mut names = Vec::new();
        // SAFETY: the handler's process; plain queries.
        unsafe {
            let n = count(self.process, address);
            let (mut context, mut index) = (0u32, 0u32);
            if n == 0 || query(self.process, address, 0, address, address, &mut context, &mut index) == 0 {
                return names;
            }
            for i in 0..n {
                if let Some((name, _, _)) = self.symbol(address, Some(context + i)) {
                    names.push(name);
                }
            }
        }
        names
    }
}

/// The module `address` lies in: its file name, its base, whether it is
/// the program.
fn module(address: u64) -> Option<(String, u64, bool)> {
    use windows_sys::Win32::System::LibraryLoader::{
        GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT, GetModuleFileNameW,
        GetModuleHandleExW, GetModuleHandleW,
    };
    let mut module = std::ptr::null_mut();
    let mut path = [0u16; 260];
    // SAFETY: an address, no name; plain queries.
    unsafe {
        let flags = GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT;
        if GetModuleHandleExW(flags, address as *const u16, &mut module) == 0 {
            return None;
        }
        let len = GetModuleFileNameW(module, path.as_mut_ptr(), path.len() as u32) as usize;
        let path = String::from_utf16_lossy(&path[..len]);
        let file = path.rsplit('\\').next().unwrap_or("?").to_string();
        Some((file, module as u64, module == GetModuleHandleW(std::ptr::null())))
    }
}

/// Names `address` (a return address unless `leaf`: the call before it is
/// named).
fn name(symbols: Option<&Symbols>, address: u64, leaf: bool) -> Named {
    let at = if leaf { address } else { address.saturating_sub(1) };
    let Some((file, base, exe)) = module(at) else {
        return Named {
            module: "?".into(),
            system: false,
            names: vec![format!("{address:#x}")],
        };
    };
    let stem = file.rsplit_once('.').map_or(file.as_str(), |(s, _)| s).to_ascii_lowercase();
    let system = matches!(stem.as_str(), "ntdll" | "win32u" | "kernelbase" | "kernel32");
    let prefix = if exe { String::new() } else { format!("{stem}!") };
    // The start of the function the code lies in, by its unwind data.
    let mut image = 0u64;
    // SAFETY: a plain lookup.
    let entry = unsafe { RtlLookupFunctionEntry(at, &mut image, std::ptr::null_mut()) };
    let begin = (!entry.is_null()).then(|| {
        // SAFETY: an entry the system gave, of the image at `image`.
        let rva = unsafe { (*entry).BeginAddress };
        image + rva as u64
    });
    let unnamed = || match begin {
        Some(begin) => format!("{stem}+{:#x}", begin - base),
        None => format!("{stem}+{:#x}", at - base),
    };
    let mut names: Vec<String> = Vec::new();
    let mut physical = None;
    if let Some(s) = symbols {
        names.extend(s.inlined(at).into_iter().map(|n| format!("{prefix}{n} [inl]")));
        if let Some((n, off, flags)) = s.symbol(at, None) {
            let start = at - off;
            physical = Some(if flags & EXPORT == 0 {
                format!("{prefix}{n}")
            } else if begin.is_some_and(|b| start < b) || (begin.is_none() && off > 0x1000) {
                // The nearest export lies before this function: not it.
                format!("{} (after {n})", unnamed())
            } else {
                format!("{prefix}{n}")
            });
        }
    }
    names.push(physical.unwrap_or_else(unnamed));
    for n in &mut names {
        if n.len() > NAME_MOST {
            let mut end = NAME_MOST;
            while !n.is_char_boundary(end) {
                end -= 1;
            }
            n.truncate(end);
            n.push_str("...");
        }
    }
    Named { module: file, system, names }
}

/// Rows by count, the largest first.
fn top<K: Clone + Ord>(counts: &HashMap<K, u64>, rows: usize) -> Vec<(K, u64)> {
    let mut v: Vec<(K, u64)> = counts.iter().map(|(k, c)| (k.clone(), *c)).collect();
    v.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    v.truncate(rows);
    v
}

fn report(number: u32, taken: &Taken, names: &mut Names, keep: bool) {
    let total = taken.stacks.values().sum::<u64>();
    let secs = taken.began.map_or(0.0, |b| b.elapsed().as_secs_f64());
    let n = taken.samples.max(1) as f64;
    let ms_each = if taken.frames > 0.0 { secs * 1000.0 / taken.frames } else { 0.0 };
    tracing::info!(
        "line samples #{number} head: {total} samples in {secs:.2} s (one per {:.0} us), {:.1} frames ({ms_each:.1} ms each); \
         {:.1} us suspended per sample ({:.2}% of the time), {:.0} KB of stack copied per sample, {} not taken, {} stacks \
         cut at {} KB, {:.1} frames deep{}",
        secs * 1e6 / n,
        taken.frames,
        taken.suspended.as_secs_f64() * 1e6 / n,
        taken.suspended.as_secs_f64() * 100.0 / secs.max(1e-9),
        taken.copied as f64 / n / 1024.0,
        taken.failed,
        taken.cut,
        COPY_MOST / 1024,
        taken.frames_walked as f64 / n,
        if keep { "; since the start (VICTORY_SAMPLE_LINE_KEEP=1)" } else { "" }
    );
    if total == 0 {
        tracing::info!("line samples #{number} end");
        return;
    }
    let ms = (taken.frames > 0.0).then(|| ms_each / total as f64);
    tables(number, "", taken, ms, names);
}

/// A group's report: its head, then its tables, their `ms/f` summed over
/// the group's threads.
fn report_group(number: u32, group: &Group, live: usize, period: (Instant, f64), names: &mut Names, keep: bool) {
    let taken = &group.taken;
    let tag = format!("[{}] ", group.name);
    let total = taken.stacks.values().sum::<u64>();
    let secs = period.0.elapsed().as_secs_f64();
    let frames = period.1;
    let n = taken.samples.max(1) as f64;
    let ms_each = if frames > 0.0 { secs * 1000.0 / frames } else { 0.0 };
    tracing::info!(
        "line samples #{number} {tag}head: {total} samples of {} threads ({live} now) in {} ticks ({} more ticks left out to \
         keep threads suspended under {:.0}% of the time) in {secs:.2} s, {frames:.1} frames ({ms_each:.1} ms each); \
         columns: % of the group's samples, ms/f summed over the group's threads (samples / ticks x ms per frame); {:.1} us \
         suspended per sample ({:.2}% of the time), {:.0} KB of stack copied per sample, {} not taken, {} stacks cut at {} KB, \
         {:.1} frames deep{}",
        group.seen.len(),
        group.ticks,
        group.skipped,
        BOUND * 100.0,
        taken.suspended.as_secs_f64() * 1e6 / n,
        taken.suspended.as_secs_f64() * 100.0 / secs.max(1e-9),
        taken.copied as f64 / n / 1024.0,
        taken.failed,
        taken.cut,
        COPY_MOST / 1024,
        taken.frames_walked as f64 / n,
        if keep { "; since the start (VICTORY_SAMPLE_LINE_KEEP=1)" } else { "" }
    );
    if total == 0 {
        tracing::info!("line samples #{number} {tag}end");
        return;
    }
    let ms = (frames > 0.0 && group.ticks > 0).then(|| ms_each / group.ticks as f64);
    tables(number, &tag, taken, ms, names);
}

/// The tables of a report, their lines tagged `tag` after the report's
/// number; `ms`: ms per frame each sample counted stands for.
fn tables(number: u32, tag: &str, taken: &Taken, ms: Option<f64>, names: &mut Names) {
    let total = taken.stacks.values().sum::<u64>();
    // Name every frame first, then count.
    let mut stacks: Vec<(Vec<usize>, u64)> = Vec::with_capacity(taken.stacks.len());
    for (frames, &count) in &taken.stacks {
        let named: Vec<usize> =
            frames.iter().enumerate().take_while(|(i, a)| *i == 0 || **a != 0).map(|(i, &a)| names.of(a, i == 0)).collect();
        stacks.push((named, count));
    }
    let named = &names.named;
    let mut own: HashMap<&str, u64> = HashMap::new();
    let mut own_fn: HashMap<&str, u64> = HashMap::new();
    let mut anywhere: HashMap<&str, u64> = HashMap::new();
    let mut whole: HashMap<String, u64> = HashMap::new();
    let mut modules: HashMap<&str, u64> = HashMap::new();
    let mut system: HashMap<&str, u64> = HashMap::new();
    let (mut waits, mut memory, mut card, mut calls, mut other) = (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut seen: Vec<&str> = Vec::with_capacity(DEPTH * 4);
    for (frames, count) in &stacks {
        let leaf = &named[frames[0]];
        let innermost = leaf.names[0].as_str();
        *own.entry(innermost).or_insert(0) += count;
        *own_fn.entry(leaf.names.last().map_or("?", |s| s.as_str())).or_insert(0) += count;
        *modules.entry(leaf.module.as_str()).or_insert(0) += count;
        if leaf.system {
            let f = leaf.names.last().map_or("?", |s| s.as_str());
            *system.entry(f).or_insert(0) += count;
            let bare = f.split_once('!').map_or(f, |(_, b)| b);
            if ["WaitFor", "DelayExecution", "AlertByThreadId", "YieldExecution", "SignalAndWait", "RemoveIoCompletion"]
                .iter()
                .any(|w| bare.contains(w))
            {
                waits += count;
            } else if ["VirtualMemory", "WriteWatch", "ViewOfSection", "FlushInstructionCache"].iter().any(|w| bare.contains(w)) {
                memory += count;
            } else if bare.contains("DDI") || bare.starts_with("D3DKMT") {
                card += count;
            } else if bare.starts_with("Nt") || bare.starts_with("Zw") {
                calls += count;
            } else {
                other += count;
            }
        }
        seen.clear();
        let mut chain: Vec<&str> = Vec::with_capacity(STACK_FRAMES);
        for &f in frames {
            for name in &named[f].names {
                let name = name.as_str();
                if !seen.contains(&name) {
                    seen.push(name);
                    *anywhere.entry(name).or_insert(0) += count;
                }
                let plumbing = ["std::", "core::", "alloc::", "ntdll", "kernelbase", "<"].iter().any(|p| name.starts_with(p));
                if chain.len() < STACK_FRAMES && (chain.is_empty() || !plumbing) {
                    chain.push(name);
                }
            }
        }
        *whole.entry(chain.join(" <- ")).or_insert(0) += count;
    }
    let row = |table: &str, rank: usize, count: u64, name: &str| {
        let share = count as f64 / total as f64;
        let ms = match ms {
            Some(each) => format!("{:6.2}", count as f64 * each),
            None => "     -".into(),
        };
        tracing::info!("line samples #{number} {tag}{table} {rank:>2}: {:5.1}% {ms} ms/f {count:>6} | {name}", share * 100.0);
    };
    for (table, counts, rows) in [
        ("self", &own, ROWS),
        ("self-fn", &own_fn, ROWS),
        ("incl", &anywhere, ROWS),
        ("module", &modules, 12),
        ("sys", &system, 15),
    ] {
        for (i, (name, count)) in top(counts, rows).into_iter().enumerate() {
            row(table, i + 1, count, name);
        }
    }
    for (i, (chain, count)) in top(&whole, STACK_ROWS).into_iter().enumerate() {
        row("stack", i + 1, count, &chain);
    }
    let pct = |c: u64| c as f64 * 100.0 / total as f64;
    tracing::info!(
        "line samples #{number} {tag}sys sum: in system modules {:.1}%: waits {:.1}%, memory calls (protect, query, write \
         watch, views) {:.1}%, card calls (D3DKMT) {:.1}%, other system calls {:.1}%, other system code {:.1}%",
        pct(waits + memory + card + calls + other),
        pct(waits),
        pct(memory),
        pct(card),
        pct(calls),
        pct(other)
    );
    tracing::info!("line samples #{number} {tag}end");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threads_group_by_their_names_without_trailing_numbers() {
        assert_eq!(group_of("victory-records-3"), "victory-records");
        assert_eq!(group_of("victory-wide_12"), "victory-wide");
        assert_eq!(group_of("victory-submit"), "victory-submit");
        assert_eq!(group_of("worker #2"), "worker");
        // A name of digits alone stays as it is.
        assert_eq!(group_of("42"), "42");
    }

    #[test]
    fn tables_keep_the_largest_rows_first() {
        let counts: HashMap<&str, u64> = [("a", 3), ("b", 9), ("c", 3), ("d", 1)].into_iter().collect();
        assert_eq!(top(&counts, 3), [("b", 9), ("a", 3), ("c", 3)]);
        assert!(top(&counts, 0).is_empty());
    }

    /// Spins in a function of its own until told to stop.
    #[inline(never)]
    fn spin_here(stop: &AtomicBool) {
        while !stop.load(Ordering::Relaxed) {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn a_running_thread_is_sampled_by_its_bounds_and_by_its_teb() {
        use std::sync::Arc;
        let stop = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let s = stop.clone();
        let spinner = std::thread::spawn(move || {
            let (mut low, mut high) = (0usize, 0usize);
            // SAFETY: writes the calling thread's stack bounds.
            unsafe { GetCurrentThreadStackLimits(&mut low, &mut high) };
            // SAFETY: plain query.
            tx.send((unsafe { GetCurrentThreadId() }, low, high)).unwrap();
            spin_here(&s);
        });
        let (id, low, high) = rx.recv().unwrap();
        let access = THREAD_SUSPEND_RESUME | THREAD_GET_CONTEXT | THREAD_QUERY_INFORMATION;
        // SAFETY: a thread of this process, alive until `stop`.
        let handle = unsafe { OpenThread(access, 0, id) };
        assert!(!handle.is_null());
        let mut copy = vec![0u64; ((high - low) / 8).max(COPY_MOST / 8) + SLACK];
        let mut taken = Taken::new();
        for _ in 0..4 {
            take(handle, Stack::Bounds { low, high }, &mut copy, &mut taken);
        }
        let teb = query_thread().map_or(0, |q| teb_of(q, handle));
        assert_ne!(teb, 0);
        let mut by_teb = Taken::new();
        for _ in 0..4 {
            take(handle, Stack::Teb(teb), &mut copy, &mut by_teb);
        }
        stop.store(true, Ordering::Relaxed);
        spinner.join().unwrap();
        // SAFETY: ours, not used again.
        unsafe { CloseHandle(handle) };
        for t in [&taken, &by_teb] {
            assert_eq!((t.samples, t.failed), (4, 0));
            assert!(t.copied > 0);
            // The spinning function and at least one caller above it.
            assert!(t.frames_walked >= 4 * 2, "{} frames walked", t.frames_walked);
        }
    }
}
