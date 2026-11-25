//! How much of what the copy-back puts into title memory the CPU touches
//! before the next merge (`VICTORY_BACK_READ_PROBE=N`, off by default): a
//! measurement for a copy-back that would merge only what the CPU reads.
//!
//! The relay merges what shaders wrote to copies in the card's memory into
//! title memory as each flight lands: some 1,250 pieces of 64 KiB a frame,
//! 34 MB of bytes that changed (an RTX 4070 run), and the title's fences
//! are set only once that is done. If the title's CPU code reads little of
//! it, the merge could wait for the first access (a fault on a page kept
//! from the CPU) and be off the fences' path.
//!
//! With the switch, one 4 KiB page in `N` of each stretch the copy-back
//! wrote is marked with `PAGE_GUARD` on top of its protection. The first
//! access to it raises a guard-page exception, which the system clears
//! itself; our handler counts it (a read or a write, by any thread of the
//! process) and lets the access go on. Our own reads and writes through
//! the kernel (`ReadProcessMemory`, `NtWriteVirtualMemory`) fail on such a
//! page instead: `crate::host` then takes the marks off the range with
//! [`unmark`], counts it as ours, and does it again. A mark that a change
//! of protection drops (the guards' own) is not seen at all, so the counts
//! of the CPU's touches are a floor.
//!
//! Counted per frame line: pages marked; of those that left the set, by the
//! CPU's read, the CPU's write, our own access (a refresh, the next merge),
//! and still marked.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

const PAGE: usize = 4 << 10;

/// `N` of `VICTORY_BACK_READ_PROBE` (1 in `N` pages marked); 0: off.
pub(crate) fn every() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        let n = std::env::var("VICTORY_BACK_READ_PROBE").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(0);
        if n > 0 {
            install();
            tracing::info!(every = n, "pages the copy-back writes are watched for the CPU's first touch (VICTORY_BACK_READ_PROBE)");
        }
        n
    })
}

/// Marked pages (page number) and the protection they had.
static MARKED: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);

/// Every page ever marked: a guard-page exception on one of them is ours
/// even if [`unmark`] took it out of [`MARKED`] at the same moment (the
/// system has cleared the bit either way; passing it on would reach the
/// crash reporter).
static EVER: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

/// Since the last frame line: marked; then read by the CPU, written by the
/// CPU, taken by our own access.
static COUNT: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

/// The copy-back wrote title memory `[at, at + len)`: one page in [`every`]
/// of its whole pages is marked.
pub(crate) fn mark(at: usize, len: usize) {
    let n = every();
    if n == 0 || len < PAGE {
        return;
    }
    use windows_sys::Win32::System::Memory::{PAGE_GUARD, PAGE_READONLY, PAGE_READWRITE, VirtualProtect};
    let (first, last) = (at.div_ceil(PAGE), (at + len) / PAGE);
    let mut marked = MARKED.lock().unwrap_or_else(|e| e.into_inner());
    let marked = marked.get_or_insert_with(HashMap::new);
    for page in first..last {
        if !page.is_multiple_of(n) || marked.contains_key(&page) {
            continue;
        }
        let Some(protect) = protection(page * PAGE) else { continue };
        if protect != PAGE_READWRITE && protect != PAGE_READONLY {
            continue;
        }
        let mut old = 0;
        // SAFETY: a committed page of ours, its protection kept and the guard
        // bit added; the handler and `unmark` give it back.
        if unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, protect | PAGE_GUARD, &mut old) } != 0 {
            marked.insert(page, protect);
            EVER.lock().unwrap_or_else(|e| e.into_inner()).get_or_insert_with(Default::default).insert(page);
            COUNT[0].fetch_add(1, Relaxed);
        }
    }
}

/// Before our own access through the kernel to `[at, at + len)`: the marks
/// there are taken off (counted as ours). Whether there were any.
pub(crate) fn unmark(at: usize, len: usize) -> bool {
    if every() == 0 || len == 0 {
        return false;
    }
    use windows_sys::Win32::System::Memory::VirtualProtect;
    let mut marked = MARKED.lock().unwrap_or_else(|e| e.into_inner());
    let Some(marked) = marked.as_mut() else { return false };
    let mut any = false;
    for page in at / PAGE..(at + len).div_ceil(PAGE) {
        if let Some(protect) = marked.remove(&page) {
            let mut old = 0;
            // SAFETY: the page's own protection back, without the guard bit.
            unsafe { VirtualProtect((page * PAGE) as *const _, PAGE, protect, &mut old) };
            COUNT[3].fetch_add(1, Relaxed);
            any = true;
        }
    }
    any
}

fn protection(at: usize) -> Option<u32> {
    use windows_sys::Win32::System::Memory::{MEM_COMMIT, MEMORY_BASIC_INFORMATION, VirtualQuery};
    let mut info: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
    // SAFETY: a query into our own structure.
    let n = unsafe { VirtualQuery(at as *const _, &mut info, size_of::<MEMORY_BASIC_INFORMATION>()) };
    (n != 0 && info.State == MEM_COMMIT).then_some(info.Protect)
}

/// The handler of guard-page exceptions on marked pages: counted, and the
/// access goes on (the system has taken the guard bit off already).
fn install() {
    use windows_sys::Win32::Foundation::{NTSTATUS, STATUS_GUARD_PAGE_VIOLATION};
    use windows_sys::Win32::System::Diagnostics::Debug::{
        AddVectoredExceptionHandler, EXCEPTION_CONTINUE_EXECUTION, EXCEPTION_CONTINUE_SEARCH, EXCEPTION_POINTERS,
    };
    unsafe extern "system" fn on_guard(info: *mut EXCEPTION_POINTERS) -> i32 {
        // SAFETY: the system passes the exception being dispatched.
        let r = unsafe { &*(*info).ExceptionRecord };
        if r.ExceptionCode != STATUS_GUARD_PAGE_VIOLATION as NTSTATUS || r.NumberParameters < 2 {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        let page = r.ExceptionInformation[1] / PAGE;
        // Not ours (the title's own guard pages): passed on. `mark` and
        // `unmark` hold these locks only around kernel calls, which never
        // raise this exception, so waiting for them cannot wait for itself.
        if !EVER.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|e| e.contains(&page)) {
            return EXCEPTION_CONTINUE_SEARCH;
        }
        if MARKED.lock().unwrap_or_else(|e| e.into_inner()).as_mut().and_then(|m| m.remove(&page)).is_some() {
            const WRITE: usize = 1;
            COUNT[if r.ExceptionInformation[0] == WRITE { 2 } else { 1 }].fetch_add(1, Relaxed);
        }
        EXCEPTION_CONTINUE_EXECUTION
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: the handler only touches the set and the counters.
    ONCE.call_once(|| unsafe {
        AddVectoredExceptionHandler(1, Some(on_guard));
    });
}

/// With the frame line.
pub(crate) fn line(frames: f64) {
    if every() == 0 {
        return;
    }
    let [marked, read, written, ours] = [0, 1, 2, 3].map(|i| COUNT[i].swap(0, Relaxed) as f64 / frames);
    let left = MARKED.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, HashMap::len);
    tracing::info!(
        "copy-back pages watched per frame (1 in {}, VICTORY_BACK_READ_PROBE): {marked:.0} marked; first touched by the CPU \
         reading {read:.1}, by the CPU writing {written:.1}, by our own access first (a refresh, the next merge) {ours:.1}; \
         {left} still marked",
        every()
    );
}
