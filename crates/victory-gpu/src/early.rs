//! Title buffer copies that read memory shaders write before those shaders'
//! results were in title memory (`VICTORY_EARLY_READS`; on by default,
//! `=0` turns it off).
//!
//! A title `CopyBufferRegion` between two buffers runs on the CPU when its
//! submission is prepared (`crate::host`'s `Step::Copy`): it reads title
//! memory as it is then. On the console the copy runs on the card, in
//! order, after every shader before it. Two ways the CPU's copy can read
//! bytes from before a shader wrote them:
//!
//! - ahead: a flight still in the air declared shader writes over the
//!   source. Upstream's `fill_ahead` let such copies go without waiting
//!   (only fills wait now, see `crate::host` `copy_ahead`); a copy read
//!   after a targeted relay wait was not looked at by its check at all;
//! - own: the copy's own submission declared shader writes over the source
//!   before it (an earlier list, or earlier in the same list). The CPU
//!   copies before the card has run any of the submission's lists.
//!
//! Either way the destination gets the old bytes, and whoever reads it
//! (the title's CPU once the fence is set: a read-back of GPU culling
//! counts, of a "visible" flag; or later GPU work: indirect arguments)
//! decides from a value the card had already replaced. The copies are
//! noted when their steps run and compared when flights land: the bytes a
//! landing's merge puts into title memory at the source against the ones
//! there (what the copy read, unless the title wrote them since). A
//! difference is logged (the first ones in full, with the copy, the
//! submission that read and the flights that wrote) and counted on the
//! frame line.
//!
//! With `VICTORY_COPY_REPAIR=1` (off by default) a copy found so also has
//! its destination put right at the landing, before the fences behind the
//! flights are set: each destination byte that still holds what the copy
//! left there takes the byte the shaders wrote. That is what the title's
//! CPU would read on the console after its fence. GPU work sent before the
//! landing still read the old bytes.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Mutex, OnceLock};

/// Copies are looked up by the 64 KiB of title memory they read.
const PIECE: usize = 1 << 16;
/// A copy is compared with landings this many frames after it at most.
const KEEP: u64 = 16;
/// Hits logged in full before only every 1024th is.
const LOGGED: u64 = 16;

/// Whether copies are watched (`VICTORY_EARLY_READS=0`: not). Read once.
pub(crate) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_EARLY_READS").is_none_or(|v| v != "0"))
}

/// `VICTORY_COPY_REPAIR=1`: a copy found to have read early has its
/// destination put right at the landing. Read once.
fn repair() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_COPY_REPAIR").is_some_and(|v| v == "1"))
}

#[derive(Clone, Copy)]
struct Copy {
    src: usize,
    end: usize,
    dst: usize,
    frame: u64,
    /// The submission whose step it is.
    number: u64,
    /// Flights in the air declared writes over the source when it ran.
    ahead: bool,
    /// Its own submission declared writes over the source before it.
    own: bool,
}

#[derive(Default)]
struct Piece {
    pruned: u64,
    copies: Vec<Copy>,
}

static COPIES: Mutex<Option<HashMap<usize, Piece>>> = Mutex::new(None);

/// The submission numbers of the flights being landed now, lowest and
/// highest; (0, 0): not known (a landing by another way: every copy counts).
static LANDING: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// Since the last frame line: copies noted ahead, noted own; landings'
/// writes compared with one. Since the start: copies that read early
/// (ahead, own), destination bytes put right.
static COUNT: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

/// The flights numbered `first..=last` are being landed (0, 0: done).
pub(crate) fn landing(first: u64, last: u64) {
    LANDING[0].store(first, Relaxed);
    LANDING[1].store(last, Relaxed);
}

/// A title copy of `len` bytes from `src` to `dst`, a step of submission
/// `number`, is about to run on the CPU; `ahead` and `own` as in [`Copy`].
pub(crate) fn copy(dst: usize, src: usize, len: usize, number: u64, ahead: bool, own: bool) {
    if len == 0 || !(ahead || own) || !on() {
        return;
    }
    COUNT[if ahead { 0 } else { 1 }].fetch_add(1, Relaxed);
    let frame = crate::frame::current();
    let entry = Copy { src, end: src + len, dst, frame, number, ahead, own };
    let mut copies = COPIES.lock().unwrap_or_else(|e| e.into_inner());
    let map = copies.get_or_insert_with(HashMap::new);
    for piece in src / PIECE..=(src + len - 1) / PIECE {
        let p = map.entry(piece).or_default();
        if p.pruned != frame {
            p.pruned = frame;
            p.copies.retain(|c| c.frame + KEEP >= frame);
        }
        p.copies.push(entry);
    }
}

/// Whether a landing of flights `first..=last` ((0, 0): not known) can hold
/// writes `c` read before they were in title memory.
fn concerns(c: &Copy, first: u64, last: u64) -> bool {
    if first == 0 {
        return true;
    }
    // Ahead: a flight sent before the copy's submission; own: that one.
    (c.ahead && first < c.number) || (c.own && first <= c.number && c.number <= last)
}

/// A landing is about to put `new` into title memory at `at`: bytes
/// shaders wrote, merged into the title's (`crate::host::shaders_write`).
pub(crate) fn wrote(at: usize, new: &[u8]) {
    if new.is_empty() || !on() {
        return;
    }
    let end = at + new.len();
    let (first, last) = (LANDING[0].load(Relaxed), LANDING[1].load(Relaxed));
    let frame = crate::frame::current();
    let hits: Vec<Copy> = {
        let copies = COPIES.lock().unwrap_or_else(|e| e.into_inner());
        let Some(map) = copies.as_ref() else { return };
        let mut seen = Vec::new();
        for piece in at / PIECE..=(end - 1) / PIECE {
            let Some(p) = map.get(&piece) else { continue };
            for c in &p.copies {
                if c.src < end && c.end > at && c.frame + KEEP >= frame && concerns(c, first, last) {
                    // A copy over several pieces is in each of them.
                    if !seen.iter().any(|s: &Copy| s.src == c.src && s.dst == c.dst && s.number == c.number) {
                        seen.push(*c);
                    }
                }
            }
        }
        seen
    };
    if hits.is_empty() {
        return;
    }
    COUNT[2].fetch_add(1, Relaxed);
    let mut now = Vec::new();
    for c in hits {
        let (a, b) = (c.src.max(at), c.end.min(end));
        now.clear();
        now.resize(b - a, 0);
        // Title memory as it is: what the copy read unless the title wrote
        // it since.
        if !crate::host::read_into(a as *const u8, &mut now) {
            continue;
        }
        let shaders = &new[a - at..b - at];
        let Some(differs) = now.iter().zip(shaders).position(|(x, y)| x != y) else {
            continue;
        };
        COUNT[if c.ahead { 3 } else { 4 }].fetch_add(1, Relaxed);
        let total = COUNT[3].load(Relaxed) + COUNT[4].load(Relaxed);
        if total <= LOGGED || total.is_multiple_of(1024) {
            let word = |bytes: &[u8], i: usize| {
                let i = i & !3;
                let mut w = [0u8; 4];
                for (k, x) in w.iter_mut().enumerate() {
                    *x = bytes.get(i + k).copied().unwrap_or(0);
                }
                u32::from_le_bytes(w)
            };
            tracing::warn!(
                "early read: a title buffer copy of {} bytes from {:#x} to {:#x} (submission {}, frame {}) read bytes that \
                 shaders of flights {}..={} then changed (landing in frame {frame}; first at {:#x}: {:#010x} copied, \
                 {:#010x} written; {} bytes compared): {}; {} such copies ahead, {} own since the start{}",
                c.end - c.src,
                c.src,
                c.dst,
                c.number,
                c.frame,
                first,
                last,
                a + differs,
                word(&now, differs),
                word(shaders, differs),
                b - a,
                match (c.ahead, c.own) {
                    (true, true) => "flights in the air and its own submission declared writes over the source",
                    (true, false) => "a flight in the air declared writes over the source and the copy did not wait for it",
                    _ => "its own submission's shaders write the source, and the copy ran on the CPU before the card ran them",
                },
                COUNT[3].load(Relaxed),
                COUNT[4].load(Relaxed),
                if repair() { "; its destination is put right (VICTORY_COPY_REPAIR=1)" } else { "" },
            );
        }
        if repair() {
            put_right(c.dst + (a - c.src), &now, shaders);
        }
    }
}

/// [`repair`]: at `dst`, where a copy put `copied`, the bytes that still
/// hold it and that the shaders changed (`written`) take the shaders'.
fn put_right(dst: usize, copied: &[u8], written: &[u8]) {
    let mut there = vec![0u8; copied.len()];
    if !crate::host::read_into(dst as *const u8, &mut there) {
        return;
    }
    let mut fixed = there.clone();
    let mut any = false;
    for i in 0..fixed.len() {
        if there[i] == copied[i] && copied[i] != written[i] {
            fixed[i] = written[i];
            any = true;
        }
    }
    if !any {
        return;
    }
    let (Some(lo), Some(hi)) = (
        fixed.iter().zip(&there).position(|(x, y)| x != y),
        fixed.iter().zip(&there).rposition(|(x, y)| x != y),
    ) else {
        return;
    };
    let n = hi + 1 - lo;
    crate::watch::writing_exact(dst + lo, n);
    crate::host::write_mapped((dst + lo) as *mut u8, fixed[lo..].as_ptr(), n);
    COUNT[5].fetch_add(n as u64, Relaxed);
}

/// With the frame line.
pub(crate) fn line(frames: f64) {
    if !on() {
        return;
    }
    let [ahead, own, looked] = [0, 1, 2].map(|i| COUNT[i].swap(0, Relaxed));
    let early = [3, 4, 5].map(|i| COUNT[i].load(Relaxed));
    if ahead == 0 && own == 0 && looked == 0 && early == [0; 3] {
        return;
    }
    let kept = {
        let frame = crate::frame::current();
        let mut copies = COPIES.lock().unwrap_or_else(|e| e.into_inner());
        copies.as_mut().map_or(0, |map| {
            map.retain(|_, p| {
                p.copies.retain(|c| c.frame + KEEP >= frame);
                !p.copies.is_empty()
            });
            map.values().map(|p| p.copies.len()).sum::<usize>()
        })
    };
    tracing::info!(
        "title buffer copies that read memory shaders write before the shaders' results were in title memory (early reads, \
         VICTORY_EARLY_READS): per frame {:.1} under flights in the air, {:.1} under their own submission's shader writes, \
         {:.1} landings' writes compared with one; copies that read bytes the shaders then changed, since the start: {} \
         ahead, {} own; destination bytes put right {} (VICTORY_COPY_REPAIR={}); {kept} kept to check",
        ahead as f64 / frames,
        own as f64 / frames,
        looked as f64 / frames,
        early[0],
        early[1],
        early[2],
        u8::from(repair()),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn c(number: u64, ahead: bool, own: bool) -> Copy {
        Copy { src: 0, end: 8, dst: 64, frame: 0, number, ahead, own }
    }

    /// A copy ahead of flights is concerned by the landing of a flight sent
    /// before its submission; an own one by its submission's landing.
    #[test]
    fn landings_concern_the_copies_they_can_have_beaten() {
        assert!(concerns(&c(10, true, false), 7, 9));
        assert!(!concerns(&c(10, true, false), 10, 12), "flights sent after the copy wrote after it on the console too");
        assert!(concerns(&c(10, false, true), 9, 11));
        assert!(!concerns(&c(10, false, true), 11, 12));
        assert!(!concerns(&c(10, false, true), 7, 9));
        assert!(concerns(&c(10, false, true), 0, 0), "a landing not numbered concerns every copy");
    }
}
