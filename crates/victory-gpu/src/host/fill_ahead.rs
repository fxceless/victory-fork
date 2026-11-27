    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};

    /// Fills are looked up by the 64 KiB of title memory they read.
    const PIECE: usize = 1 << 16;
    /// A fill is checked against flights that land this many frames after
    /// it at most (flights are settled at 256 lists in the air, and by
    /// every present that waits).
    const KEEP: u64 = 16;

    /// A flight changed bytes a fill had read ahead of it: fills wait.
    static WAIT: AtomicBool = AtomicBool::new(false);
    /// Since the last frame line: fills that read ahead; landings' writes
    /// that touched one's piece. Since the start: fills that read bytes a
    /// flight then changed.
    static COUNT: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

    #[derive(Default)]
    struct Piece {
        /// The frame its old fills were last dropped in.
        pruned: u64,
        /// From, to, frame, and the number of the last submission started
        /// when it read: the flights in the air then have that number or
        /// a lower one, and only their writes can be ones it missed.
        fills: Vec<(usize, usize, u64, u64)>,
    }

    /// The lowest submission number among the flights the landing thread
    /// is landing now; 0: not known (a landing by another way), and then
    /// every write counts. (Run 211: a read of frame 1381 was taken for
    /// missed because shaders wrote the same 80 bytes in frame 1397, in a
    /// flight sent sixteen frames after it.)
    static LANDING: AtomicU64 = AtomicU64::new(0);

    pub(super) fn landing(first: u64) {
        LANDING.store(first, Relaxed);
    }

    static READ: Mutex<Option<HashMap<usize, Piece>>> = Mutex::new(None);

    pub(super) fn on() -> bool {
        static ON: OnceLock<bool> = OnceLock::new();
        *ON.get_or_init(|| std::env::var_os("VICTORY_FILL_WAIT").is_none_or(|v| v != "1")) && !WAIT.load(Relaxed)
    }

    /// With `VICTORY_CARD_FILLS` (`super::card_fills_mode`): the 64 KiB
    /// pieces of title memory shaders have changed this run, as landings
    /// put them in ([`wrote`]), and those a copy on the card writes
    /// ([`note_written`]).
    static LEARNED: Mutex<Option<std::collections::HashSet<usize>>> = Mutex::new(None);

    /// `[at, at + len)` is written by the card from now on (a copy of ours
    /// there, `super::HostList::card_copy`): noted as shaders' changes are.
    pub(crate) fn note_written(at: usize, len: usize) {
        if super::card_fills_mode() == 0 || len == 0 {
            return;
        }
        let mut learned = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
        learned.get_or_insert_with(Default::default).extend(at / PIECE..=(at + len - 1) / PIECE);
    }

    /// Whether `len` bytes at `src` lie in a piece shaders have changed this
    /// run (only with `VICTORY_CARD_FILLS`; false without).
    pub(crate) fn shaders_wrote(src: usize, len: usize) -> bool {
        if super::card_fills_mode() == 0 || len == 0 {
            return false;
        }
        let learned = LEARNED.lock().unwrap_or_else(|e| e.into_inner());
        learned.as_ref().is_some_and(|set| (src / PIECE..=(src + len - 1) / PIECE).any(|p| set.contains(&p)))
    }

    /// A fill reads `len` bytes at `src` ahead of flights that may write
    /// them.
    pub(super) fn read(src: usize, len: usize) {
        if len == 0 {
            return;
        }
        COUNT[0].fetch_add(1, Relaxed);
        let frame = crate::frame::current();
        let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
        let map = read.get_or_insert_with(HashMap::new);
        for piece in src / PIECE..=(src + len - 1) / PIECE {
            let p = map.entry(piece).or_default();
            if p.pruned != frame {
                p.pruned = frame;
                p.fills.retain(|f| f.2 + KEEP >= frame);
            }
            p.fills.push((src, src + len, frame, super::STARTED.load(Relaxed)));
        }
    }

    /// A landing is about to put `new` into title memory at `at`: bytes
    /// shaders wrote, merged into the title's.
    pub(crate) fn wrote(at: usize, new: &[u8]) {
        note_written(at, new.len());
        if new.is_empty() || WAIT.load(Relaxed) {
            return;
        }
        let frame = crate::frame::current();
        let landing = LANDING.load(Relaxed);
        let hits: Vec<(usize, usize, u64)> = {
            let read = READ.lock().unwrap_or_else(|e| e.into_inner());
            let Some(map) = read.as_ref() else { return };
            let end = at + new.len();
            (at / PIECE..=(end - 1) / PIECE)
                .filter_map(|piece| map.get(&piece))
                .flat_map(|p| p.fills.iter().copied())
                .filter(|f| f.0 < end && f.1 > at && f.2 + KEEP >= frame && (landing == 0 || landing <= f.3))
                .map(|f| (f.0, f.1, f.2))
                .collect()
        };
        if hits.is_empty() {
            return;
        }
        COUNT[1].fetch_add(1, Relaxed);
        let mut now = Vec::new();
        for (from, to, read_in) in hits {
            let (a, b) = (from.max(at), to.min(at + new.len()));
            now.clear();
            now.resize(b - a, 0);
            // Title memory as it is, which is what the fill read unless the
            // title wrote it since; not there: nothing to say.
            if !super::read_into(a as *const u8, &mut now) || now[..] == new[a - at..b - at] {
                continue;
            }
            let n = COUNT[2].fetch_add(1, Relaxed) + 1;
            let first = now.iter().zip(&new[a - at..b - at]).position(|(x, y)| x != y).unwrap_or(0);
            if !WAIT.swap(true, Relaxed) {
                tracing::warn!(
                    "a fill of upload memory read {} bytes at {from:#x} in frame {read_in} ahead of a flight whose shaders changed \
                     them (first at {:#x}, landed in frame {frame}): the lists of that submission drew with the bytes from before. \
                     Fills wait for flights again from here on, as with VICTORY_FILL_WAIT=1 ({n} such fills)",
                    to - from,
                    a + first
                );
            }
            return;
        }
    }

    /// With the frame line.
    pub(super) fn line(frames: f64) {
        let [fills, copies, bytes] = [0, 1, 2].map(|i| super::CARD_FILLS[i].swap(0, Relaxed));
        if super::card_fills_mode() != 0 {
            let learned = LEARNED.lock().unwrap_or_else(|e| e.into_inner()).as_ref().map_or(0, |s| s.len());
            tracing::info!(
                "copies on the card where shaders wrote (VICTORY_CARD_FILLS), per frame: {:.1} constant buffers, {:.1} title \
                 buffer copies, {:.3} MB; {learned} pieces of 64 KiB known written by shaders",
                fills as f64 / frames,
                copies as f64 / frames,
                bytes as f64 / (1 << 20) as f64 / frames
            );
        }
        let [ahead, looked] = [0, 1].map(|i| COUNT[i].swap(0, Relaxed));
        if ahead == 0 && looked == 0 {
            return;
        }
        let kept = {
            let frame = crate::frame::current();
            let mut read = READ.lock().unwrap_or_else(|e| e.into_inner());
            read.as_mut().map_or(0, |map| {
                map.retain(|_, p| {
                    p.fills.retain(|f| f.2 + KEEP >= frame);
                    !p.fills.is_empty()
                });
                map.values().map(|p| p.fills.len()).sum::<usize>()
            })
        };
        tracing::info!(
            "fills of upload memory that read title memory flights may write, not waited for (VICTORY_FILL_WAIT=1 waits): {:.1} per \
             frame; {:.1} landings' writes per frame fell in the same 64 KiB as one and were compared with it; fills that read \
             bytes a flight then changed, since the start: {}{}; {kept} fills kept to check",
            ahead as f64 / frames,
            looked as f64 / frames,
            COUNT[2].load(Relaxed),
            if WAIT.load(Relaxed) { " (fills wait again)" } else { "" }
        );
    }
