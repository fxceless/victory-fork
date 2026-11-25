use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};

use crate::Converter;

/// Identifies one conversion result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CacheKey(blake3::Hash);

impl CacheKey {
    pub fn new(converter: &dyn Converter, input: &[u8]) -> Self {
        let mut h = blake3::Hasher::new();
        h.update(converter.id().as_bytes());
        h.update(&[0]);
        h.update(&converter.version().to_le_bytes());
        h.update(input);
        CacheKey(h.finalize())
    }

    pub fn to_hex(&self) -> String {
        self.0.to_hex().to_string()
    }
}

/// One result: its bytes, and whether they were made now (a miss).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Output {
    pub bytes: Vec<u8>,
    pub converted: bool,
}

/// Content-addressed store under the user's cache directory. Entries are
/// compressed, and checked when read. With a limit the store stays below it:
/// the entries used longest ago go first.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
    limit: Option<u64>,
    shared: Arc<Shared>,
}

#[derive(Debug, Default)]
struct Shared {
    /// What the entries take on disk, as far as known: counted by a trim,
    /// and added to by every entry written since.
    used: AtomicU64,
    trimming: AtomicBool,
}

/// An entry's file: this, the length of what it holds, the first eight bytes
/// of its BLAKE3, then a zlib stream.
const MAGIC: &[u8; 4] = b"VCE1";
const HEADER: usize = 4 + 8 + 8;
/// No entry is larger; a damaged length must not become an allocation.
const MAX_ENTRY: u64 = 1 << 30;

/// How old an entry's time of use may be before a read renews it: a trim
/// only needs to tell runs apart, and a write for every read would cost more
/// than the read.
const RENEW_AFTER: Duration = Duration::from_secs(3600);
/// Staging files older than this belong to no writer any more.
const STALE_STAGING: Duration = Duration::from_secs(3600);
/// What a file takes on disk is whole clusters.
const CLUSTER: u64 = 4096;

fn on_disk(len: u64) -> u64 {
    len.max(1).next_multiple_of(CLUSTER)
}

fn checksum(bytes: &[u8]) -> [u8; 8] {
    let mut out = [0; 8];
    out.copy_from_slice(&blake3::hash(bytes).as_bytes()[..8]);
    out
}

/// What an entry's file holds. Level 6: translated shaders come out 2.2
/// times smaller (1.9 at level 1), in a few ms against the 10 to 400 that
/// making one takes.
fn pack(bytes: &[u8]) -> Option<Vec<u8>> {
    let mut out = vec![0u8; HEADER + zlib_rs::compress_bound(bytes.len())];
    out[..4].copy_from_slice(MAGIC);
    out[4..12].copy_from_slice(&(bytes.len() as u64).to_le_bytes());
    out[12..HEADER].copy_from_slice(&checksum(bytes));
    let (packed, status) =
        zlib_rs::compress_slice(&mut out[HEADER..], bytes, zlib_rs::DeflateConfig::new(6));
    let len = HEADER + packed.len();
    (status == zlib_rs::ReturnCode::Ok).then(|| {
        out.truncate(len);
        out
    })
}

/// The reverse of [`pack`]; `None` for anything else, whatever its bytes.
fn unpack(file: &[u8]) -> Option<Vec<u8>> {
    let (header, stream) = file.split_at_checked(HEADER)?;
    if &header[..4] != MAGIC {
        return None;
    }
    let len = u64::from_le_bytes(header[4..12].try_into().ok()?);
    if len > MAX_ENTRY {
        return None;
    }
    let mut out = vec![0u8; len as usize];
    let (bytes, status) =
        zlib_rs::decompress_slice(&mut out, stream, zlib_rs::InflateConfig::default());
    let whole = status == zlib_rs::ReturnCode::Ok
        && bytes.len() as u64 == len
        && checksum(bytes) == header[12..];
    whole.then_some(out)
}

impl Cache {
    /// Opens (creating) a cache at `root`. Refuses locations inside the repo.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        Self::new(root.into(), None)
    }

    /// As [`Cache::open`]; the entries take about `max_bytes` on disk at
    /// most. What is already over it is let go in the background.
    pub fn open_limited(root: impl Into<PathBuf>, max_bytes: u64) -> Result<Self> {
        let cache = Self::new(root.into(), Some(max_bytes))?;
        cache.trim_behind();
        Ok(cache)
    }

    fn new(root: PathBuf, limit: Option<u64>) -> Result<Self> {
        victory_core::ensure_outside_repo(&root)?;
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        Ok(Cache {
            root,
            limit,
            shared: Arc::default(),
        })
    }

    pub fn path_for(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.root.join(&hex[..2]).join(&hex[2..])
    }

    /// The entry for `key`. A damaged one is removed and counts as none.
    pub fn get(&self, key: &CacheKey) -> Option<Vec<u8>> {
        let path = self.path_for(key);
        let file = std::fs::read(&path).ok()?;
        let Some(bytes) = unpack(&file) else {
            tracing::debug!(path = %path.display(), "damaged cache entry removed");
            let _ = std::fs::remove_file(&path);
            return None;
        };
        renew(&path);
        Some(bytes)
    }

    /// Removes the entry for `key`, if there is one.
    pub fn remove(&self, key: &CacheKey) {
        let _ = std::fs::remove_file(self.path_for(key));
    }

    /// Returns the output for `input` from the cache, or converts it and
    /// keeps the result. A result that cannot be written (a full disk, a
    /// scanner holding the file) is returned all the same: only a failed
    /// conversion is an error.
    pub fn get_or_convert(&self, converter: &dyn Converter, input: &[u8]) -> Result<Output> {
        let key = CacheKey::new(converter, input);
        if let Some(bytes) = self.get(&key) {
            tracing::trace!(converter = converter.id(), key = %key.to_hex(), "cache hit");
            return Ok(Output {
                bytes,
                converted: false,
            });
        }
        let bytes = converter
            .convert(input)
            .with_context(|| format!("converter {} v{}", converter.id(), converter.version()))?;
        if let Err(e) = self.put(&key, &bytes) {
            static WARNED: AtomicBool = AtomicBool::new(false);
            if !WARNED.swap(true, Ordering::Relaxed) {
                tracing::warn!(
                    error = %e,
                    root = %self.root.display(),
                    "a cache entry could not be written: its result is used but not kept (said once)"
                );
            }
        }
        Ok(Output {
            bytes,
            converted: true,
        })
    }

    fn put(&self, key: &CacheKey, bytes: &[u8]) -> Result<()> {
        let path = self.path_for(key);
        let packed = pack(bytes).context("compressing")?;
        let dir = path.parent().expect("hashed path has a parent");
        std::fs::create_dir_all(dir)?;
        // Write-then-rename so readers never see partial output.
        let tmp = tempfile_in(dir);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        let written = file.write_all(&packed);
        drop(file);
        let published = written.and_then(|()| std::fs::rename(&tmp, &path));
        if published.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        published?;
        let size = on_disk(packed.len() as u64);
        let used = self.shared.used.fetch_add(size, Ordering::Relaxed) + size;
        if self.limit.is_some_and(|limit| used > limit) {
            self.trim_behind();
        }
        Ok(())
    }

    /// [`Cache::trim`] on a thread of its own, unless one is at it.
    fn trim_behind(&self) {
        if self.limit.is_none() || self.shared.trimming.swap(true, Ordering::AcqRel) {
            return;
        }
        let cache = self.clone();
        let work = move || {
            cache.trim();
            cache.shared.trimming.store(false, Ordering::Release);
        };
        let started = std::thread::Builder::new()
            .name("victory-cache-trim".into())
            .spawn(work);
        if started.is_err() {
            self.shared.trimming.store(false, Ordering::Release);
        }
    }

    /// Counts what the entries take. Over the limit, removes those used
    /// longest ago until a tenth of it is free. Also removes the staging
    /// files a crash left. Returns the bytes left.
    pub fn trim(&self) -> u64 {
        let now = SystemTime::now();
        let age = |at: SystemTime| now.duration_since(at).unwrap_or_default();
        let (mut entries, mut total) = (Vec::new(), 0u64);
        for dir in std::fs::read_dir(&self.root).into_iter().flatten().flatten() {
            for file in std::fs::read_dir(dir.path()).into_iter().flatten().flatten() {
                let Ok(meta) = file.metadata() else { continue };
                if !meta.is_file() {
                    continue;
                }
                let used_at = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
                if file.file_name().to_string_lossy().starts_with(".tmp-") {
                    if age(used_at) > STALE_STAGING {
                        let _ = std::fs::remove_file(file.path());
                    }
                    continue;
                }
                let size = on_disk(meta.len());
                total += size;
                entries.push((used_at, size, file.path()));
            }
        }
        if let Some(limit) = self.limit
            && total > limit
        {
            let (before, target) = (total, limit / 10 * 9);
            entries.sort_unstable_by_key(|e| e.0);
            let mut removed = 0usize;
            for (_, size, path) in &entries {
                if total <= target {
                    break;
                }
                // Read at this moment by another thread or run: it goes on
                // reading what it opened.
                if std::fs::remove_file(path).is_ok() {
                    total -= size;
                    removed += 1;
                }
            }
            tracing::info!(
                root = %self.root.display(),
                limit_mb = limit >> 20,
                before_mb = before >> 20,
                after_mb = total >> 20,
                removed,
                "cache over its limit: the entries used longest ago were removed"
            );
        }
        self.shared.used.store(total, Ordering::Relaxed);
        total
    }
}

/// Notes a use of the file at `path`, by its time of change.
fn renew(path: &Path) {
    let now = SystemTime::now();
    let old = std::fs::metadata(path)
        .and_then(|m| m.modified())
        .is_ok_and(|at| now.duration_since(at).unwrap_or_default() > RENEW_AFTER);
    if old && let Ok(file) = std::fs::OpenOptions::new().write(true).open(path) {
        let _ = file.set_modified(now);
    }
}

fn tempfile_in(dir: &Path) -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    dir.join(format!(".tmp-{}-{n}", std::process::id()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    struct Upper {
        version: u32,
        runs: AtomicUsize,
    }

    impl Upper {
        fn new(version: u32) -> Upper {
            Upper {
                version,
                runs: AtomicUsize::new(0),
            }
        }
    }

    impl Converter for Upper {
        fn id(&self) -> &'static str {
            "test.upper"
        }
        fn version(&self) -> u32 {
            self.version
        }
        fn convert(&self, input: &[u8]) -> Result<Vec<u8>> {
            self.runs.fetch_add(1, Ordering::Relaxed);
            Ok(input.to_ascii_uppercase())
        }
    }

    /// Bytes that do not compress, so that sizes on disk are known.
    fn noise(seed: u32, len: usize) -> Vec<u8> {
        let mut h = blake3::Hasher::new();
        h.update(&seed.to_le_bytes());
        let mut out = vec![0; len];
        h.finalize_xof().fill(&mut out);
        out
    }

    fn age(path: &Path, by: Duration) {
        let file = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        file.set_modified(SystemTime::now() - by).unwrap();
    }

    #[test]
    fn converts_once_then_hits() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper::new(1);
        let a = cache.get_or_convert(&conv, b"abc").unwrap();
        let b = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!((&a.bytes[..], a.converted), (&b"ABC"[..], true));
        assert_eq!((&b.bytes[..], b.converted), (&b"ABC"[..], false));
        assert_eq!(conv.runs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn entries_are_compressed_and_every_damage_is_a_miss() {
        let raw: Vec<u8> = (0..100_000u32).flat_map(|i| (i / 7).to_le_bytes()).collect();
        let packed = pack(&raw).unwrap();
        assert!(packed.len() * 2 < raw.len(), "{} of {}", packed.len(), raw.len());
        assert_eq!(unpack(&packed).as_deref(), Some(&raw[..]));
        assert_eq!(unpack(&pack(b"").unwrap()).as_deref(), Some(&b""[..]));
        assert_eq!(unpack(b""), None);
        assert_eq!(unpack(b"VSH1 an entry of an older build, not packed"), None);
        for at in [0, 5, 13, HEADER + 1, packed.len() / 2, packed.len() - 1] {
            let mut bad = packed.clone();
            bad[at] ^= 0x10;
            assert_eq!(unpack(&bad), None, "byte {at} changed");
            assert_eq!(unpack(&packed[..at]), None, "cut at {at}");
        }
        // What a crash leaves of a file whose length was written first.
        let mut zeros = packed.clone();
        zeros[HEADER + 100..].fill(0);
        assert_eq!(unpack(&zeros), None);
        // A length that is not a length.
        let mut huge = packed;
        huge[4..12].fill(0xff);
        assert_eq!(unpack(&huge), None);
    }

    #[test]
    fn a_damaged_entry_is_converted_again() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper::new(1);
        cache.get_or_convert(&conv, b"abc").unwrap();
        let path = cache.path_for(&CacheKey::new(&conv, b"abc"));
        std::fs::write(&path, b"broken").unwrap();
        let again = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!((&again.bytes[..], again.converted), (&b"ABC"[..], true));
        assert!(!cache.get_or_convert(&conv, b"abc").unwrap().converted);
    }

    /// A result that cannot be kept is still a result: it was an error, and
    /// the draws of a shader translated while the disk was full were dropped
    /// for the rest of the run.
    #[test]
    fn a_result_that_cannot_be_written_is_returned() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper::new(1);
        let path = cache.path_for(&CacheKey::new(&conv, b"abc"));
        std::fs::create_dir_all(&path).unwrap(); // a folder is in its place
        let out = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!((&out.bytes[..], out.converted), (&b"ABC"[..], true));
        assert!(path.is_dir());
        // Its staging file is gone.
        let files = std::fs::read_dir(path.parent().unwrap()).unwrap().count();
        assert_eq!(files, 1);
    }

    #[test]
    fn a_limited_cache_lets_go_of_what_was_used_longest_ago() {
        const SIZE: usize = 40_000; // ten clusters with its header
        const EACH: u64 = 10 * CLUSTER;
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open_limited(tmp.path().join("cache"), 10 * EACH).unwrap();
        let settle = || {
            while cache.shared.trimming.load(Ordering::Acquire) {
                std::thread::yield_now();
            }
        };
        settle();
        let conv = Upper::new(1);
        let paths: Vec<PathBuf> = (0..9u32)
            .map(|i| {
                let input = noise(i, SIZE);
                let key = CacheKey::new(&conv, &input);
                cache.put(&key, &input).unwrap();
                let path = cache.path_for(&key);
                // The first is the oldest.
                age(&path, Duration::from_secs(u64::from(100 - i) * 7200));
                path
            })
            .collect();
        assert_eq!(cache.trim(), 9 * EACH, "under the limit: nothing goes");
        assert!(paths.iter().all(|p| p.exists()));

        // The oldest one is used: it is the newest now.
        assert!(cache.get(&CacheKey::new(&conv, &noise(0, SIZE))).is_some());
        for i in 9..12u32 {
            let input = noise(i, SIZE);
            cache.put(&CacheKey::new(&conv, &input), &input).unwrap();
        }
        // Going over the limit trims by itself; another writer's entry may
        // come after, so the last word is a trim of ours.
        settle();
        assert!(cache.trim() <= 10 * EACH);
        assert!(paths[0].exists(), "used just now");
        assert!(!paths[1].exists(), "used longest ago");
        assert!(paths[8].exists());
    }

    #[test]
    fn staging_files_a_crash_left_are_removed() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let dir = tmp.path().join("cache").join("ab");
        std::fs::create_dir_all(&dir).unwrap();
        let (old, new) = (dir.join(".tmp-1-0"), dir.join(".tmp-2-0"));
        std::fs::write(&old, b"x").unwrap();
        std::fs::write(&new, b"x").unwrap();
        age(&old, STALE_STAGING * 2);
        assert_eq!(cache.trim(), 0);
        assert!(!old.exists());
        assert!(new.exists(), "a writer may still be at it");
    }

    #[test]
    fn version_bump_invalidates() {
        assert_ne!(
            CacheKey::new(&Upper::new(1), b"x"),
            CacheKey::new(&Upper::new(2), b"x")
        );
    }

    #[test]
    fn refuses_cache_inside_repo() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        assert!(Cache::open(inside).is_err());
    }
}
