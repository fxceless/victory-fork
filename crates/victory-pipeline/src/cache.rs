use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};

use crate::Converter;

/// Maximum number of staging file names to try when skipping stale `.tmp-*` collisions.
const MAX_STAGING_ATTEMPTS: usize = 1024;

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

/// Content-addressed store under the user's cache directory.
#[derive(Debug, Clone)]
pub struct Cache {
    root: PathBuf,
}

impl Cache {
    /// Opens (creating) a cache at `root`. Refuses locations inside the repo.
    pub fn open(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        victory_core::ensure_outside_repo(&root)?;
        std::fs::create_dir_all(&root).with_context(|| format!("creating {}", root.display()))?;
        Ok(Cache { root })
    }

    pub fn path_for(&self, key: &CacheKey) -> PathBuf {
        let hex = key.to_hex();
        self.root.join(&hex[..2]).join(&hex[2..])
    }

    /// Returns the cached output for `input`, converting it first if needed.
    pub fn get_or_convert(&self, converter: &dyn Converter, input: &[u8]) -> Result<PathBuf> {
        let key = CacheKey::new(converter, input);
        let path = self.path_for(&key);
        if path.is_file() {
            tracing::trace!(converter = converter.id(), key = %key.to_hex(), "cache hit");
            return Ok(path);
        }
        let output = converter
            .convert(input)
            .with_context(|| format!("converter {} v{}", converter.id(), converter.version()))?;
        let dir = path.parent().expect("hashed path has a parent");
        std::fs::create_dir_all(dir)?;
        // Write-then-rename so readers never see partial output.
        let (tmp, mut file) = create_staging_file(dir)?;
        let guard = StagingFile(Some(tmp));
        let written = file.write_all(&output);
        // Close the file handle before renaming or removing on Windows.
        drop(file);
        written?;
        publish_staging(guard, &path, |from, to| std::fs::rename(from, to))?;
        Ok(path)
    }
}

struct StagingFile(Option<PathBuf>);

impl StagingFile {
    fn path(&self) -> &Path {
        self.0.as_deref().expect("staging file path present")
    }

    fn disarm(mut self) {
        self.0 = None;
    }
}

impl Drop for StagingFile {
    fn drop(&mut self) {
        if let Some(path) = self.0.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

fn create_staging_file(dir: &Path) -> io::Result<(PathBuf, std::fs::File)> {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let pid = std::process::id();
    for _ in 0..MAX_STAGING_ATTEMPTS {
        let n = COUNTER.fetch_add(1, Ordering::Relaxed);
        let tmp = dir.join(format!(".tmp-{pid}-{n}"));
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&tmp)
        {
            Ok(file) => return Ok((tmp, file)),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!(
            "exhausted {MAX_STAGING_ATTEMPTS} staging file names in {}",
            dir.display()
        ),
    ))
}

fn publish_staging(
    guard: StagingFile,
    dst: &Path,
    rename: impl FnOnce(&Path, &Path) -> io::Result<()>,
) -> io::Result<()> {
    match rename(guard.path(), dst) {
        Ok(()) => {
            guard.disarm();
            Ok(())
        }
        // Content-addressed key: if another thread or process published `dst`
        // first and Windows rejected replacing the open target, keep the
        // existing entry and let `guard` clean up our staging file.
        Err(_) if dst.is_file() => Ok(()),
        Err(err) => Err(err),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    struct Upper {
        version: u32,
        runs: AtomicUsize,
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

    #[test]
    fn converts_once_then_hits() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let a = cache.get_or_convert(&conv, b"abc").unwrap();
        let b = cache.get_or_convert(&conv, b"abc").unwrap();
        assert_eq!(a, b);
        assert_eq!(std::fs::read(&a).unwrap(), b"ABC");
        assert_eq!(conv.runs.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn failed_publication_removes_the_staging_file() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let path = cache.path_for(&CacheKey::new(&conv, b"abc"));
        std::fs::create_dir_all(&path).unwrap();

        assert!(cache.get_or_convert(&conv, b"abc").is_err());
        assert!(path.is_dir());
        assert_eq!(
            std::fs::read_dir(path.parent().unwrap()).unwrap().count(),
            1
        );
    }

    #[test]
    fn version_bump_invalidates() {
        let v1 = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let v2 = Upper {
            version: 2,
            runs: AtomicUsize::new(0),
        };
        assert_ne!(CacheKey::new(&v1, b"x"), CacheKey::new(&v2, b"x"));
    }

    #[test]
    fn refuses_cache_inside_repo() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        assert!(Cache::open(inside).is_err());
    }

    #[test]
    fn skips_stale_staging_file_collisions() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = Cache::open(tmp.path().join("cache")).unwrap();
        let conv = Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        };
        let path = cache.path_for(&CacheKey::new(&conv, b"collision"));
        let dir = path.parent().unwrap();
        std::fs::create_dir_all(dir).unwrap();
        let pid = std::process::id();
        for n in 0..64u64 {
            std::fs::write(dir.join(format!(".tmp-{pid}-{n}")), b"stale").unwrap();
        }

        let out = cache.get_or_convert(&conv, b"collision").unwrap();
        assert_eq!(out, path);
        assert_eq!(std::fs::read(&out).unwrap(), b"COLLISION");
    }

    #[test]
    fn concurrent_publish_recovers_when_rename_loses_race() {
        let tmp = tempfile::tempdir().unwrap();
        let dst = tmp.path().join("published");
        let staging = tmp.path().join(".tmp-race");
        std::fs::write(&dst, b"WINNER").unwrap();
        std::fs::write(&staging, b"LOSER").unwrap();

        publish_staging(StagingFile(Some(staging.clone())), &dst, |_, _| {
            Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "target held open by concurrent reader/writer",
            ))
        })
        .unwrap();
        assert!(!staging.exists(), "loser staging file must be removed");
        assert_eq!(std::fs::read(&dst).unwrap(), b"WINNER");

        // When `dst` is a directory (not a published file), errors still surface
        // and the staging file is removed.
        let dir_dst = tmp.path().join("dir-target");
        std::fs::create_dir_all(&dir_dst).unwrap();
        let staging2 = tmp.path().join(".tmp-dir-fail");
        std::fs::write(&staging2, b"DATA").unwrap();
        assert!(
            publish_staging(StagingFile(Some(staging2.clone())), &dir_dst, |_, _| {
                Err(io::Error::new(
                    io::ErrorKind::PermissionDenied,
                    "cannot overwrite directory",
                ))
            })
            .is_err()
        );
        assert!(!staging2.exists(), "failed staging file must be removed");
    }

    #[test]
    fn concurrent_get_or_convert_leaves_no_staging_files() {
        use std::sync::{Arc, Barrier};

        let tmp = tempfile::tempdir().unwrap();
        let cache = Arc::new(Cache::open(tmp.path().join("cache")).unwrap());
        let conv = Arc::new(Upper {
            version: 1,
            runs: AtomicUsize::new(0),
        });
        let threads = 8;
        let barrier = Arc::new(Barrier::new(threads));
        let mut handles = Vec::with_capacity(threads);
        for _ in 0..threads {
            let cache = Arc::clone(&cache);
            let conv = Arc::clone(&conv);
            let barrier = Arc::clone(&barrier);
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                cache.get_or_convert(conv.as_ref(), b"parallel").unwrap()
            }));
        }
        let expected = cache.path_for(&CacheKey::new(conv.as_ref(), b"parallel"));
        for h in handles {
            assert_eq!(h.join().unwrap(), expected);
        }
        assert_eq!(std::fs::read(&expected).unwrap(), b"PARALLEL");
        let entries: Vec<_> = std::fs::read_dir(expected.parent().unwrap())
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(entries.len(), 1);
    }
}
