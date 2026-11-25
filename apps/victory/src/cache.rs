//! What the caches take on disk, and emptying them.

use std::path::Path;

use anyhow::{Context, Result};

/// The folders `victory cache --clear` empties: what a run makes again by
/// itself. Saves and settings are elsewhere (the data folder), and the
/// prepared images are one per build of a title.
const CLEARED: [&str; 2] = ["shaders", "pipelines"];

/// The files below `root` and what they take on disk (whole clusters).
fn usage(root: &Path) -> (usize, u64) {
    walkdir::WalkDir::new(root)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| e.metadata().ok())
        .fold((0, 0), |(n, bytes), m| {
            (n + 1, bytes + m.len().next_multiple_of(4096))
        })
}

pub fn run(clear: bool) -> Result<()> {
    let cfg = victory_core::Config::load()?;
    victory_core::ensure_outside_repo(&cfg.cache_dir)?;
    if clear {
        clear_in(&cfg.cache_dir)?;
    }
    println!("cache_dir   : {}", cfg.cache_dir.display());
    let mut limited = 0;
    for name in ["shaders", "pipelines", "images"] {
        let (files, bytes) = usage(&cfg.cache_dir.join(name));
        println!("{name:<12}: {:>8.1} MB in {files} files", bytes as f64 / 1048576.0);
        if CLEARED.contains(&name) {
            limited += bytes;
        }
    }
    println!(
        "limit       : {:>8.1} MB of {} MB (shaders and pipelines; VICTORY_CACHE_MAX_MB, or max_mb under [cache])",
        limited as f64 / 1048576.0,
        cfg.cache_max_bytes >> 20
    );
    Ok(())
}

fn clear_in(cache_dir: &Path) -> Result<()> {
    for name in CLEARED {
        let dir = cache_dir.join(name);
        // A link would take the removal somewhere else.
        match std::fs::symlink_metadata(&dir) {
            Ok(m) if m.file_type().is_symlink() => {
                anyhow::bail!("{} is a link; not emptied", dir.display())
            }
            Ok(_) => std::fs::remove_dir_all(&dir).with_context(|| {
                format!("removing {} (is the title still running?)", dir.display())
            })?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clearing_takes_the_caches_and_nothing_else() {
        let tmp = tempfile::tempdir().unwrap();
        for name in ["shaders/ab", "pipelines", "images", "other"] {
            std::fs::create_dir_all(tmp.path().join(name)).unwrap();
            std::fs::write(tmp.path().join(name).join("f"), [1u8; 5000]).unwrap();
        }
        assert_eq!(usage(tmp.path()), (4, 4 * 8192));
        clear_in(tmp.path()).unwrap();
        assert_eq!(usage(&tmp.path().join("shaders")), (0, 0));
        assert_eq!(usage(&tmp.path().join("pipelines")), (0, 0));
        assert_eq!(usage(&tmp.path().join("images")), (1, 8192));
        assert_eq!(usage(&tmp.path().join("other")), (1, 8192));
        clear_in(tmp.path()).unwrap(); // nothing there: nothing to do
    }
}
