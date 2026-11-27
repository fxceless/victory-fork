//! Rule #1, enforced at runtime: game-derived data never lands in the repo.
//!
//! Everything VIctory produces from the user's files (converted assets,
//! prepared images, shader caches, crash dumps) is as much "from the game"
//! as the files themselves. Every tool that writes such output must call
//! [`ensure_outside_repo`] on the destination first. The static half of the
//! rule is `cargo xtask policy`, which checks what git tracks.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::io;
use std::path::{Component, Path, PathBuf};

/// Upper bound on symbolic link expansions while resolving a path.
const MAX_SYMLINK_HOPS: usize = 40;

/// Marker file at the root of the source tree.
pub const ROOT_MARKER: &str = ".victory-root";

/// The source tree this binary was compiled from.
fn compiled_workspace_root() -> PathBuf {
    // crates/victory-core -> workspace root
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
}

#[derive(Debug, thiserror::Error)]
pub enum PolicyError {
    #[error(
        "refusing to use {path} for game-derived data: it is inside the VIctory source tree \
         ({root}). Keep game files, caches and outputs outside the repository \
         (see docs/asset-policy.md)"
    )]
    InsideRepo { path: PathBuf, root: PathBuf },
    #[error("cannot resolve {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
}

/// Returns `Ok` only if `path` is outside every VIctory source tree.
///
/// `path` does not have to exist yet. Two checks are made:
/// 1. it is not below the source tree this binary was built from;
/// 2. none of its ancestors contains the [`ROOT_MARKER`] file (catches
///    other clones/copies of the repository).
pub fn ensure_outside_repo(path: &Path) -> Result<(), PolicyError> {
    let resolved = resolve(path)?;

    if let Ok(root) = std::fs::canonicalize(compiled_workspace_root())
        && root.join(ROOT_MARKER).is_file()
        && resolved.starts_with(&root)
    {
        return Err(PolicyError::InsideRepo {
            path: path.to_path_buf(),
            root,
        });
    }

    for ancestor in resolved.ancestors() {
        if ancestor.join(ROOT_MARKER).is_file() {
            return Err(PolicyError::InsideRepo {
                path: path.to_path_buf(),
                root: ancestor.to_path_buf(),
            });
        }
    }
    Ok(())
}

enum Step {
    Prefix(OsString),
    RootDir,
    ParentDir,
    Normal(OsString),
}

fn push_front_components(queue: &mut VecDeque<Step>, path: &Path) {
    let mut steps = Vec::new();
    for component in path.components() {
        match component {
            Component::Prefix(p) => steps.push(Step::Prefix(p.as_os_str().to_os_string())),
            Component::RootDir => steps.push(Step::RootDir),
            Component::CurDir => {}
            Component::ParentDir => steps.push(Step::ParentDir),
            Component::Normal(s) => steps.push(Step::Normal(s.to_os_string())),
        }
    }
    for step in steps.into_iter().rev() {
        queue.push_front(step);
    }
}

fn reset_for_link_target(current: &mut PathBuf, target: &Path) {
    match target.components().next() {
        Some(Component::Prefix(_)) => current.clear(),
        Some(Component::RootDir) => {
            let prefix = current.components().next().and_then(|c| match c {
                Component::Prefix(p) => Some(p.as_os_str().to_os_string()),
                _ => None,
            });
            current.clear();
            if let Some(p) = prefix {
                current.push(p);
            }
        }
        _ => {}
    }
}

/// Resolves `path` component by component, canonicalizing existing prefixes and
/// expanding symbolic links (including dangling ones) so that paths which do
/// not exist yet still cannot point inside a repository via symlinks or `..`.
fn resolve(path: &Path) -> Result<PathBuf, PolicyError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()
            .map_err(|source| PolicyError::Io {
                path: path.to_path_buf(),
                source,
            })?
            .join(path)
    };

    let mut pending = VecDeque::new();
    push_front_components(&mut pending, &absolute);
    let mut current = PathBuf::new();
    let mut hops = 0usize;

    while let Some(step) = pending.pop_front() {
        match step {
            Step::Prefix(p) => {
                current = PathBuf::from(p);
            }
            Step::RootDir => {
                current.push(Component::RootDir.as_os_str());
                if let Ok(canon) = std::fs::canonicalize(&current) {
                    current = canon;
                }
            }
            Step::ParentDir => {
                current.pop();
            }
            Step::Normal(name) => {
                current.push(&name);
                match std::fs::symlink_metadata(&current) {
                    Ok(meta) if meta.file_type().is_symlink() => {
                        expand_symlink(path, &mut current, &mut pending, &mut hops)?;
                    }
                    Ok(_) => match std::fs::canonicalize(&current) {
                        Ok(canon) => current = canon,
                        // Handle reparse points (e.g. dangling Windows junctions)
                        // where `is_symlink()` may be false but `read_link` succeeds.
                        Err(_) if std::fs::read_link(&current).is_ok() => {
                            expand_symlink(path, &mut current, &mut pending, &mut hops)?;
                        }
                        Err(source) => {
                            return Err(PolicyError::Io {
                                path: path.to_path_buf(),
                                source,
                            });
                        }
                    },
                    Err(err) if err.kind() == io::ErrorKind::NotFound => {}
                    Err(source) => {
                        return Err(PolicyError::Io {
                            path: path.to_path_buf(),
                            source,
                        });
                    }
                }
            }
        }
    }

    Ok(current)
}

fn expand_symlink(
    original: &Path,
    current: &mut PathBuf,
    pending: &mut VecDeque<Step>,
    hops: &mut usize,
) -> Result<(), PolicyError> {
    *hops += 1;
    if *hops > MAX_SYMLINK_HOPS {
        return Err(PolicyError::Io {
            path: original.to_path_buf(),
            source: io::Error::other("too many levels of symbolic links"),
        });
    }
    let target = std::fs::read_link(&*current).map_err(|source| PolicyError::Io {
        path: original.to_path_buf(),
        source,
    })?;
    current.pop();
    reset_for_link_target(current, &target);
    push_front_components(pending, &target);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_paths_inside_this_repo() {
        let inside = compiled_workspace_root().join("cache").join("textures");
        assert!(matches!(
            ensure_outside_repo(&inside),
            Err(PolicyError::InsideRepo { .. })
        ));
        let sneaky = std::env::temp_dir().join("..").join("x"); // sanity: no panic
        let _ = ensure_outside_repo(&sneaky);
    }

    #[test]
    fn rejects_dotdot_escapes_back_into_repo() {
        let root = compiled_workspace_root();
        let path = root
            .join("crates")
            .join("..")
            .join("not-yet-created")
            .join("out");
        assert!(ensure_outside_repo(&path).is_err());
    }

    #[test]
    fn rejects_other_clones_via_marker() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join(ROOT_MARKER), "").unwrap();
        assert!(ensure_outside_repo(&tmp.path().join("cache")).is_err());
    }

    #[test]
    fn accepts_unrelated_directories() {
        let tmp = tempfile::tempdir().unwrap();
        ensure_outside_repo(&tmp.path().join("does").join("not").join("exist")).unwrap();
    }

    #[test]
    fn handles_dotdot_after_nonexistent_component() {
        let tmp = tempfile::tempdir().unwrap();
        // Ending in `..` after a missing component used to fail with PolicyError::Io
        // because Path::file_name() returns None on `..`.
        ensure_outside_repo(&tmp.path().join("missing").join("..")).unwrap();
        ensure_outside_repo(&tmp.path().join("missing").join("..").join("cache")).unwrap();

        // A `..` after a missing component that lands inside a marked tree must
        // still be rejected.
        let repo = tmp.path().join("clone");
        std::fs::create_dir_all(&repo).unwrap();
        std::fs::write(repo.join(ROOT_MARKER), "").unwrap();
        assert!(matches!(
            ensure_outside_repo(&repo.join("missing").join("..")),
            Err(PolicyError::InsideRepo { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_dangling_symlinks_into_repo() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let abs_link = tmp.path().join("to_repo_cache");
        symlink(compiled_workspace_root().join("not-yet-created"), &abs_link).unwrap();
        assert!(matches!(
            ensure_outside_repo(&abs_link),
            Err(PolicyError::InsideRepo { .. })
        ));
        assert!(matches!(
            ensure_outside_repo(&abs_link.join("textures")),
            Err(PolicyError::InsideRepo { .. })
        ));
        assert!(matches!(
            ensure_outside_repo(&abs_link.join("..")),
            Err(PolicyError::InsideRepo { .. })
        ));

        // Relative dangling symlink into another clone marked with `.victory-root`.
        let clone = tmp.path().join("other-clone");
        let outside = tmp.path().join("outside");
        std::fs::create_dir_all(&clone).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(clone.join(ROOT_MARKER), "").unwrap();
        let rel_link = outside.join("rel_cache");
        symlink(Path::new("../other-clone/new-cache"), &rel_link).unwrap();
        assert!(matches!(
            ensure_outside_repo(&rel_link),
            Err(PolicyError::InsideRepo { .. })
        ));
    }

    #[cfg(unix)]
    #[test]
    fn rejects_symlink_loops() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().unwrap();
        let a = tmp.path().join("loop_a");
        let b = tmp.path().join("loop_b");
        symlink(&b, &a).unwrap();
        symlink(&a, &b).unwrap();
        assert!(matches!(
            ensure_outside_repo(&a),
            Err(PolicyError::Io { .. })
        ));
        assert!(matches!(
            ensure_outside_repo(&a.join("child")),
            Err(PolicyError::Io { .. })
        ));
    }
}
