//! Rule #1, enforced at runtime: game-derived data never lands in the repo.
//!
//! Everything VIctory produces from the user's files (converted assets,
//! prepared images, shader caches, crash dumps) is as much "from the game"
//! as the files themselves. Every tool that writes such output must call
//! [`ensure_outside_repo`] on the destination first. The static half of the
//! rule is `cargo xtask policy`, which checks what git tracks.

use std::ffi::OsString;
use std::path::{Component, Path, PathBuf};

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

enum Tail {
    Name(OsString),
    Parent,
}

/// Canonicalizes the longest existing prefix of `path` and re-appends the
/// rest, so that paths which do not exist yet can still be compared.
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

    // Pass 1 peels non-existent trailing components (including `..`) down to
    // the longest existing prefix and folds `..` onto the canonical prefix.
    // Pass 2 re-canonicalizes any existing directory or symlink/junction
    // revealed by folding `missing/..` (e.g. `<outside>/missing/../<repo>/x`).
    let folded = resolve_existing_prefix(&absolute, path)?;
    resolve_existing_prefix(&folded, path)
}

fn resolve_existing_prefix(target: &Path, original: &Path) -> Result<PathBuf, PolicyError> {
    let mut existing = target;
    let mut rest = Vec::new();
    loop {
        match std::fs::canonicalize(existing) {
            Ok(mut out) => {
                for segment in rest.into_iter().rev() {
                    match segment {
                        Tail::Parent => {
                            out.pop();
                        }
                        Tail::Name(name) => out.push(name),
                    }
                }
                return Ok(out);
            }
            Err(source) => {
                let Some(parent) = existing.parent() else {
                    return Err(PolicyError::Io {
                        path: original.to_path_buf(),
                        source,
                    });
                };
                match existing.components().next_back() {
                    Some(Component::Normal(name)) => rest.push(Tail::Name(name.to_os_string())),
                    Some(Component::ParentDir) => rest.push(Tail::Parent),
                    Some(Component::CurDir) => {}
                    _ => {
                        return Err(PolicyError::Io {
                            path: original.to_path_buf(),
                            source,
                        });
                    }
                }
                existing = parent;
            }
        }
    }
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
    fn resolves_parent_dir_after_missing_segments() {
        let tmp = tempfile::tempdir().unwrap();
        let outside = tmp.path().join("missing").join("..").join("valid-cache");
        ensure_outside_repo(&outside).unwrap();

        let root = std::fs::canonicalize(compiled_workspace_root()).unwrap();
        let repo_inside = compiled_workspace_root()
            .join("missing")
            .join("..")
            .join("cache");
        assert!(matches!(
            ensure_outside_repo(&repo_inside),
            Err(PolicyError::InsideRepo { .. })
        ));

        if let (Some(parent), Some(name)) = (compiled_workspace_root().parent(), root.file_name()) {
            let sibling_back_in = parent.join("missing").join("..").join(name).join("cache");
            assert!(matches!(
                ensure_outside_repo(&sibling_back_in),
                Err(PolicyError::InsideRepo { .. })
            ));
        }

        // A Windows junction reached via `missing/..` must also be canonicalized
        // and rejected when it targets a repository tree.
        #[cfg(windows)]
        {
            let other_repo = tmp.path().join("other-repo");
            std::fs::create_dir(&other_repo).unwrap();
            std::fs::write(other_repo.join(ROOT_MARKER), "").unwrap();
            let link = tmp.path().join("link-to-repo");
            let status = std::process::Command::new("cmd")
                .args(["/C", "mklink", "/J"])
                .arg(&link)
                .arg(&other_repo)
                .status();
            if status.is_ok_and(|s| s.success()) {
                let via_missing = tmp
                    .path()
                    .join("missing")
                    .join("..")
                    .join("link-to-repo")
                    .join("cache");
                let err = ensure_outside_repo(&via_missing);
                let _ = std::fs::remove_dir(&link);
                assert!(matches!(err, Err(PolicyError::InsideRepo { .. })));
            }
        }
    }
}
