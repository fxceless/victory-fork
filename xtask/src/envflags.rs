//! The one shape of environment read this tree refuses: asking whether a
//! switch is *set* when the question is whether it is *on*.
//!
//! `std::env::var_os(..).is_some()` answers "was this set at all". An operator
//! who writes `VICTORY_THING=0` to turn the thing off gets it turned on, and
//! the cost lands somewhere unrecognisable -- the D3D12 debug layer was
//! enabled that way and cost hundreds of milliseconds a frame, and
//! `VICTORY_GPU_TIMING` disabled `VICTORY_LATE_WAIT` and `VICTORY_RELAY` from a
//! release build that could not even measure the timing it was asked for.
//!
//! Pull request #166 fixed eight of these in `victory-gpu`. There were
//! nineteen: nine more in `victory-shader` and one in `victory-hle`, including
//! a second reader of `VICTORY_D3D12_DEBUG` that turned the debug layer back
//! on after the first had been fixed. That is why this is a scan over the tree
//! and not eight more edits: a test of the parser proves the parser works, and
//! proves nothing about whether any call site uses it.
//!
//! The rule is the shape, not a list of names: any `var_os(..)` whose result is
//! immediately tested for presence is asking the wrong question, whatever the
//! variable is called. Switches that carry a *value* (`VICTORY_LATE_WAIT=3`)
//! are parsed, not tested for presence, and are untouched by this.

use std::path::{Path, PathBuf};

/// A read that asks whether a variable is set instead of whether it is on.
pub struct PresenceRead {
    /// The file it was found in, relative to the workspace root.
    pub file: PathBuf,
    /// 1-based line.
    pub line: usize,
    /// The variable named, as written.
    pub name: String,
}

/// Every `.rs` file under `root` that reads an environment variable and then
/// asks only whether it was set.
///
/// Written against the text rather than a regex so that spaces between the
/// call and the test cannot slip past a pattern that only matches when they
/// are adjacent -- the failure mode this file exists to stop is a check that
/// is structurally unable to see the thing it polices.
pub fn presence_only_reads(root: &Path) -> Result<Vec<PresenceRead>, String> {
    Ok(scan(root, tracked_rust_files(root)?))
}

/// The files a workspace guard is about: the ones **git tracks**, and therefore
/// the ones this repository ships and compiles.
///
/// Derived from `git ls-files` rather than by walking the directory. A walk sees
/// every copy of the tree that happens to be on disk -- `.worktrees/kee-preflight`
/// while a preflight is running, a vendored checkout, `target/` -- and reports each
/// duplicate as a defect in code nobody changed. The walk this replaces carried a
/// skip list (`target`, `.git`, `node_modules`) and missed `.worktrees` the first
/// day that directory existed, which is what every hand-kept list eventually does:
/// names a set it can remember instead of the rule it means.
///
/// `Err` when git cannot answer. A guard that quietly scans nothing when its input
/// is unavailable fails open, which is the one direction a guard must never fail.
fn tracked_rust_files(root: &Path) -> Result<Vec<PathBuf>, String> {
    let out = std::process::Command::new("git")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-z", "--", "*.rs"])
        .output()
        .map_err(|e| format!("cannot run git to list the files this guard is about: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git ls-files failed in {}: {}",
            root.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .split('\0')
        .filter(|p| !p.is_empty())
        .map(|p| root.join(p))
        .collect())
}

/// Scans the given files for a switch read for its presence only.
pub(crate) fn scan(root: &Path, files: impl IntoIterator<Item = PathBuf>) -> Vec<PresenceRead> {
    let mut found = Vec::new();
    for path in files {
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        for (i, line) in text.lines().enumerate() {
            let mut from = 0;
            while let Some(at) = line[from..].find("var_os(") {
                let start = from + at + "var_os(".len();
                let rest = &line[start..];
                // The quoted name, if the argument is a literal.
                let Some(open) = rest.find('"') else { break };
                let Some(close) = rest[open + 1..].find('"').map(|c| c + open + 1) else {
                    break;
                };
                let name = rest[open + 1..close].to_string();
                // What follows the call: `).is_some()` asks presence only.
                let after = rest[close + 1..].trim_start();
                if after.starts_with(')') && after[1..].trim_start().starts_with(".is_some()") {
                    found.push(PresenceRead {
                        file: path.strip_prefix(root).unwrap_or(&path).to_path_buf(),
                        line: i + 1,
                        name,
                    });
                }
                from = start;
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn workspace_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("xtask lives in the workspace")
            .to_path_buf()
    }

    /// The finding: no environment switch in this tree is read for its
    /// presence. If this fails, the named file and line ask "was it set" about
    /// a switch whose whole purpose is to be turned off.
    #[test]
    fn nothing_reads_a_switch_for_its_presence_only() {
        let found = presence_only_reads(&workspace_root())
            .expect("the guard must be able to list the files it is about");
        assert!(
            found.is_empty(),
            "these read an environment variable for its presence, so setting it to 0 \
             turns it ON -- route them through `env_flag`:\n{}",
            found
                .iter()
                .map(|f| format!("  {}:{} {}", f.file.display(), f.line, f.name))
                .collect::<Vec<_>>()
                .join("\n")
        );
    }

    /// A guard is the one kind of code that fails silently in the direction of
    /// "everything is fine", so the scanner is fed the exact shape it exists to
    /// catch -- with and without spaces, which a naive pattern would miss --
    /// and the shapes that must keep passing.
    ///
    /// The fixture is assembled at run time so that this file, which the
    /// workspace scan also reads, does not itself contain the shape. An
    /// exclusion list for it would be a hand-kept set, and the next file to
    /// need one would not get it.
    #[test]
    fn the_scanner_finds_the_shape_and_leaves_the_others_alone() {
        let dir = std::env::temp_dir().join(format!("victory-envflag-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        let call = ["std::env::var_o", "s(\""].concat();
        let src = format!(
            "fn a() -> bool {{ {call}VICTORY_A\").is_some() }}\n\
             fn b() -> bool {{ {call}VICTORY_B\")  .is_some() }}\n\
             fn c() -> bool {{ {call}VICTORY_C\").is_none() }}\n\
             fn d() -> bool {{ crate::env::env_flag(\"VICTORY_D\") }}\n\
             fn e() -> Option<String> {{ std::env::var_os(\"VICTORY_E\").map(|v| v.to_string_lossy().into_owned()) }}\n\
             fn f() -> Option<String> {{ std::env::var(\"VICTORY_F\").ok() }}\n"
        );
        std::fs::write(dir.join("src/bad.rs"), src).unwrap();
        let found = scan(&dir, [dir.join("src/bad.rs")]);
        let names: Vec<&str> = found.iter().map(|f| f.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["VICTORY_A", "VICTORY_B"],
            "the two presence reads, and not the three that ask something else"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
