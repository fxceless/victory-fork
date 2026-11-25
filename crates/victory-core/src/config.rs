//! Where the user's files live.
//!
//! Resolution order for each value: environment variable, then the config
//! file (`VICTORY_CONFIG`, else `./victory.toml`), then the default.
//! See `victory.example.toml` at the repository root.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

use crate::policy::ensure_outside_repo;

#[derive(Debug, Default, Clone)]
pub struct Config {
    /// The user's dump of the title.
    pub game_dir: Option<PathBuf>,
    /// Root for all derived data. Always outside the repository.
    pub cache_dir: PathBuf,
    /// What titles write and must keep (persistent local storage). Never
    /// inside the cache, which may be deleted at any time.
    pub data_dir: PathBuf,
    /// Other titles used to exercise platform-level code.
    pub test_titles: Vec<PathBuf>,
    /// The config file that was read, if any.
    pub source: Option<PathBuf>,
    /// The presenting window: frame-rate limit and window mode.
    pub display: Display,
}

/// `[display]` in the config file.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Display {
    /// Presents per second at most; 0 is no limit.
    pub fps_limit: u32,
    /// Start in a borderless window over the monitor (F11 or Alt+Enter
    /// switch it while running).
    pub borderless: bool,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileConfig {
    #[serde(default)]
    paths: FilePaths,
    #[serde(default)]
    display: FileDisplay,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileDisplay {
    fps_limit: Option<u32>,
    /// "windowed" or "borderless".
    window_mode: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct FilePaths {
    game_dir: Option<PathBuf>,
    cache_dir: Option<PathBuf>,
    data_dir: Option<PathBuf>,
    #[serde(default)]
    test_titles: Vec<PathBuf>,
}

impl Config {
    /// Loads configuration from the environment and the config file.
    pub fn load() -> Result<Self> {
        let explicit = std::env::var_os("VICTORY_CONFIG").map(PathBuf::from);
        let path = explicit
            .clone()
            .unwrap_or_else(|| PathBuf::from("victory.toml"));
        let file = if path.is_file() {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            Some((path, text))
        } else if let Some(p) = explicit {
            anyhow::bail!(
                "VICTORY_CONFIG points at {}, which does not exist",
                p.display()
            );
        } else {
            None
        };
        let (source, parsed) = match file {
            Some((path, text)) => {
                let parsed: FileConfig =
                    toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))?;
                (Some(path), parsed)
            }
            None => (None, FileConfig::default()),
        };
        Self::resolve(parsed.paths, parsed.display, source, |k| std::env::var_os(k))
    }

    fn resolve(
        file: FilePaths,
        display: FileDisplay,
        source: Option<PathBuf>,
        env: impl Fn(&str) -> Option<std::ffi::OsString>,
    ) -> Result<Self> {
        let env_path = |k: &str| env(k).filter(|v| !v.is_empty()).map(PathBuf::from);
        let test_titles = match env("VICTORY_TEST_TITLES") {
            Some(v) if !v.is_empty() => std::env::split_paths(&v)
                .filter(|p| !p.as_os_str().is_empty())
                .collect(),
            _ => file.test_titles,
        };
        let cache_dir = env_path("VICTORY_CACHE_DIR")
            .or(file.cache_dir)
            .map_or_else(|| default_dir("cache"), Ok)?;
        let data_dir = env_path("VICTORY_DATA_DIR")
            .or(file.data_dir)
            .map_or_else(|| default_dir("data"), Ok)?;
        let display = resolve_display(display, &env)?;
        Ok(Config {
            game_dir: env_path("VICTORY_GAME_DIR").or(file.game_dir),
            cache_dir,
            data_dir,
            test_titles,
            source,
            display,
        })
    }

    /// The game directory, or a helpful error if it is not configured.
    pub fn require_game_dir(&self) -> Result<&Path> {
        self.game_dir.as_deref().context(
            "no game directory configured: set VICTORY_GAME_DIR or [paths].game_dir in \
             victory.toml (see victory.example.toml)",
        )
    }

    /// Returns `<cache>/<sub>`, creating it, after checking the asset policy.
    pub fn cache_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.cache_dir, sub)
    }

    /// Returns `<data>/<sub>`, creating it, after checking the asset policy.
    pub fn data_subdir(&self, sub: &str) -> Result<PathBuf> {
        subdir(&self.data_dir, sub)
    }
}

/// `VICTORY_FPS_LIMIT` and `VICTORY_WINDOW_MODE` over `[display]`.
fn resolve_display(
    file: FileDisplay,
    env: &impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Result<Display> {
    let text = |k: &str| env(k).filter(|v| !v.is_empty()).map(|v| v.to_string_lossy().into_owned());
    let fps_limit = match text("VICTORY_FPS_LIMIT") {
        Some(v) => v
            .trim()
            .parse::<u32>()
            .ok()
            .filter(|n| *n <= 1000)
            .with_context(|| format!("VICTORY_FPS_LIMIT={v:?}: expected 0 (off) to 1000"))?,
        None => file.fps_limit.unwrap_or(0),
    };
    anyhow::ensure!(fps_limit <= 1000, "[display] fps_limit = {fps_limit}: expected 0 (off) to 1000");
    let (mode, from) = match text("VICTORY_WINDOW_MODE") {
        Some(v) => (Some(v), "VICTORY_WINDOW_MODE"),
        None => (file.window_mode, "[display] window_mode"),
    };
    let borderless = match mode.as_deref().map(str::trim) {
        None => false,
        Some(m) if m.eq_ignore_ascii_case("windowed") => false,
        Some(m) if m.eq_ignore_ascii_case("borderless") => true,
        Some(m) => anyhow::bail!("{from} = {m:?}: expected \"windowed\" or \"borderless\""),
    };
    Ok(Display { fps_limit, borderless })
}

fn subdir(root: &Path, sub: &str) -> Result<PathBuf> {
    let dir = root.join(sub);
    ensure_outside_repo(&dir)?;
    std::fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

/// `%LOCALAPPDATA%/victory/<name>` (elsewhere the XDG or home cache).
fn default_dir(name: &str) -> Result<PathBuf> {
    let base = std::env::var_os("LOCALAPPDATA")
        .or_else(|| std::env::var_os("XDG_CACHE_HOME"))
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".cache")))
        .context("cannot determine a user directory; set VICTORY_CACHE_DIR and VICTORY_DATA_DIR")?;
    Ok(base.join("victory").join(name))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    #[test]
    fn environment_overrides_file() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            cache_dir: Some("cache-from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(|| OsString::from("from-env"));
        let cfg = Config::resolve(file, FileDisplay::default(), None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-env"));
        assert_eq!(cfg.cache_dir, PathBuf::from("cache-from-file"));
    }

    #[test]
    fn empty_env_values_are_ignored() {
        let file = FilePaths {
            game_dir: Some("from-file".into()),
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_GAME_DIR").then(OsString::new);
        let cfg = Config::resolve(file, FileDisplay::default(), None, env).unwrap();
        assert_eq!(cfg.game_dir.unwrap(), PathBuf::from("from-file"));
    }

    #[test]
    fn empty_test_titles_env_is_ignored() {
        let file = FilePaths {
            test_titles: vec!["from-file".into()],
            ..Default::default()
        };
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(OsString::new);
        let cfg = Config::resolve(file, FileDisplay::default(), None, env).unwrap();
        assert_eq!(cfg.test_titles, vec![PathBuf::from("from-file")]);
    }

    #[test]
    fn test_titles_env_splits_paths() {
        let value = std::env::join_paths(["first", "second"]).unwrap();
        let file = FilePaths::default();
        let env = |k: &str| (k == "VICTORY_TEST_TITLES").then(|| value.clone());
        let cfg = Config::resolve(file, FileDisplay::default(), None, env).unwrap();
        assert_eq!(
            cfg.test_titles,
            vec![PathBuf::from("first"), PathBuf::from("second")]
        );
    }

    #[test]
    fn example_config_parses() {
        let text = include_str!("../../../victory.example.toml");
        let parsed: FileConfig = toml::from_str(text).unwrap();
        assert!(parsed.paths.game_dir.is_some());
        assert!(parsed.display.fps_limit.is_none(), "the example keeps [display] commented out");
    }

    fn display(file: FileDisplay, pairs: &'static [(&str, &str)]) -> Result<Display> {
        let env = move |k: &str| pairs.iter().find(|(n, _)| *n == k).map(|(_, v)| OsString::from(*v));
        resolve_display(file, &env)
    }

    #[test]
    fn display_defaults_are_off_and_windowed() {
        assert_eq!(display(FileDisplay::default(), &[]).unwrap(), Display::default());
    }

    #[test]
    fn display_environment_overrides_file() {
        let file = FileDisplay { fps_limit: Some(30), window_mode: Some("windowed".into()) };
        let d = display(file, &[("VICTORY_FPS_LIMIT", "120"), ("VICTORY_WINDOW_MODE", "Borderless")]).unwrap();
        assert_eq!(d, Display { fps_limit: 120, borderless: true });
        let file = FileDisplay { fps_limit: Some(30), window_mode: Some("borderless".into()) };
        assert_eq!(display(file, &[("VICTORY_FPS_LIMIT", "")]).unwrap(), Display { fps_limit: 30, borderless: true });
    }

    #[test]
    fn display_errors_name_their_source() {
        let bad = |file, pairs| format!("{:#}", display(file, pairs).unwrap_err());
        assert!(bad(FileDisplay::default(), &[("VICTORY_FPS_LIMIT", "fast")]).contains("VICTORY_FPS_LIMIT"));
        assert!(bad(FileDisplay::default(), &[("VICTORY_FPS_LIMIT", "5000")]).contains("VICTORY_FPS_LIMIT"));
        assert!(bad(FileDisplay { fps_limit: Some(5000), window_mode: None }, &[]).contains("fps_limit"));
        let file = FileDisplay { fps_limit: None, window_mode: Some("exclusive".into()) };
        assert!(bad(file, &[]).contains("window_mode"));
    }

    #[test]
    fn cache_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("cache");
        let cfg = Config {
            cache_dir: inside,
            ..Default::default()
        };
        assert!(cfg.cache_subdir("x").is_err());
    }

    #[test]
    fn data_inside_repo_is_refused() {
        let inside = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
        let cfg = Config {
            data_dir: inside,
            ..Default::default()
        };
        assert!(cfg.data_subdir("x").is_err());
    }
}
