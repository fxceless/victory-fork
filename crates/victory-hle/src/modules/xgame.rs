//! The title's own id (from its manifest), which runtime areas exist, and
//! game invites and events. Offline no invite arrives and events go
//! nowhere.
#![allow(clippy::not_unsafe_ptr_arg_deref)]

use std::ffi::{CStr, c_char, c_void};

use super::xgameruntime::{
    Guid, HRESULT, Interface, Methods, Object, S_OK, add_ref, missing, put, text, token,
};
use crate::runtime;

/// Title id and launching another title.
pub const TITLE_PROVIDER: Guid = Guid::new(
    0x973a_344e,
    0x24bf,
    0x4d0f,
    [0x84, 0x57, 0x56, 0xc5, 0x34, 0x89, 0x2b, 0x29],
);
/// As Minecraft (2022) asks for it.
pub const TITLE_IID: Guid = Guid::new(
    0x2549_f142,
    0x6419,
    0x4a06,
    [0x97, 0xb5, 0x93, 0x1a, 0xab, 0x7c, 0x2f, 0x34],
);
/// Feature checks; the interface id is the provider's, as for the next two.
pub const FEATURES: Guid = Guid::new(
    0x8836_fe87,
    0xedb9,
    0x4fe3,
    [0x8d, 0xad, 0x05, 0xf0, 0xd2, 0xcd, 0x5b, 0x40],
);
pub const INVITES: Guid = Guid::new(
    0x0651_aae2,
    0x4012,
    0x4077,
    [0xbf, 0x84, 0x8b, 0x90, 0x97, 0x09, 0x0e, 0x2c],
);
pub const EVENTS: Guid = Guid::new(
    0xbbfb_dcc7,
    0xbfe7,
    0x409b,
    [0xa5, 0xca, 0xed, 0xf0, 0x54, 0x96, 0x0b, 0x4d],
);

pub static TITLE_OBJECT: Object = Object::new("title", &TITLE_VTABLE);
pub static FEATURES_OBJECT: Object = Object::new("features", &FEATURES_VTABLE);
pub static INVITES_OBJECT: Object = Object::new("invites", &INVITES_VTABLE);
pub static EVENTS_OBJECT: Object = Object::new("events", &EVENTS_VTABLE);

pub const INTERFACES: &[Interface] = &[
    (TITLE_PROVIDER, TITLE_IID, &TITLE_OBJECT),
    (FEATURES, FEATURES, &FEATURES_OBJECT),
    (INVITES, INVITES, &INVITES_OBJECT),
    (EVENTS, EVENTS, &EVENTS_OBJECT),
];

type This = *const Object;

extern "system" fn title_id(_: This, out: *mut u32) -> HRESULT {
    put(out, runtime::title().id)
}

/// As documented: the title ends (suspended first, as on any end) and the
/// game starts again in a new process; this never returns.
/// The title relaunches itself (its own path from `GetModuleFileNameA`) with
/// arguments from its configuration. The new `victory run` gets the same
/// options, the new executable, and the title's arguments after `--`
/// (the title reads the whole command line).
extern "system" fn launch_new_game(
    _: This,
    exe: *const c_char,
    args: *const c_char,
    _user: *const c_void,
) {
    let (exe, args) = (text(exe).unwrap_or_default(), text(args).unwrap_or_default());
    tracing::info!(exe, args, "XLaunchNewGame");
    let current = runtime::main_image()
        .and_then(runtime::image_path)
        .unwrap_or_default();
    let ours: Vec<_> = std::env::args_os().skip(1).collect();
    let command = relaunch_args(ours, current.as_os_str(), &exe, &args);
    super::lifecycle::end_title(
        "XLaunchNewGame",
        Box::new(move || super::lifecycle::relaunch(command)),
    );
    loop {
        std::thread::park();
    }
}

/// Normalizes a path for matching the current executable argument across
/// `/` vs `\`, `\\?\` verbatim prefixes, `.`/`..` segments, relative paths,
/// and Windows case-insensitivity.
fn normalize_path(path: &std::path::Path, cwd: Option<&std::path::Path>) -> String {
    let full: std::borrow::Cow<'_, std::path::Path> = match cwd {
        Some(dir) if path.is_relative() => std::borrow::Cow::Owned(dir.join(path)),
        _ => std::borrow::Cow::Borrowed(path),
    };
    let raw = full.to_string_lossy();
    let stripped = raw.strip_prefix(r"\\?\").unwrap_or(&raw);
    let normalized = stripped.replace('\\', "/");
    let (drive_prefix, rest) = if normalized.len() >= 2 && normalized.as_bytes()[1] == b':' {
        (Some(normalized[..2].to_ascii_lowercase()), &normalized[2..])
    } else {
        (None, normalized.as_str())
    };
    let leading_slash = rest.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for seg in rest.split('/') {
        match seg {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|&p| p != "..") {
                    parts.pop();
                } else if !leading_slash && drive_prefix.is_none() {
                    parts.push("..");
                }
            }
            other => parts.push(other),
        }
    }
    let mut out = String::new();
    if let Some(drive) = drive_prefix {
        out.push_str(&drive);
    }
    if leading_slash {
        out.push('/');
    }
    out.push_str(&parts.join("/"));
    if cfg!(windows) {
        out.to_ascii_lowercase()
    } else {
        out
    }
}

fn same_path(a: &std::ffi::OsStr, b: &std::ffi::OsStr, cwd: Option<&std::path::Path>) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    if a == b {
        return true;
    }
    if let (Ok(ca), Ok(cb)) = (std::fs::canonicalize(a), std::fs::canonicalize(b))
        && ca == cb
    {
        return true;
    }
    normalize_path(std::path::Path::new(a), cwd) == normalize_path(std::path::Path::new(b), cwd)
}

/// Splits a Windows command-line argument string following `CommandLineToArgvW`
/// quoting and backslash-escaping rules, preserving spaces inside `"..."`.
fn split_args(args: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = args.chars().peekable();
    while let Some(&c) = chars.peek() {
        if c.is_ascii_whitespace() {
            chars.next();
            continue;
        }
        let mut token = String::new();
        let mut in_quotes = false;
        while let Some(&ch) = chars.peek() {
            if !in_quotes && ch.is_ascii_whitespace() {
                break;
            }
            match ch {
                '\\' => {
                    let mut slashes = 0usize;
                    while chars.peek() == Some(&'\\') {
                        chars.next();
                        slashes += 1;
                    }
                    if chars.peek() == Some(&'"') {
                        for _ in 0..slashes / 2 {
                            token.push('\\');
                        }
                        if slashes % 2 == 1 {
                            chars.next();
                            token.push('"');
                        } else {
                            chars.next();
                            in_quotes = !in_quotes;
                        }
                    } else {
                        for _ in 0..slashes {
                            token.push('\\');
                        }
                    }
                }
                '"' => {
                    chars.next();
                    if in_quotes && chars.peek() == Some(&'"') {
                        chars.next();
                        token.push('"');
                    } else {
                        in_quotes = !in_quotes;
                    }
                }
                _ => {
                    chars.next();
                    token.push(ch);
                }
            }
        }
        out.push(token);
    }
    out
}

/// Our own arguments for the relaunch: the title's executable replaced by
/// `exe`, earlier title arguments dropped, `args` after `--`.
fn relaunch_args(
    ours: Vec<std::ffi::OsString>,
    current: &std::ffi::OsStr,
    exe: &str,
    args: &str,
) -> Vec<std::ffi::OsString> {
    let mut out: Vec<_> = ours.into_iter().take_while(|a| a != "--").collect();
    let cwd = std::env::current_dir().ok();
    if !exe.is_empty()
        && let Some(slot) = out
            .iter_mut()
            .find(|a| same_path(a.as_os_str(), current, cwd.as_deref()))
    {
        *slot = exe.into();
    }
    let parsed = split_args(args);
    if !parsed.is_empty() {
        out.push("--".into());
        out.extend(parsed.into_iter().map(Into::into));
    }
    out
}

/// Runtime areas by number, in the public feature list's order, that we
/// serve: accessibility (0), app capture (1), async calls and providers
/// (2, 3), display (4), game (5), invites (6), game saves (7), game UI (8),
/// launcher (9), networking (10), packages (11), persistent local storage
/// (12), speech synthesis (13), store (14), system (15), task queues (16),
/// users (18), error reports (19), game events (20). Not threads (17) or
/// game streaming (21).
const AVAILABLE: [u32; 20] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 19, 20,
];

extern "system" fn is_feature_available(_: This, feature: u32) -> bool {
    AVAILABLE.contains(&feature)
}

/// Nothing raises these events offline, so the callback never runs.
extern "system" fn register(
    _: This,
    _queue: *const c_void,
    _context: *mut c_void,
    _callback: *const c_void,
    out: *mut u64,
) -> HRESULT {
    put(out, token())
}
extern "system" fn unregister(_: This, _token: u64, _wait: bool) -> bool {
    true
}

/// Accepted and dropped; the name goes to the debug log.
extern "system" fn write_event(
    _: This,
    _user: *const c_void,
    _service_config: *const c_char,
    _session: *const c_char,
    name: *const c_char,
    _dimensions: *const c_char,
    _measurements: *const c_char,
) -> HRESULT {
    if !name.is_null() {
        // SAFETY: a non-null name is a NUL-terminated string.
        let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
        tracing::debug!(%name, "game event dropped");
    }
    S_OK
}

static TITLE_VTABLE: Methods<5> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    title_id as *const (),        // 0x18
    launch_new_game as *const (), // 0x20
]);

static FEATURES_VTABLE: Methods<4> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    is_feature_available as *const (), // 0x18
]);

static INVITES_VTABLE: Methods<5> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    register as *const (),   // 0x18
    unregister as *const (), // 0x20
]);

static EVENTS_VTABLE: Methods<4> = Methods([
    missing::<0> as *const (),
    add_ref as *const (),
    add_ref as *const (),
    write_event as *const (), // 0x18
]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::modules::xgameruntime::method;

    #[test]
    fn a_relaunch_keeps_our_options() {
        let ours: Vec<std::ffi::OsString> =
            ["run", "G:/game.exe", "--execute", "--console", "series-s", "--", "-old"]
                .map(Into::into)
                .into();
        let current = std::ffi::OsStr::new("G:/game.exe");
        let again = relaunch_args(ours.clone(), current, "G:/game.exe", " -config=x  -y ");
        let expected = ["run", "G:/game.exe", "--execute", "--console", "series-s", "--"];
        assert_eq!(again[..6], expected.map(std::ffi::OsString::from));
        assert_eq!(again[6..], ["-config=x", "-y"].map(std::ffi::OsString::from));
        let other = relaunch_args(ours, current, "G:/other.exe", "");
        assert_eq!(other.len(), 5);
        assert_eq!(other[1], "G:/other.exe");
    }

    #[test]
    fn relaunch_matches_normalized_paths_and_preserves_quoted_args() {
        let ours: Vec<std::ffi::OsString> = [
            "run",
            r"G:/Games/./Title/../Title/game.exe",
            "--execute",
            "--",
            "-old",
        ]
        .map(Into::into)
        .into();
        let current = std::ffi::OsStr::new(r"\\?\g:\games\title\GAME.EXE");
        let cmd = relaunch_args(
            ours,
            current,
            r"G:\Games\Title\other.exe",
            r#"-profile="My Profile" -path "C:\Program Files\Save\\" -say "a \"b\"" -empty """#,
        );
        assert_eq!(cmd[1], r"G:\Games\Title\other.exe");
        assert_eq!(cmd[3], "--");
        assert_eq!(
            cmd[4..],
            [
                "-profile=My Profile",
                "-path",
                r"C:\Program Files\Save\",
                "-say",
                r#"a "b""#,
                "-empty",
                "",
            ]
            .map(std::ffi::OsString::from)
        );
    }

    #[test]
    fn served_areas_are_available_others_not() {
        type Check = extern "system" fn(This, u32) -> bool;
        let check = method::<Check>(&FEATURES_OBJECT, 0x18);
        assert!(check(&FEATURES_OBJECT, 18), "users");
        assert!(!check(&FEATURES_OBJECT, 21), "game streaming");

        type Register = extern "system" fn(
            This,
            *const c_void,
            *mut c_void,
            *const c_void,
            *mut u64,
        ) -> HRESULT;
        type Unregister = extern "system" fn(This, u64, bool) -> bool;
        let null = std::ptr::null();
        let mut token = 0;
        let register = method::<Register>(&INVITES_OBJECT, 0x18);
        assert_eq!(
            register(&INVITES_OBJECT, null, null as _, null, &mut token),
            S_OK
        );
        assert_ne!(token, 0);
        assert!(method::<Unregister>(&INVITES_OBJECT, 0x20)(
            &INVITES_OBJECT,
            token,
            true
        ));
    }
}
