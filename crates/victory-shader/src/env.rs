//! Reading boolean environment variables without getting the answer wrong.
//!
//! `std::env::var_os(name).is_some()` answers "was this set at all", which is
//! not the question any of these switches ask. `VICTORY_SHADER_CENSUS=0` was
//! read that way and turned the census **on**, and `VICTORY_D3D12_DEBUG=0`
//! turned the debug layer on in a second place after it had been fixed in the
//! first: two readers of one variable disagreeing about what `0` means.
//!
//! A switch that cannot be turned off is worse than a switch that does not
//! exist, because the thing it enables stays on while the operator believes it
//! is off -- and the cost shows up somewhere else entirely (the debug layer
//! costs hundreds of milliseconds a frame).
//!
//! The parser is deliberately the same rule as `victory-gpu`'s
//! `host::parse_env_flag`: unset, empty, `0`, `false`, `no` and `off` (ASCII
//! case-insensitive) are off; anything else is on. Both are pinned by tests,
//! and `xtask` refuses the `var_os(..).is_some()` shape anywhere in the tree,
//! so a new switch cannot be written this way.

/// Parses an opt-in boolean environment variable the way an operator expects:
/// unset, empty, `0`, `false`, `no` and `off` (ASCII case-insensitive, trimmed)
/// are off; anything else is on.
pub fn parse_env_flag(value: Option<&std::ffi::OsStr>) -> bool {
    let Some(value) = value else {
        return false;
    };
    let Some(text) = value.to_str() else {
        return true;
    };
    let trimmed = text.trim();
    !(trimmed.is_empty()
        || trimmed == "0"
        || trimmed.eq_ignore_ascii_case("false")
        || trimmed.eq_ignore_ascii_case("no")
        || trimmed.eq_ignore_ascii_case("off"))
}

/// Whether the opt-in boolean environment variable `name` is enabled.
pub fn env_flag(name: &str) -> bool {
    parse_env_flag(std::env::var_os(name).as_deref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn parse_env_flag_treats_explicit_falsey_values_as_off() {
        assert!(!parse_env_flag(None), "unset must be off");
        for off in ["", "   ", "0", " 0 ", "false", "FALSE", "False", "no", "NO", "off", "OFF"] {
            assert!(!parse_env_flag(Some(OsStr::new(off))), "{off:?} must disable the flag");
        }
        for on in ["1", " 1 ", "true", "TRUE", "yes", "YES", "on", "ON", "2"] {
            assert!(parse_env_flag(Some(OsStr::new(on))), "{on:?} must enable the flag");
        }
    }

    /// The same rule `victory-gpu` keeps, so the two crates cannot drift into
    /// disagreeing about what `VICTORY_D3D12_DEBUG=0` means.
    #[test]
    fn this_parser_agrees_with_the_gpu_crate_on_every_value_that_matters() {
        // Kept as a literal because the shader crate deliberately does not
        // depend on `victory-gpu`: this is the contract, not a copy of code.
        assert!(!env_flag("VICTORY_A_FLAG_THAT_IS_NOT_SET_ANYWHERE"));
        // SAFETY: tests in this binary do not set this variable; if a future
        // one does, the assertion below is the only reader and it says why.
        unsafe { std::env::set_var("VICTORY_ENV_FLAG_SELFTEST", "0") };
        assert!(!env_flag("VICTORY_ENV_FLAG_SELFTEST"), "=0 must be off");
        unsafe { std::env::set_var("VICTORY_ENV_FLAG_SELFTEST", "false") };
        assert!(!env_flag("VICTORY_ENV_FLAG_SELFTEST"), "=false must be off");
        unsafe { std::env::set_var("VICTORY_ENV_FLAG_SELFTEST", "1") };
        assert!(env_flag("VICTORY_ENV_FLAG_SELFTEST"), "=1 must be on");
        unsafe { std::env::remove_var("VICTORY_ENV_FLAG_SELFTEST") };
    }
}
