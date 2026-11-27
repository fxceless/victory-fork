# Contributing

Thanks for helping. Read **rule #1** first; everything else is convenience.

## Rule #1: nothing from the title goes in the repository

No assets, no executable bytes, no shader dumps, no screenshots, nothing
decompiled or extracted, not even in a test or a comment. Build from files
the user supplies at runtime. The full list of what is and is not allowed is
in [docs/asset-policy.md](docs/asset-policy.md); `cargo xtask policy`
enforces it and the pre-commit hook runs it on every commit.

Keep your dump, cache and data folders **outside** this checkout. The tools
refuse to write game-derived output under a folder holding `.victory-root`.

## Setting up (Windows)

1. **Visual Studio Build Tools** with the *Desktop development with C++*
   workload (MSVC and a Windows 10/11 SDK).
2. **Rust** through [rustup](https://rustup.rs). The version is pinned in
   `rust-toolchain.toml`; the first `cargo` command installs it.
3. The project builds for `x86_64-pc-windows-msvc`. If your rustup default
   host is `windows-gnu` (`rustc -vV` shows `host: x86_64-pc-windows-gnu`),
   pin the MSVC toolchain for this folder only:

   ```powershell
   rustup override set 1.98.1-x86_64-pc-windows-msvc
   ```

4. In the checkout:

   ```powershell
   cargo xtask install-hooks                # policy check on every commit
   copy victory.example.toml victory.toml   # set your paths (outside the repo)
   cargo xtask check                        # what CI runs; needs no game data
   cargo run -p victory -- config           # confirms your paths
   ```

## Everyday commands

| Command | What it does |
|---------|--------------|
| `cargo xtask check` | policy check, clippy (plain and diagnostics builds, `-D warnings`), tests: the same as CI |
| `cargo xtask check --no-test` | the same without the tests (faster) |
| `cargo xtask policy` | the asset-policy check alone |
| `cargo test -p <crate>` | one crate's tests, e.g. `victory-shader` |
| `cargo build --release -p victory` | the build you run the title with (`target\release\victory.exe`) |
| `cargo build --profile diag -p victory --features diagnostics` | with debugging aids (`target\diag\victory.exe`); see the README's Debugging section |

Tests that need real files or a GPU read their paths from environment
variables (`VICTORY_GAME_DIR`, `VICTORY_TEST_TITLES`) and skip when unset or
when no suitable adapter is present.

## Making a change

- **Where things go**: [docs/architecture.md](docs/architecture.md) has the
  crate map. Title-specific formats, build identities and hooks belong in
  `victory-game`; everything else should work for any Xbox Series title.
- **New behaviour that is not proven yet goes behind a switch**, off by
  default: a `VICTORY_*` environment variable, described in a comment where
  it is read. Say in the README which switches a result was measured with.
- **Approximations are marked.** Where code approximates what the console
  does instead of doing it, say so in a comment starting `shortcut:`.
- **Unsafe code**: `unsafe_op_in_unsafe_fn` is denied; give new `unsafe`
  blocks a `// SAFETY:` comment.
- **Formatting**: the tree is not fully `rustfmt`-clean (it follows
  upstream), and CI does not check formatting. Do **not** run
  `cargo fmt --all`; it rewrites hundreds of places and makes upstream merges
  painful. Format new files; in existing ones, match the code around you.
- **Tests assert on properties, never on snapshots** of game data. Use
  synthetic data your own code generates.

## Commits and pull requests

- Commit messages: a short subject in plain words that says what changed for
  the title (`Gathers at a named mip level read level 0; fixed`), then a body
  with the why and anything measured (frame times, counts, the GPU it was
  tested on). Prefix with the area when it helps (`Shaders:`).
- Reference the issue number in the subject when there is one: `(#11)`.
- Run `cargo xtask check` before pushing; CI runs the same steps on
  `windows-latest`.
- If something from the title slips into a commit, a follow-up commit is not
  enough: see "If something slips through" in the asset policy.

## Upstream

This is a fork of [VIctory](https://github.com/Marvisak/VIctory). Changes
that are not specific to this fork's test machine are worth offering
upstream too.
