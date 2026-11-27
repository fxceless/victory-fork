# CLAUDE.md

A fork of VIctory: runs an Xbox Series title natively on Windows from files
the user supplies. Rust workspace, Windows/MSVC only. Start with
[README.md](README.md) (fork status, switches), [CONTRIBUTING.md](CONTRIBUTING.md)
(workflow) and [docs/architecture.md](docs/architecture.md) (crate map).

## Rule #1 (non-negotiable)

Nothing from the title or any proprietary source goes in the repository: no
assets, executable bytes, shader dumps, disassembly, decompiled code, tables,
strings or screenshots, in files, tests, fixtures, comments or commit
messages. Do not name the game anywhere in the tree; say "the title". Write
derived output only outside the checkout (cache/data dirs from
`victory.toml`). See [docs/asset-policy.md](docs/asset-policy.md).

## Commands

- `cargo xtask check` - everything CI runs (policy, clippy x2 with
  `-D warnings`, tests). `--no-test` skips tests.
- `cargo test -p victory-shader` (or another crate) for a quick loop.
- `cargo build --release -p victory`; diagnostics build:
  `cargo build --profile diag -p victory --features diagnostics`.
- Running the title needs the user's own files; ask instead of guessing paths.

## Conventions

- Do not run `cargo fmt --all` (the tree is not rustfmt-clean; it would
  rewrite ~800 places). Format only new files.
- New, unproven behaviour sits behind a `VICTORY_*` env switch, off by
  default, documented in a comment where it is read.
- Mark approximations of console behaviour with a `shortcut:` comment.
- New `unsafe` blocks get a `// SAFETY:` comment.
- Tests use synthetic data and assert on properties; tests needing real
  files or a GPU skip when the env var / adapter is missing.
- Commit subjects are plain sentences about the effect on the title, with
  `(#issue)` when there is one; bodies give the why and measurements.
