# Victory-fork contribution plan

Repo: github.com/fxceless/victory-gta6 (fork pushed to JordanFableFur/victory-fork).
CI: clippy --workspace --all-targets -D warnings (plain + all-features), cargo test --workspace.
Fork conventions (from merged PRs): "the title" not "the game", VICTORY_* switches off by
default, `shortcut:` markers, commit message = summary + why + testing + AI-use note,
AI use disclosed in PR body ("written with Codex and Claude Code; reviewed and posted with
my approval. No game files were used."). Style: lowercase comments, no trailing periods,
compact code, no cargo fmt (tree not fmt-clean).

Local validation:
- export PATH=$HOME/bin:/opt/homebrew/opt/rustup/bin:$PATH
- CC_x86_64_pc_windows_msvc="clang-cl.exe" cargo +1.98.1 clippy --workspace --all-targets \
    --target x86_64-pc-windows-msvc --locked -- -D warnings   (full CI clippy equivalent)
- cargo +1.98.1 test -p <crate> --lib --target aarch64-apple-darwin   (host-runnable tests:
  victory-shader 170, victory-core, victory-formats, victory-pipeline)
- Windows-only modules (victory-hle/gpu/loader) cross-check only (no run) — extra care on
  cfg(test) code there; validate via GitHub Actions run on the fork if needed.

Queue (work in order, one branch/PR per issue):
1. #246 shader scan: trailing file bytes change a raw DXBC container's hash.
   Fix: Source::Raw::bytes() in apps/victory/src/shader.rs must slice to the container's
   parsed total size (dxbc.rs parse truncates internally to `total` but Source::Raw keeps
   the whole file slice; translate.rs hashes blake3(&bytes) of the whole slice).
   Test: in shader.rs? No tests module there. Add tests to translate.rs? translate.rs has
   no test module either. Options: add #[cfg(test)] mod tests in apps/victory/src/shader.rs
   testing sources() + bytes() equality of standalone vs embedded with tail.
   Then same-hash property holds for VICTORY_TRANSLATE_ONLY and logs.
2. #248 XGameUI maxTextLength=0 must mean 1024 (GDK doc). Fix `limit()` + EM_LIMITTEXT
   (entry.max) in crates/victory-hle/src/modules/xgameui.rs: normalize 0 -> 1024 at entry,
   pass effective limit to dialog edit control and result trimming. Tests: existing test
   module covers entry at 0x88; add max=0 case with >1024 chars? ask() test returns
   "{default}typed" — limit() is called after. Extend limit tests + async flow.
3. #249 CLI: translate --jobs 0 must be rejected (NonZeroUsize or arg error), not silently
   do zero work. apps/victory/src/main.rs uses clap; change to NonZeroUsize.
4. #247 VICTORY_FAIL_DUMP: per-reason sample counts are per-worker/per-file; two files with
   same failure reason overwrite `<reason>_0.cdxil`. Fix: command-wide state (Mutex or
   atomic per sanitized reason) so sample index is global across files/workers; filename
   disambiguation when sanitized reasons collide.
5. #245 GPU rect fallback FP16 conversion NaN->inf, ties round up. crates/victory-gpu.
   Windows-only compile — cross-check clippy only; maybe run tests via CI.
6. #244 GPU frame events: concurrent FLIP waits consume same tick. crates/victory-gpu.
7. #243 XTaskQueue terminating a composite queue cancels another queue's callback. hle.
8. #242 XGameSave unreadable legacy blob dropped during migration. hle.
9. #250 DXIL lowering: one static sampler slot for Sample and SampleCmp. shader crate
   (sampler.rs); harder, needs careful reading; maybe later.

Order chosen by: testability locally first, size of fix, then risk.
Each fix: branch from origin/main, implement, local clippy+tests, push to fork, PR against
fxceless/victory-gta6, keep PR open for maintainer review.