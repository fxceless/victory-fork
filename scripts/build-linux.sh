#!/usr/bin/env bash
# Build the Windows runtime on Linux; Proton supplies its platform APIs.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
if ! command -v cargo >/dev/null || ! command -v cargo-xwin >/dev/null; then
    echo "Install Rust and cargo-xwin, or enter nix-shell. See docs/linux.md." >&2
    exit 1
fi
exec cargo xwin build --locked --release --target x86_64-pc-windows-msvc -p victory "$@"
