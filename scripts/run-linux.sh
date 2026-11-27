#!/usr/bin/env bash
# Preserve stdin/stdout/stderr and the launcher's exit status.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${1:-}" == "--help" && $# == 1 ]]; then
    cat <<'EOF'
Usage: scripts/run-linux.sh <victory command> [arguments...]
Example: scripts/run-linux.sh run "/absolute/path/to/title.exe" --execute

VICTORY_EXE overrides the release Windows binary (Linux path).
WINEPREFIX overrides the dedicated prefix; PROTONPATH defaults to GE-Proton.
The executable argument after "run" may be a Linux path. Other path arguments
and VICTORY_* path variables must use Windows paths (see docs/linux.md).
EOF
    exit 0
fi
if ! command -v umu-run >/dev/null; then
    echo "umu-run is required; install umu-launcher or enter nix-shell." >&2
    exit 1
fi
exe="${VICTORY_EXE:-$root/target/x86_64-pc-windows-msvc/release/victory.exe}"
if [[ ! -f "$exe" ]]; then
    echo "Windows binary not found: $exe. Run scripts/build-linux.sh first." >&2
    exit 1
fi
exe="$(realpath -- "$exe")"
export WINEPREFIX="${WINEPREFIX:-${XDG_DATA_HOME:-$HOME/.local/share}/victory/prefix}"
export PROTONPATH="${PROTONPATH:-GE-Proton}"
export GAMEID="${GAMEID:-umu-default}"
if [[ "$WINEPREFIX" != /* ]]; then
    echo "WINEPREFIX must be an absolute Linux path." >&2
    exit 1
fi
# The default Proton prefix maps the Linux filesystem to Z:.
if [[ "${1:-}" == run && $# -ge 2 && "$2" != *:* ]]; then
    title="$(realpath -e -- "$2")"
    shift 2
    set -- run "Z:${title//\//\\}" "$@"
fi
# Use the repository config regardless of the caller's current directory.
cd "$root"
# GE-Proton routes Unix executable paths through umu.exe, which can create a
# separate console. A Windows path takes its direct Wine launch path instead.
# Keep the Unix install directory available to umu for runtime mount discovery.
export STEAM_COMPAT_INSTALL_PATH="${STEAM_COMPAT_INSTALL_PATH:-$(dirname -- "$exe")}"
exec umu-run "Z:${exe//\//\\}" "$@"
