#!/usr/bin/env bash
# Proton wrapper for VIctory. Edit the four paths at the top, then:
#   ./run.sh ./target/x86_64-pc-windows-msvc/diag/victory.exe \
#       run 'Z:\path\to\the-title-dump\<the title exe>' --execute
# See docs/linux-proton.md for the full setup.
set -euo pipefail

repo=/path/to/victory-fork                               # where this fork is checked out
game=/path/to/the-title-dump                             # your own dump (read-only is fine)
work=$HOME/.local/share/victory-proton-test              # the Proton prefix for this title
steam=$HOME/.local/share/Steam

runtime="$steam/steamapps/common/SteamLinuxRuntime_4/_v2-entry-point"
proton="$steam/steamapps/common/Proton - Experimental/proton"

unset WINEPREFIX VICTORY_GAME_DIR VICTORY_CACHE_DIR VICTORY_DATA_DIR VICTORY_TEST_TITLES
export STEAM_COMPAT_CLIENT_INSTALL_PATH="$steam"
export STEAM_COMPAT_DATA_PATH="$work/compatdata"
export STEAM_COMPAT_INSTALL_PATH="$repo"
export STEAM_COMPAT_MOUNTS="$repo:$game:$work"
export WINEDLLOVERRIDES='d3d12,d3d12core,dxgi=n'
export VICTORY_CONFIG='C:\victory.toml'

mkdir -p "$work"
cd "$work"
exec "$runtime" --verb=run -- "$proton" runinprefix "$@"
