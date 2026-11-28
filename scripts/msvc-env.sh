#!/bin/sh
# Puts the MSVC linker and the Windows SDK library paths in this shell's
# environment, so `cargo build` can link on Windows from Git Bash.
#
# Usage:  . scripts/msvc-env.sh
#         cargo test -p victory-gpu --lib
#
# Why this exists: Git Bash ships `/usr/bin/link.exe`, GNU coreutils' `link`,
# which comes first on PATH and shadows the MSVC linker of the same name.
# rustc then fails with "linking with `link.exe` failed ... ensure the C++
# build tools workload is selected" — the message blames the workload even
# when the workload is installed and only `LIB` is unset. Pointing PATH and
# LIB at the toolchain is the whole fix.
#
# Nothing here is hardcoded to one machine. Override with:
#   VICTORY_MSVC=<dir holding cl.exe>   e.g. F:/VS C++/VC/Tools/MSVC/14.51.36231/bin/Hostx64/x64
#   VICTORY_WINDOWS_KITS=<dir holding Lib/10.*/um/x64>
# Without a usable toolchain this prints why and changes nothing, so sourcing
# it is always safe.

_msvc_env_say() { printf 'msvc-env: %s\n' "$*" >&2; }

# 1. The toolchain's own bin directory.
_msvc_bin="${VICTORY_MSVC:-}"
if [ -z "$_msvc_bin" ]; then
    _vswhere='/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe'
    if [ -x "$_vswhere" ]; then
        _vs_root="$("$_vswhere" -latest -products '*' \
            -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 \
            -property installationPath 2>/dev/null | tr -d '\r')"
    fi
    if [ -z "$_vs_root" ]; then
        # Some layouts live outside Program Files entirely; take the newest.
        _vs_root="$(ls -d /[a-z]/VS\ C++ /[a-z]/'Program Files'/Microsoft\ Visual\ Studio/*/* 2>/dev/null |
            while IFS= read -r d; do [ -d "$d/VC/Tools/MSVC" ] && printf '%s\n' "$d"; done |
            sort | tail -1)"
    fi
    if [ -n "$_vs_root" ]; then
        _msvc_root="$(ls -d "$_vs_root"/VC/Tools/MSVC/*/bin/Hostx64/x64 2>/dev/null | sort -V | tail -1)"
        _msvc_bin="$_msvc_root"
    fi
fi
# `vswhere` answers in Windows form (`F:\VS C++\...`), and a PATH entry in that
# form does not resolve here, so GNU link.exe keeps winning. Ask for the POSIX
# spelling whenever the path exists in Windows form.
case "$_msvc_bin" in
    [A-Za-z]:*) _msvc_bin="$(cygpath -u "$_msvc_bin" 2>/dev/null || printf '%s' "$_msvc_bin")" ;;
esac

# 2. The Windows SDK's library directories (um and ucrt, for this architecture).
_kits="${VICTORY_WINDOWS_KITS:-}"
if [ -z "$_kits" ]; then
    for _d in /f/C\ Drive\ Packages\ Dump/WindowsKits/Windows\ Kits/10 \
              /c/'Program Files (x86)'/Windows\ Kits/10 \
              /[a-z]/'Program Files (x86)'/Windows\ Kits/10; do
        [ -d "$_d/Lib" ] && _kits="$_d" && break
    done
fi

if [ -z "$_msvc_bin" ] || [ ! -x "$_msvc_bin/link.exe" ]; then
    _msvc_env_say "no MSVC linker found; set VICTORY_MSVC to the directory holding link.exe"
    unset _msvc_bin _kits _msvc_root _vs_root _d _vswhere
    return 0 2>/dev/null || exit 0
fi

if [ -z "$_kits" ]; then
    _msvc_env_say "the MSVC toolbox is at $_msvc_bin but no Windows SDK Lib was found;"
    _msvc_env_say "set VICTORY_WINDOWS_KITS. Linking will fail for want of the SDK libraries."
else
    # Newest SDK containing both um/x64 and ucrt/x64. The linker is a Windows
    # program, so LIB is semicolon-separated Windows paths: the toolchain's own
    # lib, then um and ucrt from the SDK.
    _ver="$(for _cand in "$_kits"/Lib/10.*; do
        [ -d "$_cand/um/x64" ] && [ -d "$_cand/ucrt/x64" ] && printf '%s\n' "$_cand"
    done 2>/dev/null | sort -V | tail -1)"
    if [ -z "$_ver" ]; then
        _msvc_env_say "no complete Windows SDK (um/x64 + ucrt/x64) found under $_kits/Lib; keeping existing LIB"
    else
        _msvc_lib="$(cygpath -w "$_msvc_bin/../../../lib/x64")"
        _um="$(cygpath -w "$_ver/um/x64")"
        _ucrt="$(cygpath -w "$_ver/ucrt/x64")"
        LIB="$_msvc_lib;$_um;$_ucrt"
        export LIB
        _msvc_env_say "SDK libraries: $_ver"
    fi
fi

PATH="$_msvc_bin:$PATH"
export PATH
export VICTORY_MSVC="$_msvc_bin"
_msvc_env_say "linker: $_msvc_bin/link.exe"

unset _msvc_bin _kits _msvc_root _vs_root _d _vswhere _ver _cand _msvc_lib _um _ucrt
