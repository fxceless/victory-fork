{ pkgs ? import <nixpkgs> { } }:
pkgs.mkShell {
  # rustup follows rust-toolchain.toml, including the Windows MSVC target.
  packages = with pkgs; [
    rustup
    cargo-xwin
    clang
    lld
    llvm
    pkg-config
    umu-launcher
    shellcheck
  ];
}
