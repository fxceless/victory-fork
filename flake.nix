{
  description = "Nix flake for Victory Fork Windows cross-builds from Linux";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    fenix = {
      url = "github:nix-community/fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    fenix,
    nixpkgs,
    ...
  }: let
    system = "x86_64-linux";
    pkgs = nixpkgs.legacyPackages.${system};
    llvm = pkgs.llvmPackages;
    wine = pkgs.wineWow64Packages.stable;
    rust = fenix.packages.${system}.fromToolchainFile {
      file = ./rust-toolchain.toml;
      # Update the Rust distribution manifest hash when changing the channel.
      sha256 = "sha256-p8h3Sl/YRByZfZTAKXdsvF6xEenXKrXSVvpphmZENH4=";
    };
    wineRunner = pkgs.writeShellScript "victory-wine" ''
      export WINEPREFIX="''${WINEPREFIX:-''${XDG_STATE_HOME:-$HOME/.local/state}/victory-wine}"
      export WINEDEBUG="''${WINEDEBUG:--all}"
      export WINEDLLOVERRIDES="''${WINEDLLOVERRIDES-mscoree,mshtml=}"
      mkdir -p -- "$(${pkgs.coreutils}/bin/dirname -- "$WINEPREFIX")"
      exec ${wine}/bin/wine "$@"
    '';
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [
        rust
        pkgs.cargo-xwin
        pkgs.git
        llvm.clang-unwrapped
        llvm.lld
        llvm.llvm
        wine
      ];

      XWIN_VERSION = "17";
      XWIN_SDK_VERSION = "10.0.26100";
      XWIN_CRT_VERSION = "14.44.17.14";
      CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_RUNNER = wineRunner;
    };

    formatter.${system} = pkgs.alejandra;
  };
}
