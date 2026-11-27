{ pkgs ? import <nixpkgs> { } }:
import ./dev/shell.nix { inherit pkgs; }
