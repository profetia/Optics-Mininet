# { pkgs ? import (builtins.fetchTarball {
#   url = "https://github.com/NixOS/nixpkgs/archive/5ed627539ac84809c78b2dd6d26a5cebeb5ae269.tar.gz";
# }) {} }:
{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.numpy
    pkgs.python312Packages.grpcio
    pkgs.python312Packages.grpcio-tools
    pkgs.python312Packages.scapy
    pkgs.python312Packages.uvloop
    pkgs.python312Packages.numba
  ];
}
