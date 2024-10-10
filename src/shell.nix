{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.python312Packages.numpy
    pkgs.python312Packages.grpcio
    pkgs.python312Packages.grpcio-tools
    pkgs.python312Packages.scapy
  ];
}
