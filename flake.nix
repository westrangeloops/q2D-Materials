{
  description = "SVC-Materials Python development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python311;
        pythonPackages = python.pkgs;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            python
            pythonPackages.pip
            pythonPackages.setuptools
            pythonPackages.wheel
            pythonPackages.numpy
            pythonPackages.pandas
            pythonPackages.ase
            pythonPackages.matplotlib
            pythonPackages.rdkit
            pythonPackages.ipython
            pythonPackages.jupyter
            pythonPackages.scipy
            pythonPackages.requests
            pythonPackages.pytest
            pythonPackages.pytest-cov
            pythonPackages.pytest-xdist
            pythonPackages.scikit-learn
            pythonPackages.networkx
            pythonPackages.rdkit
            pythonPackages.seaborn
            pythonPackages.pymatgen
            pythonPackages.rmsd
            # Add more packages as needed
          ];
          # For ase-gui visualization
          nativeBuildInputs = [ pkgs.xorg.libX11 pkgs.xorg.libXext pkgs.xorg.libSM pkgs.xorg.libICE ];
          shellHook = ''
            export PYTHONPATH=$PWD:$PYTHONPATH
            echo "SVC-Materials dev environment ready!"
          '';
        };
      });
}