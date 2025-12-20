{
  description = "q2D-Materials: Template-driven quasi-2D perovskite structure generator";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        pythonPackages = python.pkgs;
      in {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            # Core q2D-Materials dependencies
            python
            pythonPackages.numpy      # Numerical computing
            pythonPackages.pandas     # Data manipulation
            pythonPackages.scipy      # Scientific computing
            pythonPackages.ase        # Atomic Simulation Environment
            pythonPackages.pymatgen   # Materials analysis
            pythonPackages.rdkit      # Molecular chemistry
            pythonPackages.matplotlib # Plotting and visualization

            # Web framework
            pythonPackages.fastapi    # FastAPI web framework
            pythonPackages.uvicorn    # ASGI server for FastAPI
            pythonPackages.pydantic   # Data validation using Python type annotations
            pythonPackages.python-multipart  # Required for FastAPI file uploads (form data)

            # Development tools
            pythonPackages.pytest     # Testing framework
            pythonPackages.jupyter    # Interactive development
            pythonPackages.ipython    # Enhanced Python shell

            # Build tools
            pythonPackages.pip
            pythonPackages.setuptools
            pythonPackages.wheel
            pythonPackages.streamlit
          ];

          # For ase-gui visualization and matplotlib backends
          nativeBuildInputs = [
            pkgs.xorg.libX11
            pkgs.xorg.libXext
            pkgs.xorg.libSM
            pkgs.xorg.libICE
          ];

          shellHook = ''
            export PYTHONPATH=$PWD:$PYTHONPATH

            # Install additional packages from requirements.txt if needed
            if [ ! -f .venv_installed ]; then
              echo "Installing additional Python packages from requirements.txt..."
              pip install --break-system-packages -r requirements.txt
              touch .venv_installed
            fi

            echo "q2D-Materials development environment ready!"
            echo ""
            echo "Core capabilities:"
            echo "  🧱 Layer stacking - Build complex structures from JSON templates"
            echo "  🔄 Ion assignment - Automated population with mixed compositions"
            echo "  🧬 Molecular spacers - RDKit-powered organic linker attachment"
            echo "  📐 Glazer tilting - Systematic octahedral distortion patterns"
            echo "  🔍 Multi-view visualization - ASE-powered structure analysis"
            echo "  🌀 Twisted interfaces - Moiré pattern generation"
            echo ""
            echo "Quick start:"
            echo "  python3 -c \"from q2D_Materials.core.creator import q2D_creator; print('Ready!')\""
            echo "  nix develop -c python3 Examples/plot.py  # Regenerate documentation images"
          '';
        };
      });
}