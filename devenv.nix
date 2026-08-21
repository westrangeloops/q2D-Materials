{ pkgs, ... }:

{
  # Use your Cachix cache for both pulling and pushing
  cachix.pull = [ "q2dmaterials" ];
  cachix.push = "q2dmaterials";

  packages = [
    # Core q2D-Materials dependencies
    pkgs.python312
    pkgs.python312Packages.numpy
    pkgs.python312Packages.pandas
    pkgs.python312Packages.scipy
    pkgs.python312Packages.scikit-learn
    pkgs.python312Packages.ase
    pkgs.python312Packages.pymatgen
    pkgs.python312Packages.rdkit
    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.numba
    pkgs.python312Packages.pymysql  # MySQL connector for COD database

    # Web framework
    pkgs.python312Packages.fastapi
    pkgs.python312Packages.uvicorn
    pkgs.python312Packages.pydantic
    pkgs.python312Packages.python-multipart

    # Development tools
    pkgs.python312Packages.pytest
    pkgs.python312Packages.jupyter
    pkgs.python312Packages.ipython

    # Build tools
    pkgs.python312Packages.pip
    pkgs.python312Packages.setuptools
    pkgs.python312Packages.wheel
    pkgs.python312Packages.streamlit
    pkgs.cachix
    pkgs.python312Packages.pyvis
    pkgs.python312Packages.graphviz
    pkgs.python312Packages.pygraphviz
    pkgs.python312Packages.seaborn

    pkgs.graphviz
  ];

  # X11 support and C++ stdlib (needed for pip build isolation / numpy wheels)
  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.xorg.libX11
      pkgs.xorg.libXext
      pkgs.xorg.libSM
      pkgs.xorg.libICE
      pkgs.stdenv.cc.cc.lib
    ];
  };

  scripts = {
    dev.exec = ''
      # Use nix Python and nix packages; venv with system-site-packages
      if [ ! -d ".venv" ]; then
        echo "Creating virtual environment (system-site-packages to use nix deps)..."
        python3 -m venv --system-site-packages .venv
      fi
      
      source .venv/bin/activate
      pip install --upgrade pip
      
      if [ ! -f .venv/.installed ]; then
        echo "Installing q2D-Materials in editable mode..."
        if pip install -e . --config-settings editable_mode=compat; then
          touch .venv/.installed
        else
          echo "⚠️  Editable install failed; PYTHONPATH fallback is still active."
        fi
      fi
      
      echo ""
      echo "✨ q2D-Materials development environment ready!"
      echo ""
      echo "Core capabilities:"
      echo "  🧱 Layer stacking - Build complex structures from JSON templates"
      echo "  🔄 Ion assignment - Automated population with mixed compositions"
      echo "  🧬 Molecular spacers - RDKit-powered organic linker attachment"
      echo "  📐 Glazer tilting - Systematic octahedral distortion patterns"
      echo "  🔍 Multi-view visualization - ASE-powered structure analysis"
      echo "  🌀 Twisted interfaces - Moiré pattern generation"
    '';

    test.exec = "pytest tests/";
    
    example.exec = ''
      python3 -c "from q2D_Materials.core.creator import q2D_creator; print('q2D-Materials ready!')"
    '';
  };

  enterShell = ''
    # Add project root to PYTHONPATH so q2D_Materials can be imported
    export PYTHONPATH="$PWD:$PYTHONPATH"
    dev
    # Activate .venv so python3 uses nix packages
    source .venv/bin/activate
  '';
}