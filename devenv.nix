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
    pkgs.python312Packages.ase
    pkgs.python312Packages.pymatgen
    pkgs.python312Packages.rdkit
    pkgs.python312Packages.matplotlib
    pkgs.python312Packages.numba

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
  ];

  # X11 support for visualization
  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.xorg.libX11
      pkgs.xorg.libXext
      pkgs.xorg.libSM
      pkgs.xorg.libICE
    ];
    PYTHONPATH = "$PWD:$PYTHONPATH";
  };

  scripts = {
    dev.exec = ''
      # Set up virtual environment
      if [ ! -d ".venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv .venv
      fi
      
      source .venv/bin/activate
      pip install --upgrade pip
      
      if [ ! -f .venv/.installed ]; then
        echo "Installing packages from requirements.txt..."
        pip install -r requirements.txt
        touch .venv/.installed
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
    dev
  '';
}