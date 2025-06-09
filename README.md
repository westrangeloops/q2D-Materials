![Github_portada](https://github.com/westrangeloops/SVC-Materials/blob/main/Logos/Github_portada.png)

# SVC-Materials
SVC-Maestra is a Python package for creating and analyzing DJ perovskite structures, currently developed by the Theoretical Materials Science group of the University of Antioquia. SVC-Maestra simplifies the generation of initial input structures for quasi-2D perovskite systems and provides some analysis tools for structural characterization.

## Features
- Generation of initial input structures for quasi-2D perovskite systems
- Structural analysis of perovskite octahedra including:
  - Bond angles (axial and equatorial)
  - Bond lengths
  - Octahedral distortions
  - Out-of-plane distortions
- Analysis of organic spacer penetration
- Visualization tools for structural components
- Molecule isolation from crystal structures with template-based validation


## Installation
We recommend using a conda environment:
```bash
conda create -n SVC_Materials pip
conda activate SVC_Materials
pip install ase pandas numpy matplotlib networkx
```

## Usage

### Structure Generation
```python
from svc_maestra_lib.svc_maestra import q2D_creator
```

### Structure Analysis
```python
from SVC_materials.core.analyzer import q2D_analysis

# Initialize analyzer with B-site metal and X-site halogen
analyzer = q2D_analysis(B='Pb', X='I', crystal='path/to/structure.vasp')

# Analyze perovskite structure
structure_data = analyzer.analyze_perovskite_structure()

# Calculate nitrogen penetration
penetration_data = analyzer.calculate_n_penetration()

# Visualize components
analyzer.show_original()  # Show full structure
analyzer.show_spacer()    # Show isolated spacer
analyzer.show_salt()      # Show isolated salt

# Isolate spacer molecule using a template
isolated_molecule = analyzer.isolate_spacer_molecule(
    template_file='path/to/template.xyz',
    output_file='isolated_molecule.xyz'  # Optional
)
```

### Molecule Isolation
The package provides two ways to isolate molecules from crystal structures:

1. Using the analyzer class:
```python
from SVC_materials.core.analyzer import q2D_analysis

analyzer = q2D_analysis(B='Pb', X='I', crystal='structure.vasp')
isolated_molecule = analyzer.isolate_spacer_molecule(
    template_file='template.xyz',
    output_file='output.xyz'  # Optional
)
```

2. Using the utility function directly:
```python
from SVC_materials.utils.isolate_molecule import isolate_molecule

# File-based interface
success = isolate_molecule(
    crystal_file='structure.vasp',
    template_file='template.xyz',
    output_file='output.xyz',
    debug=True  # Optional, for detailed output
)

# Or work directly with ASE Atoms objects
from SVC_materials.utils.isolate_molecule import isolate_molecule_from_atoms
from ase.io import read

crystal_atoms = read('structure.vasp')
template_atoms = read('template.xyz')
isolated_molecule = isolate_molecule_from_atoms(
    crystal_atoms=crystal_atoms,
    template_atoms=template_atoms,
    debug=True  # Optional
)
```

The isolation process includes several validation checks:
- Exact size matching with template
- Chemical formula matching
- Connectivity validation
- N-N path validation
- Bond length validation

### Analysis Output
The `analyze_perovskite_structure()` method returns a dictionary containing:
- Bond angles (axial and equatorial)
- Bond lengths
- Octahedral distortions
- Distortion classification (Regular, Axially Distorted, Equatorially Distorted, Highly Distorted)
- Distortion types (Compressed, Elongated, Rhombic, Mixed)
- Out-of-plane distortions

## Examples
Check the `examples` folder for detailed Jupyter notebooks demonstrating:
- Structure generation
- Structural analysis
- Visualization
- Data processing
- Molecule isolation

## Contributing
We welcome contributions to SVC-Materials! If you would like to contribute:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
SVC-Maestra is licensed under the MIT license.
