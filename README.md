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
- Molecule isolation from crystal structures with template-based validation:
  - Exact size matching with template
  - Chemical formula validation
  - Connectivity analysis
  - N-N path validation
  - Bond length validation
- Molecule deformation analysis:
  - Bond length comparison
  - Bond angle comparison
  - Dihedral angle comparison
  - Connectivity validation

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
```

### Molecule Isolation
The package provides three ways to isolate molecules from crystal structures:

1. **High-level batch processing**:
```python
from SVC_materials.utils.isolate_molecule import process_molecules

# Process all molecules in a directory
stats = process_molecules(
    input_dir="data/original",
    output_dir="data/molecules",
    template_dir="~/templates",
    debug=True
)

# Check results
print(f"Processed {stats['successful']}/{stats['total_files']} files")
```

2. **Single file processing**:
```python
from SVC_materials.utils.isolate_molecule import isolate_molecule

# Isolate a single molecule
success = isolate_molecule(
    crystal_file="structure.vasp",
    template_file="template/CONTCAR",
    output_file="output.xyz",
    debug=True
)
```

3. **Direct Atoms object processing**:
```python
from SVC_materials.utils.isolate_molecule import isolate_molecule_from_atoms
from ase.io import read

# Work directly with ASE Atoms objects
crystal_atoms = read("structure.vasp")
template_atoms = read("template/CONTCAR")
isolated_molecule = isolate_molecule_from_atoms(
    crystal_atoms=crystal_atoms,
    template_atoms=template_atoms,
    debug=True
)
```

### Molecule Deformation Analysis
The package provides detailed analysis of molecular deformations by comparing isolated molecules with their ideal templates. This analysis includes:

1. **Bond Length Analysis**:
   - Calculates Mean Absolute Error (MAE) in bond lengths
   - Classifies deformations as:
     - "Normal bond fluctuations" (< 0.05 Å)
     - "Significant bond strain" (0.05-0.2 Å)
     - "Severe bond strain" (> 0.2 Å)

2. **Bond Angle Analysis**:
   - Calculates MAE in bond angles
   - Classifies deformations as:
     - "Normal angle flexibility" (< 5°)
     - "Moderate angle strain" (5-15°)
     - "Severe angle strain" (> 15°)

3. **Dihedral Angle Analysis**:
   - Calculates MAE in dihedral angles
   - Classifies deformations as:
     - "Local torsional oscillation" (< 30°)
     - "Moderate conformational change" (30-90°)
     - "Major conformational change" (> 90°)

Example usage:
```python
from SVC_materials.utils.isolate_molecule import analyze_molecule_deformation

# Analyze molecule deformation
metrics = analyze_molecule_deformation(
    ideal_file='template.xyz',
    extracted_file='extracted.xyz',
    output_dir='analysis_results',
    debug=True
)

# Access results
print(f"Bond length MAE: {metrics['bond_length_mae']:.4f} Å")
print(f"Bond angle MAE: {metrics['bond_angle_mae']:.4f} degrees")
print(f"Dihedral angle MAE: {metrics['dihedral_angle_mae']:.4f} degrees")
print(f"Deformation classifications: {metrics['classifications']}")
```

### Output Files

For each analyzed molecule, the following files are generated:

1. **analysis.log**: Contains a detailed summary of the deformation analysis:
   ```
   Deformation Analysis Summary:
   ---------------------------
   Chemical Formula: C10H24N2
   Isomorphic: True

   Bond Length Analysis:
     Mean Error: 0.0234 ± 0.0156 Å
     Range: [0.0012, 0.0456] Å
     Classification: Normal bond fluctuations

   Bond Angle Analysis:
     Mean Error: 3.4567 ± 2.1234°
     Range: [0.1234, 8.9012]°
     Classification: Normal angle flexibility

   Dihedral Angle Analysis:
     Mean Error: 25.6789 ± 15.4321°
     Range: [1.2345, 45.6789]°
     Classification: Local torsional oscillation
   ```

2. **bonds.csv**: Detailed bond length analysis
   - bond: Bond identifier (e.g., "C1-C2")
   - ideal_length: Template bond length (Å)
   - extracted_length: Extracted bond length (Å)
   - error: Absolute deviation (Å)

3. **angles.csv**: Detailed bond angle analysis
   - angle: Angle identifier (e.g., "C1-C2-C3")
   - ideal_angle: Template angle (degrees)
   - extracted_angle: Extracted angle (degrees)
   - error: Absolute deviation (degrees)

4. **dihedrals.csv**: Detailed dihedral angle analysis
   - dihedral: Dihedral identifier (e.g., "C1-C2-C3-C4")
   - ideal_angle: Template dihedral angle (degrees)
   - extracted_angle: Extracted dihedral angle (degrees)
   - error: Absolute deviation (degrees)

5. **deformation_summary.csv**: Overall analysis summary
   - molecule: Molecule identifier
   - bond_length_mae: Mean absolute error in bond lengths
   - bond_length_std: Standard deviation of bond length errors
   - bond_angle_mae: Mean absolute error in bond angles
   - bond_angle_std: Standard deviation of bond angle errors
   - dihedral_angle_mae: Mean absolute error in dihedral angles
   - dihedral_angle_std: Standard deviation of dihedral angle errors
   - is_isomorphic: Whether the molecule matches the template
   - chemical_formula: Chemical formula of the molecule
   - bond_classification: Classification of bond deformations
   - angle_classification: Classification of angle deformations
   - dihedral_classification: Classification of dihedral deformations

### Interpretation Guidelines

1. **Bond Length Deviations**
   - Normal fluctuations (< 0.05 Å) are typical and indicate a stable structure
   - Significant strain (0.05 - 0.2 Å) may indicate local stress or potential errors
   - Severe strain (> 0.2 Å) suggests structural issues that need investigation

2. **Bond Angle Deviations**
   - Normal flexibility (< 5°) is expected in most structures
   - Moderate strain (5° - 15°) may affect local geometry
   - Severe strain (> 15°) indicates significant distortion

3. **Dihedral Angle Deviations**
   - Local oscillation (< 30°) represents normal conformational flexibility
   - Moderate changes (30° - 90°) may indicate transitions between states
   - Major changes (> 90°) suggest significant structural reorganization

## Examples
Check the `examples` folder for detailed Jupyter notebooks demonstrating:
- Structure generation
- Structural analysis
- Visualization
- Data processing
- Molecule isolation with different interfaces
- Molecule deformation analysis

## Contributing
We welcome contributions to SVC-Materials! If you would like to contribute:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
SVC-Maestra is licensed under the MIT license.
