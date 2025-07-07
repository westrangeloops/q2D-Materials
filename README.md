![Github_portada](https://github.com/westrangeloops/SVC-Materials/blob/main/Logos/Github_portada.png)

# SVC-Materials: Computational Framework for Quasi-2D Perovskite Structure Generation and Analysis

SVC-Materials is a comprehensive Python package for creating and analyzing quasi-2D perovskite structures, developed by the Theoretical Materials Science group at the University of Antioquia. The package provides tools for generating initial input structures for quasi-2D perovskite systems and offers extensive analysis capabilities for structural characterization.

## Quick Start

Get up and running with SVC-Materials in just a few steps:

### Installation
```bash
conda create -n SVC_Materials pip
conda activate SVC_Materials
pip install ase pandas numpy matplotlib networkx
```

### Basic Usage Example

```python
from SVC_materials.core.creator import q2D_creator
from SVC_materials.core.analyzer import q2D_analysis

# 1. Create a quasi-2D perovskite structure
creator = q2D_creator(
    B='Pb',                              # Metal center
    X='I',                               # Halogen
    molecule_xyz='molecule.xyz',         # Organic spacer molecule
    perov_vasp='perovskite.vasp',       # Base perovskite structure
    P1="6.88 -0.00 10.44",              # Position coordinates
    P2="5.49 -0.00  9.90", 
    P3="6.88 -0.00  4.49",
    Q1="0.94  5.94 10.44", 
    Q2="-0.46 5.94  9.90", 
    Q3="0.94  5.94  4.49",
    name='MyPerovskite',
    vac=10                               # Vacuum spacing
)

# Save the structure
creator.write_svc()

# 2. Analyze the structure
analyzer = q2D_analysis(B='Pb', X='I', crystal='svc_MyPerovskite.vasp')

# Get structural parameters
structure_data = analyzer.analyze_perovskite_structure()
print(f"Average bond length: {np.mean(structure_data['axial_lengths']):.3f} Å")

# Calculate organic spacer penetration
penetration = analyzer.calculate_n_penetration()

# Isolate and save components
analyzer.save_spacer()  # Save organic spacer only
analyzer.save_salt()    # Save spacer + halides
```

That's it! You've created and analyzed your first quasi-2D perovskite structure.

## Features Overview

- **Structure Generation**: Create quasi-2D perovskite structures with custom organic spacers
- **Structural Analysis**: Comprehensive analysis of perovskite octahedra including bond angles, lengths, and distortions
- **Molecule Isolation**: Extract and validate individual molecules from crystal structures
- **Deformation Analysis**: Quantitative comparison of molecular structures with ideal templates
- **Visualization Tools**: Built-in visualization for structural components
- **Multiple File Formats**: Support for VASP, XYZ, and other common formats

## Core Functionality

### Structure Generation
```python
from SVC_materials.core.creator import q2D_creator

# Create structures with precise molecular positioning
creator = q2D_creator(B='Pb', X='I', molecule_xyz='spacer.xyz', 
                     perov_vasp='base.vasp', P1=..., name='structure')
creator.write_svc()      # Save supercell with vacuum
creator.write_bulk()     # Save bulk structure with multiple slabs
```

### Structure Analysis
```python
from SVC_materials.core.analyzer import q2D_analysis

analyzer = q2D_analysis(B='Pb', X='I', crystal='structure.vasp')

# Comprehensive structural analysis
data = analyzer.analyze_perovskite_structure()
penetration = analyzer.calculate_n_penetration()

# Component isolation
analyzer.show_spacer()   # Visualize organic spacer
analyzer.save_salt()     # Save spacer + coordinating halides
```

### Molecule Isolation and Analysis
```python
from SVC_materials.utils.isolate_molecule import isolate_molecule, analyze_molecule_deformation

# Extract molecules from crystal structures
success = isolate_molecule(
    crystal_file="crystal.vasp",
    template_file="template.xyz", 
    output_file="extracted.xyz"
)

# Analyze molecular deformations
metrics = analyze_molecule_deformation(
    ideal_file='template.xyz',
    extracted_file='extracted.xyz',
    output_dir='analysis_results'
)
```

## Installation

We recommend using a conda environment:
```bash
conda create -n SVC_Materials pip
conda activate SVC_Materials
pip install ase pandas numpy matplotlib networkx
```

## Examples
Check the `examples` folder for detailed Jupyter notebooks demonstrating all features.

---

# Scientific Background and Detailed Documentation

## Theoretical Foundation

### Perovskite Crystal Structure

Quasi-two-dimensional (quasi-2D) perovskites represent a fascinating class of hybrid organic-inorganic materials with unique structural properties and applications in optoelectronics, photovoltaics, and quantum confinement systems. These materials are characterized by alternating layers of inorganic perovskite slabs and organic spacer molecules, creating quantum well structures with tunable electronic and optical properties.

The general formula for perovskites is ABX₃, where:
- **A**: Organic cation (methylammonium, formamidinium, or larger organic spacers)
- **B**: Metal cation (commonly Pb²⁺ or Sn²⁺)  
- **X**: Halide anion (I⁻, Br⁻, or Cl⁻)

In quasi-2D perovskites, the structure follows the Ruddlesden-Popper formula (L)₂(A)ₙ₋₁BₙX₃ₙ₊₁, where L represents the large organic spacer cation and n indicates the number of perovskite layers.

**Structural Hierarchy:**
- **Octahedral units**: BX₆ octahedra forming fundamental building blocks
- **Perovskite slabs**: Connected octahedra creating 2D inorganic layers
- **Organic spacers**: Large organic molecules separating inorganic layers
- **Quantum confinement**: Electronic properties modulated by slab thickness

### Structural Distortions and Their Significance

Perovskite structures exhibit various distortions that significantly influence their electronic and optical properties:

1. **Octahedral tilting**: Rotation of BX₆ octahedra about crystallographic axes
2. **Bond length variations**: Deviations from ideal B-X bond lengths due to steric effects
3. **Bond angle distortions**: Departure from ideal 180° and 90° angles in octahedral geometry
4. **Out-of-plane distortions**: Displacement of atoms from ideal planar arrangements

These distortions are quantitatively characterized through statistical analysis of bond lengths, bond angles, and geometric parameters, providing insights into structural stability and electronic properties.

## Computational Methodology

### Structure Generation Algorithm

The SVC-Materials package implements sophisticated algorithms for generating quasi-2D perovskite structures through precise computational steps:

#### Molecular Alignment and Positioning

The structure generation process begins with precise positioning of organic spacer molecules within the perovskite framework using coordinate transformation methodology:

1. **Template molecule preparation**: Organic spacer molecules are loaded and aligned based on nitrogen atom positions, serving as anchoring points
2. **Coordinate transformation**: A three-point transformation system (P1, P2, P3, Q1, Q2, Q3) defines desired positions for molecular placement
3. **Geometric optimization**: Molecules are positioned to minimize steric clashes while maintaining appropriate intermolecular distances

The mathematical framework utilizes homogeneous coordinates and rotation matrices:

**r**_new = **R** · **r**_original + **t**

where **R** represents the rotation matrix and **t** is the translation vector.

#### Supercell Construction

The package constructs supercells through a systematic approach:
- **Perovskite slab generation**: Creation of inorganic layers with specified thickness
- **Vacuum layer insertion**: Addition of controlled vacuum spacing for surface calculations
- **Periodic boundary conditions**: Proper handling of crystallographic periodicity

## Module Architecture and Scientific Implementation

### Core Modules

#### q2D_creator Module

The `q2D_creator` class serves as the primary structure generation engine, implementing advanced scientific methodologies:

**Molecular Transformation Engine:**
- **Nitrogen alignment**: Automatic detection and alignment of nitrogen atoms as structural anchors
- **Rotational degrees of freedom**: Implementation of Euler angle rotations for molecular orientation
- **Inclination control**: Precise control of molecular tilt angles relative to perovskite planes

**Structure Assembly Protocol:**
1. Perovskite template loading and validation
2. Organic molecule preprocessing and alignment
3. Coordinate system transformation and molecular positioning
4. Supercell construction with appropriate boundary conditions
5. Structure optimization and validation

#### q2D_analysis Module

The analysis module provides comprehensive structural characterization through advanced computational algorithms:

**Perovskite Structure Analysis:**

The module implements sophisticated neighbor-finding algorithms using ASE neighbor list functionality:
- **Octahedral identification**: Automatic detection of BX₆ coordination environments
- **Bond length analysis**: Statistical characterization of B-X bond distances
- **Bond angle calculations**: Comprehensive analysis of X-B-X angles in octahedral geometry
- **Distortion quantification**: Mathematical characterization of structural deviations

The algorithm employs a cutoff-distance approach (typically 3.5 Å) for neighbor identification. For each metal center B, the analysis identifies six halide neighbors X and categorizes them into axial and equatorial positions.

**Octahedral Distortion Analysis:**

The package quantifies octahedral distortions through several metrics:

1. **Bond length variance**: σ²_BL = (1/N)Σ(dᵢ - ⟨d⟩)²
2. **Bond angle variance**: σ²_BA = (1/N)Σ(θᵢ - ⟨θ⟩)²
3. **Out-of-plane distortion**: Quantification of atomic displacement from ideal planar geometry

where dᵢ represents individual bond lengths, θᵢ are bond angles, and ⟨·⟩ denotes ensemble averages.

**Nitrogen Penetration Analysis:**

A novel feature is the quantitative analysis of organic spacer penetration into inorganic layers:

1. Identifies halide planes through Z-coordinate clustering
2. Locates nitrogen atoms in organic spacers
3. Calculates penetration depths using plane-to-point distance formulas
4. Provides statistical analysis of penetration distributions

The penetration depth d_pen is calculated as:
d_pen = |ax₀ + by₀ + cz₀ + d|/√(a² + b² + c²)

where (x₀, y₀, z₀) represents nitrogen atom coordinates and ax + by + cz + d = 0 defines the halide plane equation.

### Utility Modules

#### Molecular Isolation and Validation

The `isolate_molecule` module implements advanced graph-theoretical approaches for molecular extraction and validation:

**Connectivity Analysis:**
The module employs NetworkX graph algorithms to:
- Construct molecular connectivity graphs based on interatomic distances
- Perform graph isomorphism checks for template matching
- Validate chemical formulas and bonding patterns
- Identify molecular fragments within crystal structures

**Template-Based Validation:**
1. **Size matching**: Comparison of molecular dimensions with template structures
2. **Chemical formula verification**: Validation of elemental composition
3. **Connectivity analysis**: Graph-based comparison of bonding patterns
4. **Bond length validation**: Assessment of realistic interatomic distances

#### Molecular Deformation Analysis

The package provides quantitative analysis of molecular deformations through comprehensive geometric comparisons:

**Structural Metrics:**
- **Bond length MAE**: MAE_BL = (1/N)Σ|dᵢ^ideal - dᵢ^extracted|
- **Bond angle MAE**: MAE_BA = (1/N)Σ|θᵢ^ideal - θᵢ^extracted|
- **Dihedral angle MAE**: MAE_DA = (1/N)Σ|φᵢ^ideal - φᵢ^extracted|

**Deformation Classification:**

The package implements a classification system for structural deformations:

**Bond length deformations:**
- Normal fluctuations: MAE < 0.05 Å
- Significant strain: 0.05 ≤ MAE < 0.2 Å
- Severe strain: MAE ≥ 0.2 Å

**Bond angle deformations:**
- Normal flexibility: MAE < 5°
- Moderate strain: 5° ≤ MAE < 15°
- Severe strain: MAE ≥ 15°

**Dihedral angle deformations:**
- Local oscillation: MAE < 30°
- Moderate conformational change: 30° ≤ MAE < 90°
- Major conformational change: MAE ≥ 90°

### Molecule Deformation Analysis

The package provides detailed analysis of molecular deformations by comparing isolated molecules with their ideal templates:

1. **Bond Length Analysis**: Calculates Mean Absolute Error (MAE) in bond lengths with classification as normal fluctuations (< 0.05 Å), significant strain (0.05-0.2 Å), or severe strain (> 0.2 Å)

2. **Bond Angle Analysis**: Calculates MAE in bond angles with classification as normal flexibility (< 5°), moderate strain (5-15°), or severe strain (> 15°)

3. **Dihedral Angle Analysis**: Calculates MAE in dihedral angles with classification as local oscillation (< 30°), moderate conformational change (30-90°), or major conformational change (> 90°)

### Output Files

For each analyzed molecule, the following files are generated:

1. **analysis.log**: Detailed summary of deformation analysis
2. **bonds.csv**: Detailed bond length analysis
3. **angles.csv**: Detailed bond angle analysis
4. **dihedrals.csv**: Detailed dihedral angle analysis
5. **deformation_summary.csv**: Overall analysis summary

### Interpretation Guidelines

1. **Bond Length Deviations**
   - Normal fluctuations (< 0.05 Å) indicate stable structure
   - Significant strain (0.05 - 0.2 Å) may indicate local stress
   - Severe strain (> 0.2 Å) suggests structural issues

2. **Bond Angle Deviations**
   - Normal flexibility (< 5°) is expected in most structures
   - Moderate strain (5° - 15°) may affect local geometry
   - Severe strain (> 15°) indicates significant distortion

3. **Dihedral Angle Deviations**
   - Local oscillation (< 30°) represents normal conformational flexibility
   - Moderate changes (30° - 90°) may indicate transitions between states
   - Major changes (> 90°) suggest significant structural reorganization

## Scientific Applications and Validation

### Structure-Property Relationships

The SVC-Materials package enables systematic investigation of structure-property relationships in quasi-2D perovskites:

1. **Electronic band gap tuning**: Correlation between structural parameters and electronic properties
2. **Optical absorption analysis**: Relationship between quantum confinement and optical properties
3. **Stability assessment**: Evaluation of structural stability through distortion analysis
4. **Phase transition studies**: Investigation of temperature-dependent structural changes

### Computational Efficiency and Scalability

The package is designed for computational efficiency through:
- **Vectorized operations**: Extensive use of NumPy for mathematical operations
- **Efficient neighbor finding**: Optimized algorithms for identifying atomic neighbors
- **Parallel processing capabilities**: Support for concurrent analysis of multiple structures
- **Memory optimization**: Efficient data structures for large-scale calculations

## Quality Assurance and Validation Protocols

### Structural Validation

The package implements comprehensive validation protocols:
1. **Geometric consistency checks**: Verification of reasonable bond lengths and angles
2. **Chemical formula validation**: Confirmation of correct stoichiometry
3. **Connectivity verification**: Validation of molecular bonding patterns
4. **Crystallographic validation**: Assessment of unit cell parameters and symmetry

### Error Handling and Diagnostics

Robust error handling includes:
- **Input validation**: Comprehensive checking of input parameters and file formats
- **Numerical stability**: Protection against division by zero and numerical overflow
- **Diagnostic output**: Detailed logging and debugging information
- **Graceful failure modes**: Appropriate handling of exceptional conditions

## File Handling and Data Management

The `file_handlers` module provides robust I/O capabilities for various crystallographic file formats:
- **VASP format support**: Complete reading and writing of POSCAR/CONTCAR files
- **XYZ format handling**: Support for molecular coordinate files
- **Data validation**: Automatic verification of file integrity and format compliance
- **Coordinate system management**: Proper handling of Cartesian and fractional coordinates

## Future Developments and Extensions

### Planned Enhancements

Future development directions include:
1. **Machine learning integration**: Implementation of ML-based property prediction
2. **High-throughput screening**: Automated generation and analysis of large structure databases
3. **Dynamic simulations**: Integration with molecular dynamics simulation capabilities
4. **Advanced visualization**: Enhanced 3D visualization and analysis tools

### Community Contributions

The package is designed to accommodate community contributions through:
- **Modular architecture**: Easy integration of new analysis methods
- **Standardized interfaces**: Consistent API design for extensibility
- **Documentation standards**: Comprehensive documentation for developers
- **Testing frameworks**: Robust unit testing and validation procedures

## Contributing
We welcome contributions to SVC-Materials! If you would like to contribute:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License
SVC-Materials is licensed under the MIT license.

## Conclusion

The SVC-Materials package represents a comprehensive computational framework for the generation and analysis of quasi-2D perovskite structures. Through its sophisticated algorithms, robust validation protocols, and extensive analysis capabilities, it provides researchers with powerful tools for investigating the structural and electronic properties of these fascinating materials.

The scientific rigor implemented throughout the package, from fundamental algorithms to advanced analysis methods, ensures reliable and reproducible results for materials science research. The modular architecture and extensible design facilitate future developments and community contributions, positioning SVC-Materials as a valuable resource for the growing field of hybrid perovskite research.

The package's ability to bridge the gap between theoretical predictions and experimental observations through detailed structural analysis makes it an indispensable tool for understanding the complex relationships between structure, dynamics, and properties in quasi-2D perovskite materials.
