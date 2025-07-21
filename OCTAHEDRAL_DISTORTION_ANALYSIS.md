# Octahedral Distortion Analysis Documentation

## Overview

The enhanced `q2D_analyzer` class now includes comprehensive octahedral distortion analysis capabilities. This functionality allows you to:

1. **Systematically order octahedra** for consistent comparison between structures
2. **Calculate comprehensive distortion parameters** using the OctaDist library
3. **Compare distortions** between different structures
4. **Export results** for further analysis

## Key Features

### 1. Systematic Octahedra Ordering

The analyzer orders octahedra by:
- **Z coordinate first** (vertical position)
- **X coordinate second** (horizontal position)
- **Y coordinate third** (depth position)

This ensures that octahedra in different structures can be compared in a consistent manner.

### 2. Comprehensive Distortion Parameters

The analysis calculates the following distortion parameters:

#### Bond Distance Parameters
- **Bond distances**: Individual metal-ligand bond lengths
- **Mean bond distance**: Average of all bond lengths
- **Bond distance differences**: Deviations from mean bond length

#### Distortion Parameters
- **Zeta (ζ)**: Sum of absolute deviations from mean bond length
- **Delta (Δ)**: Tilting distortion parameter (normalized variance)
- **Sigma (Σ)**: Angular distortion parameter (sum of deviations from 90°)

#### Theta Parameters
- **Theta mean**: Average theta parameter across all faces
- **Theta min**: Minimum theta parameter
- **Theta max**: Maximum theta parameter
- **Eight theta**: Individual theta values for all 8 triangular faces

#### Angular Parameters
- **Cis angles**: 12 cis angles in the octahedron
- **Trans angles**: 3 trans angles in the octahedron

#### Geometric Parameters
- **Octahedral volume**: Volume of the octahedron
- **Is octahedral**: Quality check for octahedral geometry

## Usage Examples

### Basic Analysis

```python
from SVC_materials.core.analyzer import q2D_analyzer

# Initialize analyzer
analyzer = q2D_analyzer(
    file_path="your_structure.vasp",
    b='Pb',  # Central atom
    x='Cl',  # Ligand atom
    cutoff_ref_ligand=3.5  # Distance cutoff in Angstroms
)

# Calculate distortions
distortions = analyzer.calculate_octahedral_distortions()

# Print summary
analyzer.print_distortion_summary()
```

### Getting Summary Data

```python
# Get summary DataFrame
summary_df = analyzer.get_distortion_summary()
print(summary_df)

# Export to CSV
analyzer.export_distortion_data("distortion_results.csv")
```

### Detailed Analysis of Specific Octahedron

```python
# Get detailed information for octahedron at ordered index 0
octa_details = analyzer.get_octahedron_by_index(0)

print(f"Central atom: {octa_details['central_symbol']}")
print(f"Position: {octa_details['central_coord']}")
print(f"Zeta parameter: {octa_details['zeta']:.4f}")
print(f"Delta parameter: {octa_details['delta']:.6f}")
print(f"Sigma parameter: {octa_details['sigma']:.4f}°")
```

### Structure Comparison

```python
# Load two structures
analyzer1 = q2D_analyzer("structure1.vasp", b='Pb', x='Cl')
analyzer2 = q2D_analyzer("structure2.vasp", b='Pb', x='Cl')

# Compare distortions
comparison_df = analyzer1.compare_distortions(analyzer2)
print(comparison_df)

# Save comparison
comparison_df.to_csv("structure_comparison.csv", index=False)
```

## Distortion Parameters Explained

### Zeta Parameter (ζ)
- **Definition**: Sum of absolute deviations from mean bond length
- **Formula**: ζ = Σ|d_i - d_mean|
- **Units**: Angstroms
- **Interpretation**: Higher values indicate more bond length distortion

### Delta Parameter (Δ)
- **Definition**: Tilting distortion parameter
- **Formula**: Δ = (1/6) × Σ[(d_i - d_mean)/d_mean]²
- **Units**: Dimensionless
- **Interpretation**: Measures relative variance in bond lengths

### Sigma Parameter (Σ)
- **Definition**: Angular distortion parameter
- **Formula**: Σ = Σ|90° - θ_cis|
- **Units**: Degrees
- **Interpretation**: Sum of deviations from ideal 90° cis angles

### Theta Parameters
- **Definition**: Measures trigonal distortion
- **Units**: Degrees
- **Interpretation**: Deviation from ideal octahedral geometry

## Data Structure

### Ordered Octahedra Structure
Each octahedron in `ordered_octahedra` contains:
```python
{
    'ordered_index': int,      # Systematic ordering index
    'original_index': int,     # Original discovery index
    'global_index': int,       # Index in original structure
    'symbol': str,             # Central atom symbol
    'atom_octa': list,         # Atomic symbols in octahedron
    'coord_octa': array,       # Coordinates of octahedron
    'central_coord': array,    # Central atom coordinates
    'x': float, 'y': float, 'z': float  # Individual coordinates
}
```

### Distortion Data Structure
Each distortion analysis contains:
```python
{
    'ordered_index': int,
    'global_index': int,
    'central_symbol': str,
    'central_coord': list,
    'bond_distances': list,
    'mean_bond_distance': float,
    'bond_distance_differences': list,
    'zeta': float,
    'delta': float,
    'sigma': float,
    'theta_mean': float,
    'theta_min': float,
    'theta_max': float,
    'eight_theta': list,
    'cis_angles': list,
    'trans_angles': list,
    'octahedral_volume': float,
    'is_octahedral': bool,
    'atom_symbols': list,
    'coordinates': list
}
```

## Methods Reference

### Core Methods

#### `order_octahedra()`
Orders octahedra systematically for comparison.

#### `calculate_octahedral_distortions()`
Calculates comprehensive distortion parameters for all octahedra.

#### `get_distortion_summary()`
Returns a pandas DataFrame with key distortion parameters.

#### `compare_distortions(other_analyzer)`
Compares distortion parameters between two structures.

#### `export_distortion_data(filename=None)`
Exports distortion data to CSV file.

#### `get_octahedron_by_index(ordered_index)`
Gets detailed information about a specific octahedron.

#### `print_distortion_summary()`
Prints a formatted summary of distortion parameters.

## Best Practices

1. **Consistent Parameters**: Use the same `cutoff_ref_ligand` value when comparing structures
2. **Systematic Comparison**: Always use the ordered indices for structure comparison
3. **Quality Check**: Check the `is_octahedral` flag to ensure valid octahedral geometry
4. **Statistical Analysis**: Use the summary DataFrame for statistical analysis across multiple octahedra

## Troubleshooting

### Common Issues

1. **Non-octahedral structures**: Check the `cutoff_ref_ligand` parameter
2. **Missing octahedra**: Verify the central atom symbol (`b` parameter)
3. **Import errors**: Ensure the octadist module is properly installed

### Error Messages

- `"Non-octahedral structure around X at index Y"`: The geometry doesn't meet octahedral criteria
- `"No octahedron found with ordered index X"`: Invalid index requested
- `"index of the reference center atom must be equal or greater than zero"`: Invalid reference index

## References

The distortion parameters are based on established literature:

1. **Zeta parameter**: Buron-Le Cointe et al., Phys. Rev. B 85, 064114 (2012)
2. **Delta parameter**: Lufaso & Woodward, Acta Cryst. B60, 10-20 (2004)
3. **Sigma parameter**: McCusker et al., Inorg. Chem. 35, 2100 (1996)
4. **Theta parameter**: Marchivie et al., Acta Crystallogr. B61, 25 (2005) 