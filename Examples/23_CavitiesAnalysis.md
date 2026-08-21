# Example 23: Cavity Detection and Analysis

This example demonstrates the cavity detection and analysis system for quasi-2D perovskite structures. The system identifies cuboctahedral cavities hosting A-site cations and spacer molecules, and provides comprehensive deformation metrics.

## Overview

The q2D analyzer provides:

- **Geometric Classification**: Identifies cavities by cuboctahedral structure (8B + 12X for A-sites, 4B + 8X for RP spacers, 8B + 16X for DJ spacers)
- **Volume Calculation**: Convex hull volume of cavity walls
- **Penetration Depth**: NH3 group penetration into terminal atom planes (spacers only)
- **Deformation Metrics**: Six metrics (η, κ, ν, Ω, ΔV, shear) quantifying deviation from ideal geometry
- **Shearing Deformation**: Octagonal ring centroid displacement measuring asymmetric cage distortion
- **PBC-Aware**: Handles periodic boundary conditions correctly
- **Graph-Based**: All data extracted from isolated cavity subgraphs

## Basic Usage

```python
from q2D_Materials.analyzer import q2D_analyzer

# Load and analyze structure
analyzer = q2D_analyzer("structure.cif")
analyzer.analyze()

# Get cavities
cavities = analyzer.get_cavities()  # Returns CavityCollection
print(f"Found {len(cavities)} cavities")

# Access each cavity
for i, cavity in enumerate(cavities):
    print(f"\nCavity {i} ({cavity.id}):")
    print(f"  Type: {cavity.cavity_type}")
    print(f"  Geometry: {len(cavity.b_atom_indices)}B + {len(cavity.x_atom_indices)}X")
    print(f"  Volume: {cavity.get_volume():.2f} Ų")
```

## Cavity Types

- **A-site cavities** (`cavity_type='a_site'`): Closed cuboctahedra (8B + 12X) hosting A-site cations
- **RP spacer cavities** (`cavity_type='spacer_rp'`): Open antiprisms (4B + 8X) hosting RP spacers
- **DJ spacer cavities** (`cavity_type='spacer_dj'`): Open antiprisms (8B + 16X) spanning two layers, hosting DJ spacers

## Cavity Volume

Volume is calculated using the convex hull of all X atoms forming the cavity walls. Positions use PBC-unwrapped coordinates.

```python
volume = cavity.get_volume()  # Returns float in cubic Angstroms (Ų)
```

## Penetration Depth (Spacers Only)

Measures how deeply NH3 groups penetrate into terminal atom planes.

### Formula

For RP spacers (1 NH3):
```
penetration = signed_distance(NH3_center, terminal_plane)
```

For DJ spacers (2 NH3):
```
top_penetration = signed_distance(NH3_top, terminal_plane_top)
bottom_penetration = signed_distance(NH3_bottom, terminal_plane_bottom)
```

The terminal plane is fitted using SVD through 4 terminal X atoms.

### Interpretation

- **Negative values**: NH3 below plane (penetrating into layer)
- **Positive values**: NH3 above plane (protruding from layer)
- **Zero**: NH3 exactly at plane

### API

```python
penetration = cavity.calculate_penetration_depth()
# For RP: {'penetration_depth': float, 'nh3_position': array, ...}
# For DJ: {'top_penetration': float, 'bottom_penetration': float, ...}
```

## Deformation Metrics

Six metrics quantify deviation from ideal cuboctahedral (A-site) or antiprism (spacer) geometry.

### 1. Eta (η) - Square Face Distortion

**Formula:**
```
η = 90° - mean(|square_angle - 90°|)
```

**Interpretation:**
- Higher values (closer to 90°): Less distortion, ideal square faces
- Lower values: Greater distortion

### 2. Kappa (κ) - Triangular Face Distortion

**Formula:**

For cuboctahedra (A-site):
```
κ = 60° - mean(|equilateral_triangle_angle - 60°|)
```

For antiprisms (spacers):
```
κ_equilateral = 60° - mean(|equilateral_triangle_angle - 60°|)
κ_isosceles = 60° - mean(|isosceles_triangle_angle - 60°|)
```

**Interpretation:**
- Higher values (closer to 60°): Less distortion, ideal triangular faces
- Lower values: Greater distortion

### 3. Nu (ν) - Cation Displacement

**Formula:**
```
ν = ||cation_position - cage_geometric_center||
```

**Interpretation:**
- Lower values: Better centering
- Units: Angstroms (Å)
- For DJ spacers: Measured relative to each half-cage center separately

### 4. Omega (Ω) - Edge Length Distortion

**Formula:**
```
Ω = mean(|X-X_edge_length - ideal_length|)
ideal_length = BX_distance × √2
```

**Interpretation:**
- Lower values: Less edge distortion
- Units: Angstroms (Å)
- **Note**: X-X edges are geometric polyhedral edges, NOT chemical bonds

### 5. Delta Volume (ΔV) - Volume Deviation

**Formula:**

For cuboctahedra:
```
ΔV = actual_volume - ideal_volume
ideal_volume = (20/3) × d³
where d = BX_distance
```

For antiprisms:
```
ΔV = actual_volume - ideal_volume
ideal_volume = (10/3) × d³
where d = BX_distance
```

**Interpretation:**
- Negative values: Smaller than ideal
- Positive values: Larger than ideal
- Units: Cubic Angstroms (Ų)

### 6. Shearing Deformation - Octagonal Ring Displacement

**Formula:**

For antiprisms (spacers):
```
Octagon 1: 4 B atoms + 4 equatorial X atoms
Octagon 2: 4 B atoms + 4 terminal X atoms

shear_vector = centroid(Octagon2) - centroid(Octagon1)
shear_magnitude = ||shear_vector||
shear_direction = shear_vector / shear_magnitude
```

For cuboctahedra (A-sites):
```
Identifies parallel octagonal rings from structure
(may return NaN if rings cannot be identified)
```

**Interpretation:**
- **Magnitude**: Distance between octagonal ring centroids
  - Lower values: Better alignment, less shearing
  - Higher values: Greater asymmetric distortion
- **Direction**: 3D normalized vector showing shearing direction
  - Useful for understanding the direction of cage distortion
- Units: Angstroms (Å)

**Physical Meaning:**
This metric captures asymmetric distortion where the two octagonal rings (defined by B atoms and equatorial/terminal X atoms) are displaced relative to each other. This is distinct from symmetric expansion/contraction (captured by ΔV) and measures cage "twisting" or "sliding" deformation.

### API

```python
metrics = cavity.get_deformation(
    b_cation: Optional[Union[str, List[str]]] = None,
    x_anion: Optional[Union[str, List[str]]] = None,
    bx_distance: Optional[float] = None,
    mode: str = 'delta',
    full: bool = False,
) -> Dict
```

**Parameters:**
- `b_cation`: B-site species (e.g., `'Pb'` or `['Pb', 'Sn']`). If `None`, auto-detected from subgraph
- `x_anion`: X-site species (e.g., `'Br'` or `['Br', 'I']`). If `None`, auto-detected from subgraph
- `bx_distance`: B-X bond distance in Å. If `None`, calculated from species using structural constants
- `mode`: `'delta'` (default) returns deviations from ideal, `'absolute'` returns raw measurements
- `full`: If `True`, includes per-face and per-edge detailed data

**Returns:**
- **Delta mode** (default):
  ```python
  {
      'eta': float,
      'kappa': float or {'kappa_equilateral': float, 'kappa_isosceles': float},
      'nu': float,
      'omega': float,
      'delta_volume': float,
      'shear_magnitude': float,
      'shear_vector': [x, y, z],
      'shear_direction': [x, y, z]
  }
  ```
- **Absolute mode**: `{'square_angles': {...}, 'equilateral_angles': {...}, 'nu': float, 'edge_lengths': {...}, 'volume': float, ...}`

## Complete Example

```python
from q2D_Materials.analyzer import q2D_analyzer

# Load and analyze structure
analyzer = q2D_analyzer("dj_structure.vasp")
analyzer.analyze()

# Get cavities
cavities = analyzer.get_cavities()
print(f"Found {len(cavities)} cavities\n")

# Analyze each cavity
for i, cavity in enumerate(cavities):
    print(f"{'='*70}")
    print(f"Cavity {i} ({cavity.cavity_type}):")
    print(f"{'='*70}")
    
    # Basic info
    print(f"ID: {cavity.id}")
    print(f"Geometry: {len(cavity.b_atom_indices)}B + {len(cavity.x_atom_indices)}X")
    print(f"Volume: {cavity.get_volume():.2f} Ų")
    
    # Penetration depth (spacers only)
    if cavity.cavity_type in ('spacer_rp', 'spacer_dj'):
        penetration = cavity.calculate_penetration_depth()
        if cavity.cavity_type == 'spacer_rp':
            print(f"Penetration: {penetration['penetration_depth']:.3f} Å")
        else:  # DJ
            print(f"Top penetration: {penetration['top_penetration']:.3f} Å")
            print(f"Bottom penetration: {penetration['bottom_penetration']:.3f} Å")
    
    # Deformation metrics
    metrics = cavity.get_deformation()
    print(f"\nDeformation Metrics:")
    print(f"  Eta (η): {metrics['eta']:.2f}°")
    
    if cavity.cavity_type == 'a_site':
        print(f"  Kappa (κ): {metrics['kappa']:.2f}°")
    else:  # spacer
        print(f"  Kappa equilateral (κ_eq): {metrics['kappa_equilateral']:.2f}°")
        print(f"  Kappa isosceles (κ_iso): {metrics['kappa_isosceles']:.2f}°")
    
    print(f"  Nu (ν): {metrics['nu']:.3f} Å")
    print(f"  Omega (Ω): {metrics['omega']:.3f} Å")
    print(f"  Delta Volume (ΔV): {metrics['delta_volume']:.2f} Ų")
    print(f"  Shear Magnitude: {metrics['shear_magnitude']:.3f} Å")
    print(f"  Shear Direction: [{metrics['shear_direction'][0]:.3f}, {metrics['shear_direction'][1]:.3f}, {metrics['shear_direction'][2]:.3f}]")

    # Interpretation
    print(f"\nInterpretation:")
    if metrics['eta'] > 88:
        print(f"  ✓ Square faces close to ideal (η = {metrics['eta']:.1f}°)")
    else:
        print(f"  ⚠ Square faces distorted (η = {metrics['eta']:.1f}°)")
    
    if cavity.cavity_type == 'a_site':
        if metrics['kappa'] > 58:
            print(f"  ✓ Triangle faces close to ideal (κ = {metrics['kappa']:.1f}°)")
        else:
            print(f"  ⚠ Triangle faces distorted (κ = {metrics['kappa']:.1f}°)")
    else:
        if metrics['kappa_equilateral'] > 58:
            print(f"  ✓ Equilateral triangles close to ideal (κ_eq = {metrics['kappa_equilateral']:.1f}°)")
        else:
            print(f"  ⚠ Equilateral triangles distorted (κ_eq = {metrics['kappa_equilateral']:.1f}°)")
        if metrics['kappa_isosceles'] > 58:
            print(f"  ✓ Isosceles triangles close to ideal (κ_iso = {metrics['kappa_isosceles']:.1f}°)")
        else:
            print(f"  ⚠ Isosceles triangles distorted (κ_iso = {metrics['kappa_isosceles']:.1f}°)")
    
    if metrics['nu'] < 0.2:
        print(f"  ✓ Cation well-centered (ν = {metrics['nu']:.3f} Å)")
    else:
        print(f"  ⚠ Cation off-center (ν = {metrics['nu']:.3f} Å)")
    
    if abs(metrics['delta_volume']) < 5:
        print(f"  ✓ Volume close to ideal (ΔV = {metrics['delta_volume']:.1f} Ų)")
    else:
        print(f"  ⚠ Volume deviates from ideal (ΔV = {metrics['delta_volume']:.1f} Ų)")

    if metrics['shear_magnitude'] < 0.2:
        print(f"  ✓ Minimal shearing deformation (shear = {metrics['shear_magnitude']:.3f} Å)")
    else:
        print(f"  ⚠ Significant shearing (shear = {metrics['shear_magnitude']:.3f} Å)")
    print()
```

### Example Output

```
Found 2 cavities

======================================================================
Cavity 0 (spacer_dj):
======================================================================
ID: cavity_0
Geometry: 8B + 16X
Volume: 425.67 Ų
Top penetration: 0.603 Å
Bottom penetration: -0.603 Å

Deformation Metrics:
  Eta (η): 87.8°
  Kappa equilateral (κ_eq): 59.1°
  Kappa isosceles (κ_iso): 58.7°
  Nu (ν): 0.32 Å
  Omega (Ω): 0.12 Å
  Delta Volume (ΔV): -5.1 Ų
  Shear Magnitude: 0.15 Å
  Shear Direction: [0.021, -0.035, 0.999]

Interpretation:
  ✓ Square faces close to ideal (η = 87.8°)
  ✓ Equilateral triangles close to ideal (κ_eq = 59.1°)
  ✓ Isosceles triangles close to ideal (κ_iso = 58.7°)
  ⚠ Cation off-center (ν = 0.32 Å)
  ✓ Volume close to ideal (ΔV = -5.1 Ų)
  ✓ Minimal shearing deformation (shear = 0.150 Å)

======================================================================
Cavity 1 (a_site):
======================================================================
ID: cavity_1
Geometry: 8B + 12X
Volume: 312.45 Ų

Deformation Metrics:
  Eta (η): 88.5°
  Kappa (κ): 59.2°
  Nu (ν): 0.15 Å
  Omega (Ω): 0.08 Å
  Delta Volume (ΔV): -2.3 Ų
  Shear Magnitude: 0.09 Å
  Shear Direction: [0.012, 0.005, 0.999]

Interpretation:
  ✓ Square faces close to ideal (η = 88.5°)
  ✓ Triangle faces close to ideal (κ = 59.2°)
  ✓ Cation well-centered (ν = 0.15 Å)
  ✓ Volume close to ideal (ΔV = -2.3 Ų)
  ✓ Minimal shearing deformation (shear = 0.090 Å)
```

## API Summary

### Analyzer Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `analyze()` | Build structural graph | - | - |
| `get_cavities()` | Get all cavities | - | `CavityCollection` |

### Cavity Object Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `get_volume()` | Get cavity volume | - | `float` (Ų) |
| `get_deformation()` | Get deformation metrics | `b_cation`, `x_anion`, `bx_distance`, `mode='delta'`, `full=False` | `dict` |
| `calculate_penetration_depth()` | NH3 penetration (spacers only) | - | `dict` |
| `is_point_inside(point)` | Check point containment | `point: np.ndarray` | `bool` |
| `to_atoms()` | Convert to ASE Atoms | - | `ase.Atoms` |

### Cavity Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `id` | `str` | Cavity identifier (e.g., `'cavity_0'`) |
| `cavity_type` | `str` | `'a_site'`, `'spacer_rp'`, or `'spacer_dj'` |
| `b_atom_indices` | `List[int]` | B-site atom indices |
| `x_atom_indices` | `List[int]` | X-site atom indices |
| `a_site_indices` | `List[int]` | A-site/molecule atom indices |
| `center_position` | `np.ndarray` | Cavity center coordinates |
| `subgraph` | `nx.Graph` | Isolated cavity subgraph |

### CavityCollection Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `filter(**kwargs)` | Filter cavities | Attribute filters | `CavityCollection` |
| `to_atoms()` | Convert all to ASE Atoms | - | `List[ase.Atoms]` |

## Graph Structure for Cavity Analysis

Cavity subgraphs are isolated subgraphs extracted from the parent structural graph with hierarchical structure:

### A-Site Cavity (Cuboctahedron)

```
A_Site Node (a_site_*)
  ├── CONTAINS → Molecule Atoms (atom_*_img_*)
  └── CONTAINS ← Cage (cage_0)
                    └── CONTAINS (role='corner') → B Atoms (atom_*_img_*)
                                                      └── BONDED_TO (role='ligand') → X Atoms (atom_*_img_*)
```

### RP Spacer Cavity (Square Antiprism)

```
Anchor Node (anchor_0)
  ├── CONTAINS → NH3 Atoms (N + 3H)
  └── CONTAINS ← Half_Cage (half_cage_0)
                    └── CONTAINS (role='corner') → B Atoms (atom_*_img_*)
                                                      └── BONDED_TO (role='ligand') → X Atoms (atom_*_img_*)
```

### DJ Spacer Cavity (Two Square Antiprisms)

```
Anchor_0 (anchor_0)
  ├── CONTAINS → NH3_0 Atoms (N + 3H)
  └── CONTAINS ← Half_Cage_0 (half_cage_0)
                    └── CONTAINS (role='corner') → B Atoms (atom_*_img_*)
                                                      └── BONDED_TO (role='ligand') → X Atoms (atom_*_img_*)

Anchor_1 (anchor_1)
  ├── CONTAINS → NH3_1 Atoms (N + 3H)
  └── CONTAINS (role='half_cage') → Half_Cage_1 (half_cage_1)
                                        └── CONTAINS (role='corner') → B Atoms (atom_*_img_*)
                                                                          └── BONDED_TO (role='ligand') → X Atoms (atom_*_img_*)
```

**Key Properties:**
- **Coordinates**: All positions use `pbc_position` (PBC-unwrapped), NOT `x`, `y`, `z`
- **Node IDs**: Format `atom_{idx}_img_{i}_{j}_{k}` for PBC image uniqueness
- **Cage metadata**: `cage`/`half_cage` nodes have `layer_id` property
- **B-X connectivity**: Direct `BONDED_TO` edges (no octahedron nodes in subgraph)
- **Multiple PBC images**: Atoms can appear multiple times with different `image_label` values

**Accessing Subgraph:**
```python
subgraph = cavity.subgraph
for node in subgraph.nodes():
    node_data = subgraph.nodes[node]
    if node_data.get('node_type') == 'cage':
        layer_id = node_data.get('layer_id')
        # Use pbc_position for coordinates, not x/y/z
```
# Configuring Cavity Detection Weights

The cavity detection module uses weighted scoring to select the most appropriate X atoms for cavity construction. You can now control these weights to fine-tune the detection based on your specific needs.

## Weight Classes

### ASiteWeights
Controls X atom selection for A-site cavities (cuboctahedra).

**Scoring formula:**
```
score = w_dist_anchor * dist_anchor + w_z_diff * z_diff
```

**Parameters:**
- `w_dist_anchor` (float): Weight for distance to A-site geometric center. Default: 0.7
- `w_z_diff` (float): Weight for Z-coordinate difference from anchor. Default: 0.3

**Usage:**
```python
from q2D_Materials.analyzer.cavities_processing import ASiteWeights

# Use default weights
default_weights = ASiteWeights()

# Custom weights: prioritize distance to anchor more heavily
custom_weights = ASiteWeights(
    w_dist_anchor=0.8,
    w_z_diff=0.2
)

# Custom weights: prioritize Z-coordinate alignment
z_focused_weights = ASiteWeights(
    w_dist_anchor=0.5,
    w_z_diff=0.5
)
```

### SpacerWeights
Controls terminal X atom selection for spacer cavities (square antiprisms).

**Scoring formula:**
```
score = w_dist_anchor * dist_anchor +
        w_dist_b_center * dist_b_center +
        w_z_diff * z_diff
```

**Parameters:**
- `w_dist_anchor` (float): Weight for distance to NH3 N atom anchor. Default: 0.4
- `w_dist_b_center` (float): Weight for distance to B atom geometric center. Default: 0.3
- `w_z_diff` (float): Weight for Z-coordinate difference from anchor. Default: 0.3

**Usage:**
```python
from q2D_Materials.analyzer.cavities_processing import SpacerWeights

# Use default weights
default_weights = SpacerWeights()

# Custom weights: prioritize anchor distance
anchor_focused_weights = SpacerWeights(
    w_dist_anchor=0.6,
    w_dist_b_center=0.2,
    w_z_diff=0.2
)

# Custom weights: prioritize B-center alignment
b_center_focused_weights = SpacerWeights(
    w_dist_anchor=0.2,
    w_dist_b_center=0.6,
    w_z_diff=0.2
)

# Custom weights: prioritize Z-coordinate alignment
z_focused_weights = SpacerWeights(
    w_dist_anchor=0.35,
    w_dist_b_center=0.25,
    w_z_diff=0.4
)
```

## Using Custom Weights with Cavity Detection

### Example 1: Using Default Weights
```python
from q2D_Materials.analyzer import q2D_analyzer

# Create analyzer
analyzer = q2D_analyzer('structure.vasp')

# Detect cavities with default weights
cavities = analyzer.get_cavities()
```

### Example 2: Custom A-Site Weights Only
```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.cavities_processing import ASiteWeights

# Create analyzer
analyzer = q2D_analyzer('structure.vasp')

# Define custom A-site weights
a_site_weights = ASiteWeights(
    w_dist_anchor=0.8,
    w_z_diff=0.2
)

# Detect cavities with custom A-site weights
# (spacer weights use defaults)
cavities = analyzer.get_cavities(a_site_weights=a_site_weights)
```

### Example 3: Custom Spacer Weights Only
```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.cavities_processing import SpacerWeights

# Create analyzer
analyzer = q2D_analyzer('structure.vasp')

# Define custom spacer weights
spacer_weights = SpacerWeights(
    w_dist_anchor=0.6,
    w_dist_b_center=0.2,
    w_z_diff=0.2
)

# Detect cavities with custom spacer weights
# (A-site weights use defaults)
cavities = analyzer.get_cavities(spacer_weights=spacer_weights)
```

### Example 4: Custom Weights for Both
```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.cavities_processing import ASiteWeights, SpacerWeights

# Create analyzer
analyzer = q2D_analyzer('structure.vasp')

# Define custom weights for both cavity types
a_site_weights = ASiteWeights(w_dist_anchor=0.8, w_z_diff=0.2)
spacer_weights = SpacerWeights(w_dist_anchor=0.6, w_dist_b_center=0.2, w_z_diff=0.2)

# Detect cavities with both custom weight sets
cavities = analyzer.get_cavities(
    a_site_weights=a_site_weights,
    spacer_weights=spacer_weights
)
```

### Example 5: Direct API Usage
```python
from q2D_Materials.analyzer.cavities_processing import detect_all_cavities, ASiteWeights, SpacerWeights

# Assuming you have graph, positions, symbols, and cell from somewhere
# graph = ...
# atom_positions = ...
# atom_symbols = ...
# cell = ...

# Define custom weights
a_site_weights = ASiteWeights(w_dist_anchor=0.75, w_z_diff=0.25)
spacer_weights = SpacerWeights(w_dist_anchor=0.5, w_dist_b_center=0.3, w_z_diff=0.2)

# Detect cavities with custom weights
cavities = detect_all_cavities(
    graph=graph,
    atom_positions=atom_positions,
    atom_symbols=atom_symbols,
    cell=cell,
    a_site_weights=a_site_weights,
    spacer_weights=spacer_weights
)
```

## Understanding the Weights

### A-Site Weights

**w_dist_anchor (distance to anchor):**
- Higher values prioritize X atoms closer to the A-site geometric center
- Use higher values when you want tighter, more compact cavities
- Default: 0.7 (strong preference for proximity)

**w_z_diff (Z-coordinate difference):**
- Higher values prioritize X atoms at similar Z-coordinates as the anchor
- Use higher values for layered structures where Z-alignment is critical
- Default: 0.3 (moderate consideration)

### Spacer Weights

**w_dist_anchor (distance to NH3 anchor):**
- Higher values prioritize X atoms closer to the NH3 group
- Use higher values when molecular positioning is critical
- Default: 0.4 (balanced with other factors)

**w_dist_b_center (distance to B-center):**
- Higher values prioritize X atoms closer to the geometric center of B atoms
- Use higher values for symmetric, centered cavities
- Default: 0.3 (moderate consideration)

**w_z_diff (Z-coordinate difference):**
- Higher values prioritize X atoms at similar Z-coordinates as the NH3 anchor
- Use higher values for flat, layered spacer arrangements
- Default: 0.3 (moderate consideration)

## Tips for Tuning Weights

1. **Start with defaults**: The default weights work well for most standard perovskite structures.

2. **One parameter at a time**: Change one weight at a time to understand its effect.

3. **Sum to 1.0**: While not strictly required, weights that sum to 1.0 are easier to interpret as relative importance percentages.

4. **Validation**: The weight classes will warn you if weights don't sum to 1.0.

5. **Iterative refinement**: Visualize cavity detection results and adjust weights based on observed issues.

## Common Use Cases

### Highly distorted structures
For structures with significant distortions, you might want to prioritize geometric proximity:
```python
# A-sites: prioritize distance over Z-alignment
a_site_weights = ASiteWeights(w_dist_anchor=0.9, w_z_diff=0.1)

# Spacers: prioritize B-center alignment
spacer_weights = SpacerWeights(w_dist_anchor=0.4, w_dist_b_center=0.5, w_z_diff=0.1)
```

### Perfectly layered structures
For ideal layered perovskites, you might prioritize Z-coordinate alignment:
```python
# A-sites: balance distance and Z-alignment
a_site_weights = ASiteWeights(w_dist_anchor=0.5, w_z_diff=0.5)

# Spacers: prioritize Z-alignment
spacer_weights = SpacerWeights(w_dist_anchor=0.3, w_dist_b_center=0.3, w_z_diff=0.4)
```

### Molecular spacers with off-center NH3
For structures where NH3 groups are not perfectly centered:
```python
# Spacers: reduce anchor weight, increase B-center weight
spacer_weights = SpacerWeights(w_dist_anchor=0.2, w_dist_b_center=0.5, w_z_diff=0.3)
```
