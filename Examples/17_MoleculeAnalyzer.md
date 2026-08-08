# Example 17: Molecular Analysis for Spacers

This example demonstrates comprehensive structural and geometric analysis of spacer molecules in quasi-2D perovskite structures. The `SpacerAnalysis` module provides quantitative metrics for molecular compression, volume, backbone structure, and compactness.

## Overview

The molecular analyzer extracts and analyzes spacer molecules from DJ and RP structures, computing:

- **Compression Metrics**: N-N distances, backbone paths, and compression factors
- **Volume Properties**: Van der Waals volume, convex hull volume, and radius of gyration
- **Backbone Identification**: Longest path detection and side chain analysis
- **Geometric Descriptors**: Bond lengths, angles, and dihedral angles

**Use cases**: Understanding spacer conformation, identifying compressed vs. extended chains, characterizing molecular size and compactness, analyzing backbone flexibility.

## Quick Start

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.molecular_processing import SpacerAnalysis
from q2D_Materials.modifier import GraphView

# Load and analyze structure
analyzer = q2D_analyzer("dj_structure.cif")
analyzer.analyze()

# Get spacer molecules
view = GraphView(analyzer)
spacer = view.spacers[0]

# Analyze spacer
analysis = SpacerAnalysis(spacer, analyzer).compute()

# Access properties
print(f"Compression factor: {analysis.compression:.3f}")
print(f"Radius of gyration: {analysis.radius_gyration:.2f} Å")
print(f"Van der Waals volume: {analysis.vdw_volume:.2f} Å³")
print(analysis)  # Pretty-printed summary
```

## Compression Analysis (DJ Spacers)

### Overview

For DJ (Dion-Jacobson) spacers with two terminal nitrogen groups, the analyzer computes metrics that characterize the molecular extension vs. compression state.

### Metrics

#### Euclidean Distance
Through-space N-N distance (Å):
```
d = ||r_N1 - r_N2||
```

where `r_N1`, `r_N2` are the positions of terminal nitrogen atoms.

#### Path Length
Sum of bond lengths along the backbone (Å):
```
L_path = Σ ||r_i+1 - r_i||
```

where the sum is over consecutive backbone atoms.

#### Ideal Extended Length
Expected length for a fully extended sp³ carbon chain (Å):
```
L_ideal = 0.85 × L_path
```

The factor 0.85 accounts for tetrahedral bond angles (~109.5°) in an all-trans conformation.

#### Compression
Absolute compression distance in Angstroms:
```
compression = L_ideal - d = (0.85 × L_path) - d
```

**Interpretation:**
- **0.0 Å**: Fully extended (euclidean matches ideal extended length for sp³ chain)
- **> 0 Å**: Compressed by X Angstroms (molecule shorter than expected)
- **< 0 Å**: Overstretched by |X| Angstroms (molecule longer than typical)

The factor 0.85 accounts for tetrahedral bond angles (109.5°) in an all-trans sp³ conformation.

### Example

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.molecular_processing import SpacerAnalysis
from q2D_Materials.modifier import GraphView

analyzer = q2D_analyzer("dj_structure.cif")
analyzer.analyze()
view = GraphView(analyzer)

# Analyze DJ spacer
spacer = view.spacers[0]
result = SpacerAnalysis(spacer, analyzer, spacer_type="DJ").compute()

# Compression metrics
print(f"Euclidean distance: {result.euclidean_distance:.2f} Å")
print(f"Path length: {result.path_length:.2f} Å")
print(f"Ideal extended: {result.ideal_extended_length:.2f} Å")
print(f"Compression: {result.compression:+.2f} Å")

# Check if compressed
if result.is_compressed:
    print(f"⚠ Spacer is compressed by {result.compression:.2f} Å")
else:
    print("✓ Spacer is extended or normal")
```

## Volume and Size Properties

### Van der Waals Volume

Sum of atomic volumes using van der Waals radii (Å³):
```
V_vdw = Σ (4/3) π r_vdw³
```

where `r_vdw` is the van der Waals radius for each atom.

### Convex Hull Volume

Volume of the smallest convex shape containing all atoms (Å³). Computed using `scipy.spatial.ConvexHull`.

**Note**: Requires at least 4 atoms. Returns 0.0 for planar or coplanar molecules.

### Radius of Gyration (R_g)

**New in v2.3**: Mass-weighted root-mean-square distance of atoms from the center of mass (Å):

```
R_g = sqrt(Σ(m_i × ||r_i - r_com||²) / Σ(m_i))

where:
- r_com = Σ(m_i × r_i) / Σ(m_i)  (center of mass)
- r_i = position of atom i
- m_i = mass of atom i
```

**Physical Interpretation:**
- Measures molecular **compactness** and **size**
- Larger R_g indicates more extended/spread-out molecules
- Smaller R_g indicates more compact/folded molecules
- Typical values: 2-10 Å for organic spacers
- Useful for comparing molecular conformations

**Advantages over other metrics:**
- Mass-weighted (accounts for heavy atoms like halogens)
- Rotationally invariant
- Well-defined for all molecule types (DJ, RP, A-site)
- Complements compression factor (which only applies to DJ spacers)

### Example

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.molecular_processing import SpacerAnalysis
from q2D_Materials.modifier import GraphView

analyzer = q2D_analyzer("structure.cif")
analyzer.analyze()
view = GraphView(analyzer)

# Compare spacer sizes
for i, spacer in enumerate(view.spacers):
    result = SpacerAnalysis(spacer, analyzer).compute()

    print(f"\nSpacer {i}:")
    print(f"  VdW volume: {result.vdw_volume:.2f} Å³")
    print(f"  Convex hull: {result.convex_hull_volume:.2f} Å³")
    print(f"  Radius of gyration: {result.radius_gyration:.2f} Å")

    # Compactness indicator
    if result.convex_hull_volume > 0:
        packing = result.vdw_volume / result.convex_hull_volume
        print(f"  Packing efficiency: {packing:.2f}")
```

## Backbone Analysis

### Backbone Identification

For DJ spacers, the backbone is defined as the **longest path** between terminal nitrogen groups. This follows the IUPAC organic chemistry definition.

### Backbone Properties

The `BackboneQuery` interface provides chainable methods for analyzing backbone structure:

```python
result = SpacerAnalysis(spacer, analyzer).compute()

# Access backbone interface
backbone = result.backbone

# Atom indices
indices = backbone.indices()  # List[int]

# Element symbols
elements = backbone.elements()  # List[str]

# Atomic positions
positions = backbone.positions()  # np.ndarray (n_atoms, 3)

# Bond lengths
bond_lengths = backbone.bond_lengths()  # List[float] in Å

# Bond angles
bond_angles = backbone.bond_angles()  # List[float] in degrees

# Dihedral angles
dihedrals = backbone.dihedrals()  # List[float] in degrees

# Filter backbone atoms
carbon_indices = backbone.filter(element='C')  # Only carbons
```

### Side Chains

Side chains are detected by identifying branching points (atoms with degree ≥ 3) along the backbone:

```python
result = SpacerAnalysis(spacer, analyzer).compute()

print(f"Side chains: {result.n_side_chains}")
print(f"Branching points: {result.branching_points}")

for i, side_chain in enumerate(result.side_chains):
    print(f"\nSide chain {i}:")
    print(f"  Attachment: atom {side_chain['attachment_idx']}")
    print(f"  Length: {side_chain['length']} atoms")
    print(f"  Atoms: {side_chain['chain_atoms']}")
```

## Complete Workflow Example

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.molecular_processing import SpacerAnalysis
from q2D_Materials.modifier import GraphView
import matplotlib.pyplot as plt

# Load structure
analyzer = q2D_analyzer("dj_structure.cif")
analyzer.analyze()
view = GraphView(analyzer)

# Analyze all spacers
spacer_data = []

for i, spacer in enumerate(view.spacers):
    result = SpacerAnalysis(spacer, analyzer).compute()

    spacer_data.append({
        'id': i,
        'compression': result.compression,
        'radius_gyration': result.radius_gyration,
        'vdw_volume': result.vdw_volume,
        'backbone_length': result.backbone_length,
        'n_side_chains': result.n_side_chains
    })

    print(f"\n{'='*60}")
    print(f"Spacer {i}")
    print(f"{'='*60}")
    print(result)  # Pretty-printed summary

    # Detailed backbone analysis
    if result.backbone_length > 0:
        backbone = result.backbone
        print(f"\nBackbone composition: {''.join(backbone.elements())}")
        print(f"Bond lengths: {[f'{x:.2f}' for x in backbone.bond_lengths()]} Å")
        print(f"Bond angles: {[f'{x:.1f}' for x in backbone.bond_angles()]}°")

# Statistical summary
import numpy as np

compressions = [s['compression'] for s in spacer_data]
r_gyrations = [s['radius_gyration'] for s in spacer_data]

print(f"\n{'='*60}")
print("Statistical Summary")
print(f"{'='*60}")
print(f"Compression factor: μ={np.mean(compressions):.3f}, σ={np.std(compressions):.3f}")
print(f"Radius of gyration: μ={np.mean(r_gyrations):.2f} Å, σ={np.std(r_gyrations):.2f} Å")

# Correlation plot
if len(spacer_data) > 1:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(r_gyrations, compressions, s=100, alpha=0.6)
    ax.set_xlabel('Radius of gyration (Å)', fontsize=12)
    ax.set_ylabel('Compression factor', fontsize=12)
    ax.axhline(0.8, color='r', linestyle='--', label='Compression threshold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('spacer_analysis.png', dpi=300)
    print("\n✓ Saved correlation plot: spacer_analysis.png")
```

## Advanced Usage: Custom Analysis

### Analyzing Specific Molecule Types

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.molecular_processing import SpacerAnalysis
from q2D_Materials.modifier import GraphView

analyzer = q2D_analyzer("structure.cif")
analyzer.analyze()
view = GraphView(analyzer)

# DJ spacers
for spacer in view.spacers:
    if spacer.molecule_type == 'spacer':
        result = SpacerAnalysis(spacer, analyzer, spacer_type="DJ").compute()
        print(f"DJ spacer: compression={result.compression:.3f}")

# RP spacers (single-ended)
for spacer in view.spacers:
    if spacer.molecule_type == 'spacer':
        result = SpacerAnalysis(spacer, analyzer, spacer_type="RP").compute()
        print(f"RP spacer: R_g={result.radius_gyration:.2f} Å")
```

### Filtering by Geometry

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.molecular_processing import SpacerAnalysis
from q2D_Materials.modifier import GraphView

analyzer = q2D_analyzer("structure.cif")
analyzer.analyze()
view = GraphView(analyzer)

# Find compressed spacers
compressed_spacers = []
for i, spacer in enumerate(view.spacers):
    result = SpacerAnalysis(spacer, analyzer).compute()
    if result.is_compressed:  # compression > 1.0 Å
        compressed_spacers.append((i, result))

print(f"Found {len(compressed_spacers)} compressed spacers")

# Find compact molecules (small R_g)
compact_molecules = []
for i, spacer in enumerate(view.spacers):
    result = SpacerAnalysis(spacer, analyzer).compute()
    if result.radius_gyration < 3.0:  # R_g < 3 Å
        compact_molecules.append((i, result))

print(f"Found {len(compact_molecules)} compact molecules")
```

## API Reference

### SpacerAnalysis Class

```python
SpacerAnalysis(
    molecule_graph: MoleculeGraph,
    analyzer: Optional[q2D_analyzer] = None,
    spacer_type: str = "DJ"
)
```

**Parameters:**
- `molecule_graph`: MoleculeGraph instance (from GraphView)
- `analyzer`: q2D_analyzer instance (optional, for compatibility)
- `spacer_type`: "DJ" or "RP" (default: "DJ")

**Methods:**
- `compute()`: Returns `SpacerAnalysisResult` with all computed properties

### SpacerAnalysisResult Dataclass

**Compression (DJ only):**
- `euclidean_distance`: float - Through-space N-N distance (Å)
- `path_length`: float - Sum of bond lengths along backbone (Å)
- `ideal_extended_length`: float - Expected fully extended length (path_length × 0.85) (Å)
- `compression`: float - Compression distance: ideal_extended_length - euclidean_distance (Å). Positive = compressed, zero = extended, negative = overstretched

**Volume and Size:**
- `vdw_volume`: float - Van der Waals volume (Å³)
- `convex_hull_volume`: float - Convex hull volume (Å³)
- `radius_gyration`: float - Radius of gyration (Å)

**Backbone:**
- `backbone_path`: List[int] - Atom indices in longest path
- `backbone_length`: int - Number of atoms in backbone
- `backbone_symbols`: List[str] - Element symbols along backbone
- `backbone`: BackboneQuery - Queryable backbone interface

**Side Chains:**
- `side_chains`: List[Dict] - Side chain data
- `branching_points`: List[int] - Atoms with degree ≥ 3
- `n_side_chains`: int - Total number of side chains

**Metadata:**
- `molecule_indices`: List[int] - Original structure indices
- `terminal_nitrogens`: List[int] - N atom indices
- `spacer_type`: str - "DJ" or "RP"
- `graph`: nx.Graph - Molecular graph reference

**Properties:**
- `is_compressed`: bool - Whether compression < 0.8

## Performance Notes

- **Typical analysis time**: < 10 ms per spacer on modern hardware
- **Memory usage**: Minimal (~1-5 MB per analyzed structure)
- **Dependencies**: NumPy (required), SciPy (optional, for convex hull), ASE (required, for atomic data)
- **Complexity**: O(N_atoms × N_paths) for backbone identification

## Tips & Troubleshooting

**Best Practices:**
- Use compression metrics for DJ spacers with two terminal groups
- Use radius of gyration for all molecule types (DJ, RP, A-site)
- Check `result.is_compressed` property for quick classification
- Analyze backbone dihedrals to assess conformational flexibility

**Common Issues:**
- **VdW volume = 0.0**: ASE data unavailable or missing
- **Convex hull = 0.0**: Fewer than 4 atoms or coplanar molecule
- **Compression factor = 0.0**: Not a DJ spacer or missing terminal groups
- **Radius of gyration = 0.0**: ASE unavailable or invalid atomic data

**Validation:**
- R_g should be positive for valid molecules
- R_g ≤ euclidean_distance (can't be larger than molecular extent)
- Typical R_g values: 2-10 Å for organic spacers
- VdW volume < convex hull volume (molecules don't fill convex hull)

## Integration with Other Modules

### With Structure Creation

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer.molecular_processing import SpacerAnalysis
from q2D_Materials.modifier import GraphView

# Create structure
creator = q2D_creator()
structure = creator.create_structure(
    template="cubic",
    structure_type="dj",
    A_ions="MA", B_ions="Pb", X_ions="I",
    spacer="NCCCCN",
    xy_expansion=(1, 1)
)

# Analyze created structure
analyzer = q2D_analyzer(structure)
analyzer.analyze()
view = GraphView(analyzer)

# Check spacer conformation in created structure
spacer = view.spacers[0]
result = SpacerAnalysis(spacer, analyzer).compute()

print(f"Created spacer R_g: {result.radius_gyration:.2f} Å")
print(f"Created spacer compression: {result.compression:.3f}")
```

### With Molecule Validation

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.molecular_processing import SpacerAnalysis
from q2D_Materials.modifier import GraphView

analyzer = q2D_analyzer()

# Validate molecule
validation = analyzer.mol_validate("NCCCCN", spacer_type="DJ")

if validation.is_valid:
    # Load into structure and analyze
    structure = creator.create_structure(
        template="cubic",
        structure_type="dj",
        A_ions="MA", B_ions="Pb", X_ions="I",
        spacer="NCCCCN",
        xy_expansion=(1, 1)
    )

    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    view = GraphView(analyzer)

    # Geometric analysis
    spacer = view.spacers[0]
    result = SpacerAnalysis(spacer, analyzer).compute()

    print(f"Validated and analyzed: R_g={result.radius_gyration:.2f} Å")
```

## Further Reading

- See **Example 20 (MoleculeValidator)** for spacer validation before structure creation
- See **Example 12 (Analysis)** for comprehensive structure analysis workflows
- See **Example 16 (GraphQuery)** for advanced molecular graph queries
