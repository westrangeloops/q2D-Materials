# Example 22: Octahedral Tilt Analysis

This example demonstrates advanced octahedral tilting analysis for quasi-2D perovskite structures. These metrics are particularly useful for understanding depth-dependent distortions and inter-octahedral coordination in DJ and RP phases.

## Overview

The q2D analyzer provides two key metrics for octahedral tilt analysis:

- **Mean Tilt Profile (μ)**: Average tilt magnitude per layer, revealing surface relaxation effects
- **Gearing Correlation (χ)**: Mechanical coupling between neighboring octahedra, quantifying cooperative vs. counter-rotating patterns

**Note**: Glazer notation is not applicable to quasi-2D structures (DJ, RP, monolayer). These metrics provide alternative characterization for layered perovskites.

## Mean Tilt Profile (μ)

### Formula

The mean tilt magnitude is computed as:

```
μ = (1/N) Σ ||T_i||
```

where:
- `||T|| = sqrt(α² + β² + γ²)` is the tilt magnitude (Euler angles in degrees)
- `N` is the number of octahedra
- `α, β, γ` are the Euler angles describing the octahedral rotation

### Physical Interpretation

- **Higher μ**: Indicates enhanced tilting, typically found in surface layers due to missing coordination
- **Lower μ**: Indicates bulk-like behavior, typically found in central layers
- **Depth profile**: Reveals "surface relaxation" in DJ/RP phases where surface layers show enhanced tilting compared to central layers

### API

```python
analyzer.compute_mean_tilt_profile(
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    group_by: Optional[str] = None,
) -> Union[float, Dict[str, float]]
```

**Parameters:**
- `octahedra`: List of specific octahedra IDs to include (e.g., `['octahedron_0', 'octahedron_1']`)
- `layer`: Single layer ID to filter by (e.g., `'0'`)
- `layers`: Multiple layer IDs to filter by (e.g., `['0', '1']`)
- `group_by`: If `'layer'` or `'slab'`, returns dict with per-layer/slab results plus `'global'` key. If `None`, returns single global value

**Returns:**
- If `group_by` is `None`: Mean tilt magnitude (float, degrees)
- If `group_by == 'layer'`: Dict with layer IDs as keys plus `'global'` key

## Gearing Correlation (χ)

### Formula

The gearing correlation between neighboring octahedra is computed as:

```
χ(i,j) = (R_i · R_j) / (||R_i|| ||R_j||)
```

where:
- `R` is the rotation vector (axis-angle representation from rotation matrix)
- `R_i · R_j` is the dot product of rotation vectors
- `||R||` is the magnitude of the rotation vector

### Physical Interpretation

- **χ ≈ +1**: Cooperative rotation (like meshing gears), structurally stable
- **χ ≈ -1**: Counter-rotation, destabilizing
- **χ ≈ 0**: Orthogonal rotations, no mechanical constraint
- **Interlayer χ**: Measures coupling across organic spacer (typically low)

### API

```python
analyzer.compute_gearing_correlation(
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    neighbor_criterion: str = 'shared_x',
) -> Union[Dict[Tuple[str, str], float], Dict[str, Dict[Tuple[str, str], float]]]
```

**Parameters:**
- `octahedra`: List of specific octahedra IDs to include
- `layer`: Single layer ID to filter by
- `layers`: Multiple layer IDs to filter by
- `group_by`: If `'layer'` or `'slab'`, returns dict with per-layer/slab results plus `'global'` key. If `None`, returns single dict of correlations
- `neighbor_criterion`: How to define neighbors (`'shared_x'` for corner-sharing)

**Returns:**
- If `group_by` is `None`: `{(oct_i, oct_j): χ}` where `χ ∈ [-1, 1]`
- If `group_by == 'layer'`: `{layer_id: {(oct_i, oct_j): χ}, 'global': {...}}`

## Complete Example

```python
from q2D_Materials.analyzer import q2D_analyzer

# Load and analyze structure
analyzer = q2D_analyzer("dj_structure.vasp")
analyzer.analyze()

# Get raw tilt data
tilt_data = analyzer.get_octahedral_tilts()
print(f"Found {tilt_data.euler_angles.shape[0]} octahedra")
print(f"Euler angles shape: {tilt_data.euler_angles.shape}")  # (N_oct, 3)

# Mean Tilt Profile - Global
mu_global = analyzer.compute_mean_tilt_profile()
print(f"\nGlobal mean tilt: μ = {mu_global:.2f}°")

# Mean Tilt Profile - Per-layer
profile = analyzer.compute_mean_tilt_profile(group_by='layer')
print("\nPer-layer mean tilt profile:")
for layer_id, mu in profile.items():
    if layer_id == 'global':
        print(f"  Global: μ = {mu:.2f}°")
    else:
        print(f"  Layer {layer_id}: μ = {mu:.2f}°")

# Mean Tilt Profile - Specific layer
mu_layer0 = analyzer.compute_mean_tilt_profile(layer='0')
print(f"\nLayer 0 mean tilt: μ = {mu_layer0:.2f}°")

# Gearing Correlation - Global
gearing = analyzer.compute_gearing_correlation()
print(f"\nGlobal gearing correlations: {len(gearing)} pairs")
cooperative_pairs = [(i, j) for (i, j), chi in gearing.items() if chi > 0.5]
print(f"Cooperative pairs (χ > 0.5): {len(cooperative_pairs)}")
for (oct_i, oct_j), chi in list(gearing.items())[:5]:
    print(f"  {oct_i} ↔ {oct_j}: χ = {chi:.3f}")

# Gearing Correlation - Per-layer
gearing_by_layer = analyzer.compute_gearing_correlation(group_by='layer')
print("\nPer-layer gearing correlations:")
for layer_id, pairs in gearing_by_layer.items():
    if layer_id == 'global':
        avg_chi = sum(pairs.values()) / len(pairs) if pairs else 0
        print(f"  Global: {len(pairs)} pairs, avg χ = {avg_chi:.3f}")
    else:
        avg_chi = sum(pairs.values()) / len(pairs) if pairs else 0
        print(f"  Layer {layer_id}: {len(pairs)} pairs, avg χ = {avg_chi:.3f}")

# Gearing Correlation - Specific layer
layer0_gearing = analyzer.compute_gearing_correlation(layer='0')
print(f"\nLayer 0 gearing: {len(layer0_gearing)} pairs")
```

### Example Output

```
Found 8 octahedra
Euler angles shape: (8, 3)

Global mean tilt: μ = 13.45°

Per-layer mean tilt profile:
  Global: μ = 13.45°
  Layer 0: μ = 15.40°  ← Enhanced tilting (surface)
  Layer 1: μ = 12.10°  ← Bulk-like (center)
  Layer 2: μ = 15.20°  ← Enhanced tilting (surface)

Layer 0 mean tilt: μ = 15.40°

Global gearing correlations: 24 pairs
Cooperative pairs (χ > 0.5): 18
  octahedron_0 ↔ octahedron_1: χ = 0.852
  octahedron_1 ↔ octahedron_2: χ = 0.789
  octahedron_2 ↔ octahedron_3: χ = 0.815
  octahedron_0 ↔ octahedron_4: χ = -0.123
  octahedron_1 ↔ octahedron_5: χ = -0.145

Per-layer gearing correlations:
  Global: 24 pairs, avg χ = 0.623
  Layer 0: 8 pairs, avg χ = 0.785  ← Strong intralayer coupling
  Layer 1: 8 pairs, avg χ = 0.712
  Layer 2: 8 pairs, avg χ = 0.778
  interlayer_0_1: 4 pairs, avg χ = -0.152  ← Weak interlayer coupling
  interlayer_1_2: 4 pairs, avg χ = -0.138

Layer 0 gearing: 8 pairs
```

### Interpretation

1. **Mean Tilt Profile**: Shows depth-dependent tilting with surface layers (0, 2) having higher μ (15.4°, 15.2°) than the central layer (12.1°), indicating surface relaxation.

2. **Gearing Correlation**: 
   - Intralayer pairs show strong cooperative rotation (χ ≈ 0.7-0.8), indicating mechanical coupling within layers
   - Interlayer pairs show weak/negative correlation (χ ≈ -0.15), indicating decoupled rotation across the organic spacer

## API Summary

### Analyzer Methods

| Method | Description | Parameters | Returns |
|--------|-------------|------------|---------|
| `get_octahedral_tilts()` | Get raw tilt data | `octahedra` (optional) | `OctahedralTiltData` object |
| `compute_mean_tilt_profile()` | Compute mean tilt magnitude | `octahedra`, `layer`, `layers`, `group_by` | `float` or `dict` |
| `compute_gearing_correlation()` | Compute gearing correlation | `octahedra`, `layer`, `layers`, `group_by`, `neighbor_criterion` | `dict` or `dict` of `dict`s |
| `compute_delta()` | Compute bond length distortion | `octahedra`, `layer`, `layers`, `group_by` | `float` or `dict` |
| `compute_sigma()` | Compute bond length variance | `octahedra`, `layer`, `layers`, `group_by` | `float` or `dict` |
| `compute_lambda()` | Compute bond angle distortion | `octahedra`, `layer`, `layers`, `group_by` | `float` or `dict` |

### OctahedralTiltData Object

| Attribute | Type | Description |
|-----------|------|-------------|
| `euler_angles` | `np.ndarray` | Euler angles (α, β, γ) for each octahedron, shape `(N_oct, 3)` |
| `rotation_matrices` | `np.ndarray` | Rotation matrices for each octahedron, shape `(N_oct, 3, 3)` |
| `octahedron_ids` | `List[str]` | List of octahedron node IDs (e.g., `['octahedron_0', ...]`) |
| `reference_axes` | `np.ndarray` | Reference coordinate system axes |

## Octahedral Distortion Parameters (Δ, σ, λ)

Three additional metrics quantify octahedral bond length and angle distortions:

### Delta (Δ) - Bond Length Distortion

**Formula:**
```
Δ = mean(|d_i - d_avg|) / d_avg
```

where:
- `d_i` are individual B-X bond lengths
- `d_avg` is the mean B-X bond length

**Interpretation:**
- Lower values: More uniform bond lengths
- Higher values: Greater bond length variation

### Sigma (σ²) - Bond Length Variance

**Formula:**
```
σ² = Var(d_i) / d_avg²
```

**Interpretation:**
- Lower values: Lower variance in bond lengths
- Higher values: Higher variance

### Lambda (λ²) - Bond Angle Distortion

**Formula:**
```
λ² = Var(min(|θ_i - 90°|, |θ_i - 180°|))
```

where `θ_i` are X-B-X bond angles.

**Interpretation:**
- Lower values: Angles closer to ideal (90° or 180°)
- Higher values: Greater angle distortion

### API

```python
delta = analyzer.compute_delta(octahedra=None, layer=None, layers=None, group_by=None)
sigma = analyzer.compute_sigma(octahedra=None, layer=None, layers=None, group_by=None)
lambda_param = analyzer.compute_lambda(octahedra=None, layer=None, layers=None, group_by=None)
```

**Parameters:**
- `octahedra`: List of specific octahedra IDs to include
- `layer`: Single layer ID to filter by
- `layers`: Multiple layer IDs to filter by
- `group_by`: If `'layer'`, returns dict with per-layer results plus `'global'` key

**Returns:**
- If `group_by` is `None`: Single value (float)
- If `group_by == 'layer'`: Dict with layer IDs as keys plus `'global'` key

### Example

```python
# Global distortion
delta = analyzer.compute_delta()
sigma = analyzer.compute_sigma()
lambda_param = analyzer.compute_lambda()
print(f"Δ = {delta:.4f}, σ² = {sigma:.6f}, λ² = {lambda_param:.4f}")

# Per-layer distortion
delta_by_layer = analyzer.compute_delta(group_by='layer')
for layer_id, delta in delta_by_layer.items():
    print(f"Layer {layer_id}: Δ = {delta:.4f}")
```

## Graph Structure for Octahedral Analysis

The octahedral analysis (tilt and distortion) uses the structural graph to identify octahedra and their layer assignments:

```
Structure Node (structure_0)
  └── CONTAINS → Layer Nodes (layer_0, layer_1, ...)
                    └── CONTAINS → Octahedron Nodes (octahedron_0, octahedron_1, ...)
                                      ├── CONTAINS (role='center') → B Atom (atom_*)
                                      └── CONTAINS (role='ligand') → X Atoms (atom_*, 6 total)
```

**Key Graph Relationships:**
- **Layer → Octahedron**: `CONTAINS` edges connect layer nodes to octahedron nodes
- **Octahedron → B Atom**: `CONTAINS` edge with `role='center'` connects to B-site atom
- **Octahedron → X Atoms**: `CONTAINS` edges with `role='ligand'` connect to 6 X-site atoms
- **B → X Bonds**: `BONDED_TO` edges with `role='ligand'` connect B atoms to X atoms (used for distortion analysis)
- **Neighbor Detection**: Corner-sharing octahedra identified via shared X atoms (used for gearing correlation)

**Node Properties:**
- **Layer nodes**: `node_type='layer'`, `layer_id` stored as string (e.g., `'0'`, `'1'`)
- **Octahedron nodes**: `node_type='octahedron'`, node ID format: `octahedron_{index}`
- **Atom nodes**: `node_type='atom'`, `vasp_index` for original atom index

**Accessing Layer Information:**
```python
layers = analyzer.get_layers()  # Returns {layer_id: {'octahedra': [...], 'octahedra_count': N, ...}}
```

Layer IDs are strings (e.g., `'0'`, `'1'`) matching the format used in `layer` and `layers` parameters.
