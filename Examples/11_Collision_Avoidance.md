![q2D-Materials Logo](../Logos/logo.png)

# q2D-Materials Collision Avoidance — Preventing DFT Failures

**Atomic overlaps during molecular spacer placement can cause DFT calculations to fail.** The collision avoidance system automatically detects and resolves these overlaps before they break your quantum mechanical simulations.

## The Problem

When placing flexible molecular spacers (like organic cations), naive approaches can result in atoms being too close together. This causes:
- **Matrix diagonalization failures** in DFT codes
- **Force explosions** during geometry optimization
- **Unphysical structures** that don't represent real chemistry

## The Solution: Collision Strategies

q2D-Materials includes multiple collision resolution strategies. Choose the right one for your use case:

| Strategy | Speed | Robustness | When to Use |
|----------|--------|------------|-------------|
| `"rotate"` | ⚡ Fast | ⭐ Good | **Default choice** - Most molecular spacers |
| `"nudge"` | ⚡ Fast | ⭐⭐ Better | When rotation doesn't solve the problem |
| `"optimize"` | 🐌 Slow | ⭐⭐⭐ Most robust | Complex structures, when accuracy is needed |
| `"reject"` | ⚡ Instant | ❌ None | Debug mode, manual intervention preferred |
| `"off"` | ⚡ Instant | ❌ None | Trust your inputs, disable checking |

## Strategy Details

### 🔄 `"rotate"` (Default)
**Rotates molecules around their N-N axis to find collision-free orientations.**

- **How it works**: For double spacers (NH3+-R-NH3+), rotates around the molecular axis
- **Performance**: Fastest option, usually resolves 90%+ of collisions
- **Limitations**: Only works for molecules with clear N-N axes
- **Fallback**: Automatically tries `"nudge"` if rotation fails

```python
# Most common usage - let the system handle collisions
structure = q2d.create_structure(
    A_ions="MA", B_ions="Pb", X_ions="I",
    spacer=["MA", "EA"],  # Two different spacers
    collision_strategy="rotate"  # Default behavior
)
```

### ↗️ `"nudge"`
**Applies small XY translations when rotation fails.**

- **How it works**: Systematically tries small displacements (±0.1-1.0 Å) in XY plane
- **Performance**: Fast, spiral search pattern covers space efficiently
- **Use case**: When molecules are already well-oriented but slightly overlapping
- **Limitations**: May not resolve severe steric clashes

```python
# For densely packed structures
structure = q2d.create_structure(
    A_ions="FA", B_ions="Sn", X_ions="I",
    xy_expansion=(2, 2),  # Large supercell = more potential overlaps
    spacer="big_spacer_molecule",
    collision_strategy="nudge"
)
```

### 🎯 `"optimize"`
**Uses geometry optimization to push overlapping atoms apart.**

- **How it works**: Fixes existing atoms, optimizes spacer positions using ASE FIRE optimizer
- **Performance**: Slowest option (seconds per spacer), but most reliable
- **Use case**: Complex molecular geometries
- **Requirements**: ASE installed with working calculator

```python
# For complex molecular geometries
structure = q2d.create_structure(
    A_ions=["Cs", "MA"], B_ions="Pb", X_ions="I",
    spacer="complex_polymer_spacer",
    collision_strategy="optimize",  # Takes longer but more reliable
    xy_expansion=(1, 1)
)
```

### 🚫 `"reject"`
**Warns about collisions and skips problematic placements.**

- **How it works**: Detects collisions and raises warnings instead of fixing them
- **Performance**: Fastest option, minimal computational cost
- **Use case**: Debugging, when you want manual control over placements
- **Output**: Detailed warnings with collision distances and positions

```python
# Debug mode - see what collisions exist
try:
    structure = q2d.create_structure(
        A_ions="MA", B_ions="Pb", X_ions="I",
        spacer="experimental_molecule",
        collision_strategy="reject"  # Will warn and potentially skip
    )
except Warning as w:
    print(f"Collision detected: {w}")
    # Manually adjust molecule or change strategy
```

### 🚀 `"off"`
**Disables collision checking entirely.**

- **How it works**: No collision detection or resolution
- **Performance**: Maximum speed, no overhead
- **Use case**: When you know your inputs are safe, or for benchmarking
- **Risk**: May produce structures that fail DFT calculations

```python
# Maximum performance, trust your inputs
structure = q2d.create_structure(
    A_ions="Cs", B_ions="Pb", X_ions="I",
    spacer="well_tested_molecule",
    collision_strategy="off"  # Skip collision checks
)
```

## Code Examples

### Basic Usage
```python
from q2D_Materials.core.creator import q2D_creator
q2d = q2D_creator()

# Simple case - rotation usually works
perovskite = q2d.create_structure(
    structure_type="bulk",
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    spacer="NH3(CH2)4NH3",  # Butane diammonium
    collision_strategy="rotate"  # Default
)
```

### Advanced: Different Strategies per Spacer
```python
# Mix strategies (though API currently uses one per structure)
# Future enhancement could allow per-spacer strategies
structure = q2d.create_structure(
    A_ions="FA",
    B_ions="Sn",
    X_ions="Br",
    spacer=["short_spacer", "very_long_spacer"],
    collision_strategy="optimize"  # Conservative approach
)
```

### Troubleshooting Collisions
```python
# Step 1: Try default rotation
structure = q2d.create_structure(
    A_ions="MA", B_ions="Pb", X_ions="I",
    spacer="problematic_molecule",
    collision_strategy="rotate"
)

# Step 2: If that fails, try nudge
structure = q2d.create_structure(
    A_ions="MA", B_ions="Pb", X_ions="I",
    spacer="problematic_molecule",
    collision_strategy="nudge"
)

# Step 3: For complex cases, use optimization
structure = q2d.create_structure(
    A_ions="MA", B_ions="Pb", X_ions="I",
    spacer="problematic_molecule",
    collision_strategy="optimize"
)
```

## Technical Details

### Collision Detection
- **Distance thresholds**: Based on covalent radii + 0.8 Å buffer
- **Heavy atoms only**: H-H collisions ignored for performance
- **PBC-aware**: Considers periodic boundary conditions
- **Minimum distance**: Dynamically calculated per atom pair

### Performance Characteristics
- **`"rotate"`**: ~1-10 ms per spacer
- **`"nudge"`**: ~10-100 ms per spacer
- **`"optimize"`**: ~1-10 seconds per spacer
- **`"reject"`/`"off"`**: <1 ms per spacer

### Limitations
- Currently one strategy per structure (not per spacer)
- Optimization requires ASE with working calculator
- Very large molecules (>100 atoms) may need manual tuning

## Best Practices

### 🎯 Choose the Right Strategy
- **New projects**: Start with `"rotate"` (default)
- **Dense structures**: Use `"nudge"` or `"optimize"`
- **Publication work**: Always use `"optimize"` for final structures
- **Debugging**: Use `"reject"` to identify issues

### 🔧 Troubleshooting
1. **Collisions persist?** Try a different strategy
2. **Slow performance?** Use `"rotate"` for initial screening
3. **Unusual molecules?** Consider `"optimize"` for complex geometries
4. **Want manual control?** Use `"reject"` and adjust inputs

### 📊 Validation
Always validate final structures:
```python
# Check for remaining close contacts
from ase.neighborlist import neighbor_list
i, j, d = neighbor_list('ijd', structure, cutoff=1.5)
close_contacts = [(i, j, dist) for i, j, dist in zip(i, j, d) if dist < 1.5]
print(f"Close contacts found: {len(close_contacts)}")
```

## Integration with Other Features

Collision avoidance works seamlessly with:
- **Glazer tilting**: `glazer_angles` and `glazer_pattern`
- **Layer sequences**: `"DJ"`, `"RP"`, custom sequences
- **Twisted bilayers**: Moiré patterns
- **Supercell expansion**: `xy_expansion=(n,m)`

```python
# Complex example with multiple features
twisted_perovskite = q2d.create_structure(
    structure_type="bulk",
    template="cubic",
    layer_sequence="DJ",
    thickness=2,
    A_ions="FA",
    B_ions="Pb",
    X_ions="I",
    spacer="NH3(CH2)6NH3",
    glazer_angles=[0, 0, 5],
    glazer_pattern=["0", "0", "+"],
    xy_expansion=(2, 2),
    collision_strategy="optimize"  # Ensure clean final structure
)
```

---

**Remember**: Collision avoidance prevents DFT failures, but always validate your final structures before running expensive calculations!
