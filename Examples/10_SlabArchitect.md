![q2D-Materials Logo](../Logos/logo.png)

# 10. Slab Architect — designing custom layer sequences

Slab Architect is about mastering the art of custom layer sequences. Once you understand templates and basic stacking, you can architect complex structures by combining layers, controlling spacing, and mixing different stacking patterns. This guide shows you how to think like an architect when building custom perovskite slabs.

## The philosophy

- **Layers are building blocks**: Each named layer (`L1`, `L2`, `M1`, `RP1`, etc.) is a reusable component.
- **Sequences are blueprints**: `layer_sequence` tells the builder how to stack your blocks.
- **Spacing is explicit**: Control vertical gaps with explicit distances or let BX-dist handle defaults.
- **Mix and match**: Combine different layer types in one structure.

## Basic architecture patterns

### Pattern 1: Simple repetition

```python
from q2D_Materials.core.creator import q2D_creator
from ase.io import write

q2d = q2D_creator()

# Repeat a basic unit
simple = q2d.create_structure(
    template="cubic",
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-L2-L1-L2",  # Explicit repetition
    thickness=2,  # Or use thickness to repeat
)
```

### Pattern 2: Fault insertion

```python
# Insert a fault layer
faulted = q2d.create_structure(
    template="cubic",
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-L2-L1-M1-M1-L1-L2",  # M1 creates a fault
    sharp_spacer="[NH3+]CCCC[NH3+]",  # Spacer at fault
)
```

### Pattern 3: Mixed spacing

```python
# Mix explicit and default spacing
mixed = q2d.create_structure(
    template="cubic",
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-(2.5)-L2-L1-(1.8)-L2",  # First and third gaps explicit
    # Second gap uses BX-based default
)
```

## Advanced architectures

### Architecture 1: Alternating spacer layers

Create a structure where spacer layers alternate with perovskite slabs:

```python
alternating = q2d.create_structure(
    template="cubic",
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-L2-M1-M1-L1-L2-M1-M1",  # Spacer layers every 2 perovskite units
    sharp_spacer="[NH3+]CCCC[NH3+]",
    thickness=2,
)
```

### Architecture 2: Gradient spacing

Gradually increase spacing between layers:

```python
gradient = q2d.create_structure(
    template="cubic",
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-(2.0)-L2-(2.5)-L1-(3.0)-L2-(3.5)-L1",
    # Spacing increases: 2.0 → 2.5 → 3.0 → 3.5 Å
)
```

### Architecture 3: Sandwich structures

Create a perovskite core with spacer caps:

```python
sandwich = q2d.create_structure(
    template="cubic",
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="M1-M1-L1-L2-L1-L2-L1-M1-M1",  # Spacers on top and bottom
    sharp_spacer="[NH3+]CCCC[NH3+]",
    vacuum=15.0,
    thickness=1,
)
```

### Architecture 4: Superlattices

Build repeating supercells with different layer patterns:

```python
superlattice = q2d.create_structure(
    template="cubic",
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-L2-L1-L2-M1-M1-L1-L2-L1-L2-M1-M1",  # 2×2 superlattice
    sharp_spacer="[NH3+]CCCC[NH3+]",
    xy_expansion=(2, 2),  # Match superlattice periodicity
)
```

## Layer sequence syntax

### String format

```python
# Hyphen-separated
sequence = "L1-L2-L1-L2"

# No separators (also works)
sequence = "L1L2L1L2"

# With explicit distances
sequence = "L1-(2.5)-L2-(3.0)-L1"
```

### List format

```python
# Simple list
sequence = ["L1", "L2", "L1", "L2"]

# With explicit distances (not supported in list format, use string)
sequence = "L1-(2.5)-L2"  # Use string for explicit distances
```

## Spacing strategies

### Strategy 1: BX-based (default)

```python
# Uses BX_dist to calculate spacing automatically
default = q2d.create_structure(
    template="cubic",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-L2-L1-L2",  # No explicit distances
    BX_dist=3.18,  # Optional: override auto-calculation
)
```

### Strategy 2: Explicit distances

```python
# Full control over every gap
explicit = q2d.create_structure(
    template="cubic",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-(2.0)-L2-(2.5)-L1-(3.0)-L2",
    # Each gap specified in Å
)
```

### Strategy 3: Mixed approach

```python
# Some gaps explicit, others default
mixed = q2d.create_structure(
    template="cubic",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-(2.5)-L2-L1-(3.0)-L2-L1",
    # First and third gaps explicit, second uses BX
)
```

## Combining with other features

### Architecture + Glazer tilting

```python
tilted_arch = q2d.create_structure(
    template="cubic",
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="L1-L2-M1-M1-L1-L2",
    sharp_spacer="[NH3+]CCCC[NH3+]",
    glazer_angles=[0, 0, 5],
    glazer_pattern=["0", "0", "+"],
)
```

### Architecture + Penetration

```python
penetrated_arch = q2d.create_structure(
    template="cubic",
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    layer_sequence="M1-M1-L1-L2-L1",
    sharp_spacer="[NH3+]CCCC[NH3+]",
    penetration=0.3,  # Shift spacer anchors
    vacuum=15.0,
)
```

### Architecture + Pattern mixing

```python
pattern_arch = q2d.create_structure(
    template="cubic",
    structure_type="bulk",
    A_ions=["Cs", "MA", "FA"],  # Cycle through A-sites
    B_ions=["Pb", "Sn"],        # Cycle through B-sites
    X_ions=["I", "Br"],         # Cycle through X-sites
    layer_sequence="L1-L2-L1-L2-L1-L2",
    xy_expansion=(2, 2),  # 4×4 supercell for pattern mixing
)
```

## Design workflow

1. **Start simple**: Build a basic sequence first
   ```python
   test = q2d.create_structure(
       template="cubic",
       A_ions="MA", B_ions="Pb", X_ions="I",
       layer_sequence="L1-L2",
   )
   ```

2. **Add complexity gradually**: Introduce spacers, faults, or custom spacing
   ```python
   complex = q2d.create_structure(
       template="cubic",
       A_ions="MA", B_ions="Pb", X_ions="I",
       layer_sequence="L1-L2-M1-M1-L1-L2",
       sharp_spacer="[NH3+]CCCC[NH3+]",
   )
   ```

3. **Refine spacing**: Adjust explicit distances to match your needs
   ```python
   refined = q2d.create_structure(
       template="cubic",
       A_ions="MA", B_ions="Pb", X_ions="I",
       layer_sequence="L1-(2.5)-L2-(3.0)-M1-(3.5)-M1",
       sharp_spacer="[NH3+]CCCC[NH3+]",
   )
   ```

4. **Validate**: Check atom counts, visualize, verify spacing
   ```python
   print(f"Total atoms: {len(refined)}")
   write("arch.vasp", refined, sort=True)
   ```

## Tips for architects

- **Name your patterns**: Create variables for common sequences
  ```python
   DJ_PATTERN = "L1-L2-M1-M1"
   RP_PATTERN = "L2-M1-RP1-RP2-RP1-M1"
   ```

- **Use thickness wisely**: `thickness` repeats your sequence; explicit sequences give more control

- **Match XY expansion**: If using patterns, ensure `xy_expansion` matches your sequence periodicity

- **Document your designs**: Comment complex sequences to remember your architecture choices

- **Test incrementally**: Build and validate each layer addition before moving to the next

## Common architectures reference

```python
# Basic perovskite
"L1-L2"

# Dion-Jacobson
"L1-L2-M1-M1"

# Ruddlesden-Popper  
"L2-M1-RP1-RP2-RP1-M1"

# Alternating
"L1-L2-M1-M1-L1-L2-M1-M1"

# Sandwich
"M1-M1-L1-L2-L1-M1-M1"

# Fault insertion
"L1-L2-L1-M1-M1-L1-L2"
```

## Troubleshooting

- **Wrong layer count**: Check that all layer names exist in your template
- **Spacers not appearing**: Ensure consecutive layers have matching `S#` labels
- **Spacing issues**: Verify explicit distances match your intended geometry
- **Patterns not cycling**: Ensure `xy_expansion` provides enough sites for your pattern list

Regenerate examples:
```bash
nix develop -c python3 Examples/plot.py
```

