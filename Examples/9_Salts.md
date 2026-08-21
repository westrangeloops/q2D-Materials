![q2D-Materials Logo](../Logos/logo.png)

# 9. Salt Structures — spacer-only architectures

Salt structures are minimalist templates that focus on spacer molecules connecting halide layers. Unlike standard perovskites, they don't require B-site cations—just X-site halides and sharp spacers. This makes them perfect for studying organic-inorganic interfaces, spacer packing, and molecular alignment without the complexity of full perovskite networks.

## The concept

- **No B-site required**: Salt templates define only X-site halides and spacer anchor points (`S#` sites).
- **Sharp spacers connect layers**: Double-NH₃ molecules bridge between consecutive layers via `S#` anchors.
- **Custom lattice control**: Use `lattice_multipliers` to match your spacer geometry.
- **Explicit layer spacing**: Control interlayer distances directly via `layer_sequence` syntax.

## Minimal recipe

```python
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.builders.spacer import (
    calculate_double_spacer_nh3_distances as nh3_distance,
    calculate_molecule_radius as molecule_radius,
    elongate_molecule
)
from ase.io import write
import numpy as np

q2d = q2D_creator()

# Prepare your spacer molecule
molecule = "[NH3+]CCCC[NH3+]"  # SMILES string
elongated_molecule = elongate_molecule(molecule, step_size=0.5, max_iterations=1000)

# Calculate spacer geometry
N_N_distance = nh3_distance(elongated_molecule)
molecule_diameter = molecule_radius(elongated_molecule) * 2

# Calculate required lattice parameters
# For a square unit cell: diagonal = 2 * N-N distance + padding
h = 2 * N_N_distance + 6  # 6 Å padding
A = h / np.sqrt(2)  # Square cell side length

# Create salt structure
atoms = q2d.create_structure(
    template="salts",              # or "maybe_salt" for custom templates
    X_ions="I",                   # Only X-site needed (no B-site)
    spacer=elongated_molecule,
    lattice_multipliers=[A/3, A/3],  # Scale to match spacer geometry
    layer_sequence=f"L1-({molecule_diameter})-L2",  # Explicit spacing
    structure_type="monolayer",
    optimizer="Off",              # Use "Off" if molecule is pre-elongated
    vacuum=0.0,                   # No vacuum for bulk-like salts
)

write("salts.vasp", atoms, format="vasp", sort=True)
```

## Comprehensive example: batch processing multiple molecules

This example demonstrates how to systematically generate salt structures for multiple spacer molecules and halogens, with adaptive lattice parameter calculation based on molecular geometry.

```python
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.builders.spacer import (
    calculate_double_spacer_nh3_distances as nh3_distance,
    calculate_molecule_radius as molecule_radius,
    elongate_molecule
)
from ase.io import write

# Define a library of spacer molecules to test
molecules_list = [
    '[NH3+]CCCC[NH3+]',           # Simple linear chain
    '[NH3+]CCCCCC[NH3+]',         # Longer linear chain
    'CC(C)(CC[NH3+])CC[NH3+]',    # Branched molecule
    '[NH3+]C1CCC([NH3+])CC1',     # Cyclic molecule
    '[NH3+]CC1=CC=C(C[NH3+])C=C1', # Aromatic ring
    # ... more molecules
]

q2d = q2D_creator()

# Step 1: Iterate over molecules and halogens
# This creates a combinatorial library: each molecule × each halogen
for molecule in molecules_list:
    for halogen in ['Cl', 'Br', 'I']:
        
        # Step 2: Elongate the molecule
        # Why: Maximizes N-N distance to find the most extended conformation
        # This ensures spacers use their full length, preventing compression
        elongated_molecule = elongate_molecule(
            molecule, 
            step_size=0.5,        # 0.5 Å increments per iteration
            max_iterations=1000   # Allow up to 1000 steps to find maximum
        )
        
        # Step 3: Calculate N-N distance
        # Why: This is the end-to-end length of the spacer molecule
        # Used to determine how much horizontal space is needed
        N_N_distance = nh3_distance(elongated_molecule)
        print(f"Molecule: {molecule}")
        print(f"N - N distance: {N_N_distance} Å")
        
        # Step 4: Calculate molecule diameter
        # Why: The effective width of the molecule (2× radius)
        # Used for vertical spacing to prevent overlaps between layers
        molecule_diameter = molecule_radius(elongated_molecule) * 2
        print(f"Molecule diameter: {molecule_diameter} Å")
        
        # Step 5: Calculate required diagonal spacing
        # Why: In the unit cell, spacers are arranged diagonally
        # We need space for 2 N-N distances (two spacers) plus padding
        h = 2 * N_N_distance + 6  # 6 Å padding prevents edge collisions
        print(f"Required diagonal (h): {h} Å")
        
        # Step 6: Determine packing geometry based on molecule type
        # Why: Different molecular shapes pack differently in 2D
        # - Linear chains: pack in square grids (denominator = 4)
        # - Branched molecules (CC(C)): pack in rectangular grids (denominator = 2)
        # - Cyclic/aromatic (C1C or C1=C): pack in triangular grids (denominator = 3)
        denominator = 4  # Default: square packing for linear molecules
        
        if "CC(C)" in molecule:
            # Branched molecules: rectangular packing
            # These molecules are wider, so they need more space in one direction
            denominator = 2
            print("Detected: Branched molecule → rectangular packing")
            
        if "C1C" in molecule or "C1=C" in molecule:
            # Cyclic/aromatic molecules: triangular packing
            # Rings pack more efficiently in triangular arrangements
            denominator = 3
            print("Detected: Cyclic/aromatic molecule → triangular packing")
        
        # Step 7: Calculate unit cell side length
        # Why: Convert diagonal spacing to actual cell dimensions
        # Formula: For packing geometry, h² = denominator × A²
        # Therefore: A = h / √denominator
        # Add molecule_diameter/3 as extra safety margin for collision avoidance
        A = h / np.sqrt(denominator) + molecule_diameter / 3
        print(f"Unit cell side length (A): {A} Å")
        print(f"Packing geometry: denominator = {denominator}")
        
        # Step 8: Create the salt structure
        # Why: Each parameter is carefully calculated to match the molecule
        atoms = q2d.create_structure(
            template="salts",                    # Salt template (X-sites + S# anchors)
            X_ions=halogen,                      # Halide: Cl, Br, or I
            spacer=elongated_molecule,      # Pre-elongated molecule
            lattice_multipliers=[A/3, A/3],       # Scale template to match calculated size
            #                                      # Template base is 3, so divide by 3
            layer_sequence=f"L1-({molecule_diameter})-L2",  # Explicit spacing = diameter
            structure_type="monolayer",           # 2D slab structure
            optimizer="Off",                      # Use pre-elongated geometry
            vacuum=0.0,                          # No vacuum (bulk-like)
        )
        
        # Step 9: Save structure with descriptive filename
        # Why: Organize output by molecule and halogen for easy comparison
        output_path = "../SALTS"
        filename = f"{output_path}/{molecule}_{halogen}.vasp"
        write(filename, atoms, format="vasp", sort=True)
        print(f"Saved: {filename}\n")
```

### Step-by-step explanation

**Step 1: Nested loops**
- Iterates over all molecules and all halogens
- Creates a combinatorial library: 24 molecules × 3 halogens = 72 structures
- Allows systematic comparison of different spacer-halide combinations

**Step 2: Molecule elongation**
- `elongate_molecule()` finds the maximum N-N distance by iteratively stretching
- Uses kinematic constraints to respect bond angles and lengths
- `step_size=0.5` Å: balance between precision and speed
- `max_iterations=1000`: allows complex molecules to fully extend

**Step 3: N-N distance calculation**
- Measures the actual end-to-end distance between terminal NH₃⁺ groups
- Critical for determining horizontal spacing requirements
- Different conformations yield different N-N distances

**Step 4: Molecule diameter**
- Calculates the effective cylindrical radius of the molecule
- Multiplied by 2 to get diameter
- Used for vertical spacing to prevent layer overlaps

**Step 5: Diagonal spacing calculation**
- Accounts for diagonal arrangement of spacers in the unit cell
- `2 * N_N_distance`: space for two spacers (one per layer)
- `+ 6`: padding to prevent edge collisions and allow for molecular flexibility

**Step 6: Packing geometry detection**
- **Linear molecules** (`denominator=4`): Square packing, most common
- **Branched molecules** (`CC(C)`, `denominator=2`): Rectangular packing, wider molecules need more space
- **Cyclic/aromatic** (`C1C` or `C1=C`, `denominator=3`): Triangular packing, rings pack more efficiently
- Detection uses SMILES string patterns to identify molecular topology

**Step 7: Unit cell calculation**
- Converts diagonal requirement to actual cell dimensions
- `h / √denominator`: geometric conversion based on packing type
- `+ molecule_diameter/3`: additional safety margin for collision avoidance
- Ensures spacers have adequate room without overlaps

**Step 8: Structure creation**
- `lattice_multipliers=[A/3, A/3]`: Template base is 3, so divide calculated A by 3
- `layer_sequence=f"L1-({molecule_diameter})-L2"`: Explicit spacing matches molecule width
- `optimizer="Off"`: Uses pre-elongated geometry directly (faster, more predictable)

**Step 9: File organization**
- Saves with descriptive names: `{molecule}_{halogen}.vasp`
- Easy to identify and compare structures
- Organized in `../SALTS/` directory for batch processing results

### Why this approach works

1. **Adaptive geometry**: Different molecule types get appropriate packing arrangements
2. **Systematic exploration**: Batch processing enables high-throughput screening
3. **Collision avoidance**: Multiple safety margins prevent overlaps
4. **Reproducibility**: Each structure is calculated from first principles

## Understanding the geometry

Salt structures require careful matching between:
1. **Spacer N–N distance**: The end-to-end length of your double-NH₃ molecule
2. **Molecule diameter**: The effective width (2× radius) for collision avoidance
3. **Lattice parameters**: Must accommodate the spacer geometry without overlaps

### Calculating lattice parameters

For a square unit cell with spacers arranged diagonally:

```python
# The diagonal spans 2 N-N distances plus padding
h = 2 * N_N_distance + padding  # padding typically 4-8 Å

# For square cell: diagonal² = 2 × side²
# Therefore: side = diagonal / √2
A = h / np.sqrt(2)

# Scale template multipliers to match
lattice_multipliers = [A / base_multiplier, A / base_multiplier]
```

### Setting layer spacing

Use explicit distances in `layer_sequence` to match molecule diameter:

```python
layer_sequence = f"L1-({molecule_diameter})-L2"
```

This ensures the vertical gap between layers matches the spacer's effective diameter, preventing overlaps.

## Optimizer options

- **`optimizer="Off"`**: Pure geometric placement. Use when your molecule is already elongated or you want maximum control.
- **`optimizer="KS"`**: Kinematic Solver (default). Respects bond constraints and finds physically valid conformations.
- **`optimizer="UFF"`**: Universal Force Field optimization. Best for final refinement but requires RDKit.

```python
# Pre-elongated molecule (recommended for salts)
elongated = elongate_molecule("[NH3+]CCCC[NH3+]", step_size=0.5, max_iterations=1000)
atoms = q2d.create_structure(
    template="salts",
    X_ions="I",
    spacer=elongated,
    optimizer="Off",  # Already optimized
    ...
)

# Let the solver handle elongation
atoms = q2d.create_structure(
    template="salts",
    X_ions="I",
    spacer="[NH3+]CCCC[NH3+]",  # Raw SMILES
    optimizer="KS",  # Will elongate automatically
    ...
)
```

## Collision avoidance

The global optimizer automatically:
- **Detects obstacles**: X-site halides and other heavy atoms
- **Calculates molecule radius**: Uses geometric radius and center-of-mass distance
- **Finds optimal vectors**: Minimizes collisions while preserving shortest paths
- **Handles PBC**: Considers periodic boundary conditions for proper pairing

If collisions are unavoidable, the optimizer returns the solution with minimal collisions—no warnings, just the best available configuration.

## Custom salt templates

Create your own salt template by defining layers with X-sites and S# anchors:

```json
{
  "named_layers": {
    "L1": [
      ["X", 0.25, 0.25],
      ["X", 0.75, 0.75],
      ["S1", 0.5, 0.5]
    ],
    "L2": [
      ["X", 0.25, 0.75],
      ["X", 0.75, 0.25],
      ["S1", 0.5, 0.5]
    ]
  },
  "layer_sequence": ["L1", "L2"],
  "lattice_multipliers": [5.0, 5.0]
}
```

## Tips

- **Start with elongated molecules**: Pre-elongate your spacer to avoid optimization overhead
- **Match lattice to spacer**: Calculate required cell size from N–N distance and diameter
- **Use explicit spacing**: Set layer distances explicitly to match molecule geometry
- **Check for overlaps**: Visualize the structure and verify spacer placement
- **Experiment with padding**: Adjust the padding value (6 Å default) based on your needs

## Troubleshooting

- **Overlaps**: Increase `molecule_diameter` in layer spacing or increase padding in lattice calculation
- **Spacers missing**: Ensure template has matching `S#` labels on consecutive layers
- **Wrong cell size**: Recalculate `lattice_multipliers` based on actual N–N distance
- **Collisions**: The optimizer minimizes collisions automatically; check if spacing is too tight

Regenerate examples:
```bash
python3 Examples/plot.py
```

