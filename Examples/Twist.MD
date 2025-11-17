# Twisted Bilayer Structures

The `twist()` method allows you to create twisted bilayer structures from monolayer perovskites. This method is based on Gabriel Xavier Pereira's Quadratic Twisted Bilayer Generator implementation.

## Understanding (m, n) Parameters

The `(m, n)` parameters are integer values that define the twist angle and supercell size for creating commensurate twisted bilayers. These parameters control:

- **Twist angle (θ)**: The rotation angle between the two layers
- **Supercell size**: The size of the periodic unit cell needed to accommodate the twist

The twist angle is calculated as:
```
θ = arctan(2mn / (m² - n²))
```

### Common (m, n) Combinations

| (m, n) | Twist Angle (θ) | Description |
|--------|------------------|-------------|
| (2, 1) | 53.13° | Large twist angle |
| (3, 1) | 36.87° | Medium-large twist |
| (4, 1) | 28.07° | Medium twist |
| (5, 1) | 22.62° | Small-medium twist |
| (5, 2) | 43.60° | Medium-large twist |
| (7, 2) | 31.89° | Medium twist |
| (7, 3) | 46.40° | Medium-large twist |
| (8, 3) | 41.11° | Medium twist |
| (9, 4) | 47.93° | Medium-large twist |
| (10, 3) | 33.40° | Medium twist |
| (13, 5) | 42.08° | Medium twist |

**Note**: Smaller `(m, n)` values generally produce larger twist angles and smaller supercells. Larger values produce smaller twist angles but require larger supercells.

## Method Parameters

- **`m`** (int): First integer parameter for twist angle calculation
- **`n`** (int): Second integer parameter for twist angle calculation  
- **`interlayer_distance`** (float, default=11.0): Distance in Ångströms between the two twisted layers
- **`vacuum`** (float, default=12.0): Total vacuum space in Ångströms above and below the bilayer structure (split equally)

## Example 1: Atomic Spacer with Small Twist Angle

This example creates a twisted bilayer from a monolayer with an atomic spacer (Cs) using a small twist angle:

```python
from q2D_Materials.core.creator import q2D_creator
from ase.io import write

# Create a monolayer structure
creator = q2D_creator()
monolayer = creator.create_perovskite(
    'monolayer',
    A_ions='MA',      # Methylammonium
    B_ions='Pb',      # Lead
    X_ions='I',       # Iodine
    spacer='Cs',      # Cesium (atomic spacer)
    supercell=[1, 1, 1]
)

# Create twisted bilayer with (m=3, n=1) → θ ≈ 36.87°
twisted = monolayer.twist(
    m=3, 
    n=1, 
    interlayer_distance=11.0,  # 11 Å gap between layers
    vacuum=12.0                # 12 Å total vacuum (6 Å above, 6 Å below)
)

# Save the structure
write('twisted_cs_3_1.vasp', twisted)

print(f"Original monolayer: {len(monolayer)} atoms")
print(f"Twisted bilayer: {len(twisted)} atoms")
print(f"Twist angle: ~36.87°")
print(f"Twist parameters: {twisted.twist_params}")
```

**Output:**
- Creates a twisted bilayer with approximately 36.87° twist angle
- The supercell is automatically expanded to accommodate the twist
- The structure has 11 Å spacing between the two layers
- 12 Å of vacuum is added symmetrically above and below

## Example 2: Molecular Spacer with Medium Twist Angle

This example demonstrates creating a twisted bilayer with a molecular spacer and a medium twist angle:

```python
from q2D_Materials.core.creator import q2D_creator
from ase.io import write

# Create a monolayer with molecular spacer
creator = q2D_creator()
monolayer = creator.create_perovskite(
    'monolayer',
    A_ions='MA',                    # Methylammonium
    B_ions='Pb',                    # Lead
    X_ions='I',                     # Iodine
    spacer='[NH3+]CCCCC[NH3+]',    # Pentanediammonium (PEA-like molecule)
    supercell=[1, 1, 1]
)

# Create twisted bilayer with (m=5, n=2) → θ ≈ 43.60°
twisted = monolayer.twist(
    m=5, 
    n=2, 
    interlayer_distance=11.0,  # 11 Å gap between layers
    vacuum=15.0                # 15 Å total vacuum (7.5 Å above, 7.5 Å below)
)

# Save the structure
write('twisted_molecular_5_2.vasp', twisted)

print(f"Original monolayer: {len(monolayer)} atoms")
print(f"Twisted bilayer: {len(twisted)} atoms")
print(f"Twist angle: ~43.60°")
print(f"Interlayer distance: {twisted.interlayer_distance} Å")
print(f"Vacuum: {twisted.vacuum} Å")
```

**Output:**
- Creates a twisted bilayer with approximately 43.60° twist angle
- Works with both atomic and molecular spacers
- Larger supercell compared to (3, 1) due to the (5, 2) parameters
- Customizable interlayer distance and vacuum spacing

## Important Notes

1. **Only for monolayers**: The `twist()` method only works on monolayer structures. Attempting to use it on other structure types will raise a `ValueError`.

2. **Supercell size**: The supercell automatically expands based on `(m, n)` values. Larger values result in larger supercells.

3. **Performance**: The method is optimized for performance using vectorized NumPy operations and direct pymatgen conversions (no file I/O).

4. **Metadata preservation**: All original structure metadata (A_ions, B_ions, X_ions, spacer, etc.) is preserved in the twisted structure, with additional `twist_params`, `interlayer_distance`, and `vacuum` attributes.

5. **Dependencies**: Requires `pymatgen` to be installed. The method will raise an `ImportError` if pymatgen is not available.

## Choosing (m, n) Values

- **Small twist angles** (< 30°): Use larger `m` values relative to `n` (e.g., `m=5, n=1` or `m=10, n=3`)
- **Medium twist angles** (30-45°): Balanced `m` and `n` (e.g., `m=5, n=2` or `m=7, n=3`)
- **Large twist angles** (> 45°): Use smaller `m` values relative to `n` (e.g., `m=3, n=1` or `m=2, n=1`)

Keep in mind that smaller twist angles require larger supercells, which increases computational cost.

