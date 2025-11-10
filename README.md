# q2D-Materials: Quasi-2D Perovskite Structure Generation

A Python package for creating bulk and quasi-2D perovskite structures with support for mixed compositions and molecular spacers.

## Quick Start

### Installation

```bash
# Using Nix (recommended)
nix develop

# Or using pip
pip install ase numpy rdkit
```

### Basic Usage

```python
from q2D_Materials.core.creator import q2D_creator

# Initialize creator with composition
q2d = q2D_creator(B='Pb', X='I', A='MA', name='MAPbI3')

# Create bulk perovskite
bulk = q2d.create_perovskite('bulk', supercell_size=(1, 1, 1))
q2d.write_structure(bulk, 'MAPbI3_bulk.vasp')

# Create 2D structures (RP, DJ, or monolayer)
dj = q2d.create_perovskite('DJ', 
    spacer_molecule='[NH3+]CCCCC[NH3+]',  # SMILES string or XYZ file path
    supercell=[1, 1, 2]  # [nx, ny, n_layers]
)
q2d.write_structure(dj, 'MAPbI3_DJ_n2.vasp')
```

## Structure Types

### Bulk Perovskites

```python
# Simple bulk (single unit cell)
bulk = q2d.create_perovskite('bulk', supercell_size=(1, 1, 1))

# Bulk with supercell
bulk_supercell = q2d.create_perovskite('bulk', supercell_size=(2, 2, 2))

# Double perovskite
double = q2d.create_perovskite('bulk', 
    supercell_size=(2, 2, 2),
    Bp='Sn'  # Second B-site cation
)

# Pattern-based mixed composition
mixed = q2d.create_perovskite('bulk',
    A_ions=['Cs', 'MA', 'FA', 'Cs', 'MA', 'FA', 'Cs', 'MA'],  # Pattern cycles
    B_ions=['Pb', 'Sn'],  # Alternating pattern
    X_ions=['Br', 'I', 'I'],  # Pattern: 1 Br, 2 I repeating
    supercell_size=(2, 2, 2)  # Required for mixed compositions
)
```

### 2D Perovskites

#### Ruddlesden-Popper (RP)
```python
rp = q2d.create_perovskite('RP',
    spacer_molecule='[NH3+]CCCCC=O',  # SMILES or XYZ file
    supercell=[1, 1, 2],  # [nx, ny, n_layers]
    spacer_distance=2.0,  # Vacuum gap between spacers (Å)
    penet=0.3,  # Spacer penetration into layer (fraction of BX bond)
    BX_dist=None,  # Auto-calculated if None
    wrap=False  # Wrap atoms to cell
)
```

#### Dion-Jacobson (DJ)
```python
dj = q2d.create_perovskite('DJ',
    spacer_molecule='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 2],  # [nx, ny, n_layers]
    attachment_end='top',  # 'top', 'bottom', or 'both' (default: 'top')
    penet=0.3,
    Bp='Sn',  # Optional: double perovskite
    Ap_Rx=0.0,  # Optional: rotation angles in degrees
    Ap_Ry=0.0,
    Ap_Rz=0.0
)
```

#### Monolayer
```python
monolayer = q2d.create_perovskite('monolayer',
    spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]',  # Caffeine-based
    supercell=[1, 1, 1],
    vacuum=12,  # Vacuum thickness (Å)
    attachment_end='both',  # Default: 'both'
    penet=0.3
)
```

## Spacer Molecules

Spacer molecules can be provided as:
- **SMILES strings**: `'[NH3+]CCCCC[NH3+]'` (requires RDKit)
- **XYZ file paths**: `'spacer.xyz'`
- **ASE Atoms objects**: Direct molecular structure
- **List of any above**: For pattern-based mixed spacers

```python
# Using SMILES string
dj = q2d.create_perovskite('DJ', 
    spacer_molecule='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 2]
)

# Using XYZ file
dj = q2d.create_perovskite('DJ', 
    spacer_molecule='spacer.xyz',
    supercell=[1, 1, 2]
)

# Using ASE Atoms object
from ase.io import read
spacer = read('spacer.xyz')
dj = q2d.create_perovskite('DJ', 
    spacer_molecule=spacer,
    supercell=[1, 1, 2]
)

# Pattern-based mixed spacers (for supercell with multiple positions)
mixed_spacers = q2d.create_perovskite('DJ',
    spacer_molecule=['[NH3+]CCCCC[NH3+]', 'spacer2.xyz', '[NH3+]CCCCCC[NH3+]'],
    supercell=[2, 2, 2]  # 4 positions, pattern will cycle
)
```

## Complete Examples

### Example 1: Bulk Perovskite with All Parameters

```python
from q2D_Materials.core.creator import q2D_creator

# Initialize creator
q2d = q2D_creator(B='Pb', X='I', A='MA', name='MAPbI3')

# Create bulk perovskite with all available parameters
bulk = q2d.create_perovskite(
    structure_type='bulk',
    
    # Supercell size (required for mixed compositions, optional for uniform)
    supercell_size=(2, 2, 2),  # (nx, ny, nz)
    
    # Pattern-based mixed compositions (lists cycle if shorter than positions)
    A_ions=['Cs', 'MA', 'FA', 'Cs', 'MA', 'FA', 'Cs', 'MA'],  # 8 positions
    B_ions=['Pb', 'Sn'],  # Alternating pattern
    X_ions=['Br', 'I', 'I'],  # Pattern cycles: Br-I-I-Br-I-I-...
    
    # Double perovskite
    Bp='Sn',  # Second B-site cation (creates alternating B/Bp pattern)
    
    # B-X bond distance
    BX_dist=3.18  # Angstrom (auto-calculated from ionic radii if None)
)

# Save structure
q2d.write_structure(bulk, 'MAPbI3_bulk_complete.vasp')
```

### Example 2: 2D Perovskite (DJ) with All Parameters

```python
from q2D_Materials.core.creator import q2D_creator

# Initialize creator
q2d = q2D_creator(B='Pb', X='I', A='MA', name='MAPbI3')

# Create DJ structure with all available parameters
dj = q2d.create_perovskite(
    structure_type='DJ',  # or 'RP' or 'monolayer'
    
    # Spacer molecule (required for 2D)
    spacer_molecule='[NH3+]CCCCC[NH3+]',  # SMILES, XYZ file, or Atoms object
    
    # Supercell dimensions (required for 2D)
    supercell=[2, 2, 2],  # [nx, ny, n_layers] where n_layers is octahedral layers
    
    # Pattern-based mixed compositions (optional)
    A_ions=['Cs', 'MA', 'FA', 'MA'],  # Pattern for A-site cations
    B_ions=['Pb', 'Sn'],  # Pattern for B-site cations
    X_ions=['Br', 'I'],  # Pattern for X-site anions
    
    # Spacer attachment
    attachment_end='top',  # 'top', 'bottom', or 'both' (DJ default: 'top')
    
    # Spacer penetration
    penet=0.3,  # Fraction of BX bond that spacer penetrates into layer
    
    # Spacer rotation (applied as Rx → Ry → Rz)
    Ap_Rx=0.0,  # Rotation around x-axis in degrees
    Ap_Ry=0.0,  # Rotation around y-axis in degrees
    Ap_Rz=0.0,  # Rotation around z-axis in degrees
    
    # Double perovskite
    Bp='Sn',  # Second B-site cation
    
    # B-X bond distance
    BX_dist=None,  # Auto-calculated if None
    
    # Atom wrapping
    wrap=False  # Whether to wrap atoms to unit cell
)

# Save structure
q2d.write_structure(dj, 'MAPbI3_DJ_complete.vasp')

# Visualize (optional)
q2d.view_structure(dj)
```

### Example 3: RP Structure with All Parameters

```python
rp = q2d.create_perovskite(
    structure_type='RP',
    
    spacer_molecule='[NH3+]CCCCC=O',
    supercell=[1, 1, 2],
    
    # RP-specific parameters
    spacer_distance=2.0,  # Vacuum gap between opposing spacers (Å)
    attachment_end='both',  # RP always uses 'both'
    
    # Optional parameters
    penet=0.3,
    A_ions=['MA'],  # Pattern-based A-sites
    BX_dist=None,
    wrap=False
)
```

### Example 4: Monolayer with All Parameters

```python
monolayer = q2d.create_perovskite(
    structure_type='monolayer',
    
    spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]',  # Caffeine-based
    supercell=[1, 1, 1],
    
    # Monolayer-specific parameters
    vacuum=12,  # Vacuum thickness in Angstrom
    attachment_end='both',  # Default: 'both'
    
    # Optional parameters
    penet=0.3,
    BX_dist=None,
    wrap=False
)
```

## Parameter Reference

### Bulk Perovskites
- `supercell_size` (tuple): `(nx, ny, nz)` - Supercell dimensions. Required for mixed compositions.
- `A_ions` (str/list): A-site cation(s). Single value or list pattern.
- `B_ions` (str/list): B-site cation(s). Single value or list pattern.
- `X_ions` (str/list): X-site anion(s). Single value or list pattern.
- `Bp` (str): Second B-site cation for double perovskites.
- `BX_dist` (float): B-X bond distance in Angstrom (auto-calculated if None).

### 2D Perovskites (RP, DJ, Monolayer)
- `spacer_molecule` (str/Atoms/list): **Required**. Spacer molecule(s) as SMILES, XYZ file, or Atoms object.
- `supercell` (list): **Required**. `[nx, ny, n_layers]` where `n_layers` is number of octahedral layers.
- `A_ions` (str/list): A-site cation(s) pattern.
- `B_ions` (str/list): B-site cation(s) pattern.
- `X_ions` (str/list): X-site anion(s) pattern.
- `attachment_end` (str): `'top'`, `'bottom'`, or `'both'`. DJ default: `'top'`, RP: `'both'`, Monolayer: `'both'`.
- `penet` (float): Spacer penetration into inorganic layer (fraction of BX bond, default: 0.3).
- `spacer_distance` (float): Vacuum gap between opposing spacers for RP (Å, default: 2.0).
- `vacuum` (float): Vacuum thickness for monolayer (Å, default: 12).
- `Ap_Rx`, `Ap_Ry`, `Ap_Rz` (float): Rotation angles in degrees (applied as Rx→Ry→Rz).
- `Bp` (str): Second B-site cation for double perovskites.
- `BX_dist` (float): B-X bond distance (auto-calculated if None).
- `wrap` (bool): Wrap atoms to unit cell (default: False).

## Pattern-Based Mixing

Pattern-based mixing assigns ions sequentially to positions. If the pattern list is shorter than the number of positions, it cycles through the list.

### Understanding Position Counts

**Bulk Perovskites:**
- A-sites: `nx × ny × nz` positions
- B-sites: `nx × ny × nz` positions  
- X-sites: `3 × nx × ny × nz` positions

**2D Perovskites:**
- A-sites: `(n_layers - 1) × 2 × nx × ny` positions (if n_layers > 1)
- B-sites: `2 × n_layers × nx × ny` positions
- X-sites: `(2 + 8 × n_layers) × nx × ny` positions
- Spacers: `len(z_levels) × 2 × nx × ny` positions (DJ: 1 z-level, RP: 2 z-levels)

### Pattern Examples

```python
# Pattern cycles if shorter than positions
q2d = q2D_creator(B='Pb', X='I', A='MA')

# For 2×2×2 = 8 A-sites, pattern of 3 will cycle: Cs-MA-FA-Cs-MA-FA-Cs-MA
mixed = q2d.create_perovskite('bulk',
    A_ions=['Cs', 'MA', 'FA'],  # Pattern of 3
    supercell_size=(2, 2, 2)  # 8 positions
)

# For 1×1×2 = 6 X-sites, pattern of 2 will cycle: Br-I-Br-I-Br-I
mixed_halides = q2d.create_perovskite('bulk',
    X_ions=['Br', 'I'],  # Pattern of 2
    supercell_size=(1, 1, 2)  # 6 positions
)
```

## License

MIT License
