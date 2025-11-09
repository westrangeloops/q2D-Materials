# q2D-Materials: Quasi-2D Perovskite Structure Generation

A Python package for creating bulk and quasi-2D perovskite structures with support for mixed compositions and molecular spacers.

## Quick Start

### Installation

```bash
# Using Nix (recommended)
nix develop

# Or using pip
pip install ase numpy
```

### Basic Usage

```python
from q2D_Materials.core.creator import q2D_creator
from ase.io import write

# Initialize creator with composition
q2d = q2D_creator(B='Pb', X='I', A='MA', name='MAPbI3')

# Create bulk perovskite
bulk = q2d.create_perovskite('bulk')
bulk.write('MAPbI3_bulk.vasp', format='vasp')

# Create 2D structures (RP, DJ, or monolayer)
dj = q2d.create_perovskite('DJ', 
    spacer_molecule='[NH3+]CCCCC[NH3+]',  # SMILES string or XYZ file path
    n=2  # Number of inorganic layers
)
dj.write('MAPbI3_DJ_n2.vasp', format='vasp')
```

## Structure Types

### Bulk Perovskites

```python
# Simple bulk
bulk = q2d.create_perovskite('bulk')

# Double perovskite
double = q2d.create_perovskite('bulk', Bp='Sn')

# Mixed composition (triple-cation perovskite)
mixed = q2d.create_perovskite('bulk',
    A_ions=['Cs', 'MA', 'FA'],
    A_coefficients=[0.05, 0.79, 0.18]
)

# Mixed halides
mixed_halides = q2d.create_perovskite('bulk',
    X_ions=['Br', 'I'],
    X_coefficients=[0.5, 2.5]
)
```

### 2D Perovskites

#### Ruddlesden-Popper (RP)
```python
rp = q2d.create_perovskite('RP',
    spacer_molecule='[NH3+]CCCCC=O',  # SMILES or XYZ file
    n=2,  # Layer thickness
    spacer_distance=2.0  # Vacuum gap between spacers (Å)
)
```

#### Dion-Jacobson (DJ)
```python
dj = q2d.create_perovskite('DJ',
    spacer_molecule='[NH3+]CCCCC[NH3+]',
    n=2,
    attachment_end='top'  # 'top', 'bottom', or 'both'
)
```

#### Monolayer
```python
monolayer = q2d.create_perovskite('monolayer',
    spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]',
    n=1,
    vacuum=12,  # Vacuum thickness (Å)
    attachment_end='both'
)
```

## Spacer Molecules

Spacer molecules can be provided as:
- **SMILES strings**: `'[NH3+]CCCCC[NH3+]'`
- **XYZ file paths**: `'spacer.xyz'`
- **ASE Atoms objects**: Direct molecular structure

```python
# Using SMILES (requires RDKit)
dj = q2d.create_perovskite('DJ', spacer_molecule='[NH3+]CCCCC[NH3+]', n=2)

# Using XYZ file
from ase.io import read
spacer = read('spacer.xyz')
dj = q2d.create_perovskite('DJ', spacer_molecule=spacer, n=2)
```

## Advanced Parameters

### Bulk Perovskites
- `BX_dist` (float): B-X bond distance in Angstrom (auto-calculated if None)
- `Bp` (str): Second B-site cation for double perovskites
- `A_ions` (list): List of A-site cations for mixed compositions
- `A_coefficients` (list): Coefficients for A-site ions (must sum to 1.0)
- `B_ions` (list): List of B-site cations for mixed compositions
- `B_coefficients` (list): Coefficients for B-site ions (must sum to 1.0)
- `X_ions` (list): List of X-site anions for mixed compositions
- `X_coefficients` (list): Coefficients for X-site ions (must sum to 3.0)
- `supercell_size` (tuple): Supercell size for mixed compositions (auto-calculated)
- `seed` (int): Random seed for reproducible mixed distributions

### 2D Perovskites
- `n` (int): Number of inorganic octahedral layers (default: 1)
- `penet` (float): Spacer penetration into inorganic layer (fraction of BX bond, default: 0.3)
- `spacer_distance` (float): Vacuum gap between opposing spacers for RP (Å, default: 2.0)
- `vacuum` (float): Vacuum thickness for monolayer (Å, default: 12)
- `attachment_end` (str): Where to attach spacer - 'top', 'bottom', or 'both'
- `Ap_Rx`, `Ap_Ry`, `Ap_Rz` (float): Rotation angles in degrees (applied as Rx→Ry→Rz)
- `wrap` (bool): Wrap atoms to unit cell (default: False)
- `BX_dist` (float): B-X bond distance (auto-calculated if None)
- `Bp` (str): Second B-site cation for double perovskites

## Examples

### Triple-Cation Perovskite
```python
q2d = q2D_creator(B='Pb', X='I', A='Cs')
mixed = q2d.create_perovskite('bulk',
    A_ions=['Cs', 'MA', 'FA'],
    A_coefficients=[0.05, 0.79, 0.18]
)
```

### Mixed Halides
```python
q2d = q2D_creator(B='Pb', X='I', A='MA')
mixed = q2d.create_perovskite('bulk',
    X_ions=['Br', 'I'],
    X_coefficients=[0.5, 2.5]
)
```

### Complex Mixed Composition
```python
super_mixed = q2d.create_perovskite('bulk',
    A_ions=['Cs', 'MA', 'FA'],
    A_coefficients=[0.05, 0.79, 0.18],
    B_ions=['Pb'],
    B_coefficients=[1.0],
    X_ions=['Br', 'I'],
    X_coefficients=[0.5, 2.5]
)
```

## License

MIT License
