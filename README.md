![q2D-Materials Logo](Logos/SVC_logo.png)

# q2D-Materials: Quasi-2D Perovskite Structure Generation

Another Python package for creating bulk and quasi-2D perovskite structures with support for mixed compositions and molecular spacers.

## Quick Start

### Installation

```bash
# Using Nix (recommended)
nix develop

# Or using pip
pip install ase numpy rdkit
```

![Nix Logo](https://nixos.org/_astro/nixos-logo-default-gradient-black-regular-horizontal-none.BPpok6mb_JppMK.svg)

**Why Nix?** For computational chemistry and materials science, reproducibility is critical—your results should be independent of your system's Python version or library installations. Nix ensures that everyone working with q2D-Materials uses identical environments with the exact same versions of ASE, RDKit, NumPy, and all dependencies, eliminating the classic "works on my machine" problem. This is especially valuable when sharing structures with collaborators or reproducing published results, as molecular structure generation is sensitive to numerical precision and library versions.

### Basic Usage

```python
from q2D_Materials.core.creator import q2D_creator

# Initialize empty creator
q2d = q2D_creator()

# Create a structure (all composition parameters required)
bulk = q2d.create_perovskite('bulk',
    A_ions='MA', B_ions='Pb', X_ions='I',
    supercell_size=(1, 1, 1))

# You can use any way to save or modify it with ase:
from ase.io import write
write("MAPbI3.vasp", bulk) # Cif, Vasp, etc ...
```

## Examples

### Bulk Perovskite with All Parameters

![Bulk Perovskite Structure](Logos/BULK.png)

```python
from q2D_Materials.core.creator import q2D_creator

# Initialize empty creator
q2d = q2D_creator()

# Create bulk perovskite with all available parameters
bulk = q2d.create_perovskite(
    structure_type='bulk',
    
    # Required composition parameters
    A_ions=['Cs', 'MA', 'FA', 'Cs', 'MA', 'FA', 'Cs', 'MA'],  # 8 positions
    B_ions=['Pb', 'Sn'],  # Alternating pattern
    X_ions=['Br', 'I', 'I'],  # Pattern cycles: Br-I-I-Br-I-I-...
    
    # Supercell size (required for mixed compositions, optional for uniform)
    supercell_size=(2, 2, 2),  # (nx, ny, nz)
    
    # Double perovskite
    Bp='Sn',  # Second B-site cation (creates alternating B/Bp pattern)
    
    # B-X bond distance
    BX_dist=3.18  # Angstrom (auto-calculated from ionic radii if None)
)

# Save structure
from ase.io import write
write('MAPbI3_bulk_complete.vasp', bulk)
```

### Ruddlesden-Popper (RP) Structure with All Parameters

![RP Perovskite Structure](Logos/RP.png)

```python
from q2D_Materials.core.creator import q2D_creator

# Initialize empty creator
q2d = q2D_creator()

# Create RP structure with all available parameters
rp = q2d.create_perovskite(
    structure_type='RP',
    
    # Required composition parameters
    A_ions=['Cs', 'MA', 'FA', 'MA'],  # Pattern for A-site cations
    B_ions=['Pb', 'Sn'],  # Pattern for B-site cations
    X_ions=['Br', 'I'],  # Pattern for X-site anions
    
    # Required 2D parameters
    spacer_molecule='[NH3+]CCCCC=O',  # SMILES, XYZ file, or Atoms object
    supercell=[1, 1, 2],  # [nx, ny, n_layers] where n_layers is octahedral layers

    # Spacer penetration
    penet=0.3,  # Fraction of BX bond that spacer penetrates into layer
    
    # B-X bond distance
    BX_dist=None  # Auto-calculated if None from B/X ions
)

# Save structure
from ase.io import write
write('MAPbI3_RP_complete.vasp', rp)
```

## Supported Structure Types

- **Bulk**: 3D perovskite structures with optional supercells and mixed compositions
- **RP (Ruddlesden-Popper)**: 2D layered structures with organic spacers
- **DJ (Dion-Jacobson)**: 2D layered structures with divalent organic spacers
- **Monolayer**: Single-layer 2D structures with vacuum support adsorbates and rotations.

## Ion Recommender

Get recommendations for compatible ions based on database occurrence:

```python
q2d = q2D_creator()

# Get recommendations based on X = Cl
recommendations = q2d.recommend(X='Cl', top_n=5)
print(recommendations['B'])  # Top 5 B-site cations
print(recommendations['A'])  # Top 5 A-site cations
print(recommendations['spacer'])  # Top 5 spacers
```

## Documentation

For a detailed tutorials and detailed parameter reference, see [Examples/Creator.md](Examples/Creator.md).

## License

MIT License
