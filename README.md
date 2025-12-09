![q2D-Materials Logo](Logos/logo.png)

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
from ase.io import write

q2d = q2D_creator()

bulk = q2d.create_perovskite(
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(1, 1),          # tiling in xy
    template="cubic",             # geometry choice
)

write("MAPbI3.vasp", bulk, sort=True)
```

## Examples

### Bulk Perovskite with All Parameters (current API)

![Bulk Perovskite Structure](Logos/BULK.png)

```python
from q2D_Materials.core.creator import q2D_creator
from ase.io import write

q2d = q2D_creator()

bulk = q2d.create_perovskite(
    structure_type="bulk",
    A_ions=['Cs', 'MA', 'FA', 'Cs', 'MA', 'FA', 'Cs', 'MA'],  # pattern cycles over A-sites
    B_ions=['Pb', 'Sn'],                                      # alternating B/B'
    X_ions=['Br', 'I', 'I'],                                  # pattern cycles over X-sites
    xy_expansion=(2, 2),                                      # tiling in-plane
    template="cubic",
    Bp='Sn',                                                  # optional second B-site
    BX_dist=3.18,                                             # override bond length if desired
)

write('MAPbI3_bulk_complete.vasp', bulk, sort=True)
```

### Monolayer (key differences)

```python
from q2D_Materials.core.creator import q2D_creator
from ase.io import write

q2d = q2D_creator()

mono = q2d.create_perovskite(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(1, 1),
    template="cubic",
    thickness=2,            # repeats layer_sequence along c
    vacuum=15.0,            # Å of vacuum padding
    spacer=None,            # or a SMILES / Atoms for Ap-sites
    penetration=0.0,        # shift external A/Ap sites along z (fraction of BX)
)

write('MAPbI3_mono.vasp', mono, sort=True)
```

### Twist (two monolayers, one angle)

```python
from q2D_Materials.core.creator import q2D_creator
from ase.io import write

q2d = q2D_creator()

mono1 = q2d.create_perovskite(structure_type="monolayer", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1,1), template="cubic", vacuum=12.0)
mono2 = q2d.create_perovskite(structure_type="monolayer", A_ions="FA", B_ions="Sn", X_ions="Br", xy_expansion=(1,1), template="cubic", vacuum=12.0)

bilayer = q2d.twist(
    monolayers=[mono1, mono2],
    twist_angles=[(3, 1)],        # (m, n) tuple sets the angle
    interlayer_distances=[8.0],   # Å gap
    vacuum=12.0,
)

write('twisted_bilayer.vasp', bilayer, sort=True)
```

## Supported Structure Types

- **Bulk**: 3D perovskites (tiling via `xy_expansion`, optional `Bp`, Glazer tilts)
- **Monolayer**: 2D slabs with `thickness`, `vacuum`, `spacer`, `penetration`
- **Twist**: Build twisted stacks from monolayers via `twist(...)`

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

For tutorials and examples, see the `Examples/` folder (Creator, Templates, Glazer, Jagodzinski, Monolayer, Twist).

## License

MIT License
