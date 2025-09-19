# q2D-Materials:  Yet another Quasi-2D Perovskite Structure Generation and Analysis.

A Python package for creating and analyzing quasi-2D perovskite structures.

## Quick Start

### Installation
```bash
conda create -n q2D_Materials pip
conda activate q2D_Materials
pip install ase pandas numpy matplotlib networkx
```

### Create a Structure

```python
from q2D_Materials.utils.perovskite_builder import make_dj, auto_calculate_BX_distance
from q2D_Materials.utils.common_a_sites import get_a_site_object
from q2D_Materials.utils.file_handlers import mol_load
from ase import Atoms
from ase.io import write

# 1. Load spacer molecule
spacer_df = mol_load("spacer.xyz")
spacer_elements = spacer_df['Element'].tolist()
spacer_positions = spacer_df[['X', 'Y', 'Z']].values
Ap_spacer = Atoms(symbols=spacer_elements, positions=spacer_positions)

# 2. Get A-site cation (methylammonium)
A_cation = get_a_site_object("MA")

# 3. Calculate B-X distance and penetration
bx_dist = auto_calculate_BX_distance('Pb', 'I')
penetration_fraction = 0.2 / bx_dist

# 4. Create Dion-Jacobson perovskite structure
dj_structure = make_dj(
    Ap_spacer=Ap_spacer,      # Spacer molecule
    A_site_cation=A_cation,   # A-site cation
    B_site_cation='Pb',       # Metal cation
    X_site_anion='I',         # Halide anion
    n=2,                      # Number of inorganic layers
    BX_dist=bx_dist,
    penet=penetration_fraction,
    attachment_end='top',
    wrap=True
)

# 5. Save structure
write("MAPbI3_DJ_n2.vasp", dj_structure, format='vasp')
```

### Analyze a Structure

```python
from q2D_Materials.core.analyzer import q2D_analyzer

# Initialize analyzer
analyzer = q2D_analyzer(
    file_path="structure.vasp",
    b='Pb',  # Central atom
    x='I',   # Ligand atom
    cutoff_ref_ligand=3.5
)

# Calculate octahedral distortions
distortions = analyzer.calculate_octahedral_distortions()

# Get summary DataFrame
summary_df = analyzer.get_distortion_summary()

# Save spacer and salt components
analyzer.save_spacer()   # Organic spacer only
analyzer.save_salt()     # Spacer + coordinating halides
```

## Structure Types

- **Dion-Jacobson**: `make_dj()` - Alternating organic/inorganic layers
- **Ruddlesden-Popper**: `make_2drp()` - Organic spacer between inorganic slabs
- **Monolayer**: `make_monolayer()` - Single inorganic layer

## Analysis Features

- **Octahedral Distortion**: Calculate distortion parameters (ζ, Δ, Σ, θ)
- **Structural Analysis**: Bond lengths, angles, and geometric properties
- **Component Isolation**: Extract organic spacers and inorganic components
- **Systematic Ordering**: Consistent octahedra ordering for comparison

## File Formats

- **Input**: VASP (POSCAR/CONTCAR), XYZ
- **Output**: VASP, XYZ, CSV (analysis results)

## Examples

See the `Examples/` folder for detailed usage examples and batch processing scripts.

## Citation

If you use this code, please cite:

> Stanton, R., & Trivedi, D. (2023). Pyrovskite: A software package for the high throughput construction, analysis, and featurization of two- and three-dimensional perovskite systems. *Journal of Applied Physics*, 133(24), 244701.

## License

MIT License
