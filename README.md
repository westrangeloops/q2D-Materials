![q2D-Materials Logo](Logos/logo.png)

# q2D-Materials

A template-driven toolkit for stacking quasi-2D perovskite architecture (DJ, RP, ACI, and hybrids). Pick layers, set `layer_sequence`, and the library handles ion assignment, tilts, spacers, and exports in ASE/vasp-friendly formats.

## Quick start

1. `devenv shell`  ← downloads pre-built dependencies from Cachix and gives you Python 3.12 with RDKit, ASE, FastAPI, etc.
2. Inside the shell:
   ```bash
   test    # run the full pytest suite
   example # verify the core creator still builds something
   ```

## Quick examples

### 1. Build a Dion–Jacobson block

```python
from q2D_Materials.core.creator import q2D_creator

dj = q2D_creator().create_structure(
    structure_type="dj",
    template="reduced",
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    layer_sequence="M1-M1",
)
```

*See `Examples/7_DionJacobson.md` for the full recipe and rendering.*

### 2. Build a Ruddlesden–Popper slab

```python
from q2D_Materials.core.creator import q2D_creator

rp = q2D_creator().create_structure(
    structure_type="rp",
    layer_sequence="L1-RP1-RP2",
    template="glazer",
)
```

*See `Examples/8_Ruddlessden_Popper.md` for the plan/side/iso views.*

### 3. Analyze a structure you already have

```python
from q2D_Materials.analyzer.core import Analyzer

ana = Analyzer("path/to/structure.cif")
print(ana.get_octahedral_tilts())
```

*Check `Examples/12_Analysis.md` through `Examples/22_DistortionAnalysis.md` for analyzers, graph queries, and molecule tools.*

## Installation

```bash
# Preferred: Nix + cachix-powered devenv
devenv shell
```

Alternative for prototyping:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Creating structures
Lessons 1–11 in `Examples/README.md` walk through DJs, RPs, twist stacks, and monolayers. Jump there for ready-to-run scripts that call `q2D_creator()`.

### Analyzing structures
Lessons 12–22 explain how to inspect tilts, RDFs, molecule graphs, and distortion metrics. Run the analyzer snippets inside the shell or point them at your saved structures.

### Advanced features
Want to customize templates or automate up a new spacer? The Examples folder also demonstrates template JSON editing, Glazer patterns, and twisted interfaces.

## Where to go next
- `Examples/README.md` – pick a recipe and run it with `python Examples/<number>.py` (or paste the snippet into a script).
- `dev/CONTRIBUTING.md` – for contributors ready to change the code.
- `dev/CACHIX_SETUP.md` – for maintainers who keep the caches, secrets, and workflows healthy.

## Citation
1. Kunz, S. L. et al., “`hexagonal` perovskites: From stacking sequence to space group symmetry...” *Chem. Mater.* 36, 23 (2024).
2. Stanton, R. & Trivedi, D. J., “Pyrovskite: A software package for high-throughput perovskite construction.” *J. Chem. Phys.* 159, 6 (2023).

## License
GNU GENERAL PUBLIC LICENSE v3.0