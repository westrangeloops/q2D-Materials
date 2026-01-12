![q2D-Materials Logo](../Logos/logo.png)

# Molecule Validator — Checking if Molecules Work as Spacers

Check if molecules are suitable as DJ (Dion-Jacobson) or RP (Ruddlesden-Popper) spacers. Works with SMILES strings, files, or Atoms objects—no structure needed.

**New in v2.2**: Pattern-based validation with configurable SMILES patterns for terminal groups. Backbone element validation filters out molecules with unwanted elements (P, S, metals, etc.)

## Quick Start

```python
from q2D_Materials.analyzer import q2D_analyzer

analyzer = q2D_analyzer()

# Check DJ spacer (needs 2 terminal NH2/NH3 groups)
result = analyzer.analyze_molecule_as_dj_spacer("NCCCCN")
print(f"Valid: {result.is_valid}")  # True

# Check RP spacer (needs 1+ terminal NH2/NH3 group)
result = analyzer.analyze_molecule_as_rp_spacer("NCCCCC")
print(f"Valid: {result.is_valid}")  # True
```

## Simple Examples

### Basic Check

```python
analyzer = q2D_analyzer()
result = analyzer.analyze_molecule_as_dj_spacer("NCCCCN")

if result.is_valid:
    print(f"✓ Valid! {len(result.terminal_groups)} terminals, {len(result.valid_paths)} paths")
else:
    print(f"✗ Invalid: {result.reason}")
```

### Compare DJ vs RP

```python
analyzer = q2D_analyzer()

# Molecule with 2 terminals (works for both)
dj = analyzer.analyze_molecule_as_dj_spacer("NCCCCN")  # True
rp = analyzer.analyze_molecule_as_rp_spacer("NCCCCN")  # True

# Molecule with 1 terminal (only RP)
dj = analyzer.analyze_molecule_as_dj_spacer("NCCCCC")  # False
rp = analyzer.analyze_molecule_as_rp_spacer("NCCCCC")  # True
```

### From Files

```python
from ase.io import read
analyzer = q2D_analyzer()

molecule = read("spacer.xyz")
result = analyzer.analyze_molecule_as_dj_spacer(molecule)
```

## Understanding Results

```python
result = analyzer.analyze_molecule_as_dj_spacer("NCCCCN")

# Basic info
result.is_valid          # True/False
result.reason            # Explanation
result.spacer_type       # "DJ" or "RP"

# Terminal groups
for group in result.terminal_groups:
    print(f"{group.group_type} at index {group.n_index}")
    print(f"  H atoms: {group.h_indices}")
    print(f"  Carbon neighbor: {group.carbon_neighbor}")

# Paths (DJ only)
result.valid_paths      # List of valid paths between terminals

# Original molecule
atoms = result.original_atoms  # ASE Atoms object
```

## Pattern-Based Matching

### Custom Terminal Patterns

The new pattern-based API allows you to specify custom SMILES patterns for terminal groups:

```python
analyzer = q2D_analyzer()

# Use ammonium pattern [NH3+]C
result = analyzer.analyze_molecule_as_dj_spacer(
    molecule,
    initial_pattern='[NH3+]C',
    final_pattern='[NH3+]C'
)

# Use multiple patterns (matches any of them)
result = analyzer.analyze_molecule_as_dj_spacer(
    molecule,
    initial_pattern=['NH2C', '[NH3+]C'],
    final_pattern=['NH2C', '[NH3+]C']
)

# Asymmetric spacers (different initial and final patterns)
result = analyzer.analyze_molecule_as_dj_spacer(
    molecule,
    initial_pattern='[NH3+]C',
    final_pattern='NH2C'
)
```

### Converting and Elongating

### NH2 → NH3 Conversion

```python
from q2D_Materials.analyzer.characterization.molecule_candidates import convert_nh2_to_nh3

analyzer = q2D_analyzer()
result = analyzer.analyze_molecule_as_dj_spacer("NCCN", initial_pattern='NH2C')

# Convert NH2 to NH3 (requires nitrogen index)
# Note: With pattern-based API, you may need to identify N indices from pattern matches
from q2D_Materials.analyzer.characterization.molecule_candidates import clean_molecule

# Use clean_molecule to automatically convert all NH2 to NH3
modified = clean_molecule(result.original_atoms, convert_nh2_to_nh3_flag=True)
from ase.io import write
write("spacer_nh3.xyz", modified)
```

### Elongate DJ Spacers

```python
result = analyzer.analyze_molecule_as_dj_spacer("NCCN")

# Elongate to target distance
elongated = analyzer.elongate_dj_spacer(
    result.original_atoms,
    target_distance=12.0
)

# Or maximize elongation
elongated = analyzer.elongate_dj_spacer(result.original_atoms)
```

## Intermediate Examples

### Batch Processing

```python
analyzer = q2D_analyzer()

candidates = {
    "Butanediamine": "NCCCCN",
    "Ethylenediamine": "NCCN",
    "Pentylamine": "NCCCCC"
}

valid_dj = [name for name, smiles in candidates.items() 
            if analyzer.analyze_molecule_as_dj_spacer(smiles).is_valid]
valid_rp = [name for name, smiles in candidates.items() 
            if analyzer.analyze_molecule_as_rp_spacer(smiles).is_valid]

print(f"Valid DJ: {valid_dj}")
print(f"Valid RP: {valid_rp}")
```

### Complete Workflow

```python
from q2D_Materials.analyzer import q2D_analyzer
from ase.io import write

analyzer = q2D_analyzer()

# 1. Analyze
result = analyzer.analyze_molecule_as_dj_spacer("NCCCCN")
if not result.is_valid:
    exit()

# 2. Convert NH2 to NH3 if needed
molecule = result.original_atoms
if any(g.group_type == "NH2" for g in result.terminal_groups):
    molecule = analyzer.convert_molecule_nh2_to_nh3(molecule)

# 3. Elongate
elongated = analyzer.elongate_dj_spacer(molecule, target_distance=15.0)
write("spacer_final.xyz", elongated)
```

### With Modifier Module

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.modifier import from_smiles

analyzer = q2D_analyzer()
molecule = from_smiles("NCCCCN")
result = analyzer.analyze_molecule_as_dj_spacer(molecule)
```

## Advanced Examples

### Filter by Terminal Group Type

```python
analyzer = q2D_analyzer()

molecules = ["NCCCCN", "C[NH3+]", "NCCN"]

# Find molecules with exactly 2 NH3 groups
dj_with_2_nh3 = []
for smiles in molecules:
    result = analyzer.analyze_molecule_as_dj_spacer(smiles)
    if result.is_valid:
        nh3_count = sum(1 for g in result.terminal_groups if g.group_type == "NH3")
        if nh3_count == 2:
            dj_with_2_nh3.append(smiles)
```

### Custom Validation

```python
def is_good_dj_spacer(smiles, min_path_length=3):
    result = analyzer.analyze_molecule_as_dj_spacer(
        smiles, min_chain_length=min_path_length
    )
    if not result.is_valid or len(result.valid_paths) == 0:
        return False
    return max(len(p) for p in result.valid_paths) >= min_path_length + 2

# Test
for smiles in ["NCCCCN", "NCCN", "NCCCCCCN"]:
    print(f"{smiles}: {is_good_dj_spacer(smiles, min_path_length=4)}")
```

### Pattern-Based Validation with Custom Patterns

```python
# Validate with custom patterns
result = analyzer.analyze_molecule_as_dj_spacer(
    molecule,
    initial_pattern='[NH3+]C',
    final_pattern='[NH3+]C',
    min_chain_length=3,
    allowed_backbone_elements={'C', 'N', 'O'},
    forbidden_backbone_elements={'P', 'S'},
    max_non_carbon_ratio=0.2
)
```

### Integration with Structure Creation

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

analyzer = q2D_analyzer()
creator = q2D_creator()

spacer_smiles = "C(CC[NH3+])C[NH3+]"  # 1,4-butanediammonium
result = analyzer.analyze_molecule_as_dj_spacer(spacer_smiles)

if result.is_valid:
    # Prepare spacer
    molecule = result.original_atoms
    # Convert NH2 to NH3 if needed (using clean_molecule)
    from q2D_Materials.analyzer.characterization.molecule_candidates import clean_molecule
    molecule = clean_molecule(molecule, convert_nh2_to_nh3_flag=True)
    molecule = analyzer.elongate_dj_spacer(molecule, target_distance=12.0)
    
    # Use in structure
    structure = creator.create_structure(
        template="cubic",
        structure_type="dj",
        A_ions="MA", B_ions="Pb", X_ions="I",
        spacer=spacer_smiles,
        xy_expansion=(1, 1)
    )
```

## Backbone Element Validation

### Overview

Control which chemical elements are allowed in the backbone path between terminal groups. This helps filter out molecules with unwanted elements (phosphorus, metals, etc.) that aren't suitable as spacers.

### Default Settings

```python
# Default allowed backbone elements
allowed_backbone_elements = {'C', 'N', 'O'}

# Default forbidden backbone elements
forbidden_backbone_elements = {
    'P', 'Si', 'B', 'Se', 'I', 'As', 'Ge',
    'Sn', 'Pb', 'Bi', 'Al', 'Ti', 'Fe', 'Cu', 'Zn'
}

# Default maximum non-carbon ratio (excluding H, N)
max_non_carbon_ratio = 0.3  # 30% max
```

### Basic Usage

```python
from q2D_Materials.analyzer import q2D_analyzer

analyzer = q2D_analyzer()

# Use defaults (allows only C, N, O in backbone)
result = analyzer.analyze_molecule_as_dj_spacer("NCCCCN")
print(result.is_valid)  # True

# Molecule with phosphorus (P) in backbone - REJECTED
result = analyzer.analyze_molecule_as_dj_spacer(
    molecule_with_P,
    forbidden_backbone_elements={'P', 'Si', 'B', 'Se', 'I', 'As', 'Ge', 'Sn', 'Pb', 'Bi', 'Al', 'Ti', 'Fe', 'Cu', 'Zn'}
)
print(result.is_valid)  # False
print(result.reason)    # "No valid backbone path found between pattern matches"
```

### Custom Element Validation

**Example 1: Allow only carbon and nitrogen**

```python
analyzer = q2D_analyzer()

# Strict: only C and N in backbone
result = analyzer.analyze_molecule_as_dj_spacer(
    "NCCCCN",
    allowed_backbone_elements={'C', 'N'}
)
```

**Example 2: Forbid specific elements**

```python
# Explicitly forbid phosphorus and sulfur
result = analyzer.analyze_molecule_as_dj_spacer(
    molecule,
    forbidden_backbone_elements={'P', 'S', 'Si'}
)
```

**Example 3: Control carbon ratio**

```python
# Require at least 80% carbon in backbone (excluding H, N)
result = analyzer.analyze_molecule_as_dj_spacer(
    molecule,
    max_non_carbon_ratio=0.2  # Max 20% non-carbon
)
```

**Example 4: Custom terminal patterns with element validation**

```python
# Use ammonium pattern with strict backbone validation
result = analyzer.analyze_molecule_as_dj_spacer(
    molecule,
    initial_pattern='[NH3+]C',
    final_pattern='[NH3+]C',
    allowed_backbone_elements={'C', 'N', 'O'},
    forbidden_backbone_elements={'P', 'S'},
    max_non_carbon_ratio=0.25
)
```

### Complete Example: Batch Validation with Custom Rules

```python
from q2D_Materials.analyzer import q2D_analyzer
from ase.io import read
import os

analyzer = q2D_analyzer()

# Custom validation rules
ALLOWED_ELEMENTS = {'C', 'N', 'O'}
FORBIDDEN_ELEMENTS = {'P', 'S', 'Si', 'B'}
MAX_NON_CARBON = 0.25

valid_molecules = []
rejected_molecules = []

# Process all molecules in directory
for pkl_file in os.listdir("spacer_molecules/"):
    if not pkl_file.endswith(".pkl"):
        continue

    molecule = read(f"spacer_molecules/{pkl_file}")

    result = analyzer.analyze_molecule_as_dj_spacer(
        molecule,
        allowed_backbone_elements=ALLOWED_ELEMENTS,
        forbidden_backbone_elements=FORBIDDEN_ELEMENTS,
        max_non_carbon_ratio=MAX_NON_CARBON
    )

    if result.is_valid:
        valid_molecules.append(pkl_file)
    else:
        rejected_molecules.append((pkl_file, result.reason))

# Report results
print(f"✓ Valid molecules: {len(valid_molecules)}")
print(f"✗ Rejected molecules: {len(rejected_molecules)}")

for filename, reason in rejected_molecules:
    print(f"  {filename}: {reason}")
```

### Why This Matters

**Problem**: Some molecules pass structural validation (have NH2/NH3 groups, continuous paths) but contain problematic elements:

- **Phosphorus (P)**: Nucleotide derivatives, phosphates
- **Sulfur (S)**: Sulfones, sulfonates
- **Metals**: Coordination complexes
- **Silicon (Si)**: Siloxanes, silanes

**Solution**: Backbone element validation automatically filters these out:

```python
# Example: Nucleotide-like molecule (has P in backbone)
# Formula: C28H34N10O23P3
# Has 2 valid NH2 groups, but path goes through phosphorus

result = analyzer.analyze_molecule_as_dj_spacer(
    nucleotide_molecule,
    forbidden_backbone_elements={'P', 'Si', 'B', 'Se', 'I', 'As', 'Ge', 'Sn', 'Pb', 'Bi', 'Al', 'Ti', 'Fe', 'Cu', 'Zn'}
)
# Result: is_valid=False
# Reason: "No valid backbone path found between pattern matches"
```

### Advanced: Per-Project Custom Validators

```python
class SpacerValidator:
    """Custom validator for specific project requirements."""

    def __init__(self, project_type="organic"):
        self.analyzer = q2D_analyzer()

        if project_type == "organic":
            self.allowed = {'C', 'N', 'O'}
            self.forbidden = {'P', 'Si', 'B', 'As', 'Se'}
            self.carbon_ratio = 0.3
        elif project_type == "halogenated":
            self.allowed = {'C', 'N', 'O', 'F', 'Cl', 'Br'}
            self.forbidden = {'P', 'Si', 'B'}
            self.carbon_ratio = 0.4

    def validate_dj(self, molecule, initial_pattern=None, final_pattern=None):
        return self.analyzer.analyze_molecule_as_dj_spacer(
            molecule,
            initial_pattern=initial_pattern,
            final_pattern=final_pattern,
            allowed_backbone_elements=self.allowed,
            forbidden_backbone_elements=self.forbidden,
            max_non_carbon_ratio=self.carbon_ratio
        )

# Usage
validator = SpacerValidator(project_type="organic")
result = validator.validate_dj("NCCCCN")
```

### Direct API Access

For advanced users, import the functions directly:

```python
from q2D_Materials.analyzer.characterization.molecule_candidates import (
    analyze_molecule_candidate,
    DEFAULT_ALLOWED_BACKBONE_ELEMENTS,
    DEFAULT_FORBIDDEN_BACKBONE_ELEMENTS,
    DEFAULT_MAX_NON_CARBON_RATIO
)

# Check current defaults
print(f"Allowed: {DEFAULT_ALLOWED_BACKBONE_ELEMENTS}")
print(f"Forbidden: {DEFAULT_FORBIDDEN_BACKBONE_ELEMENTS}")
print(f"Max non-C ratio: {DEFAULT_MAX_NON_CARBON_RATIO}")

# Use directly
result = analyze_molecule_candidate(
    molecule,
    spacer_type="DJ",
    allowed_backbone_elements={'C', 'N'},
    forbidden_backbone_elements={'P', 'S', 'Si'},
    max_non_carbon_ratio=0.2
)
```

## Key Concepts

**DJ Spacers**: Bifunctional, need 2 terminal groups matching specified patterns, connected through carbon backbone. Pattern: `{pattern} - C - {chain} - C - {pattern}`. Example: `C(CC[NH3+])C[NH3+]` (1,4-butanediammonium).

**RP Spacers**: Monofunctional, need 1+ terminal group matching specified pattern bonded to carbon. Pattern: `{pattern} - C`. Example: `C[NH3+]` (Methylammonium).

**Pattern-Based Matching**: Uses SMILES pattern matching to identify terminal groups. Default patterns: `'NH2C'` (matches both NH2 and NH3 groups). Custom patterns: `'[NH3+]C'` for ammonium, `'NH2C'` for amine.

**Terminal Groups**: Detected via SMILES pattern matching. Common patterns: `'NH2C'` (amine), `'[NH3+]C'` (ammonium).

## Common Patterns

**Diamines (DJ)**: `C(C[NH3+])[NH3+]`, `C(CC[NH3+])C[NH3+]`, `C(CCCC[NH3+])CCC[NH3+]`

**Monoamines (RP)**: `C[NH3+]`, `CC[NH3+]`, `CCCC[NH3+]`, `C1=CC=C(C=C1)CC[NH3+]`

## Tips & Troubleshooting

- Start with simple molecules: `"NCCCCN"`, `"C[NH3+]"`
- Check `result.reason` if invalid
- Works standalone—no structure needed
- Accepts SMILES, ASE Atoms, or file paths
- Use backbone element validation to filter out unwanted chemical elements

**Common issues**:
- Invalid SMILES → check syntax
- No pattern matches → molecule doesn't match specified terminal patterns
- No valid path (DJ) → terminals not connected via carbon backbone
- Valid structure but rejected → check if backbone contains forbidden elements (P, S, metals)
- Molecule has pattern matches but no valid paths → try adjusting `allowed_backbone_elements` or `forbidden_backbone_elements` if defaults are too strict
- Pattern not found → try different patterns like `'[NH3+]C'` for ammonium groups
