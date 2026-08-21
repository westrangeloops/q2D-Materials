# Data Tables Directory

This directory contains atomic properties and lookup tables used throughout q2D-Materials.

## Atomic Properties Files

### covalent_radii.json
- **Used by**: `utils/properties/atomic_properties.py` → `utils/molecules/graph_converter.py`
- **Purpose**: Single bond covalent radii for all elements
- **Format**: JSON object mapping element symbols to radii (in Angstroms)
- **Source**: ASE (Atomic Simulation Environment) covalent radii database
- **Functions using this file**:
  - `get_covalent_radius()`: Get single bond covalent radius for an element
  - `are_atoms_bonded()`: Check if two atoms are bonded based on distance

### covalent_radii_bond_order.json
- **Used by**: `utils/properties/atomic_properties.py` → `utils/molecules/graph_converter.py`
- **Purpose**: Bond-order-specific covalent radii (single, double, triple bonds)
- **Format**: JSON object mapping element symbols to dictionaries with 'single', 'double', 'triple' keys
- **Source**: First-principle calculations for bond-order-specific radii
- **Functions using this file**:
  - `get_covalent_radius_by_bond_order()`: Get radius for specific bond order
  - `estimate_bond_order()`: Estimate bond order from interatomic distance
  - Used in charge inference and bond detection in `graph_converter.py`

### atomic_valence.json
- **Used by**: `utils/properties/atomic_properties.py`
- **Purpose**: Typical valence values (maximum number of bonds) for common elements
- **Format**: JSON object mapping element symbols to typical valence (integer)
- **Functions using this file**:
  - `get_valence()`: Get typical valence for an element
  - `validate_replacement()`: Validate atom replacement feasibility

## Perovskite Ion Data Files

### A-ion_data.csv
- **Used by**: `utils/A_sites.py`, `utils/recomender.py`
- **Purpose**: A-site ion database for perovskite structures
- **Contains**: Ion names, radii, and other properties

### B-ion_data.csv
- **Used by**: `utils/A_sites.py`, `utils/recomender.py`
- **Purpose**: B-site ion database for perovskite structures
- **Contains**: Ion names, radii, and other properties

### X-ion_data.csv
- **Used by**: `utils/A_sites.py`, `utils/recomender.py`
- **Purpose**: X-site ion database for perovskite structures
- **Contains**: Ion names, radii, and other properties

### Perovskite_ions_data.csv
- **Used by**: `utils/A_sites.py`, `utils/recomender.py`
- **Purpose**: Combined perovskite ions database
- **Contains**: Comprehensive ion data from all sites
- **Source**: See `disclaimer.MD` for citation information

## Lookup Tables

### glazer_pattern_lookup.json
- **Used by**: `builders/glazer_notation.py`
- **Purpose**: Glazer notation to space group lookup table
- **Format**: JSON object mapping Glazer pattern strings to space group identifiers

## Usage in Graph Converter

The `graph_converter.py` module uses atomic properties from this directory via the `atomic_properties.py` module:

1. **Bond Detection**: Uses `are_atoms_bonded()` which reads from `covalent_radii.json`
2. **Bond Order Estimation**: Uses `estimate_bond_order()` which reads from `covalent_radii_bond_order.json`
3. **Charge Inference**: Uses bond-order-specific radii to distinguish between single, double, and aromatic bonds

All atomic property functions automatically load data from these JSON files on first use and cache the results for performance.

## Editing Files

All files in this directory can be edited directly. Changes will be automatically loaded by the Python modules (cached data will be refreshed on next import).

**Note**: Standard JSON doesn't support comments, so we use `_comment` and `description` fields within the JSON structure for documentation.

## File Locations

All files are located at:
```
q2D_Materials/data/tables/
```

Code accesses these files via:
```python
from pathlib import Path
data_dir = Path(__file__).parent.parent.parent / "data" / "tables"
```

