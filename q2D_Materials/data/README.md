# Central Data Directory

This directory contains all JSON configuration files, templates, and data tables used by q2D-Materials.

## Directory Structure

```
data/
├── config/          # Configuration files (constants, thresholds)
├── templates/       # Structure templates (layer definitions)
└── tables/          # Data tables (atomic properties, ion data, lookup tables)
```

## Files

### config/
- **structural_constants.json**: Default constants and tolerance thresholds for structural analysis
  - Used by: `analyzer/structural_constants.py`
  - Contains: Fitting tolerances, RMSD thresholds, search radii, etc.

### templates/
- **cubic.json**: Cubic perovskite template
- **hexagonal.json**: Hexagonal perovskite template
- **jagodinxky.json**: Jagodzinski template
- **reduced.json**: Reduced cell template
- **salts.json**: Salts template variant 1
- **salts2.json**: Salts template variant 2
  - Used by: `builders/templates.py`
  - Contains: Layer definitions with site positions

### tables/
- **atomic_valence.json**: Typical valence values for common elements
  - Used by: `utils/atomic_properties.py`
- **covalent_radii.json**: Covalent radii for elements (in Angstroms)
  - Used by: `utils/atomic_properties.py`
- **glazer_pattern_lookup.json**: Glazer notation to space group lookup table
  - Used by: `builders/glazer_notation.py`
- **A-ion_data.csv**: A-site ion database
- **B-ion_data.csv**: B-site ion database
- **X-ion_data.csv**: X-site ion database
- **Perovskite_ions_data.csv**: Combined perovskite ions database
  - Used by: `utils/A_sites.py`, `utils/recomender.py`

## Editing Files

All files in this directory can be edited directly. Changes will be automatically loaded by the Python modules.

**Note**: Standard JSON doesn't support comments, so we use `_comment` and `description` fields within the JSON structure for documentation.

## License

Parts of the structural analysis code are from PDynA (https://github.com/WMD-group/PDynA):
MIT License - Copyright (c) 2022 Xia Liang

