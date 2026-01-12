# Complete List of JSON Files in q2D-Materials

This document lists all JSON files in the repository and their locations after centralization.

## Central Data Directory: `q2D_Materials/data/`

### Configuration Files (`data/config/`)
1. **structural_constants.json**
   - **Location**: `q2D_Materials/data/config/structural_constants.json`
   - **Used by**: `analyzer/structural_constants.py`
   - **Purpose**: Default constants and tolerance thresholds for structural analysis
   - **Contains**: Fitting tolerances, RMSD thresholds, search radii, hydrogen bond parameters, volume calculation tolerances
   - **Source**: Based on PDynA (https://github.com/WMD-group/PDynA)

### Template Files (`data/templates/`)
2. **cubic.json**
   - **Location**: `q2D_Materials/data/templates/cubic.json`
   - **Used by**: `builders/templates.py`
   - **Purpose**: Cubic perovskite structure template
   - **Contains**: Named layers (L1, L2, M1, RP1, RP2) with site positions

3. **hexagonal.json**
   - **Location**: `q2D_Materials/data/templates/hexagonal.json`
   - **Used by**: `builders/templates.py`
   - **Purpose**: Hexagonal perovskite structure template

4. **jagodinxky.json**
   - **Location**: `q2D_Materials/data/templates/jagodinxky.json`
   - **Used by**: `builders/templates.py`
   - **Purpose**: Jagodzinski structure template

5. **reduced.json**
   - **Location**: `q2D_Materials/data/templates/reduced.json`
   - **Used by**: `builders/templates.py`
   - **Purpose**: Reduced cell structure template

6. **salts.json**
   - **Location**: `q2D_Materials/data/templates/salts.json`
   - **Used by**: `builders/templates.py`
   - **Purpose**: Salts template variant 1

7. **salts2.json**
   - **Location**: `q2D_Materials/data/templates/salts2.json`
   - **Used by**: `builders/templates.py`
   - **Purpose**: Salts template variant 2

### Data Tables (`data/tables/`)
8. **atomic_valence.json**
   - **Location**: `q2D_Materials/data/tables/atomic_valence.json`
   - **Used by**: `utils/atomic_properties.py`
   - **Purpose**: Typical valence values for common elements
   - **Contains**: Element symbols mapped to typical valence (number of bonds)

9. **covalent_radii.json**
   - **Location**: `q2D_Materials/data/tables/covalent_radii.json`
   - **Used by**: `utils/atomic_properties.py`
   - **Purpose**: Covalent radii for elements
   - **Contains**: Element symbols mapped to covalent radii in Angstroms
   - **Source**: ASE (Atomic Simulation Environment) covalent radii database

10. **glazer_pattern_lookup.json**
    - **Location**: `q2D_Materials/data/tables/glazer_pattern_lookup.json`
    - **Used by**: `builders/glazer_notation.py`
    - **Purpose**: Glazer notation to space group lookup table
    - **Contains**: Glazer pattern strings mapped to space group identifiers

## Summary

- **Total JSON files**: 10
- **Configuration files**: 1
- **Template files**: 6
- **Data table files**: 3

## Migration Notes

All JSON files have been moved from their original locations:
- `builders/data/*.json` → `data/templates/*.json`
- `tables/*.json` → `data/tables/*.json`
- `analyzer/structural_constants.json` → `data/config/structural_constants.json`

All code references have been updated to point to the new central location.

