<!-- 71ccd070-1461-4fdc-86c7-d7ab29b9a0c3 f4d2e3de-fbcd-4720-820c-d47e66a2f5ea -->
# JSON-Based Wyckoff Position Template System

## Overview

Replace hardcoded position templates with a JSON-based template system that reads Wyckoff positions from user-defined JSON files. Templates are stored in a folder and can be extended by users. Works for both bulk and 2D structures (2D is priority).

## Architecture

### 1. Template Folder Structure

   - Create `q2D_Materials/templates/` directory
   - Store JSON template files (e.g., `Cubic_perovskite.json`, `Orthorhombic_perovskite.json`)
   - Users can add custom JSON files following the schema
   - Template files are discovered by filename pattern matching

### 2. JSON Template Schema

   - Based on provided perovskite composition schema
   - Extended to include:
     - `space_group`: Space group symbol (e.g., "Pm3m", "Pnma")
     - `point_group`: Point group symbol
     - `lattice_type`: "cubic", "orthorhombic", etc.
     - `wyckoff_positions`: Object with A, B, X site Wyckoff positions
     - `lattice_parameters`: Optional lattice parameter specification
   - Keep required fields: `ions_a_site`, `ions_b_site`, `ions_x_site` (can be placeholders for template)
   - Structure type indicator: "bulk" or "2d" (or both)

### 3. Template Loader (`q2D_Materials/utils/template_loader.py`)

   - Function to scan templates folder for JSON files
   - Load and validate JSON templates
   - Parse Wyckoff positions and space group information
   - Return template data structure for use in structure generation

### 4. Wyckoff Position Expansion (`q2D_Materials/utils/wyckoff_expander.py`)

   - Function to expand Wyckoff positions to all equivalent positions using pymatgen
   - Use pymatgen's SpacegroupAnalyzer for symmetry operations
   - Convert fractional coordinates to position lists
   - Handle special positions (e.g., 3d in cubic generates 3 positions)

### 5. Template Data Structure (JSON format)

   ```json
   {
       "space_group": "Pm3m",
       "point_group": "m3m",
       "lattice_type": "cubic",
       "structure_types": ["bulk", "2d"],
       "wyckoff_positions": {
           "B": [{"wyckoff": "1a", "position": [0, 0, 0]}],
           "A": [{"wyckoff": "1b", "position": [0.5, 0.5, 0.5]}],
           "X": [{"wyckoff": "3d", "position": [0, 0.5, 0]}]
       },
       "lattice_parameters": {
           "type": "cubic",
           "a": null,
           "b": null,
           "c": null,
           "calculation_method": "BX_distance"
       },
       "ions_a_site": [],
       "ions_b_site": [],
       "ions_x_site": []
   }
   ```

### 6. Modify Position Template Functions

   - Update `_get_bulk_position_template()` to accept `template_name` parameter
   - Update `_get_2d_layer_position_template()` to accept `template_name` parameter
   - Load template from JSON file based on name
   - Expand Wyckoff positions to full position list
   - Maintain backward compatibility with default behavior

### 7. Update API Functions

   - Add `template_name` parameter to `create_perovskite()` and `create_bulk_perovskite()`
   - Add `template_name` parameter to `create_2d_perovskite()`
   - Template name maps to JSON filename (e.g., "Cubic" -> "Cubic_perovskite.json")
   - Default behavior: use existing hardcoded templates if no template specified

## Implementation Steps

1. **Create template storage module** (`wyckoff_templates.py`)

   - Define WYCKOFF_TEMPLATES dictionary
   - Add validation functions
   - Include helper functions to query templates

2. **Create Wyckoff expander** (`wyckoff_expander.py`)

   - Implement `expand_wyckoff_position()` function
   - Use pymatgen SpacegroupAnalyzer for symmetry operations
   - Generate all equivalent positions from Wyckoff notation
   - Return list of fractional coordinates

3. **Refactor `_get_bulk_position_template()`**

   - Add `space_group` parameter (default: 'Pm3m')
   - Load template from WYCKOFF_TEMPLATES
   - Expand Wyckoff positions to full position list
   - Return same dict format as before for compatibility

4. **Update lattice parameter handling**

   - For cubic: keep current BX_dist-based calculation
   - For orthorhombic: may need separate a, b, c parameters
   - Add `lattice_parameters` parameter to `create_bulk_perovskite()`

5. **Update public API**

   - Add `space_group` parameter to `create_perovskite()` and `create_bulk_perovskite()`
   - Maintain backward compatibility (default to cubic)
   - Update docstrings

6. **Testing**

   - Test cubic structure (should match current behavior)
   - Test orthorhombic structure generation
   - Verify position counts match Wyckoff multiplicities

## Files to Modify

- `q2D_Materials/utils/perovskite_builder.py` - Update template functions and API
- `q2D_Materials/core/creator.py` - Pass through space_group parameter

## Files to Create

- `q2D_Materials/utils/wyckoff_templates.py` - Template definitions
- `q2D_Materials/utils/wyckoff_expander.py` - Symmetry expansion logic

## Considerations

- Maintain backward compatibility with existing code
- Use pymatgen for symmetry operations (already in requirements)
- Support for non-cubic lattice parameters may require additional API changes
- 2D structures remain unchanged (hardcoded templates preserved)

### To-dos

- [ ] Create wyckoff_templates.py with template data structure for Pm3m and Pnma space groups
- [ ] Create wyckoff_expander.py with function to expand Wyckoff positions using pymatgen symmetry operations
- [ ] Refactor _get_bulk_position_template() to use Wyckoff templates with space_group parameter (default 'Pm3m')
- [ ] Update lattice parameter calculation to support non-cubic systems (orthorhombic a, b, c)
- [ ] Add space_group parameter to create_perovskite() and create_bulk_perovskite() functions with backward compatibility
- [ ] Update q2D_creator class to pass through space_group parameter