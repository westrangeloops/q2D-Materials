<!-- be418b94-0b21-4a47-9bef-2dc3be75230a da4db8c8-38bb-4967-9236-c6d4750f42cf -->
# Simplify Builder Pipeline and Add Passivation

## Overview

Refactor `perovskite_builder.py` to have a clear, linear pipeline for each structure type, and add passivation functionality to convert DJ/RP structures to isolated monolayers with passivators at both ends and vacuum.

## Part 1: Simplify perovskite_builder.py Pipeline

### Current Issues
- Complex nested logic mixing monolayer creation, spacer attachment, and stacking
- Unclear separation between structure creation steps
- Duplicate code for similar operations

### New Clear Pipeline

**Bulk:**
1. Read lists (A, B, X)
2. Populate positions using `_create_unified_core()`
3. Return structure

**Monolayer:**
1. Read lists (A, B, X)
2. Populate monolayer using `_create_unified_core()` (structure_type='2d_layer')
3. Read spacer list
4. Populate spacers at positions using `_attach_spacer()`
5. Add vacuum and center
6. Return structure

**DJ:**
1. Read lists (A, B, X)
2. Create bottom monolayer slab (n_bottom layers)
3. Populate bottom monolayer with A/B/X
4. Read spacer list
5. Populate spacers on bottom slab (both attachments)
6. Create upper monolayer slab (n_upper layers)
7. Populate upper monolayer with A/B/X
8. Populate spacers on upper slab (top attachment only, or halogen if n_upper=0)
9. Stack slabs by translating Z
10. Return structure

**RP:**
1. Read lists (A, B, X)
2. Create bottom monolayer slab (n_bottom layers)
3. Populate bottom monolayer with A/B/X
4. Read spacer list
5. Populate spacers on bottom slab (both attachments)
6. Create upper monolayer slab (n_upper layers)
7. Populate upper monolayer with A/B/X
8. Populate spacers on upper slab (both attachments)
9. Create copy of upper slab
10. Rotate copy 90° around Z
11. Adjust Z to create periodic structure (no vacuum, tight packing)
12. Return structure

### Implementation Changes

**File: `q2D_Materials/utils/perovskite_builder.py`**

1. **Refactor `create_2d_perovskite()`** (lines 1220-1369)
   - Simplify to clear routing logic
   - Remove nested conditionals
   - Call helper functions in sequence

2. **Keep `_create_monolayer_slab()`** (lines 800-959)
   - Already follows the pipeline: create layer → attach spacers
   - May need minor cleanup

3. **Keep `_create_dj_from_monolayers()`** (lines 962-1081)
   - Already follows the pipeline
   - Verify it matches the described steps

4. **Keep `_create_rp_from_monolayers()`** (lines 1084-1217)
   - Already follows the pipeline
   - Verify rotation and stacking logic

5. **Remove duplicate helper functions** (lines 1517-1661)
   - There are duplicate definitions of `_setup_cell_for_spacers`, `_calculate_z_levels`, etc.
   - Keep only one set (lines 1372-1516)

6. **Fix indentation errors** (line 921, 953-956)
   - Fix incorrect indentation in `_create_monolayer_slab()`

## Part 2: Add Passivation Functionality

### New Function: `create_passivated_monolayer()`

**Location:** New file `q2D_Materials/utils/passivation.py` or add to `interface_creator.py`

**Purpose:** Convert DJ/RP structures to isolated monolayers with passivators at both ends and vacuum

**Parameters:**
- `perovskite`: q2DStructure (DJ or RP)
- `passivators`: list of Atoms/str (passivator molecules, pattern-based)
- `vacuum`: float (vacuum to add, default: 12.0)
- `which_slab`: str, optional ('bottom', 'upper', or 'both' - default: 'both' for full structure)

**DJ Passivation Process:**
1. Extract bottom and upper slabs from DJ structure
2. Identify spacer positions at exposed ends:
   - Top: spacers attached to upper slab (top attachment)
   - Bottom: spacers attached to bottom slab (bottom attachment)
3. Replace spacers with passivators:
   - Top passivators: attach at top of upper slab
   - Bottom passivators: attach at bottom of bottom slab
4. Combine: passivator (top) → upper slab → spacer (interlayer) → bottom slab → passivator (bottom)
5. Add vacuum around structure
6. Return as monolayer q2DStructure

**RP Passivation Process:**
1. Extract bottom and upper slabs from RP structure
2. Identify spacer positions:
   - Top: spacers attached to rotated upper slab (top attachment, rotated)
   - Bottom: spacers attached to bottom slab (bottom attachment)
3. Replace spacers with passivators:
   - Top passivators: attach at top of upper slab, then rotate 90° with slab
   - Bottom passivators: attach at bottom of bottom slab (no rotation)
4. Combine: passivator (top, rotated) → upper slab (rotated) → molecules (rotated) → bottom slab → passivator (bottom)
5. Add vacuum around structure
6. Return as monolayer q2DStructure

**Special Rules:**
- **RP**: If `passivators` list is empty, use spacer list from original structure
- **DJ**: Passivators must be different from spacers (validation: check for double-ended NH3 groups)
- **Pattern-based**: Passivators use same list cycling as spacers (nx × ny positions per end)

### Implementation Details

**File: `q2D_Materials/utils/passivation.py` (new file)**

1. **Function: `create_passivated_monolayer()`**
   - Extract slabs from DJ/RP structure
   - Identify spacer attachment positions
   - Replace spacers with passivators using `_attach_spacer()` logic
   - For RP: rotate top passivators with upper slab
   - Combine slabs
   - Add vacuum
   - Return q2DStructure with structure_type='monolayer'

2. **Helper: `_extract_slabs_from_dj()`**
   - Parse DJ structure to identify bottom and upper slabs
   - Return separated slabs with their z-ranges

3. **Helper: `_extract_slabs_from_rp()`**
   - Parse RP structure to identify bottom and rotated upper slabs
   - Return separated slabs

4. **Helper: `_replace_spacers_with_passivators()`**
   - Find spacer positions at specified ends
   - Remove spacers
   - Attach passivators at same positions
   - Use pattern-based assignment (cycle through passivators list)

5. **Helper: `_validate_dj_passivators()`**
   - Check that passivators don't have double-ended NH3 groups
   - Compare with original spacers to ensure they're different

**File: `q2D_Materials/core/structure.py`**

1. **Add method: `to_passivated_monolayer()`** (after `to_interface_ready()`)
   - Wrapper that calls `create_passivated_monolayer()`
   - Parameters: `passivators`, `vacuum=12.0`
   - Validates structure_type is DJ or RP
   - Returns new q2DStructure

## Files to Modify

1. `q2D_Materials/utils/perovskite_builder.py`
   - Fix indentation errors
   - Remove duplicate functions
   - Clean up pipeline logic

2. `q2D_Materials/utils/passivation.py` (new file)
   - Implement passivation functionality

3. `q2D_Materials/core/structure.py`
   - Add `to_passivated_monolayer()` method

## Key Technical Details

### DJ Passivation Structure
```
vacuum
-------
passivator (top) - attaches to upper slab top
-------
upper slab (n_upper layers)
-------
spacer (interlayer) - original spacer between slabs
-------
bottom slab (n_bottom layers)
-------
passivator (bottom) - attaches to bottom slab bottom
-------
vacuum
```

### RP Passivation Structure
```
vacuum
-------
passivator (top, rotated 90°) - attaches to upper slab top, then rotated
-------
upper slab (n_upper layers, rotated 90°)
-------
molecules (rotated 90°) - original spacers, rotated
-------
bottom slab (n_bottom layers)
-------
passivator (bottom) - attaches to bottom slab bottom
-------
vacuum
```

### Passivator Validation for DJ
- Check molecular structure for NH3 groups at both ends
- Ensure passivators are different from original spacers
- If validation fails, raise ValueError with explanation

### To-dos

- [ ] Fix indentation errors in _create_monolayer_slab() (lines 921, 953-956)
- [ ] Remove duplicate helper functions (_setup_cell_for_spacers, _calculate_z_levels, etc. at lines 1517-1661)
- [ ] Simplify create_2d_perovskite() to clear routing logic without nested conditionals
- [ ] Create new file q2D_Materials/utils/passivation.py with passivation functions
- [ ] Implement DJ passivation: extract slabs, replace spacers with passivators at both ends, add vacuum
- [ ] Implement RP passivation: extract slabs, replace spacers with passivators (top rotated), add vacuum
- [ ] Add validation for DJ passivators (no double-ended NH3, different from spacers)
- [ ] Add to_passivated_monolayer() method to q2DStructure class in structure.py