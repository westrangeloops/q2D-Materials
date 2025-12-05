<!-- de7463a4-bbf0-4624-b88f-19c21b09880c 038ffc7b-3dd9-4604-a1e1-84dd0fabaa9e -->
# Refactoring Plan: Layered Architecture for Perovskite Builder

## Overview

Refactor `perovskite_builder.py` (1578 lines) into a clean, layered architecture that separates geometry templates, structure building, and atom population. This enables better testability, extensibility, and maintainability while preserving backward compatibility.

## Directory Structure

Create new modules following the proposed architecture:

```
q2D_Materials/
├── templates/
│   └── templates.py          # Universal geometry templates
├── builders/
│   ├── q_builder.py          # Template → abstract position matrices
│   └── population.py        # Matrix → ASE Atoms (ion assignment)
├── pipeline/
│   └── perovskite.py        # Orchestration layer
└── utils/
    └── perovskite_builder.py # Deprecated (Eliminated after refactor)
```

## Phase 1: Extract Templates Module

### 1.1 Create `templates/templates.py`

**Purpose**: Pure geometry templates defining fractional positions for A, B, X sites.

**Extract from `perovskite_builder.py`**:

- `_get_bulk_position_template()` → `CubicTemplate.get_position_matrix()`
- `_get_2d_layer_position_template()` → `ReducedTemplate.get_position_matrix()`

**Key Classes**:

```python
class Template:
    name: str
    dim: str  # '3D' or '2D'
    def get_position_matrix(self, BX_dist: float, n_layers: int = 1) -> dict:
        # Returns {'A': np.ndarray, 'B': np.ndarray, 'X': np.ndarray}
        # All positions in fractional coordinates

class CubicTemplate(Template):
    # For bulk perovskites
    # Returns positions from _get_bulk_position_template()

class ReducedTemplate(Template):
    # For 2D layers (RP/DJ/monolayer base)
    # Returns positions from _get_2d_layer_position_template()
```

**Dependencies**: numpy only (pure geometry)

## Phase 2: Extract Q-Builder Module

### 2.1 Create `builders/q_builder.py`

**Purpose**: Convert templates to full supercell position matrices with proper lattice vectors.

**Extract from `perovskite_builder.py`**:

- `_validate_and_convert_BX_dist()` → `calculate_lattice_vectors()`
- `_translate_positions_to_supercell()` → `expand_to_supercell()`
- Supercell expansion logic from `_create_unified_core()` (lines 579-650)

**Key Functions**:

```python
def calculate_lattice_vectors(BX_dist: float, structure_type: str) -> np.ndarray:
    # Converts BX_dist to lattice vector sizes
    # Handles 2D transformation (sqrt calculation for 2D layers)

def build_structure_matrix(
    template: Template,
    BX_dist: float,
    supercell: tuple[int, int, int],
    n_layers: int,
    structure_type: str
) -> QBuilderOutput:
    # Returns QBuilderOutput with:
    #   - cell_vectors: np.ndarray[(3,3)]
    #   - positions: dict[str, np.ndarray]  # {'A': [...], 'B': [...], 'X': [...]}
    #   - Ap_positions: list (for spacers, empty initially)
```

**Special Handling**:

- 2D structures: only expand in x/y, not z (n_layers handled separately)
- A-site position filtering for 2D (keep [0.75, 0.25], filter [0.25, 0.75] for later spacer attachment)

**Dependencies**: numpy, templates

## Phase 3: Extract Population Module

### 3.1 Create `builders/population.py`

**Purpose**: Populate abstract position matrices with actual atoms, handling molecular alignment and placement.

**Extract from `perovskite_builder.py`**:

- `_assign_ions_to_positions()` → `assign_ions_to_sites()`
- Ion assignment logic from `_create_unified_core()` (lines 669-794)
- Molecular A-site handling (alignment, CoM correction, placement)

**Key Functions**:

```python
def assign_ions_to_sites(
    positions: dict,
    A_ions, B_ions, X_ions,
    structure_type: str
) -> list[tuple]:
    # Returns list of (site_type, ion, position) tuples
    # Handles pattern-based assignment (lists cycle through positions)

def populate_structure(
    matrix: QBuilderOutput,
    A_ions, B_ions, X_ions,
    spacer_ions=None,
    structure_type='bulk',
    lattice_vectors=None,
    metadata: dict = None
) -> Atoms:
    # Converts site matrices to ASE Atoms
    # Handles:
    #   - Atomic vs molecular A-sites
    #   - Molecular alignment (align_ase_molecule_for_perovskite)
    #   - CoM correction for bulk
    #   - Placement (add_adsorbate for bulk, place_atoms_at_location for 2D)
```

**Dependencies**: ase, numpy, molecule_builder, common_a_sites

## Phase 4: Extract Spacer Attachment Logic

### 4.1 Move spacer functions to `builders/population.py` or keep in separate module

**Functions to extract**:

- `_attach_spacer()` (lines 1517-1575)
- `_setup_cell_for_spacers()` (lines 1431-1440)
- `_calculate_z_levels()` (lines 1443-1458)
- `_generate_attachment_positions()` (lines 1461-1478)
- `_adjust_atomic_spacer_z()` (lines 1481-1496)
- `_process_spacer_attachment()` (lines 1499-1514)
- `_get_effective_spacer_size()` (lines 68-103)

**Decision**: Keep in `population.py` as `attach_spacers()` function since it's part of the population phase.

## Phase 5: Create Orchestration Layer

### 5.1 Create `pipeline/perovskite.py`

**Purpose**: Thin orchestration layer that coordinates templates → q_builder → population.

**Key Functions**:

```python
def create_perovskite(
    structure_type: str,
    A, B, X,
    spacer=None,
    BX_dist=None,
    supercell=(1,1,1),
    n_layers=1,
    **kwargs
) -> Atoms:
    """
    Unified pipeline:
 1. Get template based on structure_type
 2. Build structure matrix (q_builder)
 3. Populate with atoms (population)
 4. Attach spacers if 2D (population)
 5. Apply structure-specific transformations (RP rotation, monolayer vacuum, etc.)
    """
```

**Structure-Specific Logic**:

- **Bulk**: Simple pipeline (template → matrix → populate)
- **DJ**: Add spacer attachment after population
- **RP**: Add spacer attachment + layer duplication + 90° rotation
- **Monolayer**: Add spacer attachment + vacuum + centering

**Extract from `perovskite_builder.py`**:

- `create_bulk_perovskite()` → simplified version using pipeline
- `create_2d_perovskite()` → simplified version using pipeline
- RP-specific logic (lines 1059-1279)
- DJ-specific logic (lines 1286-1360)
- Monolayer-specific logic (lines 1373-1423)

**Dependencies**: templates, q_builder, population, molecule_builder

## Phase 6: Maintain Backward Compatibility

### 6.1 Update `utils/perovskite_builder.py`

**Strategy**: Keep as thin wrapper that imports from new modules.

```python
# Deprecated: Use pipeline.perovskite instead
from ..pipeline.perovskite import create_perovskite, create_bulk_perovskite, create_2d_perovskite
from ..builders.population import auto_calculate_BX_distance  # or move to utils

# Re-export all public functions for backward compatibility
__all__ = ['create_perovskite', 'create_bulk_perovskite', 'create_2d_perovskite', ...]
```

### 6.2 Verify `core/creator.py` compatibility

- No changes needed - it already imports from `perovskite_builder`
- All function signatures remain identical

## Implementation Order

1. **Phase 1**: Create `templates/templates.py` (pure geometry, no dependencies)
2. **Phase 2**: Create `builders/q_builder.py` (depends on templates)
3. **Phase 3**: Create `builders/population.py` (depends on q_builder)
4. **Phase 4**: Integrate spacer logic into population.py
5. **Phase 5**: Create `pipeline/perovskite.py` (orchestrates everything)
6. **Phase 6**: Update `core/creator.py` to use new pipeline, delete `utils/perovskite_builder.py`

## Testing Strategy

- **Unit tests**: Test each module independently
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Templates: verify fractional positions are correct
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Q-builder: verify supercell expansion and lattice vectors
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Population: verify ion assignment and molecular placement
- **Integration tests**: Test full pipeline end-to-end
- **Regression tests**: Compare outputs from old vs new code for identical inputs

## Key Design Decisions

1. **Template API**: Returns fractional coordinates, not Angstroms (scaling happens in q_builder). Returns ALL site types (A, B, X, Ap) so same template can produce different structure types.

2. **Template Universality**: 

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - CubicTemplate can produce bulk (direct) or 2D (with spacer attachment)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - ReducedTemplate can produce RP, DJ, or monolayer (same positions, different spacer attachment logic)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                - Structure type is determined by population/attachment logic, not template choice

3. **QBuilderOutput**: Intermediate representation (dataclass) between q_builder and population. Preserves Ap positions separately from A positions.

4. **Spacer attachment**: Part of population phase, not separate module. Uses Ap positions from template.

5. **No backward compatibility**: Clean break - delete old `perovskite_builder.py`, update `creator.py` to use new pipeline directly.

6. **Special cases**: All preserved (atomic spacers, RP rotation, monolayer vacuum, etc.) - handled in population/orchestration layer.

## Files to Create

- `q2D_Materials/templates/__init__.py`
- `q2D_Materials/templates/templates.py`
- `q2D_Materials/builders/__init__.py`
- `q2D_Materials/builders/q_builder.py`
- `q2D_Materials/builders/population.py`
- `q2D_Materials/pipeline/__init__.py`
- `q2D_Materials/pipeline/perovskite.py`

## Files to Modify

- `q2D_Materials/core/creator.py` (update imports to use new pipeline)
- `q2D_Materials/__init__.py` (add new module exports if needed)

## Files to Delete

- `q2D_Materials/utils/perovskite_builder.py` (replaced by new architecture)

## Migration Notes

- All existing code using `perovskite_builder` will continue to work
- New code should import from `pipeline.perovskite` directly
- Gradual migration path: old code works, new code uses new API

### To-dos

- [ ] Create templates/templates.py with CubicTemplate and ReducedTemplate classes, extracting _get_bulk_position_template and _get_2d_layer_position_template logic
- [ ] Create builders/q_builder.py with build_structure_matrix function, extracting lattice vector calculation and supercell expansion logic from _create_unified_core
- [ ] Create builders/population.py with populate_structure and assign_ions_to_sites functions, extracting ion assignment and molecular placement logic
- [ ] Integrate spacer attachment functions (_attach_spacer, _calculate_z_levels, etc.) into population.py as attach_spacers function
- [ ] Create pipeline/perovskite.py orchestration layer that coordinates templates → q_builder → population, extracting structure-specific logic from create_bulk_perovskite and create_2d_perovskite
- [ ] Update utils/perovskite_builder.py to import and re-export from new modules, maintaining backward compatibility