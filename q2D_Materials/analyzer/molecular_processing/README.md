# Molecular Processing Module

This module provides unified molecular graph construction and backbone identification for q2D-Materials.

## Overview

The molecular processing module creates standardized molecular graphs that work identically for:
- Standalone molecules (SMILES, XYZ, Atoms objects)
- Structure-extracted molecules (from crystal structures)

This enables seamless integration between molecular validation tools and structural analysis.

## Key Components

### 1. `molecule_graph.py`

Core functions for molecular graph construction and backbone identification:

- **`create_molecule_graph(molecule, analyze_backbone=True)`**
  - Creates unified molecular graph from SMILES, XYZ, Atoms, or existing graph
  - Returns graph with molecule node and atom nodes
  - Automatically identifies backbone if requested

- **`identify_backbone(graph, molecule_node)`**
  - Identifies backbone atoms through NH3 group analysis
  - Finds longest path between NH3 groups (avoiding H atoms)
  - Marks atoms as `role='backbone'` or `role='functional_group'`
  - Stores `nh3_count` on molecule node

### 2. `molecule_validation.py`

Pattern-based validation for DJ/RP spacers (from `characterization/molecule_candidates.py`):

- **`analyze_molecule_candidate()`** - Validate molecules as DJ or RP spacers
- **`analyze_dj_candidate()`** - DJ-specific validation
- **`analyze_rp_candidate()`** - RP-specific validation

## Graph Structure

### Molecule Node
```python
{
    'node_type': 'molecule',
    'molecule_type': 'organic',  # or 'spacer', 'a_site'
    'formula': 'C4H14N2',
    'nh3_count': 2  # Automatically detected
}
```

### Atom Nodes
```python
{
    'node_type': 'atom',
    'symbol': 'C',
    'role': 'backbone',  # or 'functional_group'
    'position': [x, y, z],  # if available
    ...
}
```

### Edges
- **`contains`**: molecule_0 → atom nodes
- **`bonded_to`**: atom → atom (molecular bonds)

## Usage Examples

### Create Molecular Graph

```python
from q2D_Materials.analyzer.molecular_processing import create_molecule_graph

# From SMILES
graph = create_molecule_graph("C(CC[NH3+])C[NH3+]")

# From Atoms object
from ase.io import read
atoms = read("molecule.xyz")
graph = create_molecule_graph(atoms)

# Access properties
mol_data = graph.nodes['molecule_0']
print(f"NH3 count: {mol_data['nh3_count']}")
print(f"Formula: {mol_data['formula']}")

# Check atom roles
for node in graph.nodes():
    if node != 'molecule_0':
        role = graph.nodes[node].get('role')
        if role:
            print(f"Atom {node}: {role}")
```

### Backbone Identification

```python
from q2D_Materials.analyzer.molecular_processing import identify_backbone

# Create graph without backbone analysis
graph = create_molecule_graph("C(CC[NH3+])C[NH3+]", analyze_backbone=False)

# Manually identify backbone
nh3_count = identify_backbone(graph, 'molecule_0')
print(f"Found {nh3_count} NH3 groups")
```

## Integration with Structure Analysis

When molecules are extracted from crystal structures via `graph_construction.py`, the backbone identification is automatically performed:

1. `_create_molecule_nodes()` creates molecule nodes
2. `_add_molecule_node()` calls `identify_backbone()` for spacer molecules
3. Molecule node gets `nh3_count` attribute
4. Atoms get `role` attribute

Downstream modules (like `cavity_tracing.py`) read `nh3_count` from the graph instead of recalculating.

## Algorithm Details

### Backbone Path Finding

1. **Find NH3 groups**: Use SMARTS pattern matching (`[NH3+]C`, `[NH2]C`)
2. **Count NH3 groups**: Store as `nh3_count` on molecule node
3. **Find longest path**:
   - If 2+ NH3: Find longest path between any pair through CHON (skip H)
   - If 1 NH3: Find longest path from NH3 through CHON
4. **Mark atoms**:
   - Path atoms → `role='backbone'`
   - Other CHON atoms → `role='functional_group'`
   - H atoms → no role (not structural)

### Why Avoid H Atoms?

Hydrogen atoms are dead ends in the molecular graph (terminal atoms with only one bond). Including them in path finding would create artificially short paths. By excluding H atoms, we find the true carbon-based backbone of the molecule.

## Design Philosophy

- **Unified format**: Same graph structure regardless of input source
- **Graph-based**: All analysis uses graph topology, not geometry
- **Separation of concerns**: 
  - `molecule_graph.py`: Graph construction and backbone identification
  - `molecule_validation.py`: Pattern-based DJ/RP validation
  - Downstream modules: Use `nh3_count` for their own purposes

## Migration Notes

### Old Approach (v2.2 and earlier)
```python
# NH3 counting scattered across modules
nh3_centers = _calculate_nh3_center(molecule_atoms, positions, symbols)
nh3_count = len(nh3_centers)
if nh3_count >= 2:
    spacer_type = 'dj'
```

### New Approach (v2.3+)
```python
# Read from graph (already computed during graph construction)
nh3_count = graph.nodes['molecule_0'].get('nh3_count', 0)
if nh3_count >= 2:
    spacer_type = 'dj'
```

## See Also

- [`Examples/20_MoleculeValidator.md`](../../../Examples/20_MoleculeValidator.md) - User documentation
- [`analyzer/core/graph_construction.py`](../core/graph_construction.py) - Structural graph construction
- [`analyzer/detection/cavity_tracing.py`](../detection/cavity_tracing.py) - Cavity detection using nh3_count

