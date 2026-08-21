# Graph Structure Reference

Complete reference for the structural graph built by `q2D_Materials/analyzer/core/graph_construction.py`.

## Design Philosophy

- **Nodes store intrinsic properties only** (what the entity IS)
- **Edges express relationships** (how entities connect)
- **Derive computed properties via traversal** (terminal status, sharing patterns)
- **No index lists in nodes** - that's what edges are for

---

## Graph Hierarchy

```
Structure (root)
  ├── CONTAINS → Layer
  │                └── CONTAINS → Octahedron
  │                                 ├── CONTAINS (role='center') → Atom (B-site)
  │                                 └── CONTAINS (role='ligand') → Atom (X-site)
  ├── CONTAINS → A_Site
  │                └── CONTAINS → Atom ← BONDED_TO → Atom
  └── CONTAINS → Spacer
                   └── CONTAINS → Atom ← BONDED_TO → Atom
```

---

## Node Types and Properties

### 1. Structure Node (`structure_0`)

**Purpose**: Single root node containing global metadata for the entire structure.

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'structure'` |
| `formula` | str | Chemical formula (Hill notation) |
| `thickness` | int | Number of octahedral layers |
| `a`, `b`, `c` | float | Cell lengths in Ångströms |
| `alpha`, `beta`, `gamma` | float | Cell angles in degrees |
| `layer_count` | int | Number of layers |
| `octahedra_count` | int | Number of octahedra |
| `a_site_count` | int | Number of A-site molecules |
| `spacer_count` | int | Number of spacer molecules |
| `atom_count` | int | Total atoms |

**Note**: There is exactly one Structure node per graph. Access via `graph.graph['structure_node']`.

**Example Node ID**: `structure_0`

---

### 2. Layer Nodes (`layer_*`)

**Purpose**: Represent 2D slabs in the perovskite structure.

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'layer'` |
| `z_coord` | float | Z-coordinate of the layer |

**Example Node ID**: `layer_0`, `layer_1`

---

### 2. Octahedron Nodes (`octahedron_*`)

**Purpose**: Represent B-X6 coordination units in the perovskite framework.

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'octahedron'` |

**Note**: Octahedra have no stored index lists. Query atoms via CONTAINS edges.

**Example Node ID**: `octahedron_0`, `octahedron_1`

---

### 3. Atom Nodes (`atom_*`)

**Purpose**: Represent individual atoms with intrinsic properties.

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'atom'` |
| `vasp_index` | int | Index in VASP structure (0-based) |
| `symbol` | str | Chemical element symbol (e.g., 'Pb', 'Cl', 'C') |
| `x` | float | Direct coordinate X |
| `y` | float | Direct coordinate Y |
| `z` | float | Direct coordinate Z |
| `is_terminal` | bool | *(X-sites only)* True if atom belongs to exactly 1 octahedron (terminal/surface atom) |
| `is_equatorial` | bool | *(X-sites only)* True if atom is in equatorial position of any octahedron |
| `is_interlayer` | bool | *(X-sites only)* True if atom is shared between 2 octahedra (axial bridging atom) |
| `hybridization` | str | *(Optional)* Hybridization for carbon atoms ('sp', 'sp2', 'sp3') |

**Note**: Structural properties (`is_terminal`, `is_equatorial`, `is_interlayer`) are computed during graph construction and stored on X-site atoms (F, Cl, Br, I, S, Se, Te, O). These are fundamental properties used frequently in organohalide perovskite chemistry.

**Example Node ID**: `atom_0`, `atom_1`

---

### 4. A_Site Nodes (`a_site_*`)

**Purpose**: Represent A-site organic molecules (cations in cuboctahedral cavities).

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'a_site'` |
| `formula` | str | Chemical formula (Hill notation) |
| `nh3_count` | int | Number of NH3 groups (0 for atomic A-sites like Cs, Rb, K) |

**Example Node ID**: `a_site_0`, `a_site_1`

---

### 5. Spacer Nodes (`spacer_*`)

**Purpose**: Represent spacer organic molecules (interact with membrane surface).

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'spacer'` |
| `formula` | str | Chemical formula (Hill notation) |
| `nh3_count` | int | Number of NH3 groups (1 for RP, 2+ for DJ spacers) |

**Example Node ID**: `spacer_0`, `spacer_1`

---

## Edge Types and Properties

### 1. CONTAINS Edges

**Purpose**: Express hierarchical containment relationships.

#### Layer → Octahedron

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |

#### Octahedron → Atom

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |
| `role` | str | `'center'` for B-site atom, `'ligand'` for X atoms |
| `geometry` | str | *(Ligands only)* `'axial'`, `'equatorial'`, or `'unknown'` |

#### A_Site/Spacer → Atom

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |

#### Structure → Layer

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |

#### Structure → A_Site

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |

#### Structure → Spacer

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |

---

### 2. BONDED_TO Edges

**Connects**: `atom_i` ↔ `atom_j` (within molecules)

**Purpose**: Represent covalent bonds within organic molecules.

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'bonded_to'` |
| `bond_type` | str | Bond type: '{element1}-{element2}' (alphabetical, e.g., 'C-H', 'C-N') |
| `distance` | float | Bond length in Ångströms |

---

## Derived Information (Query via Traversal)

Instead of storing redundant index lists, use graph traversal:

### Structure Metadata

```python
from q2D_Materials.analyzer.core.graph_construction import get_structure_node
structure = get_structure_node(graph)
print(f"Formula: {structure['formula']}")
print(f"Cell: a={structure['a']:.3f}, b={structure['b']:.3f}, c={structure['c']:.3f}")
print(f"Layers: {structure['layer_count']}, Octahedra: {structure['octahedra_count']}")
```

Or via the analyzer API:

```python
metadata = analyzer.get_structure_metadata()
print(f"Structure type: {metadata.get('structure_type')}")
```

### Terminal X Atoms
Atoms belonging to exactly 1 octahedron:

```python
from q2D_Materials.analyzer.core.graph_construction import get_terminal_atoms
terminal_atoms = get_terminal_atoms(graph)
```

### Interlayer X Atoms
Atoms shared between octahedra in different layers:

```python
from q2D_Materials.analyzer.core.graph_construction import get_interlayer_atoms
interlayer_atoms = get_interlayer_atoms(graph)
```

### Center (B-site) Atom of Octahedron

```python
from q2D_Materials.analyzer.core.graph_construction import get_center_atom
center = get_center_atom(graph, 'octahedron_0')
```

### Ligand Atoms of Octahedron

```python
from q2D_Materials.analyzer.core.graph_construction import get_ligand_atoms
# All ligands
ligands = get_ligand_atoms(graph, 'octahedron_0')
# Only axial ligands
axial = get_ligand_atoms(graph, 'octahedron_0', geometry='axial')
```

### Atoms in an A_Site or Spacer

```python
from q2D_Materials.analyzer.core.graph_construction import get_molecule_atoms
atoms = get_molecule_atoms(graph, 'a_site_0')
atoms = get_molecule_atoms(graph, 'spacer_0')
```

### Octahedra Sharing an Atom

```python
from q2D_Materials.analyzer.core.graph_construction import get_sharing_octahedra
octahedra = get_sharing_octahedra(graph, 'atom_5')
```

---

## What Was Removed (and Why)

| Old Property/Edge | Why Removed | Query Instead |
|-------------------|-------------|---------------|
| `intralayer_x_atoms` in Layer | Index list in node | Traverse: Layer → Octahedra → Atoms |
| `terminal_atoms` in Octahedron | Index list in node | Count atoms with exactly 1 containing octahedron |
| `is_a_site`, `spacer_indices` in Atom | Molecule info scattered | Traverse: Molecule → Atom |
| `clifford_6d` | Single 6D array | Use separate `x`, `y`, `z` |
| `clifford_distance` on edges | Computable | Calculate from coordinates |
| `has_center` / `is_center_of` edges | Redundant | Use `role: 'center'` on CONTAINS edge |
| `shares_atoms` edges | Derived | Find atoms with >1 containing octahedra |
| `interlayer_connection` edges | Derived | Query through shared atoms across layers |
| `terminal` on edges | Computable | Count containing octahedra |

---

## Graph Hierarchy Visualization

```
Structure (root)
  ├── CONTAINS → Layer
  │                └── CONTAINS → Octahedron
  │                                 ├── CONTAINS (role='center') → Atom (B-site)
  │                                 └── CONTAINS (role='ligand', geometry='axial'|'equatorial') → Atom (X-site)
  ├── CONTAINS → A_Site
  │                └── CONTAINS → Atom ← BONDED_TO → Atom
  └── CONTAINS → Spacer
                   └── CONTAINS → Atom ← BONDED_TO → Atom
```

---

## Usage Example

```python
import networkx as nx
from q2D_Materials.analyzer.core.graph_construction import (
    get_structure_node,
    get_terminal_atoms,
    get_center_atom,
    get_ligand_atoms,
    get_molecule_atoms,
)

# Get structure metadata
structure = get_structure_node(graph)
print(f"Formula: {structure['formula']}")
print(f"Cell: a={structure['a']:.3f} Å, b={structure['b']:.3f} Å, c={structure['c']:.3f} Å")
print(f"Angles: α={structure['alpha']:.1f}°, β={structure['beta']:.1f}°, γ={structure['gamma']:.1f}°")

# Access node properties
for node, data in graph.nodes(data=True):
    node_type = data.get('node_type')
    
    if node_type == 'structure':
        formula = data.get('formula')
        thickness = data.get('thickness')
    
    elif node_type == 'atom':
        symbol = data.get('symbol')
        x, y, z = data.get('x'), data.get('y'), data.get('z')
        
    elif node_type == 'octahedron':
        center = get_center_atom(graph, node)
        axial_ligands = get_ligand_atoms(graph, node, geometry='axial')
        
    elif node_type == 'layer':
        z_coord = data.get('z_coord')
        
    elif node_type == 'a_site':
        formula = data.get('formula')
        nh3_count = data.get('nh3_count', 0)
        atoms = get_molecule_atoms(graph, node)
        
    elif node_type == 'spacer':
        formula = data.get('formula')
        nh3_count = data.get('nh3_count', 0)
        atoms = get_molecule_atoms(graph, node)

# Find terminal atoms
terminals = get_terminal_atoms(graph)

# Get all layers from structure
structure_id = graph.graph['structure_node']
for neighbor in graph.neighbors(structure_id):
    edge_data = graph.get_edge_data(structure_id, neighbor)
    if edge_data.get('edge_type') == 'contains':
        node_data = graph.nodes[neighbor]
        if node_data.get('node_type') == 'layer':
            print(f"Layer: {neighbor}, z={node_data.get('z_coord')}")

# Access edge properties
for u, v, data in graph.edges(data=True):
    edge_type = data.get('edge_type')
    
    if edge_type == 'contains':
        role = data.get('role')  # 'center' or 'ligand' (for oct→atom)
        geometry = data.get('geometry')  # 'axial' or 'equatorial'
        
    elif edge_type == 'bonded_to':
        bond_type = data.get('bond_type')
        distance = data.get('distance')
```

---

## Molecule Classification

Molecules are classified based on 12 nearest X atoms around their center:

| Classification | Criterion | Description |
|----------------|-----------|-------------|
| **A-site** | No terminal X atoms in nearest 12 | Forms closed cuboctahedra |
| **Spacer** | Has terminal X atoms in nearest 12 | Interacts with membrane surface |

---

## Related Files

- **Graph Construction**: `q2D_Materials/analyzer/core/graph_construction.py`
- **Octahedra Detection**: `q2D_Materials/analyzer/detection/octahedral_detection.py`
- **Layer Identification**: `q2D_Materials/analyzer/detection/layer_identification.py`
- **Cavity Detection**: `q2D_Materials/analyzer/cavities_processing/cavity_tracing.py`
