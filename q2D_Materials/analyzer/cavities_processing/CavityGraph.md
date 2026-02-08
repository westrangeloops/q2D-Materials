# Cavity Subgraph Reference

Complete reference for the cavity subgraphs built by `q2D_Materials/analyzer/cavities_processing/cavity_tracing.py`.

## Design Philosophy

Cavity subgraphs are **isolated subgraphs** extracted from the parent structural graph. They represent individual cavities (A-site cuboctahedra or spacer antiprisms) with:

- **PBC-unwrapped coordinates**: All positions use `pbc_position` (not `x`, `y`, `z`)
- **Direct B-X connectivity**: B-X bonds are explicit edges (no octahedron nodes)
- **Multiple PBC images**: Atoms can appear multiple times with different `image_label` values
- **Cage metadata**: `cage` and `half_cage` nodes track geometric structure

---

## Graph Hierarchy

### A-Site Cavity (Cuboctahedron)

```
A_Site Node (a_site_*)
  ├── CONTAINS → Molecule Atoms (atom_*_img_*)
  └── CONTAINS ← Cage (cage_0)
                    └── CONTAINS (role='corner') → B Atoms (atom_*_img_*)
                                                      └── BONDED_TO (role='ligand') → X Atoms (atom_*_img_*)
```

### Spacer Cavity (Antiprism)

#### RP Spacer (1 NH3 group)

```
Anchor Node (anchor_0)
  ├── CONTAINS → NH3 Atoms (N + 3H)
  └── CONTAINS ← Half_Cage (half_cage_0)
                    └── CONTAINS (role='corner') → B Atoms (atom_*_img_*)
                                                      └── BONDED_TO (role='ligand') → X Atoms (atom_*_img_*)
```

#### DJ Spacer (2+ NH3 groups)

```
Anchor_0 (anchor_0)
  ├── CONTAINS → NH3_0 Atoms (N + 3H)
  └── CONTAINS ← Half_Cage_0 (half_cage_0)
                    └── CONTAINS (role='corner') → B Atoms (atom_*_img_*)
                                                      └── BONDED_TO (role='ligand') → X Atoms (atom_*_img_*)

Anchor_1 (anchor_1)
  ├── CONTAINS → NH3_1 Atoms (N + 3H)
  └── CONTAINS (role='half_cage') → Half_Cage_1 (half_cage_1)
                                        └── CONTAINS (role='corner') → B Atoms (atom_*_img_*)
                                                                          └── BONDED_TO (role='ligand') → X Atoms (atom_*_img_*)
```

---

## Node Types and Properties

### 1. A_Site Node (`a_site_*`)

**Purpose**: Represents the A-site molecule in a cuboctahedral cavity.

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'a_site'` |
| `formula` | str | Chemical formula (Hill notation) |
| `nh3_count` | int | Number of NH3 groups (0 for atomic A-sites) |

**Note**: Only present in A-site cavities. Inherited from parent graph.

**Example Node ID**: `a_site_0`

---

### 2. Anchor Node (`anchor_*`)

**Purpose**: Represents NH3 group centers in spacer cavities.

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'anchor'` |
| `anchor_index` | int | Index of this anchor (0 for RP, 0/1 for DJ) |

**Note**: Only present in spacer cavities. One anchor per NH3 group.

**Example Node ID**: `anchor_0`, `anchor_1`

---

### 3. Cage Node (`cage_*`)

**Purpose**: Represents complete cuboctahedral cages (A-site cavities).

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'cage'` |
| `cage_index` | int | Index of this cage (usually 0) |
| `layer_id` | str or list | Layer ID(s) this cage spans |

**Note**: Only present in A-site cavities. Contains 8 B atoms (corners) and 12 X atoms (edges).

**Example Node ID**: `cage_0`

---

### 4. Half_Cage Node (`half_cage_*`)

**Purpose**: Represents incomplete square antiprism cages (spacer cavities).

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'half_cage'` |
| `cage_index` | int | Index of this half-cage (0 for RP, 0/1 for DJ) |
| `layer_id` | str | Layer ID this half-cage belongs to |

**Note**: Only present in spacer cavities. Contains 4 B atoms (corners) and 8 X atoms (4 terminal + 4 equatorial).

**Example Node ID**: `half_cage_0`, `half_cage_1`

---

### 5. Atom Nodes (`atom_{idx}_img_{i}_{j}_{k}`)

**Purpose**: Represent atoms with PBC-unwrapped coordinates.

**Node ID Format**: `atom_{original_index}_img_{i}_{j}_{k}` where `(i, j, k)` is the PBC image label.

| Property | Type | Description |
|----------|------|-------------|
| `node_type` | str | Always `'atom'` |
| `original_index` | int | Original atom index in parent graph (vasp_index) |
| `image_label` | tuple | PBC image label `(i, j, k)` |
| `pbc_position` | np.ndarray | PBC-unwrapped position (3D) |
| `symbol` | str | Chemical element symbol |
| `is_B` | bool | *(B atoms only)* True if this is a B-site atom |
| `is_terminal` | bool | *(X atoms only)* True if terminal X atom |
| `is_equatorial` | bool | *(X atoms only)* True if equatorial X atom |
| `is_interlayer` | bool | *(X atoms only)* True if interlayer X atom |
| `hybridization` | str | *(Optional)* Hybridization for carbon atoms |

**Note**: 
- **DO NOT use `x`, `y`, `z`** - these are excluded from subgraph nodes
- **Always use `pbc_position`** for coordinates
- Atoms can appear multiple times with different `image_label` values

**Example Node ID**: `atom_5_img_0_0_0`, `atom_5_img_1_0_0`

---

## Edge Types and Properties

### 1. CONTAINS Edges

**Purpose**: Express hierarchical containment relationships.

#### A_Site → Atom

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |

#### Anchor → Atom

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |

**Note**: Anchors only contain NH3 atoms (N + 3H).

#### Cage/Half_Cage → B Atom

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |
| `role` | str | Always `'corner'` |

#### Cage → A_Site

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |
| `role` | str | Always `'a_site'` |

#### Half_Cage → Anchor

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |
| `role` | str | Always `'anchor'` |

#### Anchor → Half_Cage (DJ only)

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'contains'` |
| `role` | str | Always `'half_cage'` |

**Note**: Only `anchor_1` connects to `half_cage_1` in DJ spacers.

---

### 2. BONDED_TO Edges

**Connects**: `B atom` ↔ `X atom`

**Purpose**: Represent direct B-X bonds (no octahedron nodes in subgraph).

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'bonded_to'` |
| `role` | str | Always `'ligand'` |
| `geometry` | str | `'axial'`, `'equatorial'`, or `'unknown'` |
| `bond_type` | str | Bond type: '{B}-{X}' (e.g., 'Pb-Cl') |
| `distance` | float | Bond length in Ångströms (PBC-unwrapped) |

**Note**: 
- B-X edges are created by matching topological constraints (from parent graph) with geometric proximity (PBC-unwrapped positions)
- Each B atom connects to 3 X atoms in cuboctahedra, 2-4 X atoms in antiprisms
- All edge properties are inherited from parent graph

---

### 3. BONDED_TO Edges (Molecule Atoms)

**Connects**: `atom_i` ↔ `atom_j` (within organic molecules)

**Purpose**: Represent covalent bonds within A-site or spacer molecules.

| Property | Type | Description |
|----------|------|-------------|
| `edge_type` | str | Always `'bonded_to'` |
| `bond_type` | str | Bond type: '{element1}-{element2}' (alphabetical) |
| `distance` | float | Bond length in Ångströms |

**Note**: Inherited from parent graph. Only present between molecule atoms (not B/X atoms).

---

## Cavity Types

### A-Site Cavity (Cuboctahedron)

- **Structure**: 1 complete cage with 8 B atoms and 12 X atoms
- **Molecule**: A_Site node (atomic or molecular with NH3)
- **Anchor**: A-site center (centroid for atomic, NH3 center for molecular)
- **Cage**: `cage_0` (complete cuboctahedron)

**Example**: MA, FA, Cs, Rb in perovskite structures

---

### RP Spacer Cavity (Square Antiprism)

- **Structure**: 1 half-cage with 4 B atoms and 8 X atoms
- **Molecule**: Spacer with 1 NH3 group
- **Anchor**: `anchor_0` (NH3 center)
- **Half_Cage**: `half_cage_0` (incomplete antiprism)

**Example**: Ruddlesden-Popper (RP) phase spacers

---

### DJ Spacer Cavity (Two Square Antiprisms)

- **Structure**: 2 half-cages, each with 4 B atoms and 8 X atoms (8 B + 16 X total)
- **Molecule**: Spacer with 2+ NH3 groups
- **Anchors**: `anchor_0` (first NH3), `anchor_1` (second NH3)
- **Half_Cages**: `half_cage_0`, `half_cage_1`

**Example**: Dion-Jacobson (DJ) phase spacers

---

## Key Differences from Parent Graph

| Feature | Parent Graph | Cavity Subgraph |
|---------|--------------|-----------------|
| **Coordinates** | `x`, `y`, `z` (wrapped) | `pbc_position` (unwrapped) |
| **Node IDs** | `atom_{idx}` | `atom_{idx}_img_{i}_{j}_{k}` |
| **Octahedron nodes** | Present | Absent |
| **B-X connectivity** | Via octahedron nodes | Direct B-X edges |
| **PBC images** | Single instance | Multiple instances with `image_label` |
| **Cage metadata** | Absent | `cage`/`half_cage` nodes present |
| **Anchor nodes** | Absent | Present for spacers |

---

## Usage Examples

### Accessing Cavity Subgraph

```python
from q2D_Materials.analyzer.cavities_processing.cavity_tracing import detect_all_cavities

cavities = detect_all_cavities(graph, atom_positions, atom_symbols, cell)

for cavity in cavities:
    subgraph = cavity.subgraph
    
    # Access cavity type
    print(f"Cavity type: {cavity.cavity_type}")  # 'a_site', 'spacer_rp', 'spacer_dj'
    
    # Access center position
    center = cavity.center_position
    print(f"Cavity center: {center}")
```

### Querying A-Site Cavity

```python
# Find A_Site node
a_site_nodes = [n for n, d in subgraph.nodes(data=True) 
                if d.get('node_type') == 'a_site']

if a_site_nodes:
    a_site_node = a_site_nodes[0]
    formula = subgraph.nodes[a_site_node]['formula']
    print(f"A-site formula: {formula}")
    
    # Get all molecule atoms
    mol_atoms = [n for n in subgraph.neighbors(a_site_node)
                 if subgraph.get_edge_data(a_site_node, n).get('edge_type') == 'contains']
    
    # Get cage node
    cage_nodes = [n for n, d in subgraph.nodes(data=True)
                  if d.get('node_type') == 'cage']
    if cage_nodes:
        cage_node = cage_nodes[0]
        # Get B atoms (corners)
        b_atoms = [n for n in subgraph.neighbors(cage_node)
                   if subgraph.get_edge_data(cage_node, n).get('role') == 'corner']
        print(f"Cage has {len(b_atoms)} B atoms")
```

### Querying Spacer Cavity

```python
# Find anchor nodes
anchor_nodes = [n for n, d in subgraph.nodes(data=True)
                if d.get('node_type') == 'anchor']

for anchor_node in anchor_nodes:
    anchor_idx = subgraph.nodes[anchor_node]['anchor_index']
    print(f"Anchor {anchor_idx}")
    
    # Get NH3 atoms
    nh3_atoms = [n for n in subgraph.neighbors(anchor_node)
                 if subgraph.get_edge_data(anchor_node, n).get('edge_type') == 'contains']
    
    # Get half_cage connected to this anchor
    half_cages = [n for n in subgraph.neighbors(anchor_node)
                  if subgraph.nodes.get(n, {}).get('node_type') == 'half_cage']
    
    for half_cage in half_cages:
        # Get B atoms (corners)
        b_atoms = [n for n in subgraph.neighbors(half_cage)
                   if subgraph.get_edge_data(half_cage, n).get('role') == 'corner']
        print(f"Half-cage has {len(b_atoms)} B atoms")
```

### Accessing PBC-Unwrapped Coordinates

```python
# Get all atom positions (PBC-unwrapped)
for node, data in subgraph.nodes(data=True):
    if data.get('node_type') == 'atom':
        # Get PBC-unwrapped position
        pbc_pos = data.get('pbc_position')
        original_idx = data.get('original_index')
        image_label = data.get('image_label')
        
        print(f"Atom {original_idx} image {image_label}: {pbc_pos}")
        
        # DO NOT use x, y, z - they are not present in subgraph nodes
        # x = data.get('x')  # This will be None!
```

### Finding B-X Bonds

```python
# Find all B-X bonds in cavity
b_atoms = [n for n, d in subgraph.nodes(data=True)
           if d.get('is_B') == True]

for b_node in b_atoms:
    # Get X ligands
    x_ligands = [n for n in subgraph.neighbors(b_node)
                 if (subgraph.get_edge_data(b_node, n).get('edge_type') == 'bonded_to' and
                     subgraph.get_edge_data(b_node, n).get('role') == 'ligand')]
    
    for x_node in x_ligands:
        edge_data = subgraph.get_edge_data(b_node, x_node)
        geometry = edge_data.get('geometry')  # 'axial' or 'equatorial'
        distance = edge_data.get('distance')
        print(f"B-X bond: {b_node} → {x_node}, geometry={geometry}, distance={distance:.3f}")
```

---

## Related Files

- **Cavity Detection**: `q2D_Materials/analyzer/cavities_processing/cavity_tracing.py`
- **Cavity Class**: `q2D_Materials/analyzer/cavities_processing/cavity_class.py`
- **Parent Graph**: `q2D_Materials/analyzer/core/graph_construction.py`
- **PBC Reconstruction**: `q2D_Materials/utils/molecules/pbc_reconstruction.py`
- **PBC Distances**: `q2D_Materials/utils/geometry/pbc_distances.py`
