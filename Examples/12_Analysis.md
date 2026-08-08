![q2D-Materials Logo](../Logos/logo.png)

# 12. Graph Decomposition — from structure to graph

The `q2D_analyzer` decomposes a structure (VASP, CIF, or ASE Atoms) into a graph-based representation using connectivity analysis. This graph captures the hierarchical organization of the material: layers contain octahedra, octahedra contain atoms, and connections represent sharing and bonding relationships.

## Basic Usage

Load a structure and analyze it to build the graph:

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

# Create a structure
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    thickness=2, vacuum=15.0,
)

# Analyze it to build the graph
analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Or load from file
analyzer = q2D_analyzer("structure.vasp")
analyzer.analyze()
```

## The Graph Structure

The analyzer builds a NetworkX graph with multiple types of nodes and various edge types:

### Node Types

1. **Layer Nodes** (`node_type='layer'`)
   - Represent inorganic layers/slabs
   - Node ID format: `layer_{id}`
   - Attributes:
     - `position`: Layer position classification
     - `z_coord`: Z-coordinate of the layer
     - `intralayer_x_atoms`: X atoms within the layer
     - `interlayer_x_atoms_above`: X atoms connecting to layer above
     - `interlayer_x_atoms_below`: X atoms connecting to layer below

2. **Octahedron Nodes** (`node_type='octahedron'`)
   - Represent BX₆ octahedral units
   - Node ID format: `octahedron_{id}`
   - Attributes:
     - `central_atom`: Index of the B-site atom
     - `terminal_atoms`: X atoms at layer boundaries (includes `axial_terminal`)
     - `interlayer_atoms`: X atoms between layers (includes `axial_interlayer`)
     - `intralayer_atoms`: X atoms within the layer (includes `equatorial`)
   - **Note**: Each octahedron should have 2 axial atoms (top/bottom) and 4 equatorial atoms in ideal structures

3. **A_Site Nodes** (`node_type='a_site'`)
   - Represent A-site cations (atomic or molecular)
   - Node ID format: `a_site_{id}`
   - Attributes:
     - `formula`: Chemical formula of the A-site
     - `nh3_count`: Number of NH3 groups (for molecular A-sites, 0 for atomic)
   - Connected to atoms via `contains` edges

4. **Spacer Nodes** (`node_type='spacer'`)
   - Represent spacer molecules between layers
   - Node ID format: `spacer_{id}`
   - Attributes:
     - `formula`: Chemical formula of the spacer
     - `nh3_count`: Number of NH3 groups (1 for RP, 2+ for DJ)
   - Connected to atoms via `contains` edges

5. **Atom Nodes** (`node_type='atom'`)
   - Represent individual atoms
   - Node ID format: `atom_{index}`
   - Attributes:
     - `vasp_index`: Original atom index
     - `symbol`: Atomic symbol
     - `direct_coordinates`: [x, y, z] coordinates
     - `x_atom_type`: Type of X atom (if applicable)
     - `x_connected_octahedra`: Octahedra connected to this X atom
     - `is_X`: Whether atom is an X-site atom
     - `is_terminal`: Whether atom is terminal (not shared)
     - `is_equatorial`: Whether atom is equatorial (intralayer)
     - `is_interlayer`: Whether atom is interlayer (between layers)
   - **Note**: The `is_A` and `is_S` properties have been removed. Use A_Site and Spacer nodes instead.

### Edge Types

1. **`contains`**: 
   - Structure → Layer (structure contains layers)
   - Structure → A_Site (structure contains A-site nodes)
   - Structure → Spacer (structure contains spacer nodes)
   - Layer → Octahedron (layer contains octahedra)
   - Octahedron → Atom (octahedron contains atoms, with `role='center'` for B atoms, `role='ligand'` for X atoms)
   - A_Site → Atom (A-site contains atoms)
   - Spacer → Atom (spacer contains atoms)
2. **`interlayer_connection`**: Layer → Layer (adjacent layers connected via X atoms)
3. **`shares_atoms`**: Octahedron → Octahedron (octahedra sharing X-site atoms)
4. **`bonded_to`**: Atom → Atom (covalent bonds between atoms, with `role='ligand'` for B-X bonds)

## Accessing the Graph

Get the NetworkX graph directly:

```python
graph = analyzer.get_graph()

# Count nodes by type
layer_nodes = [n for n, d in graph.nodes(data=True) 
               if d.get('node_type') == 'layer']
octahedra_nodes = [n for n, d in graph.nodes(data=True) 
                   if d.get('node_type') == 'octahedron']
a_site_nodes = [n for n, d in graph.nodes(data=True) 
                if d.get('node_type') == 'a_site']
spacer_nodes = [n for n, d in graph.nodes(data=True) 
                if d.get('node_type') == 'spacer']
atom_nodes = [n for n, d in graph.nodes(data=True) 
              if d.get('node_type') == 'atom']

print(f"Graph has {len(layer_nodes)} layers, {len(octahedra_nodes)} octahedra, "
      f"{len(a_site_nodes)} A-sites, {len(spacer_nodes)} spacers, {len(atom_nodes)} atoms")
```

## Graph Decomposition Process

The `analyze()` method performs several steps:

1. **Octahedra Detection**: Identifies BX₆ units using bond distance analysis
2. **Atom Classification**: Classifies X atoms by geometry and connectivity:
   - **Geometry labels**: `axial` (along stacking direction) or `equatorial` (in-plane)
   - **Connectivity labels**: `terminal` (unshared), `interlayer` (shared between layers), or `intralayer` (shared within layer)
   - **Combined labels**:
     - `axial_terminal`: Axial atom at layer boundary (not shared)
     - `axial_interlayer`: Axial atom shared between layers (connects adjacent layers)
     - `equatorial`: Equatorial atom shared within layer (implicitly `equatorial/intralayer`)
     - `equatorial/terminal`: Not currently supported (reserved for future defect handling)
     - `axial/intralayer`: Does not exist (axial atoms are by definition interlayer)
3. **Layer Identification**: Groups octahedra into layers based on z-coordinates
4. **Graph Construction**: Builds the NetworkX graph with nodes and edges
5. **A_Site and Spacer Node Creation**: Creates A_Site and Spacer nodes:
   - Classifies molecules as A-sites or spacers based on proximity to terminal X atoms
   - Creates `a_site_{id}` nodes for A-site cations
   - Creates `spacer_{id}` nodes for spacer molecules
   - Connects atoms to their respective A_Site or Spacer nodes via `contains` edges
   - Handles isolated atoms in cavities by creating A_Site nodes for them
7. **Structure Type Inference**: Determines structure type from graph patterns

## Extracting Components from the Graph

### Octahedra

```python
octahedra = analyzer.get_octahedra()

print(f"Found {len(octahedra)} octahedra")
for oct in octahedra[:3]:  # Show first 3
    print(f"  Octahedron {oct['id']}:")
    print(f"    Central atom: {oct['central_atom_index']}")
    print(f"    Terminal atoms: {len(oct.get('terminal_atoms', []))}")
    print(f"    Interlayer atoms: {len(oct.get('interlayer_atoms', []))}")
    print(f"    Intralayer atoms: {len(oct.get('intralayer_atoms', []))}")
```

### Layers

```python
layers = analyzer.get_layers()

for layer_name, layer_data in layers.items():
    print(f"Layer {layer_name}:")
    print(f"  Octahedra: {len(layer_data.get('octahedra', []))}")
    print(f"  Z-range: {layer_data.get('z_min', 0):.2f} - {layer_data.get('z_max', 0):.2f} Å")
```

### Spacers and A-Sites

```python
# Get spacer molecules
spacers = analyzer.get_spacers()
print(f"Found {len(spacers)} spacer molecules")
for i, spacer in enumerate(spacers):
    print(f"  Spacer {i+1}: {spacer.get_chemical_formula()}")
    print(f"    Type: {spacer.info.get('spacer_type', 'unknown')}")

# Get A-site cations
a_sites = analyzer.get_a_sites()
print(f"Found {len(a_sites)} A-site cations")
for i, a_site in enumerate(a_sites):
    print(f"  A-site {i+1}: {a_site.get('formula', 'unknown')}")
    print(f"    Type: {a_site.get('type', 'unknown')}")
```

## Structure Type Detection

The analyzer infers structure type from graph patterns:

```python
print(f"Structure type: {analyzer.structure_type}")
# Output: 'bulk', 'dj', 'rp', or 'monolayer'
```

The structure type is determined by:
- Presence of spacer molecules
- Layer connectivity patterns
- Octahedra sharing relationships
- Z-continuity of slabs

## Atom Classification Labels

The analyzer uses combined geometry/connectivity labels for X atoms:

| Label | Geometry | Connectivity | Description |
|-------|----------|--------------|-------------|
| `axial_terminal` | Axial | Terminal | Axial atom at layer boundary, not shared |
| `axial_interlayer` | Axial | Interlayer | Axial atom shared between adjacent layers |
| `equatorial` | Equatorial | Intralayer | Equatorial atom shared within same layer |

**Important Notes:**
- `equatorial` implicitly means `equatorial/intralayer` (equatorial atoms are always intralayer)
- `axial/intralayer` does not exist (axial atoms are by definition interlayer)
- `equatorial/terminal` is not currently supported but reserved for future defect handling
- In ideal octahedra: 2 axial (1 terminal + 1 interlayer) + 4 equatorial
- In 1×1 unit cells: Special handling for self-shared equatorial atoms through PBC

## Atom Classification Algorithm

The classification uses **topology-based detection** (not geometry), relying on atom sharing patterns between octahedra. The algorithm determines atom types by analyzing how many octahedra share each atom and whether those octahedra belong to the same or different layers.

### Primary Classification Table

Based on the number of octahedra sharing each atom:

| Number of Octahedra | Condition | Classification | Geometry Label |
|---------------------|-----------|----------------|----------------|
| **1 octahedron** | - | `'terminal'` | `'axial_terminal'` |
| **2 octahedra** | Same layer | `'equatorial'` | `'equatorial'` |
| **2 octahedra** | Different layers | `'axial_interlayer'` | `'axial_interlayer'` |
| **>2 octahedra** | - | `'multi_shared'` | `'unknown'` |

### Layer Detection Table

Layers are identified by analyzing sharing patterns between octahedra pairs:

| Sharing Count | Interpretation | Layer Relationship | Atom Type |
|---------------|----------------|-------------------|-----------|
| **≥ max_sharing** (typically 4) | High sharing | Same layer | Equatorial (intralayer) |
| **< max_sharing** (typically 1-2) | Low sharing | Different layers | Axial (interlayer) |
| **1 atom** (1×1 cells) | Single shared | Different layers | Axial (interlayer) |
| **≥2 atoms** (1×1 cells) | Self-shared via PBC | Same layer | Equatorial |

### Classification Decision Tree

```
For each atom:
│
├─ Number of octahedra = 1?
│  └─ YES → 'terminal' → 'axial_terminal'
│
└─ Number of octahedra = 2?
   │
   ├─ Check layer membership of both octahedra
   │  │
   │  ├─ Same layer?
   │  │  └─ YES → 'equatorial' → 'equatorial'
   │  │
   │  └─ Different layers?
   │     └─ YES → 'axial_interlayer' → 'axial_interlayer'
   │
   └─ Number of octahedra > 2?
      └─ YES → 'multi_shared' → 'unknown'
```

### Layer Formation Rules

| Connection Type | Sharing Pattern | Forms Layer? | Connects Layers? |
|----------------|-----------------|--------------|------------------|
| **Equatorial** | High count (≥4 atoms) | ✅ YES | ❌ NO |
| **Axial** | Low count (1-2 atoms) | ❌ NO | ✅ YES |

**Key Insight:** Layers are formed by **equatorial connections** (high sharing count, same layer), while **axial atoms** connect different layers (low sharing count, interlayer).

### Example Classifications

| Atom | Octahedra | Layer IDs | Sharing Count | Classification | Geometry |
|------|-----------|-----------|---------------|----------------|----------|
| Atom_5 | [Oct_0] | - | 1 octahedron | `'terminal'` | `'axial_terminal'` |
| Atom_12 | [Oct_0, Oct_1] | [0, 0] | 2, same layer | `'equatorial'` | `'equatorial'` |
| Atom_23 | [Oct_0, Oct_2] | [0, 1] | 2, different layers | `'axial_interlayer'` | `'axial_interlayer'` |
| Atom_34 | [Oct_0, Oct_1, Oct_2] | [0, 0, 1] | 3 octahedra | `'multi_shared'` | `'unknown'` |

### Visual Connectivity Diagram

```
Layer 0 (Top Layer)
┌─────────────────────────────────┐
│  Oct_0  ────  Oct_1  ────  Oct_2  │  ← Equatorial connections (high sharing)
│    │         │         │          │     (4+ atoms shared, same layer)
│    │         │         │          │
│    X         X         X          │  ← Equatorial atoms (intralayer)
│    │         │         │          │
└────┼─────────┼─────────┼──────────┘
     │         │         │
     │         │         │
     X         X         X          ← Axial atoms (interlayer)
     │         │         │          (1-2 atoms shared, different layers)
     │         │         │
┌────┼─────────┼─────────┼──────────┐
│  Oct_3  ────  Oct_4  ────  Oct_5  │  ← Equatorial connections (high sharing)
│    │         │         │          │     (4+ atoms shared, same layer)
│    │         │         │          │
│    X         X         X          │  ← Equatorial atoms (intralayer)
│    │         │         │          │
└─────────────────────────────────┘
Layer 1 (Bottom Layer)

Legend:
  Oct_N  = Octahedron node
  X      = X-site atom
  ────   = Equatorial connection (high sharing, same layer)
  │      = Axial connection (low sharing, different layers)
```

### Algorithm Steps

1. **Build atom → octahedra mapping**: For each atom, record which octahedra it belongs to
2. **Find sharing between pairs**: Identify pairs of octahedra that share atoms
3. **Distinguish same-layer vs inter-layer pairs**: Use sharing count heuristic:
   - High sharing count (≥4) → same layer (equatorial connection)
   - Low sharing count (1-2) → different layers (axial connection)
4. **Assign layer membership**: Use union-find algorithm to group octahedra into layers
5. **Classify atoms**: Based on layer membership of the octahedra they connect

### Special Case: 1×1 Unit Cells

In 1×1 unit cells, equatorial atoms can appear to be shared with themselves through periodic boundary conditions. The algorithm handles this by:

| Sharing Pattern | Count | Classification | Reason |
|----------------|-------|---------------|--------|
| Single atom shared | 1 | `'axial_interlayer'` | True inter-layer bridge |
| Multiple atoms shared | ≥2 | `'equatorial'` | Self-shared via PBC (same layer) |

This ensures that only true inter-layer connections (single shared atom) are classified as axial, while high-count sharing is recognized as equatorial self-sharing through PBC.

### Querying Atoms by Classification

```python
graph = analyzer.get_graph()

# Find all axial terminal atoms
axial_terminal = [n for n, d in graph.nodes(data=True) 
                  if d.get('node_type') == 'atom' and 
                  any(e.get('geometry') == 'axial_terminal' 
                      for _, _, e in graph.edges(n, data=True))]

# Find all equatorial atoms
equatorial = [n for n, d in graph.nodes(data=True) 
              if d.get('node_type') == 'atom' and 
              any(e.get('geometry') == 'equatorial' 
                  for _, _, e in graph.edges(n, data=True))]

# Find interlayer connections
interlayer = [n for n, d in graph.nodes(data=True) 
              if d.get('node_type') == 'atom' and 
              any(e.get('geometry') == 'axial_interlayer' 
                  for _, _, e in graph.edges(n, data=True))]
```

## Graph Traversal Examples

### Find Octahedra Connected by Shared Atoms

```python
graph = analyzer.get_graph()

# Find octahedra that share atoms
octahedra_nodes = [n for n, d in graph.nodes(data=True) 
                   if d.get('node_type') == 'octahedron']

for node in octahedra_nodes[:3]:
    neighbors = [n for n in graph.neighbors(node) 
                 if graph.nodes[n].get('node_type') == 'octahedron']
    edges = graph.edges(node, data=True)
    shared_edges = [(u, v, d) for u, v, d in edges 
                    if d.get('edge_type') == 'shares_atoms']
    print(f"Octahedron {node} has {len(neighbors)} octahedra neighbors")
    print(f"  Shares atoms with: {[v for u, v, d in shared_edges]}")
```

### Find Atoms in a Specific Octahedron

```python
oct_node = 'octahedron_0'
atom_nodes = [n for n in graph.neighbors(oct_node) 
              if graph.nodes[n].get('node_type') == 'atom']

print(f"Octahedron {oct_node} contains {len(atom_nodes)} atoms")
for atom_node in atom_nodes:
    atom_data = graph.nodes[atom_node]
    print(f"  {atom_node}: {atom_data['symbol']} at {atom_data['direct_coordinates']}")
```

### Find Layers and Their Octahedra

```python
layer_nodes = [n for n, d in graph.nodes(data=True) 
               if d.get('node_type') == 'layer']

for layer_node in layer_nodes:
    octahedra = [n for n in graph.neighbors(layer_node) 
                 if graph.nodes[n].get('node_type') == 'octahedron']
    print(f"{layer_node} contains {len(octahedra)} octahedra")
```

### Find A_Site and Spacer Nodes

```python
# Find all A_Site nodes
a_site_nodes = [n for n, d in graph.nodes(data=True) 
                if d.get('node_type') == 'a_site']

for a_site_node in a_site_nodes:
    # Get atoms in this A-site
    atoms = [n for n in graph.neighbors(a_site_node) 
             if graph.nodes[n].get('node_type') == 'atom']
    formula = graph.nodes[a_site_node].get('formula', 'unknown')
    print(f"{a_site_node}: {formula} with {len(atoms)} atoms")

# Find all Spacer nodes
spacer_nodes = [n for n, d in graph.nodes(data=True) 
                if d.get('node_type') == 'spacer']

for spacer_node in spacer_nodes:
    # Get atoms in this spacer
    atoms = [n for n in graph.neighbors(spacer_node) 
             if graph.nodes[n].get('node_type') == 'atom']
    formula = graph.nodes[spacer_node].get('formula', 'unknown')
    nh3_count = graph.nodes[spacer_node].get('nh3_count', 0)
    print(f"{spacer_node}: {formula} with {len(atoms)} atoms, {nh3_count} NH3 groups")
```

## Structure-Level Features

The graph can be used to calculate structure-level geometric features that characterize the entire material structure, not just individual components.

### Interplane Distance

Calculate the distance between terminal atom planes using the structure graph. Terminal atoms define the boundaries between the spacer molecule region and the inorganic slab region.

```python
# After analyzing the structure
analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Calculate interplane distance
result = analyzer.inter_plane_distance()

print(f"Interplane distance: {result['interplane_distance']:.3f} Å")
print(f"Top terminal atoms: {result['top_atom_count']}")
print(f"Bottom terminal atoms: {result['bottom_atom_count']}")
print(f"Molecule mean Z: {result['molecule_mean_z']:.3f} Å")
```

**Algorithm:**
1. Gets all terminal atoms from the structure graph (atoms belonging to exactly one octahedron)
2. Samples up to 10 atoms per spacer molecule to determine molecule mean Z position
3. Determines slab configuration:
   - If molecule mean Z is between terminal Z range: splits terminals by median Z into top/bottom
   - If molecule mean Z is outside terminal range: uses closest pair of terminals
   - If no molecules: falls back to Z-clustering
4. Fits planes via SVD with edge case handling:
   - **X ≥ 4 atoms**: SVD plane fitting
   - **X = 3 atoms**: SVD plane fitting (3 points define a plane)
   - **X = 2 atoms**: Midpoint with vertical normal vector
5. Calculates perpendicular distance between planes

**Return Value:**
```python
{
    'interplane_distance': float,      # Distance between planes (Å) or NaN if invalid
    'top_centroid': list or None,      # [x, y, z] centroid of top plane
    'bottom_centroid': list or None,   # [x, y, z] centroid of bottom plane
    'top_atom_count': int,            # Number of atoms in top plane
    'bottom_atom_count': int,         # Number of atoms in bottom plane
    'molecule_mean_z': float or None,  # Mean Z position of sampled molecule atoms
}
```

**Example Usage:**
```python
from q2D_Materials.analyzer import q2D_analyzer
import numpy as np

# Load and analyze a structure
analyzer = q2D_analyzer("structure.vasp")
analyzer.analyze()

# Calculate interplane distance
result = analyzer.inter_plane_distance()

if not np.isnan(result['interplane_distance']):
    print(f"Distance between terminal planes: {result['interplane_distance']:.3f} Å")
    print(f"Top plane centroid: {result['top_centroid']}")
    print(f"Bottom plane centroid: {result['bottom_centroid']}")
else:
    print("Could not calculate interplane distance (insufficient terminal atoms)")
```

**Physical Interpretation:**
- **Interplane distance** represents the spacing between the top and bottom boundaries of the inorganic slab
- This is a **cell-level feature** that characterizes the overall structure geometry
- Useful for comparing layer spacing across different structures and compositions
- Terminal atoms are the X-site atoms that form the interface between the spacer region and the slab region

### Octahedral Volumes

Calculate octahedral volumes using convex hull of X atoms with PBC-aware positioning. This method ensures that all X atoms are in the closest periodic image relative to the B atom, preventing errors from PBC wrapping that could distort the volume calculation.

```python
# After analyzing the structure
analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Global mode: mean volume for all octahedra
mean_volume = analyzer.get_octahedral_volumes(mode='global')
print(f"Mean octahedral volume: {mean_volume:.2f} Å³")

# Local mode: volume per octahedron
volumes_dict = analyzer.get_octahedral_volumes(mode='local')
for oct_id, vol in volumes_dict.items():
    print(f"{oct_id}: {vol:.2f} Å³")
```

**Algorithm:**
1. For each B atom (octahedron center), gets the 6 X atoms bonded to it from the graph
2. Uses PBC-aware distance calculations (`calculate_pbc_distances` with `return_vectors=True`) to ensure each X atom is in the closest periodic image relative to the B atom
3. Calculates the convex hull volume of the 6 X atoms using `scipy.spatial.ConvexHull`
4. Returns either mean volume (global mode) or per-octahedron volumes (local mode)

**Parameters:**
- `mode`: str, default='global'
  - `'global'`: Return mean volume for all octahedra (single float)
  - `'local'`: Return volume per octahedron (dict mapping octahedron_id -> volume)
- `octahedra`: list of str, optional
  - Specific octahedra IDs to include (e.g., ['octahedron_0', 'octahedron_1'])
  - If None, computes all octahedra

**Return Value:**
- If `mode='global'`: Mean octahedral volume in Å³ (float)
- If `mode='local'`: Dict mapping `octahedron_id -> volume` in Å³

**Example Usage:**
```python
from q2D_Materials.analyzer import q2D_analyzer
import numpy as np

# Load and analyze a structure
analyzer = q2D_analyzer("structure.vasp")
analyzer.analyze()

# Get mean octahedral volume
mean_vol = analyzer.get_octahedral_volumes(mode='global')
print(f"Mean octahedral volume: {mean_vol:.2f} Å³")

# Get per-octahedron volumes
volumes = analyzer.get_octahedral_volumes(mode='local')
print(f"Found {len(volumes)} octahedra")
for oct_id, vol in list(volumes.items())[:5]:  # Show first 5
    print(f"  {oct_id}: {vol:.2f} Å³")

# Calculate statistics
vol_array = np.array(list(volumes.values()))
print(f"Volume statistics:")
print(f"  Mean: {np.nanmean(vol_array):.2f} Å³")
print(f"  Std: {np.nanstd(vol_array):.2f} Å³")
print(f"  Min: {np.nanmin(vol_array):.2f} Å³")
print(f"  Max: {np.nanmax(vol_array):.2f} Å³")
```

**Physical Interpretation:**
- **Octahedral volume** represents the volume enclosed by the 6 X atoms forming the octahedral cage
- Calculated using convex hull, which gives the volume of the smallest convex polyhedron containing all 6 X atoms
- PBC-aware calculations ensure accurate volumes even when X atoms are in different periodic images
- Useful for characterizing octahedral distortion and comparing volumes across different structures
- Volume changes can indicate structural phase transitions or chemical pressure effects

**Important Notes:**
- The method uses PBC-aware distance calculations to ensure all X atoms are in the minimum image relative to the B atom
- This prevents errors from periodic boundary wrapping that could artificially inflate or distort volumes
- Returns NaN for octahedra with invalid connectivity (e.g., fewer than 6 X atoms or degenerate geometry)
- The convex hull method is more robust than pyramid decomposition for distorted octahedra

## Analysis Parameters

Control the graph construction process:

```python
analyzer.analyze(
    cutoff_distance=4.0,      # Maximum distance for octahedral neighbors (Å)
    min_tolerance=0.2,        # Starting bond length tolerance (Å)
    step=0.1,                # Tolerance increment step (Å)
    max_steps=20,            # Maximum tolerance optimization steps
    non_metal_symbols=['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']  # Excluded from octahedral centers
)
```

## Complete Example

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

# Create and analyze a structure
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    thickness=2, vacuum=15.0,
)

analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Access graph structure
graph = analyzer.get_graph()

# Extract components
print(f"Structure type: {analyzer.structure_type}")
print(f"Octahedra: {len(analyzer.get_octahedra())}")
print(f"Layers: {len(analyzer.get_layers())}")
print(f"Spacers: {len(analyzer.get_spacers())}")
print(f"A-sites: {len(analyzer.get_a_sites())}")

# Graph statistics
print(f"\nGraph statistics:")
print(f"  Total nodes: {graph.number_of_nodes()}")
print(f"  Total edges: {graph.number_of_edges()}")
print(f"  Layer nodes: {len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'layer'])}")
print(f"  Octahedron nodes: {len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'octahedron'])}")
print(f"  A_Site nodes: {len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'a_site'])}")
print(f"  Spacer nodes: {len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'spacer'])}")
print(f"  Atom nodes: {len([n for n, d in graph.nodes(data=True) if d.get('node_type') == 'atom'])}")
```

## Exporting the Graph to HTML

You can export the structural graph to an interactive HTML visualization using pyvis:

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.utils.other.graph_pyvis_export import export_structure_pyvis

# After analyzing the structure
analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Get the graph and export it
graph = analyzer.get_graph()
export_structure_pyvis(graph, 'my_structure_graph.html')
```

The exported HTML file features:
- **Interactive visualization** with drag-and-drop node positioning
- **Physics simulation** for automatic node layout
- **Node colors** based on type (layers=pink, octahedra=purple, atoms=elemental colors, a_sites=orange, spacers=green)
- **Node sizes** scaled by type importance
- **Hover tooltips** showing node information (type, symbol, index)
- **Edge labels** showing relationship types (contains, shares_atoms, etc.)

### Export Parameters

```python
export_structure_pyvis(
    graph,           # The NetworkX graph from analyzer.get_graph()
    'output.html',   # Output file path (extension added automatically)
    exclude_cavities=True  # If True (default), cavity nodes are excluded
)
```

The function returns the absolute path to the created HTML file.

### Example with Full Analysis and Export

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.utils.other.graph_pyvis_export import export_structure_pyvis

# Create and analyze a structure
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    thickness=2, vacuum=15.0,
)

analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Export the graph to HTML
graph = analyzer.get_graph()
html_path = export_structure_pyvis(graph, 'my_structure.html')
print(f"Graph exported to: {html_path}")

# Open in browser
import webbrowser
webbrowser.open(f'file://{html_path}')
```

## Graph-Based Analysis Workflow

1. **Load structure** - From file or Atoms object
2. **Call `analyze()`** - Builds the graph representation
3. **Access components** - Use getter methods or direct graph access
4. **Traverse graph** - Use NetworkX functions for custom analysis
5. **Extract information** - Query nodes and edges for specific properties
6. **Calculate structure-level features** - Use characterization modules (e.g., interplane distance)
7. **Export visualization** - Convert to interactive HTML for exploration

The graph structure provides a foundation for all other analyses (Glazer detection, RDF, B-X-B angles, interplane distance) which operate on this graph representation.
