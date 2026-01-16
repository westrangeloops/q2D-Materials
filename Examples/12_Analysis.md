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

The analyzer builds a NetworkX graph with three types of nodes and various edge types:

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

3. **Atom Nodes** (`node_type='atom'`)
   - Represent individual atoms
   - Node ID format: `atom_{index}`
   - Attributes:
     - `vasp_index`: Original atom index
     - `symbol`: Atomic symbol
     - `direct_coordinates`: [x, y, z] coordinates
     - `x_atom_type`: Type of X atom (if applicable)
     - `x_connected_octahedra`: Octahedra connected to this X atom
     - `is_spacer`: Whether atom belongs to a spacer molecule (set during pre-classification)
     - `is_a_site`: Whether atom belongs to an A-site cation (set during pre-classification)
     - `spacer_indices`: List of atom indices in the spacer molecule (if `is_spacer=True`)
     - `a_site_indices`: List of atom indices in the A-site molecule (if `is_a_site=True`)
     - `spacer_formula`, `a_site_formula`: Molecular formulas (if applicable)
     - `spacer_type`: Type of spacer ('dj', 'rp', or 'unknown') - set during molecular classification

### Edge Types

1. **`contains`**: Layer → Octahedron (layer contains octahedra)
2. **`interlayer_connection`**: Layer → Layer (adjacent layers connected via X atoms)
3. **`shares_atoms`**: Octahedron → Octahedron (octahedra sharing X-site atoms)
4. **`contains_atom`**: Octahedron → Atom (octahedron contains X atoms)
   - Edge attributes:
     - `geometry`: Geometry label (`axial_terminal`, `axial_interlayer`, or `equatorial`)
     - `terminal`: Boolean indicating if atom is terminal (not shared)
     - `clifford_distance`: 6D distance using Clifford embedding (PBC-aware)
5. **`has_center`**: Octahedron → Atom (octahedron's B-site center)
6. **`is_center_of`**: Atom → Octahedron (B-site atom is center of octahedron)
7. **`covalent_bond`**: Atom → Atom (covalent bonds between atoms)

## Accessing the Graph

Get the NetworkX graph directly:

```python
graph = analyzer.get_graph()

# Count nodes by type
layer_nodes = [n for n, d in graph.nodes(data=True) 
               if d.get('node_type') == 'layer']
octahedra_nodes = [n for n, d in graph.nodes(data=True) 
                   if d.get('node_type') == 'octahedron']
atom_nodes = [n for n, d in graph.nodes(data=True) 
              if d.get('node_type') == 'atom']

print(f"Graph has {len(layer_nodes)} layers, {len(octahedra_nodes)} octahedra, {len(atom_nodes)} atoms")
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
5. **Molecular Pre-Classification**: Pre-classifies molecules as A-sites or spacers based on Z-coordinates:
   - Molecules within slab zones (Z-center within layer's inorganic atoms) → `is_a_site=True`
   - Molecules between layers (Z-center outside slab zones) → `is_spacer=True`
   - Sets `a_site_indices` or `spacer_indices` accordingly
6. **Molecular Classification**: Refines classification using continuity analysis and sets `spacer_type` ('dj' or 'rp')
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
- **Node colors** based on type (layers=pink, octahedra=purple, atoms=elemental colors, molecules=orange)
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
6. **Export visualization** - Convert to interactive HTML for exploration

The graph structure provides a foundation for all other analyses (Glazer detection, RDF, B-X-B angles) which operate on this graph representation.
