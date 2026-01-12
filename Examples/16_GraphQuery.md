![q2D-Materials Logo](../Logos/logo.png)

# 16. Graph Queries — custom analysis with NetworkX

The analyzer's graph structure is built on NetworkX, providing full access to graph algorithms and custom queries. You can traverse the graph, filter nodes, find paths, and perform complex structural analysis.

## Accessing the Graph

Get the NetworkX graph directly:

```python
from q2D_Materials.analyzer import q2D_analyzer

analyzer = q2D_analyzer("structure.vasp")
analyzer.analyze()

graph = analyzer.get_graph()
```

## Basic Graph Queries

### Count Node Types

```python
# Count nodes by type
layer_nodes = [n for n, d in graph.nodes(data=True) 
               if d.get('node_type') == 'layer']
octahedra_nodes = [n for n, d in graph.nodes(data=True) 
                   if d.get('node_type') == 'octahedron']
atom_nodes = [n for n, d in graph.nodes(data=True) 
              if d.get('node_type') == 'atom']

print(f"Layers: {len(layer_nodes)}")
print(f"Octahedra: {len(octahedra_nodes)}")
print(f"Atoms: {len(atom_nodes)}")
```

### Find Neighbors

```python
# Find octahedra connected by shared atoms
octahedra_nodes = [n for n, d in graph.nodes(data=True) 
                   if d.get('node_type') == 'octahedron']

for node in octahedra_nodes[:3]:
    neighbors = [n for n in graph.neighbors(node) 
                 if graph.nodes[n].get('node_type') == 'octahedron']
    print(f"Octahedron {node} has {len(neighbors)} octahedra neighbors")
    
    # Get shared atoms
    edges = graph.edges(node, data=True)
    for u, v, d in edges:
        if d.get('edge_type') == 'shares_atoms':
            shared = d.get('shared_atoms', [])
            print(f"  Shares {len(shared)} atoms with {v}")
```

### Filter Nodes by Attributes

```python
# Find all spacer atoms
spacer_atoms = [n for n, d in graph.nodes(data=True) 
                if d.get('node_type') == 'atom' and d.get('is_spacer')]

print(f"Found {len(spacer_atoms)} spacer atoms")

# Find all A-site atoms
a_site_atoms = [n for n, d in graph.nodes(data=True) 
                if d.get('node_type') == 'atom' and d.get('is_a_site')]

print(f"Found {len(a_site_atoms)} A-site atoms")
```

## Graph Traversal

### Find Paths Between Octahedra

```python
import networkx as nx

# Find shortest path between two octahedra
octahedra_nodes = [n for n, d in graph.nodes(data=True) 
                   if d.get('node_type') == 'octahedron']

if len(octahedra_nodes) >= 2:
    path = nx.shortest_path(graph, octahedra_nodes[0], octahedra_nodes[1])
    print(f"Path from {octahedra_nodes[0]} to {octahedra_nodes[1]}:")
    print(f"  Length: {len(path) - 1} edges")
    print(f"  Path: {' -> '.join(path)}")
```

### Find Connected Components

```python
# Find connected components of octahedra
octahedra_subgraph = graph.subgraph([
    n for n, d in graph.nodes(data=True) 
    if d.get('node_type') == 'octahedron'
])

components = list(nx.connected_components(octahedra_subgraph))
print(f"Found {len(components)} connected octahedra components")
for i, comp in enumerate(components):
    print(f"  Component {i+1}: {len(comp)} octahedra")
```

### Breadth-First Search

```python
# BFS starting from an octahedron
octahedra_nodes = [n for n, d in graph.nodes(data=True) 
                   if d.get('node_type') == 'octahedron']

if octahedra_nodes:
    start = octahedra_nodes[0]
    bfs_tree = nx.bfs_tree(graph, start)
    print(f"BFS tree from {start}: {len(bfs_tree.nodes())} nodes")
```

## Edge-Based Queries

### Find All Shared Atom Connections

```python
# Find all octahedra pairs that share atoms
shared_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                if d.get('edge_type') == 'shares_atoms']

print(f"Found {len(shared_edges)} shared-atom connections")
for u, v, d in shared_edges[:5]:
    shared = d.get('shared_atoms', [])
    print(f"  {u} <-> {v}: {len(shared)} shared atoms")
```

### Find Interlayer Connections

```python
# Find connections between layers
interlayer_edges = [(u, v, d) for u, v, d in graph.edges(data=True)
                    if d.get('edge_type') == 'interlayer_connection']

print(f"Found {len(interlayer_edges)} interlayer connections")
for u, v, d in interlayer_edges:
    via_atoms = d.get('via_x_atoms', [])
    print(f"  {u} <-> {v}: via {len(via_atoms)} X atoms")
```

## Characterization Query Builder

Use the query builder for method chaining and complex queries:

```python
# Standard analyses with method chaining
result = (analyzer.get_characterization()
          .glazer(tolerance=0.1)
          .execute())

# Graph queries
octahedra_list = (analyzer.get_characterization()
                  .octahedra()
                  .neighbors(edge_type='shares_atoms')
                  .to_list())

# Filter and chain
spacer_atoms = (analyzer.get_characterization()
                .atoms()
                .filter(is_spacer=True)
                .to_list())
```

### Query Builder Methods

```python
char = analyzer.get_characterization()

# Start from octahedra
octahedra = char.octahedra()

# Get neighbors
neighbors = octahedra.neighbors(edge_type='shares_atoms')

# Filter by attributes
filtered = neighbors.filter(central_atom=5)  # Example: specific central atom

# Convert to list
result = filtered.to_list()

# Or get subgraph
subgraph = filtered.to_subgraph()
```

## Custom Graph Analysis

### Calculate Graph Metrics

```python
import networkx as nx

# Graph density
density = nx.density(graph)
print(f"Graph density: {density:.4f}")

# Average degree
degrees = dict(graph.degree())
avg_degree = sum(degrees.values()) / len(degrees)
print(f"Average degree: {avg_degree:.2f}")

# Clustering coefficient (for octahedra subgraph)
octahedra_subgraph = graph.subgraph([
    n for n, d in graph.nodes(data=True) 
    if d.get('node_type') == 'octahedron'
])
clustering = nx.average_clustering(octahedra_subgraph)
print(f"Octahedra clustering: {clustering:.4f}")
```

### Find Central Nodes

```python
# Find most connected octahedra
octahedra_nodes = [n for n, d in graph.nodes(data=True) 
                   if d.get('node_type') == 'octahedron']

degrees = dict(graph.degree(octahedra_nodes))
most_connected = max(degrees.items(), key=lambda x: x[1])
print(f"Most connected octahedron: {most_connected[0]} ({most_connected[1]} connections)")
```

## Advanced Queries

### Find Octahedra in Specific Layer

```python
# Get layer information
layers = analyzer.get_layers()

# Find octahedra in first layer
layer_name = list(layers.keys())[0]
layer_octahedra = layers[layer_name].get('octahedra', [])

print(f"Layer {layer_name} contains {len(layer_octahedra)} octahedra")

# Access via graph
layer_node = f'layer_{layer_name}'
octahedra_in_layer = [n for n in graph.neighbors(layer_node)
                      if graph.nodes[n].get('node_type') == 'octahedron']
print(f"Graph shows {len(octahedra_in_layer)} octahedra in {layer_node}")
```

### Find Atoms in Specific Octahedron

```python
# Get octahedron node
oct_node = 'octahedron_0'

# Find all atoms in this octahedron
atom_nodes = [n for n in graph.neighbors(oct_node) 
              if graph.nodes[n].get('node_type') == 'atom']

print(f"Octahedron {oct_node} contains {len(atom_nodes)} atoms:")
for atom_node in atom_nodes:
    atom_data = graph.nodes[atom_node]
    edge_data = graph.edges[oct_node, atom_node]
    edge_type = edge_data.get('edge_type', 'unknown')
    print(f"  {atom_node}: {atom_data['symbol']} ({edge_type})")
```

### Find Covalent Bonds

```python
# Find all covalent bonds
covalent_bonds = [(u, v, d) for u, v, d in graph.edges(data=True)
                  if d.get('edge_type') == 'covalent_bond']

print(f"Found {len(covalent_bonds)} covalent bonds")
for u, v, d in covalent_bonds[:5]:
    u_symbol = graph.nodes[u].get('symbol', '?')
    v_symbol = graph.nodes[v].get('symbol', '?')
    print(f"  {u} ({u_symbol}) - {v} ({v_symbol})")
```

## Complete Example

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator
import networkx as nx

# Create and analyze structure
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    thickness=2, vacuum=15.0,
)

analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Access graph
graph = analyzer.get_graph()

# Graph statistics
print("Graph Statistics:")
print(f"  Total nodes: {graph.number_of_nodes()}")
print(f"  Total edges: {graph.number_of_edges()}")

# Node type counts
node_types = {}
for n, d in graph.nodes(data=True):
    node_type = d.get('node_type', 'unknown')
    node_types[node_type] = node_types.get(node_type, 0) + 1

for node_type, count in node_types.items():
    print(f"  {node_type}: {count}")

# Edge type counts
edge_types = {}
for u, v, d in graph.edges(data=True):
    edge_type = d.get('edge_type', 'unknown')
    edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

for edge_type, count in edge_types.items():
    print(f"  {edge_type}: {count}")

# Find most connected octahedron
octahedra_nodes = [n for n, d in graph.nodes(data=True) 
                   if d.get('node_type') == 'octahedron']
if octahedra_nodes:
    degrees = dict(graph.degree(octahedra_nodes))
    most_connected = max(degrees.items(), key=lambda x: x[1])
    print(f"\nMost connected octahedron: {most_connected[0]} ({most_connected[1]} connections)")
```

## Molecular Graph Queries

The graph query system also works with molecular graphs from SMILES strings, enabling pattern-based molecule analysis without 3D coordinate generation.

### Pattern Matching on Molecular Graphs

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.utils.molecules.smiles_parser import smiles_to_graph

# Create molecular graph directly from SMILES (no 3D coordinates)
molecule_graph = smiles_to_graph('NCCCCN')

# Pattern matching uses graph isomorphism (fast, geometry-independent)
analyzer = q2D_analyzer()
result = analyzer.analyze_molecule_as_dj_spacer(
    'NCCCCN',
    initial_pattern='NH2C',
    final_pattern='NH2C'
)

print(f"Valid DJ spacer: {result.is_valid}")
print(f"Valid paths: {len(result.valid_paths)}")
```

### Graph-Based Pattern Matching

Molecule candidate analysis uses the same graph query principles:

```python
from q2D_Materials.utils.molecules.smiles_parser import smiles_to_graph
from q2D_Materials.modifier.fragment import from_smiles
import networkx as nx

# Parse SMILES to graph (graph-based, no 3D coordinates)
pattern_graph = from_smiles('[NH3+]C')
molecule_graph = from_smiles('C[NH3+]CCCC[NH3+]C')

# Use NetworkX subgraph isomorphism for pattern matching
from networkx.algorithms import isomorphism

matcher = isomorphism.GraphMatcher(
    molecule_graph,
    pattern_graph,
    node_match=lambda n1, n2: n1.get('symbol') == n2.get('symbol')
)

matches = list(matcher.subgraph_isomorphisms_iter())
print(f"Found {len(matches)} pattern matches")
```

### Shared Graph Utilities

Both structural and molecular graphs use shared utilities:

```python
from q2D_Materials.analyzer.utils.graph_utils import (
    find_shortest_path,
    validate_path_continuity,
    get_connected_components,
    filter_nodes_by_attributes
)

# Works on any NetworkX graph (structural or molecular)
path = find_shortest_path(graph, source_node, target_node)
is_valid = validate_path_continuity(graph, path)
components = get_connected_components(graph)
carbon_nodes = filter_nodes_by_attributes(graph, symbol='C')
```

## Tips for Graph Queries

1. **Use node attributes** - Filter by `node_type`, `is_spacer`, `is_a_site`, `symbol`, etc.
2. **Use edge attributes** - Filter by `edge_type` to find specific connections
3. **Leverage NetworkX** - Use built-in algorithms (shortest_path, connected_components, etc.)
4. **Use shared utilities** - Import from `graph_utils` for common operations
5. **Build subgraphs** - Create subgraphs for focused analysis
6. **Graph-based matching** - For molecules, use graph isomorphism instead of 3D coordinates
7. **Cache results** - Store frequently accessed node/edge lists

The graph structure provides a powerful foundation for custom structural analysis beyond the built-in analysis methods. Both structural graphs (from crystal structures) and molecular graphs (from SMILES) use the same NetworkX-based query system for consistency and performance.

