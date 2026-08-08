#!/usr/bin/env python3
"""
Check how many layers are actually in a DJ n=2 structure.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

print("Creating 2×2 DJ structure (n=2)")

q2d = q2D_creator()

structure = q2d.create_structure(
    structure_type="bulk",
    template="cubic",
    layer_sequence="DJ",
    thickness=2,
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    spacer="[NH3+]CCC[NH3+]",
    glazer_angles=[0, 0, 0],
    glazer_pattern=["0", "0", "0"],
)

structure = structure * (2, 2, 1)

print(f"Formula: {structure.get_chemical_formula()}")

analyzer = q2D_analyzer(structure)
analyzer.analyze()

graph = analyzer.get_graph()

# Check layer nodes in graph
layer_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'layer']
print(f"\nLayer nodes in graph: {len(layer_nodes)}")

for layer_node in layer_nodes:
    layer_data = graph.nodes[layer_node]
    print(f"  {layer_node}: z_coord={layer_data.get('z_coord')}, position={layer_data.get('position')}")

# Check octahedra per layer
for layer_node in layer_nodes:
    oct_count = sum(1 for neighbor in graph.neighbors(layer_node) 
                    if graph.nodes[neighbor].get('node_type') == 'octahedron')
    print(f"  {layer_node}: {oct_count} octahedra")
