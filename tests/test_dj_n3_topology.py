#!/usr/bin/env python3
"""
Test DJ n=3 structure to see if it has special pairs.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

print("Creating 3×3 DJ structure (n=3)")

q2d = q2D_creator()

structure = q2d.create_structure(
    structure_type="bulk",
    template="cubic",
    layer_sequence="DJ",
    thickness=3,  # n=3
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    spacer="[NH3+]CCC[NH3+]",
    glazer_angles=[0, 0, 0],
    glazer_pattern=["0", "0", "0"],
)

structure = structure * (3, 3, 1)

print(f"Formula: {structure.get_chemical_formula()}")
print(f"Atoms: {len(structure)}")

analyzer = q2D_analyzer(structure)
analyzer.analyze()

octahedra = analyzer.get_octahedra()
print(f"\nOctahedra detected: {len(octahedra)}")

graph = analyzer.get_graph()
neighbor_indices = graph.graph.get('neighbor_indices', [])

from q2D_Materials.analyzer.octahedral_processing.octahedral_detection import find_shared_atoms

shared_atoms = find_shared_atoms(neighbor_indices)

# Analyze sharing patterns
sharing_counts = {}
for pair, atoms in shared_atoms.items():
    count = len(atoms)
    if count not in sharing_counts:
        sharing_counts[count] = 0
    sharing_counts[count] += 1

print(f"\nSharing patterns:")
for count in sorted(sharing_counts.keys()):
    print(f"  {sharing_counts[count]} pairs share {count} atom(s)")

# Check for special pairs
special_pairs = {pair: atoms for pair, atoms in shared_atoms.items() if len(atoms) >= 2}
print(f"\nSpecial pairs (≥2 atoms): {len(special_pairs)}")

if len(special_pairs) > 0:
    print("✓ Found special pairs!")
    for pair, atoms in list(special_pairs.items())[:5]:
        print(f"  Octahedra {pair[0]}-{pair[1]}: share {len(atoms)} atoms")
else:
    print("✗ No special pairs found")
