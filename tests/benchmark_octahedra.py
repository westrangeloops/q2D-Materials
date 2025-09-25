#!/usr/bin/env python3
"""
Simple benchmark to show optimization improvements.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q2D_Materials.utils.geometry import _count_octahedra, find_shared_atoms
from ase.io import read

# Load test structures
atoms1 = read('tests/test_structures/MAPbCl3_n1_l1.vasp')
atoms2 = read('tests/test_structures/MAPbBr3_n2_l1.vasp') 
atoms3 = read('tests/test_structures/MAPbI3_n3_l1.vasp')

test_structures = [
    ("MAPbCl3_n1_l1", atoms1),
    ("MAPbBr3_n2_l1", atoms2), 
    ("MAPbI3_n3_l1", atoms3)
]

print("=== Octahedra Function Performance Benchmark ===")
print()

total_time = 0
total_octahedra = 0

for name, atoms in test_structures:
    print(f"Testing {name}...")
    
    # Time the optimized function
    start_time = time.time()
    
    count, centers, symbols, neighbor_indices = _count_octahedra(
        atoms.positions,
        atom_symbols=atoms.get_chemical_symbols(),
        cutoff_distance=4.0,
        cell=atoms.get_cell(),
        pbc=atoms.get_pbc()
    )
    
    # Also time the shared atoms analysis
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    total_time += elapsed
    total_octahedra += count
    
    print(f"  Found {count} octahedra in {elapsed:.4f} seconds")
    print(f"  Shared connections: {len(shared_atoms)} pairs")
    print()

print(f"=== Summary ===")
print(f"Total octahedra analyzed: {total_octahedra}")
print(f"Total time: {total_time:.4f} seconds")
print(f"Average time per structure: {total_time/len(test_structures):.4f} seconds")
print(f"Average time per octahedron: {total_time/total_octahedra:.6f} seconds")
print()
print("Optimizations implemented:")
print("✅ Pre-computed distances (no recalculation per tolerance)")
print("✅ Vectorized index operations")
print("✅ Optimized numpy intersection for shared atoms")
print("✅ Eliminated redundant array operations")
print("✅ Cached neighbor data structure")
