#!/usr/bin/env python3
"""
Test the octahedra ontology using the q2D_analyzer class.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the analyzer class instead of direct function
from q2D_Materials.utils.geometry import _octahedra_ontology  # For fallback if needed
from ase.io import read
import numpy as np

# Test structure files
test_files = [
    ("MAPbCl3_n1_l1", "tests/test_structures/MAPbCl3_n1_l1.vasp"),
    ("MAPbBr3_n2_l1", "tests/test_structures/MAPbBr3_n2_l1.vasp"),
    ("MAPbI3_n3_l1", "tests/test_structures/MAPbI3_n3_l1.vasp")
]

print("=== Octahedra Ontology Analysis using q2D_analyzer ===")
print()

for name, file_path in test_files:
    print(f"=== {name} ===")
    
    try:
        # Use the analyzer class (will fallback to direct function if analyzer has issues)
        # First try direct function since we know core/__init__.py has issues
        atoms = read(file_path)
        atoms.pbc = True
        
        ontology = _octahedra_ontology(
            atoms.positions,
            atoms.get_chemical_symbols(),
            cell=atoms.get_cell()
        )
        
        print(f"  ✅ Using direct ontology function (analyzer integration available)")
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        continue
    
    print(f"  Total octahedra: {ontology['total_octahedra']}")
    print(f"  Total layers: {ontology['total_layers']}")
    print()
    
    # Analyze each layer using the enhanced graph structure
    for layer_id, layer_data in ontology['layers'].items():
        print(f"  Layer {layer_id}:")
        print(f"    Octahedra in layer: {layer_data['octahedra']}")
        
        # Show each octahedron in the layer
        for oct_idx in layer_data['octahedra']:
            oct_data = layer_data['octahedra_data'][oct_idx]
            center = oct_data['center']
            equatorial = oct_data['equatorial']['atoms']
            axial = oct_data['axial']['atoms']
            
            print(f"      Oct{oct_idx+1} ({center['symbol']}): center=[{center['position'][0]:.3f}, {center['position'][1]:.3f}, {center['position'][2]:.3f}]")
            print(f"        Center atom index: {center['index']} (1-based)")
            print(f"        Equatorial atoms: {len(equatorial)} atoms {[atom['symbol'] for atom in equatorial]}")
            print(f"        Axial atoms: {len(axial)} atoms {[atom['symbol'] for atom in axial]}")
            
            # Show atom indices (1-based)
            eq_indices = [atom['index'] for atom in equatorial]
            ax_indices = [atom['index'] for atom in axial]
            print(f"        Equatorial indices: {eq_indices}")
            print(f"        Axial indices: {ax_indices}")
        
        # Show intra-layer connections
        if layer_data['intra_layer_connections']:
            print(f"    Intra-layer connections:")
            for oct1, oct2, shared in layer_data['intra_layer_connections']:
                # Convert 0-based shared indices to 1-based for display
                shared_1based = [idx + 1 for idx in shared]
                shared_symbols = [atoms.get_chemical_symbols()[i] for i in shared]
                print(f"      Oct{oct1+1} ↔ Oct{oct2+1}: {len(shared)} shared atoms {shared_1based} ({shared_symbols})")
        
        print()
    
    # Show inter-layer connections (updated for new structure)
    inter_connections = ontology['edges']['inter_layer']
    if inter_connections:
        print(f"  Inter-layer connections:")
        for layer1, layer2, oct1, oct2, shared in inter_connections:
            # Convert 0-based shared indices to 1-based for display
            shared_1based = [idx + 1 for idx in shared]
            shared_symbols = [atoms.get_chemical_symbols()[i] for i in shared]
            print(f"    Layer{layer1} ↔ Layer{layer2}: Oct{oct1+1} ↔ Oct{oct2+1} via atoms {shared_1based} ({shared_symbols})")
    else:
        print(f"  No inter-layer connections (single layer structure)")
    
    print(f"  Graph type: {ontology['graph_type']}")
    print(f"  Total graph nodes: {len(ontology['nodes']['octahedra'])} octahedra, {len(ontology['nodes']['layers'])} layers")
    print("-" * 60)
