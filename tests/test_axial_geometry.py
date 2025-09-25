#!/usr/bin/env python3
"""
Concise test: Measure axial-center-axial distances in detected octahedra.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q2D_Materials.utils.geometry import _octahedra_ontology, _calculate_distances
from ase.io import read
import numpy as np

def test_axial_geometry(atoms, name):
    """Test axial-center-axial geometry for octahedra using clean entity-relationship graph."""
    print(f"=== {name} ===")
    
    # Ensure PBC is enabled for crystalline structures
    atoms.pbc = True
    
    # Get clean entity-relationship graph
    graph = _octahedra_ontology(
        atoms.positions, atoms.get_chemical_symbols(),
        cutoff_distance=4.0, cell=atoms.get_cell()
    )
    
    entities = graph['entities']
    
    # Group octahedra by layers for display
    layer_groups = {}
    for oct_id, oct_data in entities['octahedra'].items():
        layer_id = oct_data['layer']
        if layer_id not in layer_groups:
            layer_groups[layer_id] = []
        layer_groups[layer_id].append(oct_id)
    
    for layer_id, oct_list in layer_groups.items():
        print(f"Layer {layer_id}:")
        
        for oct_id in oct_list:
            oct_data = entities['octahedra'][oct_id]
            
            # Get center atom info
            center_atom_id = oct_data['center_atom']
            center_atom = entities['atoms'][center_atom_id]
            center_pos = np.array(center_atom['position'])
            center_idx = center_atom['index']
            center_symbol = center_atom['symbol']
            
            # Use relationship-based atom classification
            classification = oct_data['atom_classification']
            
            equatorial_ids = classification['equatorial']
            axial_ids = classification['axial'] 
            terminal_ids = classification['terminal']
            
            print(f"  Oct{oct_id+1} (center={center_idx} {center_symbol}):")
            print(f"    🔗 Axial: {len(axial_ids)} atoms (inter-layer shared)")
            print(f"    ⚡ Equatorial: {len(equatorial_ids)} atoms (intra-layer shared)")
            print(f"    🔚 Terminal: {len(terminal_ids)} atoms (not shared)")
            
            # Test axial geometry if we have exactly 2 axial atoms
            if len(axial_ids) == 2:
                # Get axial atom info
                ax1_atom = entities['atoms'][axial_ids[0]]
                ax2_atom = entities['atoms'][axial_ids[1]]
                
                ax1_pos = np.array(ax1_atom['position'])
                ax2_pos = np.array(ax2_atom['position'])
                ax1_idx = ax1_atom['index']
                ax2_idx = ax2_atom['index']
                ax1_symbol = ax1_atom['symbol']
                ax2_symbol = ax2_atom['symbol']
                
                # PBC-aware distances
                d1 = _calculate_distances(center_pos, [ax1_pos], atoms.get_cell())[0]
                d2 = _calculate_distances(center_pos, [ax2_pos], atoms.get_cell())[0]
                axial_distance = _calculate_distances(ax1_pos, [ax2_pos], atoms.get_cell())[0]
                
                # PBC-aware angle check (should be ~180° for linear)
                cell = atoms.get_cell()
                inv_cell = np.linalg.inv(cell)
                
                # Calculate PBC-corrected difference vectors
                diff1 = ax1_pos - center_pos
                diff1_cell = diff1 @ inv_cell.T
                diff1_cell = diff1_cell - np.round(diff1_cell)
                v1 = diff1_cell @ cell
                
                diff2 = ax2_pos - center_pos
                diff2_cell = diff2 @ inv_cell.T
                diff2_cell = diff2_cell - np.round(diff2_cell)
                v2 = diff2_cell @ cell
                
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                
                print(f"    🎯 Axial geometry: {ax1_symbol}{ax1_idx}({d1:.3f}Å)-{center_symbol}{center_idx}-{ax2_symbol}{ax2_idx}({d2:.3f}Å)")
                print(f"       Axial distance: {axial_distance:.3f}Å, Angle: {angle:.1f}°")
                
                if angle > 160:
                    print(f"       ✅ Linear octahedron (angle > 160°)")
                elif angle > 140:
                    print(f"       ⚠️  Distorted octahedron (140° < angle < 160°)")
                else:
                    print(f"       🚨 Highly distorted (angle < 140°)")
                    
            elif len(axial_ids) == 0:
                print(f"    📝 No axial atoms (single layer structure)")
                
                # For single layer, analyze terminal atoms instead
                if len(terminal_ids) == 2:
                    term1_atom = entities['atoms'][terminal_ids[0]]
                    term2_atom = entities['atoms'][terminal_ids[1]]
                    
                    term1_pos = np.array(term1_atom['position'])
                    term2_pos = np.array(term2_atom['position'])
                    
                    # Check if terminal atoms are opposite each other
                    cell = atoms.get_cell()
                    inv_cell = np.linalg.inv(cell)
                    
                    diff1 = term1_pos - center_pos
                    diff1_cell = diff1 @ inv_cell.T
                    diff1_cell = diff1_cell - np.round(diff1_cell)
                    v1 = diff1_cell @ cell
                    
                    diff2 = term2_pos - center_pos
                    diff2_cell = diff2 @ inv_cell.T
                    diff2_cell = diff2_cell - np.round(diff2_cell)
                    v2 = diff2_cell @ cell
                    
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
                    
                    print(f"    🔚 Terminal geometry: {term1_atom['symbol']}{term1_atom['index']}-{center_symbol}{center_idx}-{term2_atom['symbol']}{term2_atom['index']}")
                    print(f"       Terminal angle: {angle:.1f}° {'(quasi-linear)' if angle > 160 else '(bent)'}")
                
            else:
                print(f"    ⚠️  Unexpected axial count: {len(axial_ids)} (expected 0 or 2)")
    
    print()

# Test structures
structures = [
    ("MAPbCl3_n1_l1", read('tests/test_structures/MAPbCl3_n1_l1.vasp')),
    ("MAPbBr3_n2_l1", read('tests/test_structures/MAPbBr3_n2_l1.vasp')),
    ("MAPbI3_n3_l1", read('tests/test_structures/MAPbI3_n3_l1.vasp'))
]

print("🔍 Relationship-Based Octahedral Geometry Test")
print("Using crystallographic atom classification:")
print("  🔗 AXIAL = Inter-layer shared (corner-sharing)")
print("  ⚡ EQUATORIAL = Intra-layer shared (edge-sharing)")
print("  🔚 TERMINAL = Not shared (unique to octahedron)")
print()

for name, atoms in structures:
    test_axial_geometry(atoms, name)

print("✅ Test complete!")
print("Expected results:")
print("  🔗 Multi-layer: Axial atoms with ~180° angles")
print("  📝 Single layer: No axial atoms, terminal atoms analyzed")
print("  ⚡ Edge-sharing: 4 equatorial atoms between adjacent octahedra")
print("  🎯 Crystallographically meaningful classification!")
