#!/usr/bin/env python3
"""
Explore and visualize the unified ontology graph structure.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from q2D_Materials.utils.geometry import _octahedra_ontology
from q2D_Materials.utils.cell_analysis import _get_cell_properties
from ase.io import read
import json
import pprint

def create_unified_ontology(file_path):
    """Create the unified ontology."""
    atoms = read(file_path)
    atoms.pbc = True
    
    cell_properties = _get_cell_properties(atoms)
    octahedra_ontology = _octahedra_ontology(
        atoms.positions, 
        atoms.get_chemical_symbols(), 
        cell=atoms.get_cell()
    )
    
    return {
        'cell_properties': cell_properties,
        'octahedra_ontology': octahedra_ontology
    }

def explore_graph_structure(unified_ontology):
    """Explore the graph structure interactively."""
    print("🔍 GRAPH STRUCTURE EXPLORER")
    print("=" * 60)
    
    # Top level keys
    print("📋 TOP-LEVEL STRUCTURE:")
    for key in unified_ontology.keys():
        print(f"   ├── {key}")
    print()
    
    # Cell properties
    print("🔬 CELL PROPERTIES:")
    cell_props = unified_ontology['cell_properties']
    for key, value in cell_props.items():
        if isinstance(value, (int, float, str)):
            print(f"   ├── {key}: {value}")
        else:
            print(f"   ├── {key}: {type(value).__name__}")
    print()
    
    # Octahedral network structure
    print("🏗️  OCTAHEDRAL NETWORK STRUCTURE:")
    network = unified_ontology['octahedra_ontology']
    print(f"   ├── Graph type: {network['graph_type']}")
    print(f"   ├── Total octahedra: {network['total_octahedra']}")
    print(f"   ├── Total layers: {network['total_layers']}")
    print(f"   ├── Main sections:")
    for key in network.keys():
        print(f"   │   ├── {key}")
    print()

def show_layer_details(unified_ontology):
    """Show detailed layer structure."""
    network = unified_ontology['octahedra_ontology']
    
    print("📚 LAYER-BY-LAYER BREAKDOWN:")
    print("=" * 60)
    
    for layer_id, layer_data in network['layers'].items():
        print(f"🗂️  Layer {layer_id}:")
        print(f"   ├── Octahedra count: {len(layer_data['octahedra'])}")
        print(f"   ├── Octahedra indices: {layer_data['octahedra']}")
        print(f"   ├── Connections: {len(layer_data['intra_layer_connections'])}")
        
        # Show each octahedron
        for oct_idx in layer_data['octahedra']:
            oct_data = layer_data['octahedra_data'][oct_idx]
            center = oct_data['center']
            print(f"   │")
            print(f"   ├── 🔷 Octahedron {oct_idx+1}:")
            print(f"   │   ├── Center: atom {center['index']} ({center['symbol']})")
            print(f"   │   ├── Position: [{center['position'][0]:.3f}, {center['position'][1]:.3f}, {center['position'][2]:.3f}]")
            print(f"   │   ├── Equatorial atoms: {len(oct_data['equatorial']['atoms'])}")
            print(f"   │   ├── Axial atoms: {len(oct_data['axial']['atoms'])}")
            
            # Show atomic details
            eq_indices = [atom['index'] for atom in oct_data['equatorial']['atoms']]
            ax_indices = [atom['index'] for atom in oct_data['axial']['atoms']]
            print(f"   │   ├── Equatorial indices: {eq_indices}")
            print(f"   │   └── Axial indices: {ax_indices}")
        print()

def export_to_json(unified_ontology, filename):
    """Export the graph to JSON for external viewing."""
    print(f"📄 EXPORTING TO JSON: {filename}")
    
    # Convert numpy arrays and other non-serializable objects
    def convert_for_json(obj):
        if hasattr(obj, 'tolist'):  # numpy arrays
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(item) for item in obj]
        else:
            return obj
    
    json_data = convert_for_json(unified_ontology)
    
    with open(filename, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"   ✅ Exported to {filename}")
    print(f"   📖 You can now view it with: cat {filename} | jq")
    print(f"   🌐 Or open in any JSON viewer/editor")
    print()

def show_connectivity_graph(unified_ontology):
    """Show the connectivity as a simple graph."""
    network = unified_ontology['octahedra_ontology']
    
    print("🔗 CONNECTIVITY GRAPH:")
    print("=" * 60)
    
    # Intra-layer connections
    print("📚 INTRA-LAYER CONNECTIONS (Edge-sharing):")
    for layer_id, connections in network['edges']['intra_layer'].items():
        print(f"   Layer {layer_id}:")
        for oct1, oct2, shared in connections:
            print(f"      Oct{oct1+1} ═══ Oct{oct2+1} ({len(shared)} shared atoms)")
    print()
    
    # Inter-layer connections
    print("🌐 INTER-LAYER CONNECTIONS (Corner-sharing):")
    if network['edges']['inter_layer']:
        for layer1, layer2, oct1, oct2, shared in network['edges']['inter_layer']:
            print(f"   Layer{layer1} ──┐")
            print(f"              │ Oct{oct1+1} ··· Oct{oct2+1} ({len(shared)} shared)")
            print(f"   Layer{layer2} ──┘")
    else:
        print("   (No inter-layer connections - single layer structure)")
    print()

def interactive_explorer(unified_ontology):
    """Interactive exploration menu."""
    while True:
        print("\n🎛️  INTERACTIVE GRAPH EXPLORER")
        print("=" * 40)
        print("1. 📋 Show overall structure")
        print("2. 📚 Show layer details")
        print("3. 🔗 Show connectivity graph")
        print("4. 📄 Export to JSON")
        print("5. 🔍 Show specific octahedron")
        print("6. 📊 Show summary statistics")
        print("0. ❌ Exit")
        
        choice = input("\n🔢 Enter your choice (0-6): ").strip()
        
        if choice == '1':
            explore_graph_structure(unified_ontology)
        elif choice == '2':
            show_layer_details(unified_ontology)
        elif choice == '3':
            show_connectivity_graph(unified_ontology)
        elif choice == '4':
            filename = input("📁 Enter filename (default: unified_ontology.json): ").strip()
            if not filename:
                filename = "unified_ontology.json"
            export_to_json(unified_ontology, filename)
        elif choice == '5':
            show_specific_octahedron(unified_ontology)
        elif choice == '6':
            show_summary_stats(unified_ontology)
        elif choice == '0':
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

def show_specific_octahedron(unified_ontology):
    """Show details of a specific octahedron."""
    network = unified_ontology['octahedra_ontology']
    total_oct = network['total_octahedra']
    
    if total_oct == 0:
        print("❌ No octahedra found in this structure.")
        return
    
    print(f"\n🔷 Available octahedra: 1 to {total_oct}")
    try:
        oct_num = int(input(f"Enter octahedron number (1-{total_oct}): "))
        oct_idx = oct_num - 1
        
        if oct_idx < 0 or oct_idx >= total_oct:
            print("❌ Invalid octahedron number.")
            return
        
        # Find which layer this octahedron is in
        layer_id = network['nodes']['octahedra'][oct_idx]['layer_id']
        oct_data = network['layers'][layer_id]['octahedra_data'][oct_idx]
        
        print(f"\n🔍 OCTAHEDRON {oct_num} DETAILS:")
        print("=" * 40)
        center = oct_data['center']
        print(f"🎯 Center: atom {center['index']} ({center['symbol']})")
        print(f"📍 Position: [{center['position'][0]:.3f}, {center['position'][1]:.3f}, {center['position'][2]:.3f}]")
        print(f"📚 Layer: {layer_id}")
        print()
        
        print("🔷 Equatorial atoms:")
        for atom in oct_data['equatorial']['atoms']:
            print(f"   atom {atom['index']} ({atom['symbol']}) at [{atom['position'][0]:.3f}, {atom['position'][1]:.3f}, {atom['position'][2]:.3f}]")
        
        print("\n🔶 Axial atoms:")
        for atom in oct_data['axial']['atoms']:
            print(f"   atom {atom['index']} ({atom['symbol']}) at [{atom['position'][0]:.3f}, {atom['position'][1]:.3f}, {atom['position'][2]:.3f}]")
        
    except ValueError:
        print("❌ Please enter a valid number.")

def show_summary_stats(unified_ontology):
    """Show summary statistics."""
    cell_props = unified_ontology['cell_properties']
    network = unified_ontology['octahedra_ontology']
    
    print("\n📊 SUMMARY STATISTICS:")
    print("=" * 40)
    print(f"🔬 Cell volume: {cell_props.get('volume', 'N/A')} Å³")
    print(f"🏗️  Total octahedra: {network['total_octahedra']}")
    print(f"📚 Total layers: {network['total_layers']}")
    
    if network['total_layers'] > 0:
        layer_sizes = [len(layer_data['octahedra']) for layer_data in network['layers'].values()]
        print(f"📏 Layer distribution: {layer_sizes}")
        print(f"🔗 Total intra-layer connections: {sum(len(conns) for conns in network['edges']['intra_layer'].values())}")
        print(f"🌐 Total inter-layer connections: {len(network['edges']['inter_layer'])}")

if __name__ == "__main__":
    print("🚀 UNIFIED ONTOLOGY GRAPH EXPLORER")
    print("=" * 60)
    
    # Load and create unified ontology
    test_file = "tests/test_structures/MAPbCl3_n1_l1.vasp"
    unified_ontology = create_unified_ontology(test_file)
    
    print("✅ Unified ontology loaded!")
    print(f"📁 Structure: {test_file}")
    
    # Start interactive exploration
    interactive_explorer(unified_ontology)
