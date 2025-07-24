#!/usr/bin/env python3

import json
from SVC_materials.core.connectivity import ConnectivityAnalyzer
from SVC_materials.core.geometry import GeometryCalculator

# Load the test data
with open('tests/n3_ontology.json', 'r') as f:
    data = json.load(f)

octahedra_data = data['octahedra']

# Create mock coordinates and symbols for testing
all_coords = []
all_symbols = []
for i in range(118):  # From the original data
    all_coords.append([0, 0, 0])  # Dummy coordinates
    all_symbols.append('Cl')  # Dummy symbols

# Test the connectivity analyzer
print("=== TESTING FIXED CONNECTIVITY ANALYZER ===")
geom_calc = GeometryCalculator()
analyzer = ConnectivityAnalyzer(geom_calc)
connectivity = analyzer.analyze_octahedra_connectivity(octahedra_data, all_coords, all_symbols)

print("\n=== CONNECTION TYPES ===")
connections = connectivity['octahedra_connections']
for oct_key, conns in connections.items():
    print(f"\n{oct_key}:")
    for conn in conns:
        print(f"  -> {conn['connected_octahedron']} via {conn['connection_type']} atom {conn['shared_atom_index']}")

print("\n=== SUMMARY BY CONNECTION TYPE ===")
connection_types = {'axial': [], 'equatorial': [], 'mixed': [], 'unknown': []}
for oct_key, conns in connections.items():
    for conn in conns:
        conn_type = conn['connection_type']
        connection_types[conn_type].append(f"{oct_key} -> {conn['connected_octahedron']}")

for conn_type, connections_list in connection_types.items():
    if connections_list:
        print(f"\n{conn_type.upper()} connections:")
        for conn in connections_list:
            print(f"  {conn}") 