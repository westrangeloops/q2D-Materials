"""
Comprehensive graph ontology construction for 2D materials.

This module builds the complete structural connectivity graph including
octahedra, layers, atoms, and molecular connections.

Uses pymatgen's CovalentBondNN for proper covalent bond detection, which
prevents incorrectly merging molecules that are spatially close but not
covalently bonded.

Functions
---------
_graph_inorganic_ontology
    Build comprehensive graph-based inorganic ontology
_create_molecular_connections
    Create covalent and hydrogen bond connections for molecules
"""

import numpy as np
import networkx as nx
from ase import Atoms
from .octahedral_detection import _count_octahedra, find_shared_atoms, _classify_atoms
from .layer_identification import _identify_layers, _get_octahedron_layer
from .pymatgen_utils import get_covalent_bonds
from ..utils.geometry import _calculate_distances


def _graph_inorganic_ontology(atom_positions, atom_symbols, cell, cutoff_distance=4.0, 
                             min_tolerance=0.2, step=0.1, max_steps=20,
                             non_metal_symbols=['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']):
    """
    Build a comprehensive graph-based inorganic ontology for 2D materials.
    
    The graph structure follows this hierarchy:
    1. Unit Cell Node: Contains lattice parameters and angles
    2. Layer Nodes: Classified as Surface or Central based on terminal atoms
    3. Octahedra Nodes: Each octahedron with classified atoms
    4. Atom Nodes: Individual atoms with properties and classifications
    
    Parameters:
    atom_positions: list of [x, y, z] coordinates
    atom_symbols: list of atomic symbols
    cell: 3x3 array of unit cell vectors
    cutoff_distance: maximum distance for octahedral neighbors
    min_tolerance: starting bond length tolerance for octahedra detection
    step: increment step for tolerance optimization
    max_steps: maximum steps for tolerance optimization
    non_metal_symbols: list of non-metal symbols for filtering central atoms
    
    Returns:
    networkx.Graph: Complete inorganic ontology graph
    """
    import numpy as np
    
    # Initialize the graph
    G = nx.Graph()
    
    # 1. This part is managed by the cell_analysis.py module.
    
    # 2. DETECT OCTAHEDRA
    octahedra_count, centers_positions, center_symbols, neighbor_indices, center_atom_indices = _count_octahedra(
        atom_positions, atom_symbols, cutoff_distance, min_tolerance, step, max_steps, cell, non_metal_symbols
    )
    
    # 3. FIND SHARED ATOMS BETWEEN OCTAHEDRA
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    # 4. CLASSIFY ATOMS BASED ON SHARING (legacy, kept for compatibility)
    atom_classifications = _classify_atoms(atom_positions, atom_symbols, neighbor_indices, shared_atoms)

    # 5. IDENTIFY LAYERS BY Z-COORDINATE GROUPING (new approach)
    # Pass atom positions and center indices for Z-based layer detection
    layers, x_atom_classifications = _identify_layers(
        neighbor_indices,
        shared_atoms,
        atom_positions=np.array(atom_positions),
        center_atom_indices=center_atom_indices,
        cell=cell,
    )

    # 6. CREATE LAYER NODES
    for layer_id, layer_info in layers.items():
        G.add_node(f'layer_{layer_id}',
                   node_type='layer',
                   position=layer_info['position'],  # 'surface' or 'central'
                   z_coord=layer_info.get('z_coord'),
                   intralayer_x_atoms=layer_info.get('intralayer_x_atoms', []),
                   interlayer_x_atoms_above=layer_info.get('interlayer_x_atoms_above', []),
                   interlayer_x_atoms_below=layer_info.get('interlayer_x_atoms_below', []))

        # Note: Unit cell connections will be handled by the analyzer

    # 6b. CREATE LAYER-TO-LAYER EDGES through interlayer X-atoms
    layer_ids = sorted(layers.keys())
    for i in range(len(layer_ids) - 1):
        layer_a = layer_ids[i]
        layer_b = layer_ids[i + 1]
        connecting_x_atoms = layers[layer_a].get('interlayer_x_atoms_above', [])
        if connecting_x_atoms:
            G.add_edge(f'layer_{layer_a}', f'layer_{layer_b}',
                       edge_type='interlayer_connection',
                       via_x_atoms=connecting_x_atoms)
    
    # 7. CREATE OCTAHEDRA NODES
    for i, (center_pos, center_sym, neighbors, center_idx) in enumerate(zip(
        centers_positions, center_symbols, neighbor_indices, center_atom_indices)):

        # Classify atoms in this octahedron using new Z-based classifications
        terminal_atoms = []  # Axial (connected to only 1 octahedron)
        interlayer_atoms = []  # Interlayer (connects floors)
        intralayer_atoms = []  # Intralayer (equatorial, same floor)

        for neighbor_idx in neighbors:
            if neighbor_idx in x_atom_classifications:
                x_type = x_atom_classifications[neighbor_idx]['type']
                if x_type == 'axial':
                    terminal_atoms.append(neighbor_idx)
                elif x_type == 'interlayer':
                    interlayer_atoms.append(neighbor_idx)
                elif x_type == 'intralayer':
                    intralayer_atoms.append(neighbor_idx)
            else:
                # Fallback to legacy classification if not in new classifications
                if neighbor_idx in atom_classifications:
                    atom_class = atom_classifications[neighbor_idx]['classification']
                    if atom_class == 'terminal':
                        terminal_atoms.append(neighbor_idx)
                    elif atom_class in ['axial', 'mixed']:
                        interlayer_atoms.append(neighbor_idx)
                    elif atom_class in ['equatorial']:
                        intralayer_atoms.append(neighbor_idx)

        # Determine which layer this octahedron belongs to
        layer_id = _get_octahedron_layer(i, layers)

        G.add_node(f'octahedron_{i}',
                   node_type='octahedron',
                   central_atom=center_idx,
                   terminal_atoms=terminal_atoms,
                   interlayer_atoms=interlayer_atoms,
                   intralayer_atoms=intralayer_atoms)

        # Connect octahedron to its layer
        G.add_edge(f'layer_{layer_id}', f'octahedron_{i}', edge_type='contains')
    
    # 8. CREATE ATOM NODES
    for i, (pos, symbol) in enumerate(zip(atom_positions, atom_symbols)):
        node_data = {
            'node_type': 'atom',
            'vasp_index': i,
            'symbol': symbol,
            'direct_coordinates': pos.tolist(),
        }
        # Add X-atom classification if available
        if i in x_atom_classifications:
            x_info = x_atom_classifications[i]
            node_data['x_atom_type'] = x_info['type']
            node_data['x_connected_octahedra'] = x_info['connected_octahedra']
        G.add_node(f'atom_{i}', **node_data)
    
    # 9. CREATE OCTAHEDRA-TO-OCTAHEDRA CONNECTIONS (for shared atoms)
    for (oct_i, oct_j), shared_atom_indices in shared_atoms.items():
        G.add_edge(f'octahedron_{oct_i}', f'octahedron_{oct_j}', 
                   edge_type='shares_atoms', 
                   shared_atoms=shared_atom_indices)
    
    # 10. CREATE OCTAHEDRA-TO-ATOM CONNECTIONS
    for i, neighbors in enumerate(neighbor_indices):
        # Connect octahedron to all its neighbor atoms
        for neighbor_idx in neighbors:
            G.add_edge(f'octahedron_{i}', f'atom_{neighbor_idx}', 
                       edge_type='contains_atom')
        
        # Connect octahedron to its central atom
        center_idx = center_atom_indices[i]
        G.add_edge(f'octahedron_{i}', f'atom_{center_idx}', 
                   edge_type='has_center')
    
    # 11. CREATE CENTRAL ATOM TO OCTAHEDRA CONNECTIONS
    # Each central atom (divalent) is connected to all octahedra it belongs to
    for i, center_idx in enumerate(center_atom_indices):
        G.add_edge(f'atom_{center_idx}', f'octahedron_{i}', 
                   edge_type='is_center_of')
    
    # 12. CREATE MOLECULAR CONNECTIONS (H-N-C bonds and H-halogen bonds)
    _create_molecular_connections(G, atom_positions, atom_symbols, neighbor_indices, cell)
    
    return G

def _create_molecular_connections(G, atom_positions, atom_symbols, neighbor_indices, cell):
    """
    Create molecular connections using pymatgen's CovalentBondNN.

    Uses proper bond chemistry instead of simple distance cutoffs, which
    prevents incorrectly bonding atoms from different molecules that are
    spatially close but not covalently bonded.

    Hydrogen bonds are detected separately and marked as such to prevent
    them from merging separate molecules during graph traversal.

    Parameters:
    G: networkx graph
    atom_positions: list of atom coordinates
    atom_symbols: list of atomic symbols
    neighbor_indices: list of neighbor indices for each octahedron
    cell: unit cell vectors for PBC calculations
    """
    # Find atoms that are not part of any octahedron (isolated atoms)
    atoms_in_octahedra = set()
    for oct_neighbors in neighbor_indices:
        atoms_in_octahedra.update(oct_neighbors)

    isolated_atoms = []
    for i, symbol in enumerate(atom_symbols):
        if i not in atoms_in_octahedra:
            isolated_atoms.append(i)

    if not isolated_atoms:
        return

    # Build ASE Atoms object for pymatgen
    full_atoms = Atoms(
        symbols=atom_symbols,
        positions=atom_positions,
        cell=cell,
        pbc=True
    )

    # Only detect covalent bonds for organic atoms (not metals like Pb)
    # CovalentBondNN doesn't have bond data for metal-metal bonds
    organic_elements = {'C', 'N', 'H', 'O', 'S', 'P'}
    organic_indices = {i for i, sym in enumerate(atom_symbols) if sym in organic_elements}

    # Get covalent bonds using pymatgen's CovalentBondNN
    # This uses proper bond chemistry instead of simple distance cutoffs
    covalent_bonds = get_covalent_bonds(full_atoms, atom_indices=organic_indices)

    # Add covalent bonds to graph (H-H bonds are already filtered by pymatgen_utils)
    for i, j, bond_length in covalent_bonds:
        symbol_i = atom_symbols[i]
        symbol_j = atom_symbols[j]

        # Skip H-H bonds (double-check, should already be filtered)
        if symbol_i == 'H' and symbol_j == 'H':
            continue

        # Determine bond type label
        if {symbol_i, symbol_j} == {'H', 'N'}:
            bond_type = 'H-N'
        elif {symbol_i, symbol_j} == {'N', 'C'}:
            bond_type = 'N-C'
        elif {symbol_i, symbol_j} == {'H', 'C'}:
            bond_type = 'H-C'
        elif symbol_i == 'C' and symbol_j == 'C':
            bond_type = 'C-C'
        else:
            bond_type = f'{symbol_i}-{symbol_j}'

        G.add_edge(f'atom_{i}', f'atom_{j}',
                  edge_type='covalent_bond', bond_type=bond_type, distance=bond_length)

    # Add hydrogen bonds separately (H to halogens in octahedra)
    # These are NOT treated as covalent bonds to prevent merging molecules
    halogen_symbols = {'F', 'Cl', 'Br', 'I'}
    halogen_atoms = []
    for i, symbol in enumerate(atom_symbols):
        if i in atoms_in_octahedra and symbol in halogen_symbols:
            halogen_atoms.append(i)

    # H-bond cutoff (typically 2.5-3.5 A)
    hbond_cutoff = 3.2

    for h_atom_idx in isolated_atoms:
        if atom_symbols[h_atom_idx] != 'H':
            continue

        h_position = atom_positions[h_atom_idx]

        for halogen_idx in halogen_atoms:
            distance = _calculate_distances(h_position, [atom_positions[halogen_idx]], cell)[0]

            if distance <= hbond_cutoff:
                G.add_edge(f'atom_{h_atom_idx}', f'atom_{halogen_idx}',
                          edge_type='hydrogen_bond', distance=distance)
