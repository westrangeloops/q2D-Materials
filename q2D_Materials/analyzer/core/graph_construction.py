"""Comprehensive graph ontology construction for 2D materials.

This module builds the complete structural connectivity graph including
octahedra, layers, atoms, and molecular connections.

Uses Clifford 6D embedding for automatic PBC handling.
"""

import numpy as np
import networkx as nx
from ase import Atoms
import sys

from ..detection.layer_identification import _get_octahedron_layer, _identify_layers
from ..detection.octahedral_detection import _count_octahedra, find_shared_atoms
from ..utils.pymatgen_utils import get_covalent_bonds
from ..utils.clifford_embedding import embed_to_6d, clifford_distance, get_cell_lengths


def _graph_inorganic_ontology(
    atom_positions,
    atom_symbols,
    cell,
    cutoff_distance=4.0,
    min_tolerance=0.2,
    step=0.1,
    max_steps=20,
    non_metal_symbols=None,
    octahedra_edges=None,
    octahedra_centers=None,
    valid_halogen=None,
    valid_molecule=None,
):
    """Build comprehensive graph-based inorganic ontology for 2D materials using Clifford 6D embedding.

    Graph structure hierarchy:
    1. Layer Nodes: Classified as Surface or Central
    2. Octahedra Nodes: Each octahedron with classified atoms
    3. Atom Nodes: Individual atoms with properties and classifications

    Uses Clifford 6D embedding for automatic PBC handling - no periodic shift tracking needed.

    Parameters
    ----------
    atom_positions : list
        List of [x, y, z] coordinates
    atom_symbols : list
        List of atomic symbols
    cell : np.ndarray
        3x3 array of unit cell vectors
    cutoff_distance : float
        Maximum distance for octahedral neighbors
    min_tolerance : float
        Starting bond length tolerance for octahedra detection
    step : float
        Increment step for tolerance optimization
    max_steps : int
        Maximum steps for tolerance optimization
    non_metal_symbols : list, optional
        List of non-metal symbols for filtering central atoms
    octahedra_edges : list, optional
        Valid edge counts (e.g., [4, 6])
    octahedra_centers : list, optional
        Valid B-site atoms (e.g., ['Pb', 'Sn'])
    valid_halogen : list, optional
        Valid halogen symbols
    valid_molecule : list, optional
        Valid molecule element symbols

    Returns
    -------
    networkx.Graph
        Complete inorganic ontology graph with 6D Clifford coordinates
    """
    if non_metal_symbols is None:
        non_metal_symbols = ['C', 'O', 'N', 'H', 'F', 'Cl', 'Br', 'I']

    G = nx.Graph()
    n_atoms_total = len(atom_positions)
    
    # Extract cell lengths for Clifford embedding
    cell_lengths = get_cell_lengths(cell)
    
    # Embed all atoms to 6D Clifford space (done once)
    atom_positions_array = np.array(atom_positions)
    atom_6d_coords = embed_to_6d(atom_positions_array, cell_lengths)
    
    # Count octahedra using Clifford embedding
    octahedra_count, centers_positions, center_symbols, neighbor_indices, center_atom_indices, octahedra_geometries = _count_octahedra(
        atom_positions, atom_symbols, cutoff_distance, min_tolerance, step, max_steps, cell, non_metal_symbols,
        octahedra_edges=octahedra_edges,
        octahedra_centers=octahedra_centers,
        valid_halogen=valid_halogen,
        valid_molecule=valid_molecule,
    )
    
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    layers, x_atom_classifications = _identify_layers(
        neighbor_indices,
        shared_atoms,
        atom_positions=atom_positions_array,
        center_atom_indices=center_atom_indices,
        cell=cell,
        octahedra_geometries=octahedra_geometries,
    )

    for layer_id, layer_info in layers.items():
        G.add_node(
            f'layer_{layer_id}',
            node_type='layer',
            position=layer_info['position'],
            z_coord=layer_info.get('z_coord'),
            intralayer_x_atoms=layer_info.get('intralayer_x_atoms', []),
            interlayer_x_atoms_above=layer_info.get('interlayer_x_atoms_above', []),
            interlayer_x_atoms_below=layer_info.get('interlayer_x_atoms_below', []),
        )

    layer_ids = sorted(layers.keys())
    for i in range(len(layer_ids) - 1):
        layer_a = layer_ids[i]
        layer_b = layer_ids[i + 1]
        connecting_x_atoms = layers[layer_a].get('interlayer_x_atoms_above', [])
        if connecting_x_atoms:
            G.add_edge(
                f'layer_{layer_a}',
                f'layer_{layer_b}',
                edge_type='interlayer_connection',
                via_x_atoms=connecting_x_atoms
            )

    # Add octahedra nodes
    if octahedra_count > 0 and len(center_symbols) > 0 and len(neighbor_indices) > 0 and len(center_atom_indices) > 0:
        for i, (center_pos, center_sym, neighbors, center_idx) in enumerate(zip(
            centers_positions, center_symbols, neighbor_indices, center_atom_indices
        )):
            geometry_info = octahedra_geometries[i] if i < len(octahedra_geometries) else None
            
            if geometry_info is None:
                raise ValueError(
                    f"Missing geometry classification for octahedron {i} with center atom {center_idx} ({center_sym})"
                )
            
            # Validate geometry counts
            counts = {
                'axial_top': sum(1 for v in geometry_info.values() if v == 'axial_top'),
                'axial_bottom': sum(1 for v in geometry_info.values() if v == 'axial_bottom'),
                'equatorial': sum(1 for v in geometry_info.values() if v == 'equatorial')
            }
            
            if counts['axial_top'] != 1 or counts['axial_bottom'] != 1 or counts['equatorial'] != 4:
                raise ValueError(
                    f"Invalid geometry classification for octahedron {i} (center atom {center_idx}): "
                    f"expected 1 axial_top, 1 axial_bottom, 4 equatorial, "
                    f"got {counts['axial_top']} axial_top, {counts['axial_bottom']} axial_bottom, "
                    f"{counts['equatorial']} equatorial"
                )
            
            # Determine terminal status
            terminal_atoms = []
            interlayer_atoms = []
            intralayer_atoms = []
            
            atom_to_octahedra = {}
            for oct_idx, oct_neighbors in enumerate(neighbor_indices):
                for neighbor_idx in oct_neighbors:
                    if neighbor_idx not in atom_to_octahedra:
                        atom_to_octahedra[neighbor_idx] = []
                    atom_to_octahedra[neighbor_idx].append(oct_idx)
            
            for neighbor_idx in neighbors:
                geometry_label = geometry_info.get(neighbor_idx)
                if geometry_label is None:
                    continue
                
                n_octahedra_sharing = len(atom_to_octahedra.get(neighbor_idx, []))
                is_terminal = (n_octahedra_sharing == 1)
                
                if is_terminal:
                    terminal_atoms.append(neighbor_idx)
                elif geometry_label in ['axial_top', 'axial_bottom']:
                    interlayer_atoms.append(neighbor_idx)
                elif geometry_label == 'equatorial':
                    intralayer_atoms.append(neighbor_idx)
            
            layer_id = _get_octahedron_layer(i, layers)
            
            G.add_node(
                f'octahedron_{i}',
                node_type='octahedron',
                central_atom=center_idx,
                terminal_atoms=terminal_atoms,
                interlayer_atoms=interlayer_atoms,
                intralayer_atoms=intralayer_atoms
            )
            
            G.add_edge(f'layer_{layer_id}', f'octahedron_{i}', edge_type='contains')
    
    # Add atom nodes with 6D coordinates
    for i, (pos, symbol) in enumerate(zip(atom_positions, atom_symbols)):
        node_data = {
            'node_type': 'atom',
            'vasp_index': i,
            'symbol': symbol,
            'direct_coordinates': pos.tolist(),  # Original 3D coordinates
            'clifford_6d': atom_6d_coords[i].tolist(),  # 6D Clifford coordinates
        }
        if i in x_atom_classifications:
            x_info = x_atom_classifications[i]
            node_data['x_atom_type'] = x_info['type']
            node_data['x_connected_octahedra'] = x_info['connected_octahedra']
        G.add_node(f'atom_{i}', **node_data)
    
    # Add edges between octahedra (shared atoms)
    for (oct_i, oct_j), shared_atom_indices in shared_atoms.items():
        G.add_edge(
            f'octahedron_{oct_i}',
            f'octahedron_{oct_j}',
            edge_type='shares_atoms',
            shared_atoms=shared_atom_indices
        )
    
    # Build mapping for terminal status
    atom_to_octahedra = {}
    for oct_idx, oct_neighbors in enumerate(neighbor_indices):
        for neighbor_idx in oct_neighbors:
            if neighbor_idx not in atom_to_octahedra:
                atom_to_octahedra[neighbor_idx] = []
            atom_to_octahedra[neighbor_idx].append(oct_idx)
    
    # Add edges between octahedra and atoms with 6D distances
    for i, neighbors in enumerate(neighbor_indices):
        geometry_info = octahedra_geometries[i] if i < len(octahedra_geometries) else None
        center_idx = center_atom_indices[i]
        center_6d = atom_6d_coords[center_idx]
        
        for neighbor_idx in neighbors:
            edge_attrs = {'edge_type': 'contains_atom'}
            
            # Add geometry label
            if geometry_info and neighbor_idx in geometry_info:
                edge_attrs['geometry'] = geometry_info[neighbor_idx]
            
            # Add terminal status
            n_octahedra_sharing = len(atom_to_octahedra.get(neighbor_idx, []))
            edge_attrs['terminal'] = (n_octahedra_sharing == 1)
            
            # Calculate and store 6D distance (PBC-aware)
            neighbor_6d = atom_6d_coords[neighbor_idx]
            dist_6d = clifford_distance(center_6d, neighbor_6d)
            edge_attrs['clifford_distance'] = float(dist_6d)
            
            G.add_edge(f'octahedron_{i}', f'atom_{neighbor_idx}', **edge_attrs)
        
        # Add center atom edge
        G.add_edge(f'octahedron_{i}', f'atom_{center_idx}', edge_type='has_center')
    
    # Add reverse edges (atom -> octahedron)
    for i, center_idx in enumerate(center_atom_indices):
        G.add_edge(f'atom_{center_idx}', f'octahedron_{i}', edge_type='is_center_of')

    # Create molecular connections (uses Clifford 6D coordinates)
    _create_molecular_connections(G, atom_positions, atom_symbols, neighbor_indices, atom_6d_coords, cell)
    
    # Verify all atoms are in graph
    atom_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'atom']
    octahedra_nodes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'octahedron']
    layer_nodes = [n for n in G.nodes() if 'layer' in str(n).lower()]
    total_nodes = len(G.nodes())
    is_critical = (total_nodes == n_atoms_total and n_atoms_total > 0)
    
    if is_critical or octahedra_count == 0:
        layers_count_safe = len(layers) if layers is not None and len(layers) > 0 else 0
        msg = f"WARNING: Graph integrity issue - {n_atoms_total} atoms, {total_nodes} nodes, {octahedra_count} octahedra detected, {layers_count_safe} layers"
        print(msg, file=sys.stderr)
    
    if total_nodes == n_atoms_total and n_atoms_total > 0 and octahedra_count == 0:
        error_msg = (
            f"Graph construction failed: Structure has {n_atoms_total} atoms but graph only has {total_nodes} nodes. "
            f"No structural nodes (octahedra/layers) were added. Octahedra detection returned 0. "
            f"This indicates octahedra detection failed for this structure."
        )
        print(f"ERROR: {error_msg}", file=sys.stderr)
    
    # Ensure all atoms are in the graph
    if len(atom_nodes) < n_atoms_total:
        missing_indices = []
        for i in range(n_atoms_total):
            if f'atom_{i}' not in G:
                missing_indices.append(i)
                node_data = {
                    'node_type': 'atom',
                    'vasp_index': i,
                    'symbol': atom_symbols[i] if atom_symbols else 'Unknown',
                    'direct_coordinates': atom_positions[i].tolist(),
                    'clifford_6d': atom_6d_coords[i].tolist(),
                }
                G.add_node(f'atom_{i}', **node_data)
    
    return G

def _create_molecular_connections(G, atom_positions, atom_symbols, neighbor_indices, atom_6d_coords, cell):
    """Create molecular connections using pymatgen's CovalentBondNN.

    Uses proper bond chemistry instead of simple distance cutoffs.
    Hydrogen bonds are detected separately using Clifford 6D distances.

    Parameters
    ----------
    G : networkx.Graph
        Graph to add connections to
    atom_positions : list
        List of atom coordinates (3D)
    atom_symbols : list
        List of atomic symbols
    neighbor_indices : list
        List of neighbor indices for each octahedron
    atom_6d_coords : np.ndarray
        Array of 6D Clifford coordinates for all atoms (PBC-aware)
    cell : np.ndarray
        Unit cell matrix (3x3)
    """
    atoms_in_octahedra = set()
    for oct_neighbors in neighbor_indices:
        atoms_in_octahedra.update(oct_neighbors)

    isolated_atoms = [i for i, _ in enumerate(atom_symbols) if i not in atoms_in_octahedra]

    if not isolated_atoms:
        return

    full_atoms = Atoms(
        symbols=atom_symbols,
        positions=atom_positions,
        cell=cell,
        pbc=True
    )

    organic_elements = {'C', 'N', 'H', 'O', 'S', 'P'}
    organic_indices = {i for i, sym in enumerate(atom_symbols) if sym in organic_elements}

    covalent_bonds = get_covalent_bonds(full_atoms, atom_indices=organic_indices)

    for i, j, bond_length in covalent_bonds:
        symbol_i = atom_symbols[i]
        symbol_j = atom_symbols[j]

        if symbol_i == 'H' and symbol_j == 'H':
            continue

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

        G.add_edge(f'atom_{i}', f'atom_{j}', edge_type='covalent_bond', bond_type=bond_type, distance=bond_length)

    halogen_symbols = {'F', 'Cl', 'Br', 'I'}
    halogen_atoms = [i for i, symbol in enumerate(atom_symbols) if i in atoms_in_octahedra and symbol in halogen_symbols]

    hbond_cutoff = 3.2

    for h_atom_idx in isolated_atoms:
        if atom_symbols[h_atom_idx] != 'H':
            continue

        h_6d = atom_6d_coords[h_atom_idx]
        for halogen_idx in halogen_atoms:
            # Use Clifford 6D distance (automatically PBC-aware)
            halogen_6d = atom_6d_coords[halogen_idx]
            distance = clifford_distance(h_6d, halogen_6d)
            if distance <= hbond_cutoff:
                G.add_edge(f'atom_{h_atom_idx}', f'atom_{halogen_idx}', edge_type='hydrogen_bond', distance=float(distance))
