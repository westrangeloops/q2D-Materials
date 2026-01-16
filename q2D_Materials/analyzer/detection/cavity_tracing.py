"""Cavity detection algorithm for A-site cavities in perovskites.

This module implements cavity detection by finding the 8 nearest B-sites (octahedra centers)
around each A-site molecule using periodic boundary conditions.

Algorithm:
1. For each A-site molecule, calculate its 3D center position (PBC-aware)
2. Find the 8 nearest B-sites using 27-image periodic boundary conditions
3. Validate topology: atoms form 2 layers with at least 2 B-sites each
4. Find X-site atoms that connect to these B-sites
5. Apply bipartite matching: each B connects to exactly 3 X, each X to exactly 2 B
6. Build cavity subgraph with PBC-unwrapped coordinates

Result: Complete representation of cuboctahedral cavities as isolated subgraphs.
"""

import sys
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict

from ..utils.clifford_embedding import (
    embed_to_6d,
    clifford_distance,
    get_cell_lengths,
    unwrap_relative_coordinate,
)



def _find_valid_xb_assignment(connection_matrix, n_b, n_x):
    """
    Find a valid assignment using numpy where:
    - Each B atom connects to exactly 3 X atoms
    - Each X atom connects to exactly 2 B atoms
    - No duplicate bonds
    
    Uses efficient numpy operations instead of loops.
    
    Parameters
    ----------
    connection_matrix : np.ndarray
        Binary matrix of shape (n_x, n_b) where connection_matrix[x, b] = 1 if X atom x can connect to B atom b
    n_b : int
        Number of B atoms
    n_x : int
        Number of X atoms
        
    Returns
    -------
    np.ndarray or None
        Assignment matrix of shape (n_x, n_b) where assignment[x, b] = 1 if X x is assigned to B b
        Returns None if no valid assignment found
    """
    # Target: n_b * 3 = n_x_selected * 2
    # For 4 B atoms: need 6 X atoms
    target_n_x = (n_b * 3) // 2
    
    # Use greedy + backtracking approach
    assignment = np.zeros((n_x, n_b), dtype=np.int32)
    
    def backtrack(x_idx):
        """Backtrack to find valid assignment using numpy operations."""
        # Check if complete
        x_connections = np.sum(assignment, axis=1)  # connections per X atom
        b_connections = np.sum(assignment, axis=0)  # connections per B atom
        
        # Count complete X atoms (exactly 2 connections)
        complete_x = np.sum(x_connections == 2)
        incomplete_x = np.sum((x_connections > 0) & (x_connections < 2))
        
        # If all needed X atoms are complete and all B atoms have exactly 3 connections
        if complete_x == target_n_x and np.all(b_connections == 3):
            return True
        
        # Early termination: if any B atom has >3 connections (should not happen)
        if np.any(b_connections > 3):
            return False
        
        # If we've processed all X atoms, not complete
        if x_idx >= n_x:
            return False
        
        # If this X atom already has 2 connections, skip to next
        if x_connections[x_idx] == 2:
            return backtrack(x_idx + 1)
        
        # If this X atom has 1 connection, try to add one more
        if x_connections[x_idx] == 1:
            current_b = np.where(assignment[x_idx, :] == 1)[0]
            possible_b = np.where(connection_matrix[x_idx, :] == 1)[0]
            possible_b = possible_b[possible_b != current_b[0]]  # Exclude already connected B
            
            for b2 in possible_b:
                if b_connections[b2] < 3:
                    assignment[x_idx, b2] = 1
                    if backtrack(x_idx + 1):
                        return True
                    assignment[x_idx, b2] = 0
            
            return backtrack(x_idx + 1)
        
        # If this X atom has 0 connections, try to connect to 2 B atoms
        if x_connections[x_idx] == 0:
            # Find possible B atoms for this X
            possible_b = np.where(connection_matrix[x_idx, :] == 1)[0]
            
            # Try all combinations of 2 B atoms
            if len(possible_b) >= 2:
                for i in range(len(possible_b)):
                    for j in range(i + 1, len(possible_b)):
                        b1, b2 = possible_b[i], possible_b[j]
                        
                        # Check if both B atoms have room
                        if b_connections[b1] < 3 and b_connections[b2] < 3:
                            assignment[x_idx, b1] = 1
                            assignment[x_idx, b2] = 1
                            
                            if backtrack(x_idx + 1):
                                return True
                            
                            # Backtrack
                            assignment[x_idx, b1] = 0
                            assignment[x_idx, b2] = 0
            
            # Also try skipping this X atom
            return backtrack(x_idx + 1)
        
        return backtrack(x_idx + 1)
    
    if backtrack(0):
        return assignment
    return None


def get_octahedra_data(graph: nx.Graph) -> Dict[int, Dict[str, Any]]:
    """Extract octahedra data from graph by querying through edges.
    
    This function derives terminal/interlayer/intralayer atoms by counting
    how many octahedra share each atom and checking layer assignments.
    
    Parameters
    ----------
    graph : nx.Graph
        The structural graph
        
    Returns
    -------
    dict
        Mapping from octahedron index to octahedra data with keys:
        - 'node_id': str - Octahedron node ID
        - 'central_atom': int - Index of B-site center atom
        - 'terminal_atoms': list - Atom indices belonging to only this octahedron
        - 'interlayer_atoms': list - Atom indices shared across layers
        - 'intralayer_atoms': list - Atom indices shared within same layer
    """
    # First pass: collect all octahedra, their centers, and ligands
    octahedra_info = {}  # oct_idx -> {'center': int, 'ligands': list of (atom_idx, geometry)}
    
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'octahedron':
            oct_idx = int(node.replace('octahedron_', ''))
            center_atom = None
            ligands = []
            
            # Query edges to find center and ligands
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if not edge_data or edge_data.get('edge_type') != 'contains':
                    continue
                
                # Get the atom index
                if not neighbor.startswith('atom_'):
                    continue
                atom_idx = int(neighbor.replace('atom_', ''))
                
                role = edge_data.get('role')
                if role == 'center':
                    center_atom = atom_idx
                elif role == 'ligand':
                    geometry = edge_data.get('geometry', 'unknown')
                    ligands.append((atom_idx, geometry))
            
            octahedra_info[oct_idx] = {
                'node_id': node,
                'center': center_atom,
                'ligands': ligands
            }
    
    # Second pass: build atom -> octahedra mapping for terminal detection
    atom_to_octahedra = {}
    for oct_idx, info in octahedra_info.items():
        for atom_idx, geometry in info['ligands']:
            if atom_idx not in atom_to_octahedra:
                atom_to_octahedra[atom_idx] = []
            atom_to_octahedra[atom_idx].append(oct_idx)
    
    # Get layer assignments for each octahedron
    oct_to_layer = _get_layer_assignments(graph)
    
    # Third pass: classify atoms as terminal/interlayer/intralayer
    octahedra_data = {}
    for oct_idx, info in octahedra_info.items():
        terminal_atoms = []
        interlayer_atoms = []
        intralayer_atoms = []
        
        oct_layer = oct_to_layer.get(oct_idx)
        
        for atom_idx, geometry in info['ligands']:
            sharing_octahedra = atom_to_octahedra.get(atom_idx, [])
            n_sharing = len(sharing_octahedra)
            
            if n_sharing == 1:
                # Terminal: belongs to only this octahedron
                terminal_atoms.append(atom_idx)
            elif n_sharing >= 2:
                # Check if shared across layers or within layer
                sharing_layers = set()
                for sharing_oct in sharing_octahedra:
                    layer = oct_to_layer.get(sharing_oct)
                    if layer is not None:
                        sharing_layers.add(layer)
                
                if len(sharing_layers) > 1:
                    # Shared across different layers
                    interlayer_atoms.append(atom_idx)
                else:
                    # Shared within same layer
                    intralayer_atoms.append(atom_idx)
        
        octahedra_data[oct_idx] = {
            'node_id': info['node_id'],
            'central_atom': info['center'],
            'terminal_atoms': terminal_atoms,
            'interlayer_atoms': interlayer_atoms,
            'intralayer_atoms': intralayer_atoms,
        }
    
    return octahedra_data


def _get_layer_assignments(graph: nx.Graph) -> Dict[int, int]:
    """Get layer assignments for each octahedron.
    
    Parameters
    ----------
    graph : nx.Graph
        The structural graph
        
    Returns
    -------
    dict
        Mapping from octahedron index to layer index
    """
    oct_to_layer = {}
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'layer':
            layer_id = int(node.replace('layer_', ''))
            for neighbor in graph.neighbors(node):
                if neighbor.startswith('octahedron_'):
                    oct_idx = int(neighbor.replace('octahedron_', ''))
                    oct_to_layer[oct_idx] = layer_id
    return oct_to_layer


def _build_octahedra_neighbor_map(
    octahedra_data: Dict[int, Dict[str, Any]],
    shared_atoms: Dict[Tuple[int, int], List[int]],
) -> Dict[int, Dict[int, List[int]]]:
    """Build a map of which octahedra share which atoms.
    
    Parameters
    ----------
    octahedra_data : dict
        Octahedra data from get_octahedra_data
    shared_atoms : dict
        Shared atoms between octahedra pairs
        
    Returns
    -------
    dict
        oct_idx -> {neighbor_oct_idx -> [shared_atom_indices]}
    """
    neighbor_map = defaultdict(lambda: defaultdict(list))
    
    for (oct_i, oct_j), atoms in shared_atoms.items():
        neighbor_map[oct_i][oct_j].extend(atoms)
        neighbor_map[oct_j][oct_i].extend(atoms)
    
    return dict(neighbor_map)


def _unwrap_atom_position(atom_idx: int, ref_pos: np.ndarray, atom_positions: np.ndarray, cell_lengths: np.ndarray) -> np.ndarray:
    """Unwrap atom position relative to reference using PBC.
    
    Parameters
    ----------
    atom_idx : int
        Atom index to unwrap
    ref_pos : np.ndarray
        Reference position (usually first atom in cavity)
    atom_positions : np.ndarray
        All atom positions
    cell_lengths : np.ndarray
        Cell lengths for PBC
        
    Returns
    -------
    np.ndarray
        PBC-unwrapped position relative to ref_pos
    """
    return unwrap_relative_coordinate(ref_pos, atom_positions[atom_idx], cell_lengths)


def assign_a_sites_to_cavities(
    cavities: List[Dict[str, Any]],
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell_lengths: np.ndarray,
    cavity_centers: List[np.ndarray],
) -> List[Dict[str, Any]]:
    """Assign A-site atoms/molecules to their containing cavities.
    
    Parameters
    ----------
    cavities : list
        List of cavity data dictionaries
    graph : nx.Graph
        The structural graph
    atom_positions : np.ndarray
        Atom positions
    atom_symbols : list
        Atom symbols
    cell_lengths : np.ndarray
        Cell lengths
    cavity_centers : list
        List of cavity center positions
        
    Returns
    -------
    list
        Updated cavity data with A-site assignments
    """
    # Find A-site molecules in graph using Molecule nodes
    a_site_atoms = []
    
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'molecule' and data.get('molecule_type') == 'a_site':
            # Get atoms in this molecule via CONTAINS edges
            molecule_atoms = []
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'contains':
                    neighbor_data = graph.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'atom':
                        atom_idx = neighbor_data.get('vasp_index')
                        if atom_idx is not None:
                            molecule_atoms.append(atom_idx)
            
            if not molecule_atoms:
                continue
            
            # Calculate center of mass for molecular A-sites
            if len(molecule_atoms) > 1:
                ref_pos = atom_positions[molecule_atoms[0]]
                positions = [ref_pos]
                for idx in molecule_atoms[1:]:
                    unwrapped = unwrap_relative_coordinate(
                        ref_pos, atom_positions[idx], cell_lengths
                    )
                    positions.append(unwrapped)
                center = np.mean(positions, axis=0)
            else:
                center = atom_positions[molecule_atoms[0]]
            
            formula = data.get('formula', '?')  # Get formula from molecule node
            a_site_atoms.append({
                'indices': molecule_atoms,
                'center': center,
                'formula': formula,
                'is_molecular': len(molecule_atoms) > 1,
            })
    
    # Assign each A-site to nearest cavity
    for a_site in a_site_atoms:
        if not cavity_centers:
            continue
            
        a_center = a_site['center']
        min_dist = float('inf')
        best_cavity_idx = None
        
        for i, cav_center in enumerate(cavity_centers):
            # Use PBC-aware distance
            a_6d = embed_to_6d(a_center, cell_lengths)
            cav_6d = embed_to_6d(cav_center, cell_lengths)
            dist = clifford_distance(a_6d, cav_6d)
            
            if dist < min_dist:
                min_dist = dist
                best_cavity_idx = i
        
        if best_cavity_idx is not None:
            if 'a_site_indices' not in cavities[best_cavity_idx]:
                cavities[best_cavity_idx]['a_site_indices'] = []
                cavities[best_cavity_idx]['a_site_formulas'] = []
                cavities[best_cavity_idx]['a_site_types'] = []
            
            cavities[best_cavity_idx]['a_site_indices'].extend(a_site['indices'])
            cavities[best_cavity_idx]['a_site_formulas'].append(a_site['formula'])
            cavities[best_cavity_idx]['a_site_types'].append(
                'molecular' if a_site['is_molecular'] else 'atomic'
            )
    
    # Update cavity metadata
    for cavity in cavities:
        cavity['contains_a_site'] = len(cavity.get('a_site_indices', [])) > 0
        if cavity.get('a_site_types'):
            cavity['a_site_type'] = cavity['a_site_types'][0] if len(cavity['a_site_types']) == 1 else 'mixed'
        else:
            cavity['a_site_type'] = None
    
    return cavities


def _build_cavity_subgraph(
    cavity_data: Dict[str, Any],
    octahedra_data: Dict[int, Dict[str, Any]],
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell: np.ndarray,
    cell_lengths: np.ndarray,
) -> nx.Graph:
    """Build an isolated subgraph for a cavity with PBC-unwrapped coordinates.
    
    The subgraph contains:
    - B-atom nodes (central atoms of octahedra)
    - X-atom nodes (ligand atoms)
    - A-site nodes (molecule atoms)
    - Edges: B-X bonds and molecular bonds
    
    Uses unique node IDs (atom_{idx}_pbc_{instance}) for atoms that appear multiple
    times with different PBC coordinates.
    
    All positions are PBC-unwrapped relative to cavity center.
    
    Parameters
    ----------
    cavity_data : dict
        Cavity data with octahedra indices and PBC coordinates
    octahedra_data : dict
        Octahedra data
    graph : nx.Graph
        The main structural graph
    atom_positions : np.ndarray
        Original atom positions
    atom_symbols : list
        Atom symbols
    cell : np.ndarray
        Unit cell matrix
    cell_lengths : np.ndarray
        Cell lengths
        
    Returns
    -------
    nx.Graph
        Isolated cavity subgraph with PBC-unwrapped coordinates and unique node IDs
    """
    from collections import defaultdict
    
    subgraph = nx.Graph()
    
    # Get cavity center
    center = np.array(cavity_data.get('center_position', [0, 0, 0]))
    inv_cell = np.linalg.inv(cell)
    
    def unwrap_position(pos: np.ndarray) -> np.ndarray:
        """Unwrap position relative to cavity center."""
        center_frac = center @ inv_cell
        pos_frac = pos @ inv_cell
        delta_frac = pos_frac - center_frac
        delta_frac = delta_frac - np.round(delta_frac)
        return center + (delta_frac @ cell)
    
    # Get cavity atoms with their 27-image positions and labels
    all_cavity_octs = cavity_data.get('octahedra_indices', [])
    b_data_list = cavity_data.get('b_data_list', [])  # List of (atom_idx, cartesian_pos, image_label)
    x_data_list = cavity_data.get('x_data_list', [])  # List of (atom_idx, cartesian_pos, image_label)
    
    # For A-site cavities: include molecule atoms
    # For spacer cavities: also include the molecule atoms (the whole spacer + cage)
    cavity_type = cavity_data.get('cavity_type', 'a_site')
    a_site_indices = cavity_data.get('a_site_indices', [])
    
    # Track atom node mappings
    atom_node_mapping = {}  # Maps (atom_idx, image_label) to unique node_id
    
    # Add 8 B atom nodes - one node per Cartesian position with unique image label
    for i, (b_idx, b_pos, img_label) in enumerate(b_data_list):
        # Create unique node ID using image label
        node_id = f'atom_{b_idx}_img_{img_label[0]}_{img_label[1]}_{img_label[2]}'
        atom_node_mapping[(b_idx, img_label)] = node_id
        
        # Get properties from parent graph atom node
        parent_atom_node = f'atom_{b_idx}'
        parent_data = graph.nodes.get(parent_atom_node, {})
        
        # Start with ALL inherited properties from parent
        node_attrs = dict(parent_data)
        
        # Replace xyz coordinates with PBC-unwrapped position
        node_attrs['x'] = float(b_pos[0])
        node_attrs['y'] = float(b_pos[1])
        node_attrs['z'] = float(b_pos[2])
        
        # Add minimal cavity-specific metadata
        node_attrs['original_index'] = b_idx
        node_attrs['image_label'] = img_label
        node_attrs['pbc_position'] = b_pos
        
        subgraph.add_node(node_id, **node_attrs)
    
    # Add 12 X atom nodes - one node per Cartesian position with unique image label
    for i, (x_idx, x_pos, img_label) in enumerate(x_data_list):
        # Create unique node ID using image label
        node_id = f'atom_{x_idx}_img_{img_label[0]}_{img_label[1]}_{img_label[2]}'
        atom_node_mapping[(x_idx, img_label)] = node_id
        
        # Get properties from parent graph atom node
        parent_atom_node = f'atom_{x_idx}'
        parent_data = graph.nodes.get(parent_atom_node, {})
        
        # Start with ALL inherited properties from parent
        node_attrs = dict(parent_data)
        
        # Replace xyz coordinates with PBC-unwrapped position
        node_attrs['x'] = float(x_pos[0])
        node_attrs['y'] = float(x_pos[1])
        node_attrs['z'] = float(x_pos[2])
        
        # Add minimal cavity-specific metadata
        node_attrs['original_index'] = x_idx
        node_attrs['image_label'] = img_label
        node_attrs['pbc_position'] = x_pos
        
        # Note: is_terminal, is_equatorial, is_interlayer are now inherited from parent graph
        # These properties are computed and stored during graph construction
        
        subgraph.add_node(node_id, **node_attrs)
    
    # Add A-site molecule with its atoms (PBC-aware - find nearest image of each atom to cavity center)
    molecule_atom_data = {}  # Maps atom_idx to (cartesian_pos, image_label)
    if a_site_indices:
        from q2D_Materials.utils.geometry.pbc_distances import find_nearest_image_positions
        
        # For each molecule atom, find its nearest image to the cavity center
        # (not the N nearest neighbors total, but the nearest image of each specific atom)
        for a_idx in a_site_indices:
            # Get this atom's position
            atom_pos = atom_positions[a_idx]
            atom_idx_array = np.array([a_idx])
            
            # Find the nearest image of THIS atom to cavity center
            nearest_indices, nearest_positions, nearest_distances, nearest_labels = \
                find_nearest_image_positions(
                    center, 
                    np.array([atom_pos]), 
                    atom_idx_array, 
                    cell, 
                    n_neighbors=1,  # Get only the single nearest image
                    exclude_indices=np.array([])
                )
            
            if len(nearest_indices) > 0:
                molecule_atom_data[a_idx] = (nearest_positions[0], tuple(nearest_labels[0]))
            else:
                # If no image found, this is a real error - don't silently fall back
                raise RuntimeError(f"Could not find nearest image for molecule atom {a_idx} to cavity center {center}")
        
        # Calculate molecule center from nearest images
        mol_center = np.mean([pos for pos, _ in molecule_atom_data.values()], axis=0) if molecule_atom_data else center
        
        # Add molecule node
        molecule_formula = ''.join(atom_symbols[a_idx] for a_idx in a_site_indices)
        molecule_node = 'molecule_a_site'
        
        subgraph.add_node(
            molecule_node,
            node_type='molecule',
            a_site_type='organic',
            formula=molecule_formula,
            atom_indices=a_site_indices,
            center_position=mol_center.tolist(),
            num_atoms=len(a_site_indices),
        )
        
        # Add A-site atoms (using their nearest image positions)
        for a_idx in a_site_indices:
            pos, img_label = molecule_atom_data[a_idx]
            
            # Create unique node ID using image label
            node_id = f'atom_{a_idx}_img_{img_label[0]}_{img_label[1]}_{img_label[2]}'
            atom_node_mapping[(a_idx, img_label)] = node_id
            
            # Get properties from parent graph atom node
            parent_atom_node = f'atom_{a_idx}'
            parent_data = graph.nodes.get(parent_atom_node, {})
            
            # Start with ALL inherited properties from parent
            node_attrs = dict(parent_data)
            
            # Replace xyz coordinates with PBC-unwrapped position
            node_attrs['x'] = float(pos[0])
            node_attrs['y'] = float(pos[1])
            node_attrs['z'] = float(pos[2])
            
            # Add minimal cavity-specific metadata
            node_attrs['original_index'] = a_idx
            node_attrs['image_label'] = img_label
            node_attrs['pbc_position'] = pos
            
            subgraph.add_node(node_id, **node_attrs)
            
            # Connect atom to molecule node using CONTAINS edge
            subgraph.add_edge(
                molecule_node,
                node_id,
                edge_type='contains',
            )
    
    # Add edges for B-X bonds based on the main graph connectivity
    # Edges connect atom indices. For each bond in the original graph,
    # we create edges between all image instances that are present in the cavity.
    
    # Collect all atom indices and their image labels in the cavity
    cavity_atoms_with_labels = {}  # {atom_idx: [image_label1, image_label2, ...]}
    
    for b_idx, b_pos, img_label in b_data_list:
        if b_idx not in cavity_atoms_with_labels:
            cavity_atoms_with_labels[b_idx] = []
        cavity_atoms_with_labels[b_idx].append(img_label)
    
    for x_idx, x_pos, img_label in x_data_list:
        if x_idx not in cavity_atoms_with_labels:
            cavity_atoms_with_labels[x_idx] = []
        cavity_atoms_with_labels[x_idx].append(img_label)
    
    for a_idx in a_site_indices:
        if a_idx in molecule_atom_data:
            _, img_label = molecule_atom_data[a_idx]
            cavity_atoms_with_labels[a_idx] = [img_label]
    
    # Create edges between atoms that are bonded in the main graph
    for atom_i in cavity_atoms_with_labels:
        for atom_j in cavity_atoms_with_labels:
            if atom_i >= atom_j:
                continue
            
            # Check if bond exists in main graph (between atom indices)
            if graph.has_edge(f'atom_{atom_i}', f'atom_{atom_j}'):
                edge_data = graph.get_edge_data(f'atom_{atom_i}', f'atom_{atom_j}')
                
                # Create edge between first image instance of each atom
                # This preserves the B→3X and X→2B connectivity
                img_label_i = cavity_atoms_with_labels[atom_i][0]
                img_label_j = cavity_atoms_with_labels[atom_j][0]
                
                node_i = atom_node_mapping[(atom_i, img_label_i)]
                node_j = atom_node_mapping[(atom_j, img_label_j)]
                
                subgraph.add_edge(
                    node_i,
                    node_j,
                    **edge_data
                )
    
    # Add cavity metadata to subgraph with full octahedra information
    octahedra_metadata = {}
    for oct_idx in all_cavity_octs:
        if oct_idx in octahedra_data:
            oct_data = octahedra_data[oct_idx]
            b_idx = oct_data.get('central_atom')
            octahedra_metadata[oct_idx] = {
                'b_atom_index': b_idx,
                'b_atom_symbol': atom_symbols[b_idx] if b_idx is not None else 'B',
                'intralayer_atoms': oct_data.get('intralayer_atoms', []),
                'interlayer_atoms': oct_data.get('interlayer_atoms', []),
                'terminal_atoms': oct_data.get('terminal_atoms', []),
                'order': oct_data.get('order', 1),
            }
    
    # Extract unique atom indices from data lists
    b_atom_indices = list(set(b_idx for b_idx, _, _ in b_data_list))
    x_atom_indices = list(set(x_idx for x_idx, _, _ in x_data_list))
    
    subgraph.graph['cavity_metadata'] = {
        'octahedra_indices': all_cavity_octs,
        'x_atom_indices': x_atom_indices,
        'b_atom_indices': b_atom_indices,
        'a_site_indices': a_site_indices,
        'center_position': center.tolist(),
        'octahedra_metadata': octahedra_metadata,
        'is_pbc_wrapped': cavity_data.get('is_pbc_wrapped', False),
        'contains_a_site': len(a_site_indices) > 0,
    }
    
    return subgraph


def _count_nh3_groups_from_graph(
    molecule_node: str,
    graph: nx.Graph,
    atom_symbols: List[str],
) -> int:
    """Count NH3 groups in a molecule using graph structure.
    
    This function uses the graph connectivity to identify NH3 groups by finding
    N atoms that have exactly 3 H neighbors connected via 'bonded_to' edges in the graph.
    This is more reliable than geometric distance calculations.
    
    Parameters
    ----------
    molecule_node : str
        The molecule node ID in the graph (e.g., 'molecule_0')
    graph : nx.Graph
        The structural graph
    atom_symbols : list
        All atom symbols
    
    Returns
    -------
    int
        Number of NH3 groups found in the molecule
    """
    nh3_count = 0
    
    # Get all atom nodes that belong to this molecule
    molecule_atom_nodes = []
    for neighbor in graph.neighbors(molecule_node):
        edge_data = graph.get_edge_data(molecule_node, neighbor)
        if edge_data and edge_data.get('edge_type') == 'contains':
            neighbor_data = graph.nodes.get(neighbor, {})
            if neighbor_data.get('node_type') == 'atom':
                atom_idx = neighbor_data.get('vasp_index')
                if atom_idx is not None:
                    molecule_atom_nodes.append((neighbor, atom_idx))
    
    # Create a set of molecule atom node IDs for quick lookup
    molecule_atom_node_ids = {node_id for node_id, _ in molecule_atom_nodes}
    
    # Find all N atoms in the molecule
    n_atom_nodes = []
    for node_id, atom_idx in molecule_atom_nodes:
        if atom_symbols[atom_idx] == 'N':
            n_atom_nodes.append((node_id, atom_idx))
    
    # For each N atom, check if it has 3 H neighbors connected via bonds in the graph
    for n_node_id, n_atom_idx in n_atom_nodes:
        h_neighbor_count = 0
        
        # Check all neighbors of this N atom in the graph
        for neighbor_node in graph.neighbors(n_node_id):
            # Only consider neighbors that are in the same molecule
            if neighbor_node not in molecule_atom_node_ids:
                continue
            
            neighbor_data = graph.nodes.get(neighbor_node, {})
            if neighbor_data.get('node_type') == 'atom':
                neighbor_atom_idx = neighbor_data.get('vasp_index')
                if neighbor_atom_idx is not None and atom_symbols[neighbor_atom_idx] == 'H':
                    # Check if there's a 'bonded_to' edge between N and H
                    edge_data = graph.get_edge_data(n_node_id, neighbor_node)
                    if edge_data and edge_data.get('edge_type') == 'bonded_to':
                        # There's a bond, count it as a neighbor
                        h_neighbor_count += 1
        
        # If N has exactly 3 H neighbors, it's an NH3 group
        if h_neighbor_count == 3:
            nh3_count += 1
    
    return nh3_count


def _calculate_nh3_center(
    molecule_indices: List[int],
    atom_positions: np.ndarray,
    atom_symbols: List[str],
) -> List[np.ndarray]:
    """Calculate centers of mass for all NH3 groups in a molecule.
    
    Parameters
    ----------
    molecule_indices : list
        Indices of atoms in the molecule
    atom_positions : np.ndarray
        All atom positions in the structure
    atom_symbols : list
        All atom symbols
    
    Returns
    -------
    list of np.ndarray
        List of NH3 center of mass positions (N + 3H) / 4
        Returns empty list if no NH3 groups found
    """
    nh3_centers = []
    
    # Extract molecule atoms
    mol_symbols = [atom_symbols[idx] for idx in molecule_indices]
    mol_positions = atom_positions[molecule_indices]
    
    # Find all nitrogen atoms in molecule
    n_indices_in_mol = [i for i, sym in enumerate(mol_symbols) if sym == 'N']
    
    if not n_indices_in_mol:
        return nh3_centers
    
    # For each N, find if it's part of an NH3 group (3 H neighbors within 1.2 Å)
    nh_bond_cutoff = 1.2
    
    for n_idx_in_mol in n_indices_in_mol:
        n_idx_global = molecule_indices[n_idx_in_mol]
        n_pos = atom_positions[n_idx_global]
        
        # Find H atoms bonded to this N within the molecule
        h_indices_in_mol = []
        for i, sym in enumerate(mol_symbols):
            if sym == 'H':
                h_idx_global = molecule_indices[i]
                h_pos = atom_positions[h_idx_global]
                distance = np.linalg.norm(h_pos - n_pos)
                if distance < nh_bond_cutoff:
                    h_indices_in_mol.append(i)
        
        # If 3 H atoms nearby, it's an NH3 group
        if len(h_indices_in_mol) == 3:
            h_positions = [atom_positions[molecule_indices[i]] for i in h_indices_in_mol]
            # Center of mass: (N + 3H) / 4
            center = (n_pos + np.sum(h_positions, axis=0)) / 4.0
            nh3_centers.append(center)
    
    return nh3_centers


def _calculate_penetration_depth(membrane_points: np.ndarray, query_point: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Calculate signed distance (penetration) of a query point relative to 
    the best-fit plane of 4 membrane points using SVD.
    
    The plane is fitted to the 4 membrane points using singular value decomposition.
    The normal vector is the singular vector corresponding to the smallest singular value.
    
    Parameters
    ----------
    membrane_points : np.ndarray
        4x3 array of terminal atom coordinates (x, y, z)
    query_point : np.ndarray
        1x3 array of the NH3 center position
    
    Returns
    -------
    tuple
        (signed_distance, plane_centroid, plane_normal)
        - signed_distance (float): Signed distance. Negative usually implies penetration.
        - plane_centroid (np.ndarray): 3D center of the 4 membrane points
        - plane_normal (np.ndarray): 3D unit normal vector (pointing upward, normal[2] > 0)
    """
    if membrane_points.shape != (4, 3):
        raise ValueError(f"Expected 4x3 array of membrane points, got {membrane_points.shape}")
    
    # 1. Calculate centroid of the 4 points
    centroid = np.mean(membrane_points, axis=0)
    
    # 2. Center the points (remove translation)
    centered_points = membrane_points - centroid
    
    # 3. Compute surface normal using SVD
    # The normal is the singular vector corresponding to the SMALLEST singular value
    # (i.e., the direction with the least variance)
    u, s, vh = np.linalg.svd(centered_points, full_matrices=True)
    normal = vh[-1, :]  # The last row of Vh corresponds to the smallest singular value
    
    # 4. Enforce consistent orientation (upward-pointing normal)
    # If normal.z < 0, flip it so normal[2] > 0
    if normal[2] < 0:
        normal = -normal
    
    # Normalize to unit vector
    normal = normal / np.linalg.norm(normal)
    
    # 5. Calculate signed distance (projection)
    # Vector from centroid to query point
    vec_to_point = query_point - centroid
    
    # Project this vector onto the normal
    signed_distance = np.dot(vec_to_point, normal)
    
    return float(signed_distance), centroid, normal


def _find_cavity_for_spacer(
    molecule_indices: List[int],
    spacer_type: str,
    octahedra_data: Dict[int, Dict[str, Any]],
    oct_to_layer: Dict[int, int],
    neighbor_map: Dict[int, Dict[int, List[int]]],
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell: np.ndarray,
    cell_lengths: np.ndarray,
) -> List[Optional[Dict[str, Any]]]:
    """Find cavities enclosing a spacer molecule.
    
    For RP spacers: returns 1 cavity (4B+8X around single NH3 center)
    For DJ spacers: returns 2 cavities (4B+8X around each NH3 center)
    
    Parameters
    ----------
    molecule_indices : list
        Indices of atoms in the spacer molecule
    spacer_type : str
        Type of spacer: 'rp' or 'dj'
    octahedra_data : dict
        Octahedra data
    oct_to_layer : dict
        Octahedra layer mapping
    neighbor_map : dict
        Octahedra neighbor mapping
    atom_positions : np.ndarray
        Atom positions
    atom_symbols : list
        Atom symbols
    cell : np.ndarray
        Unit cell
    cell_lengths : np.ndarray
        Cell lengths
        
    Returns
    -------
    list of dict or None
        Cavity data dictionaries for each cavity
    """
    if not molecule_indices or not octahedra_data:
        return []
    
    # Calculate NH3 centers
    nh3_centers = _calculate_nh3_center(molecule_indices, atom_positions, atom_symbols)
    
    if not nh3_centers:
        print(f"    No NH3 centers found for spacer molecule", file=sys.stderr)
        return []
    
    # If spacer_type is unknown, infer from number of NH3 centers
    if spacer_type.lower() == 'unknown':
        if len(nh3_centers) >= 2:
            spacer_type = 'dj'
            print(f"    Inferred spacer type as 'dj' based on {len(nh3_centers)} NH3 centers", file=sys.stderr)
        else:
            spacer_type = 'rp'
            print(f"    Inferred spacer type as 'rp' based on {len(nh3_centers)} NH3 centers", file=sys.stderr)
    
    cavities_data = []
    
    # Calculate geometric center of the entire molecule
    # This is the reference point for finding the cavity around the spacer
    mol_atom_positions = np.array([atom_positions[idx] for idx in molecule_indices])
    mol_center = np.mean(mol_atom_positions, axis=0)
    
    # For RP spacer: one cavity
    # For DJ spacer: one cavity (using entire molecule center, not individual NH3 centers)
    cavity_center_to_use = mol_center
    
    # Set cavity sizes based on spacer type
    # RP spacer (between 1 layer and boundary): 4B + 8X
    # DJ spacer (between 2 layers): 8B + 16X
    if spacer_type.lower() == 'rp':
        n_b_atoms = 4
        n_x_atoms = 8
    else:  # DJ
        n_b_atoms = 8
        n_x_atoms = 16
    
    print(f"    Processing {spacer_type.upper()} spacer - using molecule geometric center as cavity reference", file=sys.stderr)
    print(f"    Looking for {n_b_atoms} B atoms and {n_x_atoms} X atoms", file=sys.stderr)
    
    # Single loop iteration for both RP and DJ (one cavity per spacer molecule)
    for loop_idx in range(1):  # Always just one cavity per spacer
        cavity_type = 'spacer_rp' if spacer_type.lower() == 'rp' else 'spacer_dj'
        
        # Find nearest B atoms around molecule center
        from q2D_Materials.utils.geometry.pbc_distances import find_nearest_image_positions
        
        # Collect all B atom positions
        b_positions = []
        b_indices = []
        b_oct_indices = []
        
        for oct_idx, oct_data in octahedra_data.items():
            b_idx = oct_data.get('central_atom')
            if b_idx is None:
                continue
            b_positions.append(atom_positions[b_idx])
            b_indices.append(b_idx)
            b_oct_indices.append(oct_idx)
        
        if len(b_positions) == 0:
            continue
        
        b_positions = np.array(b_positions)
        b_indices = np.array(b_indices)
        
        # Find nearest B positions from 27-image supercell
        nearest_b_indices, nearest_b_positions, nearest_b_distances, nearest_b_labels = \
            find_nearest_image_positions(
                cavity_center_to_use, b_positions, b_indices, cell, n_neighbors=n_b_atoms,
                exclude_indices=np.array(molecule_indices)
            )
        
        if len(nearest_b_indices) < n_b_atoms:
            print(f"    Not enough B positions for {cavity_type} cavity ({len(nearest_b_indices)} < {n_b_atoms})", file=sys.stderr)
            continue
        
        # Map B indices back to octahedron indices
        b_idx_to_oct = {b_indices[i]: b_oct_indices[i] for i in range(len(b_indices))}
        nearest_b_octs = [b_idx_to_oct[b_idx] for b_idx in nearest_b_indices]
        
        # Find nearest X positions
        x_positions = []
        x_indices = []
        
        for atom_idx, symbol in enumerate(atom_symbols):
            if symbol not in ['Cl', 'Br', 'I', 'F', 'O']:
                continue
            x_positions.append(atom_positions[atom_idx])
            x_indices.append(atom_idx)
        
        if len(x_positions) == 0:
            continue
        
        x_positions = np.array(x_positions)
        x_indices = np.array(x_indices)
        
        # Find nearest X positions
        nearest_x_indices, nearest_x_positions, nearest_x_distances, nearest_x_labels = \
            find_nearest_image_positions(
                cavity_center_to_use, x_positions, x_indices, cell, n_neighbors=n_x_atoms,
                exclude_indices=np.array(molecule_indices)
            )
        
        if len(nearest_x_indices) < n_x_atoms:
            print(f"    Not enough X positions for {cavity_type} cavity ({len(nearest_x_indices)} < {n_x_atoms})", file=sys.stderr)
            continue
        
        # Build cavity data
        b_data_list = []
        for i in range(n_b_atoms):
            b_data_list.append((
                nearest_b_indices[i],
                nearest_b_positions[i],
                tuple(nearest_b_labels[i])
            ))
        
        x_data_list = []
        for i in range(n_x_atoms):
            x_data_list.append((
                nearest_x_indices[i],
                nearest_x_positions[i],
                tuple(nearest_x_labels[i])
            ))
        
        # Cavity center from B positions
        center = np.mean(nearest_b_positions, axis=0)
        
        octahedra_info = {}
        for i, oct_idx in enumerate(nearest_b_octs):
            if oct_idx in octahedra_data:
                b_idx = nearest_b_indices[i]
                octahedra_info[b_idx] = {
                    'octahedron_index': oct_idx,
                    'octahedra_data': octahedra_data[oct_idx],
                }
        
        cavity_data = {
            'octahedra_indices': nearest_b_octs,
            'b_data_list': b_data_list,
            'x_data_list': x_data_list,
            'octahedra_info': octahedra_info,
            'a_site_indices': molecule_indices,
            'a_site_formulas': [],
            'a_site_types': [],
            'center_position': center.tolist(),
            'cavity_type': cavity_type,
            'nh3_center_index': 0,
            'closed': False,  # Spacer cavities are open
        }
        
        cavities_data.append(cavity_data)
        print(f"  Found {cavity_type} cavity for spacer molecule ({n_b_atoms}B + {n_x_atoms}X)", file=sys.stderr)
    
    return cavities_data


def _find_cavity_for_molecule(
    molecule_indices: List[int],
    octahedra_data: Dict[int, Dict[str, Any]],
    oct_to_layer: Dict[int, int],
    neighbor_map: Dict[int, Dict[int, List[int]]],
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell: np.ndarray,
    cell_lengths: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Find cavity enclosing a given A-site molecule by finding 8 nearest B atoms.
    
    A-site molecules always use closed cage geometry: 8B + 12X.
    No validation needed - A-sites go directly to closed cage pipeline.
    
    Algorithm:
    1. Calculate molecule center (3D)
    2. Find 8 nearest B atoms using 27-image PBC (3×3×3 unit cells)
    3. Find 12 X atoms closest to molecule center
    4. Validate topology: 8 B positions form 2 layers (4 per layer)
    
    Parameters
    ----------
    molecule_indices : list
        Indices of atoms in the A-site molecule
    octahedra_data : dict
        Octahedra data
    oct_to_layer : dict
        Octahedra layer mapping
    neighbor_map : dict
        Octahedra neighbor mapping
    atom_positions : np.ndarray
        Atom positions
    atom_symbols : list
        Atom symbols
    cell : np.ndarray
        Unit cell
    cell_lengths : np.ndarray
        Cell lengths
        
    Returns
    -------
    dict or None
        Cavity data dictionary or None if not found
    """
    if not molecule_indices or not octahedra_data:
        return None
    
    # Calculate molecule center (3D, using PBC-aware unwrapping)
    ref_pos = atom_positions[molecule_indices[0]]
    mol_positions = [ref_pos]
    for mol_idx in molecule_indices[1:]:
        unwrapped = unwrap_relative_coordinate(
            ref_pos, atom_positions[mol_idx], cell_lengths
        )
        mol_positions.append(unwrapped)
    
    mol_center = np.mean(mol_positions, axis=0)
    
    print(f"    Molecule center: {mol_center}", file=sys.stderr)
    
    from q2D_Materials.utils.geometry.pbc_distances import find_nearest_image_positions
    
    # Step 1: Find 8 nearest B atom Cartesian positions to molecule center
    # We work in a 27-image supercell (3×3×3 unit cells). Each atom index can
    # appear up to 27 times with different Cartesian coordinates across the images.
    # We select 8 unique Cartesian B positions (which may correspond to fewer unique indices).
    
    # Collect all B atom positions and their octahedron indices
    b_positions = []
    b_indices = []
    b_oct_indices = []
    
    for oct_idx, oct_data in octahedra_data.items():
        b_idx = oct_data.get('central_atom')
        if b_idx is None:
            continue
        b_positions.append(atom_positions[b_idx])
        b_indices.append(b_idx)
        b_oct_indices.append(oct_idx)
    
    if len(b_positions) == 0:
        print(f"    No B atoms found in octahedra data", file=sys.stderr)
        return None
    
    b_positions = np.array(b_positions)
    b_indices = np.array(b_indices)
    
    # Find 8 nearest B positions from 27-image supercell (excluding molecule atoms)
    nearest_b_indices, nearest_b_positions, nearest_b_distances, nearest_b_labels = \
        find_nearest_image_positions(
            mol_center, b_positions, b_indices, cell, n_neighbors=8,
            exclude_indices=np.array(molecule_indices)
        )
    
    if len(nearest_b_indices) < 8:
        print(f"    Not enough B positions in 27-image supercell ({len(nearest_b_indices)} < 8)", file=sys.stderr)
        return None
    
    # Map B indices back to octahedron indices
    b_idx_to_oct = {b_indices[i]: b_oct_indices[i] for i in range(len(b_indices))}
    nearest_8_octs = [b_idx_to_oct[b_idx] for b_idx in nearest_b_indices]
    
    print(f"    Found 8 nearest B positions with distances: {[f'{d:.3f}' for d in nearest_b_distances]}", 
          file=sys.stderr)
    
    # Validate topology: 8 B positions should form 2 layers (4 per layer)
    # Group B positions by Z coordinate (with tolerance for layering)
    z_coords = nearest_b_positions[:, 2]
    z_min, z_max = z_coords.min(), z_coords.max()
    z_tolerance = 1.0  # Ångströms
    
    layer1_mask = np.abs(z_coords - z_coords.min()) < z_tolerance
    layer2_mask = np.abs(z_coords - z_coords.max()) < z_tolerance
    
    n_layer1 = layer1_mask.sum()
    n_layer2 = layer2_mask.sum()
    
    # Require 4 B positions in each layer
    if n_layer1 != 4 or n_layer2 != 4:
        print(f"    Topology invalid: {n_layer1} B in layer 1, {n_layer2} in layer 2 (need 4+4)",
              file=sys.stderr)
        return None
    
    print(f"    ✓ Topology check: {n_layer1} B atoms in layer 1, {n_layer2} in layer 2", file=sys.stderr)
    
    # Step 2: Calculate cavity center from the 8 B atoms
    cavity_center = np.mean(nearest_b_positions, axis=0)
    
    # Step 3: Find X atom Cartesian positions closest to MOLECULE center
    # (same reference as the 8 B atoms were found)
    # A-site molecules always use closed cage: 8B + 12X
    n_x_needed = 12
    
    # Collect all X atom positions
    x_positions = []
    x_indices = []
    
    for atom_idx, symbol in enumerate(atom_symbols):
        # Skip non-X atoms
        if symbol not in ['Cl', 'Br', 'I', 'F', 'O']:  # Common X-site elements
            continue
        x_positions.append(atom_positions[atom_idx])
        x_indices.append(atom_idx)
    
    if len(x_positions) == 0:
        print(f"    No X atoms found in structure", file=sys.stderr)
        return None
    
    x_positions = np.array(x_positions)
    x_indices = np.array(x_indices)
    
    # Find exactly n_x_needed nearest X positions to molecule center (same reference as B search)
    # Exclude molecule atoms from X search
    nearest_x_indices, nearest_x_positions, nearest_x_distances, nearest_x_labels = \
        find_nearest_image_positions(
            mol_center, x_positions, x_indices, cell, n_neighbors=n_x_needed,
            exclude_indices=np.array(molecule_indices)
        )
    
    if len(nearest_x_indices) < n_x_needed:
        print(f"    Not enough X positions in 27-image supercell ({len(nearest_x_indices)} < {n_x_needed})", file=sys.stderr)
        return None
    
    print(f"    Selected {n_x_needed} nearest X positions with distances: {[f'{d:.3f}' for d in nearest_x_distances]}", 
          file=sys.stderr)
    
    # Step 5: Build cavity data with Cartesian PBC coordinates and image labels
    # Store the actual Cartesian positions we selected from the 27-image supercell
    
    # Store 8 B positions with their image labels
    b_data_list = []  # List of (atom_idx, cartesian_pos, image_label)
    for i in range(8):
        b_data_list.append((
            nearest_b_indices[i],
            nearest_b_positions[i],
            tuple(nearest_b_labels[i])  # Convert to tuple for hashing
        ))
    
    # Store X positions with their image labels
    x_data_list = []  # List of (atom_idx, cartesian_pos, image_label)
    for i in range(n_x_needed):
        x_data_list.append((
            nearest_x_indices[i],
            nearest_x_positions[i],
            tuple(nearest_x_labels[i])  # Convert to tuple for hashing
        ))
    
    # Final cavity center from 8 B Cartesian positions
    center = np.mean(nearest_b_positions, axis=0)
    
    # Build octahedra info for each cavity B atom
    octahedra_info = {}
    for i, oct_idx in enumerate(nearest_8_octs):
        if oct_idx in octahedra_data:
            b_idx = nearest_b_indices[i]
            octahedra_info[b_idx] = {
                'octahedron_index': oct_idx,
                'octahedra_data': octahedra_data[oct_idx],
            }
    
    return {
        'octahedra_indices': nearest_8_octs,  # 8 octahedron indices
        'b_data_list': b_data_list,  # List of (atom_idx, cartesian_pos, image_label)
        'x_data_list': x_data_list,  # List of (atom_idx, cartesian_pos, image_label)
        'octahedra_info': octahedra_info,  # Mapping: b_idx -> octahedra data
        'a_site_indices': molecule_indices,
        'a_site_formulas': [],  # Will be filled by assign_a_sites_to_cavities
        'a_site_types': [],  # Will be filled by assign_a_sites_to_cavities
        'center_position': center.tolist(),
        'cavity_type': 'a_site',  # Mark as A-site cavity
        'closed': True,
    }


def _detect_all_cavities(
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell: np.ndarray,
    neighbor_indices: List[List[int]],
) -> List['Cavity']:
    """Detect all cavities in the structure using molecule-anchored search.
    
    NEW ALGORITHM: For each A-site molecule, systematically search for enclosing
    octahedra using generalized m1×m2×n grid-based cascade with PBC handling.
    
    Each valid cavity contains exactly 1 A-site molecule (anchor) and is isolated
    with PBC-unwrapped coordinates in a subgraph.
    
    Parameters
    ----------
    graph : nx.Graph
        The structural graph
    atom_positions : np.ndarray
        Atom positions
    atom_symbols : list
        Atom symbols
    cell : np.ndarray
        Unit cell matrix
    neighbor_indices : list
        List of neighbor indices for each octahedron
        
    Returns
    -------
    list of Cavity
        List of Cavity objects with all metadata
    """
    from .cavity_class import Cavity
    from .octahedral_detection import find_shared_atoms
    
    cell_lengths = get_cell_lengths(cell)
    octahedra_data = get_octahedra_data(graph)
    oct_to_layer = _get_layer_assignments(graph)
    
    if not octahedra_data:
        return []
    
    # Build shared atoms map for neighbor connectivity
    shared_atoms = find_shared_atoms(neighbor_indices)
    neighbor_map = _build_octahedra_neighbor_map(octahedra_data, shared_atoms)
    
    # Find all A-site molecules using Molecule nodes
    a_site_molecules = []
    
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'molecule' and data.get('molecule_type') == 'a_site':
            # Get atoms in this molecule via CONTAINS edges
            molecule_atoms = []
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'contains':
                    neighbor_data = graph.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'atom':
                        atom_idx = neighbor_data.get('vasp_index')
                        if atom_idx is not None:
                            molecule_atoms.append(atom_idx)
            
            if molecule_atoms:
                a_site_molecules.append(molecule_atoms)
    
    print(f"  Found {len(a_site_molecules)} A-site molecules", file=sys.stderr)
    
    # Find all spacer molecules using Molecule nodes and classify by NH3 count
    spacer_molecules = []
    
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'molecule' and data.get('molecule_type') == 'spacer':
            # Get atoms in this molecule via CONTAINS edges
            molecule_atoms = []
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'contains':
                    neighbor_data = graph.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'atom':
                        atom_idx = neighbor_data.get('vasp_index')
                        if atom_idx is not None:
                            molecule_atoms.append(atom_idx)
            
            if not molecule_atoms:
                continue
            
            # Count NH3 groups using graph structure (more reliable than geometric)
            nh3_count = _count_nh3_groups_from_graph(node, graph, atom_symbols)
            
            # Fallback to geometric method if graph-based count is 0
            if nh3_count == 0:
                nh3_centers = _calculate_nh3_center(molecule_atoms, atom_positions, atom_symbols)
                nh3_count = len(nh3_centers)
                if nh3_count > 0:
                    print(f"  Spacer molecule {molecule_atoms[:3]}...: graph-based count failed, using geometric method ({nh3_count} NH3)", file=sys.stderr)
            
            # Classify: 2+ NH3 = DJ, 1 NH3 = RP
            if nh3_count >= 2:
                spacer_type = 'dj'
            elif nh3_count == 1:
                spacer_type = 'rp'
            else:
                # No NH3 groups found, default to 'rp'
                print(f"  WARNING: Spacer molecule {molecule_atoms[:3]}... has no NH3 groups, defaulting to 'rp'", file=sys.stderr)
                spacer_type = 'rp'  # Default fallback
            
            spacer_molecules.append((molecule_atoms, spacer_type))
            print(f"  Spacer molecule {molecule_atoms[:3]}...: {nh3_count} NH3 groups → {spacer_type}", file=sys.stderr)
    
    print(f"  Found {len(spacer_molecules)} spacer molecules", file=sys.stderr)
    total_molecules = len(a_site_molecules) + len(spacer_molecules)
    print(f"  Total molecules to process: {total_molecules} ({len(a_site_molecules)} A-site + {len(spacer_molecules)} spacer)", file=sys.stderr)
    
    cavities = []
    cavity_centers = []
    found_cavities = 0
    
    # Search for cavity around each A-site molecule
    # A-site molecules: no validation needed, always use closed cage (8B + 12X)
    for mol_idx, molecule_indices in enumerate(a_site_molecules):
        # Find cavity for this molecule
        cavity_data = _find_cavity_for_molecule(
            molecule_indices,
            octahedra_data,
            oct_to_layer,
            neighbor_map,
            atom_positions,
            atom_symbols,
            cell,
            cell_lengths,
        )
        
        if cavity_data is None:
            print(f"  Could not find cavity for molecule {mol_idx} with atoms {molecule_indices}",
                  file=sys.stderr)
            continue
        
        # Validate cavity has correct Cartesian position counts
        # A-site cavities: always 8 B + 12 X (closed cage)
        # Spacer cavities: 4 B + 8 X (half cuboctahedral)
        cavity_type = cavity_data.get('cavity_type', 'a_site')
        all_b = len(cavity_data.get('b_data_list', []))
        all_x = len(cavity_data.get('x_data_list', []))
        
        # Validate based on cavity type
        if cavity_type == 'a_site':
            # A-site molecules always use closed cage: 8B + 12X
            if all_b != 8 or all_x != 12:
                print(f"  Cavity for molecule {mol_idx}: invalid size ({all_b} B, {all_x} X, need 8 B and 12 X for a_site)",
                      file=sys.stderr)
                continue
        elif cavity_type in ('spacer_rp', 'spacer_dj'):
            if all_b != 4 or all_x != 8:
                print(f"  Cavity for molecule {mol_idx}: invalid size ({all_b} B, {all_x} X, need 4 B and 8 X for spacer)",
                      file=sys.stderr)
                continue
        else:
            print(f"  Unknown cavity type: {cavity_type}", file=sys.stderr)
            continue
        
        # Build isolated subgraph with PBC coordinates
        cavity_subgraph = _build_cavity_subgraph(
            cavity_data, octahedra_data, graph, atom_positions, atom_symbols, cell, cell_lengths
        )
        
        cavity_data['subgraph'] = cavity_subgraph
        
        # Compute hull data
        hull_data = _compute_cavity_hull_data(
            cavity_data, octahedra_data, atom_positions, cell
        )
        
        cavity_data['is_pbc_wrapped'] = _check_pbc_wrapping(
            set(cavity_data.get('octahedra_indices', [])),
            octahedra_data, atom_positions, cell_lengths
        )
        
        # Extract B and X atom indices from the data lists
        b_atom_indices = [b_idx for b_idx, _, _ in cavity_data.get('b_data_list', [])]
        x_atom_indices = [x_idx for x_idx, _, _ in cavity_data.get('x_data_list', [])]
        
        # Create Cavity object
        cavity = Cavity(
            cavity_id=f'cavity_{found_cavities}',
            b_atom_indices=b_atom_indices,
            x_atom_indices=x_atom_indices,
            a_site_indices=cavity_data.get('a_site_indices', []),
            pbc_coordinates=cavity_data.get('pbc_coordinates', {}),
            subgraph=cavity_subgraph,
            center_position=np.array(cavity_data.get('center_position', [0, 0, 0])),
            octahedra_info=cavity_data.get('octahedra_info', {}),
            contains_a_site=len(cavity_data.get('a_site_indices', [])) > 0,
            is_pbc_wrapped=cavity_data.get('is_pbc_wrapped', False),
            hull_data=hull_data,
            cavity_type=cavity_data.get('cavity_type', 'a_site'),
        )
        
        cavities.append(cavity)
        cavity_center = np.array(cavity_data['center_position'])
        cavity_centers.append(cavity_center)
        found_cavities += 1
        
        print(f"  Found cavity {found_cavities} for molecule {mol_idx}", file=sys.stderr)
    
    # Search for cavities around each spacer molecule
    # Spacer type already determined by NH3 count (2+ = DJ, 1 = RP)
    for mol_idx, (molecule_indices, spacer_type) in enumerate(spacer_molecules):
        # Find cavities for this spacer (may be 1 for RP or 2 for DJ)
        print(f"  Processing spacer molecule {mol_idx} with {len(molecule_indices)} atoms, type: {spacer_type}", file=sys.stderr)
        cavities_data_list = _find_cavity_for_spacer(
            molecule_indices,
            spacer_type,
            octahedra_data,
            oct_to_layer,
            neighbor_map,
            atom_positions,
            atom_symbols,
            cell,
            cell_lengths,
        )
        
        if not cavities_data_list:
            print(f"  WARNING: No cavities found for spacer molecule {mol_idx} (type: {spacer_type})", file=sys.stderr)
        
        for cavity_idx, cavity_data in enumerate(cavities_data_list):
            if cavity_data is None:
                continue
            
            # Validate cavity has correct size
            cavity_type = cavity_data.get('cavity_type', 'spacer_rp')
            all_b = len(cavity_data.get('b_data_list', []))
            all_x = len(cavity_data.get('x_data_list', []))
            
            if cavity_type == 'spacer_rp':
                # RP spacer: 4B + 8X
                if all_b != 4 or all_x != 8:
                    print(f"  Spacer cavity for molecule {mol_idx}: invalid size ({all_b} B, {all_x} X, need 4 B and 8 X for spacer_rp)",
                          file=sys.stderr)
                    continue
            elif cavity_type == 'spacer_dj':
                # DJ spacer: 8B + 16X (spans between 2 layers)
                if all_b != 8 or all_x != 16:
                    print(f"  Spacer cavity for molecule {mol_idx}: invalid size ({all_b} B, {all_x} X, need 8 B and 16 X for spacer_dj)",
                          file=sys.stderr)
                    continue
            else:
                print(f"  Unknown spacer cavity type: {cavity_type}", file=sys.stderr)
                continue
            
            # Build isolated subgraph with PBC coordinates
            cavity_subgraph = _build_cavity_subgraph(
                cavity_data, octahedra_data, graph, atom_positions, atom_symbols, cell, cell_lengths
            )
            
            cavity_data['subgraph'] = cavity_subgraph
            
            # Compute hull data
            hull_data = _compute_cavity_hull_data(
                cavity_data, octahedra_data, atom_positions, cell
            )
            
            cavity_data['is_pbc_wrapped'] = _check_pbc_wrapping(
                set(cavity_data.get('octahedra_indices', [])),
                octahedra_data, atom_positions, cell_lengths
            )
            
            # Create Cavity object
            cavity = Cavity(
                cavity_id=f'cavity_{found_cavities}',
                b_atom_indices=cavity_data.get('b_atom_indices', []),
                x_atom_indices=cavity_data.get('x_atom_indices', []),
                a_site_indices=cavity_data.get('a_site_indices', []),
                pbc_coordinates=cavity_data.get('pbc_coordinates', {}),
                subgraph=cavity_subgraph,
                center_position=np.array(cavity_data.get('center_position', [0, 0, 0])),
                octahedra_info=cavity_data.get('octahedra_info', {}),
                contains_a_site=len(cavity_data.get('a_site_indices', [])) > 0,
                is_pbc_wrapped=cavity_data.get('is_pbc_wrapped', False),
                hull_data=hull_data,
                cavity_type=cavity_data.get('cavity_type', 'spacer_rp'),
            )
            
            cavities.append(cavity)
            cavity_center = np.array(cavity_data['center_position'])
            cavity_centers.append(cavity_center)
            found_cavities += 1
            
            nh3_idx = cavity_data.get('nh3_center_index', 0)
            print(f"  Found {cavity_type} cavity {found_cavities} for spacer molecule {mol_idx}, cavity {cavity_idx+1}/{len(cavities_data_list)} (NH3 {nh3_idx})", file=sys.stderr)
    
    print(f"  Retained {len(cavities)} cavities with PBC subgraphs", file=sys.stderr)
    print(f"  Summary: {len(a_site_molecules)} A-site molecules processed, {len(spacer_molecules)} spacer molecules processed, {len(cavities)} total cavities created", file=sys.stderr)
    
    # Assign any remaining A-sites to cavities (fallback) - convert cavities back to dicts for this function
    cavity_dicts = [
        {
            'octahedra_indices': cav.b_atom_indices,
            'x_atom_indices': cav.x_atom_indices,
            'b_atom_indices': cav.b_atom_indices,
            'a_site_indices': cav.a_site_indices,
            'a_site_formulas': [],  # Will be filled by assign_a_sites_to_cavities
            'a_site_types': [],  # Will be filled by assign_a_sites_to_cavities
            'pbc_coordinates': cav.pbc_coordinates,
            'center_position': cav.center_position.tolist(),
            'octahedra_info': cav.octahedra_info,
        }
        for cav in cavities
    ]
    
    cavity_dicts = assign_a_sites_to_cavities(
        cavity_dicts, graph, atom_positions, atom_symbols, cell_lengths, cavity_centers
    )
    
    # Update cavities with assigned A-sites
    for cav, cav_dict in zip(cavities, cavity_dicts):
        cav.a_site_indices = cav_dict.get('a_site_indices', [])
        cav.contains_a_site = len(cav_dict.get('a_site_indices', [])) > 0
    
    return cavities


def _check_pbc_wrapping(
    octahedra_indices: Set[int],
    octahedra_data: Dict[int, Dict[str, Any]],
    atom_positions: np.ndarray,
    cell_lengths: np.ndarray,
) -> bool:
    """Check if a cavity spans periodic boundary.
    
    Parameters
    ----------
    octahedra_indices : set
        Indices of octahedra in the cavity
    octahedra_data : dict
        Octahedra data
    atom_positions : np.ndarray
        Atom positions
    cell_lengths : np.ndarray
        Cell lengths
        
    Returns
    -------
    bool
        True if cavity spans cell boundary
    """
    if len(octahedra_indices) < 2:
        return False
    
    # Get B-site positions
    positions = []
    for oct_idx in octahedra_indices:
        if oct_idx in octahedra_data:
            b_idx = octahedra_data[oct_idx]['central_atom']
            positions.append(atom_positions[b_idx])
    
    if len(positions) < 2:
        return False
    
    positions = np.array(positions)
    
    # Check if any coordinate spans more than half the cell
    for dim in range(3):
        coord_range = positions[:, dim].max() - positions[:, dim].min()
        if coord_range > cell_lengths[dim] * 0.5:
            return True
    
    return False


# Removed add_cavities_to_graph function - cavities are stored in Cavity objects with subgraphs
# No need to modify the principal graph


def calculate_cavity_deformation(
    cavity_data: Dict[str, Any],
    octahedra_data: Dict[int, Dict[str, Any]],
    atom_positions: np.ndarray,
    cell: np.ndarray,
) -> Dict[str, Any]:
    """Calculate cavity deformation from ideal cuboctahedron geometry.
    
    A cuboctahedron has:
    - 8 triangular faces (equilateral)
    - 6 square faces
    - 12 identical vertices where 2 triangles and 2 squares meet
    - Vertex angles: 60° + 60° + 90° + 90° = 300°
    - Dihedral angles:
        * Square-triangle: ~109.47° (tetrahedral angle)
        * Square-square: ~125.26° (in cupola arrangement) or ~70.53° (opposite)
        * Triangle-triangle: ~70.53°
    
    Parameters
    ----------
    cavity_data : dict
        Cavity data with octahedra indices
    octahedra_data : dict
        Octahedra data
    atom_positions : np.ndarray
        Atomic positions
    cell : np.ndarray
        Unit cell matrix
        
    Returns
    -------
    dict
        Deformation metrics:
        - 'vertex_angle_deviations': RMS deviation from ideal vertex angles
        - 'dihedral_deviations': RMS deviation from ideal dihedral angles
        - 'edge_length_variation': Coefficient of variation of edge lengths
        - 'symmetry_score': Overall symmetry metric (0=perfect, 1=highly distorted)
    """
    from ..utils.clifford_embedding import embed_to_6d, clifford_distance
    
    # Get corner X atoms (vertices of cuboctahedron)
    all_corner_atoms = cavity_data.get('x_atom_indices', [])

    
    if len(all_corner_atoms) < 4:
        return {
            'vertex_angle_deviations': None,
            'dihedral_deviations': None,
            'edge_length_variation': None,
            'symmetry_score': None,
            'error': 'Insufficient vertices for deformation analysis'
        }
    
    # Use PBC coordinates if available, otherwise use original positions
    cell_lengths = get_cell_lengths(cell)
    pbc_coords = cavity_data.get('pbc_coordinates', {})
    
    corner_positions_3d = []
    for idx in all_corner_atoms:
        if idx in pbc_coords:
            corner_positions_3d.append(pbc_coords[idx])
        else:
            corner_positions_3d.append(atom_positions[idx])
    
    corner_positions_3d = np.array(corner_positions_3d)
    corner_positions_6d = embed_to_6d(corner_positions_3d, cell_lengths)
    
    # Calculate all edge lengths (distances between corner atoms)
    n_corners = len(all_corner_atoms)
    edge_lengths = []
    for i in range(n_corners):
        for j in range(i + 1, n_corners):
            dist = clifford_distance(corner_positions_6d[i], corner_positions_6d[j])
            edge_lengths.append(dist)
    
    if not edge_lengths:
        return {
            'vertex_angle_deviations': None,
            'dihedral_deviations': None,
            'edge_length_variation': 0.0,
            'symmetry_score': None,
        }
    
    # Edge length variation (coefficient of variation)
    edge_lengths = np.array(edge_lengths)
    edge_mean = np.mean(edge_lengths)
    edge_std = np.std(edge_lengths)
    edge_cv = edge_std / edge_mean if edge_mean > 0 else 0.0
    
    # Calculate vertex angles and compare to ideal (60° and 90°)
    # For simplicity, calculate angles at each vertex from its neighbors
    vertex_angle_deviations = []
    
    # Find connected atoms (within 1.5x median edge length)
    median_edge = np.median(edge_lengths)
    connectivity_threshold = median_edge * 1.5
    
    for i, atom_i_6d in enumerate(corner_positions_6d):
        # Find neighbors of this vertex
        neighbors = []
        for j, atom_j_6d in enumerate(corner_positions_6d):
            if i == j:
                continue
            dist = clifford_distance(atom_i_6d, atom_j_6d)
            if dist < connectivity_threshold:
                neighbors.append((j, atom_j_6d))
        
        # Calculate angles between neighbors at this vertex
        if len(neighbors) >= 2:
            angles = []
            for k in range(len(neighbors)):
                for m in range(k + 1, len(neighbors)):
                    j_idx, j_6d = neighbors[k]
                    m_idx, m_6d = neighbors[m]
                    
                    # Vectors from vertex to neighbors (in 3D for angle calc)
                    # Use first 3 components of 6D embedding
                    vec_j = corner_positions_6d[j_idx][:3] - atom_i_6d[:3]
                    vec_m = corner_positions_6d[m_idx][:3] - atom_i_6d[:3]
                    
                    # Calculate angle
                    cos_angle = np.dot(vec_j, vec_m) / (np.linalg.norm(vec_j) * np.linalg.norm(vec_m) + 1e-10)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    angles.append(angle_deg)
            
            # Compare to ideal angles (60° and 90°)
            for angle in angles:
                # Find closest ideal angle
                ideal_angles = [60.0, 90.0, 120.0]  # Include 120° for some configurations
                deviations = [abs(angle - ideal) for ideal in ideal_angles]
                min_deviation = min(deviations)
                vertex_angle_deviations.append(min_deviation)
    
    # RMS deviation of vertex angles
    if vertex_angle_deviations:
        rms_vertex = np.sqrt(np.mean(np.array(vertex_angle_deviations)**2))
    else:
        rms_vertex = None
    
    # Calculate dihedral angles (angle between adjacent faces)
    # This is more complex and requires face identification
    # For now, use a simplified metric based on tetrahedral angle
    dihedral_deviations = []
    ideal_tetrahedral = 109.47  # degrees
    
    # Sample some dihedral-like angles
    if n_corners >= 4:
        for i in range(min(n_corners, 8)):
            for j in range(i + 1, min(n_corners, 8)):
                for k in range(j + 1, min(n_corners, 8)):
                    # Three points form an angle
                    vec_ij = corner_positions_6d[j][:3] - corner_positions_6d[i][:3]
                    vec_ik = corner_positions_6d[k][:3] - corner_positions_6d[i][:3]
                    
                    cos_angle = np.dot(vec_ij, vec_ik) / (np.linalg.norm(vec_ij) * np.linalg.norm(vec_ik) + 1e-10)
                    cos_angle = np.clip(cos_angle, -1.0, 1.0)
                    angle_deg = np.degrees(np.arccos(cos_angle))
                    
                    # Deviation from ideal tetrahedral
                    deviation = abs(angle_deg - ideal_tetrahedral)
                    dihedral_deviations.append(deviation)
    
    if dihedral_deviations:
        rms_dihedral = np.sqrt(np.mean(np.array(dihedral_deviations)**2))
    else:
        rms_dihedral = None
    
    # Overall symmetry score (normalized combination of metrics)
    symmetry_score = None
    if rms_vertex is not None:
        # Normalize: 0 = perfect, 1 = highly distorted
        # Assume >30° RMS deviation is highly distorted
        symmetry_score = min(1.0, (rms_vertex + edge_cv * 50) / 30.0)
    
    return {
        'vertex_angle_deviations': float(rms_vertex) if rms_vertex is not None else None,
        'dihedral_deviations': float(rms_dihedral) if rms_dihedral is not None else None,
        'edge_length_variation': float(edge_cv),
        'symmetry_score': float(symmetry_score) if symmetry_score is not None else None,
        'n_vertices': n_corners,
        'mean_edge_length': float(edge_mean),
    }


def _compute_cavity_hull_data(
    cavity_data: Dict[str, Any],
    octahedra_data: Dict[int, Dict[str, Any]],
    atom_positions: np.ndarray,
    cell: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Compute and cache convex hull data for a cavity.
    
    This function extracts corner atoms, unwraps them relative to the cavity
    center (PBC-aware), and computes the convex hull. The hull data is cached
    to avoid recalculation when API methods are called.
    
    Parameters
    ----------
    cavity_data : dict
        Cavity data with octahedra indices
    octahedra_data : dict
        Octahedra data
    atom_positions : np.ndarray
        Atomic positions
    cell : np.ndarray
        Unit cell matrix
        
    Returns
    -------
    dict or None
        Dictionary containing:
        - 'corner_atom_indices': list of corner atom indices
        - 'corner_positions_unwrapped': np.ndarray of unwrapped positions
        - 'hull_volume': float, volume in Ų
        - 'hull_equations': list of hull plane equations [a, b, c, d]
        - 'hull_vertices': list of vertex indices in corner_positions
        Returns None if hull computation fails
    """
    from scipy.spatial import ConvexHull
    
    # Get corner X atoms (vertices of cuboctahedron)
    all_corner_atoms = cavity_data.get('x_atom_indices', [])
    
    if len(all_corner_atoms) < 4:
        return None
    
    # Use PBC coordinates if available, otherwise unwrap relative to center
    center = np.array(cavity_data.get('center_position', [0, 0, 0]))
    pbc_coords = cavity_data.get('pbc_coordinates', {})
    cell_lengths = get_cell_lengths(cell)
    
    corner_positions = []
    for atom_idx in all_corner_atoms:
        if atom_idx in pbc_coords:
            # Use pre-computed PBC coordinates
            corner_positions.append(pbc_coords[atom_idx])
        else:
            # Unwrap relative to center
            atom_pos = atom_positions[atom_idx]
            unwrapped_pos = unwrap_relative_coordinate(center, atom_pos, cell_lengths)
            corner_positions.append(unwrapped_pos)
    
    corner_positions = np.array(corner_positions)
    
    # Calculate convex hull
    try:
        hull = ConvexHull(corner_positions)
        
        return {
            'corner_atom_indices': all_corner_atoms,
            'corner_positions_unwrapped': corner_positions.tolist(),
            'hull_volume': float(hull.volume),
            'hull_equations': hull.equations.tolist(),
            'hull_vertices': hull.vertices.tolist(),
        }
    except Exception as e:
        # If convex hull fails, return None
        print(f"  WARNING: Convex hull calculation failed: {e}", file=sys.stderr)
        return None


def calculate_cavity_volume(
    cavity_data: Dict[str, Any],
    octahedra_data: Dict[int, Dict[str, Any]],
    atom_positions: np.ndarray,
    cell: np.ndarray,
) -> float:
    """Calculate cavity volume using convex hull of corner atoms.
    
    The cavity is defined by the X atoms (vertices of the cuboctahedron).
    If hull data is cached in cavity_data, uses that; otherwise computes it.
    
    Parameters
    ----------
    cavity_data : dict
        Cavity data with octahedra indices (may contain cached hull_data)
    octahedra_data : dict
        Octahedra data
    atom_positions : np.ndarray
        Atomic positions
    cell : np.ndarray
        Unit cell matrix
        
    Returns
    -------
    float
        Cavity volume in Ų (cubic Angstroms)
    """
    # Try to use cached hull data first
    hull_data = cavity_data.get('hull_data')
    if hull_data is not None and 'hull_volume' in hull_data:
        return float(hull_data['hull_volume'])
    
    # Fallback: compute hull data if not cached
    hull_data = _compute_cavity_hull_data(
        cavity_data, octahedra_data, atom_positions, cell
    )
    
    if hull_data is not None:
        return float(hull_data['hull_volume'])
    
    return 0.0


def find_closest_cavity_atoms(
    molecule_atom_indices: List[int],
    cavity_data: Dict[str, Any],
    octahedra_data: Dict[int, Dict[str, Any]],
    atom_positions: np.ndarray,
    cell: np.ndarray,
) -> List[Dict[str, Any]]:
    """Find closest cavity atoms for each atom in a molecule (A-site).
    
    For each atom in the molecule, finds the closest X atom in the cavity walls
    and returns PBC-aware coordinates.
    
    Parameters
    ----------
    molecule_atom_indices : list
        Indices of atoms in the molecule
    cavity_data : dict
        Cavity data with corner atom indices
    octahedra_data : dict
        Octahedra data
    atom_positions : np.ndarray
        Atomic positions
    cell : np.ndarray
        Unit cell matrix
        
    Returns
    -------
    list of dict
        For each molecule atom:
        {
            'molecule_atom_index': int,
            'molecule_atom_position': np.ndarray (3,),
            'closest_cavity_atom_index': int,
            'closest_cavity_atom_position': np.ndarray (3,),
            'distance': float,
            'pbc_vector': np.ndarray (3,)  # vector from molecule to cavity atom
        }
    """
    from ..utils.clifford_embedding import embed_to_6d, clifford_distance
    
    # Get cavity corner atoms (X atoms forming the cavity walls)
    all_cavity_atoms = cavity_data.get('x_atom_indices', [])
    
    if not all_cavity_atoms:
        return []
    
    # Use PBC coordinates if available
    pbc_coords = cavity_data.get('pbc_coordinates', {})
    center = np.array(cavity_data.get('center_position', [0, 0, 0]))
    inv_cell = np.linalg.inv(cell)
    cell_lengths = get_cell_lengths(cell)
    
    results = []
    
    for mol_atom_idx in molecule_atom_indices:
        mol_pos = atom_positions[mol_atom_idx]
        
        # Unwrap molecule atom position relative to cavity center
        mol_pos_unwrapped = unwrap_relative_coordinate(center, mol_pos, cell_lengths)
        
        # Find closest cavity atom using PBC-aware distances
        min_distance = float('inf')
        closest_cavity_idx = None
        closest_cavity_pos = None
        
        for cavity_atom_idx in all_cavity_atoms:
            # Use PBC coordinates if available, otherwise unwrap
            if cavity_atom_idx in pbc_coords:
                cavity_pos_unwrapped = pbc_coords[cavity_atom_idx]
            else:
                cavity_pos = atom_positions[cavity_atom_idx]
                cavity_pos_unwrapped = unwrap_relative_coordinate(center, cavity_pos, cell_lengths)
            
            # Calculate distance in unwrapped space
            distance = np.linalg.norm(cavity_pos_unwrapped - mol_pos_unwrapped)
            
            if distance < min_distance:
                min_distance = distance
                closest_cavity_idx = cavity_atom_idx
                closest_cavity_pos = cavity_pos_unwrapped
        
        if closest_cavity_idx is not None:
            # Calculate PBC vector from molecule to cavity atom
            pbc_vector = closest_cavity_pos - mol_pos_unwrapped
            
            results.append({
                'molecule_atom_index': int(mol_atom_idx),
                'molecule_atom_position': mol_pos_unwrapped.tolist(),
                'closest_cavity_atom_index': int(closest_cavity_idx),
                'closest_cavity_atom_position': closest_cavity_pos.tolist(),
                'distance': float(min_distance),
                'pbc_vector': pbc_vector.tolist(),
            })
    
    return results


def is_point_in_cavity(
    point: np.ndarray,
    cavity_data: Dict[str, Any],
    octahedra_data: Dict[int, Dict[str, Any]],
    atom_positions: np.ndarray,
    cell: np.ndarray,
    tolerance: float = 1e-12
) -> bool:
    """Check if a point is inside the cavity volume (convex hull).
    
    Uses the convex hull of the cavity corner atoms (X-sites) to determine
    containment. If hull data is cached in cavity_data, uses that; otherwise
    computes it. Handles PBC by unwrapping corner atoms relative to the
    cavity center.
    
    Parameters
    ----------
    point : np.ndarray
        Point to check [x, y, z]
    cavity_data : dict
        Cavity data dictionary (may contain cached hull_data)
    octahedra_data : dict
        Octahedra data
    atom_positions : np.ndarray
        Atom positions
    cell : np.ndarray
        Unit cell matrix
    tolerance : float
        Tolerance for hull check (negative means strictly inside)
        
    Returns
    -------
    bool
        True if point is inside the cavity
    """
    # Get cavity center
    center = np.array(cavity_data.get('center_position', [0, 0, 0]))
    
    # Unwrap point relative to center
    point_unwrapped = unwrap_relative_coordinate(center, point, get_cell_lengths(cell))
    
    # Try to use cached hull data first
    hull_data = cavity_data.get('hull_data')
    if hull_data is not None and 'hull_equations' in hull_data:
        hull_equations = np.array(hull_data['hull_equations'])
        # Check if point is inside all hull equations (normal . point + offset <= 0)
        # eq = [normal_x, normal_y, normal_z, offset]
        return all(np.dot(eq[:-1], point_unwrapped) + eq[-1] <= tolerance for eq in hull_equations)
    
    # Fallback: compute hull data if not cached
    from scipy.spatial import ConvexHull
    
    hull_data = _compute_cavity_hull_data(
        cavity_data, octahedra_data, atom_positions, cell
    )
    
    if hull_data is None:
        return False
    
    hull_equations = np.array(hull_data['hull_equations'])
    return all(np.dot(eq[:-1], point_unwrapped) + eq[-1] <= tolerance for eq in hull_equations)


def calculate_displacement_from_centroid(
    target_point: np.ndarray,
    cavity_data: Dict[str, Any],
    octahedra_data: Dict[int, Dict[str, Any]],
    atom_positions: np.ndarray,
    cell: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """Calculate the magnitude and direction of the displacement vector 
    from the cuboctahedral centroid to a target point.
    
    This function works only for A-site cavities, which are true cuboctahedra
    with 8 B atoms and 12 X atoms (vertices). The centroid is calculated from
    the 12 X atoms (vertices) of the cuboctahedron.
    
    Parameters
    ----------
    target_point : np.ndarray
        The (x, y, z) coordinates of the point of interest
    cavity_data : dict
        Cavity data dictionary with X atom indices and PBC coordinates
    octahedra_data : dict
        Octahedra data (not used but kept for API consistency)
    atom_positions : np.ndarray
        Original atom positions
    cell : np.ndarray
        Unit cell matrix for PBC handling
        
    Returns
    -------
    tuple
        (magnitude, direction)
        - magnitude (float): Euclidean distance between the centroid and the target point
        - direction (np.ndarray): A (3,) unit vector indicating direction.
                                 Returns a zero vector if magnitude is 0.
    
    Raises
    ------
    ValueError
        If the cavity is not an A-site cavity (not a true cuboctahedron)
    """
    # Check that this is an A-site cavity (true cuboctahedron)
    cavity_type = cavity_data.get('cavity_type', 'a_site')
    if cavity_type != 'a_site':
        raise ValueError(
            f"calculate_displacement_from_centroid only works for A-site cavities "
            f"(true cuboctahedra), but got cavity type: {cavity_type}"
        )
    
    # Get X atoms (vertices of the cuboctahedron)
    x_atom_indices = cavity_data.get('x_atom_indices', [])
    
    if len(x_atom_indices) != 12:
        raise ValueError(
            f"A-site cavity should have exactly 12 X atoms (vertices), "
            f"but found {len(x_atom_indices)}"
        )
    
    # Get cavity center for PBC unwrapping
    center = np.array(cavity_data.get('center_position', [0, 0, 0]))
    cell_lengths = get_cell_lengths(cell)
    
    # Get vertex positions (X atoms) with PBC handling
    pbc_coords = cavity_data.get('pbc_coordinates', {})
    vertices = []
    
    for x_idx in x_atom_indices:
        if x_idx in pbc_coords:
            # Use pre-computed PBC coordinates
            vertices.append(pbc_coords[x_idx])
        else:
            # Unwrap relative to cavity center
            x_pos = atom_positions[x_idx]
            unwrapped_pos = unwrap_relative_coordinate(center, x_pos, cell_lengths)
            vertices.append(unwrapped_pos)
    
    vertices = np.array(vertices)
    
    # Unwrap target point relative to cavity center
    target = np.array(target_point)
    target_unwrapped = unwrap_relative_coordinate(center, target, cell_lengths)
    
    # Calculate centroid from vertices
    centroid = np.mean(vertices, axis=0)
    
    # Calculate vector difference (displacement)
    displacement_vector = target_unwrapped - centroid
    
    # Calculate magnitude (Euclidean norm)
    magnitude = np.linalg.norm(displacement_vector)
    
    # Calculate normalized direction vector
    # Handle division by zero if the target is exactly at the centroid
    if magnitude > 1e-8:
        direction = displacement_vector / magnitude
    else:
        direction = np.zeros(3)
    
    return float(magnitude), direction
