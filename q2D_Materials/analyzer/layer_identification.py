"""
Layer and slab identification for 2D perovskites.

This module identifies layers/slabs based on z-coordinate continuity
and edge-sharing relationships between octahedra.

Functions
---------
_identify_slabs_by_continuity
    Partition octahedra into slabs based on z-continuity
_identify_layers
    Group octahedra into Z-based layers (legacy approach)
_get_octahedron_layer
    Helper to determine which layer an octahedron belongs to
"""

import numpy as np
import networkx as nx
from .octahedral_detection import _calculate_avg_bx_distance, _classify_x_atoms_by_z


def _identify_slabs_by_continuity(
    bx_graph: nx.Graph,
    octahedra_info: list,
    atom_positions: np.ndarray,
) -> dict:
    """
    Partition octahedra into slabs based on z-continuity in the B-X network.
    
    A slab is a connected group of octahedra where z-changes between 
    adjacent octahedra are "smooth" (within expected layer spacing).
    A discontinuity (large z-jump) indicates a slab boundary.
    
    Parameters
    ----------
    bx_graph : nx.Graph
        B-X network graph from _build_bx_network
    octahedra_info : list
        List of octahedra dictionaries
    atom_positions : np.ndarray
        Array of all atom positions
        
    Returns
    -------
    dict
        Slab information:
        - 'slabs': dict mapping slab_id -> list of octahedra indices
        - 'slab_z_ranges': dict mapping slab_id -> (min_z, max_z)
        - 'discontinuity_regions': list of (z_start, z_end) tuples
        - 'max_z_jump_threshold': the calculated threshold for discontinuity
    """
    if len(bx_graph.nodes) == 0:
        return {
            'slabs': {},
            'slab_z_ranges': {},
            'discontinuity_regions': [],
            'max_z_jump_threshold': 0,
        }
    
    # Calculate expected layer spacing from B-X distances
    max_bx_distance = 5.0
    bx_distances = []
    for oct_data in octahedra_info:
        central_idx = oct_data.get('central_atom_index')
        if central_idx is None:
            continue
        b_pos = atom_positions[central_idx]
        for neighbor_list in ['terminal_atoms', 'interlayer_atoms', 'intralayer_atoms']:
            for x_idx in oct_data.get(neighbor_list, []):
                x_pos = atom_positions[x_idx]
                dist = np.linalg.norm(b_pos - x_pos)
                if dist < max_bx_distance:
                    bx_distances.append(dist)
    
    if bx_distances:
        avg_bx = np.mean(bx_distances)
        # Expected layer spacing = 2 × B-X (apical-to-apical)
        expected_layer_spacing = 2 * avg_bx
    else:
        expected_layer_spacing = 7.0  # Fallback
    
    # Max z-jump within a slab = 1.3 × expected layer spacing
    # Anything larger indicates a discontinuity (slab boundary)
    max_z_jump = expected_layer_spacing * 1.3
    
    # Create a subgraph that only includes "continuous" edges
    continuous_graph = nx.Graph()
    continuous_graph.add_nodes_from(bx_graph.nodes(data=True))
    
    for u, v, edge_data in bx_graph.edges(data=True):
        z_diff = edge_data.get('z_difference', 0)
        if z_diff <= max_z_jump:
            continuous_graph.add_edge(u, v, **edge_data)
    
    # Find connected components in the continuous graph = slabs
    slabs = {}
    slab_z_ranges = {}
    
    for slab_id, component in enumerate(nx.connected_components(continuous_graph)):
        oct_indices = list(component)
        slabs[slab_id] = oct_indices
        
        # Calculate z-range for this slab
        z_coords = [bx_graph.nodes[idx]['z_coord'] for idx in oct_indices]
        slab_z_ranges[slab_id] = (min(z_coords), max(z_coords))
    
    # Identify discontinuity regions (gaps between slabs)
    discontinuity_regions = []
    if len(slab_z_ranges) > 1:
        # Sort slabs by their min z-coordinate
        sorted_slabs = sorted(slab_z_ranges.items(), key=lambda x: x[1][0])
        
        for i in range(len(sorted_slabs) - 1):
            slab_a_id, (_, z_max_a) = sorted_slabs[i]
            slab_b_id, (z_min_b, _) = sorted_slabs[i + 1]
            
            if z_min_b > z_max_a:
                discontinuity_regions.append((z_max_a, z_min_b))
    
    return {
        'slabs': slabs,
        'slab_z_ranges': slab_z_ranges,
        'discontinuity_regions': discontinuity_regions,
        'max_z_jump_threshold': max_z_jump,
        'expected_layer_spacing': expected_layer_spacing,
    }

def _identify_layers(
    neighbor_indices: list,
    shared_atoms: dict,
    atom_positions: np.ndarray = None,
    center_atom_indices: list = None,
    cell: np.ndarray = None,
) -> tuple:
    """
    Identify layers by Z-coordinate grouping of octahedra centers.

    This replaces the old edge-sharing based approach which failed for cubic
    perovskites where all connections are corner-sharing (1 shared atom).

    Algorithm:
    1. Get Z-coords of all octahedra centers
    2. Calculate avg B-X distance → z_threshold for same-floor grouping
    3. Group octahedra by Z-level (within z_threshold)
    4. Check for axial atoms to determine if structure has floors
       - No axials → bulk (all connected, still group by Z for visualization)
       - Has axials → identify floors between interlayer connections

    Parameters
    ----------
    neighbor_indices : list
        List of neighbor indices for each octahedron
    shared_atoms : dict
        Dict mapping octahedra pairs to shared atom indices
    atom_positions : np.ndarray, optional
        Array of atom positions (required for Z-based grouping)
    center_atom_indices : list, optional
        List of central atom indices for each octahedron
    cell : np.ndarray, optional
        Unit cell matrix

    Returns
    -------
    tuple
        (layers_dict, x_atom_classifications)
        - layers_dict: {layer_id: {'position': str, 'octahedra': list, ...}}
        - x_atom_classifications: {atom_idx: {'type': str, 'connected_octahedra': list, ...}}
    """
    n_octahedra = len(neighbor_indices)
    if n_octahedra == 0:
        return {}, {}

    # Fallback to old behavior if positions not provided (backwards compatibility)
    if atom_positions is None or center_atom_indices is None:
        # Use simple grouping where each octahedron is its own layer
        layers = {}
        for oct_idx in range(n_octahedra):
            layers[oct_idx] = {
                'position': 'surface',
                'octahedra': [oct_idx],
                'octahedra_count': 1,
            }
        return layers, {}

    # Calculate average B-X distance for Z-threshold
    avg_bx = _calculate_avg_bx_distance(neighbor_indices, atom_positions, center_atom_indices)
    # Z-tolerance for grouping octahedra into same layer
    # This should be less than the B-X distance but enough to handle distortions
    z_tolerance = avg_bx * 0.5

    # Classify X-atoms by Z-difference analysis
    x_atom_classifications = _classify_x_atoms_by_z(
        neighbor_indices, shared_atoms, atom_positions, center_atom_indices
    )

    # Get Z-coordinates of octahedra centers
    oct_z_coords = []
    for oct_idx, center_idx in enumerate(center_atom_indices):
        oct_z_coords.append((oct_idx, atom_positions[center_idx][2]))

    # Sort octahedra by Z-coordinate
    oct_z_coords.sort(key=lambda x: x[1])

    # Group octahedra by Z-level
    z_levels = []  # List of (z_coord, [octahedra_indices])

    for oct_idx, z_coord in oct_z_coords:
        # Find if there's an existing Z-level within tolerance
        found_level = None
        for level_idx, (level_z, level_octs) in enumerate(z_levels):
            if abs(z_coord - level_z) < z_tolerance:
                found_level = level_idx
                break

        if found_level is not None:
            # Add to existing level, update average Z
            level_z, level_octs = z_levels[found_level]
            level_octs.append(oct_idx)
            new_avg_z = sum(atom_positions[center_atom_indices[o]][2] for o in level_octs) / len(level_octs)
            z_levels[found_level] = (new_avg_z, level_octs)
        else:
            # Create new Z-level
            z_levels.append((z_coord, [oct_idx]))

    # Sort Z-levels by their Z-coordinate
    z_levels.sort(key=lambda x: x[0])

    # Check for axial atoms to determine surface vs central layers
    has_axial_atoms = any(info['type'] == 'axial' for info in x_atom_classifications.values())

    # Create layer dict
    layers = {}
    n_levels = len(z_levels)

    for layer_id, (z_coord, octahedra_list) in enumerate(z_levels):
        # Determine position: surface if it's the first or last layer, or has axial atoms
        if n_levels == 1:
            position = 'surface'  # Single layer = monolayer = all surface
        elif layer_id == 0 or layer_id == n_levels - 1:
            position = 'surface'  # Top or bottom layer
        else:
            position = 'central'  # Middle layer

        # Count terminal (axial) atoms in this layer
        layer_atom_indices = set()
        for oct_idx in octahedra_list:
            layer_atom_indices.update(neighbor_indices[oct_idx])

        terminal_count = sum(
            1 for atom_idx in layer_atom_indices
            if atom_idx in x_atom_classifications and x_atom_classifications[atom_idx]['type'] == 'axial'
        )

        # Find intralayer X-atoms (equatorial connections within this layer)
        intralayer_x_atoms = [
            atom_idx for atom_idx in layer_atom_indices
            if atom_idx in x_atom_classifications and x_atom_classifications[atom_idx]['type'] == 'intralayer'
        ]

        layers[layer_id] = {
            'position': position,
            'octahedra': octahedra_list,
            'octahedra_count': len(octahedra_list),
            'z_coord': z_coord,
            'terminal_atoms_count': terminal_count,
            'intralayer_x_atoms': intralayer_x_atoms,
        }

    # Add interlayer connection info
    for layer_id in range(len(z_levels) - 1):
        current_layer_octs = set(layers[layer_id]['octahedra'])
        next_layer_octs = set(layers[layer_id + 1]['octahedra'])

        # Find interlayer X-atoms connecting these layers
        connecting_x_atoms = []
        for atom_idx, info in x_atom_classifications.items():
            if info['type'] == 'interlayer':
                connected_octs = set(info['connected_octahedra'])
                if connected_octs & current_layer_octs and connected_octs & next_layer_octs:
                    connecting_x_atoms.append(atom_idx)

        layers[layer_id]['interlayer_x_atoms_above'] = connecting_x_atoms
        layers[layer_id + 1]['interlayer_x_atoms_below'] = connecting_x_atoms

    # Handle edge cases for first/last layers
    if 0 in layers and 'interlayer_x_atoms_below' not in layers[0]:
        layers[0]['interlayer_x_atoms_below'] = []
    if len(layers) > 0:
        last_layer_id = max(layers.keys())
        if 'interlayer_x_atoms_above' not in layers[last_layer_id]:
            layers[last_layer_id]['interlayer_x_atoms_above'] = []

    return layers, x_atom_classifications


def _get_octahedron_layer(octahedron_id, layers):
    """
    Determine which layer an octahedron belongs to based on layer membership.

    Parameters
    ----------
    octahedron_id : int
        Index of the octahedron
    layers : dict
        Dict of layer information

    Returns
    -------
    int
        layer_id
    """
    for layer_id, layer_info in layers.items():
        if octahedron_id in layer_info['octahedra']:
            return layer_id

    # If not found, return 0 as default
    return 0
