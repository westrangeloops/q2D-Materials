"""Layer and slab identification for 2D perovskites.

This module identifies layers/slabs based on z-coordinate continuity
and edge-sharing relationships between octahedra.
"""

import numpy as np
import networkx as nx

from .octahedral_detection import _calculate_avg_bx_distance, _classify_x_atoms_by_z


def _identify_slabs_by_continuity(
    bx_graph: nx.Graph,
    octahedra_info: list,
    atom_positions: np.ndarray,
) -> dict:
    """Partition octahedra into slabs based on z-continuity in the B-X network.

    A slab is a connected group of octahedra where z-changes between
    adjacent octahedra are smooth (within expected layer spacing).
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
        Slab information with keys:
        - 'slabs': dict mapping slab_id -> list of octahedra indices
        - 'slab_z_ranges': dict mapping slab_id -> (min_z, max_z)
        - 'discontinuity_regions': list of (z_start, z_end) tuples
        - 'max_z_jump_threshold': calculated threshold for discontinuity
        - 'expected_layer_spacing': expected layer spacing
    """
    if len(bx_graph.nodes) == 0:
        return {
            'slabs': {},
            'slab_z_ranges': {},
            'discontinuity_regions': [],
            'max_z_jump_threshold': 0,
            'expected_layer_spacing': 7.0,
        }

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
        expected_layer_spacing = 2 * avg_bx
    else:
        expected_layer_spacing = 7.0

    max_z_jump = expected_layer_spacing * 1.3

    continuous_graph = nx.Graph()
    continuous_graph.add_nodes_from(bx_graph.nodes(data=True))

    for u, v, edge_data in bx_graph.edges(data=True):
        z_diff = edge_data.get('z_difference', 0)
        if z_diff <= max_z_jump:
            continuous_graph.add_edge(u, v, **edge_data)

    slabs = {}
    slab_z_ranges = {}

    for slab_id, component in enumerate(nx.connected_components(continuous_graph)):
        oct_indices = list(component)
        slabs[slab_id] = oct_indices
        z_coords = [bx_graph.nodes[idx]['z_coord'] for idx in oct_indices]
        slab_z_ranges[slab_id] = (min(z_coords), max(z_coords))

    discontinuity_regions = []
    if len(slab_z_ranges) > 1:
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
    """Identify layers by Z-coordinate grouping of octahedra centers.

    Replaces edge-sharing based approach which failed for cubic perovskites
    where all connections are corner-sharing (1 shared atom).

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

    if atom_positions is None or center_atom_indices is None:
        layers = {}
        for oct_idx in range(n_octahedra):
            layers[oct_idx] = {
                'position': 'surface',
                'octahedra': [oct_idx],
                'octahedra_count': 1,
            }
        return layers, {}

    avg_bx = _calculate_avg_bx_distance(neighbor_indices, atom_positions, center_atom_indices)
    z_tolerance = avg_bx * 0.5

    x_atom_classifications = _classify_x_atoms_by_z(
        neighbor_indices, shared_atoms, atom_positions, center_atom_indices
    )

    oct_z_coords = [(oct_idx, atom_positions[center_idx][2]) for oct_idx, center_idx in enumerate(center_atom_indices)]
    oct_z_coords.sort(key=lambda x: x[1])

    z_levels = []

    for oct_idx, z_coord in oct_z_coords:
        found_level = None
        for level_idx, (level_z, level_octs) in enumerate(z_levels):
            if abs(z_coord - level_z) < z_tolerance:
                found_level = level_idx
                break

        if found_level is not None:
            level_z, level_octs = z_levels[found_level]
            level_octs.append(oct_idx)
            new_avg_z = sum(atom_positions[center_atom_indices[o]][2] for o in level_octs) / len(level_octs)
            z_levels[found_level] = (new_avg_z, level_octs)
        else:
            z_levels.append((z_coord, [oct_idx]))

    z_levels.sort(key=lambda x: x[0])

    layers = {}
    n_levels = len(z_levels)

    for layer_id, (z_coord, octahedra_list) in enumerate(z_levels):
        if n_levels == 1:
            position = 'surface'
        elif layer_id == 0 or layer_id == n_levels - 1:
            position = 'surface'
        else:
            position = 'central'

        layer_atom_indices = set()
        for oct_idx in octahedra_list:
            layer_atom_indices.update(neighbor_indices[oct_idx])

        terminal_count = sum(
            1 for atom_idx in layer_atom_indices
            if atom_idx in x_atom_classifications and x_atom_classifications[atom_idx]['type'] == 'axial'
        )

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

    for layer_id in range(len(z_levels) - 1):
        current_layer_octs = set(layers[layer_id]['octahedra'])
        next_layer_octs = set(layers[layer_id + 1]['octahedra'])

        connecting_x_atoms = []
        for atom_idx, info in x_atom_classifications.items():
            if info['type'] == 'interlayer':
                connected_octs = set(info['connected_octahedra'])
                if connected_octs & current_layer_octs and connected_octs & next_layer_octs:
                    connecting_x_atoms.append(atom_idx)

        layers[layer_id]['interlayer_x_atoms_above'] = connecting_x_atoms
        layers[layer_id + 1]['interlayer_x_atoms_below'] = connecting_x_atoms

    if 0 in layers and 'interlayer_x_atoms_below' not in layers[0]:
        layers[0]['interlayer_x_atoms_below'] = []
    if len(layers) > 0:
        last_layer_id = max(layers.keys())
        if 'interlayer_x_atoms_above' not in layers[last_layer_id]:
            layers[last_layer_id]['interlayer_x_atoms_above'] = []

    return layers, x_atom_classifications


def _get_octahedron_layer(octahedron_id, layers):
    """Determine which layer an octahedron belongs to.

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

    return 0
