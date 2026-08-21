"""Layer and slab identification for 2D perovskites.

This module identifies layers/slabs based on z-coordinate continuity
and edge-sharing relationships between octahedra.
"""

import numpy as np
import networkx as nx

from ..octahedral_processing.octahedral_detection import _calculate_avg_bx_distance


def _mic_distance(p1: np.ndarray, p2: np.ndarray, cell: np.ndarray = None) -> float:
    """Minimum-image Cartesian distance; Euclidean if ``cell`` is None."""
    delta = np.asarray(p2, dtype=float) - np.asarray(p1, dtype=float)
    if cell is not None:
        cell = np.asarray(cell, dtype=float)
        frac = np.linalg.solve(cell.T, delta)
        frac -= np.round(frac)
        delta = cell.T @ frac
    return float(np.linalg.norm(delta))


def _identify_slabs_by_continuity(
    octahedra_info: list,
    shared_atoms: dict,
    atom_positions: np.ndarray,
    cell: np.ndarray = None,
) -> dict:
    """Partition octahedra into slabs based on continuity.

    A slab is a connected group of octahedra linked by shared ligands and/or
    B–B proximity within expected layer spacing (MIC-aware). This recovers
    slabs when ligand classification marks bridges as terminal (e.g. NMSE
    #209) and when stacking is not along cartesian z.

    Parameters
    ----------
    octahedra_info : list
        List of octahedra dictionaries with 'central_atom_index' key
    shared_atoms : dict
        Dictionary mapping (oct_i, oct_j) -> list of shared atom indices
    atom_positions : np.ndarray
        Array of all atom positions
    cell : np.ndarray, optional
        3x3 cell matrix for minimum-image distances

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
    if len(octahedra_info) == 0:
        return {
            'slabs': {},
            'slab_z_ranges': {},
            'discontinuity_regions': [],
            'max_z_jump_threshold': 0,
            'expected_layer_spacing': 7.0,
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
        expected_layer_spacing = 2 * avg_bx
    else:
        expected_layer_spacing = 7.0

    max_z_jump = expected_layer_spacing * 1.3

    # Build graph of octahedra with z-coordinates (cartesian z kept for ranges)
    oct_graph = nx.Graph()
    oct_z_coords = {}
    oct_centers = {}

    for oct_idx, oct_data in enumerate(octahedra_info):
        central_idx = oct_data.get('central_atom_index')
        if central_idx is not None:
            z_coord = atom_positions[central_idx][2]
            oct_graph.add_node(oct_idx, z_coord=z_coord)
            oct_z_coords[oct_idx] = z_coord
            oct_centers[oct_idx] = np.asarray(atom_positions[central_idx], dtype=float)

    # Add edges between octahedra that share atoms, filtering by B–B distance
    for (oct_i, oct_j), shared in shared_atoms.items():
        if oct_i in oct_centers and oct_j in oct_centers:
            bb_dist = _mic_distance(oct_centers[oct_i], oct_centers[oct_j], cell)
            if bb_dist <= max_z_jump:
                oct_graph.add_edge(
                    oct_i, oct_j,
                    z_difference=abs(oct_z_coords[oct_i] - oct_z_coords[oct_j]),
                    bb_distance=bb_dist,
                    n_shared_atoms=len(shared),
                    shared_atoms=shared,
                )

    # Fallback: connect by B–B proximity when ligand sharing was missed
    # (all-terminal misclassification). Does not bridge organic gaps (~2× spacing).
    oct_ids = list(oct_centers.keys())
    for i, oct_i in enumerate(oct_ids):
        for oct_j in oct_ids[i + 1:]:
            if oct_graph.has_edge(oct_i, oct_j):
                continue
            bb_dist = _mic_distance(oct_centers[oct_i], oct_centers[oct_j], cell)
            if bb_dist <= max_z_jump:
                oct_graph.add_edge(
                    oct_i, oct_j,
                    z_difference=abs(oct_z_coords[oct_i] - oct_z_coords[oct_j]),
                    bb_distance=bb_dist,
                    n_shared_atoms=0,
                    shared_atoms=[],
                    proximity_fallback=True,
                )

    slabs = {}
    slab_z_ranges = {}

    for slab_id, component in enumerate(nx.connected_components(oct_graph)):
        oct_indices = list(component)
        slabs[slab_id] = oct_indices
        z_coords = [oct_z_coords[idx] for idx in oct_indices]
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
    octahedra_geometries: list = None,
) -> tuple:
    """Identify layers by Z-coordinate grouping of octahedra centers.

    For bulk structures where topology-based classification fails (all atoms
    appear equatorial), this function uses Z-coordinate clustering to first
    identify layers, then reclassifies atoms based on layer connections.

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
    octahedra_geometries : list, optional
        List of geometric classifications for each octahedron

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

    # 1. Classify X-atoms based on geometry (REQUIRED - no fallback)
    if octahedra_geometries is None:
        raise ValueError(
            "octahedra_geometries is required for layer identification. "
            "Geometric classification must be performed first."
        )

    x_atom_classifications = {}

    # Aggregate roles of each atom across all octahedra
    atom_roles = {}
    atom_connected_octs = {}

    for oct_idx, geom in enumerate(octahedra_geometries):
        if not geom:
            continue
        for atom_idx, role in geom.items():
            if atom_idx not in atom_roles:
                atom_roles[atom_idx] = set()
                atom_connected_octs[atom_idx] = []
            atom_roles[atom_idx].add(role)
            atom_connected_octs[atom_idx].append(oct_idx)

    # Check if this is a bulk structure (all atoms classified as equatorial)
    # by checking if there are any terminal atoms
    has_terminal_atoms = False
    for atom_idx, roles in atom_roles.items():
        connected = atom_connected_octs[atom_idx]
        if any(r in ['axial_top', 'axial_bottom', 'axial_terminal'] for r in roles) and len(connected) == 1:
            has_terminal_atoms = True
            break

    is_bulk = not has_terminal_atoms

    # For bulk structures: use Z-coordinate clustering to identify layers FIRST
    # Then reclassify atoms based on layer connections
    if is_bulk and atom_positions is not None and center_atom_indices is not None:
        # Get Z-coordinates of octahedra centers
        z_coords = np.array([atom_positions[center_atom_indices[i]][2] for i in range(n_octahedra)])

        # Cluster by Z-coordinate using simple binning
        # Expected layer spacing is roughly 2 * B-X distance (~6-7 Angstrom)
        z_sorted_indices = np.argsort(z_coords)
        z_sorted = z_coords[z_sorted_indices]

        # Find layer boundaries using gaps in Z
        if len(z_sorted) > 1:
            z_diffs = np.diff(z_sorted)
            median_diff = np.median(z_diffs) if len(z_diffs) > 0 else 1.0
            # Gaps larger than 2x median indicate layer boundary
            gap_threshold = max(median_diff * 2.0, 2.0)

            layer_boundaries = [0]
            for i, diff in enumerate(z_diffs):
                if diff > gap_threshold:
                    layer_boundaries.append(i + 1)
            layer_boundaries.append(len(z_sorted))

            # Assign octahedra to Z-based layers
            z_layer_assignment = {}
            for layer_id in range(len(layer_boundaries) - 1):
                start_idx = layer_boundaries[layer_id]
                end_idx = layer_boundaries[layer_id + 1]
                for i in range(start_idx, end_idx):
                    oct_idx = z_sorted_indices[i]
                    z_layer_assignment[oct_idx] = layer_id
        else:
            z_layer_assignment = {0: 0}

        # Now reclassify atoms based on Z-layer connections
        for atom_idx, roles in atom_roles.items():
            connected = atom_connected_octs[atom_idx]

            # Get layers this atom connects
            connected_layers = set(z_layer_assignment.get(oct_idx, 0) for oct_idx in connected)

            if len(connected_layers) > 1:
                # Connects multiple Z-layers -> interlayer
                atype = 'interlayer'
            elif len(connected) > 1:
                # Shared within same layer -> intralayer
                atype = 'intralayer'
            else:
                # Only in one octahedron -> terminal (shouldn't happen in bulk)
                atype = 'axial'

            x_atom_classifications[atom_idx] = {
                'type': atype,
                'connected_octahedra': connected,
                'z_coord': atom_positions[atom_idx][2] if atom_positions is not None else 0.0
            }

        # Build layer connectivity using Z-layer assignment
        oct_graph = nx.Graph()
        oct_graph.add_nodes_from(range(n_octahedra))

        for (oct_i, oct_j), shared in shared_atoms.items():
            # Only connect if in same Z-layer
            if z_layer_assignment.get(oct_i, 0) == z_layer_assignment.get(oct_j, 0):
                oct_graph.add_edge(oct_i, oct_j)

        components = list(nx.connected_components(oct_graph))
    else:
        # Original logic for slab structures with clear axial/equatorial distinction
        # Determine final type based on geometric rules
        for atom_idx, roles in atom_roles.items():
            connected = atom_connected_octs[atom_idx]
            n_octs_sharing = len(connected)

            # Rule 1.2: Intralayer = equatorial in ANY octahedron
            if 'equatorial' in roles:
                atype = 'intralayer'
            # Rule 1.2: Interlayer = axial in MULTIPLE octahedra (shared axials connect layers)
            elif any(r in ['axial_top', 'axial_bottom'] for r in roles) and n_octs_sharing > 1:
                atype = 'interlayer'
            # Rule 1.2: Terminal = axial in EXACTLY ONE octahedron
            elif any(r in ['axial_top', 'axial_bottom'] for r in roles) and n_octs_sharing == 1:
                atype = 'axial'  # Terminal axial
            else:
                atype = 'unknown'

            x_atom_classifications[atom_idx] = {
                'type': atype,
                'connected_octahedra': connected,
                'z_coord': atom_positions[atom_idx][2] if atom_positions is not None else 0.0
            }

        # 2. Identify Layers via Graph Connectivity (Equatorial connections)
        # Build a graph where nodes are octahedra
        # Edges exist if they share an 'intralayer' (equatorial) atom
        oct_graph = nx.Graph()
        oct_graph.add_nodes_from(range(n_octahedra))

        for (oct_i, oct_j), shared in shared_atoms.items():
            # Check if any shared atom is intralayer
            is_intralayer_connection = False
            for atom_idx in shared:
                if atom_idx in x_atom_classifications:
                    if x_atom_classifications[atom_idx]['type'] == 'intralayer':
                        is_intralayer_connection = True
                        break

            if is_intralayer_connection:
                oct_graph.add_edge(oct_i, oct_j)

        components = list(nx.connected_components(oct_graph))

    # Find connected components -> Layers
    layers = {}
    
    # Calculate average Z for each component to sort them
    # Note: Uses Cartesian Z-coordinates for sorting. This is a heuristic for layer ordering.
    # For non-orthogonal cells, this approximates the stacking direction.
    # Layers are primarily identified by connectivity (edge-sharing), not Z-distance.
    comp_z_coords = []
    for comp in components:
        oct_indices = list(comp)
        avg_z = 0.0
        if atom_positions is not None and center_atom_indices is not None:
            z_vals = [atom_positions[center_atom_indices[i]][2] for i in oct_indices]
            avg_z = np.mean(z_vals)
        comp_z_coords.append((avg_z, oct_indices))
        
    # Sort layers by Z
    comp_z_coords.sort(key=lambda x: x[0])
    
    n_levels = len(comp_z_coords)

    for layer_id, (z_coord, octahedra_list) in enumerate(comp_z_coords):
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

    for layer_id in range(len(comp_z_coords) - 1):
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
