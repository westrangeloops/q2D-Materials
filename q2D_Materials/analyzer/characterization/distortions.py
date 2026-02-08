"""Octahedral distortion calculations.

This module provides methods for computing octahedral distortion parameters
using the graph-based analyzer.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ..utils.geometry_helpers import (
    apply_pbc_to_vector,
    calculate_angle_between_vectors,
    extract_bx_bond_vectors,
)
from ..octahedral_processing.volume_calculations import compute_octahedral_volume
from ...utils.geometry.pbc_distances import (
    find_nearest_image_positions,
    calculate_pbc_distances,
)


def _get_x_atoms_from_octahedron_node(graph, octahedron_node: str) -> List[int]:
    """
    Get all X-atom indices from an octahedron node using graph traversal.
    
    Tries two methods:
    1. Direct: Octahedron → CONTAINS (role='ligand') → Atom (X-site)
    2. Fallback: Octahedron → CONTAINS (role='center') → B atom → BONDED_TO (role='ligand') → X atom
    
    Parameters
    ----------
    graph : nx.Graph
        The structural graph
    octahedron_node : str
        Octahedron node ID (e.g., 'octahedron_0')
    
    Returns
    -------
    list of int
        List of X-atom indices (vasp_index values)
    """
    x_atom_indices = []
    
    # Method 1: Try direct CONTAINS edges (preferred according to GraphStructure.md)
    for neighbor in graph.neighbors(octahedron_node):
        edge_data = graph.get_edge_data(octahedron_node, neighbor)
        if (edge_data and
            edge_data.get('edge_type') == 'contains' and
            edge_data.get('role') == 'ligand'):
            neighbor_data = graph.nodes.get(neighbor, {})
            if neighbor_data.get('node_type') == 'atom':
                atom_idx = neighbor_data.get('vasp_index')
                if atom_idx is not None:
                    x_atom_indices.append(atom_idx)
    
    # Method 2: Fallback to BONDED_TO edges via B atom (if CONTAINS edges don't exist)
    if len(x_atom_indices) < 6:
        # Find B atom first
        b_atom_node = None
        for neighbor in graph.neighbors(octahedron_node):
            edge_data = graph.get_edge_data(octahedron_node, neighbor)
            if (edge_data and
                edge_data.get('edge_type') == 'contains' and
                edge_data.get('role') == 'center'):
                b_atom_node = neighbor
                break
        
        # If B atom found, get X atoms via BONDED_TO edges
        if b_atom_node:
            for neighbor in graph.neighbors(b_atom_node):
                edge_data = graph.get_edge_data(b_atom_node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'bonded_to' and
                    edge_data.get('role') == 'ligand'):
                    neighbor_data = graph.nodes.get(neighbor, {})
                    if neighbor_data.get('node_type') == 'atom':
                        atom_idx = neighbor_data.get('vasp_index')
                        if atom_idx is not None and atom_idx not in x_atom_indices:
                            x_atom_indices.append(atom_idx)
    
    return x_atom_indices


def _get_b_atom_from_octahedron_node(graph, octahedron_node: str) -> Optional[int]:
    """
    Get B-atom index from an octahedron node using graph traversal.
    
    According to GraphStructure.md:
    - Octahedron → CONTAINS (role='center') → Atom (B-site)
    
    Parameters
    ----------
    graph : nx.Graph
        The structural graph
    octahedron_node : str
        Octahedron node ID (e.g., 'octahedron_0')
    
    Returns
    -------
    int or None
        B-atom index (vasp_index) or None if not found
    """
    # Traverse: Octahedron → CONTAINS (role='center') → Atom
    for neighbor in graph.neighbors(octahedron_node):
        edge_data = graph.get_edge_data(octahedron_node, neighbor)
        if (edge_data and
            edge_data.get('edge_type') == 'contains' and
            edge_data.get('role') == 'center'):
            neighbor_data = graph.nodes.get(neighbor, {})
            if neighbor_data.get('node_type') == 'atom':
                atom_idx = neighbor_data.get('vasp_index')
                if atom_idx is not None:
                    return atom_idx
    return None


def _filter_octahedra_by_selection(
    analyzer,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
) -> List[str]:
    """
    Filter octahedra node IDs based on selection criteria using graph structure.
    
    According to GraphStructure.md:
    - Layer → CONTAINS → Octahedron
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzed structure with graph
    octahedra : list of str, optional
        Specific octahedra IDs to include (e.g., ['octahedron_0', 'octahedron_1'])
    layer : str, optional
        Single layer ID to filter by (e.g., '0')
    layers : list of str, optional
        Multiple layer IDs to filter by (e.g., ['0', '1'])
    
    Returns
    -------
    list of str
        List of octahedron node IDs
    """
    graph = analyzer.get_graph()
    
    # If no filters specified, return all octahedron nodes
    if octahedra is None and layer is None and layers is None:
        return [node for node, data in graph.nodes(data=True) 
                if data.get('node_type') == 'octahedron']
    
    # Build layer -> octahedra mapping from graph
    # Traverse: Layer → CONTAINS → Octahedron
    layer_oct_map = {}
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'layer':
            layer_id = node.replace('layer_', '')
            octahedra_in_layer = []
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if (edge_data and 
                    edge_data.get('edge_type') == 'contains' and
                    neighbor.startswith('octahedron_')):
                    octahedra_in_layer.append(neighbor)
            layer_oct_map[layer_id] = octahedra_in_layer
    
    # Filter by specific octahedra IDs
    if octahedra is not None:
        octahedra_set = set(octahedra)
        all_octahedra = [node for node, data in graph.nodes(data=True) 
                        if data.get('node_type') == 'octahedron']
        return [oct_id for oct_id in all_octahedra if oct_id in octahedra_set]
    
    # Filter by single layer
    if layer is not None:
        return layer_oct_map.get(layer, [])
    
    # Filter by multiple layers
    if layers is not None:
        allowed_ids = set()
        for layer_id in layers:
            allowed_ids.update(layer_oct_map.get(layer_id, []))
        return list(allowed_ids)
    
    return []


def _get_octahedral_distortions_detailed(
    analyzer,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Union[float, np.ndarray, str]]]:
    """
    Compute per-octahedron distortion metrics with layer information.

    Uses direct graph traversal according to GraphStructure.md:
    - Octahedron → CONTAINS (role='center') → Atom (B-site)
    - Octahedron → CONTAINS (role='ligand') → Atom (X-site)
    - Layer → CONTAINS → Octahedron

    Bond lengths are separated into axial and equatorial categories based on
    X atom geometry from the graph. This is important because axial B-X bonds
    (along c-axis) and equatorial B-X bonds (in ab-plane) have significantly
    different lengths in layered perovskites.

    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzed structure with graph
    octahedra : list of str, optional
        Specific octahedra IDs to include
    layer : str, optional
        Single layer ID to filter by
    layers : list of str, optional
        Multiple layer IDs to filter by

    Returns
    -------
    dict
        Octahedron ID -> metrics dict containing:
        - 'delta': Bond length distortion (using all bonds)
        - 'sigma': Bond length variance (using all bonds)
        - 'lambda': Bond angle variance
        - 'bond_lengths': Array of all B-X bond lengths (6 values)
        - 'bond_lengths_axial': Array of axial B-X bond lengths (2 values)
        - 'bond_lengths_equatorial': Array of equatorial B-X bond lengths (4 values)
        - 'bond_angles': Array of X-B-X angles
        - 'mean_bond_length': Mean of all B-X bond lengths
        - 'mean_bond_length_axial': Mean of axial B-X bond lengths
        - 'mean_bond_length_equatorial': Mean of equatorial B-X bond lengths
        - 'mean_angle': Mean X-B-X angle
        - 'volume': Octahedral volume
        - 'layer': Layer ID this octahedron belongs to
        - 'central_atom_index': Index of B-site atom
        - 'central_atom_symbol': Element symbol of B-site
        - 'geometry': Dict mapping X atom index to geometry type
    """
    graph = analyzer.get_graph()

    # Filter octahedra by selection
    selected_octahedra_nodes = _filter_octahedra_by_selection(
        analyzer, octahedra=octahedra, layer=layer, layers=layers
    )

    # Build octahedron -> layer mapping using graph traversal
    # Traverse: Layer → CONTAINS → Octahedron
    oct_to_layer = {}
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'layer':
            layer_id = node.replace('layer_', '')
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if (edge_data and
                    edge_data.get('edge_type') == 'contains' and
                    neighbor.startswith('octahedron_')):
                    oct_to_layer[neighbor] = layer_id

    # Validate that we found layer-octahedron connections
    if not oct_to_layer:
        import warnings
        warnings.warn(
            "No layer-octahedron connections found in graph. "
            "All octahedra will be assigned layer='unknown'. "
            "This may indicate a graph construction issue.",
            UserWarning
        )

    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())
    atom_symbols = analyzer.cell.get_chemical_symbols()

    # Get pre-computed X atom data from cache (computed once, reused for all octahedra)
    b_x_data = analyzer.get_b_x_atoms()
    all_x_indices = b_x_data['x_indices']
    all_x_positions = b_x_data['x_positions']

    if len(all_x_indices) < 6:
        # Not enough X atoms in structure - return empty results
        return {}

    results = {}

    for oct_node in selected_octahedra_nodes:
        # Get B atom using graph traversal
        central_idx = _get_b_atom_from_octahedron_node(graph, oct_node)

        if central_idx is None:
            continue

        # Reconstruct octahedron using PBC images
        # The graph may only show 5 atoms (e.g., "1 axial, 4 equatorial") because
        # some atoms are in different periodic images. We need to find the 6 nearest
        # X atoms using PBC to reconstruct the full octahedron.
        central_pos = atom_positions[central_idx]

        # Find 6 nearest X atoms using PBC images
        # This will find the correct periodic images to reconstruct the full octahedron
        # Even if the graph only shows 5 atoms, this will find all 6 including PBC images
        nearest_x_indices, nearest_x_positions, nearest_distances, image_labels = find_nearest_image_positions(
            reference_position=central_pos,
            candidate_positions=all_x_positions,
            candidate_indices=all_x_indices,
            cell=cell,
            n_neighbors=6,
            pbc=True,
        )

        # Use the reconstructed positions (with PBC images)
        # These positions are already in the correct PBC image relative to the B atom
        x_positions = nearest_x_positions
        bond_vectors = x_positions - central_pos

        # Get geometry classification from graph BEFORE computing bond lengths
        # This allows us to separate axial and equatorial bonds
        geometry = {}
        for x_idx in nearest_x_indices:
            atom_node = f"atom_{x_idx}"
            node_data = graph.nodes.get(atom_node, {})

            # Check if atom is axial (can be terminal, interlayer, or just axial)
            is_axial = node_data.get('is_axial', False)
            is_terminal = node_data.get('is_terminal', False)
            is_interlayer = node_data.get('is_interlayer', False)

            if is_axial or is_interlayer or is_terminal:
                # Axial atom: classify as terminal or interlayer
                if is_terminal:
                    geometry[x_idx] = 'axial_terminal'
                else:
                    geometry[x_idx] = 'axial_interlayer'
            else:
                # Equatorial atom
                geometry[x_idx] = 'equatorial'

        # Fallback: if graph gave no equatorial (e.g. graph geometry not set), classify by c-axis alignment
        # Axial = along c (2 largest |dot(bond_vec, c_hat)|), equatorial = in ab-plane (4 smallest)
        n_equatorial_from_graph = sum(1 for g in geometry.values() if g == 'equatorial')
        if n_equatorial_from_graph == 0 and len(nearest_x_indices) == 6:
            c_vec = np.array(cell[2], dtype=np.float64)
            c_norm = np.linalg.norm(c_vec)
            c_hat = c_vec / c_norm if c_norm >= 1e-10 else np.array([0.0, 0.0, 1.0])
            alignments = []
            for i in range(len(bond_vectors)):
                vec = bond_vectors[i]
                norm = np.linalg.norm(vec) + 1e-9
                alignment = np.abs(np.dot(vec / norm, c_hat))
                alignments.append((i, nearest_x_indices[i], alignment))
            alignments.sort(key=lambda x: x[2], reverse=True)
            for k, (_, x_idx, _) in enumerate(alignments):
                geometry[x_idx] = 'axial_interlayer' if k < 2 else 'equatorial'

        # Compute bond lengths separated by geometry
        bond_lengths_axial = []
        bond_lengths_equatorial = []
        all_bond_lengths = []

        for i, x_idx in enumerate(nearest_x_indices):
            bond_length = np.linalg.norm(bond_vectors[i])
            all_bond_lengths.append(bond_length)

            geom_type = geometry.get(x_idx, 'equatorial')
            if geom_type.startswith('axial'):
                bond_lengths_axial.append(bond_length)
            else:
                bond_lengths_equatorial.append(bond_length)

        bond_lengths_array = np.array(all_bond_lengths)
        bond_lengths_axial_array = np.array(bond_lengths_axial)
        bond_lengths_equatorial_array = np.array(bond_lengths_equatorial)

        # Compute X-B-X angles (using reconstructed positions)
        angles = []
        for i in range(len(x_positions)):
            for j in range(i + 1, len(x_positions)):
                vec1 = x_positions[i] - central_pos
                vec2 = x_positions[j] - central_pos
                angle = calculate_angle_between_vectors(vec1, vec2, cell=cell, apply_pbc=False)
                angles.append(angle)

        angles_array = np.array(angles)

        # Compute mean bond lengths (separate for axial and equatorial)
        mean_bond_length = float(np.mean(bond_lengths_array))
        mean_bond_length_axial = float(np.mean(bond_lengths_axial_array)) if len(bond_lengths_axial_array) > 0 else None
        mean_bond_length_equatorial = float(np.mean(bond_lengths_equatorial_array)) if len(bond_lengths_equatorial_array) > 0 else None

        # Compute distortion parameters (using all bonds for consistency with literature)
        delta_param = np.mean(np.abs(bond_lengths_array - mean_bond_length)) / mean_bond_length
        sigma_param = np.var(bond_lengths_array) / (mean_bond_length ** 2)

        # Lambda: variance from ideal angles (90° or 180°)
        angle_deviations = []
        for angle in angles_array:
            dev_from_90 = abs(angle - 90.0)
            dev_from_180 = abs(angle - 180.0)
            min_dev = min(dev_from_90, dev_from_180)
            angle_deviations.append(min_dev)

        lambda_param = np.var(angle_deviations)
        mean_angle = float(np.mean(angles_array))

        # Compute octahedral volume
        try:
            oct_volume = compute_octahedral_volume(
                central_pos=central_pos,
                x_positions=nearest_x_positions,
                x_indices=nearest_x_indices,
                geometry=geometry
            )
        except ValueError as e:
            import warnings
            warnings.warn(f"Volume calculation failed for {oct_node}: {e}")
            oct_volume = 0.0

        results[oct_node] = {
            'delta': float(delta_param),
            'sigma': float(sigma_param),
            'lambda': float(lambda_param),
            'bond_lengths': bond_lengths_array,
            'bond_lengths_axial': bond_lengths_axial_array,
            'bond_lengths_equatorial': bond_lengths_equatorial_array,
            'bond_angles': angles_array,
            'mean_bond_length': mean_bond_length,
            'mean_bond_length_axial': mean_bond_length_axial,
            'mean_bond_length_equatorial': mean_bond_length_equatorial,
            'mean_angle': mean_angle,
            'volume': float(oct_volume),
            'layer': oct_to_layer.get(oct_node, 'unknown'),
            'central_atom_index': central_idx,
            'central_atom_symbol': atom_symbols[central_idx] if central_idx is not None else None,
            'geometry': geometry,
        }

    return results


def _aggregate_bond_length_metrics(
    oct_data_list: List[Dict],
) -> Dict[str, Union[float, np.ndarray, None]]:
    """
    Aggregate bond length metrics from a list of octahedra data.

    Computes separate means for axial and equatorial bonds, as well as
    distortion parameters (delta, sigma, lambda).

    Parameters
    ----------
    oct_data_list : list of dict
        List of per-octahedron metrics dictionaries

    Returns
    -------
    dict
        Aggregated metrics with axial/equatorial separation
    """
    all_bond_lengths = []
    all_bond_lengths_axial = []
    all_bond_lengths_equatorial = []
    all_angles = []

    for metrics in oct_data_list:
        all_bond_lengths.extend(metrics['bond_lengths'])
        all_bond_lengths_axial.extend(metrics['bond_lengths_axial'])
        all_bond_lengths_equatorial.extend(metrics['bond_lengths_equatorial'])
        all_angles.extend(metrics['bond_angles'])

    bond_lengths_array = np.array(all_bond_lengths)
    bond_lengths_axial_array = np.array(all_bond_lengths_axial)
    bond_lengths_equatorial_array = np.array(all_bond_lengths_equatorial)
    angles_array = np.array(all_angles)

    # Guard against empty arrays
    if len(bond_lengths_array) == 0:
        return {
            'delta': None,
            'sigma': None,
            'lambda': None,
            'bond_lengths': bond_lengths_array,
            'bond_lengths_axial': bond_lengths_axial_array,
            'bond_lengths_equatorial': bond_lengths_equatorial_array,
            'bond_angles': angles_array,
            'mean_bond_length': None,
            'mean_bond_length_axial': None,
            'mean_bond_length_equatorial': None,
            'mean_angle': None,
        }

    mean_bond_length = float(np.mean(bond_lengths_array))

    # Guard against zero or NaN mean_bond_length
    if mean_bond_length == 0 or np.isnan(mean_bond_length):
        return {
            'delta': None,
            'sigma': None,
            'lambda': None,
            'bond_lengths': bond_lengths_array,
            'bond_lengths_axial': bond_lengths_axial_array,
            'bond_lengths_equatorial': bond_lengths_equatorial_array,
            'bond_angles': angles_array,
            'mean_bond_length': None,
            'mean_bond_length_axial': None,
            'mean_bond_length_equatorial': None,
            'mean_angle': None,
        }

    # Compute separate means for axial and equatorial
    mean_bond_length_axial = float(np.mean(bond_lengths_axial_array)) if len(bond_lengths_axial_array) > 0 else None
    mean_bond_length_equatorial = float(np.mean(bond_lengths_equatorial_array)) if len(bond_lengths_equatorial_array) > 0 else None

    # Compute distortion parameters (using all bonds)
    delta = float(np.mean(np.abs(bond_lengths_array - mean_bond_length)) / mean_bond_length)
    sigma = float(np.var(bond_lengths_array) / (mean_bond_length ** 2))

    # Handle angle calculations
    if len(angles_array) == 0:
        lambda_param = 0.0
        mean_angle = 0.0
    else:
        angle_deviations = []
        for angle in angles_array:
            dev_from_90 = abs(angle - 90.0)
            dev_from_180 = abs(angle - 180.0)
            min_dev = min(dev_from_90, dev_from_180)
            angle_deviations.append(min_dev)

        lambda_param = float(np.var(angle_deviations)) if len(angle_deviations) > 0 else 0.0
        mean_angle = float(np.mean(angles_array))

    return {
        'delta': delta,
        'sigma': sigma,
        'lambda': lambda_param,
        'bond_lengths': bond_lengths_array,
        'bond_lengths_axial': bond_lengths_axial_array,
        'bond_lengths_equatorial': bond_lengths_equatorial_array,
        'bond_angles': angles_array,
        'mean_bond_length': mean_bond_length,
        'mean_bond_length_axial': mean_bond_length_axial,
        'mean_bond_length_equatorial': mean_bond_length_equatorial,
        'mean_angle': mean_angle,
    }


def _compute_octahedral_distortions(
    analyzer,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Compute octahedral distortion parameters (Δ, σ, λ) from graph structure.

    Bond lengths are separated into axial and equatorial categories. Axial bonds
    are along the c-axis (terminal or interlayer X atoms), while equatorial bonds
    are in the ab-plane. These have significantly different lengths in layered
    perovskites.

    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzed structure with graph
    octahedra : list of str, optional
        Specific octahedra IDs to include
    layer : str, optional
        Single layer ID to filter by
    layers : list of str, optional
        Multiple layer IDs to filter by
    group_by : str, optional
        Group results by 'layer' or None for global

    Returns
    -------
    dict
        If group_by is None:
            Dictionary containing:
            - 'delta': Mean absolute deviation of B-X bond lengths (Δ)
            - 'sigma': Variance of B-X bond lengths (σ²)
            - 'lambda': Variance of X-B-X bond angles (λ²)
            - 'bond_lengths': Array of all B-X bond lengths
            - 'bond_lengths_axial': Array of axial B-X bond lengths
            - 'bond_lengths_equatorial': Array of equatorial B-X bond lengths
            - 'bond_angles': Array of all X-B-X bond angles
            - 'mean_bond_length': Mean of all B-X bond lengths
            - 'mean_bond_length_axial': Mean of axial B-X bond lengths
            - 'mean_bond_length_equatorial': Mean of equatorial B-X bond lengths
            - 'mean_angle': Mean X-B-X angle

        If group_by == 'layer':
            Dictionary with layer IDs as keys, each containing the above metrics,
            plus 'global' key with overall metrics
    """
    # Get per-octahedron detailed data
    oct_data = _get_octahedral_distortions_detailed(
        analyzer, octahedra=octahedra, layer=layer, layers=layers
    )

    if len(oct_data) == 0:
        empty_result = {
            'delta': None,
            'sigma': None,
            'lambda': None,
            'bond_lengths': np.array([]),
            'bond_lengths_axial': np.array([]),
            'bond_lengths_equatorial': np.array([]),
            'bond_angles': np.array([]),
            'mean_bond_length': None,
            'mean_bond_length_axial': None,
            'mean_bond_length_equatorial': None,
            'mean_angle': None,
        }
        if group_by == 'layer':
            return {'global': empty_result}
        return empty_result

    # If grouping by layer, compute per-layer metrics
    if group_by == 'layer':
        # Group octahedra by layer
        layer_octs = {}
        for oct_id, metrics in oct_data.items():
            layer_id = metrics['layer']
            if layer_id not in layer_octs:
                layer_octs[layer_id] = []
            layer_octs[layer_id].append(metrics)

        # Compute metrics for each layer
        results = {}
        for layer_id, layer_oct_list in layer_octs.items():
            # Skip 'unknown' layer or layers with no valid octahedra
            if layer_id == 'unknown' or not layer_oct_list:
                continue

            layer_metrics = _aggregate_bond_length_metrics(layer_oct_list)
            if layer_metrics['mean_bond_length'] is not None:
                results[layer_id] = layer_metrics

        # Also compute global metrics
        results['global'] = _aggregate_bond_length_metrics(list(oct_data.values()))

        return results

    # Otherwise, compute global metrics for selected octahedra
    return _aggregate_bond_length_metrics(list(oct_data.values()))
