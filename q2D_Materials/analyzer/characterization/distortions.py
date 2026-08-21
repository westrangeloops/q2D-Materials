"""Octahedral distortion calculations.

This module provides methods for computing octahedral distortion parameters
using the graph-based analyzer.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ase.data import atomic_numbers, covalent_radii
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


def _reconstruct_validated_octahedron(
    analyzer,
    central_idx: int,
    n_neighbors: int = 6,
    cutoff_scale: float = 1.45,
    cutoff_padding: float = 0.25,
) -> Tuple[Optional[Dict], str]:
    """Reconstruct one BX6 unit from periodic X images and validate its bonds.

    The same representation is used by bond, angle, distortion, and volume
    descriptors. Repeated crystallographic X indices are allowed when their
    periodic image labels differ, which is required for primitive cells.
    """
    b_x_data = analyzer.get_b_x_atoms()
    all_x_indices = np.asarray(b_x_data["x_indices"], dtype=int)
    all_x_positions = np.asarray(b_x_data["x_positions"], dtype=float)
    if len(all_x_indices) == 0:
        return None, "no_x_candidates"

    atom_positions = analyzer.cell.get_positions()
    atom_symbols = analyzer.cell.get_chemical_symbols()
    cell = np.asarray(analyzer.cell.get_cell(), dtype=float)
    central_pos = atom_positions[central_idx]

    x_indices, x_positions, distances, image_labels = find_nearest_image_positions(
        reference_position=central_pos,
        candidate_positions=all_x_positions,
        candidate_indices=all_x_indices,
        cell=cell,
        n_neighbors=n_neighbors,
        pbc=True,
    )
    if len(x_indices) != n_neighbors:
        return None, "insufficient_periodic_x_images"
    if not np.all(np.isfinite(distances)) or np.any(distances <= 1e-8):
        return None, "invalid_bond_distances"

    image_keys = {
        (int(idx), *(int(value) for value in image))
        for idx, image in zip(x_indices, image_labels)
    }
    if len(image_keys) != n_neighbors:
        return None, "duplicate_periodic_x_images"

    b_symbol = atom_symbols[central_idx]
    b_radius = covalent_radii[atomic_numbers[b_symbol]]
    cutoffs = []
    for x_idx in x_indices:
        x_symbol = atom_symbols[int(x_idx)]
        x_radius = covalent_radii[atomic_numbers[x_symbol]]
        cutoffs.append(cutoff_scale * (b_radius + x_radius) + cutoff_padding)
    if np.any(distances > np.asarray(cutoffs)):
        return None, "bond_distance_exceeds_chemical_cutoff"

    bond_vectors = x_positions - central_pos
    plane_normal = np.cross(cell[0], cell[1])
    norm = np.linalg.norm(plane_normal)
    if norm < 1e-10:
        plane_normal = cell[2]
        norm = np.linalg.norm(plane_normal)
    plane_normal = plane_normal / norm
    alignments = np.abs(
        np.asarray(
            [
                np.dot(vector / np.linalg.norm(vector), plane_normal)
                for vector in bond_vectors
            ]
        )
    )
    axial_positions = set(np.argsort(alignments)[-2:].tolist())
    geometry_labels = [
        "axial" if i in axial_positions else "equatorial"
        for i in range(n_neighbors)
    ]

    return {
        "indices": x_indices,
        "positions": x_positions,
        "distances": distances,
        "image_labels": image_labels,
        "bond_vectors": bond_vectors,
        "geometry_labels": geometry_labels,
    }, "valid"


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


def _octahedral_distortions_cache_key(
    octahedra: Optional[List[str]],
    layer: Optional[str],
    layers: Optional[List[str]],
) -> tuple:
    """Hashable selection key for distortion-detail caching."""
    oct_key = tuple(sorted(octahedra)) if octahedra is not None else None
    layers_key = tuple(sorted(layers)) if layers is not None else None
    return (oct_key, layer, layers_key)


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

    Results are cached on ``analyzer`` for the selection key within one
    analyze session (cleared on ``load`` / ``analyze``).

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
    cache_key = _octahedral_distortions_cache_key(octahedra, layer, layers)
    cache = getattr(analyzer, "_octahedral_distortions_cache", None)
    if cache is None:
        cache = {}
        analyzer._octahedral_distortions_cache = cache
    if cache_key in cache:
        return cache[cache_key]

    result = _compute_octahedral_distortions_detailed(
        analyzer, octahedra=octahedra, layer=layer, layers=layers
    )
    cache[cache_key] = result
    return result


def _compute_octahedral_distortions_detailed(
    analyzer,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Union[float, np.ndarray, str]]]:
    """Uncached implementation of per-octahedron distortion metrics."""
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

    cell = np.array(analyzer.cell.get_cell())
    atom_symbols = analyzer.cell.get_chemical_symbols()

    results = {}

    for oct_node in selected_octahedra_nodes:
        central_idx = _get_b_atom_from_octahedron_node(graph, oct_node)
        if central_idx is None:
            results[oct_node] = {
                "status": "failed",
                "status_reason": "missing_central_atom",
                "bond_lengths": np.array([]),
                "bond_lengths_axial": np.array([]),
                "bond_lengths_equatorial": np.array([]),
                "bond_angles": np.array([]),
                "volume": np.nan,
                "layer": oct_to_layer.get(oct_node, "unknown"),
                "central_atom_index": None,
                "central_atom_symbol": None,
                "geometry": {},
            }
            continue

        reconstructed, reason = _reconstruct_validated_octahedron(
            analyzer, central_idx
        )
        if reconstructed is None:
            results[oct_node] = {
                "status": "not_applicable",
                "status_reason": reason,
                "bond_lengths": np.array([]),
                "bond_lengths_axial": np.array([]),
                "bond_lengths_equatorial": np.array([]),
                "bond_angles": np.array([]),
                "volume": np.nan,
                "layer": oct_to_layer.get(oct_node, "unknown"),
                "central_atom_index": central_idx,
                "central_atom_symbol": atom_symbols[central_idx],
                "geometry": {},
            }
            continue

        nearest_x_indices = reconstructed["indices"]
        nearest_x_positions = reconstructed["positions"]
        image_labels = reconstructed["image_labels"]
        bond_vectors = reconstructed["bond_vectors"]
        geometry_labels = reconstructed["geometry_labels"]
        central_pos = analyzer.cell.get_positions()[central_idx]

        bond_lengths_array = np.asarray(reconstructed["distances"], dtype=float)
        bond_lengths_axial_array = np.asarray(
            [
                distance
                for distance, label in zip(bond_lengths_array, geometry_labels)
                if label == "axial"
            ],
            dtype=float,
        )
        bond_lengths_equatorial_array = np.asarray(
            [
                distance
                for distance, label in zip(bond_lengths_array, geometry_labels)
                if label == "equatorial"
            ],
            dtype=float,
        )

        # Compute X-B-X angles (using reconstructed positions)
        angles = []
        for i in range(len(nearest_x_positions)):
            for j in range(i + 1, len(nearest_x_positions)):
                vec1 = nearest_x_positions[i] - central_pos
                vec2 = nearest_x_positions[j] - central_pos
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

        volume_geometry = {
            i: (
                "axial_interlayer"
                if label == "axial"
                else "equatorial"
            )
            for i, label in enumerate(geometry_labels)
        }
        try:
            oct_volume = compute_octahedral_volume(
                central_pos=central_pos,
                x_positions=nearest_x_positions,
                x_indices=list(range(6)),
                geometry=volume_geometry,
            )
        except ValueError as e:
            import warnings
            warnings.warn(f"Volume calculation failed for {oct_node}: {e}")
            oct_volume = np.nan

        geometry = {
            f"{int(x_idx)}@{','.join(str(int(v)) for v in image)}": label
            for x_idx, image, label in zip(
                nearest_x_indices, image_labels, geometry_labels
            )
        }

        results[oct_node] = {
            'status': 'valid',
            'status_reason': 'valid',
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
            'periodic_x_indices': nearest_x_indices,
            'periodic_image_labels': image_labels,
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
    total_count = len(oct_data_list)
    oct_data_list = [
        metrics
        for metrics in oct_data_list
        if metrics.get("status") == "valid"
        and len(metrics.get("bond_lengths", [])) == 6
    ]
    valid_count = len(oct_data_list)
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
            'status': 'not_applicable' if total_count else 'failed',
            'valid_octahedra': valid_count,
            'total_octahedra': total_count,
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
            'status': 'failed',
            'valid_octahedra': valid_count,
            'total_octahedra': total_count,
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
        'status': 'valid' if valid_count == total_count else 'partial',
        'valid_octahedra': valid_count,
        'total_octahedra': total_count,
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
            'status': 'not_applicable',
            'valid_octahedra': 0,
            'total_octahedra': 0,
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
