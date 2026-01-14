"""Octahedral distortion calculations.

This module provides methods for computing octahedral distortion parameters
using the graph-based analyzer.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from ..utils.geometry_helpers import (
    apply_pbc_to_vector,
    calculate_angle_between_vectors,
    get_all_x_atoms_from_octahedron,
    extract_bx_bond_vectors,
)


def _filter_octahedra_by_selection(
    analyzer,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
) -> List[Dict]:
    """
    Filter octahedra based on selection criteria using graph structure.

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
    list of dict
        Filtered octahedra information
    """
    all_octahedra = analyzer.get_octahedra()

    # If no filters specified, return all
    if octahedra is None and layer is None and layers is None:
        return all_octahedra

    # Build layer -> octahedra mapping from graph
    layer_oct_map = {}
    layers_info = analyzer.get_layers()
    for layer_id, layer_data in layers_info.items():
        layer_oct_map[layer_id] = layer_data['octahedra']

    # Filter by specific octahedra IDs
    if octahedra is not None:
        octahedra_set = set(octahedra)
        return [oct for oct in all_octahedra if oct['id'] in octahedra_set]

    # Filter by single layer
    if layer is not None:
        if layer in layer_oct_map:
            allowed_ids = set(layer_oct_map[layer])
            return [oct for oct in all_octahedra if oct['id'] in allowed_ids]
        return []

    # Filter by multiple layers
    if layers is not None:
        allowed_ids = set()
        for layer_id in layers:
            if layer_id in layer_oct_map:
                allowed_ids.update(layer_oct_map[layer_id])
        return [oct for oct in all_octahedra if oct['id'] in allowed_ids]

    return all_octahedra


def _get_octahedral_distortions_detailed(
    analyzer,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Union[float, np.ndarray, str]]]:
    """
    Compute per-octahedron distortion metrics with layer information.

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
        - 'delta': Bond length distortion
        - 'sigma': Bond length variance
        - 'lambda': Bond angle variance
        - 'bond_lengths': Array of B-X bond lengths
        - 'bond_angles': Array of X-B-X angles
        - 'mean_bond_length': Mean B-X bond length
        - 'mean_angle': Mean X-B-X angle
        - 'layer': Layer ID this octahedron belongs to
        - 'central_atom_index': Index of B-site atom
        - 'central_atom_symbol': Element symbol of B-site
    """
    # Filter octahedra by selection
    selected_octahedra = _filter_octahedra_by_selection(
        analyzer, octahedra=octahedra, layer=layer, layers=layers
    )

    # Build octahedron -> layer mapping
    oct_to_layer = {}
    layers_info = analyzer.get_layers()
    for layer_id, layer_data in layers_info.items():
        for oct_id in layer_data['octahedra']:
            oct_to_layer[oct_id] = layer_id

    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())

    results = {}

    for oct in selected_octahedra:
        oct_id = oct['id']
        central_idx = oct['central_atom_index']

        if central_idx is None:
            continue

        # Get all X atoms around this B atom using shared utility
        x_atoms = get_all_x_atoms_from_octahedron(oct)

        if len(x_atoms) < 6:
            continue  # Skip incomplete octahedra

        # Extract B-X bond vectors with PBC using shared utility
        bond_vectors, x_positions = extract_bx_bond_vectors(
            oct, atom_positions, cell, apply_pbc=True, max_atoms=6
        )

        # Compute bond lengths from vectors
        bond_lengths = np.linalg.norm(bond_vectors, axis=1).tolist()

        # Compute X-B-X angles using shared utility
        central_pos = atom_positions[central_idx]
        angles = []
        for i in range(len(x_positions)):
            for j in range(i + 1, len(x_positions)):
                vec1 = x_positions[i] - central_pos
                vec2 = x_positions[j] - central_pos
                angle = calculate_angle_between_vectors(vec1, vec2, cell=cell, apply_pbc=False)
                angles.append(angle)

        bond_lengths_array = np.array(bond_lengths)
        angles_array = np.array(angles)

        # Compute distortion parameters for this octahedron
        mean_bond_length = np.mean(bond_lengths_array)
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
        mean_angle = np.mean(angles_array)

        results[oct_id] = {
            'delta': float(delta_param),
            'sigma': float(sigma_param),
            'lambda': float(lambda_param),
            'bond_lengths': bond_lengths_array,
            'bond_angles': angles_array,
            'mean_bond_length': float(mean_bond_length),
            'mean_angle': float(mean_angle),
            'layer': oct_to_layer.get(oct_id, 'unknown'),
            'central_atom_index': central_idx,
            'central_atom_symbol': oct['central_atom_symbol'],
        }

    return results


def _compute_octahedral_distortions(
    analyzer,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    group_by: Optional[str] = None,
) -> Dict[str, Union[float, np.ndarray, Dict]]:
    """
    Compute octahedral distortion parameters (Δ, σ, λ) from graph structure.

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
            - 'bond_angles': Array of all X-B-X bond angles
            - 'mean_bond_length': Mean B-X bond length
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
            'bond_angles': np.array([]),
            'mean_bond_length': None,
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
            layer_bond_lengths = []
            layer_angles = []

            for metrics in layer_oct_list:
                layer_bond_lengths.extend(metrics['bond_lengths'])
                layer_angles.extend(metrics['bond_angles'])

            bond_lengths_array = np.array(layer_bond_lengths)
            angles_array = np.array(layer_angles)

            mean_bond_length = np.mean(bond_lengths_array)
            delta = np.mean(np.abs(bond_lengths_array - mean_bond_length)) / mean_bond_length
            sigma = np.var(bond_lengths_array) / (mean_bond_length ** 2)

            angle_deviations = []
            for angle in angles_array:
                dev_from_90 = abs(angle - 90.0)
                dev_from_180 = abs(angle - 180.0)
                min_dev = min(dev_from_90, dev_from_180)
                angle_deviations.append(min_dev)

            lambda_param = np.var(angle_deviations)
            mean_angle = np.mean(angles_array)

            results[layer_id] = {
                'delta': float(delta),
                'sigma': float(sigma),
                'lambda': float(lambda_param),
                'bond_lengths': bond_lengths_array,
                'bond_angles': angles_array,
                'mean_bond_length': float(mean_bond_length),
                'mean_angle': float(mean_angle),
            }

        # Also compute global metrics
        all_bond_lengths = []
        all_angles = []
        for metrics in oct_data.values():
            all_bond_lengths.extend(metrics['bond_lengths'])
            all_angles.extend(metrics['bond_angles'])

        bond_lengths_array = np.array(all_bond_lengths)
        angles_array = np.array(all_angles)

        mean_bond_length = np.mean(bond_lengths_array)
        delta = np.mean(np.abs(bond_lengths_array - mean_bond_length)) / mean_bond_length
        sigma = np.var(bond_lengths_array) / (mean_bond_length ** 2)

        angle_deviations = []
        for angle in angles_array:
            dev_from_90 = abs(angle - 90.0)
            dev_from_180 = abs(angle - 180.0)
            min_dev = min(dev_from_90, dev_from_180)
            angle_deviations.append(min_dev)

        lambda_param = np.var(angle_deviations)
        mean_angle = np.mean(angles_array)

        results['global'] = {
            'delta': float(delta),
            'sigma': float(sigma),
            'lambda': float(lambda_param),
            'bond_lengths': bond_lengths_array,
            'bond_angles': angles_array,
            'mean_bond_length': float(mean_bond_length),
            'mean_angle': float(mean_angle),
        }

        return results

    # Otherwise, compute global metrics for selected octahedra
    all_bond_lengths = []
    all_angles = []

    for metrics in oct_data.values():
        all_bond_lengths.extend(metrics['bond_lengths'])
        all_angles.extend(metrics['bond_angles'])

    bond_lengths_array = np.array(all_bond_lengths)
    angles_array = np.array(all_angles)

    mean_bond_length = np.mean(bond_lengths_array)
    delta = np.mean(np.abs(bond_lengths_array - mean_bond_length)) / mean_bond_length
    sigma = np.var(bond_lengths_array) / (mean_bond_length ** 2)

    angle_deviations = []
    for angle in angles_array:
        dev_from_90 = abs(angle - 90.0)
        dev_from_180 = abs(angle - 180.0)
        min_dev = min(dev_from_90, dev_from_180)
        angle_deviations.append(min_dev)

    lambda_param = np.var(angle_deviations)
    mean_angle = np.mean(angles_array)

    return {
        'delta': float(delta),
        'sigma': float(sigma),
        'lambda': float(lambda_param),
        'bond_lengths': bond_lengths_array,
        'bond_angles': angles_array,
        'mean_bond_length': float(mean_bond_length),
        'mean_angle': float(mean_angle),
    }
