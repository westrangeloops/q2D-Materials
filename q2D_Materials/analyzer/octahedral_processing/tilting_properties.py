"""
Tilting properties analysis for quasi-2D perovskites.

This module provides advanced tilting metrics for analyzing octahedral rotations
in layered perovskite structures:

- Mean Tilt Profile (μ_layer): Average tilt magnitude per layer
- Gearing Correlation (χ): Mechanical coupling between neighboring octahedra

These metrics are particularly useful for understanding depth-dependent distortions
and inter-octahedral coordination in quasi-2D (DJ, RP) perovskite phases.

Functions
---------
compute_mean_tilt_profile
    Compute mean tilt magnitude per layer/slab
compute_gearing_correlation
    Compute gearing correlation between neighboring octahedra
"""

from typing import Dict, Tuple, Optional, List, Union
import numpy as np
from scipy.spatial.transform import Rotation


def compute_mean_tilt_profile(
    analyzer,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    group_by: Optional[str] = None,
) -> Union[float, Dict[str, float]]:
    """
    Compute mean tilt magnitude profile.

    Formula: μ = (1/N) Σ ||T_i|| where ||T|| = sqrt(α² + β² + γ²)

    Measures the average tilting magnitude, optionally grouped by layer/slab.
    Most useful for 2D layered structures (DJ, RP) to quantify surface
    relaxation effects and depth-dependent distortions.

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
    group_by : str, optional
        If 'layer' or 'slab', returns dict with per-layer/slab results plus 'global' key
        If None, returns single global value

    Returns
    -------
    float or dict
        If group_by is None: Mean tilt magnitude (float, degrees)
        If group_by == 'layer': Dict with layer IDs as keys plus 'global' key
        If group_by == 'slab': Dict with slab IDs as keys plus 'global' key

    Examples
    --------
    >>> analyzer = q2D_analyzer("rp_structure.vasp")
    >>> analyzer.analyze()
    >>>
    >>> # Global mean tilt
    >>> mu_global = analyzer.compute_mean_tilt_profile()
    >>> print(f"Global mean tilt: {mu_global:.2f}°")
    >>>
    >>> # Per-layer profile
    >>> profile = analyzer.compute_mean_tilt_profile(group_by='layer')
    >>> for layer_id, mu in profile.items():
    ...     print(f"Layer {layer_id}: μ = {mu:.2f}°")
    >>>
    >>> # Single layer
    >>> mu_layer0 = analyzer.compute_mean_tilt_profile(layer='0')
    >>> print(f"Layer 0 mean tilt: {mu_layer0:.2f}°")

    Notes
    -----
    Physical Interpretation:
    - **Surface layers**: Higher μ indicates enhanced tilting due to missing coordination
    - **Central layers**: Lower μ, closer to bulk behavior
    - **Depth profile**: Reveals "surface relaxation" in DJ/RP phases

    Example for n=3 DJ structure:
    - Layer 0 (surface):  μ ≈ 15.4°  ← Enhanced tilting
    - Layer 1 (center):   μ ≈ 12.1°  ← Bulk-like
    - Layer 2 (surface):  μ ≈ 15.2°  ← Enhanced tilting
    """
    # 1. Get tilts for filtered octahedra
    from .tilt_calculations import compute_octahedral_tilts
    tilt_data = compute_octahedral_tilts(analyzer, octahedra=octahedra)

    # 2. Build octahedron → layer/slab mapping from graph
    graph = analyzer.get_graph()
    oct_to_group = {}

    # Build mapping if group_by is specified, OR if layer/layers filtering is needed
    needs_mapping = (group_by in ['layer', 'slab']) or (layer is not None) or (layers is not None)
    
    if needs_mapping:
        # Determine node type: prefer 'layer' if group_by specifies it, otherwise default to 'layer'
        if group_by == 'slab':
            node_type = 'slab'
        else:
            node_type = 'layer'  # Default to 'layer' for filtering
        
        for node, data in graph.nodes(data=True):
            if data.get('node_type') == node_type:
                group_id = node.replace(f'{node_type}_', '')
                for neighbor in graph.neighbors(node):
                    edge_data = graph.get_edge_data(node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'contains':
                        if neighbor.startswith('octahedron_'):
                            oct_to_group[neighbor] = group_id

    # 3. Filter by layer/layers if specified
    filtered_oct_ids = tilt_data.octahedron_ids
    if layer is not None:
        filtered_oct_ids = [oid for oid in filtered_oct_ids if oct_to_group.get(oid) == layer]
    elif layers is not None:
        filtered_oct_ids = [oid for oid in filtered_oct_ids if oct_to_group.get(oid) in layers]

    # 4. Compute magnitudes
    magnitudes = []
    group_magnitudes = {}

    for i, oct_id in enumerate(tilt_data.octahedron_ids):
        if oct_id not in filtered_oct_ids:
            continue
        if tilt_data.valid_mask is not None and not tilt_data.valid_mask[i]:
            continue

        # Tilt magnitude: ||T|| = sqrt(α² + β² + γ²)
        magnitude = np.linalg.norm(tilt_data.euler_angles[i])
        magnitudes.append(magnitude)

        if group_by in ['layer', 'slab']:
            group_id = oct_to_group.get(oct_id)
            if group_id:
                if group_id not in group_magnitudes:
                    group_magnitudes[group_id] = []
                group_magnitudes[group_id].append(magnitude)

    # 5. Return based on group_by
    if group_by in ['layer', 'slab']:
        result = {'global': float(np.mean(magnitudes)) if magnitudes else None}
        for gid, mags in group_magnitudes.items():
            result[gid] = float(np.mean(mags)) if mags else None
        return result
    else:
        return float(np.mean(magnitudes)) if magnitudes else None


def compute_gearing_correlation(
    analyzer,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    group_by: Optional[str] = None,
    neighbor_criterion: str = 'shared_x',
) -> Union[Dict[Tuple[str, str], float], Dict[str, Dict[Tuple[str, str], float]]]:
    """
    Compute gearing correlation between neighboring octahedra.

    Formula: χ(i,j) = (R_i · R_j) / (||R_i|| ||R_j||)
    where R = rotation vector (axis-angle from rotation matrix)

    Measures mechanical coordination between corner-sharing octahedra.
    Quantifies cooperative vs. counter-rotating tilting patterns.

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
    group_by : str, optional
        If 'layer' or 'slab', returns dict with per-layer/slab results plus 'global' key
        If None, returns single dict of correlations
    neighbor_criterion : str, default='shared_x'
        How to define neighbors ('shared_x' for corner-sharing)

    Returns
    -------
    dict or dict of dicts
        If group_by is None: {(oct_i, oct_j): χ} where χ ∈ [-1, 1]
        If group_by == 'layer' or 'slab': {layer_id: {(oct_i, oct_j): χ}, 'global': {...}}

        χ interpretation:
        - χ ≈ +1: Cooperative rotation (like meshing gears)
        - χ ≈ -1: Counter-rotation
        - χ ≈ 0: Orthogonal rotations

    Examples
    --------
    >>> analyzer = q2D_analyzer("tilted_bulk.vasp")
    >>> analyzer.analyze()
    >>>
    >>> # Global correlations
    >>> gearing = analyzer.compute_gearing_correlation()
    >>> for (oct_i, oct_j), chi in gearing.items():
    ...     if chi > 0.5:
    ...         print(f"{oct_i} ↔ {oct_j}: Cooperative (χ = {chi:.3f})")
    >>>
    >>> # Per-layer grouped correlations
    >>> gearing_by_layer = analyzer.compute_gearing_correlation(group_by='layer')
    >>> for layer_id, pairs in gearing_by_layer.items():
    ...     if layer_id != 'global':
    ...         avg_chi = np.mean(list(pairs.values()))
    ...         print(f"Layer {layer_id} avg gearing: {avg_chi:.3f}")
    >>>
    >>> # Single layer
    >>> layer0_gearing = analyzer.compute_gearing_correlation(layer='0')
    >>> print(f"Layer 0 has {len(layer0_gearing)} octahedral pairs")

    Notes
    -----
    Physical Interpretation:
    - **χ ≈ +1**: Octahedra rotate cooperatively (like meshing gears), structurally stable
    - **χ ≈ -1**: Counter-rotation, destabilizing
    - **χ ≈ 0**: Decoupled rotations, no mechanical constraint
    - **Interlayer χ**: Measures coupling across organic spacer (typically low)

    Example for tilted perovskite:
    - Intralayer pairs:  χ ≈ +0.85  ← Strong mechanical coupling
    - Interlayer pairs:  χ ≈ -0.15  ← Weak coupling across spacer
    """
    # 1. Get tilts for filtered octahedra
    from .tilt_calculations import compute_octahedral_tilts
    tilt_data = compute_octahedral_tilts(analyzer, octahedra=octahedra)

    # 2. Build octahedron → layer/slab mapping
    graph = analyzer.get_graph()
    oct_to_group = {}

    # Build mapping if group_by is specified, OR if layer/layers filtering is needed
    needs_mapping = (group_by in ['layer', 'slab']) or (layer is not None) or (layers is not None)
    
    if needs_mapping:
        # Determine node type: prefer 'layer' if group_by specifies it, otherwise default to 'layer'
        if group_by == 'slab':
            node_type = 'slab'
        else:
            node_type = 'layer'  # Default to 'layer' for filtering
        
        for node, data in graph.nodes(data=True):
            if data.get('node_type') == node_type:
                group_id = node.replace(f'{node_type}_', '')
                for neighbor in graph.neighbors(node):
                    edge_data = graph.get_edge_data(node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'contains':
                        if neighbor.startswith('octahedron_'):
                            oct_to_group[neighbor] = group_id

    # 3. Filter by layer/layers if specified
    filtered_oct_ids = set(tilt_data.octahedron_ids)
    if layer is not None:
        filtered_oct_ids = {oid for oid in filtered_oct_ids if oct_to_group.get(oid) == layer}
    elif layers is not None:
        filtered_oct_ids = {oid for oid in filtered_oct_ids if oct_to_group.get(oid) in layers}

    # 4. Find neighbor pairs via shared X atoms
    from .octahedral_detection import find_shared_atoms
    octahedra_list = analyzer.get_octahedra()
    neighbor_indices = [oct['ligand_atoms'] for oct in octahedra_list]
    shared_atoms = find_shared_atoms(neighbor_indices)

    # 5. Compute correlations
    correlations = {}
    group_correlations = {}

    for (i, j) in shared_atoms.keys():
        if i == j:  # Skip self-sharing
            continue

        oct_i_id = tilt_data.octahedron_ids[i]
        oct_j_id = tilt_data.octahedron_ids[j]

        # Skip if either octahedron filtered out
        if oct_i_id not in filtered_oct_ids or oct_j_id not in filtered_oct_ids:
            continue

        # Convert rotation matrices to rotation vectors
        Rmat_i = tilt_data.rotation_matrices[i]
        Rmat_j = tilt_data.rotation_matrices[j]

        rot_vec_i = Rotation.from_matrix(Rmat_i).as_rotvec()
        rot_vec_j = Rotation.from_matrix(Rmat_j).as_rotvec()

        # Normalized dot product
        norm_i = np.linalg.norm(rot_vec_i)
        norm_j = np.linalg.norm(rot_vec_j)

        if norm_i > 1e-6 and norm_j > 1e-6:
            chi = np.dot(rot_vec_i, rot_vec_j) / (norm_i * norm_j)
        else:
            chi = 0.0  # No rotation

        correlations[(oct_i_id, oct_j_id)] = float(chi)

        # Group if needed
        if group_by in ['layer', 'slab']:
            group_i = oct_to_group.get(oct_i_id)
            group_j = oct_to_group.get(oct_j_id)

            # Intralayer/intraslab correlation
            if group_i == group_j and group_i is not None:
                if group_i not in group_correlations:
                    group_correlations[group_i] = {}
                group_correlations[group_i][(oct_i_id, oct_j_id)] = float(chi)
            # Interlayer/interslab correlation
            elif group_i != group_j and group_i is not None and group_j is not None:
                interlayer_key = f'interlayer_{group_i}_{group_j}'
                if interlayer_key not in group_correlations:
                    group_correlations[interlayer_key] = {}
                group_correlations[interlayer_key][(oct_i_id, oct_j_id)] = float(chi)

    # 6. Return based on group_by
    if group_by in ['layer', 'slab']:
        result = {'global': correlations}
        result.update(group_correlations)
        return result
    else:
        return correlations
