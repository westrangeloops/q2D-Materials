"""
B-X-B angle calculations for perovskite structures.

This module provides functions to calculate B-X-B bond angles using the
analyzer's graph structure.

Functions
---------
_calculate_bxb_angles
    Calculate B-X-B bond angles using analyzer graph structure
debug_bxb_angles
    Debug function to check if B-X-B angles can be calculated
"""

from typing import Tuple, Optional, List, Dict, Union
import numpy as np
from ..octahedral_processing.octahedral_detection import find_shared_atoms
from ..utils.geometry_helpers import (
    apply_pbc_to_vector,
    calculate_angle_between_vectors,
    get_all_x_atoms_from_octahedron,
    normalize_layer_id,
)
from ...utils.geometry.pbc_distances import find_nearest_image_positions
from .distortions import _filter_octahedra_by_selection


def debug_bxb_angles(analyzer) -> Dict[str, any]:
    """Debug function to check if B-X-B angles exist in structure.
    
    Returns diagnostic info about octahedral connectivity.
    """
    all_octahedra = analyzer.get_octahedra()
    graph = analyzer.get_graph()
    
    neighbor_indices = []
    for oct in all_octahedra:
        all_neighbors = get_all_x_atoms_from_octahedron(oct)
        neighbor_indices.append(all_neighbors)
    
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    # Count X atoms with properties
    x_with_properties = 0
    terminal_count = 0
    equatorial_count = 0
    interlayer_count = 0
    
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'atom' and data.get('vasp_index') is not None:
            if data.get('is_terminal') or data.get('is_equatorial') or data.get('is_interlayer'):
                x_with_properties += 1
                if data.get('is_terminal'):
                    terminal_count += 1
                if data.get('is_equatorial'):
                    equatorial_count += 1
                if data.get('is_interlayer'):
                    interlayer_count += 1
    
    return {
        'num_octahedra': len(all_octahedra),
        'num_shared_x_pairs': len(shared_atoms),
        'num_total_shared_x_angles': sum(len(v) for v in shared_atoms.values()),
        'x_atoms_with_properties': x_with_properties,
        'terminal_x_atoms': terminal_count,
        'equatorial_x_atoms': equatorial_count,
        'interlayer_x_atoms': interlayer_count,
        'shared_atom_pairs': list(shared_atoms.keys()),
    }


def _calculate_bxb_angles(
    analyzer,
    bxb_scale: float = 1.4,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    include_bp: bool = False,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    group_by: Optional[str] = None,
) -> Union[
    Tuple[Optional[np.ndarray], Optional[np.ndarray]],
    Dict[str, Union[Dict[str, Optional[np.ndarray]], Optional[np.ndarray]]]
]:
    """
    Calculate B-X-B angles using analyzer graph structure.

    Uses the analyzer's octahedra graph to identify B and X sites, then
    calculates angles for B-X-B triplets where X is shared between octahedra.
    
    Note: For layer-specific B-X-B analysis, use analyzer.layers.get_bxb() instead.
    This function is for structure-wide analysis only.

    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    bxb_scale : float
        Scale factor for bond cutoff (not used in graph-based approach, kept for API compatibility)
    supercell : Tuple[int, int, int]
        Supercell replication (not used in graph-based approach, kept for API compatibility)
    include_bp : bool
        Whether to calculate B-X-Bp angles (Bp = spacer B-site)
    octahedra : list of str, optional
        Specific octahedra IDs to include (e.g., ['octahedron_0', 'octahedron_1'])
        (DEPRECATED: layer-specific analysis should use analyzer.layers)
    layer : str, optional
        Single layer ID to filter by (DEPRECATED: use analyzer.layers.get_bxb(layer_id=...))
    layers : list of str, optional
        Multiple layer IDs to filter by (DEPRECATED: use analyzer.layers)
    group_by : str, optional
        Group results by 'layer' (DEPRECATED: use analyzer.layers)
        For backward compatibility, still supported but returns delegated to layers wrapper.

    Returns
    -------
    If group_by is None:
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        (bxb_angles, bxbp_angles) - arrays of angles in degrees
    
    If group_by == 'layer':
        Dict with keys:
        - 'global': Tuple of (bxb_angles, bxbp_angles) for all angles
        - layer IDs: Dict with 'bxb_angles' and 'bxbp_angles' for each layer
        - 'interlayer': Dict with 'bxb_angles' and 'bxbp_angles' for interlayer angles
        
        (Delegates to analyzer.layers for layer-specific analysis)
    """
    # Handle deprecated layer-specific parameters by delegating to layers wrapper
    if group_by == 'layer' or layer is not None or layers is not None:
        import warnings
        warnings.warn(
            "Layer-specific B-X-B analysis via get_bxb_angles() is deprecated. "
            "Use analyzer.layers.get_bxb() for intra-layer analysis or "
            "analyzer.layers.get_interlayer_bxb() for inter-layer analysis instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Delegate to layers wrapper for backward compatibility
        return _calculate_bxb_angles_grouped_by_layer(
            analyzer, bxb_scale, supercell, include_bp, octahedra, layer, layers, group_by
        )
    
    # Structure-wide B-X-B analysis (not grouped by layer)
    # Only calculates global angles, separated by geometry type (terminal/equatorial/interlayer)
    all_octahedra = analyzer.get_octahedra()
    if not all_octahedra:
        return None, None

    # Get graph and atomic positions
    graph = analyzer.get_graph()
    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())

    # Use all octahedra for structure-wide analysis
    octahedra = all_octahedra

    # Get cached B atom data for better performance
    b_x_data = analyzer.get_b_x_atoms()
    b_idx_to_pos = {b_idx: b_x_data['b_positions'][i] for i, b_idx in enumerate(b_x_data['b_indices'])}

    # Build neighbor indices list
    neighbor_indices = []
    for oct in octahedra:
        all_neighbors = get_all_x_atoms_from_octahedron(oct)
        neighbor_indices.append(all_neighbors)

    # Find shared atoms between octahedra
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    # Build X atom properties mapping from graph
    x_atom_properties = {}
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'atom':
            vasp_idx = data.get('vasp_index')
            if vasp_idx is not None:
                x_atom_properties[vasp_idx] = {
                    'is_terminal': data.get('is_terminal', False),
                    'is_equatorial': data.get('is_equatorial', False),
                    'is_interlayer': data.get('is_interlayer', False),
                }

    # Calculate B-X-B angles by geometry type
    # Note: Terminal X atoms (belonging to only 1 octahedron) cannot form B-X-B angles
    # Only equatorial (intra-layer) and interlayer (inter-layer) X atoms form B-X-B angles
    bxb_equatorial_angles = []
    bxb_interlayer_angles = []
    bxb_all_angles = []
    bxbp_angles = []

    # For each pair of octahedra sharing X-sites, calculate angles
    for (oct_i, oct_j), shared_x_indices in shared_atoms.items():
        if oct_i >= len(octahedra) or oct_j >= len(octahedra):
            continue

        oct_i_data = octahedra[oct_i]
        oct_j_data = octahedra[oct_j]

        b_i_idx = oct_i_data.get("central_atom_index")
        b_j_idx = oct_j_data.get("central_atom_index")

        if b_i_idx is None or b_j_idx is None:
            continue

        # Calculate angle for each shared X-site
        for x_idx in shared_x_indices:
            if x_idx >= len(atom_positions):
                continue

            # Get X atom geometry properties
            # Note: Only equatorial and interlayer X atoms can form B-X-B angles
            x_props = x_atom_properties.get(x_idx, {})
            is_equatorial = x_props.get('is_equatorial', False)
            is_interlayer = x_props.get('is_interlayer', False)

            x_pos = atom_positions[x_idx]

            # Find nearest periodic images of B_i and B_j
            b_i_base_pos = b_idx_to_pos.get(b_i_idx, atom_positions[b_i_idx])
            b_i_data = find_nearest_image_positions(
                reference_position=x_pos,
                candidate_positions=np.array([b_i_base_pos]),
                candidate_indices=np.array([b_i_idx]),
                cell=cell,
                n_neighbors=1,
                pbc=True,
            )
            _, b_i_nearest_pos, _, _ = b_i_data
            b_i_pos = b_i_nearest_pos[0]

            b_j_base_pos = b_idx_to_pos.get(b_j_idx, atom_positions[b_j_idx])
            b_j_data = find_nearest_image_positions(
                reference_position=x_pos,
                candidate_positions=np.array([b_j_base_pos]),
                candidate_indices=np.array([b_j_idx]),
                cell=cell,
                n_neighbors=1,
                pbc=True,
            )
            _, b_j_nearest_pos, _, _ = b_j_data
            b_j_pos = b_j_nearest_pos[0]

            # Calculate angle
            vec_xi = b_i_pos - x_pos
            vec_xj = b_j_pos - x_pos
            angle = calculate_angle_between_vectors(vec_xi, vec_xj, cell=cell, apply_pbc=False)

            # Classify angle by X-atom geometry (equatorial or interlayer only)
            bxb_all_angles.append(angle)
            if is_equatorial:
                bxb_equatorial_angles.append(angle)
            if is_interlayer:
                bxb_interlayer_angles.append(angle)

    # Return structure-wide angles decomposed by geometry
    return {
        'bxb_angles': np.array(bxb_all_angles) if bxb_all_angles else None,
        'bxb_equatorial_angles': np.array(bxb_equatorial_angles) if bxb_equatorial_angles else None,
        'bxb_interlayer_angles': np.array(bxb_interlayer_angles) if bxb_interlayer_angles else None,
    }, None


def _calculate_bxb_angles_grouped_by_layer(
    analyzer,
    bxb_scale: float = 1.4,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    include_bp: bool = False,
    octahedra: Optional[List[str]] = None,
    layer: Optional[str] = None,
    layers: Optional[List[str]] = None,
    group_by: Optional[str] = None,
) -> Dict:
    """
    DEPRECATED: Calculate B-X-B angles grouped by layer.
    
    This function is kept for backward compatibility. New code should use
    analyzer.layers.get_bxb() for intra-layer analysis or
    analyzer.layers.get_interlayer_bxb() for inter-layer analysis.
    
    Delegates to the Layers wrapper for layer-specific analysis.
    """
    from ..core.layer_analysis import get_intralayer_bxb, get_all_interlayer_bxb
    
    # Get all layers
    layers_dict = analyzer.get_layers()
    if not layers_dict:
        return {'global': (None, None)}
    
    # Collect all angles
    global_bxb_angles = []
    layer_bxb_angles = {}
    interlayer_bxb_angles = []
    
    # Get intra-layer angles for each layer
    for layer_id in sorted(layers_dict.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')):
        try:
            result = get_intralayer_bxb(analyzer, layer_id)
            if result['bxb_angles'] is not None and len(result['bxb_angles']) > 0:
                layer_bxb_angles[layer_id] = result['bxb_angles']
                global_bxb_angles.extend(result['bxb_angles'])
        except (ValueError, KeyError):
            continue
    
    # Get inter-layer angles
    all_interlayer_result = get_all_interlayer_bxb(analyzer)
    if all_interlayer_result['bxb_angles'] is not None and len(all_interlayer_result['bxb_angles']) > 0:
        interlayer_bxb_angles = all_interlayer_result['bxb_angles']
        global_bxb_angles.extend(interlayer_bxb_angles)
    
    # Build result dict
    result = {
        'global': (
            np.array(global_bxb_angles) if global_bxb_angles else None,
            None  # bxbp_angles not implemented
        )
    }
    
    # Add per-layer results
    for layer_id, angles in layer_bxb_angles.items():
        result[layer_id] = {
            'bxb_angles': angles,
            'bxbp_angles': None,
        }
    
    # Add interlayer results if any
    if len(interlayer_bxb_angles) > 0:
        result['interlayer'] = {
            'bxb_angles': interlayer_bxb_angles,
            'bxbp_angles': None,
        }
    
    return result
