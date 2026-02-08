"""Octahedral volume calculations using convex hull decomposition."""

from typing import Dict, List, Optional
import numpy as np
from scipy.spatial import ConvexHull
import warnings


def compute_octahedral_volume(
    central_pos: np.ndarray,
    x_positions: np.ndarray,
    x_indices: List[int],
    geometry: Dict[int, str]
) -> float:
    """
    Compute octahedral volume as sum of two pyramids.
    
    Each octahedron is decomposed into:
    - Pyramid 1: axial_1 (terminal or interlayer) + 4 equatorial atoms
    - Pyramid 2: axial_2 (terminal or interlayer) + 4 equatorial atoms
    
    Parameters
    ----------
    central_pos : np.ndarray
        Position of B-site central atom (not used in volume, but for validation)
    x_positions : np.ndarray
        Positions of 6 X-site atoms (PBC-corrected), shape (6, 3)
    x_indices : List[int]
        Indices of the 6 X-site atoms
    geometry : Dict[int, str]
        Mapping of atom_idx -> geometry label from octahedra_geometries
        Labels: 'axial_terminal', 'axial_interlayer', 'equatorial'
    
    Returns
    -------
    float
        Total octahedral volume in Å³
    
    Raises
    ------
    ValueError
        If geometry classification is invalid (not 2 axial + 4 equatorial)
    """
    # Classify atoms based on geometry dict
    axial_indices = []
    equatorial_indices = []
    
    for i, atom_idx in enumerate(x_indices):
        geom_label = geometry.get(atom_idx, 'unknown')
        
        if geom_label in ['axial_terminal', 'axial_interlayer']:
            axial_indices.append(i)
        elif geom_label == 'equatorial':
            equatorial_indices.append(i)
        else:
            raise ValueError(
                f"Unknown geometry label '{geom_label}' for atom {atom_idx}. "
                f"Expected 'axial_terminal', 'axial_interlayer', or 'equatorial'."
            )
    
    # Validate octahedral geometry: must have exactly 2 axial and 4 equatorial
    if len(axial_indices) != 2:
        raise ValueError(
            f"Invalid octahedron: found {len(axial_indices)} axial atoms, expected 2. "
            f"Axial atoms: {[x_indices[i] for i in axial_indices]}"
        )
    
    if len(equatorial_indices) != 4:
        raise ValueError(
            f"Invalid octahedron: found {len(equatorial_indices)} equatorial atoms, expected 4. "
            f"Equatorial atoms: {[x_indices[i] for i in equatorial_indices]}"
        )
    
    # Get positions
    axial_pos_1 = x_positions[axial_indices[0]]
    axial_pos_2 = x_positions[axial_indices[1]]
    equatorial_positions = x_positions[equatorial_indices]
    
    # Build two pyramids
    # Pyramid 1: axial_1 as apex, 4 equatorial as base
    pyramid1_points = np.vstack([axial_pos_1.reshape(1, 3), equatorial_positions])
    
    # Pyramid 2: axial_2 as apex, same 4 equatorial as base
    pyramid2_points = np.vstack([axial_pos_2.reshape(1, 3), equatorial_positions])
    
    # Compute volumes using ConvexHull
    try:
        hull1 = ConvexHull(pyramid1_points)
        hull2 = ConvexHull(pyramid2_points)
        
        total_volume = hull1.volume + hull2.volume
        
        return float(total_volume)
        
    except Exception as e:
        warnings.warn(
            f"ConvexHull failed for octahedron with atoms {x_indices}: {e}. "
            f"Returning 0.0 volume."
        )
        return 0.0


def compute_all_octahedral_volumes(
    analyzer,
    octahedra: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute volumes for all octahedra in the structure.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzed structure with graph
    octahedra : List[str], optional
        Specific octahedra IDs to compute. If None, compute all.
    
    Returns
    -------
    Dict[str, float]
        Mapping of octahedron_id -> volume (Å³)
    """
    from ..characterization.distortions import (
        _get_b_atom_from_octahedron_node,
        _filter_octahedra_by_selection
    )
    from ...utils.geometry.pbc_distances import find_nearest_image_positions
    
    graph = analyzer.get_graph()
    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())
    
    # Get all X atoms
    b_x_data = analyzer.get_b_x_atoms()
    all_x_indices = b_x_data['x_indices']
    all_x_positions = b_x_data['x_positions']
    
    if len(all_x_indices) < 6:
        return {}
    
    # Filter octahedra
    selected_octahedra = _filter_octahedra_by_selection(
        analyzer, octahedra=octahedra, layer=None, layers=None
    )
    
    volumes = {}
    
    for oct_node in selected_octahedra:
        # Get B atom
        central_idx = _get_b_atom_from_octahedron_node(graph, oct_node)
        if central_idx is None:
            continue
        
        central_pos = atom_positions[central_idx]
        
        # Find 6 nearest X atoms using PBC
        nearest_x_indices, nearest_x_positions, _, _ = find_nearest_image_positions(
            reference_position=central_pos,
            candidate_positions=all_x_positions,
            candidate_indices=all_x_indices,
            cell=cell,
            n_neighbors=6,
            pbc=True,
        )
        
        # Get geometry classification from graph
        geometry = {}
        for x_idx in nearest_x_indices:
            atom_node = f"atom_{x_idx}"
            node_data = graph.nodes.get(atom_node, {})
            
            # Check if atom is axial (can be terminal, interlayer, or just axial)
            is_axial = node_data.get('is_axial', False)
            is_terminal = node_data.get('is_terminal', False)
            is_interlayer = node_data.get('is_interlayer', False)
            
            if is_axial or is_interlayer:
                # Axial atom: classify as terminal or interlayer
                if is_terminal:
                    geometry[x_idx] = 'axial_terminal'
                elif is_interlayer:
                    geometry[x_idx] = 'axial_interlayer'
                else:
                    # Axial but not terminal or interlayer (shouldn't happen, but handle gracefully)
                    geometry[x_idx] = 'axial_interlayer'
            else:
                # Equatorial atom
                geometry[x_idx] = 'equatorial'
        
        # Compute volume
        try:
            volume = compute_octahedral_volume(
                central_pos=central_pos,
                x_positions=nearest_x_positions,
                x_indices=nearest_x_indices,
                geometry=geometry
            )
            volumes[oct_node] = volume
        except ValueError as e:
            warnings.warn(f"Failed to compute volume for {oct_node}: {e}")
            continue
    
    return volumes
