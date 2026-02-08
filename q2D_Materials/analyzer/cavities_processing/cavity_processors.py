"""Cavity processors for A-site and spacer (RP/DJ) cavity detection.

Orchestrates molecule reconstruction, B/X selection, subgraph building,
and Cavity object creation for each cavity type.
"""

import sys
import numpy as np
import networkx as nx
from typing import List, Dict, Any, Tuple, Optional
from scipy.spatial import ConvexHull

from .cavity_class import Cavity
from .weights import ASiteWeights, SpacerWeights
from .molecule_helpers import (
    _find_nh3_groups,
    _calculate_nh3_center,
    _calculate_nh3_center_from_reconstructed,
)
from .x_atom_selection import _get_cuboctahedron_x_atoms, _get_antiprism_x_atoms
from .subgraph_build import _build_subgraph
from .pbc_images import _find_nearest_from_cached_images
from ...utils.molecules.pbc_reconstruction import reconstruct_molecule_from_nh3, reconstruct_molecule_pbc


def _compute_hull_data(corner_positions: np.ndarray) -> Optional[Dict[str, Any]]:
    """Compute convex hull data for cavity volume/containment checks."""
    if len(corner_positions) < 4:
        return None
    
    hull = ConvexHull(corner_positions)
    
    return {
        'corner_atom_indices': list(range(len(corner_positions))),
        'corner_positions_unwrapped': corner_positions.tolist(),
        'hull_volume': float(hull.volume),
        'surface_area': float(hull.area),
        'hull_equations': hull.equations.tolist(),
        'hull_vertices': hull.vertices.tolist(),
    }


def _process_a_site(
    molecule_node_id: str,
    mol_indices: List[int],
    nh3_count: int,
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    b_positions: np.ndarray,
    b_indices: np.ndarray,
    x_positions: np.ndarray,
    x_indices: np.ndarray,
    cell: np.ndarray,
    cavity_id: int,
    b_image_positions: np.ndarray,
    b_image_indices: np.ndarray,
    b_image_labels: np.ndarray,
    x_image_positions: np.ndarray,
    x_image_indices: np.ndarray,
    x_image_labels: np.ndarray,
    weights: ASiteWeights,
    max_validation_iterations: int = 10
) -> Optional[Cavity]:
    """Process A-site molecule (atomic or molecular with NH3)."""
    if nh3_count == 0:
        initial_anchor = np.mean([atom_positions[i] for i in mol_indices], axis=0)
        mol_data = reconstruct_molecule_pbc(
            mol_indices, initial_anchor, graph, atom_positions, atom_symbols, cell
        )
        reconstructed_positions = np.array([mol_data[idx][0] for idx in mol_indices])
        anchor = np.mean(reconstructed_positions, axis=0)
        print(f"    Atomic A-site ({len(mol_indices)} atoms), anchor at centroid {anchor} (from reconstructed molecule)", file=sys.stderr)
    else:
        nh3_groups = _find_nh3_groups(graph, molecule_node_id)
        if not nh3_groups:
            print(f"    WARNING: Molecular A-site has nh3_count={nh3_count} but no NH3 groups found, skipping", file=sys.stderr)
            return None

        initial_anchor = _calculate_nh3_center(nh3_groups[0], atom_positions)
        mol_data = reconstruct_molecule_from_nh3(
            mol_indices, initial_anchor, graph, atom_positions, atom_symbols, cell
        )
        reconstructed_positions = [pos for pos, _ in mol_data.values()]
        anchor = np.mean(reconstructed_positions, axis=0)
        print(f"    Molecular A-site, anchor at reconstructed molecule geometric center {anchor}", file=sys.stderr)
        print(f"    (calculated from {len(reconstructed_positions)} reconstructed atom positions)", file=sys.stderr)
    
    b_data = _find_nearest_from_cached_images(
        anchor, b_image_positions, b_image_indices, b_image_labels,
        n_neighbors=8,
        exclude_indices=np.array(mol_indices, dtype=np.int32) if mol_indices else None
    )

    if len(b_data[0]) < 8:
        print(f"    WARNING: Not enough B atoms found ({len(b_data[0])} < 8)", file=sys.stderr)
        return None

    x_data = _get_cuboctahedron_x_atoms(graph, b_data[0], atom_positions, cell, anchor, weights, b_positions_unwrapped=b_data[1])

    if len(x_data[0]) < 12:
        print(f"    WARNING: Not enough X atoms found ({len(x_data[0])} < 12)", file=sys.stderr)
        return None

    cage_info = [{
        'nh3_group_idx': 0 if nh3_count > 0 else None,
        'b_indices': b_data[0],
        'x_indices': x_data[0],
        'is_complete': True
    }]
    subgraph = _build_subgraph(mol_data, b_data, x_data, graph, cell, molecule_node_id, cage_info, max_validation_iterations)
    
    if subgraph is None:
        print(f"    ERROR: Failed to build valid subgraph for A-site cavity. Skipping.", file=sys.stderr)
        return None

    center = np.mean(b_data[1], axis=0)
    hull_data = _compute_hull_data(x_data[1])

    cavity = Cavity(
        cavity_id=f'cavity_{cavity_id}',
        b_atom_indices=b_data[0].tolist(),
        x_atom_indices=x_data[0].tolist(),
        a_site_indices=mol_indices,
        pbc_coordinates={},
        subgraph=subgraph,
        center_position=center,
        octahedra_info={},
        contains_a_site=True,
        is_pbc_wrapped=False,
        hull_data=hull_data,
        cavity_type='a_site'
    )
    
    print(f"    ✓ Created A-site cavity (8B + 12X)", file=sys.stderr)
    return cavity


def _process_rp_spacer(
    molecule_node_id: str,
    mol_indices: List[int],
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    b_positions: np.ndarray,
    b_indices: np.ndarray,
    x_positions: np.ndarray,
    x_indices: np.ndarray,
    cell: np.ndarray,
    cavity_id: int,
    b_image_positions: np.ndarray,
    b_image_indices: np.ndarray,
    b_image_labels: np.ndarray,
    x_image_positions: np.ndarray,
    x_image_indices: np.ndarray,
    x_image_labels: np.ndarray,
    weights: SpacerWeights,
    max_validation_iterations: int = 10
) -> Optional[Cavity]:
    """Process RP spacer (1 NH3 group)."""
    nh3_groups = _find_nh3_groups(graph, molecule_node_id)
    if not nh3_groups:
        print(f"    WARNING: RP spacer has no NH3 groups", file=sys.stderr)
        return None
    
    initial_anchor = _calculate_nh3_center(nh3_groups[0], atom_positions)
    mol_data = reconstruct_molecule_from_nh3(
        mol_indices, initial_anchor, graph, atom_positions, atom_symbols, cell
    )
    anchor = _calculate_nh3_center(nh3_groups[0], atom_positions)
    
    print(f"    RP spacer, anchor at NH3 N atom {anchor} (wrapped position from original cell)", file=sys.stderr)
    
    b_data = _find_nearest_from_cached_images(
        anchor, b_image_positions, b_image_indices, b_image_labels,
        n_neighbors=4,
        exclude_indices=np.array(mol_indices, dtype=np.int32) if mol_indices else None
    )

    if len(b_data[0]) < 4:
        print(f"    WARNING: Not enough B atoms found ({len(b_data[0])} < 4)", file=sys.stderr)
        return None

    x_data = _get_antiprism_x_atoms(graph, b_data[0], atom_positions, cell,
                                     anchor, weights,
                                     b_positions_unwrapped=b_data[1])

    if x_data is None:
        return None
    
    if len(x_data[0]) < 8:
        print(f"    WARNING: Not enough X atoms found ({len(x_data[0])} < 8)", file=sys.stderr)
        return None

    cage_info = [{
        'nh3_group_idx': 0,
        'b_indices': b_data[0],
        'x_indices': x_data[0],
        'is_complete': False
    }]
    subgraph = _build_subgraph(mol_data, b_data, x_data, graph, cell, molecule_node_id, cage_info, max_validation_iterations)
    
    if subgraph is None:
        print(f"    ERROR: Failed to build valid subgraph for RP spacer cavity. Skipping.", file=sys.stderr)
        return None

    center = np.mean(b_data[1], axis=0)
    hull_data = _compute_hull_data(x_data[1])

    cavity = Cavity(
        cavity_id=f'cavity_{cavity_id}',
        b_atom_indices=b_data[0].tolist(),
        x_atom_indices=x_data[0].tolist(),
        a_site_indices=mol_indices,
        pbc_coordinates={},
        subgraph=subgraph,
        center_position=center,
        octahedra_info={},
        contains_a_site=True,
        is_pbc_wrapped=False,
        hull_data=hull_data,
        cavity_type='spacer_rp'
    )
    
    print(f"    ✓ Created RP spacer cavity (4B + 8X)", file=sys.stderr)
    return cavity


def _process_dj_spacer(
    molecule_node_id: str,
    mol_indices: List[int],
    graph: nx.Graph,
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    b_positions: np.ndarray,
    b_indices: np.ndarray,
    x_positions: np.ndarray,
    x_indices: np.ndarray,
    cell: np.ndarray,
    cavity_id: int,
    b_image_positions: np.ndarray,
    b_image_indices: np.ndarray,
    b_image_labels: np.ndarray,
    x_image_positions: np.ndarray,
    x_image_indices: np.ndarray,
    x_image_labels: np.ndarray,
    weights: SpacerWeights,
    max_validation_iterations: int = 10
) -> Optional[Cavity]:
    """Process DJ spacer (2+ NH3 groups)."""
    nh3_groups = _find_nh3_groups(graph, molecule_node_id)
    if len(nh3_groups) < 2:
        print(f"    WARNING: DJ spacer has only {len(nh3_groups)} NH3 groups", file=sys.stderr)
        return None
    
    initial_anchor1 = _calculate_nh3_center(nh3_groups[0], atom_positions)
    mol_data = reconstruct_molecule_from_nh3(
        mol_indices, initial_anchor1, graph, atom_positions, atom_symbols, cell
    )
    anchor1 = _calculate_nh3_center_from_reconstructed(nh3_groups[0], mol_data)
    anchor2 = _calculate_nh3_center_from_reconstructed(nh3_groups[1], mol_data)
    
    print(f"    DJ spacer, anchor1 at NH3_1 N atom {anchor1} (from reconstructed molecule - unwrapped)", file=sys.stderr)
    print(f"    DJ spacer, anchor2 at NH3_2 N atom {anchor2} (from reconstructed molecule - unwrapped)", file=sys.stderr)

    b1_data = _find_nearest_from_cached_images(
        anchor1, b_image_positions, b_image_indices, b_image_labels,
        n_neighbors=4,
        exclude_indices=np.array(mol_indices, dtype=np.int32) if mol_indices else None
    )

    if len(b1_data[0]) < 4:
        print(f"    WARNING: Not enough B atoms for cage 1", file=sys.stderr)
        return None

    x1_data = _get_antiprism_x_atoms(graph, b1_data[0], atom_positions, cell,
                                      anchor1, weights,
                                      b_positions_unwrapped=b1_data[1])

    if x1_data is None:
        return None
    
    if len(x1_data[0]) < 8:
        print(f"    WARNING: Not enough X atoms for cage 1 ({len(x1_data[0])} < 8)", file=sys.stderr)
        return None

    b2_data = _find_nearest_from_cached_images(
        anchor2, b_image_positions, b_image_indices, b_image_labels,
        n_neighbors=4,
        exclude_indices=np.array(mol_indices, dtype=np.int32) if mol_indices else None
    )

    if len(b2_data[0]) < 4:
        print(f"    WARNING: Not enough B atoms for cage 2", file=sys.stderr)
        return None

    x2_data = _get_antiprism_x_atoms(graph, b2_data[0], atom_positions, cell,
                                      anchor2, weights,
                                      b_positions_unwrapped=b2_data[1])

    if x2_data is None:
        return None
    
    if len(x2_data[0]) < 8:
        print(f"    WARNING: Not enough X atoms for cage 2 ({len(x2_data[0])} < 8)", file=sys.stderr)
        return None
    
    b_indices_merged = np.concatenate([b1_data[0], b2_data[0]])
    b_positions_merged = np.vstack([b1_data[1], b2_data[1]])
    b_distances_merged = np.concatenate([b1_data[2], b2_data[2]])
    b_labels_merged = np.vstack([b1_data[3], b2_data[3]])
    
    x_indices_merged = np.concatenate([x1_data[0], x2_data[0]])
    x_positions_merged = np.vstack([x1_data[1], x2_data[1]])
    x_distances_merged = np.concatenate([x1_data[2], x2_data[2]])
    x_labels_merged = np.vstack([x1_data[3], x2_data[3]])
    
    b_data_merged = (b_indices_merged, b_positions_merged, b_distances_merged, b_labels_merged)
    x_data_merged = (x_indices_merged, x_positions_merged, x_distances_merged, x_labels_merged)

    cage_info = [
        {'nh3_group_idx': 0, 'b_indices': b1_data[0], 'x_indices': x1_data[0], 'is_complete': False},
        {'nh3_group_idx': 1, 'b_indices': b2_data[0], 'x_indices': x2_data[0], 'is_complete': False}
    ]
    subgraph = _build_subgraph(mol_data, b_data_merged, x_data_merged, graph, cell, molecule_node_id, cage_info, max_validation_iterations)
    
    if subgraph is None:
        print(f"    ERROR: Failed to build valid subgraph for DJ spacer cavity. Skipping.", file=sys.stderr)
        return None
    
    center = np.mean(b_positions_merged, axis=0)
    
    hull1 = _compute_hull_data(x1_data[1])
    hull2 = _compute_hull_data(x2_data[1])
    if hull1 is not None and hull2 is not None:
        v1 = hull1['hull_volume']
        v2 = hull2['hull_volume']
        mean_halfcage_volume = (v1 + v2) / 2.0
        hull_data = hull1.copy()
        hull_data['hull_volume'] = mean_halfcage_volume
    else:
        hull_data = _compute_hull_data(x_positions_merged)
    
    cavity = Cavity(
        cavity_id=f'cavity_{cavity_id}',
        b_atom_indices=b_indices_merged.tolist(),
        x_atom_indices=x_indices_merged.tolist(),
        a_site_indices=mol_indices,
        pbc_coordinates={},
        subgraph=subgraph,
        center_position=center,
        octahedra_info={},
        contains_a_site=True,
        is_pbc_wrapped=False,
        hull_data=hull_data,
        cavity_type='spacer_dj'
    )
    
    print(f"    ✓ Created DJ spacer cavity (8B + 16X total, volume = mean of 2 half-cages)", file=sys.stderr)
    return cavity
