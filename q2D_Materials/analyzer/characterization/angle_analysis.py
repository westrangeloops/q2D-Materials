"""
B-X-B angle calculations for perovskite structures.

This module provides functions to calculate B-X-B bond angles using the
analyzer's graph structure.

Functions
---------
_calculate_bxb_angles
    Calculate B-X-B bond angles using analyzer graph structure
"""

from typing import Tuple, Optional
import numpy as np
from ..detection.octahedral_detection import find_shared_atoms
from ..utils.geometry_helpers import (
    apply_pbc_to_vector,
    calculate_angle_between_vectors,
    get_all_x_atoms_from_octahedron,
)


def _calculate_bxb_angles(
    analyzer,
    bxb_scale: float = 1.4,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    include_bp: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Calculate B-X-B angles using analyzer graph structure.

    Uses the analyzer's octahedra graph to identify B and X sites, then
    calculates angles for B-X-B triplets where X is shared between octahedra.

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

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        (bxb_angles, bxbp_angles) - arrays of angles in degrees
    """
    # Get octahedra from analyzer
    octahedra = analyzer.get_octahedra()
    if not octahedra:
        return None, None

    # Get graph to find shared atoms
    graph = analyzer.get_graph()
    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())

    # Build neighbor indices list for find_shared_atoms using shared utility
    neighbor_indices = []
    for oct in octahedra:
        all_neighbors = get_all_x_atoms_from_octahedron(oct)
        neighbor_indices.append(all_neighbors)

    # Find shared atoms between octahedra
    shared_atoms = find_shared_atoms(neighbor_indices)

    # Calculate B-X-B angles
    bxb_angles = []
    bxbp_angles = []

    # Get B-site indices (central atoms)
    b_site_indices = {oct["central_atom_index"] for oct in octahedra if oct.get("central_atom_index") is not None}

    # Get Bp sites if needed (spacer B-sites - not implemented yet, would need spacer analysis)
    bp_site_indices = set()
    if include_bp:
        # TODO: Identify Bp sites from spacers if needed
        pass

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

            # Get positions
            b_i_pos = atom_positions[b_i_idx]
            b_j_pos = atom_positions[b_j_idx]
            x_pos = atom_positions[x_idx]

            # Calculate vectors with PBC using shared utility
            vec_xi = x_pos - b_i_pos
            vec_xi = apply_pbc_to_vector(vec_xi, cell)

            vec_xj = x_pos - b_j_pos
            vec_xj = apply_pbc_to_vector(vec_xj, cell)

            # Calculate angle using shared utility
            angle = calculate_angle_between_vectors(vec_xi, vec_xj, cell=cell, apply_pbc=False)
            bxb_angles.append(angle)

            # Check for B-X-Bp angles if needed
            if include_bp and b_j_idx in bp_site_indices:
                angle_bp = calculate_angle_between_vectors(vec_xi, vec_xj, cell=cell, apply_pbc=False)
                bxbp_angles.append(angle_bp)

    bxb_array = np.array(bxb_angles) if bxb_angles else None
    bxbp_array = np.array(bxbp_angles) if bxbp_angles else None

    return bxb_array, bxbp_array

