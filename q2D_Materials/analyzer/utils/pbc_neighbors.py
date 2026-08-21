"""
Periodic boundary condition neighbor finding (now using unified API).

All functionality has been moved to q2D_Materials.utils.geometry.pbc_distances
for better performance and consistency. This module now contains only
backward-compatible wrappers.
"""

import numpy as np
from typing import Tuple, List, Optional

# Import from unified module
from ...utils.geometry.pbc_distances import (
    calculate_pbc_distances,
    find_nearest_neighbors as _find_nearest_neighbors_unified,
)


def get_27_image_offsets() -> np.ndarray:
    """
    Get the 27 offset vectors for periodic images.
    
    Returns array of shape (27, 3) with offsets [-1, 0, 1] in each dimension.
    """
    offsets = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                offsets.append([i, j, k])
    return np.array(offsets, dtype=np.float64)


def find_nearest_neighbors_pbc(
    center_position: np.ndarray,
    candidate_positions: np.ndarray,
    candidate_indices: np.ndarray,
    cell: np.ndarray,
    n_neighbors: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find N nearest neighbors considering periodic boundary conditions.
    
    This is now a wrapper around the unified find_nearest_neighbors() function.
    
    Parameters
    ----------
    center_position : np.ndarray
        Position of center atom (3,)
    candidate_positions : np.ndarray
        Positions of candidate neighbor atoms (M, 3)
    candidate_indices : np.ndarray
        Original indices of candidate atoms (M,)
    cell : np.ndarray
        Unit cell matrix (3, 3) - rows are cell vectors
    n_neighbors : int
        Number of nearest neighbors to find
    
    Returns
    -------
    neighbor_indices : np.ndarray
        Indices of N nearest neighbors
    neighbor_distances : np.ndarray
        Distances to N nearest neighbors in Angstroms
    """
    return _find_nearest_neighbors_unified(
        center_position,
        candidate_positions,
        cell,
        n_neighbors,
        pbc=True,
        return_indices=candidate_indices
    )


def find_all_octahedral_neighbors(
    atom_positions: np.ndarray,
    atom_symbols: List[str],
    cell: np.ndarray,
    b_site_elements: List[str] = ['Pb', 'Sn'],
    x_site_elements: List[str] = ['Br', 'I', 'Cl'],
    n_neighbors: int = 6,
) -> dict:
    """
    Find 6 nearest X-site neighbors for each B-site atom.
    
    Parameters
    ----------
    atom_positions : np.ndarray
        All atom positions (N, 3)
    atom_symbols : list
        All atom symbols
    cell : np.ndarray
        Unit cell matrix (3, 3)
    b_site_elements : list
        B-site element symbols
    x_site_elements : list
        X-site element symbols
    n_neighbors : int
        Number of neighbors to find (default: 6 for octahedra)
    
    Returns
    -------
    dict
        Mapping of B-site index -> (neighbor_indices, neighbor_distances)
    """
    atom_positions = np.asarray(atom_positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Identify B-sites and X-sites
    b_site_indices = [i for i, sym in enumerate(atom_symbols) if sym in b_site_elements]
    x_site_indices = np.array([i for i, sym in enumerate(atom_symbols) if sym in x_site_elements])
    
    if len(x_site_indices) < n_neighbors:
        return {}
    
    x_site_positions = atom_positions[x_site_indices]
    
    results = {}
    for b_idx in b_site_indices:
        center_pos = atom_positions[b_idx]
        neighbor_indices, neighbor_distances = find_nearest_neighbors_pbc(
            center_pos, x_site_positions, x_site_indices, cell, n_neighbors
        )
        if neighbor_indices is not None:
            results[b_idx] = {
                'indices': neighbor_indices,
                'distances': neighbor_distances,
                'position': center_pos,
                'symbol': atom_symbols[b_idx],
            }
    
    return results


# Backward compatibility: numba-accelerated versions are now automatic
NUMBA_AVAILABLE = True  # Always true in unified module


def find_nearest_neighbors_pbc_fast(
    center_position: np.ndarray,
    candidate_positions: np.ndarray,
    candidate_indices: np.ndarray,
    cell: np.ndarray,
    n_neighbors: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find N nearest neighbors with automatic numba acceleration.
    
    This is now identical to find_nearest_neighbors_pbc() as the unified
    module handles numba optimization automatically.
    """
    return find_nearest_neighbors_pbc(
        center_position,
        candidate_positions,
        candidate_indices,
        cell,
        n_neighbors
    )
