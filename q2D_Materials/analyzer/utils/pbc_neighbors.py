"""
Periodic boundary condition neighbor finding using 27-image approach.

This is a straightforward, efficient implementation that:
1. Creates 27 periodic images (including original)
2. Finds N nearest neighbors using Euclidean distance
3. Returns distances in Angstroms (directly interpretable)

Can be accelerated with numba if needed.
"""

import numpy as np
from typing import Tuple, List, Optional


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

    Uses the 27-image approach: for each candidate atom, considers all
    27 periodic images and finds the closest one to the center.

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
    offsets = get_27_image_offsets()
    n_candidates = len(candidate_positions)

    if n_candidates < n_neighbors:
        return None, None

    # For each candidate, find minimum distance across all 27 images
    min_distances = np.full(n_candidates, np.inf)

    for i, pos in enumerate(candidate_positions):
        for offset in offsets:
            # Translate position by offset * cell vectors
            translated = pos + offset @ cell
            dist = np.linalg.norm(translated - center_position)
            if dist < min_distances[i]:
                min_distances[i] = dist

    # Sort by distance and take N nearest
    sorted_idx = np.argsort(min_distances)
    nearest_idx = sorted_idx[:n_neighbors]

    return candidate_indices[nearest_idx], min_distances[nearest_idx]


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


# Optional: Numba-accelerated version for large structures
try:
    from numba import jit, prange

    @jit(nopython=True, cache=True)
    def _find_min_distance_numba(center: np.ndarray, candidate: np.ndarray, cell: np.ndarray) -> float:
        """Find minimum distance considering 27 periodic images (numba accelerated)."""
        min_dist = np.inf
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # Translate by offset * cell
                    translated = candidate.copy()
                    translated[0] += i * cell[0, 0] + j * cell[1, 0] + k * cell[2, 0]
                    translated[1] += i * cell[0, 1] + j * cell[1, 1] + k * cell[2, 1]
                    translated[2] += i * cell[0, 2] + j * cell[1, 2] + k * cell[2, 2]

                    dx = translated[0] - center[0]
                    dy = translated[1] - center[1]
                    dz = translated[2] - center[2]
                    dist = np.sqrt(dx*dx + dy*dy + dz*dz)

                    if dist < min_dist:
                        min_dist = dist
        return min_dist

    @jit(nopython=True, parallel=True, cache=True)
    def find_min_distances_batch_numba(
        center: np.ndarray,
        candidates: np.ndarray,
        cell: np.ndarray,
    ) -> np.ndarray:
        """Find minimum distances for all candidates (numba parallel)."""
        n = len(candidates)
        min_dists = np.empty(n, dtype=np.float64)
        for i in prange(n):
            min_dists[i] = _find_min_distance_numba(center, candidates[i], cell)
        return min_dists

    NUMBA_AVAILABLE = True

except ImportError:
    NUMBA_AVAILABLE = False


def find_nearest_neighbors_pbc_fast(
    center_position: np.ndarray,
    candidate_positions: np.ndarray,
    candidate_indices: np.ndarray,
    cell: np.ndarray,
    n_neighbors: int = 6,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find N nearest neighbors with optional numba acceleration.

    Falls back to pure numpy if numba is not available.
    """
    if NUMBA_AVAILABLE and len(candidate_positions) > 50:
        # Use numba for larger candidate sets
        min_distances = find_min_distances_batch_numba(
            center_position.astype(np.float64),
            candidate_positions.astype(np.float64),
            cell.astype(np.float64),
        )
    else:
        # Use pure numpy for small sets
        offsets = get_27_image_offsets()
        n_candidates = len(candidate_positions)
        min_distances = np.full(n_candidates, np.inf)

        for i, pos in enumerate(candidate_positions):
            for offset in offsets:
                translated = pos + offset @ cell
                dist = np.linalg.norm(translated - center_position)
                if dist < min_distances[i]:
                    min_distances[i] = dist

    if len(candidate_positions) < n_neighbors:
        return None, None

    sorted_idx = np.argsort(min_distances)
    nearest_idx = sorted_idx[:n_neighbors]

    return candidate_indices[nearest_idx], min_distances[nearest_idx]
