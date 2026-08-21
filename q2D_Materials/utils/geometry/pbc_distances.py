"""
Unified PBC-aware distance calculations for crystalline structures.

This module consolidates all distance calculation functionality across the codebase,
providing a single, robust, numba-optimized API that handles all use cases.

This replaces and unifies:
- utils.geometry.geometry._calculate_distances()
- utils.geometry.geometry._calculate_distances_with_extended_pbc()
- builders.populate._calculate_xy_pbc_distances()
- analyzer.utils.pbc_neighbors.find_nearest_neighbors_pbc()
- utils.geometry.structural_utils distance matrix functions

Features
--------
- Numba acceleration for large structures (automatic fallback to numpy)
- Support for full 3D PBC, XY-only PBC, or custom PBC axes
- Minimum image convention (fast) and extended search (robust)
- Distance matrices and nearest neighbor queries
- Consistent API and well-tested edge cases

Examples
--------
Basic distance calculation:

>>> import numpy as np
>>> from q2D_Materials.utils.geometry.pbc_distances import calculate_pbc_distances
>>> 
>>> cell = np.eye(3) * 10.0  # 10 Å cubic cell
>>> ref = np.array([0.0, 0.0, 0.0])
>>> atoms = np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
>>> distances = calculate_pbc_distances(ref, atoms, cell)
>>> distances  # Both are 1 Å away due to PBC
array([1., 1.])

Finding nearest neighbors:

>>> from q2D_Materials.utils.geometry.pbc_distances import find_nearest_neighbors
>>> center = np.array([5.0, 5.0, 5.0])
>>> candidates = np.random.rand(100, 3) * 10.0
>>> indices, distances = find_nearest_neighbors(center, candidates, cell, n_neighbors=6)
>>> len(indices)  # Returns 6 nearest neighbors
6
"""

import numpy as np
from typing import Union, List, Tuple, Optional

# Try to import numba for acceleration
try:
    from numba import jit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


# ============================================================================
# Numba-accelerated kernels
# ============================================================================

if NUMBA_AVAILABLE:
    @jit(nopython=True, cache=True)
    def _pbc_distance_numba(
        ref: np.ndarray,
        target: np.ndarray,
        cell: np.ndarray,
        pbc_axes: np.ndarray  # [1, 1, 1] or [1, 1, 0] for XY-only
    ) -> float:
        """
        Core numba distance calculation with configurable PBC axes.
        
        Searches 27 images (or 9 for XY-only) to find minimum distance.
        """
        min_dist = np.inf
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    # Skip Z images if pbc_axes[2] == 0
                    if k != 0 and pbc_axes[2] == 0:
                        continue
                    
                    # Skip Y images if pbc_axes[1] == 0
                    if j != 0 and pbc_axes[1] == 0:
                        continue
                    
                    # Skip X images if pbc_axes[0] == 0
                    if i != 0 and pbc_axes[0] == 0:
                        continue
                    
                    # Translate target by periodic offset
                    translated = target.copy()
                    translated[0] += i * cell[0, 0] + j * cell[1, 0] + k * cell[2, 0]
                    translated[1] += i * cell[0, 1] + j * cell[1, 1] + k * cell[2, 1]
                    translated[2] += i * cell[0, 2] + j * cell[1, 2] + k * cell[2, 2]
                    
                    dx = translated[0] - ref[0]
                    dy = translated[1] - ref[1]
                    dz = translated[2] - ref[2]
                    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if dist < min_dist:
                        min_dist = dist
        return min_dist

    @jit(nopython=True, parallel=True, cache=True)
    def _pbc_distances_batch_numba(
        ref: np.ndarray,           # Shape: (3,)
        targets: np.ndarray,       # Shape: (N, 3)
        cell: np.ndarray,          # Shape: (3, 3)
        pbc_axes: np.ndarray       # Shape: (3,) with 0/1 values
    ) -> np.ndarray:
        """Vectorized numba distance calculation with parallel execution."""
        n = len(targets)
        distances = np.empty(n, dtype=np.float64)
        for i in prange(n):
            distances[i] = _pbc_distance_numba(ref, targets[i], cell, pbc_axes)
        return distances

    @jit(nopython=True, parallel=True, cache=True)
    def _pbc_distance_matrix_numba(
        positions_a: np.ndarray,   # Shape: (M, 3)
        positions_b: np.ndarray,   # Shape: (N, 3)
        cell: np.ndarray,
        pbc_axes: np.ndarray
    ) -> np.ndarray:
        """Full distance matrix with numba parallel execution."""
        m = len(positions_a)
        n = len(positions_b)
        matrix = np.empty((m, n), dtype=np.float64)
        for i in prange(m):
            for j in range(n):
                matrix[i, j] = _pbc_distance_numba(positions_a[i], positions_b[j], cell, pbc_axes)
        return matrix

    @jit(nopython=True, cache=True)
    def _pbc_distance_with_vector_numba(
        ref: np.ndarray,
        target: np.ndarray,
        cell: np.ndarray,
        pbc_axes: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Find minimum distance and corresponding vector."""
        min_dist = np.inf
        min_vec = np.zeros(3, dtype=np.float64)
        
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if k != 0 and pbc_axes[2] == 0:
                        continue
                    if j != 0 and pbc_axes[1] == 0:
                        continue
                    if i != 0 and pbc_axes[0] == 0:
                        continue
                    
                    translated = target.copy()
                    translated[0] += i * cell[0, 0] + j * cell[1, 0] + k * cell[2, 0]
                    translated[1] += i * cell[0, 1] + j * cell[1, 1] + k * cell[2, 1]
                    translated[2] += i * cell[0, 2] + j * cell[1, 2] + k * cell[2, 2]
                    
                    dx = translated[0] - ref[0]
                    dy = translated[1] - ref[1]
                    dz = translated[2] - ref[2]
                    dist = np.sqrt(dx*dx + dy*dy + dz*dz)
                    
                    if dist < min_dist:
                        min_dist = dist
                        min_vec[0] = dx
                        min_vec[1] = dy
                        min_vec[2] = dz
        
        return min_dist, min_vec


# ============================================================================
# Pure numpy fallback implementations
# ============================================================================

def _pbc_distance_numpy(
    ref: np.ndarray,
    target: np.ndarray,
    cell: np.ndarray,
    pbc_axes: np.ndarray
) -> float:
    """Pure numpy implementation of PBC distance (fallback when numba unavailable)."""
    min_dist = np.inf
    
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if k != 0 and pbc_axes[2] == 0:
                    continue
                if j != 0 and pbc_axes[1] == 0:
                    continue
                if i != 0 and pbc_axes[0] == 0:
                    continue
                
                offset = np.array([i, j, k], dtype=np.float64)
                translated = target + offset @ cell
                dist = np.linalg.norm(translated - ref)
                
                if dist < min_dist:
                    min_dist = dist
    
    return min_dist


def _pbc_distances_batch_numpy(
    ref: np.ndarray,
    targets: np.ndarray,
    cell: np.ndarray,
    pbc_axes: np.ndarray
) -> np.ndarray:
    """Pure numpy batch distance calculation."""
    n = len(targets)
    distances = np.empty(n, dtype=np.float64)
    
    for idx in range(n):
        distances[idx] = _pbc_distance_numpy(ref, targets[idx], cell, pbc_axes)
    
    return distances


# ============================================================================
# Minimum Image Convention (Fast Path)
# ============================================================================

def _calculate_distances_minimum_image(
    reference_atom: np.ndarray,
    atom_list: np.ndarray,
    cell: np.ndarray,
    pbc_axes: np.ndarray
) -> np.ndarray:
    """
    Fast minimum image convention using vectorized numpy operations.
    
    This is ~10x faster than 27-image search but may miss neighbors at
    cell boundaries in small/distorted cells.
    """
    ref_coord = np.asarray(reference_atom, dtype=np.float64)
    atom_coords = np.asarray(atom_list, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Calculate inverse cell
    try:
        inv_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        # Singular cell - fall back to Euclidean distance
        diff_vectors = atom_coords - ref_coord
        return np.linalg.norm(diff_vectors, axis=1)
    
    # Vectorized PBC wrapping
    diff_vectors = atom_coords - ref_coord
    diff_frac = diff_vectors @ inv_cell.T
    
    # Apply PBC only to specified axes
    for axis_idx in range(3):
        if pbc_axes[axis_idx] == 1:
            diff_frac[:, axis_idx] = diff_frac[:, axis_idx] - np.round(diff_frac[:, axis_idx])
    
    pbc_diff_vectors = diff_frac @ cell
    distances = np.linalg.norm(pbc_diff_vectors, axis=1)
    
    return distances


# ============================================================================
# Public API Functions
# ============================================================================

def calculate_pbc_distances(
    reference_positions: np.ndarray,
    target_positions: np.ndarray,
    cell: np.ndarray,
    pbc: Union[bool, List[bool]] = True,
    mode: str = 'auto',
    return_vectors: bool = False,
    use_numba: Optional[bool] = None
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Universal PBC-aware distance calculation with numba acceleration.
    
    This function handles all distance calculation scenarios in the codebase,
    automatically selecting the best algorithm based on inputs.
    
    Parameters
    ----------
    reference_positions : np.ndarray
        Reference position(s). Shape: (3,) for single reference or (M, 3) for multiple.
    target_positions : np.ndarray
        Target positions to calculate distances to. Shape: (N, 3).
    cell : np.ndarray
        Unit cell matrix with cell vectors as rows. Shape: (3, 3).
    pbc : bool or list of bool, default=True
        Periodic boundary conditions:
        - True: Apply PBC in all directions [True, True, True]
        - False: No PBC [False, False, False]
        - [True, True, False]: XY-only PBC (for monolayers/slabs)
    mode : str, default='auto'
        Distance calculation mode:
        - 'auto': Use minimum_image for cells > 20 Å, extended otherwise
        - 'minimum_image': Fast, may miss neighbors at boundaries (best for large cells)
        - 'extended': Robust 27-image search (best for small cells < 15 Å)
    return_vectors : bool, default=False
        If True, return (distances, vectors) tuple instead of just distances.
        Vectors point from reference to target positions (PBC-wrapped).
    use_numba : bool or None, default=None
        Force numba on/off. If None, uses numba automatically when available
        and beneficial (large target sets).
    
    Returns
    -------
    distances : np.ndarray
        If reference_positions is (3,): returns shape (N,)
        If reference_positions is (M, 3): returns shape (M, N) distance matrix
    vectors : np.ndarray, optional
        Only returned if return_vectors=True. Same shape as distances but
        with extra dimension for 3D vectors.
    
    Examples
    --------
    Single reference to multiple targets:
    
    >>> import numpy as np
    >>> cell = np.eye(3) * 10.0
    >>> ref = np.array([0.0, 0.0, 0.0])
    >>> targets = np.array([[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
    >>> dists = calculate_pbc_distances(ref, targets, cell)
    >>> dists
    array([1., 1.])
    
    XY-only PBC for monolayer:
    
    >>> dists = calculate_pbc_distances(ref, targets, cell, pbc=[True, True, False])
    
    Multiple references (distance matrix):
    
    >>> refs = np.array([[0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])
    >>> dists = calculate_pbc_distances(refs, targets, cell)
    >>> dists.shape
    (2, 2)
    
    With distance vectors:
    
    >>> dists, vecs = calculate_pbc_distances(ref, targets, cell, return_vectors=True)
    >>> vecs.shape
    (2, 3)
    """
    # Input validation and conversion
    reference_positions = np.asarray(reference_positions, dtype=np.float64)
    target_positions = np.asarray(target_positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Handle single reference position
    single_reference = reference_positions.ndim == 1
    if single_reference:
        reference_positions = reference_positions.reshape(1, 3)
    
    # Ensure target_positions is 2D
    if target_positions.ndim == 1:
        target_positions = target_positions.reshape(1, 3)
    
    # Parse PBC specification
    if isinstance(pbc, bool):
        pbc_axes = np.array([1, 1, 1] if pbc else [0, 0, 0], dtype=np.int32)
    else:
        pbc_axes = np.array([1 if p else 0 for p in pbc], dtype=np.int32)
    
    # Determine calculation mode
    if mode == 'auto':
        # Use minimum image for large cells, extended for small cells
        cell_lengths = np.linalg.norm(cell, axis=1)
        min_length = np.min(cell_lengths[pbc_axes == 1]) if np.any(pbc_axes) else float('inf')
        mode = 'minimum_image' if min_length > 20.0 else 'extended'
    
    # Decide whether to use numba
    if use_numba is None:
        # Use numba for extended mode with many targets, or large distance matrices
        n_refs = len(reference_positions)
        n_targets = len(target_positions)
        use_numba_auto = NUMBA_AVAILABLE and (
            (mode == 'extended' and n_targets > 50) or
            (n_refs > 1 and n_targets > 20)
        )
        use_numba = use_numba_auto
    
    # Calculate distances based on mode
    if mode == 'minimum_image':
        # Fast vectorized path (no numba needed)
        if len(reference_positions) == 1:
            distances = _calculate_distances_minimum_image(
                reference_positions[0], target_positions, cell, pbc_axes
            )
        else:
            # Distance matrix using minimum image
            m, n = len(reference_positions), len(target_positions)
            distances = np.empty((m, n), dtype=np.float64)
            for i in range(m):
                distances[i, :] = _calculate_distances_minimum_image(
                    reference_positions[i], target_positions, cell, pbc_axes
                )
        
        # Vectors not supported in minimum image mode (would need recomputation)
        if return_vectors:
            raise NotImplementedError(
                "return_vectors=True not supported with mode='minimum_image'. "
                "Use mode='extended' for distance vectors."
            )
    
    elif mode == 'extended':
        # Extended search (27-image or 9-image for XY)
        if len(reference_positions) == 1:
            # Single reference
            ref = reference_positions[0]
            
            if use_numba and NUMBA_AVAILABLE:
                distances = _pbc_distances_batch_numba(ref, target_positions, cell, pbc_axes)
            else:
                distances = _pbc_distances_batch_numpy(ref, target_positions, cell, pbc_axes)
            
            # Compute vectors if requested
            if return_vectors:
                if use_numba and NUMBA_AVAILABLE:
                    # Use numba version that returns both
                    vectors = np.empty((len(target_positions), 3), dtype=np.float64)
                    for i in range(len(target_positions)):
                        _, vectors[i] = _pbc_distance_with_vector_numba(
                            ref, target_positions[i], cell, pbc_axes
                        )
                else:
                    # Recompute with vectors (fallback)
                    vectors = np.empty((len(target_positions), 3), dtype=np.float64)
                    for i in range(len(target_positions)):
                        _, vectors[i] = _compute_min_distance_and_vector_numpy(
                            ref, target_positions[i], cell, pbc_axes
                        )
        else:
            # Multiple references (distance matrix)
            if use_numba and NUMBA_AVAILABLE:
                distances = _pbc_distance_matrix_numba(
                    reference_positions, target_positions, cell, pbc_axes
                )
            else:
                m, n = len(reference_positions), len(target_positions)
                distances = np.empty((m, n), dtype=np.float64)
                for i in range(m):
                    distances[i, :] = _pbc_distances_batch_numpy(
                        reference_positions[i], target_positions, cell, pbc_axes
                    )
            
            if return_vectors:
                raise NotImplementedError(
                    "return_vectors=True not supported for distance matrices. "
                    "Use single reference position instead."
                )
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'auto', 'minimum_image', or 'extended'.")
    
    # Reshape output for single reference
    if single_reference:
        distances = distances.flatten()
        if return_vectors:
            # vectors already has correct shape
            pass
    
    if return_vectors:
        return distances, vectors
    else:
        return distances


def _compute_min_distance_and_vector_numpy(
    ref: np.ndarray,
    target: np.ndarray,
    cell: np.ndarray,
    pbc_axes: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Helper function to compute minimum distance and vector (numpy fallback)."""
    min_dist = np.inf
    min_vec = np.zeros(3, dtype=np.float64)
    
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                if k != 0 and pbc_axes[2] == 0:
                    continue
                if j != 0 and pbc_axes[1] == 0:
                    continue
                if i != 0 and pbc_axes[0] == 0:
                    continue
                
                offset = np.array([i, j, k], dtype=np.float64)
                translated = target + offset @ cell
                vec = translated - ref
                dist = np.linalg.norm(vec)
                
                if dist < min_dist:
                    min_dist = dist
                    min_vec = vec
    
    return min_dist, min_vec


def find_nearest_neighbors(
    reference_position: np.ndarray,
    candidate_positions: np.ndarray,
    cell: np.ndarray,
    n_neighbors: int,
    pbc: Union[bool, List[bool]] = True,
    return_indices: Optional[np.ndarray] = None,
    use_numba: Optional[bool] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find K nearest neighbors with PBC (numba-accelerated).
    
    This function uses the extended 27-image search to ensure robustness
    for octahedral detection and other neighbor-finding tasks.
    
    Parameters
    ----------
    reference_position : np.ndarray
        Position of center atom. Shape: (3,).
    candidate_positions : np.ndarray
        Positions of candidate neighbor atoms. Shape: (M, 3).
    cell : np.ndarray
        Unit cell matrix. Shape: (3, 3).
    n_neighbors : int
        Number of nearest neighbors to find.
    pbc : bool or list of bool, default=True
        Periodic boundary conditions (same as calculate_pbc_distances).
    return_indices : np.ndarray or None, default=None
        If provided, returns these indices (mapped from candidates) instead
        of 0-based candidate indices. Useful for subset searches.
    use_numba : bool or None, default=None
        Force numba on/off. Auto-detects if None.
    
    Returns
    -------
    neighbor_indices : np.ndarray
        Indices of N nearest neighbors (shape: (n_neighbors,)).
        If return_indices provided, maps to those indices.
        Returns None if insufficient candidates.
    distances : np.ndarray
        Distances to N nearest neighbors in Angstroms (shape: (n_neighbors,)).
        Returns None if insufficient candidates.
    
    Examples
    --------
    >>> import numpy as np
    >>> cell = np.eye(3) * 10.0
    >>> center = np.array([5.0, 5.0, 5.0])
    >>> candidates = np.random.rand(100, 3) * 10.0
    >>> indices, distances = find_nearest_neighbors(center, candidates, cell, n_neighbors=6)
    >>> len(indices)
    6
    
    With original indices:
    
    >>> # Find neighbors among subset of atoms
    >>> x_site_indices = np.array([10, 15, 20, 25, 30, 35])  # Original indices
    >>> x_site_positions = all_positions[x_site_indices]
    >>> neighbor_orig_idx, dists = find_nearest_neighbors(
    ...     center, x_site_positions, cell, 6, return_indices=x_site_indices
    ... )
    >>> neighbor_orig_idx  # Returns original indices from full structure
    array([10, 15, 20, 25, 30, 35])
    """
    candidate_positions = np.asarray(candidate_positions, dtype=np.float64)
    reference_position = np.asarray(reference_position, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    n_candidates = len(candidate_positions)
    
    # Check if we have enough candidates
    if n_candidates < n_neighbors:
        return None, None
    
    # Calculate distances using extended search (robust for boundaries)
    distances = calculate_pbc_distances(
        reference_position,
        candidate_positions,
        cell,
        pbc=pbc,
        mode='extended',
        use_numba=use_numba
    )
    
    # Sort by distance and take N nearest
    sorted_idx = np.argsort(distances)
    nearest_idx = sorted_idx[:n_neighbors]
    nearest_distances = distances[nearest_idx]
    
    # Map to original indices if provided
    if return_indices is not None:
        return_indices = np.asarray(return_indices)
        nearest_idx = return_indices[nearest_idx]
    
    return nearest_idx, nearest_distances


def distance_matrix_pbc(
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    cell: np.ndarray,
    pbc: Union[bool, List[bool]] = True,
    mode: str = 'auto',
    return_vectors: bool = False,
    use_numba: Optional[bool] = None
) -> np.ndarray:
    """
    Compute full distance matrix between two sets of positions with PBC.
    
    This replaces distance_matrix(), distance_matrix_ase(), and related functions
    from structural_utils.py.
    
    Parameters
    ----------
    positions_a : np.ndarray
        First set of positions. Shape: (M, 3).
    positions_b : np.ndarray
        Second set of positions. Shape: (N, 3).
    cell : np.ndarray
        Unit cell matrix. Shape: (3, 3).
    pbc : bool or list of bool, default=True
        Periodic boundary conditions.
    mode : str, default='auto'
        Calculation mode ('auto', 'minimum_image', or 'extended').
    return_vectors : bool, default=False
        If True, return distance vectors instead of scalar distances.
        Not supported for distance matrices yet.
    use_numba : bool or None, default=None
        Force numba on/off.
    
    Returns
    -------
    distance_matrix : np.ndarray
        Matrix of distances. Shape: (M, N).
        Element [i, j] is distance from positions_a[i] to positions_b[j].
    
    Examples
    --------
    >>> import numpy as np
    >>> cell = np.eye(3) * 10.0
    >>> pos_a = np.array([[0, 0, 0], [5, 5, 5]])
    >>> pos_b = np.array([[1, 0, 0], [9, 0, 0], [5, 6, 5]])
    >>> dmat = distance_matrix_pbc(pos_a, pos_b, cell)
    >>> dmat.shape
    (2, 3)
    >>> dmat[0, 0]  # Distance from pos_a[0] to pos_b[0]
    1.0
    """
    return calculate_pbc_distances(
        positions_a,
        positions_b,
        cell,
        pbc=pbc,
        mode=mode,
        return_vectors=return_vectors,
        use_numba=use_numba
    )


# ============================================================================
# Convenience functions for common use cases
# ============================================================================

def calculate_xy_pbc_distances(
    reference_position: np.ndarray,
    target_positions: np.ndarray,
    cell: np.ndarray
) -> np.ndarray:
    """
    Calculate PBC distances with XY-only periodicity (no Z wrapping).
    
    This is a convenience wrapper for monolayer and slab calculations
    where periodic boundary conditions only apply in the XY plane.
    
    Replaces: builders.populate._calculate_xy_pbc_distances()
    
    Parameters
    ----------
    reference_position : np.ndarray
        Reference position. Shape: (3,).
    target_positions : np.ndarray
        Target positions. Shape: (N, 3).
    cell : np.ndarray
        Unit cell matrix. Shape: (3, 3).
    
    Returns
    -------
    distances : np.ndarray
        PBC distances considering only XY periodicity. Shape: (N,).
    
    Examples
    --------
    >>> import numpy as np
    >>> cell = np.eye(3) * 10.0
    >>> ref = np.array([0.0, 0.0, 5.0])
    >>> targets = np.array([[9.0, 0.0, 5.0], [0.0, 0.0, 15.0]])
    >>> dists = calculate_xy_pbc_distances(ref, targets, cell)
    >>> dists  # [1.0 (wrapped in X), 10.0 (no Z wrapping)]
    array([ 1., 10.])
    """
    return calculate_pbc_distances(
        reference_position,
        target_positions,
        cell,
        pbc=[True, True, False],  # XY-only
        mode='extended'
    )


def calculate_minimum_image_distances(
    reference_position: np.ndarray,
    target_positions: np.ndarray,
    cell: np.ndarray,
    pbc: Union[bool, List[bool]] = True
) -> np.ndarray:
    """
    Fast minimum image convention distance calculation.
    
    This is the fastest distance calculation method but may miss neighbors
    at cell boundaries for small cells. Use for large cells (> 20 Å) or
    when performance is critical.
    
    Replaces: utils.geometry.geometry._calculate_distances()
    
    Parameters
    ----------
    reference_position : np.ndarray
        Reference position. Shape: (3,).
    target_positions : np.ndarray
        Target positions. Shape: (N, 3).
    cell : np.ndarray
        Unit cell matrix. Shape: (3, 3).
    pbc : bool or list of bool, default=True
        Periodic boundary conditions.
    
    Returns
    -------
    distances : np.ndarray
        Minimum image distances. Shape: (N,).
    """
    return calculate_pbc_distances(
        reference_position,
        target_positions,
        cell,
        pbc=pbc,
        mode='minimum_image'
    )


def calculate_extended_pbc_distances(
    reference_position: np.ndarray,
    target_positions: np.ndarray,
    cell: np.ndarray,
    pbc: Union[bool, List[bool]] = True
) -> np.ndarray:
    """
    Robust extended PBC search (27 images or 9 for XY-only).
    
    This is the most robust method for small cells and boundary detection.
    Automatically uses numba acceleration when available.
    
    Replaces: utils.geometry.geometry._calculate_distances_with_extended_pbc()
    
    Parameters
    ----------
    reference_position : np.ndarray
        Reference position. Shape: (3,).
    target_positions : np.ndarray
        Target positions. Shape: (N, 3).
    cell : np.ndarray
        Unit cell matrix. Shape: (3, 3).
    pbc : bool or list of bool, default=True
        Periodic boundary conditions.
    
    Returns
    -------
    distances : np.ndarray
        Extended search distances. Shape: (N,).
    """
    return calculate_pbc_distances(
        reference_position,
        target_positions,
        cell,
        pbc=pbc,
        mode='extended'
    )


def find_nearest_image_positions(
    reference_position: np.ndarray,
    candidate_positions: np.ndarray,
    candidate_indices: np.ndarray,
    cell: np.ndarray,
    n_neighbors: int,
    pbc: Union[bool, List[bool]] = True,
    exclude_indices: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find N nearest positions from 27-image supercell with full metadata.
    
    This function generates all 27 periodic images (3×3×3) of each candidate
    position and returns the N nearest ones with their atom indices, Cartesian
    coordinates, and image labels.
    
    Useful for cavity detection, cluster analysis, and any algorithm that needs
    to work with explicit periodic images rather than minimum image convention.
    
    Parameters
    ----------
    reference_position : np.ndarray
        Center position to measure distances from. Shape: (3,).
    candidate_positions : np.ndarray
        Positions of candidate atoms. Shape: (M, 3).
    candidate_indices : np.ndarray
        Atom indices corresponding to candidate_positions. Shape: (M,).
    cell : np.ndarray
        Unit cell matrix. Shape: (3, 3).
    n_neighbors : int
        Number of nearest image positions to return.
    pbc : bool or list of bool, default=True
        Periodic boundary conditions. If True, uses full 3D PBC.
        Can be [True, True, False] for XY-only, etc.
    exclude_indices : np.ndarray or None, default=None
        Atom indices to exclude from search (e.g., molecule atoms).
        Useful for cavity detection where you want to exclude A-site atoms.
    
    Returns
    -------
    nearest_indices : np.ndarray
        Atom indices of nearest positions. Shape: (n_neighbors,).
        These are the original atom indices from candidate_indices.
    nearest_positions : np.ndarray
        Cartesian coordinates of nearest positions. Shape: (n_neighbors, 3).
        These are the actual 27-image positions, not the original positions.
    nearest_distances : np.ndarray
        Distances from reference to nearest positions. Shape: (n_neighbors,).
    image_labels : np.ndarray
        Image offset labels for each position. Shape: (n_neighbors, 3).
        Each row is [i, j, k] where i,j,k ∈ {-1, 0, 1}.
        [0, 0, 0] is the original unit cell.
    
    Examples
    --------
    Find 8 nearest B atom positions for cavity detection:
    
    >>> import numpy as np
    >>> cell = np.eye(3) * 10.0
    >>> mol_center = np.array([5.0, 5.0, 5.0])
    >>> b_positions = np.array([[2.0, 2.0, 2.0], [8.0, 8.0, 8.0]])
    >>> b_indices = np.array([0, 1])
    >>> indices, positions, distances, labels = find_nearest_image_positions(
    ...     mol_center, b_positions, b_indices, cell, n_neighbors=8
    ... )
    >>> indices.shape
    (8,)
    >>> positions.shape
    (8, 3)
    >>> labels.shape
    (8, 3)
    
    Exclude molecule atoms when searching for cavity walls:
    
    >>> mol_atoms = np.array([10, 11, 12])  # A-site molecule
    >>> indices, positions, distances, labels = find_nearest_image_positions(
    ...     mol_center, b_positions, b_indices, cell, n_neighbors=8,
    ...     exclude_indices=mol_atoms
    ... )
    
    Notes
    -----
    - Generates up to 27 images per candidate (or 9 for XY-only PBC).
    - Returns fewer than n_neighbors if insufficient images exist.
    - Image labels can be used to create unique node IDs: f"atom_{idx}_img_{i}_{j}_{k}"
    """
    # Parse PBC axes
    if isinstance(pbc, bool):
        pbc_axes = np.array([1, 1, 1] if pbc else [0, 0, 0], dtype=np.int32)
    else:
        pbc_axes = np.array([int(p) for p in pbc], dtype=np.int32)
    
    # Generate all image offsets based on PBC axes
    image_offsets = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                # Skip images for non-periodic axes
                if i != 0 and pbc_axes[0] == 0:
                    continue
                if j != 0 and pbc_axes[1] == 0:
                    continue
                if k != 0 and pbc_axes[2] == 0:
                    continue
                image_offsets.append([i, j, k])
    
    image_offsets = np.array(image_offsets, dtype=np.float64)
    n_images = len(image_offsets)
    
    # Create exclusion set for fast lookup
    exclude_set = set(exclude_indices) if exclude_indices is not None else set()
    
    # Generate all image positions
    all_data = []  # (distance, atom_idx, cartesian_position, image_label)
    
    for atom_idx, pos in zip(candidate_indices, candidate_positions):
        # Skip excluded atoms
        if atom_idx in exclude_set:
            continue
        
        for img_offset in image_offsets:
            # Translate position by image offset
            translated = pos + img_offset @ cell
            
            # Calculate distance to reference
            dist = np.linalg.norm(translated - reference_position)
            
            # Store data
            all_data.append((dist, atom_idx, translated, img_offset))
    
    # Sort by distance
    all_data.sort(key=lambda x: x[0])
    
    # Select N nearest
    n_return = min(n_neighbors, len(all_data))
    nearest_data = all_data[:n_return]
    
    # Extract arrays
    nearest_distances = np.array([d[0] for d in nearest_data])
    nearest_indices = np.array([d[1] for d in nearest_data], dtype=np.int32)
    nearest_positions = np.array([d[2] for d in nearest_data])
    image_labels = np.array([d[3] for d in nearest_data], dtype=np.int32)
    
    return nearest_indices, nearest_positions, nearest_distances, image_labels
