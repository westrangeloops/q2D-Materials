"""
Clifford 6D Torus Embedding for Periodic Boundary Conditions.

This module implements the Clifford torus embedding that automatically handles
periodic boundary conditions by mapping 3D coordinates to 6D space. In this
embedding, points at cell boundaries map to the same 6D point, and Euclidean
distance in 6D corresponds to the shortest path across periodic boundaries.

Mathematical Foundation:
- Each 3D coordinate (x, y, z) maps to 6D: [R*cos(2πx/Lx), R*sin(2πx/Lx), 
  R*cos(2πy/Ly), R*sin(2πy/Ly), R*cos(2πz/Lz), R*sin(2πz/Lz)]
- Where R = L / (2π) for each dimension
- Euclidean distance in 6D = chord distance on torus = shortest path across PBC

All core functions are optimized with Numba JIT compilation for maximum performance.
"""

import numpy as np
from numba import jit, prange


# ============================================================================
# Numba-optimized core functions (compiled to machine code)
# ============================================================================

@jit(nopython=True, fastmath=True)
def embed_to_6d_numba(coords, cell_lengths):
    """
    Numba-optimized embedding of 3D coordinates to 6D Clifford torus space.
    
    Maps 3D coords to 6D Clifford Torus in one pass. Compiles to machine code.
    
    Parameters
    ----------
    coords : np.ndarray
        Shape (N, 3) array of 3D coordinates
    cell_lengths : np.ndarray
        Shape (3,) array of [Lx, Ly, Lz] cell dimensions
        
    Returns
    -------
    np.ndarray
        Shape (N, 6) array of 6D Clifford coordinates
    """
    n = coords.shape[0]
    out = np.empty((n, 6), dtype=np.float64)
    
    # Pre-compute constants to save CPU cycles inside loop
    inv_2pi = 1.0 / (2.0 * np.pi)
    two_pi = 2.0 * np.pi
    Rx, Ry, Rz = cell_lengths[0] * inv_2pi, cell_lengths[1] * inv_2pi, cell_lengths[2] * inv_2pi
    Sx, Sy, Sz = two_pi / cell_lengths[0], two_pi / cell_lengths[1], two_pi / cell_lengths[2]

    for i in range(n):
        out[i, 0] = Rx * np.cos(coords[i, 0] * Sx)
        out[i, 1] = Rx * np.sin(coords[i, 0] * Sx)
        out[i, 2] = Ry * np.cos(coords[i, 1] * Sy)
        out[i, 3] = Ry * np.sin(coords[i, 1] * Sy)
        out[i, 4] = Rz * np.cos(coords[i, 2] * Sz)
        out[i, 5] = Rz * np.sin(coords[i, 2] * Sz)
    return out


@jit(nopython=True, fastmath=True)
def clifford_distance_one_to_one_numba(coord1_6d, coord2_6d):
    """
    Calculate distance between two 6D points (Numba-optimized).
    
    Parameters
    ----------
    coord1_6d : np.ndarray
        Shape (6,) 6D coordinates
    coord2_6d : np.ndarray
        Shape (6,) 6D coordinates
        
    Returns
    -------
    float
        Euclidean distance in 6D space
    """
    d_sq = 0.0
    # Explicit unroll for 6D is faster in Numba
    d_sq += (coord1_6d[0] - coord2_6d[0]) ** 2
    d_sq += (coord1_6d[1] - coord2_6d[1]) ** 2
    d_sq += (coord1_6d[2] - coord2_6d[2]) ** 2
    d_sq += (coord1_6d[3] - coord2_6d[3]) ** 2
    d_sq += (coord1_6d[4] - coord2_6d[4]) ** 2
    d_sq += (coord1_6d[5] - coord2_6d[5]) ** 2
    return np.sqrt(d_sq)


@jit(nopython=True, fastmath=True)
def clifford_distance_one_to_many_numba(coord1_6d, coords2_6d):
    """
    Calculate distances from one 6D point to many 6D points (Numba-optimized).
    
    Parameters
    ----------
    coord1_6d : np.ndarray
        Shape (6,) 6D coordinates of reference point
    coords2_6d : np.ndarray
        Shape (N, 6) array of 6D coordinates
        
    Returns
    -------
    np.ndarray
        Shape (N,) array of distances
    """
    n = coords2_6d.shape[0]
    distances = np.empty(n, dtype=np.float64)
    
    for j in range(n):
        d_sq = 0.0
        # Explicit unroll for 6D
        d_sq += (coord1_6d[0] - coords2_6d[j, 0]) ** 2
        d_sq += (coord1_6d[1] - coords2_6d[j, 1]) ** 2
        d_sq += (coord1_6d[2] - coords2_6d[j, 2]) ** 2
        d_sq += (coord1_6d[3] - coords2_6d[j, 3]) ** 2
        d_sq += (coord1_6d[4] - coords2_6d[j, 4]) ** 2
        d_sq += (coord1_6d[5] - coords2_6d[j, 5]) ** 2
        distances[j] = np.sqrt(d_sq)
    
    return distances


@jit(nopython=True, parallel=True, fastmath=True)
def calc_clifford_distances_numba(coords_6d):
    """
    Calculate full N×N pairwise distance matrix (Numba-optimized, parallel).
    
    Parameters
    ----------
    coords_6d : np.ndarray
        Shape (N, 6) array of 6D coordinates
        
    Returns
    -------
    np.ndarray
        Shape (N, N) distance matrix
    """
    n = coords_6d.shape[0]
    dist_mat = np.empty((n, n), dtype=np.float64)
    
    for i in prange(n):
        for j in range(n):
            d_sq = 0.0
            # Explicit unroll for 6D is faster in Numba
            d_sq += (coords_6d[i, 0] - coords_6d[j, 0]) ** 2
            d_sq += (coords_6d[i, 1] - coords_6d[j, 1]) ** 2
            d_sq += (coords_6d[i, 2] - coords_6d[j, 2]) ** 2
            d_sq += (coords_6d[i, 3] - coords_6d[j, 3]) ** 2
            d_sq += (coords_6d[i, 4] - coords_6d[j, 4]) ** 2
            d_sq += (coords_6d[i, 5] - coords_6d[j, 5]) ** 2
            dist_mat[i, j] = np.sqrt(d_sq)
    
    return dist_mat


@jit(nopython=True, fastmath=True)
def _get_unwrapped_vector(c_origin, c_target, cell):
    """
    Helper: Calculates the shortest 3D vector using phase unwrapping.
    
    Uses 6D Clifford coordinates to find the correct periodic image,
    then converts the phase difference back to a 3D displacement vector.
    
    Parameters
    ----------
    c_origin : np.ndarray
        Shape (6,) 6D coordinates of origin atom
    c_target : np.ndarray
        Shape (6,) 6D coordinates of target atom
    cell : np.ndarray
        Shape (3,) cell dimensions [Lx, Ly, Lz]
        
    Returns
    -------
    np.ndarray
        Shape (3,) unwrapped 3D displacement vector
    """
    vec = np.empty(3, dtype=np.float64)
    pi = np.pi
    two_pi = 2.0 * np.pi
    
    for i in range(3):
        # Indices in 6D array (0,1 for x; 2,3 for y; 4,5 for z)
        idx_u, idx_v = 2 * i, 2 * i + 1
        L = cell[i]
        
        # Phase angles from 6D coordinates
        phi1 = np.arctan2(c_origin[idx_v], c_origin[idx_u])
        phi2 = np.arctan2(c_target[idx_v], c_target[idx_u])
        
        # Calculate angular difference and wrap to [-pi, pi]
        d_phi = phi2 - phi1
        if d_phi > pi:
            d_phi -= two_pi
        elif d_phi < -pi:
            d_phi += two_pi
            
        # Convert phase difference to distance
        # Distance = Angle * Radius = d_phi * (L / 2pi)
        vec[i] = d_phi * (L / two_pi)
    
    return vec


@jit(nopython=True, fastmath=True)
def clifford_angle_numba(p1_6d, center_6d, p2_6d, cell_lengths):
    """
    Calculate EXACT physical bond angle P1-Center-P2 using phase unwrapping.
    
    This function uses 6D Clifford coordinates to identify the correct periodic
    neighbors (topology), but calculates the angle using unwrapped 3D vectors
    (geometry). This ensures that linear bonds (180°) are correctly identified.
    
    The key insight: 6D coordinates are used for neighbor finding (PBC-aware),
    but angles are calculated in 3D space using the unwrapped vectors.
    
    Parameters
    ----------
    p1_6d : np.ndarray
        Shape (6,) 6D coordinates of first atom
    center_6d : np.ndarray
        Shape (6,) 6D coordinates of central atom
    p2_6d : np.ndarray
        Shape (6,) 6D coordinates of second atom
    cell_lengths : np.ndarray
        Shape (3,) cell dimensions [Lx, Ly, Lz]
        
    Returns
    -------
    float
        Angle in degrees (0 to 180), or np.nan if vectors are invalid
    """
    # 1. Unwrap 3D vectors from 6D coordinates using phase unwrapping
    vec_ba = _get_unwrapped_vector(center_6d, p1_6d, cell_lengths)
    vec_bc = _get_unwrapped_vector(center_6d, p2_6d, cell_lengths)
    
    # 2. Calculate angle using standard 3D Euclidean geometry
    # This ensures 180° really means 180°
    norm_ba_sq = 0.0
    norm_bc_sq = 0.0
    dot_product = 0.0
    
    for i in range(3):
        norm_ba_sq += vec_ba[i] * vec_ba[i]
        norm_bc_sq += vec_bc[i] * vec_bc[i]
        dot_product += vec_ba[i] * vec_bc[i]
    
    norm_ba = np.sqrt(norm_ba_sq)
    norm_bc = np.sqrt(norm_bc_sq)
    
    if norm_ba < 1e-10 or norm_bc < 1e-10:
        return np.nan
    
    cosine = dot_product / (norm_ba * norm_bc)
    
    # Clip to avoid float errors (e.g., 1.000000002)
    if cosine > 1.0:
        cosine = 1.0
    elif cosine < -1.0:
        cosine = -1.0
    
    return np.degrees(np.arccos(cosine))


@jit(nopython=True, fastmath=True)
def unwrap_relative_coordinate_numba(ref_3d, target_3d, cell_lengths):
    """
    Unwrap target coordinate relative to reference using phase difference (Numba-optimized).
    
    Parameters
    ----------
    ref_3d : np.ndarray
        Shape (3,) reference atom coordinates
    target_3d : np.ndarray
        Shape (3,) target atom coordinates (in unit cell)
    cell_lengths : np.ndarray
        Shape (3,) array of [Lx, Ly, Lz] cell dimensions
        
    Returns
    -------
    np.ndarray
        Shape (3,) unwrapped target coordinates
    """
    unwrapped_target = np.empty(3, dtype=np.float64)
    two_pi = 2.0 * np.pi
    pi = np.pi
    
    for i in range(3):
        L = cell_lengths[i]
        
        # Convert positions to angles (phases) on the Clifford ring
        theta_ref = (two_pi * ref_3d[i]) / L
        theta_target = (two_pi * target_3d[i]) / L
        
        # Calculate angular difference
        delta_theta = theta_target - theta_ref
        
        # Find shortest arc (wrap to -pi to +pi)
        # This determines if the +1 image, -1 image, or 0 image is closer
        delta_theta = (delta_theta + pi) % (two_pi) - pi
        
        # Translate back to linear distance and add to reference
        unwrapped_target[i] = ref_3d[i] + (delta_theta * L / two_pi)
    
    return unwrapped_target


# ============================================================================
# Public API functions (wrappers that maintain backward compatibility)
# ============================================================================

def embed_to_6d(coords_3d, cell_lengths):
    """
    Embed 3D coordinates into 6D Clifford torus space.
    
    This mapping automatically handles periodic boundary conditions:
    - Points at x=0 and x=L map to the same 6D point
    - Distance calculations in 6D automatically find shortest path across boundaries
    
    Uses Numba-optimized implementation for maximum performance.
    
    Parameters
    ----------
    coords_3d : np.ndarray
        Shape (3,) or (N, 3) array of 3D coordinates
    cell_lengths : np.ndarray
        Shape (3,) array of [Lx, Ly, Lz] cell dimensions
        
    Returns
    -------
    np.ndarray
        Shape (6,) for single coordinate, or (N, 6) for multiple coordinates
    """
    coords_3d = np.asarray(coords_3d, dtype=np.float64)
    cell_lengths = np.asarray(cell_lengths, dtype=np.float64)
    
    # Remember original shape to determine output shape
    was_1d = coords_3d.ndim == 1
    
    if was_1d:
        coords_3d = coords_3d.reshape(1, -1)
    
    # Use Numba-optimized version
    coords_6d = embed_to_6d_numba(coords_3d, cell_lengths)
    
    # If input was 1D, return 1D output; otherwise return 2D
    if was_1d:
        return coords_6d[0]  # Return first (and only) row as 1D array
    else:
        return coords_6d


def clifford_distance(coord1_6d, coord2_6d):
    """
    Calculate Euclidean distance in 6D Clifford space.
    
    This distance automatically corresponds to the shortest path across
    periodic boundaries (chord distance on the torus).
    
    Uses Numba-optimized implementation. Automatically routes to appropriate
    optimized function based on input shapes.
    
    Parameters
    ----------
    coord1_6d : np.ndarray
        Shape (6,) or (N, 6) array of 6D coordinates
    coord2_6d : np.ndarray
        Shape (6,) or (N, 6) array of 6D coordinates
        
    Returns
    -------
    float or np.ndarray
        Euclidean distance(s) in 6D space
    """
    coord1_6d = np.asarray(coord1_6d, dtype=np.float64)
    coord2_6d = np.asarray(coord2_6d, dtype=np.float64)
    
    # Handle input shape detection and route to appropriate Numba function
    if coord1_6d.ndim == 1 and coord2_6d.ndim == 1:
        # One-to-one: both are 1D arrays
        return clifford_distance_one_to_one_numba(coord1_6d, coord2_6d)
    elif coord1_6d.ndim == 1 and coord2_6d.ndim == 2:
        # One-to-many: first is 1D, second is 2D
        return clifford_distance_one_to_many_numba(coord1_6d, coord2_6d)
    elif coord1_6d.ndim == 2 and coord2_6d.ndim == 1:
        # Many-to-one: transpose and use one-to-many
        return clifford_distance_one_to_many_numba(coord2_6d, coord1_6d)
    elif coord1_6d.ndim == 2 and coord2_6d.ndim == 2:
        # Many-to-many: compute row by row using one-to-many
        n1 = coord1_6d.shape[0]
        n2 = coord2_6d.shape[0]
        distances = np.empty((n1, n2), dtype=np.float64)
        for i in range(n1):
            distances[i] = clifford_distance_one_to_many_numba(coord1_6d[i], coord2_6d)
        return distances
    else:
        # Fallback to original NumPy implementation for edge cases
        if coord1_6d.ndim == 1:
            coord1_6d = coord1_6d.reshape(1, -1)
        if coord2_6d.ndim == 1:
            coord2_6d = coord2_6d.reshape(1, -1)
        distances = np.linalg.norm(coord1_6d - coord2_6d, axis=-1)
        return distances.squeeze() if distances.size == 1 else distances


def clifford_angle(p1_6d, center_6d, p2_6d, cell_lengths):
    """
    Calculate EXACT physical bond angle P1-Center-P2 using phase unwrapping.
    
    This function uses 6D Clifford coordinates to identify the correct periodic
    neighbors (topology), but calculates the angle using unwrapped 3D vectors
    (geometry). This ensures that linear bonds (180°) are correctly identified,
    even when atoms span significant portions of the unit cell.
    
    The key insight from the Clifford formalism: 6D coordinates are used for
    neighbor finding (PBC-aware), but angles must be calculated in 3D space
    using the unwrapped vectors to get the correct physical geometry.
    
    Parameters
    ----------
    p1_6d : np.ndarray
        Shape (6,) 6D coordinates of first atom
    center_6d : np.ndarray
        Shape (6,) 6D coordinates of central atom
    p2_6d : np.ndarray
        Shape (6,) 6D coordinates of second atom
    cell_lengths : np.ndarray
        Shape (3,) cell dimensions [Lx, Ly, Lz]
        
    Returns
    -------
    float
        Angle in degrees (0 to 180), or np.nan if vectors are invalid
    """
    p1_6d = np.asarray(p1_6d, dtype=np.float64).flatten()
    center_6d = np.asarray(center_6d, dtype=np.float64).flatten()
    p2_6d = np.asarray(p2_6d, dtype=np.float64).flatten()
    cell_lengths = np.asarray(cell_lengths, dtype=np.float64).flatten()
    
    # Ensure correct shapes
    if p1_6d.shape[0] != 6 or center_6d.shape[0] != 6 or p2_6d.shape[0] != 6:
        raise ValueError("All 6D coordinate inputs must have shape (6,)")
    if cell_lengths.shape[0] != 3:
        raise ValueError("cell_lengths must have shape (3,)")
    
    # Use Numba-optimized version with phase unwrapping
    return clifford_angle_numba(p1_6d, center_6d, p2_6d, cell_lengths)


def unwrap_relative_coordinate(ref_3d, target_3d, cell_lengths):
    """
    Unwrap target coordinate relative to reference using phase difference logic.
    
    This function finds the nearest periodic image of the target atom relative
    to the reference atom by comparing their angular phases on the Clifford ring.
    
    Uses Numba-optimized implementation for maximum performance.
    
    Parameters
    ----------
    ref_3d : np.ndarray
        Shape (3,) reference atom coordinates
    target_3d : np.ndarray
        Shape (3,) target atom coordinates (in unit cell)
    cell_lengths : np.ndarray
        Shape (3,) array of [Lx, Ly, Lz] cell dimensions
        
    Returns
    -------
    np.ndarray
        Shape (3,) unwrapped target coordinates (nearest image relative to ref)
    """
    ref_3d = np.asarray(ref_3d, dtype=np.float64).flatten()
    target_3d = np.asarray(target_3d, dtype=np.float64).flatten()
    cell_lengths = np.asarray(cell_lengths, dtype=np.float64).flatten()
    
    # Ensure correct shapes
    if ref_3d.shape[0] != 3 or target_3d.shape[0] != 3 or cell_lengths.shape[0] != 3:
        raise ValueError("All inputs must have shape (3,) for 3D coordinates/cell lengths")
    
    # Use Numba-optimized version
    return unwrap_relative_coordinate_numba(ref_3d, target_3d, cell_lengths)


def get_cell_lengths(cell):
    """
    Extract cell lengths [Lx, Ly, Lz] from 3x3 cell matrix.
    
    Parameters
    ----------
    cell : np.ndarray
        Shape (3, 3) unit cell matrix
        
    Returns
    -------
    np.ndarray
        Shape (3,) array of cell lengths [|a|, |b|, |c|]
    """
    cell = np.asarray(cell, dtype=np.float64)
    if cell.shape != (3, 3):
        raise ValueError(f"Cell matrix must be 3x3, got shape {cell.shape}")
    
    return np.linalg.norm(cell, axis=0)

