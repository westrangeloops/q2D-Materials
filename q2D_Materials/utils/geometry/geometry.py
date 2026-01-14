"""
PBC-aware distance calculations for crystalline structures.

This module contains shared geometric utilities used by both builders and analyzers.

Note:
    Analyzer-specific functions have been moved to the q2D_Materials.analyzer module:
    - Constants and bond radii → analyzer.perovskite_constants
    - Octahedral detection → analyzer.octahedral_detection
    - Layer identification → analyzer.layer_identification
    - Molecular classification → analyzer.molecule_classification
    - A-site detection → analyzer.cavity_detection
    - Structure classification → analyzer.structure_classification
    - Graph construction → analyzer.graph_construction
"""

import numpy as np


def _calculate_distances(reference_atom, atom_list, cell, pbc=None):
    """
    Calculate PBC-aware distances between one reference atom and multiple atoms.
    Optimized for crystal structures - always uses periodic boundary conditions.

    This function is used by both the builder (collision detection) and analyzer modules.

    Parameters
    ----------
    reference_atom : array-like
        [x, y, z] coordinates of the reference atom
    atom_list : array-like
        List of [x, y, z] coordinates of atoms to calculate distances to
    cell : array-like
        3x3 array of unit cell vectors (required)
    pbc : list of bool, optional
        List of 3 booleans for periodic boundary conditions
        (optional, defaults to [True, True, True])

    Returns
    -------
    numpy.ndarray
        PBC-aware distances from reference_atom to each atom in atom_list

    Examples
    --------
    >>> import numpy as np
    >>> cell = np.eye(3) * 10  # 10 Å cubic cell
    >>> ref = np.array([0, 0, 0])
    >>> atoms = np.array([[1, 0, 0], [9, 0, 0]])  # Two atoms
    >>> distances = _calculate_distances(ref, atoms, cell)
    >>> distances
    array([1., 1.])  # Both are 1 Å away due to PBC
    """
    # Convert inputs to numpy arrays for efficiency
    ref_coord = np.asarray(reference_atom, dtype=np.float64)
    atom_coords = np.asarray(atom_list, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)

    # Always use PBC for crystal structures
    inv_cell = np.linalg.inv(cell)

    # Calculate all differences at once (vectorized)
    diff_vectors = atom_coords - ref_coord  # Shape: (n_atoms, 3)

    # Apply PBC to all vectors at once
    diff_cell_coords = diff_vectors @ inv_cell.T  # More efficient matrix multiplication
    diff_cell_coords = diff_cell_coords - np.round(diff_cell_coords)
    pbc_diff_vectors = diff_cell_coords @ cell  # Shape: (n_atoms, 3)

    # Calculate distances for all atoms at once
    distances = np.linalg.norm(pbc_diff_vectors, axis=1)

    return distances


def _calculate_distances_with_extended_pbc(
    reference_atom,
    atom_list,
    cell,
    search_radius=1
):
    """
    Calculate PBC-aware distances by searching multiple periodic images.

    Unlike _calculate_distances() which uses minimum image convention (nearest
    periodic image only), this function searches all nearby periodic images within
    the specified search radius to ensure neighbors at cell boundaries are not missed.

    This is crucial for octahedral detection where halogen atoms at cell boundaries
    might be missed by the minimum image convention.

    Parameters
    ----------
    reference_atom : array-like
        [x, y, z] coordinates of the reference atom
    atom_list : array-like
        List of [x, y, z] coordinates of atoms to calculate distances to
    cell : array-like
        3x3 array of unit cell vectors (required)
    search_radius : int, optional
        Number of periodic cells to search in each direction:
        - 1: searches 3×3×3 = 27 images (default, recommended for most cases)
        - 2: searches 5×5×5 = 125 images (for very small cells < 10 Å)
        - 0: equivalent to minimum image convention

    Returns
    -------
    numpy.ndarray
        Minimum distances from reference_atom to each atom across all periodic
        images searched

    Notes
    -----
    Performance: This function is ~27x more expensive than _calculate_distances()
    for search_radius=1. Use as fallback when standard search finds insufficient
    neighbors, not as default.

    Examples
    --------
    >>> import numpy as np
    >>> cell = np.eye(3) * 10  # 10 Å cubic cell
    >>> ref = np.array([0.1, 0.1, 0.1])  # Near cell boundary
    >>> atoms = np.array([[9.9, 0.1, 0.1]])  # Atom on opposite boundary
    >>> distances = _calculate_distances_with_extended_pbc(ref, atoms, cell)
    >>> distances
    array([0.2])  # Correctly finds nearest periodic image
    """
    # Convert inputs to numpy arrays
    ref_coord = np.asarray(reference_atom, dtype=np.float64)
    atom_coords = np.asarray(atom_list, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)

    # Handle singular cell matrix
    try:
        inv_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        # Fallback to simple Euclidean distance if cell is singular
        diff_vectors = atom_coords - ref_coord
        return np.linalg.norm(diff_vectors, axis=1)

    # Convert to fractional coordinates
    diff_vectors = atom_coords - ref_coord
    diff_frac = diff_vectors @ inv_cell.T  # Shape: (n_atoms, 3)

    # Generate all periodic image shifts within search_radius
    # For search_radius=1: (-1,-1,-1), (-1,-1,0), ..., (1,1,1) = 27 shifts
    shifts = []
    for dx in range(-search_radius, search_radius + 1):
        for dy in range(-search_radius, search_radius + 1):
            for dz in range(-search_radius, search_radius + 1):
                shifts.append([dx, dy, dz])
    shifts = np.array(shifts, dtype=np.float64)  # Shape: (n_shifts, 3)

    # For each atom, calculate distance to all periodic images
    n_atoms = len(atom_coords)
    n_shifts = len(shifts)
    min_distances = np.full(n_atoms, np.inf, dtype=np.float64)

    # Vectorized calculation: for each shift, calculate distances to all atoms
    for shift in shifts:
        # Apply periodic shift to fractional coordinates
        shifted_frac = diff_frac - shift  # Broadcasting: (n_atoms, 3) - (3,)

        # Convert back to Cartesian coordinates
        shifted_cart = shifted_frac @ cell  # Shape: (n_atoms, 3)

        # Calculate distances
        distances = np.linalg.norm(shifted_cart, axis=1)  # Shape: (n_atoms,)

        # Keep minimum distance for each atom
        min_distances = np.minimum(min_distances, distances)

    return min_distances
