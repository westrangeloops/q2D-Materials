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
