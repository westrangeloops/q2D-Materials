"""
PBC-aware distance calculations for crystalline structures.

This module now contains only wrapper functions for backward compatibility.
All distance calculations have been unified in pbc_distances.py.

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
from .pbc_distances import calculate_pbc_distances


def _calculate_distances(reference_atom, atom_list, cell, pbc=None):
    """
    Calculate PBC-aware distances (wrapper for unified API).
    
    This is now a simple wrapper around calculate_pbc_distances().
    All legacy code should work without modification.
    
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
    """
    # Convert to new API format
    pbc_arg = True if pbc is None else pbc
    return calculate_pbc_distances(
        reference_atom,
        atom_list,
        cell,
        pbc=pbc_arg,
        mode='minimum_image'  # Old function used minimum image
    )


def _calculate_distances_with_extended_pbc(
    reference_atom,
    atom_list,
    cell,
    search_radius=1
):
    """
    Calculate PBC-aware distances with extended search (wrapper for unified API).
    
    This is now a simple wrapper around calculate_pbc_distances().
    
    Parameters
    ----------
    reference_atom : array-like
        [x, y, z] coordinates of the reference atom
    atom_list : array-like
        List of [x, y, z] coordinates of atoms to calculate distances to
    cell : array-like
        3x3 array of unit cell vectors (required)
    search_radius : int, optional
        Ignored - new implementation always uses optimal search
    
    Returns
    -------
    numpy.ndarray
        Minimum distances from reference_atom to each atom
    """
    return calculate_pbc_distances(
        reference_atom,
        atom_list,
        cell,
        pbc=True,
        mode='extended'  # Always use extended for robustness
    )
