"""
Graph-centered geometry utilities for perovskite structure analysis.

This module provides general-purpose geometry calculation functions that work
with the analyzer's octahedra dictionary structure. All utilities are designed
to be structure-type agnostic and work with bulk, DJ, RP, and monolayer perovskites.

Functions
---------
apply_pbc_to_vector
    Apply periodic boundary conditions to vectors
calculate_angle_between_vectors
    Calculate angle between two vectors in degrees
get_all_x_atoms_from_octahedron
    Extract all X-atom indices from octahedra dictionary
extract_bx_bond_vectors
    Extract B-X bond vectors from octahedra with PBC
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from ...utils.geometry.structural_utils import apply_pbc_cart_vecs_single_frame


def apply_pbc_to_vector(
    vec: np.ndarray,
    cell: np.ndarray,
    inv_cell: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Apply periodic boundary conditions to a vector or array of vectors.
    
    This is the canonical PBC wrapping function for analyzer code. It works
    with any cell geometry (cubic, tetragonal, orthorhombic, etc.) and
    handles single vectors or batches efficiently.
    
    Parameters
    ----------
    vec : np.ndarray
        Vector(s) to wrap. Shape can be (3,) for single vector, (N, 3) for
        batch of vectors, or (N, M, 3) for multi-dimensional batch.
    cell : np.ndarray
        Unit cell matrix, shape (3, 3)
    inv_cell : np.ndarray, optional
        Pre-computed inverse cell matrix for performance. If None, computed
        internally.
    
    Returns
    -------
    np.ndarray
        Vector(s) with PBC applied, same shape as input
    
    Examples
    --------
    >>> import numpy as np
    >>> cell = np.eye(3) * 10.0  # 10 Å cubic cell
    >>> vec = np.array([9.0, 0.0, 0.0])  # Vector crossing boundary
    >>> wrapped = apply_pbc_to_vector(vec, cell)
    >>> np.linalg.norm(wrapped)  # Should be ~1.0 (wrapped to -1,0,0)
    """
    vec = np.asarray(vec, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Handle different input shapes
    original_shape = vec.shape
    is_single_vector = vec.ndim == 1
    
    if is_single_vector:
        vec = vec.reshape(1, 3)
    
    # Compute inverse cell if not provided
    if inv_cell is None:
        inv_cell = np.linalg.inv(cell)
    else:
        inv_cell = np.asarray(inv_cell, dtype=np.float64)
    
    # Apply PBC: convert to fractional, wrap, convert back
    vec_frac = vec @ inv_cell.T
    vec_frac = vec_frac - np.round(vec_frac)
    vec_pbc = vec_frac @ cell
    
    # Restore original shape
    if is_single_vector:
        vec_pbc = vec_pbc.reshape(3)
    else:
        vec_pbc = vec_pbc.reshape(original_shape)
    
    return vec_pbc


def apply_pbc_to_vectors_batch(
    vectors: np.ndarray,
    cell: np.ndarray
) -> np.ndarray:
    """
    Apply PBC to a batch of vectors (optimized for multiple vectors).
    
    Parameters
    ----------
    vectors : np.ndarray
        Vectors to wrap, shape (N, 3) or (N, M, 3)
    cell : np.ndarray
        Unit cell matrix, shape (3, 3)
    
    Returns
    -------
    np.ndarray
        Vectors with PBC applied, same shape as input
    """
    return apply_pbc_to_vector(vectors, cell)


def calculate_angle_between_vectors(
    vec1: np.ndarray,
    vec2: np.ndarray,
    cell: Optional[np.ndarray] = None,
    apply_pbc: bool = True
) -> float:
    """
    Calculate angle between two vectors in degrees.
    
    This function handles PBC automatically if a cell is provided, making it
    suitable for calculating angles in periodic systems. Works with any
    perovskite structure type.
    
    Parameters
    ----------
    vec1 : np.ndarray
        First vector, shape (3,)
    vec2 : np.ndarray
        Second vector, shape (3,)
    cell : np.ndarray, optional
        Unit cell matrix for PBC. If provided and apply_pbc=True, vectors
        are wrapped before angle calculation.
    apply_pbc : bool, default=True
        Whether to apply PBC if cell is provided
    
    Returns
    -------
    float
        Angle between vectors in degrees (0-180°)
    
    Examples
    --------
    >>> import numpy as np
    >>> vec1 = np.array([1.0, 0.0, 0.0])
    >>> vec2 = np.array([0.0, 1.0, 0.0])
    >>> angle = calculate_angle_between_vectors(vec1, vec2)
    >>> angle  # Should be 90.0
    90.0
    """
    vec1 = np.asarray(vec1, dtype=np.float64)
    vec2 = np.asarray(vec2, dtype=np.float64)
    
    # Apply PBC if requested
    if apply_pbc and cell is not None:
        vec1 = apply_pbc_to_vector(vec1, cell)
        vec2 = apply_pbc_to_vector(vec2, cell)
    
    # Calculate norms
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    # Handle edge cases
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    
    # Calculate cosine of angle
    cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    
    # Convert to degrees
    angle = np.degrees(np.arccos(cos_angle))
    
    return float(angle)


def get_all_x_atoms_from_octahedron(oct: Dict) -> List[int]:
    """
    Extract all X-atom indices from an octahedra dictionary.
    
    This function works with the standard octahedra dictionary structure
    from the analyzer's graph. It combines terminal, interlayer, and
    intralayer atoms into a single list.
    
    Parameters
    ----------
    oct : dict
        Octahedra dictionary with keys:
        - 'terminal_atoms': list of terminal (surface) atom indices
        - 'interlayer_atoms': list of inter-layer shared atom indices
        - 'intralayer_atoms': list of intra-layer shared atom indices
    
    Returns
    -------
    list of int
        Combined list of all X-atom indices
    
    Examples
    --------
    >>> oct = {
    ...     'terminal_atoms': [0, 1],
    ...     'interlayer_atoms': [2, 3],
    ...     'intralayer_atoms': [4, 5]
    ... }
    >>> atoms = get_all_x_atoms_from_octahedron(oct)
    >>> len(atoms)  # Should be 6
    6
    """
    terminal = oct.get('terminal_atoms', [])
    interlayer = oct.get('interlayer_atoms', [])
    intralayer = oct.get('intralayer_atoms', [])
    
    # Combine all X-atoms
    all_x_atoms = list(terminal) + list(interlayer) + list(intralayer)
    
    return all_x_atoms


def extract_bx_bond_vectors(
    oct: Dict,
    atom_positions: np.ndarray,
    cell: np.ndarray,
    apply_pbc: bool = True,
    max_atoms: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract B-X bond vectors from an octahedra dictionary with PBC.
    
    This function extracts bond vectors from the central B-site atom to all
    X-site atoms in the octahedron, applying PBC automatically. Works with
    any perovskite structure type (bulk, DJ, RP, monolayer).
    
    Parameters
    ----------
    oct : dict
        Octahedra dictionary with 'central_atom_index' and X-atom lists
    atom_positions : np.ndarray
        Array of atom positions, shape (N, 3)
    cell : np.ndarray
        Unit cell matrix, shape (3, 3)
    apply_pbc : bool, default=True
        Whether to apply periodic boundary conditions
    max_atoms : int, optional
        Maximum number of X-atoms to include (default: all available)
        Useful for incomplete octahedra or filtering
    
    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - bond_vectors: Array of B-X bond vectors, shape (n_bonds, 3)
        - x_positions: Array of X-atom positions, shape (n_bonds, 3)
    
    Examples
    --------
    >>> import numpy as np
    >>> oct = {
    ...     'central_atom_index': 0,
    ...     'terminal_atoms': [1, 2],
    ...     'interlayer_atoms': [3, 4],
    ...     'intralayer_atoms': [5, 6]
    ... }
    >>> positions = np.random.rand(10, 3) * 10
    >>> cell = np.eye(3) * 10
    >>> bonds, x_pos = extract_bx_bond_vectors(oct, positions, cell)
    >>> bonds.shape[0]  # Number of bonds
    6
    """
    central_idx = oct.get('central_atom_index')
    if central_idx is None:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Get all X-atoms
    x_atoms = get_all_x_atoms_from_octahedron(oct)
    
    if max_atoms is not None:
        x_atoms = x_atoms[:max_atoms]
    
    if len(x_atoms) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)
    
    # Get positions
    central_pos = atom_positions[central_idx]
    x_positions = atom_positions[x_atoms]
    
    # Calculate bond vectors
    bond_vectors = x_positions - central_pos
    
    # Apply PBC if requested
    if apply_pbc:
        bond_vectors = apply_pbc_to_vectors_batch(bond_vectors, cell)
        # Recalculate x_positions after PBC
        x_positions = central_pos + bond_vectors
    
    return bond_vectors, x_positions


def normalize_layer_id(layer_id):
    """Normalize layer ID to string format.
    
    Converts integers to strings and removes 'layer_' prefix if present.
    """
    if isinstance(layer_id, int):
        return str(layer_id)
    if isinstance(layer_id, str):
        if layer_id.startswith('layer_'):
            return layer_id.replace('layer_', '')
        return layer_id
    return str(layer_id)

