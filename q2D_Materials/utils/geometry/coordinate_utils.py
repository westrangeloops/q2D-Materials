"""
Coordinate system utilities for handling non-orthogonal cells.

This module provides functions for converting between Cartesian and fractional
coordinates, which is essential for proper handling of non-orthogonal unit cells
where cell angles differ from 90 degrees.

In non-orthogonal cells:
- Cartesian Z-coordinates can exceed the c parameter
- Direct coordinate comparisons can be misleading
- PBC wrapping must use fractional coordinates

Functions
---------
cartesian_to_fractional : Convert Cartesian to fractional coordinates
fractional_to_cartesian : Convert fractional to Cartesian coordinates
get_actual_cell_extent : Get actual X, Y, Z extent from cell matrix
wrap_fractional : Wrap fractional coordinates to [0, 1)
unwrap_fractional_relative : Unwrap fractional coordinates relative to reference
"""

import numpy as np
from typing import Union, Tuple


def cartesian_to_fractional(
    positions: np.ndarray,
    cell: np.ndarray
) -> np.ndarray:
    """
    Convert Cartesian coordinates to fractional coordinates.
    
    Fractional coordinates are independent of cell angles and are always
    in the range [0, 1) for atoms within the unit cell.
    
    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates. Shape: (3,) for single position or (N, 3) for multiple.
    cell : np.ndarray
        Unit cell matrix with cell vectors as rows. Shape: (3, 3).
        cell[0] = a-vector, cell[1] = b-vector, cell[2] = c-vector.
        Note: ASE uses rows for cell vectors, which is the convention used here.
    
    Returns
    -------
    np.ndarray
        Fractional coordinates. Same shape as input positions.
    
    Examples
    --------
    >>> import numpy as np
    >>> cell = np.array([[10.0, 0.0, 0.0],
    ...                  [0.0, 10.0, 0.0],
    ...                  [0.0, 0.0, 20.0]])
    >>> pos = np.array([5.0, 5.0, 10.0])
    >>> frac = cartesian_to_fractional(pos, cell)
    >>> frac
    array([0.5, 0.5, 0.5])
    
    Non-orthogonal cell example:
    >>> cell = np.array([[10.0, 0.0, 0.0],
    ...                  [0.0, 10.0, 0.0],
    ...                  [2.0, 0.0, 20.0]])  # c-vector tilted
    >>> pos = np.array([6.0, 5.0, 10.0])
    >>> frac = cartesian_to_fractional(pos, cell)
    >>> # frac[2] will be 0.5 even though Z=10 and c=20.1
    """
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Handle single position vs array of positions
    single_position = positions.ndim == 1
    if single_position:
        positions = positions.reshape(1, 3)
    
    # Calculate inverse cell matrix
    try:
        inv_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        raise ValueError(
            "Cannot invert cell matrix. Cell may be singular or degenerate. "
            f"Cell determinant: {np.linalg.det(cell):.6e}"
        )
    
    # Convert: fractional = Cartesian @ inv_cell
    # Since cell vectors are rows: cart = frac @ cell, so frac = cart @ inv(cell)
    fractional = positions @ inv_cell
    
    if single_position:
        fractional = fractional.reshape(3)
    
    return fractional


def fractional_to_cartesian(
    frac_positions: np.ndarray,
    cell: np.ndarray
) -> np.ndarray:
    """
    Convert fractional coordinates to Cartesian coordinates.
    
    Parameters
    ----------
    frac_positions : np.ndarray
        Fractional coordinates. Shape: (3,) for single position or (N, 3) for multiple.
    cell : np.ndarray
        Unit cell matrix with cell vectors as rows. Shape: (3, 3).
    
    Returns
    -------
    np.ndarray
        Cartesian coordinates. Same shape as input positions.
    
    Examples
    --------
    >>> import numpy as np
    >>> cell = np.array([[10.0, 0.0, 0.0],
    ...                  [0.0, 10.0, 0.0],
    ...                  [0.0, 0.0, 20.0]])
    >>> frac = np.array([0.5, 0.5, 0.5])
    >>> cart = fractional_to_cartesian(frac, cell)
    >>> cart
    array([5., 5., 10.])
    """
    frac_positions = np.asarray(frac_positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Handle single position vs array of positions
    single_position = frac_positions.ndim == 1
    if single_position:
        frac_positions = frac_positions.reshape(1, 3)
    
    # Convert: Cartesian = fractional @ cell
    cartesian = frac_positions @ cell
    
    if single_position:
        cartesian = cartesian.reshape(3)
    
    return cartesian


def get_actual_cell_extent(cell: np.ndarray) -> Tuple[float, float, float]:
    """
    Get the actual X, Y, Z extent of a unit cell.
    
    For non-orthogonal cells, the actual extent differs from the cell parameters
    (a, b, c) because the cell vectors are tilted.
    
    The extent is calculated by finding the maximum projection of the cell
    onto each Cartesian axis.
    
    Parameters
    ----------
    cell : np.ndarray
        Unit cell matrix with cell vectors as rows. Shape: (3, 3).
    
    Returns
    -------
    tuple of (float, float, float)
        Actual X, Y, Z extent of the cell in Angstroms.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Orthogonal cell
    >>> cell = np.array([[10.0, 0.0, 0.0],
    ...                  [0.0, 10.0, 0.0],
    ...                  [0.0, 0.0, 20.0]])
    >>> get_actual_cell_extent(cell)
    (10.0, 10.0, 20.0)
    
    >>> # Non-orthogonal cell (c-vector tilted)
    >>> cell = np.array([[10.0, 0.0, 0.0],
    ...                  [0.0, 10.0, 0.0],
    ...                  [2.0, 0.0, 20.0]])
    >>> extent = get_actual_cell_extent(cell)
    >>> # X-extent includes contribution from tilted c-vector
    >>> extent[0] > 10.0
    True
    """
    cell = np.asarray(cell, dtype=np.float64)
    
    # Generate all 8 corners of the parallelepiped
    # Corners are at fractional coordinates: (0,0,0), (1,0,0), (0,1,0), etc.
    corners_frac = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1]
    ], dtype=np.float64)
    
    # Convert to Cartesian
    corners_cart = corners_frac @ cell
    
    # Find extent along each axis
    x_extent = float(np.max(corners_cart[:, 0]) - np.min(corners_cart[:, 0]))
    y_extent = float(np.max(corners_cart[:, 1]) - np.min(corners_cart[:, 1]))
    z_extent = float(np.max(corners_cart[:, 2]) - np.min(corners_cart[:, 2]))
    
    return (x_extent, y_extent, z_extent)


def wrap_fractional(
    frac_positions: np.ndarray,
    min_val: float = 0.0,
    max_val: float = 1.0
) -> np.ndarray:
    """
    Wrap fractional coordinates into the range [min_val, max_val).
    
    This is the proper way to handle PBC wrapping for non-orthogonal cells.
    
    Parameters
    ----------
    frac_positions : np.ndarray
        Fractional coordinates. Shape: (3,) for single position or (N, 3) for multiple.
    min_val : float, default=0.0
        Minimum value of the wrapped range.
    max_val : float, default=1.0
        Maximum value of the wrapped range (exclusive).
    
    Returns
    -------
    np.ndarray
        Wrapped fractional coordinates. Same shape as input.
    
    Examples
    --------
    >>> import numpy as np
    >>> frac = np.array([1.5, -0.3, 0.7])
    >>> wrap_fractional(frac)
    array([0.5, 0.7, 0.7])
    """
    frac_positions = np.asarray(frac_positions, dtype=np.float64)
    
    range_size = max_val - min_val
    wrapped = ((frac_positions - min_val) % range_size) + min_val
    
    return wrapped


def unwrap_fractional_relative(
    frac_positions: np.ndarray,
    reference_frac: np.ndarray,
    cell_extent: float = 1.0
) -> np.ndarray:
    """
    Unwrap fractional coordinates relative to a reference position.
    
    This ensures that all positions are in the same periodic image as the
    reference, which is essential for calculating molecular properties
    across PBC boundaries.
    
    Parameters
    ----------
    frac_positions : np.ndarray
        Fractional coordinates to unwrap. Shape: (N, 3).
    reference_frac : np.ndarray
        Reference fractional position. Shape: (3,).
    cell_extent : float, default=1.0
        Extent of one cell in fractional coordinates (typically 1.0).
    
    Returns
    -------
    np.ndarray
        Unwrapped fractional coordinates. Shape: (N, 3).
    
    Examples
    --------
    >>> import numpy as np
    >>> reference = np.array([0.1, 0.1, 0.1])
    >>> positions = np.array([[0.15, 0.15, 0.15],
    ...                       [0.95, 0.95, 0.95]])  # Close to reference across PBC
    >>> unwrapped = unwrap_fractional_relative(positions, reference)
    >>> # Second position will be unwrapped to [-0.05, -0.05, -0.05]
    >>> unwrapped[1, 0] < 0
    True
    """
    frac_positions = np.asarray(frac_positions, dtype=np.float64)
    reference_frac = np.asarray(reference_frac, dtype=np.float64)
    
    # Calculate difference from reference
    diff = frac_positions - reference_frac
    
    # Unwrap: if difference > cell_extent/2, subtract cell_extent
    # if difference < -cell_extent/2, add cell_extent
    half_extent = cell_extent / 2.0
    
    unwrapped = frac_positions.copy()
    unwrapped[diff > half_extent] -= cell_extent
    unwrapped[diff < -half_extent] += cell_extent
    
    return unwrapped


def get_fractional_z_extent(
    positions: np.ndarray,
    cell: np.ndarray,
    wrapped: bool = True
) -> Tuple[float, float, float]:
    """
    Get the Z-extent of positions in fractional coordinates.
    
    This is useful for determining the actual thickness of a slab or
    the extent of atoms along the c-direction in non-orthogonal cells.
    
    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates. Shape: (N, 3).
    cell : np.ndarray
        Unit cell matrix. Shape: (3, 3).
    wrapped : bool, default=True
        If True, wrap fractional coordinates to [0, 1) before calculating extent.
        If False, use unwrapped fractional coordinates.
    
    Returns
    -------
    tuple of (float, float, float)
        (min_frac_z, max_frac_z, span_frac_z)
    
    Examples
    --------
    >>> import numpy as np
    >>> cell = np.array([[10.0, 0.0, 0.0],
    ...                  [0.0, 10.0, 0.0],
    ...                  [2.0, 0.0, 20.0]])
    >>> positions = np.array([[0.0, 0.0, 0.5],
    ...                       [0.0, 0.0, 10.0],
    ...                       [0.0, 0.0, 19.5]])
    >>> min_z, max_z, span = get_fractional_z_extent(positions, cell)
    >>> # Span should be close to 1.0 (full cell)
    """
    positions = np.asarray(positions, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Convert to fractional
    frac_positions = cartesian_to_fractional(positions, cell)
    
    # Wrap if requested
    if wrapped:
        frac_positions = wrap_fractional(frac_positions)
    
    # Get Z extent
    frac_z = frac_positions[:, 2]
    min_frac_z = float(np.min(frac_z))
    max_frac_z = float(np.max(frac_z))
    span_frac_z = max_frac_z - min_frac_z
    
    return (min_frac_z, max_frac_z, span_frac_z)
