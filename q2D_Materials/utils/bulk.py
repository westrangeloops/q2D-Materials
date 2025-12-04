"""
Bulk structure operations.

This module handles 3D bulk perovskite operations including:
- Orientation transformations

For Glazer tilting, use glazer_tilting.py directly.
"""

from __future__ import annotations

import numpy as np
from ase import Atoms


def orient_structure(
    structure: Atoms,
    plane: str = "100",
) -> Atoms:
    """
    Reorient structure so specified plane is parallel to xy.
    
    Parameters
    ----------
    structure : Atoms
        Input structure
    plane : str
        Miller plane: '100', '110', or '111'
        
    Returns
    -------
    Atoms
        Reoriented structure
    """
    atoms = structure.copy()
    
    if plane == "100":
        # Already aligned
        pass
    elif plane == "110":
        a = atoms.cell[0, 0]
        new_a = a * np.sqrt(2)
        
        pos = atoms.get_positions()
        new_pos = np.zeros_like(pos)
        new_pos[:, 0] = (pos[:, 0] - pos[:, 1]) / np.sqrt(2)
        new_pos[:, 1] = pos[:, 2]
        new_pos[:, 2] = (pos[:, 0] + pos[:, 1]) / np.sqrt(2)
        
        atoms.set_positions(new_pos)
        atoms.set_cell([new_a, a, new_a])
    elif plane == "111":
        a = atoms.cell[0, 0]
        
        R = np.array([
            [1/np.sqrt(2), -1/np.sqrt(2), 0],
            [1/np.sqrt(6), 1/np.sqrt(6), -2/np.sqrt(6)],
            [1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)],
        ])
        
        pos = atoms.get_positions()
        new_pos = pos @ R.T
        atoms.set_positions(new_pos)
        
        a_hex = a * np.sqrt(2)
        c_hex = a * np.sqrt(3)
        atoms.set_cell([a_hex, a_hex, c_hex, 90, 90, 120])
    else:
        raise ValueError(f"Unsupported plane: {plane}")
    
    return atoms
