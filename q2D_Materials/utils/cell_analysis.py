

"""
Cell Properties Module for q2D Materials.

This module contains functions for extracting cell properties from ASE atoms objects.
"""

import numpy as np
from ase import Atoms

# This function is used to get the cell and output all properties that are part of the whole cell structure.
def _get_cell_properties(atoms):
    """
    Get clean cell properties from ASE atoms object.
    Returns only essential crystallographic information without redundancy.
    """
    # Lattice parameters
    a, b, c, alpha, beta, gamma = atoms.get_cell_lengths_and_angles()
    volume = atoms.get_volume()
    
    # Chemical composition summary (non-redundant)
    symbols = atoms.get_chemical_symbols()
    composition = {}
    for symbol in symbols:
        composition[symbol] = composition.get(symbol, 0) + 1
    
    return {
        "lattice_parameters": {
            "a": float(a),
            "b": float(b), 
            "c": float(c),
            "alpha": float(alpha),
            "beta": float(beta),
            "gamma": float(gamma)
        },
        "volume": float(volume),
        "composition": composition,
        "formula": atoms.get_chemical_formula(),
        "total_atoms": len(atoms)
    }

def _get_a_b_x_composition(atoms, user_defined_B_cations=None, user_defined_X_anions=None):
    """
    Get A, B, X composition from ASE atoms object.
    """
    # These are used to help determine B-, X-site ions insofar as they're not provided.
    # This can also be passed as an argument if one is using perovskites containing more 
    # exotic materials, e.g. organic X-site anions, or TM B-site cations, etc.
    if user_defined_B_cations is not None:
        common_B = user_defined_B_cations
    else:
        common_B = ['Pb', 'Sn', 'Ge', 'Bi', 'In', 'Tl', 'Zn', 'Cu',
                    'Mn', 'Sb', 'Cd', 'Fe', 'Ag', 'Au', 'Pd', 'Cd',
                    'Hg', 'Co', 'Mg']

    if user_defined_X_anions is not None: 
        common_X = user_defined_X_anions
    else:
        common_X = ['O', 'F', 'Cl', 'Br', 'I']

    unique_types = []
    for atom in atoms.get_chemical_symbols():
        if atom not in unique_types:
            unique_types.append(atom)

    atom_types = unique_types
    return atom_types
    