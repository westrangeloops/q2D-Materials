"""Utility functions for q2D-Materials.

This module provides general-purpose utilities organized by category:
- geometry: Geometric calculations and PBC handling
- properties: Atomic and molecular properties
- sites: Site-specific utilities (A-sites, etc.)
- molecules: Molecular building and manipulation
- files: File I/O utilities
- other: Miscellaneous utilities
"""

# Re-export commonly used utilities for backward compatibility
from .geometry.geometry import _calculate_distances
from .geometry.structural_utils import (
    get_cart_from_frac,
    get_frac_from_cart,
    apply_pbc_cart_vecs,
)
from .properties import (
    get_covalent_radii,
    get_covalent_radius,
    get_atomic_valences,
    get_valence,
    determine_hybridization,
)
from .sites import (
    get_ionic_radius,
    is_molecular_a_cation,
    get_a_site_object,
)
from .molecules import (
    smiles_to_ase_atoms,
    smiles_to_xyz,
)
from .other import (
    jag_to_layers,
)

__all__ = [
    # Geometry
    '_calculate_distances',
    'get_cart_from_frac',
    'get_frac_from_cart',
    'apply_pbc_cart_vecs',
    # Properties
    'get_covalent_radii',
    'get_covalent_radius',
    'get_atomic_valences',
    'get_valence',
    'determine_hybridization',
    # Sites
    'get_ionic_radius',
    'is_molecular_a_cation',
    'get_a_site_object',
    # Molecules
    'smiles_to_ase_atoms',
    'smiles_to_xyz',
    # Other
    'jag_to_layers',
]
