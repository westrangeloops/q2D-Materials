"""
Builders module for perovskite structure construction.

This module provides builders that convert geometry templates into
populated structure matrices and eventually ASE Atoms objects.
"""

from .q_builder import QBuilderOutput, calculate_lattice_vectors, build_structure_matrix
from .populate import (
    normalize_a_site, assign_ions_to_sites, populate_structure, attach_spacers
)

__all__ = [
    'QBuilderOutput',
    'calculate_lattice_vectors',
    'build_structure_matrix',
    'normalize_a_site',
    'assign_ions_to_sites',
    'populate_structure',
    'attach_spacers',
]

