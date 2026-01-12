"""Structural processing functions for perovskite analysis.

This module provides a collection of structural processing functions for analyzing
perovskite structures, including octahedral analysis, octahedral processing, and
molecular analysis.

Parts of this code are from PDynA (https://github.com/WMD-group/PDynA):
MIT License

Copyright (c) 2022 Xia Liang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from typing import Optional, Tuple, List, Dict, Any, Union
import logging

# Import utilities
from ..utils.geometry.structural_utils import (
    get_cart_from_frac,
    get_frac_from_cart,
    apply_pbc_cart_vecs,
    apply_pbc_cart_vecs_single_frame,
    periodicity_fold,
    get_volume,
    distance_matrix,
    distance_matrix_ase,
    distance_matrix_ase_replace,
    distance_matrix_handler,
)

# Import constants
from ..utils.geometry.structural_constants import (
    DEFAULT_FITTING_TOLERANCE,
    DEFAULT_MATCH_TOLERANCE,
    DEFAULT_BB_SEARCH_RADIUS,
    DEFAULT_POPULATION_GAP_TOL,
    DEFAULT_RMSD_THRESHOLD,
    DEFAULT_ANGLE_TOLERANCE,
    DEFAULT_DISTINCT_THRESHOLD,
    DEFAULT_HBOND_MAX_DISTANCE,
    DEFAULT_HBOND_MIN_ANGLE,
    DEFAULT_VOLUME_TOL,
)

# Import octahedral analysis
from .octahedral_processing.octahedral_analysis import (
    load_octahedral_basis,
    octahedra_coords_into_bond_vectors,
    calc_distortions_from_bond_vectors,
    calc_distortions_from_bond_vectors_full,
    match_molecules_extra,
    calc_displacement,
    calc_displacement_full,
    match_bx_orthogonal,
    match_bx_orthogonal_rotated,
    quick_match_octahedron,
    match_bx_arbitrary,
    calc_rotation_from_arbitrary_order,
    find_population_gap,
    convert_xb_to_bx,
    fit_octahedral_network_frame,
    fit_octahedral_network_defect_tol,
    fit_octahedral_network_defect_tol_non_orthogonal,
    find_polytype_network,
    simply_calc_distortion,
)

# Import resolve functions
from .octahedral_processing.resolve_octahedra import (
    tqdm_joblib,
    resolve_octahedra,
    _compute_frame_distortions,
    _process_single_frame,
    _refit_octahedral_network,
)

# Import molecular analysis
from .characterization.molecular_analysis import (
    centmass_organic,
    centmass_organic_vec,
    find_b_cage_and_disp,
    match_mixed_halide_octa_dot,
)

__all__ = [
    # Utilities
    'get_cart_from_frac',
    'get_frac_from_cart',
    'apply_pbc_cart_vecs',
    'apply_pbc_cart_vecs_single_frame',
    'periodicity_fold',
    'get_volume',
    'distance_matrix',
    'distance_matrix_ase',
    'distance_matrix_ase_replace',
    'distance_matrix_handler',
    # Constants
    'DEFAULT_FITTING_TOLERANCE',
    'DEFAULT_MATCH_TOLERANCE',
    'DEFAULT_BB_SEARCH_RADIUS',
    'DEFAULT_POPULATION_GAP_TOL',
    'DEFAULT_RMSD_THRESHOLD',
    'DEFAULT_ANGLE_TOLERANCE',
    'DEFAULT_DISTINCT_THRESHOLD',
    'DEFAULT_HBOND_MAX_DISTANCE',
    'DEFAULT_HBOND_MIN_ANGLE',
    'DEFAULT_VOLUME_TOL',
    # Octahedral analysis
    'load_octahedral_basis',
    'octahedra_coords_into_bond_vectors',
    'calc_distortions_from_bond_vectors',
    'calc_distortions_from_bond_vectors_full',
    'match_molecules_extra',
    'calc_displacement',
    'calc_displacement_full',
    'match_bx_orthogonal',
    'match_bx_orthogonal_rotated',
    'quick_match_octahedron',
    'match_bx_arbitrary',
    'calc_rotation_from_arbitrary_order',
    'find_population_gap',
    'convert_xb_to_bx',
    'fit_octahedral_network_frame',
    'fit_octahedral_network_defect_tol',
    'fit_octahedral_network_defect_tol_non_orthogonal',
    'find_polytype_network',
    'simply_calc_distortion',
    # Resolve functions
    'tqdm_joblib',
    'resolve_octahedra',
    # Molecular analysis
    'centmass_organic',
    'centmass_organic_vec',
    'find_b_cage_and_disp',
    'match_mixed_halide_octa_dot',
]

