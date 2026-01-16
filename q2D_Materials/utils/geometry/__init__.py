"""Geometric utilities for crystal structures."""

# New unified distance calculation API
from .pbc_distances import (
    calculate_pbc_distances,
    find_nearest_neighbors,
    distance_matrix_pbc,
    calculate_xy_pbc_distances,
    calculate_minimum_image_distances,
    calculate_extended_pbc_distances,
)

# Legacy distance functions (will be deprecated)
from .geometry import _calculate_distances
from .structural_utils import (
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
from .structural_constants import (
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
    DEFAULT_FITTING_TOL,
    DEFAULT_CONFIDENCE_BOUND,
    DEFAULT_MATCH_TOL,
)

__all__ = [
    # New unified API (recommended)
    'calculate_pbc_distances',
    'find_nearest_neighbors',
    'distance_matrix_pbc',
    'calculate_xy_pbc_distances',
    'calculate_minimum_image_distances',
    'calculate_extended_pbc_distances',
    # Legacy functions (backward compatibility)
    '_calculate_distances',
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
    'DEFAULT_FITTING_TOL',
    'DEFAULT_CONFIDENCE_BOUND',
    'DEFAULT_MATCH_TOL',
]

