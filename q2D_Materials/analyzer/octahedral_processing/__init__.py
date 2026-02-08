"""Octahedral processing and structural characterization functions (PDynA-based).

This module provides low-level building blocks for octahedral analysis, including:
- Octahedral distortion and tilting calculations
- Octahedral connectivity and network fitting
- Trajectory-based octahedral resolution (for MD simulations)
"""

from .resolve_octahedra import (
    tqdm_joblib,
    resolve_octahedra,
)
from .octahedral_analysis import (
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
from .tilt_calculations import (
    OctahedralTiltData,
    compute_octahedral_tilts,
    calculate_euler_angles_from_bonds,
    get_reference_axes,
    compute_inclination_from_rotations,
    compute_out_of_plane_inclination,
)
from .tilting_properties import (
    compute_mean_tilt_profile,
    compute_gearing_correlation,
)
from .volume_calculations import (
    compute_octahedral_volume,
    compute_all_octahedral_volumes,
)

__all__ = [
    'tqdm_joblib',
    'resolve_octahedra',
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
    # Tilt calculations (refactored from glazer_detection)
    'OctahedralTiltData',
    'compute_octahedral_tilts',
    'calculate_euler_angles_from_bonds',
    'get_reference_axes',
    'compute_inclination_from_rotations',
    'compute_out_of_plane_inclination',
    # Tilting properties (new advanced metrics)
    'compute_mean_tilt_profile',
    'compute_gearing_correlation',
    # Volume calculations
    'compute_octahedral_volume',
    'compute_all_octahedral_volumes',
]

