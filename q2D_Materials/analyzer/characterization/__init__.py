"""Property calculation and characterization functions."""

from .glazer_detection import (
    _detect_glazer_pattern,
    DEFAULT_TOLERANCE,
    DEFAULT_TILT_SIGNIFICANCE_THRESHOLD,
    DEFAULT_ZERO_TILT_THRESHOLD,
    DEFAULT_ZERO_TILT_RELATIVE_THRESHOLD,
    DEFAULT_MAGNITUDE_EQUIVALENCE_THRESHOLD,
    DEFAULT_BOND_SELECTION_THRESHOLD,
    DEFAULT_NUMERICAL_TOLERANCE,
)
from .angle_analysis import _calculate_bxb_angles
from .rdf_analysis import (
    _calculate_partial_rdf,
    _gaussian_kernel_discrete_spectrum,
)
from .characterization import (
    analyze,
    CharacterizationQuery,
    query_octahedra,
    query_layers,
    query_atoms,
    get_octahedron_neighbors,
    get_layer_octahedra,
    find_paths,
    get_subgraph,
)
from .network_analysis import _build_bx_network
from .a_site_analysis import (
    centmass_organic,
    centmass_organic_vec,
    find_b_cage_and_disp,
    match_mixed_halide_octa_dot,
)
from .molecule_candidates import (
    analyze_molecule_candidate,
    analyze_dj_candidate,
    analyze_rp_candidate,
    convert_nh2_to_nh3,
    SpacerCandidateResult,
    TerminalGroup,
)
from .smarts_validator import (
    find_smarts_matches,
    validate_smarts_pattern,
)
from .spacer_analysis import (
    SpacerAnalysis,
    SpacerAnalysisResult,
    BackboneQuery,
)
from .distortions import (
    _compute_octahedral_distortions,
)

__all__ = [
    '_detect_glazer_pattern',
    'DEFAULT_TOLERANCE',
    'DEFAULT_TILT_SIGNIFICANCE_THRESHOLD',
    'DEFAULT_ZERO_TILT_THRESHOLD',
    'DEFAULT_ZERO_TILT_RELATIVE_THRESHOLD',
    'DEFAULT_MAGNITUDE_EQUIVALENCE_THRESHOLD',
    'DEFAULT_BOND_SELECTION_THRESHOLD',
    'DEFAULT_NUMERICAL_TOLERANCE',
    '_calculate_bxb_angles',
    '_calculate_partial_rdf',
    '_gaussian_kernel_discrete_spectrum',
    'analyze',
    'CharacterizationQuery',
    'query_octahedra',
    'query_layers',
    'query_atoms',
    'get_octahedron_neighbors',
    'get_layer_octahedra',
    'find_paths',
    'get_subgraph',
    '_build_bx_network',
    'centmass_organic',
    'centmass_organic_vec',
    'find_b_cage_and_disp',
    'match_mixed_halide_octa_dot',
    'analyze_molecule_candidate',
    'analyze_dj_candidate',
    'analyze_rp_candidate',
    'convert_nh2_to_nh3',
    'SpacerCandidateResult',
    'TerminalGroup',
    'find_smarts_matches',
    'validate_smarts_pattern',
    'SpacerAnalysis',
    'SpacerAnalysisResult',
    'BackboneQuery',
    '_compute_octahedral_distortions',
]

