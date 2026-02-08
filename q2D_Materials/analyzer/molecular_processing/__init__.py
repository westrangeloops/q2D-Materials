"""Molecular processing and analysis modules.

This package contains all modules related to molecular graph construction,
backbone identification, and spacer validation.

Modules
-------
molecule_graph : Unified molecular graph construction
    - create_molecule_graph() - Create standardized molecular graphs
    - identify_backbone() - Identify backbone atoms and count NH3 groups

molecule_candidates : Molecule validation for DJ/RP spacers
    - analyze_molecule_candidate() - Pattern-based validation
    - analyze_dj_candidate() - DJ spacer validation
    - analyze_rp_candidate() - RP spacer validation
    - convert_nh2_to_nh3() - Convert NH2 groups to NH3
    - clean_molecule() - Clean and standardize molecules

molecule_classification : Molecular component classification
    - _classify_molecules_by_continuity() - Classify molecules by layer continuity
    - _find_molecular_components() - Find molecular components in structure

spacer_analysis : Spacer molecular analysis
    - SpacerAnalysis - Comprehensive spacer analysis class
    - SpacerAnalysisResult - Analysis result data class
    - BackboneQuery - Queryable backbone interface

smarts_validator : SMARTS pattern matching utilities
    - find_smarts_matches() - Find pattern matches in molecules
    - validate_smarts_pattern() - Validate SMARTS patterns
"""

from .molecule_graph import create_molecule_graph, identify_backbone
from .molecule_candidates import (
    analyze_molecule_candidate,
    analyze_dj_candidate,
    analyze_rp_candidate,
    convert_nh2_to_nh3,
    clean_molecule,
    SpacerCandidateResult,
    TerminalGroup,
)
from .molecule_classification import (
    _classify_molecules_by_continuity,
    _find_molecular_components,
)
from .spacer_analysis import (
    SpacerAnalysis,
    SpacerAnalysisResult,
    BackboneQuery,
)
from .smarts_validator import (
    find_smarts_matches,
    validate_smarts_pattern,
)

__all__ = [
    # molecule_graph
    'create_molecule_graph',
    'identify_backbone',
    # molecule_candidates
    'analyze_molecule_candidate',
    'analyze_dj_candidate',
    'analyze_rp_candidate',
    'convert_nh2_to_nh3',
    'clean_molecule',
    'SpacerCandidateResult',
    'TerminalGroup',
    # molecule_classification
    '_classify_molecules_by_continuity',
    '_find_molecular_components',
    # spacer_analysis
    'SpacerAnalysis',
    'SpacerAnalysisResult',
    'BackboneQuery',
    # smarts_validator
    'find_smarts_matches',
    'validate_smarts_pattern',
]

