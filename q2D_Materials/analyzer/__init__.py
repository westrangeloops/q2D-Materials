"""
Graph-based analyzer for 2D quantum materials.

This module provides comprehensive analysis of perovskite structures using a
graph-based approach to identify octahedra, layers, molecules, and A-sites.

The analyzer module is organized into focused submodules:
- perovskite_constants: Bond radii and element data
- octahedral_detection: Octahedra identification and classification
- network_analysis: B-X network graph construction
- layer_identification: Layer and slab detection
- molecule_classification: Molecular component finding
- cavity_detection: A-site cation identification
- structure_classification: Structure type inference (bulk/DJ/RP/monolayer)
- graph_construction: Comprehensive graph ontology builder
- characterization: Perovskite characterization getters (Glazer patterns, BXB angles, RDFs)
- analyzer_class: Main q2D_analyzer class (public API)

Example
-------
>>> from q2D_Materials.analyzer import q2D_analyzer
>>> analyzer = q2D_analyzer("structure.vasp")
>>> analyzer.analyze()
>>> print(analyzer.structure_type)
'dj'
>>> octahedra = analyzer.get_octahedra()
>>> spacers = analyzer.get_spacers()
>>> glazer = analyzer.get_glazer_pattern()
>>> bxb_angles = analyzer.get_bxb_angles()
>>> rdf = analyzer.get_partial_rdf([["Pb", "I"]])
"""

# Import main class
from .analyzer_class import q2D_analyzer

# Import constants for external use
from .perovskite_constants import (
    PEROVSKITE_BOND_RADII,
    COVALENT_RADII,
    get_bond_cutoff,
)

# Import commonly used utilities
from .octahedral_detection import _count_octahedra, find_shared_atoms

# Import graph building (used by tests)
from .graph_construction import _graph_inorganic_ontology

# Import layer/slab functions (used by tests)
from .layer_identification import _identify_layers, _identify_slabs_by_continuity
from .network_analysis import _build_bx_network

# Import molecule functions
from .molecule_classification import (
    _find_molecular_components,
    _classify_molecules_by_continuity,
)

# Import A-site functions
from .cavity_detection import _identify_a_site_cations

# Import structure type inference
from .structure_classification import (
    _infer_structure_type,
    _infer_structure_type_from_graph,
)

__all__ = [
    # Main class
    'q2D_analyzer',

    # Constants
    'PEROVSKITE_BOND_RADII',
    'COVALENT_RADII',
    'get_bond_cutoff',

    # Utilities
    '_count_octahedra',
    'find_shared_atoms',

    # Graph building
    '_graph_inorganic_ontology',

    # Layers/slabs
    '_identify_layers',
    '_identify_slabs_by_continuity',
    '_build_bx_network',

    # Molecules
    '_find_molecular_components',
    '_classify_molecules_by_continuity',

    # A-sites
    '_identify_a_site_cations',

    # Structure type
    '_infer_structure_type',
    '_infer_structure_type_from_graph',
]
