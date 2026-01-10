"""Graph-based analyzer for 2D quantum materials.

This module provides comprehensive analysis of perovskite structures using a
graph-based approach to identify octahedra, layers, molecules, and A-sites.

Example
-------
>>> from q2D_Materials.analyzer import q2D_analyzer
>>> analyzer = q2D_analyzer("structure.vasp")
>>> analyzer.analyze()
>>> analyzer.structure_type
'dj'
>>> octahedra = analyzer.get_octahedra()
>>> spacers = analyzer.get_spacers()
>>> glazer = analyzer.get_glazer_pattern()
>>> bxb_angles = analyzer.get_bxb_angles()
>>> rdf = analyzer.get_partial_rdf([["Pb", "I"]])
"""

from .analyzer_class import q2D_analyzer
from .cavity_detection import _identify_a_site_cations
from .graph_construction import _graph_inorganic_ontology
from .layer_identification import _identify_layers, _identify_slabs_by_continuity
from .molecule_classification import (
    _classify_molecules_by_continuity,
    _find_molecular_components,
)
from .network_analysis import _build_bx_network
from .octahedral_detection import _count_octahedra, find_shared_atoms
from .perovskite_constants import (
    COVALENT_RADII,
    PEROVSKITE_BOND_RADII,
    get_bond_cutoff,
)
from .structure_classification import (
    _infer_structure_type,
    _infer_structure_type_from_graph,
)

__all__ = [
    'q2D_analyzer',
    'PEROVSKITE_BOND_RADII',
    'COVALENT_RADII',
    'get_bond_cutoff',
    '_count_octahedra',
    'find_shared_atoms',
    '_graph_inorganic_ontology',
    '_identify_layers',
    '_identify_slabs_by_continuity',
    '_build_bx_network',
    '_find_molecular_components',
    '_classify_molecules_by_continuity',
    '_identify_a_site_cations',
    '_infer_structure_type',
    '_infer_structure_type_from_graph',
]
