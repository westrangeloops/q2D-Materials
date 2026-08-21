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
>>>
>>> # New: Layer-specific B-X-B analysis
>>> intra_layer = analyzer.layers.get_bxb(index=0)
>>> print(f"Layer 0 B-X-B mean: {intra_layer['bxb_mean']:.2f}°")
>>> inter_layer = analyzer.layers.get_interlayer_bxb('0', '1')
>>> print(f"Inter-layer B-X-B mean: {inter_layer['bxb_mean']:.2f}°")
"""

from .core.analyzer_class import q2D_analyzer
from .core.graph_construction import _graph_inorganic_ontology
from .core.layer_identification import _identify_layers, _identify_slabs_by_continuity
from .core.layers_wrapper import Layers
from .core.slabs_wrapper import Slabs
from .core.layer_analysis import (
    get_intralayer_bxb,
    get_interlayer_bxb,
    get_all_interlayer_bxb,
)
from .molecular_processing.molecule_classification import (
    _classify_molecules_by_continuity,
    _find_molecular_components,
)
from .octahedral_processing.octahedral_detection import (
    _count_octahedra,
    find_shared_atoms,
    build_octahedra_ligand_info,
)
from .utils.perovskite_constants import (
    PEROVSKITE_BOND_RADII,
    get_bond_cutoff,
)
from .core.structure_classification import (
    _infer_structure_type,
    _infer_structure_type_from_graph,
)

__all__ = [
    'q2D_analyzer',
    'Layers',
    'Slabs',
    'get_intralayer_bxb',
    'get_interlayer_bxb',
    'get_all_interlayer_bxb',
    'PEROVSKITE_BOND_RADII',
    'get_bond_cutoff',
    '_count_octahedra',
    'find_shared_atoms',
    'build_octahedra_ligand_info',
    '_graph_inorganic_ontology',
    '_identify_layers',
    '_identify_slabs_by_continuity',
    '_find_molecular_components',
    '_classify_molecules_by_continuity',
    '_infer_structure_type',
    '_infer_structure_type_from_graph',
]
