"""Structure component detection functions."""

from .octahedral_detection import _count_octahedra, find_shared_atoms
from .cavity_detection import _identify_a_site_cations
from .layer_identification import _identify_layers, _identify_slabs_by_continuity
from .molecule_classification import (
    _classify_molecules_by_continuity,
    _find_molecular_components,
)

__all__ = [
    '_count_octahedra',
    'find_shared_atoms',
    '_identify_a_site_cations',
    '_identify_layers',
    '_identify_slabs_by_continuity',
    '_classify_molecules_by_continuity',
    '_find_molecular_components',
]

