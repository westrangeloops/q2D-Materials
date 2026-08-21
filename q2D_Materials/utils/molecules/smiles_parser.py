"""
DEPRECATED: Direct SMILES to Graph Parser - Replaced by RDKit-based graph converter.

This module is deprecated and will be removed in a future version.
Use q2D_Materials.utils.molecules.graph_converter instead:
- rdkit_to_graph() - Convert RDKit Mol to NetworkX graph
- graph_to_rdkit() - Convert NetworkX graph to RDKit Mol
- from_smiles() in modifier.fragment - Convert SMILES to graph using RDKit

All SMILES parsing now uses RDKit for better chemical validation and robustness.
"""

import warnings

warnings.warn(
    "smiles_parser module is deprecated. Use graph_converter.rdkit_to_graph() "
    "or modifier.fragment.from_smiles() instead. This module will be removed "
    "in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backward compatibility (temporary)
from ...modifier.fragment import from_smiles as smiles_to_graph

__all__ = ['smiles_to_graph']
