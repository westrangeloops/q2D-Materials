"""
q2D Materials Core Module

This module provides structure creation and analysis for 2D perovskite materials.

Main Classes:
- q2D_creator: Create perovskite structures from templates
- q2D_analyzer: Analyze structures using graph-based methods
- q2DStructure: Structure wrapper with metadata

The analyzer uses a hierarchical graph approach to decompose structures:
1. Load structure (VASP, CIF, Atoms)
2. Build connectivity graph
3. Detect octahedra (BX6 units)
4. Identify layers/slabs
5. Find spacer molecules
6. Classify A-site cations
7. Infer structure type (bulk, DJ, RP, monolayer)
"""

# Import the creator
from .creator import q2D_creator

# Import q2DStructure for structure creation
from .structure import q2DStructure

# Note: q2D_analyzer should be imported directly from q2D_Materials.analyzer
# to avoid circular imports (analyzer imports from core.structure)

__all__ = [
    'q2D_creator',
    'q2DStructure',
] 