"""
Pipeline module for perovskite structure creation.

This module provides the orchestration layer that coordinates templates,
q_builder, and population to create complete perovskite structures.
"""

from .perovskite import (
    create_perovskite,
    create_bulk_perovskite,
    create_monolayer_perovskite,
)
from .common import (
    auto_calculate_BX_distance,
    resolve_BX_distance,
    normalize_spacer,
    build_cell_positions,
    populate_positions,
    default_layer_sequence,
    get_template,
)

__all__ = [
    'create_perovskite',
    'create_bulk_perovskite',
    'create_monolayer_perovskite',
    'auto_calculate_BX_distance',
    'resolve_BX_distance',
    'normalize_spacer',
    'build_cell_positions',
    'populate_positions',
    'default_layer_sequence',
    'get_template',
]

