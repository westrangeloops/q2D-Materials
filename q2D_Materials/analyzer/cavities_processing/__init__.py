"""Cavity processing module for detecting and analyzing perovskite cavities.

This module provides tools for detecting A-site and spacer cavities in 2D perovskite
structures, with configurable heuristic weights for cavity X-atom selection.
"""

from .cavity_tracing import detect_all_cavities, ASiteWeights, SpacerWeights
from .cavity_class import Cavity

__all__ = [
    'detect_all_cavities',
    'ASiteWeights',
    'SpacerWeights',
    'Cavity',
]
