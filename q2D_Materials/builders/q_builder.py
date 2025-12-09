"""
Lightweight container and helpers for structure matrices.

QBuilderOutput mirrors the minimal fields consumed by populate.populate_structure.
"""

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class QBuilderOutput:
    positions: Dict[str, np.ndarray]
    lattice_vector_sizes: np.ndarray
    cell_vectors: np.ndarray


def calculate_lattice_vectors(BX_dist: float) -> np.ndarray:
    """Return cubic lattice vectors (lengths) from a BX bond distance."""
    return np.array([2 * BX_dist, 2 * BX_dist, 2 * BX_dist], dtype=float)


def build_structure_matrix(
    positions: Dict[str, np.ndarray], lattice_vector_sizes: np.ndarray
) -> QBuilderOutput:
    """
    Create a QBuilderOutput from positions and lattice vectors.

    Positions are expected in cartesian coordinates.
    """
    cell_vectors = np.array(
        [
            [lattice_vector_sizes[0], 0.0, 0.0],
            [0.0, lattice_vector_sizes[1], 0.0],
            [0.0, 0.0, lattice_vector_sizes[2]],
        ]
    )
    return QBuilderOutput(
        positions=positions,
        lattice_vector_sizes=lattice_vector_sizes,
        cell_vectors=cell_vectors,
    )

