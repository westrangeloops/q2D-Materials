"""
Matrix-based Glazer tilting with analytic lattice contraction.

Applies Glazer tilt rotations directly to position matrices and
computes lattice shrinkage from rigid-unit geometry instead of heuristics.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional

PositionMatrix = Dict[str, List[List[float]]]


def calculate_angles_from_glazer_pattern(
    tilt_pattern: List[str], default_angle: float = 2.0
) -> List[float]:
    """
    Calculate tilting angles from a Glazer pattern.
    """
    angles: List[float] = []
    for pattern in tilt_pattern:
        if pattern == "0":
            angles.append(0.0)
        else:
            angles.append(default_angle)
    return angles


def glazer_notation_to_pattern(glazer_string: str) -> List[str]:
    """
    Convert Glazer notation string (e.g., "a-a-a-") to tilt pattern list.
    """
    if len(glazer_string) != 6:
        raise ValueError(
            f"Glazer notation must be 6 characters (e.g., 'a-a-a-'), got '{glazer_string}'"
        )

    mapping = [glazer_string[1], glazer_string[3], glazer_string[5]]

    valid_patterns = ["0", "+", "-"]
    if any(p not in valid_patterns for p in mapping):
        raise ValueError(f"Invalid Glazer notation pattern '{glazer_string}'")

    return mapping


def _rotation_matrix(axis: np.ndarray, angle_deg: float) -> np.ndarray:
    """Generate Rodrigues' rotation matrix given an axis and angle in degrees."""
    ang = math.radians(angle_deg)
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12 or abs(ang) < 1e-12:
        return np.eye(3)
    axis /= norm
    x, y, z = axis
    c = math.cos(ang)
    s = math.sin(ang)
    C = 1 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ]
    )


def calculate_lattice_scaling(rot_matrices: List[np.ndarray]) -> np.ndarray:
    """
    Determine contraction factors from combined rotations.

    For each cartesian axis i, scaling_i = |(R_total @ e_i) · e_i|,
    which is the projection of the rotated unit vector back onto itself.
    """
    R_total = np.eye(3)
    for R in rot_matrices:
        R_total = R @ R_total

    unit_vectors = np.eye(3)
    scaling_factors: List[float] = []
    for i in range(3):
        original_vec = unit_vectors[i]
        rotated_vec = R_total @ original_vec
        projection = abs(np.dot(rotated_vec, original_vec))
        scaling_factors.append(projection)

    return np.asarray(scaling_factors, dtype=float)


def apply_glazer_tilting_matrix(
    position_matrix: PositionMatrix,
    lattice_vectors: Tuple[float, float, float],
    angles: Optional[List[float]] = None,
    tilt_pattern: Optional[List[str]] = None,
    default_angle_from_pattern: float = 2.0,
) -> Tuple[PositionMatrix, Tuple[float, float, float]]:
    """
    Apply Glazer tilting and adjust lattice vectors via rigid-unit projection.
    """
    if angles is None and tilt_pattern is None:
        raise ValueError("Either angles or tilt_pattern must be provided.")
    if angles is None:
        angles = calculate_angles_from_glazer_pattern(
            tilt_pattern or [], default_angle=default_angle_from_pattern
        )
    if len(angles) != 3:
        raise ValueError("angles must have length 3")

    pos = {k: np.asarray(v, dtype=float) for k, v in position_matrix.items()}
    if "B" not in pos or "X" not in pos:
        raise ValueError("position_matrix must contain 'B' and 'X' keys")

    axes = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1])]
    rotmats = [_rotation_matrix(axis, angle) for axis, angle in zip(axes, angles)]

    scaling_factors = calculate_lattice_scaling(rotmats)
    new_lattice_vectors = np.asarray(lattice_vectors, dtype=float) * scaling_factors

    X_new = pos["X"].copy()
    B_coords = pos["B"]

    for i, x_pos in enumerate(pos["X"]):
        dists = np.linalg.norm(B_coords - x_pos, axis=1)
        b_idx = np.argmin(dists)
        center_b = B_coords[b_idx]

        bond_vector = x_pos - center_b
        rotated_bond = bond_vector.copy()
        for R in rotmats:
            rotated_bond = rotated_bond @ R.T
        X_new[i] = center_b + rotated_bond

    # Apply PBC wrapping to keep atoms within the new cell
    lattice_matrix = np.array([[new_lattice_vectors[0], 0, 0],
                              [0, new_lattice_vectors[1], 0],
                              [0, 0, new_lattice_vectors[2]]])

    def wrap_positions(coords: np.ndarray) -> np.ndarray:
        """Wrap cartesian coordinates into the cell defined by lattice_matrix."""
        # Convert to fractional coordinates
        frac_coords = np.linalg.solve(lattice_matrix.T, coords.T).T
        # Wrap to [0, 1)
        wrapped_frac = frac_coords - np.floor(frac_coords)
        # Convert back to cartesian
        return (lattice_matrix @ wrapped_frac.T).T

    A_wrapped = wrap_positions(pos.get("A", np.zeros((0, 3), dtype=float)))
    B_wrapped = wrap_positions(B_coords)
    X_wrapped = wrap_positions(X_new)

    out = {
        "A": A_wrapped.tolist(),
        "B": B_wrapped.tolist(),
        "X": X_wrapped.tolist(),
    }
    return out, tuple(new_lattice_vectors.tolist())
