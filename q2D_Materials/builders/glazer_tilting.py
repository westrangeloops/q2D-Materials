"""
Glazer-style octahedral tilting applied to position matrices (no ASE atoms).

Given cartesian positions (as produced by base_cell_builder) and lattice vectors,
apply per-axis small rotations to X-site positions about their nearest B-site
center, following a Glazer tilt pattern.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np

PositionMatrix = Dict[str, List[List[float]]]


def _rotation_matrix(axis: str, angle_deg: float) -> np.ndarray:
    """Return a 3x3 rotation matrix about axis x/y/z by angle in degrees."""
    ang = np.deg2rad(angle_deg)
    c, s = np.cos(ang), np.sin(ang)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError("axis must be one of 'x', 'y', 'z'")


def _cell_index(
    vec: np.ndarray, lv: np.ndarray, supercell: Tuple[int, int, int]
) -> Tuple[int, int, int]:
    """Return cell indices in range [0, n-1] for each axis, wrapping with PBC."""
    frac = vec / lv
    ix = int(np.floor(frac[0] + 1e-6)) % supercell[0]
    iy = int(np.floor(frac[1] + 1e-6)) % supercell[1]
    iz = int(np.floor(frac[2] + 1e-6)) % supercell[2]
    return ix, iy, iz


def _nearest_b_with_pbc(
    xvec: np.ndarray, b_positions: List[List[float]], cell_len: np.ndarray
) -> np.ndarray:
    """Return nearest B-site position using minimum-image convention."""
    best = None
    best_d2 = float("inf")
    for bpos in b_positions:
        b = np.asarray(bpos, dtype=float)
        delta = b - xvec
        delta -= np.round(delta / cell_len) * cell_len
        d2 = np.dot(delta, delta)
        if d2 < best_d2:
            best_d2 = d2
            best = xvec + delta
    return best if best is not None else np.asarray(b_positions[0], dtype=float)


def _deduplicate_positions_pbc(
    positions: List[List[float]], cell_len: np.ndarray, tol: float = 0.5
) -> List[List[float]]:
    """Remove positions closer than tol using minimum-image distances."""
    unique: List[np.ndarray] = []
    for pos in positions:
        p = np.asarray(pos, dtype=float)
        is_dup = False
        for u in unique:
            delta = p - u
            delta -= np.round(delta / cell_len) * cell_len
            if np.linalg.norm(delta) < tol:
                is_dup = True
                break
        if not is_dup:
            unique.append(p)
    return [u.tolist() for u in unique]


def _kvecs_from_pattern(angles: List[float], tilt_pattern: List[str]) -> List[np.ndarray]:
    """Build k-vectors (phase) per Glazer sign."""
    if len(angles) != 3 or len(tilt_pattern) != 3:
        raise ValueError("angles and tilt_pattern must have length 3")
    kvecs = []
    for i, pat in enumerate(tilt_pattern):
        ang = angles[i]
        if ang == 0.0 and pat != "0":
            pat = "0"
        if pat == "0" and ang != 0.0:
            angles[i] = 0.0
        if pat == "0":
            kvecs.append(np.array([0, 0, 0], dtype=int))
        elif pat == "+":
            kv = np.array([1, 1, 1], dtype=int)
            kv[i] = 0
            kvecs.append(kv)
        elif pat == "-":
            kvecs.append(np.array([1, 1, 1], dtype=int))
        else:
            raise ValueError("tilt_pattern entries must be '+', '-', or '0'")
    return kvecs


def _adjust_network_connectivity(
    position_matrix: PositionMatrix,
    rotated_x_positions: List[List[float]],
    displacements: List[np.ndarray],
    supercell: Tuple[int, int, int],
    lattice_vectors: Tuple[float, float, float],
) -> PositionMatrix:
    """
    Adjust positions after rotations to maintain octahedral network connectivity.

    This implements a simplified version of the network adjustment from the reference
    DistortPerovskite code, ensuring that octahedra remain connected across supercell boundaries.
    """
    nx, ny, nz = supercell
    lv = np.asarray(lattice_vectors, dtype=float)

    # Start with rotated positions
    adjusted_positions = {
        "A": [list(p) for p in position_matrix.get("A", [])],
        "B": [list(p) for p in position_matrix.get("B", [])],
        "X": rotated_x_positions.copy(),
    }

    # Convert to numpy arrays for easier manipulation
    x_positions = np.asarray(rotated_x_positions, dtype=float)
    b_positions = np.asarray(position_matrix.get("B", []), dtype=float)

    # For each X atom, find its nearest B atom and ensure proper connectivity
    for i, x_pos in enumerate(x_positions):
        # Find nearest B atom
        distances = np.linalg.norm(b_positions - x_pos, axis=1)
        nearest_b_idx = np.argmin(distances)
        nearest_b = b_positions[nearest_b_idx]

        # The X atom should be at approximately bond_length from B
        # If it's significantly closer/farther, there might be connectivity issues
        bond_length = np.linalg.norm(x_pos - nearest_b)

        # Check if this X atom is involved in any problematic overlaps
        for j, other_x in enumerate(x_positions):
            if i == j:
                continue

            dist = np.linalg.norm(x_pos - other_x)
            # If two X atoms are very close, they might be duplicates from boundary issues
            if dist < 0.5:  # Much more conservative threshold
                # Adjust the position to avoid overlap
                # Move along the vector away from the nearest B atom
                direction = (x_pos - nearest_b) / np.linalg.norm(x_pos - nearest_b)
                # Small adjustment to separate overlapping atoms
                adjustment = direction * 0.01
                adjusted_positions["X"][i] = (x_pos + adjustment).tolist()

    return adjusted_positions


def apply_glazer_tilt(
    position_matrix: PositionMatrix,
    lattice_vectors: Tuple[float, float, float],
    supercell: Tuple[int, int, int],
    angles: List[float],
    tilt_pattern: List[str],
    adjust_cell: bool = True,
) -> Tuple[PositionMatrix, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Apply Glazer tilts to X sites about their nearest B site.

    Parameters
    ----------
    position_matrix : dict
        Cartesion positions with keys 'A','B','X' (output of base_cell_builder).
    lattice_vectors : (float, float, float)
        Lattice vector lengths (a, b, c) for the underlying unit cell.
    supercell : (int, int, int)
        Supercell replication (nx, ny, nz) that produced the positions.
    angles : list[float]
        Rotation angles [omega_x, omega_y, omega_z] in degrees.
    tilt_pattern : list[str]
        Glazer pattern entries ['+','-','0'] matching angles.

    Returns
    -------
    (positions, lv_unit, cell_lengths)
        positions : dict with tilted X; A and B shifted only if cell adjusted.
        lv_unit : updated unit-cell vector lengths (a,b,c)
        cell_lengths : full supercell lengths after tilt (A)
    """
    lv = np.asarray(lattice_vectors, dtype=float)
    nx, ny, nz = supercell
    cell_len = lv * np.asarray(supercell, dtype=float)
    kvecs = _kvecs_from_pattern(angles, tilt_pattern)

    # Build cell index -> B positions for nearest-center lookup
    b_cells: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
    for bpos in position_matrix.get("B", []):
        bx, by, bz = bpos
        ix = int(np.floor(bx / lv[0] + 1e-6))
        iy = int(np.floor(by / lv[1] + 1e-6))
        iz = int(np.floor(bz / lv[2] + 1e-6))
        key = (ix, iy, iz)
        b_cells.setdefault(key, []).append(np.asarray(bpos, dtype=float))

    rotated_x: List[List[float]] = []
    for xpos in position_matrix.get("X", []):
        xvec = np.asarray(xpos, dtype=float)
        ix = int(np.floor(xvec[0] / lv[0] + 1e-6))
        iy = int(np.floor(xvec[1] / lv[1] + 1e-6))
        iz = int(np.floor(xvec[2] / lv[2] + 1e-6))
        cell_key = (ix, iy, iz)

        # pick nearest B in this cell, else nearest overall
        candidates = b_cells.get(cell_key, [])
        if not candidates:
            candidates = [np.asarray(bpos, dtype=float) for bpos in position_matrix.get("B", [])]
        b_center = min(candidates, key=lambda b: np.linalg.norm(xvec - b))

        # signed angles per axis using (-1)^dot(shift, kvec)
        shift = np.array([ix, iy, iz], dtype=int)
        rot_total = np.eye(3)
        for ax, axis_name in enumerate(["x", "y", "z"]):
            if angles[ax] == 0.0:
                continue
            phase = (-1) ** int(np.dot(shift, kvecs[ax]))
            ang_signed = angles[ax] * phase
            rot_total = _rotation_matrix(axis_name, ang_signed) @ rot_total

        rel = xvec - b_center
        x_rot = b_center + rot_total @ rel
        rotated_x.append(x_rot.tolist())

    # Remove any duplicated X atoms that land on periodic boundaries
    rotated_x = _deduplicate_positions_pbc(rotated_x, cell_len, tol=0.5)

    positions_out: PositionMatrix = {
        "A": [list(p) for p in position_matrix.get("A", [])],
        "B": [list(p) for p in position_matrix.get("B", [])],
        "X": rotated_x,
    }
    # Preserve Ap sites (spacer attachment points) unchanged
    if "Ap" in position_matrix:
        positions_out["Ap"] = [list(p) for p in position_matrix.get("Ap", [])]

    # Optional cell adjustment: wrap positions back into the original supercell
    if adjust_cell:
        cell_len = lv * np.asarray(supercell, dtype=float)
        padding = np.maximum(cell_len * 1e-6, 1e-5)
        for k, plist in positions_out.items():
            # Do not wrap Ap sites; keep their absolute placement for spacer attachment
            if k == "Ap":
                continue
            if not plist:
                continue
            arr = np.asarray(plist, dtype=float)
            arr = np.mod(arr, cell_len)  # wrap into [0, cell_len)
            arr = np.clip(arr, 0.0, cell_len - padding)
            positions_out[k] = arr.tolist()

        return positions_out, tuple(lv.tolist()), tuple(cell_len.tolist())

    return positions_out, tuple(lv.tolist()), tuple((lv * np.asarray(supercell)).tolist())

