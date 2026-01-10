"""
Glazer-style octahedral tilting applied to position matrices (no ASE atoms).

Given cartesian positions (as produced by templates.build_bulk_cell) and lattice vectors,
apply per-axis small rotations to X-site positions about their nearest B-site
center, following a Glazer tilt pattern.
"""

from typing import Dict, List, Tuple, Optional

import numpy as np

from .glazer_notation import parse_glazer_notation, suggest_angles_for_notation

PositionMatrix = Dict[str, List[List[float]]]

# =============================================================================
# GLAZER TILTING TOLERANCE CONSTANTS
# =============================================================================
# Default tolerance values for Glazer tilting operations.
# These control numerical precision and position handling during tilt application.

# Small value added to fractional coordinates before flooring to handle
# numerical precision issues near cell boundaries
DEFAULT_NUMERICAL_TOLERANCE = 1e-6

# Maximum distance (Å) between positions to consider them duplicates
# Typical bond length precision is ~0.1 Å
DEFAULT_DEDUPLICATION_THRESHOLD = 0.1

# Minimum distance (Å) between X atoms to avoid overlap detection
# Typical bond length precision is ~0.1 Å
DEFAULT_OVERLAP_THRESHOLD = 0.1

# Distance (Å) to move overlapping atoms to resolve conflicts
DEFAULT_ADJUSTMENT_STEP = 0.01

# Fraction of cell length used for padding when wrapping positions
DEFAULT_CELL_PADDING_FACTOR = 1e-6

# Minimum absolute padding (Å) when wrapping positions
DEFAULT_MIN_CELL_PADDING = 1e-5


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
    vec: np.ndarray, lv: np.ndarray, supercell: Tuple[int, int, int], 
    numerical_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE
) -> Tuple[int, int, int]:
    """
    Return cell indices in range [0, n-1] for each axis, wrapping with PBC.
    
    Parameters
    ----------
    numerical_tolerance : float
        Small value added to fractional coordinates before flooring to handle
        numerical precision issues near cell boundaries. Default: 1e-6
    """
    frac = vec / lv
    ix = int(np.floor(frac[0] + numerical_tolerance)) % supercell[0]
    iy = int(np.floor(frac[1] + numerical_tolerance)) % supercell[1]
    iz = int(np.floor(frac[2] + numerical_tolerance)) % supercell[2]
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
    positions: List[List[float]], cell_len: np.ndarray, 
    deduplication_threshold: float = DEFAULT_DEDUPLICATION_THRESHOLD
) -> List[List[float]]:
    """
    Remove positions closer than threshold using minimum-image distances.
    
    Parameters
    ----------
    deduplication_threshold : float
        Maximum distance (Å) between positions to consider them duplicates.
        Default: 0.1 Å (typical bond length precision).
    """
    unique: List[np.ndarray] = []
    for pos in positions:
        p = np.asarray(pos, dtype=float)
        is_dup = False
        for u in unique:
            delta = p - u
            delta -= np.round(delta / cell_len) * cell_len
            if np.linalg.norm(delta) < deduplication_threshold:
                is_dup = True
                break
        if not is_dup:
            unique.append(p)
    return [u.tolist() for u in unique]


def _kvecs_from_pattern(
    angles: List[float], tilt_pattern: List[str]
) -> Tuple[List[np.ndarray], List[float]]:
    """Build k-vectors (phase) per Glazer sign, returning adjusted angles."""
    if len(angles) != 3 or len(tilt_pattern) != 3:
        raise ValueError("angles and tilt_pattern must have length 3")
    angles_adj = list(angles)
    kvecs = []
    for i, pat in enumerate(tilt_pattern):
        ang = angles_adj[i]
        if ang == 0.0 and pat != "0":
            pat = "0"
        if pat == "0" and ang != 0.0:
            angles_adj[i] = 0.0
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
    return kvecs, angles_adj


def _adjust_network_connectivity(
    position_matrix: PositionMatrix,
    rotated_x_positions: List[List[float]],
    displacements: List[np.ndarray],
    supercell: Tuple[int, int, int],
    lattice_vectors: Tuple[float, float, float],
    overlap_threshold: float = DEFAULT_OVERLAP_THRESHOLD,
    adjustment_step: float = DEFAULT_ADJUSTMENT_STEP,
) -> PositionMatrix:
    """
    Adjust positions after rotations to maintain octahedral network connectivity.

    This implements a simplified version of the network adjustment from the reference
    DistortPerovskite code, ensuring that octahedra remain connected across supercell boundaries.
    
    Parameters
    ----------
    overlap_threshold : float
        Minimum distance (Å) between X atoms to avoid overlap detection.
        Default: 0.1 Å (typical bond length precision).
    adjustment_step : float
        Distance (Å) to move overlapping atoms to resolve conflicts.
        Default: 0.01 Å (small adjustment).
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
            if dist < overlap_threshold:
                # Adjust the position to avoid overlap
                # Move along the vector away from the nearest B atom
                direction = (x_pos - nearest_b) / np.linalg.norm(x_pos - nearest_b)
                # Small adjustment to separate overlapping atoms
                adjustment = direction * adjustment_step
                adjusted_positions["X"][i] = (x_pos + adjustment).tolist()

    return adjusted_positions


def apply_glazer_tilt(
    position_matrix: PositionMatrix,
    lattice_vectors: Tuple[float, float, float],
    supercell: Tuple[int, int, int],
    angles: List[float],
    tilt_pattern: List[str],
    adjust_cell: bool = True,
    numerical_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
    deduplication_threshold: float = DEFAULT_DEDUPLICATION_THRESHOLD,
    cell_padding_factor: float = DEFAULT_CELL_PADDING_FACTOR,
    min_cell_padding: float = DEFAULT_MIN_CELL_PADDING,
) -> Tuple[PositionMatrix, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Apply Glazer tilts to X sites about their nearest B site.

    Parameters
    ----------
    position_matrix : dict
        Cartesian positions with keys 'A','B','X' (output of templates.build_bulk_cell).
    lattice_vectors : (float, float, float)
        Lattice vector lengths (a, b, c) for the underlying unit cell.
    supercell : (int, int, int)
        Supercell replication (nx, ny, nz) that produced the positions.
    angles : list[float]
        Rotation angles [omega_x, omega_y, omega_z] in degrees.
    tilt_pattern : list[str]
        Glazer pattern entries ['+','-','0'] matching angles.
    adjust_cell : bool
        Whether to wrap positions back into the supercell.
    numerical_tolerance : float
        Small value added to fractional coordinates before flooring to handle
        numerical precision issues near cell boundaries. Default: 1e-6
    deduplication_threshold : float
        Maximum distance (Å) between positions to consider them duplicates.
        Default: 0.1 Å (typical bond length precision).
    cell_padding_factor : float
        Fraction of cell length used for padding when wrapping positions.
        Default: 1e-6 (relative to cell size).
    min_cell_padding : float
        Minimum absolute padding (Å) when wrapping positions.
        Default: 1e-5 Å.

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
    kvecs, angles_eff = _kvecs_from_pattern(angles, tilt_pattern)

    # Build cell index -> B positions for nearest-center lookup
    b_cells: Dict[Tuple[int, int, int], List[np.ndarray]] = {}
    for bpos in position_matrix.get("B", []):
        bx, by, bz = bpos
        ix = int(np.floor(bx / lv[0] + numerical_tolerance))
        iy = int(np.floor(by / lv[1] + numerical_tolerance))
        iz = int(np.floor(bz / lv[2] + numerical_tolerance))
        key = (ix, iy, iz)
        b_cells.setdefault(key, []).append(np.asarray(bpos, dtype=float))

    rotated_x: List[List[float]] = []
    for xpos in position_matrix.get("X", []):
        xvec = np.asarray(xpos, dtype=float)
        ix = int(np.floor(xvec[0] / lv[0] + numerical_tolerance))
        iy = int(np.floor(xvec[1] / lv[1] + numerical_tolerance))
        iz = int(np.floor(xvec[2] / lv[2] + numerical_tolerance))
        cell_key = (ix, iy, iz)

        # pick nearest B in this cell, else nearest overall
        candidates = b_cells.get(cell_key, [])
        if not candidates:
            candidates = [np.asarray(bpos, dtype=float) for bpos in position_matrix.get("B", [])]
        b_center = min(candidates, key=lambda b: np.linalg.norm(xvec - b))

        # signed angles per axis using (-1)^dot(shift, kvec)
        # Use B-site's shift for phase calculation to match analyzer convention
        # This ensures consistent vector convention: phase is based on the octahedron's (B-site's) position
        # The shift represents which unit cell in the supercell the B-site belongs to
        # This matches the analyzer's shift calculation for consistent phase relationships
        b_ix = int(np.floor(b_center[0] / lv[0] + numerical_tolerance)) % supercell[0]
        b_iy = int(np.floor(b_center[1] / lv[1] + numerical_tolerance)) % supercell[1]
        b_iz = int(np.floor(b_center[2] / lv[2] + numerical_tolerance)) % supercell[2]
        shift = np.array([b_ix, b_iy, b_iz], dtype=int)
        rot_total = np.eye(3)
        for ax, axis_name in enumerate(["x", "y", "z"]):
            if angles_eff[ax] == 0.0:
                continue
            phase = (-1) ** int(np.dot(shift, kvecs[ax]))
            ang_signed = angles_eff[ax] * phase
            rot_total = _rotation_matrix(axis_name, ang_signed) @ rot_total

        rel = xvec - b_center
        x_rot = b_center + rot_total @ rel
        rotated_x.append(x_rot.tolist())

    # Remove any duplicated X atoms that land on periodic boundaries
    rotated_x = _deduplicate_positions_pbc(rotated_x, cell_len, deduplication_threshold)

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
        padding = np.maximum(cell_len * cell_padding_factor, min_cell_padding)
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


def apply_glazer_tilt_from_notation(
    position_matrix: PositionMatrix,
    lattice_vectors: Tuple[float, float, float],
    supercell: Tuple[int, int, int],
    glazer_notation: str,
    angles: Optional[List[float]] = None,
    base_angle: float = 10.0,
    adjust_cell: bool = True,
    numerical_tolerance: float = DEFAULT_NUMERICAL_TOLERANCE,
    deduplication_threshold: float = DEFAULT_DEDUPLICATION_THRESHOLD,
    cell_padding_factor: float = DEFAULT_CELL_PADDING_FACTOR,
    min_cell_padding: float = DEFAULT_MIN_CELL_PADDING,
) -> Tuple[PositionMatrix, Tuple[float, float, float], Tuple[float, float, float]]:
    """
    Apply Glazer tilting using full Glazer notation string.

    Parameters
    ----------
    position_matrix : dict
        Cartesian positions with keys 'A','B','X' (output of templates.build_bulk_cell).
    lattice_vectors : (float, float, float)
        Lattice vector lengths (a, b, c) for the underlying unit cell.
    supercell : (int, int, int)
        Supercell replication (nx, ny, nz) that produced the positions.
    glazer_notation : str
        Full Glazer notation like "a-b+a-" or "a0a0c+".
    angles : list[float], optional
        Explicit tilt angles [omega_x, omega_y, omega_z] in degrees.
        If None, angles will be suggested based on the notation.
    base_angle : float
        Base angle in degrees for auto-generated angles (default 10°).
    adjust_cell : bool
        Whether to wrap positions back into the supercell.
    numerical_tolerance : float
        Small value added to fractional coordinates before flooring to handle
        numerical precision issues near cell boundaries. Default: 1e-6
    deduplication_threshold : float
        Maximum distance (Å) between positions to consider them duplicates.
        Default: 0.1 Å (typical bond length precision).
    cell_padding_factor : float
        Fraction of cell length used for padding when wrapping positions.
        Default: 1e-6 (relative to cell size).
    min_cell_padding : float
        Minimum absolute padding (Å) when wrapping positions.
        Default: 1e-5 Å.

    Returns
    -------
    (positions, lv_unit, cell_lengths)
        positions : dict with tilted positions
        lv_unit : updated unit-cell vector lengths (a,b,c)
        cell_lengths : full supercell lengths after tilt (A)
    """
    # Parse the notation
    system = parse_glazer_notation(glazer_notation)

    # Use provided angles or suggest based on notation
    if angles is None:
        angles = suggest_angles_for_notation(glazer_notation, base_angle)

    # Apply tilting with parsed components
    return apply_glazer_tilt(
        position_matrix=position_matrix,
        lattice_vectors=lattice_vectors,
        supercell=supercell,
        angles=angles,
        tilt_pattern=system.tilt_pattern,
        adjust_cell=adjust_cell,
        numerical_tolerance=numerical_tolerance,
        deduplication_threshold=deduplication_threshold,
        cell_padding_factor=cell_padding_factor,
        min_cell_padding=min_cell_padding,
    )

