"""
Octahedral tilt calculations using Kabsch algorithm and Euler decomposition.

This module provides core geometric calculations for extracting octahedral tilts
from structures. It is used by both Glazer pattern detection and tilting properties analysis.

CRITICAL: All calculations use an orthonormal reference frame derived from the AB plane,
which works correctly for ALL cell types including triclinic. The reference frame is:
- x_ref: along lattice vector a
- z_ref: normal to AB plane (a × b) - the stacking direction
- y_ref: in AB plane, perpendicular to x_ref (z × x)

This ensures Euler angles and inclination measures are physically meaningful for
layered 2D perovskites regardless of crystal system.

Functions
---------
compute_octahedral_tilts
    Compute Euler angles, rotation matrices, and inclination angles for all octahedra
calculate_euler_angles_from_bonds
    Calculate Euler angles from octahedral bond vectors using Kabsch algorithm
get_reference_axes
    Determine orthonormal reference frame based on AB plane
compute_inclination_from_rotations
    Compute out-of-plane inclination angles from rotation matrices
compute_out_of_plane_inclination
    Convenience function to get inclination angles as a dict
"""

from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from ..utils.geometry_helpers import (
    apply_pbc_to_vector,
    get_all_x_atoms_from_octahedron,
)


@dataclass
class OctahedralTiltData:
    """Container for octahedral tilt calculations.

    Attributes
    ----------
    euler_angles : np.ndarray
        Shape (N_oct, 3) - [alpha, beta, gamma] in degrees.
        With AB-plane reference: alpha, beta are in-plane tilts,
        gamma is rotation about the stacking axis.
    rotation_matrices : np.ndarray
        Shape (N_oct, 3, 3) - Rotation matrices
    octahedron_ids : List[str]
        Octahedron identifiers (['octahedron_0', ...])
    b_atom_indices : np.ndarray
        Shape (N_oct,) - B-site atom indices
    reference_axes : np.ndarray
        Shape (3, 3) - Orthonormal basis from AB plane.
        Rows are [x_axis, y_axis, z_axis] where z_axis is the AB plane normal.
    inclination_angles : np.ndarray, optional
        Shape (N_oct,) - Angle (degrees) between octahedral axial direction
        and AB plane normal. 0° = upright, 90° = lying flat in AB plane.
    axial_directions : np.ndarray, optional
        Shape (N_oct, 3) - Unit vectors of the octahedral axial (z) bond direction.
    """
    euler_angles: np.ndarray
    rotation_matrices: np.ndarray
    octahedron_ids: List[str]
    b_atom_indices: np.ndarray
    reference_axes: np.ndarray
    inclination_angles: Optional[np.ndarray] = None
    axial_directions: Optional[np.ndarray] = None
    valid_mask: Optional[np.ndarray] = None


def calculate_euler_angles_from_bonds(
    v_x: Optional[np.ndarray],
    v_y: Optional[np.ndarray],
    v_z: Optional[np.ndarray],
    ref_axes: np.ndarray
) -> Tuple[List[float], np.ndarray]:
    """
    Calculate Euler angles (XYZ convention) from octahedral bond vectors.

    Uses Kabsch algorithm (SVD-based) to find optimal rotation matrix R
    that transforms ideal bonds to actual bonds, then decomposes R into
    Euler angles using scipy.Rotation.

    Convention matches builder: R_total = R_z(gamma) @ R_y(beta) @ R_x(alpha)

    Parameters
    ----------
    v_x, v_y, v_z : np.ndarray or None
        Bond vectors along each axis (in Cartesian coordinates).
        If a bond is missing (None), uses ideal axis from ref_axes as fallback.
    ref_axes : np.ndarray
        Reference axes defining ideal (untilted) orientation (3x3 matrix).
        Rows are [x_axis, y_axis, z_axis] unit vectors.

    Returns
    -------
    tuple of (List[float], np.ndarray)
        - euler_angles: [alpha, beta, gamma] in degrees (length 3)
        - rotation_matrix: R (shape 3x3)

    Notes
    -----
    Requires at least 2 bonds to determine rotation. With fewer bonds,
    returns [0, 0, 0] and identity matrix.

    The Kabsch algorithm minimizes RMSD between ideal and actual bond
    configurations, ensuring physically meaningful rotations.
    """
    # Build actual bond matrix from available vectors
    # Use ref_axes-aligned ideal vectors as fallback for missing bonds
    actual_bonds = []
    ideal_bonds = []

    # Normalize reference axes for ideal bond directions
    ref_x = ref_axes[0] / np.linalg.norm(ref_axes[0])
    ref_y = ref_axes[1] / np.linalg.norm(ref_axes[1])
    ref_z = ref_axes[2] / np.linalg.norm(ref_axes[2])

    if v_x is not None:
        actual_bonds.append(v_x / np.linalg.norm(v_x))
        ideal_bonds.append(ref_x)
    if v_y is not None:
        actual_bonds.append(v_y / np.linalg.norm(v_y))
        ideal_bonds.append(ref_y)
    if v_z is not None:
        actual_bonds.append(v_z / np.linalg.norm(v_z))
        ideal_bonds.append(ref_z)

    if len(actual_bonds) < 2:
        # Not enough bonds to determine rotation
        return [0.0, 0.0, 0.0], np.eye(3)

    # Convert to arrays
    actual_matrix = np.array(actual_bonds).T  # Shape (3, n)
    ideal_matrix = np.array(ideal_bonds).T    # Shape (3, n)

    # Find rotation matrix using Kabsch algorithm (SVD-based)
    # R transforms ideal_matrix to actual_matrix: actual = R @ ideal
    H = ideal_matrix @ actual_matrix.T  # Covariance matrix
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = +1, not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Decompose rotation matrix to Euler angles using XYZ convention
    # This matches the builder's order: R_total = R_z @ R_y @ R_x
    # scipy uses 'xyz' for extrinsic (fixed frame) which gives the same result
    try:
        rot = Rotation.from_matrix(R)
        # Use 'xyz' extrinsic (lowercase) to match R_z @ R_y @ R_x order
        euler_angles = rot.as_euler('xyz', degrees=True)
        return euler_angles.tolist(), R
    except Exception:
        # Fallback to zero if decomposition fails
        return [0.0, 0.0, 0.0], np.eye(3)


def get_reference_axes(
    atom_positions: np.ndarray,
    cell: np.ndarray,
    octahedra: List[Dict]
) -> np.ndarray:
    """
    Determine orthonormal reference frame for tilt measurements based on the AB plane.

    CRITICAL: Builds an ORTHONORMAL frame from the AB plane, which works correctly
    for ALL cell types including triclinic. The frame is defined as:
    - x_ref: along lattice vector a (normalized)
    - z_ref: normal to AB plane (a × b, normalized) - the stacking direction
    - y_ref: completes right-handed system (z × x), lies in AB plane

    This ensures Euler angles and tilts are measured relative to the AB plane,
    which is the physically meaningful reference for layered 2D perovskites.

    For sqrt(2)×sqrt(2) rotated supercells, detects rotation by examining
    bond orientations and applies 45° transformation in the AB plane.

    Parameters
    ----------
    atom_positions : np.ndarray
        All atom positions (N_atoms, 3) in Cartesian coordinates
    cell : np.ndarray
        Unit cell matrix (3, 3) with rows as lattice vectors [a, b, c]
    octahedra : list of dict
        Octahedra data from analyzer.get_octahedra()

    Returns
    -------
    np.ndarray
        Reference axes matrix (3, 3) defining orthonormal basis.
        Rows are [x_axis, y_axis, z_axis] unit vectors.
        z_axis is ALWAYS the AB plane normal (stacking direction).

    Notes
    -----
    The orthonormal frame from AB plane ensures:
    1. Works correctly for triclinic, monoclinic, and all crystal systems
    2. z_ref (row 2) is always perpendicular to the AB plane
    3. Euler angles α, β represent in-plane tilts, γ represents out-of-plane rotation
    4. Out-of-plane inclination can be computed from the rotation matrix

    Detection logic for rotated supercells:
    1. Standard supercell: x_ref along a, y_ref in AB plane perpendicular to a
    2. Rotated supercell: x_ref along [110], y_ref along [1-10] (in AB plane)
    """
    # Build orthonormal frame from AB plane (works for ALL cell types including triclinic)
    a_vec = cell[0]
    b_vec = cell[1]

    # x_ref: along lattice vector a
    x_ref = a_vec / np.linalg.norm(a_vec)

    # z_ref: normal to AB plane (stacking direction)
    ab_cross = np.cross(a_vec, b_vec)
    z_ref = ab_cross / np.linalg.norm(ab_cross)

    # y_ref: in AB plane, perpendicular to x_ref (completes right-handed system)
    y_ref = np.cross(z_ref, x_ref)
    y_ref = y_ref / np.linalg.norm(y_ref)  # Should already be unit, but ensure

    # This is our orthonormal basis aligned to the AB plane
    cell_dirs = np.array([x_ref, y_ref, z_ref])

    # Check if this is a sqrt(2)×sqrt(2) rotated supercell by examining bond orientations
    # In such cells, the octahedral bonds are rotated 45° relative to cell vectors
    is_rotated_supercell = False
    rotation_angle = 0.0

    for oct_data in octahedra:
        b_idx = oct_data.get("central_atom_index")
        if b_idx is None:
            continue

        b_pos = atom_positions[b_idx]
        x_indices = get_all_x_atoms_from_octahedron(oct_data)

        if len(x_indices) < 3:
            continue

        # Get bond vectors with PBC using shared utility
        inv_cell = np.linalg.inv(cell)
        bond_vectors = []
        for x_idx in x_indices:
            v = atom_positions[x_idx] - b_pos
            v = apply_pbc_to_vector(v, cell, inv_cell=inv_cell)
            bond_vectors.append(v)

        # Check alignment of bonds with cell directions
        # For a non-rotated cell, at least one bond should align well with each cell axis
        # For a 45° rotated cell, bonds align with diagonal directions
        max_alignments = [0.0, 0.0, 0.0]
        for v in bond_vectors:
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            for k in range(3):
                align = abs(np.dot(v_norm, cell_dirs[k]))
                if align > max_alignments[k]:
                    max_alignments[k] = align

        # If bonds don't align well with any cell direction (align < 0.7 for all),
        # this might be a rotated supercell
        if max_alignments[0] < 0.7 and max_alignments[1] < 0.7:
            # Check for 45° rotation in XY plane
            diag1 = (cell_dirs[0] + cell_dirs[1]) / np.sqrt(2)
            diag2 = (cell_dirs[0] - cell_dirs[1]) / np.sqrt(2)

            diag_alignments = [0.0, 0.0]
            for v in bond_vectors:
                v_norm = v / (np.linalg.norm(v) + 1e-9)
                d1 = abs(np.dot(v_norm, diag1))
                d2 = abs(np.dot(v_norm, diag2))
                if d1 > diag_alignments[0]:
                    diag_alignments[0] = d1
                if d2 > diag_alignments[1]:
                    diag_alignments[1] = d2

            if diag_alignments[0] > 0.7 and diag_alignments[1] > 0.7:
                is_rotated_supercell = True
                rotation_angle = 45.0

        break  # Only need to check one octahedron

    if is_rotated_supercell:
        # For 45° rotated supercells, use diagonal directions as reference
        # The pseudo-cubic axes are along [110] and [1-10] directions
        ref_axes = np.array([
            (cell_dirs[0] + cell_dirs[1]) / np.sqrt(2),  # [110] direction
            (cell_dirs[0] - cell_dirs[1]) / np.sqrt(2),  # [1-10] direction (or [-110])
            cell_dirs[2]                                   # [001] direction
        ])

        # Ensure right-handed coordinate system
        if np.dot(np.cross(ref_axes[0], ref_axes[1]), ref_axes[2]) < 0:
            ref_axes[1] = -ref_axes[1]

        return ref_axes
    else:
        # Standard supercell: use orthonormal AB-plane-based frame
        # x along a, y in AB plane perpendicular to a, z normal to AB plane
        return cell_dirs


def compute_octahedral_tilts(
    analyzer,
    octahedra: Optional[List[str]] = None,
    bond_selection_threshold: float = 0.1,
) -> OctahedralTiltData:
    """
    Compute Euler angles and rotation matrices for octahedra using graph data.

    Uses Kabsch algorithm to find rotation matrix R transforming ideal
    octahedron to actual octahedron, then decomposes to Euler angles
    following convention: R_total = R_z(gamma) @ R_y(beta) @ R_x(alpha).

    This matches the builder's rotation application order.

    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzed structure with graph
    octahedra : list of str, optional
        Specific octahedra IDs to compute. If None, computes all octahedra.
    bond_selection_threshold : float, default=0.1
        Minimum projection value for selecting bonds along reference axes.
        Ensures consistent sign convention by requiring positive projections.

    Returns
    -------
    OctahedralTiltData
        Container with Euler angles, rotation matrices, and metadata

    Notes
    -----
    The reference frame is CRITICAL for correct tilt measurement:
    - Uses CELL vectors as primary reference (ideal untilted structure)
    - Detects sqrt(2)×sqrt(2) rotated supercells and transforms accordingly
    - Ensures tilts are measured relative to crystallographic axes

    Bond selection uses POSITIVE projection constraint to ensure global
    sign consistency across the structure.

    Examples
    --------
    >>> analyzer = q2D_analyzer("structure.vasp")
    >>> analyzer.analyze()
    >>> tilt_data = compute_octahedral_tilts(analyzer)
    >>> print(tilt_data.euler_angles)  # (N_oct, 3) array of [α, β, γ]
    >>> print(tilt_data.rotation_matrices.shape)  # (N_oct, 3, 3)
    """
    # Get structure data
    octahedra_list = analyzer.get_octahedra()
    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())
    inv_cell = np.linalg.inv(cell)

    # Filter octahedra if specified
    if octahedra is not None:
        octahedra_set = set(octahedra)
        octahedra_list = [oct for oct in octahedra_list if oct['id'] in octahedra_set]

    # Get reference axes (pseudo-cubic basis)
    ref_axes = get_reference_axes(atom_positions, cell, octahedra_list)

    # Calculate local tilts for each octahedron
    euler_angles_list = []
    rotation_matrices_list = []
    octahedron_ids = []
    b_atom_indices_list = []
    valid_mask = []

    for oct_data in octahedra_list:
        b_idx = oct_data.get("central_atom_index")
        if b_idx is None:
            euler_angles_list.append([0.0, 0.0, 0.0])
            rotation_matrices_list.append(np.eye(3))
            octahedron_ids.append(oct_data['id'])
            b_atom_indices_list.append(-1)
            valid_mask.append(False)
            continue

        b_pos = atom_positions[b_idx]

        # Get all X neighbors
        x_indices = get_all_x_atoms_from_octahedron(oct_data)

        # Vectors to neighbors with PBC
        vectors = []
        for x_idx in x_indices:
            v = atom_positions[x_idx] - b_pos
            # Apply PBC using shared utility
            v = apply_pbc_to_vector(v, cell, inv_cell=inv_cell)
            vectors.append(v)

        # Project vectors onto reference axes to identify them
        # CRITICAL: Use consistent sign convention - always select bonds pointing
        # in the POSITIVE direction along each axis to ensure global sign consistency
        v_x, v_y, v_z = None, None, None
        v_x_proj, v_y_proj, v_z_proj = -np.inf, -np.inf, -np.inf  # Start with -inf to require positive

        for v in vectors:
            # Project onto ref axes
            projs = [np.dot(v, ref_axes[i]) for i in range(3)]
            abs_projs = np.abs(projs)
            max_idx = np.argmax(abs_projs)

            # CRITICAL: Only select bonds with POSITIVE projection to ensure consistent sign
            # This matches the monolith's approach of using a global reference frame
            if max_idx == 0 and projs[0] > bond_selection_threshold and projs[0] > v_x_proj:
                v_x = v
                v_x_proj = projs[0]
            elif max_idx == 1 and projs[1] > bond_selection_threshold and projs[1] > v_y_proj:
                v_y = v
                v_y_proj = projs[1]
            elif max_idx == 2 and projs[2] > bond_selection_threshold and projs[2] > v_z_proj:
                v_z = v
                v_z_proj = projs[2]

        # Calculate Euler angles and rotation matrix
        is_valid = sum(vector is not None for vector in (v_x, v_y, v_z)) >= 2
        euler, R = calculate_euler_angles_from_bonds(v_x, v_y, v_z, ref_axes)
        euler_angles_list.append(euler)
        rotation_matrices_list.append(R)
        octahedron_ids.append(oct_data['id'])
        b_atom_indices_list.append(b_idx)
        valid_mask.append(is_valid)

    # Compute inclination angles (angle between axial direction and AB plane normal)
    inclination_angles, axial_directions = compute_inclination_from_rotations(
        np.array(rotation_matrices_list), ref_axes
    )

    return OctahedralTiltData(
        euler_angles=np.array(euler_angles_list),
        rotation_matrices=np.array(rotation_matrices_list),
        octahedron_ids=octahedron_ids,
        b_atom_indices=np.array(b_atom_indices_list),
        reference_axes=ref_axes,
        inclination_angles=inclination_angles,
        axial_directions=axial_directions,
        valid_mask=np.asarray(valid_mask, dtype=bool),
    )


def compute_inclination_from_rotations(
    rotation_matrices: np.ndarray,
    reference_axes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute out-of-plane inclination angles from rotation matrices.

    The inclination angle measures how much the octahedral axial direction
    deviates from the AB plane normal (stacking direction). This is the
    physically meaningful measure for layered 2D perovskites.

    Parameters
    ----------
    rotation_matrices : np.ndarray
        Shape (N_oct, 3, 3) - Rotation matrices for each octahedron
    reference_axes : np.ndarray
        Shape (3, 3) - Orthonormal reference frame where row 2 (z_ref) is
        the AB plane normal

    Returns
    -------
    tuple of (np.ndarray, np.ndarray)
        - inclination_angles: Shape (N_oct,) - Angles in degrees.
          0° = axial direction parallel to AB normal (upright octahedron)
          90° = axial direction in the AB plane (lying flat)
        - axial_directions: Shape (N_oct, 3) - Unit vectors of octahedral
          axial direction for each octahedron

    Notes
    -----
    The octahedral axial direction is defined as the z-axis of the octahedron
    after rotation: axial = R @ z_ref, where z_ref is the reference z-axis.

    For a perfect untilted octahedron (R = I), axial = z_ref, giving θ = 0°.

    Physical interpretation:
    - θ ≈ 0°: Octahedra are upright, typical for bulk-like behavior
    - θ > 0°: Octahedra are inclined, common at surfaces and interfaces
    - Large θ: May indicate significant structural distortion or shearing
    """
    n_oct = rotation_matrices.shape[0]
    inclination_angles = np.zeros(n_oct)
    axial_directions = np.zeros((n_oct, 3))

    # The AB plane normal is z_ref (row 2 of reference_axes)
    z_ref = reference_axes[2]

    for i in range(n_oct):
        R = rotation_matrices[i]

        # The octahedral axial direction after rotation
        # Ideal axial is along z_ref, rotated axial is R @ z_ref
        axial = R @ z_ref
        axial = axial / (np.linalg.norm(axial) + 1e-12)
        axial_directions[i] = axial

        # Inclination = angle between axial and z_ref (AB plane normal)
        # cos(θ) = |axial · z_ref| (absolute value for unsigned angle)
        cos_theta = np.abs(np.dot(axial, z_ref))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Numerical safety
        theta_rad = np.arccos(cos_theta)
        inclination_angles[i] = np.degrees(theta_rad)

    return inclination_angles, axial_directions


def compute_out_of_plane_inclination(
    analyzer,
    octahedra: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute out-of-plane inclination for each octahedron.

    This is the angle between the octahedral axial direction and the
    AB plane normal. It provides a single scalar measure of how much
    each octahedron is inclined relative to the layered structure.

    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzed structure with graph
    octahedra : list of str, optional
        Specific octahedra IDs to compute. If None, computes all.

    Returns
    -------
    dict
        {octahedron_id: inclination_angle} where angle is in degrees.
        0° = upright (axial parallel to AB normal)
        90° = lying flat (axial in AB plane)

    Examples
    --------
    >>> analyzer = q2D_analyzer("structure.vasp")
    >>> analyzer.analyze()
    >>> inclinations = compute_out_of_plane_inclination(analyzer)
    >>> for oct_id, angle in inclinations.items():
    ...     print(f"{oct_id}: {angle:.2f}°")
    octahedron_0: 2.34°
    octahedron_1: 3.12°

    Notes
    -----
    This measure is particularly useful for:
    - Detecting sheared structures where octahedra lean systematically
    - Quantifying surface relaxation effects in layered perovskites
    - Comparing structures with different cell types (orthorhombic vs triclinic)

    The measure is independent of in-plane tilting patterns (Glazer-type tilts)
    and captures only the out-of-plane component of octahedral orientation.
    """
    tilt_data = compute_octahedral_tilts(analyzer, octahedra=octahedra)

    return {
        oct_id: float(angle)
        for oct_id, angle in zip(tilt_data.octahedron_ids, tilt_data.inclination_angles)
    }
