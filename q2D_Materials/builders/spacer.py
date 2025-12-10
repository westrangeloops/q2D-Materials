"""
Generic spacer placement tools.

This module provides low-level, structure-agnostic tools for placing
atomic or molecular spacers. Structure-specific logic (DJ, RP, monolayer)
lives in slab.py.
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import numpy as np
from ase import Atoms

# Constants for site role tracking
SITE_ROLE_KEY = "site_role"
SITE_SPACER = "spacer"


def prepare_spacer(
    spacer: Atoms,
    BX_dist: float,
) -> Tuple[float, float, Atoms]:
    """
    Prepare a spacer molecule/atom for placement.
    
    Parameters
    ----------
    spacer : Atoms
        Spacer molecule or single atom
    BX_dist : float
        B-X distance for calculating defaults
        
    Returns
    -------
    tuple of (spacer_length, default_penetration, aligned_spacer)
        - spacer_length: Effective length for cell calculations
        - default_penetration: Default penetration into surface
        - aligned_spacer: Prepared spacer atoms
    """
    from .molecule_builder import align_ase_molecule_for_perovskite, get_molecule_length
    from .A_sites import get_ionic_radius
    
    is_atomic = len(spacer) == 1
    
    if is_atomic:
        symbol = spacer.get_chemical_symbols()[0]
        try:
            ionic_rad = get_ionic_radius("A", symbol)
        except (ValueError, KeyError):
            ionic_rad = 1.8
        # For atoms: length = 2*radius, penetration = +radius
        return ionic_rad * 2, ionic_rad, spacer.copy()
    else:
        aligned = align_ase_molecule_for_perovskite(spacer.copy())
        
        # For molecules: use N-to-N distance
        symbols = aligned.get_chemical_symbols()
        positions = aligned.get_positions()
        n_indices = [i for i, s in enumerate(symbols) if s == 'N']
        
        if len(n_indices) >= 2:
            n_z = [positions[i, 2] for i in n_indices]
            length = max(n_z) - min(n_z)
        else:
            length = get_molecule_length(aligned)
        
        # Default penetration = -1 Å (N is 1 Å below surface X)
        return length, -1.0, aligned


def place_spacer_at_position(
    spacer: Atoms,
    x: float,
    y: float,
    z: float,
    direction: str,
    is_atomic: bool,
) -> Atoms:
    """
    Place a single spacer at the specified position.
    
    Parameters
    ----------
    spacer : Atoms
        Prepared spacer (aligned molecule or single atom)
    x, y, z : float
        Position for spacer attachment point
    direction : str
        'up' or 'down' - direction spacer points
    is_atomic : bool
        Whether spacer is a single atom
        
    Returns
    -------
    Atoms
        Positioned spacer with SITE_SPACER tag
    """
    if is_atomic:
        symbol = spacer.get_chemical_symbols()[0]
        atom = Atoms(symbol, positions=[[x, y, z]])
        atom.arrays[SITE_ROLE_KEY] = np.array([SITE_SPACER], dtype=object)
        return atom
    else:
        placed = _place_molecule(spacer, [x, y, z], direction)
        placed.arrays[SITE_ROLE_KEY] = np.array(
            [SITE_SPACER] * len(placed), dtype=object
        )
        return placed


def place_spacers_grid(
    structure: Atoms,
    spacer: Atoms,
    ax: float,
    ay: float,
    nx: int,
    ny: int,
    z: float,
    direction: str,
    is_atomic: bool,
    shift: Tuple[float, float] = (0.0, 0.0),
) -> Atoms:
    """
    Place spacers on a grid at A-site positions.
    
    For molecular spacers, automatically rotates molecules to avoid overlaps
    in the XY plane. For atomic spacers, places directly.
    
    Parameters
    ----------
    structure : Atoms
        Structure to add spacers to
    spacer : Atoms
        Prepared spacer (aligned molecule or single atom)
    ax, ay : float
        Unit cell dimensions in x and y
    nx, ny : int
        Number of unit cells in x and y
    z : float
        Z position for spacer attachment
    direction : str
        'up' or 'down' - direction spacers point
    is_atomic : bool
        Whether spacer is a single atom
    shift : tuple
        Fractional shift (e.g., (0.5, 0.5) for RP)
        
    Returns
    -------
    Atoms
        Structure with spacers added
    """
    from .molecule_builder import add_atoms
    
    result = structure.copy()
    
    # Collect all XY positions first
    xy_positions = []
    for ix in range(nx):
        for iy in range(ny):
            x = (ix + shift[0]) * ax
            y = (iy + shift[1]) * ay
            xy_positions.append((x, y))
    
    if is_atomic:
        # Atomic spacers: place directly (no overlap issues)
        for x, y in xy_positions:
            placed = place_spacer_at_position(spacer, x, y, z, direction, is_atomic)
            result = add_atoms(result, placed)
    else:
        # Molecular spacers: use rotation to avoid overlaps
        placed_molecules = place_molecules_with_rotation(
            spacer, xy_positions, z, direction, min_distance=2.0, max_rotation_attempts=36
        )
        
        # Add site_role tags and add to structure
        for placed in placed_molecules:
            placed.arrays[SITE_ROLE_KEY] = np.array(
                [SITE_SPACER] * len(placed), dtype=object
            )
            result = add_atoms(result, placed)
    
    return result


def _place_molecule(
    mol: Atoms,
    target_xyz: List[float],
    direction: str,
) -> Atoms:
    """
    Place molecule with NH3+ attachment point at target position.
    
    After alignment, molecules have NH3+ at MAX Z (top) and tail at MIN Z (bottom).
    
    For "up" direction (molecule extends upward from surface):
        - Flip molecule so NH3+ is at MIN Z, tail at MAX Z
        - Attach NH3+ (now at MIN Z) to surface
        - Result: NH3+ at surface, tail extends UP
        
    For "down" direction (molecule extends downward from surface):
        - Keep molecule orientation (NH3+ at MAX Z)
        - Attach NH3+ (at MAX Z) to surface  
        - Result: NH3+ at surface, tail extends DOWN
    """
    mol_copy = mol.copy()
    
    # For "up" direction, flip molecule so tail points up and NH3+ points down
    if direction == "up":
        mol_copy.rotate(180, "x")
    
    # Find attachment N (the NH3+ nitrogen)
    symbols = mol_copy.get_chemical_symbols()
    positions = mol_copy.get_positions()
    
    n_indices = [i for i, s in enumerate(symbols) if s == 'N']
    
    if n_indices:
        if direction == "up":
            # After flip, NH3+ is at MIN Z - attach this to surface
            attach_idx = n_indices[np.argmin([positions[i, 2] for i in n_indices])]
        else:
            # No flip, NH3+ is at MAX Z - attach this to surface
            attach_idx = n_indices[np.argmax([positions[i, 2] for i in n_indices])]
    else:
        # Fallback for molecules without N
        attach_idx = np.argmin(positions[:, 2]) if direction == "up" else np.argmax(positions[:, 2])
    
    # Shift to target position
    current = positions[attach_idx]
    shift = np.array(target_xyz) - current
    mol_copy.positions += shift
    
    return mol_copy


def _check_molecule_overlap_xy(mol1: Atoms, mol2: Atoms, min_distance: float = 2.0) -> bool:
    """
    Check if two molecules overlap in the XY plane.
    
    Only checks XY distances (ignores Z) since molecules are placed at different Z levels.
    Uses heavy atoms (non-hydrogen) for efficiency.
    
    Parameters
    ----------
    mol1, mol2 : Atoms
        Two molecules to check
    min_distance : float
        Minimum allowed XY distance between any atoms (default: 2.0 Å)
        
    Returns
    -------
    bool
        True if molecules overlap in XY, False otherwise
    """
    pos1 = mol1.get_positions()
    pos2 = mol2.get_positions()
    sym1 = mol1.get_chemical_symbols()
    sym2 = mol2.get_chemical_symbols()
    
    # Check only heavy atoms (C, N, O, etc.) for efficiency
    heavy1 = [i for i, s in enumerate(sym1) if s != 'H']
    heavy2 = [i for i, s in enumerate(sym2) if s != 'H']
    
    # If no heavy atoms, check all atoms
    atoms_to_check1 = pos1[heavy1] if heavy1 else pos1
    atoms_to_check2 = pos2[heavy2] if heavy2 else pos2
    
    # Check XY distances only (ignore Z)
    for p1 in atoms_to_check1:
        xy_distances = np.sqrt(
            (atoms_to_check2[:, 0] - p1[0])**2 + 
            (atoms_to_check2[:, 1] - p1[1])**2
        )
        if np.any(xy_distances < min_distance):
            return True  # Overlap detected
    
    return False  # No overlap


def place_molecules_with_rotation(
    spacer_template: Atoms,
    xy_positions: List[Tuple[float, float]],
    z: float,
    direction: str,
    min_distance: float = 2.0,
    max_rotation_attempts: int = 36,
) -> List[Atoms]:
    """
    Place molecules at XY positions with automatic rotation to avoid overlaps.
    
    For each position, tries placing the molecule. If it overlaps with already-placed
    molecules, rotates around z-axis (vertical rotation at attachment point) to find
    a non-overlapping orientation.
    
    Parameters
    ----------
    spacer_template : Atoms
        Prepared spacer molecule (already aligned)
    xy_positions : List[Tuple[float, float]]
        List of (x, y) positions where molecules should be placed
    z : float
        Z position for attachment point
    direction : str
        'up' or 'down' - direction molecules point
    min_distance : float
        Minimum XY distance between molecules (default: 2.0 Å)
    max_rotation_attempts : int
        Maximum number of rotation angles to try (default: 36 = 10° increments)
        
    Returns
    -------
    List[Atoms]
        List of placed molecules (with rotations applied to avoid overlaps)
    """
    placed_molecules = []
    
    for x, y in xy_positions:
        # Try placing without rotation first
        placed = _place_molecule(spacer_template.copy(), [x, y, z], direction)
        
        # Check for overlaps with already-placed molecules
        overlaps = False
        for existing in placed_molecules:
            if _check_molecule_overlap_xy(placed, existing, min_distance):
                overlaps = True
                break
        
        if not overlaps:
            # No overlap, use this orientation
            placed_molecules.append(placed)
            continue
        
        # Overlap detected - try rotations around z-axis
        # Find attachment point (N atom) in the original template for rotation center
        symbols = spacer_template.get_chemical_symbols()
        positions = spacer_template.get_positions()
        n_indices = [i for i, s in enumerate(symbols) if s == 'N']
        
        if n_indices:
            if direction == "up":
                # For "up", after flip NH3+ will be at MIN Z
                attach_idx = n_indices[np.argmin([positions[i, 2] for i in n_indices])]
            else:
                # For "down", NH3+ is at MAX Z
                attach_idx = n_indices[np.argmax([positions[i, 2] for i in n_indices])]
            rotation_center = positions[attach_idx].copy()
        else:
            # Fallback to center of mass
            rotation_center = spacer_template.get_center_of_mass()
        
        # Try rotations around z-axis (vertical) at attachment point
        angle_increment = 360.0 / max_rotation_attempts
        found_non_overlapping = False
        
        for attempt in range(1, max_rotation_attempts):
            angle = attempt * angle_increment
            
            # Create rotated copy of template
            mol_rotated = spacer_template.copy()
            
            # Rotate around z-axis at the attachment point
            # Translate to origin, rotate, translate back
            mol_rotated.positions -= rotation_center
            mol_rotated.rotate(angle, 'z', center=(0, 0, 0))
            mol_rotated.positions += rotation_center
            
            # Place rotated molecule at target position
            placed_rotated = _place_molecule(mol_rotated, [x, y, z], direction)
            
            # Check overlaps
            overlaps = False
            for existing in placed_molecules:
                if _check_molecule_overlap_xy(placed_rotated, existing, min_distance):
                    overlaps = True
                    break
            
            if not overlaps:
                # Found non-overlapping rotation
                placed_molecules.append(placed_rotated)
                found_non_overlapping = True
                break
        
        if not found_non_overlapping:
            # All rotations failed, use original (may have slight overlap)
            placed_molecules.append(placed)
    
    return placed_molecules


def get_spacer_extent(structure: Atoms) -> Tuple[float, float]:
    """
    Get the z-extent of spacers in a structure.
    
    Returns
    -------
    tuple of (z_min, z_max) for spacer atoms
    """
    roles = structure.arrays.get(SITE_ROLE_KEY)
    positions = structure.get_positions()
    
    if roles is None:
        return (0.0, 0.0)
    
    spacer_mask = roles == SITE_SPACER
    if not spacer_mask.any():
        return (0.0, 0.0)
    
    spacer_z = positions[spacer_mask, 2]
    return (spacer_z.min(), spacer_z.max())


def place_double_spacer_between_positions(
    molecule: Atoms,
    p1: np.ndarray,
    p2: np.ndarray,
) -> Atoms:
    """
    Place a double spacer molecule so its two NH3+ groups align with positions P1 and P2.
    
    The molecule is aligned so that:
    - One NH3+ group is at position P1
    - The other NH3+ group is at position P2
    - The molecule is oriented along the P1-P2 vector
    
    Parameters
    ----------
    molecule : Atoms
        Double spacer molecule with two NH3+ groups
    p1 : np.ndarray
        First attachment position [x, y, z] (e.g., S1 in M1 layer)
    p2 : np.ndarray
        Second attachment position [x, y, z] (e.g., S1 in M2 layer)
        
    Returns
    -------
    Atoms
        Positioned molecule with NH3+ groups at P1 and P2
    """
    mol_copy = molecule.copy()
    symbols = mol_copy.get_chemical_symbols()
    positions = mol_copy.get_positions()
    
    # Find all nitrogen atoms
    n_indices = [i for i, s in enumerate(symbols) if s == 'N']
    
    if len(n_indices) < 2:
        # Not a double spacer - place at center between P1 and P2
        center = (np.array(p1) + np.array(p2)) / 2.0
        com = mol_copy.get_center_of_mass()
        shift = center - com
        mol_copy.positions += shift
        return mol_copy
    
    # Find hydrogen atoms
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']
    h_positions = positions[h_indices] if len(h_indices) > 0 else np.array([]).reshape(0, 3)
    
    # Identify NH3+ groups: N atoms with 3 nearby H atoms
    nh_bond_cutoff = 1.2
    nh3_n_indices = []
    
    for n_idx in n_indices:
        n_pos = positions[n_idx]
        if len(h_positions) > 0:
            distances = np.linalg.norm(h_positions - n_pos, axis=1)
            nearby_h_count = np.sum(distances < nh_bond_cutoff)
            if nearby_h_count == 3:
                nh3_n_indices.append(n_idx)
        else:
            # No H atoms - assume all N are NH3+ (fallback)
            nh3_n_indices.append(n_idx)
    
    if len(nh3_n_indices) < 2:
        # Fewer than 2 NH3 groups - place at center
        center = (np.array(p1) + np.array(p2)) / 2.0
        com = mol_copy.get_center_of_mass()
        shift = center - com
        mol_copy.positions += shift
        return mol_copy
    
    # Get the two NH3+ nitrogen positions
    nh3_positions = positions[nh3_n_indices]
    
    # If more than 2 NH3 groups, use the two that are furthest apart
    if len(nh3_n_indices) > 2:
        max_dist = 0.0
        best_pair = (0, 1)
        for i in range(len(nh3_n_indices)):
            for j in range(i + 1, len(nh3_n_indices)):
                dist = np.linalg.norm(nh3_positions[j] - nh3_positions[i])
                if dist > max_dist:
                    max_dist = dist
                    best_pair = (i, j)
        nh3_1_idx = nh3_n_indices[best_pair[0]]
        nh3_2_idx = nh3_n_indices[best_pair[1]]
    else:
        nh3_1_idx = nh3_n_indices[0]
        nh3_2_idx = nh3_n_indices[1]
    
    # Get current NH3+ positions
    nh3_1_pos = positions[nh3_1_idx]
    nh3_2_pos = positions[nh3_2_idx]
    
    # Calculate vectors
    current_nh3_vector = nh3_2_pos - nh3_1_pos
    target_vector = np.array(p2) - np.array(p1)
    
    # Normalize vectors
    current_norm = np.linalg.norm(current_nh3_vector)
    target_norm = np.linalg.norm(target_vector)
    
    if current_norm < 1e-6 or target_norm < 1e-6:
        # Degenerate case - place at center
        center = (np.array(p1) + np.array(p2)) / 2.0
        com = mol_copy.get_center_of_mass()
        shift = center - com
        mol_copy.positions += shift
        return mol_copy
    
    current_unit = current_nh3_vector / current_norm
    target_unit = target_vector / target_norm
    
    # Calculate rotation to align current_nh3_vector with target_vector
    # Use Rodrigues' rotation formula
    v = np.cross(current_unit, target_unit)
    s = np.linalg.norm(v)  # sin(angle)
    c = np.dot(current_unit, target_unit)  # cos(angle)
    
    if s < 1e-6:
        # Vectors are already aligned (or anti-aligned)
        if c < 0:
            # Anti-aligned - rotate 180 degrees around perpendicular axis
            perp = np.array([1.0, 0.0, 0.0]) if abs(current_unit[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            axis = np.cross(current_unit, perp)
            axis = axis / np.linalg.norm(axis)
            # Rotate 180 degrees around nh3_1 position
            mol_copy.rotate(180.0, axis, center=nh3_1_pos)
    else:
        # Calculate rotation axis and angle
        axis = v / s
        angle_rad = np.arccos(np.clip(c, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Rotate molecule around nh3_1 position
        mol_copy.rotate(angle_deg, axis, center=nh3_1_pos)
    
    # Scale molecule if needed (stretch/compress to match target distance)
    # Get updated NH3 positions after rotation
    updated_positions = mol_copy.get_positions()
    nh3_1_pos_updated = updated_positions[nh3_1_idx]
    nh3_2_pos_updated = updated_positions[nh3_2_idx]
    current_vector_updated = nh3_2_pos_updated - nh3_1_pos_updated
    current_dist = np.linalg.norm(current_vector_updated)
    
    if current_dist > 1e-6:
        # Scale molecule along the NH3-NH3 axis to match target distance
        # This ensures the molecule spans exactly between P1 and P2
        scale_factor = target_norm / current_dist
        # Scale relative to nh3_1 position
        mol_copy.positions = nh3_1_pos_updated + (mol_copy.positions - nh3_1_pos_updated) * scale_factor
    
    # Translate so first NH3+ is at P1
    final_positions = mol_copy.get_positions()
    nh3_1_final = final_positions[nh3_1_idx]
    translation = np.array(p1) - nh3_1_final
    mol_copy.positions += translation
    
    # Verify second NH3+ is at P2 (or very close)
    final_positions_after = mol_copy.get_positions()
    nh3_2_final = final_positions_after[nh3_2_idx]
    p2_actual = np.array(p2)
    distance_error = np.linalg.norm(nh3_2_final - p2_actual)
    
    # If there's a significant error, adjust the second NH3+ position
    # This can happen due to rounding or if scaling wasn't perfect
    if distance_error > 0.01:  # 0.01 Å tolerance
        # Adjust by translating the molecule slightly
        correction = p2_actual - nh3_2_final
        mol_copy.positions += correction
    
    return mol_copy


def calculate_double_spacer_nh3_distance(molecule: Atoms) -> float:
    """
    Calculate the distance between two NH3+ groups in a double spacer molecule.
    
    For molecules with two NH3 groups (like DJ spacers), this calculates the
    distance between the two NH3+ nitrogen atoms. This distance is used to
    determine how the molecule spans between adjacent layers.
    
    Parameters
    ----------
    molecule : Atoms
        ASE Atoms object of the double spacer molecule (should have 2 NH3+ groups)
        
    Returns
    -------
    float
        Distance in Angstroms between the two NH3+ nitrogen atoms.
        Returns 0.0 if fewer than 2 NH3 groups are found.
        
    Raises
    ------
    ValueError
        If molecule has no atoms or is invalid
    """
    if len(molecule) == 0:
        raise ValueError("Molecule must have at least one atom")
    
    symbols = molecule.get_chemical_symbols()
    positions = molecule.get_positions()
    
    # Find all nitrogen atoms
    n_indices = [i for i, s in enumerate(symbols) if s == 'N']
    
    if len(n_indices) < 2:
        # Not a double spacer - return 0.0
        return 0.0
    
    # Find hydrogen atoms
    h_indices = [i for i, s in enumerate(symbols) if s == 'H']
    
    if len(h_indices) == 0:
        # No H atoms - can't identify NH3 groups
        return 0.0
    
    h_positions = positions[h_indices]
    
    # Identify NH3+ groups: N atoms with 3 nearby H atoms
    # Typical N-H bond distance is around 1.0-1.1 Å
    nh_bond_cutoff = 1.2
    
    nh3_n_indices = []
    for n_idx in n_indices:
        n_pos = positions[n_idx]
        # Calculate distances from this N to all H atoms
        distances = np.linalg.norm(h_positions - n_pos, axis=1)
        nearby_h_count = np.sum(distances < nh_bond_cutoff)
        
        # NH3+ groups have 3 H atoms nearby
        if nearby_h_count == 3:
            nh3_n_indices.append(n_idx)
    
    if len(nh3_n_indices) < 2:
        # Fewer than 2 NH3 groups found
        return 0.0
    
    # Calculate distance between the two NH3+ nitrogen atoms
    # For molecules with >2 NH3 groups, use the two that are furthest apart
    if len(nh3_n_indices) == 2:
        n1_pos = positions[nh3_n_indices[0]]
        n2_pos = positions[nh3_n_indices[1]]
        distance = np.linalg.norm(n2_pos - n1_pos)
    else:
        # More than 2 NH3 groups - find the two that are furthest apart
        nh3_positions = positions[nh3_n_indices]
        max_distance = 0.0
        for i in range(len(nh3_n_indices)):
            for j in range(i + 1, len(nh3_n_indices)):
                dist = np.linalg.norm(nh3_positions[j] - nh3_positions[i])
                if dist > max_distance:
                    max_distance = dist
        distance = max_distance
    
    return float(distance)
