"""
Generic spacer placement tools.

This module provides low-level, structure-agnostic tools for placing
atomic or molecular spacers. It includes kinematic solvers to ensure
molecular spacers are placed with physically valid bond lengths and angles.
Structure-specific logic (DJ, RP, monolayer) lives in slab.py.
"""

from __future__ import annotations

from typing import Optional, Tuple, List, Set, Dict

import numpy as np
from ase import Atoms

from .optimizers import place_spacer_with_optimizer

# Constants for site role tracking
SITE_ROLE_KEY = "site_role"
SITE_SPACER = "spacer"


def prepare_spacer(
    spacer: Atoms,
    BX_dist: float,
) -> Tuple[float, float, Atoms]:
    """
    Prepare a spacer molecule/atom for placement.
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
        n_indices, _ = _find_terminal_nitrogens(aligned)
        positions = aligned.get_positions()
        
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
    """Place a single spacer at the specified position."""
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
    """Place spacers on a grid with automatic overlap avoidance."""
    from .molecule_builder import add_atoms
    
    result = structure.copy()
    
    # Generate grid positions
    ix_grid, iy_grid = np.meshgrid(range(nx), range(ny), indexing='ij')
    x_pos = (ix_grid + shift[0]) * ax
    y_pos = (iy_grid + shift[1]) * ay
    xy_positions = list(zip(x_pos.flatten(), y_pos.flatten()))
    
    if is_atomic:
        for x, y in xy_positions:
            placed = place_spacer_at_position(spacer, x, y, z, direction, is_atomic)
            result = add_atoms(result, placed)
    else:
        placed_molecules = place_molecules_with_rotation(
            spacer, xy_positions, z, direction, min_distance=2.0
        )
        for placed in placed_molecules:
            placed.arrays[SITE_ROLE_KEY] = np.array([SITE_SPACER] * len(placed), dtype=object)
            result = add_atoms(result, placed)
    
    return result


def _find_terminal_nitrogens(mol: Atoms) -> Tuple[List[int], List[int]]:
    """
    Helper to identify indices of NH3+ nitrogens and all nitrogens.

    Enhanced with more robust NH3+ detection based on mofun approach:
    - N-H bond distance (1.0-1.1 Å typical)
    - H-N-H angle (~109° for sp3)
    - Coordination number

    Returns: (all_n_indices, nh3_n_indices)
    """
    symbols = mol.get_chemical_symbols()
    positions = mol.get_positions()
    n_indices = [i for i, s in enumerate(symbols) if s == 'N']

    h_indices = [i for i, s in enumerate(symbols) if s == 'H']

    if not h_indices:
        # Fallback if no hydrogens: assume all N are attachment points
        return n_indices, n_indices

    h_positions = positions[h_indices]
    nh3_n_indices = []

    # Enhanced NH3+ detection
    nh_bond_cutoff = 1.2  # Typical N-H bond distance is 1.0-1.1 Å

    for n_idx in n_indices:
        n_pos = positions[n_idx]

        # Find H atoms within bond distance
        h_dists = np.linalg.norm(h_positions - n_pos, axis=1)
        nearby_h = [h_indices[i] for i, d in enumerate(h_dists) if 0.9 < d < nh_bond_cutoff]

        if len(nearby_h) == 3:
            # Check angles (should be ~109° for sp3 NH3+)
            h_positions_nearby = positions[nearby_h]
            angles = []

            for i in range(3):
                v1 = h_positions_nearby[i] - n_pos
                v2 = h_positions_nearby[(i+1)%3] - n_pos
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                angles.append(np.arccos(np.clip(cos_angle, -1, 1)))

            avg_angle = np.mean(angles) * 180 / np.pi
            if 100 < avg_angle < 120:  # Reasonable sp3 angle range
                nh3_n_indices.append(n_idx)
        elif len(nearby_h) >= 2:
            # Allow NH2+ groups as well (for robustness)
            nh3_n_indices.append(n_idx)

    return n_indices, nh3_n_indices


def count_nh3_groups(mol: Atoms) -> int:
    """
    Count the number of NH3+ groups in a molecule.

    This is a convenience wrapper around _find_terminal_nitrogens().

    Parameters
    ----------
    mol : Atoms
        The molecule to analyze

    Returns
    -------
    int
        Number of NH3+ groups found
    """
    _, nh3_n_indices = _find_terminal_nitrogens(mol)
    return len(nh3_n_indices)


def _place_molecule(
    mol: Atoms,
    target_xyz: List[float],
    direction: str,
) -> Atoms:
    """Place molecule with terminal NH3 attached to surface."""
    mol_copy = mol.copy()
    
    if direction == "up":
        mol_copy.rotate(180, "x")
    
    n_indices, _ = _find_terminal_nitrogens(mol_copy)
    positions = mol_copy.get_positions()
    
    if n_indices:
        # For 'up': NH3+ should be at MIN Z (bottom). For 'down': MAX Z (top).
        z_vals = positions[n_indices, 2]
        target_n_idx = n_indices[np.argmin(z_vals)] if direction == "up" else n_indices[np.argmax(z_vals)]
        current = positions[target_n_idx]
    else:
        # Fallback to absolute bounds
        idx = np.argmin(positions[:, 2]) if direction == "up" else np.argmax(positions[:, 2])
        current = positions[idx]
    
    mol_copy.positions += np.array(target_xyz) - current
    return mol_copy


def _check_molecule_overlap_xy(mol1: Atoms, mol2: Atoms, min_distance: float = 2.0) -> bool:
    """Check XY overlap using heavy atoms only."""
    pos1 = mol1.get_positions()
    pos2 = mol2.get_positions()
    sym1 = mol1.get_chemical_symbols()
    sym2 = mol2.get_chemical_symbols()
    
    # Heavy atom filter
    mask1 = np.array([s != 'H' for s in sym1])
    mask2 = np.array([s != 'H' for s in sym2])
    
    p1_check = pos1[mask1] if np.any(mask1) else pos1
    p2_check = pos2[mask2] if np.any(mask2) else pos2
    
    # Vectorized XY distance check
    # Shape: (N1, 1, 2) - (1, N2, 2) -> (N1, N2, 2)
    diff = p1_check[:, :2][:, np.newaxis, :] - p2_check[:, :2][np.newaxis, :, :]
    dists_sq = np.sum(diff**2, axis=2)
    
    return np.any(dists_sq < min_distance**2)


def place_molecules_with_rotation(
    spacer_template: Atoms,
    xy_positions: List[Tuple[float, float]],
    z: float,
    direction: str,
    min_distance: float = 2.0,
    max_rotation_attempts: int = 36,
) -> List[Atoms]:
    """Place molecules grid with rotation optimization."""
    placed_molecules = []
    
    # Pre-calculate rotation center (attachment point)
    n_indices, _ = _find_terminal_nitrogens(spacer_template)
    positions = spacer_template.get_positions()
    
    if n_indices:
        z_vals = positions[n_indices, 2]
        # Attachment point logic matches _place_molecule (roughly)
        # Note: direction is handled inside _place_molecule, so here we find center relative to raw template
        # We need the logical "center" for Z-rotation.
        # If 'up', template is flipped later, so bottom becomes top.
        # Safe bet: use Center of Mass for XY rotation or the "bottom-most" N
        attach_idx = n_indices[np.argmin(positions[n_indices, 2])]
        rotation_center_local = positions[attach_idx]
    else:
        rotation_center_local = spacer_template.get_center_of_mass()

    angle_increment = 360.0 / max_rotation_attempts

    for x, y in xy_positions:
        # 1. Try default orientation
        candidate = _place_molecule(spacer_template.copy(), [x, y, z], direction)
        
        # Fast overlap check
        if not any(_check_molecule_overlap_xy(candidate, existing, min_distance) for existing in placed_molecules):
            placed_molecules.append(candidate)
            continue
            
        # 2. Try rotations
        found = False
        # Calculate the actual rotation pivot in global space (x,y,z is the attachment point)
        pivot = np.array([x, y, z])
        
        for attempt in range(1, max_rotation_attempts):
            angle = attempt * angle_increment
            
            # Rotate the CANDIDATE in place around its attachment point
            rotated_candidate = candidate.copy()
            # Since candidate is already placed at [x,y,z], we rotate around that point
            rotated_candidate.rotate(angle, 'z', center=pivot)
            
            if not any(_check_molecule_overlap_xy(rotated_candidate, existing, min_distance) for existing in placed_molecules):
                placed_molecules.append(rotated_candidate)
                found = True
                break
        
        if not found:
            placed_molecules.append(candidate) # Fallback

    return placed_molecules


def place_double_spacer_between_positions(
    molecule: Atoms,
    p1: np.ndarray,
    p2: np.ndarray,
    optimizer: str = "KS",
    cell: Optional[np.ndarray] = None,
    target_vector: Optional[np.ndarray] = None,
) -> Atoms:
    """
    Physically aligns a flexible double spacer between two points using the specified optimizer.
    
    Parameters
    ----------
    molecule : Atoms
        The molecule to place
    p1 : np.ndarray
        Target position for first terminal NH3+ nitrogen
    p2 : np.ndarray
        Target position for second terminal NH3+ nitrogen
    optimizer : str, default "KS"
        Optimizer to use: "Off" (pure geometry), "KS" (Kinematic Solver), or "UFF" (UFF optimization)
    cell : np.ndarray, optional
        Unit cell matrix (3x3) for PBC-aware shortest vector calculation
    target_vector : np.ndarray, optional
        Explicit vector from p1 to p2. If provided, overrides internal shortest path calculation.
        
    Returns
    -------
    Atoms
        Aligned molecule between p1 and p2
    """
    return place_spacer_with_optimizer(molecule, p1, p2, optimizer=optimizer, cell=cell, target_vector=target_vector)


def calculate_double_spacer_nh3_distances(molecule: Atoms) -> float:
    """Calculates distance between terminal NH3 groups."""
    if len(molecule) == 0: return 0.0
    
    _, nh3_indices = _find_terminal_nitrogens(molecule)
    
    if len(nh3_indices) < 2: return 0.0
    
    positions = molecule.get_positions()
    if len(nh3_indices) == 2:
        return float(np.linalg.norm(positions[nh3_indices[0]] - positions[nh3_indices[1]]))
        
    # Max distance logic for >2
    nh3_pos = positions[nh3_indices]
    # Vectorized max distance
    # (N, 1, 3) - (1, N, 3) -> (N, N, 3)
    diff = nh3_pos[:, np.newaxis, :] - nh3_pos[np.newaxis, :, :]
    return float(np.max(np.linalg.norm(diff, axis=2)))


def calculate_molecule_radius(molecule: Atoms, n1_index: Optional[int] = None, n2_index: Optional[int] = None) -> float:
    """Return effective cylindrical radius relative to N-N axis."""
    positions = molecule.get_positions()

    if n1_index is None or n2_index is None:
        _, nh3_indices = _find_terminal_nitrogens(molecule)
        if len(nh3_indices) < 2: return 0.0
        n1_index, n2_index = nh3_indices[0], nh3_indices[1]

    N1 = positions[n1_index]
    N2 = positions[n2_index]
    axis_vec = N2 - N1
    axis_len = np.linalg.norm(axis_vec)

    if axis_len < 1e-6: return 0.0

    # Vector rejection: v - proj_v_on_axis
    # Shift to N1 origin
    rel_pos = positions - N1
    # Projection scalar
    proj_scalar = np.dot(rel_pos, axis_vec) / (axis_len**2)
    # Projection vector
    proj_vec = proj_scalar[:, np.newaxis] * axis_vec
    # Rejection (orthogonal component)
    orth_vec = rel_pos - proj_vec
    dists = np.linalg.norm(orth_vec, axis=1)

    # Exclude the defining atoms themselves to avoid float errors
    mask = np.ones(len(molecule), dtype=bool)
    mask[[n1_index, n2_index]] = False

    return float(dists[mask].max()) if np.any(mask) else 0.0


def replace_spacer_molecule(
    structure: Atoms,
    old_spacer_pattern: Atoms,
    new_spacer: Atoms,
    cell: Optional[np.ndarray] = None
) -> Atoms:
    """
    Find and replace spacer molecules in structure.

    Based on mofun's find/replace approach, adapted for ASE Atoms.
    Uses pattern matching to find old_spacer_pattern instances,
    then replaces them with new_spacer while preserving alignment.

    Parameters
    ----------
    structure : ase.Atoms
        The structure containing spacers to replace
    old_spacer_pattern : ase.Atoms
        Pattern of the spacer to replace
    new_spacer : ase.Atoms
        New spacer molecule to insert
    cell : np.ndarray, optional
        Unit cell for PBC-aware operations

    Returns
    -------
    ase.Atoms
        Structure with spacers replaced
    """
    from .molecule_builder import find_molecule_patterns

    # Find all instances of the old pattern
    matches = find_molecule_patterns(structure, old_spacer_pattern)

    if not matches:
        return structure.copy()

    # Process each match
    new_structure = structure.copy()

    for match_indices in matches:
        # Get positions of matched atoms
        match_positions = structure.positions[list(match_indices)]

        # Align new_spacer to match the old spacer's position and orientation
        aligned_new = _align_spacer_to_positions(new_spacer, match_positions, cell)

        # Replace atoms: remove old, add new
        # For simplicity, we'll replace atom-by-atom if same size,
        # otherwise we'll need more complex logic
        if len(aligned_new) == len(match_indices):
            # Same number of atoms: direct replacement
            new_structure.positions[list(match_indices)] = aligned_new.positions
            new_structure.set_chemical_symbols([
                aligned_new.get_chemical_symbols()[i] if i < len(aligned_new) else structure.get_chemical_symbols()[match_indices[i]]
                for i in range(len(match_indices))
            ])
        else:
            # Different sizes: remove old atoms and add new ones
            # This is more complex and would require careful handling of indices
            # For now, skip this case
            continue

    return new_structure


def _align_spacer_to_positions(spacer: Atoms, target_positions: np.ndarray,
                              cell: Optional[np.ndarray] = None) -> Atoms:
    """
    Align spacer molecule to match target positions.

    Simplified alignment: translate to center of mass, then align principal axes.
    """
    aligned = spacer.copy()

    # Center both molecules
    target_com = target_positions.mean(axis=0)
    spacer_com = aligned.get_center_of_mass()

    aligned.translate(target_com - spacer_com)

    # For more sophisticated alignment, would need:
    # 1. Principal component analysis
    # 2. Rotation matrix calculation
    # 3. Quaternion-based alignment

    return aligned


def analyze_molecule_topology(molecule: Atoms, cell: Optional[np.ndarray] = None) -> dict:
    """
    Analyze molecular connectivity and topology for better kinematic solver initialization.

    Based on mofun approach: builds bond graph and identifies rotatable bonds.

    Parameters
    ----------
    molecule : ase.Atoms
        The molecule to analyze
    cell : np.ndarray, optional
        Unit cell for PBC-aware bond detection

    Returns
    -------
    dict
        Topology analysis results:
        - 'bonds': List of (i,j) bond tuples
        - 'cycles': List of cycles (rings)
        - 'rotatable_bonds': Bonds that can be rotated
        - 'graph': NetworkX graph object
        - 'terminal_groups': Indices of terminal functional groups
    """
    import networkx as nx
    from .optimizers import detect_bonds_pbc

    # Detect bonds
    if cell is not None:
        bonds = detect_bonds_pbc(molecule, cell)
    else:
        # Fallback to simple distance-based bonding
        bonds = []
        positions = molecule.get_positions()
        symbols = molecule.get_chemical_symbols()

        for i in range(len(molecule)):
            for j in range(i + 1, len(molecule)):
                dist = np.linalg.norm(positions[i] - positions[j])
                # Simple bonding criteria
                if dist < 2.0:  # Rough cutoff
                    bonds.append((i, j))

    # Build connectivity graph
    G = nx.Graph()
    G.add_nodes_from(range(len(molecule)))
    G.add_edges_from(bonds)

    # Find cycles (rigid rings)
    cycles = []
    try:
        cycles = nx.cycle_basis(G)
    except nx.NetworkXNoCycle:
        pass

    # Identify rotatable bonds (single bonds not in cycles)
    rotatable_bonds = []
    for bond in bonds:
        i, j = bond
        # Check if bond is in any cycle
        in_cycle = any(bond in cycle or (bond[1], bond[0]) in cycle for cycle in cycles)
        if not in_cycle:
            rotatable_bonds.append(bond)

    # Find terminal functional groups (NH3+, etc.)
    _, nh3_indices = _find_terminal_nitrogens(molecule)
    terminal_groups = nh3_indices

    return {
        'bonds': bonds,
        'cycles': cycles,
        'rotatable_bonds': rotatable_bonds,
        'graph': G,
        'terminal_groups': terminal_groups,
        'num_atoms': len(molecule),
        'num_bonds': len(bonds),
        'num_cycles': len(cycles),
        'num_rotatable': len(rotatable_bonds)
    }