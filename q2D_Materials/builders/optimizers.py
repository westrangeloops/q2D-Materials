"""Optimizers for flexible spacer molecules."""

from __future__ import annotations

from typing import List, Set, Tuple, Optional, Union, Dict
import warnings
import itertools
import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import build_neighbor_list



class KinematicChainSolver:
    """CCD-based inverse kinematics solver respecting chemical rigidity (rings, double bonds)."""

    # Element pairs that can form bonds (optimization: skip distance checks for non-bondable pairs)
    BONDABLE_PAIRS = {
        ('H', 'H'), ('H', 'C'), ('H', 'N'), ('H', 'O'), ('H', 'F'), ('H', 'P'), ('H', 'S'), ('H', 'Cl'),
        ('C', 'C'), ('C', 'N'), ('C', 'O'), ('C', 'F'), ('C', 'P'), ('C', 'S'), ('C', 'Cl'),
        ('N', 'N'), ('N', 'O'), ('N', 'F'), ('N', 'P'), ('N', 'S'), ('N', 'Cl'),
        ('O', 'O'), ('O', 'F'), ('O', 'P'), ('O', 'S'), ('O', 'Cl'),
        ('F', 'F'), ('F', 'P'), ('F', 'S'), ('F', 'Cl'),
        ('P', 'P'), ('P', 'S'), ('P', 'Cl'),
        ('S', 'S'), ('S', 'Cl'),
        ('Cl', 'Cl'),
        # Include common metals and halides
        ('Li', 'C'), ('Li', 'N'), ('Li', 'O'), ('Li', 'F'), ('Li', 'Cl'),
        ('Na', 'C'), ('Na', 'N'), ('Na', 'O'), ('Na', 'F'), ('Na', 'Cl'),
        ('K', 'C'), ('K', 'N'), ('K', 'O'), ('K', 'F'), ('K', 'Cl'),
        ('Rb', 'C'), ('Rb', 'N'), ('Rb', 'O'), ('Rb', 'F'), ('Rb', 'Cl'),
        ('Cs', 'C'), ('Cs', 'N'), ('Cs', 'O'), ('Cs', 'F'), ('Cs', 'Cl'),
        ('Mg', 'C'), ('Mg', 'N'), ('Mg', 'O'), ('Mg', 'F'), ('Mg', 'Cl'),
        ('Ca', 'C'), ('Ca', 'N'), ('Ca', 'O'), ('Ca', 'F'), ('Ca', 'Cl'),
        ('Sr', 'C'), ('Sr', 'N'), ('Sr', 'O'), ('Sr', 'F'), ('Sr', 'Cl'),
        ('Ba', 'C'), ('Ba', 'N'), ('Ba', 'O'), ('Ba', 'F'), ('Ba', 'Cl'),
        ('Pb', 'C'), ('Pb', 'N'), ('Pb', 'O'), ('Pb', 'F'), ('Pb', 'Cl'), ('Pb', 'I'),
        ('Sn', 'C'), ('Sn', 'N'), ('Sn', 'O'), ('Sn', 'F'), ('Sn', 'Cl'),
        ('Ge', 'C'), ('Ge', 'N'), ('Ge', 'O'), ('Ge', 'F'), ('Ge', 'Cl'),
        ('Bi', 'C'), ('Bi', 'N'), ('Bi', 'O'), ('Bi', 'F'), ('Bi', 'Cl'),
        ('Sb', 'C'), ('Sb', 'N'), ('Sb', 'O'), ('Sb', 'F'), ('Sb', 'Cl'),
        ('In', 'C'), ('In', 'N'), ('In', 'O'), ('In', 'F'), ('In', 'Cl'),
        ('Ga', 'C'), ('Ga', 'N'), ('Ga', 'O'), ('Ga', 'F'), ('Ga', 'Cl'),
        ('Al', 'C'), ('Al', 'N'), ('Al', 'O'), ('Al', 'F'), ('Al', 'Cl'),
        ('Si', 'C'), ('Si', 'N'), ('Si', 'O'), ('Si', 'F'), ('Si', 'Cl'),
        ('B', 'C'), ('B', 'N'), ('B', 'O'), ('B', 'F'), ('B', 'Cl'),
    }

    def __init__(self, atoms: Atoms, cutoff_buffer: float = 1.2, double_bond_threshold: float = 0.94):
        """Initialize solver. cutoff_buffer: covalent radii multiplier. double_bond_threshold: rigid bond threshold."""
        self.atoms = atoms.copy()
        self.num_atoms = len(atoms)
        self.positions = self.atoms.get_positions()

        # 1. Build Connectivity Graph
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.num_atoms))

        # Get radii for connectivity checks
        radii = [covalent_radii[z] for z in atoms.numbers]

        # Build neighbor list
        nl = build_neighbor_list(atoms, cutoffs=[r * cutoff_buffer for r in radii], self_interaction=False)

        self.rigid_bonds: Set[Tuple[int, int]] = set()

        # Process edges from neighbor list
        for i in range(self.num_atoms):
            neighbors, _ = nl.get_neighbors(i)
            for j in neighbors:
                j = int(j)
                if i > j:
                    continue  # handle pair once

                # EARLY ELEMENT FILTERING: Skip distance calculation if elements can't bond
                elem_i = atoms.get_chemical_symbols()[i]
                elem_j = atoms.get_chemical_symbols()[j]
                if (elem_i, elem_j) not in self.BONDABLE_PAIRS and (elem_j, elem_i) not in self.BONDABLE_PAIRS:
                    continue

                dist = np.linalg.norm(self.positions[i] - self.positions[j])

                # Add to graph
                self.G.add_edge(i, j)

                # RIGIDITY CHECK 1: bond length (double / partial bonds)
                ideal_len = radii[i] + radii[j]
                if dist < (ideal_len * double_bond_threshold):
                    self.rigid_bonds.add(tuple(sorted((i, j))))

        # RIGIDITY CHECK 2: cycles (rings)
        try:
            cycles = nx.cycle_basis(self.G)
            for cycle in cycles:
                for k in range(len(cycle)):
                    u, v = cycle[k], cycle[(k + 1) % len(cycle)]
                    self.rigid_bonds.add(tuple(sorted((u, v))))
        except nx.NetworkXNoCycle:
            pass

    def _get_subtree(self, pivot_atom: int, child_atom: int) -> List[int]:
        """Find the branch of the molecule attached to child_atom if we cut the bond."""
        if self.G.has_edge(pivot_atom, child_atom):
            self.G.remove_edge(pivot_atom, child_atom)
            subtree = list(nx.node_connected_component(self.G, child_atom))
            self.G.add_edge(pivot_atom, child_atom)
            return subtree
        return []

    def solve(self, anchor_idx: int, mover_idx: int, anchor_pos: np.ndarray,
              target_pos: np.ndarray, tolerance: float = 0.1, max_iter: int = 100) -> Atoms:
        """Align molecule: anchor_idx at anchor_pos, mover_idx near target_pos."""
        # 1. Pre-translation: pin anchor to anchor_pos
        curr_pos = self.atoms.get_positions()
        shift = np.array(anchor_pos) - curr_pos[anchor_idx]
        self.atoms.translate(shift)

        target_pos = np.array(target_pos)

        # 2. Find path in bond graph from anchor to mover
        try:
            path = nx.shortest_path(self.G, source=anchor_idx, target=mover_idx)
        except nx.NetworkXNoPath:
            return self.atoms

        # 3. CCD loop
        best_error = float("inf")
        best_atoms = self.atoms.copy()

        for _ in range(max_iter):
            current_error = np.linalg.norm(self.atoms.positions[mover_idx] - target_pos)
            if current_error < best_error:
                best_error = current_error
                best_atoms = self.atoms.copy()

            if current_error < tolerance:
                break

            # Walk backwards from mover -> anchor; skip last link
            for i in range(len(path) - 2, 0, -1):
                pivot = path[i - 1]
                child = path[i]

                # skip rigid bonds
                edge_key = tuple(sorted((pivot, child)))
                if edge_key in self.rigid_bonds:
                    continue

                moving_indices = self._get_subtree(pivot, child)
                if mover_idx not in moving_indices:
                    continue

                # Geometry
                pos = self.atoms.get_positions()
                pivot_loc = pos[pivot]
                mover_loc = pos[mover_idx]

                axis = pos[child] - pivot_loc
                axis_len = np.linalg.norm(axis)
                if axis_len < 1e-3:
                    continue
                axis /= axis_len

                r_cur = mover_loc - pivot_loc
                r_tar = target_pos - pivot_loc

                # remove component parallel to axis
                r_cur_perp = r_cur - np.dot(r_cur, axis) * axis
                r_tar_perp = r_tar - np.dot(r_tar, axis) * axis

                norm_cur = np.linalg.norm(r_cur_perp)
                norm_tar = np.linalg.norm(r_tar_perp)
                if norm_cur < 1e-2 or norm_tar < 1e-2:
                    continue

                r_cur_perp /= norm_cur
                r_tar_perp /= norm_tar

                cross = np.cross(r_cur_perp, r_tar_perp)
                dot = np.dot(r_cur_perp, r_tar_perp)
                angle = np.arctan2(np.dot(cross, axis), dot)

                if abs(angle) > 1e-4:
                    angle_deg = np.degrees(angle * 0.5)
                    moving_positions = pos[moving_indices]
                    # translate to pivot origin
                    moving_centered = moving_positions - pivot_loc
                    cos_a = np.cos(np.radians(angle_deg))
                    sin_a = np.sin(np.radians(angle_deg))
                    dot_products = np.dot(moving_centered, axis)
                    cross_products = np.cross(axis, moving_centered)
                    rotated = (
                        moving_centered * cos_a
                        + cross_products * sin_a
                        + axis * dot_products[:, np.newaxis] * (1 - cos_a)
                    )
                    new_positions = rotated + pivot_loc
                    self.atoms.positions[moving_indices] = new_positions

        # use best solution
        self.atoms = best_atoms

        # ensure anchor is at anchor_pos
        final_pos = self.atoms.get_positions()
        anchor_final = final_pos[anchor_idx]
        anchor_error = np.linalg.norm(anchor_final - anchor_pos)
        if anchor_error > 0.01:
            correction = anchor_pos - anchor_final
            self.atoms.translate(correction)

        return self.atoms


def _find_directional_xy_pbc_vector(p1: np.ndarray, p2: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """
    Find the vector from p1 to p2 deterministically based on fractional coordinates.
    Z coordinate is kept fixed (no wrapping in Z direction).
    
    This function determines which periodic cell p2 is in based on its fractional
    coordinates and selects the vector that points to that specific cell.
    
    Rules:
    - X=0.N Y=0.N: within the center cell (0,0) -> shift (0,0)
    - X=-0.N Y=0.N: cell to the left (-1,0) -> shift (-1,0)  
    - X=1.N Y=0.N: cell to the right (1,0) -> shift (1,0)
    - X=0.N Y=1.N: upper cell (0,1) -> shift (0,1)
    - X=0.N Y=-0.N: lower cell (0,-1) -> shift (0,-1)
    - Diagonals: X=-1.N Y=-1.N -> shift (-1,-1), X=1.N Y=1.N -> shift (1,1), etc.

    Parameters
    ----------
    p1 : np.ndarray
        Ground point [x, y, z]
    p2 : np.ndarray
        Sky point [x, y, z]
    cell : np.ndarray
        Unit cell matrix (3x3)

    Returns
    -------
    np.ndarray
        Vector from p1 to p2 pointing to the correct periodic cell
    """
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    cell = np.asarray(cell, dtype=np.float64)
    
    # Calculate base difference
    raw_diff = p2 - p1
    
    try:
        inv_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return raw_diff
    
    # Convert both points to fractional coordinates
    p1_frac = p1 @ inv_cell.T  # Shape: (3,)
    p2_frac = p2 @ inv_cell.T  # Shape: (3,)
    
    # Get XY fractional components
    p1_frac_xy = p1_frac[:2]
    p2_frac_xy = p2_frac[:2]
    
    # Determine which cell p2 is in based on integer part of fractional coordinates
    # Use floor to handle negative coordinates correctly
    # For example: -0.2 -> floor(-0.2) = -1, 1.2 -> floor(1.2) = 1, 0.5 -> floor(0.5) = 0
    cell_u = int(np.floor(p2_frac_xy[0]))
    cell_v = int(np.floor(p2_frac_xy[1]))
    
    # For salts templates, positions can be outside the 3x3 grid
    # Don't clamp - allow any integer cell offset
    pass
    
    # We want the vector to point TO the correct periodic image of p2
    # First, wrap p2 to the center cell
    shift = np.array([cell_u, cell_v])
    p2_frac_xy_wrapped = p2_frac_xy - shift

    # Convert wrapped p2 back to Cartesian
    p2_frac_wrapped = np.array([p2_frac_xy_wrapped[0], p2_frac_xy_wrapped[1], p2_frac[2]])
    p2_wrapped = p2_frac_wrapped @ cell

    # Vector to wrapped p2 in center cell
    vec_to_wrapped = p2_wrapped - p1

    # Add the cell offset to get vector to correct periodic image
    cell_offset = cell_u * cell[0] + cell_v * cell[1]  # Cartesian cell offset
    final_vec = vec_to_wrapped + cell_offset

    return final_vec


def place_spacer_geometric(molecule: Atoms, p1: np.ndarray, p2: np.ndarray,
                           cell: Optional[np.ndarray] = None,
                           target_vector: Optional[np.ndarray] = None) -> Atoms:
    """Rigid translation/rotation to align terminal NH3+ groups to p1 and p2."""
    from .spacer import _find_terminal_nitrogens
    
    mol_copy = molecule.copy()
    _, nh3_indices = _find_terminal_nitrogens(mol_copy)
    positions = mol_copy.get_positions()
    
    # Determine the vector to use (Explicit or Calculated)
    if target_vector is not None:
        vec = target_vector
    elif cell is not None:
        vec = _find_directional_xy_pbc_vector(p1, p2, cell)
    else:
        vec = np.array(p2) - np.array(p1)
    
    if len(nh3_indices) < 2:
        # Fallback: Rigid translation to center
        center = np.array(p1) + vec / 2.0
        mol_copy.translate(center - mol_copy.get_center_of_mass())
        return mol_copy
    
    # Pick the two furthest NH3 groups if > 2
    if len(nh3_indices) > 2:
        max_d = -1.0
        best_pair = (0, 1)
        nh3_pos = positions[nh3_indices]
        for i in range(len(nh3_indices)):
            for j in range(i+1, len(nh3_indices)):
                d = np.linalg.norm(nh3_pos[i] - nh3_pos[j])
                if d > max_d:
                    max_d = d
                    best_pair = (i, j)
        idx1, idx2 = nh3_indices[best_pair[0]], nh3_indices[best_pair[1]]
    else:
        idx1, idx2 = nh3_indices[0], nh3_indices[1]
    
    # Get current positions
    n1_pos = positions[idx1]
    n2_pos = positions[idx2]
    
    # Translate so first N is at p1
    mol_copy.translate(p1 - n1_pos)
    
    # Rotate so second N aligns with p2
    current_vec = positions[idx2] - positions[idx1]
    
    current_len = np.linalg.norm(current_vec)
    target_len = np.linalg.norm(vec)
    
    if current_len > 1e-6 and target_len > 1e-6:
        # Normalize vectors
        current_unit = current_vec / current_len
        target_unit = vec / target_len
        
        # Calculate rotation axis and angle
        cross = np.cross(current_unit, target_unit)
        dot = np.clip(np.dot(current_unit, target_unit), -1.0, 1.0)
        
        if np.linalg.norm(cross) > 1e-6:
            # Rotation needed
            axis = cross / np.linalg.norm(cross)
            angle = np.arccos(dot)
            mol_copy.rotate(np.degrees(angle), v=axis, center=p1)
        elif dot < 0:
            # Vectors are opposite, rotate 180 degrees around perpendicular axis
            perp = np.array([1, 0, 0]) if abs(current_unit[0]) < 0.9 else np.array([0, 1, 0])
            axis = np.cross(current_unit, perp)
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                mol_copy.rotate(180, v=axis, center=p1)
    
    # Final adjustment to align ends to p1 and virtual p2
    final_positions = mol_copy.get_positions()
    final_n1 = final_positions[idx1]
    final_n2 = final_positions[idx2]
    
    # The virtual P2 is P1 + the chosen vector
    virtual_p2 = p1 + vec
    
    # Adjust both positions to match targets
    correction = (p1 - final_n1 + virtual_p2 - final_n2) / 2.0
    mol_copy.translate(correction)

    # Basic check: ensure molecule is oriented along the placement vector
    final_positions = mol_copy.get_positions()
    placement_axis = (p2 - p1) / np.linalg.norm(p2 - p1)
    # Simple check without graph analysis for geometric placement
    n1_pos = final_positions[idx1]
    n2_pos = final_positions[idx2]
    rel_pos = final_positions - n1_pos
    proj_scalars = np.dot(rel_pos, placement_axis)
    proj_vecs = proj_scalars[:, np.newaxis] * placement_axis
    orth_vecs = rel_pos - proj_vecs
    distances_from_axis = np.linalg.norm(orth_vecs, axis=1)
    max_distance = np.max(distances_from_axis)

    if max_distance > 2.0:  # If not straight, try to elongate it slightly
        # If not straight, try to elongate it slightly
        try:
            elongated = _elongate_single_molecule(mol_copy, step_size=0.1, max_iterations=10,
                                                 target_distance=np.linalg.norm(p2 - p1))
            if elongated is not None:
                mol_copy = elongated
        except:
            pass  # Keep original if elongation fails

    # Wrap atoms back into the unit cell before returning
    mol_copy.wrap()

    return mol_copy


def place_spacer_with_optimizer(molecule: Atoms, p1: np.ndarray, p2: np.ndarray,
                                optimizer: str = "KS", cell: Optional[np.ndarray] = None,
                                target_vector: Optional[np.ndarray] = None) -> Atoms:
    """Place spacer between p1 and p2. Options: "Off" (geometric only), "KS" (elongate + KS, default)."""
    optimizer = optimizer.upper()
    
    if optimizer == "OFF":
        return place_spacer_geometric(molecule, p1, p2, cell=cell, target_vector=target_vector)
        
    else:  # KS or UFF: do elongation if needed, then geometric placement
        from .spacer import _find_terminal_nitrogens
        _, nh3_indices = _find_terminal_nitrogens(molecule)

        # Check if molecule needs elongation
        positions = molecule.get_positions()
        current_distance = np.linalg.norm(positions[nh3_indices[1]] - positions[nh3_indices[0]])

        if len(nh3_indices) >= 2:
            # Double spacer: Check if already reasonably elongated
            min_reasonable_distance = 8.0  # Å - reasonable minimum for double spacers

            if current_distance < min_reasonable_distance:
                # Need elongation first
                molecule = _elongate_single_molecule(molecule, step_size=0.5, max_iterations=100,
                                                   target_distance=None)  # None = use default max elongation

        # Use geometric placement for all cases (preserves molecular conformation)
        return place_spacer_geometric(molecule, p1, p2, cell=cell, target_vector=target_vector)

        # Pick the two furthest NH3 groups if > 2
        positions = elongated.get_positions()
        if len(nh3_indices) > 2:
            max_d = -1.0
            best_pair = (0, 1)
            nh3_pos = positions[nh3_indices]
            for i in range(len(nh3_indices)):
                for j in range(i+1, len(nh3_indices)):
                    d = np.linalg.norm(nh3_pos[i] - nh3_pos[j])
                    if d > max_d:
                        max_d = d
                        best_pair = (i, j)
            idx1, idx2 = nh3_indices[best_pair[0]], nh3_indices[best_pair[1]]
        else:
            idx1, idx2 = nh3_indices[0], nh3_indices[1]

        aligned_mol = solver.solve(
            anchor_idx=idx1,
            mover_idx=idx2,
            anchor_pos=p1,
            target_pos=p1 + vec,
            tolerance=0.1,
            max_iter=150
        )
        return aligned_mol


def _elongate_single_molecule(molecule: str | Atoms, step_size: float = 0.5,
                              max_iterations: int = 100, target_distance: Optional[float] = None) -> Atoms:
    """Elongate molecule towards target N-N distance using kinematic solver."""
    # Convert SMILES to Atoms if needed
    if isinstance(molecule, str):
        from .molecule_builder import smiles_to_ase_atoms
        mol_atoms = smiles_to_ase_atoms(molecule)
    else:
        mol_atoms = molecule.copy()

    # Find terminal NH3+ groups (using our enhanced detection)
    from .spacer import _find_terminal_nitrogens
    _, nh3_indices = _find_terminal_nitrogens(mol_atoms)

    if len(nh3_indices) < 2:
        # Not enough NH3+ groups to elongate
        return mol_atoms

    # Get initial positions and current distance
    positions = mol_atoms.get_positions()
    n1_pos = positions[nh3_indices[0]]
    n2_pos = positions[nh3_indices[1]]
    current_distance = np.linalg.norm(n2_pos - n1_pos)

    # Unit vector along N-N axis
    direction = (n2_pos - n1_pos) / current_distance

    # Helper function to check if molecule is straight along N-N axis
    def is_molecule_straight(positions, n1_idx, n2_idx, axis_vec, mol_atoms):
        """Check if molecule is oriented correctly along the N-N axis."""
        n1_pos = positions[n1_idx]
        n2_pos = positions[n2_idx]

        # Get all atom positions relative to N1
        rel_pos = positions - n1_pos

        # Project onto the N-N axis
        proj_scalars = np.dot(rel_pos, axis_vec)

        # Check that N2 is further along the axis than N1 (basic sanity check)
        n1_proj = proj_scalars[n1_idx]
        n2_proj = proj_scalars[n2_idx]

        if n2_proj <= n1_proj:
            return False  # N2 should be further along the axis than N1

        # Check perpendicular distances from axis - should be reasonable
        axis_length = np.linalg.norm(axis_vec)
        if axis_length < 1e-6:
            return False

        # Project positions onto axis
        axis_unit = axis_vec / axis_length
        proj_vecs = proj_scalars[:, np.newaxis] * axis_unit
        orth_vecs = rel_pos - proj_vecs
        distances_from_axis = np.linalg.norm(orth_vecs, axis=1)

        # For a straight molecule, most atoms should be close to the axis
        # Allow some tolerance for molecular structure
        max_distance = np.max(distances_from_axis)
        return max_distance <= 2.0  # 2Å tolerance for molecular width

    # If no target distance provided, use the old maximum elongation behavior
    if target_distance is None:
        target_distance = current_distance * 2.5  # Old behavior: maximize
    else:
        # Ensure target is reasonable (not less than current distance)
        target_distance = max(target_distance, current_distance)

    # Initialize best result tracking
    best_atoms = mol_atoms.copy()
    best_distance = current_distance
    best_distance_error = abs(current_distance - target_distance)

    # Check if the initial molecule is reasonably oriented
    initial_positions = mol_atoms.get_positions()
    initial_axis_vec = direction
    initial_is_straight = is_molecule_straight(initial_positions, nh3_indices[0], nh3_indices[1], initial_axis_vec, mol_atoms)

    # If initial molecule is straight, keep it as candidate
    if initial_is_straight:
        best_distance_error = abs(current_distance - target_distance)
    else:
        # Initial molecule is not straight, mark it as invalid to force improvement
        best_distance_error = float('inf')

    # Adaptive step size: start large, get smaller near target
    current_step = max(1.0, min(step_size * 4, target_distance - current_distance))  # Start with large steps
    min_step = 0.05  # Minimum step size for fine tuning

    consecutive_failures = 0
    max_consecutive_failures = 5

    # Elongation loop: try to reach the target distance
    current_target_dist = current_distance
    iteration = 0

    while iteration < max_iterations and current_step >= min_step:
        # Adapt step size based on distance to target
        dist_to_target = abs(current_target_dist - target_distance)

        if dist_to_target < 2.0:  # Close to target, use small steps
            current_step = max(min_step, current_step * 0.5)
        elif consecutive_failures > 0:  # Having trouble, reduce step size
            current_step = max(min_step, current_step * 0.5)
        elif dist_to_target > 5.0:  # Far from target, can use larger steps
            current_step = min(2.0, current_step * 1.2)

        # Determine next target distance
        if current_target_dist < target_distance:
            next_target_dist = min(target_distance, current_target_dist + current_step)
        else:
            # We're past target, reduce towards target
            next_target_dist = max(target_distance, current_target_dist - current_step)

        # Skip if we're not making progress
        if abs(next_target_dist - current_target_dist) < min_step * 0.5:
            break

        current_target_dist = next_target_dist

        # Target position for second NH3+ along the axis
        target_pos = n1_pos + direction * current_target_dist

        # Use kinematic solver
        solver = KinematicChainSolver(mol_atoms.copy())
        solver_max_iter = 100 if current_step > 0.2 else 150  # More iterations for small steps

        elongated = solver.solve(
            anchor_idx=nh3_indices[0],
            mover_idx=nh3_indices[1],
            anchor_pos=n1_pos,
            target_pos=target_pos,
            tolerance=0.1,  # Tighter tolerance for target-directed elongation
            max_iter=solver_max_iter
        )

        # Check result
        final_positions = elongated.get_positions()
        final_distance = np.linalg.norm(final_positions[nh3_indices[1]] - final_positions[nh3_indices[0]])
        distance_error = abs(final_distance - current_target_dist)

        iteration += 1

        if distance_error < 0.3:  # Distance success (within tolerance)
            # Also check if molecule is reasonably straight
            axis_vec = (final_positions[nh3_indices[1]] - final_positions[nh3_indices[0]]) / final_distance
            if is_molecule_straight(final_positions, nh3_indices[0], nh3_indices[1], axis_vec, mol_atoms):
                # Update best result if this is closer to target
                target_error = abs(final_distance - target_distance)
                if target_error < best_distance_error:
                    best_atoms = elongated
                    best_distance = final_distance
                    best_distance_error = target_error

                consecutive_failures = 0
            else:
                # Molecule is folded, treat as failure to encourage longer distances
                consecutive_failures += 1
        else:
            consecutive_failures += 1

        # Stop if too many consecutive failures
        if consecutive_failures >= max_consecutive_failures:
            break

    # Final refinement: try to get even closer to target with smaller tolerance
    if best_distance_error > 0.1 and best_distance > current_distance + 0.1:
        final_target_pos = n1_pos + direction * (target_distance if target_distance > best_distance else best_distance)
        final_solver = KinematicChainSolver(best_atoms.copy())
        final_attempt = final_solver.solve(
            anchor_idx=nh3_indices[0],
            mover_idx=nh3_indices[1],
            anchor_pos=n1_pos,
            target_pos=final_target_pos,
            tolerance=0.05,  # Very tight tolerance for final refinement
            max_iter=200
        )

        final_positions = final_attempt.get_positions()
        final_distance = np.linalg.norm(final_positions[nh3_indices[1]] - final_positions[nh3_indices[0]])
        final_error = abs(final_distance - target_distance)

        # Check if final attempt is both close to target and straight
        axis_vec = (final_positions[nh3_indices[1]] - final_positions[nh3_indices[0]]) / final_distance
        if final_error < best_distance_error and is_molecule_straight(final_positions, nh3_indices[0], nh3_indices[1], axis_vec, mol_atoms):
            best_atoms = final_attempt
            best_distance = final_distance
            best_distance_error = final_error

    # Wrap atoms back into the unit cell before returning
    if hasattr(best_atoms, 'wrap'):
        best_atoms.wrap()

    return best_atoms


def elongate_molecule(molecule: str | Atoms | List[str | Atoms],
                     step_size: float = 0.5, max_iterations: int = 100,
                     target_distance: Optional[float] = None) -> Atoms | List[Atoms]:
    """Elongate N-N distance towards target distance using kinematic constraints."""
    # Handle batch processing
    if isinstance(molecule, list):
        return [_elongate_single_molecule(mol, step_size=step_size, max_iterations=max_iterations,
                                         target_distance=target_distance)
                for mol in molecule]

    # Handle single molecule
    return _elongate_single_molecule(molecule, step_size=step_size, max_iterations=max_iterations,
                                    target_distance=target_distance)


def find_optimal_spacer_vectors_global(starts: List[np.ndarray], targets: List[np.ndarray],
                                       cell: np.ndarray, spacer_radius: float = 2.0,
                                       obstacles: List[Tuple[np.ndarray, float]] = [],
                                       attachment_tolerance: float = 3.0) -> List[np.ndarray]:
    """
    Return deterministic PBC-aware vectors between starts and targets.
    
    Uses fractional coordinates to deterministically select the correct periodic cell
    for each target point. Parameters spacer_radius, obstacles, and attachment_tolerance
    are kept for API compatibility but are not used.
    """
    vectors = []
    for start, target in zip(starts, targets):
        # Use deterministic vector selection based on fractional coordinates
        vec = _find_directional_xy_pbc_vector(start, target, cell)
        vectors.append(vec)
    
    return vectors


def _get_uc_neighbor_offsets(cell: np.ndarray) -> np.ndarray:
    """Get 27 unit cell neighbor offsets for PBC calculations."""
    multipliers = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 1, 3)
    return np.array([np.matmul(cell.T, mult[0]) for mult in multipliers])


def detect_bonds_pbc(atoms: Atoms, cell: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """Detect bonds using covalent radii with PBC awareness."""
    elements = atoms.get_chemical_symbols()
    positions = atoms.get_positions()

    # Get covalent radii
    radii = [covalent_radii[ase.data.atomic_numbers.get(e, 6)] for e in elements]  # Default to C

    # Define bondable pairs for filtering (same as in KinematicChainSolver)
    BONDABLE_PAIRS = KinematicChainSolver.BONDABLE_PAIRS

    # PBC setup
    if cell is not None:
        uc_offsets = _get_uc_neighbor_offsets(cell)
    else:
        uc_offsets = np.array([[0., 0., 0.]])

    bonds = []
    for idx1, pos1 in enumerate(positions):
        pos1_images = pos1 + uc_offsets  # All periodic images

        for idx2 in range(idx1 + 1, len(positions)):
            pos2 = positions[idx2]

            # Early element filtering
            elem1, elem2 = elements[idx1], elements[idx2]
            if (elem1, elem2) not in BONDABLE_PAIRS and (elem2, elem1) not in BONDABLE_PAIRS:
                continue

            # Check distance to all images
            dists = np.linalg.norm(pos1_images - pos2, axis=1)
            min_dist = np.min(dists)

            # Bond if within covalent radii sum + tolerance
            max_bond_len = radii[idx1] + radii[idx2] + 0.45  # Same tolerance as mofun
            if min_dist < max_bond_len:
                bonds.append((idx1, idx2))

    return bonds