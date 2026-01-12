"""Optimizers for flexible spacer molecules."""

from __future__ import annotations

from typing import List, Set, Tuple, Optional, Dict
import networkx as nx
import numpy as np
from ase import Atoms
from ase.neighborlist import build_neighbor_list

from ..utils.properties.atomic_properties import get_covalent_radius

from .primitives import Vector3D, gram_schmidt, rodrigues_rotate



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
        symbols = atoms.get_chemical_symbols()
        radii = [get_covalent_radius(symbol, default=1.5) for symbol in symbols]

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
              target_pos: np.ndarray, tolerance: float = 0.1, max_iter: int = 100,
              folding_weight: float = 0.0, backbone_path: Optional[List[int]] = None) -> Atoms:
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
            
            # Add folding penalty if requested
            if folding_weight > 0 and backbone_path is not None:
                folding_score = _calculate_folding_score(
                    self.atoms.positions, backbone_path, anchor_idx, mover_idx
                )
                total_error = current_error + folding_weight * folding_score
            else:
                total_error = current_error
            
            if total_error < best_error:
                best_error = total_error
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

                # Geometry using clean abstractions
                pos = self.atoms.get_positions()
                pivot_loc = pos[pivot]
                mover_loc = pos[mover_idx]

                # Create axis vector using geometry module
                axis_vec = Vector3D(pos[child] - pivot_loc)
                if axis_vec.length() < 1e-3:
                    continue
                axis_vec = axis_vec.normalize()
                axis = axis_vec.coords

                # Current and target vectors relative to pivot
                r_cur = Vector3D(mover_loc - pivot_loc)
                r_tar = Vector3D(target_pos - pivot_loc)

                # Remove component parallel to axis (project perpendicular)
                r_cur_perp = r_cur.subtract(axis_vec.multiply(r_cur.dot(axis_vec)))
                r_tar_perp = r_tar.subtract(axis_vec.multiply(r_tar.dot(axis_vec)))

                if r_cur_perp.length() < 1e-2 or r_tar_perp.length() < 1e-2:
                    continue

                r_cur_perp = r_cur_perp.normalize()
                r_tar_perp = r_tar_perp.normalize()

                # Calculate rotation angle
                cross_vec = r_cur_perp.cross(r_tar_perp)
                dot_val = r_cur_perp.dot(r_tar_perp)
                angle = np.arctan2(cross_vec.dot(axis_vec), dot_val)

                if abs(angle) > 1e-4:
                    # Use half the angle for damping (CCD standard practice)
                    half_angle = angle * 0.5
                    
                    # Use vectorized Rodrigues rotation from geometry module
                    moving_positions = pos[moving_indices]
                    rotated = rodrigues_rotate(moving_positions, axis_vec, half_angle, center=pivot_loc)
                    self.atoms.positions[moving_indices] = rotated

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
    """Find the vector from p1 to p2 deterministically based on fractional coordinates."""
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
        # Use geometry module for clean vector operations
        current_vec_obj = Vector3D(current_vec).normalize()
        target_vec_obj = Vector3D(vec).normalize()
        
        # Calculate rotation axis and angle
        cross_vec = current_vec_obj.cross(target_vec_obj)
        dot = np.clip(current_vec_obj.dot(target_vec_obj), -1.0, 1.0)
        
        if cross_vec.length() > 1e-6:
            # Rotation needed
            axis = cross_vec.normalize().coords
            angle = np.arccos(dot)
            mol_copy.rotate(np.degrees(angle), v=axis, center=p1)
        elif dot < 0:
            # Vectors are opposite, rotate 180 degrees
            # Use Gram-Schmidt to get a robust perpendicular axis
            perp1, perp2 = gram_schmidt(current_vec_obj)
            mol_copy.rotate(180, v=perp1.coords, center=p1)
    
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
                                target_vector: Optional[np.ndarray] = None,
                                existing_structure: Optional[Atoms] = None,
                                collision_strategy: str = "off") -> Atoms:
    """Place spacer between p1 and p2. Options: "Off" (geometric only), "KS" (elongate + KS, default)."""
    optimizer = optimizer.upper()
    
    if optimizer == "OFF":
        placed = place_spacer_geometric(molecule, p1, p2, cell=cell, target_vector=target_vector)
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
        placed = place_spacer_geometric(molecule, p1, p2, cell=cell, target_vector=target_vector)

    # Check for and resolve collisions if existing structure provided
    if existing_structure is not None and collision_strategy != "off":
        from .collision import resolve_collisions
        placed, collision_resolved = resolve_collisions(
            placed, existing_structure, cell=cell, strategy=collision_strategy
        )
        if not collision_resolved and collision_strategy == "reject":
            # Return empty atoms to indicate rejection
            return Atoms()

    return placed


def _get_torsion_angle(positions: np.ndarray, a: int, b: int, c: int, d: int) -> float:
    """Calculate the torsion (dihedral) angle for atoms a-b-c-d."""
    # Get positions
    p_a = positions[a]
    p_b = positions[b]
    p_c = positions[c]
    p_d = positions[d]
    
    # Calculate vectors
    v1 = p_b - p_a
    v2 = p_c - p_b
    v3 = p_d - p_c
    
    # Calculate normal vectors to the planes
    n1 = np.cross(v1, v2)
    n2 = np.cross(v2, v3)
    
    # Normalize
    n1_norm = np.linalg.norm(n1)
    n2_norm = np.linalg.norm(n2)
    
    if n1_norm < 1e-6 or n2_norm < 1e-6:
        return 0.0  # Degenerate case
    
    n1 = n1 / n1_norm
    n2 = n2 / n2_norm
    
    # Calculate angle
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    angle = np.arccos(cos_angle)
    
    # Determine sign using the cross product
    cross = np.cross(n1, n2)
    v2_norm = np.linalg.norm(v2)
    if v2_norm > 1e-6:
        sign = np.sign(np.dot(cross, v2 / v2_norm))
        angle = sign * angle
    
    return np.degrees(angle)


def _set_torsion_angle(atoms: Atoms, a: int, b: int, c: int, d: int, 
                       target_angle: float, moving_indices: List[int]) -> Atoms:
    """
    Set the torsion angle for atoms a-b-c-d by rotating around the b-c bond.
    
    Uses Gram-Schmidt orthogonalization for robust perpendicular axis generation,
    ensuring reliable torsion angle calculations even in near-degenerate cases.
    
    Parameters
    ----------
    atoms : Atoms
        The molecule
    a, b, c, d : int
        Atom indices defining the torsion
    target_angle : float
        Target torsion angle in degrees
    moving_indices : List[int]
        Indices of atoms that should move (typically the 'd' side of the bond)
    
    Returns
    -------
    Atoms
        Modified molecule with new torsion angle
    """
    result = atoms.copy()
    positions = result.get_positions()
    
    # Get current angle
    current_angle = _get_torsion_angle(positions, a, b, c, d)
    
    # Calculate rotation needed
    rotation_angle = target_angle - current_angle
    
    if abs(rotation_angle) < 0.1:
        return result  # Already at target
    
    # Rotation axis is the b-c bond
    p_b = positions[b]
    p_c = positions[c]
    axis_vec = Vector3D(p_c - p_b)
    
    if axis_vec.length() < 1e-6:
        return result  # Degenerate bond
    
    # Generate orthonormal basis using Gram-Schmidt
    # This ensures robust perpendicular vectors even for axis-aligned bonds
    perp1, perp2 = gram_schmidt(axis_vec)
    
    # Use vectorized Rodrigues rotation from geometry module
    angle_rad = np.radians(rotation_angle)
    moving_positions = positions[moving_indices]
    
    rotated = rodrigues_rotate(moving_positions, axis_vec, angle_rad, center=p_b)
    
    result.positions[moving_indices] = rotated
    
    return result


def _sequential_unfold(atoms: Atoms, torsion_info: Dict, n1_idx: int, n2_idx: int,
                      target_distance: float) -> Atoms:
    """
    Sequentially unfold the molecule bond-by-bond from anchor toward target.
    
    This refines the conformation after CCD by ensuring each segment of the
    backbone projects "outward" in the direction from N1 to N2.
    
    Parameters
    ----------
    atoms : Atoms
        The molecule after CCD optimization
    torsion_info : dict
        Backbone and torsion information
    n1_idx : int
        Anchor nitrogen index
    n2_idx : int
        Target nitrogen index
    target_distance : float
        Desired N-N distance
    
    Returns
    -------
    Atoms
        Refined molecule with sequential unfolding applied
    """
    result = atoms.copy()
    positions = result.get_positions()
    
    backbone_path = torsion_info['backbone_path']
    torsion_quads = torsion_info['torsion_quads']
    
    if len(backbone_path) < 3 or not torsion_quads:
        return result  # Nothing to unfold
    
    # Build connectivity graph
    from ase.neighborlist import build_neighbor_list
    
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    
    symbols = atoms.get_chemical_symbols()
    radii = [get_covalent_radius(symbol, default=1.5) for symbol in symbols]
    nl = build_neighbor_list(atoms, cutoffs=[r * 1.2 for r in radii], self_interaction=False)
    
    for i in range(len(atoms)):
        neighbors, _ = nl.get_neighbors(i)
        for j in neighbors:
            j = int(j)
            if i < j:
                G.add_edge(i, j)
    
    # Ideal direction: from N1 toward N2
    n1_pos = positions[n1_idx]
    n2_pos = positions[n2_idx]
    ideal_direction = n2_pos - n1_pos
    ideal_len = np.linalg.norm(ideal_direction)
    
    if ideal_len < 1e-6:
        return result
    
    ideal_direction = ideal_direction / ideal_len
    
    # Process torsions sequentially along the backbone
    for torsion_quad in torsion_quads:
        a, b, c, d = torsion_quad
        
        # Determine which side to rotate (away from anchor)
        if G.has_edge(b, c):
            G.remove_edge(b, c)
            
            component_b = set(nx.node_connected_component(G, b))
            component_c = set(nx.node_connected_component(G, c))
            
            # Rotate the component that doesn't contain the anchor
            if n1_idx in component_b:
                moving_indices = list(component_c)
                pivot_pos = positions[b]
            else:
                moving_indices = list(component_b)
                pivot_pos = positions[c]
            
            G.add_edge(b, c)
            
            # Calculate center of mass of moving segment
            moving_positions = positions[moving_indices]
            moving_com = moving_positions.mean(axis=0)
            
            # Vector from pivot to moving COM
            pivot_to_com = moving_com - pivot_pos
            pivot_to_com_len = np.linalg.norm(pivot_to_com)
            
            if pivot_to_com_len < 1e-6:
                continue
            
            pivot_to_com_unit = pivot_to_com / pivot_to_com_len
            
            # Check if moving segment is projecting in the ideal direction
            projection = np.dot(pivot_to_com_unit, ideal_direction)
            
            # Try to improve projection if not optimal
            if projection < 0.8:  # More aggressive threshold
                # Try different torsion angles to maximize outward projection
                current_angle = _get_torsion_angle(positions, a, b, c, d)
                best_angle = current_angle
                best_projection = projection
                
                # Try angles around trans (180) and gauche (±60, ±120)
                # Also try intermediate angles for fine-tuning
                test_angles = [180.0, 60.0, -60.0, 120.0, -120.0, 90.0, -90.0, 150.0, -150.0]
                
                for test_angle in test_angles:
                    test_atoms = _set_torsion_angle(result.copy(), a, b, c, d, 
                                                    test_angle, moving_indices)
                    test_positions = test_atoms.get_positions()
                    test_moving_com = test_positions[moving_indices].mean(axis=0)
                    test_pivot_to_com = test_moving_com - pivot_pos
                    test_len = np.linalg.norm(test_pivot_to_com)
                    
                    if test_len > 1e-6:
                        test_projection = np.dot(test_pivot_to_com / test_len, ideal_direction)
                        
                        # Prefer projections that extend the molecule
                        if test_projection > best_projection:
                            best_projection = test_projection
                            best_angle = test_angle
                
                # Apply best angle if it's a significant improvement
                if best_projection > projection + 0.1 or abs(best_angle - current_angle) > 10.0:
                    result = _set_torsion_angle(result, a, b, c, d, best_angle, moving_indices)
                    positions = result.get_positions()
    
    return result


def _calculate_folding_score(positions: np.ndarray, backbone_path: List[int], 
                             n1_idx: int, n2_idx: int) -> float:
    """
    Calculate how "folded" the molecule is.
    
    Measures:
    1. Deviation of backbone atoms from the N-N axis
    2. Atoms projecting inward toward the center (negative progress along axis)
    
    Lower score = more extended, higher score = more folded
    
    Parameters
    ----------
    positions : np.ndarray
        Atomic positions
    backbone_path : List[int]
        Ordered list of backbone atom indices
    n1_idx : int
        First nitrogen index
    n2_idx : int
        Second nitrogen index
    
    Returns
    -------
    float
        Folding score (0 = perfectly extended, higher = more folded)
    """
    if len(backbone_path) < 3:
        return 0.0  # Too short to fold
    
    n1_pos = positions[n1_idx]
    n2_pos = positions[n2_idx]
    
    # N-N axis
    axis = n2_pos - n1_pos
    axis_len = np.linalg.norm(axis)
    
    if axis_len < 1e-6:
        return 0.0  # Degenerate case
    
    axis_unit = axis / axis_len
    
    # Calculate perpendicular distances and progress along axis for backbone atoms
    folding_score = 0.0
    
    for idx in backbone_path:
        if idx == n1_idx or idx == n2_idx:
            continue  # Skip endpoints
        
        pos = positions[idx]
        rel_pos = pos - n1_pos
        
        # Progress along axis (should be between 0 and axis_len for extended molecule)
        progress = np.dot(rel_pos, axis_unit)
        
        # Perpendicular distance from axis
        proj_on_axis = progress * axis_unit
        perp_vec = rel_pos - proj_on_axis
        perp_dist = np.linalg.norm(perp_vec)
        
        # Penalty for being far from axis (folding sideways)
        folding_score += perp_dist ** 2
        
        # Penalty for negative progress (folding backwards)
        if progress < 0:
            folding_score += abs(progress) * 10.0
        
        # Penalty for going past the endpoint (folding forward)
        if progress > axis_len:
            folding_score += (progress - axis_len) * 10.0
    
    # Normalize by number of backbone atoms
    if len(backbone_path) > 2:
        folding_score /= (len(backbone_path) - 2)
    
    return folding_score


def _set_extended_conformation(atoms: Atoms, torsion_info: Dict, 
                               n1_idx: int, n2_idx: int) -> Atoms:
    """
    Set all rotatable backbone torsions to trans (180 degrees) conformation.
    
    This creates an initial extended conformation before CCD refinement,
    helping to avoid folded local minima.
    
    Parameters
    ----------
    atoms : Atoms
        The molecule
    torsion_info : dict
        Output from _identify_backbone_and_torsions
    n1_idx : int
        Anchor nitrogen index
    n2_idx : int
        Target nitrogen index
    
    Returns
    -------
    Atoms
        Molecule with extended backbone conformation
    """
    result = atoms.copy()
    
    if not torsion_info['torsion_quads']:
        return result  # No rotatable torsions
    
    # Build connectivity graph to determine which atoms move with each rotation
    from ase.neighborlist import build_neighbor_list
    
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    
    symbols = atoms.get_chemical_symbols()
    radii = [get_covalent_radius(symbol, default=1.5) for symbol in symbols]
    nl = build_neighbor_list(atoms, cutoffs=[r * 1.2 for r in radii], self_interaction=False)
    
    for i in range(len(atoms)):
        neighbors, _ = nl.get_neighbors(i)
        for j in neighbors:
            j = int(j)
            if i < j:
                G.add_edge(i, j)
    
    # Process each torsion from anchor (n1) toward target (n2)
    backbone_path = torsion_info['backbone_path']
    
    for torsion_quad in torsion_info['torsion_quads']:
        a, b, c, d = torsion_quad
        
        # Determine which side of the bond to rotate
        # We want to rotate the side away from the anchor
        if G.has_edge(b, c):
            G.remove_edge(b, c)
            
            # Find which component contains the anchor
            component_b = set(nx.node_connected_component(G, b))
            component_c = set(nx.node_connected_component(G, c))
            
            # Rotate the component that doesn't contain the anchor
            if n1_idx in component_b:
                moving_indices = list(component_c)
            else:
                moving_indices = list(component_b)
            
            G.add_edge(b, c)
            
            # Set torsion to 180 degrees (trans)
            result = _set_torsion_angle(result, a, b, c, d, 180.0, moving_indices)
    
    return result


def _identify_backbone_and_torsions(atoms: Atoms, n1_idx: int, n2_idx: int, 
                                    rigid_bonds: Set[Tuple[int, int]]) -> Dict:
    """
    Identify the backbone path between two NH3+ groups and classify bonds.
    
    Parameters
    ----------
    atoms : Atoms
        The molecule
    n1_idx : int
        Index of first NH3+ nitrogen
    n2_idx : int
        Index of second NH3+ nitrogen
    rigid_bonds : Set[Tuple[int, int]]
        Set of rigid bonds (rings, double bonds)
    
    Returns
    -------
    dict
        - backbone_path: ordered list of atom indices from N1 to N2
        - backbone_bonds: list of (i, j) tuples along the backbone
        - rotatable_bonds: backbone bonds that can rotate (not in rings, not double bonds)
        - torsion_quads: list of (a, b, c, d) for each rotatable bond defining torsion angle
    """
    # Build connectivity graph (should already exist in KinematicChainSolver, but rebuild here)
    from ase.neighborlist import build_neighbor_list
    
    G = nx.Graph()
    G.add_nodes_from(range(len(atoms)))
    
    symbols = atoms.get_chemical_symbols()
    radii = [get_covalent_radius(symbol, default=1.5) for symbol in symbols]
    nl = build_neighbor_list(atoms, cutoffs=[r * 1.2 for r in radii], self_interaction=False)
    
    for i in range(len(atoms)):
        neighbors, _ = nl.get_neighbors(i)
        for j in neighbors:
            j = int(j)
            if i < j:
                G.add_edge(i, j)
    
    # Find shortest path between N1 and N2
    try:
        backbone_path = nx.shortest_path(G, source=n1_idx, target=n2_idx)
    except nx.NetworkXNoPath:
        return {
            'backbone_path': [],
            'backbone_bonds': [],
            'rotatable_bonds': [],
            'torsion_quads': []
        }
    
    # Extract backbone bonds
    backbone_bonds = []
    for i in range(len(backbone_path) - 1):
        bond = tuple(sorted((backbone_path[i], backbone_path[i+1])))
        backbone_bonds.append(bond)
    
    # Identify rotatable bonds (not rigid, not terminal)
    rotatable_bonds = []
    for bond in backbone_bonds:
        if bond not in rigid_bonds and (bond[1], bond[0]) not in rigid_bonds:
            rotatable_bonds.append(bond)
    
    # Build torsion quads for each rotatable bond
    # A torsion is defined by 4 atoms: a-b-c-d where b-c is the rotatable bond
    torsion_quads = []
    symbols = atoms.get_chemical_symbols()
    
    for bond in rotatable_bonds:
        b, c = bond  # Central bond atoms
        
        # Find atom 'a' connected to 'b' (prefer backbone, avoid 'c')
        neighbors_b = list(G.neighbors(b))
        a = None
        for neighbor in neighbors_b:
            if neighbor != c:
                # Prefer backbone atoms
                if neighbor in backbone_path:
                    a = neighbor
                    break
        if a is None:
            # No backbone neighbor, take any non-c neighbor
            for neighbor in neighbors_b:
                if neighbor != c:
                    a = neighbor
                    break
        
        # Find atom 'd' connected to 'c' (prefer backbone, avoid 'b')
        neighbors_c = list(G.neighbors(c))
        d = None
        for neighbor in neighbors_c:
            if neighbor != b:
                # Prefer backbone atoms
                if neighbor in backbone_path:
                    d = neighbor
                    break
        if d is None:
            # No backbone neighbor, take any non-b neighbor
            for neighbor in neighbors_c:
                if neighbor != b:
                    d = neighbor
                    break
        
        # Only add if we found all 4 atoms
        if a is not None and d is not None:
            torsion_quads.append((a, b, c, d))
    
    return {
        'backbone_path': backbone_path,
        'backbone_bonds': backbone_bonds,
        'rotatable_bonds': rotatable_bonds,
        'torsion_quads': torsion_quads
    }


def _elongate_single_molecule(molecule: str | Atoms, step_size: float = 0.5,
                              max_iterations: int = 100, target_distance: Optional[float] = None) -> Atoms:
    """
    Elongate molecule towards target N-N distance using multi-phase unfolding.
    
    Phases:
    1. Identify backbone and rotatable torsions
    2. Set extended (trans) conformation
    3. Use outward-biased CCD to reach target distance
    4. Apply sequential refinement to ensure straightness
    
    Parameters
    ----------
    molecule : str | Atoms
        SMILES string or Atoms object to elongate
    step_size : float
        Step size for iterative elongation (deprecated, kept for compatibility)
    max_iterations : int
        Maximum iterations for CCD solver
    target_distance : float, optional
        Target N-N distance. If None, maximizes elongation.
    
    Returns
    -------
    Atoms
        Elongated molecule with extended conformation
    """
    # Convert SMILES to Atoms if needed
    if isinstance(molecule, str):
        from .molecule_builder import smiles_to_ase_atoms
        mol_atoms = smiles_to_ase_atoms(molecule)
    else:
        mol_atoms = molecule.copy()

    # Find terminal NH3+ groups
    from .spacer import _find_terminal_nitrogens
    _, nh3_indices = _find_terminal_nitrogens(mol_atoms)

    if len(nh3_indices) < 2:
        return mol_atoms  # Not enough NH3+ groups to elongate

    # Get initial positions and current distance
    positions = mol_atoms.get_positions()
    n1_idx = nh3_indices[0]
    n2_idx = nh3_indices[1]
    n1_pos = positions[n1_idx]
    n2_pos = positions[n2_idx]
    current_distance = np.linalg.norm(n2_pos - n1_pos)

    # Determine target distance
    if target_distance is None:
        target_distance = current_distance * 2.5  # Maximize elongation
    else:
        target_distance = max(target_distance, current_distance)

    # PHASE 1: Identify backbone and torsions
    solver_init = KinematicChainSolver(mol_atoms)
    torsion_info = _identify_backbone_and_torsions(
        mol_atoms, n1_idx, n2_idx, solver_init.rigid_bonds
    )
    
    if not torsion_info['backbone_path']:
        return mol_atoms  # No path found
    
    # PHASE 2: Set extended conformation (all torsions to trans)
    extended_mol = _set_extended_conformation(mol_atoms, torsion_info, n1_idx, n2_idx)
    
    # Update positions after extension
    positions = extended_mol.get_positions()
    n1_pos = positions[n1_idx]
    n2_pos = positions[n2_idx]
    extended_distance = np.linalg.norm(n2_pos - n1_pos)
    
    # If extended conformation is already better than original, use it as starting point
    if extended_distance > current_distance * 1.05:
        current_distance = extended_distance
    else:
        # Extended conformation didn't help much, keep original
        extended_mol = mol_atoms.copy()
        positions = extended_mol.get_positions()
        n1_pos = positions[n1_idx]
        n2_pos = positions[n2_idx]
    
    direction = (n2_pos - n1_pos) / max(current_distance, 1e-6)
    
    # PHASE 3: Outward-biased CCD elongation
    best_atoms = extended_mol.copy()
    best_distance = current_distance
    best_folding_score = _calculate_folding_score(
        extended_mol.get_positions(), torsion_info['backbone_path'], n1_idx, n2_idx
    )
    
    # Iterative elongation with folding penalty
    num_attempts = min(max_iterations // 20, 10)  # Multiple attempts with different targets
    
    for attempt in range(num_attempts):
        # Gradually increase target distance
        intermediate_target = current_distance + (target_distance - current_distance) * (attempt + 1) / num_attempts
        target_pos = n1_pos + direction * intermediate_target
        
        # Use CCD with folding penalty
        solver = KinematicChainSolver(extended_mol.copy())
        elongated = solver.solve(
            anchor_idx=n1_idx,
            mover_idx=n2_idx,
            anchor_pos=n1_pos,
            target_pos=target_pos,
            tolerance=0.1,
            max_iter=max_iterations // num_attempts,
            folding_weight=0.5,  # Moderate anti-folding bias
            backbone_path=torsion_info['backbone_path']
        )
        
        # Evaluate result
        final_positions = elongated.get_positions()
        final_distance = np.linalg.norm(final_positions[n2_idx] - final_positions[n1_idx])
        final_folding_score = _calculate_folding_score(
            final_positions, torsion_info['backbone_path'], n1_idx, n2_idx
        )
        
        # Keep if better (longer and less folded)
        if final_distance > best_distance and final_folding_score <= best_folding_score * 1.2:
            best_atoms = elongated
            best_distance = final_distance
            best_folding_score = final_folding_score
            extended_mol = elongated  # Use as starting point for next iteration
            
            # Update direction
            positions = extended_mol.get_positions()
            n1_pos = positions[n1_idx]
            n2_pos = positions[n2_idx]
            direction = (n2_pos - n1_pos) / max(np.linalg.norm(n2_pos - n1_pos), 1e-6)
    
    # PHASE 4: Sequential refinement to unfold any remaining kinks
    refined_mol = _sequential_unfold(best_atoms, torsion_info, n1_idx, n2_idx, target_distance)
    
    # Check if refinement improved the result
    refined_positions = refined_mol.get_positions()
    refined_distance = np.linalg.norm(refined_positions[n2_idx] - refined_positions[n1_idx])
    refined_folding_score = _calculate_folding_score(
        refined_positions, torsion_info['backbone_path'], n1_idx, n2_idx
    )
    
    # Use refined version if it's better or comparable
    if refined_folding_score < best_folding_score * 0.9 or refined_distance > best_distance * 1.05:
        best_atoms = refined_mol
    
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
    radii = [get_covalent_radius(e, default=1.5) for e in elements]

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