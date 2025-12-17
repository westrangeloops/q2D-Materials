"""
Optimizers for flexible spacer molecules.

This module provides different optimization strategies for aligning spacer molecules
between two anchor points (P1, P2):
- Off: Pure geometric placement (rigid translation/rotation)
- KS: Kinematic Chain Solver (CCD-based inverse kinematics)
- UFF: Universal Force Field optimization with pinned anchors
"""

from __future__ import annotations

from typing import List, Set, Tuple, Optional, Union
import warnings
import itertools
from scipy.spatial.distance import cdist
from scipy.spatial import cKDTree
import networkx as nx
import numpy as np
from ase import Atoms
from ase.data import covalent_radii
from ase.neighborlist import build_neighbor_list
from ase.io import write
from io import StringIO

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from rdkit.Geometry import Point3D
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


class KinematicChainSolver:
    """
    Inverse Kinematics solver for molecules that respects chemical rigidity.

    Uses Cyclic Coordinate Descent (CCD) to align a molecule between two points.

    Locks:
    1. Cycles (Rings) - detected via NetworkX
    2. Double/Triple Bonds - inferred from bond length vs covalent radii

    Only allows rotation around 'true' single bonds (rotors).
    """

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
        """
        Parameters
        ----------
        atoms : Atoms
            The molecule to solve.
        cutoff_buffer : float
            Multiplier for covalent radii to determine connectivity.
        double_bond_threshold : float
            Factor to determine if a bond is double/rigid.
            If dist < (r1 + r2) * threshold, it is treated as rigid.
        """
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

    def solve(
        self,
        anchor_idx: int,
        mover_idx: int,
        anchor_pos: np.ndarray,
        target_pos: np.ndarray,
        tolerance: float = 0.1,
        max_iter: int = 100,
    ) -> Atoms:
        """
        Align the molecule so that:
        - atom[anchor_idx] is at anchor_pos (P1)
        - atom[mover_idx] is as close as possible to target_pos (P2)
        """
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


def _find_shortest_pbc_vector_general(
    p1: np.ndarray,
    p2: np.ndarray,
    cell: np.ndarray,
    dimensions: Tuple[int, ...] = (0, 1, 2)
) -> np.ndarray:
    """
    General helper function for finding shortest PBC vectors in specified dimensions.

    Parameters
    ----------
    p1 : np.ndarray
        First point (Cartesian coordinates)
    p2 : np.ndarray
        Second point (Cartesian coordinates)
    cell : np.ndarray
        Unit cell matrix (3x3)
    dimensions : tuple of int
        Which dimensions to consider for PBC (e.g., (0, 1) for XY only, (0, 1, 2) for 3D)

    Returns
    -------
    np.ndarray
        Shortest vector from p1 to p2 considering PBC in specified dimensions
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

    # Convert to fractional coordinates
    diff_frac = raw_diff @ inv_cell.T  # Shape: (3,)

    # Split fractional coordinates into PBC and non-PBC dimensions
    pbc_dims = list(dimensions)
    non_pbc_dims = [i for i in range(3) if i not in pbc_dims]

    diff_frac_pbc = diff_frac[pbc_dims]  # Fractional coords for PBC dimensions
    diff_frac_fixed = diff_frac[non_pbc_dims]  # Fractional coords for fixed dimensions

    # Generate all combinations of shifts for PBC dimensions
    shift_range = [-1, 0, 1]
    shifts = np.array(np.meshgrid(*[shift_range for _ in pbc_dims])).T.reshape(-1, len(pbc_dims))

    # Find the shortest vector among all shifted images
    best_vector = None
    best_distance = np.inf

    for shift in shifts:
        # Apply shift to PBC fractional coordinates
        diff_frac_pbc_shifted = diff_frac_pbc - shift

        # Create full fractional coords with shifted PBC and original fixed dimensions
        diff_frac_shifted = np.zeros(3)
        diff_frac_shifted[pbc_dims] = diff_frac_pbc_shifted
        diff_frac_shifted[non_pbc_dims] = diff_frac_fixed

        # Convert back to Cartesian
        diff_shifted = diff_frac_shifted @ cell

        # Calculate distance
        distance = np.linalg.norm(diff_shifted)

        if distance < best_distance:
            best_distance = distance
            best_vector = diff_shifted

    return best_vector if best_vector is not None else raw_diff


def find_shortest_pbc_vector(p1: np.ndarray, p2: np.ndarray, cell: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Find the shortest vector from p1 to p2 considering periodic boundary conditions.

    Correctly handles non-rectangular (skewed) cells by checking all 27
    nearest periodic images instead of relying on fractional rounding.

    Parameters
    ----------
    p1 : np.ndarray
        First point (Cartesian coordinates)
    p2 : np.ndarray
        Second point (Cartesian coordinates)
    cell : np.ndarray, optional
        Unit cell matrix (3x3). If None, returns p2 - p1 (no PBC)

    Returns
    -------
    np.ndarray
        Shortest vector from p1 to p2 considering PBC
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    raw_diff = p2 - p1

    if cell is None:
        return raw_diff

    cell = np.array(cell)
    if cell.shape != (3, 3) or np.allclose(cell, 0):
        return raw_diff

    # Use the general helper for 3D PBC
    return _find_shortest_pbc_vector_general(p1, p2, cell, dimensions=(0, 1, 2))


def relax_spacer_with_uff(
    molecule: Atoms, 
    p1: np.ndarray, 
    p2: np.ndarray, 
    stiffness: float = 1000.0, 
    cell: Optional[np.ndarray] = None,
    target_vector: Optional[np.ndarray] = None
) -> Atoms:
    """
    Fits a molecule between p1 and p2 using UFF with Linear Scaling and PBC.
    
    Parameters
    ----------
    molecule : Atoms
        Input molecule
    p1, p2 : np.ndarray
        Start and end points
    stiffness : float
        UFF force constant (unused in current implementation but kept for API compat)
    cell : np.ndarray
        Unit cell
    target_vector : np.ndarray, optional
        Explicit vector to use instead of calculating shortest PBC vector.
    """
    if not HAS_RDKIT:
        raise ImportError("RDKit is required for UFF optimization.")

    from .spacer import _find_terminal_nitrogens
    try:
        from rdkit.Chem import rdDetermineBonds
    except ImportError:
        warnings.warn("RDKit version too old: missing rdDetermineBonds.")
        return place_spacer_geometric(molecule, p1, p2, cell=cell, target_vector=target_vector)

    # --- 1. Identify Anchors ---
    _, nh3_indices = _find_terminal_nitrogens(molecule)
    if len(nh3_indices) < 2:
        return molecule.copy()
    
    idx_a = int(nh3_indices[0])
    idx_b = int(nh3_indices[1])

    # --- 2. Rigid Alignment (PBC aware) ---
    # Pass target_vector explicitly to geometric placer
    molecule_aligned = place_spacer_geometric(molecule, p1, p2, cell=cell, target_vector=target_vector)
    
    # --- 3. ASE -> RDKit (Topology Calculation) ---
    xyz_io = StringIO()
    write(xyz_io, molecule_aligned, format='xyz')
    xyz_io.seek(0)
    rd_mol = Chem.MolFromXYZBlock(xyz_io.read())
    
    if rd_mol is None: 
        return molecule_aligned

    try:
        rdDetermineBonds.DetermineConnectivity(rd_mol)
        rdDetermineBonds.DetermineBondOrders(rd_mol)
        Chem.rdmolops.SanitizeMol(rd_mol)
    except Exception:
        return molecule_aligned

    # --- 4. Linear Scaling ---
    positions = molecule_aligned.get_positions()
    n1_pos = positions[idx_a]
    n2_pos = positions[idx_b]
    
    current_vec = n2_pos - n1_pos
    
    # Use passed vector if available, else calculate
    vec = target_vector if target_vector is not None else find_shortest_pbc_vector(p1, p2, cell)
    
    current_dist = np.linalg.norm(current_vec)
    target_dist = np.linalg.norm(vec)
    
    conf = rd_mol.GetConformer()
    
    if current_dist > 0.1:
        # Calculate stretch factor
        scale = target_dist / current_dist
        axis = vec / target_dist
        
        # Apply scaling to ALL atoms in the RDKit conformer
        for i in range(len(molecule)):
            pos = positions[i]
            rel_pos = pos - n1_pos
            
            # Project onto the N1->N2 axis
            proj = np.dot(rel_pos, axis)
            # Vector rejection (perpendicular component)
            perp = rel_pos - (proj * axis)
            
            # Scale ONLY the distance along the axis, keep width the same
            new_rel_pos = (proj * scale * axis) + perp
            new_pos_coord = n1_pos + new_rel_pos
            
            conf.SetAtomPosition(i, Point3D(float(new_pos_coord[0]), float(new_pos_coord[1]), float(new_pos_coord[2])))

    # --- 5. Pin & Relax ---
    # Pin N1 to p1, and N2 to the "virtual" p2 (p1 + pbc_vector)
    ff = AllChem.UFFGetMoleculeForceField(rd_mol)
    if ff:
        virtual_p2 = p1 + vec
        conf.SetAtomPosition(idx_a, Point3D(float(p1[0]), float(p1[1]), float(p1[2])))
        conf.SetAtomPosition(idx_b, Point3D(float(virtual_p2[0]), float(virtual_p2[1]), float(virtual_p2[2])))
        
        ff.AddFixedPoint(idx_a)
        ff.AddFixedPoint(idx_b)
        try:
            ff.Minimize(maxIts=500)
        except:
            pass

    # --- 6. Export back to ASE ---
    new_pos_array = molecule_aligned.get_positions().copy()
    for i in range(len(molecule)):
        pt = conf.GetAtomPosition(i)
        new_pos_array[i] = [pt.x, pt.y, pt.z]
        
    molecule_optimized = molecule_aligned.copy()
    molecule_optimized.set_positions(new_pos_array)
    
    return molecule_optimized


def place_spacer_geometric(
    molecule: Atoms, 
    p1: np.ndarray, 
    p2: np.ndarray, 
    cell: Optional[np.ndarray] = None,
    target_vector: Optional[np.ndarray] = None
) -> Atoms:
    """
    Pure geometric placement: rigid translation/rotation to align terminal NH3+ groups.
    
    Parameters
    ----------
    molecule : Atoms
        The molecule to place
    p1 : np.ndarray
        Target position for first terminal NH3+ nitrogen
    p2 : np.ndarray
        Target position for second terminal NH3+ nitrogen
    cell : np.ndarray, optional
        Unit cell matrix (3x3) for PBC-aware shortest vector calculation
    target_vector : np.ndarray, optional
        Explicit vector from p1 to p2 (p2_image - p1). If provided, this overrides
        the internal shortest vector calculation.
        
    Returns
    -------
    Atoms
        Molecule aligned geometrically
    """
    from .spacer import _find_terminal_nitrogens
    
    mol_copy = molecule.copy()
    _, nh3_indices = _find_terminal_nitrogens(mol_copy)
    positions = mol_copy.get_positions()
    
    # Determine the vector to use (Explicit or Calculated)
    vec = target_vector if target_vector is not None else find_shortest_pbc_vector(p1, p2, cell)
    
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
    
    return mol_copy


def place_spacer_with_optimizer(
    molecule: Atoms,
    p1: np.ndarray,
    p2: np.ndarray,
    optimizer: str = "KS",
    cell: Optional[np.ndarray] = None,
    target_vector: Optional[np.ndarray] = None
) -> Atoms:
    """
    Place a spacer molecule between two points using the specified optimizer.
    
    Parameters
    ----------
    molecule : Atoms
        The molecule to place
    p1 : np.ndarray
        Target position for first terminal NH3+ nitrogen
    p2 : np.ndarray
        Target position for second terminal NH3+ nitrogen
    optimizer : str, default "KS"
        Optimizer to use: "Off", "KS", or "UFF"
    cell : np.ndarray, optional
        Unit cell matrix (3x3) for PBC-aware shortest vector calculation
    target_vector : np.ndarray, optional
        Explicit vector from p1 to p2. If provided, overrides internal shortest path calculation.
        
    Returns
    -------
    Atoms
        Optimized molecule aligned between p1 and p2
    """
    optimizer = optimizer.upper()
    
    if optimizer == "OFF":
        return place_spacer_geometric(molecule, p1, p2, cell=cell, target_vector=target_vector)
        
    elif optimizer == "KS":
        solver = KinematicChainSolver(molecule)
        from .spacer import _find_terminal_nitrogens
        _, nh3_indices = _find_terminal_nitrogens(molecule)
        
        if len(nh3_indices) < 2:
            return place_spacer_geometric(molecule, p1, p2, cell=cell, target_vector=target_vector)
        
        # Pick the two furthest NH3 groups if > 2
        positions = molecule.get_positions()
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
        
        # Use passed vector if available, else calculate
        vec = target_vector if target_vector is not None else find_shortest_pbc_vector(p1, p2, cell)
        
        aligned_mol = solver.solve(
            anchor_idx=idx1,
            mover_idx=idx2,
            anchor_pos=p1,
            target_pos=p1 + vec, # Use the explicit vector for the target position
            tolerance=0.1,
            max_iter=150
        )
        return aligned_mol
        
    elif optimizer == "UFF":
        return relax_spacer_with_uff(molecule, p1, p2, cell=cell, target_vector=target_vector)
        
    else:
        warnings.warn(f"Unknown optimizer '{optimizer}', falling back to 'Off'")
        return place_spacer_geometric(molecule, p1, p2, cell=cell, target_vector=target_vector)


def _elongate_single_molecule(molecule: str | Atoms, step_size: float = 0.5, max_iterations: int = 100) -> Atoms:
    """
    Internal function to elongate a single molecule to maximum possible distance.

    Enhanced version that aggressively finds the most elongate conformation possible.
    
    Parameters
    ----------
    molecule : str | Atoms
        Input molecule as SMILES string or ASE Atoms object
    step_size : float, default 0.5
        Distance increment (Å) for each elongation attempt
    max_iterations : int, default 100
        Maximum number of elongation steps to attempt (increased for more aggressive elongation)
        
    Returns
    -------
    Atoms
        Elongated molecule at maximum achievable N-N distance
    """
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

    # Start elongation from current distance
    target_distance = current_distance
    best_atoms = mol_atoms.copy()
    best_distance = current_distance

    # Estimate maximum possible elongation (rough heuristic: 2x current distance is usually max)
    max_reasonable_distance = current_distance * 2.5
    max_target_distance = current_distance + (max_iterations * step_size)
    max_target_distance = min(max_target_distance, max_reasonable_distance)

    consecutive_failures = 0
    max_consecutive_failures = 5  # Allow some tolerance for solver instability
    no_progress_count = 0
    max_no_progress = 10  # Stop if no improvement for 10 iterations

    # More aggressive elongation: try larger steps when possible
    for iteration in range(max_iterations):
        # Early exit if we've exceeded reasonable maximum
        if target_distance >= max_target_distance:
            break
        target_distance += step_size

        # Target position for second NH3+ along the axis
        target_pos = n1_pos + direction * target_distance

        # Use kinematic solver with optimized settings for maximum elongation
        solver = KinematicChainSolver(mol_atoms.copy())

        # Use fewer iterations for speed - increase only if needed
        solver_max_iter = 100 if step_size > 0.1 else 150  # Fewer iterations for larger steps
        
        elongated = solver.solve(
            anchor_idx=nh3_indices[0],  # Fix first NH3+ nitrogen
            mover_idx=nh3_indices[1],   # Move second NH3+ nitrogen
            anchor_pos=n1_pos,          # Keep first NH3+ at original position
            target_pos=target_pos,      # Move second NH3+ to target
            tolerance=0.2,              # Looser tolerance for max elongation
            max_iter=solver_max_iter    # Adaptive iterations based on step size
        )

        # Check if solver succeeded (mover is close to target)
        final_positions = elongated.get_positions()
        final_distance = np.linalg.norm(final_positions[nh3_indices[1]] - final_positions[nh3_indices[0]])
        distance_error = abs(final_distance - target_distance)

        # More lenient success criteria for maximum elongation
        if distance_error < 0.5:  # Increased tolerance for max elongation
            # Check if we made progress
            if final_distance > best_distance + 0.01:  # At least 0.01 Å improvement
                best_atoms = elongated
                best_distance = final_distance
                no_progress_count = 0
            else:
                no_progress_count += 1
            
            consecutive_failures = 0  # Reset failure counter on success

            # Adaptive step size: if we're succeeding easily, try larger steps
            if distance_error < 0.1 and step_size < 1.0:
                step_size = min(step_size * 1.2, 1.0)  # Cap at 1.0 Å
        else:
            consecutive_failures += 1
            no_progress_count += 1

        # Early stopping: no progress for too long
        if no_progress_count >= max_no_progress:
            break

        # Stop if we've had too many consecutive failures
        if consecutive_failures >= max_consecutive_failures:
            break

    # Final optimization: try one more time with the best configuration found
    # This helps recover from local minima (but skip if we didn't improve much)
    if best_distance > current_distance + 0.1:  # Only if we made significant progress
        final_target_pos = n1_pos + direction * best_distance
        final_solver = KinematicChainSolver(best_atoms.copy())
        final_attempt = final_solver.solve(
            anchor_idx=nh3_indices[0],
            mover_idx=nh3_indices[1],
            anchor_pos=n1_pos,
            target_pos=final_target_pos,
            tolerance=0.05,  # Stricter tolerance for final optimization
            max_iter=150  # Reduced from 300 for speed
        )

        final_positions = final_attempt.get_positions()
        final_distance = np.linalg.norm(final_positions[nh3_indices[1]] - final_positions[nh3_indices[0]])
        if final_distance > best_distance:
            best_atoms = final_attempt
            best_distance = final_distance

    return best_atoms


def elongate_molecule(
    molecule: str | Atoms | List[str | Atoms], 
    step_size: float = 0.5, 
    max_iterations: int = 100
) -> Atoms | List[Atoms]:
    """
    Elongate a molecule or batch of molecules by maximizing the distance between terminal NH3+ groups.

    Uses kinematic constraints to find the maximum achievable distance between
    terminal NH3+ nitrogen atoms while respecting bond rigidity and connectivity.

    Parameters
    ----------
    molecule : str | Atoms | List[str | Atoms]
        Input molecule(s) as SMILES string(s), ASE Atoms object(s), or a list of either
    step_size : float, default 0.5
        Distance increment (Å) for each elongation attempt
    max_iterations : int, default 100
        Maximum number of elongation steps to attempt (increased for more aggressive elongation)

    Returns
    -------
    Atoms | List[Atoms]
        Elongated molecule(s) at maximum achievable N-N distance.
        Returns a single Atoms object if input is a single molecule,
        or a list of Atoms objects if input is a list.

    Notes
    -----
    The function identifies terminal NH3+ groups and iteratively increases the target
    distance along the N-N axis, using the kinematic solver to find the maximum
    distance that can be achieved without violating bond constraints.

    If a molecule has fewer than 2 NH3+ groups, returns the input molecule unchanged.
    
    When processing a batch, each molecule is elongated independently.
    """
    # Handle batch processing
    if isinstance(molecule, list):
        return [_elongate_single_molecule(mol, step_size=step_size, max_iterations=max_iterations) 
                for mol in molecule]
    
    # Handle single molecule
    return _elongate_single_molecule(molecule, step_size=step_size, max_iterations=max_iterations)


def check_segment_intersection_2d(p1, p2, q1, q2):
    """
    Returns True if line segment p1-p2 intersects q1-q2 in 2D (ignoring Z).
    """
    # 2D cross product helper
    def ccw(A, B, C):
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

    # Standard segment intersection test
    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))

def find_optimal_spacer_vectors_global(
    starts: List[np.ndarray], 
    targets: List[np.ndarray], 
    cell: np.ndarray,
    spacer_radius: float = 2.0,         # Radius of the spacer "cylinder"
    obstacles: List[Tuple[np.ndarray, float]] = [], # List of (position, radius) for ions/atoms
    attachment_tolerance: float = 3.0  # Exclude obstacles within this distance of start/end points
) -> List[np.ndarray]:
    """
    Finds optimal vectors minimizing length while avoiding 3D volumetric collisions.
    
    Models:
    - Spacers: Cylinders with `spacer_radius`
    - Obstacles: Spheres with specific radii
    
    Parameters
    ----------
    starts, targets : List[np.ndarray]
        Start and end points for spacers
    cell : np.ndarray
        Unit cell
    spacer_radius : float
        Radius of the spacer molecule (approx 2.0 Å for alkyl chains)
    obstacles : List[Tuple[np.ndarray, float]]
        Static atoms/ions to avoid. Format: (position_array, radius_float)
    """
    cell = np.array(cell)
    inv_cell = np.linalg.inv(cell)
    n_spacers = len(starts)
    
    # Filter obstacles: exclude those too close to attachment points
    # (spacers attach to these points, so nearby atoms are expected)
    filtered_obstacles = []
    attachment_tol_sq = attachment_tolerance ** 2
    
    for (obs_pos, obs_rad) in obstacles:
        obs_pos = np.array(obs_pos)
        too_close = False
        
        # Check distance to all start and end points
        for start in starts:
            dist_sq_to_start = np.sum(find_shortest_pbc_vector(obs_pos, start, cell)**2)
            if dist_sq_to_start < attachment_tol_sq:
                too_close = True
                break
        
        if not too_close:
            for target in targets:
                dist_sq_to_target = np.sum(find_shortest_pbc_vector(obs_pos, target, cell)**2)
                if dist_sq_to_target < attachment_tol_sq:
                    too_close = True
                    break
        
        if not too_close:
            filtered_obstacles.append((obs_pos, obs_rad))
    
    obstacles = filtered_obstacles
    
    # 1. Generate Candidates (Top 5 shortest vectors per spacer)
    candidates = []
    shifts = np.array(list(itertools.product([-1, 0, 1], repeat=3)))
    
    for i in range(n_spacers):
        p1 = starts[i]
        p2 = targets[i]
        raw_diff = p2 - p1
        
        diff_frac = raw_diff @ inv_cell
        diff_frac_wrapped = diff_frac - np.round(diff_frac)
        
        cand_frac = diff_frac_wrapped + shifts
        cand_cart = cand_frac @ cell
        dists = np.linalg.norm(cand_cart, axis=1)
        
        # Sort and take top 10 (increased from 5 for better collision avoidance)
        sorted_idx = np.argsort(dists)
        best_indices = sorted_idx[:10]
        candidates.append([cand_cart[j] for j in best_indices])

    # 2. Combinatorial Search with Volumetric Checks
    best_combination = None
    best_collision_free = None
    min_total_cost = float('inf')
    min_collision_free_cost = float('inf')
    
    # Pre-calculate squared radii for faster checks
    spacer_diam_sq = (2 * spacer_radius) ** 2
        
    for vector_set in itertools.product(*candidates):
        lengths_sq = [np.sum(v**2) for v in vector_set]  # Squared lengths (avoid sqrt)
        total_len = sum(np.sqrt(l) for l in lengths_sq)  # Only sqrt when needed for total
        collisions = 0
        
        # --- Check A: Spacer vs Spacer (Cylinder vs Cylinder) ---
        for i in range(n_spacers):
            p1_start = starts[i]
            p1_end = p1_start + vector_set[i]
            
            for j in range(i + 1, n_spacers):
                p2_start = starts[j]
                p2_end = p2_start + vector_set[j]
                
                # Fast 3D segment-segment distance check
                dist_sq = _dist_sq_segment_segment(p1_start, p1_end, p2_start, p2_end)
        
                # Check if distance is less than sum of radii squared
                if dist_sq < spacer_diam_sq:
                    collisions += 1
        
        # --- Check B: Spacer vs Obstacle (Cylinder vs Sphere) ---
        if obstacles:
            # Build KD-tree for spatial indexing (only once per function call)
            obs_positions = np.array([obs[0] for obs in obstacles])
            obs_radii = np.array([obs[1] for obs in obstacles])
            obs_tree = cKDTree(obs_positions)

            # Maximum search radius: spacer length + spacer radius + max obstacle radius
            max_search_radius = (np.max([np.linalg.norm(v) for v in vector_set]) +
                               spacer_radius + np.max(obs_radii) + 2.0)  # Extra buffer

        for i in range(n_spacers):
                p_start = starts[i]
                p_end = p_start + vector_set[i]

                # Find obstacles within search radius of both endpoints
                nearby_start = obs_tree.query_ball_point(p_start, max_search_radius)
                nearby_end = obs_tree.query_ball_point(p_end, max_search_radius)

                # Combine and deduplicate nearby obstacle indices
                nearby_obs_indices = set(nearby_start + nearby_end)

                # Check collisions only with nearby obstacles
                for obs_idx in nearby_obs_indices:
                    obs_pos = obs_positions[obs_idx]
                    obs_rad = obs_radii[obs_idx]
                    min_dist_sq = (spacer_radius + obs_rad) ** 2

                    # Point-Segment distance check (PBC-aware)
                    dist_sq = _dist_sq_point_segment(obs_pos, p_start, p_end, cell=cell)

                    if dist_sq < min_dist_sq:
                        collisions += 1
        
        # Cost Function
        # Huge penalty for collisions to force finding a clean path
        penalty_weight = 100000.0  # Increased penalty
        current_cost = total_len + (collisions * penalty_weight)
        
        # Track best collision-free solution separately
        if collisions == 0:
            if total_len < min_collision_free_cost:
                min_collision_free_cost = total_len
                best_collision_free = vector_set
        
        # Track best overall solution (may have collisions)
        if current_cost < min_total_cost:
            min_total_cost = current_cost
            best_combination = vector_set

    # Prefer collision-free solution, but accept best overall if no collision-free exists
    # (This minimizes collisions even when perfect avoidance isn't possible)
    if best_collision_free is not None:
        return list(best_collision_free)
    elif best_combination is not None:
        # Return best solution (minimal collisions) without warning
        # This is expected behavior when obstacles are unavoidable
        return list(best_combination)
    else:
        # Fallback: return shortest vectors (shouldn't happen, but safety check)
        return [candidates[i][0] for i in range(n_spacers)]

def _dist_sq_point_segment(p: np.ndarray, s1: np.ndarray, s2: np.ndarray, cell: Optional[np.ndarray] = None) -> float:
    """
    Returns squared distance from point p to segment s1-s2.
    If cell is provided, checks PBC-aware distance (checks nearest periodic image).
    """
    if cell is not None:
        # PBC-aware: find shortest distance considering periodic images
        # Get PBC-aware vector from p to s1
        vec_to_s1 = find_shortest_pbc_vector(p, s1, cell)
        # Get PBC-aware vector from s1 to s2 (segment direction)
        vec_s1_to_s2 = find_shortest_pbc_vector(s1, s2, cell)
        
        # If segment is degenerate, return distance to s1
        seg_len_sq = np.sum(vec_s1_to_s2**2)
        if seg_len_sq < 1e-12:
            return np.sum(vec_to_s1**2)
        
        # Project p onto the segment using PBC-aware vectors
        t = np.dot(vec_to_s1, vec_s1_to_s2) / seg_len_sq
        t = np.clip(t, 0.0, 1.0)
        
        # Closest point on segment (in PBC-aware space)
        closest_on_segment = s1 + t * vec_s1_to_s2
        
        # Get PBC-aware distance from p to closest point
        vec_to_closest = find_shortest_pbc_vector(p, closest_on_segment, cell)
        return np.sum(vec_to_closest**2)
    
    # Non-PBC version
    s2_s1 = s2 - s1
    if np.allclose(s2_s1, 0):
        return np.sum((p - s1)**2)
    
    t = np.dot(p - s1, s2_s1) / np.dot(s2_s1, s2_s1)
    t = np.clip(t, 0.0, 1.0)
    projection = s1 + t * s2_s1
    return np.sum((p - projection)**2)

def _get_uc_neighbor_offsets(cell: np.ndarray) -> np.ndarray:
    """Get 27 unit cell neighbor offsets for PBC calculations."""
    multipliers = np.array(np.meshgrid([-1, 0, 1], [-1, 0, 1], [-1, 0, 1])).T.reshape(-1, 1, 3)
    return np.array([np.matmul(cell.T, mult[0]) for mult in multipliers])


def detect_bonds_pbc(atoms: Atoms, cell: Optional[np.ndarray] = None) -> List[Tuple[int, int]]:
    """
    Detect bonds using covalent radii with PBC awareness (improved version).
    """
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


def apply_torque_overlap_avoidance(
    molecule: Atoms,
    existing_molecules: Union[List[Atoms], Atoms],
    n1_index: int,
    n2_index: int,
    cell: Optional[np.ndarray] = None,
    max_iterations: int = 10,
    torque_strength: float = 0.1,
    min_distance: float = 2.5,
    convergence_tol: float = 0.1
) -> Atoms:
    """
    Apply torque-based rotation around N-N axis to avoid overlaps.

    This implements a cheap force field approach that rotates molecules around their
    N-N axis to minimize overlaps with existing molecules.

    Parameters
    ----------
    molecule : Atoms
        The molecule to rotate (already placed in structure)
    existing_molecules : List[Atoms] or Atoms
        List of already-placed molecules OR full structure to avoid overlaps with.
        If Atoms object, all atoms in it will be checked for overlaps.
    n1_index, n2_index : int
        Indices of terminal N atoms defining the rotation axis
    cell : np.ndarray, optional
        Unit cell for PBC-aware distance calculations
    max_iterations : int, default 10
        Maximum rotation attempts
    torque_strength : float, default 0.1
        Strength of torque (radians per iteration)
    min_distance : float, default 2.5
        Minimum allowed distance between atoms (Å)
    convergence_tol : float, default 0.1
        Overlap reduction threshold for convergence

    Returns
    -------
    Atoms
        Molecule rotated to minimize overlaps
    """
    # Handle both list of molecules and single structure
    if isinstance(existing_molecules, Atoms):
        # Convert structure to list of atoms (each atom as a single-atom "molecule")
        # This allows checking against all atoms in the structure
        structure = existing_molecules
        if len(structure) == 0:
            return molecule
        
        # Create list of single-atom "molecules" for compatibility
        # Only include heavy atoms (exclude H) for efficiency
        existing_molecules_list = []
        symbols = structure.get_chemical_symbols()
        positions = structure.get_positions()
        for i in range(len(structure)):
            if symbols[i] != 'H':  # Skip hydrogen for efficiency
                single_atom = Atoms(symbols=[symbols[i]], 
                                   positions=[positions[i]])  # positions[i] is shape (3,), wrapping gives (1, 3)
                existing_molecules_list.append(single_atom)
        existing_molecules = existing_molecules_list
    
    if len(existing_molecules) == 0:
        return molecule

    # Get N-N axis (rotation axis)
    positions = molecule.get_positions()
    n1_pos = positions[n1_index]
    n2_pos = positions[n2_index]
    axis_direction = n2_pos - n1_pos
    axis_length = np.linalg.norm(axis_direction)

    if axis_length < 1e-6:
        return molecule  # Degenerate axis

    axis_direction = axis_direction / axis_length
    pivot_point = (n1_pos + n2_pos) / 2.0  # Midpoint of N-N bond

    # Initial overlap check
    net_force, initial_overlap = calculate_overlap_forces(molecule, existing_molecules, min_distance, cell)

    # Even if no overlap detected, try a small rotation to optimize position
    # Use a more sensitive threshold for detection
    if initial_overlap == 0:
        # Check with a slightly larger threshold to detect near-overlaps
        _, near_overlap = calculate_overlap_forces(molecule, existing_molecules, min_distance + 0.5, cell)
        if near_overlap == 0:
            # No overlaps at all, but still try a small exploratory rotation
            # Use force magnitude to determine rotation direction
            force_magnitude = np.linalg.norm(net_force)
            if force_magnitude < 1e-6:
                return molecule  # Truly no forces, no rotation needed
            # Small exploratory rotation
            initial_overlap = 0.01  # Small value to trigger rotation
        else:
            initial_overlap = near_overlap * 0.1  # Scale down but still rotate

    best_molecule = molecule.copy()
    best_overlap = initial_overlap
    total_rotation = 0.0

    # Try rotations in both directions
    for direction in [1, -1]:  # Clockwise and counterclockwise
        current_molecule = molecule.copy()
        current_overlap = initial_overlap
        consecutive_worse = 0
        max_consecutive_worse = 3

        for iteration in range(max_iterations):
            # Calculate overlap forces
            net_force, overlap_energy = calculate_overlap_forces(
                current_molecule, existing_molecules, min_distance, cell
            )

            # Check convergence
            overlap_reduction = initial_overlap - overlap_energy
            if overlap_energy < best_overlap:
                best_overlap = overlap_energy
                best_molecule = current_molecule.copy()
                consecutive_worse = 0
            else:
                consecutive_worse += 1

            # Stop if converged or too many consecutive worse steps
            if overlap_reduction > convergence_tol or consecutive_worse >= max_consecutive_worse:
                break

            # Calculate torque from forces
            torque = force_to_torque(net_force, pivot_point, axis_direction, current_molecule.get_positions())
            
            # Ensure we have a non-zero rotation
            if abs(torque) < 1e-6:
                # If torque is zero, use force direction to determine rotation
                force_magnitude = np.linalg.norm(net_force)
                if force_magnitude > 1e-6:
                    # Use a small default rotation based on force direction
                    torque = 0.1 * force_magnitude
                else:
                    break  # No force, no rotation

            # Apply rotation with minimum step size to ensure movement
            rotation_angle = direction * torque_strength * max(abs(torque), 0.05)  # Minimum 0.05 rad rotation
            current_molecule = rotate_around_axis(current_molecule, pivot_point, axis_direction, rotation_angle)
            total_rotation += rotation_angle

            # Prevent excessive rotation (max 90 degrees total)
            if abs(total_rotation) > np.pi/2:
                break

    return best_molecule


def calculate_overlap_forces(
    molecule: Atoms,
    existing_molecules: List[Atoms],
    min_distance: float = 2.5,
    cell: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, float]:
    """
    Calculate net repulsion force and total overlap energy.

    This computes a simple repulsion force model where atoms closer than
    min_distance repel each other.

    Parameters
    ----------
    molecule : Atoms
        The molecule to check for overlaps
    existing_molecules : List[Atoms]
        List of molecules to check against
    min_distance : float
        Minimum allowed distance between atoms
    cell : np.ndarray, optional
        Unit cell for PBC-aware calculations

    Returns
    -------
    Tuple[np.ndarray, float]
        (net_force_vector, total_overlap_energy)
    """
    mol_positions = molecule.get_positions()
    mol_symbols = molecule.get_chemical_symbols()

    net_force = np.zeros(3)
    total_overlap_energy = 0.0
    force_constant = 1.0

    # Only consider heavy atoms (exclude H) for performance and relevance
    heavy_atom_mask = np.array([s != 'H' for s in mol_symbols])
    mol_positions = mol_positions[heavy_atom_mask]

    if len(mol_positions) == 0:
        return net_force, total_overlap_energy

    for existing_mol in existing_molecules:
        existing_positions = existing_mol.get_positions()
        existing_symbols = existing_mol.get_chemical_symbols()

        # Only consider heavy atoms in existing molecules too
        existing_heavy_mask = np.array([s != 'H' for s in existing_symbols])
        existing_positions = existing_positions[existing_heavy_mask]

        if len(existing_positions) == 0:
            continue

        # Calculate all pairwise distances (PBC-aware)
        for mol_pos in mol_positions:
            min_dist = float('inf')
            closest_existing_pos = None

            # Find closest existing atom (check all 27 unit cell images if PBC)
            for existing_pos in existing_positions:
                if cell is not None:
                    # PBC-aware distance
                    dist_vec = find_shortest_pbc_vector(mol_pos, existing_pos, cell)
                    dist = np.linalg.norm(dist_vec)
                else:
                    # Simple distance
                    dist = np.linalg.norm(mol_pos - existing_pos)

                if dist < min_dist:
                    min_dist = dist
                    closest_existing_pos = existing_pos

            # Calculate repulsion force if too close
            if min_dist < min_distance and closest_existing_pos is not None:
                overlap = min_distance - min_dist
                energy_contribution = force_constant * overlap**2

                # Force direction: from existing atom to molecule atom
                if cell is not None:
                    force_direction = find_shortest_pbc_vector(closest_existing_pos, mol_pos, cell)
                else:
                    force_direction = mol_pos - closest_existing_pos

                force_magnitude = np.linalg.norm(force_direction)
                if force_magnitude > 1e-6:
                    force_direction = force_direction / force_magnitude
                    force_vector = force_constant * overlap * force_direction

                    net_force += force_vector
                    total_overlap_energy += energy_contribution

    return net_force, total_overlap_energy


def force_to_torque(
    force: np.ndarray,
    pivot_point: np.ndarray,
    axis_direction: np.ndarray,
    atom_positions: np.ndarray
) -> float:
    """
    Convert net force to torque magnitude around axis.

    Calculates the torque by projecting the force onto the plane perpendicular
    to the axis and computing the effective lever arm.

    Parameters
    ----------
    force : np.ndarray
        Net force vector applied to the molecule
    pivot_point : np.ndarray
        Point around which to calculate torque (midpoint of N-N bond)
    axis_direction : np.ndarray
        Direction of rotation axis (N-N bond direction)
    atom_positions : np.ndarray
        Positions of all atoms in the molecule

    Returns
    -------
    float
        Torque magnitude (positive = one direction, negative = other)
    """
    # Project force onto plane perpendicular to axis
    force_parallel = np.dot(force, axis_direction) * axis_direction
    force_perpendicular = force - force_parallel
    
    # Use center of mass as effective point of force application
    com = np.mean(atom_positions, axis=0)
    lever_arm = com - pivot_point
    
    # Project lever arm onto plane perpendicular to axis
    lever_parallel = np.dot(lever_arm, axis_direction) * axis_direction
    lever_perpendicular = lever_arm - lever_parallel
    
    # Torque = r_perp × F_perp (projected onto axis)
    torque_vector = np.cross(lever_perpendicular, force_perpendicular)
    torque_magnitude = np.dot(torque_vector, axis_direction)
    
    return torque_magnitude


def rotate_around_axis(
    molecule: Atoms,
    pivot_point: np.ndarray,
    axis_direction: np.ndarray,
    angle: float
) -> Atoms:
    """
    Rotate molecule around axis by given angle.

    Uses Rodrigues' rotation formula for efficient rotation around arbitrary axis.

    Parameters
    ----------
    molecule : Atoms
        Molecule to rotate
    pivot_point : np.ndarray
        Point to rotate around
    axis_direction : np.ndarray
        Rotation axis direction (should be normalized)
    angle : float
        Rotation angle in radians

    Returns
    -------
    Atoms
        Rotated molecule
    """
    rotated_molecule = molecule.copy()
    positions = rotated_molecule.get_positions()

    # Translate so pivot point is at origin
    positions_centered = positions - pivot_point

    # Rodrigues' rotation formula
    # R = cosθI + (1-cosθ)kk^T + sinθK
    # where K = [[0,-k_z,k_y], [k_z,0,-k_x], [-k_y,k_x,0]]

    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)
    k = axis_direction

    # Cross product matrix K
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])

    # Rodrigues' formula
    R = (cos_theta * np.eye(3) +
         (1 - cos_theta) * np.outer(k, k) +
         sin_theta * K)

    # Apply rotation
    positions_rotated = positions_centered @ R.T

    # Translate back
    rotated_molecule.set_positions(positions_rotated + pivot_point)

    return rotated_molecule


def _dist_sq_segment_segment(p1: np.ndarray, p2: np.ndarray, q1: np.ndarray, q2: np.ndarray) -> float:
    """
    Returns squared minimum distance between two line segments p1-p2 and q1-q2.
    Implementation based on "Real-Time Collision Detection" (Ericson).
    """
    d1 = p2 - p1
    d2 = q2 - q1
    r = p1 - q1
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    # Check if segments degenerate into points
    if a <= 1e-8 and e <= 1e-8:
        return np.dot(r, r)
    if a <= 1e-8:
        return _dist_sq_point_segment(p1, q1, q2)
    if e <= 1e-8:
        return _dist_sq_point_segment(q1, p1, p2)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    
    # Parallel lines case
    if denom != 0.0:
        s = np.clip((b * f - c * e) / denom, 0.0, 1.0)
    else:
        s = 0.0
        
    # Compute point on L1 closest to L2, then clamp t
    t = (b * s + f) / e
    if t < 0.0:
        t = 0.0
        s = np.clip(-c / a, 0.0, 1.0)
    elif t > 1.0:
        t = 1.0
        s = np.clip((b - c) / a, 0.0, 1.0)
        
    p_closest = p1 + s * d1
    q_closest = q1 + t * d2
    return np.sum((p_closest - q_closest)**2)