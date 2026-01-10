"""
Collision Detection and Avoidance Module for Molecular Structures.

This module provides comprehensive collision detection and resolution strategies
to prevent atomic overlaps during molecular placement, ensuring DFT calculations
can proceed without matrix diagonalization failures or force explosions.

Key Features:
- PBC-aware 3D distance calculations
- Covalent radii-based minimum distance thresholds
- Multiple resolution strategies (rotate, nudge, optimize, reject)
- Efficient vectorized operations for performance
- Envelope-based collision detection using 3D convex hulls (fast O(hull_size) vs O(n_atoms^2))
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from ase import Atoms
from ase.data import covalent_radii, vdw_radii
from ase.optimize import FIRE
from ase.constraints import FixAtoms

from ..utils.geometry import _calculate_distances
from .primitives import (
    calculate_molecular_envelope,
    envelopes_overlap,
)


@dataclass
class CollisionInfo:
    """Information about a detected collision."""
    atom1_idx: int  # Index in molecule
    atom2_idx: int  # Index in existing structure
    distance: float  # Actual distance
    min_distance: float  # Required minimum distance
    position1: np.ndarray  # Position of atom1
    position2: np.ndarray  # Position of atom2


class CollisionDetector:
    """PBC-aware collision detection using covalent radii and geometric constraints."""

    def __init__(self, cell: Optional[np.ndarray] = None, pbc: Optional[List[bool]] = None):
        """
        Initialize collision detector.

        Parameters
        ----------
        cell : np.ndarray, optional
            Unit cell matrix (3x3) for PBC calculations
        pbc : list of bool, optional
            Periodic boundary conditions [x, y, z]. Defaults to [True, True, True] if cell provided
        """
        self.cell = cell
        if pbc is None and cell is not None:
            self.pbc = [True, True, True]
        else:
            self.pbc = pbc or [False, False, False]

    def _get_min_distance(self, symbol1: str, symbol2: str, buffer: float = 0.8) -> float:
        """
        Calculate minimum allowed distance between two atoms.

        Parameters
        ----------
        symbol1, symbol2 : str
            Atomic symbols
        buffer : float
            Additional safety buffer in Angstroms

        Returns
        -------
        float
            Minimum allowed distance
        """
        from ase.data import atomic_numbers

        try:
            atomic_num1 = atomic_numbers[symbol1]
            r1 = covalent_radii[atomic_num1]
        except KeyError:
            r1 = 1.5  # Default for unknown atoms

        try:
            atomic_num2 = atomic_numbers[symbol2]
            r2 = covalent_radii[atomic_num2]
        except KeyError:
            r2 = 1.5  # Default for unknown atoms

        return r1 + r2 + buffer

    def detect_collisions(
        self,
        molecule: Atoms,
        existing_structure: Atoms,
        min_distance_override: Optional[float] = None,
        heavy_atoms_only: bool = True,
    ) -> List[CollisionInfo]:
        """
        Detect all collisions between molecule and existing structure.

        Parameters
        ----------
        molecule : Atoms
            Molecule to check for collisions
        existing_structure : Atoms
            Existing atomic structure
        min_distance_override : float, optional
            Override minimum distance threshold (ignores covalent radii if provided)
        heavy_atoms_only : bool
            If True, skip hydrogen-hydrogen collision checks

        Returns
        -------
        List[CollisionInfo]
            List of detected collisions
        """
        if len(molecule) == 0:
            return []

        collisions = []

        # Get positions and symbols
        mol_positions = molecule.get_positions()
        mol_symbols = molecule.get_chemical_symbols()
        existing_positions = existing_structure.get_positions()
        existing_symbols = existing_structure.get_chemical_symbols()

        # Filter atoms if heavy_atoms_only is enabled
        if heavy_atoms_only:
            mol_mask = np.array([s != 'H' for s in mol_symbols])
            existing_mask = np.array([s != 'H' for s in existing_symbols])

            if not np.any(mol_mask):
                mol_mask = np.ones(len(mol_symbols), dtype=bool)  # Fallback if no heavy atoms
            if not np.any(existing_mask):
                existing_mask = np.ones(len(existing_symbols), dtype=bool)  # Fallback

            mol_positions_filtered = mol_positions[mol_mask]
            mol_symbols_filtered = [s for s, m in zip(mol_symbols, mol_mask) if m]
            existing_positions_filtered = existing_positions[existing_mask]
            existing_symbols_filtered = [s for s, m in zip(existing_symbols, existing_mask) if m]
        else:
            mol_positions_filtered = mol_positions
            mol_symbols_filtered = mol_symbols
            existing_positions_filtered = existing_positions
            existing_symbols_filtered = existing_symbols

        # Check each atom in molecule against all atoms in existing structure
        for i, (pos1, sym1) in enumerate(zip(mol_positions_filtered, mol_symbols_filtered)):
            if min_distance_override is not None:
                min_dist = min_distance_override
            else:
                # Calculate minimum distance for each pair
                min_dists = [self._get_min_distance(sym1, sym2) for sym2 in existing_symbols_filtered]
                min_dists = np.array(min_dists)

            # Use PBC-aware distance calculation if cell is available
            if self.cell is not None and any(self.pbc):
                distances = _calculate_distances(pos1, existing_positions_filtered, self.cell, self.pbc)
            else:
                # Simple Euclidean distance
                diff = existing_positions_filtered - pos1
                distances = np.linalg.norm(diff, axis=1)

            # Find violations
            if min_distance_override is not None:
                violations = distances < min_distance_override
            else:
                violations = distances < min_dists

            # Record collisions
            for j, violates in enumerate(violations):
                if violates:
                    dist = distances[j]
                    min_dist = min_distance_override if min_distance_override is not None else min_dists[j]

                    # Get original indices (accounting for filtering)
                    if heavy_atoms_only:
                        orig_mol_idx = np.where(mol_mask)[0][i]
                        orig_existing_idx = np.where(existing_mask)[0][j]
                    else:
                        orig_mol_idx = i
                        orig_existing_idx = j

                    collision = CollisionInfo(
                        atom1_idx=orig_mol_idx,
                        atom2_idx=orig_existing_idx,
                        distance=dist,
                        min_distance=min_dist,
                        position1=mol_positions[orig_mol_idx],
                        position2=existing_positions[orig_existing_idx]
                    )
                    collisions.append(collision)

        return collisions

    def detect_collisions_envelope(
        self,
        molecule: Atoms,
        existing_structure: Atoms,
        envelope_tolerance: float = 0.5,
        detailed_if_overlap: bool = True,
    ) -> Tuple[bool, List[CollisionInfo]]:
        """
        Fast envelope-based collision detection using convex hulls.
        
        This is a two-phase approach:
        1. Quick check: Calculate convex hull envelopes and check for overlap
        2. Detailed check: Only if envelopes overlap, do atom-by-atom check
        
        This provides O(hull_size) complexity for non-colliding molecules,
        much faster than O(n_atoms^2) atom-by-atom checks.
        
        Parameters
        ----------
        molecule : Atoms
            Molecule to check for collisions
        existing_structure : Atoms
            Existing atomic structure
        envelope_tolerance : float
            Additional buffer for envelope overlap detection
        detailed_if_overlap : bool
            If True, perform detailed atom-by-atom check when envelopes overlap
            
        Returns
        -------
        Tuple[bool, List[CollisionInfo]]
            (has_collision, collision_details). If detailed_if_overlap=False and
            envelopes overlap, collision_details will be empty but has_collision=True.
        """
        if len(molecule) == 0:
            return False, []
        
        # Get positions and radii for envelope calculation
        mol_positions = molecule.get_positions()
        existing_positions = existing_structure.get_positions()
        
        # Get van der Waals radii (more appropriate for envelope than covalent)
        mol_radii = self._get_vdw_radii(molecule)
        existing_radii = self._get_vdw_radii(existing_structure)
        
        # Calculate molecular envelopes (convex hulls expanded by atomic radii)
        mol_envelope = calculate_molecular_envelope(mol_positions, mol_radii)
        existing_envelope = calculate_molecular_envelope(existing_positions, existing_radii)
        
        # Quick envelope overlap check
        if not envelopes_overlap(mol_envelope, existing_envelope, tolerance=envelope_tolerance):
            return False, []  # No collision possible
        
        # Envelopes overlap - need detailed check if requested
        if detailed_if_overlap:
            collisions = self.detect_collisions(molecule, existing_structure)
            return len(collisions) > 0, collisions
        else:
            # Just report that envelopes overlap
            return True, []
    
    def _get_vdw_radii(self, atoms: Atoms) -> np.ndarray:
        """Get van der Waals radii for atoms."""
        from ase.data import atomic_numbers
        
        radii = []
        for symbol in atoms.get_chemical_symbols():
            atomic_num = atomic_numbers.get(symbol, 6)  # Default to C
            # vdw_radii returns nan for some elements, fallback to covalent + buffer
            vdw = vdw_radii[atomic_num]
            if np.isnan(vdw):
                vdw = covalent_radii[atomic_num] + 0.5
            radii.append(vdw)
        return np.array(radii)


def detect_collisions(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    min_distance: float = 0.8,
    heavy_atoms_only: bool = True,
    pbc: Optional[List[bool]] = None,
) -> List[CollisionInfo]:
    """
    Convenience function for collision detection.

    Parameters
    ----------
    molecule : Atoms
        Molecule to check for collisions
    existing_structure : Atoms
        Existing atomic structure
    cell : np.ndarray, optional
        Unit cell matrix for PBC calculations
    min_distance : float
        Minimum distance threshold (overrides covalent radii if > 0.8)
    heavy_atoms_only : bool
        Skip H-H collision checks for performance

    Returns
    -------
    List[CollisionInfo]
        Detected collisions
    """
    detector = CollisionDetector(cell=cell, pbc=pbc)

    if min_distance > 0.8:
        # Use override if significantly larger than default buffer
        return detector.detect_collisions(molecule, existing_structure,
                                        min_distance_override=min_distance,
                                        heavy_atoms_only=heavy_atoms_only)
    else:
        return detector.detect_collisions(molecule, existing_structure,
                                        heavy_atoms_only=heavy_atoms_only)


def detect_collisions_fast(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    envelope_tolerance: float = 0.5,
    detailed_if_overlap: bool = True,
    pbc: Optional[List[bool]] = None,
) -> Tuple[bool, List[CollisionInfo]]:
    """
    Fast envelope-based collision detection using convex hulls.
    
    This uses a two-phase approach for efficiency:
    1. Quick check: Calculate convex hull envelopes and check for overlap
    2. Detailed check: Only if envelopes overlap, do atom-by-atom check
    
    For non-colliding molecules, this is O(hull_size) instead of O(n_atoms^2).
    
    Parameters
    ----------
    molecule : Atoms
        Molecule to check for collisions
    existing_structure : Atoms
        Existing atomic structure
    cell : np.ndarray, optional
        Unit cell matrix for PBC calculations
    envelope_tolerance : float
        Additional buffer for envelope overlap detection
    detailed_if_overlap : bool
        If True, perform detailed atom-by-atom check when envelopes overlap
    pbc : List[bool], optional
        Periodic boundary conditions
        
    Returns
    -------
    Tuple[bool, List[CollisionInfo]]
        (has_collision, collision_details)
    """
    detector = CollisionDetector(cell=cell, pbc=pbc)
    return detector.detect_collisions_envelope(
        molecule, existing_structure,
        envelope_tolerance=envelope_tolerance,
        detailed_if_overlap=detailed_if_overlap
    )


# ============================================================================
# Resolution Strategies
# ============================================================================

def _rotate_molecule_around_axis(
    molecule: Atoms,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    angle_degrees: float
) -> Atoms:
    """Rotate molecule around specified axis by given angle."""
    rotated = molecule.copy()

    # Translate to axis point as origin
    rotated.translate(-axis_point)

    # Rotate around axis
    rotated.rotate(angle_degrees, v=axis_direction, center=[0, 0, 0])

    # Translate back
    rotated.translate(axis_point)

    return rotated


def _nudge_molecule_xy(
    molecule: Atoms,
    dx: float,
    dy: float,
    reference_point: Optional[np.ndarray] = None
) -> Atoms:
    """Apply small XY translation to molecule."""
    nudged = molecule.copy()

    if reference_point is None:
        # Use center of mass
        com = nudged.get_center_of_mass()
        translation = np.array([dx, dy, 0])
    else:
        # Translate relative to reference point
        translation = np.array([dx, dy, 0])

    nudged.translate(translation)
    return nudged


def _find_nh3_axis(molecule: Atoms) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Find the N-N axis for double spacers.

    Returns
    -------
    Tuple[Optional[np.ndarray], Optional[np.ndarray]]
        (axis_start, axis_direction) or (None, None) if not a double spacer
    """
    from .spacer import _find_terminal_nitrogens

    n_indices, nh3_indices = _find_terminal_nitrogens(molecule)

    if len(nh3_indices) < 2:
        return None, None

    positions = molecule.get_positions()
    n1_pos = positions[nh3_indices[0]]
    n2_pos = positions[nh3_indices[1]]

    axis_start = n1_pos
    axis_direction = n2_pos - n1_pos
    axis_length = np.linalg.norm(axis_direction)

    if axis_length < 1e-6:
        return None, None

    axis_direction = axis_direction / axis_length
    return axis_start, axis_direction


def resolve_collisions_rotate(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    max_attempts: int = 36,
    angle_step: float = 10.0,
    min_distance: float = 0.8,
    heavy_atoms_only: bool = True,
) -> Tuple[Atoms, bool]:
    """
    Resolve collisions by rotating molecule around its N-N axis.

    Parameters
    ----------
    molecule : Atoms
        Molecule to resolve collisions for
    existing_structure : Atoms
        Existing structure to avoid collisions with
    cell : np.ndarray, optional
        Unit cell for PBC calculations
    max_attempts : int
        Maximum rotation attempts
    angle_step : float
        Rotation angle increment in degrees
    min_distance : float
        Minimum distance threshold
    heavy_atoms_only : bool
        Skip H-H checks

    Returns
    -------
    Tuple[Atoms, bool]
        (resolved_molecule, success)
    """
    # Find rotation axis
    axis_start, axis_direction = _find_nh3_axis(molecule)

    if axis_start is None or axis_direction is None:
        # Not a double spacer, try nudge strategy instead
        return resolve_collisions_nudge(molecule, existing_structure, cell,
                                      max_attempts, min_distance, heavy_atoms_only)

    detector = CollisionDetector(cell=cell)

    # Check initial collisions
    initial_collisions = detector.detect_collisions(molecule, existing_structure,
                                                  heavy_atoms_only=heavy_atoms_only)
    if not initial_collisions:
        return molecule, True

    # Try rotations
    for attempt in range(max_attempts):
        angle = attempt * angle_step

        # Skip 0-degree rotation (already checked)
        if attempt == 0:
            continue

        rotated = _rotate_molecule_around_axis(molecule, axis_start, axis_direction, angle)

        # Check for collisions
        collisions = detector.detect_collisions(rotated, existing_structure,
                                              heavy_atoms_only=heavy_atoms_only)
        if not collisions:
            return rotated, True

    return molecule, False  # Failed to resolve


def resolve_collisions_nudge(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    max_attempts: int = 25,
    nudge_step: float = 0.2,
    min_distance: float = 0.8,
    heavy_atoms_only: bool = True,
) -> Tuple[Atoms, bool]:
    """
    Resolve collisions by small XY translations.

    Parameters
    ----------
    molecule : Atoms
        Molecule to resolve collisions for
    existing_structure : Atoms
        Existing structure to avoid collisions with
    cell : np.ndarray, optional
        Unit cell for PBC calculations
    max_attempts : int
        Maximum nudge attempts (arranged in spiral pattern)
    nudge_step : float
        Distance increment for nudges
    min_distance : float
        Minimum distance threshold
    heavy_atoms_only : bool
        Skip H-H checks

    Returns
    -------
    Tuple[Atoms, bool]
        (resolved_molecule, success)
    """
    detector = CollisionDetector(cell=cell)

    # Check initial collisions
    initial_collisions = detector.detect_collisions(molecule, existing_structure,
                                                  heavy_atoms_only=heavy_atoms_only)
    if not initial_collisions:
        return molecule, True

    # Generate spiral nudge pattern
    nudges = []
    for layer in range(1, int(np.sqrt(max_attempts)) + 1):
        for dx in [-layer, 0, layer]:
            for dy in [-layer, 0, layer]:
                if dx == 0 and dy == 0:
                    continue  # Skip center
                nudges.append((dx * nudge_step, dy * nudge_step))

                if len(nudges) >= max_attempts:
                    break
            if len(nudges) >= max_attempts:
                break
        if len(nudges) >= max_attempts:
            break

    # Try each nudge
    for dx, dy in nudges:
        nudged = _nudge_molecule_xy(molecule, dx, dy)

        collisions = detector.detect_collisions(nudged, existing_structure,
                                              heavy_atoms_only=heavy_atoms_only)
        if not collisions:
            return nudged, True

    return molecule, False  # Failed to resolve


def resolve_collisions_optimize(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    max_steps: int = 50,
    force_threshold: float = 0.1,
    min_distance: float = 0.8,
    heavy_atoms_only: bool = True,
) -> Tuple[Atoms, bool]:
    """
    Resolve collisions using geometry optimization to push atoms apart.

    This is the most robust but slowest method. Uses ASE FIRE optimizer
    with fixed existing structure atoms.

    Parameters
    ----------
    molecule : Atoms
        Molecule to resolve collisions for
    existing_structure : Atoms
        Existing structure (will be fixed during optimization)
    cell : np.ndarray, optional
        Unit cell for PBC calculations
    max_steps : int
        Maximum optimization steps
    force_threshold : float
        Force convergence threshold
    min_distance : float
        Minimum distance threshold
    heavy_atoms_only : bool
        Skip H-H checks

    Returns
    -------
    Tuple[Atoms, bool]
        (resolved_molecule, success)
    """
    detector = CollisionDetector(cell=cell)

    # Check initial collisions
    initial_collisions = detector.detect_collisions(molecule, existing_structure,
                                                  heavy_atoms_only=heavy_atoms_only)
    if not initial_collisions:
        return molecule, True

    # Combine structures for optimization
    combined = existing_structure.copy()
    molecule_start_idx = len(combined)

    # Add molecule atoms
    for atom in molecule:
        combined.append(atom)

    # Fix existing structure atoms
    fix_indices = list(range(molecule_start_idx))
    constraints = [FixAtoms(indices=fix_indices)]
    combined.set_constraint(constraints)

    # Set calculator (use Lennard-Jones for repulsive forces)
    from ase.calculators.lj import LennardJones
    calc = LennardJones()
    combined.set_calculator(calc)

    # Optimize
    optimizer = FIRE(combined)
    optimizer.run(fmax=force_threshold, steps=max_steps)

    # Extract optimized molecule
    optimized_molecule = combined[molecule_start_idx:]

    # Check if collisions resolved
    final_collisions = detector.detect_collisions(optimized_molecule, existing_structure,
                                                heavy_atoms_only=heavy_atoms_only)

    success = len(final_collisions) == 0
    return optimized_molecule, success


def resolve_collisions_reject(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    min_distance: float = 0.8,
    heavy_atoms_only: bool = True,
) -> Tuple[Atoms, bool]:
    """
    Reject placement if collisions detected (user gets warning).

    This strategy never resolves collisions - it just reports them.

    Parameters
    ----------
    molecule : Atoms
        Molecule to check for collisions
    existing_structure : Atoms
        Existing structure
    cell : np.ndarray, optional
        Unit cell for PBC calculations
    min_distance : float
        Minimum distance threshold
    heavy_atoms_only : bool
        Skip H-H checks

    Returns
    -------
    Tuple[Atoms, bool]
        (original_molecule, False) if collisions found, (molecule, True) if no collisions
    """
    detector = CollisionDetector(cell=cell)
    collisions = detector.detect_collisions(molecule, existing_structure,
                                          heavy_atoms_only=heavy_atoms_only)

    if collisions:
        # Report collision details
        warnings.warn(
            f"Collision detected during molecular placement! "
            f"Found {len(collisions)} atomic overlaps. "
            f"Closest collision: {min(c.distance for c in collisions):.3f} Å "
            f"(required: {min(c.min_distance for c in collisions):.3f} Å). "
            f"Consider using a different collision_strategy or adjusting geometry."
        )
        return molecule, False

    return molecule, True


def resolve_collisions(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    strategy: str = "rotate",
    **kwargs
) -> Tuple[Atoms, bool]:
    """
    Resolve collisions using specified strategy.

    Parameters
    ----------
    molecule : Atoms
        Molecule to resolve collisions for
    existing_structure : Atoms
        Existing structure to avoid collisions with
    cell : np.ndarray, optional
        Unit cell for PBC calculations
    strategy : str
        Resolution strategy: "rotate", "nudge", "optimize", "reject", "off"
    **kwargs
        Additional arguments passed to specific strategy functions

    Returns
    -------
    Tuple[Atoms, bool]
        (resolved_molecule, success). If strategy="off", returns (molecule, True)
    """
    strategy = strategy.lower()

    if strategy == "off":
        return molecule, True

    elif strategy == "rotate":
        return resolve_collisions_rotate(molecule, existing_structure, cell, **kwargs)

    elif strategy == "nudge":
        return resolve_collisions_nudge(molecule, existing_structure, cell, **kwargs)

    elif strategy == "optimize":
        return resolve_collisions_optimize(molecule, existing_structure, cell, **kwargs)

    elif strategy == "reject":
        return resolve_collisions_reject(molecule, existing_structure, cell, **kwargs)

    else:
        raise ValueError(f"Unknown collision resolution strategy: {strategy}. "
                        "Choose from: 'rotate', 'nudge', 'optimize', 'reject', 'off'")
