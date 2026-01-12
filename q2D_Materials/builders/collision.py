"""Collision detection and resolution for molecular placement."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
from ase import Atoms
from ase.data import vdw_radii
from ..utils.properties.atomic_properties import get_covalent_radius
from ase.optimize import FIRE
from ase.constraints import FixAtoms

from ..utils.geometry.geometry import _calculate_distances
from .primitives import (
    calculate_molecular_envelope,
    envelopes_overlap,
)


@dataclass
class CollisionInfo:
    """Information about a detected collision."""
    atom1_idx: int
    atom2_idx: int
    distance: float
    min_distance: float
    position1: np.ndarray
    position2: np.ndarray


class CollisionDetector:
    """PBC-aware collision detection using covalent radii."""

    def __init__(self, cell: Optional[np.ndarray] = None, pbc: Optional[List[bool]] = None):
        """
        Initialize collision detector.

        Parameters
        ----------
        cell : np.ndarray, optional
            Unit cell matrix (3x3) for PBC calculations
        pbc : list of bool, optional
            Periodic boundary conditions [x, y, z]
        """
        self.cell = cell
        if pbc is None and cell is not None:
            self.pbc = [True, True, True]
        else:
            self.pbc = pbc or [False, False, False]

    def _get_min_distance(self, symbol1: str, symbol2: str, buffer: float = 0.8) -> float:
        """Calculate minimum allowed distance between two atoms."""
        r1 = get_covalent_radius(symbol1, default=1.5)
        r2 = get_covalent_radius(symbol2, default=1.5)
        return r1 + r2 + buffer

    def detect_collisions(
        self,
        molecule: Atoms,
        existing_structure: Atoms,
        min_distance_override: Optional[float] = None,
        heavy_atoms_only: bool = True,
    ) -> List[CollisionInfo]:
        """Detect all collisions between molecule and existing structure."""
        if len(molecule) == 0:
            return []

        collisions = []

        mol_positions = molecule.get_positions()
        mol_symbols = molecule.get_chemical_symbols()
        existing_positions = existing_structure.get_positions()
        existing_symbols = existing_structure.get_chemical_symbols()

        if heavy_atoms_only:
            mol_mask = np.array([s != 'H' for s in mol_symbols])
            existing_mask = np.array([s != 'H' for s in existing_symbols])

            if not np.any(mol_mask):
                mol_mask = np.ones(len(mol_symbols), dtype=bool)
            if not np.any(existing_mask):
                existing_mask = np.ones(len(existing_symbols), dtype=bool)

            mol_positions_filtered = mol_positions[mol_mask]
            mol_symbols_filtered = [s for s, m in zip(mol_symbols, mol_mask) if m]
            existing_positions_filtered = existing_positions[existing_mask]
            existing_symbols_filtered = [s for s, m in zip(existing_symbols, existing_mask) if m]
        else:
            mol_positions_filtered = mol_positions
            mol_symbols_filtered = mol_symbols
            existing_positions_filtered = existing_positions
            existing_symbols_filtered = existing_symbols

        for i, (pos1, sym1) in enumerate(zip(mol_positions_filtered, mol_symbols_filtered)):
            if min_distance_override is not None:
                min_dist = min_distance_override
            else:
                min_dists = [self._get_min_distance(sym1, sym2) for sym2 in existing_symbols_filtered]
                min_dists = np.array(min_dists)

            if self.cell is not None and any(self.pbc):
                distances = _calculate_distances(pos1, existing_positions_filtered, self.cell, self.pbc)
            else:
                diff = existing_positions_filtered - pos1
                distances = np.linalg.norm(diff, axis=1)

            if min_distance_override is not None:
                violations = distances < min_distance_override
            else:
                violations = distances < min_dists

            for j, violates in enumerate(violations):
                if violates:
                    dist = distances[j]
                    min_dist = min_distance_override if min_distance_override is not None else min_dists[j]

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
            (has_collision, collision_details)
        """
        if len(molecule) == 0:
            return False, []

        mol_positions = molecule.get_positions()
        existing_positions = existing_structure.get_positions()
        mol_radii = self._get_vdw_radii(molecule)
        existing_radii = self._get_vdw_radii(existing_structure)
        mol_envelope = calculate_molecular_envelope(mol_positions, mol_radii)
        existing_envelope = calculate_molecular_envelope(existing_positions, existing_radii)

        if not envelopes_overlap(mol_envelope, existing_envelope, tolerance=envelope_tolerance):
            return False, []

        if detailed_if_overlap:
            collisions = self.detect_collisions(molecule, existing_structure)
            return len(collisions) > 0, collisions
        return True, []
    
    def _get_vdw_radii(self, atoms: Atoms) -> np.ndarray:
        """Get van der Waals radii for atoms."""
        from ase.data import atomic_numbers

        radii = []
        for symbol in atoms.get_chemical_symbols():
            atomic_num = atomic_numbers.get(symbol, 6)
            vdw = vdw_radii[atomic_num]
            if np.isnan(vdw):
                # Fallback: use covalent radius + 0.5 if vdw radius not available
                vdw = get_covalent_radius(symbol, default=1.5) + 0.5
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
    """Convenience function for collision detection."""
    detector = CollisionDetector(cell=cell, pbc=pbc)
    if min_distance > 0.8:
        return detector.detect_collisions(molecule, existing_structure,
                                        min_distance_override=min_distance,
                                        heavy_atoms_only=heavy_atoms_only)
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
    """Fast envelope-based collision detection using convex hulls."""
    detector = CollisionDetector(cell=cell, pbc=pbc)
    return detector.detect_collisions_envelope(
        molecule, existing_structure,
        envelope_tolerance=envelope_tolerance,
        detailed_if_overlap=detailed_if_overlap
    )


# Resolution Strategies

def _rotate_molecule_around_axis(
    molecule: Atoms,
    axis_point: np.ndarray,
    axis_direction: np.ndarray,
    angle_degrees: float
) -> Atoms:
    """Rotate molecule around specified axis by given angle."""
    rotated = molecule.copy()
    rotated.translate(-axis_point)
    rotated.rotate(angle_degrees, v=axis_direction, center=[0, 0, 0])
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
    nudged.translate(np.array([dx, dy, 0]))
    return nudged


def _find_nh3_axis(molecule: Atoms) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Find the N-N axis for double spacers."""
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
    """Resolve collisions by rotating molecule around its N-N axis."""
    axis_start, axis_direction = _find_nh3_axis(molecule)
    if axis_start is None or axis_direction is None:
        return resolve_collisions_nudge(molecule, existing_structure, cell,
                                      max_attempts, min_distance, heavy_atoms_only)

    detector = CollisionDetector(cell=cell)
    initial_collisions = detector.detect_collisions(molecule, existing_structure,
                                                  heavy_atoms_only=heavy_atoms_only)
    if not initial_collisions:
        return molecule, True

    for attempt in range(max_attempts):
        if attempt == 0:
            continue
        angle = attempt * angle_step
        rotated = _rotate_molecule_around_axis(molecule, axis_start, axis_direction, angle)
        collisions = detector.detect_collisions(rotated, existing_structure,
                                              heavy_atoms_only=heavy_atoms_only)
        if not collisions:
            return rotated, True

    return molecule, False


def resolve_collisions_nudge(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    max_attempts: int = 25,
    nudge_step: float = 0.2,
    min_distance: float = 0.8,
    heavy_atoms_only: bool = True,
) -> Tuple[Atoms, bool]:
    """Resolve collisions by small XY translations."""
    detector = CollisionDetector(cell=cell)
    initial_collisions = detector.detect_collisions(molecule, existing_structure,
                                                  heavy_atoms_only=heavy_atoms_only)
    if not initial_collisions:
        return molecule, True

    nudges = []
    for layer in range(1, int(np.sqrt(max_attempts)) + 1):
        for dx in [-layer, 0, layer]:
            for dy in [-layer, 0, layer]:
                if dx == 0 and dy == 0:
                    continue
                nudges.append((dx * nudge_step, dy * nudge_step))
                if len(nudges) >= max_attempts:
                    break
            if len(nudges) >= max_attempts:
                break
        if len(nudges) >= max_attempts:
            break

    for dx, dy in nudges:
        nudged = _nudge_molecule_xy(molecule, dx, dy)
        collisions = detector.detect_collisions(nudged, existing_structure,
                                              heavy_atoms_only=heavy_atoms_only)
        if not collisions:
            return nudged, True

    return molecule, False


def resolve_collisions_optimize(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    max_steps: int = 50,
    force_threshold: float = 0.1,
    min_distance: float = 0.8,
    heavy_atoms_only: bool = True,
) -> Tuple[Atoms, bool]:
    """Resolve collisions using geometry optimization."""
    detector = CollisionDetector(cell=cell)
    initial_collisions = detector.detect_collisions(molecule, existing_structure,
                                                  heavy_atoms_only=heavy_atoms_only)
    if not initial_collisions:
        return molecule, True

    combined = existing_structure.copy()
    molecule_start_idx = len(combined)
    for atom in molecule:
        combined.append(atom)

    fix_indices = list(range(molecule_start_idx))
    constraints = [FixAtoms(indices=fix_indices)]
    combined.set_constraint(constraints)

    from ase.calculators.lj import LennardJones
    calc = LennardJones()
    combined.set_calculator(calc)

    optimizer = FIRE(combined)
    optimizer.run(fmax=force_threshold, steps=max_steps)

    optimized_molecule = combined[molecule_start_idx:]
    final_collisions = detector.detect_collisions(optimized_molecule, existing_structure,
                                                heavy_atoms_only=heavy_atoms_only)
    return optimized_molecule, len(final_collisions) == 0


def resolve_collisions_reject(
    molecule: Atoms,
    existing_structure: Atoms,
    cell: Optional[np.ndarray] = None,
    min_distance: float = 0.8,
    heavy_atoms_only: bool = True,
) -> Tuple[Atoms, bool]:
    """Reject placement if collisions detected."""
    detector = CollisionDetector(cell=cell)
    collisions = detector.detect_collisions(molecule, existing_structure,
                                          heavy_atoms_only=heavy_atoms_only)

    if collisions:
        warnings.warn(
            f"Collision detected: {len(collisions)} overlaps. "
            f"Closest: {min(c.distance for c in collisions):.3f} Å "
            f"(required: {min(c.min_distance for c in collisions):.3f} Å)."
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
    """Resolve collisions using specified strategy."""
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
