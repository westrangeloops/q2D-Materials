"""
Pipeline entry points for building perovskite structures.

This module wires templates -> base cell builder -> population to produce
ASE Atoms objects. Currently only bulk is implemented here.
"""

from typing import List, Tuple, Optional, Dict

import numpy as np
from ase import Atoms
from ase.build import add_vacuum

from q2D_Materials.builders.templates import (
    load_template,
    available_templates,
    build_bulk_cell,
    build_structure_matrix,
)
from q2D_Materials.builders.population import populate_structure
from q2D_Materials.builders.molecule_builder import com_to_origin
from q2D_Materials.utils.glazer_tilting import (
    apply_glazer_tilting_matrix,
    glazer_notation_to_pattern,
    calculate_angles_from_glazer_pattern,
)
from q2D_Materials.utils.A_sites import calculate_BX_distance


def get_template(template_name: str) -> str:
    """Return a template name, validating it exists (based on available JSONs)."""
    name = template_name.lower()
    valid = available_templates()
    if name not in valid:
        raise ValueError(f"template must be one of {valid}")
    return name


def auto_calculate_BX_distance(B: str, X: str) -> float:
    """Calculate BX distance from ionic radii data with a small fallback."""
    try:
        return calculate_BX_distance(B, X)
    except ValueError:
        return 2.0


def _build_positions(
    template_name: str,
    thickness: int,
    BX_dist: float,
    jahn_teller_dist: float,
    include_terminals: bool = False,
) -> Dict[str, object]:
    """Select the correct builder for a template and return positions/lattice/cell for unit cell."""
    tpl_data = load_template(
        template_name,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
    )

    return build_bulk_cell(
        tpl_data,
        supercell=(1, 1, 1),
        include_terminals=include_terminals,
    )


def _apply_glazer_if_any(
    positions: Dict[str, List[List[float]]],
    cell_matrix: np.ndarray,
    glazer_angles: Optional[List[float]],
    glazer_pattern: Optional[List[str]],
) -> Tuple[Dict[str, List[List[float]]], np.ndarray, np.ndarray]:
    """Apply Glazer tilt if requested, otherwise return inputs unchanged."""
    lattice_vec_sizes = np.linalg.norm(cell_matrix, axis=1)

    if glazer_angles is None and glazer_pattern is None:
        return positions, lattice_vec_sizes, cell_matrix

    if glazer_pattern is not None and glazer_angles is None:
        glazer_angles = calculate_angles_from_glazer_pattern(list(glazer_pattern))

    pos_out, lv_out = apply_glazer_tilting_matrix(
        position_matrix=positions,
        lattice_vectors=tuple(lattice_vec_sizes.tolist()),
        angles=list(glazer_angles) if glazer_angles is not None else None,
        tilt_pattern=list(glazer_pattern) if glazer_pattern is not None else None,
    )
    new_cell_matrix = np.array(
        [
            [lv_out[0], 0.0, 0.0],
            [0.0, lv_out[1], 0.0],
            [0.0, 0.0, lv_out[2]],
        ]
    )
    return pos_out, np.asarray(lv_out, dtype=float), new_cell_matrix


def _populate(
    positions: Dict[str, List[List[float]]],
    lattice_vec_sizes: np.ndarray,
    cell_matrix: np.ndarray,
    A,
    B,
    X,
) -> Atoms:
    """Populate ions using the population toolchain."""
    positions_np = {site: np.asarray(coords, dtype=float) for site, coords in positions.items()}
    matrix = build_structure_matrix(positions_np, lattice_vec_sizes, cell_vectors=cell_matrix)
    return populate_structure(
        matrix=matrix,
        A_ions=A,
        B_ions=B,
        X_ions=X, # This must be expanded before being passed to the population toolchain in fact all of them is a good idea A, B, X, Ap must be expanded before being passed to the population toolchain
    )


def create_bulk_perovskite(
    A: str | List[str] | Atoms,
    B: str | List[str] | Atoms,
    X: str | List[str] | Atoms,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    BX_dist: float = None,
    template: str = "cubic",
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str]] = None,
    jahn_teller_dist: float = 1.0,
    thickness: int = 1,
) -> Atoms:
    """
    Build a bulk perovskite using a chosen geometry template and populate it.
    Creates a unit cell first, then applies ASE supercell multiplication.
    """
    if BX_dist is None:
        B_first = B[0] if isinstance(B, (list, tuple)) else B
        X_first = X[0] if isinstance(X, (list, tuple)) else X
        BX_dist = auto_calculate_BX_distance(B_first, X_first)

    tpl = get_template(template)
    cell_data = _build_positions(
        tpl, thickness=1, BX_dist=BX_dist, jahn_teller_dist=jahn_teller_dist,
    )
    positions = cell_data["positions"]
    unit_cell_matrix = cell_data["unit_cell_matrix"]

    positions, lattice_vec_sizes, cell_matrix = _apply_glazer_if_any(
        positions,
        unit_cell_matrix,
        glazer_angles,
        glazer_pattern,
    )

    atoms = _populate(positions, lattice_vec_sizes, cell_matrix, A, B, X)
    
    # Apply ASE supercell multiplication
    if supercell != (1, 1, 1):
        atoms = atoms * supercell
    
    return atoms

def create_monolayer_perovskite(
    A: str | List[str] | Atoms,
    B: str | List[str] | Atoms,
    X: str | List[str] | Atoms,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    BX_dist: float = None,
    template: str = "cubic",
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str]] = None,
    jahn_teller_dist: float = 1.0,
    thickness: int = 1,
    vacuum: float = 10.0,
) -> Atoms:
    """
    Build a monolayer perovskite using a chosen geometry template and populate it.
    Replace the X of terminal positions with the X of the spacer.
    Creates a unit cell first, then applies ASE supercell multiplication.
    """
    if BX_dist is None:
        B_first = B[0] if isinstance(B, (list, tuple)) else B
        X_first = X[0] if isinstance(X, (list, tuple)) else X
        BX_dist = auto_calculate_BX_distance(B_first, X_first)

    tpl = get_template(template)
    cell_data = _build_positions(
        tpl,
        thickness=thickness,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        include_terminals=True,
    )
    positions = cell_data["positions"]
    unit_cell_matrix = cell_data["unit_cell_matrix"]

    positions, lattice_vec_sizes, cell_matrix = _apply_glazer_if_any(
        positions,
        unit_cell_matrix,
        glazer_angles,
        glazer_pattern,
    )

    atoms = _populate(positions, lattice_vec_sizes, cell_matrix, A, B, X)
    
    # Apply ASE supercell multiplication
    if supercell != (1, 1, 1):
        atoms = atoms * supercell

    # Add vacuum then center the slab symmetrically around mid-cell in Z
    add_vacuum(atoms, vacuum=vacuum)
    z_min = atoms.positions[:, 2].min()
    z_max = atoms.positions[:, 2].max()
    cell_z = atoms.get_cell().lengths()[2]
    center = 0.5 * (z_min + z_max)
    shift = 0.5 * cell_z - center
    atoms.positions[:, 2] += shift


    return atoms


def create_perovskite(structure_type="bulk", **kwargs):
    """Dispatch to the appropriate creator based on structure_type."""
    stype = structure_type.lower()
    if stype == "bulk":
        return create_bulk_perovskite(**kwargs)
    if stype == "monolayer":
        return create_monolayer_perovskite(**kwargs)
    raise NotImplementedError(f"{structure_type} creation is not implemented yet.")
