"""
Pipeline entry points for building perovskite structures.

This module wires templates -> base cell builder -> population to produce
ASE Atoms objects. Currently only bulk is implemented here.
"""

from typing import List, Tuple, Optional, Dict

import numpy as np
from ase import Atoms

from q2D_Materials.builders.base_cell_builder import build_bulk_cell, build_slab_cell
from q2D_Materials.builders.population import populate_structure
from q2D_Materials.builders.q_builder import build_structure_matrix
from q2D_Materials.builders.glazer_tilting import apply_glazer_tilt
from q2D_Materials.templates.templates import CubicTemplate, ReducedTemplate, Template
from q2D_Materials.utils.common_a_sites import calculate_BX_distance


def get_template(template_name: str) -> Template:
    """Return a template instance by name ('cubic' or 'reduced')."""
    name = template_name.lower()
    if name == "cubic":
        return CubicTemplate()
    if name == "reduced":
        return ReducedTemplate()
    raise ValueError("template must be 'cubic' or 'reduced'")


def auto_calculate_BX_distance(B: str, X: str) -> float:
    """Calculate BX distance from ionic radii data with a small fallback."""
    try:
        return calculate_BX_distance(B, X)
    except ValueError:
        return 2.0


def _build_positions(
    tpl: Template, supercell_size: Tuple[int, int, int], BX_dist: float
) -> Dict[str, List[List[float]]]:
    """Select the correct builder for a template and return positions/lattice/cell."""
    if getattr(tpl, "dim", "3D").upper() == "3D":
        return build_bulk_cell(tpl, BX_dist=BX_dist, supercell=supercell_size)
    if len(supercell_size) != 3:
        raise ValueError("supercell_size must be a 3-tuple (nx, ny, nz)")
    nx, ny, nz = supercell_size
    return build_slab_cell(tpl, BX_dist=BX_dist, n_layers=nz, supercell=(nx, ny))


def _apply_glazer_if_any(
    positions: Dict[str, List[List[float]]],
    lattice_vectors: Tuple[float, float, float],
    supercell_size: Tuple[int, int, int],
    glazer_angles: Optional[List[float]],
    glazer_pattern: Optional[List[str]],
) -> Tuple[Dict[str, List[List[float]]], np.ndarray, np.ndarray]:
    """Apply Glazer tilt if requested, otherwise return inputs unchanged."""
    lattice_vec_sizes = np.asarray(lattice_vectors, dtype=float)
    cell_lengths_arr = lattice_vec_sizes * np.asarray(supercell_size, dtype=float)

    if glazer_angles is None or glazer_pattern is None:
        return positions, lattice_vec_sizes, cell_lengths_arr

    pos_out, lv_unit, cell_lengths = apply_glazer_tilt(
        position_matrix=positions,
        lattice_vectors=tuple(lattice_vec_sizes.tolist()),
        supercell=supercell_size,
        angles=list(glazer_angles),
        tilt_pattern=list(glazer_pattern),
        adjust_cell=True,
    )
    return pos_out, np.asarray(lv_unit, dtype=float), np.asarray(cell_lengths, dtype=float)


def _populate(
    positions: Dict[str, List[List[float]]],
    lattice_vec_sizes: np.ndarray,
    cell_lengths: np.ndarray,
    A,
    B,
    X,
) -> Atoms:
    """Populate ions using the population toolchain."""
    positions_np = {site: np.asarray(coords, dtype=float) for site, coords in positions.items()}
    matrix = build_structure_matrix(positions_np, lattice_vec_sizes)
    matrix.cell_vectors = np.array(
        [
            [cell_lengths[0], 0.0, 0.0],
            [0.0, cell_lengths[1], 0.0],
            [0.0, 0.0, cell_lengths[2]],
        ]
    )
    return populate_structure(
        matrix=matrix,
        A_ions=A,
        B_ions=B,
        X_ions=X,
    )


def create_bulk_perovskite(
    A,
    B,
    X,
    supercell_size: Tuple[int, int, int],
    BX_dist: float = None,
    template: str = "cubic",
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str]] = None,
) -> Atoms:
    """
    Build a bulk perovskite using a chosen geometry template and populate it.
    """
    if BX_dist is None:
        B_first = B[0] if isinstance(B, (list, tuple)) else B
        X_first = X[0] if isinstance(X, (list, tuple)) else X
        BX_dist = auto_calculate_BX_distance(B_first, X_first)

    tpl = get_template(template)
    cell_data = _build_positions(tpl, supercell_size, BX_dist)
    positions = cell_data["positions"]

    positions, lattice_vec_sizes, cell_lengths_arr = _apply_glazer_if_any(
        positions,
        cell_data["lattice_vectors"],
        supercell_size,
        glazer_angles,
        glazer_pattern,
    )

    return _populate(positions, lattice_vec_sizes, cell_lengths_arr, A, B, X)


def create_perovskite(structure_type="bulk", **kwargs):
    """Dispatch to the appropriate creator based on structure_type."""
    stype = structure_type.lower()
    if stype == "bulk":
        return create_bulk_perovskite(**kwargs)
    raise NotImplementedError(f"{structure_type} creation is not implemented yet.")
