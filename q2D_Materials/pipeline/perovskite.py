"""
Pipeline entry points for building perovskite structures.

This module wires templates -> base cell builder -> population to produce
ASE Atoms objects. Currently only bulk is implemented here.
"""

from typing import List, Tuple, Optional

import numpy as np
from ase import Atoms
from ase.build import add_vacuum

from q2D_Materials.pipeline.common import (
    build_cell_positions,
    default_layer_sequence,
    get_template,
    normalize_spacer,
    populate_positions,
    resolve_BX_distance,
)


def create_bulk_perovskite(
    A: str | List[str] | Atoms,
    B: str | List[str] | Atoms,
    X: str | List[str] | Atoms,
    xy_expansion: Tuple[int, int] = (1, 1),
    BX_dist: float = None,
    template: str = "cubic",
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str]] = None,
    jahn_teller_dist: float = 1.0,
    thickness: int = 1,
    layer_sequence: Optional[List[str] | str] = None,
    dj_spacer: Optional[List[str | Atoms]] = None,
) -> Atoms:
    """
    Build a bulk perovskite using a chosen geometry template and populate it.
    Applies XY expansion within the layer plane.
    """
    BX_dist = resolve_BX_distance(B, X, BX_dist)
    layer_sequence = default_layer_sequence(layer_sequence, thickness, structure_type="bulk")
    
    # Normalize dj_spacer early to calculate N-N distances
    dj_spacer_normalized = None
    dj_spacer_nn_distance = None
    if dj_spacer is not None:
        dj_spacer_normalized = []
        for ds in dj_spacer:
            if isinstance(ds, Atoms):
                dj_spacer_normalized.append(ds.copy())
            else:
                dj_spacer_normalized.append(normalize_spacer(ds))
        
        # Calculate maximum N-N distance from dj_spacer molecules
        from q2D_Materials.pipeline.common import calculate_max_dj_spacer_nn_distance
        dj_spacer_nn_distance = calculate_max_dj_spacer_nn_distance(dj_spacer_normalized)
    
    tpl = get_template(template)
    cell_data = build_cell_positions(
        tpl,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        layer_sequence=layer_sequence,
        xy_expansion=xy_expansion,
        glazer_angles=glazer_angles,
        glazer_pattern=glazer_pattern,
        dj_spacer_nn_distance=dj_spacer_nn_distance,
    )
    positions = cell_data["positions"]
    unit_cell_matrix = cell_data["unit_cell_matrix"]
    lattice_vec_sizes = np.linalg.norm(unit_cell_matrix, axis=1)
    site_labels = cell_data.get("site_labels", {})
    
    # Normalize dj_spacer for population
    dj_spacer_for_populate = None
    if dj_spacer_normalized is not None:
        dj_spacer_for_populate = dj_spacer_normalized
    
    atoms = populate_positions(positions, lattice_vec_sizes, unit_cell_matrix, A, B, X, dj_spacer=dj_spacer_for_populate, site_labels=site_labels)
    
    return atoms

def create_monolayer_perovskite(
    A: str | List[str] | Atoms,
    B: str | List[str] | Atoms,
    X: str | List[str] | Atoms,
    xy_expansion: Tuple[int, int] = (1, 1),
    BX_dist: float = None,
    template: str = "cubic",
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str]] = None,
    jahn_teller_dist: float = 1.0,
    thickness: int = 1,
    vacuum: float = 10.0,
    layer_sequence: Optional[List[str] | str] = None,
    spacer: str | List[str] | Atoms = None,
    penetration: float = 0.0,
    attachment_end: Optional[str] = None,
    dj_spacer: Optional[List[str | Atoms]] = None,
) -> Atoms:
    """
    Build a monolayer perovskite using a chosen geometry template and populate it.
    Replace the X of terminal positions with the X of the spacer.
    Applies XY expansion within the layer plane.
    """
    BX_dist = resolve_BX_distance(B, X, BX_dist)
    layer_sequence = default_layer_sequence(layer_sequence, thickness, structure_type="monolayer")

    # Normalize dj_spacer early to calculate N-N distances
    dj_spacer_normalized = None
    dj_spacer_nn_distance = None
    if dj_spacer is not None:
        dj_spacer_normalized = []
        for ds in dj_spacer:
            if isinstance(ds, Atoms):
                dj_spacer_normalized.append(ds.copy())
            else:
                dj_spacer_normalized.append(normalize_spacer(ds))
        
        # Calculate maximum N-N distance from dj_spacer molecules
        from q2D_Materials.pipeline.common import calculate_max_dj_spacer_nn_distance
        dj_spacer_nn_distance = calculate_max_dj_spacer_nn_distance(dj_spacer_normalized)

    tpl = get_template(template)
    cell_data = build_cell_positions(
        tpl,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        layer_sequence=layer_sequence,
        xy_expansion=xy_expansion,
        penetration=penetration,
        spacer_provided=spacer is not None,
        attachment_end=attachment_end,
        glazer_angles=glazer_angles,
        glazer_pattern=glazer_pattern,
        dj_spacer_nn_distance=dj_spacer_nn_distance,
    )
    positions = cell_data["positions"]
    site_labels = cell_data.get("site_labels", {})
    unit_cell_matrix = cell_data["unit_cell_matrix"]
    lattice_vec_sizes = np.linalg.norm(unit_cell_matrix, axis=1)

    # Normalize spacer (Ap_ions) - spacers can be SMILES strings or abbreviations
    # "HOLE" string in lists or as single value creates holes (unpopulated Ap positions)
    Ap_ions = None
    if spacer is not None:
        if isinstance(spacer, list):
            Ap_ions = []
            for s in spacer:
                if isinstance(s, str) and s.upper() == "HOLE":
                    Ap_ions.append("HOLE")  # Keep "HOLE" to create unpopulated positions
                else:
                    Ap_ions.append(normalize_spacer(s))
        else:
            # Check if single spacer value is "HOLE"
            if isinstance(spacer, str) and spacer.upper() == "HOLE":
                Ap_ions = "HOLE"
            else:
                Ap_ions = normalize_spacer(spacer)

    # dj_spacer_normalized already calculated above

    atoms = populate_positions(positions, lattice_vec_sizes, unit_cell_matrix, A, B, X, Ap_ions, dj_spacer=dj_spacer_normalized, site_labels=site_labels)

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
    if stype == "bilayer":
        return create_bilayer_perovskite(**kwargs)
    raise NotImplementedError(f"{structure_type} creation is not implemented yet.")
