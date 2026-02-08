"""
Pipeline entry points for building perovskite structures.

This module wires templates -> base cell builder -> population to produce
ASE Atoms objects. Currently only bulk is implemented here.
"""

from typing import List, Tuple, Optional, Dict

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
    B: str | List[str] | Atoms | None,
    X: str | List[str] | Atoms | None,
    xy_expansion: Tuple[int, int] = (1, 1),
    BX_dist: float = None,
    template: str | Dict = "cubic",
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str]] = None,
    jahn_teller_dist: float = 1.0,
    thickness: int = 1,
    layer_sequence: Optional[List[str] | str] = None,
    sharp_spacer: Optional[List[str | Atoms]] = None,
    lattice_multipliers: Optional[List[float]] = None,
    optimizer: str = "KS",
    spacer_orientation: Optional[List[str]] = None,
    collision_strategy: str = "rotate",
) -> Atoms:
    """
    Build a bulk perovskite (or related) structure using a geometry template.
    Applies XY expansion within the layer plane.

    B and X can be omitted for templates that do not define B-sites (e.g. salts);
    in that case a small default BX_dist is used and Glazer tilts are effectively
    disabled by the template logic.
    """
    if B is None or X is None:
        if BX_dist is None:
            BX_dist = 3.0
    else:
        BX_dist = resolve_BX_distance(B, X, BX_dist)
    layer_sequence, interlayer_distances = default_layer_sequence(layer_sequence, thickness, structure_type="bulk")
    
    # Normalize sharp_spacer early to calculate required spans
    sharp_spacer_normalized = None
    sharp_spacer_span = None
    if sharp_spacer is not None:
        sharp_spacer_normalized = []
        for ds in sharp_spacer:
            if isinstance(ds, Atoms):
                sharp_spacer_normalized.append(ds.copy())
            else:
                sharp_spacer_normalized.append(normalize_spacer(ds))

        from q2D_Materials.pipeline.common import calculate_max_sharp_spacer_span

        sharp_spacer_span = calculate_max_sharp_spacer_span(
            sharp_spacer_normalized,
            atomic_span=BX_dist,
        )
    
    tpl = get_template(template)
    cell_data = build_cell_positions(
        tpl,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        layer_sequence=layer_sequence,
        xy_expansion=xy_expansion,
        glazer_angles=glazer_angles,
        glazer_pattern=glazer_pattern,
        sharp_spacer_span=sharp_spacer_span,
        lattice_multipliers=lattice_multipliers,
        interlayer_distances=interlayer_distances,
    )
    positions = cell_data["positions"]
    unit_cell_matrix = cell_data["unit_cell_matrix"]
    lattice_vec_sizes = np.linalg.norm(unit_cell_matrix, axis=1)
    site_labels = cell_data.get("site_labels", {})
    floors_cart = cell_data.get("floors_cart")
    
    # Normalize sharp_spacer for population
    sharp_spacer_for_populate = None
    if sharp_spacer_normalized is not None:
        sharp_spacer_for_populate = sharp_spacer_normalized

    atoms = populate_positions(
        positions,
        lattice_vec_sizes,
        unit_cell_matrix,
        floors_cart,
        A,
        B,
        X,
        sharp_spacer=sharp_spacer_for_populate,
        site_labels=site_labels,
        optimizer=optimizer,
        BX_dist=BX_dist,
        spacer_orientation=spacer_orientation,
        collision_strategy=collision_strategy,
    )

    nx, ny = xy_expansion
    expected_oct = nx * ny * thickness
    b_sym = (B[0] if isinstance(B, list) and B else B) if B else None
    if hasattr(b_sym, "get_chemical_symbols"):
        b_sym = b_sym.get_chemical_symbols()[0]
    n_b = sum(1 for s in atoms.get_chemical_symbols() if s == b_sym) if isinstance(b_sym, str) else 0

    return atoms

def create_monolayer_perovskite(
    A: str | List[str] | Atoms,
    B: str | List[str] | Atoms | None,
    X: str | List[str] | Atoms | None,
    xy_expansion: Tuple[int, int] = (1, 1),
    BX_dist: float = None,
    template: str | Dict = "cubic",
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str]] = None,
    jahn_teller_dist: float = 1.0,
    thickness: int = 1,
    vacuum: float = 10.0,
    layer_sequence: Optional[List[str] | str] = None,
    passivator: str | List[str] | Atoms = None,
    penetration: float = 0.0,
    attachment_end: Optional[str] = None,
    sharp_spacer: Optional[List[str | Atoms]] = None,
    lattice_multipliers: Optional[List[float]] = None,
    optimizer: str = "KS",
    spacer_orientation: Optional[List[str]] = None,
    collision_strategy: str = "rotate",
) -> Atoms:
    """
    Build a monolayer perovskite (or related) structure using a geometry template.
    Replace the X of terminal positions with the X of the passivator.
    Applies XY expansion within the layer plane.

    B and X can be omitted for templates that do not define B-sites (e.g. salts);
    in that case a small default BX_dist is used and Glazer tilts are effectively
    disabled by the template logic.
    """
    if B is None or X is None:
        if BX_dist is None:
            BX_dist = 3.0
    else:
        BX_dist = resolve_BX_distance(B, X, BX_dist)
    layer_sequence, interlayer_distances = default_layer_sequence(layer_sequence, thickness, structure_type="monolayer")

    # Normalize sharp_spacer early to calculate required spans
    sharp_spacer_normalized = None
    sharp_spacer_span = None
    if sharp_spacer is not None:
        sharp_spacer_normalized = []
        for ds in sharp_spacer:
            if isinstance(ds, Atoms):
                sharp_spacer_normalized.append(ds.copy())
            else:
                sharp_spacer_normalized.append(normalize_spacer(ds))

        from q2D_Materials.pipeline.common import calculate_max_sharp_spacer_span

        sharp_spacer_span = calculate_max_sharp_spacer_span(
            sharp_spacer_normalized,
            atomic_span=BX_dist,
        )

    tpl = get_template(template)
    # Default attachment_end to 'both' for monolayers when passivator is provided
    # This ensures Ap-sites are created on both top and bottom surfaces
    effective_attachment_end = attachment_end
    if passivator is not None and attachment_end is None:
        effective_attachment_end = 'both'
    
    cell_data = build_cell_positions(
        tpl,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        layer_sequence=layer_sequence,
        xy_expansion=xy_expansion,
        penetration=penetration,
        spacer_provided=passivator is not None,
        attachment_end=effective_attachment_end,
        glazer_angles=glazer_angles,
        glazer_pattern=glazer_pattern,
        sharp_spacer_span=sharp_spacer_span,
        lattice_multipliers=lattice_multipliers,
        interlayer_distances=interlayer_distances,
    )
    positions = cell_data["positions"]
    site_labels = cell_data.get("site_labels", {})
    unit_cell_matrix = cell_data["unit_cell_matrix"]
    lattice_vec_sizes = np.linalg.norm(unit_cell_matrix, axis=1)
    floors_cart = cell_data.get("floors_cart")

    # Normalize passivator (Ap_ions) - passivators can be SMILES strings or abbreviations
    # "HOLE" string in lists or as single value creates holes (unpopulated Ap positions)
    Ap_ions = None
    if passivator is not None:
        if isinstance(passivator, list):
            Ap_ions = []
            for s in passivator:
                if isinstance(s, str) and s.upper() == "HOLE":
                    Ap_ions.append("HOLE")  # Keep "HOLE" to create unpopulated positions
                else:
                    Ap_ions.append(normalize_spacer(s))
        else:
            # Check if single passivator value is "HOLE"
            if isinstance(passivator, str) and passivator.upper() == "HOLE":
                Ap_ions = "HOLE"
            else:
                Ap_ions = normalize_spacer(passivator)

    atoms = populate_positions(
        positions,
        lattice_vec_sizes,
        unit_cell_matrix,
        floors_cart,
        A,
        B,
        X,
        Ap_ions,
        sharp_spacer=sharp_spacer_normalized,
        site_labels=site_labels,
        optimizer=optimizer,
        BX_dist=BX_dist,
        spacer_orientation=spacer_orientation,
        collision_strategy=collision_strategy,
    )

    if len(atoms) == 0:
        return atoms

    # Always align bottom at Z=0, then add vacuum above if needed
    z_min = atoms.positions[:, 2].min()
    atoms.positions[:, 2] -= z_min  # Shift so bottom is at Z=0
    
    if vacuum > 0.0:
        add_vacuum(atoms, vacuum=vacuum)
        # After adding vacuum, shift up by half vacuum to center
        shift = vacuum / 2.0
        atoms.positions[:, 2] += shift


    return atoms


def create_structure(structure_type="bulk", **kwargs):
    """Dispatch to the appropriate structure creator based on structure_type."""
    stype = structure_type.lower()
    if stype == "bulk":
        return create_bulk_perovskite(**kwargs)
    if stype == "monolayer":
        return create_monolayer_perovskite(**kwargs)
    if stype == "bilayer":
        return create_bilayer_perovskite(**kwargs)
    raise NotImplementedError(f"{structure_type} creation is not implemented yet.")
