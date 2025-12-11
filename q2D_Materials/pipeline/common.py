"""
Shared helpers for perovskite pipeline entry points.

This module centralizes reusable steps such as template selection,
Glazer tilting, BX distance resolution, spacer normalization, and population.
"""

from typing import List, Tuple, Optional, Dict

import numpy as np
from ase import Atoms

from q2D_Materials.builders.templates import (
    available_templates,
    build_floor_schema,
    build_structure_matrix,
    flatten_floor_schema,
)
from q2D_Materials.builders.populate import populate_structure
from q2D_Materials.utils.A_sites import (
    calculate_BX_distance,
    get_a_site_object,
)
from q2D_Materials.utils.molecule_builder import smiles_to_ase_atoms


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


def resolve_BX_distance(B, X, BX_dist: float | None) -> float:
    """
    Return the BX distance, computing it when not provided.
    Accepts either single species or lists/tuples.
    """
    if BX_dist is not None:
        return BX_dist
    B_first = B[0] if isinstance(B, (list, tuple)) else B
    X_first = X[0] if isinstance(X, (list, tuple)) else X
    return auto_calculate_BX_distance(B_first, X_first)


def normalize_spacer(spacer):
    """
    Normalize spacer input to handle strings (SMILES or abbreviations) and Atoms objects.
    
    Supports both atomic spacers (e.g., "Cs", "Rb", "K") and molecular spacers (SMILES strings).

    Parameters
    ----------
    spacer : str or Atoms
        Spacer molecule as SMILES string, abbreviation, or Atoms object

    Returns
    -------
    Atoms
        ASE Atoms object of the spacer molecule or atom
    """
    if isinstance(spacer, Atoms):
        return spacer.copy()

    if isinstance(spacer, str):
        try:
            normalized = get_a_site_object(spacer)
            if isinstance(normalized, Atoms):
                return normalized
            if isinstance(normalized, str):
                try:
                    from pymatgen.core.periodic_table import Element

                    try:
                        Element(normalized)
                        return Atoms(normalized, positions=[[0, 0, 0]])
                    except (ValueError, KeyError):
                        pass
                except ImportError:
                    atomic_spacers = ['Cs', 'K', 'Rb', 'Na', 'Li', 'Ca', 'Sr', 'Ba', 'Mg']
                    if normalized in atomic_spacers:
                        return Atoms(normalized, positions=[[0, 0, 0]])
                    if len(normalized) <= 2 and normalized[0].isupper():
                        return Atoms(normalized, positions=[[0, 0, 0]])
        except (ValueError, ImportError):
            pass

        try:
            if smiles_to_ase_atoms is not None:
                return smiles_to_ase_atoms(spacer)
            else:
                raise ValueError("RDKit not available for SMILES processing")
        except Exception as e:
            raise ValueError(f"Failed to parse spacer '{spacer}' as SMILES or abbreviation: {e}")

    raise ValueError(f"Spacer must be a string (SMILES or abbreviation) or Atoms object, got {type(spacer)}")


def _count_nh3_groups(spacer_atoms: Atoms) -> int:
    """Return the number of NH3-like nitrogens (N with 3 nearby H)."""
    symbols = spacer_atoms.get_chemical_symbols()
    positions = spacer_atoms.get_positions()

    n_indices = [i for i, s in enumerate(symbols) if s == "N"]
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]
    if not n_indices or not h_indices:
        return 0

    h_positions = positions[h_indices]
    nh3_count = 0
    nh_bond_cutoff = 1.2

    for n_idx in n_indices:
        n_pos = positions[n_idx]
        distances = np.linalg.norm(h_positions - n_pos, axis=1)
        nearby_h = np.sum(distances < nh_bond_cutoff)
        if nearby_h == 3:
            nh3_count += 1

    return nh3_count


def calculate_max_sharp_spacer_span(sharp_spacer: Optional[List]) -> Optional[float]:
    """
    Return the maximum span required by sharp_spacer molecules.

    For double-NH3 spacers, uses the N–N distance. For mono-NH3 or atomic
    spacers, uses the single-molecule length along its main axis.
    """
    if sharp_spacer is None or len(sharp_spacer) == 0:
        return None

    from q2D_Materials.builders.spacer import calculate_double_spacer_nh3_distance
    from q2D_Materials.pipeline.common import normalize_spacer
    from q2D_Materials.utils.molecule_builder import get_molecule_length, align_ase_molecule_for_perovskite

    max_span = 0.0

    for spacer in sharp_spacer:
        if isinstance(spacer, str):
            try:
                spacer_atoms = normalize_spacer(spacer)
            except Exception:
                continue
        elif isinstance(spacer, Atoms):
            spacer_atoms = spacer
        else:
            continue

        if len(spacer_atoms) == 0:
            continue

        nh3_count = _count_nh3_groups(spacer_atoms)
        if nh3_count >= 2:
            span = calculate_double_spacer_nh3_distance(spacer_atoms)
        else:
            aligned = align_ase_molecule_for_perovskite(spacer_atoms.copy())
            span = get_molecule_length(aligned)

        if span > max_span:
            max_span = span

    return max_span if max_span > 0 else None


def build_cell_positions(
    template_name: str,
    BX_dist: float,
    jahn_teller_dist: float,
    layer_sequence: Optional[List[str] | str] = None,
    xy_expansion: Tuple[int, int] = (1, 1),
    penetration: float = 0.0,
    spacer_provided: bool = False,
    attachment_end: Optional[str] = None,
    sharp_spacer_span: Optional[float] = None,
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Build floor schema then flatten to site-indexed positions.

    Penetration and spacer_provided are retained for signature compatibility
    but no longer alter the template geometry in this refactor.
    """
    schema = build_floor_schema(
        template_name=template_name,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        layer_sequence=layer_sequence,
        xy_expansion=xy_expansion,
        sharp_spacer_nn_distance=sharp_spacer_span,
        glazer_angles=glazer_angles,
        glazer_pattern=glazer_pattern,
    )

    if spacer_provided and attachment_end:
        _apply_attachment_end_ap(schema, attachment_end)

    positions, site_labels = flatten_floor_schema(schema)
    cell_matrix = schema.cell
    lattice_vec_sizes = np.linalg.norm(cell_matrix, axis=1)

    if penetration != 0.0:
        from q2D_Materials.builders.populate import apply_penetration_offsets

        positions = apply_penetration_offsets(positions, penetration, BX_dist=BX_dist)

    return {
        "schema": schema,
        "positions": positions,
        "site_labels": site_labels,
        "unit_cell_matrix": cell_matrix,
        "lattice_vec_sizes": lattice_vec_sizes,
        "floors_cart": list(schema.floors.values()),
    }


def _apply_attachment_end_ap(schema, attachment_end: str) -> None:
    """
    Convert A sites on selected floors to Ap based on attachment_end ('bottom', 'top', 'both').
    """
    if attachment_end is None:
        return

    end = attachment_end.lower()
    keys = list(schema.floors.keys())
    if not keys:
        return

    floors_to_convert: List[str] = []
    if end in ("bottom", "bot"):
        floors_to_convert.append(keys[0])
    elif end == "top":
        floors_to_convert.append(keys[-1])
    elif end == "both":
        floors_to_convert.extend([keys[0], keys[-1]])
    else:
        return

    for fk in floors_to_convert:
        entries = schema.floors.get(fk, [])
        for entry in entries:
            if entry and entry[0] == "A":
                entry[0] = "Ap"


def populate_positions(
    positions: Dict[str, List[List[float]]],
    lattice_vec_sizes: np.ndarray,
    cell_matrix: np.ndarray,
    floors_cart: List[List[List[float]]],
    A,
    B,
    X,
    Ap_ions=None,
    sharp_spacer=None,
    site_labels=None,
) -> Atoms:
    """Populate ions using the population toolchain."""
    positions_np = {site: np.asarray(coords, dtype=float) for site, coords in positions.items()}
    matrix = build_structure_matrix(positions_np, lattice_vec_sizes, cell_vectors=cell_matrix)
    return populate_structure(
        matrix=matrix,
        floors_cart=floors_cart,
        A_ions=A,
        B_ions=B,
        X_ions=X,
        Ap_ions=Ap_ions,
        sharp_spacer=sharp_spacer,
        site_labels=site_labels,
    )


def default_layer_sequence(layer_sequence: Optional[str | List[str]], thickness: int, structure_type: str = "monolayer") -> str:
    """Return the default layer sequence for a given thickness if not provided."""
    if layer_sequence is None:
        if structure_type.lower() == "bulk":
            return "-".join(["L1-L2"] * thickness)
        else:
            return "-".join(["L1-L2"] * thickness)
    elif isinstance(layer_sequence, str) and layer_sequence.upper() == "DJ":
        # DJ keyword: (L1-L2) * thickness + "-M1-M2" for bulk
        if structure_type.lower() == "bulk" and thickness > 1:
            parts = ["L2", "L1"] * thickness
            parts.pop()
            base_sequence = "-".join(parts)
            return f"{base_sequence}-M1-M1"
        elif structure_type.lower() == "bulk" and thickness == 1:
            return "L2-M1-M1"
    elif isinstance(layer_sequence, str) and layer_sequence.upper() == "RP":
        # RP keyword: fixed sequence using RP layers
        return "L2-M1-RP1-RP2-RP1-M1"
    else:
        return layer_sequence
