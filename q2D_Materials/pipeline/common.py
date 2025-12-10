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


def calculate_max_dj_spacer_nn_distance(dj_spacer: Optional[List]) -> Optional[float]:
    """
    Calculate the maximum N-N distance from dj_spacer molecules.
    
    Parameters
    ----------
    dj_spacer : Optional[List]
        List of double spacer molecules (Atoms objects or strings)
        
    Returns
    -------
    Optional[float]
        Maximum N-N distance in Angstroms, or None if no valid molecules found.
        Returns None if dj_spacer is None or empty, or if all are atomic spacers.
    """
    if dj_spacer is None or len(dj_spacer) == 0:
        return None
    
    from q2D_Materials.builders.spacer import calculate_double_spacer_nh3_distance
    from q2D_Materials.pipeline.common import normalize_spacer
    
    max_nn_distance = 0.0
    has_molecular_spacer = False
    
    for ds in dj_spacer:
        if isinstance(ds, str):
            # Normalize string to Atoms
            try:
                ds_atoms = normalize_spacer(ds)
            except:
                continue
        elif isinstance(ds, Atoms):
            ds_atoms = ds
        else:
            continue
        
        # Check if atomic (single atom)
        if len(ds_atoms) == 1:
            continue  # Skip atomic spacers for N-N distance calculation
        
        has_molecular_spacer = True
        nn_dist = calculate_double_spacer_nh3_distance(ds_atoms)
        if nn_dist > max_nn_distance:
            max_nn_distance = nn_dist
    
    if has_molecular_spacer and max_nn_distance > 0:
        return max_nn_distance
    return None


def build_cell_positions(
    template_name: str,
    BX_dist: float,
    jahn_teller_dist: float,
    layer_sequence: Optional[List[str] | str] = None,
    xy_expansion: Tuple[int, int] = (1, 1),
    penetration: float = 0.0,
    spacer_provided: bool = False,
    attachment_end: Optional[str] = None,
    dj_spacer_nn_distance: Optional[float] = None,
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
        dj_spacer_nn_distance=dj_spacer_nn_distance,
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
    A,
    B,
    X,
    Ap_ions=None,
    dj_spacer=None,
    site_labels=None,
) -> Atoms:
    """Populate ions using the population toolchain."""
    positions_np = {site: np.asarray(coords, dtype=float) for site, coords in positions.items()}
    matrix = build_structure_matrix(positions_np, lattice_vec_sizes, cell_vectors=cell_matrix)
    return populate_structure(
        matrix=matrix,
        A_ions=A,
        B_ions=B,
        X_ions=X,
        Ap_ions=Ap_ions,
        dj_spacer=dj_spacer,
        site_labels=site_labels,
    )


def default_layer_sequence(layer_sequence: Optional[str | List[str]], thickness: int, structure_type: str = "monolayer") -> str:
    """Return the default layer sequence for a given thickness if not provided."""
    if layer_sequence is None:
        if structure_type.lower() == "bulk":
            return "-".join(["L1-L2"] * thickness) + "-L1"
        else:
            return "-".join(["L1-L2"] * thickness) + "-L1"
    elif isinstance(layer_sequence, str) and layer_sequence.upper() == "DJ":
        # DJ keyword: (L1-L2) * thickness + "-M1-M2" for bulk
        if structure_type.lower() == "bulk" and thickness > 1:
            base_sequence = "-".join(["L1-L2"] * thickness)
            return f"{base_sequence}-M1-M1"
        elif structure_type.lower() == "bulk" and thickness == 1:
            return "L2-M1-M1"
        else:
            # For monolayer, just use the base sequence
            return "-".join(["L1-L2"] * thickness) + "-L1"
    else:
        return layer_sequence

