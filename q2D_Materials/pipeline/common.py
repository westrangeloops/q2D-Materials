"""
Shared helpers for perovskite pipeline entry points.

This module centralizes reusable steps such as template selection,
Glazer tilting, BX distance resolution, spacer normalization, and population.
"""

from typing import List, Tuple, Optional, Dict

import numpy as np
from ase import Atoms

from q2D_Materials.builders.templates import (
    load_template,
    available_templates,
    build_bulk_cell,
    build_structure_matrix,
)
from q2D_Materials.builders.glazer_tilting import apply_glazer_tilt
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


def build_cell_positions(
    template_name: str,
    BX_dist: float,
    jahn_teller_dist: float,
    layer_sequence: Optional[List[str] | str] = None,
    xy_expansion: Tuple[int, int] = (1, 1),
    penetration: float = 0.0,
    spacer_provided: bool = False,
) -> Dict[str, object]:
    """Select the correct builder for a template and return positions/lattice/cell for unit cell."""
    tpl_data = load_template(
        template_name,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        layer_sequence=layer_sequence,
        penetration=penetration,
        spacer_provided=spacer_provided,
    )

    return build_bulk_cell(
        tpl_data,
        xy_expansion=xy_expansion,
    )


def apply_glazer_tilting(
    positions: Dict[str, List[List[float]]],
    cell_matrix: np.ndarray,
    glazer_angles: Optional[List[float]],
    glazer_pattern: Optional[List[str]],
) -> Tuple[Dict[str, List[List[float]]], np.ndarray, np.ndarray]:
    """
    Apply Glazer tilt if requested, otherwise return inputs unchanged.

    Glazer tilting applies octahedral rotations to X-sites about their nearest B-site centers,
    following the specified angle and pattern parameters.
    """
    lattice_vec_sizes = np.linalg.norm(cell_matrix, axis=1)

    if glazer_angles is not None and glazer_pattern is not None:
        if len(glazer_angles) == 3 and len(glazer_pattern) == 3:
            supercell_size = (1, 1, 1)

            tilted_positions, _, _ = apply_glazer_tilt(
                position_matrix=positions,
                lattice_vectors=tuple(lattice_vec_sizes),
                supercell=supercell_size,
                angles=glazer_angles,
                tilt_pattern=glazer_pattern,
                adjust_cell=True
            )
            return tilted_positions, lattice_vec_sizes, cell_matrix

    return positions, lattice_vec_sizes, cell_matrix


def populate_positions(
    positions: Dict[str, List[List[float]]],
    lattice_vec_sizes: np.ndarray,
    cell_matrix: np.ndarray,
    A,
    B,
    X,
    Ap_ions=None,
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
    )


def default_layer_sequence(layer_sequence: Optional[str], thickness: int) -> str:
    """Return the default layer sequence for a given monolayer thickness if not provided."""
    if layer_sequence is None:
        return "-".join(["L1-L2"] * thickness) + "-L1"
    else:
        return layer_sequence

