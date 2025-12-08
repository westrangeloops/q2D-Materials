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
from q2D_Materials.builders.glazer_tilting import apply_glazer_tilt
from q2D_Materials.builders.population import populate_structure
from q2D_Materials.builders.molecule_builder import com_to_origin
# Glazer tilting functionality removed for minimal implementation
from q2D_Materials.utils.A_sites import calculate_BX_distance
from q2D_Materials.builders.population import normalize_a_site
from q2D_Materials.utils.molecule_builder import smiles_to_ase_atoms
from q2D_Materials.utils.A_sites import get_a_site_object, is_molecular_a_cation


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


def normalize_spacer(spacer):
    """
    Normalize spacer input to handle strings (SMILES or abbreviations) and Atoms objects.

    Parameters
    ----------
    spacer : str or Atoms
        Spacer molecule as SMILES string, abbreviation, or Atoms object

    Returns
    -------
    Atoms
        ASE Atoms object of the spacer molecule
    """
    if isinstance(spacer, Atoms):
        return spacer.copy()

    if isinstance(spacer, str):
        # First try to treat as an A-site cation abbreviation
        try:
            normalized = get_a_site_object(spacer)
            if isinstance(normalized, Atoms):
                return normalized
            # If it's a string (atomic cation), we still need to convert to Atoms
            # For spacers, we expect molecular cations, so treat as SMILES
        except (ValueError, ImportError):
            pass

        # Try to treat as raw SMILES string
        try:
            if smiles_to_ase_atoms is not None:
                return smiles_to_ase_atoms(spacer)
            else:
                raise ValueError("RDKit not available for SMILES processing")
        except Exception as e:
            raise ValueError(f"Failed to parse spacer '{spacer}' as SMILES or abbreviation: {e}")

    raise ValueError(f"Spacer must be a string (SMILES or abbreviation) or Atoms object, got {type(spacer)}")


def _build_positions(
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


def _apply_glazer_if_any(
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

    # Apply Glazer tilting if angles and pattern are provided
    if glazer_angles is not None and glazer_pattern is not None:
        if len(glazer_angles) == 3 and len(glazer_pattern) == 3:
            # Determine supercell size from cell_matrix dimensions
            # The cell_matrix represents the supercell vectors
            supercell_size = (1, 1, 1)  # Default for monolayers

            # Apply Glazer tilting
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


def _populate(
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
        X_ions=X, # This must be expanded before being passed to the population toolchain in fact all of them is a good idea A, B, X, Ap must be expanded before being passed to the population toolchain
        Ap_ions=Ap_ions,
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
) -> Atoms:
    """
    Build a bulk perovskite using a chosen geometry template and populate it.
    Applies XY expansion within the layer plane.
    """
    if BX_dist is None:
        B_first = B[0] if isinstance(B, (list, tuple)) else B
        X_first = X[0] if isinstance(X, (list, tuple)) else X
        BX_dist = auto_calculate_BX_distance(B_first, X_first)

    tpl = get_template(template)
    cell_data = _build_positions(
        tpl,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        layer_sequence=layer_sequence,
        xy_expansion=xy_expansion,
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
) -> Atoms:
    """
    Build a monolayer perovskite using a chosen geometry template and populate it.
    Replace the X of terminal positions with the X of the spacer.
    Applies XY expansion within the layer plane.
    """
    if BX_dist is None:
        B_first = B[0] if isinstance(B, (list, tuple)) else B
        X_first = X[0] if isinstance(X, (list, tuple)) else X
        BX_dist = auto_calculate_BX_distance(B_first, X_first)
    
    if layer_sequence is None:
        layer_sequence = "-".join(["L1-L2"] * thickness) + "-L1"



    tpl = get_template(template)
    cell_data = _build_positions(
        tpl,
        BX_dist=BX_dist,
        jahn_teller_dist=jahn_teller_dist,
        layer_sequence=layer_sequence,
        xy_expansion=xy_expansion,
        penetration=penetration,
        spacer_provided=spacer is not None,
    )
    positions = cell_data["positions"]
    unit_cell_matrix = cell_data["unit_cell_matrix"]

    positions, lattice_vec_sizes, cell_matrix = _apply_glazer_if_any(
        positions,
        unit_cell_matrix,
        glazer_angles,
        glazer_pattern,
    )

    # Normalize spacer (Ap_ions) - spacers can be SMILES strings or abbreviations
    Ap_ions = None
    if spacer is not None:
        if isinstance(spacer, list):
            Ap_ions = [normalize_spacer(s) for s in spacer]
        else:
            Ap_ions = normalize_spacer(spacer)

    atoms = _populate(positions, lattice_vec_sizes, cell_matrix, A, B, X, Ap_ions)

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
