"""
Template loader and cell builders for perovskite structures.

Loads geometric templates from JSON files and provides helpers to expand them
into cartesian position matrices. Supports general triclinic lattices.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np


def available_templates() -> List[str]:
    """Return list of template names based on JSON files in data/."""
    template_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(template_dir, "data")
    if not os.path.exists(data_dir):
        return []

    names: List[str] = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".json"):
            names.append(os.path.splitext(fname)[0])
    return sorted(names)


def load_template(
    template_name: str,
    BX_dist: float = 3.0,
    jahn_teller_dist: float = 1.0,
) -> Dict[str, object]:
    """
    Load template positions and lattice parameters from JSON file.

    Parameters
    ----------
    template_name : str
        Name of the template (e.g., 'cubic', 'orthorhombic')
    BX_dist : float
        Base bond distance used to scale the multipliers.
    jahn_teller_dist : float
        Factor to scale the c-axis length (for cubic-derived distortions).

    Returns
    -------
    dict
        {
            "positions": { "A": [...], ... },
            "lattice_lengths": [a, b, c],
            "angles": [alpha, beta, gamma]
        }
    """
    template_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(template_dir, 'data', f'{template_name}.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    _validate_template_data(data, template_name, json_path)

    # Deep copy positions
    positions = {k: [list(p) for p in v] for k, v in data['positions'].items()}

    multipliers = data.get("lattice_multipliers", [2.0, 2.0, 2.0])

    # Apply JT distortion to the c-axis if requested (usually for cubic templates)
    if template_name == "cubic" and len(multipliers) == 3:
        multipliers = [
            multipliers[0],
            multipliers[1],
            multipliers[2] * jahn_teller_dist,
        ]

    lattice_lengths = [float(m) * BX_dist for m in multipliers]
    # Load Angles (Default to 90 if not present)
    angles = data.get("angles", [90.0, 90.0, 90.0])

    return {
        "positions": positions,
        "lattice_lengths": lattice_lengths,
        "angles": angles,
    }


def _validate_template_data(data: Dict, template_name: str, json_path: str) -> None:
    """Validate template JSON structure."""
    if not isinstance(data, dict):
        raise ValueError(f"Template {template_name} must be a JSON object.")

    required_keys = {"name", "positions"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Template {template_name} missing keys {missing}.")

    pos = data.get("positions", {})
    if not isinstance(pos, dict):
        raise ValueError(f"'positions' must be an object.")

    for site in ("A", "B", "X"):
        if site not in pos or not isinstance(pos[site], list):
            raise ValueError(f"'positions.{site}' must be a list of [x,y,z].")
        for entry in pos[site]:
            if not (isinstance(entry, list) and len(entry) == 3 and all(isinstance(v, (int, float)) for v in entry)):
                raise ValueError(f"'positions.{site}' entries must be [x,y,z] numbers.")

    if "lattice_multipliers" in data:
        lm = data["lattice_multipliers"]
        if not (isinstance(lm, list) and len(lm) == 3 and all(isinstance(v, (int, float)) for v in lm)):
            raise ValueError(f"'lattice_multipliers' must be [ax, ay, az] numbers.")

    if "angles" in data:
        if not (len(data["angles"]) == 3 and all(isinstance(a, (int, float)) for a in data["angles"])):
            raise ValueError("Angles must be a list of 3 floats.")


# -----------------------------------------------------------------------------
# Lattice math helpers
# -----------------------------------------------------------------------------

def cell_matrix_from_parameters(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """
    Convert lattice lengths and angles (in degrees) to a 3x3 Cartesian matrix.

    Aligns c with Z, b in YZ plane.
    """
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

    # Clamp cosine term to avoid floating-point drift
    val = (np.cos(alpha_r) * np.cos(beta_r) - np.cos(gamma_r)) / (
        np.sin(alpha_r) * np.sin(beta_r)
    )
    val = max(-1.0, min(1.0, val))

    vol_factor = np.sqrt(
        1
        - np.cos(alpha_r) ** 2
        - np.cos(beta_r) ** 2
        - np.cos(gamma_r) ** 2
        + 2 * np.cos(alpha_r) * np.cos(beta_r) * np.cos(gamma_r)
    )

    # Vector c along z
    v_c = [0.0, 0.0, c]
    # Vector b in the YZ plane
    v_b = [0.0, b * np.sin(alpha_r), b * np.cos(alpha_r)]
    # General vector a
    v_a_x = a * vol_factor / np.sin(alpha_r)
    v_a_y = a * (np.cos(gamma_r) - np.cos(beta_r) * np.cos(alpha_r)) / np.sin(alpha_r)
    v_a_z = a * np.cos(beta_r)
    v_a = [v_a_x, v_a_y, v_a_z]

    # Return as rows: matrix[0] is vector a
    return np.array([v_a, v_b, v_c])


# -----------------------------------------------------------------------------
# Position expansion helpers
# -----------------------------------------------------------------------------
PositionMatrix = Dict[str, List[List[float]]]


def _add_terminal_sites(position_matrix: PositionMatrix, cell_matrix: np.ndarray) -> PositionMatrix:
    """
    Duplicate atoms that sit at the bottom face (z≈0) to the top face.

    Uses the c-vector (cell_matrix[2]) for translation.
    """
    out: PositionMatrix = {k: [list(p) for p in v] for k, v in position_matrix.items()}
    c_vec = cell_matrix[2]  # The c-vector

    for site in ("A", "B", "X"):
        site_positions = out.get(site, [])
        terminals = []
        for p in site_positions:
            if abs(p[2]) < 1e-6:
                new_pos = np.array(p) + c_vec
                terminals.append(new_pos.tolist())
        site_positions.extend(terminals)
        out[site] = site_positions
    return out


def _ensure_site_keys(position_matrix: PositionMatrix) -> PositionMatrix:
    """Copy a position matrix and guarantee A/B/X keys exist."""
    sites: PositionMatrix = {"A": [], "B": [], "X": []}
    for key, vals in position_matrix.items():
        sites[key] = [list(pos) for pos in vals]
    return sites


def _expand_supercell(
    position_matrix: PositionMatrix,
    unit_cell_matrix: np.ndarray,
    supercell: Tuple[int, int, int],
) -> PositionMatrix:
    """
    Translate fractional positions to cartesian and tile to the supercell.

    Handles non-orthogonal lattices via matrix multiplication.
    """
    nx, ny, nz = supercell
    expanded: PositionMatrix = {k: [] for k in position_matrix}

    # Grid of integer offsets
    grid = np.indices((nx, ny, nz)).reshape(3, -1).T  # (N_cells, 3)
    cartesian_shifts = grid @ unit_cell_matrix  # (N_cells, 3)

    for site, frac_coords_list in position_matrix.items():
        if not frac_coords_list:
            continue

        frac_coords = np.array(frac_coords_list, dtype=float)
        base_cartesian = frac_coords @ unit_cell_matrix
        all_positions = (base_cartesian[:, None, :] + cartesian_shifts[None, :, :]).reshape(-1, 3)

        expanded[site] = all_positions.tolist()

    return expanded


# -----------------------------------------------------------------------------
# Cell builders
# -----------------------------------------------------------------------------

def build_bulk_cell(
    template_data: Dict[str, object],
    supercell: Tuple[int, int, int] = (1, 1, 1),
    include_terminals: bool = False,
) -> Dict[str, object]:
    """
    Create a bulk position matrix from loaded template data.

    Parameters
    ----------
    template_data : dict
        Output from load_template()
    """
    frac_positions = template_data["positions"]
    lengths = template_data["lattice_lengths"]
    angles = template_data["angles"]

    unit_cell_matrix = cell_matrix_from_parameters(*lengths, *angles)
    base_positions = _ensure_site_keys(frac_positions)
    cartesian_positions = _expand_supercell(base_positions, unit_cell_matrix, supercell)

    supercell_matrix = unit_cell_matrix * np.array(supercell)[:, None]

    if include_terminals:
        cartesian_positions = _add_terminal_sites(cartesian_positions, supercell_matrix)

    return {
        "positions": cartesian_positions,
        "unit_cell_matrix": unit_cell_matrix,
        "supercell_matrix": supercell_matrix,
    }


# -----------------------------------------------------------------------------
# Structure matrix helpers
# -----------------------------------------------------------------------------


@dataclass
class QBuilderOutput:
    positions: Dict[str, np.ndarray]
    lattice_vector_sizes: np.ndarray
    cell_vectors: np.ndarray


def calculate_lattice_vectors(BX_dist: float) -> np.ndarray:
    """Return cubic lattice vectors (lengths) from a BX bond distance."""
    return np.array([2 * BX_dist, 2 * BX_dist, 2 * BX_dist], dtype=float)


def build_structure_matrix(
    positions: Dict[str, np.ndarray],
    lattice_vector_sizes: np.ndarray,
    cell_vectors: Optional[np.ndarray] = None,
) -> QBuilderOutput:
    """
    Create a QBuilderOutput from positions and lattice vectors.

    Positions are expected in cartesian coordinates.
    """
    if cell_vectors is None:
        cell_vectors = np.array(
            [
                [lattice_vector_sizes[0], 0.0, 0.0],
                [0.0, lattice_vector_sizes[1], 0.0],
                [0.0, 0.0, lattice_vector_sizes[2]],
            ]
        )
    else:
        cell_vectors = np.asarray(cell_vectors, dtype=float)

    return QBuilderOutput(
        positions=positions,
        lattice_vector_sizes=lattice_vector_sizes,
        cell_vectors=cell_vectors,
    )


