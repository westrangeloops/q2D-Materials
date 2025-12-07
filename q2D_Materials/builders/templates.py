"""
Template loader and cell builders for perovskite structures.

Loads geometric templates from JSON files and provides helpers to expand them
into cartesian position matrices.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np


def available_templates() -> List[str]:
    """Return list of template names based on JSON files in data/."""
    template_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(template_dir, "data")
    names: List[str] = []
    for fname in os.listdir(data_dir):
        if fname.endswith(".json"):
            names.append(os.path.splitext(fname)[0])
    return sorted(names)


def load_template(
    template_name: str,
    BX_dist: float = 3.0,
    jahn_teller_dist: float = 1.0,
    include_terminals: bool = False,
) -> Dict[str, object]:
    """
    Load template positions from JSON file.

    Parameters
    ----------
    template_name : str
        Name of the template ('cubic' or 'reduced')
    BX_dist : float, optional
        B-X bond distance used to scale lattice vectors
    jahn_teller_dist : float, optional
        Jahn-Teller distortion factor (applied to cubic z-axis)
    include_terminal_x : bool, optional
        If True, add terminal X at [0.5, 0.5, 1.0]

    Returns
    -------
    dict
        dict with:
        - positions: position matrix with 'A', 'B', 'X' keys
        - lattice_vectors: tuple[float, float, float]
    """
    template_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(template_dir, 'data', f'{template_name}.json')

    with open(json_path, 'r') as f:
        data = json.load(f)

    _validate_template_data(data, template_name, json_path)

    # Return a deep-ish copy so callers don't mutate the JSON structure
    positions = {k: [list(p) for p in v] for k, v in data['positions'].items()}

    multipliers = data.get("lattice_multipliers", [2.0, 2.0, 2.0])
    # Apply JT distortion to cubic z
    if template_name == "cubic" and len(multipliers) == 3:
        multipliers = [multipliers[0], multipliers[1], multipliers[2] * jahn_teller_dist]
    lattice_vectors = tuple(float(m) * BX_dist for m in multipliers)

    return {
        "positions": positions,
        "lattice_vectors": lattice_vectors,
    }


def _validate_template_data(data: Dict, template_name: str, json_path: str) -> None:
    """Validate template JSON structure."""
    if not isinstance(data, dict):
        raise ValueError(f"Template {template_name} must be a JSON object. See Examples/Templates.MD.")

    required_keys = {"name", "positions"}
    missing = required_keys - set(data.keys())
    if missing:
        raise ValueError(f"Template {template_name} missing keys {missing} in {json_path}. See Examples/Templates.MD.")

    pos = data.get("positions", {})
    if not isinstance(pos, dict):
        raise ValueError(f"'positions' must be an object with A/B/X arrays. See Examples/Templates.MD.")

    for site in ("A", "B", "X"):
        if site not in pos or not isinstance(pos[site], list):
            raise ValueError(f"'positions.{site}' must be a list of [x,y,z]. See Examples/Templates.MD.")
        for entry in pos[site]:
            if not (isinstance(entry, list) and len(entry) == 3 and all(isinstance(v, (int, float)) for v in entry)):
                raise ValueError(f"'positions.{site}' entries must be [x,y,z] numbers. See Examples/Templates.MD.")

    if "lattice_multipliers" in data:
        lm = data["lattice_multipliers"]
        if not (isinstance(lm, list) and len(lm) == 3 and all(isinstance(v, (int, float)) for v in lm)):
            raise ValueError(f"'lattice_multipliers' must be [ax, ay, az] numbers. See Examples/Templates.MD.")


# -----------------------------------------------------------------------------
# Position expansion helpers
# -----------------------------------------------------------------------------
PositionMatrix = Dict[str, List[List[float]]]


def _add_terminal_sites(position_matrix: PositionMatrix, z_top: float) -> PositionMatrix:
    """
    Duplicate atoms that sit at the bottom face (z≈0) to the top face (z=z_top).
    """
    out: PositionMatrix = {k: [list(p) for p in v] for k, v in position_matrix.items()}
    for site in ("A", "B", "X"):
        site_positions = out.get(site, [])
        terminals = [[x, y, z_top] for x, y, z in site_positions if abs(z) < 1e-6]
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
    lattice_vectors: Tuple[float, float, float],
    supercell: Tuple[int, int, int],
) -> PositionMatrix:
    """Translate fractional positions to cartesian and tile to the supercell."""
    lv = np.asarray(lattice_vectors, dtype=float)
    nx, ny, nz = supercell
    expanded: PositionMatrix = {k: [] for k in position_matrix}

    tx = np.arange(nx, dtype=float)
    ty = np.arange(ny, dtype=float)
    tz = np.arange(nz, dtype=float)
    translations = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    translations *= lv

    for site, positions in position_matrix.items():
        if not positions:
            continue
        pos = np.asarray(positions, dtype=float) * lv  # fractional -> cartesian
        tiled = (pos[:, None, :] + translations[None, :, :]).reshape(-1, 3)
        expanded[site] = tiled.tolist()

    return expanded


# -----------------------------------------------------------------------------
# Cell builders
# -----------------------------------------------------------------------------

def build_bulk_cell(
    position_matrix: PositionMatrix,
    BX_dist: float = 3.0,
    supercell: Tuple[int, int, int] = (1, 1, 1),
    jahn_teller_dist: float = 1.0,
    include_terminals: bool = False,
) -> Dict[str, object]:
    """
    Create a bulk position matrix from position data.
    """
    lattice_vectors = (2 * BX_dist, 2 * BX_dist, 2 * BX_dist * jahn_teller_dist)
    base_positions = _ensure_site_keys(position_matrix)
    positions = _expand_supercell(base_positions, lattice_vectors, supercell)

    cell = (
        supercell[0] * lattice_vectors[0],
        supercell[1] * lattice_vectors[1],
        supercell[2] * lattice_vectors[2],
    )

    if include_terminals:
        positions = _add_terminal_sites(positions, z_top=cell[2])

    return {"positions": positions, "cell": cell, "lattice_vectors": lattice_vectors}


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
    positions: Dict[str, np.ndarray], lattice_vector_sizes: np.ndarray
) -> QBuilderOutput:
    """
    Create a QBuilderOutput from positions and lattice vectors.

    Positions are expected in cartesian coordinates.
    """
    cell_vectors = np.array(
        [
            [lattice_vector_sizes[0], 0.0, 0.0],
            [0.0, lattice_vector_sizes[1], 0.0],
            [0.0, 0.0, lattice_vector_sizes[2]],
        ]
    )
    return QBuilderOutput(
        positions=positions,
        lattice_vector_sizes=lattice_vector_sizes,
        cell_vectors=cell_vectors,
    )


