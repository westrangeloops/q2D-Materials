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
    layer_sequence: Optional[List[str] | str] = None,
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
    layer_sequence : list[str] | str, optional
        For named-layer templates, the order to stack. If omitted, template default is used.

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

    positions, n_layers = _build_layer_stack(
        data=data,
        layer_sequence=layer_sequence,
    )

    multipliers = data.get("lattice_multipliers", [2.0, 2.0, 2.0])
    if len(multipliers) == 2:
        multipliers = [multipliers[0], multipliers[1], 1.0]

    # Apply JT distortion to the c-axis if requested (usually for cubic templates)
    if template_name == "cubic" and len(multipliers) == 3:
        multipliers = [
            multipliers[0],
            multipliers[1],
            multipliers[2] * jahn_teller_dist,
        ]

    lattice_lengths = [float(m) * BX_dist for m in multipliers]
    # For layer-based templates, set c from stacked layers (one BX_dist per layer) and JT
    if n_layers is not None:
        lattice_lengths[2] = BX_dist * float(n_layers) * jahn_teller_dist
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

    if "name" not in data:
        raise ValueError(f"Template {template_name} missing required key 'name'.")

    has_layers = "layers" in data or "Layers" in data
    has_named_layers = "named_layers" in data or "NamedLayers" in data
    has_positions = "positions" in data
    if not has_layers and not has_named_layers and not has_positions:
        raise ValueError(f"Template {template_name} must define 'layers' or 'named_layers'. See {json_path}.")

    if has_layers:
        layers = data.get("layers") or data.get("Layers")
        if not isinstance(layers, list) or len(layers) == 0:
            raise ValueError(f"'layers' must be a non-empty list of layer definitions.")
        for li, layer in enumerate(layers):
            if not isinstance(layer, list):
                raise ValueError(f"Layer {li} in template {template_name} must be a list.")
            for entry in layer:
                if not (isinstance(entry, list) and len(entry) == 3):
                    raise ValueError("Each layer entry must be [site, x, y].")
                site, x, y = entry
                if site not in ("A", "B", "X"):
                    raise ValueError("Layer entries must use site labels A, B, or X.")
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    raise ValueError("Layer coordinates must be numbers.")
    elif has_named_layers:
        named_layers = data.get("named_layers") or data.get("NamedLayers")
        if not isinstance(named_layers, dict) or not named_layers:
            raise ValueError("'named_layers' must be a non-empty object of layer_name -> entries.")
        for lname, layer in named_layers.items():
            if not isinstance(layer, list) or len(layer) == 0:
                raise ValueError(f"Named layer '{lname}' must be a non-empty list.")
            for entry in layer:
                if not (isinstance(entry, list) and len(entry) == 3):
                    raise ValueError("Each layer entry must be [site, x, y].")
                site, x, y = entry
                if site not in ("A", "B", "X"):
                    raise ValueError("Layer entries must use site labels A, B, or X.")
                if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                    raise ValueError("Layer coordinates must be numbers.")
    elif has_positions:
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
        if not (isinstance(lm, list) and len(lm) in (2, 3) and all(isinstance(v, (int, float)) for v in lm)):
            raise ValueError(f"'lattice_multipliers' must be [ax, ay] or [ax, ay, az] numbers.")

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


def _layers_to_positions(layers: List[List[List[float]]]) -> PositionMatrix:
    """
    Convert 2D layer definitions into 3D fractional positions stacked along c.

    Each layer entry is [site, x, y]. Layers are evenly spaced along z.
    """
    n_layers = len(layers)
    if n_layers == 0:
        return {"A": [], "B": [], "X": []}

    dz = 1.0 / n_layers
    positions: PositionMatrix = {"A": [], "B": [], "X": []}

    for idx, layer in enumerate(layers):
        z = idx * dz
        for entry in layer:
            site, x, y = entry
            positions[site].append([float(x), float(y), float(z)])

    return positions


def _named_layers_to_layers(named_layers: Dict[str, List[List[float]]], sequence: Optional[List[str]]) -> List[List[List[float]]]:
    """
    Expand named layer definitions into an ordered list of layers.
    If no sequence is provided, the insertion order of the mapping is used.
    """
    if sequence is None:
        return [named_layers[name] for name in named_layers]
    layers: List[List[List[float]]] = []
    for name in sequence:
        if name not in named_layers:
            raise ValueError(f"Layer '{name}' not found in named_layers.")
        layers.append(named_layers[name])
    return layers


def _normalize_layer_sequence(seq: Optional[List[str] | str]) -> Optional[List[str]]:
    """Normalize a user-provided sequence (list or dash/space separated string) to a list."""
    if seq is None:
        return None
    if isinstance(seq, list):
        return [str(s) for s in seq]
    # string: allow separators "-", ",", or whitespace
    raw = str(seq)
    for sep in ("-", ","):
        raw = raw.replace(sep, " ")
    parts = [part for part in raw.split() if part]
    # If user supplied a contiguous string of letters (e.g., "AcBcA"), split into characters
    if len(parts) == 1 and len(parts[0]) > 1:
        return list(parts[0])
    return parts


def _build_layer_stack(
    data: Dict,
    layer_sequence: Optional[List[str] | str],
) -> Tuple[PositionMatrix, Optional[int]]:
    """Resolve layers/named_layers and repeat by thickness."""
    layers = data.get("layers") or data.get("Layers")
    named_layers = data.get("named_layers") or data.get("NamedLayers")

    # Resolve base layers list
    if layers is not None:
        base_layers = layers
    elif named_layers is not None:
        seq = _normalize_layer_sequence(layer_sequence) or data.get("layer_sequence")
        base_layers = _named_layers_to_layers(named_layers, seq)
    else:
        # Fallback for legacy templates that still ship "positions"
        pos = {k: [list(p) for p in v] for k, v in data["positions"].items()}
        return pos, None

    # layer_sequence already includes all repetitions, so use base_layers directly
    stacked_layers = base_layers

    return _layers_to_positions(stacked_layers), len(stacked_layers)


def _expand_xy(
    position_matrix: PositionMatrix,
    unit_cell_matrix: np.ndarray,
    xy_expansion: Tuple[int, int],
) -> PositionMatrix:
    """
    Translate fractional positions to cartesian and tile in XY plane only.

    Creates copies of the layer in X and Y directions, preserving Z coordinates.
    Handles non-orthogonal lattices via matrix multiplication.
    """
    nx, ny = xy_expansion
    expanded: PositionMatrix = {k: [] for k in position_matrix}

    # Grid of integer offsets in XY plane only (Z=0)
    grid = np.indices((nx, ny, 1)).reshape(3, -1).T  # (N_cells, 3) with Z=0
    # Only use X and Y components of the unit cell matrix
    xy_matrix = unit_cell_matrix.copy()
    xy_matrix[2] = 0  # Zero out Z vector so Z expansion doesn't happen
    cartesian_shifts = grid @ xy_matrix  # (N_cells, 3)

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
    xy_expansion: Tuple[int, int] = (1, 1),
) -> Dict[str, object]:
    """
    Create a bulk position matrix from loaded template data.

    Parameters
    ----------
    template_data : dict
        Output from load_template()
    xy_expansion : tuple[int, int]
        Expansion factors in X and Y directions within the layer plane.
    """
    frac_positions = template_data["positions"]
    lengths = template_data["lattice_lengths"]
    angles = template_data["angles"]

    unit_cell_matrix = cell_matrix_from_parameters(*lengths, *angles)
    base_positions = _ensure_site_keys(frac_positions)
    cartesian_positions = _expand_xy(base_positions, unit_cell_matrix, xy_expansion)

    # For XY expansion, the effective cell matrix is expanded only in X and Y
    xy_expanded_matrix = unit_cell_matrix.copy()
    xy_expanded_matrix[0] *= xy_expansion[0]  # Expand X vector
    xy_expanded_matrix[1] *= xy_expansion[1]  # Expand Y vector
    # Z vector remains the same

    return {
        "positions": cartesian_positions,
        "unit_cell_matrix": unit_cell_matrix,
        "expanded_cell_matrix": xy_expanded_matrix,
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


