"""Template loader for stacked floors with cartesian coordinates."""

import json
import os
import re
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from q2D_Materials.builders.glazer_tilting import apply_glazer_tilt, apply_glazer_tilt_from_notation
from q2D_Materials.builders.glazer_notation import resolve_glazer_input


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


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


@dataclass
class FloorSchema:
    floors: "OrderedDict[str, List[List[float]]]"
    floor_names: Dict[str, str]
    cell: np.ndarray
    lattice_lengths: Tuple[float, float, float]
    angles: Tuple[float, float, float]
    xy_expansion: Tuple[int, int]


def build_floor_schema(
    template_name: str | Dict,
    BX_dist: float = 3.0,
    jahn_teller_dist: float = 1.0,
    layer_sequence: Optional[List[str] | str] = None,
    xy_expansion: Tuple[int, int] = (1, 1),
    sharp_spacer_nn_distance: Optional[float] = None,
    glazer_angles: Optional[List[float]] = None,
    glazer_pattern: Optional[List[str] | str] = None,
    lattice_multipliers: Optional[List[float]] = None,
    interlayer_distances: Optional[Dict[int, float]] = None,
) -> FloorSchema:
    """Build a numbered-floor schema with cartesian coordinates."""
    data = _load_template_json(template_name)
    angles = tuple(data.get("angles", [90.0, 90.0, 90.0]))

    layers, layer_names = _resolve_layers(data, layer_sequence)
    z_positions, total_height = _compute_layer_heights(
        layers, layer_names, BX_dist=BX_dist, sharp_spacer_nn_distance=sharp_spacer_nn_distance, interlayer_distances=interlayer_distances
    )

    lattice_lengths = _calculate_lattice_lengths(
        data,
        BX_dist,
        total_height,
        jahn_teller_dist=jahn_teller_dist,
        lattice_multipliers_override=lattice_multipliers,
    )
    cell_matrix = cell_matrix_from_parameters(*lattice_lengths, *angles)

    floors = OrderedDict()
    floor_names: Dict[str, str] = {}

    for idx, (layer, z_abs, lname) in enumerate(zip(layers, z_positions, layer_names)):
        floor_key = str(idx + 1)
        floor_names[floor_key] = lname
        entries: List[List[float]] = []
        for site, x_frac, y_frac in layer:
            frac = np.array([float(x_frac), float(y_frac), z_abs / lattice_lengths[2]], dtype=float)
            cart = frac @ cell_matrix
            entries.append([site, float(cart[0]), float(cart[1]), float(cart[2])])
        floors[floor_key] = entries

    if xy_expansion != (1, 1):
        floors, cell_matrix = _apply_xy_expansion(floors, cell_matrix, xy_expansion)

    if glazer_pattern is not None:
        floors, cell_matrix, lattice_lengths = _apply_glazer_to_floors(
            floors, lattice_lengths, glazer_angles, glazer_pattern,
            xy_expansion=xy_expansion, BX_dist=BX_dist
        )

    return FloorSchema(
        floors=floors,
        floor_names=floor_names,
        cell=cell_matrix,
        lattice_lengths=tuple(lattice_lengths),
        angles=angles,
        xy_expansion=xy_expansion,
    )


def flatten_floor_schema(schema: FloorSchema) -> Tuple[Dict[str, np.ndarray], Dict[str, List[str]]]:
    """Collapse floors into a site-indexed position matrix."""
    positions: Dict[str, List[List[float]]] = {}
    labels: Dict[str, List[str]] = {}

    for floor_entries in schema.floors.values():
        for site, x, y, z in floor_entries:
            positions.setdefault(site, []).append([x, y, z])
            labels.setdefault(site, []).append(site)

    positions_np = {k: np.asarray(v, dtype=float) for k, v in positions.items()}
    return positions_np, labels


# -----------------------------------------------------------------------------
# Lattice math helpers
# -----------------------------------------------------------------------------


def cell_matrix_from_parameters(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> np.ndarray:
    """Convert lattice lengths and angles (in degrees) to a 3x3 Cartesian matrix."""
    alpha_r = np.radians(alpha)
    beta_r = np.radians(beta)
    gamma_r = np.radians(gamma)

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

    v_c = [0.0, 0.0, c]
    v_b = [0.0, b * np.sin(alpha_r), b * np.cos(alpha_r)]
    v_a_x = a * vol_factor / np.sin(alpha_r)
    v_a_y = a * (np.cos(gamma_r) - np.cos(beta_r) * np.cos(alpha_r)) / np.sin(alpha_r)
    v_a_z = a * np.cos(beta_r)
    v_a = [v_a_x, v_a_y, v_a_z]

    return np.array([v_a, v_b, v_c])


# -----------------------------------------------------------------------------
# Helper routines for floor builder
# -----------------------------------------------------------------------------


def _load_template_json(template_name: str | Dict) -> Dict:
    """Load a template from a JSON file or return it directly if it's already a dict/string."""
    if isinstance(template_name, dict):
        return template_name
        
    if isinstance(template_name, str) and template_name.strip().startswith("{"):
        try:
            return json.loads(template_name)
        except json.JSONDecodeError:
            pass
    template_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(template_dir, "data", f"{template_name}.json")
    with open(json_path, "r") as f:
        return json.load(f)


def _normalize_layer_sequence(seq: Optional[List[str] | str]) -> Optional[List[str]]:
    if seq is None:
        return None
    if isinstance(seq, list):
        return [str(s) for s in seq]
    raw = str(seq)
    for sep in ("-", ","):
        raw = raw.replace(sep, " ")
    parts = [part for part in raw.split() if part]
    if len(parts) == 1 and len(parts[0]) > 1:
        return list(parts[0])
    return parts


def _resolve_layers(data: Dict, layer_sequence: Optional[List[str] | str]) -> Tuple[List[List[List[float]]], List[str]]:
    named_layers = data.get("named_layers") or data.get("NamedLayers")
    layers = data.get("layers") or data.get("Layers")

    if named_layers is not None:
        seq = _normalize_layer_sequence(layer_sequence) or data.get("layer_sequence") or list(named_layers.keys())
        resolved_layers = [named_layers[name] for name in seq]
        layer_names = [str(name) for name in seq]
        return resolved_layers, layer_names

    if layers is not None:
        layer_names = _normalize_layer_sequence(layer_sequence) or [f"L{i + 1}" for i in range(len(layers))]
        return layers, layer_names

    raise ValueError("Template must define either 'named_layers' or 'layers'.")


def _layer_has_spacer_sites(layer: List[List[float]]) -> bool:
    """Return True if a layer contains any S# site."""
    for entry in layer:
        if not entry:
            continue
        site = entry[0]
        if isinstance(site, str) and site.startswith("S") and len(site) > 1 and site[1:].isdigit():
            return True
    return False


def _compute_layer_heights(
    layers: List[List[List[float]]],
    layer_names: List[str],
    BX_dist: float,
    sharp_spacer_nn_distance: Optional[float],
    interlayer_distances: Optional[Dict[int, float]] = None,
) -> Tuple[List[float], float]:
    """Return absolute z for each layer start and total c-length."""
    if not layer_names:
        return [], 0.0

    z_positions = [0.0]
    cumulative = 0.0

    for idx in range(1, len(layer_names)):
        prev_layer = layers[idx - 1]
        curr_layer = layers[idx]
        gap = BX_dist
        if interlayer_distances is not None and idx - 1 in interlayer_distances:
            gap = interlayer_distances[idx - 1]
        elif _layer_has_spacer_sites(prev_layer) and _layer_has_spacer_sites(curr_layer):
            gap = sharp_spacer_nn_distance if sharp_spacer_nn_distance is not None else 2.0 * BX_dist
        cumulative += gap
        z_positions.append(cumulative)

    terminal_gap_idx = len(layer_names) - 1
    if interlayer_distances is not None and terminal_gap_idx in interlayer_distances:
        terminal_gap = interlayer_distances[terminal_gap_idx]
    else:
        terminal_gap = BX_dist

    total_height = cumulative + terminal_gap
    return z_positions, total_height


def _is_s_label(site: str) -> bool:
    return isinstance(site, str) and re.fullmatch(r"S\d+", site) is not None


def _parse_s_index(site: str) -> Optional[int]:
    if not _is_s_label(site):
        return None
    try:
        return int(site[1:])
    except ValueError:
        return None


def _max_s_index(floors: "OrderedDict[str, List[List[float]]]"):
    max_idx = 0
    for entries in floors.values():
        for site, *_ in entries:
            if _is_s_label(site):
                idx = _parse_s_index(site)
                if idx is not None:
                    max_idx = max(max_idx, idx)
    return max_idx


def _calculate_lattice_lengths(
    data: Dict,
    BX_dist: float,
    total_height: float,
    jahn_teller_dist: float,
    lattice_multipliers_override: Optional[List[float]] = None,
) -> Tuple[float, float, float]:
    multipliers = lattice_multipliers_override or data.get("lattice_multipliers", [2.0, 2.0, 2.0])
    if len(multipliers) == 2:
        multipliers = [multipliers[0], multipliers[1], 1.0]

    a = float(multipliers[0]) * BX_dist
    b = float(multipliers[1]) * BX_dist
    c = float(total_height) * jahn_teller_dist
    return a, b, c


def _apply_xy_expansion(
    floors: "OrderedDict[str, List[List[float]]]",
    cell_matrix: np.ndarray,
    xy_expansion: Tuple[int, int],
) -> Tuple["OrderedDict[str, List[List[float]]]", np.ndarray]:
    nx, ny = xy_expansion
    expanded: "OrderedDict[str, List[List[float]]]" = OrderedDict()
    a_vec = cell_matrix[0]
    b_vec = cell_matrix[1]
    next_s_idx = _max_s_index(floors) + 1
    s_label_map: Dict[Tuple[int, int, int], str] = {}

    for key, entries in floors.items():
        expanded_entries: List[List[float]] = []
        for site, x, y, z in entries:
            base_vec = np.array([x, y, z], dtype=float)
            base_s_idx = _parse_s_index(site) if _is_s_label(site) else None
            for ix in range(nx):
                for iy in range(ny):
                    shift = a_vec * ix + b_vec * iy
                    vec = base_vec + shift
                    new_site = site
                    if base_s_idx is not None:
                        key_map = (base_s_idx, ix, iy)
                        if key_map not in s_label_map:
                            s_label_map[key_map] = f"S{next_s_idx}"
                            next_s_idx += 1
                        new_site = s_label_map[key_map]
                    expanded_entries.append([new_site, float(vec[0]), float(vec[1]), float(vec[2])])
        expanded[key] = expanded_entries

    expanded_cell = cell_matrix.copy()
    expanded_cell[0] *= nx
    expanded_cell[1] *= ny
    return expanded, expanded_cell


def _apply_glazer_to_floors(
    floors: "OrderedDict[str, List[List[float]]]",
    lattice_lengths: Tuple[float, float, float],
    glazer_angles: Optional[List[float]],
    glazer_pattern: List[str] | str,
    xy_expansion: Tuple[int, int] = (1, 1),
    BX_dist: float = 3.0,
) -> Tuple["OrderedDict[str, List[List[float]]]", np.ndarray, Tuple[float, float, float]]:
    """Apply Glazer tilting to X-sites and return updated floors and cell."""
    # Build position matrix grouped by site
    position_matrix: Dict[str, List[List[float]]] = {}
    for entries in floors.values():
        for site, x, y, z in entries:
            position_matrix.setdefault(site, []).append([x, y, z])
    original_positions = {k: [list(p) for p in v] for k, v in position_matrix.items()}

    if (
        "X" not in position_matrix
        or len(position_matrix["X"]) == 0
        or "B" not in position_matrix
        or len(position_matrix["B"]) == 0
    ):
        a, b, c = lattice_lengths
        nx, ny = xy_expansion
        cell_matrix = cell_matrix_from_parameters(a, b, c, 90.0, 90.0, 90.0)
        cell_matrix[0] *= nx
        cell_matrix[1] *= ny
        return floors, cell_matrix, (a*nx, b*ny, c)

    a_unit = lattice_lengths[0]
    b_unit = lattice_lengths[1]
    c_unit_ref = 2.0 * BX_dist
    total_height = lattice_lengths[2]
    nz = max(1, int(round(total_height / c_unit_ref)))
    c_unit = total_height / nz
    
    unit_lattice_vectors = (a_unit, b_unit, c_unit)
    supercell_dims = (xy_expansion[0], xy_expansion[1], nz)

    if isinstance(glazer_pattern, str):
        notation = resolve_glazer_input(glazer_pattern)
        tilted_positions, lv_unit, cell_lengths = apply_glazer_tilt_from_notation(
            position_matrix,
            lattice_vectors=unit_lattice_vectors,
            supercell=supercell_dims,
            glazer_notation=notation,
            angles=glazer_angles,
            adjust_cell=False,
        )
    else:
        if glazer_angles is None:
             a, b, c = lattice_lengths
             nx, ny = xy_expansion
             cell_matrix = cell_matrix_from_parameters(a, b, c, 90.0, 90.0, 90.0)
             cell_matrix[0] *= nx
             cell_matrix[1] *= ny
             return floors, cell_matrix, (a*nx, b*ny, c)

        tilted_positions, lv_unit, cell_lengths = apply_glazer_tilt(
            position_matrix,
            lattice_vectors=unit_lattice_vectors,
            supercell=supercell_dims,
            angles=glazer_angles,
            tilt_pattern=glazer_pattern,
            adjust_cell=False,
        )

    for site, coords in original_positions.items():
        if site not in tilted_positions:
            tilted_positions[site] = coords

    updated_floors: "OrderedDict[str, List[List[float]]]" = OrderedDict()
    site_iterators: Dict[str, List[List[float]]] = {k: list(v) for k, v in tilted_positions.items()}
    for floor_key, entries in floors.items():
        new_entries: List[List[float]] = []
        for site, _, _, _ in entries:
            if site in site_iterators and site_iterators[site]:
                pos = site_iterators[site].pop(0)
                new_entries.append([site, pos[0], pos[1], pos[2]])
            else:
                orig = original_positions.get(site, [[0.0, 0.0, 0.0]])
                pos = orig[0]
                new_entries.append([site, pos[0], pos[1], pos[2]])
        updated_floors[floor_key] = new_entries
    nx, ny, nz = supercell_dims
    full_lengths = (lv_unit[0] * nx, lv_unit[1] * ny, lv_unit[2] * nz)
    cell_matrix = np.diag(full_lengths)
    
    return updated_floors, cell_matrix, full_lengths


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
    """Create a QBuilderOutput from positions and lattice vectors."""
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

