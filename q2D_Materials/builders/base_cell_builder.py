"""
Lightweight helpers to expand geometry templates into 3D position matrices.

Each helper takes a template (cubic or reduced) and produces cartesian
positions with real z values so the structure can later be populated with
species. Output positions include A sites and are expanded to a requested
supercell.
"""

from typing import Dict, List, Tuple

import numpy as np

from q2D_Materials.templates.templates import Template


PositionMatrix = Dict[str, List[List[float]]]


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

    # Precompute all translation vectors for the requested supercell
    tx = np.arange(nx, dtype=float)
    ty = np.arange(ny, dtype=float)
    tz = np.arange(nz, dtype=float)
    translations = np.stack(np.meshgrid(tx, ty, tz, indexing="ij"), axis=-1).reshape(
        -1, 3
    )
    translations *= lv  # scale to cartesian

    for site, positions in position_matrix.items():
        if not positions:
            continue
        pos = np.asarray(positions, dtype=float) * lv  # fractional -> cartesian
        # Broadcast positions over all translations, then flatten back to list
        tiled = (pos[:, None, :] + translations[None, :, :]).reshape(-1, 3)
        expanded[site] = tiled.tolist()

    return expanded


def build_bulk_cell(
    template: Template,
    BX_dist: float = 3.0,
    supercell: Tuple[int, int, int] = (1, 1, 1),
) -> Dict[str, object]:
    """
    Create a bulk position matrix from a cubic template.

    Returns a dictionary with cartesian positions and cell lengths.
    """
    lattice_vectors = (2 * BX_dist, 2 * BX_dist, 2 * BX_dist)
    base_positions = _ensure_site_keys(template.get_position_matrix())
    positions = _expand_supercell(base_positions, lattice_vectors, supercell)
    cell = (
        supercell[0] * lattice_vectors[0],
        supercell[1] * lattice_vectors[1],
        supercell[2] * lattice_vectors[2],
    )
    return {"positions": positions, "cell": cell, "lattice_vectors": lattice_vectors}


def build_slab_cell(
    template: Template,
    BX_dist: float = 3.0,
    n_layers: int = 1,
    supercell: Tuple[int, int] = (1, 1),
) -> Dict[str, object]:
    """
    Create a layered 2D slab position matrix from a reduced template.

    In-plane lattice vectors are scaled to keep octahedra square; z is taken
    directly from the template so returned positions already contain layer
    stacking information.
    """
    lv_xy_base = 2 * BX_dist
    lv_xy = np.sqrt(lv_xy_base**2 / 2.0) * 2.0
    lv_z = 2 * BX_dist
    lattice_vectors = (lv_xy, lv_xy, lv_z)

    base_positions = _ensure_site_keys(template.get_position_matrix(n_layers=n_layers))
    positions = _expand_supercell(
        base_positions, lattice_vectors, (supercell[0], supercell[1], 1)
    )

    # Normalize so the lowest atom sits at z=0 for cleaner downstream placement.
    all_z = np.concatenate(
        [np.asarray(plist, dtype=float)[:, 2] for plist in positions.values() if plist]
    )
    min_z = float(all_z.min(initial=0.0))
    if min_z != 0.0:
        for key, plist in positions.items():
            if not plist:
                continue
            arr = np.asarray(plist, dtype=float)
            arr[:, 2] -= min_z
            positions[key] = arr.tolist()
        all_z = all_z - min_z
    max_z = float(all_z.max(initial=0.0))

    cell = (supercell[0] * lv_xy, supercell[1] * lv_xy, max_z)
    return {
        "positions": positions,
        "cell": cell,
        "lattice_vectors": lattice_vectors,
        "n_layers": n_layers,
    }

