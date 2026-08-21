"""Stacking registry analysis at inter-slab interfaces.

Applies to any structure with 2+ z-discontinuous inorganic slabs (ordinary
RP/DJ cells as well as user-declared twisters). Ported from
stacking_analysis.ipynb.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.analyzer_class import q2D_analyzer


@dataclass
class SummaryStats:
    n: int = 0
    mean: float = float('nan')
    std: float = float('nan')
    min: float = float('nan')
    max: float = float('nan')


@dataclass
class StackingRegistryResult:
    x_to_cation: SummaryStats = field(default_factory=SummaryStats)
    x_to_x: SummaryStats = field(default_factory=SummaryStats)
    cation_to_cation: SummaryStats = field(default_factory=SummaryStats)
    ratio_per_x: np.ndarray = field(default_factory=lambda: np.array([]))
    xy_x_interface: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 2))
    xy_cation_interface: np.ndarray = field(default_factory=lambda: np.array([]).reshape(0, 2))
    lattice: np.ndarray = field(default_factory=lambda: np.eye(3))
    slab_pair: Tuple[str, str] = ('0', '1')
    z_interface: float = float('nan')
    vdw_gap: float = float('nan')
    raw: Dict[str, Any] = field(default_factory=dict)


def _summarize(arr: np.ndarray) -> SummaryStats:
    if len(arr) == 0:
        return SummaryStats()
    return SummaryStats(
        n=len(arr),
        mean=float(arr.mean()),
        std=float(arr.std()),
        min=float(arr.min()),
        max=float(arr.max()),
    )


def min_image_distances(
    ref_point_cart: np.ndarray,
    candidates_cart: np.ndarray,
    lattice: np.ndarray,
    pbc: Tuple[bool, bool, bool] = (True, True, True),
) -> np.ndarray:
    """Minimum-image distances from one point to many candidates."""
    inv_lat = np.linalg.inv(lattice)
    ref_frac = ref_point_cart @ inv_lat
    cand_frac = candidates_cart @ inv_lat
    delta_frac = cand_frac - ref_frac[None, :]
    for dim in range(3):
        if pbc[dim]:
            delta_frac[:, dim] -= np.round(delta_frac[:, dim])
    delta_cart = delta_frac @ lattice
    return np.linalg.norm(delta_cart, axis=1)


def split_layers_by_gap(z_values_ref: np.ndarray) -> float:
    """Return z-cut at the midpoint of the largest gap in a 1D z array.

    Matches the notebook helper used on B-site (e.g. Sn) centers.
    """
    z_sorted = np.sort(np.asarray(z_values_ref, dtype=float))
    if len(z_sorted) < 2:
        return float(z_sorted[0]) if len(z_sorted) else 0.0
    gaps = np.diff(z_sorted)
    i_max_gap = int(np.argmax(gaps))
    return float(0.5 * (z_sorted[i_max_gap] + z_sorted[i_max_gap + 1]))


def _b_site_z_values(analyzer: "q2D_analyzer") -> np.ndarray:
    """Z coordinates of octahedron B-site centers (notebook uses Sn)."""
    positions = analyzer.cell.get_positions()
    b_x = analyzer.get_b_x_atoms()
    b_indices = b_x.get('b_indices', np.array([], dtype=int))
    if len(b_indices) == 0:
        return positions[:, 2]
    return positions[b_indices, 2]


def _resolve_species_lists(
    analyzer: "q2D_analyzer",
    x_symbols: Optional[List[str]],
    cation_symbols: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    symbols = analyzer.cell.get_chemical_symbols()
    b_x = analyzer.get_b_x_atoms()
    x_set = set(int(i) for i in b_x['x_indices'].tolist())

    if x_symbols is None:
        x_symbols = sorted({symbols[i] for i in x_set})

    if cation_symbols is None:
        cation_symbols = []
        for a in analyzer.get_a_sites():
            sym = a.get('symbol', '')
            if sym and sym not in x_symbols:
                cation_symbols.append(sym)
        cation_symbols = sorted(set(cation_symbols)) or ['Cs', 'Rb', 'K']

    return list(x_symbols), list(cation_symbols)


def _select_interface_atoms(
    positions: np.ndarray,
    indices: np.ndarray,
    z_interface: float,
    frac: float,
) -> np.ndarray:
    """Select the frac*N atoms closest in z to the interface plane."""
    if len(indices) == 0 or frac <= 0:
        return np.array([], dtype=int)
    n = int(round(frac * len(indices)))
    if n == 0:
        return np.array([], dtype=int)
    order = np.argsort(np.abs(positions[indices, 2] - z_interface))
    return indices[order[:n]]


def analyze_stack_interface(
    analyzer: "q2D_analyzer",
    slab_id1: str = '0',
    slab_id2: str = '1',
    x_frac: float = 0.25,
    cation_frac: float = 0.5,
    x_symbols: Optional[List[str]] = None,
    cation_symbols: Optional[List[str]] = None,
) -> StackingRegistryResult:
    """Compute inter-slab stacking registry metrics at a slab interface.

    Works for any structure with 2+ discontinuous slabs (RP, DJ, or
    user-declared twister). Methodology mirrors ``stacking_analysis.ipynb``:

    1. Split atoms into bottom/top using the z-gap between the requested
       slabs (fallback: largest z-gap among B-site centers).
    2. Define the interface plane as the midpoint between the topmost X of the
       bottom layer and the bottommost X of the top layer.
    3. Select ``x_frac`` of X atoms and ``cation_frac`` of cations in each layer
       nearest to that plane.
    4. Compute PBC nearest-neighbor distances across layers.

    The per-X stacking ratio stored in ``ratio_per_x`` uses the documented form
    ``R = 2 * d(X→cation) / (d(X→X) + d(X→cation))`` (heatmap colormap centred
    at R=1). Raw distance arrays are retained in ``raw`` so alternate ratios
    (e.g. the notebook's ``d(X→cation)/d(X→X)``) can be reconstructed.
    """
    pair = tuple(sorted((str(slab_id1), str(slab_id2)), key=lambda x: int(x) if x.isdigit() else 0))

    positions = np.asarray(analyzer.cell.get_positions(), dtype=float)
    symbols = list(analyzer.cell.get_chemical_symbols())
    lattice = np.array(analyzer.cell.get_cell(), dtype=float)

    x_symbols, cation_symbols = _resolve_species_lists(analyzer, x_symbols, cation_symbols)
    b_x = analyzer.get_b_x_atoms()
    x_set = set(int(i) for i in b_x['x_indices'].tolist())

    # --- 1. Layer split: prefer requested slab z-ranges, else B-site z-gap ---
    z_cut = None
    slabs = analyzer.get_slabs()
    info1 = slabs.get(str(slab_id1), {})
    info2 = slabs.get(str(slab_id2), {})
    z1 = info1.get('z_range') or (None, None)
    z2 = info2.get('z_range') or (None, None)
    if z1[0] is not None and z1[1] is not None and z2[0] is not None and z2[1] is not None:
        if float(z1[0]) <= float(z2[0]):
            z_cut = 0.5 * (float(z1[1]) + float(z2[0]))
        else:
            z_cut = 0.5 * (float(z2[1]) + float(z1[0]))
    if z_cut is None:
        z_cut = split_layers_by_gap(_b_site_z_values(analyzer))
    layer_bottom = positions[:, 2] < z_cut

    is_x = np.array([
        (symbols[i] in x_symbols) and (i in x_set)
        for i in range(len(symbols))
    ], dtype=bool)
    # Fall back to symbol-only if graph X-set is empty / incomplete
    if not is_x.any():
        is_x = np.array([symbols[i] in x_symbols for i in range(len(symbols))], dtype=bool)

    is_cation = np.array([symbols[i] in cation_symbols for i in range(len(symbols))], dtype=bool)

    x_bot_mask = is_x & layer_bottom
    x_top_mask = is_x & ~layer_bottom
    cat_bot_mask = is_cation & layer_bottom
    cat_top_mask = is_cation & ~layer_bottom

    # --- 2. Interface plane from facing X atoms ---
    if x_bot_mask.any() and x_top_mask.any():
        z_top_bot = float(positions[x_bot_mask, 2].max())
        z_bot_top = float(positions[x_top_mask, 2].min())
        z_interface = 0.5 * (z_top_bot + z_bot_top)
        vdw_gap = z_bot_top - z_top_bot
    else:
        # Fallback: midpoint of slab B-site z ranges
        z_bot_max = slabs.get(pair[0], {}).get('z_range', (None, None))[1]
        z_top_min = slabs.get(pair[1], {}).get('z_range', (None, None))[0]
        if z_bot_max is not None and z_top_min is not None:
            z_interface = 0.5 * (float(z_bot_max) + float(z_top_min))
            vdw_gap = float(z_top_min) - float(z_bot_max)
        else:
            z_interface = float(z_cut)
            vdw_gap = float('nan')

    # --- 3. Select interface atoms ---
    x_bot_idx = np.where(x_bot_mask)[0]
    x_top_idx = np.where(x_top_mask)[0]
    cat_bot_idx = np.where(cat_bot_mask)[0]
    cat_top_idx = np.where(cat_top_mask)[0]

    x_bot_if = _select_interface_atoms(positions, x_bot_idx, z_interface, x_frac)
    x_top_if = _select_interface_atoms(positions, x_top_idx, z_interface, x_frac)
    cat_bot_if = _select_interface_atoms(positions, cat_bot_idx, z_interface, cation_frac)
    cat_top_if = _select_interface_atoms(positions, cat_top_idx, z_interface, cation_frac)

    # --- 4. Cross-layer nearest-neighbor distances ---
    def nearest(src_idx: np.ndarray, tgt_idx: np.ndarray) -> np.ndarray:
        if len(src_idx) == 0 or len(tgt_idx) == 0:
            return np.array([])
        tgt_pos = positions[tgt_idx]
        return np.array([
            min_image_distances(positions[i], tgt_pos, lattice).min()
            for i in src_idx
        ])

    d_x_to_cat = np.concatenate([
        nearest(x_bot_if, cat_top_if),
        nearest(x_top_if, cat_bot_if),
    ])
    d_x_to_x = np.concatenate([
        nearest(x_bot_if, x_top_if),
        nearest(x_top_if, x_bot_if),
    ])
    d_cat_to_cat = np.concatenate([
        nearest(cat_bot_if, cat_top_if),
        nearest(cat_top_if, cat_bot_if),
    ])

    ratio_per_x = np.array([])
    if len(d_x_to_cat) > 0 and len(d_x_to_x) > 0:
        n = min(len(d_x_to_cat), len(d_x_to_x))
        denom = d_x_to_x[:n] + d_x_to_cat[:n]
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio_per_x = np.where(denom > 0, 2.0 * d_x_to_cat[:n] / denom, np.nan)

    xy_x = np.vstack([
        positions[x_bot_if, :2] if len(x_bot_if) else np.empty((0, 2)),
        positions[x_top_if, :2] if len(x_top_if) else np.empty((0, 2)),
    ]) if (len(x_bot_if) + len(x_top_if)) else np.empty((0, 2))

    xy_cat = np.vstack([
        positions[cat_bot_if, :2] if len(cat_bot_if) else np.empty((0, 2)),
        positions[cat_top_if, :2] if len(cat_top_if) else np.empty((0, 2)),
    ]) if (len(cat_bot_if) + len(cat_top_if)) else np.empty((0, 2))

    return StackingRegistryResult(
        x_to_cation=_summarize(d_x_to_cat),
        x_to_x=_summarize(d_x_to_x),
        cation_to_cation=_summarize(d_cat_to_cat),
        ratio_per_x=ratio_per_x,
        xy_x_interface=xy_x,
        xy_cation_interface=xy_cat,
        lattice=lattice,
        slab_pair=pair,
        z_interface=float(z_interface),
        vdw_gap=float(vdw_gap),
        raw={
            'd_x_to_cation': d_x_to_cat,
            'd_x_to_x': d_x_to_x,
            'd_cation_to_cation': d_cat_to_cat,
            'z_cut': z_cut,
            'n_x_bot_interface': int(len(x_bot_if)),
            'n_x_top_interface': int(len(x_top_if)),
            'n_cation_bot_interface': int(len(cat_bot_if)),
            'n_cation_top_interface': int(len(cat_top_if)),
        },
    )
