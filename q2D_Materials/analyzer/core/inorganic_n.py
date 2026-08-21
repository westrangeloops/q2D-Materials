"""Infer inorganic perovskite layer thickness n from B-site geometry."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np


def _cluster_frac_planes(frac_vals, tol: float = 0.06) -> list[float]:
    """Cluster fractional coordinates into plane means with PBC unwrap.

    Returns sorted unwrapped plane positions (not wrapped to [0, 1)).
    """
    vals = sorted(float(v) for v in frac_vals)
    if not vals:
        return []
    ref = vals[0]
    unwrapped = sorted(ref + (v - ref - round(v - ref)) for v in vals)
    planes: list[list[float]] = []
    current: list[float] = []
    for u in unwrapped:
        if not current or abs(u - current[-1]) <= tol:
            current.append(u)
        else:
            planes.append(current)
            current = [u]
    if current:
        planes.append(current)
    return [float(np.mean(p)) for p in planes]


def _equatorial_bx_distances(graph, cell, max_bx: float = 5.0) -> list[float]:
    """Collect B–X bond lengths for equatorial ligands when tagged."""
    positions = cell.get_positions()
    distances: list[float] = []

    for node, data in graph.nodes(data=True):
        if data.get("node_type") != "octahedron":
            continue
        center_idx = None
        for neigh in graph.neighbors(node):
            edge = graph.get_edge_data(node, neigh) or {}
            if edge.get("edge_type") == "contains" and edge.get("role") == "center":
                center_idx = graph.nodes[neigh].get("vasp_index")
                break
        if center_idx is None:
            continue
        b_pos = positions[center_idx]

        eq_indices: list[int] = []
        for neigh in graph.neighbors(node):
            edge = graph.get_edge_data(node, neigh) or {}
            if edge.get("edge_type") != "contains" or edge.get("role") != "ligand":
                continue
            nd = graph.nodes[neigh]
            idx = nd.get("vasp_index")
            if idx is None:
                continue
            if nd.get("is_equatorial"):
                eq_indices.append(idx)

        if not eq_indices:
            center_node = f"atom_{center_idx}"
            if center_node in graph:
                for neigh in graph.neighbors(center_node):
                    nd = graph.nodes[neigh]
                    if nd.get("is_equatorial"):
                        idx = nd.get("vasp_index")
                        if idx is not None:
                            eq_indices.append(idx)

        for x_idx in eq_indices:
            dist = float(np.linalg.norm(b_pos - positions[x_idx]))
            if dist < max_bx:
                distances.append(dist)

    if distances:
        return distances

    # Fallback: all octahedron–ligand contains edges
    for node, data in graph.nodes(data=True):
        if data.get("node_type") != "octahedron":
            continue
        center_idx = None
        ligand_idxs: list[int] = []
        for neigh in graph.neighbors(node):
            edge = graph.get_edge_data(node, neigh) or {}
            if edge.get("edge_type") != "contains":
                continue
            idx = graph.nodes[neigh].get("vasp_index")
            if idx is None:
                continue
            if edge.get("role") == "center":
                center_idx = idx
            elif edge.get("role") == "ligand":
                ligand_idxs.append(idx)
        if center_idx is None:
            continue
        b_pos = positions[center_idx]
        for x_idx in ligand_idxs:
            dist = float(np.linalg.norm(b_pos - positions[x_idx]))
            if dist < max_bx:
                distances.append(dist)
    return distances


def _expected_layer_spacing(graph, cell) -> float:
    """``2 * mean(equatorial B–X)``, matching layer_identification convention."""
    bx = _equatorial_bx_distances(graph, cell)
    if bx:
        return float(2.0 * np.mean(bx))
    return 6.3


def _b_frac_coords(graph, cell) -> np.ndarray:
    frac = cell.get_scaled_positions()
    idxs: list[int] = []
    for node, data in graph.nodes(data=True):
        if data.get("node_type") != "octahedron":
            continue
        for neigh in graph.neighbors(node):
            edge = graph.get_edge_data(node, neigh) or {}
            if edge.get("edge_type") == "contains" and edge.get("role") == "center":
                idx = graph.nodes[neigh].get("vasp_index")
                if idx is not None:
                    idxs.append(idx)
    if not idxs:
        return np.zeros((0, 3))
    return np.asarray([frac[i] for i in idxs], dtype=float)


def _n_from_planes_along_axis(
    plane_means: list[float],
    axis_length: float,
    expected: float,
    *,
    lo_ratio: float = 0.85,
    hi_ratio: float = 1.30,
    spacer_ratio: float = 1.5,
) -> Optional[int]:
    """Count inorganic n from B-planes along one axis, handling PBC + spacer gaps.

    Planes linked by gaps in ``[lo, hi] * expected`` belong to the same slab.
    Larger gaps (organic spacers) and the periodic wrap are treated as cuts.
    Returns the size of the largest slab segment, or None if no intra-slab
    linkage exists (corrugated monolayer / no credible stacking).
    """
    n_planes = len(plane_means)
    if n_planes == 0:
        return None
    if n_planes == 1:
        return 1

    lo = expected * lo_ratio
    hi = expected * hi_ratio
    spacer = expected * spacer_ratio

    # Consecutive gaps + wrap-around gap (fractional span of unwrapped planes)
    gaps: list[float] = []
    for i in range(n_planes - 1):
        gaps.append((plane_means[i + 1] - plane_means[i]) * axis_length)
    span = plane_means[-1] - plane_means[0]
    if span >= 1.0 - 1e-6:
        wrap = max(0.0, (1.0 - span) * axis_length)
    else:
        wrap = (plane_means[0] + 1.0 - plane_means[-1]) * axis_length
    gaps.append(wrap)

    # Classify edges between plane i and plane (i+1) % n
    intra = [lo <= g <= hi for g in gaps]
    if not any(intra):
        return None

    # Uniform fill of the cell (tiny wrap, all other gaps intra, no spacer):
    # not a 2D slab stack along this axis (e.g. NMSE #759).
    if (
        n_planes >= 2
        and gaps[-1] < expected * 0.15
        and all(intra[:-1])
        and not any(g > spacer for g in gaps[:-1])
    ):
        return None

    # Connected components of planes under intra-slab edges (circular)
    visited = [False] * n_planes
    best = 0
    for start in range(n_planes):
        if visited[start]:
            continue
        size = 0
        stack = [start]
        visited[start] = True
        while stack:
            i = stack.pop()
            size += 1
            for j, edge_idx in (
                ((i + 1) % n_planes, i),
                ((i - 1) % n_planes, (i - 1) % n_planes),
            ):
                if intra[edge_idx] and not visited[j]:
                    visited[j] = True
                    stack.append(j)
        best = max(best, size)

    if best <= 0:
        return None
    _ = spacer
    return int(best)


def infer_inorganic_n_from_slabs(
    slabs: dict,
    graph,
    cell,
    *,
    frac_tol: float = 0.06,
    gap_lo_ratio: float = 0.85,
    gap_hi_ratio: float = 1.25,
) -> Optional[int]:
    """Estimate inorganic n (octahedral sheets per slab).

    Uses **all** B-sites in the cell (not fragmented continuity slabs). Along
    each crystal axis, cluster B into planes and count the largest run of
    planes linked by gaps ≈ ``2 * mean(equatorial B–X)``, treating larger
    gaps as organic spacers (PBC-aware). Prefers the longest cell axis
    (typical spacer/stacking direction). Falls back to n=1 when no credible
    multilayer stacking axis exists (rejects buckling / lateral supercells).
    """
    expected = _expected_layer_spacing(graph, cell)
    lengths = np.asarray(cell.cell.lengths(), dtype=float)
    coords = _b_frac_coords(graph, cell)
    if len(coords) == 0:
        if not slabs:
            return None
        frac = cell.get_scaled_positions()
        idxs: list[int] = []
        for sdata in slabs.values():
            for on in sdata.get("octahedra", []):
                if on not in graph:
                    continue
                for neigh in graph.neighbors(on):
                    edge = graph.get_edge_data(on, neigh) or {}
                    if edge.get("edge_type") == "contains" and edge.get("role") == "center":
                        idx = graph.nodes[neigh].get("vasp_index")
                        if idx is not None:
                            idxs.append(idx)
        if not idxs:
            return None
        coords = np.asarray([frac[i] for i in idxs], dtype=float)

    # For 1–2 B sites, any coplanar axis ⇒ n=1 (avoids in-plane lattice
    # mistaken for stacking, e.g. #650/#695). Do not apply when nB>=3:
    # true multilayers often have all B coplanar along a short lateral axis.
    if len(coords) <= 2:
        for axis in range(3):
            if len(_cluster_frac_planes(coords[:, axis], tol=frac_tol)) == 1:
                return 1

    order = list(np.argsort(-lengths))  # longest first
    long = order[0]
    planes_long = _cluster_frac_planes(coords[:, long], tol=frac_tol)
    n_long = _n_from_planes_along_axis(
        planes_long,
        float(lengths[long]),
        expected,
        lo_ratio=gap_lo_ratio,
        hi_ratio=gap_hi_ratio,
    )
    if n_long is not None:
        return int(n_long)

    # Longest axis is the spacer direction but has no corner-sharing multilayer
    # (organic gaps only / buckling). Do not trust shorter axes (lateral lattice).
    if len(planes_long) >= 1:
        return 1

    for axis in order[1:]:
        planes = _cluster_frac_planes(coords[:, axis], tol=frac_tol)
        n_est = _n_from_planes_along_axis(
            planes,
            float(lengths[axis]),
            expected,
            lo_ratio=gap_lo_ratio,
            hi_ratio=gap_hi_ratio,
        )
        if n_est is None:
            continue
        aspect = float(lengths[long]) / float(lengths[axis])
        if n_est == 2 and aspect < 1.4:
            continue
        return int(n_est)

    return 1


def infer_inorganic_n(analyzer: Any) -> Optional[int]:
    """Convenience wrapper using an analyzed ``q2D_analyzer`` instance."""
    return infer_inorganic_n_from_slabs(
        analyzer.get_slabs(),
        analyzer.get_graph(),
        analyzer.cell,
    )
