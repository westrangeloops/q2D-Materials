"""Cleanup helpers for experimental CIF disorder (split-site hydrogens).

Converts crystallographic disorder models into an approximate single-site
structure usable for bonding, valence, and molecule validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from ase import Atoms


def _nearest_heavy(atoms: Atoms, h_idx: int, heavy_indices: List[int]) -> Optional[int]:
    """Index of the nearest non-H atom to this hydrogen (PBC-aware)."""
    if not heavy_indices:
        return None
    dists = atoms.get_distances(h_idx, heavy_indices, mic=True)
    return int(heavy_indices[int(np.argmin(dists))])


def _get_occupancies(atoms: Atoms) -> Optional[np.ndarray]:
    """Return per-atom float occupancies if ASE stored a numeric array, else None.

    ASE CIF readers often put a label→occupancy *dict* in ``atoms.info['occupancy']``;
    that is not usable as a per-index array and must be ignored.
    """
    if "occupancy" in getattr(atoms, "arrays", {}):
        raw = np.asarray(atoms.arrays["occupancy"])
        try:
            occ = raw.astype(float)
        except (TypeError, ValueError):
            return None
        if occ.shape == (len(atoms),):
            return occ
    return None


def _cluster_representative_position(
    atoms: Atoms,
    members: List[int],
    occupancies: Optional[np.ndarray],
) -> np.ndarray:
    """Mean or occupancy-weighted centroid of a disordered H cluster (PBC unwrap)."""
    positions = atoms.get_positions()
    cell = atoms.get_cell()
    ref = positions[members[0]]
    unwrapped = []
    weights = []
    for idx in members:
        # Unwrap relative to first member so the mean is not cell-split
        d = atoms.get_distance(members[0], idx, mic=True, vector=True)
        unwrapped.append(ref + d)
        if occupancies is not None:
            weights.append(max(float(occupancies[idx]), 1e-6))
        else:
            weights.append(1.0)
    unwrapped = np.asarray(unwrapped, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights /= weights.sum()
    return (unwrapped * weights[:, None]).sum(axis=0)


# Only near-identical split sites (Å). Larger H–H separation is a real geometry
# / disorder problem and must not be collapsed away.
DEFAULT_H_MERGE_CUTOFF = 0.01


def merge_disordered_hydrogens(
    atoms: Atoms,
    cutoff: float = DEFAULT_H_MERGE_CUTOFF,
) -> Tuple[Atoms, Dict[str, Any]]:
    """Merge split-site / disordered hydrogens into one approximate site.

    Only merges H–H pairs closer than ``cutoff`` (default 0.01 Å) that share
    the same nearest heavy atom (same parent). That collapses near-duplicate
    CIF sites without erasing chemically distinct hydrogens.

    For each cluster, keep one H:
    - If occupancies exist: keep the highest-occupancy index, place it at the
      occupancy-weighted centroid of the cluster.
    - Else: keep the lowest index, place it at the geometric mean.

    Returns
    -------
    cleaned : Atoms
        Structure with duplicate H removed
    report : dict
        ``n_h_merged``, ``n_clusters_merged``
    """
    symbols = atoms.get_chemical_symbols()
    h_indices = [i for i, s in enumerate(symbols) if s == "H"]
    heavy_indices = [i for i, s in enumerate(symbols) if s != "H"]
    empty_report = {"n_h_merged": 0, "n_clusters_merged": 0}
    if len(h_indices) < 2:
        return atoms.copy(), empty_report

    parent_heavy = {i: _nearest_heavy(atoms, i, heavy_indices) for i in h_indices}
    occupancies = _get_occupancies(atoms)

    uf = {i: i for i in h_indices}

    def find(x: int) -> int:
        while uf[x] != x:
            uf[x] = uf[uf[x]]
            x = uf[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            uf[rb] = ra

    for a, i in enumerate(h_indices):
        for j in h_indices[a + 1 :]:
            if parent_heavy[i] is None or parent_heavy[i] != parent_heavy[j]:
                continue
            if atoms.get_distance(i, j, mic=True) < cutoff:
                union(i, j)

    clusters: dict[int, List[int]] = {}
    for i in h_indices:
        clusters.setdefault(find(i), []).append(i)

    drop = set()
    keep_position: dict[int, np.ndarray] = {}
    n_clusters = 0

    for members in clusters.values():
        if len(members) < 2:
            continue
        n_clusters += 1
        members_sorted = sorted(members)
        if occupancies is not None:
            keep_idx = max(members_sorted, key=lambda i: float(occupancies[i]))
        else:
            keep_idx = members_sorted[0]
        keep_position[keep_idx] = _cluster_representative_position(
            atoms, members_sorted, occupancies
        )
        for idx in members_sorted:
            if idx != keep_idx:
                drop.add(idx)

    if not drop:
        return atoms.copy(), empty_report

    keep = [i for i in range(len(atoms)) if i not in drop]
    cleaned = atoms[keep]
    # Remap keep_position indices onto cleaned Atoms
    old_to_new = {old: new for new, old in enumerate(keep)}
    positions = cleaned.get_positions()
    for old_idx, pos in keep_position.items():
        positions[old_to_new[old_idx]] = pos
    cleaned.set_positions(positions)

    report = {
        "n_h_merged": len(drop),
        "n_clusters_merged": n_clusters,
    }
    return cleaned, report


def merge_close_hydrogens(
    atoms: Atoms,
    cutoff: float = DEFAULT_H_MERGE_CUTOFF,
) -> Tuple[Atoms, int]:
    """Alias for :func:`merge_disordered_hydrogens` returning ``(atoms, n_removed)``."""
    cleaned, report = merge_disordered_hydrogens(atoms, cutoff=cutoff)
    return cleaned, int(report["n_h_merged"])


def select_cif_atoms(
    path: str,
    hint: Optional[str] = None,
) -> Tuple[Atoms, Dict[str, Any]]:
    """Read a CIF, choosing the best data block when several are present.

    ASE ``read(path)`` alone returns the *last* block. NMSE files often pack
    iodide (with H) and bromide (without H) in one file.

    Preference order:
    1. Halogen matching ``hint`` / filename (I vs Br vs Cl)
    2. More hydrogens
    3. More atoms

    Returns
    -------
    atoms : Atoms
    meta : dict
        ``cif_blocks_considered``, ``selected_halogens``
    """
    from ase.io import read

    structures = read(path, index=":")
    if not isinstance(structures, list):
        structures = [structures]

    hint_text = (hint or path).lower()
    want_i = any(tok in hint_text for tok in ("pbi", "sni", "gei", "i4", "i7", "i10", "i13", "i16"))
    want_br = any(tok in hint_text for tok in ("pbbr", "snbr", "br4", "br7"))
    want_cl = any(tok in hint_text for tok in ("pbcl", "cl4", "cl10"))

    def score(a: Atoms) -> tuple:
        syms = a.get_chemical_symbols()
        n_h = syms.count("H")
        n_i = syms.count("I")
        n_br = syms.count("Br")
        n_cl = syms.count("Cl")
        halogen_bonus = 0
        if want_i and n_i > 0:
            halogen_bonus += 10_000
        if want_br and n_br > 0 and not (want_i and n_i > 0):
            halogen_bonus += 10_000
        if want_cl and n_cl > 0 and n_i == 0 and n_br == 0:
            halogen_bonus += 10_000
        return (halogen_bonus, n_h, len(a))

    chosen = max(structures, key=score)
    syms = chosen.get_chemical_symbols()
    meta = {
        "cif_blocks_considered": len(structures),
        "selected_halogens": {
            "I": syms.count("I"),
            "Br": syms.count("Br"),
            "Cl": syms.count("Cl"),
        },
    }
    return chosen, meta


def prepare_experimental_structure(
    source: Union[str, Path, Atoms],
    *,
    hint: Optional[str] = None,
    h_cutoff: float = DEFAULT_H_MERGE_CUTOFF,
) -> Tuple[Atoms, Dict[str, Any]]:
    """Build an analysis-ready approximate structure from an experimental CIF/Atoms.

    Pipeline:
    1. If ``source`` is a CIF path, select the best data block.
    2. Merge near-duplicate hydrogens (same parent, H–H < ``h_cutoff``).

    Parameters
    ----------
    source : str, Path, or Atoms
        CIF path or ASE Atoms
    hint : str, optional
        Filename/compound hint for halogen block selection (defaults to path)
    h_cutoff : float
        Max H–H distance (Å) for a duplicate site (default 0.01 Å)

    Returns
    -------
    atoms : Atoms
        Approximate single-site structure
    report : dict
        Cleanup metadata for logging / downstream flags
    """
    report: Dict[str, Any] = {
        "n_h_merged": 0,
        "n_clusters_merged": 0,
        "cif_blocks_considered": 1,
        "selected_halogens": {},
        "source_type": None,
    }

    if isinstance(source, Atoms):
        report["source_type"] = "atoms"
        cleaned, merge_report = merge_disordered_hydrogens(source.copy(), cutoff=h_cutoff)
        report.update(merge_report)
        syms = cleaned.get_chemical_symbols()
        report["selected_halogens"] = {
            "I": syms.count("I"),
            "Br": syms.count("Br"),
            "Cl": syms.count("Cl"),
        }
        cleaned.pbc = True
        return cleaned, report

    path = str(source)
    report["source_type"] = "cif" if path.lower().endswith(".cif") else "file"
    if path.lower().endswith(".cif"):
        atoms, block_meta = select_cif_atoms(path, hint=hint or path)
        report.update(block_meta)
    else:
        from ase.io import read

        atoms = read(path)
        report["cif_blocks_considered"] = 1
        syms = atoms.get_chemical_symbols()
        report["selected_halogens"] = {
            "I": syms.count("I"),
            "Br": syms.count("Br"),
            "Cl": syms.count("Cl"),
        }

    cleaned, merge_report = merge_disordered_hydrogens(atoms, cutoff=h_cutoff)
    report["n_h_merged"] = merge_report["n_h_merged"]
    report["n_clusters_merged"] = merge_report["n_clusters_merged"]
    cleaned.pbc = True
    return cleaned, report
