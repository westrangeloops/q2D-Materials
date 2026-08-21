#!/usr/bin/env python3
"""Compare notebook stacking analysis vs repo q2D_analyzer implementation.

Runs both pipelines on the same POSCAR fixtures and reports pass/fail.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
TWISTER_DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load frozen notebook reference (verbatim from stacking_analysis.ipynb cell 0)
_ref_path = TWISTER_DATA / "reference_notebook_impl.py"
_spec = importlib.util.spec_from_file_location("reference_notebook_impl", _ref_path)
_ref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ref)

from ase.io import read
from q2D_Materials.analyzer import q2D_analyzer


def _fmt_stats(d: dict) -> str:
    if d.get("n", 0) == 0:
        return "(empty)"
    return (
        f"n={d['n']:4d} mean={d['mean']:.6f} std={d['std']:.6f} "
        f"min={d['min']:.6f} max={d['max']:.6f}"
    )


def _stats_to_dict(stats) -> dict:
    """Convert SummaryStats or notebook dict to plain dict."""
    if isinstance(stats, dict):
        return stats
    return {
        "n": int(stats.n),
        "mean": float(stats.mean),
        "std": float(stats.std),
        "min": float(stats.min),
        "max": float(stats.max),
    }


def _compare_stats(name: str, ref: dict, repo: dict, atol: float = 1e-3) -> list[str]:
    """Return list of failure messages (empty = pass)."""
    failures = []
    if ref["n"] != repo["n"]:
        failures.append(f"  {name}: n mismatch ref={ref['n']} repo={repo['n']}")
        return failures  # other fields not comparable
    if ref["n"] == 0:
        return failures
    for key in ("mean", "std", "min", "max"):
        rv, pv = ref[key], repo[key]
        if not (np.isnan(rv) and np.isnan(pv)):
            if abs(rv - pv) > atol:
                failures.append(
                    f"  {name}.{key}: ref={rv:.6f} repo={pv:.6f} "
                    f"diff={abs(rv - pv):.6e} (atol={atol})"
                )
    return failures


def _compare_ratio_dist(ref_ratio: np.ndarray, repo_ratio: np.ndarray, atol: float = 1e-3) -> list[str]:
    failures = []
    if len(ref_ratio) == 0 and len(repo_ratio) == 0:
        return failures
    if len(ref_ratio) != len(repo_ratio):
        failures.append(
            f"  ratio length: ref={len(ref_ratio)} repo={len(repo_ratio)}"
        )
        # Still compare distribution summaries on overlapping length
    if len(ref_ratio) == 0 or len(repo_ratio) == 0:
        return failures

    # Notebook stores ratio_per_I = d_I_to_Cs / d_I_to_I (NOT the 2d/(d+d) form).
    # Repo stores ratio_per_x = 2*d(X->cat)/(d(X->X)+d(X->cat)).
    # Compare both raw means and also convert notebook ratio if needed.
    ref_mean, ref_std = float(np.nanmean(ref_ratio)), float(np.nanstd(ref_ratio))
    repo_mean, repo_std = float(np.nanmean(repo_ratio)), float(np.nanstd(repo_ratio))

    # Convert notebook R_nb = d_cs/d_ii into the 2d/(d+d) form is not possible
    # without raw distances; report both and also compare sorted values if same length
    # after converting repo -> notebook form if raw available.
    failures.append(
        f"  ratio_note: notebook formula is d(I->Cs)/d(I->I); "
        f"repo formula is 2*d(X->cat)/(d(X->X)+d(X->cat))"
    )
    failures.append(
        f"  ratio_dist: ref_mean={ref_mean:.6f}±{ref_std:.6f} "
        f"repo_mean={repo_mean:.6f}±{repo_std:.6f}"
    )
    return failures  # informational; judged separately


def run_one(poscar_path: Path, atol: float = 1e-3) -> dict:
    print("=" * 72)
    print(f"POSCAR: {poscar_path}")
    print("=" * 72)

    # --- Reference (notebook) ---
    ref = _ref.analyze_interface(str(poscar_path), frac_I=0.25, frac_Cs=0.5, verbose=True)
    print("\n[REFERENCE notebook]")
    for key in ("I_to_Cs", "I_to_I", "Cs_to_Cs"):
        print(f"  {key}: {_fmt_stats(ref[key])}")
    ref_ratio = ref["_raw"]["ratio_per_I"]
    print(f"  ratio_per_I: n={len(ref_ratio)} mean={np.nanmean(ref_ratio):.6f} "
          f"std={np.nanstd(ref_ratio):.6f}")
    print(f"  z_iface={ref['_meta']['z_interface']:.6f} "
          f"vdw_gap={ref['_meta']['vdw_gap']:.6f}")

    # --- Repo implementation ---
    atoms = read(str(poscar_path))
    analyzer = q2D_analyzer(atoms)
    analyzer.analyze()
    print("\n[REPO q2D_analyzer]")
    print(f"  structure_type={analyzer.structure_type} "
          f"is_twister={analyzer.is_twister} n_slabs={analyzer.n_slabs}")
    slabs = analyzer.get_slabs()
    for sid, info in slabs.items():
        print(f"  slab_{sid}: octahedra={info['octahedra_count']} "
              f"layers={info['layer_ids']} z_range={info['z_range']}")

    if analyzer.n_slabs < 2:
        print("  FAIL: expected n_slabs >= 2")
        return {"path": str(poscar_path), "ok": False, "failures": ["n_slabs < 2"]}

    registry = analyzer.get_stacking_registry(
        slab_id1="0",
        slab_id2="1",
        x_frac=0.25,
        cation_frac=0.5,
        x_symbols=["I"],
        cation_symbols=["Cs"],
    )
    repo_map = {
        "I_to_Cs": _stats_to_dict(registry.x_to_cation),
        "I_to_I": _stats_to_dict(registry.x_to_x),
        "Cs_to_Cs": _stats_to_dict(registry.cation_to_cation),
    }
    for key in ("I_to_Cs", "I_to_I", "Cs_to_Cs"):
        print(f"  {key}: {_fmt_stats(repo_map[key])}")
    print(f"  ratio_per_x: n={len(registry.ratio_per_x)} "
          f"mean={np.nanmean(registry.ratio_per_x) if len(registry.ratio_per_x) else float('nan'):.6f} "
          f"std={np.nanstd(registry.ratio_per_x) if len(registry.ratio_per_x) else float('nan'):.6f}")
    print(f"  z_iface={registry.z_interface:.6f} vdw_gap={registry.vdw_gap:.6f}")

    # --- Compare distance stats ---
    failures = []
    for key in ("I_to_Cs", "I_to_I", "Cs_to_Cs"):
        failures.extend(_compare_stats(key, ref[key], repo_map[key], atol=atol))

    # Intermediate diagnostics
    print("\n[COMPARE]")
    print(f"  z_iface diff: {abs(ref['_meta']['z_interface'] - registry.z_interface):.6e}")
    print(f"  vdw_gap diff: {abs(ref['_meta']['vdw_gap'] - registry.vdw_gap):.6e}")

    # Rebuild notebook-equivalent ratio from repo raw distances for fair compare
    d_x_cat = registry.raw.get("d_x_to_cation", np.array([]))
    d_x_x = registry.raw.get("d_x_to_x", np.array([]))
    if len(d_x_cat) > 0 and len(d_x_x) > 0:
        n = min(len(d_x_cat), len(d_x_x))
        # Notebook formula: R = d(I->Cs) / d(I->I)
        repo_nb_ratio = d_x_cat[:n] / d_x_x[:n]
        # Sorted comparison (atom ordering may differ)
        ref_sorted = np.sort(ref_ratio)
        repo_sorted = np.sort(repo_nb_ratio)
        if len(ref_sorted) == len(repo_sorted):
            max_diff = float(np.max(np.abs(ref_sorted - repo_sorted)))
            mean_diff = float(np.mean(np.abs(ref_sorted - repo_sorted)))
            print(f"  ratio (notebook formula, sorted): max_abs_diff={max_diff:.6e} "
                  f"mean_abs_diff={mean_diff:.6e}")
            if max_diff > atol:
                failures.append(
                    f"  sorted notebook-formula ratio max_diff={max_diff:.6e} > {atol}"
                )
        else:
            failures.append(
                f"  ratio length mismatch for notebook-formula compare: "
                f"ref={len(ref_sorted)} repo={len(repo_sorted)}"
            )
        # Also report mean of both formulas
        print(f"  notebook R=dCs/dII mean={np.nanmean(ref_ratio):.6f}")
        print(f"  repo-as-notebook R mean={np.nanmean(repo_nb_ratio):.6f}")
        print(f"  repo R=2d/(d+d) mean={np.nanmean(registry.ratio_per_x):.6f}")

    if failures:
        print("\n  RESULT: FAIL")
        for f in failures:
            print(f)
    else:
        print("\n  RESULT: PASS")

    return {
        "path": str(poscar_path),
        "ok": len(failures) == 0,
        "failures": failures,
        "analyzer": analyzer,
        "registry": registry,
        "ref": ref,
    }


def main():
    fixtures = [
        TWISTER_DATA / "MACE" / "2_1" / "POSCAR",
        TWISTER_DATA / "MACE" / "13_1" / "POSCAR",
    ]
    results = []
    for path in fixtures:
        if not path.exists():
            print(f"MISSING: {path}")
            results.append({"path": str(path), "ok": False, "failures": ["missing file"]})
            continue
        results.append(run_one(path))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    all_ok = True
    for r in results:
        status = "PASS" if r["ok"] else "FAIL"
        print(f"  [{status}] {r['path']}")
        all_ok = all_ok and r["ok"]

    # Save heatmap for 13_1 if available
    for r in results:
        if "13_1" in r["path"] and r.get("analyzer") is not None and r.get("registry") is not None:
            out = TWISTER_DATA / "heatmap_13_1_repro.png"
            print(f"\nWriting heatmap to {out}")
            r["analyzer"].plot_stacking_heatmap(r["registry"], output_path=str(out))
            print(f"  wrote {out}")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
