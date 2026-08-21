#!/usr/bin/env python3
"""Generate dual-ratio stacking maps for CREATOR_TWISTERS examples.

Usage (from repo root, devenv shell):

    python tests/data/twister/generate_creator_example_maps.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[3]
TESTS = ROOT / "tests"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

from twister_creator_examples import CREATOR_TWISTERS, analyze_creator_twister  # noqa: E402

from q2D_Materials.analyzer.twister_processing.stacking_plots import (  # noqa: E402
    RATIO_LABEL_NORMALIZED,
    RATIO_LABEL_UNNORMALIZED,
    plot_stacking_heatmap,
    unnormalized_ratio,
)

OUT_DIR = Path(__file__).resolve().parent / "creator_examples"
GRID_PTS = 120
DPI = 200


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in CREATOR_TWISTERS:
        example_id = spec["id"]
        print(f"=== {example_id} ===")
        bilayer, analyzer, registry = analyze_creator_twister(spec)
        print(
            f"  atoms={len(bilayer)} n_slabs={analyzer.n_slabs} "
            f"n_x={registry.x_to_x.n} n_cat={registry.x_to_cation.n}"
        )
        ratio_raw = unnormalized_ratio(registry)
        n = min(len(ratio_raw), len(registry.xy_x_interface), len(registry.ratio_per_x))
        if n == 0:
            print("  SKIP: empty registry")
            continue

        raw_path = OUT_DIR / f"{example_id}_unnormalized.png"
        norm_path = OUT_DIR / f"{example_id}_normalized.png"

        plot_stacking_heatmap(
            replace(registry, ratio_per_x=ratio_raw),
            output_path=str(raw_path),
            grid_pts=GRID_PTS,
            colorbar_label=RATIO_LABEL_UNNORMALIZED,
            dpi=DPI,
        )
        plot_stacking_heatmap(
            registry,
            output_path=str(norm_path),
            grid_pts=GRID_PTS,
            colorbar_label=RATIO_LABEL_NORMALIZED,
            dpi=DPI,
        )
        print(f"  wrote {raw_path.name} {norm_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
