#!/usr/bin/env python3
"""Generate dual-ratio stacking maps for the five creator twisters in example 18.

Usage (from repo root):

    devenv shell -- python Examples/generate_twister_stacking_maps.py
"""
from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.twister_processing.stacking_plots import (
    RATIO_LABEL_NORMALIZED,
    RATIO_LABEL_UNNORMALIZED,
    plot_stacking_heatmap,
    unnormalized_ratio,
)
from q2D_Materials.core.creator import q2D_creator

OUT_DIR = Path(__file__).resolve().parent / "images" / "twister"
GRID_PTS = 120
DPI = 200

EXAMPLES: List[Dict[str, Any]] = [
    dict(id="cspbi_homobilayer_2_1", title="CsPbI / CsPbI (2,1)", B1="Pb", X1="I", B2="Pb", X2="I", mn=(2, 1)),
    dict(id="cssni_homobilayer_2_1", title="CsSnI / CsSnI (2,1)", B1="Sn", X1="I", B2="Sn", X2="I", mn=(2, 1)),
    dict(id="cspbi_cssni_heterobilayer_3_1", title="CsPbI / CsSnI (3,1)", B1="Pb", X1="I", B2="Sn", X2="I", mn=(3, 1)),
    dict(id="cspbbr_homobilayer_2_1", title="CsPbBr / CsPbBr (2,1)", B1="Pb", X1="Br", B2="Pb", X2="Br", mn=(2, 1)),
    dict(id="cspbi_homobilayer_3_1", title="CsPbI / CsPbI (3,1)", B1="Pb", X1="I", B2="Pb", X2="I", mn=(3, 1)),
]


def _monolayer(q2d: q2D_creator, b_ion: str, x_ion: str):
    return q2d.create_structure(
        structure_type="monolayer",
        A_ions="Cs",
        B_ions=b_ion,
        X_ions=x_ion,
        xy_expansion=(1, 1),
        template="cubic",
        vacuum=12.0,
    )


def build_and_analyze(spec: Dict[str, Any]):
    q2d = q2D_creator()
    m1 = _monolayer(q2d, spec["B1"], spec["X1"])
    m2 = _monolayer(q2d, spec["B2"], spec["X2"])
    mn: Tuple[int, int] = spec["mn"]
    bilayer = q2d.twist(
        monolayers=[m1, m2],
        twist_angles=[mn],
        interlayer_distances=[8.0],
        vacuum=12.0,
    )
    analyzer = q2D_analyzer(bilayer)
    analyzer.analyze()
    x_symbols = sorted({spec["X1"], spec["X2"]})
    registry = analyzer.get_stacking_registry(
        cation_symbols=["Cs"],
        x_symbols=x_symbols,
    )
    return bilayer, analyzer, registry


def write_overview(out_dir: Path) -> Path:
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg

    fig, axes = plt.subplots(len(EXAMPLES), 2, figsize=(8.2, 18.5))
    for i, spec in enumerate(EXAMPLES):
        stem = spec["id"]
        for j, kind in enumerate(("unnormalized", "normalized")):
            ax = axes[i, j]
            img = mpimg.imread(out_dir / f"{stem}_{kind}.png")
            ax.imshow(img)
            ax.set_axis_off()
            if i == 0:
                ax.set_title(
                    r"Unnormalized $r=d_{X\to\mathrm{cation}}/d_{X\to X}$"
                    if j == 0
                    else r"Normalized $R=2d/(d+d)$",
                    fontsize=10,
                )
            if j == 0:
                ax.text(
                    -0.02,
                    0.5,
                    spec["title"],
                    transform=ax.transAxes,
                    va="center",
                    ha="right",
                    fontsize=8,
                    rotation=90,
                )
    fig.suptitle("Creator twisters — dual stacking maps", fontsize=12, y=0.995)
    fig.tight_layout()
    out = out_dir / "overview_unnormalized_vs_normalized.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for spec in EXAMPLES:
        example_id = spec["id"]
        print(f"=== {example_id} ===")
        bilayer, analyzer, registry = build_and_analyze(spec)
        ratio_raw = unnormalized_ratio(registry)
        print(
            f"  atoms={len(bilayer)} n_slabs={analyzer.n_slabs} "
            f"n_x={registry.x_to_x.n} n_cat={registry.x_to_cation.n} "
            f"vdw_gap={registry.vdw_gap:.3f} Å"
        )
        print(
            f"  mean R={float(registry.ratio_per_x.mean()) if len(registry.ratio_per_x) else float('nan'):.4f} "
            f"mean r={float(ratio_raw.mean()) if len(ratio_raw) else float('nan'):.4f}"
        )
        if len(ratio_raw) == 0 or len(registry.ratio_per_x) == 0:
            print("  SKIP: empty registry")
            continue

        plot_stacking_heatmap(
            replace(registry, ratio_per_x=ratio_raw),
            output_path=str(OUT_DIR / f"{example_id}_unnormalized.png"),
            grid_pts=GRID_PTS,
            colorbar_label=RATIO_LABEL_UNNORMALIZED,
            dpi=DPI,
        )
        plot_stacking_heatmap(
            registry,
            output_path=str(OUT_DIR / f"{example_id}_normalized.png"),
            grid_pts=GRID_PTS,
            colorbar_label=RATIO_LABEL_NORMALIZED,
            dpi=DPI,
        )
        print(f"  wrote {example_id}_unnormalized.png {example_id}_normalized.png")

    overview = write_overview(OUT_DIR)
    print(f"wrote {overview.name}")
    expected = 2 * len(EXAMPLES) + 1
    n_png = len(list(OUT_DIR.glob("*.png")))
    print(f"PNG count in {OUT_DIR}: {n_png} (expected {expected})")
    return 0 if n_png >= expected else 1


if __name__ == "__main__":
    raise SystemExit(main())
