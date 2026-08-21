"""Creator-built Cs twister examples: stack detection, registry, and map smoke test."""

from pathlib import Path

import numpy as np
import pytest

from twister_creator_examples import (
    CREATOR_TWISTERS,
    analyze_creator_twister,
    build_creator_twister,
    registry_x_symbols,
)
from q2D_Materials.analyzer.twister_processing.stacking_plots import (
    RATIO_LABEL_NORMALIZED,
    unnormalized_ratio,
    plot_stacking_heatmap,
)

EXAMPLE_IDS = [spec["id"] for spec in CREATOR_TWISTERS]
MAP_DIR = Path(__file__).resolve().parent / "data" / "twister" / "creator_examples"


@pytest.mark.parametrize("spec", CREATOR_TWISTERS, ids=EXAMPLE_IDS)
def test_creator_twister_is_twister(spec):
    bilayer = build_creator_twister(spec)
    assert bilayer.structure_type == "twister"
    assert bilayer.is_twister is True


@pytest.mark.parametrize("spec", CREATOR_TWISTERS, ids=EXAMPLE_IDS)
def test_creator_twister_stack_and_registry(spec):
    bilayer, analyzer, registry = analyze_creator_twister(spec)
    assert analyzer.n_slabs >= 2
    assert analyzer.is_twister is True
    assert analyzer.structure_type == "twister"
    assert registry.x_to_cation.n > 0
    assert registry.x_to_x.n > 0
    assert registry.x_to_cation.mean > 0
    assert registry.x_to_x.mean > 0

    ratio_norm = np.asarray(registry.ratio_per_x)
    ratio_raw = unnormalized_ratio(registry)
    n_xy = len(registry.xy_x_interface)
    assert len(ratio_norm) > 0
    assert len(ratio_raw) > 0
    assert len(ratio_norm) == n_xy or min(len(ratio_norm), n_xy) > 0
    n = min(len(ratio_norm), len(ratio_raw), n_xy)
    assert n > 0
    assert np.isfinite(ratio_norm[:n]).all()
    assert np.isfinite(ratio_raw[:n]).all()
    assert registry_x_symbols(spec) == sorted({spec["X1"], spec["X2"]})
    del bilayer


@pytest.mark.parametrize("spec", CREATOR_TWISTERS, ids=EXAMPLE_IDS)
def test_creator_twister_maps_committed(spec):
    stem = spec["id"]
    unnorm = MAP_DIR / f"{stem}_unnormalized.png"
    norm = MAP_DIR / f"{stem}_normalized.png"
    assert unnorm.is_file(), f"missing {unnorm}; run generate_creator_example_maps.py"
    assert norm.is_file(), f"missing {norm}; run generate_creator_example_maps.py"
    assert unnorm.stat().st_size > 0
    assert norm.stat().st_size > 0


def test_plot_stacking_heatmap_smoke(tmp_path):
    spec = CREATOR_TWISTERS[0]
    _, analyzer, registry = analyze_creator_twister(spec)
    out = tmp_path / "cspbi_homobilayer_2_1_smoke.png"
    plot_stacking_heatmap(
        registry,
        output_path=str(out),
        grid_pts=40,
        colorbar_label=RATIO_LABEL_NORMALIZED,
        dpi=80,
    )
    assert out.is_file()
    assert out.stat().st_size > 0
    del analyzer
