"""Regression tests for validated PBC octahedral reconstruction and volumes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator


NMSE_CIFS = Path(__file__).resolve().parents[1] / (
    "TEMPORAL/nmse_2d_perovskite_db/CORRECT/cifs"
)


@pytest.fixture(scope="module")
def creator():
    return q2D_creator()


def test_perfect_bulk_has_valid_local_descriptors(creator):
    structure = creator.create_structure(
        A_ions="Cs",
        B_ions="Pb",
        X_ions="I",
        structure_type="bulk",
        template="cubic",
        xy_expansion=(2, 2),
        thickness=2,
        glazer_pattern="a0a0a0",
        glazer_angles=[0, 0, 0],
    )
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()

    detailed = analyzer.get_octahedral_distortions()
    assert detailed
    assert all(metrics.get("status") == "valid" for metrics in detailed.values())
    assert analyzer.compute_delta() is not None
    assert analyzer.compute_delta() < 0.01

    volumes = analyzer.get_octahedral_volumes(mode="local")
    assert all(np.isfinite(volume) and volume > 0 for volume in volumes.values())

    bxb = analyzer.get_bxb_angles()
    assert bxb.get("bxb_angles") is not None
    assert len(bxb["bxb_angles"]) > 0
    assert abs(bxb["bxb_mean"] - 180.0) < 2.0

    validity = analyzer.get_octahedral_validity()
    assert all(entry["status"] == "valid" for entry in validity.values())


def test_volume_uses_same_reconstruction_as_distortions():
    """Structures that previously lost volume under graph-bond counting."""
    if not NMSE_CIFS.exists():
        pytest.skip("NMSE CORRECT CIF directory not available")

    for prefix in ("0049", "0058"):
        path = next(NMSE_CIFS.glob(f"{prefix}_*"))
        analyzer = q2D_analyzer(str(path))
        analyzer.analyze()
        detailed = analyzer.get_octahedral_distortions()
        volumes = analyzer.get_octahedral_volumes(mode="local")
        assert detailed, f"{prefix}: expected octahedra"
        assert all(metrics.get("status") == "valid" for metrics in detailed.values())
        assert all(np.isfinite(volume) and volume > 0 for volume in volumes.values())
        assert analyzer.compute_delta() is not None


def test_empty_bxb_does_not_crash_without_octahedra(creator):
    structure = creator.create_structure(
        A_ions="Cs",
        B_ions="Pb",
        X_ions="I",
        structure_type="bulk",
        template="cubic",
        xy_expansion=(1, 1),
        thickness=1,
        glazer_pattern="a0a0a0",
        glazer_angles=[0, 0, 0],
    )
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    # Even with zero shared-X angles, API must return a dict.
    bxb = analyzer.get_bxb_angles()
    assert isinstance(bxb, dict)
    assert "bxb_angles" in bxb
    assert "bxb_mean" in bxb
