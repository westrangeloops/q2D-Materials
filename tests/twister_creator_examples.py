"""Cs-based twisted bilayers built with q2D_creator.twist()."""

from typing import Any, Dict, List, Tuple

from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure

CREATOR_TWISTERS: List[Dict[str, Any]] = [
    dict(id="cspbi_homobilayer_2_1", B1="Pb", X1="I", B2="Pb", X2="I", mn=(2, 1)),
    dict(id="cssni_homobilayer_2_1", B1="Sn", X1="I", B2="Sn", X2="I", mn=(2, 1)),
    dict(id="cspbi_cssni_heterobilayer_3_1", B1="Pb", X1="I", B2="Sn", X2="I", mn=(3, 1)),
    dict(id="cspbbr_homobilayer_2_1", B1="Pb", X1="Br", B2="Pb", X2="Br", mn=(2, 1)),
    dict(id="cspbi_homobilayer_3_1", B1="Pb", X1="I", B2="Pb", X2="I", mn=(3, 1)),
]


def _monolayer(q2d: q2D_creator, b_ion: str, x_ion: str) -> q2DStructure:
    return q2d.create_structure(
        structure_type="monolayer",
        A_ions="Cs",
        B_ions=b_ion,
        X_ions=x_ion,
        xy_expansion=(1, 1),
        template="cubic",
        vacuum=12.0,
    )


def build_creator_twister(spec: Dict[str, Any]) -> q2DStructure:
    """Build a Cs twisted bilayer from a CREATOR_TWISTERS row."""
    q2d = q2D_creator()
    m1 = _monolayer(q2d, spec["B1"], spec["X1"])
    m2 = _monolayer(q2d, spec["B2"], spec["X2"])
    mn: Tuple[int, int] = spec["mn"]
    return q2d.twist(
        monolayers=[m1, m2],
        twist_angles=[mn],
        interlayer_distances=[8.0],
        vacuum=12.0,
    )


def registry_x_symbols(spec: Dict[str, Any]) -> List[str]:
    return sorted({spec["X1"], spec["X2"]})


def analyze_creator_twister(spec: Dict[str, Any]):
    """Return (structure, analyzed q2D_analyzer, stacking registry)."""
    bilayer = build_creator_twister(spec)
    analyzer = q2D_analyzer(bilayer)
    analyzer.analyze()
    registry = analyzer.get_stacking_registry(
        cation_symbols=["Cs"],
        x_symbols=registry_x_symbols(spec),
    )
    return bilayer, analyzer, registry
