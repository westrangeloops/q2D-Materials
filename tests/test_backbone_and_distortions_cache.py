"""Parity tests for optimized backbone ID and octahedra distortion cache."""

from __future__ import annotations

from typing import List, Optional, Set

import networkx as nx
import numpy as np
import pytest

from q2D_Materials.analyzer.characterization.distortions import (
    _compute_octahedral_distortions_detailed,
    _get_octahedral_distortions_detailed,
)
from q2D_Materials.analyzer.molecular_processing.molecule_graph import (
    _find_longest_path_from_node,
    _find_path_avoiding_h,
    _longest_path_between_anchors,
    _non_h_subgraph,
    create_molecule_graph,
    identify_backbone,
)


def _legacy_longest_between_anchors(graph: nx.Graph, anchors: List[int]) -> List[int]:
    """Old O(K^2) path search that rebuilt the non-H subgraph every pair."""
    longest: List[int] = []
    for i, a1 in enumerate(anchors):
        for a2 in anchors[i + 1 :]:
            path = _find_path_avoiding_h(graph, a1, a2)  # rebuilds non-H each call
            if path and len(path) > len(longest):
                longest = path
    return longest


def _legacy_longest_from_node(graph: nx.Graph, start: int) -> List[int]:
    non_h = _non_h_subgraph(graph)
    if start not in non_h:
        return [start]
    longest = [start]
    for target in non_h.nodes():
        if target == start:
            continue
        try:
            path = nx.shortest_path(non_h, start, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue
        if len(path) > len(longest):
            longest = path
    return longest


def _backbone_roles(graph: nx.Graph, molecule_node: str) -> Set[int]:
    roles = set()
    for node, data in graph.nodes(data=True):
        if node == molecule_node:
            continue
        if data.get("role") == "backbone":
            roles.add(node)
    return roles


class TestBackbonePathOptimization:
    def test_diamine_smiles_backbone_roles(self):
        # Putrescine-like diamine: two NH3 termini.
        G = create_molecule_graph(
            "C(CC[NH3+])C[NH3+]", analyze_backbone=False
        )
        mol_node = "molecule_0"
        assert mol_node in G
        nh3 = identify_backbone(G, mol_node)
        assert nh3 >= 2
        backbone = _backbone_roles(G, mol_node)
        assert len(backbone) >= 2
        # Non-H CHON atoms should all have a role.
        for node, data in G.nodes(data=True):
            if node == mol_node or data.get("node_type") != "atom":
                continue
            if data.get("symbol") in {"C", "O", "N"}:
                assert data.get("role") in {"backbone", "functional_group"}

    def test_longest_path_between_anchors_matches_legacy(self):
        G = create_molecule_graph(
            "[NH3+]CCCC[NH3+]", analyze_backbone=False
        )
        mol_node = "molecule_0"
        # Collect NH3 anchors the same way identify_backbone does.
        from q2D_Materials.analyzer.molecular_processing.molecule_candidates import (
            _find_pattern_matches,
            _parse_pattern,
        )

        atom_nodes = [
            n
            for n in G.nodes()
            if n != mol_node and G.nodes[n].get("node_type") == "atom"
        ]
        mol_sub = G.subgraph(atom_nodes).copy()
        matches = []
        for pattern in ("[NH3+]C", "[NH2]C"):
            matches.extend(_find_pattern_matches(mol_sub, _parse_pattern(pattern)))
        anchors = list(
            {m["anchor_idx"] for m in matches if m["anchor_idx"] is not None}
        )
        assert len(anchors) >= 2

        non_h = _non_h_subgraph(mol_sub)
        optimized = _longest_path_between_anchors(non_h, anchors)
        legacy = _legacy_longest_between_anchors(mol_sub, anchors)
        assert set(optimized) == set(legacy)
        assert len(optimized) == len(legacy)

    def test_longest_from_single_anchor_matches_legacy(self):
        G = create_molecule_graph("CCCC[NH3+]", analyze_backbone=False)
        mol_node = "molecule_0"
        from q2D_Materials.analyzer.molecular_processing.molecule_candidates import (
            _find_pattern_matches,
            _parse_pattern,
        )

        atom_nodes = [
            n
            for n in G.nodes()
            if n != mol_node and G.nodes[n].get("node_type") == "atom"
        ]
        mol_sub = G.subgraph(atom_nodes).copy()
        matches = _find_pattern_matches(mol_sub, _parse_pattern("[NH3+]C"))
        anchors = list(
            {m["anchor_idx"] for m in matches if m["anchor_idx"] is not None}
        )
        assert len(anchors) == 1
        start = anchors[0]
        non_h = _non_h_subgraph(mol_sub)
        optimized = _find_longest_path_from_node(mol_sub, start, non_h_graph=non_h)
        legacy = _legacy_longest_from_node(mol_sub, start)
        assert set(optimized) == set(legacy)
        assert len(optimized) == len(legacy)


class TestOctahedralDistortionsCache:
    def test_cache_returns_identical_and_avoids_recompute(self, tmp_path):
        pytest.importorskip("ase")
        from ase import Atoms
        from q2D_Materials.analyzer import q2D_analyzer

        # Tiny synthetic BX6-like fragment is hard; use a real small CIF if present,
        # otherwise skip when analyze finds no octahedra.
        from pathlib import Path

        cif = (
            Path(__file__).resolve().parents[1]
            / "TEMPORAL"
            / "nmse_2d_perovskite_db"
            / "CORRECT"
            / "cifs"
            / "0001_CH3_CH2_3NH3_2PbI4.cif"
        )
        if not cif.is_file():
            pytest.skip("NMSE sample CIF not available")

        analyzer = q2D_analyzer(str(cif))
        analyzer.analyze(octahedra_centers=["Pb"], valid_halogen=["I"])

        first = _get_octahedral_distortions_detailed(analyzer)
        assert first, "expected octahedra distortions"
        assert analyzer._octahedral_distortions_cache
        cache_key = next(iter(analyzer._octahedral_distortions_cache))

        # Second call must hit cache (same object identity for cached dict).
        second = _get_octahedral_distortions_detailed(analyzer)
        assert second is first

        # Fresh uncached compute matches numeric content.
        fresh = _compute_octahedral_distortions_detailed(analyzer)
        assert set(fresh) == set(first)
        for oct_id in first:
            assert first[oct_id]["delta"] == pytest.approx(fresh[oct_id]["delta"])
            assert first[oct_id]["sigma"] == pytest.approx(fresh[oct_id]["sigma"])
            np.testing.assert_allclose(
                first[oct_id]["bond_lengths"],
                fresh[oct_id]["bond_lengths"],
                rtol=1e-12,
                atol=1e-12,
            )

        # analyze() clears cache
        analyzer.analyze(octahedra_centers=["Pb"], valid_halogen=["I"])
        assert analyzer._octahedral_distortions_cache in (None, {})
        _ = cache_key  # silence unused if asserts above passed
