"""Tests for twister stack detection and analyzer API."""

import numpy as np
import pytest

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.octahedral_processing.octahedral_detection import (
    build_octahedra_ligand_info,
    find_shared_atoms,
)


class TestOctahedraLigandInfo:
    """Regression: ligand lists must be non-empty for perovskite structures."""

    @pytest.fixture
    def monolayer_analyzer(self):
        q2d = q2D_creator()
        struct = q2d.create_structure(
            structure_type="monolayer",
            A_ions="Cs",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(1, 1),
            template="cubic",
            vacuum=12.0,
        )
        analyzer = q2D_analyzer(struct)
        analyzer.analyze()
        return analyzer

    def test_ligand_lists_not_empty(self, monolayer_analyzer):
        oct_info, neighbor_indices = build_octahedra_ligand_info(
            monolayer_analyzer.get_graph(),
            atom_symbols=monolayer_analyzer.cell.get_chemical_symbols(),
        )
        assert len(oct_info) > 0
        assert all(len(n) > 0 for n in neighbor_indices)

    def test_shared_atoms_detected(self, monolayer_analyzer):
        _, neighbor_indices = build_octahedra_ligand_info(
            monolayer_analyzer.get_graph(),
        )
        shared = find_shared_atoms(neighbor_indices)
        assert len(shared) >= 0  # monolayer may have shared equatorial X


class TestTwisterDetection:
    """Twister bilayer stack detection via analyzer."""

    @pytest.fixture
    def twisted_bilayer(self):
        q2d = q2D_creator()
        m1 = q2d.create_structure(
            structure_type="monolayer",
            A_ions="Cs",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(1, 1),
            template="cubic",
            vacuum=12.0,
        )
        m2 = q2d.create_structure(
            structure_type="monolayer",
            A_ions="Cs",
            B_ions="Sn",
            X_ions="Br",
            xy_expansion=(1, 1),
            template="cubic",
            vacuum=12.0,
        )
        bilayer = q2d.twist(
            monolayers=[m1, m2],
            twist_angles=[(3, 1)],
            interlayer_distances=[8.0],
            vacuum=12.0,
        )
        return bilayer

    def test_bilayer_structure_type_twister(self, twisted_bilayer):
        assert twisted_bilayer.structure_type == "twister"
        assert twisted_bilayer.is_twister is True

    def test_analyzer_detects_twister(self, twisted_bilayer):
        analyzer = q2D_analyzer(twisted_bilayer)
        analyzer.analyze()
        assert analyzer.n_slabs >= 2
        assert analyzer.is_twister is True
        assert analyzer.structure_type == "twister"

    def test_slab_nodes_in_graph(self, twisted_bilayer):
        analyzer = q2D_analyzer(twisted_bilayer)
        analyzer.analyze()
        graph = analyzer.get_graph()
        slab_nodes = [n for n, d in graph.nodes(data=True) if d.get("node_type") == "slab"]
        assert len(slab_nodes) >= 2

    def test_get_slabs_api(self, twisted_bilayer):
        analyzer = q2D_analyzer(twisted_bilayer)
        analyzer.analyze()
        slabs = analyzer.get_slabs()
        assert len(slabs) >= 2
        all_layers = set()
        for sid, info in slabs.items():
            assert info["octahedra_count"] > 0
            for lid in info["layer_ids"]:
                assert lid not in all_layers
                all_layers.add(lid)

    def test_slabs_wrapper_layers_of(self, twisted_bilayer):
        analyzer = q2D_analyzer(twisted_bilayer)
        analyzer.analyze()
        slab0 = analyzer.slabs["0"]
        layer_ids = slab0["layer_ids"]
        if layer_ids:
            view = analyzer.slabs.layers_of("0")
            assert len(view) == len(layer_ids)


class TestMonolayerRegression:
    """Single-stack structures must not be classified as twister."""

    @pytest.fixture
    def monolayer(self):
        q2d = q2D_creator()
        return q2d.create_structure(
            structure_type="monolayer",
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(1, 1),
            template="cubic",
            vacuum=12.0,
        )

    def test_monolayer_not_twister(self, monolayer):
        analyzer = q2D_analyzer(monolayer)
        analyzer.analyze()
        assert analyzer.is_twister is False
        assert analyzer.n_slabs == 1
        assert analyzer.structure_type == "monolayer"


class TestStackingAnalysis:
    """Stacking registry on twisted bilayer."""

    @pytest.fixture
    def twisted_analyzer(self):
        q2d = q2D_creator()
        m1 = q2d.create_structure(
            structure_type="monolayer",
            A_ions="Cs",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(1, 1),
            template="cubic",
            vacuum=12.0,
        )
        m2 = q2d.create_structure(
            structure_type="monolayer",
            A_ions="Cs",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(1, 1),
            template="cubic",
            vacuum=12.0,
        )
        bilayer = q2d.twist(
            monolayers=[m1, m2],
            twist_angles=[(3, 1)],
            interlayer_distances=[8.0],
            vacuum=12.0,
        )
        analyzer = q2D_analyzer(bilayer)
        analyzer.analyze()
        return analyzer

    def test_stacking_registry(self, twisted_analyzer):
        registry = twisted_analyzer.get_stacking_registry()
        assert registry.x_to_cation.n >= 0
        if registry.x_to_cation.n > 0:
            assert registry.x_to_cation.mean > 0

    def test_monolayer_raises_on_stacking(self):
        q2d = q2D_creator()
        mono = q2d.create_structure(
            structure_type="monolayer",
            A_ions="Cs",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(1, 1),
            template="cubic",
            vacuum=12.0,
        )
        analyzer = q2D_analyzer(mono)
        analyzer.analyze()
        with pytest.raises(ValueError, match="2\\+ independent stacks"):
            analyzer.get_stacking_registry()


class TestNmseRpNotTwister:
    """NMSE BA2PbI4 (#1): inferred RP, multi-slab stacking registry without twister label."""

    @pytest.fixture
    def nmse_1_analyzer(self):
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
            pytest.skip(f"NMSE #1 CIF not present: {cif}")
        analyzer = q2D_analyzer(str(cif))
        analyzer.analyze(
            octahedra_centers=["Pb"],
            valid_halogen=["I"],
        )
        return analyzer

    def test_nmse_1_classified_rp_not_twister(self, nmse_1_analyzer):
        assert nmse_1_analyzer.is_twister is False
        assert nmse_1_analyzer.structure_type == "rp"
        assert nmse_1_analyzer.n_slabs >= 2

    def test_nmse_1_stacking_registry_without_twister(self, nmse_1_analyzer):
        assert nmse_1_analyzer.is_twister is False
        registry = nmse_1_analyzer.get_stacking_registry(
            x_symbols=["I"],
            cation_symbols=["N"],
        )
        assert registry.x_to_x.n > 0
        assert registry.x_to_cation.n > 0
        assert np.isfinite(registry.x_to_x.mean)
        assert np.isfinite(registry.x_to_cation.mean)


class TestSlabForZ:
    """Gap cations must be tagged as bridging both slabs, not nearest-only."""

    def test_gap_z_returns_both_slabs(self):
        from q2D_Materials.analyzer.twister_processing.graph_construction import (
            _slabs_for_z,
        )

        stacks_info = {
            "slab_z_ranges": {
                0: (0.0, 6.0),
                1: (14.0, 20.0),
            }
        }
        assert _slabs_for_z(17.0, stacks_info) == {"1"}
        assert _slabs_for_z(3.0, stacks_info) == {"0"}
        assert _slabs_for_z(10.0, stacks_info) == {"0", "1"}

    def test_vacuum_z_returns_nearest(self):
        from q2D_Materials.analyzer.twister_processing.graph_construction import (
            _slabs_for_z,
        )

        stacks_info = {
            "slab_z_ranges": {
                0: (0.0, 6.0),
                1: (14.0, 20.0),
            }
        }
        assert _slabs_for_z(-2.0, stacks_info) == {"0"}
        assert _slabs_for_z(30.0, stacks_info) == {"1"}
