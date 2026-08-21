"""Labeled perovskite roundtrips for molecule chemistry validation.

Plant known chemistry labels in creator-built cells, run analyze(), and assert
validate_molecules() recovers the planted label. Non-logical cases are mutated
after the cell is built (creator/RDKit would otherwise add hydrogens).
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.neighborlist import neighbor_list

from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator


def _build_rp() -> Atoms:
    creator = q2D_creator()
    structure = creator.create_structure(
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        structure_type="bulk",
        template="cubic",
        layer_sequence="RP",
        thickness=1,
        xy_expansion=(1, 1),
        spacer="[NH3+]CCC",
    )
    return structure.atoms if hasattr(structure, "atoms") else structure


def _build_dj() -> Atoms:
    creator = q2D_creator()
    structure = creator.create_structure(
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        structure_type="bulk",
        template="cubic",
        layer_sequence="DJ",
        thickness=1,
        xy_expansion=(1, 1),
        spacer="[NH3+]CCCC[NH3+]",
    )
    return structure.atoms if hasattr(structure, "atoms") else structure


def _as_atoms(structure) -> Atoms:
    if isinstance(structure, Atoms) and type(structure) is Atoms:
        return structure
    # q2DStructure subclasses Atoms but slicing needs a plain Atoms
    return Atoms(
        symbols=structure.get_chemical_symbols(),
        positions=structure.get_positions(),
        cell=structure.get_cell(),
        pbc=structure.get_pbc(),
    )


def _delete_all_h(atoms: Atoms) -> Atoms:
    plain = _as_atoms(atoms)
    keep = [i for i, s in enumerate(plain.get_chemical_symbols()) if s != "H"]
    return plain[keep]


def _delete_c_bonded_h(atoms: Atoms) -> Atoms:
    """Remove hydrogens bonded only to carbon (leave NH3 intact)."""
    plain = _as_atoms(atoms)
    symbols = plain.get_chemical_symbols()
    # cutoff covers typical C–H (~1.1) and N–H (~1.0)
    i_idx, j_idx = neighbor_list("ij", plain, 1.3)
    neighbors = {i: [] for i in range(len(plain))}
    for i, j in zip(i_idx, j_idx):
        neighbors[i].append(j)

    drop = set()
    for i, sym in enumerate(symbols):
        if sym != "H":
            continue
        heavy = [j for j in neighbors[i] if symbols[j] != "H"]
        if len(heavy) == 1 and symbols[heavy[0]] == "C":
            drop.add(i)
        elif not heavy:
            dists = plain.get_distances(i, range(len(plain)), mic=True)
            nearest = int(np.argsort(dists)[1])
            if symbols[nearest] == "C" and dists[nearest] < 1.3:
                drop.add(i)

    keep = [i for i in range(len(plain)) if i not in drop]
    assert len(drop) > 0, "expected at least one C-bonded H to delete"
    return plain[keep]


def _has_organic_nodes(analyzer: q2D_analyzer) -> bool:
    graph = analyzer._graph
    for node, data in graph.nodes(data=True):
        if data.get("node_type") not in ("spacer", "a_site"):
            continue
        for neighbor in graph.neighbors(node):
            edge = graph.get_edge_data(node, neighbor) or {}
            if edge.get("edge_type") != "contains":
                continue
            if graph.nodes[neighbor].get("symbol") in ("C", "N", "O"):
                return True
    return False


def _reason_codes(result: dict) -> list[str]:
    codes = []
    for mol in result["molecules"]:
        reason = mol.get("reason", "")
        code = reason.split(":")[0].strip()
        codes.append(code)
    return codes


@pytest.fixture(scope="module")
def rp_atoms():
    return _as_atoms(_build_rp())


@pytest.fixture(scope="module")
def dj_atoms():
    return _as_atoms(_build_dj())


class TestLabeledMoleculeValidity:
    def test_rp_logical_ok(self, rp_atoms):
        """Planted label ok: RP propylammonium + MA, untouched."""
        analyzer = q2D_analyzer(rp_atoms.copy())
        analyzer.analyze()
        assert _has_organic_nodes(analyzer)
        result = analyzer.validate_molecules()
        assert result["n_organic"] > 0
        assert result["is_valid"] is True, result
        assert result["n_failed"] == 0
        assert all(c == "ok" or mol["is_valid"] for mol, c in zip(
            result["molecules"], _reason_codes(result)
        ))

    def test_dj_logical_ok(self, dj_atoms):
        """Planted label ok: DJ BDA, untouched (includes ammonium N degree 4)."""
        analyzer = q2D_analyzer(dj_atoms.copy())
        analyzer.analyze()
        assert _has_organic_nodes(analyzer)
        result = analyzer.validate_molecules()
        assert result["n_organic"] > 0
        assert result["is_valid"] is True, result
        assert result["n_failed"] == 0

    def test_rp_delete_all_h_missing_hydrogens(self, rp_atoms):
        """Planted label missing_hydrogens: strip all H after build."""
        mutated = _delete_all_h(rp_atoms.copy())
        assert "H" not in mutated.get_chemical_symbols()
        analyzer = q2D_analyzer(mutated)
        analyzer.analyze()
        result = analyzer.validate_molecules()
        assert result["is_valid"] is False, result
        codes = _reason_codes(result)
        assert any(c == "missing_hydrogens" for c in codes), codes

    def test_rp_delete_c_h_bad_valence(self, rp_atoms):
        """Planted label bad_valence: strip C–H only, leave NH3."""
        mutated = _delete_c_bonded_h(rp_atoms.copy())
        assert "H" in mutated.get_chemical_symbols()
        analyzer = q2D_analyzer(mutated)
        analyzer.analyze()
        result = analyzer.validate_molecules()
        assert result["is_valid"] is False, result
        codes = _reason_codes(result)
        assert any(c == "bad_valence" for c in codes), codes


class TestMolValidateChemistryOnly:
    def test_smiles_chemistry_ok(self):
        analyzer = q2D_analyzer()
        result = analyzer.mol_validate("C[NH3+]", spacer_type=None)
        assert result.is_valid is True
        assert result.reason == "ok"

    def test_smiles_dj_still_works(self):
        analyzer = q2D_analyzer()
        result = analyzer.mol_validate(
            "NCCCCN",
            spacer_type="DJ",
            initial_pattern="[NH2]C",
            final_pattern="[NH2]C",
        )
        assert result.is_valid is True

    def test_aromatic_smiles_ok(self):
        analyzer = q2D_analyzer()
        # phenyl-like ammonium: degree-3 aromatic C must pass stage 1
        result = analyzer.mol_validate("c1ccccc1C[NH3+]", spacer_type=None)
        assert result.is_valid is True, result.reason


class TestHydrogenCleanup:
    def test_merge_same_parent_duplicates_mean_position(self):
        from q2D_Materials.utils.molecules.hydrogen_cleanup import (
            merge_disordered_hydrogens,
            merge_close_hydrogens,
        )

        # C with three near-identical H sites (true split-site, < 0.01 Å)
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [1.09, 0.005, 0.0],
                [1.09, -0.003, 0.004],
            ]
        )
        atoms = Atoms(
            symbols=["C", "H", "H", "H"],
            positions=positions,
            cell=[20, 20, 20],
            pbc=True,
        )
        cleaned, report = merge_disordered_hydrogens(atoms)
        assert report["n_h_merged"] == 2
        assert report["n_clusters_merged"] == 1
        assert cleaned.get_chemical_symbols().count("H") == 1
        # Kept H sits near the mean of the three disordered sites
        h_pos = cleaned.get_positions()[1]
        expected = positions[1:].mean(axis=0)
        assert np.allclose(h_pos, expected, atol=1e-6)

        # Alias still returns (atoms, n_removed)
        _, n = merge_close_hydrogens(atoms)
        assert n == 2

    def test_occupancy_weighted_keep(self):
        from q2D_Materials.utils.molecules.hydrogen_cleanup import merge_disordered_hydrogens

        atoms = Atoms(
            symbols=["C", "H", "H"],
            positions=[[0, 0, 0], [1.0, 0, 0], [1.005, 0, 0]],
            cell=[20, 20, 20],
            pbc=True,
        )
        atoms.set_array("occupancy", np.array([1.0, 0.3, 0.7]))
        cleaned, report = merge_disordered_hydrogens(atoms)
        assert report["n_h_merged"] == 1
        # Occupancy-weighted centroid: 0.3*1.0 + 0.7*1.005 = 1.0035
        assert abs(cleaned.get_positions()[1][0] - 1.0035) < 1e-6

    def test_does_not_merge_beyond_default_tolerance(self):
        from q2D_Materials.utils.molecules.hydrogen_cleanup import merge_disordered_hydrogens

        # Same parent, but 0.1 Å apart — not a duplicate site under 0.01 Å default
        atoms = Atoms(
            symbols=["C", "H", "H"],
            positions=[[0, 0, 0], [1.09, 0, 0], [1.19, 0, 0]],
            cell=[20, 20, 20],
            pbc=True,
        )
        cleaned, report = merge_disordered_hydrogens(atoms)
        assert report["n_h_merged"] == 0
        assert len(cleaned) == 3

    def test_does_not_merge_different_parents(self):
        from q2D_Materials.utils.molecules.hydrogen_cleanup import merge_close_hydrogens

        atoms = Atoms(
            symbols=["C", "H", "C", "H"],
            positions=[[0, 0, 0], [1.09, 0, 0], [3.0, 0, 0], [4.09, 0, 0]],
            cell=[20, 20, 20],
            pbc=True,
        )
        cleaned, n = merge_close_hydrogens(atoms)
        assert n == 0
        assert len(cleaned) == 4

    def test_select_cif_prefers_iodide_with_h(self, tmp_path):
        from q2D_Materials.utils.molecules.hydrogen_cleanup import select_cif_atoms

        # Minimal two-block CIF: last block is Br without H (ASE default trap)
        cif = tmp_path / "test_PbI4.cif"
        cif.write_text(
            """
data_iodide
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Pb Pb 0 0 0
I1 I 0.5 0 0
H1 H 0.2 0.2 0.2

data_bromide
_cell_length_a 10
_cell_length_b 10
_cell_length_c 10
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Pb Pb 0 0 0
Br1 Br 0.5 0 0
"""
        )
        atoms, meta = select_cif_atoms(str(cif), hint=str(cif))
        assert meta["cif_blocks_considered"] == 2
        assert atoms.get_chemical_symbols().count("I") == 1
        assert atoms.get_chemical_symbols().count("H") == 1
        assert atoms.get_chemical_symbols().count("Br") == 0

    def test_prepare_experimental_structure(self):
        from q2D_Materials.utils.molecules.hydrogen_cleanup import (
            prepare_experimental_structure,
        )

        atoms = Atoms(
            symbols=["C", "H", "H", "H"],
            positions=[
                [0, 0, 0],
                [1.09, 0, 0],
                [1.09, 0.005, 0],
                [1.09, -0.003, 0.004],
            ],
            cell=[20, 20, 20],
            pbc=True,
        )
        cleaned, report = prepare_experimental_structure(atoms)
        assert report["source_type"] == "atoms"
        assert report["n_h_merged"] == 2
        assert cleaned.get_chemical_symbols().count("H") == 1
