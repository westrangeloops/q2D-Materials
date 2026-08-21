"""Parity tests for vectorized detect_bonds vs the legacy pairwise loop."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import pytest

from q2D_Materials.utils.geometry.pbc_distances import calculate_pbc_distances
from q2D_Materials.utils.properties.atomic_properties import (
    are_atoms_bonded,
    detect_bonds,
    estimate_bond_order,
    _enforce_hydrogen_single_bond,
)


def _detect_bonds_reference(
    symbols: List[str],
    positions: np.ndarray,
    tolerance: float = 0.45,
    cell: Optional[np.ndarray] = None,
    pbc: Union[bool, List[bool], None] = None,
    enforce_hydrogen_rules: bool = True,
    estimate_bond_orders: bool = False,
) -> List[Tuple[int, int, float, int]]:
    """Legacy O(N^2) detect_bonds used as an oracle for parity tests."""
    bonds: List[Tuple[int, int, float, int]] = []
    n_atoms = len(symbols)
    positions = np.asarray(positions, dtype=np.float64)

    for i in range(n_atoms):
        for j in range(i + 1, n_atoms):
            if symbols[i] == "H" and symbols[j] == "H":
                continue
            if cell is not None:
                pbc_arg = True if pbc is None else pbc
                distance = float(
                    calculate_pbc_distances(
                        positions[i],
                        positions[j : j + 1],
                        cell,
                        pbc=pbc_arg,
                        mode="auto",
                    )[0]
                )
            else:
                distance = float(np.linalg.norm(positions[i] - positions[j]))

            if are_atoms_bonded(distance, symbols[i], symbols[j], tolerance):
                if estimate_bond_orders:
                    bond_order = estimate_bond_order(
                        distance, symbols[i], symbols[j], tolerance=0.15
                    )
                else:
                    bond_order = 1
                bonds.append((i, j, distance, bond_order))

    if enforce_hydrogen_rules:
        bonds = _enforce_hydrogen_single_bond(bonds, symbols)
    return bonds


def _bond_pair_set(bonds: List[Tuple[int, int, float, int]]):
    return {(min(i, j), max(i, j)) for i, j, _d, _o in bonds}


def _assert_bonds_match(
    actual: List[Tuple[int, int, float, int]],
    expected: List[Tuple[int, int, float, int]],
    *,
    rtol: float = 1e-9,
    atol: float = 1e-9,
) -> None:
    assert _bond_pair_set(actual) == _bond_pair_set(expected)
    actual_map = {(min(i, j), max(i, j)): (d, o) for i, j, d, o in actual}
    expected_map = {(min(i, j), max(i, j)): (d, o) for i, j, d, o in expected}
    for key, (d_exp, o_exp) in expected_map.items():
        d_act, o_act = actual_map[key]
        assert o_act == o_exp
        assert d_act == pytest.approx(d_exp, rel=rtol, abs=atol)


class TestDetectBondsParity:
    def test_methane_like_no_pbc(self):
        # Rough tetrahedral CH4 geometry (C at origin).
        symbols = ["C", "H", "H", "H", "H"]
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.09, 0.0, 0.0],
                [-0.36, 1.03, 0.0],
                [-0.36, -0.51, 0.89],
                [-0.36, -0.51, -0.89],
            ],
            dtype=np.float64,
        )
        actual = detect_bonds(symbols, positions, enforce_hydrogen_rules=True)
        expected = _detect_bonds_reference(
            symbols, positions, enforce_hydrogen_rules=True
        )
        _assert_bonds_match(actual, expected)
        # Four C–H bonds; no H–H.
        assert len(actual) == 4
        assert all("H" in (symbols[i], symbols[j]) for i, j, *_ in actual)

    def test_ethane_no_pbc_skips_hh(self):
        symbols = ["C", "C", "H", "H", "H", "H", "H", "H"]
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.54, 0.0, 0.0],
                [-0.5, 0.9, 0.0],
                [-0.5, -0.45, 0.78],
                [-0.5, -0.45, -0.78],
                [2.04, 0.9, 0.0],
                [2.04, -0.45, 0.78],
                [2.04, -0.45, -0.78],
            ],
            dtype=np.float64,
        )
        actual = detect_bonds(symbols, positions)
        expected = _detect_bonds_reference(symbols, positions)
        _assert_bonds_match(actual, expected)
        assert (0, 1) in _bond_pair_set(actual)

    def test_pbc_wraps_across_cell(self):
        # Two carbons ~1.5 Å apart across a cubic cell face → bonded via PBC.
        cell = np.eye(3) * 5.0
        symbols = ["C", "C", "H", "H"]
        positions = np.array(
            [
                [0.2, 2.5, 2.5],
                [3.7, 2.5, 2.5],  # wrap: 0.2 + (5.0 - 3.7) = 1.5 Å
                [0.2, 3.5, 2.5],
                [3.7, 3.5, 2.5],
            ],
            dtype=np.float64,
        )

        actual = detect_bonds(symbols, positions, cell=cell, pbc=True)
        expected = _detect_bonds_reference(symbols, positions, cell=cell, pbc=True)
        _assert_bonds_match(actual, expected)
        assert (0, 1) in _bond_pair_set(actual)

    def test_hydrogen_single_bond_rule(self):
        # H equidistant-ish between two carbons: keep only shortest H bond.
        symbols = ["C", "C", "H"]
        positions = np.array(
            [
                [0.0, 0.0, 0.0],
                [3.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],  # closer to C0
            ],
            dtype=np.float64,
        )
        actual = detect_bonds(
            symbols, positions, tolerance=2.0, enforce_hydrogen_rules=True
        )
        expected = _detect_bonds_reference(
            symbols, positions, tolerance=2.0, enforce_hydrogen_rules=True
        )
        _assert_bonds_match(actual, expected)
        h_bonds = [(i, j) for i, j, *_ in actual if 2 in (i, j)]
        assert len(h_bonds) == 1
        assert set(h_bonds[0]) == {0, 2}

    def test_estimate_bond_orders_parity(self):
        symbols = ["C", "C"]
        positions = np.array([[0.0, 0.0, 0.0], [1.20, 0.0, 0.0]], dtype=np.float64)
        actual = detect_bonds(
            symbols,
            positions,
            estimate_bond_orders=True,
            enforce_hydrogen_rules=False,
        )
        expected = _detect_bonds_reference(
            symbols,
            positions,
            estimate_bond_orders=True,
            enforce_hydrogen_rules=False,
        )
        _assert_bonds_match(actual, expected)

    def test_empty_and_singleton(self):
        assert detect_bonds([], np.zeros((0, 3))) == []
        assert detect_bonds(["C"], np.array([[0.0, 0.0, 0.0]])) == []
