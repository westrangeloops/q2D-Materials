"""
Utilities for generating perovskite structures from Jagodzinski stacking
sequences before applying Glazer tilts.

This module adapts the layer templates and sequence logic from the PerovGen
project (MIT license, 2024 by mhaefner-chem) to provide programmatic access
to Jagodzinski-based stacking inside the q2D_Materials codebase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from ase import Atoms

# Layer templates (fractional coordinates) taken from PerovGen's `positions`.
# Uppercase letters correspond to A/X layers, lowercase to interlayer B sites.
LAYER_TEMPLATES: Dict[str, List[Tuple[str, float, float]]] = {
    "A": [("A", 0 / 6, 0 / 6), ("X", 3 / 6, 3 / 6), ("X", 3 / 6, 0 / 6), ("X", 0 / 6, 3 / 6)],
    "B": [("A", 4 / 6, 2 / 6), ("X", 4 / 6, 5 / 6), ("X", 1 / 6, 5 / 6), ("X", 1 / 6, 2 / 6)],
    "C": [("A", 2 / 6, 4 / 6), ("X", 2 / 6, 1 / 6), ("X", 5 / 6, 1 / 6), ("X", 5 / 6, 4 / 6)],
    "a": [("B", 0 / 6, 0 / 6)],
    "b": [("B", 4 / 6, 2 / 6)],
    "c": [("B", 2 / 6, 4 / 6)],
}


class JagodzinskiError(ValueError):
    """Raised when a Jagodzinski sequence cannot be processed."""


def _validate_sequence(sequence: str) -> str:
    seq = sequence.strip().lower()
    if not seq:
        raise JagodzinskiError("Jagodzinski sequence must be a non-empty string of 'c' and 'h'.")
    if any(ch not in ("c", "h") for ch in seq):
        raise JagodzinskiError(f"Invalid character in Jagodzinski sequence '{sequence}'. Only 'c'/'h' allowed.")
    return seq


def _add_layer(layer_str: str, char: str, *, final: bool = False) -> str:
    last_layer = layer_str[-1]
    second_last = layer_str[-3] if len(layer_str) >= 3 else layer_str[-1]

    if final:
        candidates = ["A", "B", "C"]
        for candidate in candidates:
            if candidate == last_layer:
                continue
            pair = sorted([last_layer, candidate])
            if pair in (["A", "B"], ["A", "C"], ["B", "C"]):
                new_layer = candidate
                break
        else:
            raise JagodzinskiError(
                f"Unable to close Jagodzinski sequence. Invalid terminal transition from '{last_layer}'."
            )
    else:
        if char == "h":
            new_layer = second_last
        else:  # char == 'c'
            options = ["A", "B", "C"]
            if second_last in options:
                options.remove(second_last)
            if last_layer in options:
                options.remove(last_layer)
            new_layer = options[0] if options else second_last

    pair = sorted([last_layer, new_layer])
    if pair == ["A", "B"]:
        interlayer = "c"
    elif pair == ["A", "C"]:
        interlayer = "b"
    elif pair == ["B", "C"]:
        interlayer = "a"
    else:
        raise JagodzinskiError(f"Invalid layer transition '{last_layer}->{new_layer}' for Jagodzinski sequence.")

    return interlayer + ("" if final else new_layer)


def generate_layer_sequence(sequence: str) -> str:
    """
    Expand a Jagodzinski sequence of 'c'/'h' into alternating layer/interlayer
    symbols (e.g., 'AcBa...' ). Equivalent to PerovGen's get_layer_sequence.
    """
    seq = _validate_sequence(sequence)

    layer_sequence = "AcB"
    for char in seq:
        layer_sequence += _add_layer(layer_sequence, char)
    layer_sequence += _add_layer(layer_sequence, seq[-1], final=True)

    # Remove preset closing "AcB"
    return layer_sequence[:-4]


def _elements_map(A: str, B: str, X: str) -> Dict[str, str]:
    if not all(isinstance(elem, str) for elem in (A, B, X)):
        raise JagodzinskiError("Jagodzinski stacking currently requires string element symbols for A/B/X sites.")
    return {"A": A, "B": B, "X": X}


def _get_vectors(a: float, b: float, c: float, alpha: float, beta: float, gamma: float) -> np.ndarray:
    vec_c = np.array([0.0, 0.0, c])
    vec_a = np.array(
        [
            np.sin(np.deg2rad(beta)) * a,
            0.0,
            np.cos(np.deg2rad(beta)) * a,
        ]
    )

    alpha_spread = np.array(
        [
            np.sin(np.deg2rad(alpha)) * b,
            0.0,
            np.cos(np.deg2rad(alpha)) * b,
        ]
    )
    vec_b = np.array(
        [
            np.cos(np.deg2rad(gamma)) * alpha_spread[0],
            np.sin(np.deg2rad(gamma)) * alpha_spread[0],
            alpha_spread[2],
        ]
    )

    return np.vstack([vec_a, vec_b, vec_c])


@dataclass
class JagodzinskiStructure:
    atoms: Atoms
    layer_sequence: str


def generate_jagodzinski_structure(
    sequence: str,
    elements: Dict[str, str],
    *,
    a_cubic: float = 4.0,
) -> JagodzinskiStructure:
    """
    Build an ASE Atoms object from a Jagodzinski stacking sequence.

    Parameters
    ----------
    sequence : str
        String of 'c'/'h' characters describing stacking.
    elements : dict
        Mapping of {"A": "Ca", "B": "Ti", "X": "O"}.
    a_cubic : float
        Cubic perovskite lattice parameter used to scale the hexagonal cell.
    """
    layer_sequence = generate_layer_sequence(sequence)
    elem_map = _elements_map(elements["A"], elements["B"], elements["X"])

    n_layers = len(layer_sequence)
    if n_layers == 0:
        raise JagodzinskiError("Generated layer sequence is empty.")

    symbols: List[str] = []
    frac_positions: List[Tuple[float, float, float]] = []

    for idx, layer_key in enumerate(layer_sequence):
        layer_template = LAYER_TEMPLATES.get(layer_key)
        if layer_template is None:
            raise JagodzinskiError(f"Unknown layer key '{layer_key}' in generated sequence.")
        z = idx / n_layers
        for site_type, x, y in layer_template:
            symbols.append(elem_map[site_type])
            frac_positions.append((x, y, z))

    a = a_cubic * np.sqrt(2)
    c = n_layers / 2.0 * a_cubic / np.sqrt(3)
    cell = _get_vectors(a, a, c, 90, 90, 120)

    atoms = Atoms(symbols=symbols, scaled_positions=frac_positions, cell=cell, pbc=True)
    return JagodzinskiStructure(atoms=atoms, layer_sequence=layer_sequence)


