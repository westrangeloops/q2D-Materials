"""
Jagodzinski (c/h) string to explicit layer sequence converter.

Takes a Jagodzinski code of cubic/hex stacking (`c` / `h`) and expands it
to the explicit layer + interlayer sequence expected by the named-layer
templates (e.g., `jagodinxky.json` which defines A, B, C, a, b, c).

- Seed with `AcB` (layer/interlayer/layer)
- For each code character:
  - `h`: repeat the layer from two steps back (ABA-style)
  - `c`: choose the third layer not used by the previous two (ABC-style)
  - Insert the appropriate interlayer (a, b, or c) between successive layers
- Close with a final interlayer back toward the reference layer (default B)
"""

from typing import Iterable, List


_LAYERS: tuple[str, str, str] = ("A", "B", "C")
# Interlayer mapping between successive layer letters
_INTER = {
    ("A", "B"): "c",
    ("B", "A"): "c",
    ("A", "C"): "b",
    ("C", "A"): "b",
    ("B", "C"): "a",
    ("C", "B"): "a",
}


def _third_layer(prev: str, prev_prev: str) -> str:
    """Return the layer in A/B/C that is not one of the previous two."""
    for layer in _LAYERS:
        if layer not in (prev, prev_prev):
            return layer
    raise ValueError("Could not determine third layer; inputs must be in A/B/C.")


def _interlayer(layer_a: str, layer_b: str) -> str:
    """Map successive layers to the corresponding interlayer (a, b, or c)."""
    key = (layer_a, layer_b)
    if key not in _INTER:
        raise ValueError(f"No interlayer mapping for pair {key}.")
    return _INTER[key]


def jag_to_layers(
    code: Iterable[str],
    *,
    seed: tuple[str, str, str] = ("A", "c", "B"),
    close_with: str = "B",
) -> List[str]:
    """
    Convert a Jagodzinski c/h string into an explicit layer_sequence list.

    Parameters
    ----------
    code : Iterable[str]
        Sequence of characters containing only 'c' or 'h'.
    seed : tuple[str, str, str], optional
        Starting trio (layer, interlayer, layer), default ("A", "c", "B")
        to match the inspiration script's AcB seed.
    close_with : str, optional
        Reference layer to close the stack with a final interlayer (default "B").

    Returns
    -------
    list[str]
        Expanded sequence of layer and interlayer names, e.g.,
        ['A', 'c', 'B', 'a', 'C', 'b'] for code='ch'.
    """
    seq: List[str] = list(seed)
    if len(seq) != 3:
        raise ValueError("Seed must have exactly three entries: layer, interlayer, layer.")
    if seq[0] not in _LAYERS or seq[2] not in _LAYERS:
        raise ValueError("Seed layers must be one of A/B/C.")

    for ch in code:
        if ch not in ("c", "h"):
            raise ValueError("Jagodzinski code must contain only 'c' or 'h'.")

        prev_layer = seq[-1]       # last layer placed
        prev_prev_layer = seq[-3]  # layer before the interlayer

        if ch == "h":
            new_layer = prev_prev_layer  # repeat layer (ABA)
        else:  # ch == "c"
            new_layer = _third_layer(prev_layer, prev_prev_layer)  # ABC

        inter = _interlayer(prev_layer, new_layer)
        seq.extend([inter, new_layer])

    # Close with final interlayer toward the reference layer
    inter_close = _interlayer(seq[-1], close_with)
    seq.append(inter_close)
    return seq

