![q2D-Materials Logo](../Logos/logo.png)

# 4. Jagodzinski stacking — a plain-language guide

Jagodzinski is just a compact spelling of stacking. Uppercase letters are A-site layers (`A`, `B`, `C`); lowercase letters are the B-only interlayers (`a`, `b`, `c`). Crystallographers like `c/h` codes: `c` means “take the third layer you’re not using” (ABC turn), `h` means “go back to the layer from two steps ago” (ABA repeat). Our templates need the explicit letters, so we translate once and then build.

### A single, worked example (with frames)
```python
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.utils import jag_to_layers

code = "chc"               # c/h Jagodzinski code
seq = jag_to_layers(code)  # -> ['A','c','B','a','C','b','A','c']

q2d = q2D_creator()
bulk = q2d.create_perovskite(
    structure_type='bulk',
    A_ions='MA', B_ions='Pb', X_ions='I',
    xy_expansion=(1, 1),
    template='jagodinxky',
    layer_sequence=seq,
)
```

If you prefer seeing it, here’s the storyboard: start from a B-only interlayer (`a`), add an A-layer (`C`), then insert another B-only interlayer (`c`).
![Jagodzinski frames](./images/jago-a.png) ![Jagodzinski frames](./images/jago-aC.png) ![Jagodzinski frames](./images/jago-aCc.png)

What the helper is really doing: seed with `AcB`, take `c` as the “third layer” step, `h` as the “repeat two steps back” step, drop the right interlayer (`a`, `b`, or `c`) every time you move, and close toward `B` so the sequence stays consistent with the crystallographic code.

### The template you’re feeding (jagodinxky)

```json
{
  "name": "jagodinxky",
  "named_layers": {
    "A": [["A", 0.0, 0.0], ["X", 0.5, 0.5], ...],
    "B": [["A", 0.666, 0.333], ["X", 0.666, 0.833], ...],
    "C": [["A", 0.333, 0.666], ["X", 0.333, 0.166], ...],
    "a": [["B", 0.0, 0.0]],
    "b": [["B", 0.666, 0.333]],
    "c": [["B", 0.333, 0.666]]
  },
  "layer_sequence": ["A", "B", "C"],
  "lattice_multipliers": [1.5, 1.5],
  "angles": [90.0, 90.0, 120.0]
}
```

### Things to keep in mind
- Strings or lists are equivalent: `"ABc"` == `["A","B","c"]`; separators like `- ,` are ignored.
- Case matters: `A` ≠ `a`.
- In monolayers, `thickness` just repeats your `layer_sequence`.
- Jagodzinski only sets stacking order; you can still mix compositions, tilts, spacers, or penetration.

Regenerate the frames any time with:
```bash
nix develop -c python3 Examples/plot.py
```
