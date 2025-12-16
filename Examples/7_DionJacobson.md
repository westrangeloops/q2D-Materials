![q2D-Materials Logo](../Logos/logo.png)

# 7. Dion–Jacobson spacers (DJ)

Use DJ spacers when you need a molecule bridging two adjacent spacer layers (S# ↔ S#). This guide is short and practical.

## How it works
- Templates carry spacer sites as `S#` entries in consecutive layers. Any back‑to‑back layers that both contain `S#` sites are treated as a DJ gap.
- XY expansion auto‑relabels `S#` per image (S1→S3…); each label is injective so pairing is unambiguous.
- The gap between spacer layers uses the spacer N–N distance (fallback `2*BX_dist` if unknown).
- Glazer tilting now happens before population; spacer anchors are kept and wrapped.

## Minimal recipe
```python
from q2D_Materials.core.creator import q2D_creator
q2d = q2D_creator()

dj = q2d.create_structure(
    structure_type="bulk",
    template="cubic",          # or reduced
    layer_sequence="DJ",       # expands to L1-L2-...-M1-M1
    thickness=2,
    A_ions="MA", B_ions="Pb", X_ions="I",
    sharp_spacer="[NH3+]CCCC[NH3+]",  # SMILES or Atoms; list cycles
    glazer_angles=[0, 0, 3],
    glazer_pattern=["0", "0", "+"],
)
```

Render (from `Examples/plot.py`):

- Bulk DJ with spacer (isometric): `Examples/images/dj_bulk.png`
- Bulk DJ with atomic spacer (Cs, isometric): `Examples/images/dj_atomic.png`

![Bulk DJ spacer](images/dj_bulk.png)

![Bulk DJ atomic spacer (Cs)](images/dj_atomic.png)

## Atomic spacer example
```python
from q2D_Materials.core.creator import q2D_creator
q2d = q2D_creator()

dj_atomic = q2d.create_structure(
    structure_type="bulk",
    template="cubic",
    layer_sequence="DJ",
    thickness=3,
    A_ions="MA",
    B_ions="Pb",
    X_ions="I",
    sharp_spacer="Cs",
    glazer_angles=[3, 2, 6],
    glazer_pattern=["+", "-", "-"],
)
```

## Tips
- Provide `sharp_spacer` as SMILES or pre‑built ASE `Atoms`; lists cycle across S# labels.
- `penetration` shifts Ap/S# along ±c by `BX_dist * penetration` (per Ap site, uniform for S#).
- `attachment_end` controls whether L1 A sites convert to Ap (bottom/top/both) for monolayers; DJ spacers rely on `S#` only.
- For custom sequences, include two consecutive spacer layers (any names) that each contain `S#` sites; pairing is label-based, not name-based.

## Troubleshooting
- Spacer missing: ensure the template has S# sites on two consecutive layers and `sharp_spacer` is a molecule (not a bare atom).
- Overlap after tilting: reduce Glazer angles or increase `BX_dist`; large tilts can crowd X sites near spacer anchors.

