# Twister / multi-slab processing module

Analysis of structures with **independent inorganic stacks (slabs)**: stack
detection, graph ontology extension, and inter-slab stacking registry.

A **twister** is a user/creator-declared label (`structure_type='twister'`, e.g.
from `twist_monolayer` / `q2D_creator.twist`). It is **not** inferred from
geometry. Ordinary RP and DJ unit cells commonly also contain 2+
z-discontinuous slabs separated by the organic spacer bilayer; those structures
are classified as `rp` / `dj` (or other inferred types), while still exposing
the same stack and stacking-registry APIs whenever `n_slabs >= 2`.

## Graph hierarchy

For structures with 2+ independent perovskite stacks:

```
structure_0 (n_slabs=N, is_twister=True only if user-declared)
├── slab_0
│   ├── layer_0 … layer_k
│   └── a_site / spacer (intra-stack)
├── slab_1
│   └── …
└── spacer / a_site (bridges_slabs=(0,1))  ← interface cations/molecules
```

Single-slab structures get one `slab_0` node wrapping all layers. Multi-slab
geometry alone does **not** set `is_twister`.

## Stacking registry

`get_stacking_registry()` measures nearest-neighbor distances across the gap
between two slabs (PBC) for interface X atoms and A-site / interface cations.
These are **distance ratios**, not chemical stoichiometry. The API is available
for **any** structure with `n_slabs >= 2` (RP, DJ, declared twister, …).

The API default stored in `registry.ratio_per_x` is the normalized form

\[
R = \frac{2\, d_{X \to \mathrm{cation}}}{d_{X \to X} + d_{X \to \mathrm{cation}}}
\]

The notebook/raw form \(r = d_{X \to \mathrm{cation}} / d_{X \to X}\) is reconstructed with `unnormalized_ratio(registry)`. \(R < 1\) is I-over-Cs-like; \(R > 1\) is I-over-I-like. Always report `registry.vdw_gap` with the map: both ratios include the vertical gap.

Pass `x_symbols` and `cation_symbols` explicitly for mixed chemistry.

## Public API (via `q2D_analyzer`)

```python
from dataclasses import replace
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.twister_processing.stacking_plots import (
    plot_stacking_heatmap,
    unnormalized_ratio,
    RATIO_LABEL_UNNORMALIZED,
)

analyzer = q2D_analyzer("twisted_bilayer.vasp")  # or any multi-slab CIF
analyzer.analyze()

print(analyzer.structure_type)  # 'twister' only if declared on the structure
print(analyzer.is_twister)      # True only when structure_type == 'twister'
print(analyzer.n_slabs)         # 2+ for multi-slab RP/DJ/twister cells
print(analyzer.get_slabs())     # {'0': {...}, '1': {...}}

analyzer.slabs.layers_of('0').get_bxb(index=0)

registry = analyzer.get_stacking_registry(
    slab_id1='0', slab_id2='1',
    x_symbols=['I'], cation_symbols=['Cs'],
)
analyzer.plot_stacking_heatmap(registry)  # plots normalized R

plot_stacking_heatmap(
    replace(registry, ratio_per_x=unnormalized_ratio(registry)),
    colorbar_label=RATIO_LABEL_UNNORMALIZED,
)
```

## Modules

| File | Role |
|------|------|
| `stack_detection.py` | `detect_stacks()` — z-continuity slab partition |
| `graph_construction.py` | `enrich_graph_with_slabs()` — add `slab_*` nodes |
| `stacking_analysis.py` | `analyze_stack_interface()` — distances and normalized \(R\) |
| `stacking_plots.py` | `plot_stacking_heatmap()`, `unnormalized_ratio()` |

## See also

- [Examples/18_TwisterAnalyzer.md](../../../Examples/18_TwisterAnalyzer.md) — meaning of \(r\)/\(R\), API, five creator maps
- [Examples/6_Twist.md](../../../Examples/6_Twist.md) — building twisted bilayers
- [Examples/generate_twister_stacking_maps.py](../../../Examples/generate_twister_stacking_maps.py) — regenerate documentation maps
