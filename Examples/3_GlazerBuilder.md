![q2D-Materials Logo](../Logos/logo.png)

# 3. Glazer tilting — rotations in plain language

Glazer tilting is just rotating the X-site anions around each B-site center. You choose angles `[x, y, z]` (degrees) and a sign pattern that says whether adjacent octahedra rotate in-phase (`+`), out-of-phase (`-`), or not at all (`0`). Zero angle and `'0'` go together.

## Specifying Glazer Patterns

The `glazer_pattern` parameter accepts two formats:

1. **List format** (explicit): `["0", "0", "+"]` — requires `glazer_angles` to be provided
2. **String format** (convenient): `"a-b+a-"` or `"Pnma"` — automatically suggests angles (default 10°)

### List Format (Explicit Control)

When using a list, you must provide both `glazer_pattern` and `glazer_angles`:

```python
from q2D_Materials.core.creator import q2D_creator
q2d = q2D_creator()

# Untilted structure
untilted = q2d.create_structure(
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(1, 1), template="cubic",
)

# Tilted structure with explicit angles and pattern
tilting = q2d.create_structure(
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(1, 1), template="cubic",
    glazer_angles=[0, 0, 10],    # +10° about z-axis
    glazer_pattern=["0", "0", "+"],  # No tilt on x/y, in-phase on z
)
```

### String Format (Convenient)

When using a string, you can specify either:
- **Glazer notation** (e.g., `"a-b+a-"`): Standard 6-character notation
- **Space group symbol** (e.g., `"Pnma"`): Automatically resolves to notation

Angles default to 10° for nonzero tilts, but you can override them:

```python
# Using Glazer notation string (defaults to 10° angles)
pnma_style = q2d.create_structure(
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    glazer_pattern="a-b+a-",  # Automatically uses [10, 10, 10] angles
)

# Custom angles with notation string
pnma_custom = q2d.create_structure(
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    glazer_pattern="a-b+a-",
    glazer_angles=[8, 8, 12],  # Override default angles
)

# Using Space Group symbol (case insensitive)
pnma_by_group = q2d.create_structure(
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    glazer_pattern="Pnma",    # Resolves to "a-b+a-" with [10, 10, 10] angles
)

# Cubic (untilted) structure
cubic = q2d.create_structure(
    structure_type="bulk",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(1, 1), template="cubic",
    glazer_pattern="Pm-3m",  # or "a0a0a0"
)
```

**Complete Space Group to Glazer Notation Mapping (All 15 Systems):**

The following table lists all 15 conventional Glazer tilt systems from Howard & Stokes (1998):

| # | Space Group | Glazer Notation | System | Template Used |
|---|-------------|----------------|--------|---------------|
| 1 | `Pm-3m` | `a0a0a0` | Cubic | cubic |
| 2 | `P4/mbm` | `a0a0c+` | Tetragonal | cubic |
| 3 | `I4/mmm` | `a0b+b+` | Tetragonal | cubic |
| 4 | `Im-3` | `a+a+a+` | Cubic | cubic |
| 5 | `Immm` | `a+b+c+` | Orthorhombic | cubic |
| 6 | `I4/mcm` | `a0a0c-` | Tetragonal | cubic |
| 7 | `Imma` | `a0b-b-` | Orthorhombic | cubic |
| 8 | `R-3c` | `a-a-a-` | Rhombohedral | cubic |
| 9 | `C2/m` | `a0b-c-` | Monoclinic | cubic |
| 10 | `C2/c` | `a-b-b-` | Monoclinic | cubic |
| 11 | `P-1` | `a-b-c-` | Triclinic | reduced |
| 12 | `Cmcm` | `a0b+c-` | Orthorhombic | reduced |
| 13 | `Pnma` | `a+b-b-` | Orthorhombic | reduced |
| 14 | `P21/m` | `a+b-c-` | Monoclinic | reduced |
| 15 | `P42/nmc` | `a+a+c-` | Tetragonal | reduced |

**Note:** All structures are generated as **monolayers** with vacuum spacing. The first 10 systems use the `cubic` template, while systems 11-15 use the `reduced` template for better visualization of their complex tilt patterns. All visualizations use isometric views (`45x,45y,45z`) to show the full 3D tilt patterns.

### Visual Examples

**Basic tilting (top-down, 4×4 cell):**
- First frame: no Glazer tilt (reference) — `glazer_angles=[0,0,0]`
- Second frame: positive tilt about z — `glazer_angles=[0,0,10]`, `glazer_pattern=['0','0','+']`

![untilted](./images/glazer-untitled-top.png) ![tilted](./images/glazer-tilted-top.png)

### Understanding Viewing Angles and Axis-Specific Tilting

The **viewing angle** is crucial for visualizing Glazer tilting because tilts occur about specific axes (x, y, or z). Different viewing angles reveal different aspects of the tilt pattern:

- **[0 0 1] (Top view)**: Looking down the z-axis → best for seeing tilts about x and y axes
- **[0 1 0] (Front view)**: Looking along the y-axis → best for seeing tilts about x and z axes  
- **[1 0 0] (Side view)**: Looking along the x-axis → best for seeing tilts about y and z axes

Each of the 15 space group visualizations is shown from all three viewing angles in the gallery table below, allowing you to see how tilting appears from different perspectives and identify which axes have tilts.

## Understanding Glazer Notation

Glazer notation is a 6-character string describing octahedral tilting:
- **Magnitude letters** (`a`, `b`, `c`): Relative tilt magnitudes (a ≤ b ≤ c)
- **Phase signs** (`+`, `-`, `0`): Correlation between adjacent octahedra
  - `+`: In-phase (same sense) — neighbors rotate the same way
  - `-`: Out-of-phase (opposite sense) — neighbors alternate
  - `0`: No tilt on this axis

### Pattern Examples

- `a0a0a0`: No tilting (cubic, Pm-3m)
- `a0a0c+`: Tilt only about z-axis, in-phase (tetragonal, P4/mbm)
- `a0a0c-`: Tilt only about z-axis, out-of-phase (tetragonal, I4/mcm)
- `a-b+a-`: Different tilts on all axes, mixed phases (orthorhombic, Pnma)
- `a-a-a-`: Equal tilts on all axes, all out-of-phase (rhombohedral, R-3c)

### Understanding Tilt Angles and Axes

Glazer tilting rotates octahedra about three orthogonal axes (x, y, z). The notation `[angle_x, angle_y, angle_z]` specifies the rotation angle in degrees about each axis:

- **`angle_x`**: Rotation about the x-axis (left-right tilting)
- **`angle_y`**: Rotation about the y-axis (front-back tilting)  
- **`angle_z`**: Rotation about the z-axis (in-plane rotation, most visible from top view)

**Important**: The viewing angle determines which tilts are most visible:
- Tilts about the **z-axis** are most visible from **top view** (looking down)
- Tilts about **x or y axes** are most visible from **side views** (looking along perpendicular axes)
- **Isometric views** show all tilts simultaneously but may obscure individual axis contributions

**Example patterns:**
- `a0a0c-`: Only z-axis tilt → best viewed from top or side
- `a0b-b-`: Y and z axis tilts → best viewed from front or isometric
- `a+b-b-`: All three axes with different phases → requires isometric view to see full pattern

### Picking Patterns and Supercells

- **In-phase (`+`) patterns**: Can work with 1×1 supercells
- **Out-of-phase (`-`) patterns**: Usually need 2×2 or larger supercells to accommodate alternation
- **Angles**: Typically small (0–15°) for perovskites. Default is 10° when using string notation
- **Magnitude equivalence**: The system automatically detects when tilt angles are similar and assigns the same letter (a, b, or c)
- **Viewing recommendations**: The visualization automatically selects the best viewing axis:
  - **[0 0 1]** (top view): When z-axis has no tilt, to see x and y tilts
  - **[0 1 0]** (front view): When y-axis has no tilt, to see x and z tilts
  - **[1 0 0]** (side view): When x-axis has no tilt, to see y and z tilts

### Detecting Glazer Patterns

You can also detect Glazer patterns from existing structures using the analyzer:

```python
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer.analyzer_class import q2D_analyzer
from ase.io import read

# Create a structure with Glazer tilting
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    thickness=2, vacuum=15.0,
    glazer_pattern="a+b-b-",  # Pnma notation
)

# Analyze and detect the pattern
analyzer = q2D_analyzer(structure)
glazer_info = analyzer.get_glazer_pattern()

print(f"Detected notation: {glazer_info['notation']}")
print(f"Tilt angles: {glazer_info['tilt_angles']}")
print(f"Space group: {glazer_info['space_group']}")
```

The analyzer uses tolerance-based detection, so you can adjust sensitivity:

```python
# More sensitive detection (smaller tolerance)
glazer_info = analyzer.get_glazer_pattern(tolerance=0.05)

# Less sensitive (larger tolerance, for noisy structures)
glazer_info = analyzer.get_glazer_pattern(tolerance=0.5)
```

### Combining with Everything Else

Tilting is applied after the template is built, so it works with:
- Any template (cubic, reduced, custom)
- Jagodzinski layer sequences
- Spacers and pattern mixing
- Monolayers and bulk structures

Just ensure your `glazer_pattern` matches where angles are nonzero. For out-of-phase patterns, use `xy_expansion=(2, 2)` or larger.

### All 15 Glazer Space Group Visualizations

All 15 space group systems are visualized as **monolayer structures** with three orthogonal viewing angles to demonstrate how tilting appears from different perspectives. Each space group is shown from three axis views:
- **[0 0 1]** (Top view): Looking down the z-axis
- **[0 1 0]** (Front view): Looking along the y-axis  
- **[1 0 0]** (Side view): Looking along the x-axis

Each structure uses optimized tilt magnitudes based on the Glazer notation, and comparing the three views reveals which axes have tilts and how they manifest from different perspectives.

**Gallery of all 15 systems (three views per system):**

Each space group is shown from three viewing angles to demonstrate how different perspectives reveal tilting on different axes:

| System | Space Group | Glazer Notation | [0 0 1]<br/>Top View | [0 1 0]<br/>Front View | [1 0 0]<br/>Side View |
|--------|-------------|----------------|---------------------|----------------------|---------------------|
| 1 | Pm-3m | a0a0a0 | ![Pm-3m [001]](./images/glazer_01_pm-3m_001.png) | ![Pm-3m [010]](./images/glazer_01_pm-3m_010.png) | ![Pm-3m [100]](./images/glazer_01_pm-3m_100.png) |
| 2 | P4/mbm | a0a0c+ | ![P4/mbm [001]](./images/glazer_02_p4_mbm_001.png) | ![P4/mbm [010]](./images/glazer_02_p4_mbm_010.png) | ![P4/mbm [100]](./images/glazer_02_p4_mbm_100.png) |
| 3 | I4/mmm | a0b+b+ | ![I4/mmm [001]](./images/glazer_03_i4_mmm_001.png) | ![I4/mmm [010]](./images/glazer_03_i4_mmm_010.png) | ![I4/mmm [100]](./images/glazer_03_i4_mmm_100.png) |
| 4 | Im-3 | a+a+a+ | ![Im-3 [001]](./images/glazer_04_im-3_001.png) | ![Im-3 [010]](./images/glazer_04_im-3_010.png) | ![Im-3 [100]](./images/glazer_04_im-3_100.png) |
| 5 | Immm | a+b+c+ | ![Immm [001]](./images/glazer_05_immm_001.png) | ![Immm [010]](./images/glazer_05_immm_010.png) | ![Immm [100]](./images/glazer_05_immm_100.png) |
| 6 | I4/mcm | a0a0c- | ![I4/mcm [001]](./images/glazer_06_i4_mcm_001.png) | ![I4/mcm [010]](./images/glazer_06_i4_mcm_010.png) | ![I4/mcm [100]](./images/glazer_06_i4_mcm_100.png) |
| 7 | Imma | a0b-b- | ![Imma [001]](./images/glazer_07_imma_001.png) | ![Imma [010]](./images/glazer_07_imma_010.png) | ![Imma [100]](./images/glazer_07_imma_100.png) |
| 8 | R-3c | a-a-a- | ![R-3c [001]](./images/glazer_08_r-3c_001.png) | ![R-3c [010]](./images/glazer_08_r-3c_010.png) | ![R-3c [100]](./images/glazer_08_r-3c_100.png) |
| 9 | C2/m | a0b-c- | ![C2/m [001]](./images/glazer_09_c2_m_001.png) | ![C2/m [010]](./images/glazer_09_c2_m_010.png) | ![C2/m [100]](./images/glazer_09_c2_m_100.png) |
| 10 | C2/c | a-b-b- | ![C2/c [001]](./images/glazer_10_c2_c_001.png) | ![C2/c [010]](./images/glazer_10_c2_c_010.png) | ![C2/c [100]](./images/glazer_10_c2_c_100.png) |
| 11 | P-1 | a-b-c- | ![P-1 [001]](./images/glazer_11_p-1_001.png) | ![P-1 [010]](./images/glazer_11_p-1_010.png) | ![P-1 [100]](./images/glazer_11_p-1_100.png) |
| 12 | Cmcm | a0b+c- | ![Cmcm [001]](./images/glazer_12_cmcm_001.png) | ![Cmcm [010]](./images/glazer_12_cmcm_010.png) | ![Cmcm [100]](./images/glazer_12_cmcm_100.png) |
| 13 | Pnma | a+b-b- | ![Pnma [001]](./images/glazer_13_pnma_001.png) | ![Pnma [010]](./images/glazer_13_pnma_010.png) | ![Pnma [100]](./images/glazer_13_pnma_100.png) |
| 14 | P21/m | a+b-c- | ![P21/m [001]](./images/glazer_14_p21_m_001.png) | ![P21/m [010]](./images/glazer_14_p21_m_010.png) | ![P21/m [100]](./images/glazer_14_p21_m_100.png) |
| 15 | P42/nmc | a+a+c- | ![P42/nmc [001]](./images/glazer_15_p42_nmc_001.png) | ![P42/nmc [010]](./images/glazer_15_p42_nmc_010.png) | ![P42/nmc [100]](./images/glazer_15_p42_nmc_100.png) |

**Viewing Angle Legend:**
- **[0 0 1] (Top View)**: Looking down the z-axis → best for seeing tilts about x and y axes
- **[0 1 0] (Front View)**: Looking along the y-axis → best for seeing tilts about x and z axes
- **[1 0 0] (Side View)**: Looking along the x-axis → best for seeing tilts about y and z axes

By comparing the three views, you can see how tilting on different axes becomes visible from different perspectives. For example, a tilt about the z-axis (like in `a0a0c-`) is most clearly visible in the [0 0 1] top view, while tilts about y and z axes (like in `a0b-b-`) are best seen in the [1 0 0] side view.

**Note:** To regenerate all visualizations, run:
```bash
python3 Examples/plot.py
```
