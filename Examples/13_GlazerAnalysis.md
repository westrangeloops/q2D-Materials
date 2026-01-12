![q2D-Materials Logo](../Logos/logo.png)

# 13. Glazer Pattern Detection — reverse engineering tilts

The analyzer can detect Glazer tilt patterns from existing structures by analyzing octahedral tilting angles. This is the inverse operation of creating structures with Glazer tilting—given a structure, it determines what Glazer pattern was used (or would best describe it).

## Basic Usage

Detect the Glazer pattern from an analyzed structure:

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

# Create a structure with known Glazer pattern
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    thickness=2, vacuum=15.0,
    glazer_pattern="a+b-b-",  # Pnma
)

# Analyze and detect Glazer pattern
analyzer = q2D_analyzer(structure)
analyzer.analyze()

glazer_info = analyzer.get_glazer_pattern()

print(f"Glazer notation: {glazer_info['notation']}")
print(f"Tilt angles: {glazer_info['tilt_angles']}°")
print(f"Tilt pattern: {glazer_info['tilt_pattern']}")
print(f"Space group: {glazer_info['space_group']}")
```

## Detection Results

The `get_glazer_pattern()` method returns a dictionary with:

- `notation`: Glazer notation string (e.g., `"a+b-b-"`)
- `tilt_angles`: List of tilt angles in degrees `[x, y, z]`
- `tilt_pattern`: List of phase patterns `["+", "-", "-"]`
- `space_group`: Associated space group (e.g., `"Pnma"`)

## Adjusting Detection Sensitivity

The detection uses tolerance-based thresholds. For noisy or optimized structures, you may need to adjust:

```python
# More sensitive (smaller tolerance)
glazer_info = analyzer.get_glazer_pattern(tolerance=0.05)

# Less sensitive (larger tolerance, for noisy structures)
glazer_info = analyzer.get_glazer_pattern(tolerance=0.5)
```

## Fine-Tuning Detection Parameters

For structures with subtle tilts or experimental data, fine-tune specific thresholds:

```python
glazer_info = analyzer.get_glazer_pattern(
    tolerance=0.1,
    tilt_significance_threshold=0.15,  # Minimum tilt to consider (degrees)
    zero_tilt_threshold=2.0,            # Maximum to classify as zero (degrees)
    magnitude_equivalence_threshold=0.2,  # Difference for same magnitude (degrees)
    bond_selection_threshold=0.1,      # Bond selection tolerance
    numerical_tolerance=1e-6          # Numerical comparison tolerance
)
```

**Parameter explanations:**

- `tolerance`: Overall tolerance for angle comparisons
- `tilt_significance_threshold`: Minimum tilt angle to be considered significant (default: 0.1°)
- `zero_tilt_threshold`: Maximum angle to classify as zero tilt (default: 1.0°)
- `magnitude_equivalence_threshold`: Maximum difference for tilts to be considered the same magnitude (default: 0.15°)
- `bond_selection_threshold`: Tolerance for selecting bonds in angle calculations
- `numerical_tolerance`: Numerical precision for comparisons

## Detection Algorithm

The Glazer detection algorithm:

1. **Extracts B-X-B angles** from the graph structure
2. **Groups angles by axis** (x, y, z directions)
3. **Determines tilt magnitudes** by comparing angles to 180° (untilted)
4. **Identifies phase relationships** by comparing tilts of adjacent octahedra
5. **Classifies magnitudes** as same (a, b, c) or different
6. **Assigns Glazer notation** based on magnitude and phase patterns
7. **Maps to space group** using the Glazer notation lookup

## Example: Detecting from Experimental Structures

For experimental or optimized structures, you may need relaxed tolerances:

```python
# Load experimental structure
analyzer = q2D_analyzer("experimental_structure.vasp")
analyzer.analyze()

# Use relaxed tolerances for noisy data
glazer_info = analyzer.get_glazer_pattern(
    tolerance=0.3,
    tilt_significance_threshold=0.5,
    zero_tilt_threshold=3.0,
    magnitude_equivalence_threshold=0.5
)

print(f"Detected pattern: {glazer_info['notation']}")
print(f"Space group: {glazer_info['space_group']}")
```

## Example: Verifying Created Structures

Verify that a created structure matches the intended Glazer pattern:

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

# Create structure with specific pattern
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    glazer_pattern="a0a0c+",  # P4/mbm
)

# Detect and verify
analyzer = q2D_analyzer(structure)
analyzer.analyze()

detected = analyzer.get_glazer_pattern()
print(f"Created: a0a0c+")
print(f"Detected: {detected['notation']}")
print(f"Match: {detected['notation'] == 'a0a0c+'}")
```

## Common Issues and Solutions

**Issue: Detection fails or returns unexpected pattern**

- **Solution**: Increase `tolerance` and `magnitude_equivalence_threshold`
- **Solution**: Check that structure has been properly analyzed (`analyzer.analyze()`)

**Issue: Small tilts not detected**

- **Solution**: Decrease `tilt_significance_threshold`
- **Solution**: Decrease `zero_tilt_threshold`

**Issue: Different magnitudes incorrectly grouped**

- **Solution**: Decrease `magnitude_equivalence_threshold`
- **Solution**: Use smaller `tolerance` for more precise detection

## Complete Example

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

# Create structure with Glazer tilting
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    thickness=2, vacuum=15.0,
    glazer_pattern="a+b-b-",  # Pnma
)

# Analyze and detect
analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Detect Glazer pattern
glazer = analyzer.get_glazer_pattern()

print("Glazer Detection Results:")
print(f"  Notation: {glazer['notation']}")
print(f"  Tilt angles: {glazer['tilt_angles']}°")
print(f"  Tilt pattern: {glazer['tilt_pattern']}")
print(f"  Space group: {glazer['space_group']}")

# Verify detection
if glazer['notation'] == 'a+b-b-':
    print("\n✓ Detection successful!")
else:
    print(f"\n⚠ Detected {glazer['notation']} instead of a+b-b-")
```

