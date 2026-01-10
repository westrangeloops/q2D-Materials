![q2D-Materials Logo](../Logos/logo.png)

# 14. Radial Distribution Functions — pair correlations

Radial distribution functions (RDF) describe the probability of finding atom pairs at specific distances. The analyzer computes partial RDFs for element pairs, revealing local structure, coordination environments, and bonding distances.

## Basic Usage

Compute RDF for a single element pair:

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

# Create and analyze structure
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    thickness=2, vacuum=15.0,
)

analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Compute RDF for Pb-I pairs
rdf = analyzer.get_partial_rdf([["Pb", "I"]])

# Access results
distances = rdf['Pb-I']['distances']  # Distance array (Å)
rdf_values = rdf['Pb-I']['rdf']       # RDF values
print(f"Pb-I RDF: {len(distances)} points")
print(f"First peak at: {distances[np.argmax(rdf_values[:100])]:.2f} Å")
```

## Multiple Element Pairs

Compute RDFs for multiple pairs simultaneously:

```python
# Multiple pairs
rdf = analyzer.get_partial_rdf([
    ["Pb", "I"],
    ["Pb", "Pb"],
    ["I", "I"]
])

# Access each pair
for pair_name, data in rdf.items():
    distances = data['distances']
    rdf_values = data['rdf']
    print(f"{pair_name} RDF: {len(distances)} points")
    print(f"  First peak: {distances[np.argmax(rdf_values[:100])]:.2f} Å")
```

## RDF Parameters

Control RDF computation with parameters:

```python
rdf = analyzer.get_partial_rdf(
    element_pairs=[["Pb", "I"]],
    max_dist=15.0,      # Maximum distance (Å)
    npoints=300,        # Number of grid points
    mode="pyrovskite",  # Normalization mode: 'ase' or 'pyrovskite'
    ss_norm=False       # Self-scattering normalization
)
```

**Parameter explanations:**

- `max_dist`: Maximum distance for RDF computation (default: 10.0 Å)
- `npoints`: Number of grid points. If `None`, uses `max_dist * 10` for 'ase' mode or `max_dist * 50` for 'pyrovskite' mode
- `mode`: 
  - `'ase'`: Uses ASE's binning method
  - `'pyrovskite'`: Uses gaussian kernel smoothing (default, recommended for perovskites)
- `ss_norm`: Whether to apply structure-specific normalization (default: False)

## RDF Modes

### ASE Mode

Uses simple binning:

```python
rdf = analyzer.get_partial_rdf(
    element_pairs=[["Pb", "I"]],
    mode="ase",
    max_dist=10.0,
    npoints=100
)
```

### Pyrovskite Mode (Recommended)

Uses gaussian kernel smoothing for smoother RDFs:

```python
rdf = analyzer.get_partial_rdf(
    element_pairs=[["Pb", "I"]],
    mode="pyrovskite",  # Default
    max_dist=15.0
)
```

## Interpreting RDF Results

RDF peaks indicate preferred distances:

```python
import numpy as np

rdf = analyzer.get_partial_rdf([["Pb", "I"]])

distances = rdf['Pb-I']['distances']
rdf_values = rdf['Pb-I']['rdf']

# Find first coordination shell (first major peak)
first_peak_idx = np.argmax(rdf_values[:200])  # Search first 200 points
first_peak_dist = distances[first_peak_idx]

print(f"First coordination shell: {first_peak_dist:.2f} Å")
print(f"Peak height: {rdf_values[first_peak_idx]:.2f}")

# Find all peaks above threshold
peak_threshold = np.max(rdf_values) * 0.3  # 30% of max
peaks = []
for i in range(1, len(rdf_values) - 1):
    if (rdf_values[i] > rdf_values[i-1] and 
        rdf_values[i] > rdf_values[i+1] and 
        rdf_values[i] > peak_threshold):
        peaks.append((distances[i], rdf_values[i]))

print(f"\nFound {len(peaks)} peaks:")
for dist, height in peaks[:5]:  # Show first 5
    print(f"  {dist:.2f} Å (height: {height:.2f})")
```

## Common Use Cases

### Bond Distance Analysis

```python
# Analyze Pb-I bond distances
rdf = analyzer.get_partial_rdf([["Pb", "I"]], max_dist=5.0)

distances = rdf['Pb-I']['distances']
rdf_values = rdf['Pb-I']['rdf']

# First peak is typically the bond distance
bond_distance = distances[np.argmax(rdf_values[:100])]
print(f"Pb-I bond distance: {bond_distance:.2f} Å")
```

### Coordination Environment

```python
# Analyze coordination by integrating RDF
rdf = analyzer.get_partial_rdf([["Pb", "I"]], max_dist=4.0)

distances = rdf['Pb-I']['distances']
rdf_values = rdf['Pb-I']['rdf']

# Integrate to first minimum (coordination number estimate)
first_min_idx = None
for i in range(50, len(rdf_values) - 1):
    if rdf_values[i] < rdf_values[i-1] and rdf_values[i] < rdf_values[i+1]:
        first_min_idx = i
        break

if first_min_idx:
    coordination_radius = distances[first_min_idx]
    print(f"Coordination shell radius: {coordination_radius:.2f} Å")
```

### Structure Comparison

```python
# Compare RDFs from different structures
rdf1 = analyzer1.get_partial_rdf([["Pb", "I"]])
rdf2 = analyzer2.get_partial_rdf([["Pb", "I"]])

# Compare first peak positions
peak1 = rdf1['Pb-I']['distances'][np.argmax(rdf1['Pb-I']['rdf'][:100])]
peak2 = rdf2['Pb-I']['distances'][np.argmax(rdf2['Pb-I']['rdf'][:100])]

print(f"Structure 1 first peak: {peak1:.2f} Å")
print(f"Structure 2 first peak: {peak2:.2f} Å")
print(f"Difference: {abs(peak1 - peak2):.2f} Å")
```

## Complete Example

```python
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator
import numpy as np

# Create and analyze structure
q2d = q2D_creator()
structure = q2d.create_structure(
    structure_type="monolayer",
    A_ions="MA", B_ions="Pb", X_ions="I",
    xy_expansion=(2, 2), template="cubic",
    thickness=2, vacuum=15.0,
)

analyzer = q2D_analyzer(structure)
analyzer.analyze()

# Compute RDFs for multiple pairs
rdf = analyzer.get_partial_rdf(
    element_pairs=[
        ["Pb", "I"],
        ["Pb", "Pb"],
        ["I", "I"]
    ],
    max_dist=15.0,
    mode="pyrovskite"
)

# Analyze each pair
for pair_name, data in rdf.items():
    distances = data['distances']
    rdf_values = data['rdf']
    
    # Find first peak
    first_peak_idx = np.argmax(rdf_values[:200])
    first_peak_dist = distances[first_peak_idx]
    first_peak_height = rdf_values[first_peak_idx]
    
    print(f"{pair_name} RDF:")
    print(f"  First peak: {first_peak_dist:.2f} Å (height: {first_peak_height:.2f})")
    print(f"  Total points: {len(distances)}")
```

