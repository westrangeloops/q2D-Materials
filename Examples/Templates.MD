# Templates for q2D_Materials

This project loads perovskite geometry templates from JSON files in
`q2D_Materials/builders/data/`. You can add your own templates by dropping a
JSON there. This guide explains the expected format for the layer-based system.

## File location

- Place your template at `q2D_Materials/builders/data/<name>.json`.
- The `<name>` (without `.json`) is what you pass as `template` in creator/pipeline.

## Required JSON structure

```json
{
  "name": "cubic",
  "named_layers": {
    "L1": [
      ["A", 0.0, 0.0],
      ["X", 0.5, 0.5]
    ],
    "L2": [
      ["B", 0.5, 0.5],
      ["X", 0.0, 0.5],
      ["X", 0.5, 0.0]
    ]
  },
  "layer_sequence": ["L1", "L2"],
  "lattice_multipliers": [2.0, 2.0]
}
```

### Fields

- `name` (string): Identifier for the template.
- `named_layers` (object): Dictionary of layer names to their atomic content.
  - Each layer name maps to an array of entries `[site, x, y]`.
  - `site` must be "A", "B", or "X".
  - `x, y` are fractional coordinates within the layer.
- `layer_sequence` (array): Default stacking order of layers.
- `lattice_multipliers` (array of 2 numbers): In-plane lattice scaling factors.
- `layer_spacing` (object, optional): Custom spacing between layer pairs.
- `angles` (array of 3 numbers, optional): Lattice angles in degrees.

### Layer Entry Format

Each layer entry is `[site, x, y]` where:
- `site`: "A", "B", or "X" (atom type)
- `x, y`: fractional coordinates (0-1) within the 2D layer

### Custom Layer Spacing

```json
{
  "layer_spacing": {
    "L1->L2": 2.8,
    "L2->L1": 2.9
  }
}
```

### Notes

- Layers stack in the order specified by `layer_sequence`.
- The c-axis length is calculated as `BX_dist × n_layers × jahn_teller_dist`.
- XY expansion tiles layers within the plane using `xy_expansion=(nx, ny)`.
- Thickness controls how many times the layer sequence repeats.

## Examples

- `cubic.json`: Simple cubic perovskite (L1 + L2 stacking)
- `reduced.json`: Reduced symmetry with rotated in-plane lattice
- `hexagonal.json`: Hexagonal perovskite structure
- `jagodinxky.json`: Complex Jagodszinski stacking sequences

## Validation errors

If a template is malformed, the loader will raise an error. Common issues:
- Missing `name` or `named_layers`
- Invalid layer entries (not `[site, x, y]`)
- `site` not "A", "B", or "X"
- `lattice_multipliers` not a 2-number array
- `layer_sequence` referencing undefined layer names

Fix the JSON accordingly and retry.

