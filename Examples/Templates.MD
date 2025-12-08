# Templates for q2D_Materials

This project loads perovskite geometry templates from JSON files in
`q2D_Materials/builders/data/`. You can add your own templates by dropping a
JSON there. This guide explains the expected format.

## File location

- Place your template at `q2D_Materials/builders/data/<name>.json`.
- The `<name>` (without `.json`) is what you pass as `template` in creator/pipeline.

## Required JSON structure

```json
{
  "name": "cubic",
  "positions": {
    "A": [[0.0, 0.0, 0.0]],
    "B": [[0.5, 0.5, 0.5]],
    "X": [
      [0.5, 0.5, 0.0],
      [0.5, 0.0, 0.5],
      [0.0, 0.5, 0.5]
    ]
  },
  "lattice_multipliers": [2.0, 2.0, 2.0]
}
```

### Fields

- `name` (string): Identifier for the template.
- `positions` (object):
  - `A`, `B`, `X`: arrays of fractional coordinates `[x, y, z]`.
  - Each coordinate must be a list of three numbers.
- `lattice_multipliers` (array of 3 numbers, optional):
  - Scales the lattice by `BX_dist` to produce lattice vectors.
  - If omitted, defaults to `[2.0, 2.0, 2.0]`.

### Notes

- Fractional coordinates are scaled by the lattice vectors during build.
- For cubic templates, Jahn–Teller distortion (if used) is applied to the z
  multiplier at runtime.
- Keep arrays numeric and length-3 to pass validation.

## Examples

- `cubic.json` (provided): standard ABX3 cubic cell.
- `reduced.json` (provided): reduced 2D template with in-plane rotated lattice.

## Validation errors

If a template is malformed, the loader will raise an error with a pointer to
this file. Common issues:
- Missing `name` or `positions`
- `positions.A` / `positions.B` / `positions.X` not arrays of `[x, y, z]`
- `lattice_multipliers` not a 3-number array

Fix the JSON accordingly and retry.

