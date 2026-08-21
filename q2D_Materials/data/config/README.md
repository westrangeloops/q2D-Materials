# Configuration Files

This directory contains JSON configuration files for q2D-Materials.

## structural_constants.json

Contains all default constants and tolerance thresholds used throughout the structural analysis modules. These values control:

- Octahedral matching and distortion calculations
- Time-averaging parameters
- Hydrogen bond detection
- Volume calculations
- Population gap detection

### Structure

Each constant is stored as an object with:
- `value`: The actual constant value
- `description`: What the constant controls
- `used_in`: List of modules/functions where it's used

### Editing

You can edit these values directly in the JSON file. The changes will be automatically loaded when the Python modules import from `structural_constants.py`.

**Note**: Standard JSON doesn't support comments, so we use `_comment` fields and `description` fields within the JSON structure itself.

### Example

```json
{
  "fitting_tolerance": {
    "value": 0.3,
    "description": "Used in match_bx_orthogonal() for matching octahedral bond vectors",
    "used_in": ["octahedral_analysis.py:match_bx_orthogonal"]
  }
}
```

## License

Parts of this code are from PDynA (https://github.com/WMD-group/PDynA):
MIT License - Copyright (c) 2022 Xia Liang

