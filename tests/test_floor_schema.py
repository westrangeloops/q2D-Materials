import numpy as np
import pytest
from ase import Atoms

from q2D_Materials.builders.populate import apply_penetration_offsets, populate_from_floor_schema
from q2D_Materials.builders.templates import build_floor_schema, flatten_floor_schema
from q2D_Materials.pipeline.common import _apply_attachment_end_ap


def test_floor_schema_cubic_basic():
    schema = build_floor_schema("cubic", BX_dist=3.0, layer_sequence="L1-L2-L1")

    assert list(schema.floors.keys()) == ["1", "2", "3"]
    assert schema.floor_names["1"] == "L1"
    assert schema.floor_names["2"] == "L2"

    positions, labels = flatten_floor_schema(schema)
    assert "A" in positions and "B" in positions and "X" in positions
    assert all(isinstance(v, np.ndarray) for v in positions.values())
    assert labels["A"][0] == "A"

    z_vals = [p[3] for p in schema.floors["1"]] + [p[3] for p in schema.floors["2"]]
    assert min(z_vals) == 0.0
    assert schema.cell.shape == (3, 3)

    atoms = populate_from_floor_schema(schema, A_ions="Cs", B_ions="Pb", X_ions="I")
    assert len(atoms) > 0


def test_penetration_shifts_ap_and_s():
    schema = build_floor_schema("cubic", BX_dist=3.0, layer_sequence="L1-L2")
    schema.floors["1"].append(["Ap", 0.0, 0.0, 0.0])
    schema.floors["2"].append(["S1", 0.0, 0.0, 3.0])

    positions, _ = flatten_floor_schema(schema)
    shifted = apply_penetration_offsets(positions, penetration=0.5, BX_dist=3.0)

    ap_z = shifted["Ap"][:, 2].tolist()
    s_z = shifted["S1"][:, 2].tolist()

    assert ap_z and s_z
    # Ap bottom goes down by 1.5, S1 top goes up by 1.5
    assert ap_z[0] == -1.5
    assert s_z[0] == 4.5


def test_penetration_cycles_ap_sites():
    schema = build_floor_schema("cubic", BX_dist=3.0, layer_sequence="L1-L2")
    # Add two Ap sites on first floor
    schema.floors["1"].append(["Ap", 0.0, 0.0, 0.0])
    schema.floors["1"].append(["Ap", 0.1, 0.1, 0.0])

    positions, _ = flatten_floor_schema(schema)
    shifted = apply_penetration_offsets(positions, penetration=[0.2, 0.4], BX_dist=3.0)

    ap_z = shifted["Ap"][:, 2].tolist()
    # Both Ap on bottom: shifts are -0.6, -1.2
    assert pytest.approx(ap_z) == [-0.6, -1.2]


def test_attachment_end_converts_floors():
    schema = build_floor_schema("cubic", BX_dist=3.0, layer_sequence="L1-L2")
    _apply_attachment_end_ap(schema, "bottom")
    positions, _ = flatten_floor_schema(schema)
    assert "Ap" in positions
    assert positions["Ap"].shape[0] == 1  # only bottom layer A converted

    schema_top = build_floor_schema("cubic", BX_dist=3.0, layer_sequence="L1-L2-L1")
    _apply_attachment_end_ap(schema_top, "top")
    positions_top, _ = flatten_floor_schema(schema_top)
    assert positions_top["Ap"].shape[0] == 1  # only top layer A converted

    schema_both = build_floor_schema("cubic", BX_dist=3.0, layer_sequence="L1-L2-L1")
    _apply_attachment_end_ap(schema_both, "both")
    positions_both, _ = flatten_floor_schema(schema_both)
    assert positions_both["Ap"].shape[0] == 2  # both floors converted

