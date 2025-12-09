from pathlib import Path
import sys

from ase.io import write
from q2D_Materials.core.creator import q2D_creator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

IMAGES = Path(__file__).resolve().parent / "images"
IMAGES.mkdir(exist_ok=True)

q2d = q2D_creator()

rotation_iso = "45x,45y,0z"
rotation_top = "0x,0y,0z"

# Bases
cubic_base = dict(structure_type="bulk", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic")
jago_base = dict(structure_type="bulk", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="jagodinxky")

# Creator / Templates (cubic)
creator_L1 = q2d.create_perovskite(**cubic_base, layer_sequence=["L1"])
write(IMAGES / "creator-L1.png", creator_L1, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2 = q2d.create_perovskite(**cubic_base, layer_sequence=["L1", "L2"])
write(IMAGES / "creator-L1-L2.png", creator_L1_L2, rotation=rotation_iso, show_unit_cell=2)

creator_L1_L2_L1 = q2d.create_perovskite(**cubic_base, layer_sequence=["L1", "L2", "L1"])
write(IMAGES / "creator-L1-L2-L1.png", creator_L1_L2_L1, rotation=rotation_iso, show_unit_cell=2)

# Jagodzinski storyboard
jago_a = q2d.create_perovskite(**jago_base, layer_sequence=["a"])
write(IMAGES / "jago-a.png", jago_a, rotation=rotation_iso, show_unit_cell=2)

jago_aC = q2d.create_perovskite(**jago_base, layer_sequence=["a", "C"])
write(IMAGES / "jago-aC.png", jago_aC, rotation=rotation_iso, show_unit_cell=2)

jago_aCc = q2d.create_perovskite(**jago_base, layer_sequence=["a", "C", "c"])
write(IMAGES / "jago-aCc.png", jago_aCc, rotation=rotation_iso, show_unit_cell=2)

# Glazer: untilted vs tilted (4x4, top-down)
glazer_untilted = q2d.create_perovskite(**(cubic_base | {"xy_expansion": (4, 4), "glazer_angles": [0, 0, 0], "glazer_pattern": ["0", "0", "0"]}), layer_sequence=["L1", "L2"])
write(IMAGES / "glazer-untitled-top.png", glazer_untilted, rotation=rotation_top, show_unit_cell=2)

glazer_tilted = q2d.create_perovskite(**(cubic_base | {"xy_expansion": (4, 4), "glazer_angles": [0, 0, 10], "glazer_pattern": ["0", "0", "+"]}), layer_sequence=["L1", "L2"])
write(IMAGES / "glazer-tilted-top.png", glazer_tilted, rotation=rotation_top, show_unit_cell=2)

# Monolayer: single and double thickness
mono1 = q2d.create_perovskite(structure_type="monolayer", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic", vacuum=15.0, thickness=1, layer_sequence=["L1", "L2"])
write(IMAGES / "mono-1layer.png", mono1, rotation=rotation_iso, show_unit_cell=2)

mono2 = q2d.create_perovskite(structure_type="monolayer", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic", vacuum=15.0, thickness=2, layer_sequence=["L1", "L2", "L1"])
write(IMAGES / "mono-2layer.png", mono2, rotation=rotation_iso, show_unit_cell=2)

# Twist storyboard: build two monolayers and twist them
twist_mono1 = q2d.create_perovskite(structure_type="monolayer", A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic", vacuum=12.0)
twist_mono2 = q2d.create_perovskite(structure_type="monolayer", A_ions="FA", B_ions="Sn", X_ions="Br", xy_expansion=(1, 1), template="cubic", vacuum=12.0)

write(IMAGES / "twist-mono1.png", twist_mono1, rotation=rotation_top, show_unit_cell=2)
write(IMAGES / "twist-mono2.png", twist_mono2, rotation=rotation_top, show_unit_cell=2)

twisted = q2d.twist(monolayers=[twist_mono1, twist_mono2], twist_angles=[(3, 1)], interlayer_distances=[8.0], vacuum=12.0)
write(IMAGES / "twist-bilayer.png", twisted, rotation=rotation_iso, show_unit_cell=2)
