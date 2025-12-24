import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure
from ase.io import write
from pathlib import Path
import sys
import json
import numpy as np
from ase.neighborlist import neighbor_list

path = "../IMAGES/"

rotation_iso = "45x,45y,0z"
rotation_top = "0x,0y,0z"
rotation_side = "90x,0y,0z"

# Bases
cubic_base = dict(structure_type="bulk", xy_expansion=(2, 2), template="cubic")
jago_base = dict(structure_type="bulk", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="jagodinxky")
dj_base = dict(structure_type="bulk", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="reduced")
ruddlesden_popper_base = dict(structure_type="bulk", xy_expansion=(1, 1), template="reduced", layer_sequence="RP")
dion_jacobson_base = dict(structure_type="bulk", xy_expansion=(1, 1), template="reduced", layer_sequence="DJ")

# Using ASE to create high quality images for the thesis:
# Bulk perovskite Cs Pb I:
q2d = q2D_creator()
bulk_perovskite = q2d.create_structure(**cubic_base, thickness=2, spacer="Cs", B_ions="Pb", X_ions="I", A_ions="Cs")
# Save as cif file
write(path + "bulk_perovskite.cif", bulk_perovskite, format="cif")

# Ruddlesden-Popper perovskite Cs Pb I:
ruddlesden_popper = q2d.create_structure(**ruddlesden_popper_base, thickness=2, spacer="Cs", B_ions="Pb", X_ions="I", A_ions="Cs")
write(path + "ruddlesden_popper_atomic.cif", ruddlesden_popper, format="cif")
# Ruddlesden-Popper perovskite Cs Pb I with NH3 spacers:
ruddlesden_popper_spacer = q2d.create_structure(**ruddlesden_popper_base, thickness=2, spacer="CCCC[NH3+]", B_ions="Pb", X_ions="I", A_ions="MA")
write(path + "ruddlesden_popper_spacer.cif", ruddlesden_popper_spacer, format="cif")

#Dion-Jacobson perovskite Cs Pb I:
dion_jacobson = q2d.create_structure(**dion_jacobson_base, thickness=2, spacer="Cs", B_ions="Pb", X_ions="I", A_ions="Cs")
write(path + "dion_jacobson_atomic.cif", dion_jacobson, format="cif")
# Dion-Jacobson perovskite Cs Pb I with NH3 spacers:
dion_jacobson_spacer = q2d.create_structure(**dion_jacobson_base, thickness=2, spacer="[NH3+]CC=CC[NH3+]", B_ions="Pb", X_ions="I", A_ions="MA")
write(path + "dion_jacobson_spacer.cif", dion_jacobson_spacer, format="cif")

# ACI Spacer:
aci_spacer = q2d.create_structure(**dion_jacobson_base, thickness=2, spacer=["CC[NH3+]", "[NH3+]CCCCCCC[NH3+]"], B_ions="Pb", X_ions="I", A_ions="MA")
write(path + "aci_spacer.cif", aci_spacer, format="cif")