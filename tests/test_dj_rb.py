#!/usr/bin/env python3
"""Test DJ phase with atomic spacers - outputs VASP files for visual inspection."""

import sys

sys.path.insert(0, "/home/dotempo/Documents/DJ/q2D-Materials")

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

creator = q2D_creator()

print("DJ Phase Tests")
print("=" * 40)

# 1. DJ with Rb spacer (Sr-Nb-O)
print("\n1. DJ SrNbO3 with Rb spacer")
dj1 = creator.create_perovskite(
    "DJ", A_ions="Sr", B_ions="Nb", X_ions="O", spacer="Rb", supercell=[1, 1, 2]
)
write("dj_rb_sr.vasp", dj1, format="vasp", sort=True)
print(f"   {len(dj1)} atoms")

# 2. DJ with Rb spacer (Ca-Ta-O)
print("\n2. DJ CaTaO3 with Rb spacer")
dj2 = creator.create_perovskite(
    "DJ", A_ions="Ca", B_ions="Ta", X_ions="O", spacer="Rb", supercell=[1, 1, 2]
)
write("dj_rb_ca.vasp", dj2, format="vasp", sort=True)
print(f"   {len(dj2)} atoms")

# 3. DJ with Cs spacer (Pb-I)
print("\n3. DJ CsPbI3 with Cs spacer")
dj3 = creator.create_perovskite(
    "DJ", A_ions="Cs", B_ions="Pb", X_ions="I", spacer="Cs", supercell=[1, 1, 1]
)
write("dj_cs_pb.vasp", dj3, format="vasp", sort=True)
print(f"   {len(dj3)} atoms")

# 4. RP for comparison
print("\n4. RP CsPbI3 with Cs spacer (comparison)")
rp = creator.create_perovskite(
    "RP", A_ions="Cs", B_ions="Pb", X_ions="I", spacer="Cs", supercell=[1, 1, 1]
)
write("rp_cs_pb.vasp", rp, format="vasp", sort=True)
print(f"   {len(rp)} atoms")

print("\n" + "=" * 40)
print("Files written:")
print("  - dj_rb_sr.vasp")
print("  - dj_rb_ca.vasp")
print("  - dj_cs_pb.vasp")
print("  - rp_cs_pb.vasp")
