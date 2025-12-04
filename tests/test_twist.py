#!/usr/bin/env python3
"""Test twist method - outputs VASP files for visual inspection."""

import sys
sys.path.insert(0, '/home/dotempo/Documents/DJ/q2D-Materials')

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

creator = q2D_creator()
    
print("Twist Bilayer Tests")
print("=" * 40)

# NOTE: All structures must use at least 2x2 supercell in XY plane
# because a single octahedron cannot physically exhibit tilting/twist patterns

# 1. Monolayer with atomic spacer (2x2x1)
print("\n1. Monolayer with Cs spacer (2x2)")
mono_cs = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs', supercell=[2, 2, 1]
)
write('mono_cs.vasp', mono_cs, format='vasp', sort=True)
print(f"   {len(mono_cs)} atoms")

# 2. Twist (m=3, n=1)
print("\n2. Twisted (3,1) from Cs monolayer")
twist_3_1 = mono_cs.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_cs_3_1.vasp', twist_3_1, format='vasp', sort=True)
print(f"   {len(twist_3_1)} atoms, params: {twist_3_1.twist_params}")

# 3. Twist (m=5, n=2)
print("\n3. Twisted (5,2) from Cs monolayer")
twist_5_2 = mono_cs.twist(m=5, n=2, interlayer_distance=11.0, vacuum=15.0)
write('twist_cs_5_2.vasp', twist_5_2, format='vasp', sort=True)
print(f"   {len(twist_5_2)} atoms, params: {twist_5_2.twist_params}")

# 4. Monolayer with molecular spacer (2x2x1)
print("\n4. Monolayer with PDA spacer (2x2)")
mono_pda = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]', supercell=[2, 2, 1]
)
write('mono_pda.vasp', mono_pda, format='vasp', sort=True)
print(f"   {len(mono_pda)} atoms")

# 5. Twist molecular spacer
print("\n5. Twisted (3,1) from PDA monolayer")
twist_pda = mono_pda.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_pda_3_1.vasp', twist_pda, format='vasp', sort=True)
print(f"   {len(twist_pda)} atoms, params: {twist_pda.twist_params}")

# ===================================================================
# TWIST WITH GLAZER TILTING
# ===================================================================
print("\n" + "=" * 40)
print("TWIST WITH GLAZER TILTING")
print("=" * 40)

# 6. Monolayer with pattern a-a-a-, then twist (2x2 required for Glazer)
print("\n6. Monolayer Cs with pattern a-a-a-, then twist (3,1)")
mono_aaa = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs', supercell=[2, 2, 1],
    glazer_notation='a-a-a-'
)
write('mono_aaa.vasp', mono_aaa, format='vasp', sort=True)
print(f"   {len(mono_aaa)} atoms")
twist_aaa = mono_aaa.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_aaa_3_1.vasp', twist_aaa, format='vasp', sort=True)
print(f"   Twisted: {len(twist_aaa)} atoms, params: {twist_aaa.twist_params}")

# 7. Monolayer with pattern a0a0c- (tetragonal), then twist
print("\n7. Monolayer Cs with pattern a0a0c-, then twist (3,1)")
mono_tetragonal = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs', supercell=[2, 2, 1],
    glazer_notation='a0a0c-'
)
write('mono_tetragonal.vasp', mono_tetragonal, format='vasp', sort=True)
print(f"   {len(mono_tetragonal)} atoms")
twist_tetragonal = mono_tetragonal.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_tetragonal_3_1.vasp', twist_tetragonal, format='vasp', sort=True)
print(f"   Twisted: {len(twist_tetragonal)} atoms, params: {twist_tetragonal.twist_params}")

# 8. Monolayer with pattern a+b-c- (mixed), then twist
print("\n8. Monolayer Cs with pattern a+b-c-, then twist (3,1)")
mono_mixed = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs', supercell=[2, 2, 1],
    glazer_pattern=['+', '-', '-'],
    glazer_default_angle=2.0
)
write('mono_mixed.vasp', mono_mixed, format='vasp', sort=True)
print(f"   {len(mono_mixed)} atoms")
twist_mixed = mono_mixed.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_mixed_3_1.vasp', twist_mixed, format='vasp', sort=True)
print(f"   Twisted: {len(twist_mixed)} atoms, params: {twist_mixed.twist_params}")

# 9. Monolayer PDA with pattern a-a-a-, then twist
print("\n9. Monolayer PDA with pattern a-a-a-, then twist (3,1)")
mono_pda_aaa = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]', supercell=[2, 2, 1],
    glazer_notation='a-a-a-'
)
write('mono_pda_aaa.vasp', mono_pda_aaa, format='vasp', sort=True)
print(f"   {len(mono_pda_aaa)} atoms")
twist_pda_aaa = mono_pda_aaa.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_pda_aaa_3_1.vasp', twist_pda_aaa, format='vasp', sort=True)
print(f"   Twisted: {len(twist_pda_aaa)} atoms, params: {twist_pda_aaa.twist_params}")

# 10. Monolayer PDA with pattern a0a0c-, then twist
print("\n10. Monolayer PDA with pattern a0a0c-, then twist (3,1)")
mono_pda_tetragonal = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]', supercell=[2, 2, 1],
    glazer_notation='a0a0c-'
)
write('mono_pda_tetragonal.vasp', mono_pda_tetragonal, format='vasp', sort=True)
print(f"   {len(mono_pda_tetragonal)} atoms")
twist_pda_tetragonal = mono_pda_tetragonal.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_pda_tetragonal_3_1.vasp', twist_pda_tetragonal, format='vasp', sort=True)
print(f"   Twisted: {len(twist_pda_tetragonal)} atoms, params: {twist_pda_tetragonal.twist_params}")

# 11. Monolayer with pattern and larger Glazer angles, then twist
print("\n11. Monolayer Cs with pattern a-a-a- (5°), then twist (3,1)")
mono_tilt = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs', supercell=[2, 2, 1],
    glazer_notation='a-a-a-',
    glazer_default_angle=5.0
)
write('mono_tilt.vasp', mono_tilt, format='vasp', sort=True)
print(f"   {len(mono_tilt)} atoms")
twist_tilt = mono_tilt.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_tilt_3_1.vasp', twist_tilt, format='vasp', sort=True)
print(f"   Twisted: {len(twist_tilt)} atoms, params: {twist_tilt.twist_params}")

# 12. Monolayer PDA with pattern and larger Glazer angles, then twist
print("\n12. Monolayer PDA with pattern a-a-a- (5°), then twist (3,1)")
mono_pda_tilt = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]', supercell=[2, 2, 1],
    glazer_notation='a-a-a-',
    glazer_default_angle=5.0
)
write('mono_pda_tilt.vasp', mono_pda_tilt, format='vasp', sort=True)
print(f"   {len(mono_pda_tilt)} atoms")
twist_pda_tilt = mono_pda_tilt.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_pda_tilt_3_1.vasp', twist_pda_tilt, format='vasp', sort=True)
print(f"   Twisted: {len(twist_pda_tilt)} atoms, params: {twist_pda_tilt.twist_params}")

# 13. Monolayer with larger supercell for Glazer pattern visibility
print("\n13. Monolayer Cs (3x3) with pattern a-a-a-, then twist (3,1)")
mono_pattern = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs', supercell=[3, 3, 1],
    glazer_notation='a-a-a-'
)
write('mono_pattern.vasp', mono_pattern, format='vasp', sort=True)
print(f"   {len(mono_pattern)} atoms")
twist_pattern = mono_pattern.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('twist_pattern_3_1.vasp', twist_pattern, format='vasp', sort=True)
print(f"   Twisted: {len(twist_pattern)} atoms, params: {twist_pattern.twist_params}")

print("\n" + "=" * 40)
print("Files written:")
print("  Basic (2x2 supercell):")
print("    - mono_cs.vasp")
print("    - twist_cs_3_1.vasp")
print("    - twist_cs_5_2.vasp")
print("    - mono_pda.vasp")
print("    - twist_pda_3_1.vasp")
print("  With Glazer Tilting (atomic spacer, 2x2):")
print("    - mono_aaa.vasp (a-a-a-)")
print("    - twist_aaa_3_1.vasp")
print("    - mono_tetragonal.vasp (a0a0c-)")
print("    - twist_tetragonal_3_1.vasp")
print("    - mono_mixed.vasp (a+b-c-)")
print("    - twist_mixed_3_1.vasp")
print("    - mono_tilt.vasp (a-a-a- 5°)")
print("    - twist_tilt_3_1.vasp")
print("  With Glazer Tilting (molecular spacer, 2x2):")
print("    - mono_pda_aaa.vasp (a-a-a-)")
print("    - twist_pda_aaa_3_1.vasp")
print("    - mono_pda_tetragonal.vasp (a0a0c-)")
print("    - twist_pda_tetragonal_3_1.vasp")
print("    - mono_pda_tilt.vasp (a-a-a- 5°)")
print("    - twist_pda_tilt_3_1.vasp")
print("  With larger supercell (3x3):")
print("    - mono_pattern.vasp (a-a-a-)")
print("    - twist_pattern_3_1.vasp")
