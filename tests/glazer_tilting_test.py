#!/usr/bin/env python3
"""Test Glazer tilting - outputs VASP files for visual inspection.

NOTE: All structures must use at least 2x2 supercell in the XY plane
because a single octahedron cannot physically exhibit tilting patterns.
Glazer notation describes octahedral tilting correlations between
neighboring octahedra, requiring multiple octahedra to be meaningful.
"""

import sys
sys.path.insert(0, '/home/dotempo/Documents/DJ/q2D-Materials')

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

creator = q2D_creator()

print("Glazer Tilting Tests")
print("=" * 40)
print("NOTE: All structures use at least 2x2x2 (bulk) or 2x2xN (2D)")
print("      supercell as required for physical octahedral tilting")
print("=" * 40)

# 1. Small angles (2x2x2 minimum required)
print("\n1. SrTiO3 with small angles (3°)")
s1 = creator.create_perovskite('bulk',
    A_ions='Sr', B_ions='Ti', X_ions='O',
    supercell_size=(2, 2, 2),
    glazer_angles=[3.0, 3.0, 3.0]
)
write('glazer_small_angles.vasp', s1, format='vasp', sort=True)
print(f"   {len(s1)} atoms, cell: {s1.cell.lengths()}")

# 2. Pattern only (a-a-a-)
print("\n2. SrTiO3 with pattern a-a-a-")
s2 = creator.create_perovskite('bulk',
    A_ions='Sr', B_ions='Ti', X_ions='O',
    supercell_size=(2, 2, 2),
    glazer_notation='a-a-a-'
)
write('glazer_pattern_aaa.vasp', s2, format='vasp', sort=True)
print(f"   {len(s2)} atoms, cell: {s2.cell.lengths()}")

# 3. Mixed pattern (a+b-c-)
print("\n3. LaNiO3 with pattern a+b-c-")
s3 = creator.create_perovskite('bulk',
    A_ions='La', B_ions='Ni', X_ions='O',
    supercell_size=(2, 2, 2),
    glazer_pattern=['+', '-', '-'],
    glazer_default_angle=3.0
)
write('glazer_pattern_abc.vasp', s3, format='vasp', sort=True)
print(f"   {len(s3)} atoms, cell: {s3.cell.lengths()}")

# 4. Large angles (5°)
print("\n4. SrTiO3 with large angles (5°)")
s4 = creator.create_perovskite('bulk',
    A_ions='Sr', B_ions='Ti', X_ions='O',
    supercell_size=(2, 2, 2),
    glazer_angles=[5.0, 5.0, 5.0]
)
write('glazer_large_angles.vasp', s4, format='vasp', sort=True)
print(f"   {len(s4)} atoms, cell: {s4.cell.lengths()}")

# 5. Tetragonal pattern (a0a0c-)
print("\n5. BaTiO3 tetragonal (a0a0c-)")
s5 = creator.create_perovskite('bulk',
    A_ions='Ba', B_ions='Ti', X_ions='O',
    supercell_size=(2, 2, 2),
    glazer_notation='a0a0c-'
)
write('glazer_tetragonal.vasp', s5, format='vasp', sort=True)
print(f"   {len(s5)} atoms, cell: {s5.cell.lengths()}")

# ===================================================================
# 2D PEROVSKITES WITH GLAZER TILTING
# All 2D structures require at least 2x2 in XY plane
# ===================================================================
print("\n" + "=" * 40)
print("2D PEROVSKITES WITH GLAZER TILTING")
print("(All use 2x2 in XY plane as physically required)")
print("=" * 40)

# 6. DJ oxide with pattern a-a-a- (n=2 octahedral layers)
print("\n6. DJ SrNbO3 with Rb spacer, pattern a-a-a-, n=2")
s6 = creator.create_perovskite('DJ',
    A_ions='Sr', B_ions='Nb', X_ions='O',
    spacer='Rb',
    supercell=[2, 2], n_layers=2,
    glazer_notation='a-a-a-'
)
write('glazer_dj_aaa.vasp', s6, format='vasp', sort=True)
print(f"   {len(s6)} atoms, cell: {s6.cell.lengths()}")

# 7. DJ oxide with pattern a0a0c- (tetragonal)
print("\n7. DJ CaTaO3 with K spacer, pattern a0a0c-, n=2")
s7 = creator.create_perovskite('DJ',
    A_ions='Ca', B_ions='Ta', X_ions='O',
    spacer='K',
    supercell=[2, 2], n_layers=2,
    glazer_notation='a0a0c-'
)
write('glazer_dj_tetragonal.vasp', s7, format='vasp', sort=True)
print(f"   {len(s7)} atoms, cell: {s7.cell.lengths()}")

# 8. DJ oxide with pattern a+b-c- (mixed)
print("\n8. DJ LaNiO3 with Rb spacer, pattern a+b-c-, n=2")
s8 = creator.create_perovskite('DJ',
    A_ions='La', B_ions='Ni', X_ions='O',
    spacer='Rb',
    supercell=[2, 2], n_layers=2,
    glazer_pattern=['+', '-', '-'],
    glazer_default_angle=3.0
)
write('glazer_dj_mixed.vasp', s8, format='vasp', sort=True)
print(f"   {len(s8)} atoms, cell: {s8.cell.lengths()}")

# 9. RP oxide with pattern a-a-a-
print("\n9. RP SrTiO3 with Rb spacer, pattern a-a-a-, n=1")
s9 = creator.create_perovskite('RP',
    A_ions='Sr', B_ions='Ti', X_ions='O',
    spacer='Rb',
    supercell=[2, 2], n_layers=1,
    glazer_notation='a-a-a-'
)
write('glazer_rp_aaa.vasp', s9, format='vasp', sort=True)
print(f"   {len(s9)} atoms, cell: {s9.cell.lengths()}")

# 10. RP oxide with pattern a0a0c- (tetragonal)
print("\n10. RP CaTiO3 with Cs spacer, pattern a0a0c-, n=1")
s10 = creator.create_perovskite('RP',
    A_ions='MA', B_ions='Ti', X_ions='O',
    spacer='HDA',
    supercell=[2, 2], n_layers=2,
    glazer_notation='a0a0c-'
)
write('glazer_rp_tetragonal.vasp', s10, format='vasp', sort=True)
print(f"   {len(s10)} atoms, cell: {s10.cell.lengths()}")

# 11. RP oxide with pattern a+b-c- (mixed, n=3 layers)
print("\n11. RP LaNiO3 with Rb spacer, pattern a+b-c-, n=3")
s11 = creator.create_perovskite('RP',
    A_ions=['La', 'MA'], B_ions=['Ni', 'Zn'], X_ions=['O', 'Cl', 'Br'],
    spacer='Rb',
    supercell=[4, 4], n_layers=3,
    glazer_pattern=['+', '-', '-'],
    glazer_default_angle=3.0,
)
write('glazer_rp_mixed.vasp', s11, format='vasp', sort=True)
print(f"   {len(s11)} atoms, cell: {s11.cell.lengths()}")

# 12. DJ halide with pattern a-a-a-
print("\n12. DJ CsPbI3 with Cs spacer, pattern a-a-a-, n=1")
s12 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=2,
    glazer_notation='a-a-a-',
)
write('glazer_dj_halide_aaa.vasp', s12, format='vasp', sort=True)
print(f"   {len(s12)} atoms, cell: {s12.cell.lengths()}")

# 13. DJ halide with pattern a0a0c-
print("\n13. DJ CsPbI3 with Cs spacer, pattern a0a0c-, n=1")
s13 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=1,
    glazer_notation='a0a0c-'
)
write('glazer_dj_halide_tetragonal.vasp', s13, format='vasp', sort=True)
print(f"   {len(s13)} atoms, cell: {s13.cell.lengths()}")

# 14. RP halide with pattern a-a-a-
print("\n14. RP CsPbI3 with Cs spacer, pattern a-a-a-, n=1")
s14 = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=1,
    glazer_notation='a-a-a-'
)
write('glazer_rp_halide_aaa.vasp', s14, format='vasp', sort=True)
print(f"   {len(s14)} atoms, cell: {s14.cell.lengths()}")

# 15. RP halide with pattern a0a0c-
print("\n15. RP CsPbI3 with Cs spacer, pattern a0a0c-, n=1")
s15 = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=1,
    glazer_notation='a0a0c-'
)
write('glazer_rp_halide_tetragonal.vasp', s15, format='vasp', sort=True)
print(f"   {len(s15)} atoms, cell: {s15.cell.lengths()}")

# 16. DJ with smaller angles (to show cubic vs orthorhombic cells)
print("\n16. DJ CsPbI3 with Cs spacer, small angle (1°), n=1")
s16 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=1,
    glazer_notation='a-a-a-',
    glazer_default_angle=1.0
)
write('glazer_dj_small.vasp', s16, format='vasp', sort=True)
print(f"   {len(s16)} atoms, cell: {s16.cell.lengths()}")

# 17. RP with smaller angles
print("\n17. RP CsPbI3 with Cs spacer, small angle (1°), n=1")
s17 = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=1,
    glazer_notation='a-a-a-',
    glazer_default_angle=1.0
)
write('glazer_rp_small.vasp', s17, format='vasp', sort=True)
print(f"   {len(s17)} atoms, cell: {s17.cell.lengths()}")

# 18. DJ halide with mixed pattern
print("\n18. DJ CsPbI3 with Cs spacer, pattern a+b-c-, n=1")
s18 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=1,
    glazer_pattern=['+', '-', '-'],
    glazer_default_angle=2.0
)
write('glazer_dj_halide.vasp', s18, format='vasp', sort=True)
print(f"   {len(s18)} atoms, cell: {s18.cell.lengths()}")

# 19. RP halide with mixed pattern
print("\n19. RP CsPbI3 with Cs spacer, pattern a+b-c-, n=1")
s19 = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=1,
    glazer_pattern=['+', '-', '-'],
    glazer_default_angle=2.0
)
write('glazer_rp_halide.vasp', s19, format='vasp', sort=True)
print(f"   {len(s19)} atoms, cell: {s19.cell.lengths()}")

# 20. DJ with n=2 octahedral layers and Glazer pattern
print("\n20. DJ CsPbI3 n=2 with Cs spacer, pattern a-a-a-")
s20 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=2,
    glazer_notation='a-a-a-'
)
write('glazer_dj_pattern.vasp', s20, format='vasp', sort=True)
print(f"   {len(s20)} atoms, cell: {s20.cell.lengths()}")

# 21. RP with n=2 octahedral layers and Glazer pattern
print("\n21. RP CsPbI3 n=2 with Cs spacer, pattern a-a-a-")
s21 = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2], n_layers=2,
    glazer_notation='a-a-a-'
)
write('glazer_rp_pattern.vasp', s21, format='vasp', sort=True)
print(f"   {len(s21)} atoms, cell: {s21.cell.lengths()}")

print("\n" + "=" * 40)
print("Files written:")
print("  Bulk (2x2x2 minimum):")
print("    - glazer_small_angles.vasp")
print("    - glazer_pattern_aaa.vasp")
print("    - glazer_pattern_abc.vasp")
print("    - glazer_large_angles.vasp")
print("    - glazer_tetragonal.vasp")
print("  DJ oxide (2x2xN):")
print("    - glazer_dj_aaa.vasp (a-a-a-, n=2)")
print("    - glazer_dj_tetragonal.vasp (a0a0c-, n=2)")
print("    - glazer_dj_mixed.vasp (a+b-c-, n=2)")
print("  RP oxide (2x2xN):")
print("    - glazer_rp_aaa.vasp (a-a-a-, n=1)")
print("    - glazer_rp_tetragonal.vasp (a0a0c-, n=1)")
print("    - glazer_rp_mixed.vasp (a+b-c-, n=3)")
print("  DJ halide (2x2xN):")
print("    - glazer_dj_halide_aaa.vasp (a-a-a-, n=1)")
print("    - glazer_dj_halide_tetragonal.vasp (a0a0c-, n=1)")
print("    - glazer_dj_halide.vasp (a+b-c-, n=1)")
print("    - glazer_dj_small.vasp (a-a-a- 1°, n=1)")
print("    - glazer_dj_pattern.vasp (n=2)")
print("  RP halide (2x2xN):")
print("    - glazer_rp_halide_aaa.vasp (a-a-a-, n=1)")
print("    - glazer_rp_halide_tetragonal.vasp (a0a0c-, n=1)")
print("    - glazer_rp_halide.vasp (a+b-c-, n=1)")
print("    - glazer_rp_small.vasp (a-a-a- 1°, n=1)")
print("    - glazer_rp_pattern.vasp (n=2)")
