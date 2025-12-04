#!/usr/bin/env python3
"""Comprehensive test for DJ, RP, normal, mixed positions, different spacers, Glazer, and twist.
Outputs VASP files for visual inspection."""

import sys
sys.path.insert(0, '/home/dotempo/Documents/DJ/q2D-Materials')

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

creator = q2D_creator()

print("Comprehensive Structure Tests")
print("=" * 60)

# ===================================================================
# DJ STRUCTURES
# ===================================================================
print("\n--- DJ (DION-JACOBSON) STRUCTURES ---")

# 1. DJ normal (complete cell)
print("\n1. DJ n=1, normal (complete cell), CsPbI3, PDA spacer")
dj_normal = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1],
    reduced=False
)
write('test_dj_normal.vasp', dj_normal, format='vasp', sort=True)
print(f"   {len(dj_normal)} atoms")

# 2. DJ with mixed A-site (MA and Cs)
print("\n2. DJ n=1, mixed A-site (MA + Cs), PbI3, PDA spacer")
dj_mixed_A = creator.create_perovskite('DJ',
    A_ions=['MA', 'Cs'], B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 1],
    mixing_ratios={'A': [0.5, 0.5]},
    seed=42
)
write('test_dj_mixed_A.vasp', dj_mixed_A, format='vasp', sort=True)
print(f"   {len(dj_mixed_A)} atoms (mixed MA/Cs)")

# 3. DJ with different spacer (Rb atomic)
print("\n3. DJ n=1, CsPbI3, Rb atomic spacer")
dj_rb = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Rb',
    supercell=[1, 1, 1]
)
write('test_dj_rb_spacer.vasp', dj_rb, format='vasp', sort=True)
print(f"   {len(dj_rb)} atoms")

# 4. DJ with Glazer tilting
print("\n4. DJ n=1, CsPbI3, PDA spacer, Glazer a-a-a-")
dj_glazer = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 1],
    glazer_notation='a-a-a-',
    glazer_default_angle=2.0
)
write('test_dj_glazer.vasp', dj_glazer, format='vasp', sort=True)
print(f"   {len(dj_glazer)} atoms (with Glazer tilting)")

# ===================================================================
# RP STRUCTURES
# ===================================================================
print("\n--- RP (RUDDLESDEN-POPPER) STRUCTURES ---")

# 5. RP normal (complete cell)
print("\n5. RP n=1, normal (complete cell), CsPbI3, PDA spacer")
rp_normal = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1],
    reduced=False
)
write('test_rp_normal.vasp', rp_normal, format='vasp', sort=True)
print(f"   {len(rp_normal)} atoms")

# 6. RP with mixed A-site (FA and Cs)
print("\n6. RP n=1, mixed A-site (FA + Cs), PbI3, PDA spacer")
rp_mixed_A = creator.create_perovskite('RP',
    A_ions=['FA', 'Cs'], B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 1],
    mixing_ratios={'A': [0.5, 0.5]},
    seed=42
)
write('test_rp_mixed_A.vasp', rp_mixed_A, format='vasp', sort=True)
print(f"   {len(rp_mixed_A)} atoms (mixed FA/Cs)")

# 7. RP with different spacer (Cs atomic)
print("\n7. RP n=1, CsPbI3, Cs atomic spacer")
rp_cs = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[1, 1, 1]
)
write('test_rp_cs_spacer.vasp', rp_cs, format='vasp', sort=True)
print(f"   {len(rp_cs)} atoms")

# ===================================================================
# MIXED POSITIONS (MOLECULE + Cs)
# ===================================================================
print("\n--- MIXED A-SITE POSITIONS (MOLECULE + Cs) ---")

# 8. DJ with triple mix (MA, FA, Cs)
print("\n8. DJ n=1, triple A-site mix (MA + FA + Cs), PbI3, PDA spacer")
dj_triple_A = creator.create_perovskite('DJ',
    A_ions=['MA', 'FA', 'Cs'], B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 1],
    mixing_ratios={'A': [0.33, 0.33, 0.34]},
    seed=42
)
write('test_dj_triple_A.vasp', dj_triple_A, format='vasp', sort=True)
print(f"   {len(dj_triple_A)} atoms (mixed MA/FA/Cs)")

# 9. RP with triple mix (MA, FA, Cs)
print("\n9. RP n=1, triple A-site mix (MA + FA + Cs), PbI3, PDA spacer")
rp_triple_A = creator.create_perovskite('RP',
    A_ions=['MA', 'FA', 'Cs'], B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 1],
    mixing_ratios={'A': [0.33, 0.33, 0.34]},
    seed=42
)
write('test_rp_triple_A.vasp', rp_triple_A, format='vasp', sort=True)
print(f"   {len(rp_triple_A)} atoms (mixed MA/FA/Cs)")

# ===================================================================
# TWO DIFFERENT SPACERS
# ===================================================================
print("\n--- TWO DIFFERENT SPACERS ---")

# 10. DJ with PDA spacer (molecular)
print("\n10. DJ n=1, CsPbI3, PDA molecular spacer")
dj_pda = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1]
)
write('test_dj_spacer1_pda.vasp', dj_pda, format='vasp', sort=True)
print(f"   {len(dj_pda)} atoms (PDA spacer)")

# 11. DJ with Rb spacer (atomic) - different from above
print("\n11. DJ n=1, CsPbI3, Rb atomic spacer")
dj_rb2 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Rb',
    supercell=[1, 1, 1]
)
write('test_dj_spacer2_rb.vasp', dj_rb2, format='vasp', sort=True)
print(f"   {len(dj_rb2)} atoms (Rb spacer)")

# ===================================================================
# GLAZER TILTING
# ===================================================================
print("\n--- GLAZER TILTING ---")

# 12. RP with Glazer tilting
print("\n12. RP n=1, CsPbI3, PDA spacer, Glazer a-a-a-")
rp_glazer = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 1],
    glazer_notation='a-a-a-',
    glazer_default_angle=2.0
)
write('test_rp_glazer.vasp', rp_glazer, format='vasp', sort=True)
print(f"   {len(rp_glazer)} atoms (with Glazer tilting)")

# 13. DJ with different Glazer pattern
print("\n13. DJ n=1, CsPbI3, PDA spacer, Glazer a0a0c-")
dj_glazer2 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 1],
    glazer_notation='a0a0c-',
    glazer_default_angle=2.0
)
write('test_dj_glazer2.vasp', dj_glazer2, format='vasp', sort=True)
print(f"   {len(dj_glazer2)} atoms (Glazer a0a0c-)")

# ===================================================================
# TWISTED STRUCTURES
# ===================================================================
print("\n--- TWISTED STRUCTURES ---")

# 14. Monolayer for twisting
print("\n14. Monolayer n=1, CsPbI3, PDA spacer (base for twist)")
mono_base = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 1],
    vacuum=15.0
)
write('test_mono_base.vasp', mono_base, format='vasp', sort=True)
print(f"   {len(mono_base)} atoms")

# 15. Twisted bilayer
print("\n15. Twisted bilayer (3,1) from monolayer")
twist_31 = mono_base.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('test_twist_31.vasp', twist_31, format='vasp', sort=True)
print(f"   {len(twist_31)} atoms (twisted m=3, n=1)")

# 16. Twisted with different parameters
print("\n16. Twisted bilayer (5,2) from monolayer")
twist_52 = mono_base.twist(m=5, n=2, interlayer_distance=11.0, vacuum=15.0)
write('test_twist_52.vasp', twist_52, format='vasp', sort=True)
print(f"   {len(twist_52)} atoms (twisted m=5, n=2)")

# ===================================================================
# COMBINED: GLAZER + TWIST
# ===================================================================
print("\n--- COMBINED: GLAZER + TWIST ---")

# 17. Monolayer with Glazer, then twist
print("\n17. Monolayer with Glazer a-a-a-, then twist (3,1)")
mono_glazer = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 1],
    glazer_notation='a-a-a-',
    glazer_default_angle=2.0,
    vacuum=15.0
)
write('test_mono_glazer.vasp', mono_glazer, format='vasp', sort=True)
print(f"   {len(mono_glazer)} atoms (with Glazer)")
twist_glazer = mono_glazer.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
write('test_twist_glazer.vasp', twist_glazer, format='vasp', sort=True)
print(f"   Twisted: {len(twist_glazer)} atoms (Glazer + twist)")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 60)
print("Files written:")
print("  DJ Structures:")
print("    - test_dj_normal.vasp (normal, complete cell)")
print("    - test_dj_mixed_A.vasp (mixed MA/Cs)")
print("    - test_dj_rb_spacer.vasp (Rb atomic spacer)")
print("    - test_dj_glazer.vasp (with Glazer a-a-a-)")
print("    - test_dj_triple_A.vasp (mixed MA/FA/Cs)")
print("    - test_dj_spacer1_pda.vasp (PDA molecular spacer)")
print("    - test_dj_spacer2_rb.vasp (Rb atomic spacer)")
print("    - test_dj_glazer2.vasp (Glazer a0a0c-)")
print("  RP Structures:")
print("    - test_rp_normal.vasp (normal, complete cell)")
print("    - test_rp_mixed_A.vasp (mixed FA/Cs)")
print("    - test_rp_cs_spacer.vasp (Cs atomic spacer)")
print("    - test_rp_triple_A.vasp (mixed MA/FA/Cs)")
print("    - test_rp_glazer.vasp (with Glazer a-a-a-)")
print("  Twisted Structures:")
print("    - test_mono_base.vasp (base monolayer)")
print("    - test_twist_31.vasp (twisted m=3, n=1)")
print("    - test_twist_52.vasp (twisted m=5, n=2)")
print("    - test_mono_glazer.vasp (monolayer with Glazer)")
print("    - test_twist_glazer.vasp (Glazer + twist)")
print("=" * 60)

