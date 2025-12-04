#!/usr/bin/env python3
"""Test creator API - outputs VASP files for visual inspection."""

import sys
sys.path.insert(0, '/home/dotempo/Documents/DJ/q2D-Materials')

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

creator = q2D_creator()

print("Creator API Tests")
print("=" * 50)

# ===================================================================
# BULK PEROVSKITE TESTS
# ===================================================================
print("\n--- BULK PEROVSKITES ---")

# 1. Simple bulk
print("\n1. Simple bulk CsPbI3 (2x2x2)")
bulk = creator.create_perovskite('bulk',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    supercell_size=(2, 2, 2)
)
write('bulk_CsPbI3.vasp', bulk, format='vasp', sort=True)
print(f"   {len(bulk)} atoms")

# 2. Mixed B-site (Pb/Sn alloy)
print("\n2. Mixed B-site Cs(Pb,Sn)I3")
mixed_B = creator.create_perovskite('bulk',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    supercell_size=(2, 2, 2),
    substitutions=[{'old': 'Pb', 'new': 'Sn', 'fraction': 0.5, 'seed': 42}]
)
write('bulk_CsPbSnI3.vasp', mixed_B, format='vasp', sort=True)
print(f"   {len(mixed_B)} atoms")

# 3. Mixed halides
print("\n3. Bulk with X-site vacancies")
vac_X = creator.create_perovskite('bulk',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    supercell_size=(2, 2, 2),
    vacancies={'site_type': 'X_site', 'fraction': 0.1, 'seed': 42}
)
write('bulk_CsPbI3_Ivac.vasp', vac_X, format='vasp', sort=True)
print(f"   {len(vac_X)} atoms (with vacancies)")

# 4. SrTiO3 oxide
print("\n4. Oxide perovskite SrTiO3")
oxide = creator.create_perovskite('bulk',
    A_ions='Sr', B_ions='Ti', X_ions='O',
    supercell_size=(2, 2, 2)
)
write('bulk_SrTiO3.vasp', oxide, format='vasp', sort=True)
print(f"   {len(oxide)} atoms")

# ===================================================================
# 2D PEROVSKITE TESTS
# ===================================================================
print("\n--- 2D PEROVSKITES ---")

# 5. DJ n=1 with molecular spacer
print("\n5. DJ n=1 with PDA spacer")
dj_n1 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1]
)
write('dj_n1_pda.vasp', dj_n1, format='vasp', sort=True)
print(f"   {len(dj_n1)} atoms")

# 6. DJ n=2
print("\n6. DJ n=2 with PDA spacer")
dj_n2 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 2]
)
write('dj_n2_pda.vasp', dj_n2, format='vasp', sort=True)
print(f"   {len(dj_n2)} atoms")

# 7. DJ with atomic spacer
print("\n7. DJ n=2 with Rb spacer")
dj_rb = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Rb',
    supercell=[1, 1, 2]
)
write('dj_n2_rb.vasp', dj_rb, format='vasp', sort=True)
print(f"   {len(dj_rb)} atoms")

# 8. RP n=1
print("\n8. RP n=1 with PDA spacer")
rp_n1 = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1]
)
write('rp_n1_pda.vasp', rp_n1, format='vasp', sort=True)
print(f"   {len(rp_n1)} atoms")

# 9. RP n=2
print("\n9. RP n=2 with PDA spacer")
rp_n2 = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 2]
)
write('rp_n2_pda.vasp', rp_n2, format='vasp', sort=True)
print(f"   {len(rp_n2)} atoms")

# 10. RP with atomic spacer
print("\n10. RP n=1 with Cs spacer")
rp_cs = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[1, 1, 1]
)
write('rp_n1_cs.vasp', rp_cs, format='vasp', sort=True)
print(f"   {len(rp_cs)} atoms")

# 11. ACI phase
print("\n11. ACI n=1 with PDA spacer")
aci = creator.create_perovskite('ACI',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1]
)
write('aci_n1_pda.vasp', aci, format='vasp', sort=True)
print(f"   {len(aci)} atoms")

# 12. Monolayer
print("\n12. Monolayer n=1 with PDA spacer")
mono = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1],
    vacuum=15.0
)
write('mono_n1_pda.vasp', mono, format='vasp', sort=True)
print(f"   {len(mono)} atoms")

# 13. Monolayer with Cs
print("\n13. Monolayer n=1 with Cs spacer")
mono_cs = creator.create_perovskite('monolayer',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[1, 1, 1],
    vacuum=15.0
)
write('mono_n1_cs.vasp', mono_cs, format='vasp', sort=True)
print(f"   {len(mono_cs)} atoms")

# ===================================================================
# LARGER SUPERCELLS
# ===================================================================
print("\n--- LARGER SUPERCELLS ---")

# 14. DJ 2x2 supercell
print("\n14. DJ 2x2 supercell, n=2")
dj_2x2 = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 2]
)
write('dj_2x2_n2.vasp', dj_2x2, format='vasp', sort=True)
print(f"   {len(dj_2x2)} atoms")

# 15. RP 2x2 supercell
print("\n15. RP 2x2 supercell, n=1")
rp_2x2 = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCC[NH3+]',
    supercell=[2, 2, 1]
)
write('rp_2x2_n1.vasp', rp_2x2, format='vasp', sort=True)
print(f"   {len(rp_2x2)} atoms")

# ===================================================================
# OXIDE PEROVSKITES
# ===================================================================
print("\n--- OXIDE 2D PEROVSKITES ---")

# 16. DJ oxide
print("\n16. DJ SrNbO3 with K spacer")
dj_oxide = creator.create_perovskite('DJ',
    A_ions='Sr', B_ions='Nb', X_ions='O',
    spacer='K',
    supercell=[1, 1, 2]
)
write('dj_SrNbO3_K.vasp', dj_oxide, format='vasp', sort=True)
print(f"   {len(dj_oxide)} atoms")

# 17. RP oxide
print("\n17. RP CaTiO3 with Rb spacer")
rp_oxide = creator.create_perovskite('RP',
    A_ions='Ca', B_ions='Ti', X_ions='O',
    spacer='Rb',
    supercell=[1, 1, 1]
)
write('rp_CaTiO3_Rb.vasp', rp_oxide, format='vasp', sort=True)
print(f"   {len(rp_oxide)} atoms")

# ===================================================================
# MIXED COMPOSITION TESTS
# ===================================================================
print("\n--- MIXED COMPOSITIONS ---")

# 18. Mixed B-site in 2D
print("\n18. DJ n=2 with 50% Pb/Sn substitution")
dj_mixed_B = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Rb',
    supercell=[2, 2, 2],
    substitutions=[{'old': 'Pb', 'new': 'Sn', 'fraction': 0.5, 'seed': 42}]
)
write('dj_mixed_B.vasp', dj_mixed_B, format='vasp', sort=True)
print(f"   {len(dj_mixed_B)} atoms")

# 19. Mixed X-site (halides)
print("\n19. DJ n=1 with mixed halides via substitution")
dj_mixed_X = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Rb',
    supercell=[2, 2, 1],
    substitutions=[{'old': 'I', 'new': 'Br', 'fraction': 0.33, 'seed': 42}]
)
write('dj_mixed_X.vasp', dj_mixed_X, format='vasp', sort=True)
print(f"   {len(dj_mixed_X)} atoms")

# 20. X-site vacancies in 2D
print("\n20. RP n=1 with I vacancies")
rp_vac = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='Cs',
    supercell=[2, 2, 1],
    vacancies={'site_type': 'X_site', 'fraction': 0.1, 'seed': 42}
)
write('rp_vacancies.vasp', rp_vac, format='vasp', sort=True)
print(f"   {len(rp_vac)} atoms")

# ===================================================================
# REDUCED CELL TESTS (2-OCTAHEDRA)
# ===================================================================
print("\n--- REDUCED CELL (2-OCTAHEDRA) ---")

# 21. DJ with reduced cell
print("\n21. DJ n=1 with reduced cell (2-octahedra)")
dj_reduced = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1],
    reduced=True
)
write('dj_n1_reduced.vasp', dj_reduced, format='vasp', sort=True)
print(f"   {len(dj_reduced)} atoms (reduced cell)")

# 22. RP with reduced cell
print("\n22. RP n=1 with reduced cell (2-octahedra)")
rp_reduced = creator.create_perovskite('RP',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1],
    reduced=True
)
write('rp_n1_reduced.vasp', rp_reduced, format='vasp', sort=True)
print(f"   {len(rp_reduced)} atoms (reduced cell)")

# 23. Normal (complete cell) for comparison
print("\n23. DJ n=1 with complete cell (4-octahedra) - normal")
dj_normal = creator.create_perovskite('DJ',
    A_ions='Cs', B_ions='Pb', X_ions='I',
    spacer='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 1],
    reduced=False
)
write('dj_n1_normal.vasp', dj_normal, format='vasp', sort=True)
print(f"   {len(dj_normal)} atoms (complete cell)")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "=" * 50)
print("Files written:")
print("  Bulk:")
print("    - bulk_CsPbI3.vasp")
print("    - bulk_CsPbSnI3.vasp (50% Sn)")
print("    - bulk_CsPbI3_Ivac.vasp (10% I vacancies)")
print("    - bulk_SrTiO3.vasp")
print("  DJ:")
print("    - dj_n1_pda.vasp, dj_n2_pda.vasp")
print("    - dj_n2_rb.vasp (atomic spacer)")
print("    - dj_2x2_n2.vasp (2x2 supercell)")
print("    - dj_SrNbO3_K.vasp (oxide)")
print("    - dj_mixed_B.vasp (Pb/Sn alloy)")
print("    - dj_mixed_X.vasp (I/Br mixed)")
print("  RP:")
print("    - rp_n1_pda.vasp, rp_n2_pda.vasp")
print("    - rp_n1_cs.vasp (atomic spacer)")
print("    - rp_2x2_n1.vasp (2x2 supercell)")
print("    - rp_CaTiO3_Rb.vasp (oxide)")
print("    - rp_vacancies.vasp (with vacancies)")
print("  ACI:")
print("    - aci_n1_pda.vasp")
print("  Monolayer:")
print("    - mono_n1_pda.vasp, mono_n1_cs.vasp")
print("  Reduced Cell (2-octahedra):")
print("    - dj_n1_reduced.vasp (DJ with reduced cell)")
print("    - rp_n1_reduced.vasp (RP with reduced cell)")
print("    - dj_n1_normal.vasp (DJ with complete cell for comparison)")
print("=" * 50)
