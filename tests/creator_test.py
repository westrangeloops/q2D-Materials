from q2D_Materials.core.creator import q2D_creator
from ase.io import write

q2d = q2D_creator(B='Pb', X='I', A='MA', name='MAPbI3')

# Bulk - only relevant parameters
bulk = q2d.create_perovskite('bulk', BX_dist=None, Bp=None, penet=0.3)

# DJ - only DJ-relevant parameters  
dj = q2d.create_perovskite('DJ', spacer_molecule='[NH3+]CCCCC[NH3+]', n=2, A='MA', penet=0.3)

# RP - only RP-relevant parameters
rp = q2d.create_perovskite('RP', spacer_molecule='[NH3+]CCCCC=O', n=2, A='MA', penet=0.3)

# DJ Big Molecules test
dj_big = q2d.create_perovskite('DJ', spacer_molecule='[NH3+]CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=2, A='FA', penet=0.3)

# RP Big Molecules test
rp_big = q2d.create_perovskite('RP', spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=2, A='FA', penet=0.3)

# Monolayer - only monolayer-relevant parameters
monolayer = q2d.create_perovskite('monolayer', spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=1, A='MA', penet=0.3, vacuum=10, attachment_end='top')

# N != 1 tests:
dj_n2 = q2d.create_perovskite('DJ', spacer_molecule='[NH3+]CCCCC[NH3+]', n=2, A='MA', penet=0.3)
rp_n2 = q2d.create_perovskite('RP', spacer_molecule='[NH3+]CCCCC=O', n=2, A='MA', penet=0.3)
rp_big_n2 = q2d.create_perovskite('RP', spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=2, A='MA', penet=0.3)
monolayer_n2 = q2d.create_perovskite('monolayer', spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=2, A='MA', penet=0.3, vacuum=10, attachment_end='top')

bulk.write('MAPbI3_bulk_n1_A_Cs.vasp', format='vasp', sort=True)
dj.write('MAPbI3_DJ_n1_A_Cs.vasp', format='vasp', sort=True)
dj_big.write('MAPbI3_DJ_big_n1_A_Cs.vasp', format='vasp', sort=True)
rp.write('MAPbI3_RP_n1_A_Cs.vasp', format='vasp', sort=True)
rp_big.write('MAPbI3_RP_big_n1_A_Cs.vasp', format='vasp', sort=True)
monolayer.write('MAPbI3_monolayer_n1_A_Cs.vasp', format='vasp', sort=True)
dj_n2.write('MAPbI3_DJ_n2_A_Cs.vasp', format='vasp', sort=True)
rp_n2.write('MAPbI3_RP_n2_A_Cs.vasp', format='vasp', sort=True)
rp_big_n2.write('MAPbI3_RP_big_n2_A_Cs.vasp', format='vasp', sort=True)
monolayer_n2.write('MAPbI3_monolayer_n2_A_Cs.vasp', format='vasp', sort=True)

# Triple-cation perovskite (Cs₀.₀₅MA₀.₇₉FA₀.₁₈PbI₃)
mixed = q2d.create_mixed_bulk(
    A_ions=['Cs', 'MA', 'FA'],
    A_coefficients=[0.05, 0.79, 0.18]
)

mixed.write('MAPbI3_mixed_n1_A_Cs.vasp', format='vasp', sort=True)

# Mixed halides
mixed = q2d.create_mixed_bulk(
    X_ions=['Br', 'I'],
    X_coefficients=[0.5, 2.5]
)
mixed.write('MAPbI3_mixed_n1_X_Cs.vasp', format='vasp', sort=True)

# SuperMix
superMix = q2d.create_mixed_bulk(
    A_ions=['Cs', 'MA', 'FA'],
    A_coefficients=[0.05, 0.79, 0.18],
    B_ions=['Pb'],
    B_coefficients=[1.0],
    X_ions=['Br', 'I'],
    X_coefficients=[0.5, 2.5]
)
superMix.write('MAPbI3_superMix_n1_A_Cs.vasp', format='vasp', sort=True)