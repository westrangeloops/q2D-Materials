import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

q2d = q2D_creator(B='Pb', X='I', A='MA', name='MAPbI3')

# Bulk - simple bulk perovskite
bulk = q2d.create_perovskite('bulk')

# DJ - Dion-Jacobson structure
dj = q2d.create_perovskite('DJ', spacer_molecule='[NH3+]CCCCC[NH3+]', n=2)

# RP - Ruddlesden-Popper structure
rp = q2d.create_perovskite('RP', spacer_molecule='[NH3+]CCCCC=O', n=2)

# DJ Big Molecules test
q2d_fa = q2D_creator(B='Pb', X='I', A='FA', name='FAPbI3')
dj_big = q2d_fa.create_perovskite('DJ', spacer_molecule='[NH3+]CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=2)

# RP Big Molecules test
rp_big = q2d_fa.create_perovskite('RP', spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=2)

# Monolayer - monolayer structure
monolayer = q2d.create_perovskite('monolayer', spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=1, vacuum=10, attachment_end='top')

# N != 1 tests:
dj_n2 = q2d.create_perovskite('DJ', spacer_molecule='[NH3+]CCCCC[NH3+]', n=2)
rp_n2 = q2d.create_perovskite('RP', spacer_molecule='[NH3+]CCCCC=O', n=2)
rp_big_n2 = q2d.create_perovskite('RP', spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=2)
monolayer_n2 = q2d.create_perovskite('monolayer', spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]', n=2, vacuum=10, attachment_end='top')

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
mixed = q2d.create_perovskite('bulk',
    A_ions=['Cs', 'MA', 'FA'],
    A_coefficients=[0.05, 0.79, 0.18]
)

mixed.write('MAPbI3_mixed_n1_A_Cs.vasp', format='vasp', sort=True)

# Mixed halides
mixed = q2d.create_perovskite('bulk',
    X_ions=['Br', 'I'],
    X_coefficients=[0.5, 2.5]
)
mixed.write('MAPbI3_mixed_n1_X_Cs.vasp', format='vasp', sort=True)

# SuperMix
superMix = q2d.create_perovskite('bulk',
    A_ions=['Cs', 'MA', 'FA'],
    A_coefficients=[0.05, 0.79, 0.18],
    B_ions=['Pb'],
    B_coefficients=[1.0],
    X_ions=['Br', 'I'],
    X_coefficients=[0.5, 2.5]
)
superMix.write('MAPbI3_superMix_n1_A_Cs.vasp', format='vasp', sort=True)


# Mixed spacers - Pattern-based (deterministic) - no Ap_coefficients needed
mixed_spacers_alt = q2d.create_perovskite('DJ',
    spacer_molecule=['[NH3+]CCCCC[NH3+]', '[NH3+]CCCCC=O'],
    spacer_pattern='alternating',  # Equal weights automatically
    n=2
)
mixed_spacers_alt.write('MAPbI3_mixed_spacers_alternating_n1_A_Cs.vasp', format='vasp', sort=True)

# Mixed spacers - Pattern-based (checkerboard pattern)
mixed_spacers_check = q2d.create_perovskite('DJ',
    spacer_molecule=['[NH3+]CCCCC[NH3+]', '[NH3+]CCCCC=O'],
    spacer_pattern='checkerboard',
    n=2
)
mixed_spacers_check.write('MAPbI3_mixed_spacers_checkerboard_n1_A_Cs.vasp', format='vasp', sort=True)

# Mixed spacers - Pattern-based (random pattern)
mixed_spacers_rand = q2d.create_perovskite('DJ',
    spacer_molecule=['[NH3+]CCCCC[NH3+]', '[NH3+]CCCCC=O'],
    spacer_pattern='random',
    n=2,
    seed=42
)
mixed_spacers_rand.write('MAPbI3_mixed_spacers_random_pattern_n1_A_Cs.vasp', format='vasp', sort=True)

# Mixed spacers - Probability-based (random) - requires Ap_coefficients
mixed_spacers_prob = q2d.create_perovskite('DJ',
    spacer_molecule=['[NH3+]CCCCC[NH3+]', '[NH3+]CCCCC=O'],
    Ap_coefficients=[0.7, 0.3],  # 70/30 random distribution
    n=2,
    seed=42
)
mixed_spacers_prob.write('MAPbI3_mixed_spacers_probability_n1_A_Cs.vasp', format='vasp', sort=True)

# This will error with a helpful message - demonstrating mutually exclusive parameters
# Uncomment to test the error message:
#dj_error = q2d.create_perovskite('DJ',
#     spacer_molecule=['[NH3+]CCCCC[NH3+]', '[NH3+]CCCCC=O'],
#     Ap_coefficients=[0.5, 0.5],
#     spacer_pattern='alternating',  # ERROR: mutually exclusive!
#     n=2
# )
