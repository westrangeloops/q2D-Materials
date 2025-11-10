import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

q2d = q2D_creator(B='Pb', X='I', A='MA', name='MAPbI3')

print("="*60)
print("Testing Pattern-Based Perovskite Creation")
print("="*60)

# ===================================================================
# BULK PEROVSKITE TESTS
# ===================================================================
print("\n" + "="*60)
print("BULK PEROVSKITE TESTS")
print("="*60)

# Bulk - simple bulk perovskite (single unit cell)
print("\n1. Simple bulk perovskite...")
bulk = q2d.create_perovskite('bulk', supercell_size=(1, 1, 1))
bulk.write('MAPbI3_bulk_simple.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_bulk_simple.vasp")

# Triple-cation perovskite with explicit pattern
print("\n2. Mixed A-site perovskite (pattern-based)...")
mixed = q2d.create_perovskite('bulk',
    A_ions=['Cs', 'MA', 'FA', 'Cs', 'MA', 'FA', 'Cs', 'MA'],  # Explicit pattern
    supercell_size=(2, 2, 2)  # 2x2x2 = 8 A-site positions
)
mixed.write('MAPbI3_mixed_A_pattern.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_mixed_A_pattern.vasp")

# Mixed halides with pattern
print("\n3. Mixed X-site (halides) perovskite...")
mixed_X = q2d.create_perovskite('bulk',
    X_ions=['Br', 'I', 'I', 'Br', 'I', 'I'],  # Pattern: 1 Br, 2 I repeating
    supercell_size=(1, 1, 2)  # 1x1x2 = 2 unit cells = 6 X-site positions
)
mixed_X.write('MAPbI3_mixed_X_pattern.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_mixed_X_pattern.vasp")

# Full mixed composition
print("\n4. Super-mixed perovskite (A, B, X all mixed)...")
superMix = q2d.create_perovskite('bulk',
    A_ions=['Cs', 'MA', 'FA', 'MA', 'MA', 'FA', 'Cs', 'MA'],  # Pattern
    B_ions=['Pb', 'Sn', 'Pb', 'Pb', 'Sn', 'Pb', 'Pb', 'Sn'],  # Pattern
    X_ions=['Br'] * 12 + ['I'] * 12,  # Half Br, half I
    supercell_size=(2, 2, 2)
)
superMix.write('MAPbI3_superMix_pattern.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_superMix_pattern.vasp")

# ===================================================================
# 2D PEROVSKITE TESTS
# ===================================================================
print("\n" + "="*60)
print("2D PEROVSKITE TESTS")
print("="*60)

# Simple DJ structure (n=1)
print("\n5. Dion-Jacobson (DJ) structure, n=1...")
dj_n1 = q2d.create_perovskite('DJ',
    spacer_molecule='[NH3+]CCCCC[NH3+]',  # SMILES string
    supercell=[1, 1, 1]  # [nx, ny, n_layers]
)
dj_n1.write('MAPbI3_DJ_n1.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_DJ_n1.vasp")

# DJ structure with n=2
print("\n6. Dion-Jacobson (DJ) structure, n=2...")
dj_n2 = q2d.create_perovskite('DJ',
    spacer_molecule='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 2]  # 2 layers
)
dj_n2.write('MAPbI3_DJ_n2.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_DJ_n2.vasp")

# RP structure with n=2
print("\n7. Ruddlesden-Popper (RP) structure, n=2...")
rp_n2 = q2d.create_perovskite('RP',
    spacer_molecule='[NH3+]CCCCC=O',  # Different spacer for RP
    supercell=[1, 1, 2],
    spacer_distance=2.0
)
rp_n2.write('MAPbI3_RP_n2.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_RP_n2.vasp")

# Monolayer structure
print("\n8. Monolayer structure, n=1...")
monolayer = q2d.create_perovskite('monolayer',
    spacer_molecule='CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]',  # Caffeine-based spacer
    supercell=[1, 1, 1],
    vacuum=12
)
monolayer.write('MAPbI3_monolayer.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_monolayer.vasp")

# ===================================================================
# 2D WITH PATTERN-BASED MIXING
# ===================================================================
print("\n" + "="*60)
print("2D PEROVSKITE WITH PATTERN-BASED MIXING")
print("="*60)

# DJ with mixed A-site cations
print("\n9. DJ structure with mixed A-site cations (pattern-based)...")
# For supercell=[1, 1, 2] (n_layers=2): 
#   Total A-sites = 2 * (2-1) * 1 * 1 = 2
#   Spacers occupy = 1 * 1 = 1 position
#   Available A-sites = 2 - 1 = 1
dj_mixed_A = q2d.create_perovskite('DJ',
    spacer_molecule='[NH3+]CCCCC[NH3+]',
    supercell=[1, 1, 2],
    A_ions=['Cs']  # Correct: 1 A-site (spacers occupy the other position)
)
dj_mixed_A.write('MAPbI3_DJ_mixed_A.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_DJ_mixed_A.vasp")

# DJ with larger supercell to show more A-sites
print("\n10. DJ structure with larger supercell (2x2, n=2)...")
# For supercell=[2, 2, 2] with n_layers=2: 
#   A-sites = (n_layers - 1) × nx × ny = (2-1) × 2 × 2 = 4
#   Spacers (DJ uses 'top' attachment) = 1 z-level × 1 base position × 2×2 = 4
dj_large = q2d.create_perovskite('DJ',
    spacer_molecule='[NH3+]CCCCC[NH3+]',
    supercell=[2, 2, 2],
    A_ions=['Cs', 'MA', 'FA', 'MA']  # Correct: 4 A-sites
)
dj_large.write('MAPbI3_DJ_2x2_n2.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_DJ_2x2_n2.vasp")

# RP with mixed halides
print("\n11. RP structure with mixed halides (pattern-based)...")
# For supercell=[1, 1, 2] (n_layers=2): Expected X-sites = (2 + 8*2) * 1 * 1 = 18
# Using a shorter pattern to demonstrate cycling
rp_mixed_X = q2d.create_perovskite('RP',
    spacer_molecule='[NH3+]CCCCC=O',
    supercell=[1, 1, 2],
    X_ions=['Br', 'I']  # Pattern will cycle to fill 18 positions
)
rp_mixed_X.write('MAPbI3_RP_mixed_X.vasp', format='vasp', sort=True)
print("✓ Wrote MAPbI3_RP_mixed_X.vasp")

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "="*60)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nGenerated files:")
print("  Bulk:")
print("    - MAPbI3_bulk_simple.vasp")
print("    - MAPbI3_mixed_A_pattern.vasp")
print("    - MAPbI3_mixed_X_pattern.vasp")
print("    - MAPbI3_superMix_pattern.vasp")
print("  2D:")
print("    - MAPbI3_DJ_n1.vasp")
print("    - MAPbI3_DJ_n2.vasp")
print("    - MAPbI3_RP_n2.vasp")
print("    - MAPbI3_monolayer.vasp")
print("    - MAPbI3_DJ_mixed_A.vasp")
print("    - MAPbI3_DJ_2x2_n2.vasp")
print("    - MAPbI3_RP_mixed_X.vasp")
print("\nNote: Pattern validation will warn if pattern length doesn't match")
print("expected position count. Patterns will cycle if shorter than expected.")
print("="*60)
