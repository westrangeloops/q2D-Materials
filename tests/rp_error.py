import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure
from ase.io import write

q2d = q2D_creator()

# RP structure with n=2
print("\n7. Ruddlesden-Popper (RP) structure, n=2...")
rp_n2 = q2d.create_perovskite(
    "RP",
    A_ions=["MA", "FA"],
    B_ions=["Pb", "Sn"],
    X_ions=["I", "Cl", "Br"],
    spacer_molecule="[NH3+]CCCCC=O",  # Different spacer for RP
    supercell=[1, 1, 2],
    spacer_distance=2.0,
)
# Verify q2DStructure
assert isinstance(rp_n2, q2DStructure), "create_perovskite should return q2DStructure"
assert rp_n2.structure_type == 'rp', "Structure type should be 'rp'"
print(f"✓ Structure type: {rp_n2.structure_type}")
print(f"✓ BX_dist: {rp_n2.BX_dist:.3f} Å")
write("MAPbI3_RP_ERROR.vasp", rp_n2, format="vasp", sort=True)
print("✓ Wrote MAPbI3_RP_ERROR.vasp")
