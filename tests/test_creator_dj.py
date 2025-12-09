from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure
from ase.io import write

def main():
    q2d = q2D_creator()
    atoms = q2d.create_perovskite(
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        structure_type="bilayer",
        thickness=2,
        spacer="[NH3+]CCCC[NH3+]",
        penetration=0.0,
        glazer_angles=[0, 0, 3],
        glazer_pattern=["0", "0", "+"],
    )
    write("DJ_perovskite.vasp", atoms, format="vasp", sort=True)
    print("✓ Wrote DJ_perovskite.vasp")

if __name__ == "__main__":
    main()