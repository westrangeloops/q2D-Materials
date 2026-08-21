from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure
from ase.io import write

smiles = "C1=CC=CC=C1C[NH3+]"
smiles_dj = "[NH3+]C1C=CCC=CC1C[NH3+]"

def main():
    q2d = q2D_creator()
    glazer_cases_rp = [
        ("rp_orientation_A", [0, 0, 20], ["0", "0", "+"], 1, smiles, 0.5, "A", "RP"),
        ("rp_orientation_B", [0, 1.5, 10], ["0", "+", "+"], 2, smiles, [-0.4, 0.4], "B", "RP"),
        ("rp_orientation_mixed", [1.5, 1.5, 15], ["-", "-", "-"], 3, smiles, 0.3, ["A", "B"], "RP"),
        ("dj_orientation_A", [0, 2.5, 5], ["0", "-", "-"], 4, smiles_dj, 0.7, "A", "DJ"),
        ("dj_orientation_B", [3, 2, 6], ["+", "-", "-"], 5, smiles_dj, 0.0, "B", "DJ"),
        ("dj_orientation_mixed", [3, 2, 6], ["+", "-", "-"], 5, smiles_dj, 0.0, ["A", "B"], "DJ"),
    ]
    for name, angles, pattern, thickness, spacer, penetration, spacer_orientation, RP_DJ in glazer_cases_rp:
        print(f"\nRP orientation: {name} angles={angles} pattern={pattern} spacer_orientation={spacer_orientation}")
        atoms = q2d.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            template="Reduced",
            structure_type="bulk",
            thickness=thickness,
            spacer=spacer,
            penetration=penetration,
            glazer_angles=angles,
            glazer_pattern=pattern,
            layer_sequence=RP_DJ,
            spacer_orientation=spacer_orientation,
        )
        write(f"{name}.vasp", atoms, format="vasp", sort=True)
        print(f"✓ Wrote {name}.vasp")
    

if __name__ == "__main__":
    main()
