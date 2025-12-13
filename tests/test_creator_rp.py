from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure
from ase.io import write

def main():
    q2d = q2D_creator()
    glazer_cases_rp = [
        ("rp_glazer_a0a0c+", [0, 0, 20], ["0", "0", "+"], 1, (1, 1, 1), "CCCCC[NH3+]", 0.5),
        ("rp_glazer_a0b+b+", [0, 1.5, 10], ["0", "+", "+"], 2, (1, 1, 2), "[NH3+]CCCCCCC", [-0.4, 0.4]),
        ("rp_glazer_a-a-a-", [1.5, 1.5, 15], ["-", "-", "-"], 3, (2, 2, 2), "[NH3+]CCCCCCCCC=O", 0.3),
        ("rp_double-", [0, 2.5, 5], ["0", "-", "-"], 4, (1, 1, 5), ["C=CCC=CCC[NH3+]", "CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]"], 0.7),
        ("rp_glazer_a+b-c-", [3, 2, 6], ["+", "-", "-"], 5, (1, 1, 3), "CCC=C[NH3+]", 0.0),
        ("rp_glazer_atomic", [3, 2, 6], ["+", "-", "-"], 5, (1, 1, 3), "Cs", -0.5),
        ("rpdj", [0, 2.5, 5], ["0", "-", "-"], 4, (1, 1, 5), ["C=CCC=CCC[NH3+]", "[NH3+]CCCC[NH3+]"], 0.0),
    ]
    for name, angles, pattern, thickness, sc, spacer, penetration in glazer_cases_rp:
        print(f"\nGlazer RP: {name} angles={angles} pattern={pattern} sc={sc}")
        atoms = q2d.create_perovskite(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            xy_expansion=(sc[0], sc[1]),
            thickness=thickness,
            sharp_spacer=spacer,
            penetration=penetration,
            glazer_angles=angles,
            glazer_pattern=pattern,
            layer_sequence="RP",
        )
        write(f"{name}.vasp", atoms, format="vasp", sort=True)
        print(f"✓ Wrote {name}.vasp")

if __name__ == "__main__":
    main()