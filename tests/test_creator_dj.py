from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure
from ase.io import write

def main():
    q2d = q2D_creator()
    glazer_cases_dj = [
        ("dj_glazer_a0a0c+", [0, 0, 20], ["0", "0", "+"], 1, (1, 1, 1), "[NH3+]CCCCC[NH3+]", 0.5),
        ("dj_glazer_a0b+b+", [0, 1.5, 10], ["0", "+", "+"], 2, (1, 1, 2), "[NH3+]CCCCCCC[NH3+]", [-0.4, 0.4]),
        ("dj_glazer_a-a-a-", [1.5, 1.5, 15], ["-", "-", "-"], 3, (2, 2, 2), "[NH3+]CCCCCCCCC[NH3+]", 0.3),
        ("dj_glazer_a0b-c-", [0, 2.5, 5], ["0", "-", "-"], 4, (1, 1, 5), ["[NH3+]CCCCCCC[NH3+]", "CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]"], 0.7),
        ("dj_glazer_a+b-c-", [3, 2, 6], ["+", "-", "-"], 5, (1, 1, 3), "[NH3+]CCC=CCC[NH3+]", 0.0),
        ("dj_glazer_atomic", [3, 2, 6], ["+", "-", "-"], 5, (1, 1, 3), "Cs", 0.0),
    ]
    
    optimizers = ["Off", "KS", "UFF"]
    
    for optimizer in optimizers:
        print(f"\n--- Testing optimizer: {optimizer} ---")
            
        for name, angles, pattern, thickness, sc, spacer, penetration in glazer_cases_dj:
            output_name = f"{name}_opt{optimizer}"
            print(f"\nGlazer DJ: {name} angles={angles} pattern={pattern} optimizer={optimizer}")
            atoms = q2d.create_structure(
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
                        layer_sequence="DJ",
                        optimizer=optimizer,
                    )
            write(f"{output_name}.vasp", atoms, format="vasp", sort=True)
            print(f"✓ Wrote {output_name}.vasp ({len(atoms)} atoms)")

if __name__ == "__main__":
    main()