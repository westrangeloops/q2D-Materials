from q2D_Materials.core.creator import q2D_creator
from ase.io import write


def main():
    q2d = q2D_creator()

    glazer_cases_rp = [
        ("rp_glazer_a0a0c+", [0, 0, 20], ["0", "0", "+"], 1, (1, 1, 1), "[NH3+]CCCCC", 0.5),
        ("rp_glazer_a0b+b+", [0, 1.5, 10], ["0", "+", "+"], 2, (1, 1, 2), "[NH3+]CCCCOCCC", [-0.4, 0.4]),
        ("rp_glazer_a-a-a-", [1.5, 1.5, 15], ["-", "-", "-"], 3, (2, 2, 2), "[NH3+]CCCCCCCCC=O", 0.3),
    ]

    slabs = []

    for name, angles, pattern, thickness, sc, spacer, penetration in glazer_cases_rp:
        print(f"\nGlazer RP: {name} angles={angles} pattern={pattern} sc={sc}")
        atoms = q2d.create_perovskite(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="monolayer",
            xy_expansion=(sc[0], sc[1]),
            thickness=thickness,
            spacer=spacer,
            penetration=penetration,
            glazer_angles=angles,
            glazer_pattern=pattern,
        )
        write(f"{name}.vasp", atoms, format="vasp", sort=True)
        slabs.append(atoms)

    A = q2d.Twist(
        m1=slabs[0],
        m2=slabs[1],
        m=3,
        n=1,
        interlayer_distance=4.0,
        vacuum=4.0,
    )
    write("A_rp_slab1_slab2.vasp", A, format="vasp", sort=True)


if __name__ == "__main__":
    main()