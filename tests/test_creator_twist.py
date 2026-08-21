import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure
from ase.io import write


def main():
    q2d = q2D_creator()
    vacuum = 15.0  # shared vacuum padding for monolayers

    print("=" * 60)
    print("Testing twist method with 3 monolayers and their permutations")
    print("=" * 60)

    # Create 3 monolayers (m1, m2, m3)
    monolayers = []

    # m1: Super-mixed monolayer (reduced template, similar to test_creator_monolayer.py)
    print("\n1. Creating m1: super-mixed monolayer (reduced, mixed A/B/X, spacers, Glazer)...")
    m1 = q2d.create_structure(
        A_ions=["Cs", "MA", "FA", "MA", "MA", "FA", "Cs", "MA"],
        B_ions=["Pb", "Sn", "Pb", "Pb", "Sn", "Pb", "Pb", "Sn"],
        passivator=["CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]", "CCCC[NH3+]"],
        X_ions=["Br"] * 12 + ["I"] * 12,
        xy_expansion=(2, 2),
        template="reduced",
        structure_type="monolayer",
        vacuum=vacuum,
        thickness=2,
        penetration=[-0.3, 0.5],
        glazer_angles=[0, 0, 3],
        glazer_pattern=["0", "0", "+"],
    )
    write("m1_superMix.vasp", m1, format="vasp", sort=True)
    monolayers.append(m1)
    print(f"✓ Wrote m1_superMix.vasp ({len(m1)} atoms, xy_expansion={m1.xy_expansion})")

    # m2: Mixed A-site with different xy_expansion (cubic template)
    print("\n2. Creating m2: mixed A-site monolayer (cubic, xy_expansion=(3, 3))...")
    m2 = q2d.create_structure(
        A_ions=["Cs", "MA", "FA", "Cs", "MA", "FA", "Cs", "MA", "FA"],
        B_ions="Pb",
        X_ions="I",
        xy_expansion=(3, 3),
        template="cubic",
        structure_type="monolayer",
        vacuum=vacuum,
        passivator="Cs",
    )
    write("m2_mixedA_cubic.vasp", m2, format="vasp", sort=True)
    monolayers.append(m2)
    print(f"✓ Wrote m2_mixedA_cubic.vasp ({len(m2)} atoms, xy_expansion={m2.xy_expansion})")

    # m3: Mixed B/X sites with Glazer tilting (reduced template)
    print("\n3. Creating m3: mixed B/X monolayer with Glazer tilting (reduced)...")
    m3 = q2d.create_structure(
        A_ions="MA",
        B_ions=["Pb", "Sn", "Pb", "Sn"],
        X_ions=["Br", "I", "I", "Br", "I", "I", "Br", "I"],
        xy_expansion=(2, 2),
        template="reduced",
        structure_type="monolayer",
        vacuum=vacuum,
        passivator="CCCC[NH3+]",
        glazer_angles=[0, 2, 2],
        glazer_pattern=["0", "+", "+"],
        thickness=2,
    )
    write("m3_mixedBX_glazer.vasp", m3, format="vasp", sort=True)
    monolayers.append(m3)
    print(f"✓ Wrote m3_mixedBX_glazer.vasp ({len(m3)} atoms, xy_expansion={m3.xy_expansion})")

    # Summary of monolayers
    print("\n" + "=" * 60)
    print("Summary of created monolayers:")
    print("=" * 60)
    for i, mono in enumerate(monolayers, 1):
        print(f"  m{i}: {len(mono)} atoms, xy_expansion={mono.xy_expansion}, "
              f"structure_type={mono.structure_type}")

    print("\nAll 3 monolayers (m1, m2, m3) created and saved successfully!")

    # Create bilayer permutations
    print("\n" + "=" * 60)
    print("Creating bilayer permutations")
    print("=" * 60)

    # A = Twist(m1, m2)
    print("\nCreating A = Twist(m1, m2)...")
    try:
        A = q2d.Twist(
            m1=monolayers[0],  # m1
            m2=monolayers[1],  # m2
            m=3, n=1,
            interlayer_distance=4.0,
            vacuum=4.0
        )
        write("A_twist_m1_m2.vasp", A, format="vasp", sort=True)
        print(f"✓ Wrote A_twist_m1_m2.vasp ({len(A)} atoms)")
    except Exception as e:
        print(f"✗ Error creating A: {e}")
        A = None

    # B = Twist(m1, m3)
    print("\nCreating B = Twist(m1, m3)...")
    try:
        B = q2d.Twist(
            m1=monolayers[0],  # m1
            m2=monolayers[2],  # m3
            m=3, n=1,
            interlayer_distance=4.0,
            vacuum=4.0
        )
        write("B_twist_m1_m3.vasp", B, format="vasp", sort=True)
        print(f"✓ Wrote B_twist_m1_m3.vasp ({len(B)} atoms)")
    except Exception as e:
        print(f"✗ Error creating B: {e}")
        B = None

    # C = Twist(m2, m3)
    print("\nCreating C = Twist(m2, m3)...")
    try:
        C = q2d.Twist(
            m1=monolayers[1],  # m2
            m2=monolayers[2],  # m3
            m=3, n=1,
            interlayer_distance=4.0,
            vacuum=4.0
        )
        write("C_twist_m2_m3.vasp", C, format="vasp", sort=True)
        print(f"✓ Wrote C_twist_m2_m3.vasp ({len(C)} atoms)")
    except Exception as e:
        print(f"✗ Error creating C: {e}")
        C = None

    # Self-twists: Twist(m1, m1), Twist(m2, m2), Twist(m3, m3)
    print("\n" + "=" * 60)
    print("Creating self-twists")
    print("=" * 60)

    for i, mono in enumerate(monolayers, 1):
        print(f"\nCreating self-twist: Twist(m{i}, m{i})...")
        try:
            self_twist = q2d.Twist(
                m1=mono,
                m2=mono,
                m=3, n=1,
                interlayer_distance=4.0,
                vacuum=4.0
            )
            write(f"self_twist_m{i}_m{i}.vasp", self_twist, format="vasp", sort=True)
            print(f"✓ Wrote self_twist_m{i}_m{i}.vasp ({len(self_twist)} atoms)")
        except Exception as e:
            print(f"✗ Error creating self-twist m{i}: {e}")

    # Create trilayer: D = Twist(A, B)
    print("\n" + "=" * 60)
    print("Creating trilayer: D = Twist(A, B)")
    print("=" * 60)

    if A is not None and B is not None:
        try:
            D = q2d.Twist(
                m1=A,  # A (m1+m2)
                m2=B,  # B (m1+m3)
                m=2, n=1,  # Different twist angle for trilayer
                interlayer_distance=4.0,
                vacuum=15.0  # Larger vacuum for trilayer
            )
            write("D_twist_A_B_trilayer.vasp", D, format="vasp", sort=True)
            print(f"✓ Wrote D_twist_A_B_trilayer.vasp ({len(D)} atoms)")
            print("  This is a trilayer structure: (m1+m2) + (m1+m3)")
        except Exception as e:
            print(f"✗ Error creating D: {e}")
    else:
        print("✗ Cannot create D because A or B failed")

    print("\nTwist permutations test completed!")
    print("Created bilayers: A(m1+m2), B(m1+m3), C(m2+m3)")
    print("Created self-twists: m1+m1, m2+m2, m3+m3")
    print("Created trilayer: D(A+B) = (m1+m2) + (m1+m3)")


if __name__ == "__main__":
    main()

