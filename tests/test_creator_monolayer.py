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
    print("Testing monolayer perovskite creation (cubic/reduced templates)")
    print("=" * 60)

    # 0. Wrapper check (cubic)
    print("\n0. Verifying q2DStructure wrapper (cubic monolayer)...")
    test_structure = q2d.create_perovskite(
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        xy_expansion=(1, 1),
        template="cubic",
        structure_type="monolayer",
        vacuum=vacuum,
    )
    assert isinstance(test_structure, q2DStructure)
    assert test_structure.structure_type == "monolayer"
    assert test_structure.BX_dist is not None
    assert test_structure.A_ions == "MA"
    print("✓ q2DStructure wrapper working (cubic monolayer)")

    # 1. Simple monolayer cubic
    print("\n1. Simple monolayer perovskite (cubic)...")
    mono_cubic = q2d.create_perovskite(
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        xy_expansion=(1, 1),
        template="cubic",
        structure_type="monolayer",
        vacuum=vacuum,
    )
    write("MAPbI3_monolayer_simple_cubic.vasp", mono_cubic, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_monolayer_simple_cubic.vasp")

    # 2. Simple monolayer reduced
    print("\n2. Simple monolayer perovskite (reduced)...")
    mono_reduced = q2d.create_perovskite(
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        xy_expansion=(1, 1),
        template="reduced",
        structure_type="monolayer",
        vacuum=vacuum,
    )
    write("MAPbI3_monolayer_simple_reduced.vasp", mono_reduced, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_monolayer_simple_reduced.vasp")

    # 3. Mixed A-site pattern (cubic)
    print("\n3. Mixed A-site monolayer (pattern, cubic)...")
    mixed_a_cubic = q2d.create_perovskite(
        A_ions=["Cs", "MA", "FA", "Cs", "MA", "FA", "Cs", "MA"],
        B_ions="Pb",
        X_ions="I",
        xy_expansion=(2, 2),
        template="cubic",
        structure_type="monolayer",
        vacuum=vacuum,
    )
    write("MAPbI3_monolayer_mixed_A_pattern_cubic.vasp", mixed_a_cubic, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_monolayer_mixed_A_pattern_cubic.vasp")

    # 4. Mixed X-site pattern (reduced)
    print("\n4. Mixed X-site monolayer (pattern, reduced)...")
    mixed_x_reduced = q2d.create_perovskite(
        A_ions="MA",
        B_ions="Pb",
        X_ions=["Br", "I", "I", "Br", "I", "I"],
        xy_expansion=(1, 1),
        template="reduced",
        structure_type="monolayer",
        vacuum=vacuum,
    )
    write("MAPbI3_monolayer_mixed_X_pattern_reduced.vasp", mixed_x_reduced, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_monolayer_mixed_X_pattern_reduced.vasp")

    # 5. Fully mixed composition (reduced)
    print("\n5. Super-mixed monolayer (A/B/X patterns, reduced)...")
    super_mix_reduced = q2d.create_perovskite(
        A_ions=["Cs", "MA", "FA", "MA", "MA", "FA", "Cs", "MA"],
        B_ions=["Pb", "Sn", "Pb", "Pb", "Sn", "Pb", "Pb", "Sn"],
        spacer=["CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]", "CCCC[NH3+]"],
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
    write("MAPbI3_monolayer_superMix_pattern_reduced.vasp", super_mix_reduced, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_monolayer_superMix_pattern_reduced.vasp")

    # 6-10: Glazer tilts (cubic)
    glazer_cases_cubic = [
        ("monolayer_cubic_glazer_a0a0c+", [0, 0, 3], ["0", "0", "+"], (1, 1, 1)),
        ("monolayer_cubic_glazer_a0b+b+", [0, 2, 2], ["0", "+", "+"], (1, 1, 3)),
        ("monolayer_cubic_glazer_a-a-a-", [2, 2, 2], ["-", "-", "-"], (2, 4, 2)),
        ("monolayer_cubic_glazer_a0b-c-", [0, 3, 3], ["0", "-", "-"], (1, 1, 5)),
        ("monolayer_cubic_glazer_a+b-c-", [4, 2, 2], ["+", "-", "-"], (2, 3, 2)),
    ]
    for name, angles, pattern, sc in glazer_cases_cubic:
        print(f"\nGlazer cubic monolayer: {name} angles={angles} pattern={pattern} sc={sc}")
        struct = q2d.create_perovskite(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(sc[0], sc[1]),
            template="cubic",
            glazer_angles=angles,
            glazer_pattern=pattern,
            structure_type="monolayer",
            vacuum=vacuum,
            thickness=sc[2],
        )
        write(f"{name}.vasp", struct, format="vasp", sort=True)
        print(f"✓ Wrote {name}.vasp")

    # 11-15: Glazer tilts (reduced)
    glazer_cases_reduced = [
        ("monolayer_reduced_glazer_a0a0c+", [0, 0, 2], ["0", "0", "+"], (1, 1, 1), "CCCC[NH3+]", "bottom", 0.5),
        ("monolayer_reduced_glazer_a0b+b+", [0, 1.5, 1.5], ["0", "+", "+"], (1, 1, 2), "CCCC[NH3+]", "top", [-0.4, 0.4]),
        ("monolayer_reduced_glazer_a-a-a-", [1.5, 1.5, 1.5], ["-", "-", "-"], (2, 2, 2), "CCCC[NH3+]", "bottom", 0.3),
        ("monolayer_reduced_glazer_a0b-c-", [0, 2.5, 2.5], ["0", "-", "-"], (1, 1, 5), ["CCCC[NH3+]", "CN1C=NC2=C1C(=O)N(C(=O)N2C)CC[NH3+]"], "both", 0.7),
        ("monolayer_reduced_glazer_a+b-c-", [3, 2, 2], ["+", "-", "-"], (1, 1, 3), "CCCC[NH3+]", "top", 0.9),
    ]
    for name, angles, pattern, sc, spacer, attachment_end, penetration in glazer_cases_reduced:
        print(f"\nGlazer reduced monolayer: {name} angles={angles} pattern={pattern} sc={sc}")
        struct = q2d.create_perovskite(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(sc[0], sc[1]),
            template="reduced",
            glazer_angles=angles,
            glazer_pattern=pattern,
            structure_type="monolayer",
            spacer=spacer,
            attachment_end=attachment_end,
            penetration=penetration,
            vacuum=vacuum,
            thickness=sc[2],
        )
        write(f"{name}.vasp", struct, format="vasp", sort=True)
        print(f"✓ Wrote {name}.vasp")

    print("\nAll monolayer tests completed.")


if __name__ == "__main__":
    main()

