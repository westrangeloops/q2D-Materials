import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.core.structure import q2DStructure
from ase.io import write


def main():
    q2d = q2D_creator()

    print("=" * 60)
    print("Testing bulk perovskite creation (cubic/reduced templates)")
    print("=" * 60)

    # 0. Wrapper check (cubic)
    print("\n0. Verifying q2DStructure wrapper (cubic)...")
    test_structure = q2d.create_structure(
        A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic"
    )
    assert isinstance(test_structure, q2DStructure)
    assert test_structure.structure_type == "bulk"
    assert test_structure.BX_dist is not None
    assert test_structure.A_ions == "MA"
    print("✓ q2DStructure wrapper working (cubic)")

    # 1. Simple bulk cubic
    print("\n1. Simple bulk perovskite (cubic)...")
    bulk_cubic = q2d.create_structure(
        A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="cubic"
    )
    write("MAPbI3_bulk_simple_cubic.vasp", bulk_cubic, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_bulk_simple_cubic.vasp")

    # 2. Simple bulk reduced
    print("\n2. Simple bulk perovskite (reduced)...")
    bulk_reduced = q2d.create_structure(
        A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="reduced"
    )
    write("MAPbI3_bulk_simple_reduced.vasp", bulk_reduced, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_bulk_simple_reduced.vasp")

    # 2b. Simple bulk hexagonal
    print("\n2b. Simple bulk perovskite (hexagonal)...")
    bulk_hex = q2d.create_structure(
        A_ions="MA", B_ions="Pb", X_ions="I", xy_expansion=(1, 1), template="hexagonal"
    )
    write("MAPbI3_bulk_simple_hexagonal.vasp", bulk_hex, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_bulk_simple_hexagonal.vasp")

    # 3. Mixed A-site pattern (cubic)
    print("\n3. Mixed A-site perovskite (pattern, cubic)...")
    mixed_a_cubic = q2d.create_structure(
        A_ions=["Cs", "MA", "FA", "Cs", "MA", "FA", "Cs", "MA"],
        B_ions="Pb",
        X_ions="I",
        xy_expansion=(2, 2),
        template="cubic",
    )
    write("MAPbI3_mixed_A_pattern_cubic.vasp", mixed_a_cubic, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_mixed_A_pattern_cubic.vasp")

    # 4. Mixed X-site pattern (reduced)
    print("\n4. Mixed X-site perovskite (pattern, reduced)...")
    mixed_x_reduced = q2d.create_structure(
        A_ions="MA",
        B_ions="Pb",
        X_ions=["Br", "I", "I", "Br", "I", "I"],
        xy_expansion=(1, 1),
        template="reduced",
    )
    write("MAPbI3_mixed_X_pattern_reduced.vasp", mixed_x_reduced, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_mixed_X_pattern_reduced.vasp")

    # 4b. Mixed X-site pattern (hexagonal)
    print("\n4b. Mixed X-site perovskite (pattern, hexagonal)...")
    mixed_x_hex = q2d.create_structure(
        A_ions="MA",
        B_ions="Pb",
        X_ions=["Br", "I", "I", "Br", "I", "I", "Br", "I"],
        xy_expansion=(1, 1),
        template="hexagonal",
    )
    write("MAPbI3_mixed_X_pattern_hexagonal.vasp", mixed_x_hex, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_mixed_X_pattern_hexagonal.vasp")

    # 5. Fully mixed composition (reduced)
    print("\n5. Super-mixed perovskite (A/B/X patterns, reduced)...")
    super_mix_reduced = q2d.create_structure(
        A_ions=["Cs", "MA", "FA", "MA", "MA", "FA", "Cs", "MA"],
        B_ions=["Pb", "Sn", "Pb", "Pb", "Sn", "Pb", "Pb", "Sn"],
        X_ions=["Br"] * 12 + ["I"] * 12,
        xy_expansion=(2, 2),
        template="reduced",
    )
    write("MAPbI3_superMix_pattern_reduced.vasp", super_mix_reduced, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_superMix_pattern_reduced.vasp")

    # 5b. Super-mixed composition (hexagonal)
    print("\n5b. Super-mixed perovskite (A/B/X patterns, hexagonal)...")
    super_mix_hexagonal = q2d.create_structure(
        A_ions=["Cs", "MA", "FA", "MA", "MA", "FA", "Cs", "MA"],
        B_ions=["Pb", "Sn", "Pb", "Pb", "Sn", "Pb", "Pb", "Sn"],
        X_ions=["Br"] * 12 + ["I"] * 12,
        xy_expansion=(2, 2),
        template="hexagonal",
    )
    write("MAPbI3_superMix_pattern_hexagonal.vasp", super_mix_hexagonal, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_superMix_pattern_hexagonal.vasp")

    # 6-10: Glazer tilts (cubic)
    glazer_cases_cubic = [
        ("cubic_glazer_a0a0c+", [0, 0, 3], ["0", "0", "+"], (1, 1, 1)),
        ("cubic_glazer_a0b+b+", [0, 2, 2], ["0", "+", "+"], (1, 1, 1)),
        ("cubic_glazer_a-a-a-", [2, 2, 2], ["-", "-", "-"], (2, 2, 2)),
        ("cubic_glazer_a0b-c-", [0, 3, 3], ["0", "-", "-"], (2, 2, 1)),
        ("cubic_glazer_a+b-c-", [4, 2, 2], ["+", "-", "-"], (2, 2, 2)),
    ]
    for name, angles, pattern, sc in glazer_cases_cubic:
        print(f"\nGlazer cubic: {name} angles={angles} pattern={pattern} sc={sc}")
        struct = q2d.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
        xy_expansion=(sc[0], sc[1]),
            template="cubic",
            glazer_angles=angles,
            glazer_pattern=pattern,
        )
        write(f"{name}.vasp", struct, format="vasp", sort=True)
        print(f"✓ Wrote {name}.vasp")

    # 11-15: Glazer tilts (reduced)
    glazer_cases_reduced = [
        ("reduced_glazer_a0a0c+", [0, 0, 2], ["0", "0", "+"], (1, 1, 1)),
        ("reduced_glazer_a0b+b+", [0, 1.5, 1.5], ["0", "+", "+"], (1, 1, 2)),
        ("reduced_glazer_a-a-a-", [1.5, 1.5, 1.5], ["-", "-", "-"], (2, 2, 2)),
        ("reduced_glazer_a0b-c-", [0, 2.5, 2.5], ["0", "-", "-"], (2, 2, 1)),
        ("reduced_glazer_a+b-c-", [3, 2, 2], ["+", "-", "-"], (2, 2, 2)),
    ]
    for name, angles, pattern, sc in glazer_cases_reduced:
        print(f"\nGlazer reduced: {name} angles={angles} pattern={pattern} sc={sc}")
        struct = q2d.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
        xy_expansion=(sc[0], sc[1]),
            template="reduced",
            glazer_angles=angles,
            glazer_pattern=pattern,
        )
        write(f"{name}.vasp", struct, format="vasp", sort=True)
        print(f"✓ Wrote {name}.vasp")

    print("\nAll bulk tests completed.")
    # Run the jagodinxky custom sequence example so the VASP file is emitted
    test_jagodinxky_custom_sequence()

    # Test interlayer distance control
    print("\nTesting interlayer distance control...")
    test_interlayer = q2d.create_structure(
        A_ions="MA", B_ions="Pb", X_ions="I",
        xy_expansion=(1, 1), template="cubic",
        layer_sequence="L1-(1.5)-L2-(2.0)-L1"
    )
    assert isinstance(test_interlayer, q2DStructure)
    assert test_interlayer.structure_type == "bulk"

    # Get z positions of floors to verify distances
    positions = test_interlayer.get_positions()
    z_coords = positions[:, 2]
    min_z = z_coords.min()
    max_z = z_coords.max()

    # For a 3-layer structure with explicit gaps: L1-(1.5)-L2-(2.0)-L1
    # Layer 1 (L1) starts at z=0
    # Gap 1.5, then Layer 2 (L2) starts around z=1.5 + layer_thickness (~2.0-2.5)
    # Gap 2.0, then Layer 3 (L1) starts around z=2.5 + 2.0 + layer_thickness (~4.5-5.0)
    expected_z_ranges = [
        (0.0, 1.5),    # First layer (L1) around 0
        (1.5, 3.5),    # Second layer (L2) after gap of 1.5
        (3.5, 6.5)     # Third layer (L1) after gap of 2.0
    ]

    # Check that atoms exist in expected z ranges
    for i, (z_min, z_max) in enumerate(expected_z_ranges):
        atoms_in_range = ((z_coords >= z_min) & (z_coords <= z_max)).sum()
        assert atoms_in_range > 0, f"No atoms found in expected z-range {z_min}-{z_max} for layer {i+1}"

    write("MAPbI3_bulk_custom_distances.vasp", test_interlayer, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_bulk_custom_distances.vasp with interlayer distances L1-(1.5)-L2-(2.0)-L1")

    # Test mixed explicit and default distances
    print("\nTesting mixed explicit/default interlayer distances...")
    test_mixed = q2d.create_structure(
        A_ions="MA", B_ions="Pb", X_ions="I",
        xy_expansion=(1, 1), template="cubic",
        layer_sequence="L1-(2.5)-L2-L1-(1.8)-L2"  # First gap explicit, second gap default BX, third gap explicit
    )
    assert isinstance(test_mixed, q2DStructure)
    write("MAPbI3_bulk_mixed_distances.vasp", test_mixed, format="vasp", sort=True)
    print("✓ Wrote MAPbI3_bulk_mixed_distances.vasp with mixed distances L1-(2.5)-L2-L1-(1.8)-L2")


def test_jagodinxky_custom_sequence():
    q2d = q2D_creator()
    seq = "AcBcAcBaCbAbCbAcBaCaBaCb"
    struct = q2d.create_structure(
        A_ions="Ca",
        B_ions="Ti",
        X_ions="O",
        xy_expansion=(1, 1),
        template="jagodinxky",
        layer_sequence=seq,
    )
    assert isinstance(struct, q2DStructure)
    assert struct.structure_type == "bulk"
    assert struct.BX_dist is not None
    write("CaTiO3_jagodinxky_custom.vasp", struct, format="vasp", sort=True)
    print("✓ Wrote CaTiO3_jagodinxky_custom.vasp")


if __name__ == "__main__":
    main()

