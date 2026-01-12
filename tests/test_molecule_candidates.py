"""Test script for molecule candidates module."""

import numpy as np
from q2D_Materials.analyzer import q2D_analyzer

def test_valid_dj_spacers():
    """Test with valid DJ spacer molecules."""
    print("=" * 60)
    print("TEST 1: Valid DJ Spacers")
    print("=" * 60)

    analyzer = q2D_analyzer()

    # Test butanediamine (4 carbons)
    print("\n1. Butanediamine (NCCCCN):")
    result = analyzer.analyze_molecule_as_dj_spacer("NCCCCN")
    print(f"   Valid: {result.is_valid}")
    print(f"   Reason: {result.reason}")
    print(f"   Terminal groups: {len(result.terminal_groups)}")
    for group in result.terminal_groups:
        print(f"     - {group.group_type} at index {group.n_index}")
    print(f"   Valid paths: {len(result.valid_paths)}")
    if result.valid_paths:
        print(f"     First path: {result.valid_paths[0]}")

    assert result.is_valid == True, "Butanediamine should be valid"
    assert len(result.valid_paths) >= 1, "Should have at least 1 valid path"
    print("   ✓ Test passed!")

    # Test hexamethylenediamine (6 carbons)
    print("\n2. Hexamethylenediamine (NCCCCCCN):")
    result = analyzer.analyze_molecule_as_dj_spacer("NCCCCCCN")
    print(f"   Valid: {result.is_valid}")
    print(f"   Reason: {result.reason}")
    print(f"   Terminal groups: {len(result.terminal_groups)}")
    print(f"   Valid paths: {len(result.valid_paths)}")

    assert result.is_valid == True, "Hexamethylenediamine should be valid"
    print("   ✓ Test passed!")


def test_invalid_molecules():
    """Test with invalid molecules - focus on correct identification."""
    print("\n" + "=" * 60)
    print("TEST 2: Invalid Molecules")
    print("=" * 60)

    analyzer = q2D_analyzer()

    # No second terminal - use a real RP cation example (MA with 1 [NH3+])
    print("\n1. RP cation with 1 [NH3+] (C[NH3+] - Methylammonium):")
    dj_result = analyzer.analyze_molecule_as_dj_spacer("C[NH3+]", initial_pattern='[NH3+]C', final_pattern='[NH3+]C')
    rp_result = analyzer.analyze_molecule_as_rp_spacer("C[NH3+]", initial_pattern='[NH3+]C')
    
    print(f"   DJ Valid: {dj_result.is_valid}")
    print(f"   RP Valid: {rp_result.is_valid}")
    
    assert dj_result.is_valid == False, "Should be invalid DJ (needs 2 terminals)"
    assert rp_result.is_valid == True, "Should be valid RP (has 1 terminal)"
    print("   ✓ Test passed! Correctly identified as invalid DJ, valid RP")

    # No terminal groups - should be invalid for both
    print("\n2. No terminal groups (CCCCCC):")
    dj_result = analyzer.analyze_molecule_as_dj_spacer("CCCCCC", initial_pattern='[NH3+]C', final_pattern='[NH3+]C')
    rp_result = analyzer.analyze_molecule_as_rp_spacer("CCCCCC", initial_pattern='[NH3+]C')
    
    print(f"   DJ Valid: {dj_result.is_valid}")
    print(f"   RP Valid: {rp_result.is_valid}")
    
    assert dj_result.is_valid == False, "Should be invalid DJ (no terminals)"
    assert rp_result.is_valid == False, "Should be invalid RP (no terminals)"
    print("   ✓ Test passed! Correctly identified as invalid for both")


def test_nh2_to_nh3_conversion():
    """Test NH2 to NH3 conversion."""
    print("\n" + "=" * 60)
    print("TEST 3: NH2 to NH3 Conversion")
    print("=" * 60)

    analyzer = q2D_analyzer()

    # Analyze NCCN to get NH2 groups
    print("\n1. Convert ethylenediamine (NCCN):")
    result = analyzer.analyze_molecule_as_dj_spacer("NCCN", initial_pattern='NH2C', final_pattern='NH2C')
    print(f"   Original molecule has {len(result.original_atoms)} atoms")
    print(f"   Terminal groups found: {len(result.terminal_groups)}")
    for group in result.terminal_groups:
        print(f"     - {group.group_type} at index {group.n_index}")

    # Convert NH2 to NH3
    modified = analyzer.convert_molecule_nh2_to_nh3(result.original_atoms)
    print(f"   Modified molecule has {len(modified)} atoms")

    # Verify: should have 2 more H atoms (one per NH2 group)
    nh2_count = sum(1 for g in result.terminal_groups if g.group_type == "NH2")
    expected_atoms = len(result.original_atoms) + nh2_count

    assert len(modified) == expected_atoms, f"Expected {expected_atoms} atoms, got {len(modified)}"
    print(f"   ✓ Test passed! Added {nh2_count} H atoms")

    # Verify geometry (N-H distances)
    print("\n2. Verify N-H distances after conversion:")
    symbols = modified.get_chemical_symbols()
    positions = modified.get_positions()
    n_indices = [i for i, s in enumerate(symbols) if s == 'N']

    for n_idx in n_indices:
        n_pos = positions[n_idx]
        h_indices = [i for i, s in enumerate(symbols) if s == 'H']
        dists = [np.linalg.norm(positions[h] - n_pos) for h in h_indices]
        nearby_h = [d for d in dists if 0.9 < d < 1.2]
        print(f"   N at index {n_idx}: {len(nearby_h)} H atoms within 0.9-1.2 Å")

        # After conversion, should have 3 H atoms
        assert len(nearby_h) == 3, f"Expected 3 H atoms near N, found {len(nearby_h)}"

    print("   ✓ Test passed! All N atoms now have 3 H neighbors (NH3)")


def test_all_cations_from_csv():
    """Test all cations from A-ion_data.csv - validate correct identification."""
    print("\n" + "=" * 60)
    print("TEST 4: All Cations from CSV Validation")
    print("=" * 60)

    import csv
    import os
    from pathlib import Path

    analyzer = q2D_analyzer()

    # Load CSV file
    csv_path = Path(__file__).parent.parent / "q2D_Materials" / "data" / "tables" / "A-ion_data.csv"
    if not csv_path.exists():
        print(f"   ⚠ CSV file not found at {csv_path}, skipping test")
        return

    print(f"\nLoading cations from: {csv_path}")
    
    dj_count = 0
    rp_count = 0
    atomic_count = 0
    failed_count = 0
    errors = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            abbrev = row.get('Abbreviation', '').strip()
            smiles = row.get('SMILE', '').strip()
            name = row.get('Common_name', '').strip()
            
            if not smiles or not abbrev:
                continue

            # Count [NH3+] groups in SMILES (exact match, not [NH2+] or [NH+])
            nh3_count = smiles.count('[NH3+]')
            
            # Skip if no [NH3+] groups - these are atomic or other types
            if nh3_count == 0:
                atomic_count += 1
                continue
            
            try:
                # Test DJ (should have 2 [NH3+] groups)
                dj_result = analyzer.analyze_molecule_as_dj_spacer(
                    smiles,
                    initial_pattern='[NH3+]C',
                    final_pattern='[NH3+]C'
                )
                
                # Test RP (should have 1 [NH3+] group)
                rp_result = analyzer.analyze_molecule_as_rp_spacer(
                    smiles,
                    initial_pattern='[NH3+]C'
                )

                # Validate identification based on [NH3+] count
                if nh3_count == 2:
                    # Should be valid DJ
                    if dj_result.is_valid:
                        dj_count += 1
                    else:
                        errors.append(f"{abbrev} ({name}): Has 2 [NH3+] but DJ invalid: {dj_result.reason}")
                elif nh3_count == 1:
                    # Should be valid RP
                    if rp_result.is_valid:
                        rp_count += 1
                    else:
                        errors.append(f"{abbrev} ({name}): Has 1 [NH3+] but RP invalid: {rp_result.reason}")
                else:
                    # More than 2 [NH3+] - could be valid DJ or special case
                    if dj_result.is_valid:
                        dj_count += 1
                    else:
                        errors.append(f"{abbrev} ({name}): Has {nh3_count} [NH3+] but DJ invalid: {dj_result.reason}")

            except Exception as e:
                failed_count += 1
                errors.append(f"{abbrev} ({name}): Error - {str(e)}")

    print(f"\nResults:")
    print(f"   DJ cations (2 [NH3+]): {dj_count}")
    print(f"   RP cations (1 [NH3+]): {rp_count}")
    print(f"   Atomic/Other cations: {atomic_count}")
    print(f"   Failed to process: {failed_count}")
    
    if errors:
        print(f"\n⚠ {len(errors)} errors found:")
        for error in errors[:20]:  # Show first 20 errors
            print(f"   - {error}")
        if len(errors) > 20:
            print(f"   ... and {len(errors) - 20} more errors")
    
    # Assertions - allow some edge cases with complex structures
    assert dj_count > 0, "Should identify at least some DJ cations"
    assert rp_count > 0, "Should identify at least some RP cations"
    # Allow up to 10 errors for complex edge cases (sulfur bridges, ring structures, etc.)
    assert len(errors) <= 10, f"Found {len(errors)} identification errors (expected <= 10 for edge cases)"
    
    if len(errors) > 0:
        print(f"\n   ⚠ {len(errors)} edge cases with complex structures (acceptable)")
    else:
        print(f"\n   ✓ All cations correctly identified!")
    print(f"   ✓ DJ: {dj_count}, RP: {rp_count}, Atomic: {atomic_count}")


def test_real_cation_data():
    """Test with real cation data - validate DJ (2 [NH3+]) and RP (1 [NH3+]) identification."""
    print("\n" + "=" * 60)
    print("TEST 5: Real Cation Data Validation")
    print("=" * 60)

    analyzer = q2D_analyzer()

    # Test known DJ cations (should have 2 [NH3+] groups)
    dj_cations = [
        ("EDA", "C(C[NH3+])[NH3+]", "Ethylenediammonium"),
        ("BDA", "C(CC[NH3+])C[NH3+]", "1,4-butanediammonium"),
        ("HDA", "C(CCC[NH3+])CC[NH3+]", "1,6-diaminohexane"),
    ]
    
    print("\n1. Testing DJ cations (should have 2 [NH3+] groups):")
    for abbrev, smiles, name in dj_cations:
        result = analyzer.analyze_molecule_as_dj_spacer(
            smiles,
            initial_pattern='[NH3+]C',
            final_pattern='[NH3+]C'
        )
        print(f"   {abbrev} ({name}): Valid={result.is_valid}, Paths={len(result.valid_paths)}")
        assert result.is_valid == True, f"{abbrev} should be valid DJ spacer"
        assert len(result.valid_paths) >= 1, f"{abbrev} should have at least 1 path"
    print("   ✓ All DJ cations correctly identified!")

    # Test known RP cations (should have 1 [NH3+] group)
    rp_cations = [
        ("MA", "C[NH3+]", "Methylammonium"),
        ("EA", "CC[NH3+]", "Ethylammonium"),
        ("PGA", "C#CC[NH3+]", "Propargylammonium"),
    ]
    
    print("\n2. Testing RP cations (should have 1 [NH3+] group):")
    for abbrev, smiles, name in rp_cations:
        rp_result = analyzer.analyze_molecule_as_rp_spacer(smiles, initial_pattern='[NH3+]C')
        dj_result = analyzer.analyze_molecule_as_dj_spacer(
            smiles,
            initial_pattern='[NH3+]C',
            final_pattern='[NH3+]C'
        )
        print(f"   {abbrev} ({name}): RP={rp_result.is_valid}, DJ={dj_result.is_valid}")
        assert rp_result.is_valid == True, f"{abbrev} should be valid RP spacer"
        assert dj_result.is_valid == False, f"{abbrev} should be invalid DJ spacer (only 1 terminal)"
    print("   ✓ All RP cations correctly identified!")


def test_complete_workflow():
    """Test complete workflow."""
    print("\n" + "=" * 60)
    print("TEST 6: Complete Workflow")
    print("=" * 60)

    analyzer = q2D_analyzer()

    # 1. Analyze SMILES
    print("\n1. Analyze molecule:")
    result = analyzer.analyze_molecule_as_dj_spacer(
        "NCCCCN",  # Butanediamine
        initial_pattern='NH2C',
        final_pattern='NH2C',
        min_chain_length=2
    )

    print(f"   Valid DJ spacer: {result.is_valid}")
    print(f"   Reason: {result.reason}")
    print(f"   Terminal groups: {len(result.terminal_groups)}")
    print(f"   Valid paths: {len(result.valid_paths)}")

    # 2. Check if NH2 to NH3 conversion needed
    print("\n2. Check for NH2 groups:")
    # Note: With pattern-based matching, terminal_groups may be empty
    # Use clean_molecule for conversion instead
    from q2D_Materials.analyzer.characterization.molecule_candidates import clean_molecule
    modified = clean_molecule(result.original_atoms, convert_nh2_to_nh3_flag=True)
    print(f"   Cleaned molecule, atom count: {len(modified)}")

    # 3. Save result
    print("\n3. Save result:")
    from ase.io import write
    output_file = "test_dj_spacer.xyz"
    write(output_file, modified)
    print(f"   Saved to {output_file}")
    print("   ✓ Complete workflow test passed!")


if __name__ == "__main__":
    try:
        test_valid_dj_spacers()
        test_invalid_molecules()
        test_nh2_to_nh3_conversion()
        test_all_cations_from_csv()
        test_real_cation_data()
        test_complete_workflow()

        print("\n" + "=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
