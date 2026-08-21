"""Test script for molecule candidates module.

Tests the unified molecule validation API:
- analyzer.mol_validate(molecule, spacer_type="DJ" or "RP", ...)
- Uses only SMILES strings as input
- Simple pass/fail validation
"""

from q2D_Materials.analyzer import q2D_analyzer


def test_dj_spacers():
    """Test DJ spacer validation - molecules with 2 terminal groups."""
    analyzer = q2D_analyzer()
    
    # Valid DJ spacers (2 terminals) - use appropriate patterns
    test_cases = [
        ("NCCCCN", "[NH2]C", "[NH2]C"),  # Butanediamine (NH2)
        ("NCCCCCCN", "[NH2]C", "[NH2]C"),  # Hexamethylenediamine (NH2)
        ("C(C[NH3+])[NH3+]", "[NH3+]C", "[NH3+]C"),  # Ethylenediammonium (NH3+)
        ("C(CC[NH3+])C[NH3+]", "[NH3+]C", "[NH3+]C"),  # 1,4-butanediammonium (NH3+)
    ]
    
    for smiles, init_pattern, final_pattern in test_cases:
        dj_result = analyzer.mol_validate(
            smiles,
            spacer_type="DJ",
            initial_pattern=init_pattern,
            final_pattern=final_pattern
        )
        rp_result = analyzer.mol_validate(
            smiles,
            spacer_type="RP",
            initial_pattern=init_pattern
        )
        dj_valid = "T" if dj_result.is_valid else "F"
        rp_valid = "T" if rp_result.is_valid else "F"
        print(f"{smiles} DJ={dj_valid} RP={rp_valid}")
        assert dj_result.is_valid == True, f"{smiles} should be valid DJ spacer"
    
    # Invalid DJ spacers (1 or 0 terminals)
    invalid_dj = [
        ("C[NH3+]", "[NH3+]C", "[NH3+]C"),  # Methylammonium (1 terminal)
        ("CC[NH3+]", "[NH3+]C", "[NH3+]C"),  # Ethylammonium (1 terminal)
        ("CCCCCC", "[NH3+]C", "[NH3+]C"),  # Hexane (0 terminals)
    ]
    
    for smiles, init_pattern, final_pattern in invalid_dj:
        dj_result = analyzer.mol_validate(
            smiles,
            spacer_type="DJ",
            initial_pattern=init_pattern,
            final_pattern=final_pattern
        )
        rp_result = analyzer.mol_validate(
            smiles,
            spacer_type="RP",
            initial_pattern=init_pattern
        )
        dj_valid = "T" if dj_result.is_valid else "F"
        rp_valid = "T" if rp_result.is_valid else "F"
        print(f"{smiles} DJ={dj_valid} RP={rp_valid}")
        assert dj_result.is_valid == False, f"{smiles} should be invalid DJ spacer"


def test_rp_spacers():
    """Test RP spacer validation - molecules with 1+ terminal group."""
    analyzer = q2D_analyzer()
    
    # Valid RP spacers (1+ terminal) - use appropriate patterns
    test_cases = [
        ("C[NH3+]", "[NH3+]C"),  # Methylammonium
        ("CC[NH3+]", "[NH3+]C"),  # Ethylammonium
        ("C#CC[NH3+]", "[NH3+]C"),  # Propargylammonium
        ("NCCCCN", "[NH2]C"),  # Butanediamine (NH2)
    ]
    
    for smiles, pattern in test_cases:
        dj_result = analyzer.mol_validate(
            smiles,
            spacer_type="DJ",
            initial_pattern=pattern,
            final_pattern=pattern
        )
        rp_result = analyzer.mol_validate(
            smiles,
            spacer_type="RP",
            initial_pattern=pattern
        )
        dj_valid = "T" if dj_result.is_valid else "F"
        rp_valid = "T" if rp_result.is_valid else "F"
        print(f"{smiles} DJ={dj_valid} RP={rp_valid}")
        assert rp_result.is_valid == True, f"{smiles} should be valid RP spacer"
    
    # Invalid RP spacers (0 terminals)
    invalid_rp = [
        ("CCCCCC", "[NH3+]C"),  # Hexane
        ("CCCC", "[NH3+]C"),  # Butane
    ]
    
    for smiles, pattern in invalid_rp:
        dj_result = analyzer.mol_validate(
            smiles,
            spacer_type="DJ",
            initial_pattern=pattern,
            final_pattern=pattern
        )
        rp_result = analyzer.mol_validate(
            smiles,
            spacer_type="RP",
            initial_pattern=pattern
        )
        dj_valid = "T" if dj_result.is_valid else "F"
        rp_valid = "T" if rp_result.is_valid else "F"
        print(f"{smiles} DJ={dj_valid} RP={rp_valid}")
        assert rp_result.is_valid == False, f"{smiles} should be invalid RP spacer"


def test_dj_vs_rp():
    """Test that molecules are correctly classified as DJ vs RP."""
    analyzer = q2D_analyzer()
    
    test_cases = [
        # (smiles, pattern, expected_dj, expected_rp)
        ("C[NH3+]", "[NH3+]C", False, True),  # 1 terminal - RP only
        ("NCCCCN", "[NH2]C", True, True),  # 2 terminals - both
        ("CCCCCC", "[NH3+]C", False, False),  # 0 terminals - neither
        ("C(C[NH3+])[NH3+]", "[NH3+]C", True, True),  # 2 terminals - both
        ("CC[NH3+]", "[NH3+]C", False, True),  # 1 terminal - RP only
    ]
    
    for smiles, pattern, expected_dj, expected_rp in test_cases:
        dj_result = analyzer.mol_validate(
            smiles,
            spacer_type="DJ",
            initial_pattern=pattern,
            final_pattern=pattern
        )
        rp_result = analyzer.mol_validate(
            smiles,
            spacer_type="RP",
            initial_pattern=pattern
        )
        
        dj_valid = "T" if dj_result.is_valid else "F"
        rp_valid = "T" if rp_result.is_valid else "F"
        print(f"{smiles} DJ={dj_valid} RP={rp_valid}")
        
        assert dj_result.is_valid == expected_dj, (
            f"{smiles}: Expected DJ={expected_dj}, got {dj_result.is_valid}"
        )
        assert rp_result.is_valid == expected_rp, (
            f"{smiles}: Expected RP={expected_rp}, got {rp_result.is_valid}"
        )


def test_backbone_validation():
    """Test backbone element validation."""
    analyzer = q2D_analyzer()
    
    # Molecule with only C, N, O should pass
    result = analyzer.mol_validate(
        "NCCCCN",
        spacer_type="DJ",
        initial_pattern='[NH2]C',
        final_pattern='[NH2]C',
        allowed_backbone_elements={'C', 'N', 'O'}
    )
    assert result.is_valid == True, "Molecule with C, N, O should be valid"
    
    # Molecule with forbidden elements should be rejected
    # (This test assumes we have a molecule with forbidden elements)
    # For now, just test that the parameter is accepted


def test_pattern_matching():
    """Test that pattern matching works correctly."""
    analyzer = q2D_analyzer()
    
    # Test with NH2 pattern
    result = analyzer.mol_validate(
        "NCCCCN",
        spacer_type="DJ",
        initial_pattern='[NH2]C',
        final_pattern='[NH2]C'
    )
    assert result.is_valid == True, "Should match NH2 pattern"
    
    # Test with NH3+ pattern
    result = analyzer.mol_validate(
        "C[NH3+]",
        spacer_type="RP",
        initial_pattern='[NH3+]C'
    )
    assert result.is_valid == True, "Should match NH3+ pattern"
    
    # Test with wrong pattern (should not match)
    result = analyzer.mol_validate(
        "CCCCCC",
        spacer_type="RP",
        initial_pattern='[NH3+]C'
    )
    assert result.is_valid == False, "Should not match pattern"


def test_all_cations_from_csv():
    """Test all cations from CSV file - print SMILES DJ=T/F RP=T/F."""
    import csv
    from pathlib import Path
    
    analyzer = q2D_analyzer()
    
    # Load CSV file
    csv_path = Path(__file__).parent.parent / "q2D_Materials" / "data" / "tables" / "A-ion_data.csv"
    if not csv_path.exists():
        print(f"⚠ CSV file not found at {csv_path}, skipping test")
        return
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles = row.get('SMILE', '').strip()
            if not smiles:
                continue
            
            # Validate as DJ spacer
            try:
                dj_result = analyzer.mol_validate(
                    smiles,
                    spacer_type="DJ",
                    initial_pattern='[NH3+]C',
                    final_pattern='[NH3+]C'
                )
                dj_valid = "T" if dj_result.is_valid else "F"
            except Exception:
                dj_valid = "F"
            
            # Validate as RP spacer
            try:
                rp_result = analyzer.mol_validate(
                    smiles,
                    spacer_type="RP",
                    initial_pattern='[NH3+]C'
                )
                rp_valid = "T" if rp_result.is_valid else "F"
            except Exception:
                rp_valid = "F"
            
            print(f"{smiles} DJ={dj_valid} RP={rp_valid}")


if __name__ == "__main__":
    test_dj_spacers()
    test_rp_spacers()
    test_dj_vs_rp()
    test_backbone_validation()
    test_pattern_matching()
    test_all_cations_from_csv()
    print("✓ All tests passed!")
