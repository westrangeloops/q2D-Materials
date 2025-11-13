import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.utils.recomender import get_recommendations

print("="*60)
print("Testing Perovskite Ion Recommender System")
print("="*60)

q2d = q2D_creator()

# ===================================================================
# BASIC RECOMMENDATION TESTS
# ===================================================================
print("\n" + "="*60)
print("BASIC RECOMMENDATION TESTS")
print("="*60)

# Test 1: Recommendations based on X-site (Iodine)
print("\n1. Getting recommendations based on X = 'I' (Iodine)...")
try:
    recs_X = q2d.recommend(X='I', top_n=5)
    
    print(f"   ✓ Recommendations retrieved successfully")
    print(f"   ✓ Found {len(recs_X['A'])} A-site recommendations")
    print(f"   ✓ Found {len(recs_X['B'])} B-site recommendations")
    print(f"   ✓ Found {len(recs_X['X'])} X-site entry (input)")
    print(f"   ✓ Found {len(recs_X['spacer'])} spacer recommendations")
    
    # Verify structure
    assert 'A' in recs_X, "Recommendations should contain 'A' key"
    assert 'B' in recs_X, "Recommendations should contain 'B' key"
    assert 'X' in recs_X, "Recommendations should contain 'X' key"
    assert 'spacer' in recs_X, "Recommendations should contain 'spacer' key"
    
    # Verify X entry matches input
    assert len(recs_X['X']) == 1, "X should contain exactly one entry (the input)"
    assert recs_X['X'][0]['abbreviation'].upper() == 'I', "X entry should match input"
    
    # Verify recommendations are sorted by occurrence (descending)
    if len(recs_X['B']) > 1:
        occurrences = [r['occurrences'] for r in recs_X['B']]
        assert occurrences == sorted(occurrences, reverse=True), "B recommendations should be sorted by occurrence"
    
    # Print top recommendations
    print(f"\n   Top 3 B-site recommendations:")
    for i, rec in enumerate(recs_X['B'][:3], 1):
        print(f"      {i}. {rec['abbreviation']} ({rec['common_name']}) - {rec['occurrences']} occurrences")
    
    print("   ✓ Test 1 passed")
except Exception as e:
    print(f"   ✗ Test 1 failed: {e}")
    raise

# Test 2: Recommendations based on B-site (Lead)
print("\n2. Getting recommendations based on B = 'Pb' (Lead)...")
try:
    recs_B = q2d.recommend(B='Pb', top_n=5)
    
    print(f"   ✓ Recommendations retrieved successfully")
    print(f"   ✓ Found {len(recs_B['A'])} A-site recommendations")
    print(f"   ✓ Found {len(recs_B['B'])} B-site entry (input)")
    print(f"   ✓ Found {len(recs_B['X'])} X-site recommendations")
    print(f"   ✓ Found {len(recs_B['spacer'])} spacer recommendations")
    
    # Verify B entry matches input
    assert len(recs_B['B']) == 1, "B should contain exactly one entry (the input)"
    assert recs_B['B'][0]['abbreviation'].upper() == 'PB', "B entry should match input"
    
    # Print top recommendations
    print(f"\n   Top 3 X-site recommendations:")
    for i, rec in enumerate(recs_B['X'][:3], 1):
        print(f"      {i}. {rec['abbreviation']} ({rec['common_name']}) - {rec['occurrences']} occurrences")
    
    print("   ✓ Test 2 passed")
except Exception as e:
    print(f"   ✗ Test 2 failed: {e}")
    raise

# Test 3: Recommendations based on spacer (PEA)
print("\n3. Getting recommendations based on spacer = 'PEA'...")
try:
    recs_spacer = q2d.recommend(spacer='PEA', top_n=5)
    
    print(f"   ✓ Recommendations retrieved successfully")
    print(f"   ✓ Found {len(recs_spacer['A'])} A-site recommendations")
    print(f"   ✓ Found {len(recs_spacer['B'])} B-site recommendations")
    print(f"   ✓ Found {len(recs_spacer['X'])} X-site recommendations")
    print(f"   ✓ Found {len(recs_spacer['spacer'])} spacer entry (input)")
    
    # Verify spacer entry matches input
    assert len(recs_spacer['spacer']) == 1, "spacer should contain exactly one entry (the input)"
    assert recs_spacer['spacer'][0]['abbreviation'].upper() == 'PEA', "spacer entry should match input"
    
    print("   ✓ Test 3 passed")
except Exception as e:
    print(f"   ✗ Test 3 failed: {e}")
    raise

# ===================================================================
# EDGE CASES AND ERROR HANDLING
# ===================================================================
print("\n" + "="*60)
print("EDGE CASES AND ERROR HANDLING")
print("="*60)

# Test 4: No input provided
print("\n4. Testing error handling - no input provided...")
try:
    try:
        recs_none = q2d.recommend()
        print("   ✗ Should have raised ValueError")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")
    print("   ✓ Test 4 passed")
except Exception as e:
    print(f"   ✗ Test 4 failed: {e}")
    raise

# Test 5: Invalid X-site
print("\n5. Testing error handling - invalid X-site...")
try:
    try:
        recs_invalid_X = q2d.recommend(X='InvalidX')
        print("   ✗ Should have raised ValueError")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")
    print("   ✓ Test 5 passed")
except Exception as e:
    print(f"   ✗ Test 5 failed: {e}")
    raise

# Test 6: Invalid B-site
print("\n6. Testing error handling - invalid B-site...")
try:
    try:
        recs_invalid_B = q2d.recommend(B='InvalidB')
        print("   ✗ Should have raised ValueError")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")
    print("   ✓ Test 6 passed")
except Exception as e:
    print(f"   ✗ Test 6 failed: {e}")
    raise

# Test 7: Invalid spacer
print("\n7. Testing error handling - invalid spacer...")
try:
    try:
        recs_invalid_spacer = q2d.recommend(spacer='InvalidSpacer')
        print("   ✗ Should have raised ValueError")
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"   ✓ Correctly raised ValueError: {e}")
    print("   ✓ Test 7 passed")
except Exception as e:
    print(f"   ✗ Test 7 failed: {e}")
    raise

# ===================================================================
# PARAMETER VARIATIONS
# ===================================================================
print("\n" + "="*60)
print("PARAMETER VARIATIONS")
print("="*60)

# Test 8: Custom top_n
print("\n8. Testing custom top_n parameter...")
try:
    recs_top3 = q2d.recommend(X='I', top_n=3)
    recs_top10 = q2d.recommend(X='I', top_n=10)
    
    assert len(recs_top3['B']) <= 3, "top_n=3 should return at most 3 recommendations"
    assert len(recs_top10['B']) <= 10, "top_n=10 should return at most 10 recommendations"
    assert len(recs_top10['B']) >= len(recs_top3['B']), "top_n=10 should return at least as many as top_n=3"
    
    print(f"   ✓ top_n=3 returned {len(recs_top3['B'])} B-site recommendations")
    print(f"   ✓ top_n=10 returned {len(recs_top10['B'])} B-site recommendations")
    print("   ✓ Test 8 passed")
except Exception as e:
    print(f"   ✗ Test 8 failed: {e}")
    raise

# Test 9: Minimum occurrences filter
print("\n9. Testing min_occurrences filter...")
try:
    recs_min1 = q2d.recommend(X='I', top_n=20, min_occurrences=1)
    recs_min10 = q2d.recommend(X='I', top_n=20, min_occurrences=10)
    
    # All recommendations should meet the minimum threshold
    for rec in recs_min10['B']:
        assert rec['occurrences'] >= 10, f"All recommendations should have >= 10 occurrences, got {rec['occurrences']}"
    
    # min_occurrences=10 should return fewer or equal recommendations
    assert len(recs_min10['B']) <= len(recs_min1['B']), "Higher min_occurrences should filter out more"
    
    print(f"   ✓ min_occurrences=1 returned {len(recs_min1['B'])} B-site recommendations")
    print(f"   ✓ min_occurrences=10 returned {len(recs_min10['B'])} B-site recommendations")
    print("   ✓ Test 9 passed")
except Exception as e:
    print(f"   ✗ Test 9 failed: {e}")
    raise

# ===================================================================
# RECOMMENDATION DATA STRUCTURE VALIDATION
# ===================================================================
print("\n" + "="*60)
print("RECOMMENDATION DATA STRUCTURE VALIDATION")
print("="*60)

# Test 10: Verify recommendation structure
print("\n10. Verifying recommendation data structure...")
try:
    recs = q2d.recommend(X='I', top_n=3)
    
    # Check that all recommendations have required fields
    required_fields = ['abbreviation', 'common_name', 'occurrences', 'molecular_formula', 'smile', 'ion_type']
    
    for category in ['A', 'B', 'X', 'spacer']:
        for rec in recs[category]:
            for field in required_fields:
                assert field in rec, f"Recommendation should have '{field}' field"
            # Verify types
            assert isinstance(rec['abbreviation'], str), "abbreviation should be string"
            assert isinstance(rec['common_name'], str), "common_name should be string"
            assert isinstance(rec['occurrences'], (int, float)), "occurrences should be numeric"
            assert isinstance(rec['molecular_formula'], str), "molecular_formula should be string"
            assert isinstance(rec['smile'], str), "smile should be string"
            assert rec['ion_type'] in ['A', 'B', 'X'], "ion_type should be A, B, or X"
    
    print("   ✓ All recommendations have required fields")
    print("   ✓ All field types are correct")
    print("   ✓ Test 10 passed")
except Exception as e:
    print(f"   ✗ Test 10 failed: {e}")
    raise

# Test 11: Verify sorting by occurrence
print("\n11. Verifying recommendations are sorted by occurrence...")
try:
    recs = q2d.recommend(X='I', top_n=10)
    
    for category in ['A', 'B', 'spacer']:
        if len(recs[category]) > 1:
            occurrences = [r['occurrences'] for r in recs[category]]
            # Should be sorted in descending order
            assert occurrences == sorted(occurrences, reverse=True), \
                f"{category} recommendations should be sorted by occurrence (descending)"
    
    print("   ✓ All recommendations are sorted by occurrence (descending)")
    print("   ✓ Test 11 passed")
except Exception as e:
    print(f"   ✗ Test 11 failed: {e}")
    raise

# ===================================================================
# CASE INSENSITIVITY AND ALTERNATIVE ABBREVIATIONS
# ===================================================================
print("\n" + "="*60)
print("CASE INSENSITIVITY AND ALTERNATIVE ABBREVIATIONS")
print("="*60)

# Test 12: Case insensitive input
print("\n12. Testing case insensitive input...")
try:
    recs_lower = q2d.recommend(X='i', top_n=3)  # lowercase
    recs_upper = q2d.recommend(X='I', top_n=3)  # uppercase
    
    # Should return same results
    assert len(recs_lower['B']) == len(recs_upper['B']), "Case should not affect results"
    assert recs_lower['X'][0]['abbreviation'].upper() == recs_upper['X'][0]['abbreviation'].upper(), \
        "Case should not affect X entry"
    
    print("   ✓ Case insensitive input works correctly")
    print("   ✓ Test 12 passed")
except Exception as e:
    print(f"   ✗ Test 12 failed: {e}")
    raise

# ===================================================================
# SUMMARY
# ===================================================================
print("\n" + "="*60)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("="*60)
print("\nTest Summary:")
print("  ✓ Basic recommendation functionality (X, B, spacer inputs)")
print("  ✓ Error handling for invalid inputs")
print("  ✓ Parameter variations (top_n, min_occurrences)")
print("  ✓ Data structure validation")
print("  ✓ Sorting verification")
print("  ✓ Case insensitivity")
print("="*60)

