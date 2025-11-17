"""
Test for twist() method on monolayer structures.

Tests the twisted bilayer generation with different (m, n) parameters.
Based on Gabriel Xavier Pereira's Quadratic Twisted Bilayer Generator.
"""

from q2D_Materials.core.creator import q2D_creator
from ase.io import write

def test_twist_monolayer():
    """Test twist method with atomic and molecular spacers."""
    
    creator = q2D_creator()
    
    # ===== Test 1: Atomic spacer (Cs) =====
    print("=" * 60)
    print("TEST 1: Monolayer with atomic spacer (Cs)")
    print("=" * 60)
    
    monolayer_atomic = creator.create_perovskite(
        'monolayer',
        A_ions='MA',
        B_ions='Pb',
        X_ions='I',
        spacer='Cs',
        supercell=[1, 1, 1]
    )
    
    print(f"Original monolayer: {len(monolayer_atomic)} atoms")
    print(f"Structure type: {monolayer_atomic.structure_type}")
    
    # Test 1a: Small twist angle (m=3, n=1)
    print("\n--- Test 1a: (m=3, n=1) with atomic spacer ---")
    twisted_1a = monolayer_atomic.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
    print(f"Twisted bilayer atoms: {len(twisted_1a)}")
    print(f"Twist params: {twisted_1a.twist_params}")
    print(f"Interlayer distance: {twisted_1a.interlayer_distance} Å")
    print(f"Vacuum: {twisted_1a.vacuum} Å")
    write('twisted_atomic_3_1.vasp', twisted_1a)
    print("Saved to twisted_atomic_3_1.vasp")
    
    # Test 1b: Medium twist angle (m=5, n=2)
    print("\n--- Test 1b: (m=5, n=2) with atomic spacer ---")
    twisted_1b = monolayer_atomic.twist(m=5, n=2, interlayer_distance=11.0, vacuum=15.0)
    print(f"Twisted bilayer atoms: {len(twisted_1b)}")
    print(f"Twist params: {twisted_1b.twist_params}")
    print(f"Interlayer distance: {twisted_1b.interlayer_distance} Å")
    print(f"Vacuum: {twisted_1b.vacuum} Å")
    write('twisted_atomic_5_2.vasp', twisted_1b)
    print("Saved to twisted_atomic_5_2.vasp")
    
    # ===== Test 2: Molecular spacer (PEA-like) =====
    print("\n" + "=" * 60)
    print("TEST 2: Monolayer with molecular spacer")
    print("=" * 60)
    
    monolayer_molecular = creator.create_perovskite(
        'monolayer',
        A_ions='MA',
        B_ions='Pb',
        X_ions='I',
        spacer='[NH3+]CCCCC[NH3+]',  # Pentanediammonium (PEA-like)
        supercell=[1, 1, 1]
    )
    
    print(f"Original monolayer: {len(monolayer_molecular)} atoms")
    print(f"Structure type: {monolayer_molecular.structure_type}")
    
    # Test 2a: Small twist angle (m=3, n=1)
    print("\n--- Test 2a: (m=3, n=1) with molecular spacer ---")
    twisted_2a = monolayer_molecular.twist(m=3, n=1, interlayer_distance=11.0, vacuum=12.0)
    print(f"Twisted bilayer atoms: {len(twisted_2a)}")
    print(f"Twist params: {twisted_2a.twist_params}")
    print(f"Interlayer distance: {twisted_2a.interlayer_distance} Å")
    print(f"Vacuum: {twisted_2a.vacuum} Å")
    write('twisted_molecular_3_1.vasp', twisted_2a)
    print("Saved to twisted_molecular_3_1.vasp")
    
    # Test 2b: Medium twist angle (m=5, n=2)
    print("\n--- Test 2b: (m=5, n=2) with molecular spacer ---")
    twisted_2b = monolayer_molecular.twist(m=5, n=2, interlayer_distance=11.0, vacuum=15.0)
    print(f"Twisted bilayer atoms: {len(twisted_2b)}")
    print(f"Twist params: {twisted_2b.twist_params}")
    print(f"Interlayer distance: {twisted_2b.interlayer_distance} Å")
    print(f"Vacuum: {twisted_2b.vacuum} Å")
    write('twisted_molecular_5_2.vasp', twisted_2b)
    print("Saved to twisted_molecular_5_2.vasp")
    
    # ===== Test 3: Different molecular spacer (BA-like) =====
    print("\n" + "=" * 60)
    print("TEST 3: Monolayer with different molecular spacer (BA-like)")
    print("=" * 60)
    
    monolayer_ba = creator.create_perovskite(
        'monolayer',
        A_ions='MA',
        B_ions='Pb',
        X_ions='I',
        spacer='[NH3+]CCCC[NH3+]',  # Butanediammonium (BA-like)
        supercell=[1, 1, 1]
    )
    
    print(f"Original monolayer: {len(monolayer_ba)} atoms")
    
    # Test 3a: Large twist angle (m=7, n=3)
    print("\n--- Test 3a: (m=7, n=3) with BA-like spacer ---")
    twisted_3a = monolayer_ba.twist(m=7, n=3, interlayer_distance=11.0, vacuum=12.0)
    print(f"Twisted bilayer atoms: {len(twisted_3a)}")
    print(f"Twist params: {twisted_3a.twist_params}")
    print(f"Interlayer distance: {twisted_3a.interlayer_distance} Å")
    print(f"Vacuum: {twisted_3a.vacuum} Å")
    write('twisted_ba_7_3.vasp', twisted_3a)
    print("Saved to twisted_ba_7_3.vasp")
    
    # Verify all are q2DStructure objects with correct attributes
    print("\n" + "=" * 60)
    print("Verifying test results...")
    print("=" * 60)
    
    all_twisted = [twisted_1a, twisted_1b, twisted_2a, twisted_2b, twisted_3a]
    
    for i, twisted in enumerate(all_twisted, 1):
        assert hasattr(twisted, 'twist_params'), f"twisted_{i} should have twist_params"
        assert hasattr(twisted, 'interlayer_distance'), f"twisted_{i} should have interlayer_distance"
        assert hasattr(twisted, 'vacuum'), f"twisted_{i} should have vacuum"
        assert len(twisted) > 0, f"twisted_{i} should have atoms"
    
    assert twisted_1a.twist_params == (3, 1), "twisted_1a should have correct twist_params"
    assert twisted_1b.twist_params == (5, 2), "twisted_1b should have correct twist_params"
    assert twisted_2a.twist_params == (3, 1), "twisted_2a should have correct twist_params"
    assert twisted_2b.twist_params == (5, 2), "twisted_2b should have correct twist_params"
    assert twisted_3a.twist_params == (7, 3), "twisted_3a should have correct twist_params"
    
    assert twisted_1a.vacuum == 12.0, "twisted_1a should have correct vacuum"
    assert twisted_1b.vacuum == 15.0, "twisted_1b should have correct vacuum"
    
    print("\n✓ All twist tests passed!")
    print(f"✓ Tested {len(all_twisted)} different configurations")
    print("✓ All structures saved successfully")

if __name__ == "__main__":
    test_twist_monolayer()

