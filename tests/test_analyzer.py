import pytest
import os
from SVC_materials.core.analyzer import q2D_analysis
import numpy as np

# Base path for test structures
TEST_STRUCTURES_DIR = 'tests/structures'

# Test structure paths
TEST_STRUCTURES = {
    'slab1': os.path.join(TEST_STRUCTURES_DIR, 'post_slab1.vasp'),
    'slab2': os.path.join(TEST_STRUCTURES_DIR, 'post_slab2.vasp'),
    'slab3': os.path.join(TEST_STRUCTURES_DIR, 'post_slab3.vasp')
}

@pytest.fixture(scope="session")
def analyzer_instance():
    """Create an analyzer instance for testing"""
    # Verify test structure exists
    test_structure = TEST_STRUCTURES['slab1']  # Start with slab1
    assert os.path.exists(test_structure), f"Test structure file {test_structure} not found"
    
    # Create analyzer instance
    analyzer = q2D_analysis(B='Pb', X='Br', crystal=test_structure)
    return analyzer

def test_analyzer_initialization(analyzer_instance):
    """Test that the analyzer initializes correctly"""
    assert analyzer_instance is not None
    assert analyzer_instance.B == 'Pb'
    assert analyzer_instance.X == 'Br'
    assert os.path.exists(TEST_STRUCTURES['slab1'])

def test_plane_detection(analyzer_instance):
    """Test that the planes are correctly detected"""
    # Test B planes
    lower_plane, upper_plane, has_two_planes = analyzer_instance._find_planes(element_type='B')
    assert lower_plane is not None, "Should have lower B plane"
    assert upper_plane is not None, "Should have upper B plane"
    
    # For single slab case, both planes should be the same
    if lower_plane == upper_plane:
        print("Single slab case detected - using same plane for both lower and upper")
    else:
        assert lower_plane < upper_plane, "Lower B plane should be below upper B plane"
    
    # Test X planes
    lower_plane, upper_plane, has_two_planes = analyzer_instance._find_planes(element_type='X')
    assert has_two_planes, "Should detect two X planes"
    assert lower_plane is not None, "Should have lower X plane"
    assert upper_plane is not None, "Should have upper X plane"
    assert lower_plane < upper_plane, "Lower X plane should be below upper X plane"
    
    # Print plane positions for debugging
    print("\nPlane positions:")
    b_lower, b_upper, _ = analyzer_instance._find_planes(element_type='B')
    x_lower, x_upper, _ = analyzer_instance._find_planes(element_type='X')
    print(f"B planes: lower={b_lower:.3f} Å, upper={b_upper:.3f} Å")
    print(f"X planes: lower={x_lower:.3f} Å, upper={x_upper:.3f} Å")

def test_nitrogen_atom_count(analyzer_instance):
    """Test that we correctly identify the 4 N atoms"""
    n_atoms = analyzer_instance.perovskite_df[analyzer_instance.perovskite_df['Element'] == 'N']
    assert len(n_atoms) == 4, "Should find exactly 4 N atoms"

def test_penetration_calculation(analyzer_instance):
    """Test the nitrogen penetration calculation"""
    # Calculate penetrations
    result = analyzer_instance.calculate_n_penetration()
    
    # Check that we got a result
    assert result is not None, "Penetration calculation failed"
    
    # Check that we have both upper and lower penetrations
    assert 'lower_penetration' in result, "Missing lower plane penetration"
    assert 'upper_penetration' in result, "Missing upper plane penetration"
    assert 'total_penetration' in result, "Missing total penetration"
    
    # Check that we have exactly 2 N atoms for each plane
    assert len(result['lower_atoms']) == 2, "Should have 2 N atoms in lower plane"
    assert len(result['upper_atoms']) == 2, "Should have 2 N atoms in upper plane"
    
    # Check that penetrations are within physical limits (1.5 Å)
    max_allowed_penetration = 1.5  # Maximum allowed penetration in Å
    
    # Get the plane positions
    lower_plane, upper_plane, _ = analyzer_instance._find_planes(element_type='X')
    
    # Check lower plane penetrations
    for atom in result['lower_atoms']:
        penetration = abs(atom['Z'] - lower_plane)
        assert penetration <= max_allowed_penetration, \
            f"Lower plane N atom penetration {penetration:.3f} Å exceeds maximum allowed {max_allowed_penetration} Å"
    
    # Check upper plane penetrations
    for atom in result['upper_atoms']:
        penetration = abs(atom['Z'] - upper_plane)
        assert penetration <= max_allowed_penetration, \
            f"Upper plane N atom penetration {penetration:.3f} Å exceeds maximum allowed {max_allowed_penetration} Å"
    
    # Check that average penetrations are reasonable
    assert result['lower_penetration'] <= max_allowed_penetration, \
        f"Average lower plane penetration {result['lower_penetration']:.3f} Å exceeds maximum allowed {max_allowed_penetration} Å"
    assert result['upper_penetration'] <= max_allowed_penetration, \
        f"Average upper plane penetration {result['upper_penetration']:.3f} Å exceeds maximum allowed {max_allowed_penetration} Å"
    assert result['total_penetration'] <= max_allowed_penetration, \
        f"Total average penetration {result['total_penetration']:.3f} Å exceeds maximum allowed {max_allowed_penetration} Å"
    
    # Print actual penetration values for debugging
    print(f"\nActual penetration values:")
    print(f"Lower plane: {result['lower_penetration']:.3f} Å")
    print(f"Upper plane: {result['upper_penetration']:.3f} Å")
    print(f"Total: {result['total_penetration']:.3f} Å")
    
    # Print individual atom penetrations
    print("\nIndividual atom penetrations:")
    print("Lower plane atoms:")
    for i, atom in enumerate(result['lower_atoms'], 1):
        penetration = abs(atom['Z'] - lower_plane)
        print(f"  N atom {i}: Z = {atom['Z']:.3f} Å, penetration = {penetration:.3f} Å")
    print("Upper plane atoms:")
    for i, atom in enumerate(result['upper_atoms'], 1):
        penetration = abs(atom['Z'] - upper_plane)
        print(f"  N atom {i}: Z = {atom['Z']:.3f} Å, penetration = {penetration:.3f} Å")

def test_penetration_atom_grouping(analyzer_instance):
    """Test that N atoms are correctly grouped into upper and lower planes"""
    result = analyzer_instance.calculate_n_penetration()
    
    # Get the middle Z value between planes
    lower_plane, upper_plane, _ = analyzer_instance._find_planes(element_type='X')
    middle_z = (lower_plane + upper_plane) / 2
    
    # Check that lower atoms are below middle
    for atom in result['lower_atoms']:
        assert atom['Z'] < middle_z, f"Lower plane N atom at Z={atom['Z']:.3f} Å should be below middle Z={middle_z:.3f} Å"
    
    # Check that upper atoms are above middle
    for atom in result['upper_atoms']:
        assert atom['Z'] >= middle_z, f"Upper plane N atom at Z={atom['Z']:.3f} Å should be above middle Z={middle_z:.3f} Å"

def test_penetration_consistency(analyzer_instance):
    """Test that penetration calculations are consistent"""
    # Calculate penetrations twice
    result1 = analyzer_instance.calculate_n_penetration()
    result2 = analyzer_instance.calculate_n_penetration()
    
    # Check that results are consistent
    assert abs(result1['lower_penetration'] - result2['lower_penetration']) < 1e-6, \
        "Lower plane penetration should be consistent between calculations"
    assert abs(result1['upper_penetration'] - result2['upper_penetration']) < 1e-6, \
        "Upper plane penetration should be consistent between calculations"
    assert abs(result1['total_penetration'] - result2['total_penetration']) < 1e-6, \
        "Total penetration should be consistent between calculations" 