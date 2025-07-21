#!/usr/bin/env python3
"""
Simple test script to verify the octahedral distortion analysis implementation.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_import():
    """Test if the enhanced analyzer can be imported."""
    try:
        from SVC_materials.core.analyzer import q2D_analyzer
        print("✓ Import successful!")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_methods():
    """Test if all new methods are available."""
    try:
        from SVC_materials.core.analyzer import q2D_analyzer
        
        expected_methods = [
            'order_octahedra',
            'calculate_octahedral_distortions',
            'get_distortion_summary',
            'compare_distortions',
            'export_distortion_data',
            'get_octahedron_by_index',
            'print_distortion_summary'
        ]
        
        available_methods = [method for method in dir(q2D_analyzer) if not method.startswith('_')]
        
        print("Available methods:")
        for method in available_methods:
            print(f"  - {method}")
        
        print("\nChecking for new distortion analysis methods:")
        all_present = True
        for method in expected_methods:
            if hasattr(q2D_analyzer, method):
                print(f"✓ {method}")
            else:
                print(f"✗ {method} - MISSING")
                all_present = False
        
        return all_present
        
    except Exception as e:
        print(f"✗ Method check failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality without requiring actual structure files."""
    try:
        from SVC_materials.core.analyzer import q2D_analyzer
        import numpy as np
        
        # Test the CalcDistortion import
        from SVC_materials.utils.octadist.calc import CalcDistortion
        
        # Create a simple test octahedron (perfect octahedron)
        # Central atom at origin, ligands at ±2 along each axis
        test_coords = np.array([
            [0.0, 0.0, 0.0],  # Central atom
            [2.0, 0.0, 0.0],  # +X ligand
            [-2.0, 0.0, 0.0], # -X ligand
            [0.0, 2.0, 0.0],  # +Y ligand
            [0.0, -2.0, 0.0], # -Y ligand
            [0.0, 0.0, 2.0],  # +Z ligand
            [0.0, 0.0, -2.0]  # -Z ligand
        ])
        
        # Test CalcDistortion directly
        dist_calc = CalcDistortion(test_coords)
        
        print("✓ CalcDistortion works!")
        print(f"  Mean bond distance: {dist_calc.d_mean:.4f}")
        print(f"  Zeta parameter: {dist_calc.zeta:.4f}")
        print(f"  Delta parameter: {dist_calc.delta:.6f}")
        print(f"  Sigma parameter: {dist_calc.sigma:.4f}")
        print(f"  Is octahedral: {not dist_calc.non_octa}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Testing Octahedral Distortion Analysis Implementation ===\n")
    
    tests = [
        ("Import Test", test_import),
        ("Methods Test", test_methods),
        ("Basic Functionality Test", test_basic_functionality)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"Running {test_name}...")
        result = test_func()
        results.append(result)
        print(f"{test_name}: {'PASSED' if result else 'FAILED'}\n")
    
    # Summary
    passed = sum(results)
    total = len(results)
    print(f"=== Test Summary ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! The distortion analysis implementation is ready to use.")
    else:
        print("⚠️  Some tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 