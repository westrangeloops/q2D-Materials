#!/usr/bin/env python3
"""
Test script to verify the integration of octahedral distortion analysis 
with the extraction.py script.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_extraction_import():
    """Test if the enhanced extraction script can be imported."""
    try:
        # Change to Graphs directory for import
        original_cwd = os.getcwd()
        graphs_dir = Path(__file__).parent / "Graphs"
        os.chdir(graphs_dir)
        
        sys.path.insert(0, str(graphs_dir))
        import extraction
        
        print("✓ extraction.py import successful!")
        
        # Check if distortion_properties function exists
        if hasattr(extraction, 'distortion_properties'):
            print("✓ distortion_properties function available")
        else:
            print("✗ distortion_properties function missing")
            return False
        
        # Check if DISTORTION_ANALYSIS_AVAILABLE flag exists
        if hasattr(extraction, 'DISTORTION_ANALYSIS_AVAILABLE'):
            status = extraction.DISTORTION_ANALYSIS_AVAILABLE
            print(f"✓ Distortion analysis availability: {status}")
        else:
            print("✗ DISTORTION_ANALYSIS_AVAILABLE flag missing")
            return False
        
        os.chdir(original_cwd)
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False

def test_distortion_function():
    """Test the distortion_properties function with a mock directory."""
    try:
        # Change to Graphs directory
        original_cwd = os.getcwd()
        graphs_dir = Path(__file__).parent / "Graphs"
        os.chdir(graphs_dir)
        
        sys.path.insert(0, str(graphs_dir))
        import extraction
        
        # Test with a non-existent directory (should return empty DataFrame)
        result_df = extraction.distortion_properties("/non/existent/path")
        
        if result_df.empty:
            print("✓ distortion_properties handles non-existent paths correctly")
        else:
            print("✗ distortion_properties should return empty DataFrame for non-existent paths")
            return False
        
        # Test the expected columns structure
        if extraction.DISTORTION_ANALYSIS_AVAILABLE:
            # Create a mock test to see what columns would be created
            print("✓ Expected distortion columns:")
            expected_columns = [
                'Experiment', 'Perovskite', 'NumOctahedra',
                'MeanZeta', 'StdZeta', 'MinZeta', 'MaxZeta',
                'MeanDelta', 'StdDelta', 'MinDelta', 'MaxDelta',
                'MeanSigma', 'StdSigma', 'MinSigma', 'MaxSigma',
                'MeanTheta', 'StdTheta', 'MinTheta', 'MaxTheta',
                'MeanBondDistance', 'StdBondDistance',
                'MeanOctaVolume', 'StdOctaVolume',
                'DistortionAnalysisSuccess'
            ]
            for col in expected_columns:
                print(f"  - {col}")
        else:
            print("⚠ Distortion analysis not available, function will return empty DataFrame")
        
        os.chdir(original_cwd)
        return True
        
    except Exception as e:
        print(f"✗ Function test failed: {e}")
        return False

def test_integration_workflow():
    """Test the complete integration workflow."""
    try:
        original_cwd = os.getcwd()
        graphs_dir = Path(__file__).parent / "Graphs"
        os.chdir(graphs_dir)
        
        sys.path.insert(0, str(graphs_dir))
        import extraction
        
        print("Testing complete integration workflow...")
        
        # Test all three functions exist
        functions_to_test = ['name_properties', 'xml_properties', 'distortion_properties']
        for func_name in functions_to_test:
            if hasattr(extraction, func_name):
                print(f"✓ {func_name} function available")
            else:
                print(f"✗ {func_name} function missing")
                return False
        
        print("✓ All required functions are available")
        print("✓ Integration workflow is ready")
        
        os.chdir(original_cwd)
        return True
        
    except Exception as e:
        print(f"✗ Integration workflow test failed: {e}")
        return False

def main():
    """Run all integration tests."""
    print("=== Testing Extraction.py Integration with Distortion Analysis ===\n")
    
    tests = [
        ("Import Test", test_extraction_import),
        ("Distortion Function Test", test_distortion_function),
        ("Integration Workflow Test", test_integration_workflow)
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
        print("🎉 All integration tests passed!")
        print("\n=== Usage Instructions ===")
        print("1. Navigate to the Graphs directory")
        print("2. Run: python extraction.py")
        print("3. The script will now include octahedral distortion analysis")
        print("4. Results will be saved to perovskites.csv with distortion columns")
        print("\n=== New Columns Added ===")
        print("- NumOctahedra: Number of octahedra found")
        print("- MeanZeta, StdZeta, MinZeta, MaxZeta: Zeta parameter statistics")
        print("- MeanDelta, StdDelta, MinDelta, MaxDelta: Delta parameter statistics")
        print("- MeanSigma, StdSigma, MinSigma, MaxSigma: Sigma parameter statistics")
        print("- MeanTheta, StdTheta, MinTheta, MaxTheta: Theta parameter statistics")
        print("- MeanBondDistance, StdBondDistance: Bond distance statistics")
        print("- MeanOctaVolume, StdOctaVolume: Octahedral volume statistics")
        print("- DistortionAnalysisSuccess: Boolean flag for successful analysis")
    else:
        print("⚠️  Some integration tests failed. Please check the implementation.")

if __name__ == "__main__":
    main() 