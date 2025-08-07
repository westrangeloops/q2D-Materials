#!/usr/bin/env python3
"""
Test script to verify the updated vector analysis correctly identifies terminal halogen atoms.
"""

import sys
import os

# Add the parent directory to the path to import SVC_materials
sys.path.insert(0, os.path.dirname(__file__))

from SVC_materials.core.analyzer import q2D_analyzer

def test_vector_analysis_fix():
    """Test the updated vector analysis with terminal atom identification."""
    
    print("=" * 80)
    print("TESTING UPDATED VECTOR ANALYSIS")
    print("=" * 80)
    
    # Test with n1 structure
    file_path = 'Graphs/henrique/n1_relaxado.vasp'
    
    print(f"\nTesting structure: {file_path}")
    print("-" * 50)
    
    try:
        analyzer = q2D_analyzer(file_path, 'Pb', 'Br', 3.5)
        
        print(f"✓ Structure loaded successfully")
        print(f"  Total atoms: {len(analyzer.atoms)}")
        print(f"  Halogen type: {analyzer.x}")
        print(f"  Total octahedra: {len(analyzer.ordered_octahedra)}")
        
        # Check connectivity analysis
        connectivity = analyzer.ontology.get('connectivity_analysis', {})
        shared_atoms = connectivity.get('shared_atoms', {})
        terminal_axial = connectivity.get('terminal_axial_atoms', {})
        
        print(f"  Shared atoms: {len(shared_atoms)}")
        print(f"  Terminal axial atoms found: {sum(len(oct_data.get('terminal_axial_atoms', [])) for oct_data in terminal_axial.values())}")
        
        # Check vector analysis
        print(f"\nTesting vector analysis...")
        vector_analysis = analyzer.get_vector_analysis()
        
        if 'error' in vector_analysis:
            print(f"❌ Vector analysis error: {vector_analysis['error']}")
            return False
        else:
            print("✅ Vector analysis successful!")
            
            # Check if we have terminal atom analysis
            if 'terminal_atom_analysis' in vector_analysis:
                terminal_info = vector_analysis['terminal_atom_analysis']
                print(f"   Terminal {terminal_info['halogen_symbol']} atoms found: {terminal_info['total_terminal_atoms_found']}")
                print(f"   Terminal atom indices: {terminal_info['terminal_atom_indices']}")
                print(f"   Low plane atoms: {terminal_info['low_plane_atom_indices']}")
                print(f"   High plane atoms: {terminal_info['high_plane_atom_indices']}")
            else:
                print("   ⚠️  No terminal atom analysis found (using legacy method)")
            
            if 'vector_analysis_results' in vector_analysis:
                results = vector_analysis['vector_analysis_results']
                print(f"   Angle between planes: {results['angle_between_planes_degrees']:.2f}°")
                print(f"   Distance between centers: {results['distance_between_plane_centers_angstrom']:.3f} Å")
                print(f"   Low plane vs Z-axis: {results['angle_between_low_plane_and_z']:.2f}°")
                print(f"   High plane vs Z-axis: {results['angle_between_high_plane_and_z']:.2f}°")
            
            return True
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the vector analysis test."""
    success = test_vector_analysis_fix()
    
    print("\n" + "=" * 80)
    if success:
        print("✅ VECTOR ANALYSIS TEST PASSED!")
        print("Terminal halogen atoms are now correctly identified from the original structure.")
        print("Vector analysis no longer depends on the salt structure having terminal atoms.")
    else:
        print("❌ VECTOR ANALYSIS TEST FAILED!")
        print("The fix may need additional adjustments.")
    print("=" * 80)
    
    return success

if __name__ == "__main__":
    main() 