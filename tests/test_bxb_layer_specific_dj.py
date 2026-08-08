"""
Test layer-specific B-X-B angle calculations for DJ structures without tilting.

Validates that perfect octahedra (no tilting, glazer_pattern='a0a0a0') have:
- Intra-layer B-X-B angles ≈ 180° (equatorial X atoms, in-plane)
- Inter-layer B-X-B angles ≈ 180° (axial/interlayer X atoms, out-of-plane)

Tests:
- DJ structure n=2 (bilayer): L0, L1 intra-layer, L0↔L1 inter-layer
- DJ structure n=3 (trilayer): L0, L1, L2 intra-layer, L0↔L1, L1↔L2 inter-layer
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer


# Tolerance for perfect octahedra validation
ANGLE_TOLERANCE = 1.0  # degrees - allow small numerical errors
BXB_ANGLE_TARGET = 180.0  # Default B-X-B angle (can be overridden per structure)


def print_bxb_results(analyzer, structure_name):
    """
    Print detailed B-X-B angle results for a structure by iterating over layers.
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzed structure
    structure_name : str
        Name of structure
    """
    # Get structure metadata
    metadata = analyzer.get_structure_metadata()
    thickness = metadata.get('thickness', len(analyzer.layers))
    
    print("\n" + "="*80)
    print(f"{structure_name} (n={thickness})")
    print("="*80)
    
    # Get global B-X-B angles
    print("\n--- Global B-X-B Angles ---")
    bxb_data = analyzer.get_bxb_angles()
    
    if bxb_data.get('bxb_angles') is not None:
        bxb_all = bxb_data['bxb_angles']
        print(f"  All B-X-B angles: {len(bxb_all)} angles")
        print(f"    Mean: {np.mean(bxb_all):.2f}° (target: {BXB_ANGLE_TARGET}°)")
        print(f"    Std:  {np.std(bxb_all):.2f}°")
        print(f"    Min:  {np.min(bxb_all):.2f}°")
        print(f"    Max:  {np.max(bxb_all):.2f}°")
        deviation = abs(np.mean(bxb_all) - BXB_ANGLE_TARGET)
        status = "✓" if deviation < ANGLE_TOLERANCE else "✗"
        print(f"    {status} Deviation from 180°: {deviation:.2f}°")
    
    if bxb_data.get('bxb_equatorial_angles') is not None:
        bxb_eq = bxb_data['bxb_equatorial_angles']
        print(f"\n  Equatorial B-X-B angles: {len(bxb_eq)} angles")
        print(f"    Mean: {np.mean(bxb_eq):.2f}°")
        print(f"    Std:  {np.std(bxb_eq):.2f}°")
    
    if bxb_data.get('bxb_interlayer_angles') is not None:
        bxb_il = bxb_data['bxb_interlayer_angles']
        print(f"\n  Interlayer B-X-B angles: {len(bxb_il)} angles")
        print(f"    Mean: {np.mean(bxb_il):.2f}°")
        print(f"    Std:  {np.std(bxb_il):.2f}°")
    
    # Get layer-specific B-X-B angles by iterating over layers
    print("\n--- Intra-Layer B-X-B Angles (B-Xeq-B, equatorial X only) ---")
    for layer_id in analyzer.layers:
        try:
            layer_bxb = analyzer.layers.get_bxb(layer_id=layer_id)
            if layer_bxb['count'] > 0 and layer_bxb['bxb_angles'] is not None:
                angles = layer_bxb['bxb_angles']
                print(f"\n  Layer {layer_id} (L{layer_id}):")
                print(f"    Count: {layer_bxb['count']} angles")
                print(f"    Mean: {layer_bxb['bxb_mean']:.2f}° (target: {BXB_ANGLE_TARGET}°)")
                print(f"    Std:  {layer_bxb['bxb_std']:.2f}°")
                print(f"    Min:  {np.min(angles):.2f}°")
                print(f"    Max:  {np.max(angles):.2f}°")
                deviation = abs(layer_bxb['bxb_mean'] - BXB_ANGLE_TARGET)
                status = "✓" if deviation < ANGLE_TOLERANCE else "✗"
                print(f"    {status} Deviation from 180°: {deviation:.2f}°")
            else:
                print(f"\n  Layer {layer_id} (L{layer_id}): No angles found")
        except (ValueError, KeyError) as e:
            print(f"\n  Layer {layer_id} (L{layer_id}): Error - {e}")
    
    # Get inter-layer B-X-B angles
    if thickness > 1:
        print("\n--- Inter-Layer B-X-B Angles (B-Xaxial-B, axial X only) ---")
        all_interlayer = analyzer.layers.get_all_interlayer_bxb()
        
        if all_interlayer['count'] > 0:
            print(f"\n  All Inter-Layer Combined:")
            print(f"    Total count: {all_interlayer['count']} angles")
            if all_interlayer['bxb_mean'] is not None:
                print(f"    Mean: {all_interlayer['bxb_mean']:.2f}° (target: {BXB_ANGLE_TARGET}°)")
                print(f"    Std:  {all_interlayer['bxb_std']:.2f}°")
                deviation = abs(all_interlayer['bxb_mean'] - BXB_ANGLE_TARGET)
                status = "✓" if deviation < ANGLE_TOLERANCE else "✗"
                print(f"    {status} Deviation from 180°: {deviation:.2f}°")
            
            print(f"\n  Per-Pair Results:")
            for (layer1, layer2), data in all_interlayer['pair_data'].items():
                if data['count'] > 0 and data['bxb_angles'] is not None:
                    angles = data['bxb_angles']
                    print(f"\n    L{layer1} ↔ L{layer2}:")
                    print(f"      Count: {data['count']} angles")
                    print(f"      Mean: {data['bxb_mean']:.2f}° (target: {BXB_ANGLE_TARGET}°)")
                    print(f"      Std:  {data['bxb_std']:.2f}°")
                    print(f"      Min:  {np.min(angles):.2f}°")
                    print(f"      Max:  {np.max(angles):.2f}°")
                    deviation = abs(data['bxb_mean'] - BXB_ANGLE_TARGET)
                    status = "✓" if deviation < ANGLE_TOLERANCE else "✗"
                    print(f"      {status} Deviation from 180°: {deviation:.2f}°")
        else:
            print("\n  No inter-layer angles found")
    
    print("\n" + "="*80)


def test_structure_bxb_layers(structure_name, structure_params):
    """
    Test B-X-B angles for a structure with given parameters.
    
    Parameters
    ----------
    structure_name : str
        Name of the structure for display
    structure_params : dict
        Dictionary of parameters to pass to create_structure()
        Can include 'Expected_BXB_Angle' to override default 180.0°
    """
    print("\n" + "="*80)
    print(f"TEST: {structure_name} - Layer-Specific B-X-B Analysis")
    print("="*80)
    
    # Extract expected B-X-B angle (if provided, otherwise use default)
    expected_bxb_angle = structure_params.pop('Expected_BXB_Angle', BXB_ANGLE_TARGET)
    
    # Create structure with no tilting (default)
    default_params = {
        'glazer_pattern': "a0a0a0",  # No tilting
        'glazer_angles': [0, 0, 0],
    }
    default_params.update(structure_params)
    
    creator = q2D_creator()
    structure = creator.create_structure(**default_params)
    
    # Analyze structure
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    
    # Print detailed results
    print_bxb_results(analyzer, structure_name)
    
    # Validate results
    print("\n--- Validation ---")
    print(f"  Expected B-X-B angle: {expected_bxb_angle}° (tolerance: ±{ANGLE_TOLERANCE}°)")
    
    # Check intra-layer angles by iterating over layers
    for layer_id in analyzer.layers:
        try:
            layer_bxb = analyzer.layers.get_bxb(layer_id=layer_id)
            if layer_bxb['count'] > 0:
                deviation = abs(layer_bxb['bxb_mean'] - expected_bxb_angle)
                assert deviation < ANGLE_TOLERANCE, (
                    f"Layer {layer_id} intra-layer B-X-B mean {layer_bxb['bxb_mean']:.2f}° "
                    f"should be close to {expected_bxb_angle}° (deviation: {deviation:.2f}°)"
                )
                print(f"  ✓ Layer {layer_id} intra-layer B-X-B: {layer_bxb['bxb_mean']:.2f}° "
                      f"(expected: {expected_bxb_angle}°, deviation: {deviation:.2f}°)")
        except (ValueError, KeyError):
            pass
    
    # Check inter-layer angles
    all_interlayer = analyzer.layers.get_all_interlayer_bxb()
    if all_interlayer['count'] > 0:
        deviation = abs(all_interlayer['bxb_mean'] - expected_bxb_angle)
        assert deviation < ANGLE_TOLERANCE, (
            f"Inter-layer B-X-B mean {all_interlayer['bxb_mean']:.2f}° "
            f"should be close to {expected_bxb_angle}° (deviation: {deviation:.2f}°)"
        )
        print(f"  ✓ Inter-layer B-X-B: {all_interlayer['bxb_mean']:.2f}° "
              f"(expected: {expected_bxb_angle}°, deviation: {deviation:.2f}°)")
        
        # Check each pair
        for (layer1, layer2), data in all_interlayer['pair_data'].items():
            if data['count'] > 0:
                deviation = abs(data['bxb_mean'] - expected_bxb_angle)
                assert deviation < ANGLE_TOLERANCE, (
                    f"L{layer1}↔L{layer2} inter-layer B-X-B mean {data['bxb_mean']:.2f}° "
                    f"should be close to {expected_bxb_angle}° (deviation: {deviation:.2f}°)"
                )
                print(f"  ✓ L{layer1}↔L{layer2} inter-layer B-X-B: {data['bxb_mean']:.2f}° "
                      f"(expected: {expected_bxb_angle}°, deviation: {deviation:.2f}°)")
    
    print("\n  ✓ All validations passed!")


# Define test structures - easily extendable list
TEST_STRUCTURES = [
    {
        'name': 'DJ n=2 (Bilayer)',
        'params': {
            'A_ions': "MA",
            'B_ions': "Pb",
            'X_ions': "I",
            'structure_type': "bulk",
            'layer_sequence': "DJ",
            'template': "cubic",
            'xy_expansion': (2, 2),
            'thickness': 2,
            'spacer': "[NH3+]CCCC[NH3+]",
        }
    },
    {
        'name': 'DJ n=3 (Trilayer)',
        'params': {
            'A_ions': "MA",
            'B_ions': "Pb",
            'X_ions': "I",
            'structure_type': "bulk",
            'layer_sequence': "DJ",
            'template': "cubic",
            'xy_expansion': (2, 2),
            'thickness': 3,
            'spacer': "[NH3+]CCCC[NH3+]",
        }
    },
    # Add more structures here as needed
    {
        'name': 'DJ n=5 (Pentalayer)',
        'params': {
            'A_ions': "MA",
            'B_ions': "Pb",
            'X_ions': "I",
            'structure_type': "bulk",
            'layer_sequence': "DJ",
            'template': "cubic",
            'xy_expansion': (1, 1),
            'thickness': 5,
            'spacer': "[NH3+]CCCC[NH3+]",
            'glazer_pattern': "a+a0a0",
            'glazer_angles': [10, 0, 0],
            'Expected_BXB_Angle': 160.0,
        }
    },
        {
        'name': 'RP n=3 (Trilayer)',
        'params': {
            'A_ions': "MA",
            'B_ions': "Pb",
            'X_ions': "I",
            'structure_type': "bulk",
            'layer_sequence': "DJ",
            'template': "cubic",
            'xy_expansion': (1, 1),
            'thickness': 5,
            'spacer': "[NH3+]CCCC[NH3+]",
            'glazer_pattern': "a+a+a0",
            'glazer_angles': [10, 10, 0],
            'Expected_BXB_Angle': 152.0,
        }
    },
]


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Layer-Specific B-X-B Angle Test for DJ Structures")
    print("="*80)
    print("\nTesting perfect octahedra (no tilting) to validate:")
    print("  - Intra-layer B-X-B angles (equatorial X) ≈ 180°")
    print("  - Inter-layer B-X-B angles (axial X) ≈ 180°")
    print("="*80)
    
    # Run tests for all structures in the list
    for test_struct in TEST_STRUCTURES:
        test_structure_bxb_layers(test_struct['name'], test_struct['params'])
    
    print("\n" + "="*80)
    print(f"All tests completed! ({len(TEST_STRUCTURES)} structures tested)")
    print("="*80 + "\n")
