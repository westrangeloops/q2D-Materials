"""
Test interplane distance calculation for structure-level features.

This test validates the analyzer.inter_plane_distance() method which
calculates the distance between terminal atom planes using the structure graph.

Tests:
- DJ structures with spacers (should have valid interplane distance)
- Structures with terminal atoms (should calculate distance)
- Edge cases: insufficient atoms, no spacers, etc.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer


def test_interplane_distance_dj_structure():
    """Test interplane distance calculation for a DJ structure with spacers."""
    # Create a DJ structure
    creator = q2D_creator()
    structure = creator.create_structure(
        structure_type="dj",
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        spacer="BDA",  # 1,4-butanediammonium
        xy_expansion=(2, 2),
        template="cubic",
        thickness=2,
        vacuum=15.0,
    )
    
    # Analyze structure
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    
    # Calculate interplane distance
    result = analyzer.inter_plane_distance()
    
    # Verify result structure
    assert 'interplane_distance' in result
    assert 'top_centroid' in result
    assert 'bottom_centroid' in result
    assert 'top_atom_count' in result
    assert 'bottom_atom_count' in result
    assert 'molecule_mean_z' in result
    
    # Verify interplane distance is a valid number (not NaN)
    assert not np.isnan(result['interplane_distance']), "Interplane distance should not be NaN"
    assert result['interplane_distance'] > 0, "Interplane distance should be positive"
    
    # Verify we have terminal atoms
    assert result['top_atom_count'] > 0, "Should have top terminal atoms"
    assert result['bottom_atom_count'] > 0, "Should have bottom terminal atoms"
    
    # Verify centroids are valid
    assert result['top_centroid'] is not None, "Top centroid should not be None"
    assert result['bottom_centroid'] is not None, "Bottom centroid should not be None"
    assert len(result['top_centroid']) == 3, "Top centroid should have 3 coordinates"
    assert len(result['bottom_centroid']) == 3, "Bottom centroid should have 3 coordinates"
    
    # Verify molecule mean Z is calculated (since we have spacers)
    assert result['molecule_mean_z'] is not None, "Molecule mean Z should be calculated"
    assert not np.isnan(result['molecule_mean_z']), "Molecule mean Z should not be NaN"
    
    print(f"\nDJ Structure Interplane Distance Test:")
    print(f"  Interplane distance: {result['interplane_distance']:.3f} Å")
    print(f"  Top atoms: {result['top_atom_count']}")
    print(f"  Bottom atoms: {result['bottom_atom_count']}")
    print(f"  Molecule mean Z: {result['molecule_mean_z']:.3f} Å")


def test_interplane_distance_monolayer():
    """Test interplane distance calculation for a monolayer structure."""
    # Create a monolayer structure
    creator = q2D_creator()
    structure = creator.create_structure(
        structure_type="monolayer",
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        xy_expansion=(2, 2),
        template="cubic",
        thickness=1,
        vacuum=15.0,
    )
    
    # Analyze structure
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    
    # Calculate interplane distance
    result = analyzer.inter_plane_distance()
    
    # For monolayer, we should still get a distance (between top and bottom terminals)
    # But it might be smaller or the structure might not have clear separation
    assert 'interplane_distance' in result
    
    # The distance might be valid or NaN depending on structure
    if not np.isnan(result['interplane_distance']):
        assert result['interplane_distance'] >= 0, "Interplane distance should be non-negative"
    
    print(f"\nMonolayer Interplane Distance Test:")
    print(f"  Interplane distance: {result['interplane_distance']:.3f} Å" if not np.isnan(result['interplane_distance']) else "  Interplane distance: NaN")
    print(f"  Top atoms: {result['top_atom_count']}")
    print(f"  Bottom atoms: {result['bottom_atom_count']}")


def test_interplane_distance_rp_structure():
    """Test interplane distance calculation for an RP structure."""
    # Create an RP structure
    creator = q2D_creator()
    structure = creator.create_structure(
        structure_type="rp",
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        spacer="PEA",  # Phenethylammonium
        xy_expansion=(2, 2),
        template="cubic",
        thickness=2,
        vacuum=15.0,
    )
    
    # Analyze structure
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    
    # Calculate interplane distance
    result = analyzer.inter_plane_distance()
    
    # Verify result structure
    assert 'interplane_distance' in result
    
    # RP structures should have valid interplane distance
    if not np.isnan(result['interplane_distance']):
        assert result['interplane_distance'] > 0, "Interplane distance should be positive"
        assert result['top_atom_count'] > 0, "Should have top terminal atoms"
        assert result['bottom_atom_count'] > 0, "Should have bottom terminal atoms"
    
    print(f"\nRP Structure Interplane Distance Test:")
    print(f"  Interplane distance: {result['interplane_distance']:.3f} Å" if not np.isnan(result['interplane_distance']) else "  Interplane distance: NaN")
    print(f"  Top atoms: {result['top_atom_count']}")
    print(f"  Bottom atoms: {result['bottom_atom_count']}")


def test_interplane_distance_edge_cases():
    """Test edge cases for interplane distance calculation."""
    # Create a structure that might have edge cases
    creator = q2D_creator()
    structure = creator.create_structure(
        structure_type="monolayer",
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        xy_expansion=(1, 1),  # Small structure
        template="cubic",
        thickness=1,
        vacuum=15.0,
    )
    
    # Analyze structure
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    
    # Calculate interplane distance
    result = analyzer.inter_plane_distance()
    
    # Verify result structure exists
    assert 'interplane_distance' in result
    assert 'top_atom_count' in result
    assert 'bottom_atom_count' in result
    
    print(f"\nEdge Cases Test:")
    print(f"  Interplane distance: {result['interplane_distance']}")
    print(f"  Top atoms: {result['top_atom_count']}")
    print(f"  Bottom atoms: {result['bottom_atom_count']}")


if __name__ == "__main__":
    print("Running interplane distance tests...")
    test_interplane_distance_dj_structure()
    test_interplane_distance_monolayer()
    test_interplane_distance_rp_structure()
    test_interplane_distance_edge_cases()
    print("\nAll tests completed!")
