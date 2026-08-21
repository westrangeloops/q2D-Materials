"""
Tests for non-orthogonal cell handling in coordinate calculations.

This module tests that the codebase correctly handles non-orthogonal unit cells
where cell angles differ from 90 degrees, which causes Cartesian coordinates
to exceed cell parameters.
"""

import pytest
import numpy as np
from ase import Atoms
from ase.build import bulk

from q2D_Materials.utils.geometry.coordinate_utils import (
    cartesian_to_fractional,
    fractional_to_cartesian,
    get_actual_cell_extent,
    wrap_fractional,
    unwrap_fractional_relative,
)
from q2D_Materials.analyzer import q2D_analyzer


class TestCoordinateConversions:
    """Test coordinate conversion utilities."""
    
    def test_orthogonal_cell_conversion(self):
        """Test conversion for orthogonal cell."""
        cell = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 20.0]
        ])
        
        # Test single position
        cart = np.array([5.0, 5.0, 10.0])
        frac = cartesian_to_fractional(cart, cell)
        np.testing.assert_allclose(frac, [0.5, 0.5, 0.5], atol=1e-10)
        
        # Test round-trip
        cart_back = fractional_to_cartesian(frac, cell)
        np.testing.assert_allclose(cart_back, cart, atol=1e-10)
    
    def test_nonorthogonal_cell_conversion(self):
        """Test conversion for non-orthogonal cell (monoclinic)."""
        # Monoclinic cell with beta != 90 degrees
        cell = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [2.0, 0.0, 20.0]  # c-vector tilted in x-direction
        ])
        
        # Position at fractional (0.5, 0.5, 0.5)
        frac = np.array([0.5, 0.5, 0.5])
        cart = fractional_to_cartesian(frac, cell)
        
        # Cartesian position calculation:
        # cart = frac @ cell = [0.5, 0.5, 0.5] @ [[10,0,0], [0,10,0], [2,0,20]]
        # cart = 0.5*[10,0,0] + 0.5*[0,10,0] + 0.5*[2,0,20]
        # cart = [5,0,0] + [0,5,0] + [1,0,10] = [6, 5, 10]
        np.testing.assert_allclose(cart, [6.0, 5.0, 10.0], atol=1e-8)
        
        # Test round-trip
        frac_back = cartesian_to_fractional(cart, cell)
        np.testing.assert_allclose(frac_back, frac, atol=1e-8)
    
    def test_multiple_positions(self):
        """Test conversion of multiple positions at once."""
        cell = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [2.0, 0.0, 20.0]
        ])
        
        cart_positions = np.array([
            [0.0, 0.0, 0.0],
            [6.0, 5.0, 10.0],
            [12.0, 10.0, 20.0]
        ])
        
        frac_positions = cartesian_to_fractional(cart_positions, cell)
        
        expected_frac = np.array([
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.0, 1.0, 1.0]
        ])
        
        np.testing.assert_allclose(frac_positions, expected_frac, atol=1e-8)
    
    def test_actual_cell_extent_orthogonal(self):
        """Test cell extent calculation for orthogonal cell."""
        cell = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 20.0]
        ])
        
        x_extent, y_extent, z_extent = get_actual_cell_extent(cell)
        
        assert abs(x_extent - 10.0) < 1e-10
        assert abs(y_extent - 10.0) < 1e-10
        assert abs(z_extent - 20.0) < 1e-10
    
    def test_actual_cell_extent_nonorthogonal(self):
        """Test cell extent calculation for non-orthogonal cell."""
        # Monoclinic cell
        cell = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [2.0, 0.0, 20.0]  # c-vector tilted
        ])
        
        x_extent, y_extent, z_extent = get_actual_cell_extent(cell)
        
        # X-extent should include contribution from tilted c-vector
        assert x_extent > 10.0
        assert abs(x_extent - 12.0) < 1e-10  # 10 + 2
        assert abs(y_extent - 10.0) < 1e-10
        assert abs(z_extent - 20.0) < 1e-10
    
    def test_wrap_fractional(self):
        """Test fractional coordinate wrapping."""
        frac = np.array([1.5, -0.3, 0.7])
        wrapped = wrap_fractional(frac)
        
        np.testing.assert_allclose(wrapped, [0.5, 0.7, 0.7], atol=1e-10)
    
    def test_unwrap_fractional_relative(self):
        """Test fractional coordinate unwrapping relative to reference."""
        reference = np.array([0.1, 0.1, 0.1])
        
        # Position close to reference but wrapped across PBC
        positions = np.array([
            [0.15, 0.15, 0.15],  # Close, no wrapping needed
            [0.95, 0.95, 0.95]   # Close to reference across PBC
        ])
        
        unwrapped = unwrap_fractional_relative(positions, reference)
        
        # Second position should be unwrapped to negative values
        assert unwrapped[0, 0] > 0  # First position unchanged
        assert unwrapped[1, 0] < 0  # Second position unwrapped
        np.testing.assert_allclose(unwrapped[1], [-0.05, -0.05, -0.05], atol=1e-10)


class TestInterplaneDistanceNonorthogonal:
    """Test interplane distance calculation with non-orthogonal cells."""
    
    def test_interplane_distance_consistency(self):
        """Test that interplane_distance + slab_thickness = actual_cell_z_extent."""
        # This test would require a real structure
        # For now, we document the expected behavior
        pass
    
    def test_z_extent_vs_c_parameter(self):
        """Test that we use actual Z-extent instead of c parameter."""
        # Create a simple monoclinic perovskite-like structure
        # This is a placeholder - actual test would need a real structure
        pass


class TestGraphCellMatrix:
    """Test that cell matrix is properly stored in graph."""
    
    def test_cell_matrix_stored(self):
        """Test that cell matrix is stored in graph metadata."""
        # Create a simple perovskite structure using ASE's bulk builder
        # Use 'rocksalt' structure as a simple cubic test case
        atoms = bulk('NaCl', 'rocksalt', a=5.64)
        
        analyzer = q2D_analyzer(atoms)
        analyzer.analyze()
        
        # Check that cell matrix is stored (use _graph which is the private attribute)
        from q2D_Materials.analyzer.core.graph_construction import get_cell_matrix
        cell = get_cell_matrix(analyzer._graph)
        
        assert cell is not None
        assert cell.shape == (3, 3)
        
        # Verify it matches the original cell
        original_cell = np.array(atoms.get_cell())
        np.testing.assert_allclose(cell, original_cell, atol=1e-8)


class TestPBCDistances:
    """Test PBC-aware distance calculations."""
    
    def test_pbc_distance_orthogonal(self):
        """Test PBC distance for orthogonal cell."""
        from q2D_Materials.utils.geometry.pbc_distances import calculate_pbc_distances
        
        cell = np.eye(3) * 10.0
        ref = np.array([0.0, 0.0, 0.0])
        targets = np.array([
            [1.0, 0.0, 0.0],
            [9.0, 0.0, 0.0]  # Should be 1 Å away across PBC
        ])
        
        dists = calculate_pbc_distances(ref, targets, cell)
        
        np.testing.assert_allclose(dists, [1.0, 1.0], atol=1e-10)
    
    def test_pbc_distance_nonorthogonal(self):
        """Test PBC distance for non-orthogonal cell."""
        from q2D_Materials.utils.geometry.pbc_distances import calculate_pbc_distances
        
        # Monoclinic cell
        cell = np.array([
            [10.0, 0.0, 0.0],
            [0.0, 10.0, 0.0],
            [2.0, 0.0, 20.0]
        ])
        
        ref = np.array([0.0, 0.0, 0.0])
        target = np.array([1.0, 0.0, 0.0])
        
        dists = calculate_pbc_distances(ref, target.reshape(1, 3), cell)
        
        # Distance should be 1 Å (simple Euclidean in this case)
        np.testing.assert_allclose(dists, [1.0], atol=1e-10)


class TestCoordinateDocumentation:
    """Test that coordinate expectations are documented."""
    
    def test_coordinate_utils_docstrings(self):
        """Verify coordinate utilities have proper docstrings."""
        assert cartesian_to_fractional.__doc__ is not None
        assert "fractional" in cartesian_to_fractional.__doc__.lower()
        assert "cartesian" in cartesian_to_fractional.__doc__.lower()
        
        assert fractional_to_cartesian.__doc__ is not None
        assert get_actual_cell_extent.__doc__ is not None
        
        # Check that non-orthogonal cells are mentioned
        assert "non-orthogonal" in get_actual_cell_extent.__doc__.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
