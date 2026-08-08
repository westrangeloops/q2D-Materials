"""
Test B-X-B bond angles and bond lengths for perfect octahedra.

Validates that perfect octahedra (no tilting, glazer_pattern='a0a0a0') have:
- B-X-B angles ≈ 180° (perfect linear)
- X-B-X angles ≈ 90° or 180° (perfect octahedra)
- B-X bond lengths are consistent (delta ≈ 0)

Tests:
- 3 bulk structures of different sizes
- 3 RP (Ruddlesden-Popper) structures of different sizes
- 4 DJ (Dion-Jacobson) structures with different sizes and spacers
- 2 DJ structures with n>3 (thickness > 3) with layer grouping validation
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.characterization.distortions import _compute_octahedral_distortions


# Tolerance for perfect octahedra validation
ANGLE_TOLERANCE = 2.0  # degrees - allow small numerical errors
BXB_ANGLE_TARGET = 180.0  # Perfect B-X-B angle
XBX_ANGLE_TARGETS = [90.0, 180.0]  # Perfect X-B-X angles
DELTA_TOLERANCE = 0.01  # Delta should be very small for perfect octahedra
BOND_LENGTH_TOLERANCE = 0.05  # Å - bond lengths should be very consistent


class TestPerfectOctahedraBXB:
    """Test B-X-B angles and bond lengths for perfect octahedra."""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up test fixtures."""
        self.creator = q2D_creator()
    
    def _validate_perfect_octahedra(self, analyzer, structure_name):
        """
        Validate that structure has perfect octahedra properties.
        
        Parameters
        ----------
        analyzer : q2D_analyzer
            Analyzed structure
        structure_name : str
            Name of structure for error messages
        """
        # Check that we have octahedra
        octahedra = analyzer.get_octahedra()
        assert len(octahedra) > 0, f"{structure_name}: No octahedra found"
        
        # Get B-X-B angles
        bxb_data = analyzer.get_bxb_angles()
        bxb_angles = bxb_data['bxb_angles']
        bxb_mean = bxb_data['bxb_mean']
        bxb_std = bxb_data['bxb_std']
        
        assert bxb_angles is not None, f"{structure_name}: B-X-B angles should not be None"
        assert len(bxb_angles) > 0, f"{structure_name}: Should have B-X-B angles"
        
        # Validate B-X-B angles are close to 180°
        mean_deviation = abs(bxb_mean - BXB_ANGLE_TARGET)
        assert mean_deviation < ANGLE_TOLERANCE, (
            f"{structure_name}: Mean B-X-B angle {bxb_mean:.2f}° should be close to "
            f"{BXB_ANGLE_TARGET}° (deviation: {mean_deviation:.2f}°)"
        )
        
        # Get X-B-X angles and bond lengths
        distortions = _compute_octahedral_distortions(analyzer)
        xbx_angles = distortions['bond_angles']
        bond_lengths = distortions['bond_lengths']
        mean_bond_length = distortions['mean_bond_length']
        delta = distortions['delta']
        
        assert len(xbx_angles) > 0, f"{structure_name}: Should have X-B-X angles"
        assert len(bond_lengths) > 0, f"{structure_name}: Should have bond lengths"
        assert mean_bond_length > 0, f"{structure_name}: Mean bond length should be positive"
        
        # Validate X-B-X angles are close to 90° or 180°
        # Check that most angles are close to ideal values
        angles_90 = np.abs(xbx_angles - 90.0) < ANGLE_TOLERANCE
        angles_180 = np.abs(xbx_angles - 180.0) < ANGLE_TOLERANCE
        valid_angles = angles_90 | angles_180
        
        valid_ratio = np.sum(valid_angles) / len(xbx_angles)
        assert valid_ratio > 0.8, (
            f"{structure_name}: At least 80% of X-B-X angles should be close to 90° or 180°. "
            f"Found {valid_ratio*100:.1f}% valid angles"
        )
        
        # Validate bond length consistency (delta should be very small)
        assert delta is not None, f"{structure_name}: Delta should not be None"
        assert delta < DELTA_TOLERANCE, (
            f"{structure_name}: Delta (bond length distortion) {delta:.6f} should be < {DELTA_TOLERANCE} "
            f"for perfect octahedra"
        )
        
        # Validate bond lengths are consistent (std should be small)
        bond_std = np.std(bond_lengths)
        assert bond_std < BOND_LENGTH_TOLERANCE, (
            f"{structure_name}: Bond length std {bond_std:.4f} Å should be < {BOND_LENGTH_TOLERANCE} Å "
            f"for perfect octahedra"
        )
        
        return {
            'bxb_mean': bxb_mean,
            'bxb_std': bxb_std,
            'xbx_mean': distortions['mean_angle'],
            'bond_length_mean': mean_bond_length,
            'bond_length_std': bond_std,
            'delta': delta,
            'sigma': distortions['sigma'],
            'lambda': distortions['lambda'],
        }
    
    def _validate_perfect_octahedra_by_layer(self, analyzer, structure_name):
        """
        Validate that structure has perfect octahedra properties grouped by layer.
        
        Parameters
        ----------
        analyzer : q2D_analyzer
            Analyzed structure
        structure_name : str
            Name of structure for error messages
            
        Returns
        -------
        dict
            Dictionary with validation results per layer
        """
        # Check that we have octahedra
        octahedra = analyzer.get_octahedra()
        assert len(octahedra) > 0, f"{structure_name}: No octahedra found"
        
        # Get layers
        layers = analyzer.get_layers()
        layer_ids = [lid for lid in layers.keys() if lid != 'unknown']
        assert len(layer_ids) > 0, f"{structure_name}: No layers found"
        
        # Get B-X-B angles grouped by layer
        bxb_data_by_layer = analyzer.get_bxb_angles(group_by='layer')
        
        assert 'global' in bxb_data_by_layer, f"{structure_name}: Should have 'global' key in grouped B-X-B data"
        assert len(bxb_data_by_layer) > 1, f"{structure_name}: Should have at least one layer-specific result"
        
        # Get distortions grouped by layer
        distortions_by_layer = _compute_octahedral_distortions(analyzer, group_by='layer')
        
        assert 'global' in distortions_by_layer, f"{structure_name}: Should have 'global' key in grouped distortions"
        
        results = {
            'global': {},
            'layers': {}
        }
        
        # Validate global results
        global_bxb = bxb_data_by_layer['global']
        global_distortions = distortions_by_layer['global']
        
        assert global_bxb['bxb_mean'] is not None, f"{structure_name}: Global B-X-B mean should not be None"
        mean_deviation = abs(global_bxb['bxb_mean'] - BXB_ANGLE_TARGET)
        assert mean_deviation < ANGLE_TOLERANCE, (
            f"{structure_name}: Global mean B-X-B angle {global_bxb['bxb_mean']:.2f}° should be close to "
            f"{BXB_ANGLE_TARGET}° (deviation: {mean_deviation:.2f}°)"
        )
        
        assert global_distortions['delta'] is not None, f"{structure_name}: Global delta should not be None"
        assert global_distortions['delta'] < DELTA_TOLERANCE, (
            f"{structure_name}: Global delta {global_distortions['delta']:.6f} should be < {DELTA_TOLERANCE}"
        )
        
        results['global'] = {
            'bxb_mean': global_bxb['bxb_mean'],
            'delta': global_distortions['delta'],
            'sigma': global_distortions['sigma'],
            'lambda': global_distortions['lambda'],
        }
        
        # Validate per-layer results
        for layer_id in layer_ids:
            if layer_id not in bxb_data_by_layer:
                continue  # Skip layers without B-X-B data
            
            layer_bxb = bxb_data_by_layer[layer_id]
            if layer_id not in distortions_by_layer:
                continue  # Skip layers without distortion data
            
            layer_distortions = distortions_by_layer[layer_id]
            
            # Validate layer B-X-B angles
            if layer_bxb['bxb_mean'] is not None:
                layer_mean_deviation = abs(layer_bxb['bxb_mean'] - BXB_ANGLE_TARGET)
                assert layer_mean_deviation < ANGLE_TOLERANCE, (
                    f"{structure_name} Layer {layer_id}: Mean B-X-B angle {layer_bxb['bxb_mean']:.2f}° "
                    f"should be close to {BXB_ANGLE_TARGET}° (deviation: {layer_mean_deviation:.2f}°)"
                )
            
            # Validate layer delta
            if layer_distortions['delta'] is not None:
                assert layer_distortions['delta'] < DELTA_TOLERANCE, (
                    f"{structure_name} Layer {layer_id}: Delta {layer_distortions['delta']:.6f} "
                    f"should be < {DELTA_TOLERANCE}"
                )
            
            results['layers'][layer_id] = {
                'bxb_mean': layer_bxb.get('bxb_mean'),
                'delta': layer_distortions.get('delta'),
                'sigma': layer_distortions.get('sigma'),
                'lambda': layer_distortions.get('lambda'),
            }
        
        return results
    
    def test_bulk_small(self):
        """Test perfect octahedra in small bulk structure."""
        print("\n" + "="*60)
        print("TEST: Bulk Small (2x2, thickness=2)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            xy_expansion=(2, 2),
            thickness=2,
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "Bulk Small")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print(f"  Sigma: {results['sigma']:.6f}")
        print(f"  Lambda: {results['lambda']:.6f}")
        print("  ✓ Bulk Small validation passed")
    
    def test_bulk_medium(self):
        """Test perfect octahedra in medium bulk structure."""
        print("\n" + "="*60)
        print("TEST: Bulk Medium (3x3, thickness=2)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            xy_expansion=(3, 3),
            thickness=2,
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "Bulk Medium")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print("  ✓ Bulk Medium validation passed")
    
    def test_bulk_large(self):
        """Test perfect octahedra in large bulk structure."""
        print("\n" + "="*60)
        print("TEST: Bulk Large (4x4, thickness=3)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            xy_expansion=(4, 4),
            thickness=3,
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "Bulk Large")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print("  ✓ Bulk Large validation passed")
    
    def test_rp_small(self):
        """Test perfect octahedra in small RP structure."""
        print("\n" + "="*60)
        print("TEST: RP Small (2x2, thickness=2)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="RP",
            xy_expansion=(2, 2),
            thickness=2,
            spacer="[NH3+]CCC",  # Propylammonium (monofunctional)
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "RP Small")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print("  ✓ RP Small validation passed")
    
    def test_rp_medium(self):
        """Test perfect octahedra in medium RP structure."""
        print("\n" + "="*60)
        print("TEST: RP Medium (3x3, thickness=3)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="RP",
            xy_expansion=(3, 3),
            thickness=3,
            spacer="[NH3+]CCCC[NH3+]",  # Butanediammonium (bifunctional)
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "RP Medium")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print("  ✓ RP Medium validation passed")
    
    def test_rp_large(self):
        """Test perfect octahedra in large RP structure."""
        print("\n" + "="*60)
        print("TEST: RP Large (4x4, thickness=4)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="RP",
            xy_expansion=(4, 4),
            thickness=4,
            spacer="[NH3+]CCCCCC[NH3+]",  # Hexanediammonium
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "RP Large")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print("  ✓ RP Large validation passed")
    
    def test_dj_small_molecular(self):
        """Test perfect octahedra in small DJ structure with molecular spacer."""
        print("\n" + "="*60)
        print("TEST: DJ Small Molecular (2x2, thickness=2, BDA)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="DJ",
            xy_expansion=(2, 2),
            thickness=2,
            spacer="[NH3+]CCCC[NH3+]",  # BDA (butanediammonium)
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "DJ Small Molecular")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print("  ✓ DJ Small Molecular validation passed")
    
    def test_dj_medium_atomic(self):
        """Test perfect octahedra in medium DJ structure with atomic spacer."""
        print("\n" + "="*60)
        print("TEST: DJ Medium Atomic (3x3, thickness=3, Cs)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="DJ",
            xy_expansion=(3, 3),
            thickness=3,
            spacer="Cs",  # Atomic spacer
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "DJ Medium Atomic")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print("  ✓ DJ Medium Atomic validation passed")
    
    def test_dj_large_molecular(self):
        """Test perfect octahedra in large DJ structure with molecular spacer."""
        print("\n" + "="*60)
        print("TEST: DJ Large Molecular (4x4, thickness=4, PDA)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="DJ",
            xy_expansion=(4, 4),
            thickness=4,
            spacer="[NH3+]CCCCCC[NH3+]",  # PDA (propanediammonium-like, longer chain)
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "DJ Large Molecular")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print("  ✓ DJ Large Molecular validation passed")
    
    def test_dj_medium_different_spacer(self):
        """Test perfect octahedra in medium DJ structure with different molecular spacer."""
        print("\n" + "="*60)
        print("TEST: DJ Medium Different Spacer (3x3, thickness=3, EDBE)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="EDBE",  # Using EDBE as A-ion, but spacer will be different
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="DJ",
            xy_expansion=(3, 3),
            thickness=3,
            spacer="[NH3+]CCCCCCCC[NH3+]",  # Octanediammonium (longer chain)
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        results = self._validate_perfect_octahedra(analyzer, "DJ Medium Different Spacer")
        
        print(f"  B-X-B mean: {results['bxb_mean']:.2f}°")
        print(f"  X-B-X mean: {results['xbx_mean']:.2f}°")
        print(f"  B-X bond length: {results['bond_length_mean']:.4f} Å (std: {results['bond_length_std']:.4f} Å)")
        print(f"  Delta: {results['delta']:.6f}")
        print("  ✓ DJ Medium Different Spacer validation passed")
    
    def test_dj_thick_molecular_grouped(self):
        """Test perfect octahedra in thick DJ structure (n=5) with layer grouping."""
        print("\n" + "="*60)
        print("TEST: DJ Thick Molecular Grouped (3x3, thickness=5, BDA)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="DJ",
            xy_expansion=(3, 3),
            thickness=5,
            spacer="[NH3+]CCCC[NH3+]",  # BDA (butanediammonium)
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        # First validate global properties
        global_results = self._validate_perfect_octahedra(analyzer, "DJ Thick Molecular")
        
        # Then validate grouped by layer
        layer_results = self._validate_perfect_octahedra_by_layer(analyzer, "DJ Thick Molecular")
        
        print(f"\n  Global Results:")
        print(f"    B-X-B mean: {layer_results['global']['bxb_mean']:.2f}°")
        print(f"    Delta: {layer_results['global']['delta']:.6f}")
        print(f"    Sigma: {layer_results['global']['sigma']:.6f}")
        print(f"    Lambda: {layer_results['global']['lambda']:.6f}")
        
        print(f"\n  Per-Layer Results:")
        for layer_id, layer_data in layer_results['layers'].items():
            if layer_data['bxb_mean'] is not None and layer_data['delta'] is not None:
                print(f"    Layer {layer_id}:")
                print(f"      B-X-B mean: {layer_data['bxb_mean']:.2f}°")
                print(f"      Delta: {layer_data['delta']:.6f}")
                print(f"      Sigma: {layer_data['sigma']:.6f}")
                print(f"      Lambda: {layer_data['lambda']:.6f}")
        
        print(f"\n  ✓ DJ Thick Molecular Grouped validation passed")
        print(f"  ✓ Layer grouping support confirmed")
    
    def test_dj_very_thick_atomic_grouped(self):
        """Test perfect octahedra in very thick DJ structure (n=6) with layer grouping."""
        print("\n" + "="*60)
        print("TEST: DJ Very Thick Atomic Grouped (4x4, thickness=6, Cs)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="DJ",
            xy_expansion=(4, 4),
            thickness=6,
            spacer="Cs",  # Atomic spacer
            glazer_pattern="a0a0a0",
            glazer_angles=[0, 0, 0],
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        # First validate global properties
        global_results = self._validate_perfect_octahedra(analyzer, "DJ Very Thick Atomic")
        
        # Then validate grouped by layer
        layer_results = self._validate_perfect_octahedra_by_layer(analyzer, "DJ Very Thick Atomic")
        
        print(f"\n  Global Results:")
        print(f"    B-X-B mean: {layer_results['global']['bxb_mean']:.2f}°")
        print(f"    Delta: {layer_results['global']['delta']:.6f}")
        print(f"    Sigma: {layer_results['global']['sigma']:.6f}")
        print(f"    Lambda: {layer_results['global']['lambda']:.6f}")
        
        print(f"\n  Per-Layer Results:")
        for layer_id, layer_data in layer_results['layers'].items():
            if layer_data['bxb_mean'] is not None and layer_data['delta'] is not None:
                print(f"    Layer {layer_id}:")
                print(f"      B-X-B mean: {layer_data['bxb_mean']:.2f}°")
                print(f"      Delta: {layer_data['delta']:.6f}")
                print(f"      Sigma: {layer_data['sigma']:.6f}")
                print(f"      Lambda: {layer_data['lambda']:.6f}")
        
        print(f"\n  ✓ DJ Very Thick Atomic Grouped validation passed")
        print(f"  ✓ Layer grouping support confirmed")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
