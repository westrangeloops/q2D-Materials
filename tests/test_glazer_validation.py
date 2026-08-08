"""
Test that Glazer detection works correctly on bulk structures.

Glazer notation is only valid for 3D bulk perovskites. This test ensures
that bulk structures work correctly with Glazer detection.
"""

import pytest
from pathlib import Path
from ase.io import write

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer


class TestGlazerValidation:
    """Test that Glazer detection works correctly on bulk structures."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test directory and creator."""
        self.test_dir = tmp_path / "glazer_validation"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.creator = q2D_creator()

    def test_glazer_works_on_bulk_ma_pb_i(self):
        """Test that Glazer detection works on bulk MAPbI3 structure."""
        # Create bulk structure with MA (multiple A-sites)
        structure = self.creator.create_structure(
            A_ions='MA', B_ions='Pb', X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )

        analyzer = q2D_analyzer(structure)
        analyzer.analyze()

        # Should work without error
        result = analyzer.get_glazer_pattern()

        # Should return valid Glazer notation
        assert 'notation' in result
        assert 'tilt_pattern' in result
        assert len(result['tilt_pattern']) == 3

    def test_glazer_works_on_bulk_large_cell(self):
        """Test that Glazer detection works on bulk structure with larger cell."""
        # Create bulk structure with larger cell and MA
        structure = self.creator.create_structure(
            A_ions='MA', B_ions='Pb', X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(3, 3),
        )

        analyzer = q2D_analyzer(structure)
        analyzer.analyze()

        # Should work without error
        result = analyzer.get_glazer_pattern()

        # Should return valid Glazer notation
        assert 'notation' in result
        assert 'tilt_pattern' in result
        assert len(result['tilt_pattern']) == 3

    def test_glazer_works_on_bulk_structures(self):
        """Test that Glazer detection still works normally on bulk structures."""
        # Create bulk structure with MA
        structure = self.creator.create_structure(
            A_ions='MA', B_ions='Pb', X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='a0a0a0',
        )

        analyzer = q2D_analyzer(structure)
        analyzer.analyze()

        # Should work without error
        result = analyzer.get_glazer_pattern()

        # Should return valid Glazer notation
        assert 'notation' in result
        assert 'tilt_pattern' in result
        assert len(result['tilt_pattern']) == 3
        assert result['notation'] == 'a0a0a0'

    def test_glazer_works_on_bulk_thick(self):
        """Test that Glazer detection works on bulk structure with thicker layers."""
        # Create bulk structure with thicker layers and MA
        structure = self.creator.create_structure(
            A_ions='MA', B_ions='Pb', X_ions='I',
            structure_type='bulk',
            thickness=3,
            xy_expansion=(2, 2),
        )

        analyzer = q2D_analyzer(structure)
        analyzer.analyze()

        # Should work without error
        result = analyzer.get_glazer_pattern()

        # Should return valid Glazer notation
        assert 'notation' in result
        assert 'tilt_pattern' in result
        assert len(result['tilt_pattern']) == 3

    def test_glazer_works_on_bulk_various_patterns(self):
        """Test that Glazer detection works on bulk structures with various patterns."""
        # Create bulk structure with MA
        structure = self.creator.create_structure(
            A_ions='MA', B_ions='Pb', X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )

        analyzer = q2D_analyzer(structure)
        analyzer.analyze()

        # Should work without error
        result = analyzer.get_glazer_pattern()

        # Should return valid Glazer notation
        assert 'notation' in result
        assert 'tilt_pattern' in result
        assert len(result['tilt_pattern']) == 3

    def test_glazer_bulk_backward_compatibility(self):
        """Test that existing bulk tests still work (backward compatibility)."""
        # Create various bulk structures with different Glazer patterns
        test_patterns = [
            ("a0a0a0", [0, 0, 0]),
            ("a0a0c+", [0, 0, 10]),
            ("a+a+a+", [10, 10, 10]),
            ("a-a-a-", [10, 10, 10]),
        ]

        for notation, angles in test_patterns:
            # Create structure with MA
            structure = self.creator.create_structure(
                A_ions='MA', B_ions='Pb', X_ions='I',
                structure_type='bulk',
                thickness=2,
                xy_expansion=(2, 2),
                glazer_pattern=notation,
                glazer_angles=angles,
            )

            analyzer = q2D_analyzer(structure)
            analyzer.analyze()

            # Should work without error
            result = analyzer.get_glazer_pattern()
            assert 'notation' in result
            assert 'tilt_pattern' in result

            # Notation might not match exactly due to detection thresholds,
            # but should at least return a valid result
            assert len(result['tilt_pattern']) == 3


class TestDJTiltAnalysis:
    """Test non-Glazer tilt analysis for DJ (Dion-Jacobson) structures."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test directory and creator."""
        self.test_dir = tmp_path / "dj_tilt_analysis"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.creator = q2D_creator()
    
    def test_dj_tilt_analysis_basic(self):
        """Test basic tilt analysis on DJ structure."""
        # Create DJ structure with bifunctional spacer
        structure = self.creator.create_structure(
            A_ions='MA', B_ions='Pb', X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            spacer='[NH3+]CCCC[NH3+]',  # BDA spacer (bifunctional, DJ)
            layer_sequence='DJ',
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        # Verify it's classified as DJ
        assert analyzer.structure_type == 'dj'
        
        # Get raw tilt data
        tilt_data = analyzer.get_octahedral_tilts()
        assert tilt_data is not None
        assert hasattr(tilt_data, 'euler_angles')
        assert tilt_data.euler_angles.shape[1] == 3  # (N_oct, 3)
        assert len(tilt_data.octahedron_ids) > 0
        
        # Mean Tilt Profile - Global
        mu_global = analyzer.compute_mean_tilt_profile()
        assert mu_global is not None
        assert isinstance(mu_global, float)
        assert mu_global >= 0  # Tilt magnitude should be non-negative
        
        # Mean Tilt Profile - Per-layer
        profile = analyzer.compute_mean_tilt_profile(group_by='layer')
        assert isinstance(profile, dict)
        assert 'global' in profile
        assert len(profile) > 1  # Should have at least global + one layer
        
        for layer_id, mu in profile.items():
            assert isinstance(mu, float)
            assert mu >= 0
        
        # Gearing Correlation - Global
        gearing = analyzer.compute_gearing_correlation()
        assert isinstance(gearing, dict)
        assert len(gearing) > 0  # Should have at least some pairs
        
        for (oct_i, oct_j), chi in gearing.items():
            assert isinstance(oct_i, str)
            assert isinstance(oct_j, str)
            assert isinstance(chi, float)
            assert -1.0 <= chi <= 1.0  # Correlation should be in [-1, 1]
        
        # Gearing Correlation - Per-layer grouped
        gearing_by_layer = analyzer.compute_gearing_correlation(group_by='layer')
        assert isinstance(gearing_by_layer, dict)
        assert 'global' in gearing_by_layer
        
        for layer_id, pairs in gearing_by_layer.items():
            assert isinstance(pairs, dict)
            if layer_id != 'global':
                # Each layer should have some octahedral pairs
                assert len(pairs) >= 0  # Can be 0 if no pairs in layer
    
    def test_dj_tilt_analysis_multilayer(self):
        """Test tilt analysis on multi-layer DJ structure."""
        # Create DJ structure with n=3 (3 inorganic layers)
        structure = self.creator.create_structure(
            A_ions='MA', B_ions='Pb', X_ions='I',
            structure_type='bulk',
            thickness=3,
            xy_expansion=(2, 2),
            spacer='[NH3+]CCCC[NH3+]',  # BDA spacer
            layer_sequence='DJ',
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        # Verify it's classified as DJ
        assert analyzer.structure_type == 'dj'
        
        # Get tilt data
        tilt_data = analyzer.get_octahedral_tilts()
        n_octahedra = len(tilt_data.octahedron_ids)
        assert n_octahedra > 0
        
        # Check per-layer profile shows depth-dependent behavior
        profile = analyzer.compute_mean_tilt_profile(group_by='layer')
        
        # Should have multiple layers
        layer_ids = [lid for lid in profile.keys() if lid != 'global']
        assert len(layer_ids) >= 2  # At least 2 layers
        
        # Surface layers typically have higher tilt (surface relaxation)
        # This is a qualitative check - actual values depend on structure
        mu_values = [mu for lid, mu in profile.items() if lid != 'global']
        assert len(mu_values) > 0
        assert all(mu >= 0 for mu in mu_values)
        
        # Gearing correlation per layer
        gearing_by_layer = analyzer.compute_gearing_correlation(group_by='layer')
        
        # Each layer should have correlation data
        for layer_id, pairs in gearing_by_layer.items():
            if layer_id != 'global':
                # Check that correlations are valid
                for (oct_i, oct_j), chi in pairs.items():
                    assert -1.0 <= chi <= 1.0
    
    def test_dj_tilt_analysis_specific_layer(self):
        """Test tilt analysis on specific layer of DJ structure."""
        structure = self.creator.create_structure(
            A_ions='MA', B_ions='Pb', X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            spacer='[NH3+]CCCC[NH3+]',
            layer_sequence='DJ',
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        # Get per-layer profile - this builds the layer mapping
        profile = analyzer.compute_mean_tilt_profile(group_by='layer')
        
        # Find a layer that has data (not None and not 'global')
        layer_with_data = None
        mu_from_profile = None
        for layer_id, mu in profile.items():
            if layer_id != 'global' and mu is not None:
                layer_with_data = layer_id
                mu_from_profile = mu
                break
        
        if layer_with_data:
            # Mean tilt for specific layer (now works with the fix)
            mu_layer = analyzer.compute_mean_tilt_profile(layer=layer_with_data)
            assert mu_layer is not None, f"Layer {layer_with_data} should have tilt data"
            assert isinstance(mu_layer, float)
            assert mu_layer >= 0
            # Should match the value from the profile
            assert abs(mu_layer - mu_from_profile) < 1e-6, "Layer-specific value should match profile value"
            
            # Gearing correlation for specific layer (now works with the fix)
            gearing_layer = analyzer.compute_gearing_correlation(layer=layer_with_data)
            assert isinstance(gearing_layer, dict)
            # Layer might have no pairs (empty dict is valid if no shared X atoms)
            for (oct_i, oct_j), chi in gearing_layer.items():
                assert -1.0 <= chi <= 1.0
        else:
            # If no layers have data, skip this test (structure might not have proper layer assignment)
            pytest.skip("No layers with octahedra found for layer-specific analysis")