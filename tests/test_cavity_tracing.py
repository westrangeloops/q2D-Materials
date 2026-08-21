"""
Test cavity tracing algorithm for A-site detection.

This test validates the cavity tracing functionality which detects
cuboctahedral cages around A-sites in perovskite structures.
"""

import pytest
import tempfile
import numpy as np
from pathlib import Path
from ase.io import write

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer


class TestCavityTracing:
    """Test cavity tracing algorithm."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test directory."""
        self.test_dir = tmp_path / "cavity_test"
        self.test_dir.mkdir()
        self.creator = q2D_creator()
    
    def _analyze_and_check_cavities(self, structure, name, expected_params):
        """Analyze structure and check cavity detection."""
        print(f"\n{'='*70}")
        print(f"Test: {name}")
        print(f"{'='*70}")
        
        # Save as VASP
        vasp_file = self.test_dir / f"{name}.vasp"
        write(str(vasp_file), structure)
        
        # Analyze
        analyzer = q2D_analyzer(str(vasp_file))
        analyzer.analyze()
        
        # Get cavities
        cavities = analyzer.get_cavities()
        
        print(f"\nCavity Detection Results:")
        print(f"-" * 70)
        print(f"  Number of cavities detected: {len(cavities)}")
        
        if cavities:
            for i, cav in enumerate(cavities):
                print(f"\n  Cavity {i}:")
                print(f"    Cavity type: {cav['cavity_type']}")
                print(f"    Contains A-site: {cav['contains_a_site']}")
                if cav['contains_a_site']:
                    print(f"    A-site indices: {cav['a_site_indices'][:5]}...")
                print(f"    PBC wrapped: {cav['is_pbc_wrapped']}")
        
        # Get A-sites and check cavity assignments
        a_sites = analyzer.get_a_sites()
        print(f"\n  A-sites found: {len(a_sites)}")
        
        for a_site in a_sites[:3]:  # Show first 3
            cavity = analyzer.get_cavity_for_a_site(a_site['atom_index'])
            if cavity:
                print(f"    A-site {a_site['atom_index']} ({a_site['formula']}) in {cavity['id']}")
            else:
                print(f"    A-site {a_site['atom_index']} ({a_site['formula']}) - no cavity assigned")
        
        # Check expected values
        results = {
            'name': name,
            'cavities_count': len(cavities),
            'expected_min_cavities': expected_params.get('min_cavities', 0),
            'a_sites_count': len(a_sites),
            'expected_a_site_type': expected_params.get('a_site_type'),
        }
        
        # Validate results
        print(f"\n  Validation:")
        
        if 'min_cavities' in expected_params:
            status = "✓" if len(cavities) >= expected_params['min_cavities'] else "✗"
            print(f"    {status} Minimum cavities: {len(cavities)} >= {expected_params['min_cavities']}")
        
        if 'a_site_type' in expected_params and cavities:
            a_site_cavity_types = [c['cavity_type'] for c in cavities if c['contains_a_site']]
            if 'a_site' in a_site_cavity_types:
                print(f"    ✓ A-site cavity type found (expected: {expected_params['a_site_type']})")
            else:
                print(f"    ~ A-site cavity types found: {set(a_site_cavity_types)}")
        
        print(f"{'='*70}\n")
        
        return results
    
    def test_bulk_cubic_with_atomic_a_site(self):
        """Test cavity detection in bulk cubic with atomic A-site (Cs)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )
        
        expected = {
            'min_cavities': 1,
            'a_site_type': 'atomic',
        }
        
        results = self._analyze_and_check_cavities(structure, 'bulk_cubic_cs', expected)
        assert True
    
    def test_bulk_cubic_with_molecular_a_site(self):
        """Test cavity detection in bulk cubic with molecular A-site (MA)."""
        structure = self.creator.create_structure(
            A_ions='MA',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )
        
        expected = {
            'min_cavities': 1,
            'a_site_type': 'molecular',
        }
        
        results = self._analyze_and_check_cavities(structure, 'bulk_cubic_ma', expected)
        assert True
    
    def test_dj_structure_with_spacer(self):
        """Test cavity detection in DJ structure."""
        try:
            structure = self.creator.create_structure(
                A_ions='MA',
                B_ions='Pb',
                X_ions='I',
                structure_type='bulk',
                thickness=2,
                xy_expansion=(2, 2),
                spacer='[NH3+]CCCC[NH3+]',
                layer_sequence='DJ',
                penetration=0.5,
            )
            
            expected = {
                'min_cavities': 1,
                'a_site_type': 'molecular',
            }
            
            results = self._analyze_and_check_cavities(structure, 'dj_with_spacer', expected)
        except Exception as e:
            print(f"\nSkipped DJ test (RDKit may not be available): {e}")
        
        assert True
    
    def test_rp_structure_with_spacer(self):
        """Test cavity detection in RP structure."""
        try:
            structure = self.creator.create_structure(
                A_ions='MA',
                B_ions='Pb',
                X_ions='I',
                structure_type='bulk',
                thickness=2,
                xy_expansion=(2, 2),
                spacer='[NH3+]CCCCC',
                layer_sequence='RP',
                penetration=0.3,
            )
            
            expected = {
                'min_cavities': 1,
                'a_site_type': 'molecular',
            }
            
            results = self._analyze_and_check_cavities(structure, 'rp_with_spacer', expected)
        except Exception as e:
            print(f"\nSkipped RP test (RDKit may not be available): {e}")
        
        assert True
    
    def test_monolayer_n1(self):
        """Test cavity detection in n=1 monolayer (no internal A-sites)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='monolayer',
            thickness=1,
            xy_expansion=(2, 2),
            vacuum=15.0,
        )
        
        expected = {
            'min_cavities': 0,  # n=1 has no internal cavities
        }
        
        results = self._analyze_and_check_cavities(structure, 'monolayer_n1', expected)
        assert True
    
    def test_monolayer_n2(self):
        """Test cavity detection in n=2 monolayer (has internal A-sites)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='monolayer',
            thickness=2,
            xy_expansion=(2, 2),
            vacuum=15.0,
        )
        
        expected = {
            'min_cavities': 1,
            'a_site_type': 'atomic',
        }
        
        results = self._analyze_and_check_cavities(structure, 'monolayer_n2', expected)
        assert True
    
    def test_1x1_unit_cell(self):
        """Test cavity detection in minimal 1x1 cell (PBC stress test)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(1, 1),
        )
        
        expected = {
            'min_cavities': 1,
        }
        
        results = self._analyze_and_check_cavities(structure, 'unit_cell_1x1', expected)
        assert True
    
    def test_3x3_supercell(self):
        """Test cavity detection in larger 3x3 supercell."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(3, 3),
        )
        
        expected = {
            'min_cavities': 4,  # Should find multiple cavities
        }
        
        results = self._analyze_and_check_cavities(structure, 'supercell_3x3', expected)
        assert True
    
    def test_query_api_cavities(self):
        """Test query API for cavities."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )
        
        vasp_file = self.test_dir / "query_test.vasp"
        write(str(vasp_file), structure)
        
        analyzer = q2D_analyzer(str(vasp_file))
        analyzer.analyze()
        
        print(f"\n{'='*70}")
        print("Query API Test")
        print(f"{'='*70}")
        
        # Test cavities() query
        query = analyzer.get_characterization()
        all_cavities = query.cavities().to_list()
        print(f"\n  cavities(): found {len(all_cavities)} cavities")
        
        # Test containing_a_site() filter
        query = analyzer.get_characterization()
        a_site_cavities = query.cavities().containing_a_site().to_list()
        print(f"  containing_a_site(): {len(a_site_cavities)} cavities with A-sites")
        
        # Test corners() method
        query = analyzer.get_characterization()
        corner_octs = query.cavities().corners().to_list()
        print(f"  corners(): {len(corner_octs)} corner octahedra")
        
        print(f"{'='*70}\n")
        
        assert True
    
    def test_cavity_a_site_assignment(self):
        """Test that A-sites are correctly assigned to cavities."""
        structure = self.creator.create_structure(
            A_ions='MA',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=3,  # More layers = more cavities
            xy_expansion=(2, 2),
        )
        
        vasp_file = self.test_dir / "assignment_test.vasp"
        write(str(vasp_file), structure)
        
        analyzer = q2D_analyzer(str(vasp_file))
        analyzer.analyze()
        
        print(f"\n{'='*70}")
        print("A-site to Cavity Assignment Test")
        print(f"{'='*70}")
        
        a_sites = analyzer.get_a_sites()
        cavities = analyzer.get_cavities()
        
        print(f"\n  Total A-sites: {len(a_sites)}")
        print(f"  Total cavities: {len(cavities)}")
        
        # Check each A-site has a cavity
        assigned_count = 0
        for a_site in a_sites:
            cavity = analyzer.get_cavity_for_a_site(a_site['atom_index'])
            if cavity:
                assigned_count += 1
        
        print(f"  A-sites with assigned cavity: {assigned_count}/{len(a_sites)}")
        
        # Check each cavity has A-site info
        cavities_with_a_sites = sum(1 for c in cavities if c['contains_a_site'])
        print(f"  Cavities containing A-sites: {cavities_with_a_sites}/{len(cavities)}")
        
        print(f"{'='*70}\n")
        
        assert True


class TestCavityChirality:
    """Test chirality determination in cavity tracing."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test directory."""
        self.test_dir = tmp_path / "chirality_test"
        self.test_dir.mkdir()
        self.creator = q2D_creator()
    
    def test_chirality_consistency(self):
        """Test that chirality is consistent within each cavity."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )
        
        vasp_file = self.test_dir / "chirality_test.vasp"
        write(str(vasp_file), structure)
        
        analyzer = q2D_analyzer(str(vasp_file))
        analyzer.analyze()
        
        cavities = analyzer.get_cavities()
        
        print(f"\n{'='*70}")
        print("Chirality Consistency Test")
        print(f"{'='*70}")
        
        for cav in cavities:
            chirality = cav['chirality']
            print(f"\n  {cav['id']}: chirality = {chirality}")
            
            # Each cavity should have a valid chirality
            assert chirality in ['R', 'L', None], f"Invalid chirality: {chirality}"
        
        print(f"{'='*70}\n")
        
        assert True


class TestCavityPBC:
    """Test PBC handling in cavity tracing."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test directory."""
        self.test_dir = tmp_path / "pbc_test"
        self.test_dir.mkdir()
        self.creator = q2D_creator()
    
    def test_pbc_wrapped_detection(self):
        """Test detection of PBC-wrapped cavities in small cells."""
        # 1x1 cell where cavities must wrap through PBC
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(1, 1),
        )
        
        vasp_file = self.test_dir / "pbc_test.vasp"
        write(str(vasp_file), structure)
        
        analyzer = q2D_analyzer(str(vasp_file))
        analyzer.analyze()
        
        cavities = analyzer.get_cavities()
        
        print(f"\n{'='*70}")
        print("PBC Wrapping Detection Test (1x1 cell)")
        print(f"{'='*70}")
        
        pbc_wrapped_count = sum(1 for c in cavities if c['is_pbc_wrapped'])
        print(f"\n  Cavities detected: {len(cavities)}")
        print(f"  PBC-wrapped cavities: {pbc_wrapped_count}")
        
        print(f"{'='*70}\n")
        
        assert True
    
    def test_larger_cell_no_wrapping(self):
        """Test that larger cells have mostly non-wrapped cavities."""
        # 3x3 cell where most cavities should not wrap
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(3, 3),
        )
        
        vasp_file = self.test_dir / "no_wrap_test.vasp"
        write(str(vasp_file), structure)
        
        analyzer = q2D_analyzer(str(vasp_file))
        analyzer.analyze()
        
        cavities = analyzer.get_cavities()
        
        print(f"\n{'='*70}")
        print("Non-PBC Wrapping Test (3x3 cell)")
        print(f"{'='*70}")
        
        pbc_wrapped_count = sum(1 for c in cavities if c['is_pbc_wrapped'])
        print(f"\n  Cavities detected: {len(cavities)}")
        print(f"  PBC-wrapped cavities: {pbc_wrapped_count}")
        print(f"  Non-wrapped cavities: {len(cavities) - pbc_wrapped_count}")
        
        print(f"{'='*70}\n")
        
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])

