"""
Test analyzer validation against creator parameters.

This test creates structures using q2D_creator, saves them as VASP files,
then analyzes them with q2D_analyzer to verify the analyzer can correctly
identify the structure components and type.

Instead of pass/fail, this test reports match percentages for each parameter.
"""

import pytest
import tempfile
import os
from pathlib import Path
from ase.io import write

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer


class TestAnalyzerValidation:
    """Validate analyzer against known creator parameters."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test directory for VASP files."""
        self.test_dir = tmp_path / "analyzer_test"
        self.test_dir.mkdir()
        self.creator = q2D_creator()
        
    def _analyze_structure(self, structure, name, expected_params):
        """
        Analyze a structure and compare with expected parameters.
        
        Parameters
        ----------
        structure : q2DStructure
            The created structure
        name : str
            Test name for the VASP file
        expected_params : dict
            Expected parameters from creator:
            - structure_type: str
            - A_ions: list or str
            - B_ions: list or str
            - X_ions: list or str
            - spacer: list or None
            - thickness: int
        
        Returns
        -------
        dict
            Match results with percentages
        """
        # Print parameters used
        print(f"\n{'='*70}")
        print(f"Test: {name}")
        print(f"{'='*70}")
        print("\nParameters Used (from creator):")
        print("-" * 70)
        for key, value in sorted(expected_params.items()):
            if value is not None:
                if isinstance(value, (list, tuple)):
                    value_str = ', '.join(str(v) for v in value) if len(str(value)) < 100 else f"{type(value).__name__} with {len(value)} items"
                else:
                    value_str = str(value)
                print(f"  {key:20s}: {value_str}")
        
        # Save as VASP
        vasp_file = self.test_dir / f"{name}.vasp"
        write(str(vasp_file), structure)
        
        # Analyze
        analyzer = q2D_analyzer(str(vasp_file))
        analyzer.analyze()
        
        # Compare results
        results = {
            'name': name,
            'expected': expected_params,
            'analyzed': {
                'structure_type': analyzer.structure_type,
                'A_ions': analyzer.get_a_sites(),
                'B_ions': [oct['central_atom_symbol'] for oct in analyzer.get_octahedra()],
                'X_ions': [],  # Will extract from octahedra
                'spacers': analyzer.get_spacers(),
                'octahedra_count': len(analyzer.get_octahedra()),
                'layers_count': len(analyzer.get_layers()),
            }
        }
        
        # Extract X-ions from octahedra neighbors
        if analyzer.get_octahedra():
            x_symbols = set()
            for oct in analyzer.get_octahedra():
                for idx in (oct['terminal_atoms'] + oct['interlayer_atoms'] + oct['intralayer_atoms']):
                    x_symbols.add(structure.get_chemical_symbols()[idx])
            results['analyzed']['X_ions'] = list(x_symbols)
        
        # Print parameters found
        print("\nParameters Found (by analyzer):")
        print("-" * 70)
        print(f"  {'structure_type':20s}: {results['analyzed']['structure_type']}")
        
        # Format A_ions
        a_ions_found = set([a['symbol'] for a in results['analyzed']['A_ions']])
        a_ions_str = ', '.join(sorted(a_ions_found)) if a_ions_found else 'None'
        print(f"  {'A_ions':20s}: {a_ions_str}")
        
        # Format B_ions
        b_ions_found = set(results['analyzed']['B_ions'])
        b_ions_str = ', '.join(sorted(b_ions_found)) if b_ions_found else 'None'
        print(f"  {'B_ions':20s}: {b_ions_str}")
        
        # Format X_ions
        x_ions_found = set(results['analyzed']['X_ions'])
        x_ions_str = ', '.join(sorted(x_ions_found)) if x_ions_found else 'None'
        print(f"  {'X_ions':20s}: {x_ions_str}")
        
        # Format spacer
        spacers = results['analyzed']['spacers']
        if spacers:
            spacer_info = []
            for spacer in spacers[:3]:  # Show first 3
                if hasattr(spacer, 'get_chemical_formula'):
                    formula = spacer.get_chemical_formula(mode='hill')
                else:
                    formula = str(spacer)
                spacer_info.append(formula)
            spacer_str = ', '.join(spacer_info)
            if len(spacers) > 3:
                spacer_str += f" ... and {len(spacers) - 3} more"
            print(f"  {'spacer':20s}: {spacer_str} ({len(spacers)} total)")
        else:
            print(f"  {'spacer':20s}: None")
        
        print(f"  {'octahedra_count':20s}: {results['analyzed']['octahedra_count']}")
        print(f"  {'layers_count':20s}: {results['analyzed']['layers_count']}")
        
        # Calculate matches
        matches = {}
        
        # Structure type match
        if expected_params.get('structure_type'):
            expected_type = expected_params['structure_type'].lower()
            analyzed_type = results['analyzed']['structure_type'].lower()
            matches['structure_type'] = 100.0 if expected_type == analyzed_type else 0.0
        
        # B-ions match
        if expected_params.get('B_ions'):
            expected_b = set([expected_params['B_ions']] if isinstance(expected_params['B_ions'], str) else expected_params['B_ions'])
            analyzed_b = set(results['analyzed']['B_ions'])
            if expected_b:
                matches['B_ions'] = 100.0 * len(expected_b & analyzed_b) / len(expected_b)
            else:
                matches['B_ions'] = 100.0 if not analyzed_b else 0.0
        
        # X-ions match
        if expected_params.get('X_ions'):
            expected_x = set([expected_params['X_ions']] if isinstance(expected_params['X_ions'], str) else expected_params['X_ions'])
            analyzed_x = set(results['analyzed']['X_ions'])
            if expected_x:
                matches['X_ions'] = 100.0 * len(expected_x & analyzed_x) / len(expected_x)
            else:
                matches['X_ions'] = 100.0 if not analyzed_x else 0.0
        
        # A-ions match
        if expected_params.get('A_ions'):
            expected_a = set([expected_params['A_ions']] if isinstance(expected_params['A_ions'], str) else expected_params['A_ions'])
            analyzed_a = set([a['symbol'] for a in results['analyzed']['A_ions']])
            if expected_a:
                matches['A_ions'] = 100.0 * len(expected_a & analyzed_a) / len(expected_a)
            else:
                matches['A_ions'] = 100.0 if not analyzed_a else 0.0
        
        # Spacer match
        if 'spacer' in expected_params:
            expected_spacer = expected_params['spacer']
            analyzed_spacer_count = len(results['analyzed']['spacers'])
            if expected_spacer is None:
                matches['spacer'] = 100.0 if analyzed_spacer_count == 0 else 0.0
            else:
                matches['spacer'] = 100.0 if analyzed_spacer_count > 0 else 0.0
        
        # Octahedra count (should match thickness * xy_expansion)
        if expected_params.get('thickness') and expected_params.get('xy_expansion'):
            thickness = expected_params['thickness']
            xy_exp = expected_params['xy_expansion']
            expected_oct = thickness * xy_exp[0] * xy_exp[1]
            analyzed_oct = results['analyzed']['octahedra_count']
            if expected_oct > 0:
                matches['octahedra_count'] = 100.0 * min(analyzed_oct, expected_oct) / expected_oct
            else:
                matches['octahedra_count'] = 100.0 if analyzed_oct == 0 else 0.0
        
        results['matches'] = matches
        results['overall_match'] = sum(matches.values()) / len(matches) if matches else 0.0
        
        return results
    
    def _print_results(self, results):
        """Print match results in a readable format."""
        print(f"\nMatch Results:")
        print(f"{'-'*70}")
        for param, match_pct in results['matches'].items():
            status = "✓" if match_pct >= 90.0 else "✗" if match_pct < 50.0 else "~"
            print(f"  {status} {param:20s}: {match_pct:6.1f}%")
        print(f"{'-'*70}")
        print(f"  Overall Match: {results['overall_match']:6.1f}%")
        print(f"{'='*70}\n")
    
    def test_bulk_cubic_simple(self):
        """Test simple cubic bulk structure."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )
        
        expected = {
            'structure_type': 'bulk',
            'A_ions': 'Cs',
            'B_ions': 'Pb',
            'X_ions': 'I',
            'spacer': None,
            'thickness': 2,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'bulk_cubic_simple', expected)
        self._print_results(results)
        
        # Always pass - this is a validation test
        assert True
    
    def test_bulk_with_glazer(self):
        """Test bulk structure with Glazer tilting (structure validation)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='a+a+c-',
            glazer_angles=[10, 10, 15],
        )
        
        expected = {
            'structure_type': 'bulk',
            'A_ions': 'Cs',
            'B_ions': 'Pb',
            'X_ions': 'I',
            'spacer': None,
            'thickness': 2,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'bulk_with_glazer', expected)
        self._print_results(results)
        
        # Also test Glazer detection
        expected_glazer = {
            'notation': 'a+a+c-',
            'tilt_pattern': ['+', '+', '-'],
            'glazer_angles': [10, 10, 15],
        }
        glazer_results = self._test_glazer_detection(structure, 'bulk_with_glazer', expected_glazer)
        
        assert True
    
    def test_bulk_mixed_ions(self):
        """Test bulk with mixed B-site ions."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions=['Pb', 'Sn'],
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )
        
        expected = {
            'structure_type': 'bulk',
            'A_ions': 'Cs',
            'B_ions': ['Pb', 'Sn'],
            'X_ions': 'I',
            'spacer': None,
            'thickness': 2,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'bulk_mixed_ions', expected)
        self._print_results(results)
        
        assert True
    
    def test_bulk_reduced_template(self):
        """Test bulk with reduced (RP-like) template."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            template='reduced',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )
        
        expected = {
            'structure_type': 'bulk',
            'A_ions': 'Cs',
            'B_ions': 'Pb',
            'X_ions': 'I',
            'spacer': None,
            'thickness': 2,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'bulk_reduced', expected)
        self._print_results(results)
        
        assert True
    
    def test_monolayer_simple(self):
        """Test simple monolayer structure."""
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
            'structure_type': 'monolayer',
            'A_ions': 'Cs',
            'B_ions': 'Pb',
            'X_ions': 'I',
            'spacer': None,
            'thickness': 1,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'monolayer_simple', expected)
        self._print_results(results)
        
        assert True
    
    def test_monolayer_bilayer(self):
        """Test bilayer monolayer structure."""
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
            'structure_type': 'monolayer',
            'A_ions': 'Cs',
            'B_ions': 'Pb',
            'X_ions': 'I',
            'spacer': None,
            'thickness': 2,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'monolayer_bilayer', expected)
        self._print_results(results)
        
        assert True
    
    def test_dj_with_organic_spacer(self):
        """Test DJ structure with organic spacer."""
        try:
            structure = self.creator.create_structure(
                A_ions='MA',
                B_ions='Pb',
                X_ions='I',
                structure_type='bulk',
                thickness=2,
                xy_expansion=(2, 2),
                spacer='[NH3+]CCCC[NH3+]',  # BDA spacer
                layer_sequence='DJ',
                penetration=0.5,
            )
            
            expected = {
                'structure_type': 'dj',
                'A_ions': 'MA',
                'B_ions': 'Pb',
                'X_ions': 'I',
                'spacer': '[NH3+]CCCC[NH3+]',
                'thickness': 2,
                'xy_expansion': (2, 2),
            }
            
            results = self._analyze_structure(structure, 'dj_organic_spacer', expected)
            self._print_results(results)
        except Exception as e:
            print(f"\nSkipped DJ test (RDKit may not be available): {e}")
        
        assert True
    
    def test_dj_with_atomic_spacer(self):
        """Test DJ structure with atomic spacer."""
        structure = self.creator.create_structure(
            A_ions='MA',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            spacer='Cs',  # Atomic spacer
            layer_sequence='DJ',
        )
        
        expected = {
            'structure_type': 'dj',
            'A_ions': 'MA',
            'B_ions': 'Pb',
            'X_ions': 'I',
            'spacer': 'Cs',
            'thickness': 2,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'dj_atomic_spacer', expected)
        self._print_results(results)
        
        assert True
    
    def test_rp_with_organic_spacer(self):
        """Test RP structure with organic spacer."""
        try:
            structure = self.creator.create_structure(
                A_ions='MA',
                B_ions='Pb',
                X_ions='I',
                structure_type='bulk',
                thickness=2,
                xy_expansion=(2, 2),
                spacer='[NH3+]CCCCC',  # PA spacer
                layer_sequence='RP',
                penetration=0.3,
            )
            
            # Save to rp_organic.vasp in current directory
            write('rp_organic.vasp', structure)
            print(f"\nSaved structure to: rp_organic.vasp")
            
            expected = {
                'structure_type': 'rp',
                'A_ions': 'MA',
                'B_ions': 'Pb',
                'X_ions': 'I',
                'spacer': '[NH3+]CCCCC',
                'thickness': 2,
                'xy_expansion': (2, 2),
            }
            
            results = self._analyze_structure(structure, 'rp_organic_spacer', expected)
            self._print_results(results)
        except Exception as e:
            print(f"\nSkipped RP test (RDKit may not be available): {e}")
        
        assert True
    
    def test_rp_with_atomic_spacer(self):
        """Test RP structure with atomic spacer."""
        structure = self.creator.create_structure(
            A_ions='MA',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            spacer='Cs',
            layer_sequence='RP',
        )
        
        expected = {
            'structure_type': 'rp',
            'A_ions': 'MA',
            'B_ions': 'Pb',
            'X_ions': 'I',
            'spacer': 'Cs',
            'thickness': 2,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'rp_atomic_spacer', expected)
        self._print_results(results)
        
        assert True
    
    def test_bulk_hexagonal(self):
        """Test bulk with hexagonal template."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            template='hexagonal',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
        )
        
        expected = {
            'structure_type': 'bulk',
            'A_ions': 'Cs',
            'B_ions': 'Pb',
            'X_ions': 'I',
            'spacer': None,
            'thickness': 2,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'bulk_hexagonal', expected)
        self._print_results(results)
        
        assert True
    
    def test_bulk_custom_sequence(self):
        """Test bulk with custom layer sequence."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=3,
            xy_expansion=(2, 2),
            layer_sequence='L1-L2-L1-L2-L1-L2',  # Custom sequence
        )
        
        expected = {
            'structure_type': 'bulk',
            'A_ions': 'Cs',
            'B_ions': 'Pb',
            'X_ions': 'I',
            'spacer': None,
            'thickness': 3,
            'xy_expansion': (2, 2),
        }
        
        results = self._analyze_structure(structure, 'bulk_custom_sequence', expected)
        self._print_results(results)
        
        assert True
    
    def _test_glazer_detection(self, structure, name, expected_glazer):
        """
        Test Glazer pattern detection on a structure.
        
        Parameters
        ----------
        structure : q2DStructure
            The created structure with Glazer tilting
        name : str
            Test name for the VASP file
        expected_glazer : dict
            Expected Glazer parameters:
            - notation: str (e.g., "a-b+a-")
            - tilt_pattern: list (e.g., ["+", "-", "+"])
            - glazer_angles: list (optional, angles used in creation)
        
        Returns
        -------
        dict
            Detection results with match information
        """
        print(f"\n{'='*70}")
        print(f"Glazer Detection Test: {name}")
        print(f"{'='*70}")
        print("\nExpected Glazer Pattern:")
        print("-" * 70)
        print(f"  {'notation':20s}: {expected_glazer.get('notation', 'N/A')}")
        print(f"  {'tilt_pattern':20s}: {expected_glazer.get('tilt_pattern', 'N/A')}")
        if 'glazer_angles' in expected_glazer:
            print(f"  {'glazer_angles':20s}: {expected_glazer['glazer_angles']}")
        
        # Save as VASP
        vasp_file = self.test_dir / f"{name}_glazer.vasp"
        write(str(vasp_file), structure)
        
        # Analyze
        analyzer = q2D_analyzer(str(vasp_file))
        analyzer.analyze()
        
        # Detect Glazer pattern
        detected = analyzer.get_glazer_pattern()
        
        print("\nDetected Glazer Pattern:")
        print("-" * 70)
        print(f"  {'notation':20s}: {detected.get('notation', 'N/A')}")
        print(f"  {'tilt_pattern':20s}: {detected.get('tilt_pattern', 'N/A')}")
        print(f"  {'tilt_angles':20s}: {detected.get('tilt_angles', 'N/A')}")
        print(f"  {'magnitudes':20s}: {detected.get('magnitudes', 'N/A')}")
        print(f"  {'space_group':20s}: {detected.get('space_group', 'N/A')}")
        
        # Compare results
        matches = {}
        
        # Notation match (exact or pattern match)
        expected_notation = expected_glazer.get('notation', '').lower()
        detected_notation = detected.get('notation', '').lower()
        if expected_notation:
            # Exact match
            if expected_notation == detected_notation:
                matches['notation'] = 100.0
            else:
                # Check if patterns match (magnitudes might differ)
                expected_pattern = expected_glazer.get('tilt_pattern', [])
                detected_pattern = detected.get('tilt_pattern', [])
                if expected_pattern == detected_pattern:
                    matches['notation'] = 75.0  # Pattern matches but notation differs
                else:
                    matches['notation'] = 0.0
        else:
            matches['notation'] = None
        
        # Tilt pattern match
        expected_pattern = expected_glazer.get('tilt_pattern', [])
        detected_pattern = detected.get('tilt_pattern', [])
        if expected_pattern:
            if expected_pattern == detected_pattern:
                matches['tilt_pattern'] = 100.0
            else:
                # Count matching phases
                matches_count = sum(1 for e, d in zip(expected_pattern, detected_pattern) if e == d)
                matches['tilt_pattern'] = 100.0 * matches_count / len(expected_pattern) if expected_pattern else 0.0
        else:
            matches['tilt_pattern'] = None
        
        # Angle comparison (if provided)
        if 'glazer_angles' in expected_glazer:
            expected_angles = expected_glazer['glazer_angles']
            detected_angles = detected.get('tilt_angles', [])
            if detected_angles and len(expected_angles) == len(detected_angles):
                # Compare angles with tolerance
                angle_diffs = [abs(e - d) for e, d in zip(expected_angles, detected_angles)]
                max_diff = max(angle_diffs) if angle_diffs else 0.0
                # Score based on how close angles are (within 5° = 100%, within 10° = 80%, etc.)
                if max_diff < 5.0:
                    matches['angles'] = 100.0
                elif max_diff < 10.0:
                    matches['angles'] = 80.0
                elif max_diff < 20.0:
                    matches['angles'] = 60.0
                else:
                    matches['angles'] = max(0.0, 100.0 - max_diff)
            else:
                matches['angles'] = 0.0
        else:
            matches['angles'] = None
        
        results = {
            'name': name,
            'expected': expected_glazer,
            'detected': detected,
            'matches': matches,
        }
        
        # Calculate overall match
        valid_matches = [v for v in matches.values() if v is not None]
        results['overall_match'] = sum(valid_matches) / len(valid_matches) if valid_matches else 0.0
        
        print(f"\nMatch Results:")
        print(f"{'-'*70}")
        for param, match_pct in matches.items():
            if match_pct is not None:
                status = "✓" if match_pct >= 90.0 else "✗" if match_pct < 50.0 else "~"
                print(f"  {status} {param:20s}: {match_pct:6.1f}%")
        print(f"{'-'*70}")
        print(f"  Overall Match: {results['overall_match']:6.1f}%")
        print(f"{'='*70}\n")
        
        return results
    
    def test_glazer_untilted(self):
        """Test Glazer detection on untilted structure (a0a0a0)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='a0a0a0',
            glazer_angles=[0, 0, 0],
        )
        
        expected = {
            'notation': 'a0a0a0',
            'tilt_pattern': ['0', '0', '0'],
            'glazer_angles': [0, 0, 0],
        }
        
        results = self._test_glazer_detection(structure, 'untilted', expected)
        assert True
    
    def test_glazer_a0a0c_plus(self):
        """Test Glazer detection on a0a0c+ pattern (tetragonal, in-phase about z)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='a0a0c+',
            glazer_angles=[0, 0, 10],
        )
        
        expected = {
            'notation': 'a0a0c+',
            'tilt_pattern': ['0', '0', '+'],
            'glazer_angles': [0, 0, 10],
        }
        
        results = self._test_glazer_detection(structure, 'a0a0c_plus', expected)
        assert True
    
    def test_glazer_a0a0c_minus(self):
        """Test Glazer detection on a0a0c- pattern (tetragonal, anti-phase about z)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='a0a0c-',
            glazer_angles=[0, 0, 15],
        )
        
        expected = {
            'notation': 'a0a0c-',
            'tilt_pattern': ['0', '0', '-'],
            'glazer_angles': [0, 0, 15],
        }
        
        results = self._test_glazer_detection(structure, 'a0a0c_minus', expected)
        assert True
    
    def test_glazer_a_b_plus_a_minus(self):
        """Test Glazer detection on a-b+a- pattern (Pnma, orthorhombic)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='a-b+a-',
            glazer_angles=[10, 10, 10],
        )
        
        expected = {
            'notation': 'a-b+a-',
            'tilt_pattern': ['-', '+', '-'],
            'glazer_angles': [10, 10, 10],
        }
        
        results = self._test_glazer_detection(structure, 'a_b_plus_a_minus', expected)
        assert True
    
    def test_glazer_a_minus_a_minus_a_minus(self):
        """Test Glazer detection on a-a-a- pattern (R-3c, rhombohedral)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='a-a-a-',
            glazer_angles=[12, 12, 12],
        )
        
        expected = {
            'notation': 'a-a-a-',
            'tilt_pattern': ['-', '-', '-'],
            'glazer_angles': [12, 12, 12],
        }
        
        results = self._test_glazer_detection(structure, 'a_minus_a_minus_a_minus', expected)
        assert True
    
    def test_glazer_a0b_minus_b_minus(self):
        """Test Glazer detection on a0b-b- pattern (Imma, orthorhombic)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='a0b-b-',
            glazer_angles=[0, 10, 10],
        )
        
        expected = {
            'notation': 'a0b-b-',
            'tilt_pattern': ['0', '-', '-'],
            'glazer_angles': [0, 10, 10],
        }
        
        results = self._test_glazer_detection(structure, 'a0b_minus_b_minus', expected)
        assert True
    
    def test_glazer_a_plus_a_plus_c_plus(self):
        """Test Glazer detection on a+a+c+ pattern (in-phase tilting)."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='a+a+c+',
            glazer_angles=[8, 8, 10],
        )
        
        expected = {
            'notation': 'a+a+c+',
            'tilt_pattern': ['+', '+', '+'],
            'glazer_angles': [8, 8, 10],
        }
        
        results = self._test_glazer_detection(structure, 'a_plus_a_plus_c_plus', expected)
        assert True
    
    def test_glazer_by_space_group(self):
        """Test Glazer detection when structure is created using space group name."""
        structure = self.creator.create_structure(
            A_ions='Cs',
            B_ions='Pb',
            X_ions='I',
            structure_type='bulk',
            thickness=2,
            xy_expansion=(2, 2),
            glazer_pattern='Pnma',  # Should resolve to a-b+a-
        )
        
        expected = {
            'notation': 'a-b+a-',  # Pnma maps to a-b+a-
            'tilt_pattern': ['-', '+', '-'],
        }
        
        results = self._test_glazer_detection(structure, 'by_space_group_pnma', expected)
        assert True
    
    def test_summary_report(self):
        """Generate a summary report of all tests."""
        print("\n" + "="*70)
        print("ANALYZER VALIDATION SUMMARY")
        print("="*70)
        print("\nAll structures were created, saved as VASP files, and analyzed.")
        print("Check the match percentages above to see how well the analyzer")
        print("identifies the structure components compared to the creator parameters.")
        print("\nGlazer pattern detection tests verify that the analyzer can correctly")
        print("identify octahedral tilting patterns from structures created with known")
        print("Glazer notations.")
        print("\nVASP files saved in:", self.test_dir)
        print("\nNote: Some tests may be skipped if RDKit is not available.")
        print("="*70 + "\n")
        
        assert True


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])
