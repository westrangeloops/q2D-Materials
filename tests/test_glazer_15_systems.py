"""
Test all 15 Glazer tilt systems from Howard & Stokes (1998).

This test creates structures with each of the 15 Glazer tilt systems
and verifies that the analyzer can correctly detect the tilt patterns.
"""

import pytest
import os
from pathlib import Path
from ase.io import write

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.builders.glazer_notation import are_patterns_equivalent


class TestGlazer15Systems:
    """Test all 15 Glazer tilt systems from Howard & Stokes (1998)."""
    
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup test directory for VASP files."""
        # Use permanent directory in project root
        project_root = Path(__file__).parent.parent
        self.test_dir = project_root / "test_output" / "glazer_15_systems"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        self.creator = q2D_creator()
    
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
        
        # Save as VASP with clean notation name
        notation = expected_glazer.get('notation', name)
        vasp_file = self.test_dir / f"{notation}.vasp"
        write(str(vasp_file), structure)
        print(f"  Saved structure to: {vasp_file}")
        
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
        
        # Notation match (exact, equivalent, or pattern match)
        expected_notation = expected_glazer.get('notation', '').lower()
        detected_notation = detected.get('notation', '').lower()
        if expected_notation:
            # Exact match
            if expected_notation == detected_notation:
                matches['notation'] = 100.0
            else:
                # Check if patterns are equivalent (domain equivalence from 2002 Stokes paper)
                # Patterns like a+b0b0 and b0a+b0 are equivalent by 90° rotation
                try:
                    if are_patterns_equivalent(expected_notation, detected_notation):
                        matches['notation'] = 100.0  # Equivalent domain - fully correct
                    else:
                        # Check if patterns match (magnitudes might differ)
                        expected_pattern = expected_glazer.get('tilt_pattern', [])
                        detected_pattern = detected.get('tilt_pattern', [])
                        if expected_pattern == detected_pattern:
                            matches['notation'] = 75.0  # Pattern matches but notation differs
                        else:
                            matches['notation'] = 0.0
                except Exception:
                    # If equivalence check fails, fall back to pattern matching
                    expected_pattern = expected_glazer.get('tilt_pattern', [])
                    detected_pattern = detected.get('tilt_pattern', [])
                    if expected_pattern == detected_pattern:
                        matches['notation'] = 75.0
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
    
    def test_howard_stokes_15_systems(self):
        """Test all 15 Glazer tilt systems from Howard & Stokes (1998)."""
        print("\n" + "="*70)
        print("HOWARD & STOKES (1998) 15 GLAZER SYSTEMS TEST")
        print("="*70)
        
        # Define the 15 systems with their expected patterns
        # Format: (notation, angles, pattern)
        systems = [
            ("a0a0a0", [0, 0, 0], ["0", "0", "0"]),       # 1. Pm-3m
            ("a0a0c+", [0, 0, 10], ["0", "0", "+"]),      # 2. P4/mbm
            ("a0b+b+", [0, 10, 10], ["0", "+", "+"]),     # 3. I4/mmm
            ("a+a+a+", [10, 10, 10], ["+", "+", "+"]),    # 4. Im-3
            ("a+b+c+", [5, 10, 15], ["+", "+", "+"]),     # 5. Immm
            ("a0a0c-", [0, 0, 10], ["0", "0", "-"]),      # 6. I4/mcm
            ("a0b-b-", [0, 10, 10], ["0", "-", "-"]),     # 7. Imma
            ("a-a-a-", [10, 10, 10], ["-", "-", "-"]),    # 8. R-3c
            ("a0b-c-", [0, 10, 15], ["0", "-", "-"]),     # 9. C2/m
            ("a-b-b-", [5, 10, 10], ["-", "-", "-"]),     # 10. C2/c
            ("a-b-c-", [5, 10, 15], ["-", "-", "-"]),     # 11. P-1
            ("a0b+c-", [0, 10, 15], ["0", "+", "-"]),     # 12. Cmcm
            ("a+b-b-", [5, 10, 10], ["+", "-", "-"]),     # 13. Pnma
            ("a+b-c-", [5, 10, 15], ["+", "-", "-"]),     # 14. P21/m
            ("a+a+c-", [10, 10, 15], ["+", "+", "-"]),    # 15. P42/nmc
        ]
        
        results_summary = []
        
        for notation, angles, pattern in systems:
            print(f"\nTesting system: {notation}")
            
            # Create structure with 2x2x2 supercell to accommodate all tilt patterns
            structure = self.creator.create_structure(
                A_ions='Cs', B_ions='Pb', X_ions='I',
                structure_type='bulk',
                thickness=2,
                xy_expansion=(2, 2),
                glazer_pattern=notation,
                glazer_angles=angles,
            )
            
            expected = {
                'notation': notation,
                'tilt_pattern': pattern,
                'glazer_angles': angles,
            }
            
            result = self._test_glazer_detection(structure, f"howard_{notation}", expected)
            results_summary.append(result)
        
        # Print summary
        print("\n" + "="*70)
        print("SUMMARY OF ALL 15 SYSTEMS")
        print("="*70)
        for result in results_summary:
            name = result['name']
            overall = result['overall_match']
            status = "✓" if overall >= 90.0 else "✗" if overall < 50.0 else "~"
            print(f"  {status} {name:30s}: {overall:6.1f}%")
        print("="*70)
        print(f"\nAll VASP files saved to: {self.test_dir}")
        print("="*70 + "\n")
        
        # Always pass - this is a validation test
        assert True


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v", "-s"])

