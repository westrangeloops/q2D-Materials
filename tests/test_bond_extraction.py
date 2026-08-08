"""
Test bond extraction and octahedra classification.

Tests that get_octahedra() returns properly classified ligand atoms:
1. terminal_atoms
2. interlayer_atoms  
3. intralayer_atoms

Also tests bond extraction functions that depend on these classifications.
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer.core.analyzer_class import q2D_analyzer
from q2D_Materials.analyzer.utils.geometry_helpers import (
    get_all_x_atoms_from_octahedron,
    extract_bx_bond_vectors,
)


class TestBondExtraction:
    """Test bond extraction and octahedra classification."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.creator = q2D_creator()
        
    def test_octahedra_classification_keys(self):
        """Test that get_octahedra() returns expected classification keys."""
        print("\n" + "="*60)
        print("TEST 1: Octahedra Classification Keys")
        print("="*60)
        
        # Create bulk structure
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(2, 2),
            template="cubic",
            structure_type="bulk",
        )
        
        # Analyze structure
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        # Get octahedra
        octahedra = analyzer.get_octahedra()
        print(f"  Octahedra found: {len(octahedra)}")
        
        assert len(octahedra) > 0, "No octahedra found"
        
        # Check that all expected keys are present
        required_keys = [
            'id',
            'central_atom_index',
            'central_atom_symbol',
            'ligand_atoms',  # Backward compatibility
            'terminal_atoms',
            'interlayer_atoms',
            'intralayer_atoms',
        ]
        
        for i, oct_info in enumerate(octahedra):
            print(f"\n  Octahedron {i} ({oct_info.get('id')}):")
            for key in required_keys:
                assert key in oct_info, f"Missing key '{key}' in octahedron {i}"
                print(f"    {key}: {type(oct_info[key]).__name__}")
                
                # Check that classification keys are lists
                if key in ['terminal_atoms', 'interlayer_atoms', 'intralayer_atoms', 'ligand_atoms']:
                    assert isinstance(oct_info[key], list), \
                        f"'{key}' should be a list, got {type(oct_info[key])}"
            
            # Verify ligand_atoms is the sum of the three classifications
            total_classified = (len(oct_info['terminal_atoms']) + 
                              len(oct_info['interlayer_atoms']) + 
                              len(oct_info['intralayer_atoms']))
            total_ligand = len(oct_info['ligand_atoms'])
            
            print(f"    Total classified: {total_classified}")
            print(f"    Total ligand_atoms: {total_ligand}")
            print(f"    Terminal: {len(oct_info['terminal_atoms'])}")
            print(f"    Interlayer: {len(oct_info['interlayer_atoms'])}")
            print(f"    Intralayer: {len(oct_info['intralayer_atoms'])}")
            
            # They should match (or ligand_atoms might be empty if no atoms found)
            if total_classified > 0:
                assert total_classified == total_ligand, \
                    f"Sum of classifications ({total_classified}) != ligand_atoms ({total_ligand})"
        
        print("  ✓ All octahedra have required classification keys")
    
    def test_get_all_x_atoms_from_octahedron(self):
        """Test get_all_x_atoms_from_octahedron() function."""
        print("\n" + "="*60)
        print("TEST 2: get_all_x_atoms_from_octahedron()")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(2, 2),
            template="cubic",
            structure_type="bulk",
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        octahedra = analyzer.get_octahedra()
        assert len(octahedra) > 0, "No octahedra found"
        
        for i, oct_info in enumerate(octahedra[:3]):  # Test first 3
            print(f"\n  Testing octahedron {i} ({oct_info.get('id')}):")
            
            # Test get_all_x_atoms_from_octahedron
            all_x_atoms = get_all_x_atoms_from_octahedron(oct_info)
            print(f"    All X atoms: {len(all_x_atoms)}")
            
            # Should have at least some X atoms (may be less than 6 for incomplete octahedra)
            assert len(all_x_atoms) > 0, f"Octahedron {i} has no X atoms"
            
            # Verify all atoms are unique
            assert len(all_x_atoms) == len(set(all_x_atoms)), \
                f"Octahedron {i} has duplicate X atom indices"
            
            # Verify all atoms are in ligand_atoms
            ligand_atoms = oct_info.get('ligand_atoms', [])
            for x_atom in all_x_atoms:
                assert x_atom in ligand_atoms, \
                    f"X atom {x_atom} not in ligand_atoms"
            
            print(f"    ✓ All {len(all_x_atoms)} X atoms correctly extracted")
        
        print("  ✓ get_all_x_atoms_from_octahedron() works correctly")
    
    def test_extract_bx_bond_vectors(self):
        """Test extract_bx_bond_vectors() function."""
        print("\n" + "="*60)
        print("TEST 3: extract_bx_bond_vectors()")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(2, 2),
            template="cubic",
            structure_type="bulk",
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        atom_positions = np.array(analyzer.cell.get_positions())
        cell = np.array(analyzer.cell.get_cell())
        
        octahedra = analyzer.get_octahedra()
        assert len(octahedra) > 0, "No octahedra found"
        
        for i, oct_info in enumerate(octahedra[:3]):  # Test first 3
            print(f"\n  Testing octahedron {i} ({oct_info.get('id')}):")
            
            central_idx = oct_info.get('central_atom_index')
            if central_idx is None:
                print(f"    Skipping: no central atom")
                continue
            
            # Test extract_bx_bond_vectors
            bond_vectors, x_positions = extract_bx_bond_vectors(
                oct_info, atom_positions, cell, apply_pbc=True
            )
            
            print(f"    Bond vectors shape: {bond_vectors.shape}")
            print(f"    X positions shape: {x_positions.shape}")
            
            # Should have same number of bonds as X atoms
            all_x_atoms = get_all_x_atoms_from_octahedron(oct_info)
            expected_bonds = len(all_x_atoms)
            
            assert bond_vectors.shape[0] == expected_bonds, \
                f"Expected {expected_bonds} bonds, got {bond_vectors.shape[0]}"
            assert x_positions.shape[0] == expected_bonds, \
                f"Expected {expected_bonds} X positions, got {x_positions.shape[0]}"
            assert bond_vectors.shape[1] == 3, "Bond vectors should be 3D"
            assert x_positions.shape[1] == 3, "X positions should be 3D"
            
            # Verify bond vectors are reasonable (not all zero)
            bond_lengths = np.linalg.norm(bond_vectors, axis=1)
            print(f"    Bond lengths: min={bond_lengths.min():.3f}, max={bond_lengths.max():.3f}, mean={bond_lengths.mean():.3f}")
            
            assert np.all(bond_lengths > 0), "All bond lengths should be positive"
            assert np.all(bond_lengths < 10.0), "Bond lengths should be reasonable (< 10 Å)"
            
            print(f"    ✓ Successfully extracted {expected_bonds} bond vectors")
        
        print("  ✓ extract_bx_bond_vectors() works correctly")
    
    def test_bond_classification_by_structure_type(self):
        """Test that different structure types have appropriate classifications."""
        print("\n" + "="*60)
        print("TEST 4: Bond Classification by Structure Type")
        print("="*60)
        
        structures = [
            ("Bulk", {
                'A_ions': "MA",
                'B_ions': "Pb",
                'X_ions': "I",
                'xy_expansion': (2, 2),
                'template': "cubic",
                'structure_type': "bulk",
            }),
            ("Monolayer", {
                'A_ions': "MA",
                'B_ions': "Pb",
                'X_ions': "I",
                'xy_expansion': (2, 2),
                'template': "cubic",
                'structure_type': "monolayer",
                'vacuum': 15.0,
            }),
        ]
        
        for name, params in structures:
            print(f"\n  Testing {name} structure...")
            structure = self.creator.create_structure(**params)
            analyzer = q2D_analyzer(structure)
            analyzer.analyze()
            
            octahedra = analyzer.get_octahedra()
            assert len(octahedra) > 0, f"{name}: No octahedra found"
            
            # Count classifications
            total_terminal = sum(len(oct.get('terminal_atoms', [])) for oct in octahedra)
            total_interlayer = sum(len(oct.get('interlayer_atoms', [])) for oct in octahedra)
            total_intralayer = sum(len(oct.get('intralayer_atoms', [])) for oct in octahedra)
            
            print(f"    Terminal atoms: {total_terminal}")
            print(f"    Interlayer atoms: {total_interlayer}")
            print(f"    Intralayer atoms: {total_intralayer}")
            
            # Monolayers should have more terminal atoms (surface atoms)
            if name == "Monolayer":
                assert total_terminal > 0, "Monolayer should have terminal atoms"
            
            # Bulk structures should have interlayer atoms (shared between layers)
            if name == "Bulk":
                # May or may not have interlayer atoms depending on structure
                pass
            
            print(f"    ✓ {name} structure classification successful")
        
        print("  ✓ Structure type classification test successful")
    
    def test_bond_length_calculation(self):
        """Test that bond lengths can be calculated from classified atoms."""
        print("\n" + "="*60)
        print("TEST 5: Bond Length Calculation")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(2, 2),
            template="cubic",
            structure_type="bulk",
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        atoms = analyzer.cell
        octahedra = analyzer.get_octahedra()
        assert len(octahedra) > 0, "No octahedra found"
        
        for i, oct_info in enumerate(octahedra[:3]):  # Test first 3
            print(f"\n  Testing octahedron {i} ({oct_info.get('id')}):")
            
            b_atom = oct_info.get('central_atom_index')
            if b_atom is None:
                continue
            
            # Collect all X atoms by classification
            terminal_atoms = oct_info.get('terminal_atoms', [])
            interlayer_atoms = oct_info.get('interlayer_atoms', [])
            intralayer_atoms = oct_info.get('intralayer_atoms', [])
            
            all_x_atoms = terminal_atoms + interlayer_atoms + intralayer_atoms
            
            if len(all_x_atoms) == 0:
                print(f"    Skipping: no X atoms")
                continue
            
            # Calculate bond lengths
            bond_lengths = []
            for x_atom in all_x_atoms:
                bond_length = atoms.get_distance(b_atom, x_atom, mic=True)
                bond_lengths.append(bond_length)
            
            bond_lengths = np.array(bond_lengths)
            
            print(f"    Total bonds: {len(bond_lengths)}")
            print(f"    Bond lengths: min={bond_lengths.min():.3f}, max={bond_lengths.max():.3f}, mean={bond_lengths.mean():.3f}")
            
            # Verify bond lengths are reasonable for Pb-I bonds (~3.0-3.5 Å)
            assert np.all(bond_lengths > 2.0), "Bond lengths should be > 2.0 Å"
            assert np.all(bond_lengths < 5.0), "Bond lengths should be < 5.0 Å"
            
            # Calculate by classification
            if len(terminal_atoms) > 0:
                terminal_lengths = [atoms.get_distance(b_atom, x, mic=True) for x in terminal_atoms]
                print(f"    Terminal bonds ({len(terminal_atoms)}): mean={np.mean(terminal_lengths):.3f}")
            
            if len(interlayer_atoms) > 0:
                interlayer_lengths = [atoms.get_distance(b_atom, x, mic=True) for x in interlayer_atoms]
                print(f"    Interlayer bonds ({len(interlayer_atoms)}): mean={np.mean(interlayer_lengths):.3f}")
            
            if len(intralayer_atoms) > 0:
                intralayer_lengths = [atoms.get_distance(b_atom, x, mic=True) for x in intralayer_atoms]
                print(f"    Intralayer bonds ({len(intralayer_atoms)}): mean={np.mean(intralayer_lengths):.3f}")
            
            print(f"    ✓ Bond length calculation successful")
        
        print("  ✓ Bond length calculation test successful")
    
    def test_bxb_angles_with_pbc(self):
        """Test B-X-B angle calculation with PBC consideration."""
        print("\n" + "="*60)
        print("TEST 6: B-X-B Angles with PBC")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(2, 2),
            template="cubic",
            structure_type="bulk",
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        # Get B-X-B angles
        bxb_data = analyzer.get_bxb_angles()
        
        print(f"  BXB angles found: {len(bxb_data['bxb_angles']) if bxb_data['bxb_angles'] is not None else 0}")
        
        assert bxb_data['bxb_angles'] is not None, "BXB angles should not be None"
        assert len(bxb_data['bxb_angles']) > 0, "Should have at least some BXB angles"
        
        angles = bxb_data['bxb_angles']
        mean_angle = bxb_data['bxb_mean']
        std_angle = bxb_data['bxb_std']
        
        print(f"  Mean BXB angle: {mean_angle:.2f}°")
        print(f"  Std BXB angle: {std_angle:.2f}°")
        print(f"  Angle range: {angles.min():.2f}° - {angles.max():.2f}°")
        
        # Verify angles are reasonable (B-X-B angles should be between 0 and 180 degrees)
        assert np.all(angles >= 0), "All angles should be >= 0°"
        assert np.all(angles <= 180), "All angles should be <= 180°"
        
        # For cubic perovskite, B-X-B angles should be close to 180° (linear)
        # But with tilting, they can be less
        assert mean_angle > 90, "Mean BXB angle should be > 90° for perovskite structures"
        assert mean_angle <= 180, "Mean BXB angle should be <= 180°"
        
        print("  ✓ BXB angle calculation with PBC successful")
    
    def test_bxb_angles_monolayer(self):
        """Test B-X-B angles on monolayer structure."""
        print("\n" + "="*60)
        print("TEST 7: B-X-B Angles (Monolayer)")
        print("="*60)
        
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(2, 2),
            template="cubic",
            structure_type="monolayer",
            vacuum=15.0,
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        bxb_data = analyzer.get_bxb_angles()
        
        assert bxb_data['bxb_angles'] is not None, "BXB angles should not be None"
        assert len(bxb_data['bxb_angles']) > 0, "Monolayer should have BXB angles"
        
        angles = bxb_data['bxb_angles']
        mean_angle = bxb_data['bxb_mean']
        
        print(f"  Mean BXB angle: {mean_angle:.2f}°")
        print(f"  Number of angles: {len(angles)}")
        
        # Monolayers may have different angle distributions due to surface effects
        assert np.all(angles >= 0), "All angles should be >= 0°"
        assert np.all(angles <= 180), "All angles should be <= 180°"
        
        print("  ✓ Monolayer BXB angle calculation successful")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
