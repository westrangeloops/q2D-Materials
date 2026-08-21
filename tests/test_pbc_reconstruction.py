"""
Test PBC-aware molecule reconstruction.

This test validates that molecules are correctly reconstructed with proper
PBC coordinates when they span cell boundaries.
"""

import pytest
import numpy as np
import networkx as nx

from q2D_Materials.utils.molecules.pbc_reconstruction import (
    reconstruct_molecule_pbc,
    reconstruct_molecule_from_nh3,
)


class TestPBCReconstruction:
    """Test PBC-aware molecule reconstruction."""
    
    def test_simple_molecule_no_pbc(self):
        """Test reconstruction of simple molecule without PBC wrapping."""
        # Create a simple linear molecule: C-C-C
        molecule_indices = [0, 1, 2]
        atom_positions = np.array([
            [0.0, 0.0, 0.0],  # C1
            [1.5, 0.0, 0.0],  # C2
            [3.0, 0.0, 0.0],  # C3
        ])
        atom_symbols = ['C', 'C', 'C']
        cell = np.eye(3) * 10.0  # 10 Å cell
        anchor_position = np.array([1.5, 0.0, 0.0])  # Center atom
        
        # Create graph with bonds
        graph = nx.Graph()
        graph.add_node('atom_0', vasp_index=0, symbol='C')
        graph.add_node('atom_1', vasp_index=1, symbol='C')
        graph.add_node('atom_2', vasp_index=2, symbol='C')
        graph.add_edge('atom_0', 'atom_1', edge_type='bonded_to', distance=1.5)
        graph.add_edge('atom_1', 'atom_2', edge_type='bonded_to', distance=1.5)
        
        # Reconstruct
        result = reconstruct_molecule_pbc(
            molecule_indices,
            anchor_position,
            graph,
            atom_positions,
            atom_symbols,
            cell,
            pbc=True
        )
        
        # Check all atoms are placed
        assert len(result) == 3
        assert all(idx in result for idx in molecule_indices)
        
        # Check positions are reasonable (within 5 Å of anchor)
        for idx, (pos, img_label) in result.items():
            dist = np.linalg.norm(pos - anchor_position)
            assert dist < 5.0, f"Atom {idx} too far from anchor: {dist:.2f} Å"
            assert img_label == (0, 0, 0), "Should be in original cell"
    
    def test_molecule_crossing_boundary(self):
        """Test reconstruction when molecule crosses PBC boundary."""
        # Create molecule that wraps around: atom at [9.0, 0, 0] in 10 Å cell
        # should be placed near [0.0, 0, 0] when anchor is at [0.5, 0, 0]
        molecule_indices = [0, 1]
        atom_positions = np.array([
            [0.5, 0.0, 0.0],  # Atom 0 (anchor)
            [9.0, 0.0, 0.0],  # Atom 1 (wraps around)
        ])
        atom_symbols = ['C', 'C']
        cell = np.eye(3) * 10.0
        anchor_position = np.array([0.5, 0.0, 0.0])
        
        # Create graph with bond
        graph = nx.Graph()
        graph.add_node('atom_0', vasp_index=0, symbol='C')
        graph.add_node('atom_1', vasp_index=1, symbol='C')
        graph.add_edge('atom_0', 'atom_1', edge_type='bonded_to', distance=1.5)
        
        # Reconstruct
        result = reconstruct_molecule_pbc(
            molecule_indices,
            anchor_position,
            graph,
            atom_positions,
            atom_symbols,
            cell,
            pbc=True
        )
        
        # Check atom 1 is placed correctly (should be at [-1.0, 0, 0] or similar)
        pos_1, img_label_1 = result[1]
        dist_from_anchor = np.linalg.norm(pos_1 - anchor_position)
        
        # Bond length should be ~1.5 Å (not 8.5 Å)
        assert dist_from_anchor < 2.0, f"Bond length too long: {dist_from_anchor:.2f} Å"
        assert img_label_1 != (0, 0, 0), "Atom 1 should be in different PBC image"
    
    def test_nh3_reconstruction(self):
        """Test reconstruction starting from NH3 group."""
        # Create NH3 group: N at center, 3H around it
        molecule_indices = [0, 1, 2, 3]  # N, H1, H2, H3
        atom_positions = np.array([
            [5.0, 5.0, 5.0],  # N
            [5.0, 6.0, 5.0],  # H1
            [5.7, 4.5, 5.0],  # H2
            [4.3, 4.5, 5.0],  # H3
        ])
        atom_symbols = ['N', 'H', 'H', 'H']
        cell = np.eye(3) * 10.0
        nh3_center = np.array([5.0, 5.0, 5.0])  # Center of NH3
        
        # Create graph
        graph = nx.Graph()
        for i, idx in enumerate(molecule_indices):
            graph.add_node(f'atom_{idx}', vasp_index=idx, symbol=atom_symbols[i])
        
        # N-H bonds
        for h_idx in [1, 2, 3]:
            graph.add_edge('atom_0', f'atom_{h_idx}', edge_type='bonded_to', distance=1.0)
        
        # Reconstruct from NH3
        result = reconstruct_molecule_from_nh3(
            molecule_indices,
            nh3_center,
            graph,
            atom_positions,
            atom_symbols,
            cell,
            pbc=True
        )
        
        # Check all atoms placed
        assert len(result) == 4
        
        # Check NH3 atoms are near center
        for idx in molecule_indices:
            pos, _ = result[idx]
            dist = np.linalg.norm(pos - nh3_center)
            assert dist < 2.0, f"NH3 atom {idx} too far from center: {dist:.2f} Å"
    
    def test_bond_lengths_preserved(self):
        """Test that bond lengths are preserved in reconstruction."""
        # Create molecule with known bond lengths
        molecule_indices = [0, 1, 2]
        atom_positions = np.array([
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],  # 1.5 Å bond
            [3.0, 0.0, 0.0],  # 1.5 Å bond
        ])
        atom_symbols = ['C', 'C', 'C']
        cell = np.eye(3) * 10.0
        anchor_position = np.array([1.5, 0.0, 0.0])
        
        # Create graph
        graph = nx.Graph()
        for i, idx in enumerate(molecule_indices):
            graph.add_node(f'atom_{idx}', vasp_index=idx, symbol=atom_symbols[i])
        graph.add_edge('atom_0', 'atom_1', edge_type='bonded_to', distance=1.5)
        graph.add_edge('atom_1', 'atom_2', edge_type='bonded_to', distance=1.5)
        
        # Reconstruct
        result = reconstruct_molecule_pbc(
            molecule_indices,
            anchor_position,
            graph,
            atom_positions,
            atom_symbols,
            cell,
            pbc=True
        )
        
        # Check bond lengths
        pos_0, _ = result[0]
        pos_1, _ = result[1]
        pos_2, _ = result[2]
        
        bond_01 = np.linalg.norm(pos_1 - pos_0)
        bond_12 = np.linalg.norm(pos_2 - pos_1)
        
        # Bond lengths should be ~1.5 Å (within 0.1 Å tolerance)
        assert abs(bond_01 - 1.5) < 0.1, f"Bond 0-1 length incorrect: {bond_01:.3f} Å"
        assert abs(bond_12 - 1.5) < 0.1, f"Bond 1-2 length incorrect: {bond_12:.3f} Å"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

