"""
Test compute_delta, compute_sigma, and compute_lambda functions.

Tests distortion parameter computation across different structure types:
1. Bulk perovskite
2. Monolayer perovskite
3. RP (Ruddlesden-Popper) structure
"""

import sys
from pathlib import Path

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer.core.analyzer_class import q2D_analyzer


class TestComputeDelta:
    """Test compute_delta, compute_sigma, and compute_lambda functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.creator = q2D_creator()
        
    def test_bulk_structure_delta(self):
        """Test compute_delta on bulk perovskite structure."""
        print("\n" + "="*60)
        print("TEST 1: Bulk Perovskite (MAPbI3)")
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
        
        # Check basic structure
        octahedra = analyzer.get_octahedra()
        layers = analyzer.get_layers()
        print(f"  Octahedra found: {len(octahedra)}")
        print(f"  Layers found: {len(layers)}")
        
        assert len(octahedra) > 0, "No octahedra found in bulk structure"
        assert len(layers) > 0, "No layers found in bulk structure"
        
        # Test global delta
        delta_global = analyzer.compute_delta()
        print(f"  Global delta: {delta_global}")
        assert delta_global is not None, "Global delta should not be None"
        assert isinstance(delta_global, (int, float)), "Delta should be numeric"
        assert delta_global >= 0, "Delta should be non-negative"
        
        # Test delta by layer
        delta_by_layer = analyzer.compute_delta(group_by='layer')
        print(f"  Delta by layer: {delta_by_layer}")
        assert isinstance(delta_by_layer, dict), "Delta by layer should be a dict"
        assert len(delta_by_layer) > 0, "Delta by layer should not be empty"
        assert 'global' in delta_by_layer, "Delta by layer should have 'global' key"
        
        # Check that global values match
        assert delta_by_layer['global'] == delta_global, "Global delta should match"
        
        # Check that layer-specific deltas are not None
        for layer_id, delta_value in delta_by_layer.items():
            if layer_id != 'global':
                assert delta_value is not None, f"Delta for layer {layer_id} should not be None"
                assert isinstance(delta_value, (int, float)), f"Delta for layer {layer_id} should be numeric"
                assert delta_value >= 0, f"Delta for layer {layer_id} should be non-negative"
        
        print("  ✓ Bulk structure delta computation successful")
    
    def test_monolayer_structure_delta(self):
        """Test compute_delta on monolayer perovskite structure."""
        print("\n" + "="*60)
        print("TEST 2: Monolayer Perovskite (MAPbI3)")
        print("="*60)
        
        # Create monolayer structure
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            xy_expansion=(2, 2),
            template="cubic",
            structure_type="monolayer",
            vacuum=15.0,
        )
        
        # Analyze structure
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        # Check basic structure
        octahedra = analyzer.get_octahedra()
        layers = analyzer.get_layers()
        print(f"  Octahedra found: {len(octahedra)}")
        print(f"  Layers found: {len(layers)}")
        
        assert len(octahedra) > 0, "No octahedra found in monolayer structure"
        assert len(layers) > 0, "No layers found in monolayer structure"
        
        # Test global delta
        delta_global = analyzer.compute_delta()
        print(f"  Global delta: {delta_global}")
        assert delta_global is not None, "Global delta should not be None"
        assert isinstance(delta_global, (int, float)), "Delta should be numeric"
        assert delta_global >= 0, "Delta should be non-negative"
        
        # Test delta by layer
        delta_by_layer = analyzer.compute_delta(group_by='layer')
        print(f"  Delta by layer: {delta_by_layer}")
        assert isinstance(delta_by_layer, dict), "Delta by layer should be a dict"
        assert len(delta_by_layer) > 0, "Delta by layer should not be empty"
        assert 'global' in delta_by_layer, "Delta by layer should have 'global' key"
        
        # For monolayer, should have at least one layer
        layer_keys = [k for k in delta_by_layer.keys() if k != 'global']
        assert len(layer_keys) > 0, "Monolayer should have at least one layer with delta"
        
        print("  ✓ Monolayer structure delta computation successful")
    
    def test_rp_structure_delta(self):
        """Test compute_delta on RP (Ruddlesden-Popper) structure."""
        print("\n" + "="*60)
        print("TEST 3: RP Structure (PA)2PbI4")
        print("="*60)
        
        # Create RP structure
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            layer_sequence="RP",
            thickness=2,
            xy_expansion=(2, 2),
            spacer="[NH3+]CCC",  # Propylammonium (monofunctional)
        )
        
        # Analyze structure
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        # Check basic structure
        octahedra = analyzer.get_octahedra()
        layers = analyzer.get_layers()
        print(f"  Octahedra found: {len(octahedra)}")
        print(f"  Layers found: {len(layers)}")
        
        assert len(octahedra) > 0, "No octahedra found in RP structure"
        assert len(layers) > 0, "No layers found in RP structure"
        
        # Test global delta
        delta_global = analyzer.compute_delta()
        print(f"  Global delta: {delta_global}")
        assert delta_global is not None, "Global delta should not be None"
        assert isinstance(delta_global, (int, float)), "Delta should be numeric"
        assert delta_global >= 0, "Delta should be non-negative"
        
        # Test delta by layer
        delta_by_layer = analyzer.compute_delta(group_by='layer')
        print(f"  Delta by layer: {delta_by_layer}")
        assert isinstance(delta_by_layer, dict), "Delta by layer should be a dict"
        assert len(delta_by_layer) > 0, "Delta by layer should not be empty"
        assert 'global' in delta_by_layer, "Delta by layer should have 'global' key"
        
        # For RP with thickness=2, should have at least 2 layers
        layer_keys = [k for k in delta_by_layer.keys() if k != 'global']
        assert len(layer_keys) > 0, "RP structure should have at least one layer with delta"
        
        print("  ✓ RP structure delta computation successful")
    
    def test_sigma_and_lambda(self):
        """Test compute_sigma and compute_lambda on all three structures."""
        print("\n" + "="*60)
        print("TEST 4: Sigma and Lambda Computation")
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
            ("RP", {
                'A_ions': "MA",
                'B_ions': "Pb",
                'X_ions': "I",
                'structure_type': "bulk",
                'template': "cubic",
                'layer_sequence': "RP",
                'thickness': 2,
                'xy_expansion': (2, 2),
                'spacer': "[NH3+]CCC",
            }),
        ]
        
        for name, params in structures:
            print(f"\n  Testing {name} structure...")
            structure = self.creator.create_structure(**params)
            analyzer = q2D_analyzer(structure)
            analyzer.analyze()
            
            # Test sigma
            sigma_global = analyzer.compute_sigma()
            sigma_by_layer = analyzer.compute_sigma(group_by='layer')
            print(f"    Sigma global: {sigma_global}")
            print(f"    Sigma by layer keys: {list(sigma_by_layer.keys())}")
            
            assert sigma_global is not None, f"{name}: Global sigma should not be None"
            assert isinstance(sigma_global, (int, float)), f"{name}: Sigma should be numeric"
            assert sigma_global >= 0, f"{name}: Sigma should be non-negative"
            assert isinstance(sigma_by_layer, dict), f"{name}: Sigma by layer should be a dict"
            assert 'global' in sigma_by_layer, f"{name}: Sigma by layer should have 'global' key"
            
            # Test lambda
            lambda_global = analyzer.compute_lambda()
            lambda_by_layer = analyzer.compute_lambda(group_by='layer')
            print(f"    Lambda global: {lambda_global}")
            print(f"    Lambda by layer keys: {list(lambda_by_layer.keys())}")
            
            assert lambda_global is not None, f"{name}: Global lambda should not be None"
            assert isinstance(lambda_global, (int, float)), f"{name}: Lambda should be numeric"
            assert lambda_global >= 0, f"{name}: Lambda should be non-negative"
            assert isinstance(lambda_by_layer, dict), f"{name}: Lambda by layer should be a dict"
            assert 'global' in lambda_by_layer, f"{name}: Lambda by layer should have 'global' key"
            
            print(f"    ✓ {name} sigma and lambda computation successful")
    
    def test_layer_filtering(self):
        """Test compute_delta with layer filtering."""
        print("\n" + "="*60)
        print("TEST 5: Layer Filtering")
        print("="*60)
        
        # Create structure with multiple layers
        structure = self.creator.create_structure(
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            structure_type="bulk",
            template="cubic",
            thickness=3,
            xy_expansion=(2, 2),
        )
        
        analyzer = q2D_analyzer(structure)
        analyzer.analyze()
        
        layers = analyzer.get_layers()
        layer_ids = list(layers.keys())
        print(f"  Available layers: {layer_ids}")
        
        if len(layer_ids) > 0:
            # Test single layer
            first_layer = layer_ids[0]
            delta_layer = analyzer.compute_delta(layer=first_layer)
            print(f"  Delta for layer {first_layer}: {delta_layer}")
            assert delta_layer is not None, f"Delta for layer {first_layer} should not be None"
            
            # Test multiple layers
            if len(layer_ids) > 1:
                selected_layers = layer_ids[:2]
                delta_layers = analyzer.compute_delta(layers=selected_layers)
                print(f"  Delta for layers {selected_layers}: {delta_layers}")
                assert delta_layers is not None, "Delta for multiple layers should not be None"
        
        print("  ✓ Layer filtering test successful")
    
    def test_empty_results_diagnosis(self):
        """Diagnostic test to identify why compute_delta might return empty results."""
        print("\n" + "="*60)
        print("TEST 6: Empty Results Diagnosis")
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
        
        # Check graph structure
        graph = analyzer.get_graph()
        layer_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'layer']
        oct_nodes = [n for n, d in graph.nodes(data=True) if d.get('node_type') == 'octahedron']
        
        print(f"  Layer nodes in graph: {len(layer_nodes)}")
        print(f"  Octahedron nodes in graph: {len(oct_nodes)}")
        
        # Check CONTAINS edges
        contains_edges = 0
        for layer_node in layer_nodes:
            for neighbor in graph.neighbors(layer_node):
                edge_data = graph.get_edge_data(layer_node, neighbor)
                if edge_data and edge_data.get('edge_type') == 'contains':
                    if neighbor.startswith('octahedron_'):
                        contains_edges += 1
        
        print(f"  Layer→Octahedron CONTAINS edges: {contains_edges}")
        
        # Check octahedra details
        octahedra = analyzer.get_octahedra()
        print(f"  Octahedra from get_octahedra(): {len(octahedra)}")
        
        if len(octahedra) > 0:
            first_oct = octahedra[0]
            print(f"  First octahedron: {first_oct.get('id')}")
            print(f"    central_atom_index: {first_oct.get('central_atom_index')}")
            print(f"    ligand_atoms count: {len(first_oct.get('ligand_atoms', []))}")
        
        # Try compute_delta
        delta_by_layer = analyzer.compute_delta(group_by='layer')
        print(f"  Delta by layer result: {delta_by_layer}")
        print(f"  Result keys: {list(delta_by_layer.keys())}")
        
        # Check if result is empty or all None
        non_global_keys = [k for k in delta_by_layer.keys() if k != 'global']
        if len(non_global_keys) == 0:
            print("  ⚠ WARNING: No layer-specific deltas found!")
            print("  This may indicate:")
            print("    - Missing CONTAINS edges between layers and octahedra")
            print("    - Octahedra not properly connected to layers")
            print("    - All octahedra assigned to 'unknown' layer")
        else:
            print(f"  ✓ Found {len(non_global_keys)} layer-specific deltas")
        
        assert len(delta_by_layer) > 0, "Delta by layer should not be completely empty"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
