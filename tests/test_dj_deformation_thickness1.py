#!/usr/bin/env python3
"""Test script to debug DJ spacer deformation metrics with thickness=1.

Creates DJ structures with thickness=1 and prints detailed deformation analysis
including the new centroid_distance, shear_magnitude, and shear_angle features.
"""

import numpy as np
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.analyzer.cavities_processing.cavity_deformation import (
    calculate_antiprism_deformation,
    measure_cavity_properties,
    _identify_square_faces_topology,
    _identify_equilateral_triangles_from_graph,
    _identify_isosceles_triangles_from_graph,
    _extract_x_positions,
    _get_b_atoms_for_cage,
)
from q2D_Materials.utils.other.graph_pyvis_export import export_structure_pyvis


def print_cavity_info(cavity, name):
    """Print detailed information about a cavity."""
    print(f"\n{'='*80}")
    print(f"CAVITY: {name} (ID: {cavity.id})")
    print(f"{'='*80}")
    print(f"Type: {cavity.cavity_type}")
    print(f"B atoms: {len(cavity.b_atom_indices)}")
    print(f"X atoms: {len(cavity.x_atom_indices)}")
    print(f"Center position: {cavity.center_position}")
    
    # Print subgraph info
    print(f"\nSubgraph nodes: {len(cavity.subgraph.nodes())}")
    print(f"Subgraph edges: {len(cavity.subgraph.edges())}")
    
    # Check for cage nodes
    cage_nodes = [
        node for node in cavity.subgraph.nodes()
        if cavity.subgraph.nodes[node].get('node_type') in ('cage', 'half_cage')
    ]
    print(f"Cage nodes: {cage_nodes}")
    print(f"Number of cage nodes: {len(cage_nodes)}")
    
    # Print X atom properties
    print("\nX atoms in subgraph:")
    x_atoms = []
    for node in cavity.subgraph.nodes():
        node_data = cavity.subgraph.nodes[node]
        if node_data.get('node_type') == 'atom' and node_data.get('is_X', False):
            x_atoms.append({
                'node': node,
                'symbol': node_data.get('symbol', '?'),
                'is_terminal': node_data.get('is_terminal', False),
                'is_equatorial': node_data.get('is_equatorial', False),
                'is_interlayer': node_data.get('is_interlayer', False),
            })
    
    print(f"Found {len(x_atoms)} X atoms in subgraph")
    terminal_count = sum(1 for x in x_atoms if x['is_terminal'])
    equatorial_count = sum(1 for x in x_atoms if x['is_equatorial'])
    interlayer_count = sum(1 for x in x_atoms if x['is_interlayer'])
    print(f"  Terminal: {terminal_count}, Equatorial: {equatorial_count}, Interlayer: {interlayer_count}")
    
    for x in x_atoms[:10]:  # Print first 10
        print(f"  {x}")
    if len(x_atoms) > 10:
        print(f"  ... and {len(x_atoms) - 10} more")
    
    # Check B-X edge geometry properties
    print("\nB-X edge geometry properties:")
    geometry_counts = {}
    for u, v in cavity.subgraph.edges():
        edge_data = cavity.subgraph.get_edge_data(u, v)
        if edge_data and edge_data.get('edge_type') == 'bonded_to':
            geometry = edge_data.get('geometry', 'unknown')
            geometry_counts[geometry] = geometry_counts.get(geometry, 0) + 1
    for geom, count in geometry_counts.items():
        print(f"  {geom}: {count}")
    
    # Print B atom properties
    print("\nB atoms in subgraph:")
    b_atoms = []
    for node in cavity.subgraph.nodes():
        node_data = cavity.subgraph.nodes[node]
        if node_data.get('node_type') == 'atom' and node_data.get('is_B', False):
            b_atoms.append({
                'node': node,
                'symbol': node_data.get('symbol', '?'),
            })
    
    print(f"Found {len(b_atoms)} B atoms in subgraph")
    for b in b_atoms[:10]:  # Print first 10
        print(f"  {b}")
    if len(b_atoms) > 10:
        print(f"  ... and {len(b_atoms) - 10} more")
    
    # Print edge types
    print("\nEdge types in subgraph:")
    edge_types = {}
    for u, v in cavity.subgraph.edges():
        edge_data = cavity.subgraph.get_edge_data(u, v)
        edge_type = edge_data.get('edge_type', 'unknown') if edge_data else 'unknown'
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    for edge_type, count in edge_types.items():
        print(f"  {edge_type}: {count}")


def test_face_identification(cavity, name):
    """Test face identification functions."""
    print(f"\n{'='*80}")
    print(f"FACE IDENTIFICATION TEST: {name}")
    print(f"{'='*80}")
    
    # Check for cage nodes
    cage_nodes = [
        node for node in cavity.subgraph.nodes()
        if cavity.subgraph.nodes[node].get('node_type') in ('cage', 'half_cage')
    ]
    print(f"Cage nodes found: {cage_nodes}")
    print(f"Number of cage nodes: {len(cage_nodes)}")
    
    if len(cage_nodes) >= 2:
        print("\nTesting per-cage face identification:")
        for cage_node in cage_nodes:
            print(f"\n--- Cage: {cage_node} ---")
            
            # Get B atoms for this cage
            try:
                b_atoms = _get_b_atoms_for_cage(cavity, cage_node_id=cage_node)
                print(f"B atoms for this cage: {len(b_atoms)}")
            except Exception as e:
                print(f"ERROR getting B atoms: {e}")
                continue
            
            # Get X positions
            try:
                x_positions, x_node_ids = _extract_x_positions(cavity, cage_node_id=cage_node)
                print(f"X positions extracted: {len(x_positions)}")
            except Exception as e:
                print(f"ERROR extracting X positions: {e}")
                import traceback
                traceback.print_exc()
                continue
            
            # Test square faces
            try:
                squares = _identify_square_faces_topology(cavity, cage_node_id=cage_node)
                print(f"Square faces: {len(squares)}")
                for i, sq in enumerate(squares):
                    print(f"  Square {i}: {sq}")
            except Exception as e:
                print(f"ERROR identifying square faces: {e}")
                import traceback
                traceback.print_exc()
            
            # Test equilateral triangles
            try:
                equilateral = _identify_equilateral_triangles_from_graph(cavity, cage_node_id=cage_node)
                print(f"Equilateral triangles: {len(equilateral)}")
                for i, tri in enumerate(equilateral[:5]):  # First 5
                    print(f"  Triangle {i}: {tri}")
                if len(equilateral) > 5:
                    print(f"  ... and {len(equilateral) - 5} more")
            except Exception as e:
                print(f"ERROR identifying equilateral triangles: {e}")
                import traceback
                traceback.print_exc()
            
            # Test isosceles triangles
            try:
                isosceles = _identify_isosceles_triangles_from_graph(cavity, cage_node_id=cage_node)
                print(f"Isosceles triangles: {len(isosceles)}")
                for i, tri in enumerate(isosceles[:5]):  # First 5
                    print(f"  Triangle {i}: {tri}")
                if len(isosceles) > 5:
                    print(f"  ... and {len(isosceles) - 5} more")
            except Exception as e:
                print(f"ERROR identifying isosceles triangles: {e}")
                import traceback
                traceback.print_exc()
            
            # Test measure_cavity_properties
            try:
                props = measure_cavity_properties(cavity, cage_node_id=cage_node)
                print(f"\nMeasured properties:")
                print(f"  Square angles: {len(props['square_angles'])}")
                print(f"  Equilateral angles: {len(props['equilateral_angles'])}")
                print(f"  Isosceles angles: {len(props['isosceles_angles'])}")
                print(f"  Edge lengths: {len(props['edge_lengths'])}")
                print(f"  Volume: {props['volume']:.2f}")
                print(f"  Terminal count: {props['terminal_count']}")
                print(f"  X center: {props['x_center']}")
            except Exception as e:
                print(f"ERROR measuring properties: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"\n⚠️  WARNING: Only {len(cage_nodes)} cage node(s) found. Expected 2 for DJ spacers.")
        print("Testing without cage filtering:")
        # Test without cage filtering
        try:
            squares = _identify_square_faces_topology(cavity, cage_node_id=None)
            print(f"Square faces: {len(squares)}")
        except Exception as e:
            print(f"ERROR identifying square faces: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            equilateral = _identify_equilateral_triangles_from_graph(cavity, cage_node_id=None)
            print(f"Equilateral triangles: {len(equilateral)}")
        except Exception as e:
            print(f"ERROR identifying equilateral triangles: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            isosceles = _identify_isosceles_triangles_from_graph(cavity, cage_node_id=None)
            print(f"Isosceles triangles: {len(isosceles)}")
        except Exception as e:
            print(f"ERROR identifying isosceles triangles: {e}")
            import traceback
            traceback.print_exc()
        
        try:
            props = measure_cavity_properties(cavity, cage_node_id=None)
            print(f"\nMeasured properties:")
            print(f"  Square angles: {len(props['square_angles'])}")
            print(f"  Equilateral angles: {len(props['equilateral_angles'])}")
            print(f"  Isosceles angles: {len(props['isosceles_angles'])}")
            print(f"  Edge lengths: {len(props['edge_lengths'])}")
        except Exception as e:
            print(f"ERROR measuring properties: {e}")
            import traceback
            traceback.print_exc()


def test_deformation(cavity, name):
    """Test deformation calculation."""
    print(f"\n{'='*80}")
    print(f"DEFORMATION CALCULATION TEST: {name}")
    print(f"{'='*80}")
    
    # Try to get deformation
    try:
        # Use a reasonable BX distance (Pb-I is about 3.2 Å, Pb-Br is about 3.0 Å)
        bx_distance = 3.2
        
        print(f"Using BX distance: {bx_distance} Å")
        
        # Test delta mode
        print("\n" + "-"*80)
        print("DELTA MODE")
        print("-"*80)
        result_delta = calculate_antiprism_deformation(
            cavity, bx_distance, mode='delta', full=False
        )
        print("\nTop-level results:")
        print(f"  eta: {result_delta.get('eta', 'N/A')}")
        print(f"  kappa: {result_delta.get('kappa', 'N/A')} (combined)")
        print(f"  kappa_equilateral: {result_delta.get('kappa_equilateral', 'N/A')}")
        print(f"  kappa_isosceles: {result_delta.get('kappa_isosceles', 'N/A')}")
        print(f"  nu: {result_delta.get('nu', 'N/A')}")
        print(f"  omega: {result_delta.get('omega', 'N/A')}")
        print(f"  delta_volume: {result_delta.get('delta_volume', 'N/A')}")
        print(f"  volume: {result_delta.get('volume', 'N/A')} (actual volume)")
        
        # NEW FEATURES
        print("\nNew features (half-cage metrics):")
        if 'centroid_distance' in result_delta:
            print(f"  ✅ centroid_distance: {result_delta['centroid_distance']:.4f} Å")
        else:
            print(f"  ❌ centroid_distance: NOT FOUND")
        
        if 'shear_magnitude' in result_delta:
            print(f"  ✅ shear_magnitude: {result_delta['shear_magnitude']:.4f} Å")
        else:
            print(f"  ❌ shear_magnitude: NOT FOUND")
        
        if 'shear_angle' in result_delta:
            print(f"  ✅ shear_angle: {result_delta['shear_angle']:.2f}°")
        else:
            print(f"  ❌ shear_angle: NOT FOUND")
        
        if 'cages' in result_delta:
            print(f"\nPer-cage results: {len(result_delta['cages'])} cages")
            for i, cage_result in enumerate(result_delta['cages']):
                print(f"\n  Cage {i} ({cage_result.get('cage_node_id', 'unknown')}):")
                print(f"    eta: {cage_result.get('eta', 'N/A')}")
                print(f"    kappa: {cage_result.get('kappa', 'N/A')} (combined)")
                print(f"    kappa_equilateral: {cage_result.get('kappa_equilateral', 'N/A')}")
                print(f"    kappa_isosceles: {cage_result.get('kappa_isosceles', 'N/A')}")
                print(f"    nu: {cage_result.get('nu', 'N/A')}")
                print(f"    omega: {cage_result.get('omega', 'N/A')}")
                print(f"    volume: {cage_result.get('volume', 'N/A')} (actual volume)")
                print(f"    shear_magnitude (octagonal): {cage_result.get('shear_magnitude', 'N/A')}")
                print(f"    shear_vector (octagonal): {cage_result.get('shear_vector', 'N/A')}")
        else:
            print("\n⚠️  No 'cages' key in results - this might indicate a single-cage structure")
        
        # Test absolute mode
        print("\n" + "-"*80)
        print("ABSOLUTE MODE")
        print("-"*80)
        result_abs = calculate_antiprism_deformation(
            cavity, bx_distance, mode='absolute', full=False
        )
        print("\nTop-level results:")
        if 'cages' in result_abs:
            print(f"  Per-cage results: {len(result_abs['cages'])} cages")
            for i, cage_result in enumerate(result_abs['cages']):
                print(f"\n  Cage {i}:")
                print(f"    Square angles: {cage_result.get('square_angles', {}).get('mean', 'N/A')}")
                print(f"    Equilateral angles: {cage_result.get('equilateral_angles', {}).get('mean', 'N/A')}")
                print(f"    Isosceles angles: {cage_result.get('isosceles_angles', {}).get('mean', 'N/A')}")
                print(f"    Volume: {cage_result.get('volume', 'N/A')}")
        else:
            print(f"  Square angles: {result_abs.get('square_angles', {}).get('mean', 'N/A')}")
            print(f"  Volume: {result_abs.get('volume', 'N/A')}")
        
        # NEW FEATURES in absolute mode
        print("\nNew features (half-cage metrics) in absolute mode:")
        if 'centroid_distance' in result_abs:
            print(f"  ✅ centroid_distance: {result_abs['centroid_distance']:.4f} Å")
        else:
            print(f"  ❌ centroid_distance: NOT FOUND")
        
        if 'shear_magnitude' in result_abs:
            print(f"  ✅ shear_magnitude: {result_abs['shear_magnitude']:.4f} Å")
        else:
            print(f"  ❌ shear_magnitude: NOT FOUND")
        
        if 'shear_angle' in result_abs:
            print(f"  ✅ shear_angle: {result_abs['shear_angle']:.2f}°")
        else:
            print(f"  ❌ shear_angle: NOT FOUND")
        
    except Exception as e:
        print(f"\n❌ ERROR calculating deformation: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main test function."""
    print("="*80)
    print("DJ SPACER DEFORMATION DEBUG TEST - THICKNESS=1, TEMPLATE=REDUCED")
    print("="*80)
    print("\n⚠️  Testing with thickness=1 and template=reduced to identify issues")
    
    # Create two DJ structures with thickness=1
    creator = q2D_creator()
    
    print("\n" + "="*80)
    print("Creating DJ structure 1 (thickness=1, template=reduced)...")
    print("="*80)
    try:
        dj1 = creator.create_structure(
            structure_type="bulk",
            template="reduced",  # ⚠️ REDUCED TEMPLATE
            layer_sequence="DJ",
            thickness=1,  # ⚠️ THICKNESS=1
            A_ions="MA",
            B_ions="Pb",
            X_ions="I",
            spacer="[NH3+]CCCC[NH3+]",  # 1,4-butanediammonium
            xy_expansion=(1, 1),
        )
        print("✅ Structure 1 created successfully")
    except Exception as e:
        print(f"❌ ERROR creating structure 1: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("Creating DJ structure 2 (thickness=1, template=reduced)...")
    print("="*80)
    try:
        dj2 = creator.create_structure(
            structure_type="bulk",
            template="reduced",  # ⚠️ REDUCED TEMPLATE
            layer_sequence="DJ",
            thickness=1,  # ⚠️ THICKNESS=1
            A_ions="MA",
            B_ions="Pb",
            X_ions="Br",
            spacer="[NH3+]CCCCCC[NH3+]",  # 1,6-hexanediammonium
            xy_expansion=(1, 1),
        )
        print("✅ Structure 2 created successfully")
    except Exception as e:
        print(f"❌ ERROR creating structure 2: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Analyze structures
    print("\n" + "="*80)
    print("Analyzing structure 1...")
    print("="*80)
    try:
        analyzer1 = q2D_analyzer(dj1)
        analyzer1.analyze()
        
        # Export and print graph for n=1 structure
        print("\n" + "-"*80)
        print("Graph information for n=1 (thickness=1, template=reduced) structure 1...")
        print("-"*80)
        try:
            graph1 = analyzer1.get_graph()
            
            # Print graph statistics
            print(f"\nGraph nodes: {len(graph1.nodes())}")
            print(f"Graph edges: {len(graph1.edges())}")
            
            # Find B atoms
            b_atoms = []
            for node in graph1.nodes():
                node_data = graph1.nodes[node]
                if node_data.get('node_type') == 'atom':
                    # Check if it's a B atom by looking for octahedron connections
                    for neighbor in graph1.neighbors(node):
                        edge_data = graph1.get_edge_data(node, neighbor)
                        if (edge_data and 
                            edge_data.get('edge_type') == 'contains' and
                            edge_data.get('role') == 'center'):
                            b_atoms.append(node)
                            break
            
            print(f"\nB atoms found: {len(b_atoms)}")
            for b_node in b_atoms[:5]:  # Print first 5
                print(f"  B atom: {b_node}")
                b_data = graph1.nodes[b_node]
                print(f"    Symbol: {b_data.get('symbol', '?')}")
                print(f"    Index: {b_data.get('vasp_index', '?')}")
                
                # Check X neighbors and their geometry
                x_neighbors = []
                for neighbor in graph1.neighbors(b_node):
                    edge_data = graph1.get_edge_data(b_node, neighbor)
                    if (edge_data and
                        edge_data.get('edge_type') == 'bonded_to' and
                        edge_data.get('role') == 'ligand'):
                        neighbor_data = graph1.nodes[neighbor]
                        geometry = edge_data.get('geometry', 'unknown')
                        is_terminal = neighbor_data.get('is_terminal', False)
                        is_equatorial = neighbor_data.get('is_equatorial', False)
                        x_neighbors.append({
                            'node': neighbor,
                            'symbol': neighbor_data.get('symbol', '?'),
                            'index': neighbor_data.get('vasp_index', '?'),
                            'geometry': geometry,
                            'is_terminal': is_terminal,
                            'is_equatorial': is_equatorial,
                        })
                
                print(f"    X neighbors: {len(x_neighbors)}")
                for x in x_neighbors[:6]:  # Print first 6
                    print(f"      X {x['node']} (idx={x['index']}, sym={x['symbol']}): "
                          f"geometry={x['geometry']}, is_terminal={x['is_terminal']}, "
                          f"is_equatorial={x['is_equatorial']}")
                if len(x_neighbors) > 6:
                    print(f"      ... and {len(x_neighbors) - 6} more")
            
            # Export graph visualization
            graph1_path = export_structure_pyvis(graph1, 'dj_thickness1_structure1_graph.html')
            print(f"\n✅ Graph exported to: {graph1_path}")
        except Exception as e:
            print(f"⚠️  Could not process/export graph: {e}")
            import traceback
            traceback.print_exc()
        
        cavities1 = analyzer1.get_cavities()
        spacer_cavities1 = [c for c in cavities1 if c.cavity_type == 'spacer_dj']
        print(f"✅ Found {len(spacer_cavities1)} DJ spacer cavities in structure 1")
    except Exception as e:
        print(f"❌ ERROR analyzing structure 1: {e}")
        import traceback
        traceback.print_exc()
        spacer_cavities1 = []
    
    print("\n" + "="*80)
    print("Analyzing structure 2...")
    print("="*80)
    try:
        analyzer2 = q2D_analyzer(dj2)
        analyzer2.analyze()
        
        # Export and print graph for n=1 structure
        print("\n" + "-"*80)
        print("Graph information for n=1 (thickness=1, template=reduced) structure 2...")
        print("-"*80)
        try:
            graph2 = analyzer2.get_graph()
            
            # Print graph statistics
            print(f"\nGraph nodes: {len(graph2.nodes())}")
            print(f"Graph edges: {len(graph2.edges())}")
            
            # Find B atoms
            b_atoms = []
            for node in graph2.nodes():
                node_data = graph2.nodes[node]
                if node_data.get('node_type') == 'atom':
                    # Check if it's a B atom by looking for octahedron connections
                    for neighbor in graph2.neighbors(node):
                        edge_data = graph2.get_edge_data(node, neighbor)
                        if (edge_data and 
                            edge_data.get('edge_type') == 'contains' and
                            edge_data.get('role') == 'center'):
                            b_atoms.append(node)
                            break
            
            print(f"\nB atoms found: {len(b_atoms)}")
            for b_node in b_atoms[:5]:  # Print first 5
                print(f"  B atom: {b_node}")
                b_data = graph2.nodes[b_node]
                print(f"    Symbol: {b_data.get('symbol', '?')}")
                print(f"    Index: {b_data.get('vasp_index', '?')}")
                
                # Check X neighbors and their geometry
                x_neighbors = []
                for neighbor in graph2.neighbors(b_node):
                    edge_data = graph2.get_edge_data(b_node, neighbor)
                    if (edge_data and
                        edge_data.get('edge_type') == 'bonded_to' and
                        edge_data.get('role') == 'ligand'):
                        neighbor_data = graph2.nodes[neighbor]
                        geometry = edge_data.get('geometry', 'unknown')
                        is_terminal = neighbor_data.get('is_terminal', False)
                        is_equatorial = neighbor_data.get('is_equatorial', False)
                        x_neighbors.append({
                            'node': neighbor,
                            'symbol': neighbor_data.get('symbol', '?'),
                            'index': neighbor_data.get('vasp_index', '?'),
                            'geometry': geometry,
                            'is_terminal': is_terminal,
                            'is_equatorial': is_equatorial,
                        })
                
                print(f"    X neighbors: {len(x_neighbors)}")
                for x in x_neighbors[:6]:  # Print first 6
                    print(f"      X {x['node']} (idx={x['index']}, sym={x['symbol']}): "
                          f"geometry={x['geometry']}, is_terminal={x['is_terminal']}, "
                          f"is_equatorial={x['is_equatorial']}")
                if len(x_neighbors) > 6:
                    print(f"      ... and {len(x_neighbors) - 6} more")
            
            # Export graph visualization
            graph2_path = export_structure_pyvis(graph2, 'dj_thickness1_structure2_graph.html')
            print(f"\n✅ Graph exported to: {graph2_path}")
        except Exception as e:
            print(f"⚠️  Could not process/export graph: {e}")
            import traceback
            traceback.print_exc()
        
        cavities2 = analyzer2.get_cavities()
        spacer_cavities2 = [c for c in cavities2 if c.cavity_type == 'spacer_dj']
        print(f"✅ Found {len(spacer_cavities2)} DJ spacer cavities in structure 2")
    except Exception as e:
        print(f"❌ ERROR analyzing structure 2: {e}")
        import traceback
        traceback.print_exc()
        spacer_cavities2 = []
    
    # Test first cavity from each structure
    if len(spacer_cavities1) > 0:
        cavity1 = spacer_cavities1[0]
        print_cavity_info(cavity1, "DJ Structure 1 (thickness=1, template=reduced) - Cavity 0")
        test_face_identification(cavity1, "DJ Structure 1 (thickness=1, template=reduced) - Cavity 0")
        test_deformation(cavity1, "DJ Structure 1 (thickness=1, template=reduced) - Cavity 0")
    else:
        print("\n⚠️  No DJ spacer cavities found in structure 1")
    
    if len(spacer_cavities2) > 0:
        cavity2 = spacer_cavities2[0]
        print_cavity_info(cavity2, "DJ Structure 2 (thickness=1, template=reduced) - Cavity 0")
        test_face_identification(cavity2, "DJ Structure 2 (thickness=1, template=reduced) - Cavity 0")
        test_deformation(cavity2, "DJ Structure 2 (thickness=1, template=reduced) - Cavity 0")
    else:
        print("\n⚠️  No DJ spacer cavities found in structure 2")
    
    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
