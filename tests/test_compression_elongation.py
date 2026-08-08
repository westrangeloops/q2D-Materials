#!/usr/bin/env python3
"""Test compression calculation with non-elongated and elongated molecules.

This test verifies that compression values change when molecules are elongated,
ensuring the compression calculation is working correctly.

Running the tests:
-----------------
1. Using pytest (recommended):
   # Run all parametrized tests
   pytest tests/test_compression_elongation.py::test_compression_changes_with_elongation -v
   
   # Run with verbose output showing print statements
   pytest tests/test_compression_elongation.py::test_compression_changes_with_elongation -v -s
   
   # Run a specific test case by index
   pytest tests/test_compression_elongation.py::test_compression_changes_with_elongation[smiles0] -v
   
   # Run all tests in the file
   pytest tests/test_compression_elongation.py -v

2. Directly with Python:
   # Run all parametrized tests
   python tests/test_compression_elongation.py
   
   # Or run just the single molecule test (modify __main__ section)
"""

import numpy as np
import pytest
from ase import Atoms
import networkx as nx

from q2D_Materials.builders.molecule_builder import smiles_to_ase_atoms
from q2D_Materials.builders.optimizers import elongate_molecule
from q2D_Materials.analyzer.molecular_processing.spacer_analysis import SpacerAnalysis
from q2D_Materials.utils.molecules.graph_converter import atoms_to_graph
from q2D_Materials.modifier.molecule_graph import MoleculeGraph


class MockGraphView:
    """Minimal mock GraphView for MoleculeGraph compatibility."""
    pass


def create_molecule_graph_from_atoms(atoms: Atoms) -> MoleculeGraph:
    """Create a MoleculeGraph from standalone Atoms object.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object
        
    Returns
    -------
    MoleculeGraph
        MoleculeGraph instance suitable for SpacerAnalysis
    """
    # Convert atoms to graph
    graph = atoms_to_graph(atoms, preserve_coords=True)
    
    # Create original_indices (0 to N-1 for standalone molecule)
    original_indices = list(range(len(atoms)))
    
    # Create MoleculeGraph
    mock_view = MockGraphView()
    mol_graph = MoleculeGraph(
        graph=graph,
        original_indices=original_indices,
        attachment_points=[],  # No attachments for standalone molecule
        molecule_type='spacer',
        molecule_index=0,
        atoms_object=atoms,
        parent_view=mock_view
    )
    
    return mol_graph


def analyze_compression(atoms: Atoms) -> float:
    """Analyze compression for a molecule.
    
    Parameters
    ----------
    atoms : Atoms
        ASE Atoms object to analyze
        
    Returns
    -------
    float
        Compression value in Angstroms
    """
    mol_graph = create_molecule_graph_from_atoms(atoms)
    analyzer = SpacerAnalysis(mol_graph, spacer_type="DJ")
    result = analyzer.compute()
    return result.compression


@pytest.mark.parametrize("smiles", [
    "[NH3+]CCCCCC[NH3+]",  # Diammonium hexane
    "[NH3+]CCCC[NH3+]",    # Diammonium butane
    "[NH3+]CCCCCCCC[NH3+]", # Diammonium octane
    "[NH3+]CCCCCCCCCCCC[NH3+]", # Diammonium dodecane
    "[NH3+]CCCCCCCCCCCCCCCC[NH3+]", # Diammonium tetradecane
    "[NH3+]CCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium hexadecane
    "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium octadecane
    "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium nonadecane
    "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium eicosane
    "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium docosane
    "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium tricosane
    "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium tetracosane
    "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium pentacosane
    "[NH3+]CCC1CCC(CCC[NH3+])CC1", # Diammonium hexacosane
    "[NH3+]CCC1=CC=C(CCC[NH3+])C=C1", # Diammonium heptacosane
    "[NH3+]CCC1=CC=C(CCC[NH3+])C=C1", # Diammonium octacosane
    "[NH3+]CCC1=CC=C(CCC[NH3+])C=C1", # Diammonium nonacosane
    "[NH3+]CCC1=CC=C(CCC[NH3+])C=C1", # Diammonium hentacosane
    "[NH3+]CCC1=CC=C(CCC[NH3+])C=C1", # Diammonium octacosane
    "[NH3+]CCC1=CC=C(CCC[NH3+])C=C1", # Diammonium nonacosane
    "[NH3+]CCC1=CC=C(CCC[NH3+])C=C1", # Diammonium hentacosane
])
def test_compression_changes_with_elongation(smiles):
    """Test that compression changes when molecules are elongated.
    
    This test:
    1. Creates a molecule from SMILES
    2. Analyzes compression of the original (non-elongated) molecule
    3. Elongates the molecule
    4. Analyzes compression of the elongated molecule
    5. Verifies that the compression difference is significant (not near zero)
    
    Parameters
    ----------
    smiles : str
        SMILES string for a diammonium spacer molecule
    """
    # Create molecule from SMILES
    original_atoms = smiles_to_ase_atoms(smiles)
    
    # Analyze compression of original molecule
    compression_original = analyze_compression(original_atoms)
    
    # Elongate the molecule
    elongated_atoms = elongate_molecule(original_atoms, max_iterations=100)
    
    # Analyze compression of elongated molecule
    compression_elongated = analyze_compression(elongated_atoms)
    
    # Calculate the difference
    compression_diff = abs(compression_elongated - compression_original)
    
    # Print diagnostic information
    print(f"\n{'='*80}")
    print(f"SMILES: {smiles}")
    print(f"Original compression: {compression_original:.3f} Å")
    print(f"Elongated compression: {compression_elongated:.3f} Å")
    print(f"Compression difference: {compression_diff:.3f} Å")
    print(f"{'='*80}")
    
    # The compression should change significantly when elongated
    # If the difference is near zero, the compression calculation is not working
    tolerance = 0.1  # Minimum expected difference in Angstroms
    assert compression_diff > tolerance, (
        f"Compression difference is too small ({compression_diff:.3f} Å). "
        f"This suggests the compression calculation is not working correctly. "
        f"Original: {compression_original:.3f} Å, "
        f"Elongated: {compression_elongated:.3f} Å"
    )


def test_compression_elongation_single_molecule():
    """Test compression with a single specific molecule."""
    smiles = "[NH3+]CCCCCC[NH3+]"  # Diammonium hexane
    
    # Create molecule from SMILES
    original_atoms = smiles_to_ase_atoms(smiles)
    
    # Analyze compression of original molecule
    compression_original = analyze_compression(original_atoms)
    print(f"\nOriginal compression: {compression_original:.3f} Å")
    
    # Elongate the molecule
    elongated_atoms = elongate_molecule(original_atoms, max_iterations=100)
    
    # Analyze compression of elongated molecule
    compression_elongated = analyze_compression(elongated_atoms)
    print(f"Elongated compression: {compression_elongated:.3f} Å")
    
    # Calculate the difference
    compression_diff = abs(compression_elongated - compression_original)
    print(f"Compression difference: {compression_diff:.3f} Å")
    
    # The compression should change significantly when elongated
    tolerance = 0.1
    assert compression_diff > tolerance, (
        f"Compression difference is too small ({compression_diff:.3f} Å). "
        f"Original: {compression_original:.3f} Å, "
        f"Elongated: {compression_elongated:.3f} Å"
    )


def test_compression_in_dj_structures():
    """Test compression calculation on 15 different DJ structures with different molecules."""
    from q2D_Materials.core.creator import q2D_creator
    from q2D_Materials.analyzer import q2D_analyzer
    from q2D_Materials.modifier import GraphView
    
    # 15 different spacer molecules to test
    test_molecules = [
        ("[NH3+]CCCC[NH3+]", "Diammonium butane"),
        ("[NH3+]CCCCCC[NH3+]", "Diammonium hexane"),
        ("[NH3+]CCCCCCCC[NH3+]", "Diammonium octane"),
        ("[NH3+]CCCCCCCCCC[NH3+]", "Diammonium decane"),
        ("[NH3+]CCCCCCCCCCCC[NH3+]", "Diammonium dodecane"),
        ("[NH3+]CCCCCCCCCCCCCC[NH3+]", "Diammonium tetradecane"),
        ("[NH3+]CCCCCCCCCCCCCCCC[NH3+]", "Diammonium hexadecane"),
        ("[NH3+]CCC1CCC(CCC[NH3+])CC1", "Diammonium cyclohexyl"),
        ("[NH3+]CCC1=CC=C(CCC[NH3+])C=C1", "Diammonium phenyl"),
        ("[NH3+]CCCC(C)CC[NH3+]", "Diammonium branched hexane"),
        ("[NH3+]CCCCCCCCCCCCCCCCCC[NH3+]", "Diammonium eicosane"),
        ("[NH3+]CCCCCCCCCCCCCCCCCCCCCC[NH3+]", "Diammonium docosane"),
        ("[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", "Diammonium octacosane"),
        ("[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", "Diammonium dotriacontane"),
        ("[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", "Diammonium hexatriacontane"),
    ]
    
    creator = q2D_creator()
    results = []
    
    print("\n" + "="*80)
    print("TESTING COMPRESSION IN 15 DJ STRUCTURES")
    print("="*80)
    
    for i, (smiles, name) in enumerate(test_molecules, 1):
        print(f"\n[{i}/15] Testing: {name} ({smiles})")
        print("-" * 80)
        
        try:
            # Create DJ structure
            structure = creator.create_structure(
                structure_type="bulk",
                template="cubic",
                layer_sequence="DJ",
                thickness=2,
                A_ions="MA",
                B_ions="Pb",
                X_ions="I",
                spacer=smiles,
                xy_expansion=(1, 1),
            )
            
            # Write structure to VASP format
            from ase.io import write
            import os
            output_dir = "dj_structures_vasp"
            os.makedirs(output_dir, exist_ok=True)
            
            # Create filename from SMILES (sanitize for filesystem)
            safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
            vasp_filename = os.path.join(output_dir, f"{i:02d}_{safe_name}.vasp")
            write(vasp_filename, structure, format='vasp')
            print(f"  Written: {vasp_filename}")
            
            # Analyze structure
            analyzer = q2D_analyzer(structure)
            analyzer.analyze()
            
            # Get spacers
            view = GraphView(analyzer)
            spacers = view.spacers.list()
            
            # Print structural graph diagnostics for spacer nodes
            print(f"\n  Structural Graph Diagnostics:")
            graph = analyzer.get_graph()
            spacer_nodes = [(node, data) for node, data in graph.nodes(data=True) if data.get('node_type') == 'spacer']
            print(f"    Found {len(spacer_nodes)} spacer nodes in structural graph")
            for spacer_node, spacer_data in spacer_nodes:
                # Count atoms connected via CONTAINS edges
                atom_indices = []
                for neighbor in graph.neighbors(spacer_node):
                    edge_data = graph.get_edge_data(spacer_node, neighbor)
                    if edge_data and edge_data.get('edge_type') == 'contains':
                        neighbor_data = graph.nodes.get(neighbor, {})
                        if neighbor_data.get('node_type') == 'atom':
                            atom_idx = neighbor_data.get('vasp_index')
                            if atom_idx is not None:
                                atom_indices.append(atom_idx)
                
                symbols = [structure[i].symbol for i in atom_indices]
                n_count = symbols.count('N')
                c_count = symbols.count('C')
                h_count = symbols.count('H')
                print(f"    {spacer_node}: {len(atom_indices)} atoms (N={n_count}, C={c_count}, H={h_count})")
                print(f"      Formula: {spacer_data.get('formula', '?')}")
                print(f"      Atom indices: {atom_indices[:10]}{'...' if len(atom_indices) > 10 else ''}")
            
            if len(spacers) == 0:
                print(f"  ✗ No spacers found in structure")
                results.append({
                    'smiles': smiles,
                    'name': name,
                    'success': False,
                    'error': 'No spacers found',
                    'compression': None,
                    'compressions': []
                })
                continue
            
            # Analyze compression for each spacer
            compressions = []
            for j, spacer in enumerate(spacers):
                try:
                    # Print graph structure diagnostics
                    print(f"\n  Spacer {j+1} Graph Diagnostics:")
                    print(f"    Graph nodes: {len(spacer.graph.nodes())} nodes")
                    print(f"    Atoms object: {len(spacer.atoms_object)} atoms")
                    print(f"    Original indices: {len(spacer.original_indices)} indices")
                    
                    # Count N atoms in graph vs atoms_object
                    n_in_graph = 0
                    n_in_atoms = 0
                    graph_n_nodes = []
                    atoms_n_indices = []
                    
                    for node, data in spacer.graph.nodes(data=True):
                        if isinstance(node, (int, np.integer)):
                            symbol = data.get('symbol')
                            if symbol == 'N':
                                n_in_graph += 1
                                graph_n_nodes.append((node, len(list(spacer.graph.neighbors(node)))))
                    
                    for i, atom in enumerate(spacer.atoms_object):
                        if atom.symbol == 'N':
                            n_in_atoms += 1
                            orig_idx = spacer.original_indices[i] if i < len(spacer.original_indices) else None
                            atoms_n_indices.append((i, orig_idx, atom.symbol))
                    
                    print(f"    N atoms in graph: {n_in_graph}")
                    print(f"    N atoms in atoms_object: {n_in_atoms}")
                    if n_in_graph > 0:
                        print(f"    Graph N nodes (node_id, neighbors): {graph_n_nodes}")
                    if n_in_atoms > 0:
                        print(f"    Atoms_object N indices (local_idx, orig_idx, symbol): {atoms_n_indices[:5]}")
                    
                    # Print graph node details (first 10 nodes)
                    print(f"    Graph node details (first 10):")
                    for idx, (node, data) in enumerate(list(spacer.graph.nodes(data=True))[:10]):
                        symbol = data.get('symbol', '?')
                        node_type = data.get('node_type', '?')
                        neighbors = len(list(spacer.graph.neighbors(node)))
                        print(f"      Node {node}: symbol={symbol}, type={node_type}, neighbors={neighbors}")
                    
                    analysis = spacer.analyze_spacer(analyzer)
                    compression = analysis.compression
                    compressions.append(compression)
                    
                    print(f"\n  Spacer {j+1}: compression={compression:.3f} Å, "
                          f"euclidean={analysis.euclidean_distance:.3f} Å, "
                          f"path_len={analysis.path_length:.3f} Å, "
                          f"ideal={analysis.ideal_extended_length:.3f} Å")
                    
                    # Check if compression was calculated
                    if compression == 0.0 and analysis.euclidean_distance == 0.0:
                        print(f"    ⚠ Warning: Compression appears to be default 0.0")
                        print(f"    Terminal N count: {len(analysis.terminal_nitrogens)}")
                        print(f"    Terminal N indices: {analysis.terminal_nitrogens}")
                except Exception as e:
                    print(f"  ✗ Error analyzing spacer {j+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    compressions.append(None)
            
            # Calculate mean compression (exclude None and NaN)
            valid_compressions = [
                c for c in compressions 
                if c is not None and not np.isnan(c)
            ]
            if valid_compressions:
                mean_compression = np.mean(valid_compressions)
                print(f"  Mean compression: {mean_compression:.3f} Å")
                
                # Check if compression is always 0 (potential bug)
                if all(abs(c) < 1e-6 for c in valid_compressions):
                    print(f"  ⚠ WARNING: All compression values are near 0!")
                    success = False
                    error = "Compression values all near 0"
                else:
                    success = True
                    error = None
            else:
                mean_compression = None
                success = False
                error = "No valid compression values"
                print(f"  ✗ No valid compression values calculated")
            
            results.append({
                'smiles': smiles,
                'name': name,
                'success': success,
                'error': error,
                'compression': mean_compression,
                'compressions': compressions,
                'n_spacers': len(spacers)
            })
            
        except Exception as e:
            print(f"  ✗ Error creating/analyzing structure: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'smiles': smiles,
                'name': name,
                'success': False,
                'error': str(e),
                'compression': None,
                'compressions': []
            })
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    print(f"Total structures tested: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Show compression statistics (exclude None and NaN)
    valid_compressions = [
        r['compression'] for r in results 
        if r['compression'] is not None and not np.isnan(r['compression'])
    ]
    
    if valid_compressions:
        print(f"\nCompression Statistics:")
        print(f"  Mean: {np.mean(valid_compressions):.3f} Å")
        print(f"  Std: {np.std(valid_compressions):.3f} Å")
        print(f"  Min: {np.min(valid_compressions):.3f} Å")
        print(f"  Max: {np.max(valid_compressions):.3f} Å")
        
        # Check if all are near 0
        if all(abs(c) < 1e-6 for c in valid_compressions):
            print(f"\n  ⚠ CRITICAL: All compression values are near 0!")
            print(f"  This suggests the compression calculation is not working correctly.")
    else:
        print(f"\n⚠ WARNING: No valid compression values found (all None or NaN)")
    
    # Show failed tests
    if failed > 0:
        print(f"\nFailed tests:")
        for r in results:
            if not r['success']:
                print(f"  - {r['name']}: {r['error']}")
    
    # Assert that at least some compressions are non-zero
    if len(valid_compressions) == 0:
        raise AssertionError(
            f"No valid compression values found! All values are None or NaN. "
            f"This indicates the compression calculation is failing for all structures."
        )
    
    non_zero_compressions = [c for c in valid_compressions if abs(c) > 0.1]
    assert len(non_zero_compressions) > 0, (
        f"All compression values are near zero! This indicates the compression "
        f"calculation is not working. Found {len(valid_compressions)} valid values, "
        f"all near zero. Values: {[f'{c:.3f}' for c in valid_compressions[:10]]}"
    )
    
    print(f"\n✓ Test passed: Found {len(non_zero_compressions)} structures with non-zero compression")
    # Don't return results to avoid pytest warning


if __name__ == "__main__":
    # Option 1: Run the single molecule test
    print("Running single molecule test...")
    test_compression_elongation_single_molecule()
    print("\n✓ Single molecule test passed!")
    
    # Option 2: Run all parametrized tests
    print("\n" + "="*80)
    print("Running all parametrized tests...")
    print("="*80)
    
    test_smiles = [
        "[NH3+]CCCCCC[NH3+]",  # Diammonium hexane
        "[NH3+]CCCC[NH3+]",    # Diammonium butane
        "[NH3+]CCCCCCCC[NH3+]", # Diammonium octane
        "[NH3+]CCCCCCCCCCCC[NH3+]", # Diammonium dodecane
        "[NH3+]CCCCCCCCCCCCCCCC[NH3+]", # Diammonium tetradecane
        "[NH3+]CCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium hexadecane
        "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium octadecane
        "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium nonadecane
        "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium eicosane
        "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium docosane
        "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium tricosane
        "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium tetracosane
        "[NH3+]CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC[NH3+]", # Diammonium pentacosane
        "[NH3+]CCC1CCC(CCC[NH3+])CC1", # Diammonium hexacosane
        "[NH3+]CCC1=CC=C(CCC[NH3+])C=C1", # Diammonium heptacosane
    ]
    
    passed = 0
    failed = 0
    
    for i, smiles in enumerate(test_smiles, 1):
        try:
            print(f"\n[{i}/{len(test_smiles)}] Testing: {smiles}")
            test_compression_changes_with_elongation(smiles)
            passed += 1
            print(f"✓ Test {i} passed")
        except AssertionError as e:
            failed += 1
            print(f"✗ Test {i} failed: {e}")
        except Exception as e:
            failed += 1
            print(f"✗ Test {i} error: {e}")
    
    print("\n" + "="*80)
    print(f"Test Summary: {passed} passed, {failed} failed out of {len(test_smiles)} tests")
    print("="*80)
    
    # Option 3: Test compression in actual DJ structures
    print("\n" + "="*80)
    print("Testing compression in 15 DJ structures...")
    print("="*80)
    dj_results = test_compression_in_dj_structures()
    
    if failed > 0:
        exit(1)
