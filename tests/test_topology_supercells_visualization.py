#!/usr/bin/env python3
"""
Test topology classification with HTML visualization for different supercell sizes.

Creates DJ structures (n=2) with 1×1, 2×2, 3×3, 4×4, and 5×5 supercells
and exports graph visualizations to HTML files.
"""

import sys
from pathlib import Path
import numpy as np

# Add parent directory to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.utils.other.graph_pyvis_export import export_structure_pyvis


def create_and_analyze_dj(supercell_size: tuple, output_name: str):
    """
    Create a DJ structure with given supercell size and export HTML visualization.
    
    Parameters
    ----------
    supercell_size : tuple
        (nx, ny, nz) supercell dimensions
    output_name : str
        Name for the output HTML file (without extension)
    """
    print("="*70)
    print(f"Creating {supercell_size[0]}×{supercell_size[1]} DJ structure (n=2)")
    print("="*70)
    
    # Create base structure
    q2d = q2D_creator()
    
    structure = q2d.create_structure(
        structure_type="bulk",
        template="cubic",
        layer_sequence="DJ",
        thickness=2,
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        spacer="[NH3+]CCC[NH3+]",  # propylamine diamine
        glazer_angles=[0, 0, 0],
        glazer_pattern=["0", "0", "0"],
    )
    
    # Create supercell
    structure = structure * supercell_size

    print(f"\nStructure created:")
    print(f"  Formula: {structure.get_chemical_formula()}")
    print(f"  Cell: {structure.cell.cellpar()}")
    print(f"  Atoms: {len(structure)}")

    # Save structure as VASP file: wrap into cell, unique species + counts on lines 6-7, atoms ordered by element
    from collections import Counter
    from ase import Atoms
    from ase.io import write
    copy = structure.copy()
    copy.wrap()
    symbols = copy.get_chemical_symbols()
    positions = copy.get_positions()
    counts = Counter(symbols)
    symbol_count = sorted(counts.items())  # e.g. [("C", 100), ("H", 450), ("I", 175), ("N", 75), ("Pb", 50)]
    # Reorder so atoms match symbol_count order (all of first species, then second, etc.)
    order = []
    for sym, _ in symbol_count:
        order.extend(i for i, s in enumerate(symbols) if s == sym)
    # Build plain ASE Atoms (q2DStructure.__getitem__ would fail on copy[order])
    atoms_ordered = Atoms(
        symbols=[symbols[i] for i in order],
        positions=positions[order],
        cell=copy.cell,
        pbc=copy.pbc,
    )
    vasp_path = ROOT / "tests" / f"{output_name}.vasp"
    write(str(vasp_path), atoms_ordered, format="vasp", vasp5=True, symbol_count=symbol_count)
    print(f"  Saved structure to: {vasp_path}")
    
    # Analyze
    print("\n" + "="*70)
    print("Analyzing structure")
    print("="*70)
    
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    
    # Get octahedra info
    octahedra = analyzer.get_octahedra()
    print(f"\nOctahedra detected: {len(octahedra)}")
    
    # Get graph
    graph = analyzer.get_graph()
    
    # Get neighbor indices from graph metadata
    neighbor_indices = graph.graph.get('neighbor_indices', [])
    print(f"Neighbor indices list length: {len(neighbor_indices)}")
    
    # Import the classification function
    from q2D_Materials.analyzer.octahedral_processing.octahedral_detection import (
        classify_atoms_by_topology,
        find_shared_atoms
    )
    
    # Run classification
    print("\n" + "="*70)
    print("Running topology classification")
    print("="*70)
    
    # Get atom positions and center indices for classification
    atom_positions = np.array([structure.positions[i] for i in range(len(structure))])
    center_atom_indices = []
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'octahedron':
            # Find center atom
            for neighbor in graph.neighbors(node):
                edge_data = graph.get_edge_data(node, neighbor)
                if edge_data and edge_data.get('role') == 'center':
                    center_atom_indices.append(data.get('vasp_index', 0))
                    break
    
    atom_classification, layer_membership, octahedra_geometries = classify_atoms_by_topology(
        neighbor_indices,
        atom_positions=atom_positions,
        center_atom_indices=center_atom_indices,
        cell=structure.cell.array
    )
    
    # Find shared atoms
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    # Print statistics
    print("\n" + "="*70)
    print("Classification Results")
    print("="*70)
    
    # Count atom types
    terminal_count = sum(1 for cls in atom_classification.values() if cls == 'terminal')
    equatorial_count = sum(1 for cls in atom_classification.values() if cls == 'equatorial')
    axial_count = sum(1 for cls in atom_classification.values() if cls == 'axial_interlayer')
    
    print(f"\nAtom classification:")
    print(f"  Terminal: {terminal_count}")
    print(f"  Equatorial: {equatorial_count}")
    print(f"  Axial interlayer: {axial_count}")
    
    # Count layers
    unique_layers = set(layer_membership.values())
    print(f"\nLayer membership:")
    print(f"  Total layers: {len(unique_layers)}")
    for layer_id in sorted(unique_layers):
        oct_count = sum(1 for layer in layer_membership.values() if layer == layer_id)
        print(f"  Layer {layer_id}: {oct_count} octahedra")
    
    # Analyze sharing patterns
    print(f"\nSharing patterns:")
    sharing_counts = {}
    for pair, atoms in shared_atoms.items():
        count = len(atoms)
        if count not in sharing_counts:
            sharing_counts[count] = 0
        sharing_counts[count] += 1
    
    for count in sorted(sharing_counts.keys()):
        print(f"  {sharing_counts[count]} pairs share {count} atom(s)")
    
    # Export HTML visualization
    print("\n" + "="*70)
    print("Exporting HTML visualization")
    print("="*70)
    
    output_path = ROOT / "tests" / f"{output_name}.html"
    try:
        export_structure_pyvis(
            graph,
            str(output_path),
            exclude_cavities=True
        )
        print(f"\n✓ HTML visualization saved to: {output_path}")
    except Exception as e:
        print(f"\n✗ Failed to export HTML: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n")
    return analyzer, graph


def main():
    """Run tests for all supercell sizes."""
    
    supercell_configs = [
        ((1, 1, 1), "dj_1x1_n2"),
        ((2, 2, 1), "dj_2x2_n2"),
        ((3, 3, 1), "dj_3x3_n2"),
        ((4, 4, 1), "dj_4x4_n2"),
        ((5, 5, 1), "dj_5x5_n2"),
    ]
    
    results = {}
    
    for supercell_size, output_name in supercell_configs:
        try:
            analyzer, graph = create_and_analyze_dj(supercell_size, output_name)
            results[output_name] = {
                'success': True,
                'analyzer': analyzer,
                'graph': graph
            }
        except Exception as e:
            print(f"\n✗ FAILED for {output_name}: {e}")
            import traceback
            traceback.print_exc()
            results[output_name] = {
                'success': False,
                'error': str(e)
            }
        
        print("\n" + "="*70 + "\n")
    
    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, result in results.items():
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{name}: {status}")
        if not result['success']:
            print(f"  Error: {result['error']}")
    
    print("\n" + "="*70)
    print(f"HTML files saved in: {ROOT / 'tests'}")
    print("="*70)


if __name__ == "__main__":
    main()
