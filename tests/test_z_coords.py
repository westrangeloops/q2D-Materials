#!/usr/bin/env python3
"""
Check Z-coordinates of octahedra in different supercells.
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.core.creator import q2D_creator

for size in [(1, 1, 1), (2, 2, 1), (3, 3, 1), (4, 4, 1), (5, 5, 1)]:
    print(f"\n{'='*50}")
    print(f"{size[0]}×{size[1]} DJ structure (n=2)")
    print('='*50)
    
    q2d = q2D_creator()
    
    structure = q2d.create_structure(
        structure_type="bulk",
        template="cubic",
        layer_sequence="DJ",
        thickness=2,
        A_ions="MA",
        B_ions="Pb",
        X_ions="I",
        spacer="[NH3+]CCC[NH3+]",
        glazer_angles=[0, 0, 0],
        glazer_pattern=["0", "0", "0"],
    )
    
    structure = structure * size
    
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    
    graph = analyzer.get_graph()
    
    # Get B-site Z-coordinates
    b_z_coords = []
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'atom' and data.get('symbol') == 'Pb':
            b_z_coords.append(data.get('z'))
    
    b_z_coords = sorted(b_z_coords)
    
    print(f"B-site Z-coordinates ({len(b_z_coords)} atoms):")
    print(f"  Min: {min(b_z_coords):.4f}")
    print(f"  Max: {max(b_z_coords):.4f}")
    print(f"  Range: {max(b_z_coords) - min(b_z_coords):.4f}")
    
    # Check for gaps
    if len(b_z_coords) > 1:
        z_diffs = np.diff(b_z_coords)
        print(f"  Z-differences: min={min(z_diffs):.4f}, max={max(z_diffs):.4f}, median={np.median(z_diffs):.4f}")
        print(f"  Gap threshold (2×median): {np.median(z_diffs) * 2:.4f}")
        
        large_gaps = [i for i, diff in enumerate(z_diffs) if diff > np.median(z_diffs) * 2]
        print(f"  Large gaps (>{np.median(z_diffs) * 2:.4f}): {len(large_gaps)}")
        if large_gaps:
            for gap_idx in large_gaps:
                print(f"    Gap at index {gap_idx}: {z_diffs[gap_idx]:.4f}")
