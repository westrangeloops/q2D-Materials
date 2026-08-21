#!/usr/bin/env python3
"""Test script for spacer molecular analysis (volume, penetration, compression).

Creates multiple DJ structures with different spacers and prints detailed analysis.
"""

import numpy as np
from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.modifier import GraphView
from q2D_Materials.analyzer.molecular_processing.spacer_analysis import SpacerAnalysis


def print_spacer_analysis(result, name, spacer_idx=0, cavity=None):
    """Print detailed spacer analysis results.
    
    Parameters
    ----------
    result : SpacerAnalysisResult
        Spacer analysis result
    name : str
        Structure name
    spacer_idx : int
        Spacer index
    cavity : Cavity, optional
        Associated cavity for penetration depth (if available)
    """
    print(f"\n{'='*80}")
    print(f"SPACER ANALYSIS: {name} (Spacer {spacer_idx})")
    print(f"{'='*80}")
    
    # ========== VOLUME ==========
    print(f"\n📦 MOLECULAR VOLUME:")
    print(f"  Van der Waals volume: {result.vdw_volume:.2f} Å³")
    print(f"  Convex hull volume: {result.convex_hull_volume:.2f} Å³")
    
    if result.vdw_volume > 0 and result.convex_hull_volume > 0:
        volume_ratio = result.convex_hull_volume / result.vdw_volume
        print(f"  Volume ratio (hull/vdw): {volume_ratio:.3f}")
        if volume_ratio < 0.5:
            print("  → Significant atomic overlap")
        elif volume_ratio < 0.8:
            print("  → Moderate atomic overlap")
        else:
            print("  → Low atomic overlap")
    
    # ========== PENETRATION (from cavity) ==========
    print(f"\n⬇️  PENETRATION DEPTH:")
    if cavity is not None and cavity.cavity_type in ('spacer_rp', 'spacer_dj'):
        try:
            penetration = cavity.calculate_penetration_depth()
            if cavity.cavity_type == 'spacer_dj':
                top_pen = penetration.get('top_penetration', 0.0)
                bottom_pen = penetration.get('bottom_penetration', 0.0)
                print(f"  Top penetration: {top_pen:.3f} Å")
                print(f"  Bottom penetration: {bottom_pen:.3f} Å")
                avg_pen = (top_pen + bottom_pen) / 2.0
                print(f"  Average penetration: {avg_pen:.3f} Å")
                print(f"  Is penetrating: {avg_pen < 0}")
            else:  # spacer_rp
                pen = penetration.get('penetration_depth', 0.0)
                print(f"  Penetration depth: {pen:.3f} Å")
                print(f"  Is penetrating: {pen < 0}")
        except Exception as e:
            print(f"  ⚠️  Could not calculate penetration: {e}")
    else:
        print("  ⚠️  No cavity available for penetration calculation")
    
    # ========== COMPRESSION ==========
    print(f"\n📏 COMPRESSION:")
    if result.euclidean_distance > 0:
        print(f"  Euclidean distance: {result.euclidean_distance:.2f} Å")
        print(f"  Path length: {result.path_length:.2f} Å")
        print(f"  Ideal extended length: {result.ideal_extended_length:.2f} Å")
        print(f"  Compression factor: {result.compression_factor:.3f}")
        print(f"  Is compressed: {result.is_compressed}")
        
        # Interpretation
        if 0.8 <= result.compression_factor <= 0.9:
            print("  → Fully extended (typical for sp3 chains)")
        elif result.compression_factor < 0.8:
            print("  → Compressed/folded (shorter than expected)")
        elif result.compression_factor > 0.9:
            print("  → Stretched (longer than typical, may indicate strain)")
    else:
        print("  ⚠️  No compression data available (requires 2 terminal N atoms)")
    
    # ========== BACKBONE ==========
    print(f"\n🔗 BACKBONE:")
    print(f"  Backbone length: {result.backbone_length} atoms")
    if result.backbone_symbols:
        print(f"  Backbone sequence: {''.join(result.backbone_symbols[:20])}")
        if len(result.backbone_symbols) > 20:
            print(f"    ... ({len(result.backbone_symbols) - 20} more atoms)")
    
    # ========== SIDE CHAINS ==========
    print(f"\n🌿 SIDE CHAINS:")
    print(f"  Number of side chains: {result.n_side_chains}")
    if result.branching_points:
        print(f"  Branching points: {result.branching_points}")
    if result.side_chains:
        for i, sc in enumerate(result.side_chains):
            print(f"  Side chain {i}:")
            print(f"    Attachment point: {sc['attachment_idx']}")
            print(f"    Chain length: {sc['length']} atoms")
            print(f"    Chain atoms: {sc['chain_atoms'][:10]}")
            if len(sc['chain_atoms']) > 10:
                print(f"      ... ({len(sc['chain_atoms']) - 10} more)")
    
    # ========== METADATA ==========
    print(f"\n📋 METADATA:")
    print(f"  Spacer type: {result.spacer_type}")
    print(f"  Terminal nitrogens: {len(result.terminal_nitrogens)}")
    print(f"  Molecule indices: {len(result.molecule_indices)} atoms")
    
    # ========== FULL REPRESENTATION ==========
    print(f"\n📄 Full Result:")
    print(result)


def test_structure(analyzer, name, spacer_smiles):
    """Test spacer analysis for a structure."""
    print(f"\n{'#'*80}")
    print(f"# TESTING: {name}")
    print(f"# Spacer: {spacer_smiles}")
    print(f"{'#'*80}")
    
    # Create GraphView
    view = GraphView(analyzer)
    spacers = view.spacers.list()
    
    # Get cavities for penetration depth
    cavities = analyzer.get_cavities()
    spacer_cavities = [c for c in cavities if c.cavity_type in ('spacer_rp', 'spacer_dj')]
    
    print(f"\nFound {len(spacers)} spacer molecules")
    print(f"Found {len(spacer_cavities)} spacer cavities")
    
    if len(spacers) == 0:
        print("⚠️  No spacers found in structure!")
        return
    
    # Analyze each spacer
    for i, spacer in enumerate(spacers):
        try:
            # Create analyzer instance
            analysis = SpacerAnalysis(spacer, analyzer, spacer_type="DJ")
            
            # Compute results
            result = analysis.compute()
            
            # Find associated cavity (if available)
            cavity = spacer_cavities[i] if i < len(spacer_cavities) else None
            
            # Print detailed analysis
            print_spacer_analysis(result, name, spacer_idx=i, cavity=cavity)
            
        except Exception as e:
            print(f"\n❌ ERROR analyzing spacer {i}: {e}")
            import traceback
            traceback.print_exc()


def main():
    """Main test function."""
    print("="*80)
    print("SPACER MOLECULAR ANALYSIS TEST")
    print("="*80)
    print("\nTesting volume, penetration, and compression calculations")
    print("for multiple DJ structures with different spacers.\n")
    
    creator = q2D_creator()
    
    # Test structures with different spacer lengths
    test_cases = [
        {
            "name": "DJ Structure 1 - Short Spacer",
            "spacer": "[NH3+]CC[NH3+]",  # 1,2-ethanediammonium
            "X_ions": "I",
        },
        {
            "name": "DJ Structure 2 - Medium Spacer",
            "spacer": "[NH3+]CCCC[NH3+]",  # 1,4-butanediammonium
            "X_ions": "I",
        },
        {
            "name": "DJ Structure 3 - Long Spacer",
            "spacer": "[NH3+]CCCCCC[NH3+]",  # 1,6-hexanediammonium
            "X_ions": "I",
        },
        {
            "name": "DJ Structure 4 - Very Long Spacer",
            "spacer": "[NH3+]CCCCCCCC[NH3+]",  # 1,8-octanediammonium
            "X_ions": "I",
        },
        {
            "name": "DJ Structure 5 - Bromide",
            "spacer": "[NH3+]CCCC[NH3+]",  # 1,4-butanediammonium
            "X_ions": "Br",
        },
    ]
    
    # Create and test each structure
    for test_case in test_cases:
        try:
            print(f"\n{'='*80}")
            print(f"Creating {test_case['name']}...")
            print(f"{'='*80}")
            
            structure = creator.create_structure(
                structure_type="bulk",
                template="cubic",
                layer_sequence="DJ",
                thickness=2,
                A_ions="MA",
                B_ions="Pb",
                X_ions=test_case["X_ions"],
                spacer=test_case["spacer"],
                xy_expansion=(1, 1),
            )
            
            print(f"Structure created: {len(structure)} atoms")
            
            # Analyze structure
            print(f"Analyzing structure...")
            analyzer = q2D_analyzer(structure)
            analyzer.analyze()
            
            # Test spacer analysis
            test_structure(analyzer, test_case["name"], test_case["spacer"])
            
        except Exception as e:
            print(f"\n❌ ERROR with {test_case['name']}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    
    # Re-analyze all structures for summary
    all_results = []
    for test_case in test_cases:
        try:
            structure = creator.create_structure(
                structure_type="bulk",
                template="cubic",
                layer_sequence="DJ",
                thickness=2,
                A_ions="MA",
                B_ions="Pb",
                X_ions=test_case["X_ions"],
                spacer=test_case["spacer"],
                xy_expansion=(1, 1),
            )
            
            analyzer = q2D_analyzer(structure)
            analyzer.analyze()
            view = GraphView(analyzer)
            spacers = view.spacers.list()
            
            for spacer in spacers:
                analysis = SpacerAnalysis(spacer, analyzer, spacer_type="DJ")
                result = analysis.compute()
                all_results.append({
                    'name': test_case['name'],
                    'spacer': test_case['spacer'],
                    'result': result
                })
        except Exception as e:
            print(f"⚠️  Skipping {test_case['name']} in summary: {e}")
    
    if all_results:
        print(f"\nAnalyzed {len(all_results)} spacer molecules")
        
        # Volume statistics
        volumes = [r['result'].vdw_volume for r in all_results if r['result'].vdw_volume > 0]
        if volumes:
            print(f"\n📦 Volume Statistics:")
            print(f"  Mean vdw volume: {np.mean(volumes):.2f} Å³")
            print(f"  Min vdw volume: {np.min(volumes):.2f} Å³")
            print(f"  Max vdw volume: {np.max(volumes):.2f} Å³")
            print(f"  Std vdw volume: {np.std(volumes):.2f} Å³")
        
        # Penetration statistics (from cavities)
        penetrations = []
        for test_case in test_cases:
            try:
                structure = creator.create_structure(
                    structure_type="bulk",
                    template="cubic",
                    layer_sequence="DJ",
                    thickness=2,
                    A_ions="MA",
                    B_ions="Pb",
                    X_ions=test_case["X_ions"],
                    spacer=test_case["spacer"],
                    xy_expansion=(1, 1),
                )
                
                analyzer = q2D_analyzer(structure)
                analyzer.analyze()
                cavities = analyzer.get_cavities()
                spacer_cavities = [c for c in cavities if c.cavity_type in ('spacer_rp', 'spacer_dj')]
                
                for cavity in spacer_cavities:
                    try:
                        penetration = cavity.calculate_penetration_depth()
                        if cavity.cavity_type == 'spacer_dj':
                            penetrations.append(penetration.get('top_penetration', 0.0))
                            penetrations.append(penetration.get('bottom_penetration', 0.0))
                        else:  # spacer_rp
                            penetrations.append(penetration.get('penetration_depth', 0.0))
                    except Exception:
                        pass
            except Exception:
                pass
        
        if penetrations:
            print(f"\n⬇️  Penetration Statistics (from cavities):")
            print(f"  Mean penetration: {np.mean(penetrations):.3f} Å")
            print(f"  Min penetration: {np.min(penetrations):.3f} Å")
            print(f"  Max penetration: {np.max(penetrations):.3f} Å")
            penetrating_count = sum(1 for p in penetrations if p < 0)
            print(f"  Penetrating NH3 groups: {penetrating_count}/{len(penetrations)}")
        
        # Compression statistics
        compressions = [r['result'].compression_factor for r in all_results if r['result'].compression_factor > 0]
        if compressions:
            print(f"\n📏 Compression Statistics:")
            print(f"  Mean compression factor: {np.mean(compressions):.3f}")
            print(f"  Min compression factor: {np.min(compressions):.3f}")
            print(f"  Max compression factor: {np.max(compressions):.3f}")
            compressed_count = sum(1 for r in all_results if r['result'].is_compressed)
            print(f"  Compressed spacers: {compressed_count}/{len(all_results)}")
        
        # Backbone statistics
        backbone_lengths = [r['result'].backbone_length for r in all_results if r['result'].backbone_length > 0]
        if backbone_lengths:
            print(f"\n🔗 Backbone Statistics:")
            print(f"  Mean backbone length: {np.mean(backbone_lengths):.1f} atoms")
            print(f"  Min backbone length: {np.min(backbone_lengths)} atoms")
            print(f"  Max backbone length: {np.max(backbone_lengths)} atoms")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
