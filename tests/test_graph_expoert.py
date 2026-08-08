"""
Test script for graph export and cavity deformation analysis.

Updated to use the new auto-extraction feature from cavity_deformation.py:
- get_deformation() now automatically extracts B/X species from cavity subgraph
- Supports mixed compositions (double perovskites, mixed halides)
- Handles both cuboctahedron (a_site) and antiprism (spacer_dj/spacer_rp) cavity types
- Antiprism cavities return kappa_equilateral and kappa_isosceles instead of single kappa
"""

from q2D_Materials.analyzer import q2D_analyzer
from q2D_Materials.utils.other.graph_pyvis_export import export_structure_pyvis, export_cavity_pyvis
from q2D_Materials.core.creator import q2D_creator
from ase.io import write, read
import pandas as pd
import numpy as np
import os

def analyze_structure(structure, name, save_cavities=False):
    """Analyze a structure and return comprehensive cavity data."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {name}")
    print(f"{'='*60}\n")
    
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    graph = analyzer.get_graph()
    export_structure_pyvis(graph, f'{name.lower().replace(" ", "_")}_structure.html')
    
    cavities = analyzer.get_cavities()
    print(f"Found {len(cavities)} cavities")
    
    # Export cavity graphs
    cavity_output_dir = f'{name.lower().replace(" ", "_")}_cavities'
    os.makedirs(cavity_output_dir, exist_ok=True)
    print(f"Exporting cavity graphs to {cavity_output_dir}/")
    
    for i, cavity in enumerate(cavities):
        if hasattr(cavity, 'subgraph') and cavity.subgraph is not None:
            try:
                export_cavity_pyvis(
                    cavity.subgraph,
                    i,
                    cavity_output_dir
                )
                print(f"  Exported {cavity.id} graph")
            except Exception as e:
                print(f"  Warning: Could not export graph for {cavity.id}: {e}")
    
    # Save cavities if requested
    if save_cavities:
        cavity_atoms_list = cavities.to_atoms()
        for i, cavity_atoms in enumerate(cavity_atoms_list):
            write(f'{name.lower().replace(" ", "_")}_cavity_{i}.xyz', cavity_atoms)
    
    # Collect cavity data
    cavity_data = []
    
    for cavity in cavities:
        data = {
            'id': cavity.id,
            'type': cavity.cavity_type,
            'volume': cavity.get_volume(),
        }
        
        # Get deformation metrics using auto-extraction (new feature from CODE_CHANGES_SUMMARY.md)
        # This automatically extracts B/X species from the cavity subgraph if not provided
        # Supports mixed compositions (double perovskites, mixed halides) automatically
        deformation = cavity.get_deformation(mode='delta')
        if deformation:
            data.update({
                'eta': deformation.get('eta'),
                'nu': deformation.get('nu'),
                'omega': deformation.get('omega'),
                'delta_volume': deformation.get('delta_volume'),
            })
            # Handle different kappa formats for different cavity types
            # Cuboctahedra (a_site) have 'kappa', antiprisms have 'kappa_equilateral' and 'kappa_isosceles'
            if 'kappa' in deformation:
                data['kappa'] = deformation.get('kappa')
            else:
                # For antiprism cavities, store both kappa values
                kappa_eq = deformation.get('kappa_equilateral')
                kappa_iso = deformation.get('kappa_isosceles')
                if kappa_eq is not None:
                    data['kappa_equilateral'] = kappa_eq
                if kappa_iso is not None:
                    data['kappa_isosceles'] = kappa_iso
                # Also compute a combined kappa for comparison purposes
                # Simple average if both are available, otherwise use the available one
                if kappa_eq is not None and kappa_iso is not None:
                    data['kappa'] = (kappa_eq + kappa_iso) / 2.0
                elif kappa_eq is not None:
                    data['kappa'] = kappa_eq
                elif kappa_iso is not None:
                    data['kappa'] = kappa_iso
                else:
                    data['kappa'] = 60.0  # Default fallback
        
        # Get absolute mode to see all angles
        absolute_deformation = cavity.get_deformation(mode='absolute')
        if absolute_deformation:
            # Handle triangle angles - different formats for different cavity types
            if 'triangle_angles' in absolute_deformation:
                data['triangle_angles'] = absolute_deformation['triangle_angles'].get('values', [])
                data['triangle_angles_mean'] = absolute_deformation['triangle_angles'].get('mean')
                data['triangle_angles_std'] = absolute_deformation['triangle_angles'].get('std')
            elif 'equilateral_angles' in absolute_deformation:
                # For antiprism cavities, combine equilateral and isosceles angles
                equi_angles = absolute_deformation['equilateral_angles'].get('values', [])
                iso_angles = absolute_deformation.get('isosceles_angles', {}).get('values', [])
                all_triangle_angles = equi_angles + iso_angles
                data['triangle_angles'] = all_triangle_angles
                if all_triangle_angles:
                    data['triangle_angles_mean'] = np.mean(all_triangle_angles)
                    data['triangle_angles_std'] = np.std(all_triangle_angles)
            if 'square_angles' in absolute_deformation:
                data['square_angles'] = absolute_deformation['square_angles'].get('values', [])
                data['square_angles_mean'] = absolute_deformation['square_angles'].get('mean')
                data['square_angles_std'] = absolute_deformation['square_angles'].get('std')
        
        # Get penetration depth for spacer cavities
        if cavity.cavity_type in ('spacer_rp', 'spacer_dj'):
            try:
                penetration = cavity.calculate_penetration_depth()
                if cavity.cavity_type == 'spacer_dj':
                    data['top_penetration'] = penetration.get('top_penetration')
                    data['bottom_penetration'] = penetration.get('bottom_penetration')
                else:  # spacer_rp
                    data['penetration_depth'] = penetration.get('penetration_depth')
            except Exception as e:
                print(f"Warning: Could not calculate penetration for {cavity.id}: {e}")
        
        cavity_data.append(data)
    
    return analyzer, cavities, cavity_data

def compare_structures(distorted_data, perfect_data):
    """Compare two structures side-by-side."""
    print(f"\n{'='*80}")
    print("COMPARISON: Distorted vs Perfect Structure")
    print(f"{'='*80}\n")
    
    # Overall statistics
    print("OVERALL STATISTICS:")
    print(f"  Distorted: {len(distorted_data)} cavities")
    print(f"  Perfect:   {len(perfect_data)} cavities")
    print()
    
    # Create DataFrames for easier comparison
    df_distorted = pd.DataFrame(distorted_data)
    df_perfect = pd.DataFrame(perfect_data)
    
    # Compare by cavity type
    print("CAVITY COUNTS BY TYPE:")
    if 'type' in df_distorted.columns and 'type' in df_perfect.columns:
        for cavity_type in set(df_distorted['type'].tolist() + df_perfect['type'].tolist()):
            count_dist = len(df_distorted[df_distorted['type'] == cavity_type])
            count_perf = len(df_perfect[df_perfect['type'] == cavity_type])
            print(f"  {cavity_type:15s}: Distorted={count_dist:2d}, Perfect={count_perf:2d}")
    else:
        print("  (no cavity type data)")
    print()
    
    # Compare deformation metrics (for cavities that have them)
    print("DEFORMATION METRICS COMPARISON:")
    print(f"{'Metric':<15} {'Distorted (mean)':<20} {'Perfect (mean)':<20} {'Difference':<15}")
    print("-" * 70)
    
    metrics_to_compare = ['eta', 'kappa', 'nu', 'omega', 'delta_volume']
    for metric in metrics_to_compare:
        if metric in df_distorted.columns and metric in df_perfect.columns:
            dist_values = df_distorted[metric].dropna()
            perf_values = df_perfect[metric].dropna()
            if len(dist_values) > 0 and len(perf_values) > 0:
                dist_mean = dist_values.mean()
                perf_mean = perf_values.mean()
                diff = dist_mean - perf_mean
                print(f"{metric:<15} {dist_mean:>18.4f}     {perf_mean:>18.4f}     {diff:>13.4f}")
    
    # Also compare kappa components for antiprism cavities if available
    if 'kappa_equilateral' in df_distorted.columns and 'kappa_equilateral' in df_perfect.columns:
        dist_kappa_eq = df_distorted['kappa_equilateral'].dropna()
        perf_kappa_eq = df_perfect['kappa_equilateral'].dropna()
        if len(dist_kappa_eq) > 0 and len(perf_kappa_eq) > 0:
            print(f"{'kappa_equilateral':<15} {dist_kappa_eq.mean():>18.4f}     {perf_kappa_eq.mean():>18.4f}     {dist_kappa_eq.mean()-perf_kappa_eq.mean():>13.4f}")
    
    if 'kappa_isosceles' in df_distorted.columns and 'kappa_isosceles' in df_perfect.columns:
        dist_kappa_iso = df_distorted['kappa_isosceles'].dropna()
        perf_kappa_iso = df_perfect['kappa_isosceles'].dropna()
        if len(dist_kappa_iso) > 0 and len(perf_kappa_iso) > 0:
            print(f"{'kappa_isosceles':<15} {dist_kappa_iso.mean():>18.4f}     {perf_kappa_iso.mean():>18.4f}     {dist_kappa_iso.mean()-perf_kappa_iso.mean():>13.4f}")
    print()
    
    # Compare penetration depths for spacer cavities
    print("PENETRATION DEPTH COMPARISON (Spacer Cavities):")
    if 'type' in df_distorted.columns and 'type' in df_perfect.columns:
        spacer_dist = df_distorted[df_distorted['type'].isin(['spacer_rp', 'spacer_dj'])]
        spacer_perf = df_perfect[df_perfect['type'].isin(['spacer_rp', 'spacer_dj'])]
    else:
        spacer_dist = pd.DataFrame()
        spacer_perf = pd.DataFrame()
    
    if not spacer_dist.empty and not spacer_perf.empty:
        if 'top_penetration' in spacer_dist.columns:
            dist_top = spacer_dist['top_penetration'].dropna()
            perf_top = spacer_perf['top_penetration'].dropna()
            if len(dist_top) > 0 and len(perf_top) > 0:
                print(f"  Top Penetration:")
                print(f"    Distorted: {dist_top.mean():.4f} Å (mean)")
                print(f"    Perfect:   {perf_top.mean():.4f} Å (mean)")
                print(f"    Difference: {dist_top.mean() - perf_top.mean():.4f} Å")
        if 'bottom_penetration' in spacer_dist.columns:
            dist_bottom = spacer_dist['bottom_penetration'].dropna()
            perf_bottom = spacer_perf['bottom_penetration'].dropna()
            if len(dist_bottom) > 0 and len(perf_bottom) > 0:
                print(f"  Bottom Penetration:")
                print(f"    Distorted: {dist_bottom.mean():.4f} Å (mean)")
                print(f"    Perfect:   {perf_bottom.mean():.4f} Å (mean)")
                print(f"    Difference: {dist_bottom.mean() - perf_bottom.mean():.4f} Å")
        if 'penetration_depth' in spacer_dist.columns:
            dist_rp = spacer_dist['penetration_depth'].dropna()
            perf_rp = spacer_perf['penetration_depth'].dropna()
            if len(dist_rp) > 0 and len(perf_rp) > 0:
                print(f"  Penetration Depth (RP):")
                print(f"    Distorted: {dist_rp.mean():.4f} Å (mean)")
                print(f"    Perfect:   {perf_rp.mean():.4f} Å (mean)")
                print(f"    Difference: {dist_rp.mean() - perf_rp.mean():.4f} Å")
    else:
        print("  No spacer cavities found in one or both structures")
    print()
    
    # Compare volumes
    print("VOLUME COMPARISON:")
    if 'volume' in df_distorted.columns and 'volume' in df_perfect.columns:
        dist_vol = df_distorted['volume'].dropna()
        perf_vol = df_perfect['volume'].dropna()
        if len(dist_vol) > 0 and len(perf_vol) > 0:
            print(f"  Distorted: {dist_vol.mean():.4f} Ų (mean), {dist_vol.sum():.4f} Ų (total)")
            print(f"  Perfect:   {perf_vol.mean():.4f} Ų (mean), {perf_vol.sum():.4f} Ų (total)")
            print(f"  Difference: {dist_vol.mean() - perf_vol.mean():.4f} Ų (mean)")
    print()
    
    # Detailed per-cavity comparison (for matching cavity IDs)
    print("DETAILED PER-CAVITY COMPARISON:")
    if 'id' in df_distorted.columns and 'id' in df_perfect.columns:
        common_ids = set(df_distorted['id']) & set(df_perfect['id'])
    else:
        common_ids = set()
    if common_ids:
        print(f"  Found {len(common_ids)} cavities with matching IDs\n")
        for cavity_id in sorted(common_ids):
            dist_row = df_distorted[df_distorted['id'] == cavity_id].iloc[0]
            perf_row = df_perfect[df_perfect['id'] == cavity_id].iloc[0]
            
            print(f"  {cavity_id} ({dist_row.get('type', '?')}):")
            if 'eta' in dist_row and pd.notna(dist_row['eta']):
                print(f"    Eta:    Distorted={dist_row['eta']:.4f}, Perfect={perf_row['eta']:.4f}, Δ={dist_row['eta']-perf_row['eta']:.4f}")
            # Handle kappa - can be single value (cuboctahedron) or split (antiprism)
            if 'kappa' in dist_row and pd.notna(dist_row['kappa']):
                print(f"    Kappa:  Distorted={dist_row['kappa']:.4f}, Perfect={perf_row['kappa']:.4f}, Δ={dist_row['kappa']-perf_row['kappa']:.4f}")
            if 'kappa_equilateral' in dist_row and pd.notna(dist_row['kappa_equilateral']):
                print(f"    Kappa (equilateral):  Distorted={dist_row['kappa_equilateral']:.4f}, Perfect={perf_row.get('kappa_equilateral', 'N/A')}, Δ={dist_row['kappa_equilateral']-perf_row.get('kappa_equilateral', 0):.4f}")
            if 'kappa_isosceles' in dist_row and pd.notna(dist_row['kappa_isosceles']):
                print(f"    Kappa (isosceles):  Distorted={dist_row['kappa_isosceles']:.4f}, Perfect={perf_row.get('kappa_isosceles', 'N/A')}, Δ={dist_row['kappa_isosceles']-perf_row.get('kappa_isosceles', 0):.4f}")
            if 'nu' in dist_row and pd.notna(dist_row['nu']):
                print(f"    Nu:     Distorted={dist_row['nu']:.4f}, Perfect={perf_row['nu']:.4f}, Δ={dist_row['nu']-perf_row['nu']:.4f}")
            if 'omega' in dist_row and pd.notna(dist_row['omega']):
                print(f"    Omega:  Distorted={dist_row['omega']:.4f}, Perfect={perf_row['omega']:.4f}, Δ={dist_row['omega']-perf_row['omega']:.4f}")
            if 'delta_volume' in dist_row and pd.notna(dist_row['delta_volume']):
                print(f"    ΔVol:   Distorted={dist_row['delta_volume']:.4f}, Perfect={perf_row['delta_volume']:.4f}, Δ={dist_row['delta_volume']-perf_row['delta_volume']:.4f}")
            if 'volume' in dist_row and pd.notna(dist_row['volume']):
                print(f"    Volume: Distorted={dist_row['volume']:.4f}, Perfect={perf_row['volume']:.4f}, Δ={dist_row['volume']-perf_row['volume']:.4f}")
            
            # Show triangle angles in detail
            if 'triangle_angles' in dist_row and isinstance(dist_row['triangle_angles'], list):
                dist_tri = dist_row['triangle_angles']
                perf_tri = perf_row.get('triangle_angles', []) if isinstance(perf_row.get('triangle_angles'), list) else []
                print(f"    Triangle Angles:")
                print(f"      Distorted: {len(dist_tri)} angles", end="")
                if dist_tri:
                    print(f", mean={dist_row.get('triangle_angles_mean', 'N/A'):.4f}°, std={dist_row.get('triangle_angles_std', 'N/A'):.4f}°")
                    print(f"      Values: {[f'{a:.2f}' for a in sorted(dist_tri)]}")
                    print(f"      Min: {min(dist_tri):.2f}°, Max: {max(dist_tri):.2f}°")
                if perf_tri:
                    print(f"      Perfect: {len(perf_tri)} angles, mean={perf_row.get('triangle_angles_mean', 'N/A'):.4f}°, std={perf_row.get('triangle_angles_std', 'N/A'):.4f}°")
                    print(f"      Values: {[f'{a:.2f}' for a in sorted(perf_tri)]}")
                    print(f"      Min: {min(perf_tri):.2f}°, Max: {max(perf_tri):.2f}°")
            
            # Show square angles in detail
            if 'square_angles' in dist_row and isinstance(dist_row['square_angles'], list):
                dist_sq = dist_row['square_angles']
                perf_sq = perf_row.get('square_angles', []) if isinstance(perf_row.get('square_angles'), list) else []
                print(f"    Square Angles:")
                sq_mean_dist = dist_row.get('square_angles_mean')
                sq_std_dist = dist_row.get('square_angles_std')
                print(f"      Distorted: {len(dist_sq)} angles", end="")
                if sq_mean_dist is not None:
                    print(f", mean={sq_mean_dist:.4f}°", end="")
                if sq_std_dist is not None:
                    print(f", std={sq_std_dist:.4f}°", end="")
                print()
                if dist_sq:
                    print(f"      Values: {[f'{a:.2f}' for a in sorted(dist_sq)]}")
                    print(f"      Min: {min(dist_sq):.2f}°, Max: {max(dist_sq):.2f}°")
                if perf_sq:
                    sq_mean_perf = perf_row.get('square_angles_mean')
                    sq_std_perf = perf_row.get('square_angles_std')
                    print(f"      Perfect: {len(perf_sq)} angles", end="")
                    if sq_mean_perf is not None:
                        print(f", mean={sq_mean_perf:.4f}°", end="")
                    if sq_std_perf is not None:
                        print(f", std={sq_std_perf:.4f}°", end="")
                    print()
                    print(f"      Values: {[f'{a:.2f}' for a in sorted(perf_sq)]}")
                    print(f"      Min: {min(perf_sq):.2f}°, Max: {max(perf_sq):.2f}°")
            print()
    else:
        print("  No matching cavity IDs found between structures")
    print()

# ============================================================================
# MAIN ANALYSIS
# ============================================================================

# Analyze distorted structure
structure_distorted = read('structure.vasp')
analyzer_dist, cavities_dist, data_dist = analyze_structure(
    structure_distorted, 
    "Distorted", 
    save_cavities=True
)

# Analyze perfect structure
q2d = q2D_creator()
structure_perfect = q2d.create_structure(
    A_ions=['Cs','MA','FA', 'K'],
    B_ions='Pb',
    X_ions='I',
    structure_type='bulk',
    template='cubic',
    layer_sequence='DJ',
    spacer='[NH3+]CCCC[NH3+]',
    thickness=2,
    xy_expansion=(2, 2),
)

write('perfect.vasp', structure_perfect, format='vasp', sort=True)

analyzer_perf, cavities_perf, data_perf = analyze_structure(
    structure_perfect, 
    "Perfect", 
    save_cavities=True
)

# Compare the two structures
compare_structures(data_dist, data_perf)