import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def extract_layer_data(base_dir):
    """
    Extract layer distortion statistics from all layers_ontology.json files.
    """
    data_records = []
    processed_files = 0
    error_count = 0
    
    print("Starting layer data extraction...")
    
    # Get all analysis directories
    analysis_dirs = [d for d in os.listdir(base_dir) if d.endswith('_analysis')]
    
    # Filter out utility directories
    utility_dirs = ['energy_analysis', 'lattice_parameter_analysis', 'penetration_depth', 'error_calculations']
    analysis_dirs = [d for d in analysis_dirs if d not in utility_dirs]
    
    print(f"Found {len(analysis_dirs)} experiment directories to process")
    
    for analysis_dir in analysis_dirs:
        try:
            # Parse experiment information
            experiment_name = analysis_dir.replace('_analysis', '')
            
            # Extract halogen
            if 'Cl3' in experiment_name:
                halogen = 'Cl'
            elif 'Br3' in experiment_name:
                halogen = 'Br'
            elif 'I3' in experiment_name:
                halogen = 'I'
            else:
                print(f"Warning: Could not determine halogen for {experiment_name}")
                continue
            
            # Extract number of layers
            if '_n1_' in experiment_name:
                n_layers = 1
            elif '_n2_' in experiment_name:
                n_layers = 2
            elif '_n3_' in experiment_name:
                n_layers = 3
            else:
                print(f"Warning: Could not determine n_layers for {experiment_name}")
                continue
            
            # Load layers ontology JSON
            layers_json_path = os.path.join(base_dir, analysis_dir, f"{experiment_name}_layers_ontology.json")
            
            if not os.path.exists(layers_json_path):
                print(f"Warning: layers_ontology.json not found for {experiment_name}")
                error_count += 1
                continue
            
            with open(layers_json_path, 'r') as f:
                layers_data = json.load(f)
            
            # Extract layer distortion statistics
            layer_stats = layers_data.get('layers_analysis', {}).get('layer_distortion_stats', {})
            
            if not layer_stats:
                print(f"Warning: No layer distortion stats found for {experiment_name}")
                error_count += 1
                continue
            
            # Process each layer
            for layer_key, layer_data in layer_stats.items():
                layer_number = int(layer_key.replace('layer_', ''))
                
                # Extract all properties with mean and std
                properties = ['zeta', 'delta', 'sigma', 'theta_mean', 'theta_min', 'theta_max', 
                             'mean_bond_distance', 'bond_distance_variance', 'octahedral_volume']
                
                layer_record = {
                    'experiment_name': experiment_name,
                    'halogen': halogen,
                    'n_layers': n_layers,
                    'layer_number': layer_number,
                    'octahedra_count': layer_data.get('zeta', {}).get('count', 0)
                }
                
                # Extract mean and std for each property
                for prop in properties:
                    if prop in layer_data:
                        layer_record[f'{prop}_mean'] = layer_data[prop].get('mean', np.nan)
                        layer_record[f'{prop}_std'] = layer_data[prop].get('std', np.nan)
                        layer_record[f'{prop}_min'] = layer_data[prop].get('min', np.nan)
                        layer_record[f'{prop}_max'] = layer_data[prop].get('max', np.nan)
                        layer_record[f'{prop}_count'] = layer_data[prop].get('count', np.nan)
                    else:
                        layer_record[f'{prop}_mean'] = np.nan
                        layer_record[f'{prop}_std'] = np.nan
                        layer_record[f'{prop}_min'] = np.nan
                        layer_record[f'{prop}_max'] = np.nan
                        layer_record[f'{prop}_count'] = np.nan
                
                data_records.append(layer_record)
            
            processed_files += 1
            
        except Exception as e:
            print(f"Error processing {analysis_dir}: {e}")
            error_count += 1
            continue
    
    print(f"Successfully processed {processed_files} files")
    print(f"Errors encountered: {error_count}")
    
    df = pd.DataFrame(data_records)
    
    if len(df) > 0:
        print(f"\nLayer data summary:")
        print(f"Total layer records: {len(df)}")
        print(f"Halogens: {sorted(df['halogen'].unique())}")
        print(f"N_layers: {sorted(df['n_layers'].unique())}")
        print(f"Layer numbers: {sorted(df['layer_number'].unique())}")
        
        # Show distribution
        print(f"\nDistribution by halogen and n_layers:")
        distribution = df.groupby(['halogen', 'n_layers', 'layer_number']).size().reset_index(name='count')
        for _, row in distribution.iterrows():
            print(f"  {row['halogen']}-n{row['n_layers']}-layer{row['layer_number']}: {row['count']} experiments")
    
    return df

def create_layer_comparison_plots(df, output_dir):
    """
    Create comprehensive layer comparison plots as requested.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define properties to analyze
    properties = ['zeta', 'delta', 'sigma', 'theta_mean', 'theta_min', 'theta_max', 
                  'mean_bond_distance', 'bond_distance_variance', 'octahedral_volume']
    
    # Property labels for better visualization
    property_labels = {
        'zeta': 'ζ (Zeta) Distortion',
        'delta': 'δ (Delta) Distortion',
        'sigma': 'σ (Sigma) Angular Variance',
        'theta_mean': 'θ Mean (degrees)',
        'theta_min': 'θ Min (degrees)',
        'theta_max': 'θ Max (degrees)',
        'mean_bond_distance': 'Mean Bond Distance (Å)',
        'bond_distance_variance': 'Bond Distance Variance',
        'octahedral_volume': 'Octahedral Volume (Å³)'
    }
    
    # Set style
    plt.style.use('default')
    colors = {'Cl': '#2E8B57', 'Br': '#CD853F', 'I': '#8B008B'}
    
    # ============================================================================
    # PART 1: Layer_1 comparison across n=1, n=2, n=3 for all halogens
    # ============================================================================
    print("Creating Layer_1 comparison plots...")
    
    layer_1_data = df[df['layer_number'] == 1].copy()
    
    if len(layer_1_data) > 0:
        # Create a large figure with subplots for all properties
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Layer 1 Properties Comparison\nAcross Halogens (Cl, Br, I) and Number of Layers (n=1, n=2, n=3)', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        axes = axes.flatten()
        
        for idx, prop in enumerate(properties):
            ax = axes[idx]
            
            # Prepare data for plotting
            plot_data = []
            for halogen in ['Cl', 'Br', 'I']:
                for n in [1, 2, 3]:
                    subset = layer_1_data[(layer_1_data['halogen'] == halogen) & 
                                         (layer_1_data['n_layers'] == n)]
                    
                    means = subset[f'{prop}_mean'].dropna()
                    stds = subset[f'{prop}_std'].dropna()
                    
                    for mean_val, std_val in zip(means, stds):
                        plot_data.append({
                            'halogen': halogen,
                            'n_layers': n,
                            'group': f'{halogen}-n{n}',
                            'mean': mean_val,
                            'std': std_val
                        })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Create violin plot with means and stds
                sns.violinplot(data=plot_df, x='n_layers', y='mean', hue='halogen', 
                             palette=colors, ax=ax, alpha=0.7)
                
                # Add error bars for std
                for halogen in ['Cl', 'Br', 'I']:
                    for n in [1, 2, 3]:
                        subset = plot_df[(plot_df['halogen'] == halogen) & (plot_df['n_layers'] == n)]
                        if len(subset) > 0:
                            mean_val = subset['mean'].mean()
                            std_val = subset['std'].mean()
                            x_pos = n - 1 + (list(colors.keys()).index(halogen) - 1) * 0.25
                            ax.errorbar(x_pos, mean_val, yerr=std_val, 
                                      color=colors[halogen], alpha=0.8, capsize=3)
            
            ax.set_title(property_labels[prop], fontweight='bold')
            ax.set_xlabel('Number of Layers (n)')
            ax.set_ylabel('Property Value')
            ax.grid(True, alpha=0.3)
            
            if idx == 0:  # Only show legend on first subplot
                ax.legend(title='Halogen', loc='upper right')
            else:
                ax.get_legend().remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_1_comparison_all_properties.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # ============================================================================
    # PART 2: Layer_2 comparison for n=2 and n=3 only
    # ============================================================================
    print("Creating Layer_2 comparison plots...")
    
    layer_2_data = df[df['layer_number'] == 2].copy()
    
    if len(layer_2_data) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Layer 2 Properties Comparison\nAcross Halogens (Cl, Br, I) for n=2 and n=3 Only', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        axes = axes.flatten()
        
        for idx, prop in enumerate(properties):
            ax = axes[idx]
            
            # Prepare data for plotting (only n=2 and n=3)
            plot_data = []
            for halogen in ['Cl', 'Br', 'I']:
                for n in [2, 3]:  # Only n=2 and n=3 have layer_2
                    subset = layer_2_data[(layer_2_data['halogen'] == halogen) & 
                                         (layer_2_data['n_layers'] == n)]
                    
                    means = subset[f'{prop}_mean'].dropna()
                    stds = subset[f'{prop}_std'].dropna()
                    
                    for mean_val, std_val in zip(means, stds):
                        plot_data.append({
                            'halogen': halogen,
                            'n_layers': n,
                            'group': f'{halogen}-n{n}',
                            'mean': mean_val,
                            'std': std_val
                        })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Create violin plot
                sns.violinplot(data=plot_df, x='n_layers', y='mean', hue='halogen', 
                             palette=colors, ax=ax, alpha=0.7)
                
                # Add error bars
                for halogen in ['Cl', 'Br', 'I']:
                    for n in [2, 3]:
                        subset = plot_df[(plot_df['halogen'] == halogen) & (plot_df['n_layers'] == n)]
                        if len(subset) > 0:
                            mean_val = subset['mean'].mean()
                            std_val = subset['std'].mean()
                            x_pos = (n-2) + (list(colors.keys()).index(halogen) - 1) * 0.25
                            ax.errorbar(x_pos, mean_val, yerr=std_val, 
                                      color=colors[halogen], alpha=0.8, capsize=3)
            
            ax.set_title(property_labels[prop], fontweight='bold')
            ax.set_xlabel('Number of Layers (n)')
            ax.set_ylabel('Property Value')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['n=2', 'n=3'])
            ax.grid(True, alpha=0.3)
            
            if idx == 0:
                ax.legend(title='Halogen', loc='upper right')
            else:
                ax.get_legend().remove()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_2_comparison_all_properties.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # ============================================================================
    # PART 3: Layer_3 comparison for n=3 only
    # ============================================================================
    print("Creating Layer_3 comparison plots...")
    
    layer_3_data = df[df['layer_number'] == 3].copy()
    
    if len(layer_3_data) > 0:
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle('Layer 3 Properties Comparison\nAcross Halogens (Cl, Br, I) for n=3 Only', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        axes = axes.flatten()
        
        for idx, prop in enumerate(properties):
            ax = axes[idx]
            
            # Prepare data for plotting (only n=3)
            plot_data = []
            for halogen in ['Cl', 'Br', 'I']:
                subset = layer_3_data[(layer_3_data['halogen'] == halogen) & 
                                     (layer_3_data['n_layers'] == 3)]
                
                means = subset[f'{prop}_mean'].dropna()
                stds = subset[f'{prop}_std'].dropna()
                
                for mean_val, std_val in zip(means, stds):
                    plot_data.append({
                        'halogen': halogen,
                        'n_layers': 3,
                        'group': f'{halogen}-n3',
                        'mean': mean_val,
                        'std': std_val
                    })
            
            if plot_data:
                plot_df = pd.DataFrame(plot_data)
                
                # Create violin plot
                sns.violinplot(data=plot_df, x='halogen', y='mean', 
                             palette=colors, ax=ax, alpha=0.7)
                
                # Add error bars
                for i, halogen in enumerate(['Cl', 'Br', 'I']):
                    subset = plot_df[plot_df['halogen'] == halogen]
                    if len(subset) > 0:
                        mean_val = subset['mean'].mean()
                        std_val = subset['std'].mean()
                        ax.errorbar(i, mean_val, yerr=std_val, 
                                  color=colors[halogen], alpha=0.8, capsize=3)
            
            ax.set_title(property_labels[prop], fontweight='bold')
            ax.set_xlabel('Halogen')
            ax.set_ylabel('Property Value')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'layer_3_comparison_all_properties.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    # ============================================================================
    # PART 4: Statistical Summary Tables
    # ============================================================================
    print("Creating statistical summary tables...")
    
    # Layer 1 summary
    layer_1_summary = create_statistical_summary(layer_1_data, properties, "Layer 1")
    layer_1_summary.to_csv(os.path.join(output_dir, 'layer_1_statistical_summary.csv'), index=False)
    
    # Layer 2 summary
    if len(layer_2_data) > 0:
        layer_2_summary = create_statistical_summary(layer_2_data, properties, "Layer 2")
        layer_2_summary.to_csv(os.path.join(output_dir, 'layer_2_statistical_summary.csv'), index=False)
    
    # Layer 3 summary
    if len(layer_3_data) > 0:
        layer_3_summary = create_statistical_summary(layer_3_data, properties, "Layer 3")
        layer_3_summary.to_csv(os.path.join(output_dir, 'layer_3_statistical_summary.csv'), index=False)
    
    print("Layer comparison analysis completed!")

def create_statistical_summary(layer_data, properties, layer_name):
    """
    Create statistical summary table for a specific layer.
    """
    summary_records = []
    
    for prop in properties:
        for halogen in ['Cl', 'Br', 'I']:
            for n in [1, 2, 3]:
                subset = layer_data[(layer_data['halogen'] == halogen) & 
                                   (layer_data['n_layers'] == n)]
                
                if len(subset) > 0:
                    means = subset[f'{prop}_mean'].dropna()
                    stds = subset[f'{prop}_std'].dropna()
                    
                    if len(means) > 0:
                        summary_records.append({
                            'layer': layer_name,
                            'property': prop,
                            'halogen': halogen,
                            'n_layers': n,
                            'n_experiments': len(means),
                            'mean_of_means': means.mean(),
                            'std_of_means': means.std(),
                            'mean_of_stds': stds.mean() if len(stds) > 0 else np.nan,
                            'min_mean': means.min(),
                            'max_mean': means.max()
                        })
    
    return pd.DataFrame(summary_records)

def main():
    """
    Main function to run the layer comparative analysis.
    """
    base_dir = '/home/dotempo/Documents/REPO/SVC-Materials/Graphs/BULKS_RESULTS'
    output_dir = os.path.join(base_dir, 'layer_comparative')
    
    print("="*80)
    print("LAYER COMPARATIVE ANALYSIS")
    print("="*80)
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    
    # Extract layer data
    df = extract_layer_data(base_dir)
    
    if len(df) == 0:
        print("No data extracted. Exiting.")
        return
    
    # Save the complete dataset
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'complete_layer_data.csv'), index=False)
    print(f"Complete dataset saved to: {os.path.join(output_dir, 'complete_layer_data.csv')}")
    
    # Create comparison plots
    create_layer_comparison_plots(df, output_dir)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("LAYER ANALYSIS SUMMARY")
    print(f"{'='*80}")
    print(f"Total layer records processed: {len(df)}")
    print(f"Unique experiments: {df['experiment_name'].nunique()}")
    print(f"Halogens analyzed: {', '.join(sorted(df['halogen'].unique()))}")
    print(f"Layer configurations analyzed: {', '.join(sorted(df['n_layers'].astype(str).unique()))}")
    print(f"Layer numbers found: {', '.join(sorted(df['layer_number'].astype(str).unique()))}")
    
    print(f"\nOutput files created:")
    print(f"- layer_1_comparison_all_properties.png")
    print(f"- layer_2_comparison_all_properties.png")
    print(f"- layer_3_comparison_all_properties.png")
    print(f"- layer_1_statistical_summary.csv")
    print(f"- layer_2_statistical_summary.csv")
    print(f"- layer_3_statistical_summary.csv")
    print(f"- complete_layer_data.csv")
    
    print(f"\nAnalysis completed successfully!")

if __name__ == "__main__":
    main() 