"""
Batch Analysis Module for q2D Materials.

This module provides batch analysis capabilities for comparing multiple experiments,
including statistical analysis and comparative visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

# Use matplotlib without x server
matplotlib.use('Agg')


class BatchAnalyzer:
    """
    Batch analyzer for comparing multiple perovskite experiments.
    
    Provides statistical analysis and comparative visualization across
    different structures, compositions, and experimental conditions.
    """
    
    def __init__(self):
        """Initialize batch analyzer."""
        self.experiments = {}
        self.comparison_data = {}
        
    def add_experiment(self, name: str, analyzer_instance):
        """
        Add an experiment to the batch analysis.
        
        Parameters:
        name: Unique identifier for the experiment
        analyzer_instance: q2D_analyzer instance with completed analysis
        """
        self.experiments[name] = analyzer_instance
        print(f"Added experiment: {name}")
        
    def load_experiments_from_directory(self, directory: str, pattern: str = "*.vasp"):
        """
        Load multiple experiments from a directory.
        
        Parameters:
        directory: Path to directory containing VASP files
        pattern: File pattern to match (default: "*.vasp")
        """
        from .analyzer import q2D_analyzer
        from ase.io import read
        
        directory_path = Path(directory)
        vasp_files = list(directory_path.glob(pattern))
        
        if not vasp_files:
            print(f"No files found matching pattern '{pattern}' in {directory}")
            return
            
        for vasp_file in vasp_files:
            try:
                # Extract experiment name from filename
                exp_name = vasp_file.stem
                
                # Auto-detect X-site from structure
                atoms = read(str(vasp_file))
                symbols = atoms.get_chemical_symbols()
                
                # Look for halogen atoms (Br, I, Cl)
                x_site = None
                for symbol in symbols:
                    if symbol in ['Br', 'I', 'Cl']:
                        x_site = symbol
                        break
                
                if x_site is None:
                    print(f"Warning: No halogen found in {vasp_file}, using default 'Cl'")
                    x_site = 'Cl'
                
                # Initialize analyzer with auto-detected X-site
                analyzer = q2D_analyzer(
                    file_path=str(vasp_file),
                    b='Pb',  # Default, can be made configurable
                    x=x_site,  # Auto-detected from structure
                    cutoff_ref_ligand=4.0
                )
                
                self.add_experiment(exp_name, analyzer)
                
            except Exception as e:
                print(f"Failed to load {vasp_file}: {str(e)}")
                
    def extract_comparison_data(self):
        """
        Extract key parameters from all experiments for comparison.
        
        Returns:
        dict: Structured comparison data
        """
        comparison_data = {
            'experiments': {},
            'summary_stats': {},
            'grouping_info': {}
        }
        
        for exp_name, analyzer in self.experiments.items():
            try:
                # Get ontology data
                ontology = analyzer.get_ontology()
                
                # Extract basic info
                cell_props = ontology.get('cell_properties', {})
                composition = cell_props.get('composition', {})
                
                # Extract layer information
                layer_data = ontology.get('layer_analysis', {})
                layer_thickness = getattr(analyzer, 'layer_thickness', None)
                
                # Extract distortion data
                distortion_data = ontology.get('distortion_analysis', {})
                
                # Extract angular data
                octahedra_data = ontology.get('octahedra', {})
                angular_stats = analyzer.angular_analyzer.get_angular_distribution_statistics(octahedra_data)
                
                # Store experiment data
                exp_data = {
                    'basic_info': {
                        'B_site': composition.get('metal_B', 'Unknown'),
                        'X_site': composition.get('halogen_X', 'Unknown'),
                        'layer_thickness': layer_thickness,
                        'n_octahedra': composition.get('number_of_octahedra', 0),
                        'n_atoms': composition.get('number_of_atoms', 0),
                        'cell_volume': cell_props.get('structure_info', {}).get('cell_volume', 0)
                    },
                    'distortion_analysis': {
                        'delta': distortion_data.get('delta_analysis', {}).get('overall_delta'),
                        'sigma': distortion_data.get('sigma_analysis', {}).get('overall_sigma'),
                        'lambda_3': distortion_data.get('lambda_analysis', {}).get('lambda_3'),
                        'lambda_2': distortion_data.get('lambda_analysis', {}).get('lambda_2'),
                        'goldschmidt_tolerance': distortion_data.get('tolerance_factors', {}).get('goldschmidt_tolerance'),
                        'octahedral_tolerance': distortion_data.get('tolerance_factors', {}).get('octahedral_tolerance')
                    },
                    'angular_analysis': angular_stats,
                    'raw_angles': self._extract_raw_angles(octahedra_data)
                }
                
                comparison_data['experiments'][exp_name] = exp_data
                
            except Exception as e:
                print(f"Failed to extract data from {exp_name}: {str(e)}")
                comparison_data['experiments'][exp_name] = {'error': str(e)}
        
        # Calculate summary statistics
        comparison_data['summary_stats'] = self._calculate_summary_statistics(comparison_data['experiments'])
        
        # Group experiments by parameters
        comparison_data['grouping_info'] = self._group_experiments(comparison_data['experiments'])
        
        self.comparison_data = comparison_data
        return comparison_data
        
    def _extract_raw_angles(self, octahedra_data: Dict) -> Dict[str, List[float]]:
        """Extract raw angle data from octahedra for distribution analysis."""
        raw_angles = {
            'cis_angles': [],
            'trans_angles': [],
            'axial_central_axial': []
        }
        
        for oct_key, oct_data in octahedra_data.items():
            angular_data = oct_data.get('angular_analysis', {})
            
            # Extract cis angles
            cis_trans = angular_data.get('cis_trans_analysis', {})
            if 'cis_angles' in cis_trans:
                raw_angles['cis_angles'].extend(cis_trans['cis_angles'])
            
            if 'trans_angles' in cis_trans:
                raw_angles['trans_angles'].extend(cis_trans['trans_angles'])
            
            # Extract axial-central-axial angles
            if 'axial_central_axial' in angular_data:
                aca_data = angular_data['axial_central_axial']
                if 'angle_degrees' in aca_data:
                    raw_angles['axial_central_axial'].append(aca_data['angle_degrees'])
        
        return raw_angles
        
    def _calculate_summary_statistics(self, experiments_data: Dict) -> Dict:
        """Calculate summary statistics across all experiments."""
        summary = {
            'distortion_stats': {},
            'angular_stats': {},
            'composition_stats': {}
        }
        
        # Collect all values for each parameter
        distortion_params = ['delta', 'sigma', 'lambda_3', 'lambda_2', 'goldschmidt_tolerance', 'octahedral_tolerance']
        angular_params = ['cis_angles', 'trans_angles', 'axial_central_axial']
        
        for param in distortion_params:
            values = []
            for exp_data in experiments_data.values():
                if 'error' not in exp_data:
                    value = exp_data.get('distortion_analysis', {}).get(param)
                    if value is not None:
                        values.append(value)
            
            if values:
                summary['distortion_stats'][param] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        for param in angular_params:
            all_angles = []
            for exp_data in experiments_data.values():
                if 'error' not in exp_data:
                    angles = exp_data.get('raw_angles', {}).get(param, [])
                    all_angles.extend(angles)
            
            if all_angles:
                summary['angular_stats'][param] = {
                    'mean': np.mean(all_angles),
                    'std': np.std(all_angles),
                    'min': np.min(all_angles),
                    'max': np.max(all_angles),
                    'count': len(all_angles)
                }
        
        return summary
        
    def _group_experiments(self, experiments_data: Dict) -> Dict:
        """Group experiments by X-site and layer thickness."""
        groups = {
            'by_X_site': {},
            'by_layer_thickness': {}
        }
        
        for exp_name, exp_data in experiments_data.items():
            if 'error' in exp_data:
                continue
                
            basic_info = exp_data.get('basic_info', {})
            X_site = basic_info.get('X_site', 'Unknown')
            layer_thickness = basic_info.get('layer_thickness', 'Unknown')
            
            # Group by X-site
            if X_site not in groups['by_X_site']:
                groups['by_X_site'][X_site] = []
            groups['by_X_site'][X_site].append(exp_name)
            
            # Group by layer thickness
            if layer_thickness not in groups['by_layer_thickness']:
                groups['by_layer_thickness'][layer_thickness] = []
            groups['by_layer_thickness'][layer_thickness].append(exp_name)
        
        return groups
        
    def create_bond_angle_distribution_comparison(self, 
                                                group_by: str = 'B_site',
                                                angle_type: str = 'cis_angles',
                                                show_mean_std: bool = True,
                                                save: bool = True,
                                                filename: str = None) -> plt.Figure:
        """
        Create comparative bond angle distribution plot.
        
        Parameters:
        group_by: Parameter to group experiments by ('B_site', 'X_site', 'composition')
        angle_type: Type of angles to plot ('cis_angles', 'trans_angles', 'axial_central_axial')
        show_mean_std: Whether to show mean ± std as vertical lines
        save: Whether to save the plot
        filename: Output filename
        
        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if not self.comparison_data:
            self.extract_comparison_data()
        
        # Get grouping information
        grouping_key = f'by_{group_by}'
        groups = self.comparison_data['grouping_info'].get(grouping_key, {})
        
        if not groups:
            print(f"No groups found for {group_by}")
            return None
        
        # Create figure with larger size for publication
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Different color schemes for halogens vs thickness
        if group_by == 'X_site':
            # Halogen-specific colors: Br=orange, I=blue, Cl=green
            halogen_colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
            colors = [halogen_colors.get(name, '#888888') for name in groups.keys()]
        else:
            # Layer thickness colors: n=1=pink, n=2=teal, n=3=light blue
            thickness_colors = {1: '#FF9797', 2: '#83DBD5', 3: '#7CCCDE'}
            colors = [thickness_colors.get(name, '#888888') for name in groups.keys()]
        
        # Collect all angles to determine global min/max for x-axis
        all_angles = []
        
        for i, (group_name, exp_names) in enumerate(groups.items()):
            if not exp_names:
                continue
                
            # Collect all angles for this group
            group_angles = []
            group_means = []
            group_stds = []
            
            for exp_name in exp_names:
                exp_data = self.comparison_data['experiments'].get(exp_name, {})
                if 'error' in exp_data:
                    continue
                    
                raw_angles = exp_data.get('raw_angles', {}).get(angle_type, [])
                if raw_angles:
                    group_angles.extend(raw_angles)
                    all_angles.extend(raw_angles)
                    
                    # Calculate mean and std for this experiment
                    exp_mean = np.mean(raw_angles)
                    exp_std = np.std(raw_angles)
                    group_means.append(exp_mean)
                    group_stds.append(exp_std)
            
            if not group_angles:
                continue
                
            # Create KDE for smooth distribution
            angles_array = np.array(group_angles)
            kde = gaussian_kde(angles_array)
            
            # Create angle range for plotting
            angle_min = np.min(angles_array)
            angle_max = np.max(angles_array)
            angle_range = np.linspace(angle_min - 5, angle_max + 5, 300)
            kde_values = kde(angle_range)
            
            # Normalize KDE to make it more visible
            kde_values = kde_values / np.max(kde_values) * 0.8
            
            # Plot KDE curve with better legend
            if group_by == 'X_site':
                legend_label = f'{group_name} (n={len(exp_names)} structures)'
            else:
                legend_label = f'n={group_name} layers (n={len(exp_names)} structures)'
            
            ax.plot(angle_range, kde_values, 
                   color=colors[i], linewidth=3, 
                   label=legend_label)
            
            # Fill area under curve
            ax.fill_between(angle_range, kde_values, alpha=0.2, color=colors[i])
            
            # Show mean ± std if requested
            if show_mean_std and group_means:
                group_mean = np.mean(group_means)
                group_std = np.mean(group_stds)  # Average std across experiments
                
                # Add mean line in the group color with dashed style
                ax.axvline(group_mean, color=colors[i], linestyle='--', alpha=0.9, linewidth=2.5)
                # Add std lines in the group color
                ax.axvline(group_mean - group_std, color=colors[i], linestyle=':', alpha=0.7, linewidth=2)
                ax.axvline(group_mean + group_std, color=colors[i], linestyle=':', alpha=0.7, linewidth=2)
        
        # Set x-axis limits based on actual data range
        if all_angles:
            data_min = np.min(all_angles)
            data_max = np.max(all_angles)
            # Add small margin (5% of range)
            margin = (data_max - data_min) * 0.05
            ax.set_xlim(data_min - margin, data_max + margin)
        else:
            ax.set_xlim(0, 180)  # Fallback
        
        # Customize plot with publication-ready styling
        angle_type_title = angle_type.replace("_", " ").title()
        if angle_type == 'cis_angles':
            angle_type_title = 'Cis Angles'
        elif angle_type == 'trans_angles':
            angle_type_title = 'Trans Angles'
        elif angle_type == 'axial_central_axial':
            angle_type_title = 'Axial-Central-Axial Angles'
        
        # Publication-ready font sizes
        ax.set_xlabel(f'{angle_type_title} (degrees)', fontsize=20, fontweight='bold', color='#333333')
        ax.set_ylabel('Normalized Density', fontsize=20, fontweight='bold', color='#333333')
        
        if group_by == 'X_site':
            title = f'Perovskite {angle_type_title} Distribution by Halogen'
        else:
            title = f'Perovskite {angle_type_title} Distribution by Layer Thickness'
        
        ax.set_title(title, fontsize=24, fontweight='bold', color='#222222', pad=30)
        
        # Remove grid lines for clean look
        ax.grid(False)
        
        # Clean axis styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Make tick labels bigger for publication
        ax.tick_params(axis='both', which='major', labelsize=18, colors='#333333', width=2, length=6)
        
        # Legend positioned at 1/4 from the left with simplified explanation
        legend_elements = []
        for i, (group_name, exp_names) in enumerate(groups.items()):
            if not exp_names:
                continue
                
            if group_by == 'X_site':
                legend_label = f'{group_name} (n={len(exp_names)})'
            else:
                legend_label = f'n={group_name} layers (n={len(exp_names)})'
            
            # Create legend entry with solid line for the distribution (keep original colors)
            legend_elements.append(plt.Line2D([0], [0], color=colors[i], linewidth=4, 
                                            linestyle='-', label=legend_label))
        
        # Add explanation lines in black/gray to avoid repetition
        legend_elements.append(plt.Line2D([0], [0], color='black', linewidth=3, 
                                        linestyle='--', label='Mean'))
        legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=3, 
                                        linestyle='-', label='Std'))
        
        # Position legend at 1/4 from the left (bbox_to_anchor=(0.25, 0.98))
        legend = ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.25, 0.98), 
                          fontsize=16, frameon=True, fancybox=False, shadow=False, 
                          framealpha=0.9, facecolor='white', edgecolor='black')
        # Set legend frame linewidth after creation
        legend.get_frame().set_linewidth(1)
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f'bond_angle_distribution_by_{group_by}_{angle_type}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Bond angle distribution plot saved to: {filename}")
        
        return fig
        

        
    def print_batch_summary(self):
        """Print a comprehensive summary of all experiments."""
        if not self.comparison_data:
            self.extract_comparison_data()
        
        print("\n" + "=" * 80)
        print("BATCH ANALYSIS SUMMARY")
        print("=" * 80)
        
        # Basic experiment info
        n_experiments = len(self.comparison_data['experiments'])
        print(f"Total experiments: {n_experiments}")
        
        # Grouping summary
        grouping_info = self.comparison_data['grouping_info']
        print(f"\nGrouping by X-site: {list(grouping_info['by_X_site'].keys())}")
        print(f"Grouping by layer thickness: {list(grouping_info['by_layer_thickness'].keys())}")
        
        # Summary statistics
        summary_stats = self.comparison_data['summary_stats']
        
        print(f"\nDistortion Parameters Summary:")
        for param, stats in summary_stats['distortion_stats'].items():
            print(f"  {param}: {stats['mean']:.6f} ± {stats['std']:.6f} (n={stats['count']})")
        
        print(f"\nAngular Parameters Summary:")
        for param, stats in summary_stats['angular_stats'].items():
            print(f"  {param}: {stats['mean']:.2f}° ± {stats['std']:.2f}° (n={stats['count']} angles)")
        
        # Show detailed angle counts
        print(f"\nDetailed Angle Counts:")
        total_cis = summary_stats['angular_stats'].get('cis_angles', {}).get('count', 0)
        total_trans = summary_stats['angular_stats'].get('trans_angles', {}).get('count', 0)
        total_axial = summary_stats['angular_stats'].get('axial_central_axial', {}).get('count', 0)
        total_angles = total_cis + total_trans + total_axial
        
        print(f"  Cis angles: {total_cis}")
        print(f"  Trans angles: {total_trans}")
        print(f"  Axial-central-axial angles: {total_axial}")
        print(f"  Total angles analyzed: {total_angles}")
        
        # Show angle statistics by X-site
        print(f"\nAngle Statistics by X-site:")
        for x_site, exp_names in grouping_info['by_X_site'].items():
            x_site_cis = 0
            x_site_trans = 0
            x_site_axial = 0
            
            for exp_name in exp_names:
                exp_data = self.comparison_data['experiments'].get(exp_name, {})
                if 'error' not in exp_data:
                    angular_stats = exp_data.get('angular_analysis', {})
                    x_site_cis += angular_stats.get('cis_angles', {}).get('count', 0)
                    x_site_trans += angular_stats.get('trans_angles', {}).get('count', 0)
                    x_site_axial += angular_stats.get('axial_central_axial', {}).get('count', 0)
            
            total_x_site = x_site_cis + x_site_trans + x_site_axial
            print(f"  {x_site}: {x_site_cis} cis, {x_site_trans} trans, {x_site_axial} axial (total: {total_x_site})")
        
        # Show angle statistics by layer thickness
        print(f"\nAngle Statistics by Layer Thickness:")
        for layer_thickness, exp_names in grouping_info['by_layer_thickness'].items():
            layer_cis = 0
            layer_trans = 0
            layer_axial = 0
            
            for exp_name in exp_names:
                exp_data = self.comparison_data['experiments'].get(exp_name, {})
                if 'error' not in exp_data:
                    angular_stats = exp_data.get('angular_analysis', {})
                    layer_cis += angular_stats.get('cis_angles', {}).get('count', 0)
                    layer_trans += angular_stats.get('trans_angles', {}).get('count', 0)
                    layer_axial += angular_stats.get('axial_central_axial', {}).get('count', 0)
            
            total_layer = layer_cis + layer_trans + layer_axial
            print(f"  n={layer_thickness}: {layer_cis} cis, {layer_trans} trans, {layer_axial} axial (total: {total_layer})")
        
        print("\n" + "=" * 80)
