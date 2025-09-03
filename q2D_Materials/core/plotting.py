"""
Plotting Module for q2D Materials.

This module implements comprehensive plotting capabilities including:
- Angle distribution plots with Gaussian smearing
- Distance distribution plots
- Radial distribution functions (RDF)
- Interactive 3D visualizations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d

# Use matplotlib without x server
matplotlib.use('Agg')


class PlottingAnalyzer:
    """
    Plotting analyzer implementing comprehensive visualization capabilities.
    """
    
    def __init__(self):
        """Initialize plotting analyzer."""
        self.smearing_default = 1.0
        self.gridpoints_default = 300
        
    def _gaussian_kernel_discrete_spectrum(self, spectrum, smearing=None, gridpoints=200):
        """
        Apply Gaussian kernel smoothing to discrete spectrum data.
        
        Parameters:
        spectrum: Array of discrete values
        smearing: Standard deviation for Gaussian kernel
        gridpoints: Number of grid points for output
        
        Returns:
        tuple: (x_values, smoothed_spectrum)
        """
        if smearing is None:
            smearing = self.smearing_default
        
        if len(spectrum) == 0:
            return np.array([]), np.array([])
        
        # Create grid
        x_min = np.min(spectrum) - 3 * smearing
        x_max = np.max(spectrum) + 3 * smearing
        x_values = np.linspace(x_min, x_max, gridpoints)
        
        # Apply Gaussian kernel smoothing
        smoothed = np.zeros_like(x_values)
        
        for value in spectrum:
            # Gaussian kernel centered at each data point
            kernel = np.exp(-0.5 * ((x_values - value) / smearing) ** 2)
            kernel = kernel / (smearing * np.sqrt(2 * np.pi))  # Normalize
            smoothed += kernel
        
        return x_values, smoothed
    
    def plot_angle_distributions(self, octahedra_data, smearing=1, gridpoints=300, 
                                fignum=123, show=True, save=False, filename=None):
        """
        Plot angle distributions with Gaussian smearing, inspired by Pyrovskite.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        smearing: Standard deviation for Gaussian kernel
        gridpoints: Number of grid points for smooth curves
        fignum: Figure number
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        
        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if not octahedra_data:
            print("No octahedra data available for plotting")
            return None
        
        # Collect all angles from octahedra
        all_cis_angles = []
        all_trans_angles = []
        
        for oct_key, oct_data in octahedra_data.items():
            # Get angular analysis data
            angular_analysis = oct_data.get('angular_analysis', {})
            
            # Collect cis angles (from bond_angles if available)
            bond_angles = oct_data.get('bond_angles', {})
            cis_angles = bond_angles.get('cis_angles', {})
            for angle_key, angle_data in cis_angles.items():
                if isinstance(angle_data, dict) and 'value' in angle_data:
                    all_cis_angles.append(angle_data['value'])
                elif isinstance(angle_data, (int, float)):
                    all_cis_angles.append(float(angle_data))
            
            # Collect trans angles
            trans_angles = bond_angles.get('trans_angles', {})
            for angle_key, angle_data in trans_angles.items():
                if isinstance(angle_data, dict) and 'value' in angle_data:
                    all_trans_angles.append(angle_data['value'])
                elif isinstance(angle_data, (int, float)):
                    all_trans_angles.append(float(angle_data))
        
        # Convert to numpy arrays
        all_cis_angles = np.array(all_cis_angles)
        all_trans_angles = np.array(all_trans_angles)
        
        if len(all_cis_angles) == 0 and len(all_trans_angles) == 0:
            print("No angle data found in octahedra")
            return None
        
        # Create figure
        fig = plt.figure(fignum, figsize=(12, 6))
        font = {'size': 18}
        matplotlib.rc('font', **font)
        
        plt.title("Perovskite Bond Angle Distributions", font=font)
        
        # Plot trans angles
        if len(all_trans_angles) > 0:
            xts, smeared_trans_angs = self._gaussian_kernel_discrete_spectrum(
                all_trans_angles, smearing=smearing, gridpoints=gridpoints
            )
            plt.plot(xts, smeared_trans_angs, label="Trans X-B-X", 
                    color='red', linewidth=2)
        
        # Plot cis angles
        if len(all_cis_angles) > 0:
            xcs, smeared_cis_angs = self._gaussian_kernel_discrete_spectrum(
                all_cis_angles, smearing=smearing, gridpoints=gridpoints
            )
            plt.plot(xcs, smeared_cis_angs, label="Cis X-B-X", 
                    color='blue', linewidth=2)
        
        plt.ylim(ymin=0.01)
        plt.xlabel("Bond Angle [°]")
        plt.ylabel("Intensity [arb. u.]")
        plt.legend()
        
        if save:
            if filename is None:
                filename = "angle_distributions.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Angle distribution plot saved to: {filename}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_distance_distributions(self, octahedra_data, smearing=0.02, gridpoints=300,
                                   fignum=12, show=True, save=False, filename=None):
        """
        Plot bond distance distributions with Gaussian smearing.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        smearing: Standard deviation for Gaussian kernel
        gridpoints: Number of grid points for smooth curves
        fignum: Figure number
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        
        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if not octahedra_data:
            print("No octahedra data available for plotting")
            return None
        
        # Collect all bond distances
        all_distances = []
        
        for oct_key, oct_data in octahedra_data.items():
            bond_distances = oct_data.get('bond_distances', {})
            for bond_key, bond_data in bond_distances.items():
                if isinstance(bond_data, dict) and 'distance' in bond_data:
                    all_distances.append(bond_data['distance'])
                elif isinstance(bond_data, (int, float)):
                    all_distances.append(float(bond_data))
        
        all_distances = np.array(all_distances)
        
        if len(all_distances) == 0:
            print("No bond distance data found in octahedra")
            return None
        
        # Create figure
        fig = plt.figure(fignum, figsize=(12, 6))
        font = {'size': 18}
        matplotlib.rc('font', **font)
        
        plt.title("Bond Distance Distribution")
        
        # Apply Gaussian smearing
        xs, smeared_distances = self._gaussian_kernel_discrete_spectrum(
            all_distances, smearing=smearing, gridpoints=gridpoints
        )
        
        plt.plot(xs, smeared_distances, color='black', linewidth=2)
        plt.ylim(ymin=0.01)
        plt.xlabel("Distance [Å]")
        plt.ylabel("Intensity [arb. u.]")
        
        if save:
            if filename is None:
                filename = "distance_distributions.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Distance distribution plot saved to: {filename}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_octahedral_distortion_summary(self, octahedra_data, fignum=456, 
                                         show=True, save=False, filename=None):
        """
        Create a comprehensive summary plot of octahedral distortion parameters.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        fignum: Figure number
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        
        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if not octahedra_data:
            print("No octahedra data available for plotting")
            return None
        
        # Extract distortion parameters
        zeta_values = []
        delta_values = []
        sigma_values = []
        theta_values = []
        
        for oct_key, oct_data in octahedra_data.items():
            distortion_params = oct_data.get('distortion_parameters', {})
            
            if 'zeta' in distortion_params:
                zeta_values.append(distortion_params['zeta'])
            if 'delta' in distortion_params:
                delta_values.append(distortion_params['delta'])
            if 'sigma' in distortion_params:
                sigma_values.append(distortion_params['sigma'])
            if 'theta_mean' in distortion_params:
                theta_values.append(distortion_params['theta_mean'])
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Octahedral Distortion Parameters Summary', fontsize=16)
        
        # Plot zeta distribution
        if zeta_values:
            axes[0, 0].hist(zeta_values, bins=20, alpha=0.7, color='blue', edgecolor='black')
            axes[0, 0].set_title('Zeta Distribution')
            axes[0, 0].set_xlabel('Zeta')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].axvline(np.mean(zeta_values), color='red', linestyle='--', 
                              label=f'Mean: {np.mean(zeta_values):.4f}')
            axes[0, 0].legend()
        
        # Plot delta distribution
        if delta_values:
            axes[0, 1].hist(delta_values, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Delta Distribution')
            axes[0, 1].set_xlabel('Delta')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].axvline(np.mean(delta_values), color='red', linestyle='--',
                              label=f'Mean: {np.mean(delta_values):.6f}')
            axes[0, 1].legend()
        
        # Plot sigma distribution
        if sigma_values:
            axes[1, 0].hist(sigma_values, bins=20, alpha=0.7, color='orange', edgecolor='black')
            axes[1, 0].set_title('Sigma Distribution')
            axes[1, 0].set_xlabel('Sigma')
            axes[1, 0].set_ylabel('Count')
            axes[1, 0].axvline(np.mean(sigma_values), color='red', linestyle='--',
                              label=f'Mean: {np.mean(sigma_values):.4f}')
            axes[1, 0].legend()
        
        # Plot theta distribution
        if theta_values:
            axes[1, 1].hist(theta_values, bins=20, alpha=0.7, color='purple', edgecolor='black')
            axes[1, 1].set_title('Theta Distribution')
            axes[1, 1].set_xlabel('Theta (degrees)')
            axes[1, 1].set_ylabel('Count')
            axes[1, 1].axvline(np.mean(theta_values), color='red', linestyle='--',
                              label=f'Mean: {np.mean(theta_values):.2f}°')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = "octahedral_distortion_summary.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Distortion summary plot saved to: {filename}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_enhanced_distortion_comparison(self, enhanced_distortion_data, fignum=789,
                                          show=True, save=False, filename=None):
        """
        Plot comparison of enhanced distortion parameters (delta, sigma, lambda).
        
        Parameters:
        enhanced_distortion_data: Dictionary with enhanced distortion analysis results
        fignum: Figure number
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        
        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if not enhanced_distortion_data or 'error' in enhanced_distortion_data:
            print("No enhanced distortion data available for plotting")
            return None
        
        # Extract data
        delta_analysis = enhanced_distortion_data.get('delta_analysis', {})
        sigma_analysis = enhanced_distortion_data.get('sigma_analysis', {})
        lambda_analysis = enhanced_distortion_data.get('lambda_analysis', {})
        tolerance_factors = enhanced_distortion_data.get('tolerance_factors', {})
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Enhanced Distortion Analysis Summary', fontsize=16)
        
        # Plot delta values
        if 'octahedra_delta' in delta_analysis and delta_analysis['octahedra_delta']:
            delta_values = delta_analysis['octahedra_delta']
            axes[0, 0].bar(range(len(delta_values)), delta_values, alpha=0.7, color='blue')
            axes[0, 0].set_title('Delta Distortion per Octahedron')
            axes[0, 0].set_xlabel('Octahedron Index')
            axes[0, 0].set_ylabel('Delta')
            axes[0, 0].axhline(delta_analysis.get('overall_delta', 0), color='red', 
                              linestyle='--', label=f'Overall: {delta_analysis.get("overall_delta", 0):.6f}')
            axes[0, 0].legend()
        
        # Plot sigma values
        if 'octahedra_sigma' in sigma_analysis and sigma_analysis['octahedra_sigma']:
            sigma_values = sigma_analysis['octahedra_sigma']
            axes[0, 1].bar(range(len(sigma_values)), sigma_values, alpha=0.7, color='green')
            axes[0, 1].set_title('Sigma Distortion per Octahedron')
            axes[0, 1].set_xlabel('Octahedron Index')
            axes[0, 1].set_ylabel('Sigma (degrees)')
            axes[0, 1].axhline(sigma_analysis.get('overall_sigma', 0), color='red',
                              linestyle='--', label=f'Overall: {sigma_analysis.get("overall_sigma", 0):.2f}°')
            axes[0, 1].legend()
        
        # Plot lambda values
        if 'octahedra_lambda_3' in lambda_analysis and lambda_analysis['octahedra_lambda_3']:
            lambda_3_values = lambda_analysis['octahedra_lambda_3']
            lambda_2_values = lambda_analysis.get('octahedra_lambda_2', [])
            
            x_pos = np.arange(len(lambda_3_values))
            width = 0.35
            
            axes[1, 0].bar(x_pos - width/2, lambda_3_values, width, label='Lambda-3', 
                          alpha=0.7, color='orange')
            if lambda_2_values:
                axes[1, 0].bar(x_pos + width/2, lambda_2_values, width, label='Lambda-2',
                              alpha=0.7, color='red')
            
            axes[1, 0].set_title('Lambda Distortion per Octahedron')
            axes[1, 0].set_xlabel('Octahedron Index')
            axes[1, 0].set_ylabel('Lambda')
            axes[1, 0].legend()
        
        # Plot tolerance factors
        tolerance_data = []
        tolerance_labels = []
        
        if 'goldschmidt_tolerance' in tolerance_factors and tolerance_factors['goldschmidt_tolerance'] is not None:
            tolerance_data.append(tolerance_factors['goldschmidt_tolerance'])
            tolerance_labels.append('Goldschmidt')
        
        if 'octahedral_tolerance' in tolerance_factors and tolerance_factors['octahedral_tolerance'] is not None:
            tolerance_data.append(tolerance_factors['octahedral_tolerance'])
            tolerance_labels.append('Octahedral')
        
        if tolerance_data:
            axes[1, 1].bar(tolerance_labels, tolerance_data, alpha=0.7, color='purple')
            axes[1, 1].set_title('Tolerance Factors')
            axes[1, 1].set_ylabel('Tolerance Factor')
            axes[1, 1].axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Ideal')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = "enhanced_distortion_comparison.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Enhanced distortion comparison plot saved to: {filename}")
        
        if show:
            plt.show()
        
        return fig
    
    def create_comprehensive_analysis_plot(self, ontology_data, fignum=999,
                                         show=True, save=False, filename=None):
        """
        Create a comprehensive analysis plot combining multiple analysis results.
        
        Parameters:
        ontology_data: Complete ontology data from analyzer
        fignum: Figure number
        show: Whether to display the plot
        save: Whether to save the plot
        filename: Output filename
        
        Returns:
        matplotlib.figure.Figure: The created figure
        """
        if not ontology_data:
            print("No ontology data available for plotting")
            return None
        
        # Create a large figure with multiple subplots
        fig = plt.figure(fignum, figsize=(20, 16))
        fig.suptitle('Comprehensive q2D Materials Analysis', fontsize=20)
        
        # Create a grid layout
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 1. Octahedra count and composition
        ax1 = fig.add_subplot(gs[0, 0])
        cell_props = ontology_data.get('cell_properties', {})
        composition = cell_props.get('composition', {})
        
        if composition:
            elements = list(composition.keys())
            counts = list(composition.values())
            ax1.bar(elements, counts, alpha=0.7, color='skyblue')
            ax1.set_title('Element Composition')
            ax1.set_ylabel('Count')
        
        # 2. Distortion parameters summary
        ax2 = fig.add_subplot(gs[0, 1])
        octahedra_data = ontology_data.get('octahedra', {})
        
        if octahedra_data:
            zeta_values = []
            for oct_data in octahedra_data.values():
                distortion = oct_data.get('distortion_parameters', {})
                if 'zeta' in distortion:
                    zeta_values.append(distortion['zeta'])
            
            if zeta_values:
                ax2.hist(zeta_values, bins=15, alpha=0.7, color='lightgreen', edgecolor='black')
                ax2.set_title('Zeta Distribution')
                ax2.set_xlabel('Zeta')
                ax2.set_ylabel('Count')
        
        # 3. Connectivity analysis
        ax3 = fig.add_subplot(gs[0, 2])
        connectivity = ontology_data.get('connectivity_analysis', {})
        network_stats = connectivity.get('network_statistics', {})
        
        if network_stats:
            connections_per_oct = network_stats.get('connections_per_octahedron', {})
            if connections_per_oct:
                oct_keys = list(connections_per_oct.keys())
                connection_counts = list(connections_per_oct.values())
                ax3.bar(range(len(oct_keys)), connection_counts, alpha=0.7, color='orange')
                ax3.set_title('Connectivity per Octahedron')
                ax3.set_xlabel('Octahedron')
                ax3.set_ylabel('Connections')
        
        # 4. Layer analysis
        ax4 = fig.add_subplot(gs[0, 3])
        layers_analysis = ontology_data.get('layers_analysis', {})
        layers = layers_analysis.get('layers', {})
        
        if layers:
            layer_names = list(layers.keys())
            octahedra_counts = [layers[layer]['octahedra_count'] for layer in layer_names]
            ax4.bar(layer_names, octahedra_counts, alpha=0.7, color='pink')
            ax4.set_title('Octahedra per Layer')
            ax4.set_ylabel('Count')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Enhanced distortion analysis
        ax5 = fig.add_subplot(gs[1, :2])
        enhanced_distortion = ontology_data.get('enhanced_distortion_analysis', {})
        
        if enhanced_distortion and 'error' not in enhanced_distortion:
            distortion_params = []
            param_names = []
            
            delta_analysis = enhanced_distortion.get('delta_analysis', {})
            if 'overall_delta' in delta_analysis and delta_analysis['overall_delta'] is not None:
                distortion_params.append(delta_analysis['overall_delta'])
                param_names.append('Delta')
            
            sigma_analysis = enhanced_distortion.get('sigma_analysis', {})
            if 'overall_sigma' in sigma_analysis and sigma_analysis['overall_sigma'] is not None:
                distortion_params.append(sigma_analysis['overall_sigma'])
                param_names.append('Sigma')
            
            lambda_analysis = enhanced_distortion.get('lambda_analysis', {})
            if 'lambda_3' in lambda_analysis and lambda_analysis['lambda_3'] is not None:
                distortion_params.append(lambda_analysis['lambda_3'])
                param_names.append('Lambda-3')
            
            if distortion_params:
                ax5.bar(param_names, distortion_params, alpha=0.7, color='lightcoral')
                ax5.set_title('Enhanced Distortion Parameters')
                ax5.set_ylabel('Distortion Value')
        
        # 6. Vector analysis
        ax6 = fig.add_subplot(gs[1, 2:])
        vector_analysis = ontology_data.get('vector_analysis', {})
        
        if vector_analysis and 'error' not in vector_analysis:
            vector_results = vector_analysis.get('vector_analysis_results', {})
            if vector_results:
                angle_between_planes = vector_results.get('angle_between_planes_degrees', 0)
                distance_between_planes = vector_results.get('distance_between_plane_centers_angstrom', 0)
                
                # Create a simple visualization
                ax6.text(0.5, 0.7, f'Angle between planes: {angle_between_planes:.2f}°', 
                        transform=ax6.transAxes, fontsize=12, ha='center')
                ax6.text(0.5, 0.5, f'Distance between planes: {distance_between_planes:.3f} Å', 
                        transform=ax6.transAxes, fontsize=12, ha='center')
                ax6.set_title('Vector Analysis Results')
                ax6.axis('off')
        
        # 7. Molecule analysis (if available)
        ax7 = fig.add_subplot(gs[2, :2])
        # This would be populated if molecule analysis is available
        
        # 8. Summary statistics
        ax8 = fig.add_subplot(gs[2, 2:])
        if octahedra_data:
            total_octahedra = len(octahedra_data)
            total_atoms = sum(len(oct_data.get('ligand_atoms', {}).get('all_ligand_global_indices', [])) 
                            for oct_data in octahedra_data.values()) + total_octahedra
            
            ax8.text(0.5, 0.8, f'Total Octahedra: {total_octahedra}', 
                    transform=ax8.transAxes, fontsize=14, ha='center')
            ax8.text(0.5, 0.6, f'Total Atoms: {total_atoms}', 
                    transform=ax8.transAxes, fontsize=14, ha='center')
            ax8.text(0.5, 0.4, f'Structure Type: 2D Perovskite', 
                    transform=ax8.transAxes, fontsize=14, ha='center')
            ax8.set_title('Structure Summary')
            ax8.axis('off')
        
        # 9. Tolerance factors
        ax9 = fig.add_subplot(gs[3, :2])
        if enhanced_distortion and 'error' not in enhanced_distortion:
            tolerance_factors = enhanced_distortion.get('tolerance_factors', {})
            if tolerance_factors:
                factors = []
                factor_names = []
                
                if 'goldschmidt_tolerance' in tolerance_factors and tolerance_factors['goldschmidt_tolerance'] is not None:
                    factors.append(tolerance_factors['goldschmidt_tolerance'])
                    factor_names.append('Goldschmidt')
                
                if 'octahedral_tolerance' in tolerance_factors and tolerance_factors['octahedral_tolerance'] is not None:
                    factors.append(tolerance_factors['octahedral_tolerance'])
                    factor_names.append('Octahedral')
                
                if factors:
                    bars = ax9.bar(factor_names, factors, alpha=0.7, color='gold')
                    ax9.set_title('Tolerance Factors')
                    ax9.set_ylabel('Tolerance Factor')
                    ax9.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Ideal')
                    ax9.legend()
        
        # 10. Analysis metadata
        ax10 = fig.add_subplot(gs[3, 2:])
        experiment = ontology_data.get('experiment', {})
        if experiment:
            ax10.text(0.1, 0.8, f'Experiment: {experiment.get("name", "Unknown")}', 
                     transform=ax10.transAxes, fontsize=12)
            ax10.text(0.1, 0.6, f'Timestamp: {experiment.get("timestamp", "Unknown")}', 
                     transform=ax10.transAxes, fontsize=12)
            ax10.text(0.1, 0.4, f'File: {experiment.get("file_path", "Unknown")}', 
                     transform=ax10.transAxes, fontsize=10)
            ax10.set_title('Analysis Metadata')
            ax10.axis('off')
        
        if save:
            if filename is None:
                filename = "comprehensive_analysis.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Comprehensive analysis plot saved to: {filename}")
        
        if show:
            plt.show()
        
        return fig
