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
import os
import re
import seaborn as sns

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
        self.bandgap_data = {}
        
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
            'axial_central_axial': [],
            'central_axial_central': []
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
            
            # Extract central-axial-central angles (B-X-B angles)
            if 'central_axial_central' in angular_data:
                cac_data = angular_data['central_axial_central']
                for cac_angle in cac_data:
                    if 'angle_degrees' in cac_angle:
                        raw_angles['central_axial_central'].append(cac_angle['angle_degrees'])
        
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
        angular_params = ['cis_angles', 'trans_angles', 'axial_central_axial', 'central_axial_central']
        
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
    
    def read_vbm_cbm_from_outcar(self, outcar_path: str) -> Optional[Dict[str, float]]:
        """
        Read VBM and CBM values from OUTCAR file.
        
        Parameters:
        outcar_path: Path to OUTCAR file
        
        Returns:
        dict: Dictionary with 'vbm', 'cbm', and 'bandgap' keys, or None if not found
        """
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
            
            # Pattern 1: Look for VBM and CBM explicitly (BANDGAP=COMPACT output)
            vbm_pattern = r'val\. band max:\s*([0-9.-]+)'
            cbm_pattern = r'cond\. band min:\s*([0-9.-]+)'
            
            vbm_match = re.search(vbm_pattern, content, re.IGNORECASE)
            cbm_match = re.search(cbm_pattern, content, re.IGNORECASE)
            
            if vbm_match and cbm_match:
                vbm = float(vbm_match.group(1))
                cbm = float(cbm_match.group(1))
                bandgap = cbm - vbm
                if bandgap > 0:  # Only return positive bandgap
                    return {'vbm': vbm, 'cbm': cbm, 'bandgap': bandgap}
            
            # Pattern 2: Look for valence and conduction band info (fallback)
            valence_pattern = r'valence band maximum\s*:\s*([0-9.-]+)'
            conduction_pattern = r'conduction band minimum\s*:\s*([0-9.-]+)'
            
            valence_match = re.search(valence_pattern, content, re.IGNORECASE)
            conduction_match = re.search(conduction_pattern, content, re.IGNORECASE)
            
            if valence_match and conduction_match:
                vbm = float(valence_match.group(1))
                cbm = float(conduction_match.group(1))
                bandgap = cbm - vbm
                if bandgap > 0:  # Only return positive bandgap
                    return {'vbm': vbm, 'cbm': cbm, 'bandgap': bandgap}
            
            return None
            
        except Exception as e:
            print(f"Error reading VBM/CBM from {outcar_path}: {str(e)}")
            return None

    def read_bandgap_from_outcar(self, outcar_path: str) -> Optional[float]:
        """
        Read bandgap from OUTCAR file.
        
        This method looks for bandgap information in VASP OUTCAR files,
        particularly when BANDGAP=COMPACT is used, which reports VBM, CBM, and fundamental gap.
        
        Parameters:
        outcar_path: Path to OUTCAR file
        
        Returns:
        float: Bandgap in eV, or None if not found
        """
        try:
            with open(outcar_path, 'r') as f:
                content = f.read()
            
            # Pattern 1: Look for "fundamental gap" (BANDGAP=COMPACT output)
            # This is the most reliable method when BANDGAP=COMPACT is used
            fundamental_gap_pattern = r'fundamental gap:\s*([0-9.-]+)'
            match = re.search(fundamental_gap_pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            # Pattern 1b: Look for "band gap" section (BANDGAP=COMPACT output)
            bandgap_section_pattern = r'band gap\s*:\s*([0-9.-]+)\s*eV'
            match = re.search(bandgap_section_pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            # Pattern 2: Look for VBM and CBM explicitly (BANDGAP=COMPACT output)
            vbm_pattern = r'val\. band max:\s*([0-9.-]+)'
            cbm_pattern = r'cond\. band min:\s*([0-9.-]+)'
            
            vbm_match = re.search(vbm_pattern, content, re.IGNORECASE)
            cbm_match = re.search(cbm_pattern, content, re.IGNORECASE)
            
            if vbm_match and cbm_match:
                vbm = float(vbm_match.group(1))
                cbm = float(cbm_match.group(1))
                bandgap = cbm - vbm
                if bandgap > 0:  # Only return positive bandgap
                    return bandgap
            
            # Pattern 3: Direct bandgap
            direct_pattern = r'band gap \(direct\)\s*=\s*([0-9.-]+)\s*eV'
            match = re.search(direct_pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            # Pattern 4: Indirect bandgap
            indirect_pattern = r'band gap \(indirect\)\s*=\s*([0-9.-]+)\s*eV'
            match = re.search(indirect_pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            # Pattern 5: General bandgap
            general_pattern = r'band gap\s*=\s*([0-9.-]+)\s*eV'
            match = re.search(general_pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            # Pattern 6: Look for E-fermi and band structure info (fallback)
            efermi_pattern = r'E-fermi\s*:\s*([0-9.-]+)'
            efermi_match = re.search(efermi_pattern, content)
            
            if efermi_match:
                efermi = float(efermi_match.group(1))
                # Look for valence and conduction band info
                valence_pattern = r'valence band maximum\s*:\s*([0-9.-]+)'
                conduction_pattern = r'conduction band minimum\s*:\s*([0-9.-]+)'
                
                valence_match = re.search(valence_pattern, content, re.IGNORECASE)
                conduction_match = re.search(conduction_pattern, content, re.IGNORECASE)
                
                if valence_match and conduction_match:
                    valence_max = float(valence_match.group(1))
                    conduction_min = float(conduction_match.group(1))
                    bandgap = conduction_min - valence_max
                    if bandgap > 0:  # Only return positive bandgap
                        return bandgap
            
            # Pattern 7: Look for bandgap in the electronic structure section
            # Sometimes VASP reports bandgap in a different format
            electronic_section_pattern = r'electronic structure\s*:\s*band gap\s*=\s*([0-9.-]+)\s*eV'
            match = re.search(electronic_section_pattern, content, re.IGNORECASE)
            if match:
                return float(match.group(1))
            
            return None
            
        except Exception as e:
            print(f"Error reading bandgap from {outcar_path}: {str(e)}")
            return None
    
    def read_bandgap_from_xml(self, xml_path: str) -> Optional[float]:
        """
        Read bandgap from vasprun.xml file using ASE.
        
        Parameters:
        xml_path: Path to vasprun.xml file
        
        Returns:
        float: Bandgap in eV, or None if not found
        """
        try:
            from ase.io import read
            from ase.calculators.vasp import Vasp
            
            # Read the XML file
            atoms = read(xml_path, format='vasp-xml')
            
            # Try to get bandgap from calculator
            if hasattr(atoms, 'calc') and atoms.calc is not None:
                if hasattr(atoms.calc, 'get_band_gap'):
                    bandgap = atoms.calc.get_band_gap()
                    if bandgap is not None:
                        return bandgap
            
            # Alternative: try to read from file content
            with open(xml_path, 'r') as f:
                content = f.read()
            
            # Look for bandgap in XML
            bandgap_pattern = r'<i name="bandgap">\s*([0-9.-]+)\s*</i>'
            match = re.search(bandgap_pattern, content)
            if match:
                return float(match.group(1))
            
            return None
            
        except Exception as e:
            print(f"Error reading bandgap from {xml_path}: {str(e)}")
            return None
    
    def read_bandgap_from_directory(self, directory: str, verbose: bool = False) -> Optional[float]:
        """
        Read bandgap from OUTCAR or vasprun.xml in a directory.
        
        Parameters:
        directory: Path to directory containing VASP output files
        verbose: Whether to print detailed information about bandgap reading
        
        Returns:
        float: Bandgap in eV, or None if not found
        """
        # Try OUTCAR first
        outcar_path = os.path.join(directory, 'OUTCAR')
        if os.path.exists(outcar_path):
            if verbose:
                print(f"    Reading bandgap from OUTCAR: {outcar_path}")
            bandgap = self.read_bandgap_from_outcar(outcar_path)
            if bandgap is not None:
                if verbose:
                    print(f"    Found bandgap in OUTCAR: {bandgap:.3f} eV")
                return bandgap
            elif verbose:
                print(f"    No bandgap found in OUTCAR")
        
        # Try vasprun.xml
        xml_path = os.path.join(directory, 'vasprun.xml')
        if os.path.exists(xml_path):
            if verbose:
                print(f"    Reading bandgap from vasprun.xml: {xml_path}")
            bandgap = self.read_bandgap_from_xml(xml_path)
            if bandgap is not None:
                if verbose:
                    print(f"    Found bandgap in vasprun.xml: {bandgap:.3f} eV")
                return bandgap
            elif verbose:
                print(f"    No bandgap found in vasprun.xml")
        
        if verbose:
            print(f"    No VASP output files found in {directory}")
        
        return None

    def read_vbm_cbm_from_directory(self, directory: str, verbose: bool = False) -> Optional[Dict[str, float]]:
        """
        Read VBM and CBM values from OUTCAR in a directory.
        
        Parameters:
        directory: Path to directory containing VASP output files
        verbose: Whether to print detailed information about VBM/CBM reading
        
        Returns:
        dict: Dictionary with 'vbm', 'cbm', and 'bandgap' keys, or None if not found
        """
        # Try OUTCAR first
        outcar_path = os.path.join(directory, 'OUTCAR')
        if os.path.exists(outcar_path):
            if verbose:
                print(f"    Reading VBM/CBM from OUTCAR: {outcar_path}")
            vbm_cbm_data = self.read_vbm_cbm_from_outcar(outcar_path)
            if vbm_cbm_data is not None:
                if verbose:
                    print(f"    Found VBM: {vbm_cbm_data['vbm']:.3f} eV, CBM: {vbm_cbm_data['cbm']:.3f} eV")
                return vbm_cbm_data
            elif verbose:
                print(f"    No VBM/CBM found in OUTCAR")
        
        if verbose:
            print(f"    No VASP output files found in {directory}")
        
        return None
        
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
        elif angle_type == 'central_axial_central':
            angle_type_title = 'B-X-B (Central-Axial-Central) Angles'
        
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
        
    def create_distortion_comparison_plot(self, 
                                        group_by: str = 'X_site',
                                        distortion_param: str = 'delta',
                                        show_mean_std: bool = True,
                                        save: bool = True,
                                        filename: str = None) -> plt.Figure:
        """
        Create comparative distortion parameter plot.
        
        Parameters:
        group_by: Parameter to group experiments by ('X_site', 'layer_thickness')
        distortion_param: Distortion parameter to plot ('delta', 'sigma', 'lambda_3', 'lambda_2')
        show_mean_std: Whether to show mean ± std
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
        
        # Collect data for each group
        group_data = []
        group_names = []
        
        for i, (group_name, exp_names) in enumerate(groups.items()):
            if not exp_names:
                continue
                
            # Collect distortion parameter values for this group
            group_values = []
            for exp_name in exp_names:
                exp_data = self.comparison_data['experiments'].get(exp_name, {})
                if 'error' in exp_data:
                    continue
                    
                value = exp_data.get('distortion_analysis', {}).get(distortion_param)
                if value is not None:
                    group_values.append(value)
            
            if group_values:
                group_data.append(group_values)
                group_names.append(group_name)
        
        if not group_data:
            print(f"No data found for {distortion_param}")
            return None
        
        # Create box plot
        box_plot = ax.boxplot(group_data, labels=group_names, patch_artist=True, 
                             showmeans=True, meanline=True, showfliers=True)
        
        # Color the boxes
        for patch, color in zip(box_plot['boxes'], colors[:len(group_data)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Customize plot with publication-ready styling
        param_title = distortion_param.replace("_", " ").title()
        if distortion_param == 'lambda_3':
            param_title = 'λ₃'
        elif distortion_param == 'lambda_2':
            param_title = 'λ₂'
        
        ax.set_xlabel('Group', fontsize=20, fontweight='bold', color='#333333')
        ax.set_ylabel(f'{param_title}', fontsize=20, fontweight='bold', color='#333333')
        
        if group_by == 'X_site':
            title = f'Perovskite {param_title} by Halogen'
        else:
            title = f'Perovskite {param_title} by Layer Thickness'
        
        ax.set_title(title, fontsize=24, fontweight='bold', color='#222222', pad=30)
        
        # Clean axis styling
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.spines['left'].set_linewidth(2)
        ax.spines['bottom'].set_linewidth(2)
        
        # Make tick labels bigger for publication
        ax.tick_params(axis='both', which='major', labelsize=18, colors='#333333', width=2, length=6)
        
        # Rotate x-axis labels if needed
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add legend with simplified explanation
        legend_elements = []
        for i, group_name in enumerate(group_names):
            if group_by == 'X_site':
                legend_label = f'{group_name} (n={len(group_data[i])})'
            else:
                legend_label = f'n={group_name} layers (n={len(group_data[i])})'
            
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=colors[i], alpha=0.7, label=legend_label))
        
        # Position legend at 1/4 from the left
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.25, 0.98), 
                 fontsize=16, frameon=True, fancybox=False, shadow=False, 
                 framealpha=0.9, facecolor='white', edgecolor='black')
        
        plt.tight_layout()
        
        if save:
            if filename is None:
                filename = f'distortion_{distortion_param}_by_{group_by}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Distortion comparison plot saved to: {filename}")
        
        return fig
    
    def create_bandgap_scatter_plots_with_family_background(self, 
                                    save: bool = True,
                                    filename_prefix: str = 'MAPbX3_bandgap_vs') -> List[plt.Figure]:
        """
        Create scatter plots of bandgap vs distortion parameters and angles with molecular family background colors.
        
        Parameters:
        save: Whether to save the plots
        filename_prefix: Prefix for saved filenames
        
        Returns:
        List[matplotlib.figure.Figure]: List of created figures
        """
        if not self.comparison_data:
            self.extract_comparison_data()
        
        # Load the CSV data to get molecular family information
        import pandas as pd
        try:
            df = pd.read_csv("unified_octahedral_molecular_dataset.csv")
            print("✓ Loaded molecular dataset for family classification")
        except FileNotFoundError:
            print("✗ Could not find unified_octahedral_molecular_dataset.csv")
            return []
        
        # Classify molecules into families
        def classify_molecule_family(smiles):
            import re
            carbon_count = len(re.findall(r'C(?![A-Z])', smiles))
            if 'C1=CC=C' in smiles or 'C1=CC=CC=C1' in smiles:
                family = 'Aromatic'
            elif 'C1CCC' in smiles and 'C1=CC' not in smiles:
                family = 'Cyclic'
            elif 'C(C)' in smiles or 'CC(C)' in smiles or 'C(C)(C)' in smiles:
                family = 'Branched'
            else:
                family = 'Linear'
            return family, carbon_count
        
        # Add family information to dataframe
        df['Family'] = df['Molecule'].apply(lambda x: classify_molecule_family(x)[0])
        df['Carbon_Count'] = df['Molecule'].apply(lambda x: classify_molecule_family(x)[1])
        
        # Collect data for scatter plots
        scatter_data = []
        
        for exp_name, exp_data in self.comparison_data['experiments'].items():
            if 'error' in exp_data:
                continue
            
            # Get basic info
            basic_info = exp_data.get('basic_info', {})
            X_site = basic_info.get('X_site', 'Unknown')
            layer_thickness = basic_info.get('layer_thickness', 'Unknown')
            
            # Get distortion data
            distortion_data = exp_data.get('distortion_analysis', {})
            
            # Get angular data
            angular_data = exp_data.get('angular_analysis', {})
            
            # Get bandgap (if available)
            bandgap = self.bandgap_data.get(exp_name)
            
            # Get molecular family from CSV data
            molecule_family = 'Unknown'
            if exp_name in df['Experiment'].values:
                molecule_family = df[df['Experiment'] == exp_name]['Family'].iloc[0]
            
            if bandgap is not None:
                scatter_data.append({
                    'exp_name': exp_name,
                    'X_site': X_site,
                    'layer_thickness': layer_thickness,
                    'molecule_family': molecule_family,
                    'bandgap': bandgap,
                    'delta': distortion_data.get('delta'),
                    'sigma': distortion_data.get('sigma'),
                    'lambda_3': distortion_data.get('lambda_3'),
                    'lambda_2': distortion_data.get('lambda_2'),
                    'cis_angle_mean': angular_data.get('cis_angles', {}).get('mean'),
                    'trans_angle_mean': angular_data.get('trans_angles', {}).get('mean'),
                    'axial_central_axial_mean': angular_data.get('axial_central_axial', {}).get('mean'),
                    'central_axial_central_mean': angular_data.get('central_axial_central', {}).get('mean')
                })
        
        if not scatter_data:
            print("No bandgap data available for scatter plots")
            return []
        
        # Create scatter plots
        figures = []
        
        # Parameters to plot against bandgap
        params = [
            ('lambda_2', 'λ₂'),
            ('lambda_3', 'λ₃'),
            ('delta', 'Δ'),
            ('sigma', 'σ'),
            ('cis_angle_mean', 'Cis Angle Mean'),
            ('trans_angle_mean', 'Trans Angle Mean'),
            ('axial_central_axial_mean', 'Axial-Central-Axial Mean'),
            ('central_axial_central_mean', 'B-X-B Mean')
        ]
        
        for param, param_label in params:
            # Filter out None values
            valid_data = [d for d in scatter_data if d.get(param) is not None]
            
            if not valid_data:
                print(f"No valid data for {param}")
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Define colors
            halogen_colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
            family_bg_colors = {
                'Linear': '#E3F2FD',      # Light blue
                'Branched': '#FFF3E0',    # Light orange
                'Cyclic': '#E8F5E8',      # Light green
                'Aromatic': '#F3E5F5',    # Light purple
                'Unknown': '#F5F5F5'      # Light gray
            }
            
            # Get unique families and create background regions
            families = list(set(d['molecule_family'] for d in valid_data))
            families = [f for f in families if f != 'Unknown']  # Remove Unknown for cleaner background
            
            if families:
                # Sort families for consistent ordering
                family_order = ['Linear', 'Branched', 'Cyclic', 'Aromatic']
                families = [f for f in family_order if f in families]
                
                # Create background regions for each family
                x_vals = [d[param] for d in valid_data]
                y_vals = [d['bandgap'] for d in valid_data]
                
                if x_vals and y_vals:
                    x_min, x_max = min(x_vals), max(x_vals)
                    y_min, y_max = min(y_vals), max(y_vals)
                    
                    # Add some padding
                    x_padding = (x_max - x_min) * 0.02
                    y_padding = (y_max - y_min) * 0.02
                    
                    # Create background rectangles for each family
                    for i, family in enumerate(families):
                        family_data = [d for d in valid_data if d['molecule_family'] == family]
                        if family_data:
                            family_x_vals = [d[param] for d in family_data]
                            family_y_vals = [d['bandgap'] for d in family_data]
                            
                            if family_x_vals and family_y_vals:
                                family_x_min, family_x_max = min(family_x_vals), max(family_x_vals)
                                family_y_min, family_y_max = min(family_y_vals), max(family_y_vals)
                                
                                # Add background rectangle
                                rect = plt.Rectangle((family_x_min - x_padding, family_y_min - y_padding),
                                                   family_x_max - family_x_min + 2*x_padding,
                                                   family_y_max - family_y_min + 2*y_padding,
                                                   facecolor=family_bg_colors.get(family, '#F5F5F5'),
                                                   alpha=0.3, zorder=0)
                                ax.add_patch(rect)
            
            # Separate data by X-site for point coloring
            x_sites = list(set(d['X_site'] for d in valid_data))
            
            for x_site in x_sites:
                site_data = [d for d in valid_data if d['X_site'] == x_site]
                
                if not site_data:
                    continue
                
                x_values = [d[param] for d in site_data]
                y_values = [d['bandgap'] for d in site_data]
                
                # Create scatter plot
                ax.scatter(x_values, y_values, 
                          c=halogen_colors.get(x_site, '#888888'), 
                          s=100, alpha=0.8, 
                          label=f'{x_site} (n={len(site_data)})',
                          edgecolors='black', linewidth=0.5, zorder=3)
            
            # Add trend line
            if len(valid_data) > 1:
                x_vals = [d[param] for d in valid_data]
                y_vals = [d['bandgap'] for d in valid_data]
                
                # Calculate correlation
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                
                # Add trend line
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                ax.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2, zorder=4)
                
                # Add correlation coefficient to plot
                ax.text(0.05, 0.95, f'R = {correlation:.3f}', 
                       transform=ax.transAxes, fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Set appropriate x-axis limits based on parameter type
            x_vals = [d[param] for d in valid_data]
            if x_vals:
                x_min, x_max = min(x_vals), max(x_vals)
                
                # Define expected ranges for specific parameters
                param_ranges = {
                    'goldschmidt_tolerance': (0.8, 1.2),  # Goldschmidt tolerance factor
                    'octahedral_tolerance': (0.3, 0.8),   # Octahedral tolerance factor
                    'delta': (0, 0.1),                    # Delta distortion parameter
                    'sigma': (0, 100),                    # Sigma distortion parameter (degrees²)
                    'lambda_2': (1.0, 1.15),             # Lambda_2 distortion parameter
                    'lambda_3': (1.0, 1.15),             # Lambda_3 distortion parameter
                    'cis_angle_mean': (80, 100),         # Cis angles (degrees)
                    'trans_angle_mean': (160, 180),      # Trans angles (degrees)
                    'axial_central_axial_mean': (160, 180), # Axial-central-axial angles (degrees)
                    'central_axial_central_mean': (140, 180) # B-X-B angles (degrees)
                }
                
                if param in param_ranges:
                    expected_min, expected_max = param_ranges[param]
                    # Use actual data range but constrain to reasonable limits
                    ax.set_xlim(max(x_min * 0.95, expected_min), min(x_max * 1.05, expected_max))
                else:
                    # For other parameters, use data range with 5% padding
                    padding = (x_max - x_min) * 0.05
                    ax.set_xlim(x_min - padding, x_max + padding)
            
            # Customize plot
            ax.set_xlabel(f'{param_label}', fontsize=18, fontweight='bold', color='#333333')
            ax.set_ylabel('Bandgap (eV)', fontsize=18, fontweight='bold', color='#333333')
            ax.set_title(f'Bandgap vs {param_label}', fontsize=20, fontweight='bold', color='#222222', pad=20)
            
            # Clean axis styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#666666')
            ax.spines['bottom'].set_color('#666666')
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            
            # Make tick labels bigger
            ax.tick_params(axis='both', which='major', labelsize=16, colors='#333333', width=2, length=6)
            
            # Add legend
            ax.legend(fontsize=14, frameon=True, fancybox=False, shadow=False, 
                     framealpha=0.9, facecolor='white', edgecolor='black')
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            if save:
                filename = f'{filename_prefix}_{param}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Bandgap scatter plot with family background saved to: {filename}")
            
            figures.append(fig)
        
        return figures

    def create_bandgap_scatter_plots(self, 
                                    save: bool = True,
                                    filename_prefix: str = 'MAPbX3_bandgap_vs') -> List[plt.Figure]:
        """
        Create scatter plots of bandgap vs distortion parameters and angles.
        
        Parameters:
        save: Whether to save the plots
        filename_prefix: Prefix for saved filenames
        
        Returns:
        List[matplotlib.figure.Figure]: List of created figures
        """
        if not self.comparison_data:
            self.extract_comparison_data()
        
        # Collect data for scatter plots
        scatter_data = []
        
        for exp_name, exp_data in self.comparison_data['experiments'].items():
            if 'error' in exp_data:
                continue
            
            # Get basic info
            basic_info = exp_data.get('basic_info', {})
            X_site = basic_info.get('X_site', 'Unknown')
            layer_thickness = basic_info.get('layer_thickness', 'Unknown')
            
            # Get distortion data
            distortion_data = exp_data.get('distortion_analysis', {})
            
            # Get angular data
            angular_data = exp_data.get('angular_analysis', {})
            
            # Get bandgap (if available)
            bandgap = self.bandgap_data.get(exp_name)
            
            if bandgap is not None:
                scatter_data.append({
                    'exp_name': exp_name,
                    'X_site': X_site,
                    'layer_thickness': layer_thickness,
                    'bandgap': bandgap,
                    'delta': distortion_data.get('delta'),
                    'sigma': distortion_data.get('sigma'),
                    'lambda_3': distortion_data.get('lambda_3'),
                    'lambda_2': distortion_data.get('lambda_2'),
                    'cis_angle_mean': angular_data.get('cis_angles', {}).get('mean'),
                    'trans_angle_mean': angular_data.get('trans_angles', {}).get('mean'),
                    'axial_central_axial_mean': angular_data.get('axial_central_axial', {}).get('mean'),
                    'central_axial_central_mean': angular_data.get('central_axial_central', {}).get('mean')
                })
        
        if not scatter_data:
            print("No bandgap data available for scatter plots")
            return []
        
        # Create scatter plots
        figures = []
        
        # Parameters to plot against bandgap
        params = [
            ('lambda_2', 'λ₂'),
            ('lambda_3', 'λ₃'),
            ('delta', 'Δ'),
            ('sigma', 'σ'),
            ('cis_angle_mean', 'Cis Angle Mean'),
            ('trans_angle_mean', 'Trans Angle Mean'),
            ('axial_central_axial_mean', 'Axial-Central-Axial Mean'),
            ('central_axial_central_mean', 'B-X-B Mean')
        ]
        
        for param, param_label in params:
            # Filter out None values
            valid_data = [d for d in scatter_data if d.get(param) is not None]
            
            if not valid_data:
                print(f"No valid data for {param}")
                continue
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Separate data by X-site for coloring
            x_sites = list(set(d['X_site'] for d in valid_data))
            colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
            
            for x_site in x_sites:
                site_data = [d for d in valid_data if d['X_site'] == x_site]
                
                if not site_data:
                    continue
                
                x_values = [d[param] for d in site_data]
                y_values = [d['bandgap'] for d in site_data]
                
                # Create scatter plot
                ax.scatter(x_values, y_values, 
                          c=colors.get(x_site, '#888888'), 
                          s=100, alpha=0.7, 
                          label=f'{x_site} (n={len(site_data)})',
                          edgecolors='black', linewidth=0.5)
            
            # Add trend line
            if len(valid_data) > 1:
                x_vals = [d[param] for d in valid_data]
                y_vals = [d['bandgap'] for d in valid_data]
                
                # Calculate correlation
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                
                # Add trend line
                z = np.polyfit(x_vals, y_vals, 1)
                p = np.poly1d(z)
                x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                ax.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2)
                
                # Add correlation coefficient to plot
                ax.text(0.05, 0.95, f'R = {correlation:.3f}', 
                       transform=ax.transAxes, fontsize=14, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            
            # Set appropriate x-axis limits based on parameter type
            x_vals = [d[param] for d in valid_data]
            if x_vals:
                x_min, x_max = min(x_vals), max(x_vals)
                
                # Define expected ranges for specific parameters
                param_ranges = {
                    'goldschmidt_tolerance': (0.8, 1.2),  # Goldschmidt tolerance factor
                    'octahedral_tolerance': (0.3, 0.8),   # Octahedral tolerance factor
                    'delta': (0, 0.1),                    # Delta distortion parameter
                    'sigma': (0, 100),                    # Sigma distortion parameter (degrees²)
                    'lambda_2': (1.0, 1.15),             # Lambda_2 distortion parameter
                    'lambda_3': (1.0, 1.15),             # Lambda_3 distortion parameter
                    'cis_angle_mean': (80, 100),         # Cis angles (degrees)
                    'trans_angle_mean': (160, 180),      # Trans angles (degrees)
                    'axial_central_axial_mean': (160, 180), # Axial-central-axial angles (degrees)
                    'central_axial_central_mean': (140, 180) # B-X-B angles (degrees)
                }
                
                if param in param_ranges:
                    expected_min, expected_max = param_ranges[param]
                    # Use actual data range but constrain to reasonable limits
                    ax.set_xlim(max(x_min * 0.95, expected_min), min(x_max * 1.05, expected_max))
                else:
                    # For other parameters, use data range with 5% padding
                    padding = (x_max - x_min) * 0.05
                    ax.set_xlim(x_min - padding, x_max + padding)
            
            # Customize plot
            ax.set_xlabel(f'{param_label}', fontsize=18, fontweight='bold', color='#333333')
            ax.set_ylabel('Bandgap (eV)', fontsize=18, fontweight='bold', color='#333333')
            ax.set_title(f'Bandgap vs {param_label}', fontsize=20, fontweight='bold', color='#222222', pad=20)
            
            # Clean axis styling
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#666666')
            ax.spines['bottom'].set_color('#666666')
            ax.spines['left'].set_linewidth(2)
            ax.spines['bottom'].set_linewidth(2)
            
            # Make tick labels bigger
            ax.tick_params(axis='both', which='major', labelsize=16, colors='#333333', width=2, length=6)
            
            # Add legend
            ax.legend(fontsize=14, frameon=True, fancybox=False, shadow=False, 
                     framealpha=0.9, facecolor='white', edgecolor='black')
            
            # Add grid
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            plt.tight_layout()
            
            if save:
                filename = f'{filename_prefix}_{param}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"Bandgap scatter plot saved to: {filename}")
            
            figures.append(fig)
        
        return figures
    
    def print_bandgap_summary(self):
        """Print a summary of bandgap data found."""
        if not self.bandgap_data:
            print("\nNo bandgap data available.")
            return
        
        print("\n" + "=" * 60)
        print("BANDGAP DATA SUMMARY")
        print("=" * 60)
        
        bandgaps = list(self.bandgap_data.values())
        print(f"Total structures with bandgap data: {len(bandgaps)}")
        print(f"Bandgap range: {min(bandgaps):.3f} - {max(bandgaps):.3f} eV")
        print(f"Average bandgap: {np.mean(bandgaps):.3f} ± {np.std(bandgaps):.3f} eV")
        
        # Group by X-site
        x_site_bandgaps = {}
        for exp_name, bandgap in self.bandgap_data.items():
            exp_data = self.comparison_data.get('experiments', {}).get(exp_name, {})
            if 'error' not in exp_data:
                x_site = exp_data.get('basic_info', {}).get('X_site', 'Unknown')
                if x_site not in x_site_bandgaps:
                    x_site_bandgaps[x_site] = []
                x_site_bandgaps[x_site].append(bandgap)
        
        print(f"\nBandgap by X-site:")
        for x_site, bg_list in x_site_bandgaps.items():
            print(f"  {x_site}: {np.mean(bg_list):.3f} ± {np.std(bg_list):.3f} eV (n={len(bg_list)})")
        
        # Group by layer thickness
        layer_bandgaps = {}
        for exp_name, bandgap in self.bandgap_data.items():
            exp_data = self.comparison_data.get('experiments', {}).get(exp_name, {})
            if 'error' not in exp_data:
                layer_thickness = exp_data.get('basic_info', {}).get('layer_thickness', 'Unknown')
                if layer_thickness not in layer_bandgaps:
                    layer_bandgaps[layer_thickness] = []
                layer_bandgaps[layer_thickness].append(bandgap)
        
        print(f"\nBandgap by layer thickness:")
        for layer, bg_list in layer_bandgaps.items():
            print(f"  n={layer}: {np.mean(bg_list):.3f} ± {np.std(bg_list):.3f} eV (n={len(bg_list)})")
        
        print("\n" + "=" * 60)

        
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

    def create_bandgap_by_layer_thickness_plots(self, 
                                               save: bool = True,
                                               filename_prefix: str = 'MAPbX3_bandgap_by_layer_thickness') -> List[plt.Figure]:
        """
        Create bandgap plots grouped by layer thickness (n=1,2,3).
        
        Parameters:
        save: Whether to save the plots
        filename_prefix: Prefix for saved filenames
        
        Returns:
        List[matplotlib.figure.Figure]: List of created figures
        """
        if not self.comparison_data:
            self.extract_comparison_data()
        
        # Collect data for plots
        plot_data = []
        
        for exp_name, exp_data in self.comparison_data['experiments'].items():
            if 'error' in exp_data:
                continue
            
            # Get basic info
            basic_info = exp_data.get('basic_info', {})
            layer_thickness = basic_info.get('layer_thickness', 'Unknown')
            X_site = basic_info.get('X_site', 'Unknown')
            
            # Get bandgap (if available)
            bandgap = self.bandgap_data.get(exp_name)
            
            if bandgap is not None and layer_thickness != 'Unknown':
                plot_data.append({
                    'exp_name': exp_name,
                    'layer_thickness': layer_thickness,
                    'X_site': X_site,
                    'bandgap': bandgap
                })
        
        if not plot_data:
            print("No bandgap data available for layer thickness plots")
            return []
        
        # Create plots
        figures = []
        
        # 1. Box plot of bandgap by layer thickness
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        # Prepare data for box plot
        layer_data = {}
        for data in plot_data:
            n = data['layer_thickness']
            if n not in layer_data:
                layer_data[n] = []
            layer_data[n].append(data['bandgap'])
        
        # Create box plot
        layer_names = sorted(layer_data.keys())
        box_data = [layer_data[n] for n in layer_names]
        
        bp = ax1.boxplot(box_data, labels=[f'n={n}' for n in layer_names], 
                        patch_artist=True, showmeans=True)
        
        # Color the boxes
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax1.set_ylabel('Bandgap (eV)', fontsize=16, fontweight='bold')
        ax1.set_xlabel('Layer Thickness (n)', fontsize=16, fontweight='bold')
        ax1.set_title('Bandgap Distribution by Layer Thickness', fontsize=18, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = []
        for i, n in enumerate(layer_names):
            data = layer_data[n]
            mean_val = np.mean(data)
            std_val = np.std(data)
            stats_text.append(f'n={n}: {mean_val:.3f}±{std_val:.3f} eV (n={len(data)})')
        
        ax1.text(0.02, 0.98, '\n'.join(stats_text), transform=ax1.transAxes, 
                fontsize=12, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        if save:
            filename1 = f'{filename_prefix}_boxplot.png'
            plt.savefig(filename1, dpi=300, bbox_inches='tight')
            print(f"Bandgap box plot by layer thickness saved to: {filename1}")
        
        figures.append(fig1)
        
        # 2. Scatter plot with X-site coloring
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        # Color by X-site
        colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
        
        for x_site in ['Br', 'I', 'Cl']:
            site_data = [d for d in plot_data if d['X_site'] == x_site]
            if not site_data:
                continue
            
            x_vals = [d['layer_thickness'] for d in site_data]
            y_vals = [d['bandgap'] for d in site_data]
            
            ax2.scatter(x_vals, y_vals, c=colors[x_site], s=100, alpha=0.7, 
                       label=f'{x_site} (n={len(site_data)})', edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Layer Thickness (n)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Bandgap (eV)', fontsize=16, fontweight='bold')
        ax2.set_title('Bandgap vs Layer Thickness by X-site', fontsize=18, fontweight='bold')
        ax2.legend(fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Set x-axis to show only integer values
        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(['n=1', 'n=2', 'n=3'])
        
        plt.tight_layout()
        
        if save:
            filename2 = f'{filename_prefix}_scatter.png'
            plt.savefig(filename2, dpi=300, bbox_inches='tight')
            print(f"Bandgap scatter plot by layer thickness saved to: {filename2}")
        
        figures.append(fig2)
        
        return figures

    def create_vbm_cbm_plots(self, 
                           save: bool = True,
                           filename_prefix: str = 'MAPbX3_vbm_cbm') -> List[plt.Figure]:
        """
        Create VBM and CBM plots.
        
        Parameters:
        save: Whether to save the plots
        filename_prefix: Prefix for saved filenames
        
        Returns:
        List[matplotlib.figure.Figure]: List of created figures
        """
        if not self.comparison_data:
            self.extract_comparison_data()
        
        # Collect VBM/CBM data
        vbm_cbm_data = []
        
        for exp_name, exp_data in self.comparison_data['experiments'].items():
            if 'error' in exp_data:
                continue
            
            # Get basic info
            basic_info = exp_data.get('basic_info', {})
            layer_thickness = basic_info.get('layer_thickness', 'Unknown')
            X_site = basic_info.get('X_site', 'Unknown')
            
            # Try to get VBM/CBM data from vbm_cbm_data (if it was stored)
            vbm_cbm_info = getattr(self, 'vbm_cbm_data', {}).get(exp_name)
            bandgap = self.bandgap_data.get(exp_name)
            
            if vbm_cbm_info is not None and layer_thickness != 'Unknown':
                # Use actual VBM/CBM data from OUTCAR files
                vbm_cbm_data.append({
                    'exp_name': exp_name,
                    'layer_thickness': layer_thickness,
                    'X_site': X_site,
                    'bandgap': vbm_cbm_info['bandgap'],
                    'vbm': vbm_cbm_info['vbm'],
                    'cbm': vbm_cbm_info['cbm']
                })
            elif bandgap is not None and layer_thickness != 'Unknown':
                # Fallback: estimate VBM and CBM from bandgap
                vbm = -bandgap/2  # Rough estimate
                cbm = bandgap/2   # Rough estimate
                
                vbm_cbm_data.append({
                    'exp_name': exp_name,
                    'layer_thickness': layer_thickness,
                    'X_site': X_site,
                    'bandgap': bandgap,
                    'vbm': vbm,
                    'cbm': cbm
                })
        
        if not vbm_cbm_data:
            print("No VBM/CBM data available for plotting")
            return []
        
        # Create plots
        figures = []
        
        # 1. VBM vs Layer Thickness
        fig1, ax1 = plt.subplots(figsize=(12, 8))
        
        # Color by X-site
        colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
        
        for x_site in ['Br', 'I', 'Cl']:
            site_data = [d for d in vbm_cbm_data if d['X_site'] == x_site]
            if not site_data:
                continue
            
            x_vals = [d['layer_thickness'] for d in site_data]
            y_vals = [d['vbm'] for d in site_data]
            
            ax1.scatter(x_vals, y_vals, c=colors[x_site], s=100, alpha=0.7, 
                       label=f'{x_site} (n={len(site_data)})', edgecolors='black', linewidth=0.5)
        
        ax1.set_xlabel('Layer Thickness (n)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('VBM (eV)', fontsize=16, fontweight='bold')
        ax1.set_title('Valence Band Maximum vs Layer Thickness', fontsize=18, fontweight='bold')
        ax1.legend(fontsize=14)
        ax1.grid(True, alpha=0.3)
        ax1.set_xticks([1, 2, 3])
        ax1.set_xticklabels(['n=1', 'n=2', 'n=3'])
        
        plt.tight_layout()
        
        if save:
            filename1 = f'{filename_prefix}_vbm_by_layer.png'
            plt.savefig(filename1, dpi=300, bbox_inches='tight')
            print(f"VBM plot saved to: {filename1}")
        
        figures.append(fig1)
        
        # 2. CBM vs Layer Thickness
        fig2, ax2 = plt.subplots(figsize=(12, 8))
        
        for x_site in ['Br', 'I', 'Cl']:
            site_data = [d for d in vbm_cbm_data if d['X_site'] == x_site]
            if not site_data:
                continue
            
            x_vals = [d['layer_thickness'] for d in site_data]
            y_vals = [d['cbm'] for d in site_data]
            
            ax2.scatter(x_vals, y_vals, c=colors[x_site], s=100, alpha=0.7, 
                       label=f'{x_site} (n={len(site_data)})', edgecolors='black', linewidth=0.5)
        
        ax2.set_xlabel('Layer Thickness (n)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('CBM (eV)', fontsize=16, fontweight='bold')
        ax2.set_title('Conduction Band Minimum vs Layer Thickness', fontsize=18, fontweight='bold')
        ax2.legend(fontsize=14)
        ax2.grid(True, alpha=0.3)
        ax2.set_xticks([1, 2, 3])
        ax2.set_xticklabels(['n=1', 'n=2', 'n=3'])
        
        plt.tight_layout()
        
        if save:
            filename2 = f'{filename_prefix}_cbm_by_layer.png'
            plt.savefig(filename2, dpi=300, bbox_inches='tight')
            print(f"CBM plot saved to: {filename2}")
        
        figures.append(fig2)
        
        # 3. Combined VBM/CBM plot
        fig3, ax3 = plt.subplots(figsize=(14, 10))
        
        for x_site in ['Br', 'I', 'Cl']:
            site_data = [d for d in vbm_cbm_data if d['X_site'] == x_site]
            if not site_data:
                continue
            
            x_vals = [d['layer_thickness'] for d in site_data]
            vbm_vals = [d['vbm'] for d in site_data]
            cbm_vals = [d['cbm'] for d in site_data]
            
            ax3.scatter(x_vals, vbm_vals, c=colors[x_site], s=100, alpha=0.7, 
                       marker='o', label=f'{x_site} VBM (n={len(site_data)})', 
                       edgecolors='black', linewidth=0.5)
            ax3.scatter(x_vals, cbm_vals, c=colors[x_site], s=100, alpha=0.7, 
                       marker='s', label=f'{x_site} CBM (n={len(site_data)})', 
                       edgecolors='black', linewidth=0.5)
        
        ax3.set_xlabel('Layer Thickness (n)', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Energy (eV)', fontsize=16, fontweight='bold')
        ax3.set_title('VBM and CBM vs Layer Thickness', fontsize=18, fontweight='bold')
        ax3.legend(fontsize=12, ncol=2)
        ax3.grid(True, alpha=0.3)
        ax3.set_xticks([1, 2, 3])
        ax3.set_xticklabels(['n=1', 'n=2', 'n=3'])
        
        plt.tight_layout()
        
        if save:
            filename3 = f'{filename_prefix}_combined.png'
            plt.savefig(filename3, dpi=300, bbox_inches='tight')
            print(f"Combined VBM/CBM plot saved to: {filename3}")
        
        figures.append(fig3)
        
        return figures

    def create_correlation_analysis(self, 
                                  save: bool = True,
                                  filename_prefix: str = 'MAPbX3_correlation') -> Dict:
        """
        Create comprehensive correlation analysis (linear and non-linear) for all properties.
        
        Parameters:
        save: Whether to save the plots and data
        filename_prefix: Prefix for saved filenames
        
        Returns:
        dict: Dictionary containing correlation results and figures
        """
        if not self.comparison_data:
            self.extract_comparison_data()
        
        # Collect all data for correlation analysis
        correlation_data = []
        
        for exp_name, exp_data in self.comparison_data['experiments'].items():
            if 'error' in exp_data:
                continue
            
            # Get basic info
            basic_info = exp_data.get('basic_info', {})
            layer_thickness = basic_info.get('layer_thickness', 'Unknown')
            X_site = basic_info.get('X_site', 'Unknown')
            
            # Get distortion data
            distortion_data = exp_data.get('distortion_analysis', {})
            
            # Get angular data
            angular_data = exp_data.get('angular_analysis', {})
            
            # Get bandgap data
            bandgap = self.bandgap_data.get(exp_name)
            
            # Get VBM/CBM data
            vbm_cbm_info = getattr(self, 'vbm_cbm_data', {}).get(exp_name)
            
            # Prepare data row
            data_row = {
                'exp_name': exp_name,
                'layer_thickness': layer_thickness,
                'X_site': X_site,
                'bandgap': bandgap,
                'vbm': vbm_cbm_info['vbm'] if vbm_cbm_info else None,
                'cbm': vbm_cbm_info['cbm'] if vbm_cbm_info else None,
                'delta': distortion_data.get('delta'),
                'sigma': distortion_data.get('sigma'),
                'lambda_3': distortion_data.get('lambda_3'),
                'lambda_2': distortion_data.get('lambda_2'),
                'cis_angle_mean': angular_data.get('cis_angles', {}).get('mean'),
                'trans_angle_mean': angular_data.get('trans_angles', {}).get('mean'),
                'axial_central_axial_mean': angular_data.get('axial_central_axial', {}).get('mean'),
                'central_axial_central_mean': angular_data.get('central_axial_central', {}).get('mean')
            }
            
            # Only include rows with bandgap data
            if data_row['bandgap'] is not None:
                correlation_data.append(data_row)
        
        if not correlation_data:
            print("No data available for correlation analysis")
            return {}
        
        # Convert to DataFrame for easier analysis
        import pandas as pd
        df = pd.DataFrame(correlation_data)
        
        # Select numeric columns for correlation
        numeric_cols = ['bandgap', 'vbm', 'cbm', 'delta', 'sigma', 'lambda_3', 'lambda_2', 
                       'cis_angle_mean', 'trans_angle_mean', 'axial_central_axial_mean', 'central_axial_central_mean']
        
        # Filter out None values and create correlation matrix
        df_numeric = df[numeric_cols].dropna()
        
        if len(df_numeric) < 2:
            print("Insufficient data for correlation analysis")
            return {}
        
        # Calculate correlations
        results = {}
        
        # 1. Linear Pearson correlations
        pearson_corr = df_numeric.corr(method='pearson')
        results['pearson_correlations'] = pearson_corr
        
        # 2. Non-linear Spearman correlations
        spearman_corr = df_numeric.corr(method='spearman')
        results['spearman_correlations'] = spearman_corr
        
        # 3. Kendall tau correlations
        kendall_corr = df_numeric.corr(method='kendall')
        results['kendall_correlations'] = kendall_corr
        
        # Create plots
        figures = []
        
        # 1. Pearson correlation heatmap
        fig1, ax1 = plt.subplots(figsize=(14, 12))
        im1 = ax1.imshow(pearson_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add correlation values as text
        for i in range(len(pearson_corr.columns)):
            for j in range(len(pearson_corr.columns)):
                text = ax1.text(j, i, f'{pearson_corr.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax1.set_xticks(range(len(pearson_corr.columns)))
        ax1.set_yticks(range(len(pearson_corr.columns)))
        ax1.set_xticklabels(pearson_corr.columns, rotation=45, ha='right')
        ax1.set_yticklabels(pearson_corr.columns)
        ax1.set_title('Pearson Linear Correlations', fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Correlation Coefficient', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            filename1 = f'{filename_prefix}_pearson_heatmap.png'
            plt.savefig(filename1, dpi=300, bbox_inches='tight')
            print(f"Pearson correlation heatmap saved to: {filename1}")
        
        figures.append(fig1)
        
        # 2. Spearman correlation heatmap
        fig2, ax2 = plt.subplots(figsize=(14, 12))
        im2 = ax2.imshow(spearman_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        
        # Add correlation values as text
        for i in range(len(spearman_corr.columns)):
            for j in range(len(spearman_corr.columns)):
                text = ax2.text(j, i, f'{spearman_corr.iloc[i, j]:.3f}',
                               ha="center", va="center", color="black", fontweight='bold')
        
        ax2.set_xticks(range(len(spearman_corr.columns)))
        ax2.set_yticks(range(len(spearman_corr.columns)))
        ax2.set_xticklabels(spearman_corr.columns, rotation=45, ha='right')
        ax2.set_yticklabels(spearman_corr.columns)
        ax2.set_title('Spearman Non-Linear Correlations', fontsize=16, fontweight='bold', pad=20)
        
        # Add colorbar
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8)
        cbar2.set_label('Correlation Coefficient', fontsize=12)
        
        plt.tight_layout()
        
        if save:
            filename2 = f'{filename_prefix}_spearman_heatmap.png'
            plt.savefig(filename2, dpi=300, bbox_inches='tight')
            print(f"Spearman correlation heatmap saved to: {filename2}")
        
        figures.append(fig2)
        
        # 3. Comparison plot (Pearson vs Spearman)
        fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Pearson
        im3a = ax3a.imshow(pearson_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax3a.set_xticks(range(len(pearson_corr.columns)))
        ax3a.set_yticks(range(len(pearson_corr.columns)))
        ax3a.set_xticklabels(pearson_corr.columns, rotation=45, ha='right')
        ax3a.set_yticklabels(pearson_corr.columns)
        ax3a.set_title('Pearson (Linear)', fontsize=14, fontweight='bold')
        plt.colorbar(im3a, ax=ax3a, shrink=0.8)
        
        # Spearman
        im3b = ax3b.imshow(spearman_corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax3b.set_xticks(range(len(spearman_corr.columns)))
        ax3b.set_yticks(range(len(spearman_corr.columns)))
        ax3b.set_xticklabels(spearman_corr.columns, rotation=45, ha='right')
        ax3b.set_yticklabels(spearman_corr.columns)
        ax3b.set_title('Spearman (Non-Linear)', fontsize=14, fontweight='bold')
        plt.colorbar(im3b, ax=ax3b, shrink=0.8)
        
        plt.tight_layout()
        
        if save:
            filename3 = f'{filename_prefix}_comparison.png'
            plt.savefig(filename3, dpi=300, bbox_inches='tight')
            print(f"Correlation comparison plot saved to: {filename3}")
        
        figures.append(fig3)
        
        # Save correlation matrices to text files
        if save:
            # Pearson correlations
            pearson_file = f'{filename_prefix}_pearson_correlations.txt'
            with open(pearson_file, 'w') as f:
                f.write("PEARSON LINEAR CORRELATIONS\n")
                f.write("=" * 50 + "\n\n")
                f.write("Correlation Matrix:\n")
                f.write(pearson_corr.to_string())
                f.write("\n\n")
                
                # Find strongest correlations
                f.write("STRONGEST CORRELATIONS (|r| > 0.5):\n")
                f.write("-" * 40 + "\n")
                
                # Get upper triangle of correlation matrix
                upper_tri = pearson_corr.where(np.triu(np.ones(pearson_corr.shape), k=1).astype(bool))
                
                # Flatten and sort by absolute value
                corr_pairs = []
                for i in range(len(upper_tri.columns)):
                    for j in range(i+1, len(upper_tri.columns)):
                        corr_val = upper_tri.iloc[i, j]
                        if not pd.isna(corr_val) and abs(corr_val) > 0.5:
                            corr_pairs.append((upper_tri.columns[i], upper_tri.columns[j], corr_val))
                
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                for var1, var2, corr_val in corr_pairs:
                    f.write(f"{var1} vs {var2}: r = {corr_val:.4f}\n")
                
                f.write(f"\nTotal data points: {len(df_numeric)}\n")
                f.write(f"Variables analyzed: {len(numeric_cols)}\n")
            
            print(f"Pearson correlations saved to: {pearson_file}")
            
            # Spearman correlations
            spearman_file = f'{filename_prefix}_spearman_correlations.txt'
            with open(spearman_file, 'w') as f:
                f.write("SPEARMAN NON-LINEAR CORRELATIONS\n")
                f.write("=" * 50 + "\n\n")
                f.write("Correlation Matrix:\n")
                f.write(spearman_corr.to_string())
                f.write("\n\n")
                
                # Find strongest correlations
                f.write("STRONGEST CORRELATIONS (|ρ| > 0.5):\n")
                f.write("-" * 40 + "\n")
                
                # Get upper triangle of correlation matrix
                upper_tri = spearman_corr.where(np.triu(np.ones(spearman_corr.shape), k=1).astype(bool))
                
                # Flatten and sort by absolute value
                corr_pairs = []
                for i in range(len(upper_tri.columns)):
                    for j in range(i+1, len(upper_tri.columns)):
                        corr_val = upper_tri.iloc[i, j]
                        if not pd.isna(corr_val) and abs(corr_val) > 0.5:
                            corr_pairs.append((upper_tri.columns[i], upper_tri.columns[j], corr_val))
                
                corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
                
                for var1, var2, corr_val in corr_pairs:
                    f.write(f"{var1} vs {var2}: ρ = {corr_val:.4f}\n")
                
                f.write(f"\nTotal data points: {len(df_numeric)}\n")
                f.write(f"Variables analyzed: {len(numeric_cols)}\n")
            
            print(f"Spearman correlations saved to: {spearman_file}")
            
            # Combined analysis file
            combined_file = f'{filename_prefix}_combined_analysis.txt'
            with open(combined_file, 'w') as f:
                f.write("COMPREHENSIVE CORRELATION ANALYSIS\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("ANALYSIS SUMMARY:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total experiments: {len(correlation_data)}\n")
                f.write(f"Valid data points: {len(df_numeric)}\n")
                f.write(f"Variables analyzed: {len(numeric_cols)}\n")
                f.write(f"Missing data points: {len(correlation_data) - len(df_numeric)}\n\n")
                
                f.write("VARIABLES INCLUDED:\n")
                f.write("-" * 20 + "\n")
                for i, var in enumerate(numeric_cols, 1):
                    f.write(f"{i:2d}. {var}\n")
                
                f.write("\n" + "=" * 60 + "\n")
                f.write("PEARSON LINEAR CORRELATIONS\n")
                f.write("=" * 60 + "\n")
                f.write(pearson_corr.to_string())
                
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("SPEARMAN NON-LINEAR CORRELATIONS\n")
                f.write("=" * 60 + "\n")
                f.write(spearman_corr.to_string())
                
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("KENDALL TAU CORRELATIONS\n")
                f.write("=" * 60 + "\n")
                f.write(kendall_corr.to_string())
                
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("INTERPRETATION GUIDE\n")
                f.write("=" * 60 + "\n")
                f.write("Pearson (r): Measures linear relationships\n")
                f.write("  |r| > 0.8: Very strong correlation\n")
                f.write("  0.6 < |r| ≤ 0.8: Strong correlation\n")
                f.write("  0.4 < |r| ≤ 0.6: Moderate correlation\n")
                f.write("  0.2 < |r| ≤ 0.4: Weak correlation\n")
                f.write("  |r| ≤ 0.2: Very weak or no correlation\n\n")
                
                f.write("Spearman (ρ): Measures monotonic relationships\n")
                f.write("  |ρ| > 0.8: Very strong monotonic relationship\n")
                f.write("  0.6 < |ρ| ≤ 0.8: Strong monotonic relationship\n")
                f.write("  0.4 < |ρ| ≤ 0.6: Moderate monotonic relationship\n")
                f.write("  0.2 < |ρ| ≤ 0.4: Weak monotonic relationship\n")
                f.write("  |ρ| ≤ 0.2: Very weak or no monotonic relationship\n\n")
                
                f.write("Kendall (τ): Measures rank correlation\n")
                f.write("  |τ| > 0.7: Very strong rank correlation\n")
                f.write("  0.5 < |τ| ≤ 0.7: Strong rank correlation\n")
                f.write("  0.3 < |τ| ≤ 0.5: Moderate rank correlation\n")
                f.write("  0.1 < |τ| ≤ 0.3: Weak rank correlation\n")
                f.write("  |τ| ≤ 0.1: Very weak or no rank correlation\n")
            
            print(f"Combined analysis saved to: {combined_file}")
        
        results['figures'] = figures
        results['data_points'] = len(df_numeric)
        results['variables'] = numeric_cols
        
        return results

def export_comprehensive_data_to_csv(batch_analyzer, band_edges_data=None, output_file="comprehensive_analysis_data.csv"):
    """
    Export all structural and electronic data to a comprehensive CSV file.
    
    Args:
        batch_analyzer: BatchAnalyzer instance with loaded experiments
        band_edges_data: Dictionary with band edges data from CSV files
        output_file: Output CSV filename
    """
    print("\n" + "=" * 60)
    print("EXPORTING COMPREHENSIVE DATA TO CSV")
    print("=" * 60)
    
    all_data = []
    
    # Process each experiment
    for exp_name, analyzer in batch_analyzer.experiments.items():
        try:
            # Get basic experiment info
            row_data = {
                'Experiment': exp_name,
                'X_Site': 'Unknown',
                'Layer_Thickness': getattr(analyzer, 'layer_thickness', None),
                'Spacer_SMILES': 'Unknown',
                'Molecule': 'Unknown'
            }
            
            # Extract X-site from experiment name
            if 'MAPbBr3' in exp_name:
                row_data['X_Site'] = 'Br'
            elif 'MAPbI3' in exp_name:
                row_data['X_Site'] = 'I'
            elif 'MAPbCl3' in exp_name:
                row_data['X_Site'] = 'Cl'
            
            # Extract spacer SMILES from experiment name
            # Expected format: MAPbX3_n{number}_{spacer_smiles}
            parts = exp_name.split('_')
            if len(parts) >= 3:
                spacer_smiles = '_'.join(parts[2:])  # Everything after MAPbX3_n{number}_
                row_data['Spacer_SMILES'] = spacer_smiles
                row_data['Molecule'] = spacer_smiles  # Use same value for both columns
            
            # Get ontology data
            ontology = analyzer.get_ontology()
            
            # Basic structural info
            cell_props = ontology.get('cell_properties', {})
            composition = cell_props.get('composition', {})
            row_data.update({
                'B_Site': composition.get('metal_B', 'Unknown'),
                'N_Octahedra': composition.get('number_of_octahedra', 0),
                'N_Atoms': composition.get('number_of_atoms', 0),
                'Cell_Volume': cell_props.get('structure_info', {}).get('cell_volume', 0)
            })
            
            # Distortion parameters
            distortion_data = ontology.get('distortion_analysis', {})
            row_data.update({
                'Delta': distortion_data.get('delta_analysis', {}).get('overall_delta'),
                'Sigma': distortion_data.get('sigma_analysis', {}).get('overall_sigma'),
                'Lambda_3': distortion_data.get('lambda_analysis', {}).get('lambda_3'),
                'Lambda_2': distortion_data.get('lambda_analysis', {}).get('lambda_2'),
                'Goldschmidt_Tolerance': distortion_data.get('tolerance_factors', {}).get('goldschmidt_tolerance'),
                'Octahedral_Tolerance': distortion_data.get('tolerance_factors', {}).get('octahedral_tolerance')
            })
            
            # Angular parameters
            octahedra_data = ontology.get('octahedra', {})
            angular_stats = analyzer.angular_analyzer.get_angular_distribution_statistics(octahedra_data)
            row_data.update({
                'Cis_Angle_Mean': angular_stats.get('cis_angles', {}).get('mean'),
                'Cis_Angle_Std': angular_stats.get('cis_angles', {}).get('std'),
                'Trans_Angle_Mean': angular_stats.get('trans_angles', {}).get('mean'),
                'Trans_Angle_Std': angular_stats.get('trans_angles', {}).get('std'),
                'Axial_Central_Axial_Mean': angular_stats.get('axial_central_axial', {}).get('mean'),
                'Axial_Central_Axial_Std': angular_stats.get('axial_central_axial', {}).get('std'),
                'Central_Axial_Central_Mean': angular_stats.get('central_axial_central', {}).get('mean'),
                'Central_Axial_Central_Std': angular_stats.get('central_axial_central', {}).get('std')
            })
            
            # Bandgap data
            if exp_name in batch_analyzer.bandgap_data:
                row_data['Bandgap'] = batch_analyzer.bandgap_data[exp_name]
            else:
                row_data['Bandgap'] = None
            
            # VBM/CBM data
            if hasattr(batch_analyzer, 'vbm_cbm_data') and exp_name in batch_analyzer.vbm_cbm_data:
                vbm_cbm = batch_analyzer.vbm_cbm_data[exp_name]
                row_data.update({
                    'VBM': vbm_cbm.get('vbm'),
                    'CBM': vbm_cbm.get('cbm')
                })
            else:
                row_data.update({
                    'VBM': None,
                    'CBM': None
                })
            
            # Band edges data from CSV files
            if band_edges_data and exp_name in band_edges_data:
                band_data = band_edges_data[exp_name]['data']
                
                # Spacer data
                if 'SPACER' in band_data:
                    spacer = band_data['SPACER']
                    row_data.update({
                        'Spacer_VBM': spacer['VBM'],
                        'Spacer_CBM': spacer['CBM'],
                        'Spacer_Band_Gap': spacer['Band_Gap'],
                        'Spacer_HOMO': spacer['VBM'],
                        'Spacer_LUMO': spacer['CBM'],
                        'Spacer_Has_DOS_at_Fermi': spacer['Has_DOS_at_Fermi']
                    })
                
                # Slab data
                if 'SLAB' in band_data:
                    slab = band_data['SLAB']
                    row_data.update({
                        'Slab_VBM': slab['VBM'],
                        'Slab_CBM': slab['CBM'],
                        'Slab_Band_Gap': slab['Band_Gap'],
                        'Slab_Has_DOS_at_Fermi': slab['Has_DOS_at_Fermi']
                    })
            
            all_data.append(row_data)
            
        except Exception as e:
            print(f"  ⚠ Error processing {exp_name}: {e}")
            continue
    
    # Create DataFrame and save to CSV
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(output_file, index=False)
        print(f"✓ Comprehensive data exported to: {output_file}")
        print(f"  Total experiments: {len(df)}")
        print(f"  Total columns: {len(df.columns)}")
        print(f"  Columns: {', '.join(df.columns)}")
        return df
    else:
        print("✗ No data to export")
        return None

def create_plots_from_csv(csv_file="unified_octahedral_molecular_dataset.csv"):
    """
    Create all possible plots from the comprehensive CSV file.
    
    Args:
        csv_file: Path to the comprehensive CSV file
    """
    print("\n" + "=" * 60)
    print("CREATING ALL PLOTS FROM CSV")
    print("=" * 60)
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded data from {csv_file}")
        print(f"  Total experiments: {len(df)}")
        print(f"  Total variables: {len(df.columns)}")
        
        all_figures = []
        
        # 1. Create correlation analysis
        print("\n" + "=" * 40)
        print("CREATING CORRELATION ANALYSIS")
        print("=" * 40)
        
        correlation_figs = create_correlation_analysis_from_csv(csv_file)
        if correlation_figs:
            all_figures.extend(correlation_figs)
            print(f"✓ Created {len(correlation_figs)} correlation plots")
        
        # 2. Create bandgap scatter plots if bandgap data is available
        if 'Bandgap' in df.columns and df['Bandgap'].notna().any():
            print("\n" + "=" * 40)
            print("CREATING BANDGAP SCATTER PLOTS")  
            print("=" * 40)
            
            # Create scatter plots for key variables
            variables = ['Delta', 'Sigma', 'Lambda_3', 'Lambda_2', 
                        'Cis_Angle_Mean', 'Trans_Angle_Mean', 'Axial_Central_Axial_Mean', 'Central_Axial_Central_Mean']
            
            for var in variables:
                if var in df.columns and df[var].notna().any():
                    try:
                        fig, ax = plt.subplots(figsize=(10, 8))
                        
                        # Filter data with both variables
                        plot_data = df[[var, 'Bandgap', 'X_Site']].dropna()
                        
                        if len(plot_data) > 0:
                            # Color by X-site
                            colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
                            
                            for x_site in plot_data['X_Site'].unique():
                                site_data = plot_data[plot_data['X_Site'] == x_site]
                                ax.scatter(site_data[var], site_data['Bandgap'], 
                                          c=colors.get(x_site, '#888888'), 
                                          s=100, alpha=0.7, 
                                          label=f'{x_site} (n={len(site_data)})',
                                          edgecolors='black', linewidth=0.5)
                            
                            # Add trend line
                            x_vals = plot_data[var].values
                            y_vals = plot_data['Bandgap'].values
                            if len(x_vals) > 1:
                                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                                z = np.polyfit(x_vals, y_vals, 1)
                                p = np.poly1d(z)
                                x_trend = np.linspace(min(x_vals), max(x_vals), 100)
                                ax.plot(x_trend, p(x_trend), "k--", alpha=0.8, linewidth=2)
                                
                                # Add correlation to plot
                                ax.text(0.05, 0.95, f'R = {correlation:.3f}', 
                                       transform=ax.transAxes, fontsize=14, 
                                       bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                            
                            # Set appropriate x-axis limits based on variable type
                            x_vals = plot_data[var].values
                            if len(x_vals) > 0:
                                x_min, x_max = min(x_vals), max(x_vals)
                                
                                # Define expected ranges for specific variables
                                var_ranges = {
                                    'Goldschmidt_Tolerance': (0.8, 1.2),  # Goldschmidt tolerance factor
                                    'Octahedral_Tolerance': (0.3, 0.8),   # Octahedral tolerance factor  
                                    'Delta': (0, 0.1),                    # Delta distortion parameter
                                    'Sigma': (0, 100),                    # Sigma distortion parameter (degrees²)
                                    'Lambda_2': (1.0, 1.15),             # Lambda_2 distortion parameter
                                    'Lambda_3': (1.0, 1.15),             # Lambda_3 distortion parameter
                                    'Cis_Angle_Mean': (80, 100),         # Cis angles (degrees)
                                    'Trans_Angle_Mean': (160, 180),      # Trans angles (degrees)
                                    'Axial_Central_Axial_Mean': (160, 180), # Axial-central-axial angles (degrees)
                                    'Central_Axial_Central_Mean': (140, 180) # B-X-B angles (degrees)
                                }
                                
                                if var in var_ranges:
                                    expected_min, expected_max = var_ranges[var]
                                    # Use actual data range but constrain to reasonable limits
                                    ax.set_xlim(max(x_min * 0.95, expected_min), min(x_max * 1.05, expected_max))
                                else:
                                    # For other parameters, use data range with 5% padding
                                    padding = (x_max - x_min) * 0.05 if x_max != x_min else 0.1
                                    ax.set_xlim(x_min - padding, x_max + padding)
                            
                            ax.set_xlabel(var.replace('_', ' '), fontsize=16, fontweight='bold')
                            ax.set_ylabel('Bandgap (eV)', fontsize=16, fontweight='bold')
                            ax.set_title(f'Bandgap vs {var.replace("_", " ")}', fontsize=18, fontweight='bold')
                            ax.legend(fontsize=12)
                            ax.grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            
                            filename = f'MAPbX3_bandgap_vs_{var.lower()}.png'
                            plt.savefig(filename, dpi=300, bbox_inches='tight')
                            print(f"  ✓ Saved: {filename}")
                            all_figures.append(fig)
                    
                    except Exception as e:
                        print(f"  ⚠ Error creating plot for {var}: {e}")
        
        # 3. Create VBM/CBM plots if available
        if all(col in df.columns for col in ['VBM', 'CBM', 'Layer_Thickness']):
            if df[['VBM', 'CBM', 'Layer_Thickness']].notna().any().any():
                print("\n" + "=" * 40)
                print("CREATING VBM/CBM PLOTS")
                print("=" * 40)
                
                try:
                    # VBM vs Layer Thickness
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plot_data = df[['VBM', 'Layer_Thickness', 'X_Site']].dropna()
                    
                    colors = {'Br': '#FD7949', 'I': '#5A9AE4', 'Cl': '#419667'}
                    for x_site in plot_data['X_Site'].unique():
                        site_data = plot_data[plot_data['X_Site'] == x_site]
                        ax.scatter(site_data['Layer_Thickness'], site_data['VBM'], 
                                  c=colors.get(x_site, '#888888'), s=100, alpha=0.7, 
                                  label=f'{x_site} (n={len(site_data)})',
                                  edgecolors='black', linewidth=0.5)
                    
                    ax.set_xlabel('Layer Thickness (n)', fontsize=16, fontweight='bold')
                    ax.set_ylabel('VBM (eV)', fontsize=16, fontweight='bold')
                    ax.set_title('VBM vs Layer Thickness', fontsize=18, fontweight='bold')
                    ax.legend(fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.set_xticks([1, 2, 3])
                    
                    plt.tight_layout()
                    filename = 'MAPbX3_vbm_by_X_site_and_layer.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"  ✓ Saved: {filename}")
                    all_figures.append(fig)
                    
                    # CBM vs Layer Thickness
                    fig, ax = plt.subplots(figsize=(10, 8))
                    plot_data = df[['CBM', 'Layer_Thickness', 'X_Site']].dropna()
                    
                    for x_site in plot_data['X_Site'].unique():
                        site_data = plot_data[plot_data['X_Site'] == x_site]
                        ax.scatter(site_data['Layer_Thickness'], site_data['CBM'], 
                                  c=colors.get(x_site, '#888888'), s=100, alpha=0.7, 
                                  label=f'{x_site} (n={len(site_data)})',
                                  edgecolors='black', linewidth=0.5)
                    
                    ax.set_xlabel('Layer Thickness (n)', fontsize=16, fontweight='bold')
                    ax.set_ylabel('CBM (eV)', fontsize=16, fontweight='bold')
                    ax.set_title('CBM vs Layer Thickness', fontsize=18, fontweight='bold')
                    ax.legend(fontsize=12)
                    ax.grid(True, alpha=0.3)
                    ax.set_xticks([1, 2, 3])
                    
                    plt.tight_layout()
                    filename = 'MAPbX3_cbm_by_X_site_and_layer.png'
                    plt.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"  ✓ Saved: {filename}")
                    all_figures.append(fig)
                
                except Exception as e:
                    print(f"  ⚠ Error creating VBM/CBM plots: {e}")
        
        print(f"\n✓ Total plots created: {len(all_figures)}")
        return all_figures
        
    except Exception as e:
        print(f"✗ Error creating plots from CSV: {e}")
        return []

def create_correlation_analysis_from_csv(csv_file="unified_octahedral_molecular_dataset.csv"):
    """
    Create correlation analysis from the comprehensive CSV file.
    
    Args:
        csv_file: Path to the comprehensive CSV file
    """
    print("\n" + "=" * 60)
    print("CREATING CORRELATION ANALYSIS FROM CSV")
    print("=" * 60)
    
    try:
        # Load data
        df = pd.read_csv(csv_file)
        print(f"✓ Loaded data from {csv_file}")
        print(f"  Total experiments: {len(df)}")
        print(f"  Total variables: {len(df.columns)}")
        
        # Select numeric columns for correlation
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        correlation_data = df[numeric_cols].dropna()
        
        if len(correlation_data) < 2:
            print("Not enough numeric data for correlation analysis")
            return None
        
        print(f"  Numeric variables: {len(numeric_cols)}")
        
        # Create correlation matrices
        pearson_corr = correlation_data.corr(method='pearson')
        spearman_corr = correlation_data.corr(method='spearman')
        
        # Create heatmaps
        figures = []
        
        # Pearson correlation heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(pearson_corr, dtype=bool))
        sns.heatmap(pearson_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Pearson Correlation Matrix (Comprehensive Data)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = 'MAPbX3_correlation_pearson_comprehensive.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        figures.append(fig)
        
        # Spearman correlation heatmap
        fig, ax = plt.subplots(figsize=(14, 12))
        mask = np.triu(np.ones_like(spearman_corr, dtype=bool))
        sns.heatmap(spearman_corr, mask=mask, annot=True, cmap='RdBu_r', center=0,
                    square=True, fmt='.3f', cbar_kws={"shrink": .8}, ax=ax)
        ax.set_title('Spearman Correlation Matrix (Comprehensive Data)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        filename = 'MAPbX3_correlation_spearman_comprehensive.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {filename}")
        figures.append(fig)
        
        # Save correlation data
        pearson_filename = 'MAPbX3_correlation_pearson_comprehensive.txt'
        with open(pearson_filename, 'w') as f:
            f.write("Pearson Correlation Matrix (Comprehensive Data)\n")
            f.write("=" * 60 + "\n\n")
            f.write(pearson_corr.to_string())
        
        spearman_filename = 'MAPbX3_correlation_spearman_comprehensive.txt'
        with open(spearman_filename, 'w') as f:
            f.write("Spearman Correlation Matrix (Comprehensive Data)\n")
            f.write("=" * 60 + "\n\n")
            f.write(spearman_corr.to_string())
        
        print(f"  ✓ Saved correlation data files")
        print(f"✓ Created {len(figures)} correlation plots")
        return figures
        
    except Exception as e:
        print(f"✗ Error creating correlation analysis: {e}")
        return None
