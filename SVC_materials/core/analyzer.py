import pandas as pd
import numpy as np
import os
from ase.io import read
from ..utils.file_handlers import vasp_load, save_vasp
from ..utils.plots import plot_gaussian_projection, plot_multi_element_comparison, save_beautiful_plot, create_summary_plot
from ..utils.octadist.io import extract_octa
from ..utils.octadist.calc import CalcDistortion

class q2D_analyzer:
    """
    A class for analyzing 2D quantum materials using VASP results.
    """
    
    def __init__(self, file_path=None, b='Pb', x='Cl', cutoff_ref_ligand=3.5):
        """
        Initialize the q2D_analyzer.
        
        Parameters:
        file_path (str, optional): Path to the VASP file to load initially
        b (str): Central atom symbol (e.g., 'Pb')
        x (str): Ligand atom symbol (e.g., 'Cl')
        cutoff_ref_ligand (float): Distance cutoff for identifying ligands
        """
        # Load the structure as atom object
        self.atoms = read(file_path)
        self.b = b
        self.x = x
        self.cutoff_ref_ligand = cutoff_ref_ligand
        self.all_octahedra = self.find_all_octahedra()
        self.ordered_octahedra = self.order_octahedra()
        self.distortion_data = None

    def find_all_octahedra(self):
        """
        Automatically identify all octahedra in the given ASE Atoms object.
        
        Returns:
            List of tuples: (central_atom_symbol, central_index, atom_octa, coord_octa)
        """
        all_octa = []
        
        # Central symbols - fix the attribute reference
        central_symbols = [self.b]

        # Get atomic symbols and coordinates
        atom_symbols = self.atoms.get_chemical_symbols()
        coord = self.atoms.get_positions()
        
        # Find indices of potential central atoms
        central_indices = [i for i, sym in enumerate(atom_symbols) if sym in central_symbols]
        
        for ref_index in central_indices:
            # Extract octahedron around this center
            atom_octa, coord_octa = extract_octa(atom_symbols, coord, ref_index=ref_index, cutoff_ref_ligand=self.cutoff_ref_ligand)
            
            # Verify if it's a proper octahedron
            dist = CalcDistortion(coord_octa)
            if dist.non_octa:
                print(f"Warning: Non-octahedral structure around {atom_symbols[ref_index]} at index {ref_index}")
                continue
            
            all_octa.append((atom_symbols[ref_index], ref_index, atom_octa, coord_octa))
        
        return all_octa

    def order_octahedra(self):
        """
        Order octahedra systematically for comparison between structures.
        Orders by Z coordinate first, then by A (x,y) coordinates.
        
        Returns:
            List of ordered octahedra with consistent indexing
        """
        if not self.all_octahedra:
            return []
        
        # Extract central atom coordinates for sorting
        octahedra_with_coords = []
        for i, (symbol, index, atom_octa, coord_octa) in enumerate(self.all_octahedra):
            # Central atom is always the first in coord_octa
            central_coord = coord_octa[0]
            octahedra_with_coords.append({
                'original_index': i,
                'symbol': symbol,
                'global_index': index,
                'atom_octa': atom_octa,
                'coord_octa': coord_octa,
                'central_coord': central_coord,
                'z': central_coord[2],
                'x': central_coord[0],
                'y': central_coord[1]
            })
        
        # Sort by Z first, then by X, then by Y
        ordered_octahedra = sorted(octahedra_with_coords, 
                                 key=lambda octa: (octa['z'], octa['x'], octa['y']))
        
        # Add ordered index for reference
        for i, octa in enumerate(ordered_octahedra):
            octa['ordered_index'] = i
        
        return ordered_octahedra

    def calculate_octahedral_distortions(self):
        """
        Calculate distortion parameters for all ordered octahedra.
        
        Returns:
            List of dictionaries containing distortion parameters for each octahedron
        """
        distortion_results = []
        
        for octa_data in self.ordered_octahedra:
            coord_octa = octa_data['coord_octa']
            
            # Calculate distortion parameters
            dist_calc = CalcDistortion(coord_octa)
            
            # Create comprehensive distortion dictionary
            distortion_dict = {
                # Identification
                'ordered_index': octa_data['ordered_index'],
                'global_index': octa_data['global_index'],
                'central_symbol': octa_data['symbol'],
                'central_coord': octa_data['central_coord'].tolist(),
                
                # Bond distances
                'bond_distances': dist_calc.bond_dist.tolist(),
                'mean_bond_distance': dist_calc.d_mean,
                'bond_distance_differences': dist_calc.diff_dist.tolist(),
                
                # Distortion parameters
                'zeta': dist_calc.zeta,  # Sum of absolute deviations from mean bond length
                'delta': dist_calc.delta,  # Tilting distortion parameter
                'sigma': dist_calc.sigma,  # Angular distortion parameter
                
                # Theta parameters
                'theta_mean': dist_calc.theta,  # Mean theta parameter
                'theta_min': dist_calc.theta_min,  # Minimum theta parameter
                'theta_max': dist_calc.theta_max,  # Maximum theta parameter
                'eight_theta': dist_calc.eight_theta,  # All 8 theta values
                
                # Bond angles
                'cis_angles': dist_calc.cis_angle,  # 12 cis angles
                'trans_angles': dist_calc.trans_angle,  # 3 trans angles
                
                # Volume
                'octahedral_volume': dist_calc.oct_vol,
                
                # Quality check
                'is_octahedral': not dist_calc.non_octa,
                
                # Atomic composition
                'atom_symbols': octa_data['atom_octa'],
                'coordinates': coord_octa.tolist()
            }
            
            distortion_results.append(distortion_dict)
        
        self.distortion_data = distortion_results
        return distortion_results

    def get_distortion_summary(self):
        """
        Get a summary DataFrame of distortion parameters for all octahedra.
        
        Returns:
            pandas.DataFrame: Summary of key distortion parameters
        """
        if self.distortion_data is None:
            self.calculate_octahedral_distortions()
        
        summary_data = []
        for dist_data in self.distortion_data:
            summary_row = {
                'ordered_index': dist_data['ordered_index'],
                'global_index': dist_data['global_index'],
                'central_symbol': dist_data['central_symbol'],
                'x': dist_data['central_coord'][0],
                'y': dist_data['central_coord'][1],
                'z': dist_data['central_coord'][2],
                'mean_bond_distance': dist_data['mean_bond_distance'],
                'zeta': dist_data['zeta'],
                'delta': dist_data['delta'],
                'sigma': dist_data['sigma'],
                'theta_mean': dist_data['theta_mean'],
                'theta_min': dist_data['theta_min'],
                'theta_max': dist_data['theta_max'],
                'octahedral_volume': dist_data['octahedral_volume'],
                'is_octahedral': dist_data['is_octahedral']
            }
            summary_data.append(summary_row)
        
        return pd.DataFrame(summary_data)

    def compare_distortions(self, other_analyzer):
        """
        Compare distortion parameters between two structures.
        
        Parameters:
            other_analyzer: Another q2D_analyzer instance
            
        Returns:
            pandas.DataFrame: Comparison of distortion parameters
        """
        if self.distortion_data is None:
            self.calculate_octahedral_distortions()
        if other_analyzer.distortion_data is None:
            other_analyzer.calculate_octahedral_distortions()
        
        # Get summary DataFrames
        self_summary = self.get_distortion_summary()
        other_summary = other_analyzer.get_distortion_summary()
        
        # Ensure same number of octahedra for comparison
        min_octahedra = min(len(self_summary), len(other_summary))
        
        comparison_data = []
        for i in range(min_octahedra):
            self_row = self_summary.iloc[i]
            other_row = other_summary.iloc[i]
            
            comparison_row = {
                'octahedron_index': i,
                'structure1_zeta': self_row['zeta'],
                'structure2_zeta': other_row['zeta'],
                'zeta_difference': abs(self_row['zeta'] - other_row['zeta']),
                'structure1_delta': self_row['delta'],
                'structure2_delta': other_row['delta'],
                'delta_difference': abs(self_row['delta'] - other_row['delta']),
                'structure1_sigma': self_row['sigma'],
                'structure2_sigma': other_row['sigma'],
                'sigma_difference': abs(self_row['sigma'] - other_row['sigma']),
                'structure1_theta': self_row['theta_mean'],
                'structure2_theta': other_row['theta_mean'],
                'theta_difference': abs(self_row['theta_mean'] - other_row['theta_mean'])
            }
            comparison_data.append(comparison_row)
        
        return pd.DataFrame(comparison_data)

    def export_distortion_data(self, filename=None):
        """
        Export distortion data to CSV file.
        
        Parameters:
            filename (str, optional): Output filename. If None, uses default naming.
        """
        if self.distortion_data is None:
            self.calculate_octahedral_distortions()
        
        if filename is None:
            filename = "octahedral_distortion_analysis.csv"
        
        summary_df = self.get_distortion_summary()
        summary_df.to_csv(filename, index=False)
        print(f"Distortion data exported to {filename}")

    def get_octahedron_by_index(self, ordered_index):
        """
        Get detailed information about a specific octahedron by its ordered index.
        
        Parameters:
            ordered_index (int): The ordered index of the octahedron
            
        Returns:
            dict: Detailed distortion data for the specified octahedron
        """
        if self.distortion_data is None:
            self.calculate_octahedral_distortions()
        
        for dist_data in self.distortion_data:
            if dist_data['ordered_index'] == ordered_index:
                return dist_data
        
        raise ValueError(f"No octahedron found with ordered index {ordered_index}")

    def print_distortion_summary(self):
        """
        Print a formatted summary of distortion parameters.
        """
        if self.distortion_data is None:
            self.calculate_octahedral_distortions()
        
        print(f"\n=== Octahedral Distortion Analysis ===")
        print(f"Total octahedra found: {len(self.distortion_data)}")
        print(f"Central atom: {self.b}")
        print(f"Ligand cutoff: {self.cutoff_ref_ligand} Å")
        
        summary_df = self.get_distortion_summary()
        print(f"\n{summary_df.to_string(index=False)}")
        
        # Statistical summary
        print(f"\n=== Statistical Summary ===")
        numeric_cols = ['zeta', 'delta', 'sigma', 'theta_mean', 'mean_bond_distance']
        for col in numeric_cols:
            if col in summary_df.columns:
                print(f"{col}: mean={summary_df[col].mean():.4f}, std={summary_df[col].std():.4f}")
