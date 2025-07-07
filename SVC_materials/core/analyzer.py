import pandas as pd
import numpy as np
import os
from ..utils.file_handlers import vasp_load, save_vasp
from ..utils.plots import plot_gaussian_projection, plot_multi_element_comparison, save_beautiful_plot, create_summary_plot

# Ionic radii database (in Angstroms) - common ionic states
IONIC_RADII = {
    'H': 0.31,    # H-
    'Li': 0.76,   # Li+
    'Be': 0.45,   # Be2+
    'B': 0.27,    # B3+
    'C': 0.30,    # C4+ (estimated)
    'N': 1.46,    # N3-
    'O': 1.40,    # O2-
    'F': 1.33,    # F-
    'Na': 1.02,   # Na+
    'Mg': 0.72,   # Mg2+
    'Al': 0.54,   # Al3+
    'Si': 0.40,   # Si4+
    'P': 2.12,    # P3-
    'S': 1.84,    # S2-
    'Cl': 1.81,   # Cl-
    'K': 1.38,    # K+
    'Ca': 1.00,   # Ca2+
    'Ti': 0.61,   # Ti4+
    'V': 0.59,    # V5+
    'Cr': 0.62,   # Cr3+
    'Mn': 0.83,   # Mn2+
    'Fe': 0.78,   # Fe2+
    'Co': 0.75,   # Co2+
    'Ni': 0.69,   # Ni2+
    'Cu': 0.73,   # Cu2+
    'Zn': 0.74,   # Zn2+
    'Ga': 0.62,   # Ga3+
    'Ge': 0.53,   # Ge4+
    'As': 0.58,   # As3+
    'Se': 1.98,   # Se2-
    'Br': 1.96,   # Br-
    'Rb': 1.52,   # Rb+
    'Sr': 1.18,   # Sr2+
    'Y': 0.90,    # Y3+
    'Zr': 0.72,   # Zr4+
    'Nb': 0.64,   # Nb5+
    'Mo': 0.65,   # Mo6+
    'Tc': 0.64,   # Tc7+
    'Ru': 0.68,   # Ru4+
    'Rh': 0.67,   # Rh3+
    'Pd': 0.86,   # Pd2+
    'Ag': 1.15,   # Ag+
    'Cd': 0.95,   # Cd2+
    'In': 0.80,   # In3+
    'Sn': 0.69,   # Sn4+
    'Sb': 0.76,   # Sb3+
    'Te': 2.21,   # Te2-
    'I': 2.20,    # I-
    'Cs': 1.67,   # Cs+
    'Ba': 1.35,   # Ba2+
    'La': 1.03,   # La3+
    'Ce': 1.01,   # Ce3+
    'Pr': 0.99,   # Pr3+
    'Nd': 0.98,   # Nd3+
    'Pm': 0.97,   # Pm3+
    'Sm': 0.96,   # Sm3+
    'Eu': 0.95,   # Eu3+
    'Gd': 0.94,   # Gd3+
    'Tb': 0.92,   # Tb3+
    'Dy': 0.91,   # Dy3+
    'Ho': 0.90,   # Ho3+
    'Er': 0.89,   # Er3+
    'Tm': 0.88,   # Tm3+
    'Yb': 0.87,   # Yb3+
    'Lu': 0.86,   # Lu3+
    'Hf': 0.71,   # Hf4+
    'Ta': 0.64,   # Ta5+
    'W': 0.62,    # W6+
    'Re': 0.63,   # Re7+
    'Os': 0.63,   # Os6+
    'Ir': 0.68,   # Ir4+
    'Pt': 0.80,   # Pt2+
    'Au': 1.37,   # Au+
    'Hg': 1.02,   # Hg2+
    'Tl': 1.50,   # Tl+
    'Pb': 1.19,   # Pb2+
    'Bi': 1.03,   # Bi3+
    'Po': 1.94,   # Po4-
    'At': 2.27,   # At- (estimated)
    'Rn': 1.20,   # Rn (estimated)
    'Fr': 1.80,   # Fr+ (estimated)
    'Ra': 1.48,   # Ra2+
    'Ac': 1.12,   # Ac3+
    'Th': 0.94,   # Th4+
    'Pa': 1.04,   # Pa4+
    'U': 1.03,    # U4+
}

def get_ionic_radius(element):
    """
    Get the ionic radius for an element in Angstroms.
    
    Parameters:
    element (str): Element symbol
    
    Returns:
    float: Ionic radius in Angstroms
    """
    return IONIC_RADII.get(element, 1.0)  # Default to 1.0 Å if not found

def ionic_radius_to_sigma(ionic_radius):
    """
    Convert ionic radius to gaussian sigma where the gaussian base width equals 2 × ionic_radius.
    
    The gaussian effectively reaches zero at ±3σ from the center.
    To make the base width (6σ total) equal to 2 × ionic_radius:
    - Base width = 6σ = 2 × ionic_radius
    - Therefore: σ = ionic_radius / 3
    
    This ensures the gaussian "footprint" spans exactly the ionic diameter.
    
    Parameters:
    ionic_radius (float): Ionic radius in Angstroms
    
    Returns:
    float: Gaussian sigma value (ionic_radius / 3)
    """
    sigma = ionic_radius / 3  # Base width = 2 × ionic_radius
    return sigma

class q2D_analyzer:
    """
    A class for analyzing 2D quantum materials using VASP results.
    """
    
    def __init__(self, file_path=None):
        """
        Initialize the q2D_analyzer.
        
        Parameters:
        file_path (str, optional): Path to the VASP file to load initially
        """
        self.data = None
        self.box = None
        self.file_path = None
        
        if file_path:
            self.load_vasp(file_path)
    
    def load_vasp(self, file_path):
        """
        Load a VASP file using the utilities from file_handlers.py
        
        Parameters:
        file_path (str): Path to the VASP file
        
        Returns:
        bool: True if successful, False otherwise
        """
        try:
            result = vasp_load(file_path)
            if result is not None:
                self.data, self.box = result
                self.file_path = file_path
                print(f"Successfully loaded VASP file: {file_path}")
                return True
            else:
                print(f"Failed to load VASP file: {file_path}")
                return False
        except Exception as e:
            print(f"Error loading VASP file {file_path}: {e}")
            return False
    
    def get_data(self):
        """
        Get the loaded atomic data as a pandas DataFrame.
        
        Returns:
        pd.DataFrame: DataFrame containing atomic coordinates and elements
        """
        return self.data
    
    def get_box(self):
        """
        Get the box information from the VASP file.
        
        Returns:
        list: Box information containing lattice vectors and elements
        """
        return self.box
    
    def get_elements(self):
        """
        Get unique elements in the loaded structure.
        
        Returns:
        list: List of unique element symbols
        """
        if self.data is not None:
            return self.data['Element'].unique().tolist()
        return []
    
    def get_atom_count(self):
        """
        Get the total number of atoms in the structure.
        
        Returns:
        int: Total number of atoms
        """
        if self.data is not None:
            return len(self.data)
        return 0
    
    def get_element_counts(self):
        """
        Get the count of each element in the structure.
        
        Returns:
        dict: Dictionary with element symbols as keys and counts as values
        """
        if self.data is not None:
            return self.data['Element'].value_counts().to_dict()
        return {}
    
    def summary(self):
        """
        Print a summary of the loaded structure.
        """
        if self.data is None:
            print("No data loaded. Please load a VASP file first.")
            return
        
        print(f"File: {self.file_path}")
        print(f"Total atoms: {self.get_atom_count()}")
        print(f"Elements: {', '.join(self.get_elements())}")
        print("Element counts:")
        for element, count in self.get_element_counts().items():
            print(f"  {element}: {count}")
        
        if self.box:
            print(f"Lattice vectors: {self.box[0]}")
    
    def _process_gaussian_data(self):
        """
        Process gaussian projection data for all elements.
        
        Returns:
        dict: Dictionary containing processed data for plotting
        """
        if self.data is None:
            return None
        
        # Get lattice vectors from box information for reference
        if self.box is None:
            c_vector_length = 0.0
        else:
            # Extract c-axis lattice vector (third vector) - for display purposes
            c_vector = np.array(self.box[0][2], dtype=float)
            c_vector_length = np.linalg.norm(c_vector)
        
        # Get unique elements
        elements = self.get_elements()
        
        # Define gaussian kernel function (MBTR-style with ionic radius-based sigma)
        def gaussian_kernel(x, center, sigma):
            """Gaussian kernel centered at 'center' with standard deviation 'sigma'"""
            return np.exp(-0.5 * ((x - center) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
        
        # Get overall z-range for consistent plotting
        all_z_coords = self.data['Z'].values
        z_min_global = all_z_coords.min()
        z_max_global = all_z_coords.max()
        z_range = np.linspace(z_min_global - 1, z_max_global + 1, 1000)
        
        # Process data for each element
        processed_data = {}
        element_sigmas = {}
        
        for element in elements:
            # Extract z-coordinates for this element (already in Cartesian coordinates/Angstroms)
            element_data = self.data[self.data['Element'] == element]
            z_coords = element_data['Z'].values
            
            if len(z_coords) == 0:
                continue
            
            # Calculate element-specific sigma based on ionic radius
            ionic_radius = get_ionic_radius(element)
            sigma = ionic_radius_to_sigma(ionic_radius)
            element_sigmas[element] = sigma
            
            # Create gaussian kernel density by summing gaussians at each atom position
            # MBTR-style: each atom contributes equally, height proportional to atom count
            kernel_density = np.zeros_like(z_range)
            
            for z_pos in z_coords:
                kernel_density += gaussian_kernel(z_range, z_pos, sigma)
            
            # Try to fit a single gaussian to the overall distribution
            fitted_gaussian = None
            fit_params = None
            
            try:
                from scipy.optimize import curve_fit
                
                def gaussian_fit(x, amp, mean, std):
                    return amp * np.exp(-((x - mean) ** 2) / (2 * std ** 2))
                
                # Initial guess
                amp_guess = kernel_density.max()
                mean_guess = z_coords.mean()
                std_guess = z_coords.std()
                
                # Fit gaussian to the kernel density
                popt, _ = curve_fit(gaussian_fit, z_range, kernel_density, 
                                  p0=[amp_guess, mean_guess, std_guess],
                                  maxfev=10000)
                
                fitted_gaussian = gaussian_fit(z_range, *popt)
                fit_params = popt
                
            except Exception as e:
                print(f"Could not fit gaussian for {element}: {e}")
            
            # Store processed data
            processed_data[element] = {
                'z_coords': z_coords,
                'kernel_density': kernel_density,
                'fitted_gaussian': fitted_gaussian,
                'fit_params': fit_params,
                'ionic_radius': ionic_radius,
                'sigma': sigma
            }
        
        return {
            'elements_data': processed_data,
            'z_range': z_range,
            'element_sigmas': element_sigmas,
            'c_vector_length': c_vector_length,
            'z_min_global': z_min_global,
            'z_max_global': z_max_global
        }
    
    def _gaussian_proyection(self, output_dir=None):
        """
        Generate gaussian projection plots for all elements.
        
        Parameters:
        output_dir (str, optional): Custom output directory for plots
        """
        if self.data is None:
            print("No data loaded. Please load a VASP file first.")
            return
        
        # Create output directory
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tests')
        os.makedirs(output_dir, exist_ok=True)
        
        # Process gaussian data
        processed_data = self._process_gaussian_data()
        if processed_data is None:
            return
        
        # Extract data
        elements_data = processed_data['elements_data']
        z_range = processed_data['z_range']
        element_sigmas = processed_data['element_sigmas']
        c_vector_length = processed_data['c_vector_length']
        
        # Structure name for plots
        structure_name = os.path.basename(self.file_path) if self.file_path else "Unknown"
        
        # Create individual plots for each element
        for element, data in elements_data.items():
            z_coords = data['z_coords']
            kernel_density = data['kernel_density']
            fitted_gaussian = data['fitted_gaussian']
            fit_params = data['fit_params']
            ionic_radius = data['ionic_radius']
            sigma = data['sigma']
            
            # Create beautiful plot
            fig = plot_gaussian_projection(
                z_range=z_range,
                kernel_density=kernel_density,
                z_coords=z_coords,
                element=element,
                structure_name=structure_name,
                c_vector_length=c_vector_length,
                sigma=sigma,
                fitted_gaussian=fitted_gaussian,
                fit_params=fit_params,
                ionic_radius=ionic_radius
            )
            
            # Save plot
            filename = f'{element}_gaussian_projection.png'
            filepath = os.path.join(output_dir, filename)
            save_beautiful_plot(fig, filepath)
            
            print(f"Saved beautiful plot for {element}: {filepath}")
            print(f"  - Number of atoms: {len(z_coords)}")
            print(f"  - Ionic radius: {ionic_radius:.3f} Å")
            print(f"  - Kernel sigma: {sigma:.3f} Å (= ionic_radius / 3)")
            print(f"  - Base width: {6 * sigma:.3f} Å (= 2 × ionic_radius)")
            print(f"  - Z-range: {z_coords.min():.3f} to {z_coords.max():.3f} Å")
            print(f"  - Peak density: {kernel_density.max():.3f}")
            if fit_params is not None:
                print(f"  - Fitted gaussian: μ={fit_params[1]:.3f} Å, σ={fit_params[2]:.3f} Å")
        
        # Create multi-element comparison plot
        if len(elements_data) > 1:
            fig = plot_multi_element_comparison(
                element_data_dict=elements_data,
                structure_name=structure_name,
                c_vector_length=c_vector_length,
                z_range=z_range,
                sigma=None  # Will use individual sigmas from element data
            )
            
            # Save multi-element plot
            filename = 'multi_element_comparison.png'
            filepath = os.path.join(output_dir, filename)
            save_beautiful_plot(fig, filepath)
            print(f"Saved multi-element comparison plot: {filepath}")
        
        print(f"Gaussian projection analysis completed with beautiful plots!")
        print(f"All plots saved to: {output_dir}")
        print(f"Cell c-axis length: {c_vector_length:.3f} Å")
        print(f"Global Z-range: {processed_data['z_min_global']:.3f} to {processed_data['z_max_global']:.3f} Å")
        
        # Print summary of ionic radii used
        print("\nIonic radii and sigma values used:")
        for element, sigma in element_sigmas.items():
            ionic_radius = get_ionic_radius(element)
            base_width = 6 * sigma
            print(f"  {element}: r_ionic = {ionic_radius:.3f} Å, σ = {sigma:.3f} Å, base_width = {base_width:.3f} Å")
    
    def gaussian_projection(self, output_dir=None):
        """
        Public method to perform gaussian projection analysis on loaded atomic data.
        
        This method creates gaussian kernel approximations of atom frequencies along the z-axis,
        grouped by element, and saves beautiful plots to the specified output directory.
        
        Parameters:
        output_dir (str, optional): Custom output directory for plots. If None, uses tests/ directory.
        
        The plots show:
        - Gaussian kernel density with elegant styling
        - Individual atom positions
        - Fitted gaussian function (when possible)
        - Multi-element comparison plot
        
        All plots follow beautiful scientific design principles inspired by Federica Fragapane.
        """
        self._gaussian_proyection(output_dir)
    
    def batch_gaussian_projection(self, structures_dir=None):
        """
        Batch process multiple VASP files and create beautiful gaussian projections for each.
        
        Parameters:
        structures_dir (str, optional): Directory containing VASP files. 
                                      If None, uses tests/structures/ directory.
        
        Creates a subfolder for each structure file and saves beautiful plots there.
        Also creates a summary comparison plot.
        """
        if structures_dir is None:
            structures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tests', 'structures')
        
        if not os.path.exists(structures_dir):
            print(f"Structures directory not found: {structures_dir}")
            return
        
        # Get all VASP files
        vasp_files = [f for f in os.listdir(structures_dir) if f.endswith('.vasp')]
        
        if not vasp_files:
            print(f"No VASP files found in {structures_dir}")
            return
        
        print(f"Found {len(vasp_files)} VASP files to process:")
        for vf in vasp_files:
            print(f"  - {vf}")
        
        # Base output directory
        base_output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'tests')
        
        # Summary data for comparison plot
        structures_summary = {}
        
        # Process each file
        for vasp_file in vasp_files:
            file_path = os.path.join(structures_dir, vasp_file)
            
            # Create output directory for this structure
            structure_name = os.path.splitext(vasp_file)[0]  # Remove .vasp extension
            output_dir = os.path.join(base_output_dir, structure_name)
            
            print(f"\n{'='*60}")
            print(f"Processing: {vasp_file}")
            print(f"Output directory: {output_dir}")
            print(f"{'='*60}")
            
            # Load the structure
            success = self.load_vasp(file_path)
            
            if success:
                # Show summary
                print("\nStructure Summary:")
                self.summary()
                
                # Perform gaussian projection analysis
                print("\nPerforming Beautiful Gaussian Projection Analysis...")
                self.gaussian_projection(output_dir)
                
                # Collect summary data
                processed_data = self._process_gaussian_data()
                if processed_data:
                    structures_summary[structure_name] = {
                        'total_atoms': self.get_atom_count(),
                        'elements': self.get_element_counts(),
                        'c_axis_length': processed_data['c_vector_length'],
                        'z_range': processed_data['z_max_global'] - processed_data['z_min_global']
                    }
                
                print(f"\nCompleted processing: {vasp_file}")
            else:
                print(f"Failed to load: {vasp_file}")
        
        # Create summary comparison plot
        if structures_summary:
            summary_path = os.path.join(base_output_dir, 'structures_summary.png')
            create_summary_plot(structures_summary, summary_path)
            print(f"\nCreated beautiful summary plot: {summary_path}")
        
        print(f"\n{'='*60}")
        print("Batch processing completed with beautiful visualizations!")
        print(f"All plots saved in organized subfolders within: {base_output_dir}")
        print(f"{'='*60}")