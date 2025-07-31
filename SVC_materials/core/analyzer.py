"""
Refactored q2D Analyzer - Main analysis class using modular components.

This is the main analyzer class that orchestrates the octahedral analysis
using specialized modules for geometry, angular analysis, and connectivity.
"""

import pandas as pd
import numpy as np
import os
from ase.io import read
from ase import Atoms
from ..utils.file_handlers import vasp_load, save_vasp
from ..utils.plots import plot_gaussian_projection, plot_multi_element_comparison, save_beautiful_plot, create_summary_plot
from ..utils.octadist.io import extract_octa
from ..utils.octadist.calc import CalcDistortion
from ..utils.octadist.linear import angle_btw_vectors

# Import our modular components
from .geometry import GeometryCalculator
from .angular_analysis import AngularAnalyzer
from .connectivity import ConnectivityAnalyzer
from .layers_analysis import LayersAnalyzer

# Use matplotlib without x server
import matplotlib
matplotlib.use('Agg')


class q2D_analyzer:
    """
    Refactored analyzer for 2D quantum materials using VASP results.
    
    Creates a unified ontology:
    Experiment -> Cell Properties -> Octahedra (with atomic indices and properties)
    
    Uses modular components for:
    - Geometry calculations (GeometryCalculator)
    - Angular analysis (AngularAnalyzer) 
    - Connectivity analysis (ConnectivityAnalyzer)
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
        # Basic properties
        self.file_path = file_path
        self.experiment_name = file_path.split('/')[-1].split('.')[0]
        self.atoms = read(file_path)
        self.b = b
        self.x = x
        self.chemical_symbols = self.atoms.get_chemical_symbols()
        self.coord = self.atoms.get_positions()
        self.cutoff_ref_ligand = cutoff_ref_ligand
        
        # Initialize modular components
        self.geometry_calc = GeometryCalculator(atoms=self.atoms)
        self.angular_analyzer = AngularAnalyzer(self.geometry_calc)
        self.connectivity_analyzer = ConnectivityAnalyzer(self.geometry_calc)
        self.layers_analyzer = LayersAnalyzer()

        # Initialize analysis first to get connectivity data
        self.all_octahedra = self.find_all_octahedra()
        self.ordered_octahedra = self.order_octahedra()
        
        # Create unified ontology
        self.ontology = self._create_unified_ontology()

        # Get terminal X atoms from connectivity analysis after ontology is created
        connectivity_analysis = self.ontology.get('connectivity_analysis', {})
        self.terminal_x_atoms = connectivity_analysis.get('terminal_axial_atoms', {})

        # Isolate the spacer, the idea is to eliminate all the octahedra, we just need to eliminate X, B and MA and save the structure:
        self.spacer = self.isolate_spacer()

        # Analyze molecules in the isolated spacer and create molecule ontology
        if self.spacer is not None and len(self.spacer) > 0:
            self.spacer_molecules = self.analyze_spacer_molecules()
            self.molecule_ontology = self._create_molecule_ontology()
        else:
            self.spacer_molecules = None
            self.molecule_ontology = None

    def find_all_octahedra(self):
        """
        Automatically identify all octahedra in the given ASE Atoms object.
        
        Returns:
            List of tuples: (central_atom_symbol, central_index, atom_octa, coord_octa)
        """
        all_octa = []
        central_symbols = [self.b]
        atom_symbols = self.chemical_symbols
        coord = self.coord
        
        # Get cell and PBC information from ASE atoms object
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        use_pbc = np.any(pbc) and self.atoms.get_volume() > 0
        
        # Find indices of potential central atoms
        central_indices = [i for i, sym in enumerate(atom_symbols) if sym in central_symbols]
        
        for ref_index in central_indices:
            # Extract octahedron around this center with PBC support
            if use_pbc:
                atom_octa, coord_octa = extract_octa(
                    atom_symbols, coord, 
                    ref_index=ref_index, 
                    cutoff_ref_ligand=self.cutoff_ref_ligand,
                    cell=cell, pbc=pbc
                )
            else:
                atom_octa, coord_octa = extract_octa(
                    atom_symbols, coord, 
                    ref_index=ref_index, 
                    cutoff_ref_ligand=self.cutoff_ref_ligand
                )
            
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
        Orders by Z coordinate first, then by X, then by Y coordinates.
        
        Returns:
            List of ordered octahedra with consistent indexing
        """
        if not self.all_octahedra:
            return []
        
        # Extract central atom coordinates for sorting
        octahedra_with_coords = []
        for i, (symbol, index, atom_octa, coord_octa) in enumerate(self.all_octahedra):
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

    def _get_cell_properties(self):
        """
        Extract cell properties in a standardized format.
        
        Returns:
            dict: Cell properties including lattice parameters and composition
        """
        cell_lengths_angles = self.atoms.get_cell_lengths_and_angles()
        a, b, c, alpha, beta, gamma = cell_lengths_angles
        
        return {
            "lattice_parameters": {
                "A": float(a),
                "B": float(b), 
                "C": float(c),
                "Alpha": float(alpha),
                "Beta": float(beta),
                "Gamma": float(gamma)
            },
            "composition": {
                "metal_B": str(self.b),
                "halogen_X": str(self.x),
                "number_of_atoms": len(self.atoms),
                "number_of_octahedra": len(self.ordered_octahedra)
            },
            "structure_info": {
                "cell_volume": float(self.atoms.get_volume()),
                "cutoff_ref_ligand": float(self.cutoff_ref_ligand)
            }
        }

    def _process_octahedron_data(self, octa_data):
        """
        Process individual octahedron data and calculate all properties.
        
        Parameters:
            octa_data: Dictionary containing octahedron information
            
        Returns:
            dict: Complete octahedron analysis with atomic indices and properties
        """
        coord_octa = octa_data['coord_octa']
        atom_symbols = octa_data['atom_octa']
        central_coord = np.array(octa_data['central_coord'])
        global_index = octa_data['global_index']
        
        # Calculate distortion parameters
        dist_calc = CalcDistortion(coord_octa)
        
        # Identify axial and equatorial positions using geometry calculator
        axial_positions, equatorial_positions = self.geometry_calc.identify_axial_equatorial_positions(
            coord_octa, atom_symbols, central_coord
        )
        
        # Get global indices for all ligand atoms using PBC-aware distance calculation
        # Process each ligand exactly once to avoid duplicates
        ligand_global_indices = []
        axial_global_indices = []
        equatorial_global_indices = []
        
        # Get cell and PBC information for distance calculations
        cell = self.atoms.get_cell()
        pbc = self.atoms.get_pbc()
        use_pbc = np.any(pbc) and self.atoms.get_volume() > 0
        
        # Process all ligands from coord_octa (skip index 0 which is central atom)
        for i in range(1, len(coord_octa)):
            ligand_coord = coord_octa[i]
            ligand_symbol = atom_symbols[i]
            
            # Find the global index using PBC-aware distance calculation
            global_atom_index = self._find_global_index_pbc_aware(
                ligand_coord, central_coord, ligand_symbol, cell, pbc, use_pbc
            )
            
            if global_atom_index is not None:
                ligand_global_indices.append(global_atom_index)
                
                # Check if this ligand is axial or equatorial
                # Find matching position in axial_positions
                is_axial = False
                is_equatorial = False
                
                for pos in axial_positions:
                    if pos['local_index'] == i - 1:  # -1 because local_index doesn't include central atom
                        axial_global_indices.append(global_atom_index)
                        is_axial = True
                        break
                
                if not is_axial:
                    for pos in equatorial_positions:
                        if pos['local_index'] == i - 1:  # -1 because local_index doesn't include central atom
                            equatorial_global_indices.append(global_atom_index)
                            is_equatorial = True
                            break
                
                # If not found in either, this indicates an issue with octahedron identification
                if not is_axial and not is_equatorial:
                    print(f"Warning: Ligand {i} ({ligand_symbol}) at global index {global_atom_index} "
                          f"not classified as axial or equatorial for octahedron {global_index}")
            else:
                # If we can't find the global index, this octahedron is incomplete
                print(f"Error: Could not find global index for ligand {i} ({ligand_symbol}) "
                      f"at coord {ligand_coord} for central atom {global_index}")
                print(f"This octahedron will be marked as non-octahedral.")
                # Mark as non-octahedral if we can't find all ligands
                dist_calc.non_octa = True
        
        # Verify we have exactly 6 ligands
        if len(ligand_global_indices) != 6:
            print(f"Warning: Found {len(ligand_global_indices)} ligands instead of 6 for octahedron {global_index}")
            dist_calc.non_octa = True
        
        # Check for duplicates
        if len(set(ligand_global_indices)) != len(ligand_global_indices):
            print(f"Error: Duplicate ligand indices found for octahedron {global_index}: {ligand_global_indices}")
            print(f"Unique indices: {list(set(ligand_global_indices))}")
            dist_calc.non_octa = True
        
        # Sort ligand indices for consistency
        ligand_global_indices.sort()
        axial_global_indices.sort()
        equatorial_global_indices.sort()
        
        # Create angles dictionary with atom indices
        cis_angles_with_indices = {}
        trans_angles_with_indices = {}
        
        if hasattr(dist_calc, 'cis_angle'):
            for i, angle_val in enumerate(list(dist_calc.cis_angle)):
                cis_angles_with_indices[f"cis_angle_{i+1}"] = {
                    "value": float(angle_val),
                    "atom_indices": f"See ligand_global_indices for atoms involved"
                }
        
        if hasattr(dist_calc, 'trans_angle'):
            for i, angle_val in enumerate(list(dist_calc.trans_angle)):
                trans_angles_with_indices[f"trans_angle_{i+1}"] = {
                    "value": float(angle_val),
                    "atom_indices": f"See ligand_global_indices for atoms involved"
                }
        
        # Create bond distances with atom indices
        bond_distances_with_indices = {}
        bond_distance_list = dist_calc.bond_dist.tolist()
        for i, distance in enumerate(bond_distance_list):
            ligand_idx = ligand_global_indices[i] if i < len(ligand_global_indices) else None
            bond_distances_with_indices[f"bond_{i+1}"] = {
                "distance": float(distance),
                "central_atom_index": int(global_index),
                "ligand_atom_index": int(ligand_idx) if ligand_idx is not None else None
            }
        
        # Create eight_theta data with indices
        eight_theta_with_indices = {}
        if hasattr(dist_calc, 'eight_theta'):
            for i, theta_val in enumerate(list(dist_calc.eight_theta)):
                eight_theta_with_indices[f"theta_face_{i+1}"] = {
                    "value": float(theta_val),
                    "description": f"Theta angle for octahedral face {i+1}"
                }
        
        # Calculate comprehensive angular analysis using angular analyzer
        angular_analysis = self.angular_analyzer.calculate_angular_analysis(
            coord_octa, axial_positions, equatorial_positions, central_coord, 
            octa_data['ordered_index'], self.ordered_octahedra, 
            self.coord, self.chemical_symbols, self.cutoff_ref_ligand
        )
        
        return {
            "central_atom": {
                "global_index": int(global_index),
                "symbol": str(octa_data['symbol']),
                "coordinates": {
                    "x": float(central_coord[0]),
                    "y": float(central_coord[1]),
                    "z": float(central_coord[2])
                }
            },
            "ligand_atoms": {
                "axial_global_indices": [int(idx) for idx in axial_global_indices],
                "equatorial_global_indices": [int(idx) for idx in equatorial_global_indices],
                "all_ligand_global_indices": [int(idx) for idx in sorted(ligand_global_indices)]
            },
            "distortion_parameters": {
                "zeta": float(dist_calc.zeta),
                "delta": float(dist_calc.delta),
                "sigma": float(dist_calc.sigma),
                "theta_mean": float(dist_calc.theta),
                "theta_min": float(dist_calc.theta_min),
                "theta_max": float(dist_calc.theta_max),
                "eight_theta": eight_theta_with_indices
            },
            "bond_distances": bond_distances_with_indices,
            "bond_distance_analysis": {
                "mean_bond_distance": float(dist_calc.d_mean),
                "bond_distance_variance": float(np.var(dist_calc.bond_dist))
            },
            "bond_angles": {
                "cis_angles": cis_angles_with_indices,
                "trans_angles": trans_angles_with_indices
            },
            "geometric_properties": {
                "octahedral_volume": float(dist_calc.oct_vol),
                "is_octahedral": bool(not dist_calc.non_octa)
            },
            "detailed_atom_info": {
                "all_ligand_symbols": [atom_symbols[i] for i in range(1, len(atom_symbols))],
                "axial_atom_symbols": [pos['symbol'] for pos in axial_positions],
                "equatorial_atom_symbols": [pos['symbol'] for pos in equatorial_positions],
                "ligand_coordinates": [coord_octa[i].tolist() for i in range(1, len(coord_octa))]
            },
            "angular_analysis": angular_analysis
        }

    def _find_global_index_pbc_aware(self, ligand_coord, central_coord, ligand_symbol, cell, pbc, use_pbc):
        """
        Find global index of ligand atom using PBC-aware distance calculation.
        
        Parameters:
        ligand_coord: coordinates of ligand atom from octahedron extraction
        central_coord: coordinates of central atom
        ligand_symbol: expected symbol of ligand atom
        cell: unit cell matrix (not used - kept for compatibility)
        pbc: periodic boundary conditions (not used - kept for compatibility)
        use_pbc: whether to use PBC calculations (not used - kept for compatibility)
        
        Returns:
        global index of ligand atom, or None if not found
        """
        ligand_coord = np.array(ligand_coord)
        central_coord = np.array(central_coord)
        tolerance = 1e-3
        
        best_match_index = None
        min_distance = float('inf')
        
        for i, coord in enumerate(self.coord):
            # Skip if atom symbol doesn't match
            if self.chemical_symbols[i] != ligand_symbol:
                continue
            
            coord = np.array(coord)
            
            # Use geometry calculator's PBC-aware distance calculation
            distance_to_ligand = self.geometry_calc.calculate_distance(coord, ligand_coord)
            dist_to_central = self.geometry_calc.calculate_distance(coord, central_coord)
            
            if dist_to_central <= self.cutoff_ref_ligand and distance_to_ligand < min_distance:
                if distance_to_ligand < tolerance:
                    min_distance = distance_to_ligand
                    best_match_index = i
        
        return best_match_index

    def _create_unified_ontology(self):
        """
        Create the unified ontology structure:
        Experiment -> Cell Properties -> Octahedra (with atomic indices and properties)
        
        Returns:
            dict: Complete unified ontology
        """
        # Get cell properties
        cell_properties = self._get_cell_properties()
        
        # Process all octahedra
        octahedra_data = {}
        for octa_data in self.ordered_octahedra:
            octahedron_index = octa_data['ordered_index']
            octahedra_data[f"octahedron_{octahedron_index + 1}"] = self._process_octahedron_data(octa_data)
        
        # Analyze connectivity between octahedra using connectivity analyzer
        connectivity_analysis = self.connectivity_analyzer.analyze_octahedra_connectivity(
            octahedra_data, self.coord, self.chemical_symbols
        )
        
        # Analyze layers using layers analyzer with connectivity information
        layers_analysis = self.layers_analyzer.identify_layers(octahedra_data, connectivity_analysis)
        
        
        # Create unified structure
        unified_ontology = {
            "experiment": {
                "name": str(self.experiment_name),
                "file_path": str(self.file_path),
                "timestamp": str(pd.Timestamp.now())
            },
            "cell_properties": cell_properties,
            "octahedra": octahedra_data,
            "connectivity_analysis": connectivity_analysis,
            "layers_analysis": layers_analysis,
        }
        
        return unified_ontology

    def get_ontology(self):
        """
        Get the unified ontology structure.
        
        Returns:
            dict: Complete unified ontology
        """
        return self.ontology

    def _make_json_serializable(self, obj):
        """
        Recursively convert any sets to lists and NumPy types to native Python types
        to make the ontology JSON serializable.
        
        Parameters:
            obj: Object to convert
            
        Returns:
            JSON-serializable object
        """
        if isinstance(obj, set):
            return list(obj)
        elif isinstance(obj, dict):
            return {key: self._make_json_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._make_json_serializable(item) for item in obj)
        elif hasattr(obj, 'dtype'):  # NumPy arrays and scalars
            if obj.ndim == 0:  # NumPy scalar
                return obj.item()  # Convert to native Python type
            else:  # NumPy array
                return obj.tolist()  # Convert to Python list
        elif isinstance(obj, (np.integer, np.floating, np.complexfloating)):
            return obj.item()  # Convert NumPy scalars to native Python types
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert NumPy arrays to Python lists
        else:
            return obj

    def export_ontology_json(self, filename=None):
        """
        Export the unified ontology to JSON format.
        
        Parameters:
            filename (str, optional): Output filename
        """
        if filename is None:
            filename = f"{self.experiment_name}_unified_ontology.json"
        
        import json
        
        # Ensure all data is JSON serializable
        serializable_ontology = self._make_json_serializable(self.ontology)
        
        with open(filename, 'w') as f:
            json.dump(serializable_ontology, f, indent=2)
        
        print(f"Unified ontology exported to {filename}")
        return serializable_ontology

            
    def get_layers_analysis(self):
        """
        Get the layers analysis results.
        
        Returns:
            dict: Layers analysis data
        """
        return self.ontology.get('layers_analysis', {})
    
    def get_layer_summary(self):
        """
        Get a formatted summary of the layer analysis.
        
        Returns:
            str: Formatted layer summary
        """
        return self.layers_analyzer.get_layer_summary()
    
    def get_octahedra_in_layer(self, layer_key):
        """
        Get all octahedra in a specific layer.
        
        Parameters:
            layer_key (str): Key of the layer (e.g., 'layer_1')
            
        Returns:
            list: List of octahedron keys in the layer
        """
        return self.layers_analyzer.get_octahedra_in_layer(layer_key)
    
    def get_layer_by_octahedron(self, octahedron_key):
        """
        Find which layer contains a specific octahedron.
        
        Parameters:
            octahedron_key (str): Key of the octahedron (e.g., 'octahedron_1')
            
        Returns:
            str or None: Layer key containing the octahedron
        """
        return self.layers_analyzer.get_layer_by_octahedron(octahedron_key)
    
    def export_layer_analysis(self, filename=None):
        """
        Export layer analysis to JSON format.
        
        Parameters:
            filename (str, optional): Output filename
            
        Returns:
            dict: Layer analysis data
        """
        return self.layers_analyzer.export_layer_analysis(filename)
    
    def set_layer_window(self):
        """
        Update layer analysis Z-window parameter and re-run the analysis.
        
        Parameters:
            z_window (float): Z-coordinate window in Angstroms for same layer
        """
        self.layers_analyzer = LayersAnalyzer()
        
        # Re-run layer analysis with new parameters
        octahedra_data = self.ontology.get('octahedra', {})
        if octahedra_data:
            layers_analysis = self.layers_analyzer.identify_layers(octahedra_data)
            self.ontology['layers_analysis'] = layers_analysis
            
        print(f"Layer analysis updated with z_window={z_window} Å")
        return self.layers_analyzer.get_layer_summary()

    def isolate_spacer(self):
        """
        Isolate the spacer by eliminating all octahedral elements (B and X atoms).
        This creates a new ASE Atoms object containing only the spacer molecules.
        
        Returns:
            ase.Atoms: New atoms object containing only spacer atoms
        """
        # Get all indices that belong to octahedra (central B atoms and ligand X atoms)
        octahedral_indices = set()
        
        # Add all central B atom indices
        for i, symbol in enumerate(self.chemical_symbols):
            if symbol == self.b:
                octahedral_indices.add(i)
        
        # Add all ligand X atom indices that are within cutoff of any B atom
        b_indices = [i for i, symbol in enumerate(self.chemical_symbols) if symbol == self.b]
        
        for i, symbol in enumerate(self.chemical_symbols):
            if symbol == self.x:
                # Check if this X atom is within cutoff of any B atom
                x_coord = self.coord[i]
                for b_idx in b_indices:
                    b_coord = self.coord[b_idx]
                    # Use geometry calculator for PBC-aware distance calculation
                    distance = self.geometry_calc.calculate_distance(x_coord, b_coord)
                    if distance <= self.cutoff_ref_ligand:
                        octahedral_indices.add(i)
                        break
        
        # Create list of indices to keep (all atoms except octahedral ones)
        spacer_indices = []
        spacer_symbols = []
        spacer_positions = []
        
        for i in range(len(self.chemical_symbols)):
            if i not in octahedral_indices:
                spacer_indices.append(i)
                spacer_symbols.append(self.chemical_symbols[i])
                spacer_positions.append(self.coord[i])
        
        if not spacer_indices:
            print("Warning: No spacer atoms found after eliminating octahedral elements")
            # Return empty atoms object with same cell
            spacer_atoms = Atoms(cell=self.atoms.get_cell(), pbc=self.atoms.get_pbc())
            return spacer_atoms
        
        # Create new ASE Atoms object with only spacer atoms
        spacer_atoms = Atoms(
            symbols=spacer_symbols,
            positions=spacer_positions,
            cell=self.atoms.get_cell(),
            pbc=self.atoms.get_pbc()
        )
        
        print(f"Spacer isolation complete:")
        print(f"  - Original structure: {len(self.chemical_symbols)} atoms")
        print(f"  - Eliminated octahedral atoms: {len(octahedral_indices)} atoms")
        print(f"    - B ({self.b}) atoms: {sum(1 for s in self.chemical_symbols if s == self.b)}")
        print(f"    - X ({self.x}) atoms in octahedra: {len([i for i in octahedral_indices if self.chemical_symbols[i] == self.x])}")
        print(f"  - Spacer atoms remaining: {len(spacer_indices)} atoms")
        
        # Print composition of spacer
        spacer_composition = {}
        for symbol in spacer_symbols:
            spacer_composition[symbol] = spacer_composition.get(symbol, 0) + 1
        
        if spacer_composition:
            print(f"  - Spacer composition: {spacer_composition}")
        
        return spacer_atoms
    
    def save_spacer_structure(self, output_path):
        """
        Save the isolated spacer structure to a file.
        
        Parameters:
            output_path (str): Path where to save the spacer structure
        """
        if hasattr(self, 'spacer') and self.spacer is not None:
            self.spacer.write(output_path)
            print(f"Spacer structure saved to: {output_path}")
        else:
            print("Error: No spacer structure available. Run isolate_spacer() first.")
    
    def analyze_spacer_molecules(self):
        """
        Analyze and identify individual molecules in the isolated spacer.
        
        Returns:
            dict: Molecule analysis results
        """
        if not hasattr(self, 'spacer') or self.spacer is None:
            print("Error: No spacer structure available. Run isolate_spacer() first.")
            return None
        
        # Use the connectivity analyzer to identify molecules
        molecule_analysis = self.connectivity_analyzer.analyze_spacer_molecules(self.spacer)
        
        # Store results for later use
        self.spacer_molecules = molecule_analysis
        
        return molecule_analysis

    def _create_molecule_ontology(self):
        """
        Create a comprehensive molecule ontology from the spacer_molecules analysis.
        
        Returns:
            dict: Complete molecule ontology
        """
        if self.spacer_molecules is None:
            return {}

        molecule_ontology = {}
        for mol_id, mol_data in self.spacer_molecules.get('molecules', {}).items():
            molecule_ontology[f"molecule_{mol_id}"] = {
                "molecule_id": mol_id,
                "formula": mol_data['formula'],
                "number_of_atoms": len(mol_data['atom_indices']),
                "atom_indices": mol_data['atom_indices'],
                "symbols": mol_data['symbols'],
                "coordinates": mol_data['coordinates']
            }
        return molecule_ontology

    def get_molecule_ontology(self):
        """
        Get the comprehensive molecule ontology.
        
        Returns:
            dict: Complete molecule ontology
        """
        return self.molecule_ontology

    def get_spacer_molecule_summary(self):
        """
        Get a human-readable summary of spacer molecules.
        
        Returns:
            str: Formatted summary of molecule analysis
        """
        if not hasattr(self, 'spacer_molecules'):
            print("Error: No spacer molecule analysis available. Run analyze_spacer_molecules() first.")
            return None
        
        return self.connectivity_analyzer.molecule_identifier.get_molecule_summary(self.spacer_molecules)
    
    def save_spacer_molecules_info(self, output_path):
        """
        Save spacer molecule analysis to a JSON file.
        
        Parameters:
            output_path (str): Path where to save the molecule analysis
        """
        if not hasattr(self, 'spacer_molecules'):
            print("Error: No spacer molecule analysis available. Run analyze_spacer_molecules() first.")
            return
        
        import json
        
        with open(output_path, 'w') as f:
            json.dump(self.spacer_molecules, f, indent=2)
        
        print(f"Spacer molecule analysis saved to: {output_path}")
    
    def save_individual_molecules(self, output_dir):
        """
        Save each identified molecule as a separate XYZ structure file.
        
        Parameters:
            output_dir (str): Directory where to save individual molecule files
        """
        if not hasattr(self, 'spacer_molecules') or not hasattr(self, 'spacer'):
            print("Error: No spacer molecule analysis available. Run analyze_spacer_molecules() first.")
            return
        
        import os
        from ase import Atoms
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        molecules = self.spacer_molecules.get('molecules', {})
        
        for mol_id, mol_data in molecules.items():
            # Extract atom information for this molecule
            atom_indices = mol_data['atom_indices']
            mol_symbols = mol_data['symbols']
            mol_coordinates = mol_data['coordinates']
            formula = mol_data['formula']
            
            # Create a new ASE Atoms object for this molecule (no PBC for individual molecules)
            mol_atoms = Atoms(
                symbols=mol_symbols,
                positions=mol_coordinates,
                cell=[20.0, 20.0, 20.0],  # Large vacuum cell
                pbc=False  # No periodic boundary conditions for isolated molecules
            )
            
            # Generate filename with XYZ extension
            filename = f"molecule_{mol_id}_{formula}.xyz"
            filepath = os.path.join(output_dir, filename)
            
            # Save the molecule as XYZ format
            mol_atoms.write(filepath, format='xyz')
            
            print(f"Molecule {mol_id} ({formula}) saved to: {filepath}")
        
        print(f"All {len(molecules)} molecules saved as XYZ files to directory: {output_dir}")

    def isolate_individual_molecules(self):
        """
        Isolate each molecule as a separate ASE Atoms object.
        
        Returns:
            dict: Dictionary of isolated molecules with their ASE Atoms objects
        """
        if not hasattr(self, 'spacer_molecules') or not hasattr(self, 'spacer'):
            print("Error: No spacer molecule analysis available. Run analyze_spacer_molecules() first.")
            return {}
        
        isolated_molecules = {}
        molecules = self.spacer_molecules.get('molecules', {})
        
        print(f"Isolating {len(molecules)} molecules from spacer structure...")
        
        for mol_id, mol_data in molecules.items():
            # Extract atom information for this molecule
            mol_symbols = mol_data['symbols']
            mol_coordinates = mol_data['coordinates']
            formula = mol_data['formula']
            
            # Create a new ASE Atoms object for this molecule
            # Note: We don't use PBC for individual isolated molecules
            mol_atoms = Atoms(
                symbols=mol_symbols,
                positions=mol_coordinates,
                cell=[20.0, 20.0, 20.0],  # Large vacuum cell
                pbc=False  # No periodic boundary conditions for isolated molecules
            )
            
            isolated_molecules[mol_id] = {
                'atoms': mol_atoms,
                'formula': formula,
                'size': len(mol_symbols),
                'original_indices': mol_data['atom_indices']
            }
            
            print(f"  - Molecule {mol_id}: {formula} ({len(mol_symbols)} atoms)")
        
        self.isolated_molecules = isolated_molecules
        return isolated_molecules
    
    def export_molecule_ontology_json(self, filename=None):
        """
        Export the molecule ontology to JSON format.
        
        Parameters:
            filename (str, optional): Output filename
        """
        if filename is None:
            filename = f"{self.experiment_name}_molecule_ontology.json"
        
        import json
        
        if self.molecule_ontology is None:
            print("Error: No molecule ontology available.")
            return
        
        # Create comprehensive molecule ontology with additional metadata
        comprehensive_ontology = {
            "experiment": {
                "name": str(self.experiment_name),
                "file_path": str(self.file_path),
                "timestamp": str(pd.Timestamp.now())
            },
            "spacer_info": {
                "total_spacer_atoms": len(self.spacer) if self.spacer is not None else 0,
                "spacer_composition": self._get_spacer_composition(),
                "cell_parameters": {
                    "a": float(self.spacer.get_cell()[0][0]) if self.spacer is not None else 0,
                    "b": float(self.spacer.get_cell()[1][1]) if self.spacer is not None else 0,
                    "c": float(self.spacer.get_cell()[2][2]) if self.spacer is not None else 0,
                }
            },
            "molecule_analysis": {
                "total_molecules": len(self.molecule_ontology),
                "unique_formulas": list(set([mol['formula'] for mol in self.molecule_ontology.values()])),
                "molecule_distribution": self._get_molecule_distribution()
            },
            "molecules": self.molecule_ontology
        }
        
        # Ensure all data is JSON serializable
        serializable_ontology = self._make_json_serializable(comprehensive_ontology)
        
        with open(filename, 'w') as f:
            json.dump(serializable_ontology, f, indent=2)
        
        print(f"Molecule ontology exported to {filename}")
        return serializable_ontology
    
    def _get_spacer_composition(self):
        """
        Get the composition of the spacer structure.
        
        Returns:
            dict: Element counts in spacer
        """
        if self.spacer is None:
            return {}
        
        composition = {}
        for symbol in self.spacer.get_chemical_symbols():
            composition[symbol] = composition.get(symbol, 0) + 1
        
        return composition
    
    def _get_molecule_distribution(self):
        """
        Get the distribution of molecule types.
        
        Returns:
            dict: Formula distribution counts
        """
        if self.molecule_ontology is None:
            return {}
        
        distribution = {}
        for mol_data in self.molecule_ontology.values():
            formula = mol_data['formula']
            distribution[formula] = distribution.get(formula, 0) + 1
        
        return distribution
    
    def create_molecule_analysis_report(self, output_dir):
        """
        Create a comprehensive analysis report for all molecules.
        
        Parameters:
            output_dir (str): Directory where to save the analysis report
        """
        import os
        
        if not hasattr(self, 'spacer_molecules'):
            print("Error: No spacer molecule analysis available.")
            return
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Export molecule ontology
        ontology_file = os.path.join(output_dir, f"{self.experiment_name}_molecule_ontology.json")
        self.export_molecule_ontology_json(ontology_file)
        
        # 2. Save spacer structure (VASP format)
        spacer_file = os.path.join(output_dir, f"{self.experiment_name}_spacer.vasp")
        self.save_spacer_structure(spacer_file)
        
        # 3. Save salt structure (spacer + 4 terminal halogens, VASP format)
        salt_file = os.path.join(output_dir, f"{self.experiment_name}_salt.vasp")
        self.save_salt_structure(salt_file)
        
        # 4. Save separated structures (spacer and A-sites, VASP format)
        separated_dir = os.path.join(output_dir, "separated_structures")
        spacer_file, a_sites_file = self.save_separated_structures(separated_dir)
        
        # 5. Isolate and save individual molecules (XYZ format)
        molecule_dir = os.path.join(output_dir, "individual_molecules")
        self.save_individual_molecules(molecule_dir)
        
        # 6. Create analysis summary
        summary_file = os.path.join(output_dir, f"{self.experiment_name}_molecule_summary.txt")
        self._create_molecule_summary_report(summary_file)
        
        print(f"Complete molecule analysis report created in: {output_dir}")
    
    def _create_molecule_summary_report(self, output_file):
        """
        Create a human-readable summary report of the molecule analysis.
        
        Parameters:
            output_file (str): Path to output summary file
        """
        if self.spacer_molecules is None:
            return
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"MOLECULE ANALYSIS SUMMARY - {self.experiment_name}\n")
            f.write("=" * 80 + "\n\n")
            
            # Basic information
            f.write(f"Experiment: {self.experiment_name}\n")
            f.write(f"File: {self.file_path}\n")
            f.write(f"Analysis timestamp: {pd.Timestamp.now()}\n\n")
            
            # Spacer information
            f.write("SPACER STRUCTURE ANALYSIS:\n")
            f.write("-" * 40 + "\n")
            if self.spacer is not None:
                f.write(f"Total spacer atoms: {len(self.spacer)}\n")
                composition = self._get_spacer_composition()
                f.write("Spacer composition:\n")
                for element, count in sorted(composition.items()):
                    f.write(f"  {element}: {count} atoms\n")
            else:
                f.write("No spacer structure available\n")
            f.write("\n")
            
            # Molecule analysis
            f.write("MOLECULE IDENTIFICATION:\n")
            f.write("-" * 40 + "\n")
            molecules = self.spacer_molecules.get('molecules', {})
            f.write(f"Total molecules identified: {len(molecules)}\n")
            
            # Molecule distribution
            distribution = self._get_molecule_distribution()
            f.write("Molecule distribution:\n")
            for formula, count in sorted(distribution.items()):
                f.write(f"  {formula}: {count} molecules\n")
            f.write("\n")
            
            # Individual molecule details
            f.write("INDIVIDUAL MOLECULES:\n")
            f.write("-" * 40 + "\n")
            for mol_id, mol_data in molecules.items():
                f.write(f"Molecule {mol_id}:\n")
                f.write(f"  Formula: {mol_data['formula']}\n")
                f.write(f"  Atoms: {len(mol_data['symbols'])}\n")
                f.write(f"  Atom types: {', '.join(sorted(set(mol_data['symbols'])))}\n")
                f.write(f"  Original indices: {mol_data['atom_indices']}\n")
                f.write("\n")
            
            # Spacer and salt structure information
            f.write("SPACER AND SALT STRUCTURES:\n")
            f.write("-" * 40 + "\n")
            spacer_molecules, a_site_molecules = self.separate_molecules_by_size()
            
            f.write("Spacer molecules (2 largest molecules):\n")
            total_spacer_atoms = 0
            for mol_id, mol_data in spacer_molecules.items():
                f.write(f"  Molecule {mol_id}: {mol_data['formula']} ({len(mol_data['symbols'])} atoms)\n")
                total_spacer_atoms += len(mol_data['symbols'])
            
            f.write(f"\nSalt structure composition:\n")
            f.write(f"  - Spacer molecules: {len(spacer_molecules)} molecules, {total_spacer_atoms} atoms\n")
            
            # Count actual terminal halogens from connectivity analysis
            terminal_count = 0
            if self.terminal_x_atoms:
                for oct_key, oct_terminal_data in self.terminal_x_atoms.items():
                    terminal_atoms = oct_terminal_data.get('terminal_axial_atoms', [])
                    terminal_count += len(terminal_atoms)
            
            f.write(f"  - Terminal halogens ({self.x}): {terminal_count} atoms\n")
            f.write(f"  - Total salt atoms: {total_spacer_atoms + terminal_count} atoms\n")
            
            if a_site_molecules:
                f.write(f"\nA-site molecules (smaller molecules):\n")
                for mol_id, mol_data in a_site_molecules.items():
                    f.write(f"  Molecule {mol_id}: {mol_data['formula']} ({len(mol_data['symbols'])} atoms)\n")
            f.write("\n")
        
        print(f"Molecule summary report saved to: {output_file}")

    def separate_molecules_by_size(self):
        """
        Separate molecules into largest (spacer) and smaller (A-sites) groups.
        
        Returns:
            tuple: (spacer_molecules, a_site_molecules) dictionaries
        """
        if not hasattr(self, 'spacer_molecules') or not self.spacer_molecules:
            print("Error: No spacer molecule analysis available. Run analyze_spacer_molecules() first.")
            return {}, {}
        
        molecules = self.spacer_molecules.get('molecules', {})
        
        if len(molecules) == 0:
            return {}, {}
        
        # Sort molecules by size (number of atoms)
        sorted_molecules = sorted(
            molecules.items(), 
            key=lambda x: len(x[1]['symbols']), 
            reverse=True
        )
        
        # Take the 2 largest molecules as spacer
        spacer_molecules = {}
        a_site_molecules = {}
        
        for i, (mol_id, mol_data) in enumerate(sorted_molecules):
            if i < 2:  # First 2 largest molecules
                spacer_molecules[mol_id] = mol_data
            else:  # Remaining smaller molecules
                a_site_molecules[mol_id] = mol_data
        
        print(f"Molecule separation by size:")
        print(f"  - Spacer molecules (2 largest): {len(spacer_molecules)} molecules")
        for mol_id, mol_data in spacer_molecules.items():
            print(f"    • Molecule {mol_id}: {mol_data['formula']} ({len(mol_data['symbols'])} atoms)")
        
        print(f"  - A-site molecules (remaining): {len(a_site_molecules)} molecules")
        for mol_id, mol_data in a_site_molecules.items():
            print(f"    • Molecule {mol_id}: {mol_data['formula']} ({len(mol_data['symbols'])} atoms)")
        
        return spacer_molecules, a_site_molecules
    
    def create_spacer_structure(self, spacer_molecules):
        """
        Create a VASP structure containing only the 2 largest molecules (spacer).
        
        Parameters:
            spacer_molecules (dict): Dictionary of spacer molecule data
            
        Returns:
            ase.Atoms: ASE Atoms object with spacer molecules
        """
        if not spacer_molecules:
            print("No spacer molecules to create structure from")
            return None
        
        all_symbols = []
        all_positions = []
        
        for mol_id, mol_data in spacer_molecules.items():
            all_symbols.extend(mol_data['symbols'])
            all_positions.extend(mol_data['coordinates'])
        
        if not all_symbols:
            return None
        
        # Create ASE Atoms object with original cell parameters but only spacer molecules
        spacer_structure = Atoms(
            symbols=all_symbols,
            positions=all_positions,
            cell=self.spacer.get_cell() if self.spacer is not None else [20.0, 20.0, 20.0],
            pbc=self.spacer.get_pbc() if self.spacer is not None else [True, True, True]
        )
        
        return spacer_structure
    
    def create_a_sites_structure(self, a_site_molecules):
        """
        Create a VASP structure containing only the smaller molecules (A-sites).
        
        Parameters:
            a_site_molecules (dict): Dictionary of A-site molecule data
            
        Returns:
            ase.Atoms: ASE Atoms object with A-site molecules
        """
        if not a_site_molecules:
            print("No A-site molecules to create structure from")
            return None
        
        all_symbols = []
        all_positions = []
        
        for mol_id, mol_data in a_site_molecules.items():
            all_symbols.extend(mol_data['symbols'])
            all_positions.extend(mol_data['coordinates'])
        
        if not all_symbols:
            return None
        
        # Create ASE Atoms object with original cell parameters but only A-site molecules
        a_sites_structure = Atoms(
            symbols=all_symbols,
            positions=all_positions,
            cell=self.spacer.get_cell() if self.spacer is not None else [20.0, 20.0, 20.0],
            pbc=self.spacer.get_pbc() if self.spacer is not None else [True, True, True]
        )
        
        return a_sites_structure
    
    def create_salt_structure(self, spacer_molecules):
        """
        Create a VASP structure containing the spacer molecules plus terminal halogens.
        This creates a "salt" structure with the organic molecules and their terminal atoms.
        
        Parameters:
            spacer_molecules (dict): Dictionary of spacer molecule data
            
        Returns:
            ase.Atoms: ASE Atoms object with spacer molecules and terminal halogens
        """
        if not spacer_molecules:
            print("No spacer molecules to create salt structure from")
            return None
        
        all_symbols = []
        all_positions = []
        
        # Add spacer molecules first
        for mol_id, mol_data in spacer_molecules.items():
            all_symbols.extend(mol_data['symbols'])
            all_positions.extend(mol_data['coordinates'])
        
        # Add terminal X atoms from connectivity analysis
        terminal_count = 0
        if self.terminal_x_atoms:
            for oct_key, oct_terminal_data in self.terminal_x_atoms.items():
                terminal_atoms = oct_terminal_data.get('terminal_axial_atoms', [])
                for terminal_atom in terminal_atoms:
                    atom_index = terminal_atom['atom_index']
                    # Get atom symbol and position from original structure
                    atom_symbol = self.chemical_symbols[atom_index]
                    atom_position = self.coord[atom_index]
                    
                    all_symbols.append(atom_symbol)
                    all_positions.append(atom_position.tolist())
                    terminal_count += 1
        
        if not all_symbols:
            print("No atoms found for salt structure")
            return None
        
        print(f"Salt structure: Added {terminal_count} terminal halogen atoms to spacer molecules")
        
        # Create ASE Atoms object
        salt_structure = Atoms(
            symbols=all_symbols,
            positions=all_positions,
            cell=self.spacer.get_cell() if self.spacer is not None else [20.0, 20.0, 20.0],
            pbc=self.spacer.get_pbc() if self.spacer is not None else [True, True, True]
        )
        
        return salt_structure
    
    def create_layers_only_structure(self):
        """
        Create a VASP structure containing only the inorganic layers (without large molecules).
        This removes the 2 largest molecules but keeps the octahedral framework and any small molecules.
        
        Returns:
            ase.Atoms: ASE Atoms object with layers only
        """
        if not hasattr(self, 'spacer_molecules') or not self.spacer_molecules:
            print("No molecule analysis available - returning original structure")
            return self.atoms.copy()
        
        # Get spacer molecules and identify which ones to remove (2 largest)
        spacer_molecules, a_site_molecules = self.separate_molecules_by_size()
        
        # Get indices of atoms in large molecules that should be removed
        large_mol_indices = set()
        if spacer_molecules:
            for mol_id, mol_data in spacer_molecules.items():
                # Find atom indices in original structure corresponding to this molecule
                mol_positions = np.array(mol_data['coordinates'])
                
                for mol_pos in mol_positions:
                    # Find corresponding atom in original structure
                    for i, orig_pos in enumerate(self.atoms.get_positions()):
                        dist = self.geometry_calc.calculate_distance(mol_pos, orig_pos)
                        if dist < 0.1:  # Very small tolerance for exact match
                            large_mol_indices.add(i)
                            break
        
        # Create mask for atoms to keep (all except large molecules)
        all_indices = set(range(len(self.atoms)))
        keep_indices = sorted(all_indices - large_mol_indices)
        
        if not keep_indices:
            print("Warning: No atoms left after removing large molecules")
            return self.atoms.copy()
        
        # Create new structure with only the atoms we want to keep
        layers_structure = self.atoms[keep_indices]
        
        print(f"Layers-only structure: {len(layers_structure)} atoms (removed {len(large_mol_indices)} from large molecules)")
        
        return layers_structure

    def save_separated_structures(self, output_dir):
        """
        Save separate VASP structures for spacer and A-site molecules.
        
        Parameters:
            output_dir (str): Directory where to save the structures
        """
        import os
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Separate molecules by size
        spacer_molecules, a_site_molecules = self.separate_molecules_by_size()
        
        # Create and save spacer structure (2 largest molecules)
        if spacer_molecules:
            spacer_structure = self.create_spacer_structure(spacer_molecules)
            if spacer_structure is not None:
                spacer_file = os.path.join(output_dir, "spacer.vasp")
                spacer_structure.write(spacer_file)
                print(f"✓ Spacer structure saved: {spacer_file}")
                print(f"  Contains {len(spacer_structure)} atoms from {len(spacer_molecules)} molecules")
            else:
                print("✗ Failed to create spacer structure")
        else:
            print("✗ No spacer molecules found")
        
        # Create and save A-sites structure (remaining smaller molecules)
        if a_site_molecules:
            a_sites_structure = self.create_a_sites_structure(a_site_molecules)
            if a_sites_structure is not None:
                a_sites_file = os.path.join(output_dir, "a_sites.vasp")
                a_sites_structure.write(a_sites_file)
                print(f"✓ A-sites structure saved: {a_sites_file}")
                print(f"  Contains {len(a_sites_structure)} atoms from {len(a_site_molecules)} molecules")
            else:
                print("✗ Failed to create A-sites structure")
        else:
            print("✗ No A-site molecules found")
        
        return spacer_file if spacer_molecules else None, a_sites_file if a_site_molecules else None

    def save_salt_structure(self, output_path):
        """
        Save the salt structure (spacer molecules + 4 terminal halogens) to a VASP file.
        
        Parameters:
            output_path (str): Path where to save the salt structure
        """
        if not hasattr(self, 'spacer_molecules') or not self.spacer_molecules:
            print("Error: No spacer molecule analysis available. Run analyze_spacer_molecules() first.")
            return
        
        # Get spacer molecules (2 largest)
        spacer_molecules, _ = self.separate_molecules_by_size()
        
        if not spacer_molecules:
            print("Error: No spacer molecules found to create salt structure.")
            return
        
        # Create salt structure
        salt_structure = self.create_salt_structure(spacer_molecules)
        
        if salt_structure is not None:
            salt_structure.write(output_path)
            print(f"Salt structure saved to: {output_path}")
            print(f"  Contains {len(salt_structure)} atoms (spacer molecules + 4 terminal halogens)")
        else:
            print("Error: Failed to create salt structure.")

