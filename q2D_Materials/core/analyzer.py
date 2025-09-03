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
from .vector_analysis import VectorAnalyzer

# Use matplotlib without x server
import matplotlib
matplotlib.use('Agg')


class q2D_analyzer:
    """
    Refactored analyzer for 2D quantum materials using VASP results.
    
    Creates a unified ontology:
    Experiment -> Cell Properties -> Octahedra (with atomic indices and properties)
    -> Vector Analysis (if salt structure available)
    
    Uses modular components for:
    - Geometry calculations (GeometryCalculator)
    - Angular analysis (AngularAnalyzer) 
    - Connectivity analysis (ConnectivityAnalyzer)
    - Vector analysis (VectorAnalyzer)
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
        self.vector_analyzer = VectorAnalyzer()

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
            
            # Create and store salt structure for vector analysis
            if self.spacer_molecules:
                spacer_mols, _ = self.separate_molecules_by_size()
                self.salt_structure = self.create_salt_structure(spacer_mols)
                
                # Perform vector analysis using original structure and connectivity analysis
                # Pass the analyzer instance to access connectivity analysis
                self.vector_analysis = self.vector_analyzer.analyze_salt_structure_vectors(
                    self.salt_structure, x_symbol=self.x, analyzer=self
                )
            else:
                self.salt_structure = None
                self.vector_analysis = {'error': 'No spacer molecules found for salt structure'}
        else:
            self.spacer_molecules = None
            self.molecule_ontology = None
            self.salt_structure = None
            self.vector_analysis = {'error': 'No spacer structure available'}
        
        # Update ontology with vector analysis
        self._update_ontology_with_vector_analysis()

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
        Note: Vector analysis is added later via _update_ontology_with_vector_analysis()
        
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

    def _update_ontology_with_vector_analysis(self):
        """
        Update the unified ontology with vector analysis and penetration depth results.
        This is called after vector analysis is completed.
        """
        if hasattr(self, 'ontology') and hasattr(self, 'vector_analysis'):
            self.ontology['vector_analysis'] = self.vector_analysis
            
            # Also add salt structure information to the ontology
            if hasattr(self, 'salt_structure') and self.salt_structure is not None:
                salt_info = {
                    'total_atoms': len(self.salt_structure),
                    'composition': {},
                    'cell_parameters': {
                        'a': float(self.salt_structure.get_cell()[0][0]),
                        'b': float(self.salt_structure.get_cell()[1][1]), 
                        'c': float(self.salt_structure.get_cell()[2][2])
                    }
                }
                
                # Get salt composition
                for symbol in self.salt_structure.get_chemical_symbols():
                    salt_info['composition'][symbol] = salt_info['composition'].get(symbol, 0) + 1
                
                self.ontology['salt_structure_info'] = salt_info
                
                # Add penetration depth analysis to ontology
                penetration_results = self.get_penetration_depth_analysis()
                if 'error' not in penetration_results:
                    self.ontology['penetration_analysis'] = penetration_results

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
        
        # 7. Export vector analysis if available
        if hasattr(self, 'vector_analysis') and 'error' not in self.vector_analysis:
            vector_file = os.path.join(output_dir, f"{self.experiment_name}_vector_analysis.json")
            self.export_vector_analysis_json(vector_file)
            
            # Add vector summary to the text report
            vector_summary_file = os.path.join(output_dir, f"{self.experiment_name}_vector_summary.txt")
            with open(vector_summary_file, 'w') as f:
                f.write(self.get_vector_summary())
            print(f"Vector analysis summary saved to: {vector_summary_file}")
        
        # 8. Export penetration depth analysis
        penetration_results = self.get_penetration_depth_analysis()
        if 'error' not in penetration_results:
            penetration_file = os.path.join(output_dir, f"{self.experiment_name}_penetration_analysis.json")
            self.export_penetration_analysis_json(penetration_file)
            
            # Add penetration summary to the text report
            penetration_summary_file = os.path.join(output_dir, f"{self.experiment_name}_penetration_summary.txt")
            with open(penetration_summary_file, 'w') as f:
                f.write(self.get_penetration_summary())
            print(f"Penetration depth analysis summary saved to: {penetration_summary_file}")
        else:
            print(f"Penetration depth analysis skipped: {penetration_results['error']}")
        
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

    def get_vector_analysis(self):
        """
        Get the vector analysis results for the salt structure.
        
        Returns:
            dict: Vector analysis results including angle between planes,
                  angles vs z-axis, and distance between plane centers
        """
        return getattr(self, 'vector_analysis', {'error': 'Vector analysis not available'})
    
    def get_vector_summary(self):
        """
        Get the key vector analysis values.
        
        Returns:
            dict: Key vector analysis values
        """
        if hasattr(self, 'vector_analysis'):
            return self.vector_analyzer.get_vector_summary(self.vector_analysis)
        else:
            return {'error': 'Vector analysis not available'}
    
    def create_vector_plot(self, output_filename=None):
        """
        Create an interactive vector analysis plot.
        
        Parameters:
            output_filename (str, optional): Output HTML filename
            
        Returns:
            str: Path to created HTML file or error message
        """
        if output_filename is None:
            output_filename = f"{self.experiment_name}_vector_plot.html"
        
        if hasattr(self, 'vector_analysis'):
            return self.vector_analyzer.create_interactive_plot(
                self.salt_structure, self.vector_analysis, output_filename, analyzer=self
            )
        else:
            return "Error: Vector analysis not available"
    
    def get_salt_structure(self):
        """
        Get the salt structure (spacer molecules + terminal halogens).
        
        Returns:
            ase.Atoms: Salt structure or None if not available
        """
        return getattr(self, 'salt_structure', None)
    
    def export_vector_analysis_json(self, filename=None):
        """
        Export the vector analysis results to JSON format.
        
        Parameters:
            filename (str, optional): Output filename
        """
        if filename is None:
            filename = f"{self.experiment_name}_vector_analysis.json"
        
        import json
        
        if not hasattr(self, 'vector_analysis'):
            print("Error: No vector analysis available.")
            return
        
        # Create comprehensive vector analysis export
        comprehensive_analysis = {
            "experiment": {
                "name": str(self.experiment_name),
                "file_path": str(self.file_path),
                "timestamp": str(pd.Timestamp.now())
            },
            "salt_structure_info": {
                "total_atoms": len(self.salt_structure) if self.salt_structure is not None else 0,
                "halogen_symbol": str(self.x)
            },
            "vector_analysis": self.vector_analysis
        }
        
        # Ensure all data is JSON serializable
        serializable_analysis = self._make_json_serializable(comprehensive_analysis)
        
        with open(filename, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"Vector analysis exported to {filename}")
        return serializable_analysis

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

    def get_penetration_depth_analysis(self):
        """
        Perform penetration depth analysis for the spacer molecules.
        This method reuses the already calculated spacer molecules and vector analysis.
        
        Returns:
            dict: Penetration depth analysis results
        """
        if not hasattr(self, 'spacer_molecules') or not self.spacer_molecules:
            return {'error': 'No spacer molecule analysis available. Run analyze_spacer_molecules() first.'}
        
        if not hasattr(self, 'salt_structure') or self.salt_structure is None:
            return {'error': 'No salt structure available for penetration depth analysis.'}
        
        # Get the 2 largest spacer molecules (already calculated)
        spacer_molecules, _ = self.separate_molecules_by_size()
        
        if len(spacer_molecules) < 2:
            return {'error': f'Need 2 spacer molecules for penetration analysis, found {len(spacer_molecules)}'}
        
        # Perform penetration depth analysis using vector analyzer
        penetration_results = self.vector_analyzer.get_penetration_depth(
            self.salt_structure, 
            spacer_molecules, 
            x_symbol=self.x,
            analyzer=self
        )
        
        return penetration_results
    
    def get_penetration_summary(self):
        """
        Get a formatted summary of the penetration depth analysis.
        
        Returns:
            str: Formatted penetration depth summary
        """
        penetration_results = self.get_penetration_depth_analysis()
        
        if 'error' in penetration_results:
            return f"Penetration Analysis Error: {penetration_results['error']}"
        
        summary_lines = []
        summary_lines.append("=" * 60)
        summary_lines.append("PENETRATION DEPTH ANALYSIS SUMMARY")
        summary_lines.append("=" * 60)
        
        # Process individual molecules
        molecule_count = 0
        for key, mol_data in penetration_results.items():
            if key.startswith('molecule_') and 'error' not in mol_data:
                molecule_count += 1
                mol_id = key.split('_')[1]
                
                summary_lines.append(f"\nMolecule {mol_id} ({mol_data['formula']}):")
                summary_lines.append("-" * 30)
                
                segments = mol_data['penetration_segments']
                summary_lines.append(f"  N1-N2 straight distance: {segments['n1_n2_straight_distance']:.3f} Å")
                summary_lines.append(f"  Penetration segments:")
                summary_lines.append(f"    N1 → Low plane:      {segments['n1_to_low_plane']:.3f} Å")
                summary_lines.append(f"    Low → High plane:    {segments['low_plane_to_high_plane']:.3f} Å")
                summary_lines.append(f"    High plane → N2:     {segments['high_plane_to_n2']:.3f} Å")
                summary_lines.append(f"    Total calculated:    {segments['total_calculated']:.3f} Å")
                summary_lines.append(f"    Length difference:   {segments['length_difference']:.6f} Å")
                
                # Show N1 and N2 positions
                n1_pos = mol_data['n1_position']
                n2_pos = mol_data['n2_position']
                summary_lines.append(f"  N1 position: ({n1_pos[0]:.3f}, {n1_pos[1]:.3f}, {n1_pos[2]:.3f})")
                summary_lines.append(f"  N2 position: ({n2_pos[0]:.3f}, {n2_pos[1]:.3f}, {n2_pos[2]:.3f})")
        
        # Add comparative analysis if available
        if 'comparative_analysis' in penetration_results:
            comp_data = penetration_results['comparative_analysis']
            summary_lines.append(f"\nCOMPARATIVE ANALYSIS:")
            summary_lines.append("-" * 30)
            
            comp = comp_data['penetration_comparison']
            summary_lines.append(f"  Segment differences between molecules:")
            summary_lines.append(f"    N1 → Low plane difference:   {comp['n1_to_low_diff']:.3f} Å")
            summary_lines.append(f"    Low → High plane difference: {comp['low_to_high_diff']:.3f} Å")
            summary_lines.append(f"    High plane → N2 difference:  {comp['high_to_n2_diff']:.3f} Å")
        
        # Add plane information
        if 'plane_reference' in penetration_results:
            plane_ref = penetration_results['plane_reference']['vector_analysis_results']
            summary_lines.append(f"\nPLANE INFORMATION:")
            summary_lines.append("-" * 30)
            summary_lines.append(f"  Angle between planes: {plane_ref['angle_between_planes_degrees']:.2f}°")
            summary_lines.append(f"  Distance between plane centers: {plane_ref['distance_between_plane_centers_angstrom']:.3f} Å")
        
        summary_lines.append("\n" + "=" * 60)
        
        return "\n".join(summary_lines)
    
    def export_penetration_analysis_json(self, filename=None):
        """
        Export the penetration depth analysis to JSON format.
        
        Parameters:
            filename (str, optional): Output filename
        """
        if filename is None:
            filename = f"{self.experiment_name}_penetration_analysis.json"
        
        import json
        
        penetration_results = self.get_penetration_depth_analysis()
        
        if 'error' in penetration_results:
            print(f"Error: {penetration_results['error']}")
            return
        
        # Create comprehensive export with metadata
        comprehensive_analysis = {
            "experiment": {
                "name": str(self.experiment_name),
                "file_path": str(self.file_path),
                "timestamp": str(pd.Timestamp.now())
            },
            "analysis_parameters": {
                "halogen_symbol": str(self.x),
                "central_metal": str(self.b),
                "total_spacer_molecules": len(self.spacer_molecules.get('molecules', {})) if self.spacer_molecules else 0,
                "analyzed_molecules": 2  # We analyze the 2 largest
            },
            "penetration_analysis": penetration_results
        }
        
        # Ensure all data is JSON serializable
        serializable_analysis = self._make_json_serializable(comprehensive_analysis)
        
        with open(filename, 'w') as f:
            json.dump(serializable_analysis, f, indent=2)
        
        print(f"Penetration depth analysis exported to {filename}")
        return serializable_analysis

    def print_ontology_summary(self):
        """
        Print a comprehensive summary of the ontology structure including penetration depth analysis.
        """
        print(f"\n=== UNIFIED ONTOLOGY SUMMARY ===")
        print(f"Experiment: {self.ontology['experiment']['name']}")
        print(f"\nCell Properties:")
        cell_props = self.ontology['cell_properties']
        lattice = cell_props['lattice_parameters']
        print(f"  A={lattice['A']:.3f}, B={lattice['B']:.3f}, C={lattice['C']:.3f}")
        print(f"  α={lattice['Alpha']:.1f}°, β={lattice['Beta']:.1f}°, γ={lattice['Gamma']:.1f}°")
        print(f"  Volume: {cell_props['structure_info']['cell_volume']:.3f}")
        print(f"  Composition: {cell_props['composition']['metal_B']}-{cell_props['composition']['halogen_X']}")
        
        print(f"\nOctahedra Analysis:")
        for oct_key, oct_data in self.ontology['octahedra'].items():
            central = oct_data['central_atom']
            distortion = oct_data['distortion_parameters']
            ligands = oct_data['ligand_atoms']
            bond_analysis = oct_data['bond_distance_analysis']
            geometric = oct_data['geometric_properties']
            detailed = oct_data['detailed_atom_info']
            angular_data = oct_data.get('angular_analysis', {})
            
            print(f"  {oct_key}: Central atom {central['symbol']} (index {central['global_index']})")
            print(f"    Distortion: ζ={distortion['zeta']:.4f}, δ={distortion['delta']:.4f}, σ={distortion['sigma']:.4f}")
            print(f"    Theta: mean={distortion['theta_mean']:.2f}°, min={distortion['theta_min']:.2f}°, max={distortion['theta_max']:.2f}°")
            print(f"    Bond distances: mean={bond_analysis['mean_bond_distance']:.3f}Å, variance={bond_analysis['bond_distance_variance']:.6f}")
            print(f"    Volume: {geometric['octahedral_volume']:.3f}Å³, Is octahedral: {geometric['is_octahedral']}")
            print(f"    Ligand indices: {ligands['all_ligand_global_indices']}")
            print(f"    Ligand symbols: {detailed['all_ligand_symbols']}")
            
            # Add angular analysis information
            if angular_data:
                if 'axial_central_axial' in angular_data:
                    aca = angular_data['axial_central_axial']
                    print(f"    Axial-Central-Axial angle: {aca['angle_degrees']:.2f}° (deviation from 180°: {aca['deviation_from_180']:.2f}°)")
                    print(f"    Is linear: {aca['is_linear']}")
                
                if 'central_axial_central' in angular_data:
                    cac_angles = angular_data['central_axial_central']
                    if cac_angles:
                        print(f"    Central-Axial-Central bridges: {len(cac_angles)}")
                        for i, cac in enumerate(cac_angles):
                            print(f"      Bridge {i+1}: {cac['angle_degrees']:.2f}° to {cac['connected_octahedron']} via atom {cac['axial_atom_global_index']}")
                
                if 'central_equatorial_central' in angular_data:
                    cec_angles = angular_data['central_equatorial_central']
                    if cec_angles:
                        print(f"    Central-Equatorial-Central bridges: {len(cec_angles)}")
                        for i, cec in enumerate(cec_angles):
                            print(f"      Bridge {i+1}: {cec['angle_degrees']:.2f}° to {cec['connected_octahedron']} via atom {cec['equatorial_atom_global_index']}")
                
                if 'summary' in angular_data:
                    summary = angular_data['summary']
                    print(f"    Angular summary: {summary['total_axial_bridges']} axial + {summary['total_equatorial_bridges']} equatorial bridges")
                    if summary['average_central_axial_central_angle'] > 0:
                        print(f"    Average C-A-C angle: {summary['average_central_axial_central_angle']:.2f}°")
                    if summary['average_central_equatorial_central_angle'] > 0:
                        print(f"    Average C-E-C angle: {summary['average_central_equatorial_central_angle']:.2f}°")
            
            if ligands['axial_global_indices'] and ligands['equatorial_global_indices']:
                print(f"    Axial: {ligands['axial_global_indices']} ({detailed['axial_atom_symbols']})")
                print(f"    Equatorial: {ligands['equatorial_global_indices']} ({detailed['equatorial_atom_symbols']})")
            
            # Show connectivity information
            connectivity = self.ontology.get('connectivity_analysis', {})
            connections = connectivity.get('octahedra_connections', {}).get(oct_key, [])
            if connections:
                shared_info = []
                for conn in connections:
                    shared_info.append(f"{conn['connected_octahedron']} (atom {conn['shared_atom_index']}:{conn['atom_symbol']})")
                print(f"    Connected to: {', '.join(shared_info)}")
            print()
        
        # Show overall connectivity summary using connectivity analyzer
        connectivity = self.ontology.get('connectivity_analysis', {})
        if connectivity:
            print(self.connectivity_analyzer.get_connectivity_summary(connectivity))
        
        # Show layer analysis summary
        layers_analysis = self.ontology.get('layers_analysis', {})
        if layers_analysis:
            print(self.layers_analyzer.get_layer_summary())
        
        # Show molecule analysis summary
        if hasattr(self, 'spacer_molecules') and self.spacer_molecules:
            print("\n=== MOLECULE ANALYSIS ===")
            molecules = self.spacer_molecules.get('molecules', {})
            print(f"Total molecules identified: {len(molecules)}")
            
            # Show molecule distribution
            distribution = self._get_molecule_distribution()
            if distribution:
                print("Molecule distribution:")
                for formula, count in sorted(distribution.items()):
                    print(f"  {formula}: {count} molecules")
        
        # Show vector analysis summary
        vector_analysis = self.ontology.get('vector_analysis', {})
        if vector_analysis and 'error' not in vector_analysis:
            print("\n=== VECTOR ANALYSIS ===")
            results = vector_analysis['vector_analysis_results']
            print(f"Angle between planes: {results['angle_between_planes_degrees']:.2f}°")
            print(f"Distance between plane centers: {results['distance_between_plane_centers_angstrom']:.3f} Å")
        
        # Show penetration depth analysis summary
        penetration_analysis = self.ontology.get('penetration_analysis', {})
        if penetration_analysis and 'error' not in penetration_analysis:
            print("\n=== PENETRATION DEPTH ANALYSIS ===")
            molecule_count = 0
            for key, mol_data in penetration_analysis.items():
                if key.startswith('molecule_') and 'error' not in mol_data:
                    molecule_count += 1
                    mol_id = key.split('_')[1]
                    segments = mol_data['penetration_segments']
                    print(f"Molecule {mol_id} ({mol_data['formula']}):")
                    print(f"  N1 → Low plane: {segments['n1_to_low_plane']:.3f} Å")
                    print(f"  Low → High plane: {segments['low_plane_to_high_plane']:.3f} Å") 
                    print(f"  High plane → N2: {segments['high_plane_to_n2']:.3f} Å")
                    print(f"  Total length: {segments['molecular_length']:.3f} Å")
                    print()
            
            if 'comparative_analysis' in penetration_analysis:
                comp = penetration_analysis['comparative_analysis']['penetration_comparison']
                print(f"Segment differences between molecules:")
                print(f"  ΔN1-Low: {comp['n1_to_low_diff']:.3f} Å")
                print(f"  ΔLow-High: {comp['low_to_high_diff']:.3f} Å") 
                print(f"  ΔHigh-N2: {comp['high_to_n2_diff']:.3f} Å")
        
        print(f"\n=== END SUMMARY ===")

    def get_penetration_depth_values(self):
        """
        Get just the penetration depth segment values for easy access.
        
        Returns:
            dict: Simple dictionary with penetration depth values for each molecule
        """
        penetration_results = self.get_penetration_depth_analysis()
        
        if 'error' in penetration_results:
            return {'error': penetration_results['error']}
        
        simple_results = {}
        
        for key, mol_data in penetration_results.items():
            if key.startswith('molecule_') and 'error' not in mol_data:
                mol_id = key.split('_')[1]
                segments = mol_data['penetration_segments']
                
                simple_results[f'molecule_{mol_id}'] = {
                    'formula': mol_data['formula'],
                    'n1_to_low_plane': segments['n1_to_low_plane'],
                    'low_plane_to_high_plane': segments['low_plane_to_high_plane'],
                    'high_plane_to_n2': segments['high_plane_to_n2'],
                    'total_length': segments['molecular_length']
                }
        
        # Add comparison if available
        if 'comparative_analysis' in penetration_results:
            comp = penetration_results['comparative_analysis']['penetration_comparison']
            simple_results['differences'] = {
                'n1_to_low_diff': comp['n1_to_low_diff'],
                'low_to_high_diff': comp['low_to_high_diff'],
                'high_to_n2_diff': comp['high_to_n2_diff']
            }
        
        return simple_results

