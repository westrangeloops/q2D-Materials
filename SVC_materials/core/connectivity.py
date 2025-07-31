"""
Connectivity analysis for octahedral structures.

This module handles connectivity analysis between octahedra,
including shared atom detection and network topology.
"""

import numpy as np
from .geometry import GeometryCalculator


class MoleculeIdentifier:
    """
    Identifies and clusters individual molecules in a structure using PBC-aware connectivity.
    """
    
    def __init__(self, geometry_calculator=None, bond_cutoff_multiplier=1.2):
        """
        Initialize molecule identifier.
        
        Parameters:
        geometry_calculator: GeometryCalculator instance
        bond_cutoff_multiplier: Multiplier for covalent radii to determine bonds
        """
        self.geometry_calc = geometry_calculator or GeometryCalculator()
        self.bond_cutoff_multiplier = bond_cutoff_multiplier
        
        # Standard covalent radii in Angstroms (from ASE or common literature)
        self.covalent_radii = {
            'H': 0.31, 'C': 0.76, 'N': 0.71, 'O': 0.66, 'F': 0.57,
            'S': 1.05, 'Cl': 0.99, 'Br': 1.20, 'I': 1.39, 'P': 1.07,
            'Si': 1.11, 'B': 0.84, 'Al': 1.21, 'Mg': 1.41, 'Ca': 1.76,
            'Na': 1.66, 'K': 2.03, 'Li': 1.28, 'Be': 0.96, 'Ne': 0.58,
            'Ar': 1.06, 'Kr': 1.16, 'Xe': 1.40, 'Rn': 1.50, 'He': 0.28
        }
    
    def identify_molecules(self, atoms_obj, symbols, coordinates):
        """
        Identify individual molecules using connectivity analysis with PBC.
        
        Parameters:
        atoms_obj: ASE Atoms object (for PBC information)
        symbols: List of atomic symbols
        coordinates: Array of atomic coordinates
        
        Returns:
        dict: Molecule analysis containing molecule assignments and statistics
        """
        # Create adjacency matrix using PBC-aware distance calculations
        adjacency_matrix = self._create_adjacency_matrix(atoms_obj, symbols, coordinates)
        
        # Find connected components (molecules)
        molecule_assignments = self._find_connected_components(adjacency_matrix)
        
        # Analyze molecules
        molecule_analysis = self._analyze_molecules(
            molecule_assignments, symbols, coordinates, atoms_obj
        )
        
        return molecule_analysis
    
    def _create_adjacency_matrix(self, atoms_obj, symbols, coordinates):
        """
        Create adjacency matrix based on PBC-aware bond distances.
        
        Returns:
        numpy.ndarray: Boolean adjacency matrix
        """
        n_atoms = len(symbols)
        adjacency_matrix = np.zeros((n_atoms, n_atoms), dtype=bool)
        
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                # Calculate PBC-aware distance
                distance = self.geometry_calc.calculate_distance(
                    coordinates[i], coordinates[j]
                )
                
                # Get bond cutoff based on covalent radii
                symbol_i = symbols[i]
                symbol_j = symbols[j]
                
                radius_i = self.covalent_radii.get(symbol_i, 1.5)  # Default fallback
                radius_j = self.covalent_radii.get(symbol_j, 1.5)
                
                bond_cutoff = (radius_i + radius_j) * self.bond_cutoff_multiplier
                
                # Check if atoms are bonded
                if distance <= bond_cutoff:
                    adjacency_matrix[i, j] = True
                    adjacency_matrix[j, i] = True
        
        return adjacency_matrix
    
    def _find_connected_components(self, adjacency_matrix):
        """
        Find connected components using depth-first search.
        
        Returns:
        numpy.ndarray: Array of molecule IDs for each atom
        """
        n_atoms = adjacency_matrix.shape[0]
        visited = np.zeros(n_atoms, dtype=bool)
        molecule_id = np.zeros(n_atoms, dtype=int)
        current_molecule = 0
        
        def dfs(atom_idx):
            """Depth-first search to find connected atoms."""
            visited[atom_idx] = True
            molecule_id[atom_idx] = current_molecule
            
            # Visit all connected neighbors
            for neighbor in np.where(adjacency_matrix[atom_idx])[0]:
                if not visited[neighbor]:
                    dfs(neighbor)
        
        # Find all connected components
        for atom_idx in range(n_atoms):
            if not visited[atom_idx]:
                dfs(atom_idx)
                current_molecule += 1
        
        return molecule_id
    
    def _analyze_molecules(self, molecule_assignments, symbols, coordinates, atoms_obj):
        """
        Analyze identified molecules and provide statistics.
        
        Returns:
        dict: Comprehensive molecule analysis
        """
        n_molecules = len(np.unique(molecule_assignments))
        
        molecule_analysis = {
            'total_molecules': n_molecules,
            'molecule_assignments': molecule_assignments.tolist(),
            'molecules': {},
            'molecule_formulas': {},
            'formula_counts': {},
            'center_of_mass': {},
            'molecule_sizes': {},
            'largest_molecule': None,
            'smallest_molecule': None
        }
        
        # Analyze each molecule
        for mol_id in range(n_molecules):
            atom_indices = np.where(molecule_assignments == mol_id)[0]
            mol_symbols = [symbols[i] for i in atom_indices]
            mol_coordinates = coordinates[atom_indices]
            
            # Calculate molecular formula
            formula = self._calculate_molecular_formula(mol_symbols)
            
            # Calculate center of mass (PBC-aware)
            com = self._calculate_center_of_mass_pbc(
                mol_coordinates, mol_symbols, atoms_obj
            )
            
            molecule_analysis['molecules'][mol_id] = {
                'atom_indices': atom_indices.tolist(),
                'symbols': mol_symbols,
                'coordinates': mol_coordinates.tolist(),
                'formula': formula,
                'size': len(atom_indices),
                'center_of_mass': com.tolist()
            }
            
            molecule_analysis['molecule_formulas'][mol_id] = formula
            molecule_analysis['center_of_mass'][mol_id] = com.tolist()
            molecule_analysis['molecule_sizes'][mol_id] = len(atom_indices)
        
        # Count formula occurrences
        all_formulas = list(molecule_analysis['molecule_formulas'].values())
        unique_formulas = list(set(all_formulas))
        
        for formula in unique_formulas:
            count = all_formulas.count(formula)
            molecule_analysis['formula_counts'][formula] = count
        
        # Find largest and smallest molecules
        sizes = list(molecule_analysis['molecule_sizes'].values())
        if sizes:
            max_size = max(sizes)
            min_size = min(sizes)
            
            for mol_id, size in molecule_analysis['molecule_sizes'].items():
                if size == max_size:
                    molecule_analysis['largest_molecule'] = mol_id
                    break
            
            for mol_id, size in molecule_analysis['molecule_sizes'].items():
                if size == min_size:
                    molecule_analysis['smallest_molecule'] = mol_id
                    break
        
        return molecule_analysis
    
    def _calculate_molecular_formula(self, symbols):
        """
        Calculate molecular formula from list of atomic symbols.
        
        Returns:
        str: Molecular formula (e.g., 'C6H12N2O2')
        """
        from collections import Counter
        
        element_counts = Counter(symbols)
        
        # Sort elements: C first, H second, then alphabetically
        formula_parts = []
        
        # Carbon first
        if 'C' in element_counts:
            count = element_counts['C']
            formula_parts.append(f"C{count}" if count > 1 else "C")
            del element_counts['C']
        
        # Hydrogen second
        if 'H' in element_counts:
            count = element_counts['H']
            formula_parts.append(f"H{count}" if count > 1 else "H")
            del element_counts['H']
        
        # Other elements alphabetically
        for element in sorted(element_counts.keys()):
            count = element_counts[element]
            formula_parts.append(f"{element}{count}" if count > 1 else element)
        
        return ''.join(formula_parts) if formula_parts else ''
    
    def _calculate_center_of_mass_pbc(self, coordinates, symbols, atoms_obj):
        """
        Calculate center of mass with PBC considerations.
        
        Returns:
        numpy.ndarray: Center of mass coordinates
        """
        # Standard atomic masses (in u/amu)
        atomic_masses = {
            'H': 1.008, 'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998,
            'S': 32.065, 'Cl': 35.453, 'Br': 79.904, 'I': 126.90, 'P': 30.974,
            'Si': 28.085, 'B': 10.811, 'Al': 26.982, 'Mg': 24.305, 'Ca': 40.078,
            'Na': 22.990, 'K': 39.098, 'Li': 6.941, 'Be': 9.012
        }
        
        masses = np.array([atomic_masses.get(symbol, 12.0) for symbol in symbols])
        total_mass = np.sum(masses)
        
        # For PBC, we need to unwrap coordinates to calculate proper center of mass
        # Simple approach: use the first atom as reference and unwrap others
        if len(coordinates) == 1:
            return coordinates[0]
        
        reference_coord = coordinates[0]
        unwrapped_coords = [reference_coord]
        
        # Get cell parameters
        cell = atoms_obj.get_cell()
        pbc = atoms_obj.get_pbc()
        use_pbc = np.any(pbc) and atoms_obj.get_volume() > 0
        
        if use_pbc:
            # Unwrap coordinates relative to first atom
            for coord in coordinates[1:]:
                # Find minimum image relative to reference
                displacement = coord - reference_coord
                
                # Apply PBC correction if needed
                displacement_pbc = self.geometry_calc._calculate_distance_pbc_signo(
                    reference_coord[0], reference_coord[1], reference_coord[2],
                    coord[0], coord[1], coord[2],
                    cell[0], cell[1], cell[2]
                )
                
                unwrapped_coord = reference_coord + np.array(displacement_pbc)
                unwrapped_coords.append(unwrapped_coord)
        else:
            unwrapped_coords = coordinates
        
        unwrapped_coords = np.array(unwrapped_coords)
        
        # Calculate weighted center of mass
        center_of_mass = np.sum(unwrapped_coords * masses[:, np.newaxis], axis=0) / total_mass
        
        return center_of_mass
    
    def get_molecule_summary(self, molecule_analysis):
        """
        Generate a human-readable summary of molecule analysis.
        
        Parameters:
        molecule_analysis: Result from identify_molecules
        
        Returns:
        str: Formatted summary
        """
        summary_lines = []
        summary_lines.append("=== MOLECULE IDENTIFICATION SUMMARY ===")
        
        # Overall statistics
        total_molecules = molecule_analysis['total_molecules']
        summary_lines.append(f"Total molecules identified: {total_molecules}")
        
        # Formula distribution
        formula_counts = molecule_analysis['formula_counts']
        if formula_counts:
            summary_lines.append(f"\nMolecular formulas found:")
            for formula, count in sorted(formula_counts.items()):
                summary_lines.append(f"  {formula}: {count} molecule(s)")
        
        # Size statistics
        sizes = list(molecule_analysis['molecule_sizes'].values())
        if sizes:
            summary_lines.append(f"\nMolecule size statistics:")
            summary_lines.append(f"  Average size: {np.mean(sizes):.1f} atoms")
            summary_lines.append(f"  Size range: {min(sizes)} - {max(sizes)} atoms")
        
        # Largest and smallest molecules
        largest_id = molecule_analysis['largest_molecule']
        smallest_id = molecule_analysis['smallest_molecule']
        
        if largest_id is not None:
            largest_formula = molecule_analysis['molecule_formulas'][largest_id]
            largest_size = molecule_analysis['molecule_sizes'][largest_id]
            summary_lines.append(f"  Largest molecule: {largest_formula} ({largest_size} atoms)")
        
        if smallest_id is not None and smallest_id != largest_id:
            smallest_formula = molecule_analysis['molecule_formulas'][smallest_id]
            smallest_size = molecule_analysis['molecule_sizes'][smallest_id]
            summary_lines.append(f"  Smallest molecule: {smallest_formula} ({smallest_size} atoms)")
        
        return '\n'.join(summary_lines)


class ConnectivityAnalyzer:
    """
    Handles connectivity analysis between octahedral structures.
    """
    
    def __init__(self, geometry_calculator=None):
        """
        Initialize connectivity analyzer.
        
        Parameters:
        geometry_calculator: GeometryCalculator instance
        """
        self.geometry_calc = geometry_calculator or GeometryCalculator()
        self.molecule_identifier = MoleculeIdentifier(geometry_calculator)
    
    def analyze_octahedra_connectivity(self, octahedra_data, all_coords, all_symbols):
        """
        Analyze which atoms are shared between octahedra.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        all_coords: All atomic coordinates
        all_symbols: All atomic symbols
        
        Returns:
        dict: Connectivity analysis including shared atoms and terminal axial information
        """
        connectivity = {
            "shared_atoms": {},
            "octahedra_connections": {},
            "sharing_types": {},
            "network_statistics": {},
            "terminal_axial_atoms": {}
        }
        
        # Create a mapping of which octahedra each atom belongs to
        atom_to_octahedra = {}
        
        for oct_key, oct_data in octahedra_data.items():
            central_index = oct_data['central_atom']['global_index']
            ligand_indices = oct_data['ligand_atoms']['all_ligand_global_indices']
            
            # Track central atoms
            if central_index not in atom_to_octahedra:
                atom_to_octahedra[central_index] = []
            atom_to_octahedra[central_index].append((oct_key, 'central'))
            
            # Track ligand atoms
            for ligand_idx in ligand_indices:
                if ligand_idx not in atom_to_octahedra:
                    atom_to_octahedra[ligand_idx] = []
                atom_to_octahedra[ligand_idx].append((oct_key, 'ligand'))
        
        # Find shared atoms
        for atom_idx, octahedra_list in atom_to_octahedra.items():
            if len(octahedra_list) > 1:
                connectivity["shared_atoms"][int(atom_idx)] = [
                    {"octahedron": oct_key, "role": role} for oct_key, role in octahedra_list
                ]
        
        # Analyze connections between octahedra
        for oct_key in octahedra_data.keys():
            connectivity["octahedra_connections"][oct_key] = []
            
        for atom_idx, octahedra_list in atom_to_octahedra.items():
            if len(octahedra_list) > 1:
                # This atom is shared between multiple octahedra
                ligand_octahedra = [oct_key for oct_key, role in octahedra_list if role == 'ligand']
                
                # Add connections with enhanced information
                for i, oct1 in enumerate(ligand_octahedra):
                    for oct2 in ligand_octahedra[i+1:]:
                        # Determine connection type (axial or equatorial)
                        connection_type = self._determine_connection_type(oct1, oct2, atom_idx, octahedra_data)
                        
                        connection_info = {
                            "connected_octahedron": oct2,
                            "shared_atom_index": int(atom_idx),
                            "atom_symbol": str(all_symbols[atom_idx]),
                            "connection_type": connection_type
                        }
                        
                        # Avoid duplicates
                        if connection_info not in connectivity["octahedra_connections"][oct1]:
                            connectivity["octahedra_connections"][oct1].append(connection_info)
                        
                        reverse_connection = {
                            "connected_octahedron": oct1,
                            "shared_atom_index": int(atom_idx),
                            "atom_symbol": str(all_symbols[atom_idx]),
                            "connection_type": connection_type
                        }
                        
                        if reverse_connection not in connectivity["octahedra_connections"][oct2]:
                            connectivity["octahedra_connections"][oct2].append(reverse_connection)
        
        # Identify terminal axial atoms using PBC-aware calculations
        connectivity["terminal_axial_atoms"] = self._identify_terminal_axial_atoms(
            octahedra_data, connectivity["shared_atoms"]
        )
        
        # Analyze sharing types
        connectivity["sharing_types"] = self._analyze_sharing_types(
            connectivity["shared_atoms"], octahedra_data, all_coords, all_symbols
        )
        
        # Calculate network statistics
        connectivity["network_statistics"] = self._calculate_network_statistics(
            connectivity["octahedra_connections"], connectivity["shared_atoms"]
        )
        
        return connectivity
    
    def analyze_spacer_molecules(self, spacer_atoms):
        """
        Analyze molecules in the isolated spacer structure.
        
        Parameters:
        spacer_atoms: ASE Atoms object containing only spacer atoms
        
        Returns:
        dict: Molecule analysis
        """
        if spacer_atoms is None or len(spacer_atoms) == 0:
            return {
                'total_molecules': 0,
                'molecule_assignments': [],
                'molecules': {},
                'molecule_formulas': {},
                'formula_counts': {},
                'error': 'No spacer atoms available'
            }
        
        symbols = spacer_atoms.get_chemical_symbols()
        coordinates = spacer_atoms.get_positions()
        
        return self.molecule_identifier.identify_molecules(spacer_atoms, symbols, coordinates)
    
    def _determine_connection_type(self, oct1_key, oct2_key, shared_atom_idx, octahedra_data):
        """
        Determine if a connection is through axial or equatorial atoms.
        
        Parameters:
        oct1_key, oct2_key (str): Keys of connected octahedra
        shared_atom_idx (int): Index of shared atom
        octahedra_data (dict): Dictionary of octahedra data
        
        Returns:
        str: 'axial', 'equatorial', or 'unknown'
        """
        for oct_key in [oct1_key, oct2_key]:
            if oct_key in octahedra_data:
                oct_data = octahedra_data[oct_key]
                axial_indices = oct_data['ligand_atoms']['axial_global_indices']
                equatorial_indices = oct_data['ligand_atoms']['equatorial_global_indices']
                
                if shared_atom_idx in axial_indices:
                    return 'axial'
                elif shared_atom_idx in equatorial_indices:
                    return 'equatorial'
        
        return 'unknown'
    
    def _identify_terminal_axial_atoms(self, octahedra_data, shared_atoms):
        """
        Identify axial atoms that are not connected to other octahedral centers.
        Uses PBC-aware distance calculations through the geometry calculator.
        
        Parameters:
        octahedra_data (dict): Dictionary of octahedra data
        shared_atoms (dict): Shared atoms data from connectivity analysis
        
        Returns:
        dict: Terminal axial atoms data
        """
        terminal_axial = {}
        
        for oct_key, oct_data in octahedra_data.items():
            axial_indices = oct_data['ligand_atoms']['axial_global_indices']
            
            terminal_atoms = []
            connected_atoms = []
            
            for axial_idx in axial_indices:
                is_terminal = True
                
                # Check if this axial atom is shared with other octahedra
                if axial_idx in shared_atoms:
                    sharing_info = shared_atoms[axial_idx]
                    # Check if it's shared with other octahedra as ligand
                    other_octahedra = [
                        info['octahedron'] for info in sharing_info 
                        if info['octahedron'] != oct_key and info['role'] == 'ligand'
                    ]
                    
                    if other_octahedra:
                        is_terminal = False
                        connected_atoms.append({
                            'atom_index': int(axial_idx),
                            'connected_to_octahedra': other_octahedra
                        })
                
                if is_terminal:
                    terminal_atoms.append({
                        'atom_index': int(axial_idx),
                        'atom_type': 'terminal_axial'
                    })
            
            # Only include octahedra that have terminal or connected axial atoms
            if terminal_atoms or connected_atoms:
                terminal_axial[oct_key] = {
                    'terminal_axial_atoms': terminal_atoms,
                    'connected_axial_atoms': connected_atoms
                }
        
        return terminal_axial
    
    def _analyze_sharing_types(self, shared_atoms, octahedra_data, all_coords, all_symbols):
        """
        Analyze the types of sharing between octahedra.
        
        Returns:
        dict: Analysis of sharing types (corner, edge, face sharing)
        """
        sharing_types = {
            "corner_sharing": [],  # 1 shared atom
            "edge_sharing": [],    # 2 shared atoms
            "face_sharing": [],    # 3+ shared atoms
            "summary": {}
        }
        
        # Group shared atoms by octahedra pairs
        octahedra_pairs = {}
        for atom_idx, sharing_info in shared_atoms.items():
            if len(sharing_info) == 2:  # Only consider pairs
                oct1 = sharing_info[0]["octahedron"]
                oct2 = sharing_info[1]["octahedron"]
                
                # Create a sorted tuple for consistent pair identification
                pair = tuple(sorted([oct1, oct2]))
                
                if pair not in octahedra_pairs:
                    octahedra_pairs[pair] = []
                octahedra_pairs[pair].append(atom_idx)
        
        # Classify sharing types
        for pair, shared_atom_indices in octahedra_pairs.items():
            num_shared = len(shared_atom_indices)
            sharing_info = {
                "octahedron_1": pair[0],
                "octahedron_2": pair[1],
                "shared_atoms": shared_atom_indices,
                "num_shared_atoms": num_shared,
                "shared_atom_symbols": [all_symbols[idx] for idx in shared_atom_indices]
            }
            
            if num_shared == 1:
                sharing_types["corner_sharing"].append(sharing_info)
            elif num_shared == 2:
                sharing_types["edge_sharing"].append(sharing_info)
            elif num_shared >= 3:
                sharing_types["face_sharing"].append(sharing_info)
        
        # Summary statistics
        sharing_types["summary"] = {
            "total_corner_sharing_pairs": len(sharing_types["corner_sharing"]),
            "total_edge_sharing_pairs": len(sharing_types["edge_sharing"]),
            "total_face_sharing_pairs": len(sharing_types["face_sharing"]),
            "total_connected_pairs": len(octahedra_pairs)
        }
        
        return sharing_types
    
    def _calculate_network_statistics(self, octahedra_connections, shared_atoms):
        """
        Calculate network topology statistics.
        
        Returns:
        dict: Network statistics
        """
        # Count connections per octahedron
        connections_per_octahedron = {
            oct_key: len(connections) 
            for oct_key, connections in octahedra_connections.items()
        }
        
        # Calculate degree distribution
        degrees = list(connections_per_octahedron.values())
        
        statistics = {
            "total_octahedra": len(octahedra_connections),
            "total_shared_atoms": len(shared_atoms),
            "connections_per_octahedron": connections_per_octahedron,
            "average_connectivity": float(np.mean(degrees)) if degrees else 0.0,
            "max_connectivity": int(max(degrees)) if degrees else 0,
            "min_connectivity": int(min(degrees)) if degrees else 0,
            "connectivity_variance": float(np.var(degrees)) if degrees else 0.0
        }
        
        # Find most and least connected octahedra
        if degrees:
            max_connections = max(degrees)
            min_connections = min(degrees)
            
            statistics["most_connected_octahedra"] = [
                oct_key for oct_key, count in connections_per_octahedron.items() 
                if count == max_connections
            ]
            
            statistics["least_connected_octahedra"] = [
                oct_key for oct_key, count in connections_per_octahedron.items() 
                if count == min_connections
            ]
        
        return statistics
    
    def get_connectivity_summary(self, connectivity_analysis):
        """
        Generate a human-readable summary of connectivity analysis.
        
        Parameters:
        connectivity_analysis: Result from analyze_octahedra_connectivity
        
        Returns:
        str: Formatted summary
        """
        shared_atoms = connectivity_analysis.get('shared_atoms', {})
        sharing_types = connectivity_analysis.get('sharing_types', {})
        network_stats = connectivity_analysis.get('network_statistics', {})
        terminal_axial = connectivity_analysis.get('terminal_axial_atoms', {})
        
        summary_lines = []
        summary_lines.append("=== CONNECTIVITY ANALYSIS SUMMARY ===")
        
        # Overall statistics
        summary_lines.append(f"Total shared atoms: {len(shared_atoms)}")
        summary_lines.append(f"Total octahedra: {network_stats.get('total_octahedra', 0)}")
        summary_lines.append(f"Average connectivity: {network_stats.get('average_connectivity', 0):.2f}")
        
        # Terminal axial statistics
        total_terminal = sum(len(data['terminal_axial_atoms']) for data in terminal_axial.values())
        total_connected_axial = sum(len(data['connected_axial_atoms']) for data in terminal_axial.values())
        summary_lines.append(f"Terminal axial atoms: {total_terminal}")
        summary_lines.append(f"Connected axial atoms: {total_connected_axial}")
        
        # Sharing types
        sharing_summary = sharing_types.get('summary', {})
        summary_lines.append(f"\nSharing Types:")
        summary_lines.append(f"  Corner sharing pairs: {sharing_summary.get('total_corner_sharing_pairs', 0)}")
        summary_lines.append(f"  Edge sharing pairs: {sharing_summary.get('total_edge_sharing_pairs', 0)}")
        summary_lines.append(f"  Face sharing pairs: {sharing_summary.get('total_face_sharing_pairs', 0)}")
        
        # Most/least connected
        most_connected = network_stats.get('most_connected_octahedra', [])
        least_connected = network_stats.get('least_connected_octahedra', [])
        
        if most_connected:
            summary_lines.append(f"\nMost connected: {', '.join(most_connected)} ({network_stats.get('max_connectivity', 0)} connections)")
        if least_connected:
            summary_lines.append(f"Least connected: {', '.join(least_connected)} ({network_stats.get('min_connectivity', 0)} connections)")
        
        # Terminal axial details
        if terminal_axial:
            summary_lines.append(f"\nTerminal Axial Atoms:")
            for oct_key, data in terminal_axial.items():
                if data['terminal_axial_atoms']:
                    terminal_atoms = [str(atom['atom_index']) for atom in data['terminal_axial_atoms']]
                    summary_lines.append(f"  {oct_key}: {', '.join(terminal_atoms)}")
        
        return '\n'.join(summary_lines) 