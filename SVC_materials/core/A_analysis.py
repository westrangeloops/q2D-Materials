"""
A-site atom identification for perovskite structures.

This module identifies A-site cations (like methylammonium, Cs, etc.) that are located
between octahedral layers and analyzes their molecular structure and connectivity.
"""

import numpy as np
from collections import defaultdict
from ase import Atoms
from ase.neighborlist import neighbor_list
from ase.data import covalent_radii, atomic_numbers


class ASiteIdentifier:
    """
    Identifies A-site atoms and molecules in perovskite structures.
    """
    
    def __init__(self, atoms=None, geometry_calculator=None, bond_tolerance=0.3):
        """
        Initialize A-site identifier.
        
        Parameters:
        atoms: ASE Atoms object
        geometry_calculator: GeometryCalculator instance (optional)
        bond_tolerance: Additional tolerance for bond detection (Angstrom)
        """
        self.atoms = atoms
        self.geometry_calc = geometry_calculator
        self.bond_tolerance = bond_tolerance
        self.a_site_analysis = {}
        
        # Common A-site elements and molecular fragments
        self.common_a_site_elements = ['Cs', 'Rb', 'K', 'Na', 'Li']
        self.organic_elements = ['C', 'H', 'O', 'N']
        
    def identify_a_site_atoms(self, octahedra_data, layers_analysis, all_coords, all_symbols):
        """
        Identify A-site atoms and molecules between octahedral layers.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        layers_analysis: Layer analysis results
        all_coords: All atomic coordinates
        all_symbols: All atomic symbols
        
        Returns:
        dict: A-site analysis with molecular structures and connectivity
        """
        # Step 1: Identify all atoms that are part of octahedra
        octahedral_atoms = self._get_octahedral_atoms(octahedra_data)
        
        # Step 2: Find non-octahedral atoms (potential A-site candidates)
        non_octahedral_atoms = self._find_non_octahedral_atoms(
            octahedral_atoms, all_coords, all_symbols
        )
        
        # Step 3: Identify molecules and their connectivity
        molecules = self._identify_molecules(non_octahedral_atoms, all_coords, all_symbols)
        
        # Step 4: Assign molecules to interlayer regions
        interlayer_molecules = self._assign_molecules_to_layers(
            molecules, layers_analysis, octahedra_data
        )
        
        # Step 5: Analyze molecular structures
        molecular_analysis = self._analyze_molecular_structures(
            interlayer_molecules, all_coords, all_symbols
        )
        
        self.a_site_analysis = {
            "octahedral_atoms": octahedral_atoms,
            "non_octahedral_atoms": non_octahedral_atoms,
            "molecules": molecules,
            "interlayer_molecules": interlayer_molecules,
            "molecular_analysis": molecular_analysis,
            "summary": self._create_summary(molecular_analysis)
        }
        
        return self.a_site_analysis
    
    def _get_octahedral_atoms(self, octahedra_data):
        """
        Get all atom indices that are part of octahedra.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        
        Returns:
        set: Set of atom indices that are part of octahedra
        """
        octahedral_atoms = set()
        
        for oct_key, oct_data in octahedra_data.items():
            # Add central atom
            central_idx = oct_data['central_atom']['global_index']
            octahedral_atoms.add(central_idx)
            
            # Add ligand atoms
            ligand_indices = oct_data['ligand_atoms']['all_ligand_global_indices']
            octahedral_atoms.update(ligand_indices)
        
        return octahedral_atoms
    
    def _find_non_octahedral_atoms(self, octahedral_atoms, all_coords, all_symbols):
        """
        Find atoms that are not part of any octahedron.
        
        Parameters:
        octahedral_atoms: Set of octahedral atom indices
        all_coords: All atomic coordinates
        all_symbols: All atomic symbols
        
        Returns:
        dict: Non-octahedral atoms with their properties
        """
        non_octahedral = {}
        
        for i, (coord, symbol) in enumerate(zip(all_coords, all_symbols)):
            if i not in octahedral_atoms:
                non_octahedral[i] = {
                    'global_index': i,
                    'symbol': symbol,
                    'coordinates': {
                        'x': float(coord[0]),
                        'y': float(coord[1]),
                        'z': float(coord[2])
                    },
                    'is_organic': symbol in self.organic_elements,
                    'is_common_a_site': symbol in self.common_a_site_elements
                }
        
        return non_octahedral
    
    def _identify_molecules(self, non_octahedral_atoms, all_coords, all_symbols):
        """
        Identify molecular structures among non-octahedral atoms using ASE.
        
        Parameters:
        non_octahedral_atoms: Dictionary of non-octahedral atoms
        all_coords: All atomic coordinates
        all_symbols: All atomic symbols
        
        Returns:
        dict: Identified molecules with connectivity
        """
        if not non_octahedral_atoms:
            return {}
        
        # Create ASE atoms object for just the non-octahedral atoms
        non_oct_indices = list(non_octahedral_atoms.keys())
        non_oct_coords = [all_coords[i] for i in non_oct_indices]
        non_oct_symbols = [all_symbols[i] for i in non_oct_indices]
        
        # Create temporary atoms object for bond analysis
        temp_atoms = Atoms(symbols=non_oct_symbols, positions=non_oct_coords)
        
        # Use ASE neighbor list to find bonds
        bonds = self._find_bonds_ase(temp_atoms)
        
        # Group atoms into molecules based on connectivity
        molecules = self._group_atoms_into_molecules(bonds, non_oct_indices, non_octahedral_atoms)
        
        return molecules
    
    def _find_bonds_ase(self, atoms):
        """
        Find bonds using ASE neighbor list with covalent radii.
        
        Parameters:
        atoms: ASE Atoms object
        
        Returns:
        list: List of bond tuples (i, j)
        """
        # Get neighbor list based on covalent radii + tolerance
        cutoffs = [covalent_radii[atomic_numbers[symbol]] + self.bond_tolerance 
                  for symbol in atoms.get_chemical_symbols()]
        
        i_indices, j_indices = neighbor_list('ij', atoms, cutoffs)
        
        # Create list of unique bonds
        bonds = []
        for i, j in zip(i_indices, j_indices):
            if i < j:  # Avoid duplicates
                bonds.append((i, j))
        
        return bonds
    
    def _group_atoms_into_molecules(self, bonds, non_oct_indices, non_octahedral_atoms):
        """
        Group atoms into molecules based on bond connectivity.
        
        Parameters:
        bonds: List of bond tuples (local indices)
        non_oct_indices: List of global indices for non-octahedral atoms
        non_octahedral_atoms: Dictionary of non-octahedral atom data
        
        Returns:
        dict: Molecules with their constituent atoms
        """
        # Create adjacency list
        adjacency = defaultdict(set)
        for i, j in bonds:
            adjacency[i].add(j)
            adjacency[j].add(i)
        
        # Find connected components (molecules)
        visited = set()
        molecules = {}
        molecule_id = 1
        
        for local_idx in range(len(non_oct_indices)):
            if local_idx not in visited:
                # Find all atoms in this molecule using DFS
                molecule_atoms = []
                stack = [local_idx]
                
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        global_idx = non_oct_indices[current]
                        molecule_atoms.append(global_idx)
                        
                        # Add neighbors to stack
                        for neighbor in adjacency[current]:
                            if neighbor not in visited:
                                stack.append(neighbor)
                
                # Create molecule data
                if molecule_atoms:
                    molecule_key = f"molecule_{molecule_id}"
                    molecules[molecule_key] = {
                        "molecule_id": molecule_id,
                        "atom_indices": sorted(molecule_atoms),
                        "atom_count": len(molecule_atoms),
                        "atom_symbols": [non_octahedral_atoms[idx]['symbol'] for idx in molecule_atoms],
                        "molecular_formula": self._get_molecular_formula(
                            [non_octahedral_atoms[idx]['symbol'] for idx in molecule_atoms]
                        ),
                        "is_organic": any(non_octahedral_atoms[idx]['is_organic'] for idx in molecule_atoms),
                        "is_inorganic_cation": all(non_octahedral_atoms[idx]['is_common_a_site'] for idx in molecule_atoms),
                        "centroid": self._calculate_centroid([non_octahedral_atoms[idx]['coordinates'] for idx in molecule_atoms])
                    }
                    molecule_id += 1
        
        return molecules
    
    def _get_molecular_formula(self, symbols):
        """
        Get molecular formula from list of atomic symbols.
        
        Parameters:
        symbols: List of atomic symbols
        
        Returns:
        str: Molecular formula (e.g., "CH3NH3", "Cs")
        """
        symbol_counts = defaultdict(int)
        for symbol in symbols:
            symbol_counts[symbol] += 1
        
        # Sort by common convention: C, H, N, O, then alphabetically
        order = ['C', 'H', 'N', 'O']
        formula_parts = []
        
        # Add elements in preferred order
        for element in order:
            if element in symbol_counts:
                count = symbol_counts[element]
                if count == 1:
                    formula_parts.append(element)
                else:
                    formula_parts.append(f"{element}{count}")
                del symbol_counts[element]
        
        # Add remaining elements alphabetically
        for element in sorted(symbol_counts.keys()):
            count = symbol_counts[element]
            if count == 1:
                formula_parts.append(element)
            else:
                formula_parts.append(f"{element}{count}")
        
        return ''.join(formula_parts)
    
    def _calculate_centroid(self, coordinates):
        """
        Calculate centroid of a set of coordinates.
        
        Parameters:
        coordinates: List of coordinate dictionaries
        
        Returns:
        dict: Centroid coordinates
        """
        if not coordinates:
            return {'x': 0.0, 'y': 0.0, 'z': 0.0}
        
        x_sum = sum(coord['x'] for coord in coordinates)
        y_sum = sum(coord['y'] for coord in coordinates)
        z_sum = sum(coord['z'] for coord in coordinates)
        n = len(coordinates)
        
        return {
            'x': float(x_sum / n),
            'y': float(y_sum / n),
            'z': float(z_sum / n)
        }
    
    def _assign_molecules_to_layers(self, molecules, layers_analysis, octahedra_data):
        """
        Assign molecules to interlayer regions based on their position relative to octahedral layers.
        
        Parameters:
        molecules: Dictionary of molecules
        layers_analysis: Layer analysis results
        octahedra_data: Octahedra data
        
        Returns:
        dict: Molecules assigned to interlayer regions
        """
        if not layers_analysis or not molecules:
            return {}
        
        # Get layer z-coordinates
        layer_z_ranges = self._get_layer_z_ranges(layers_analysis, octahedra_data)
        
        interlayer_molecules = {}
        
        for mol_key, mol_data in molecules.items():
            mol_z = mol_data['centroid']['z']
            
            # Find which interlayer region this molecule belongs to
            interlayer_region = self._find_interlayer_region(mol_z, layer_z_ranges)
            
            if interlayer_region not in interlayer_molecules:
                interlayer_molecules[interlayer_region] = []
            
            mol_data_copy = mol_data.copy()
            mol_data_copy['interlayer_region'] = interlayer_region
            interlayer_molecules[interlayer_region].append(mol_data_copy)
        
        return interlayer_molecules
    
    def _get_layer_z_ranges(self, layers_analysis, octahedra_data):
        """
        Get z-coordinate ranges for each octahedral layer.
        
        Parameters:
        layers_analysis: Layer analysis results
        octahedra_data: Octahedra data
        
        Returns:
        dict: Layer z-ranges
        """
        layer_z_ranges = {}
        layers = layers_analysis.get('layers', {})
        
        for layer_key, layer_data in layers.items():
            octahedra_keys = layer_data['octahedra_keys']
            z_coords = []
            
            for oct_key in octahedra_keys:
                if oct_key in octahedra_data:
                    z_coord = octahedra_data[oct_key]['central_atom']['coordinates']['z']
                    z_coords.append(z_coord)
            
            if z_coords:
                layer_z_ranges[layer_key] = {
                    'min_z': min(z_coords),
                    'max_z': max(z_coords),
                    'mean_z': np.mean(z_coords)
                }
        
        return layer_z_ranges
    
    def _find_interlayer_region(self, mol_z, layer_z_ranges):
        """
        Find which interlayer region a molecule belongs to based on its z-coordinate.
        
        Parameters:
        mol_z: Molecule z-coordinate
        layer_z_ranges: Dictionary of layer z-ranges
        
        Returns:
        str: Interlayer region identifier
        """
        # Sort layers by mean z-coordinate
        sorted_layers = sorted(layer_z_ranges.items(), key=lambda x: x[1]['mean_z'])
        
        # Check if molecule is between layers
        for i in range(len(sorted_layers) - 1):
            current_layer = sorted_layers[i]
            next_layer = sorted_layers[i + 1]
            
            current_max_z = current_layer[1]['max_z']
            next_min_z = next_layer[1]['min_z']
            
            if current_max_z <= mol_z <= next_min_z:
                return f"interlayer_{current_layer[0]}_to_{next_layer[0]}"
        
        # Check if molecule is above the topmost layer
        if sorted_layers and mol_z > sorted_layers[-1][1]['max_z']:
            return f"above_{sorted_layers[-1][0]}"
        
        # Check if molecule is below the bottommost layer
        if sorted_layers and mol_z < sorted_layers[0][1]['min_z']:
            return f"below_{sorted_layers[0][0]}"
        
        # Default: within a layer (shouldn't happen for A-site molecules)
        return "within_layer"
    
    def _analyze_molecular_structures(self, interlayer_molecules, all_coords, all_symbols):
        """
        Analyze molecular structures including bonds, angles, and dihedrals.
        
        Parameters:
        interlayer_molecules: Dictionary of molecules by interlayer region
        all_coords: All atomic coordinates
        all_symbols: All atomic symbols
        
        Returns:
        dict: Detailed molecular analysis
        """
        molecular_analysis = {}
        
        for region_key, molecules in interlayer_molecules.items():
            region_analysis = {
                "region": region_key,
                "molecule_count": len(molecules),
                "molecules": {}
            }
            
            for mol_data in molecules:
                mol_key = f"molecule_{mol_data['molecule_id']}"
                atom_indices = mol_data['atom_indices']
                
                # Create ASE atoms object for this molecule
                mol_coords = [all_coords[i] for i in atom_indices]
                mol_symbols = [all_symbols[i] for i in atom_indices]
                mol_atoms = Atoms(symbols=mol_symbols, positions=mol_coords)
                
                # Analyze molecular structure
                mol_analysis = self._analyze_single_molecule(mol_atoms, atom_indices, mol_data)
                region_analysis["molecules"][mol_key] = mol_analysis
            
            molecular_analysis[region_key] = region_analysis
        
        return molecular_analysis
    
    def _analyze_single_molecule(self, mol_atoms, global_indices, mol_data):
        """
        Analyze a single molecule's structure including connectivity.
        
        Parameters:
        mol_atoms: ASE Atoms object for the molecule
        global_indices: Global indices of atoms in the molecule
        mol_data: Molecule data dictionary
        
        Returns:
        dict: Detailed molecular analysis
        """
        # Find bonds using ASE
        bonds = self._find_bonds_ase(mol_atoms)
        
        # Calculate bond distances and angles
        bond_analysis = self._analyze_bonds(mol_atoms, bonds)
        angle_analysis = self._analyze_angles(mol_atoms, bonds)
        dihedral_analysis = self._analyze_dihedrals(mol_atoms, bonds)
        
        return {
            "molecular_formula": mol_data['molecular_formula'],
            "atom_count": mol_data['atom_count'],
            "atom_symbols": mol_data['atom_symbols'],
            "global_atom_indices": global_indices,
            "centroid": mol_data['centroid'],
            "is_organic": mol_data['is_organic'],
            "is_inorganic_cation": mol_data['is_inorganic_cation'],
            "connectivity": {
                "bonds": bond_analysis,
                "angles": angle_analysis,
                "dihedrals": dihedral_analysis
            },
            "structural_classification": self._classify_molecule_structure(mol_data, bond_analysis)
        }
    
    def _analyze_bonds(self, mol_atoms, bonds):
        """
        Analyze bond distances in a molecule.
        
        Parameters:
        mol_atoms: ASE Atoms object
        bonds: List of bond tuples
        
        Returns:
        dict: Bond analysis
        """
        positions = mol_atoms.get_positions()
        symbols = mol_atoms.get_chemical_symbols()
        
        bond_analysis = {
            "bond_count": len(bonds),
            "bonds": []
        }
        
        for i, j in bonds:
            distance = np.linalg.norm(positions[i] - positions[j])
            bond_info = {
                "atom_1": {"index": i, "symbol": symbols[i]},
                "atom_2": {"index": j, "symbol": symbols[j]},
                "distance": float(distance),
                "bond_type": f"{symbols[i]}-{symbols[j]}"
            }
            bond_analysis["bonds"].append(bond_info)
        
        # Calculate bond statistics
        if bonds:
            distances = [bond["distance"] for bond in bond_analysis["bonds"]]
            bond_analysis["statistics"] = {
                "mean_distance": float(np.mean(distances)),
                "std_distance": float(np.std(distances)),
                "min_distance": float(np.min(distances)),
                "max_distance": float(np.max(distances))
            }
        
        return bond_analysis
    
    def _analyze_angles(self, mol_atoms, bonds):
        """
        Analyze bond angles in a molecule.
        
        Parameters:
        mol_atoms: ASE Atoms object
        bonds: List of bond tuples
        
        Returns:
        dict: Angle analysis
        """
        if len(bonds) < 2:
            return {"angle_count": 0, "angles": []}
        
        positions = mol_atoms.get_positions()
        symbols = mol_atoms.get_chemical_symbols()
        
        # Create adjacency list
        adjacency = defaultdict(list)
        for i, j in bonds:
            adjacency[i].append(j)
            adjacency[j].append(i)
        
        angles = []
        
        # Find all angles (three connected atoms)
        for center_atom in adjacency:
            neighbors = adjacency[center_atom]
            if len(neighbors) >= 2:
                for i in range(len(neighbors)):
                    for j in range(i + 1, len(neighbors)):
                        atom1 = neighbors[i]
                        atom3 = neighbors[j]
                        
                        # Calculate angle
                        vec1 = positions[atom1] - positions[center_atom]
                        vec2 = positions[atom3] - positions[center_atom]
                        
                        cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Handle numerical errors
                        angle_rad = np.arccos(cos_angle)
                        angle_deg = np.degrees(angle_rad)
                        
                        angle_info = {
                            "atom_1": {"index": atom1, "symbol": symbols[atom1]},
                            "center_atom": {"index": center_atom, "symbol": symbols[center_atom]},
                            "atom_3": {"index": atom3, "symbol": symbols[atom3]},
                            "angle_degrees": float(angle_deg),
                            "angle_type": f"{symbols[atom1]}-{symbols[center_atom]}-{symbols[atom3]}"
                        }
                        angles.append(angle_info)
        
        angle_analysis = {
            "angle_count": len(angles),
            "angles": angles
        }
        
        # Calculate angle statistics
        if angles:
            angle_values = [angle["angle_degrees"] for angle in angles]
            angle_analysis["statistics"] = {
                "mean_angle": float(np.mean(angle_values)),
                "std_angle": float(np.std(angle_values)),
                "min_angle": float(np.min(angle_values)),
                "max_angle": float(np.max(angle_values))
            }
        
        return angle_analysis
    
    def _analyze_dihedrals(self, mol_atoms, bonds):
        """
        Analyze dihedral angles in a molecule.
        
        Parameters:
        mol_atoms: ASE Atoms object
        bonds: List of bond tuples
        
        Returns:
        dict: Dihedral analysis
        """
        if len(bonds) < 3:
            return {"dihedral_count": 0, "dihedrals": []}
        
        positions = mol_atoms.get_positions()
        symbols = mol_atoms.get_chemical_symbols()
        
        # Create adjacency list
        adjacency = defaultdict(list)
        for i, j in bonds:
            adjacency[i].append(j)
            adjacency[j].append(i)
        
        dihedrals = []
        
        # Find all dihedral angles (four connected atoms in sequence)
        for bond in bonds:
            atom2, atom3 = bond
            
            for atom1 in adjacency[atom2]:
                if atom1 == atom3:
                    continue
                for atom4 in adjacency[atom3]:
                    if atom4 == atom2 or atom4 == atom1:
                        continue
                    
                    # Calculate dihedral angle
                    dihedral_angle = self._calculate_dihedral_angle(
                        positions[atom1], positions[atom2], 
                        positions[atom3], positions[atom4]
                    )
                    
                    dihedral_info = {
                        "atom_1": {"index": atom1, "symbol": symbols[atom1]},
                        "atom_2": {"index": atom2, "symbol": symbols[atom2]},
                        "atom_3": {"index": atom3, "symbol": symbols[atom3]},
                        "atom_4": {"index": atom4, "symbol": symbols[atom4]},
                        "dihedral_degrees": float(dihedral_angle),
                        "dihedral_type": f"{symbols[atom1]}-{symbols[atom2]}-{symbols[atom3]}-{symbols[atom4]}"
                    }
                    dihedrals.append(dihedral_info)
        
        dihedral_analysis = {
            "dihedral_count": len(dihedrals),
            "dihedrals": dihedrals
        }
        
        # Calculate dihedral statistics
        if dihedrals:
            dihedral_values = [abs(dihedral["dihedral_degrees"]) for dihedral in dihedrals]
            dihedral_analysis["statistics"] = {
                "mean_dihedral": float(np.mean(dihedral_values)),
                "std_dihedral": float(np.std(dihedral_values)),
                "min_dihedral": float(np.min(dihedral_values)),
                "max_dihedral": float(np.max(dihedral_values))
            }
        
        return dihedral_analysis
    
    def _calculate_dihedral_angle(self, p1, p2, p3, p4):
        """
        Calculate dihedral angle between four points.
        
        Parameters:
        p1, p2, p3, p4: Position vectors
        
        Returns:
        float: Dihedral angle in degrees
        """
        # Calculate vectors
        v1 = p2 - p1
        v2 = p3 - p2
        v3 = p4 - p3
        
        # Calculate normal vectors to the planes
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        
        # Normalize normal vectors
        n1_norm = np.linalg.norm(n1)
        n2_norm = np.linalg.norm(n2)
        
        if n1_norm < 1e-6 or n2_norm < 1e-6:
            return 0.0  # Degenerate case
        
        n1 = n1 / n1_norm
        n2 = n2 / n2_norm
        
        # Calculate dihedral angle
        cos_angle = np.dot(n1, n2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)
        
        # Determine sign of angle
        if np.dot(np.cross(n1, n2), v2) < 0:
            angle_rad = -angle_rad
        
        return np.degrees(angle_rad)