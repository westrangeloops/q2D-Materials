from ..utils.file_handlers import vasp_load, save_vasp
from ase.io import read, write
from ase.visualize import view
import os
import numpy as np
from ase.neighborlist import NeighborList
from itertools import combinations

class q2D_analysis:
    def __init__(self, B, X, crystal):
        self.path = '/'.join(crystal.split('/')[0:-1])
        self.name = crystal.split('/')[-1]
        self.B = B
        self.X = X
        self.perovskite_df, self.box = vasp_load(crystal)
    
    def _find_planes(self, element_type='B'):
        """Helper function to find the planes of the perovskite
        
        Args:
            element_type (str): 'B' for metal atoms or 'X' for halide atoms
        """
        crystal_df = self.perovskite_df
        element = self.B if element_type == 'B' else self.X
        
        # Find all atoms of the specified element
        element_atoms = crystal_df.query("Element == @element").sort_values(by='Z')
        
        if len(element_atoms) == 0:
            print(f'No {element} atoms found in the structure')
            return None, None, None
            
        # Group atoms into planes based on Z coordinate
        planes = []
        current_plane = []
        current_z = None
        
        for _, atom in element_atoms.iterrows():
            z = atom['Z']
            
            # If this is the first atom or it's close to the current plane
            if current_z is None or abs(z - current_z) <= 0.5:
                current_plane.append(atom)
                current_z = z
            else:
                # Start a new plane
                if current_plane:
                    planes.append(current_plane)
                current_plane = [atom]
                current_z = z
        
        # Add the last plane
        if current_plane:
            planes.append(current_plane)
            
        # Calculate average Z for each plane
        plane_z_values = [np.mean([atom['Z'] for atom in plane]) for plane in planes]
        
        if len(plane_z_values) > 1:
            # Sort planes by Z coordinate
            sorted_planes = sorted(zip(plane_z_values, planes), key=lambda x: x[0])
            plane_z_values = [z for z, _ in sorted_planes]
            
            # For B atoms, we expect exactly 2 planes
            if element_type == 'B':
                if len(plane_z_values) != 2:
                    print(f"Warning: Found {len(plane_z_values)} B planes, expected 2")
                    # If we have more than 2 planes, use the outermost ones
                    lower_plane = plane_z_values[0]
                    upper_plane = plane_z_values[-1]
                else:
                    lower_plane = plane_z_values[0]
                    upper_plane = plane_z_values[1]
                
                print(f"\nPlane detection for {element} atoms:")
                print(f"Found {len(planes)} planes, using main planes at:")
                print(f"Lower plane at Z = {lower_plane:.3f} Å")
                print(f"Upper plane at Z = {upper_plane:.3f} Å")
                
                return lower_plane, upper_plane, True
            else:
                # For X atoms, find the two main planes
                if len(plane_z_values) > 2:
                    # Find the largest gap between planes
                    gaps = [plane_z_values[i+1] - plane_z_values[i] for i in range(len(plane_z_values)-1)]
                    max_gap_idx = np.argmax(gaps)
                    
                    # Use the planes on either side of the largest gap
                    lower_plane = plane_z_values[max_gap_idx]
                    upper_plane = plane_z_values[max_gap_idx + 1]
                else:
                    lower_plane = plane_z_values[0]
                    upper_plane = plane_z_values[1]
                
                print(f"\nPlane detection for {element} atoms:")
                print(f"Found {len(planes)} planes, using main planes at:")
                print(f"Lower plane at Z = {lower_plane:.3f} Å")
                print(f"Upper plane at Z = {upper_plane:.3f} Å")
                
                return lower_plane, upper_plane, True
            
        elif len(plane_z_values) == 1:
            # For B atoms in single slab case, use the same plane for both
            if element_type == 'B':
                print(f"Note: Single slab case - using same plane for both lower and upper")
                plane_z = plane_z_values[0]
                return plane_z, plane_z, True
            return plane_z_values[0], None, False
        else:
            print(f'Could not identify planes for {element} atoms')
            return None, None, None

    def _get_plane_equation(self, plane_z):
        """Calculate the equation of a plane at a given Z coordinate"""
        # For a horizontal plane, the equation is z = plane_z
        # In standard form: 0x + 0y + 1z - plane_z = 0
        return np.array([0, 0, 1, -plane_z])

    def _point_to_plane_distance(self, point, plane_eq):
        """Calculate the absolute distance from a point to a plane"""
        # plane_eq is [a, b, c, d] for plane ax + by + cz + d = 0
        # point is [x, y, z]
        a, b, c, d = plane_eq
        x, y, z = point
        # Use absolute value for the numerator to get positive distance
        numerator = abs(a*x + b*y + c*z + d)
        denominator = np.sqrt(a*a + b*b + c*c)
        return abs(numerator / denominator)  # Ensure positive distance

    def calculate_n_penetration(self):
        """Calculate the penetration depth of the 4 Nitrogen atoms from the organic molecule into both upper and lower perovskite layers"""
        # Find the X (halide) planes instead of B planes
        b_down_plane, b_up_plane, has_two_planes = self._find_planes(element_type='X')
        
        if b_down_plane is None:
            print("Error: Could not find halide planes")
            return None

        if not has_two_planes:
            print("Error: Structure must have two halide planes for penetration analysis")
            return None

        # Get all Nitrogen atoms
        n_atoms = self.perovskite_df[self.perovskite_df['Element'] == 'N']
        if len(n_atoms) == 0:
            print("Error: No Nitrogen atoms found in the structure")
            return None

        # Sort N atoms by Z coordinate
        n_atoms = n_atoms.sort_values('Z')
        
        # Find the middle Z value to separate upper and lower N atoms
        middle_z = (b_down_plane + b_up_plane) / 2
        
        # Split N atoms into upper and lower groups based on Z coordinate
        lower_n_atoms = n_atoms[n_atoms['Z'] < middle_z]
        upper_n_atoms = n_atoms[n_atoms['Z'] >= middle_z]
        
        # If we don't have exactly 2 atoms in each group, try to redistribute
        if len(lower_n_atoms) != 2 or len(upper_n_atoms) != 2:
            print(f"Warning: Uneven distribution of N atoms. Attempting to redistribute...")
            # Sort all N atoms by Z coordinate
            all_n_atoms = n_atoms.sort_values('Z')
            
            # Take the 2 lowest Z atoms for lower plane and 2 highest Z atoms for upper plane
            lower_n_atoms = all_n_atoms.head(2)
            upper_n_atoms = all_n_atoms.tail(2)
            
            print(f"Redistributed: {len(lower_n_atoms)} lower N atoms and {len(upper_n_atoms)} upper N atoms")

        # Calculate distances to both planes
        lower_plane_eq = self._get_plane_equation(b_down_plane)
        upper_plane_eq = self._get_plane_equation(b_up_plane)
        
        # Calculate distances for lower plane N atoms
        lower_distances = []
        lower_atoms = []
        for _, atom in lower_n_atoms.iterrows():
            point = np.array([atom['X'], atom['Y'], atom['Z']])
            dist = self._point_to_plane_distance(point, lower_plane_eq)
            lower_distances.append(dist)
            lower_atoms.append(atom)
        
        # Calculate distances for upper plane N atoms
        upper_distances = []
        upper_atoms = []
        for _, atom in upper_n_atoms.iterrows():
            point = np.array([atom['X'], atom['Y'], atom['Z']])
            dist = self._point_to_plane_distance(point, upper_plane_eq)
            upper_distances.append(dist)
            upper_atoms.append(atom)
        
        # Calculate averages
        avg_lower = np.mean(lower_distances)
        avg_upper = np.mean(upper_distances)
        total_avg = (avg_lower + avg_upper) / 2
        
        # Print detailed information
        print(f"\nNitrogen Penetration Analysis (Organic Molecule):")
        print(f"\nLower Plane Penetration:")
        print(f"Number of N atoms: {len(lower_atoms)}")
        print(f"Average penetration: {avg_lower:.3f} Å")
        print("Individual penetrations:")
        for i, (dist, atom) in enumerate(zip(lower_distances, lower_atoms), 1):
            print(f"N atom {i}: {dist:.3f} Å (Z = {atom['Z']:.3f})")
        
        print(f"\nUpper Plane Penetration:")
        print(f"Number of N atoms: {len(upper_atoms)}")
        print(f"Average penetration: {avg_upper:.3f} Å")
        print("Individual penetrations:")
        for i, (dist, atom) in enumerate(zip(upper_distances, upper_atoms), 1):
            print(f"N atom {i}: {dist:.3f} Å (Z = {atom['Z']:.3f})")
        
        print(f"\nTotal average penetration: {total_avg:.3f} Å")
        
        return {
            'lower_penetration': avg_lower,
            'upper_penetration': avg_upper,
            'total_penetration': total_avg,
            'lower_atoms': lower_atoms,
            'upper_atoms': upper_atoms
        }

    def _isolate_spacer(self, order=None):
        """Private method to isolate the molecules between the perovskite planes, excluding B and X atoms"""
        # First get the salt structure using _isolate_salt
        salt_df, salt_box = self._isolate_salt(order=order)
        
        if salt_df is None:
            return None
            
        # Remove B and X atoms from the salt structure
        spacer_df = salt_df[~salt_df['Element'].isin([self.B, self.X])]
        
        print(f"\nSpacer isolation:")
        print(f"Removed {len(salt_df) - len(spacer_df)} B and X atoms")
        print(f"Remaining atoms: {len(spacer_df)}")
        
        return spacer_df, salt_box

    def _isolate_salt(self, order=None):
        """Private method to isolate the molecules plus the 4 halogens between the perovskite planes"""
        # Get all B atoms and sort by Z coordinate
        b_atoms = self.perovskite_df[self.perovskite_df['Element'] == self.B].sort_values('Z')
        
        if len(b_atoms) == 0:
            print(f"No {self.B} atoms found in structure")
            return None
            
        # Group B atoms into planes (atoms within 0.5 Å of each other)
        planes = []
        current_plane = []
        current_z = None
        
        for _, atom in b_atoms.iterrows():
            z = atom['Z']
            if current_z is None or abs(z - current_z) <= 0.5:
                current_plane.append(atom)
                current_z = z
            else:
                if current_plane:
                    planes.append(current_plane)
                current_plane = [atom]
                current_z = z
        
        if current_plane:
            planes.append(current_plane)
            
        # Calculate average Z for each plane
        plane_z_values = [np.mean([atom['Z'] for atom in plane]) for plane in planes]
        plane_z_values.sort()  # Sort by Z coordinate
        
        if len(plane_z_values) >= 2:
            # Case 1: Two or more planes exist
            # Use the two lowest planes
            lower_plane = plane_z_values[0] + 1.0  # Add 1 Å to lower plane
            upper_plane = plane_z_values[1] - 1.0  # Subtract 1 Å from upper plane
            
            print(f"\nSalt isolation (two planes):")
            print(f"Lower plane (adjusted): {lower_plane:.3f} Å")
            print(f"Upper plane (adjusted): {upper_plane:.3f} Å")
            
            # Keep only atoms between adjusted planes
            iso_df = self.perovskite_df.query('Z <= @upper_plane and Z >= @lower_plane')
            
        else:
            # Case 2: Single plane
            lower_plane = plane_z_values[0] + 1.0  # Add 1 Å to the plane
            
            print(f"\nSalt isolation (single plane):")
            print(f"Plane (adjusted): {lower_plane:.3f} Å")
            
            # Keep only atoms above adjusted plane
            iso_df = self.perovskite_df.query('Z >= @lower_plane')
        
        # Update the box to be 10 angstrom in Z
        try:
            iso_df.loc[:, 'Z'] = iso_df['Z'] - iso_df['Z'].min()
            box = self.box.copy()
            box[0][2][2] = iso_df['Z'].sort_values(ascending=False).iloc[0] + 10
            return iso_df, box

        except Exception as e:
            print(f"Error creating the salt: {e}")
            return None

    def save_spacer(self, order=None, name=None):
        """Save the isolated spacer structure as a VASP file"""
        spacer_df, spacer_box = self._isolate_spacer(order=order)
        if spacer_df is not None:
            if name is None:
                name = self.path + '/' + 'spacer_' + self.name
            else:
                # Ensure the file has .vasp extension
                if not name.endswith('.vasp'):
                    name += '.vasp'
                # If no path is provided, save in the same directory as the original
                if '/' not in name:
                    name = self.path + '/' + name
            
            try:
                save_vasp(spacer_df, spacer_box, name, order=order)
                print(f'Spacer structure saved as: {name}')
                return True
            except Exception as e:
                print(f"Error saving spacer structure: {e}")
                return False
        return False

    def save_salt(self, order=None, name=None):
        """Save the isolated salt structure as a VASP file"""
        salt_df, salt_box = self._isolate_salt(order=order)
        if salt_df is not None:
            if name is None:
                name = self.path + '/' + 'salt_' + self.name
            else:
                # Ensure the file has .vasp extension
                if not name.endswith('.vasp'):
                    name += '.vasp'
                # If no path is provided, save in the same directory as the original
                if '/' not in name:
                    name = self.path + '/' + name
            
            try:
                save_vasp(salt_df, salt_box, name, order=order)
                print(f'Salt structure saved as: {name}')
                return True
            except Exception as e:
                print(f"Error saving salt structure: {e}")
                return False
        return False

    def show_spacer(self, order=None):
        """Visualize the isolated spacer structure"""
        spacer_df, spacer_box = self._isolate_spacer(order=order)
        if spacer_df is not None:
            # Save temporarily for visualization
            temp_name = self.path + '/temp_spacer.vasp'
            try:
                save_vasp(spacer_df, spacer_box, temp_name, order=order)
                atoms = read(temp_name)
                view(atoms)
                # Clean up temporary file
                if os.path.exists(temp_name):
                    os.remove(temp_name)
            except Exception as e:
                print(f"Error visualizing spacer: {e}")
                if os.path.exists(temp_name):
                    os.remove(temp_name)

    def show_salt(self, order=None):
        """Visualize the isolated salt structure"""
        salt_df, salt_box = self._isolate_salt(order=order)
        if salt_df is not None:
            # Save temporarily for visualization
            temp_name = self.path + '/temp_salt.vasp'
            try:
                save_vasp(salt_df, salt_box, temp_name, order=order)
                atoms = read(temp_name)
                view(atoms)
                # Clean up temporary file
                if os.path.exists(temp_name):
                    os.remove(temp_name)
            except Exception as e:
                print(f"Error visualizing salt: {e}")
                if os.path.exists(temp_name):
                    os.remove(temp_name)

    def show_original(self):
        """Visualize the original structure"""
        try:
            atoms = read(self.path + '/' + self.name)
            view(atoms)
        except Exception as e:
            print(f"Error visualizing original structure: {e}")


    def analyze_perovskite_structure(self, cutoff_distance=3.5):
        """Analyze the perovskite structure including bond angles, lengths, and distortions
        
        Args:
            cutoff_distance (float): Maximum distance to consider atoms as neighbors (in Å)
            
        Returns:
            dict: Dictionary containing all structural parameters
        """
        try:
            # Convert the structure to ASE Atoms object
            atoms = read(self.path + '/' + self.name)
            print(f"\nAnalyzing perovskite structure:")
            print(f"  Formula: {atoms.get_chemical_formula()}")
            print(f"  Cell dimensions (Å): {np.diag(atoms.cell)}")
            
            # Find the indices of B atoms
            b_atom_indices = [atom.index for atom in atoms if atom.symbol == self.B]
            if not b_atom_indices:
                print(f"ERROR: No '{self.B}' atoms found in the structure.")
                return None
                
            print(f"Found {len(b_atom_indices)} '{self.B}' atom(s).")
            
            # Set up neighbor list
            cutoffs = [cutoff_distance / 2.0] * len(atoms)
            nl = NeighborList(cutoffs, skin=0.3, self_interaction=False, bothways=True)
            nl.update(atoms)
            
            # Initialize storage for all measurements
            structure_data = {
                'inter_octahedral_angles': [],  # Pb-X-Pb angles between octahedra
                'in_plane_angles': [],          # Pb-X-Pb angles in the same plane
                'axial_angles': [],            # Angles between axial bonds
                'equatorial_angles': [],       # Angles between equatorial bonds
                'axial_lengths': [],           # Axial Pb-X bond lengths
                'equatorial_lengths': [],      # Equatorial Pb-X bond lengths
                'out_of_plane_distortions': [], # Deviation from normal direction
                'per_octahedron': {},          # Store data for each octahedron
                'octahedral_variances': []     # Store variance for each octahedron
            }
            
            # Loop through each B atom
            for b_index in b_atom_indices:
                # Get neighbors of the current B atom
                neighbor_indices, offsets = nl.get_neighbors(b_index)
                
                # Filter for X neighbors and calculate bond vectors
                x_neighbors_info = []
                for i, offset in zip(neighbor_indices, offsets):
                    if atoms[i].symbol == self.X:
                        x_pos = atoms.get_positions()[i] + np.dot(offset, atoms.get_cell())
                        b_pos = atoms.get_positions()[b_index]
                        bond_vector = x_pos - b_pos
                        bond_length = np.linalg.norm(bond_vector)
                        x_neighbors_info.append({
                            'index': i,
                            'vector': bond_vector,
                            'length': bond_length,
                            'position': x_pos
                        })
                
                if len(x_neighbors_info) != 6:
                    print(f"  > WARNING: B atom at index {b_index} has {len(x_neighbors_info)} {self.X} neighbors, expected 6.")
                    continue
                
                # Identify axial and equatorial bonds
                # First, find the two bonds that are most parallel to each other (axial)
                max_parallel = -1
                axial_pair = None
                for i, j in combinations(range(len(x_neighbors_info)), 2):
                    v1 = x_neighbors_info[i]['vector']
                    v2 = x_neighbors_info[j]['vector']
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    if abs(cos_theta) > max_parallel:
                        max_parallel = abs(cos_theta)
                        axial_pair = (i, j)
                
                # Separate axial and equatorial bonds
                axial_bonds = [x_neighbors_info[i] for i in axial_pair]
                equatorial_bonds = [x for i, x in enumerate(x_neighbors_info) if i not in axial_pair]
                
                # Calculate axial bond angle
                v1 = axial_bonds[0]['vector']
                v2 = axial_bonds[1]['vector']
                cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                axial_angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
                structure_data['axial_angles'].append(axial_angle)
                
                # Calculate equatorial bond angles for this octahedron
                octahedron_eq_angles = []
                octahedron_axial_angles = []
                print(f"\nOctahedron centered at B atom {b_index}:")
                
                # First identify the axial direction
                axial_direction = np.mean([b['vector'] for b in axial_bonds], axis=0)
                axial_direction = axial_direction / np.linalg.norm(axial_direction)
                
                # Calculate angles between all bonds
                for v1, v2 in combinations(equatorial_bonds, 2):
                    vec1 = v1['vector']
                    vec2 = v2['vector']
                    
                    # Calculate angle between bonds
                    cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    angle = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
                    
                    # Check if this is an axial angle (close to 180°)
                    if angle > 150:  # If angle is close to 180°
                        octahedron_axial_angles.append(angle)
                        print(f"  Axial angle: {angle:.2f}°")
                    else:  # Otherwise it's an equatorial angle
                        octahedron_eq_angles.append(angle)
                        print(f"  Equatorial angle: {angle:.2f}°")
                
                # Calculate variances separately
                ideal_equatorial = 90.0
                ideal_axial = 180.0
                
                eq_deviations = [(angle - ideal_equatorial)**2 for angle in octahedron_eq_angles]
                axial_deviations = [(angle - ideal_axial)**2 for angle in octahedron_axial_angles]
                
                eq_variance = np.mean(eq_deviations) if eq_deviations else 0
                axial_variance = np.mean(axial_deviations) if axial_deviations else 0
                
                structure_data['octahedral_variances'].append(eq_variance)  # Store only equatorial variance
                
                print(f"\n  Equatorial angles (target: 90°):")
                for angle, deviation in zip(octahedron_eq_angles, eq_deviations):
                    print(f"    {angle:.2f}° -> {(angle - ideal_equatorial):.2f}° deviation -> {deviation:.2f}°²")
                print(f"  Equatorial variance: {eq_variance:.2f}°²")
                
                print(f"\n  Axial angles (target: 180°):")
                for angle, deviation in zip(octahedron_axial_angles, axial_deviations):
                    print(f"    {angle:.2f}° -> {(angle - ideal_axial):.2f}° deviation -> {deviation:.2f}°²")
                print(f"  Axial variance: {axial_variance:.2f}°²")
                
                # Detailed distortion analysis
                print("\n  Distortion Pattern Analysis:")
                
                # 1. Axial Distortion Analysis
                if axial_deviations:
                    axial_angles = np.array(octahedron_axial_angles)
                    axial_deviations = np.array([angle - ideal_axial for angle in axial_angles])
                    print("  1. Axial Distortion:")
                    print(f"    • Average axial angle: {np.mean(axial_angles):.2f}°")
                    print(f"    • Maximum deviation: {np.max(np.abs(axial_deviations)):.2f}°")
                    print(f"    • Minimum deviation: {np.min(np.abs(axial_deviations)):.2f}°")
                    print(f"    • Standard deviation: {np.std(axial_angles):.2f}°")
                    
                    # Determine if distortion is symmetric or asymmetric
                    if np.all(axial_deviations < 0):
                        print("    • Pattern: Symmetric inward bending")
                    elif np.all(axial_deviations > 0):
                        print("    • Pattern: Symmetric outward bending")
                    else:
                        print("    • Pattern: Asymmetric bending")
                
                # 2. Equatorial Distortion Analysis
                if eq_deviations:
                    eq_angles = np.array(octahedron_eq_angles)
                    eq_deviations = np.array([angle - ideal_equatorial for angle in eq_angles])
                    print("\n  2. Equatorial Distortion:")
                    print(f"    • Average equatorial angle: {np.mean(eq_angles):.2f}°")
                    print(f"    • Maximum deviation: {np.max(np.abs(eq_deviations)):.2f}°")
                    print(f"    • Minimum deviation: {np.min(np.abs(eq_deviations)):.2f}°")
                    print(f"    • Standard deviation: {np.std(eq_angles):.2f}°")
                    
                    # Analyze equatorial distortion pattern
                    if np.all(eq_angles < ideal_equatorial):
                        print("    • Pattern: All angles compressed")
                    elif np.all(eq_angles > ideal_equatorial):
                        print("    • Pattern: All angles expanded")
                    else:
                        print("    • Pattern: Mixed compression/expansion")
                
                # 3. Overall Octahedral Distortion
                print("\n  3. Overall Octahedral Distortion:")
                print(f"    • Total variance: {eq_variance + axial_variance:.2f}°²")
                print(f"    • Distortion ratio (axial/equatorial): {axial_variance/eq_variance:.2f}" if eq_variance > 0 else "    • Distortion ratio: N/A (no equatorial variance)")
                
                # 4. Distortion Classification
                print("\n  4. Distortion Classification:")
                if axial_variance > 100 and eq_variance < 10:
                    print("    • Primary: Axial distortion dominant")
                elif eq_variance > 10 and axial_variance < 100:
                    print("    • Primary: Equatorial distortion dominant")
                elif axial_variance > 100 and eq_variance > 10:
                    print("    • Primary: Mixed axial and equatorial distortion")
                else:
                    print("    • Primary: Minimal distortion")
                
                # Store bond lengths
                structure_data['axial_lengths'].extend([b['length'] for b in axial_bonds])
                structure_data['equatorial_lengths'].extend([b['length'] for b in equatorial_bonds])
                
                # Calculate out-of-plane distortion
                # For each equatorial bond, calculate deviation from normal to axial direction
                axial_direction = np.mean([b['vector'] for b in axial_bonds], axis=0)
                axial_direction = axial_direction / np.linalg.norm(axial_direction)
                
                for bond in equatorial_bonds:
                    bond_vector = bond['vector'] / np.linalg.norm(bond['vector'])
                    # Project bond vector onto plane perpendicular to axial direction
                    projection = bond_vector - np.dot(bond_vector, axial_direction) * axial_direction
                    projection = projection / np.linalg.norm(projection)
                    # Calculate angle between projection and ideal equatorial direction
                    distortion = np.arccos(np.clip(np.dot(projection, bond_vector), -1.0, 1.0)) * 180 / np.pi
                    structure_data['out_of_plane_distortions'].append(distortion)
                
                # Store data for this octahedron
                structure_data['per_octahedron'][b_index] = {
                    'axial_angle': axial_angle,
                    'axial_lengths': [b['length'] for b in axial_bonds],
                    'equatorial_lengths': [b['length'] for b in equatorial_bonds],
                    'out_of_plane_distortions': structure_data['out_of_plane_distortions'][-4:],  # Last 4 distortions
                    'equatorial_variance': eq_variance
                }
            
            # Calculate overall statistics
            print("\n--- Analysis Complete ---")
            print(f"\nBond Angles:")
            print(f"Average axial angle: {np.mean(structure_data['axial_angles']):.2f}°")
            print(f"Average equatorial angle: {np.mean(structure_data['equatorial_angles']):.2f}°")
            
            print(f"\nBond Lengths:")
            print(f"Average axial length: {np.mean(structure_data['axial_lengths']):.3f} Å")
            print(f"Average equatorial length: {np.mean(structure_data['equatorial_lengths']):.3f} Å")
            
            print(f"\nDistortions:")
            print(f"Average out-of-plane distortion: {np.mean(structure_data['out_of_plane_distortions']):.2f}°")
            
            # Calculate bond angle variance
            ideal_axial = 180.0  # Ideal axial angle
            axial_variance = np.mean([(angle - ideal_axial)**2 for angle in structure_data['axial_angles']])
            equatorial_variance = np.mean(structure_data['octahedral_variances'])  # Average of per-octahedron variances
            
            print(f"\nBond Angle Variance:")
            print(f"Axial variance: {axial_variance:.2f}°²")
            print(f"Equatorial variance: {equatorial_variance:.2f}°²")
            
            # Calculate bond length quadratic elongation
            ideal_length = np.mean(structure_data['axial_lengths'])  # Use average as reference
            axial_elongation = np.mean([(length/ideal_length - 1)**2 for length in structure_data['axial_lengths']])
            equatorial_elongation = np.mean([(length/ideal_length - 1)**2 for length in structure_data['equatorial_lengths']])
            
            print(f"\nBond Length Quadratic Elongation:")
            print(f"Axial elongation: {axial_elongation:.4f}")
            print(f"Equatorial elongation: {equatorial_elongation:.4f}")
            
            # Add variance and elongation to the return data
            structure_data['axial_angle_variance'] = axial_variance
            structure_data['equatorial_angle_variance'] = equatorial_variance
            structure_data['axial_elongation'] = axial_elongation
            structure_data['equatorial_elongation'] = equatorial_elongation
            
            return structure_data
                
        except Exception as e:
            print(f"Error analyzing structure: {e}")
            return None 