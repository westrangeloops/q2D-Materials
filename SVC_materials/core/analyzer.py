from ..utils.file_handlers import vasp_load, save_vasp
from ase.io import read, write
from ase.visualize import view
import os
import numpy as np

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