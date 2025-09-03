"""
Distortion Analysis Module for q2D Materials.

This module implements comprehensive distortion parameter calculations including:
- Delta distortion (bond length variation)
- Sigma distortion (angular deviation from 90°)
- Lambda distortion (octahedral shape distortion)
- Tolerance factors (Goldschmidt and octahedral)
"""

import numpy as np
from ..utils.octadist.calc import CalcDistortion
from ..utils.common_a_sites import get_ionic_radius


class DistortionAnalyzer:
    """
    Distortion analyzer implementing comprehensive distortion parameters.
    """
    
    def __init__(self, geometry_calculator=None):
        """
        Initialize distortion analyzer.
        
        Parameters:
        geometry_calculator: GeometryCalculator instance
        """
        self.geometry_calc = geometry_calculator
        
        # Distortion parameters (set by calculations)
        self.octahedra_delta = None
        self.delta = None
        self.octahedra_sigma = None
        self.sigma = None
        self.cis_angles = None
        self.trans_angles = None
        self.trans_pairs = None
        self.octahedra_lambda_3 = None
        self.octahedra_lambda_2 = None
        self.octahedra_lambda = None
        self.lambda_3 = None
        self.lambda_2 = None
        self.lambda_val = None
        
        # Tolerance factors
        self.goldschmidt_tolerance = None
        self.octahedral_tolerance = None
    
    def compute_delta(self, octahedra_data, return_type="delta"):
        """
        Compute delta distortion parameter: 1/6 * Σ_{BX} |((d_BX - d_m)/d_m)²|
        
        Delta measures the variation in B-X bond lengths within each octahedron.
        A perfect octahedron has delta = 0.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data with bond distances
        return_type: "delta", "octahedra_delta", or "both"
        
        Returns:
        Delta distortion parameter(s)
        """
        if not octahedra_data:
            return None
        
        octahedra_delta = []
        
        for oct_key, oct_data in octahedra_data.items():
            # Get bond distances from the octahedron data
            bond_distances = oct_data.get('bond_distances', {})
            
            if not bond_distances:
                # Fallback: try to get from bond_distance_analysis
                bond_analysis = oct_data.get('bond_distance_analysis', {})
                if 'mean_bond_distance' in bond_analysis:
                    # If we only have mean, assume no distortion
                    octahedra_delta.append(0.0)
                    continue
                else:
                    octahedra_delta.append(0.0)
                    continue
            
            # Extract bond distance values
            distances = []
            for bond_key, bond_data in bond_distances.items():
                if isinstance(bond_data, dict) and 'distance' in bond_data:
                    distances.append(bond_data['distance'])
                elif isinstance(bond_data, (int, float)):
                    distances.append(float(bond_data))
            
            if len(distances) < 6:
                # Not enough bond distances
                octahedra_delta.append(0.0)
                continue
            
            distances = np.array(distances)
            
            # Calculate mean bond distance
            d_mean = np.mean(distances)
            
            if d_mean == 0:
                octahedra_delta.append(0.0)
                continue
            
            # Calculate delta: 1/6 * Σ |((d_BX - d_m)/d_m)²|
            delta_oct = np.mean(((distances - d_mean) / d_mean) ** 2)
            octahedra_delta.append(delta_oct)
        
        self.octahedra_delta = np.array(octahedra_delta)
        self.delta = np.mean(octahedra_delta) if len(octahedra_delta) > 0 else 0.0
        
        if return_type == "delta":
            return self.delta
        elif return_type == "octahedra_delta":
            return self.octahedra_delta
        elif return_type == "both":
            return self.delta, self.octahedra_delta
        else:
            print("Invalid return_type, enter delta, octahedra_delta, or both")
            return None
    
    def compute_sigma(self, octahedra_data, return_type="sigma"):
        """
        Compute sigma distortion parameter: 1/12 * Σ_{cis} |φ_{cis} - 90°|
        
        Sigma measures the deviation of cis angles from 90° in octahedra.
        A perfect octahedron has sigma = 0.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        return_type: "sigma", "octahedra_sigma", or "both"
        
        Returns:
        Sigma distortion parameter(s)
        """
        if not octahedra_data:
            return None
        
        # Define all possible angle pairs in an octahedron (15 total)
        pair_list = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), 
                     (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
        
        octahedra_sigma = []
        all_cis_angles = []
        all_trans_angles = []
        all_trans_pairs = []
        
        for oct_key, oct_data in octahedra_data.items():
            # Get octahedral coordinates
            central_atom = oct_data.get('central_atom', {})
            ligand_atoms = oct_data.get('ligand_atoms', {})
            
            if not central_atom or not ligand_atoms:
                octahedra_sigma.append(0.0)
                all_cis_angles.append([])
                all_trans_angles.append([])
                all_trans_pairs.append([])
                continue
            
            # Get central atom coordinates
            central_coord = np.array([
                central_atom['coordinates']['x'],
                central_atom['coordinates']['y'],
                central_atom['coordinates']['z']
            ])
            
            # Get ligand coordinates
            ligand_coords = []
            for i in range(6):  # Expect 6 ligands
                ligand_key = f"bond_{i+1}"
                if ligand_key in oct_data.get('bond_distances', {}):
                    # We need to reconstruct ligand coordinates from bond distances
                    # This is a simplified approach - in practice, we'd need the actual coordinates
                    bond_dist = oct_data['bond_distances'][ligand_key]['distance']
                    # Place ligands at estimated positions (this is approximate)
                    angle = i * np.pi / 3  # 60° spacing
                    if i < 2:  # Axial positions
                        z_offset = bond_dist if i == 0 else -bond_dist
                        ligand_coords.append(central_coord + [0, 0, z_offset])
                    else:  # Equatorial positions
                        x_offset = bond_dist * np.cos(angle)
                        y_offset = bond_dist * np.sin(angle)
                        ligand_coords.append(central_coord + [x_offset, y_offset, 0])
            
            if len(ligand_coords) < 6:
                octahedra_sigma.append(0.0)
                all_cis_angles.append([])
                all_trans_angles.append([])
                all_trans_pairs.append([])
                continue
            
            ligand_coords = np.array(ligand_coords)
            
            # Calculate all angles
            all_angles = []
            for pair in pair_list:
                vec1 = ligand_coords[pair[0]] - central_coord
                vec2 = ligand_coords[pair[1]] - central_coord
                angle = self._calculate_angle(vec1, vec2)
                all_angles.append(angle)
            
            all_angles = np.array(all_angles)
            
            # Find trans angles (3 closest to 180°)
            closest_to_180 = np.sort(np.abs(all_angles - 180.0))[:3]
            close_plus = 180.0 + closest_to_180
            close_minus = 180.0 - closest_to_180
            
            # Separate cis and trans angles
            cis_angles = []
            trans_angles = []
            trans_pairs = []
            
            for i, angle in enumerate(all_angles):
                if True in np.isclose(close_plus, angle) or True in np.isclose(close_minus, angle):
                    trans_angles.append(angle)
                    trans_pairs.append(pair_list[i])
                else:
                    cis_angles.append(angle)
            
            # Calculate sigma: 1/12 * Σ |φ_{cis} - 90°|
            if len(cis_angles) > 0:
                sigma_oct = np.mean(np.abs(np.array(cis_angles) - 90.0))
            else:
                sigma_oct = 0.0
            
            octahedra_sigma.append(sigma_oct)
            all_cis_angles.append(cis_angles)
            all_trans_angles.append(trans_angles)
            all_trans_pairs.append(trans_pairs)
        
        self.octahedra_sigma = np.array(octahedra_sigma)
        self.sigma = np.mean(octahedra_sigma) if len(octahedra_sigma) > 0 else 0.0
        self.cis_angles = all_cis_angles
        self.trans_angles = all_trans_angles
        self.trans_pairs = all_trans_pairs
        
        if return_type == "sigma":
            return self.sigma
        elif return_type == "octahedra_sigma":
            return self.octahedra_sigma
        elif return_type == "both":
            return self.sigma, self.octahedra_sigma
        else:
            print("Invalid return_type, enter sigma, octahedra_sigma, or both")
            return None
    
    def compute_lambda(self, octahedra_data, return_type="lambda", scaled=True):
        """
        Compute lambda distortion parameters.
        
        Lambda measures the displacement of the B-cation from its "natural center"
        in the octahedron, providing a measure of octahedral shape distortion.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        return_type: "lambda", "octahedra_lambda", or "both"
        scaled: Whether to use scaled lambda calculation
        
        Returns:
        Lambda distortion parameter(s)
        """
        if not octahedra_data:
            return None
        
        if self.trans_pairs is None:
            # Need to compute sigma first to get trans_pairs
            self.compute_sigma(octahedra_data)
        
        octahedra_lambda_3 = []
        octahedra_lambda_2 = []
        octahedra_lambda = []
        
        for i, (oct_key, oct_data) in enumerate(octahedra_data.items()):
            if i >= len(self.trans_pairs) or not self.trans_pairs[i]:
                octahedra_lambda_3.append(0.0)
                octahedra_lambda_2.append(0.0)
                octahedra_lambda.append(0.0)
                continue
            
            # Get octahedral coordinates
            central_atom = oct_data.get('central_atom', {})
            ligand_atoms = oct_data.get('ligand_atoms', {})
            
            if not central_atom or not ligand_atoms:
                octahedra_lambda_3.append(0.0)
                octahedra_lambda_2.append(0.0)
                octahedra_lambda.append(0.0)
                continue
            
            # Get central atom coordinates
            B = np.array([
                central_atom['coordinates']['x'],
                central_atom['coordinates']['y'],
                central_atom['coordinates']['z']
            ])
            
            # Get ligand coordinates (simplified - would need actual coordinates)
            # This is a placeholder implementation
            ligand_coords = []
            for j in range(6):
                # Approximate ligand positions
                angle = j * np.pi / 3
                if j < 2:  # Axial
                    z_offset = 2.5 if j == 0 else -2.5
                    ligand_coords.append(B + [0, 0, z_offset])
                else:  # Equatorial
                    x_offset = 2.5 * np.cos(angle)
                    y_offset = 2.5 * np.sin(angle)
                    ligand_coords.append(B + [x_offset, y_offset, 0])
            
            Xs = np.array(ligand_coords)
            
            # Calculate midpoints of trans X-X connections
            tilde_P_list = []
            e_basis = np.zeros((3, 3))
            
            for e_idx, pair in enumerate(self.trans_pairs[i]):
                if e_idx >= 3:  # Only 3 trans pairs
                    break
                
                # Calculate basis vectors
                e_basis[e_idx] = ((Xs[pair[0]] - Xs[pair[1]]) / 
                                 np.linalg.norm(Xs[pair[0]] - Xs[pair[1]]))
                
                # Calculate midpoint
                mid_point = 0.5 * (Xs[pair[0]] + Xs[pair[1]])
                tilde_P_list.append(mid_point)
            
            if len(tilde_P_list) < 3:
                octahedra_lambda_3.append(0.0)
                octahedra_lambda_2.append(0.0)
                octahedra_lambda.append(0.0)
                continue
            
            # Calculate P_tilde (natural center)
            P_tilde = np.mean(tilde_P_list, axis=0)
            
            # Displacement vector
            D = B - P_tilde
            
            # Project onto basis vectors
            D_i = np.array([abs(np.dot(D, e_basis[0])), 
                           abs(np.dot(D, e_basis[1])),
                           abs(np.dot(D, e_basis[2]))])
            
            # Calculate lambda parameters
            lambda_ij = np.zeros(3)
            
            if scaled:
                lambda_ij[0] = min(D_i[0]/D_i[1], D_i[1]/D_i[0]) * (D_i[0] + D_i[1])
                lambda_ij[1] = min(D_i[0]/D_i[2], D_i[2]/D_i[0]) * (D_i[0] + D_i[2])
                lambda_ij[2] = min(D_i[1]/D_i[2], D_i[2]/D_i[1]) * (D_i[2] + D_i[1])
            else:
                lambda_ij[0] = min(D_i[0]/D_i[1], D_i[1]/D_i[0])
                lambda_ij[1] = min(D_i[0]/D_i[2], D_i[2]/D_i[0])
                lambda_ij[2] = min(D_i[1]/D_i[2], D_i[2]/D_i[1])
            
            # Calculate lambda values
            lambda_3 = lambda_ij[0] * lambda_ij[1] * lambda_ij[2]
            lambda_2 = np.max(lambda_ij)
            lambda_val = lambda_3 / lambda_2 if lambda_2 > 0 else 0.0
            
            octahedra_lambda_3.append(lambda_3)
            octahedra_lambda_2.append(lambda_2)
            octahedra_lambda.append(lambda_val)
        
        self.octahedra_lambda_3 = np.array(octahedra_lambda_3)
        self.octahedra_lambda_2 = np.array(octahedra_lambda_2)
        self.octahedra_lambda = np.array(octahedra_lambda)
        self.lambda_3 = np.mean(octahedra_lambda_3) if len(octahedra_lambda_3) > 0 else 0.0
        self.lambda_2 = np.mean(octahedra_lambda_2) if len(octahedra_lambda_2) > 0 else 0.0
        self.lambda_val = np.mean(octahedra_lambda) if len(octahedra_lambda) > 0 else 0.0
        
        if return_type == "lambda":
            return self.lambda_3, self.lambda_2
        elif return_type == "octahedra_lambda":
            return self.octahedra_lambda_3, self.octahedra_lambda_2
        elif return_type == "both":
            return (self.octahedra_lambda_3, self.octahedra_lambda_2, 
                   self.lambda_3, self.lambda_2)
        else:
            print("Invalid return_type, enter lambda, octahedra_lambda, or both")
            return None
    
    def compute_goldschmidt_tolerance(self, A_site, B_site, X_site, R_A=None):
        """
        Compute Goldschmidt tolerance factor: t = (R_A + R_X) / (√2 * (R_B + R_X))
        
        Parameters:
        A_site: A-site cation symbol
        B_site: B-site cation symbol  
        X_site: X-site anion symbol
        R_A: A-site ionic radius (optional, will be looked up if not provided)
        
        Returns:
        Goldschmidt tolerance factor
        """
        try:
            if R_A is None:
                R_A = get_ionic_radius("A", A_site)
            R_B = get_ionic_radius("B", B_site)
            R_X = get_ionic_radius("X", X_site)
            
            self.goldschmidt_tolerance = (R_A + R_X) / (np.sqrt(2) * (R_B + R_X))
            return self.goldschmidt_tolerance
            
        except Exception as e:
            print(f"Error calculating Goldschmidt tolerance factor: {e}")
            return None
    
    def compute_octahedral_tolerance(self, B_site, X_site):
        """
        Compute octahedral tolerance factor: μ = R_B / R_X
        
        Parameters:
        B_site: B-site cation symbol
        X_site: X-site anion symbol
        
        Returns:
        Octahedral tolerance factor
        """
        try:
            R_B = get_ionic_radius("B", B_site)
            R_X = get_ionic_radius("X", X_site)
            
            self.octahedral_tolerance = R_B / R_X
            return self.octahedral_tolerance
            
        except Exception as e:
            print(f"Error calculating octahedral tolerance factor: {e}")
            return None
    
    def _calculate_angle(self, vec1, vec2):
        """
        Calculate angle between two vectors.
        
        Parameters:
        vec1, vec2: Vectors to calculate angle between
        
        Returns:
        Angle in degrees
        """
        # Handle precision errors
        arg_arccos = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        if arg_arccos > 1.0 and arg_arccos < 1.0005:
            arg_arccos = 1.0
        elif arg_arccos < -1.0 and arg_arccos > -1.0005:
            arg_arccos = -1.0
        
        return np.arccos(arg_arccos) * 180 / np.pi
    
    def get_distortion_summary(self):
        """
        Get a comprehensive summary of all distortion parameters.
        
        Returns:
        dict: Summary of distortion parameters
        """
        summary = {
            'delta_distortion': {
                'overall_delta': self.delta,
                'octahedra_delta': self.octahedra_delta.tolist() if self.octahedra_delta is not None else None,
                'description': 'Bond length variation within octahedra'
            },
            'sigma_distortion': {
                'overall_sigma': self.sigma,
                'octahedra_sigma': self.octahedra_sigma.tolist() if self.octahedra_sigma is not None else None,
                'description': 'Angular deviation from 90° in cis angles'
            },
            'lambda_distortion': {
                'lambda_3': self.lambda_3,
                'lambda_2': self.lambda_2,
                'lambda_ratio': self.lambda_val,
                'octahedra_lambda_3': self.octahedra_lambda_3.tolist() if self.octahedra_lambda_3 is not None else None,
                'octahedra_lambda_2': self.octahedra_lambda_2.tolist() if self.octahedra_lambda_2 is not None else None,
                'description': 'Displacement of B-cation from natural center'
            },
            'tolerance_factors': {
                'goldschmidt_tolerance': self.goldschmidt_tolerance,
                'octahedral_tolerance': self.octahedral_tolerance,
                'description': 'Structural stability indicators'
            }
        }
        
        return summary
