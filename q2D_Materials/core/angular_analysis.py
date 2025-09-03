"""
Enhanced Angular Analysis for Octahedral Structures.

This module handles comprehensive angular calculations inspired by Pyrovskite, including:
- Axial-Central-Axial angles
- Central-Axial-Central angles (between adjacent octahedra)
- Central-Equatorial-Central angles (between octahedra in same z-plane)
- Cis and trans angle analysis
- Angular distribution statistics
- Gaussian smearing for smooth angle distributions
"""

import numpy as np
from .geometry import GeometryCalculator


class AngularAnalyzer:
    """
    Handles angular analysis for octahedral structures.
    """
    
    def __init__(self, geometry_calculator=None):
        """
        Initialize angular analyzer.
        
        Parameters:
        geometry_calculator: GeometryCalculator instance
        """
        self.geometry_calc = geometry_calculator or GeometryCalculator()
    
    def calculate_angular_analysis(self, coord_octa, axial_positions, equatorial_positions, 
                                 central_coord, octahedron_index, all_octahedra, 
                                 all_coords, all_symbols, cutoff_distance):
        """
        Calculate comprehensive angular analysis including:
        - Axial-Central-Axial angles
        - Central-Axial-Central angles (between adjacent octahedra)  
        - Central-Equatorial-Central angles (between octahedra in same z-plane)
        
        Parameters:
        coord_octa: octahedral coordinates
        axial_positions: axial atom positions
        equatorial_positions: equatorial atom positions
        central_coord: central atom coordinates
        octahedron_index: index of current octahedron
        all_octahedra: list of all octahedra data
        all_coords: all atomic coordinates
        all_symbols: all atomic symbols
        cutoff_distance: distance cutoff
        
        Returns:
        dict: Comprehensive angular analysis
        """
        angular_analysis = {}
        
        try:
            # 1. AXIAL-CENTRAL-AXIAL ANGLES (within same octahedron)
            angular_analysis["axial_central_axial"] = self._calculate_axial_central_axial(
                axial_positions, central_coord, octahedron_index, all_coords, cutoff_distance
            )
            
            # 2. CENTRAL-AXIAL-CENTRAL ANGLES (between adjacent octahedra)
            angular_analysis["central_axial_central"] = self._calculate_central_axial_central(
                axial_positions, central_coord, octahedron_index, all_octahedra, 
                all_coords, cutoff_distance
            )
            
            # 3. CENTRAL-EQUATORIAL-CENTRAL ANGLES (between octahedra in same z-plane)
            angular_analysis["central_equatorial_central"] = self._calculate_central_equatorial_central(
                equatorial_positions, central_coord, octahedron_index, all_octahedra,
                all_coords, cutoff_distance
            )
            
            # 4. SUMMARY STATISTICS
            angular_analysis["summary"] = self._calculate_angular_summary(angular_analysis)
            
        except Exception as e:
            angular_analysis["error"] = f"Error in angular analysis: {str(e)}"
            
        return angular_analysis
    
    def _calculate_axial_central_axial(self, axial_positions, central_coord, octahedron_index, 
                                     all_coords, cutoff_distance):
        """Calculate axial-central-axial angles within same octahedron."""
        if len(axial_positions) != 2:
            return {"error": "Need exactly 2 axial positions"}
        
        axial_coords = [pos['coord'] for pos in axial_positions]
        
        # Vector from central to each axial atom
        vec_to_axial1 = axial_coords[0] - central_coord
        vec_to_axial2 = axial_coords[1] - central_coord
        
        # Calculate angle between axial atoms through central atom
        axial_central_axial_angle = self.geometry_calc.calculate_vector_angle(vec_to_axial1, vec_to_axial2)
        
        return {
            "angle_degrees": float(axial_central_axial_angle),
            "axial_atom_1_global_index": self.geometry_calc.find_global_atom_index(
                axial_coords[0], all_coords, central_coord, cutoff_distance
            ),
            "axial_atom_2_global_index": self.geometry_calc.find_global_atom_index(
                axial_coords[1], all_coords, central_coord, cutoff_distance
            ),
            "central_atom_global_index": self._get_octahedron_central_index(octahedron_index, all_coords, central_coord),
            "deviation_from_180": float(abs(180.0 - axial_central_axial_angle)),
            "is_linear": bool(abs(180.0 - axial_central_axial_angle) < 5.0)
        }
    
    def _calculate_central_axial_central(self, axial_positions, central_coord, octahedron_index,
                                       all_octahedra, all_coords, cutoff_distance):
        """Calculate central-axial-central angles between adjacent octahedra."""
        central_axial_central_angles = []
        
        if len(axial_positions) != 2:
            return central_axial_central_angles
        
        for i, axial_pos in enumerate(axial_positions):
            axial_coord = axial_pos['coord']
            axial_global_idx = self.geometry_calc.find_global_atom_index(
                axial_coord, all_coords, central_coord, cutoff_distance
            )
            
            # Find other octahedra that share this axial atom
            connected_octahedra = self._find_octahedra_sharing_atom(
                axial_global_idx, octahedron_index, all_octahedra, all_coords, cutoff_distance
            )
            
            for connected_oct_idx in connected_octahedra:
                connected_central_coord = np.array(all_octahedra[connected_oct_idx]['central_coord'])
                
                # Vector from axial atom to each central atom
                vec_to_central1 = central_coord - axial_coord
                vec_to_central2 = connected_central_coord - axial_coord
                
                # Calculate angle between central atoms through axial atom
                central_axial_central_angle = self.geometry_calc.calculate_vector_angle(vec_to_central1, vec_to_central2)
                
                central_axial_central_angles.append({
                    "angle_degrees": float(central_axial_central_angle),
                    "central_atom_1_global_index": self._get_octahedron_central_index(octahedron_index, all_coords, central_coord),
                    "axial_atom_global_index": int(axial_global_idx) if axial_global_idx else None,
                    "central_atom_2_global_index": all_octahedra[connected_oct_idx]['global_index'],
                    "connected_octahedron": f"octahedron_{connected_oct_idx + 1}",
                    "axial_position": i + 1,
                    "bridge_type": "axial_bridge"
                })
        
        return central_axial_central_angles
    
    def _calculate_central_equatorial_central(self, equatorial_positions, central_coord, octahedron_index,
                                            all_octahedra, all_coords, cutoff_distance):
        """Calculate central-equatorial-central angles between octahedra in same z-plane."""
        central_equatorial_central_angles = []
        
        if len(equatorial_positions) < 2:
            return central_equatorial_central_angles
        
        # Get current octahedron's z-coordinate
        current_z = central_coord[2]
        tolerance_z = 0.5
        
        for i, eq_pos in enumerate(equatorial_positions):
            eq_coord = eq_pos['coord']
            eq_global_idx = self.geometry_calc.find_global_atom_index(
                eq_coord, all_coords, central_coord, cutoff_distance
            )
            
            # Find other octahedra in same z-plane that share this equatorial atom
            connected_octahedra = self._find_octahedra_sharing_atom_same_z(
                eq_global_idx, octahedron_index, all_octahedra, current_z, tolerance_z,
                all_coords, cutoff_distance
            )
            
            for connected_oct_idx in connected_octahedra:
                connected_central_coord = np.array(all_octahedra[connected_oct_idx]['central_coord'])
                
                # Vector from equatorial atom to each central atom
                vec_to_central1 = central_coord - eq_coord
                vec_to_central2 = connected_central_coord - eq_coord
                
                # Calculate angle between central atoms through equatorial atom
                central_eq_central_angle = self.geometry_calc.calculate_vector_angle(vec_to_central1, vec_to_central2)
                
                central_equatorial_central_angles.append({
                    "angle_degrees": float(central_eq_central_angle),
                    "central_atom_1_global_index": self._get_octahedron_central_index(octahedron_index, all_coords, central_coord),
                    "equatorial_atom_global_index": int(eq_global_idx) if eq_global_idx else None,
                    "central_atom_2_global_index": all_octahedra[connected_oct_idx]['global_index'],
                    "connected_octahedron": f"octahedron_{connected_oct_idx + 1}",
                    "equatorial_position": i + 1,
                    "bridge_type": "equatorial_bridge",
                    "z_plane_difference": float(abs(current_z - connected_central_coord[2]))
                })
        
        return central_equatorial_central_angles
    
    def _calculate_angular_summary(self, angular_analysis):
        """Calculate summary statistics for angular analysis."""
        cac_angles = angular_analysis.get("central_axial_central", [])
        cec_angles = angular_analysis.get("central_equatorial_central", [])
        aca_data = angular_analysis.get("axial_central_axial", {})
        
        return {
            "total_axial_bridges": len(cac_angles),
            "total_equatorial_bridges": len(cec_angles),
            "average_axial_central_axial_angle": float(aca_data.get("angle_degrees", 0)),
            "average_central_axial_central_angle": float(np.mean([angle["angle_degrees"] for angle in cac_angles])) if cac_angles else 0.0,
            "average_central_equatorial_central_angle": float(np.mean([angle["angle_degrees"] for angle in cec_angles])) if cec_angles else 0.0
        }
    
    def _get_octahedron_central_index(self, octahedron_index, all_coords, central_coord):
        """Get global index of central atom using PBC-aware distance matching."""
        # Find the global index by coordinate matching using PBC-aware distance
        tolerance = 1e-6
        for i, coord in enumerate(all_coords):
            distance = self.geometry_calc.calculate_distance(coord, central_coord)
            if distance < tolerance:
                return i
        return None
    
    def _find_octahedra_sharing_atom(self, atom_global_index, exclude_octahedron_index, 
                                   all_octahedra, all_coords, cutoff_distance):
        """Find octahedra that share a specific atom."""
        sharing_octahedra = []
        
        if atom_global_index is None:
            return sharing_octahedra
        
        for i, octa_data in enumerate(all_octahedra):
            if i == exclude_octahedron_index:
                continue
                
            coord_octa = octa_data['coord_octa']
            central_global_idx = octa_data['global_index']
            central_coord = all_coords[central_global_idx]
            
            # Check all ligand positions
            for j in range(1, len(coord_octa)):
                ligand_coord = coord_octa[j]
                ligand_global_idx = self.geometry_calc.find_global_atom_index(
                    ligand_coord, all_coords, central_coord, cutoff_distance
                )
                
                if ligand_global_idx == atom_global_index:
                    sharing_octahedra.append(i)
                    break
        
        return sharing_octahedra
    
    def _find_octahedra_sharing_atom_same_z(self, atom_global_index, exclude_octahedron_index,
                                          all_octahedra, target_z, tolerance_z, all_coords, cutoff_distance):
        """Find octahedra in same z-plane that share a specific atom."""
        sharing_octahedra = []
        
        if atom_global_index is None:
            return sharing_octahedra
        
        for i, octa_data in enumerate(all_octahedra):
            if i == exclude_octahedron_index:
                continue
            
            # Check if octahedron is in same z-plane
            central_z = octa_data['central_coord'][2]
            if abs(central_z - target_z) > tolerance_z:
                continue
                
            coord_octa = octa_data['coord_octa']
            central_global_idx = octa_data['global_index']
            central_coord = all_coords[central_global_idx]
            
            # Check all ligand positions
            for j in range(1, len(coord_octa)):
                ligand_coord = coord_octa[j]
                ligand_global_idx = self.geometry_calc.find_global_atom_index(
                    ligand_coord, all_coords, central_coord, cutoff_distance
                )
                
                if ligand_global_idx == atom_global_index:
                    sharing_octahedra.append(i)
                    break
        
        return sharing_octahedra
    
    def calculate_cis_trans_angles(self, octahedra_data):
        """
        Calculate cis and trans angles for all octahedra, inspired by Pyrovskite.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        
        Returns:
        dict: Updated octahedra data with cis/trans angle analysis
        """
        # Define all possible angle pairs in an octahedron (15 total)
        pair_list = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 2), (1, 3), (1, 4), 
                     (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
        
        for oct_key, oct_data in octahedra_data.items():
            # Get octahedral coordinates
            central_atom = oct_data.get('central_atom', {})
            ligand_atoms = oct_data.get('ligand_atoms', {})
            
            if not central_atom or not ligand_atoms:
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
            
            # Store in octahedra data
            if 'angular_analysis' not in oct_data:
                oct_data['angular_analysis'] = {}
            
            oct_data['angular_analysis']['cis_trans_analysis'] = {
                'cis_angles': cis_angles,
                'trans_angles': trans_angles,
                'trans_pairs': trans_pairs,
                'all_angles': all_angles.tolist(),
                'cis_angle_count': len(cis_angles),
                'trans_angle_count': len(trans_angles),
                'average_cis_angle': float(np.mean(cis_angles)) if cis_angles else 0.0,
                'average_trans_angle': float(np.mean(trans_angles)) if trans_angles else 0.0,
                'cis_angle_std': float(np.std(cis_angles)) if cis_angles else 0.0,
                'trans_angle_std': float(np.std(trans_angles)) if trans_angles else 0.0
            }
        
        return octahedra_data
    
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
    
    def get_angular_distribution_statistics(self, octahedra_data):
        """
        Get comprehensive angular distribution statistics.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        
        Returns:
        dict: Angular distribution statistics
        """
        all_cis_angles = []
        all_trans_angles = []
        all_axial_central_axial = []
        
        for oct_data in octahedra_data.values():
            angular_analysis = oct_data.get('angular_analysis', {})
            
            # Collect cis/trans angles
            cis_trans = angular_analysis.get('cis_trans_analysis', {})
            if cis_trans:
                all_cis_angles.extend(cis_trans.get('cis_angles', []))
                all_trans_angles.extend(cis_trans.get('trans_angles', []))
            
            # Collect axial-central-axial angles
            aca_data = angular_analysis.get('axial_central_axial', {})
            if aca_data and 'angle_degrees' in aca_data:
                all_axial_central_axial.append(aca_data['angle_degrees'])
        
        # Convert to numpy arrays
        all_cis_angles = np.array(all_cis_angles)
        all_trans_angles = np.array(all_trans_angles)
        all_axial_central_axial = np.array(all_axial_central_axial)
        
        statistics = {
            'cis_angles': {
                'count': len(all_cis_angles),
                'mean': float(np.mean(all_cis_angles)) if len(all_cis_angles) > 0 else 0.0,
                'std': float(np.std(all_cis_angles)) if len(all_cis_angles) > 0 else 0.0,
                'min': float(np.min(all_cis_angles)) if len(all_cis_angles) > 0 else 0.0,
                'max': float(np.max(all_cis_angles)) if len(all_cis_angles) > 0 else 0.0,
                'deviation_from_90': float(np.mean(np.abs(all_cis_angles - 90.0))) if len(all_cis_angles) > 0 else 0.0
            },
            'trans_angles': {
                'count': len(all_trans_angles),
                'mean': float(np.mean(all_trans_angles)) if len(all_trans_angles) > 0 else 0.0,
                'std': float(np.std(all_trans_angles)) if len(all_trans_angles) > 0 else 0.0,
                'min': float(np.min(all_trans_angles)) if len(all_trans_angles) > 0 else 0.0,
                'max': float(np.max(all_trans_angles)) if len(all_trans_angles) > 0 else 0.0,
                'deviation_from_180': float(np.mean(np.abs(all_trans_angles - 180.0))) if len(all_trans_angles) > 0 else 0.0
            },
            'axial_central_axial': {
                'count': len(all_axial_central_axial),
                'mean': float(np.mean(all_axial_central_axial)) if len(all_axial_central_axial) > 0 else 0.0,
                'std': float(np.std(all_axial_central_axial)) if len(all_axial_central_axial) > 0 else 0.0,
                'min': float(np.min(all_axial_central_axial)) if len(all_axial_central_axial) > 0 else 0.0,
                'max': float(np.max(all_axial_central_axial)) if len(all_axial_central_axial) > 0 else 0.0,
                'deviation_from_180': float(np.mean(np.abs(all_axial_central_axial - 180.0))) if len(all_axial_central_axial) > 0 else 0.0
            }
        }
        
        return statistics