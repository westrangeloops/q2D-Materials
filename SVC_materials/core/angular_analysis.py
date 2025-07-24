"""
Angular analysis for octahedral structures.

This module handles angular calculations including:
- Axial-Central-Axial angles
- Central-Axial-Central angles (between adjacent octahedra)
- Central-Equatorial-Central angles (between octahedra in same z-plane)
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