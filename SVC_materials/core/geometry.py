"""
Geometry utilities for octahedral analysis.

This module contains geometric calculations, coordinate transformations,
and spatial analysis functions used by the octahedral analyzer.
"""

import numpy as np
from scipy.spatial.distance import euclidean as distance
from ..utils.octadist.linear import angle_btw_vectors


class GeometryCalculator:
    """
    Handles geometric calculations for octahedral structures.
    """
    
    def __init__(self, tolerance=1e-3, atoms=None):
        """
        Initialize geometry calculator.
        
        Parameters:
        tolerance: coordinate matching tolerance
        atoms: ASE atoms object to get cell and PBC information
        """
        self.tolerance = tolerance
        self.atoms = atoms
        
        # Set up PBC parameters if atoms object is provided
        if atoms is not None:
            self.cell = atoms.get_cell()
            self.pbc = atoms.get_pbc()
            self.use_pbc = np.any(self.pbc) and atoms.get_volume() > 0
        else:
            self.cell = None
            self.pbc = None
            self.use_pbc = False
    
    def update_atoms(self, atoms):
        """Update the atoms object and PBC parameters."""
        self.atoms = atoms
        if atoms is not None:
            self.cell = atoms.get_cell()
            self.pbc = atoms.get_pbc()
            self.use_pbc = np.any(self.pbc) and atoms.get_volume() > 0
        else:
            self.cell = None
            self.pbc = None
            self.use_pbc = False
    
    def _calculate_distance_pbc_aware(self, coord1, coord2):
        """
        Calculate distance between two coordinates using PBC if available.
        
        Parameters:
        coord1, coord2: coordinates to calculate distance between
        
        Returns:
        distance value
        """
        if self.use_pbc and self.cell is not None:
            # Use PBC-aware distance calculation
            vector_a, vector_b, vector_c = self.cell[0], self.cell[1], self.cell[2]
            dx, dy, dz = self._calculate_distance_pbc_signo(
                coord1[0], coord1[1], coord1[2],
                coord2[0], coord2[1], coord2[2],
                vector_a, vector_b, vector_c
            )
            return np.linalg.norm([dx, dy, dz])
        else:
            # Standard Euclidean distance
            return np.linalg.norm(np.array(coord1) - np.array(coord2))
    
    def _calculate_distance_pbc_signo(self, x1, y1, z1, x2, y2, z2, vector_a, vector_b, vector_c):
        """
        Calculate minimum image distance with periodic boundary conditions.
        
        Parameters:
        x1, y1, z1: coordinates of first point
        x2, y2, z2: coordinates of second point
        vector_a, vector_b, vector_c: unit cell vectors
        
        Returns:
        [dx, dy, dz]: distance vector considering PBC
        """
        # Calculate differences in coordinates
        dx = x2 - x1
        dy = y2 - y1
        dz = z2 - z1
        
        # Apply periodic boundary conditions using minimum image convention
        inv_cell = np.linalg.inv([vector_a, vector_b, vector_c])
        positions_cell_1 = np.dot([dx, dy, dz], inv_cell)
        positions_cell_2 = positions_cell_1 - np.round(positions_cell_1)
        dx, dy, dz = np.dot(positions_cell_2, [vector_a, vector_b, vector_c])
        
        return [dx, dy, dz]
    
    def find_global_atom_index(self, target_coord, all_coords, central_coord, cutoff_distance):
        """
        Find the global index of an atom by matching coordinates using PBC-aware distances.
        
        Parameters:
        target_coord: coordinates of the atom to find
        all_coords: all atomic coordinates
        central_coord: central atom coordinates for reference
        cutoff_distance: maximum search distance
        
        Returns:
        global index of the atom, or None if not found
        """
        target_coord = np.array(target_coord)
        central_coord = np.array(central_coord)
        max_search_distance = cutoff_distance + 2.0
        
        # First try: exact coordinate match within reasonable distance
        for i, coord in enumerate(all_coords):
            dist_to_central = self._calculate_distance_pbc_aware(coord, central_coord)
            if dist_to_central <= max_search_distance:
                if np.allclose(coord, target_coord, atol=self.tolerance):
                    return i
        
        # Second try: try with larger tolerance for PBC effects
        larger_tolerance = self.tolerance * 10
        for i, coord in enumerate(all_coords):
            dist_to_central = self._calculate_distance_pbc_aware(coord, central_coord)
            if dist_to_central <= max_search_distance:
                if np.allclose(coord, target_coord, atol=larger_tolerance):
                    return i
        
        # Third try: find closest atom within search radius using PBC-aware distance
        min_distance = float('inf')
        closest_index = None
        for i, coord in enumerate(all_coords):
            dist_to_central = self._calculate_distance_pbc_aware(coord, central_coord)
            if dist_to_central <= max_search_distance:
                distance_val = self._calculate_distance_pbc_aware(coord, target_coord)
                if distance_val < min_distance and distance_val < cutoff_distance:
                    min_distance = distance_val
                    closest_index = i
        
        return closest_index
    
    def find_closest_atom_by_symbol(self, target_coord, all_coords, all_symbols, symbol, max_distance=None):
        """
        Find the closest atom with matching symbol to the target coordinates using PBC-aware distance.
        
        Parameters:
        target_coord: target coordinates
        all_coords: all atomic coordinates
        all_symbols: all atomic symbols
        symbol: atomic symbol to match
        max_distance: maximum search distance
        
        Returns:
        global index of closest matching atom, or None if not found
        """
        if max_distance is None:
            max_distance = 10.0  # Default large search radius
        
        target_coord = np.array(target_coord)
        min_distance = float('inf')
        closest_index = None
        
        for i, (coord, atom_symbol) in enumerate(zip(all_coords, all_symbols)):
            if atom_symbol == symbol:
                distance_val = self._calculate_distance_pbc_aware(coord, target_coord)
                if distance_val < min_distance and distance_val <= max_distance:
                    min_distance = distance_val
                    closest_index = i
        
        return closest_index
    
    def identify_axial_equatorial_positions(self, coord_octa, atom_symbols, central_coord):
        """
        Identify which atoms are in axial vs equatorial positions in the octahedron.
        Axial positions are defined as the trans ligands along the Z-axis.
        Equatorial positions are the remaining ligands in the XY plane.
        
        Parameters:
        coord_octa: octahedral coordinates
        atom_symbols: atomic symbols
        central_coord: central atom coordinates
        
        Returns:
        tuple: (axial_positions, equatorial_positions)
        """
        ligand_coords = coord_octa[1:]
        ligand_symbols = atom_symbols[1:]
        
        # Always use Z-axis as reference for axial positions
        z_positions = [(i, coord[2]) for i, coord in enumerate(ligand_coords)]
        z_positions.sort(key=lambda x: x[1])
        
        # The ligands with highest and lowest Z coordinates are axial
        axial_indices = [z_positions[0][0], z_positions[-1][0]]
        
        # Classify atoms as axial or equatorial
        axial_positions = []
        equatorial_positions = []
        
        for idx in range(len(ligand_coords)):
            atom_data = {
                'local_index': idx,
                'symbol': ligand_symbols[idx],
                'coord': ligand_coords[idx]
            }
            
            if idx in axial_indices:
                axial_positions.append(atom_data)
            else:
                equatorial_positions.append(atom_data)
        
        return axial_positions, equatorial_positions
    
    def calculate_vector_angle(self, vec1, vec2):
        """
        Calculate angle between two vectors using octadist function.
        
        Parameters:
        vec1, vec2: vectors to calculate angle between
        
        Returns:
        angle in degrees
        """
        return angle_btw_vectors(vec1, vec2)
    
    def calculate_distance(self, coord1, coord2):
        """
        Calculate distance between two coordinates using PBC-aware method.
        
        Parameters:
        coord1, coord2: coordinates
        
        Returns:
        distance
        """
        return self._calculate_distance_pbc_aware(coord1, coord2) 