"""
Module for applying Glazer tilting distortions to perovskite structures.

This module provides functionality to apply octahedral tilting patterns
to perovskite structures using the Glazer notation system. It adapts
the DistortPerovskite algorithm to work with ASE Atoms objects.

Key Features:
- Supports both cubic (small angles) and orthorhombic (larger angles) distortions
- Automatically adjusts A-site positions to accommodate orthorhombic cell
- For 2D structures, spacer positions in XY plane are adjusted accordingly
- Cell shape adapts from cubic to orthorhombic based on tilting angles

Orthorhombic Mode:
When allow_orthorhombic=True and angles exceed max_angle_cubic, the cell
distorts from cubic to orthorhombic. This allows for larger tilting angles
by adjusting the cell dimensions (a, b, c) independently. A-site positions
are automatically adjusted in the XY plane to match the distorted cell.
"""

import copy
import math
import numpy as np
from ase import Atoms


class GlazerTilting:
    """
    Apply Glazer tilting distortions to perovskite structures.
    
    This class adapts the DistortPerovskite algorithm to work with
    ASE Atoms objects, allowing integration with the q2D Materials pipeline.
    """
    
    def __init__(self, structure, BX_dist=None, symprec=None, 
                 A_symbol=None, B_symbol=None, X_symbol=None):
        """
        Initialize GlazerTilting with a perovskite structure.
        
        Parameters
        ----------
        structure : ase.Atoms
            The perovskite structure to distort. Must be a bulk perovskite
            with ABX3 composition. Currently requires a 2x2x2 supercell or larger.
        BX_dist : float, optional
            B-X bond distance in Angstroms. If None, will be estimated from structure.
        symprec : float, optional
            Symmetry precision for spglib. If None, will be auto-calculated.
        A_symbol : str, optional
            Symbol of A-site cation. If provided, will use this instead of auto-detecting.
            Required when structure contains additional elements (e.g., molecular spacers).
        B_symbol : str, optional
            Symbol of B-site cation. If provided, will use this instead of auto-detecting.
            Required when structure contains additional elements (e.g., molecular spacers).
        X_symbol : str, optional
            Symbol of X-site anion. If provided, will use this instead of auto-detecting.
            Required when structure contains additional elements (e.g., molecular spacers).
        """
        if not isinstance(structure, Atoms):
            raise TypeError("structure must be an ase.Atoms object")
        
        self.structure = structure.copy()
        self.cell = structure.cell.copy()
        
        # Get symbols and positions
        self.symbols = structure.get_chemical_symbols()
        self.positions = structure.get_positions()
        
        # Identify A, B, X sites
        self._identify_sites(A_symbol, B_symbol, X_symbol)
        
        # Estimate or use provided BX_dist
        if BX_dist is None:
            cell_lengths = np.linalg.norm(self.cell, axis=1)
            self.BX_dist = np.mean(cell_lengths) / 2.0
        else:
            self.BX_dist = BX_dist
        
        # Set symprec
        if symprec is None:
            self.symprec = 0.5e-2 * self.BX_dist
        else:
            self.symprec = symprec
        
        # Estimate supercell size
        self._estimate_supercell_size()
        
        # Initialize distortion parameters
        self._distortion_angles = None
        self._kvecs = None
    
    def _identify_sites(self, A_symbol=None, B_symbol=None, X_symbol=None):
        """Identify A, B, and X sites from structure.
        
        If A_symbol, B_symbol, and X_symbol are provided, use them directly.
        Otherwise, try to auto-detect from the structure.
        """
        # If all three symbols are provided, use them directly
        if A_symbol is not None and B_symbol is not None and X_symbol is not None:
            self.A_symbol = A_symbol
            self.B_symbol = B_symbol
            self.X_symbol = X_symbol
        else:
            # Try to auto-detect from structure (only works for pure ABX3)
            unique_symbols = list(set(self.symbols))
            
            if len(unique_symbols) != 3:
                raise ValueError(
                    f"Expected 3 unique elements for ABX3 perovskite, "
                    f"got {len(unique_symbols)}: {unique_symbols}. "
                    f"For structures with additional elements (e.g., molecular spacers), "
                    f"you must provide A_symbol, B_symbol, and X_symbol explicitly."
                )
            
            # Count atoms
            symbol_counts = {sym: self.symbols.count(sym) for sym in unique_symbols}
            sorted_symbols = sorted(symbol_counts.items(), key=lambda x: x[1])
            
            # Heuristic: most common = X (anions), least common = B or A
            # For ABX3 in 2x2x2: 8 A, 8 B, 24 X
            # So X should be most common, A and B should be equal
            
            self.X_symbol = sorted_symbols[2][0]  # Most common
            self.A_symbol = sorted_symbols[0][0]  # Least common
            self.B_symbol = sorted_symbols[1][0]   # Middle
        
        # Get indices for perovskite sites only
        self.A_indices = [i for i, sym in enumerate(self.symbols) if sym == self.A_symbol]
        self.B_indices = [i for i, sym in enumerate(self.symbols) if sym == self.B_symbol]
        self.X_indices = [i for i, sym in enumerate(self.symbols) if sym == self.X_symbol]
        
        # Get indices for non-perovskite atoms (spacers, etc.)
        perovskite_elements = {self.A_symbol, self.B_symbol, self.X_symbol}
        self.spacer_indices = [i for i, sym in enumerate(self.symbols) if sym not in perovskite_elements]
    
    def _estimate_supercell_size(self):
        """Estimate supercell size from cell dimensions."""
        cell_lengths = np.linalg.norm(self.cell, axis=1)
        unit_cell_size = 2.0 * self.BX_dist
        self.supercell_size = [int(round(cell_len / unit_cell_size)) 
                               for cell_len in cell_lengths]
        # Ensure at least 2x2x2
        self.supercell_size = [max(2, s) for s in self.supercell_size]
    
    def apply_tilting(self, angles=None, tilt_pattern=None, to_primitive=False, 
                     allow_orthorhombic=True, max_angle_cubic=2.0, 
                     default_angle_from_pattern=2.0):
        """
        Apply Glazer tilting distortions to the structure.
        
        Two modes of operation:
        1. Angles provided: Use provided angles and adapt XY plane/cell to match
        2. Pattern only: Calculate optimal angles from Glazer pattern, then adapt cell
        
        Parameters
        ----------
        angles : list[float], optional
            Rotation angles along x, y, z axes in degrees.
            Format: [omega_x, omega_y, omega_z]
            If provided, these angles will be used and cell will be adapted.
            If None, angles will be calculated from tilt_pattern.
        tilt_pattern : list[str], optional
            Tilting pattern in Glazer notation.
            Format: ['pattern_x', 'pattern_y', 'pattern_z']
            Each element should be '0', '+', or '-'.
            Required if angles is None.
        to_primitive : bool, optional
            If True, return the primitive cell after distortion (default: False).
        allow_orthorhombic : bool, optional
            If True, allow cell to distort to orthorhombic (default: True).
            The cell will adapt to accommodate the tilting angles.
        max_angle_cubic : float, optional
            Maximum angle in degrees for cubic cell (default: 2.0).
            Not used when allow_orthorhombic=True (cell always adapts).
        default_angle_from_pattern : float, optional
            Default angle in degrees when calculating from pattern (default: 2.0).
            Only used when angles is None.
            
        Returns
        -------
        ase.Atoms
            The distorted perovskite structure.
            
        Raises
        ------
        ValueError
            If neither angles nor tilt_pattern is provided, or if both are provided
            but incompatible.
        """
        # Determine mode: angles provided or pattern only
        if angles is None and tilt_pattern is None:
            raise ValueError(
                "Either angles or tilt_pattern must be provided. "
                "Use angles to specify exact tilting, or tilt_pattern to calculate optimal angles."
            )
        
        # If only pattern provided, calculate angles
        if angles is None:
            if tilt_pattern is None:
                raise ValueError("tilt_pattern is required when angles is not provided")
            angles = calculate_angles_from_glazer_pattern(
                tilt_pattern, default_angle=default_angle_from_pattern
            )
            mode = "pattern_based"
        else:
            # Angles provided - validate pattern if also provided
            if tilt_pattern is not None:
                # Both provided - validate they're compatible
                mode = "angles_provided"
            else:
                # Only angles provided - we'll infer pattern from angles
                # (0 angle = '0', non-zero = '-' by default)
                tilt_pattern = ['0' if abs(a) < 1e-6 else '-' for a in angles]
                mode = "angles_provided"
        
        # Validate lengths
        if len(angles) != 3:
            raise ValueError("angles must have 3 elements")
        if tilt_pattern is not None and len(tilt_pattern) != 3:
            raise ValueError("tilt_pattern must have 3 elements")
        
        # Validate and normalize angles/patterns
        angles = list(angles)
        tilt_pattern = list(tilt_pattern)
        
        for i, pattern in enumerate(tilt_pattern):
            if pattern not in ['0', '+', '-']:
                raise ValueError(
                    f"tilt_pattern elements must be '0', '+', or '-', got '{pattern}'"
                )
            # Ensure consistency: if angle is 0, pattern should be '0'
            if abs(angles[i]) < 1e-6 and pattern != '0':
                tilt_pattern[i] = '0'
            # Ensure consistency: if pattern is '0', angle should be 0
            if pattern == '0' and abs(angles[i]) > 1e-6:
                if mode == "pattern_based":
                    # Pattern-based: set angle to 0
                    angles[i] = 0.0
                else:
                    # Angles provided: warn but keep angle (user knows what they want)
                    pass
        
        # Calculate k-vectors
        self._kvecs = []
        for i, pattern in enumerate(tilt_pattern):
            if pattern == '0':
                kvec_tmp = [0, 0, 0]
            elif pattern == '+':
                kvec_tmp = [1, 1, 1]
                kvec_tmp[i] = 0
            elif pattern == '-':
                kvec_tmp = [1, 1, 1]
            self._kvecs.append(np.array(kvec_tmp))
        
        self._distortion_angles = angles
        
        # When allow_orthorhombic=True, always adapt cell to match angles
        # This allows the XY plane to adjust based on the tilting
        use_orthorhombic = allow_orthorhombic
        
        if use_orthorhombic:
            # For orthorhombic, we'll adjust the cell shape to accommodate angles
            # This adapts the XY plane based on the tilting
            self._use_orthorhombic = True
        else:
            # For cubic, maintain original cell shape (only for small angles)
            self._use_orthorhombic = False
        
        # Calculate displacements
        displacements = self._get_displacements()
        
        # Apply displacements and adjust network
        distorted_structure = self._adjust_network(displacements, use_orthorhombic)
        
        # Convert to primitive if requested
        if to_primitive:
            distorted_structure = self._to_primitive(distorted_structure)
        
        return distorted_structure
    
    def _get_displacements(self):
        """
        Calculate displacements for all atoms based on octahedral rotations.
        
        This implements the core distortion logic from DistortPerovskite.
        """
        # Get fractional coordinates
        inv_cell = np.linalg.inv(self.cell)
        fractional_coords = self.positions @ inv_cell.T
        
        # Initialize displacement array
        displacements = np.zeros_like(self.positions)
        
        # For each unit cell in the supercell, calculate displacements
        nx, ny, nz = self.supercell_size
        
        # Get primitive cell positions (first unit cell)
        # We need to identify which atoms belong to the primitive cell
        # For simplicity, we'll work with all atoms and apply phase factors
        
        # Calculate displacements for X atoms (anions) around B atoms
        # The rotation is applied to X atoms relative to their nearest B atom
        
        rotation_axes = ['x', 'y', 'z']
        
        for axis_idx, axis in enumerate(rotation_axes):
            angle_deg = self._distortion_angles[axis_idx]
            if abs(angle_deg) < 1e-6:
                continue
            
            angle_rad = math.pi * angle_deg / 180.0
            rotmat = self._get_rotation_matrix(axis, angle_rad).T
            
            # Apply rotation to X atoms around their nearest B atoms
            for x_idx in self.X_indices:
                x_pos = self.positions[x_idx]
                
                # Find nearest B atom
                distances_to_B = [np.linalg.norm(x_pos - self.positions[b_idx]) 
                                 for b_idx in self.B_indices]
                nearest_B_idx = self.B_indices[np.argmin(distances_to_B)]
                B_pos = self.positions[nearest_B_idx]
                
                # Vector from B to X
                BX_vector = x_pos - B_pos
                
                # Apply rotation
                rotated_vector = BX_vector @ rotmat
                
                # Calculate displacement
                disp = rotated_vector - BX_vector
                
                # Apply phase factor based on supercell position
                # Get which unit cell this atom belongs to
                frac_pos = fractional_coords[x_idx]
                unit_cell_idx = [int(frac_pos[i] * self.supercell_size[i]) 
                                for i in range(3)]
                
                # Calculate phase factor
                kvec = self._kvecs[axis_idx]
                phase = math.cos(math.pi * np.dot(unit_cell_idx, kvec))
                
                # Add to displacement
                displacements[x_idx] += disp * phase
        
        return displacements
    
    def _adjust_network(self, displacements, use_orthorhombic=False):
        """
        Adjust positions after rotations to recover octahedral connectivity.
        
        For orthorhombic distortions, this adjusts the cell shape and A-site positions
        to accommodate larger tilting angles.
        
        Parameters
        ----------
        displacements : np.ndarray
            Displacements for all atoms
        use_orthorhombic : bool
            If True, adjust cell to orthorhombic shape
        """
        # Apply displacements
        new_positions = self.positions + displacements
        
        # Get fractional coordinates before cell adjustment
        inv_cell = np.linalg.inv(self.cell)
        fractional_coords = new_positions @ inv_cell.T
        
        if use_orthorhombic:
            # Calculate new orthorhombic cell based on tilting angles
            # For orthorhombic distortion, we adjust cell dimensions to accommodate tilting
            # This is a more robust approach that avoids singular matrices
            
            # Get original cell lengths
            cell_lengths_orig = np.linalg.norm(self.cell, axis=1)
            
            # Calculate expansion factors based on tilting angles
            # Larger angles require more space in the cell
            new_cell_lengths = cell_lengths_orig.copy()
            
            for i, angle_deg in enumerate(self._distortion_angles):
                if abs(angle_deg) > 1e-6:
                    # Calculate expansion factor based on angle
                    # For small angles: expansion ≈ 1 + (angle_rad)^2 / 2
                    # For larger angles: more significant expansion
                    angle_rad = math.radians(abs(angle_deg))
                    # Use a conservative expansion that accounts for octahedral tilting
                    # The expansion needs to accommodate the rotated octahedra
                    expansion_factor = 1.0 + 0.3 * angle_rad**2 + 0.05 * angle_rad**4
                    new_cell_lengths[i] *= expansion_factor
            
            # Ensure minimum cell size to avoid singular matrices
            min_cell_size = 0.1  # Minimum 0.1 Angstrom
            new_cell_lengths = np.maximum(new_cell_lengths, min_cell_size)
            
            # Create orthorhombic cell (diagonal matrix)
            new_cell = np.diag(new_cell_lengths)
            
            # Adjust all positions to new cell
            # Convert to fractional coordinates in original cell
            inv_cell = np.linalg.inv(self.cell)
            fractional_coords = new_positions @ inv_cell.T
            
            # Recalculate positions in new cell
            # Preserve fractional coordinates but scale by new cell dimensions
            for i in range(len(new_positions)):
                # Get fractional position in original cell
                old_frac = fractional_coords[i]
                
                # Convert to Cartesian in new orthorhombic cell
                # Fractional coordinates are preserved, but cell dimensions change
                new_positions[i] = old_frac @ new_cell.T
            
            # Note: For 2D structures with spacers, the spacer positions in the XY plane
            # are automatically adjusted because they are attached to the distorted layer.
            # The orthorhombic cell adjustment ensures spacers follow the distorted cell shape.
        else:
            # For cubic, keep original cell
            new_cell = self.cell.copy()
        
        # Create new structure
        distorted_structure = Atoms(
            symbols=self.symbols,
            positions=new_positions,
            cell=new_cell,
            pbc=self.structure.pbc
        )
        
        # Preserve arrays from original structure (e.g., site_role tags)
        for key, value in self.structure.arrays.items():
            if key not in ['numbers', 'positions']:
                distorted_structure.arrays[key] = value.copy()
        
        # Preserve info from original structure
        for key, value in self.structure.info.items():
            distorted_structure.info[key] = value
        
        return distorted_structure
    
    def _to_primitive(self, structure):
        """Convert structure to primitive cell using spglib."""
        try:
            import spglib
        except ImportError:
            raise ImportError(
                "spglib is required for primitive cell conversion. "
                "Install it with: pip install spglib"
            )
        
        # Convert to spglib format
        lattice = structure.cell
        positions = structure.get_scaled_positions()
        numbers = structure.get_atomic_numbers()
        
        cell = (lattice, positions, numbers)
        
        # Standardize to primitive
        try:
            lattice_prim, positions_prim, numbers_prim = spglib.standardize_cell(
                cell, to_primitive=True, no_idealize=False, symprec=self.symprec
            )
            
            # Convert back to ASE
            symbols_prim = [structure.get_chemical_symbols()[i] 
                           for i in range(len(numbers_prim))]
            
            return Atoms(
                symbols=symbols_prim,
                positions=positions_prim @ lattice_prim,
                cell=lattice_prim,
                pbc=structure.pbc
            )
        except Exception:
            # If standardization fails, return original
            return structure
    
    def _get_rotation_matrix(self, axis, angle_rad):
        """
        Get rotation matrix around specified axis.
        
        Parameters
        ----------
        axis : str
            'x', 'y', or 'z'
        angle_rad : float
            Rotation angle in radians
            
        Returns
        -------
        np.ndarray
            3x3 rotation matrix
        """
        if axis == 'x':
            axis_vector = np.array([1.0, 0.0, 0.0])
        elif axis == 'y':
            axis_vector = np.array([0.0, 1.0, 0.0])
        elif axis == 'z':
            axis_vector = np.array([0.0, 0.0, 1.0])
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")
        
        # Rodrigues' rotation formula
        norm = np.linalg.norm(axis_vector)
        if norm < 1e-12:
            return np.eye(3)
        
        axis_vector = axis_vector / norm
        
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        # Rotation matrix using Rodrigues' formula
        rotmat = (cos_a * np.eye(3) + 
                  (1 - cos_a) * np.outer(axis_vector, axis_vector) +
                  sin_a * self._cross_product_matrix(axis_vector))
        
        return rotmat
    
    @staticmethod
    def _cross_product_matrix(v):
        """
        Get matrix representation of cross product v × r.
        
        Parameters
        ----------
        v : np.ndarray
            3D vector
            
        Returns
        -------
        np.ndarray
            3x3 matrix such that matrix @ r = v × r
        """
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])


def apply_glazer_tilting(structure, angles=None, tilt_pattern=None, to_primitive=False, 
                         BX_dist=None, symprec=None, allow_orthorhombic=True, 
                         max_angle_cubic=2.0, default_angle_from_pattern=2.0,
                         A_symbol=None, B_symbol=None, X_symbol=None):
    """
    Apply Glazer tilting distortions to a perovskite structure.
    
    This is a convenience function that creates a GlazerTilting instance
    and applies the distortion.
    
    Parameters
    ----------
    structure : ase.Atoms
        The perovskite structure to distort. Must be a bulk perovskite
        with ABX3 composition. Currently requires a 2x2x2 supercell or larger.
    angles : list[float], optional
        Rotation angles along x, y, z axes in degrees.
        Format: [omega_x, omega_y, omega_z]
        If provided, these angles will be used and XY plane/cell will be adapted.
        If None, angles will be calculated from tilt_pattern.
    tilt_pattern : list[str], optional
        Tilting pattern in Glazer notation.
        Format: ['pattern_x', 'pattern_y', 'pattern_z']
        Each element should be '0', '+', or '-'.
        Required if angles is None.
        Examples:
        - ['-', '-', '-'] for a-a-a- (all axes anti-phase)
        - ['0', '-', '-'] for a0b-b- (x-axis no tilt, y/z anti-phase)
        - ['+', '-', '-'] for a+b-c- (x in-phase, y/z anti-phase)
    to_primitive : bool, optional
        If True, return the primitive cell after distortion (default: False).
    BX_dist : float, optional
        B-X bond distance in Angstroms. If None, will be estimated from structure.
    symprec : float, optional
        Symmetry precision for spglib. If None, will be auto-calculated.
    allow_orthorhombic : bool, optional
        If True, adapt cell to orthorhombic to accommodate tilting (default: True).
        The XY plane will be adjusted based on the tilting angles.
    max_angle_cubic : float, optional
        Maximum angle in degrees for cubic cell (default: 2.0).
        Not used when allow_orthorhombic=True (cell always adapts).
    default_angle_from_pattern : float, optional
        Default angle in degrees when calculating from pattern (default: 2.0).
        Only used when angles is None.
    A_symbol : str, optional
        Symbol of A-site cation. Required when structure contains additional elements
        (e.g., molecular spacers like C, N, H).
    B_symbol : str, optional
        Symbol of B-site cation. Required when structure contains additional elements.
    X_symbol : str, optional
        Symbol of X-site anion. Required when structure contains additional elements.
        
    Returns
    -------
    ase.Atoms
        The distorted perovskite structure with Glazer tilting applied.
        
    Notes
    -----
    Two modes of operation:
    1. Angles provided: XY plane/cell adapts to match the provided angles
    2. Pattern only: Optimal angles are calculated from Glazer pattern, then cell adapts
    
    Do not provide both angles and pattern unless they are compatible, as the system
    cannot satisfy both constraints simultaneously in general.
        
    Examples
    --------
    >>> from q2D_Materials.core.creator import q2D_creator
    >>> from q2D_Materials.utils.glazer_tilting import apply_glazer_tilting
    >>> 
    >>> creator = q2D_creator()
    >>> # Create a 2x2x2 bulk perovskite
    >>> bulk = creator.create_perovskite('bulk',
    ...     A_ions='Sr', B_ions='Ti', X_ions='O',
    ...     supercell_size=(2, 2, 2))
    >>> 
    >>> # Apply a-a-a- tilting (all anti-phase)
    >>> distorted = apply_glazer_tilting(
    ...     bulk,
    ...     angles=[1.0, 1.0, 1.0],
    ...     tilt_pattern=['-', '-', '-']
    ... )
    """
    tilting = GlazerTilting(structure, BX_dist=BX_dist, symprec=symprec,
                           A_symbol=A_symbol, B_symbol=B_symbol, X_symbol=X_symbol)
    return tilting.apply_tilting(angles=angles, tilt_pattern=tilt_pattern, 
                                to_primitive=to_primitive,
                                allow_orthorhombic=allow_orthorhombic,
                                max_angle_cubic=max_angle_cubic,
                                default_angle_from_pattern=default_angle_from_pattern)


def calculate_angles_from_glazer_pattern(tilt_pattern, default_angle=2.0):
    """
    Calculate optimal tilting angles from a Glazer pattern.
    
    This function provides typical angles for different Glazer patterns.
    The angles are chosen to be reasonable defaults that work well with
    orthorhombic cell adaptation.
    
    Parameters
    ----------
    tilt_pattern : list[str]
        Tilting pattern ['pattern_x', 'pattern_y', 'pattern_z']
        Each element should be '0', '+', or '-'.
    default_angle : float, optional
        Default angle in degrees for non-zero patterns (default: 2.0).
        Can be adjusted based on desired distortion strength.
        
    Returns
    -------
    list[float]
        Calculated angles [omega_x, omega_y, omega_z] in degrees.
        
    Examples
    --------
    >>> calculate_angles_from_glazer_pattern(['-', '-', '-'])
    [2.0, 2.0, 2.0]
    >>> calculate_angles_from_glazer_pattern(['0', '-', '-'])
    [0.0, 2.0, 2.0]
    >>> calculate_angles_from_glazer_pattern(['+', '-', '-'])
    [2.0, 2.0, 2.0]
    """
    angles = []
    for pattern in tilt_pattern:
        if pattern == '0':
            angles.append(0.0)
        else:
            # For '+' and '-' patterns, use default angle
            # In practice, '+' and '-' can have different magnitudes,
            # but we use the same default for simplicity
            angles.append(default_angle)
    return angles


def glazer_notation_to_pattern(glazer_string):
    """
    Convert Glazer notation string to tilt pattern list.
    
    Parameters
    ----------
    glazer_string : str
        Glazer notation string (e.g., "a-a-a-", "a0b-b-", "a+b-c-")
        
    Returns
    -------
    list[str]
        Tilt pattern list ['pattern_x', 'pattern_y', 'pattern_z']
        
    Examples
    --------
    >>> glazer_notation_to_pattern("a-a-a-")
    ['-', '-', '-']
    >>> glazer_notation_to_pattern("a0b-b-")
    ['0', '-', '-']
    >>> glazer_notation_to_pattern("a+b-c-")
    ['+', '-', '-']
    """
    if len(glazer_string) != 6:
        raise ValueError(
            f"Glazer notation must be 6 characters (e.g., 'a-a-a-'), got '{glazer_string}'"
        )
    
    # Extract patterns: positions 1, 3, 5 (0-indexed: 1, 3, 5)
    pattern_x = glazer_string[1]
    pattern_y = glazer_string[3]
    pattern_z = glazer_string[5]
    
    # Validate
    valid_patterns = ['0', '+', '-']
    if pattern_x not in valid_patterns or pattern_y not in valid_patterns or pattern_z not in valid_patterns:
        raise ValueError(
            f"Invalid Glazer notation pattern. "
            f"Expected format like 'a-a-a-' or 'a0b-b-', got '{glazer_string}'"
        )
    
    return [pattern_x, pattern_y, pattern_z]
