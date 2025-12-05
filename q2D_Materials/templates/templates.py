"""
Universal geometry templates for perovskite structures.

This module provides pure, declarative templates that define the geometric
arrangement of A, B, and X sites in fractional coordinates.
"""

import numpy as np
from abc import ABC, abstractmethod


class Template(ABC):
    """
    Base class for perovskite geometry templates.
    
    Templates define fractional positions for A, B, and X sites in a unit cell.
    They are pure geometry - no chemical species or lattice parameters.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Template name identifier."""
        pass
    
    @property
    @abstractmethod
    def dim(self) -> str:
        """Dimensionality: '3D' or '2D'."""
        pass
    
    @abstractmethod
    def get_position_matrix(self, n_layers: int = 1) -> dict:
        """
        Get position matrix for this template.
        
        Parameters
        ----------
        n_layers : int, optional
            Number of octahedral layers (for 2D templates, default: 1)
            
        Returns
        -------
        dict
            Dictionary with keys 'A', 'B', 'X' containing lists of fractional positions.
            Each position is [x, y, z] in fractional coordinates.
        """
        pass


class CubicTemplate(Template):
    """
    Template for cubic bulk perovskite (ABX₃).
    
    This template defines the standard cubic perovskite unit cell with:
    - A-site at [0, 0, 0]
    - B-site at [0.5, 0.5, 0.5]
    - X-sites at [0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]
    """
    
    @property
    def name(self) -> str:
        return "cubic"
    
    @property
    def dim(self) -> str:
        return "3D"
    
    def get_position_matrix(self, n_layers: int = 1) -> dict:
        """
        Return dimensionless fractional positions for the cubic topology.
        
        This is pure geometry (no distances, no chemistry). Positions can be
        scaled, stacked, or transformed by the builder using BX_dist.
        """
        return {
            'A': [[0.0, 0.0, 0.0]],
            'B': [[0.5, 0.5, 0.5]],
            'X': [[0.5, 0.5, 0.0], [0.5, 0.0, 0.5], [0.0, 0.5, 0.5]]
        }


class ReducedTemplate(Template):
    """
    Template for reduced 2D perovskite layers (RP/DJ/monolayer base).
    
    This template defines the octahedral layer structure used for 2D perovskites.
    It includes:
    - Initial X atoms at z=0
    - B and X atoms in octahedral layers
    - A-site positions between layers (if n_layers > 1)
    """
    
    @property
    def name(self) -> str:
        return "reduced"
    
    @property
    def dim(self) -> str:
        return "2D"
    
    def get_position_matrix(self, n_layers: int = 1) -> dict:
        """
        Return dimensionless fractional positions for reduced 2D perovskite layers.
        
        This is a geometric blueprint: builder assigns scale, z-spacing, and
        A/Ap classification.
        """
        # Initial X atoms at z=0
        positions = {
            'X': [[0.25, 0.25, 0.0], [0.75, 0.75, 0.0]],
            'B': [],
            'A': []
        }
        
        # Base positions for each layer (8 atoms per layer)
        # Pattern: [X, X, B, X, X, B, X, X]
        base_positions = [
            [0, 0, 0.5],        # X (index 0)
            [0.5, 0, 0.5],      # X (index 1)
            [0.25, 0.25, 0.5],  # B (index 2)
            [0, 0.5, 0.5],      # X (index 3)
            [0.5, 0.5, 0.5],    # X (index 4)
            [0.75, 0.75, 0.5],  # B (index 5)
            [0.25, 0.25, 1.0],  # X (index 6)
            [0.75, 0.75, 1.0]   # X (index 7)
        ]
        
        # Add positions for each layer
        for i in range(n_layers):
            z_shift = float(i)
            for j, base_pos in enumerate(base_positions):
                pos = [base_pos[0], base_pos[1], base_pos[2] + z_shift]
                if j in (2, 5):  # B-site positions
                    positions['B'].append(pos)
                else:  # X-site positions
                    positions['X'].append(pos)
        
        # A-like positions at every layer boundary (z=0..n_layers) so stacking
        # repeats MOL–Perov–MOL with no gaps between periodic images.
        # A-sites on every integer z boundary for seamless stacking across PBC
        positions['A'].extend(
            [[0.25, 0.75, float(i)], [0.75, 0.25, float(i)]]
            for i in range(n_layers + 1)
        )
        # Flatten the list of lists produced above
        positions['A'] = [item for sub in positions['A'] for item in sub]
        
        return positions

