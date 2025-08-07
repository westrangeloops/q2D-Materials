"""
Vector Analysis Module for q2D Materials.

This module analyzes planes defined by terminal halogen atoms and calculates
angles between these planes and relative to the crystal z-axis.
"""

import numpy as np
from ase import Atoms


class VectorAnalyzer:
    """
    Analyzer for vector and plane calculations in salt structures.
    
    Analyzes planes defined by terminal halogen atoms and calculates:
    - Angle between two planes
    - Angles of each plane relative to z-axis
    - Distance between plane centers
    """
    
    def __init__(self):
        """Initialize the VectorAnalyzer."""
        pass
    
    def create_plane_from_vectors(self, p1, p2, z_axis):
        """
        Calculate a plane normal vector from two points and a Z-axis vector.
        
        Parameters:
            p1 (array): First 3D point [x, y, z]
            p2 (array): Second 3D point [x, y, z]
            z_axis (array): Z-axis vector [x, y, z]
            
        Returns:
            numpy.array: Normal vector of the plane
        """
        p1 = np.array(p1)
        p2 = np.array(p2)
        z_axis = np.array(z_axis)
        
        # Calculate the vector between the two points
        couple_vector = p2 - p1
        
        # Calculate the cross product of the couple_vector and the z_axis vector
        result_vector = np.cross(couple_vector, z_axis)
        
        # Generate the plane normal from the two vectors
        plane_normal = np.cross(result_vector, couple_vector)
        
        return plane_normal
    
    def analyze_salt_structure_vectors(self, salt_structure, x_symbol='Cl', analyzer=None):
        """
        Analyze vector relationships in salt structure using terminal halogen atoms.
        
        Parameters:
            salt_structure (ase.Atoms): ASE Atoms object with salt structure
            x_symbol (str): Chemical symbol of the halogen atoms (default: 'Cl') - will be auto-detected if analyzer provided
            analyzer: q2D_analyzer instance for accessing connectivity analysis
            
        Returns:
            dict: Vector analysis results
        """
        # If analyzer is provided, use the original structure and connectivity analysis
        if analyzer is not None:
            return self._analyze_original_structure_vectors(analyzer)
        
        # Legacy method using salt structure (kept for backward compatibility)
        if salt_structure is None or len(salt_structure) == 0:
            return {
                'error': 'No salt structure available for vector analysis'
            }
        
        # Get positions and cell information
        positions = salt_structure.get_positions()
        symbols = salt_structure.get_chemical_symbols()
        cell = salt_structure.get_cell()
        
        # Find all halogen atoms and sort by z-coordinate
        halogen_indices = [i for i, symbol in enumerate(symbols) if symbol == x_symbol]
        
        if len(halogen_indices) < 4:
            return {
                'error': f'Need at least 4 {x_symbol} atoms for plane analysis, found {len(halogen_indices)}'
            }
        
        # Get halogen positions and sort by z-coordinate
        halogen_positions = positions[halogen_indices]
        z_coords = halogen_positions[:, 2]
        sorted_indices = np.argsort(z_coords)
        halogen_positions_sorted = halogen_positions[sorted_indices]
        
        # Get the z-axis vector from the cell
        z_axis = cell[2]
        z_axis_normalized = z_axis / np.linalg.norm(z_axis)
        
        # Define planes using the lowest and highest pairs of halogen atoms
        # Low plane: two atoms with lowest z-coordinates
        cl_low_1 = halogen_positions_sorted[0]
        cl_low_2 = halogen_positions_sorted[1]
        
        # High plane: two atoms with highest z-coordinates
        cl_high_1 = halogen_positions_sorted[-2]
        cl_high_2 = halogen_positions_sorted[-1]
        
        # Calculate plane normal vectors
        plane_normal_low = self.create_plane_from_vectors(cl_low_1, cl_low_2, z_axis)
        plane_normal_high = self.create_plane_from_vectors(cl_high_1, cl_high_2, z_axis)
        
        # Normalize the plane vectors
        normal_vector_low = plane_normal_low / np.linalg.norm(plane_normal_low)
        normal_vector_high = plane_normal_high / np.linalg.norm(plane_normal_high)
        
        # Calculate plane centers
        plane_center_low = (cl_low_1 + cl_low_2) / 2
        plane_center_high = (cl_high_1 + cl_high_2) / 2
        
        # Calculate angle between the two planes
        dot_product = np.dot(normal_vector_low, normal_vector_high)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Calculate angles between each plane and z-axis
        angle_z_axis_low = np.arccos(np.clip(np.dot(z_axis_normalized, normal_vector_low), -1.0, 1.0))
        angle_z_axis_deg_low = np.degrees(angle_z_axis_low)
        
        angle_z_axis_high = np.arccos(np.clip(np.dot(z_axis_normalized, normal_vector_high), -1.0, 1.0))
        angle_z_axis_deg_high = np.degrees(angle_z_axis_high)
        
        # Calculate distance between plane centers
        plane_separation = np.linalg.norm(plane_center_high - plane_center_low)
        
        # Calculate distances between halogen atoms in each plane
        vector_low = cl_low_2 - cl_low_1
        vector_high = cl_high_2 - cl_high_1
        distance_low = np.linalg.norm(vector_low)
        distance_high = np.linalg.norm(vector_high)
        
        return {
            'plane_analysis': {
                'low_plane': {
                    'normal_vector': normal_vector_low.tolist(),
                    'center_position': plane_center_low.tolist(),
                    'angle_vs_z_axis_degrees': float(angle_z_axis_deg_low)
                },
                'high_plane': {
                    'normal_vector': normal_vector_high.tolist(),
                    'center_position': plane_center_high.tolist(),
                    'angle_vs_z_axis_degrees': float(angle_z_axis_deg_high)
                }
            },
            'vector_analysis_results': {
                'angle_between_planes_degrees': float(angle_deg),
                'distance_between_plane_centers_angstrom': float(plane_separation),
                'angle_between_low_plane_and_z': float(angle_z_axis_deg_low),
                'angle_between_high_plane_and_z': float(angle_z_axis_deg_high)
            }
        }
    
    def _analyze_original_structure_vectors(self, analyzer):
        """
        Analyze vector relationships using terminal halogen atoms from the original structure.
        Uses connectivity analysis to identify terminal atoms correctly.
        
        Parameters:
            analyzer: q2D_analyzer instance with connectivity analysis
            
        Returns:
            dict: Vector analysis results
        """
        if not hasattr(analyzer, 'ontology') or not analyzer.ontology:
            return {
                'error': 'No ontology available for connectivity analysis'
            }
        
        # Get connectivity analysis from ontology
        connectivity_analysis = analyzer.ontology.get('connectivity_analysis', {})
        terminal_axial_atoms = connectivity_analysis.get('terminal_axial_atoms', {})
        shared_atoms = connectivity_analysis.get('shared_atoms', {})
        
        # Get original structure data
        positions = analyzer.atoms.get_positions()
        symbols = analyzer.atoms.get_chemical_symbols()
        cell = analyzer.atoms.get_cell()
        
        # Auto-detect halogen symbol from analyzer
        x_symbol = analyzer.x
        
        # Collect all terminal halogen atoms
        terminal_halogen_indices = []
        
        # Method 1: Check terminal axial atoms first
        for oct_key, oct_data in terminal_axial_atoms.items():
            terminal_atoms = oct_data.get('terminal_axial_atoms', [])
            for terminal_atom in terminal_atoms:
                atom_index = terminal_atom['atom_index']
                if symbols[atom_index] == x_symbol:
                    terminal_halogen_indices.append(atom_index)
        
        # Method 2: If no terminal axial atoms, find terminal equatorial atoms
        if len(terminal_halogen_indices) < 4:
            # Get all octahedra data to find equatorial atoms
            octahedra = analyzer.ontology.get('octahedra', {})
            shared_atom_indices = set(int(k) for k in shared_atoms.keys())
            
            for oct_key, oct_data in octahedra.items():
                # Get equatorial atoms
                equatorial_indices = oct_data.get('ligand_atoms', {}).get('equatorial_global_indices', [])
                
                # Find equatorial atoms that are not shared (terminal)
                for eq_idx in equatorial_indices:
                    if eq_idx not in shared_atom_indices and symbols[eq_idx] == x_symbol:
                        terminal_halogen_indices.append(eq_idx)
        
        # Method 3: If still not enough, find any non-shared halogen atoms
        if len(terminal_halogen_indices) < 4:
            shared_atom_indices = set(int(k) for k in shared_atoms.keys())
            
            for i, symbol in enumerate(symbols):
                if symbol == x_symbol and i not in shared_atom_indices:
                    terminal_halogen_indices.append(i)
        
        # Remove duplicates and ensure we have unique indices
        terminal_halogen_indices = list(set(terminal_halogen_indices))
        
        if len(terminal_halogen_indices) < 4:
            return {
                'error': f'Need at least 4 terminal {x_symbol} atoms for plane analysis, found {len(terminal_halogen_indices)}. '
                        f'Shared atoms: {len(shared_atoms)}, Total {x_symbol} atoms: {symbols.count(x_symbol)}'
            }
        
        # Get terminal halogen positions and sort by z-coordinate
        terminal_positions = positions[terminal_halogen_indices]
        z_coords = terminal_positions[:, 2]
        sorted_indices = np.argsort(z_coords)
        terminal_positions_sorted = terminal_positions[sorted_indices]
        terminal_indices_sorted = [terminal_halogen_indices[i] for i in sorted_indices]
        
        # Get the z-axis vector from the cell
        z_axis = cell[2]
        z_axis_normalized = z_axis / np.linalg.norm(z_axis)
        
        # Define planes using the lowest and highest pairs of terminal halogen atoms
        # Low plane: two atoms with lowest z-coordinates
        cl_low_1 = terminal_positions_sorted[0]
        cl_low_2 = terminal_positions_sorted[1]
        low_indices = [terminal_indices_sorted[0], terminal_indices_sorted[1]]
        
        # High plane: two atoms with highest z-coordinates
        cl_high_1 = terminal_positions_sorted[-2]
        cl_high_2 = terminal_positions_sorted[-1]
        high_indices = [terminal_indices_sorted[-2], terminal_indices_sorted[-1]]
        
        # Calculate plane normal vectors
        plane_normal_low = self.create_plane_from_vectors(cl_low_1, cl_low_2, z_axis)
        plane_normal_high = self.create_plane_from_vectors(cl_high_1, cl_high_2, z_axis)
        
        # Normalize the plane vectors
        normal_vector_low = plane_normal_low / np.linalg.norm(plane_normal_low)
        normal_vector_high = plane_normal_high / np.linalg.norm(plane_normal_high)
        
        # Calculate plane centers
        plane_center_low = (cl_low_1 + cl_low_2) / 2
        plane_center_high = (cl_high_1 + cl_high_2) / 2
        
        # Calculate angle between the two planes
        dot_product = np.dot(normal_vector_low, normal_vector_high)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        
        # Calculate angles between each plane and z-axis
        angle_z_axis_low = np.arccos(np.clip(np.dot(z_axis_normalized, normal_vector_low), -1.0, 1.0))
        angle_z_axis_deg_low = np.degrees(angle_z_axis_low)
        
        angle_z_axis_high = np.arccos(np.clip(np.dot(z_axis_normalized, normal_vector_high), -1.0, 1.0))
        angle_z_axis_deg_high = np.degrees(angle_z_axis_high)
        
        # Calculate distance between plane centers
        plane_separation = np.linalg.norm(plane_center_high - plane_center_low)
        
        return {
            'terminal_atom_analysis': {
                'total_terminal_atoms_found': len(terminal_halogen_indices),
                'terminal_atom_indices': terminal_halogen_indices,
                'low_plane_atom_indices': low_indices,
                'high_plane_atom_indices': high_indices,
                'halogen_symbol': x_symbol
            },
            'plane_analysis': {
                'low_plane': {
                    'normal_vector': normal_vector_low.tolist(),
                    'center_position': plane_center_low.tolist(),
                    'angle_vs_z_axis_degrees': float(angle_z_axis_deg_low),
                    'atom_indices': low_indices
                },
                'high_plane': {
                    'normal_vector': normal_vector_high.tolist(),
                    'center_position': plane_center_high.tolist(),
                    'angle_vs_z_axis_degrees': float(angle_z_axis_deg_high),
                    'atom_indices': high_indices
                }
            },
            'vector_analysis_results': {
                'angle_between_planes_degrees': float(angle_deg),
                'distance_between_plane_centers_angstrom': float(plane_separation),
                'angle_between_low_plane_and_z': float(angle_z_axis_deg_low),
                'angle_between_high_plane_and_z': float(angle_z_axis_deg_high)
            }
        }
    
    def create_interactive_plot(self, salt_structure, vector_analysis_results, output_filename='interactive_vector_analysis.html', analyzer=None):
        """
        Create an interactive Plotly visualization of the vector analysis.
        
        Parameters:
            salt_structure (ase.Atoms): Salt structure with halogen atoms (may be None)
            vector_analysis_results (dict): Results from analyze_salt_structure_vectors
            output_filename (str): Output HTML filename
            analyzer: q2D_analyzer instance for accessing original structure
            
        Returns:
            str: Path to created HTML file or error message
        """
        if 'error' in vector_analysis_results:
            return f"Vector Analysis Error: {vector_analysis_results['error']}"
        
        try:
            import plotly.graph_objects as go
            import plotly.express as px
        except ImportError:
            return "Error: Plotly not available. Install with: pip install plotly"
        
        # Use original structure if analyzer is provided and has terminal atom analysis
        if analyzer is not None and 'terminal_atom_analysis' in vector_analysis_results:
            return self._create_plot_with_original_structure(analyzer, vector_analysis_results, output_filename)
        
        # Legacy method using salt structure (kept for backward compatibility)
        if salt_structure is None or len(salt_structure) == 0:
            return "Error: No salt structure available for visualization"
        
        # Get structure data
        positions = salt_structure.get_positions()
        symbols = salt_structure.get_chemical_symbols()
        cell = salt_structure.get_cell()
        
        # Get analysis results
        results = vector_analysis_results['vector_analysis_results']
        plane_data = vector_analysis_results['plane_analysis']
        
        # Extract halogen symbol from the first halogen found
        halogen_symbol = None
        for symbol in symbols:
            if symbol in ['F', 'Cl', 'Br', 'I']:
                halogen_symbol = symbol
                break
        
        if halogen_symbol is None:
            return "Error: No halogen atoms found in salt structure"
        
        # Find halogen atoms and sort by z-coordinate
        halogen_indices = [i for i, symbol in enumerate(symbols) if symbol == halogen_symbol]
        if len(halogen_indices) < 4:
            return f"Error: Need at least 4 {halogen_symbol} atoms for visualization"
        
        halogen_positions = positions[halogen_indices]
        z_coords = halogen_positions[:, 2]
        sorted_indices = np.argsort(z_coords)
        halogen_positions_sorted = halogen_positions[sorted_indices]
        
        # Get plane data
        cl_low_1 = halogen_positions_sorted[0]
        cl_low_2 = halogen_positions_sorted[1]
        cl_high_1 = halogen_positions_sorted[-2]
        cl_high_2 = halogen_positions_sorted[-1]
        
        # Get z-axis and plane centers
        z_axis = cell[2]
        z_axis_normalized = z_axis / np.linalg.norm(z_axis)
        plane_center_low = np.array(plane_data['low_plane']['center_position'])
        plane_center_high = np.array(plane_data['high_plane']['center_position'])
        normal_vector_low = np.array(plane_data['low_plane']['normal_vector'])
        normal_vector_high = np.array(plane_data['high_plane']['normal_vector'])
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Color mapping for atoms
        colors = {'Cl': 'green', 'Br': 'brown', 'I': 'purple', 'F': 'cyan', 'Na': 'yellow', 'other': 'gray'}
        unique_symbols = list(set(symbols))
        
        # Group atoms by type for better legend handling
        for symbol in unique_symbols:
            mask = np.array(symbols) == symbol
            atom_positions = positions[mask]
            color = colors.get(symbol, colors['other'])
            
            fig.add_trace(go.Scatter3d(
                x=atom_positions[:, 0],
                y=atom_positions[:, 1],
                z=atom_positions[:, 2],
                mode='markers',
                marker=dict(size=8, color=color, opacity=0.8),
                name=f'{symbol} atoms',
                hovertemplate=f'{symbol}<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
            ))
        
        # Highlight the halogen atoms used for planes
        fig.add_trace(go.Scatter3d(
            x=[cl_low_1[0]], y=[cl_low_1[1]], z=[cl_low_1[2]],
            mode='markers', marker=dict(size=15, color='red', symbol='circle'),
            name=f'{halogen_symbol}1 (low plane)',
            hovertemplate=f'{halogen_symbol}1 (low plane)<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[cl_low_2[0]], y=[cl_low_2[1]], z=[cl_low_2[2]],
            mode='markers', marker=dict(size=15, color='darkred', symbol='circle'),
            name=f'{halogen_symbol}2 (low plane)',
            hovertemplate=f'{halogen_symbol}2 (low plane)<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[cl_high_1[0]], y=[cl_high_1[1]], z=[cl_high_1[2]],
            mode='markers', marker=dict(size=15, color='blue', symbol='circle'),
            name=f'{halogen_symbol}3 (high plane)',
            hovertemplate=f'{halogen_symbol}3 (high plane)<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[cl_high_2[0]], y=[cl_high_2[1]], z=[cl_high_2[2]],
            mode='markers', marker=dict(size=15, color='darkblue', symbol='circle'),
            name=f'{halogen_symbol}4 (high plane)',
            hovertemplate=f'{halogen_symbol}4 (high plane)<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
        ))
        
        # Draw vectors between halogen atoms
        fig.add_trace(go.Scatter3d(
            x=[cl_low_1[0], cl_low_2[0]], y=[cl_low_1[1], cl_low_2[1]], z=[cl_low_1[2], cl_low_2[2]],
            mode='lines', line=dict(color='red', width=8), name='Low plane vector',
            hovertemplate='Low plane vector<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[cl_high_1[0], cl_high_2[0]], y=[cl_high_1[1], cl_high_2[1]], z=[cl_high_1[2], cl_high_2[2]],
            mode='lines', line=dict(color='blue', width=8), name='High plane vector',
            hovertemplate='High plane vector<extra></extra>'
        ))
        
        # Create plane surfaces
        plane_size = np.max(np.max(positions, axis=0) - np.min(positions, axis=0)) * 0.8
        u = np.linspace(-plane_size/2, plane_size/2, 20)
        v = np.linspace(-plane_size/2, plane_size/2, 20)
        U, V = np.meshgrid(u, v)
        
        # Low plane
        v1_low = cl_low_2 - cl_low_1
        v1_low = v1_low / np.linalg.norm(v1_low)
        v2_low = np.cross(normal_vector_low, v1_low)
        v2_low = v2_low / np.linalg.norm(v2_low)
        
        plane_points_low = plane_center_low[:, np.newaxis, np.newaxis] + U[np.newaxis, :, :] * v1_low[:, np.newaxis, np.newaxis] + V[np.newaxis, :, :] * v2_low[:, np.newaxis, np.newaxis]
        
        fig.add_trace(go.Surface(
            x=plane_points_low[0], y=plane_points_low[1], z=plane_points_low[2],
            opacity=0.3, colorscale=[[0, 'yellow'], [1, 'yellow']], showscale=False,
            name='Low Plane', hovertemplate='Low Plane<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
        ))
        
        # High plane
        v1_high = cl_high_2 - cl_high_1
        v1_high = v1_high / np.linalg.norm(v1_high)
        v2_high = np.cross(normal_vector_high, v1_high)
        v2_high = v2_high / np.linalg.norm(v2_high)
        
        plane_points_high = plane_center_high[:, np.newaxis, np.newaxis] + U[np.newaxis, :, :] * v1_high[:, np.newaxis, np.newaxis] + V[np.newaxis, :, :] * v2_high[:, np.newaxis, np.newaxis]
        
        fig.add_trace(go.Surface(
            x=plane_points_high[0], y=plane_points_high[1], z=plane_points_high[2],
            opacity=0.3, colorscale=[[0, 'lightblue'], [1, 'lightblue']], showscale=False,
            name='High Plane', hovertemplate='High Plane<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
        ))
        
        # Draw normal vectors
        normal_scale = plane_size * 0.3
        
        # Low plane normal vector
        normal_end_low = plane_center_low + normal_vector_low * normal_scale
        fig.add_trace(go.Scatter3d(
            x=[plane_center_low[0], normal_end_low[0]],
            y=[plane_center_low[1], normal_end_low[1]],
            z=[plane_center_low[2], normal_end_low[2]],
            mode='lines+markers', line=dict(color='orange', width=6),
            marker=dict(size=[8, 12], color='orange', symbol=['circle', 'diamond']),
            name='Low Normal vector', hovertemplate='Low Normal vector<extra></extra>'
        ))
        
        # High plane normal vector
        normal_end_high = plane_center_high + normal_vector_high * normal_scale
        fig.add_trace(go.Scatter3d(
            x=[plane_center_high[0], normal_end_high[0]],
            y=[plane_center_high[1], normal_end_high[1]],
            z=[plane_center_high[2], normal_end_high[2]],
            mode='lines+markers', line=dict(color='purple', width=6),
            marker=dict(size=[8, 12], color='purple', symbol=['circle', 'diamond']),
            name='High Normal vector', hovertemplate='High Normal vector<extra></extra>'
        ))
        
        # Draw z-axis vector
        origin = np.mean(positions, axis=0)
        z_axis_end = origin + z_axis_normalized * normal_scale * 0.8
        
        fig.add_trace(go.Scatter3d(
            x=[origin[0], z_axis_end[0]], y=[origin[1], z_axis_end[1]], z=[origin[2], z_axis_end[2]],
            mode='lines+markers', line=dict(color='cyan', width=4),
            marker=dict(size=[6, 10], color='cyan', symbol=['circle', 'diamond']),
            name='Z-axis', hovertemplate='Z-axis vector<extra></extra>'
        ))
        
        # Update layout
        title_text = f'Interactive Vector Analysis - Angle: {results["angle_between_planes_degrees"]:.2f}°, Distance: {results["distance_between_plane_centers_angstrom"]:.3f} Å'
        
        fig.update_layout(
            title=title_text,
            scene=dict(
                xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)',
                aspectmode='cube', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1200, height=800, showlegend=True
        )
        
        # Save HTML file
        fig.write_html(output_filename)
        return output_filename
    
    def _create_plot_with_original_structure(self, analyzer, vector_analysis_results, output_filename):
        """
        Create an interactive plot using the original structure and terminal atom analysis.
        
        Parameters:
            analyzer: q2D_analyzer instance
            vector_analysis_results (dict): Results with terminal atom analysis
            output_filename (str): Output HTML filename
            
        Returns:
            str: Path to created HTML file or error message
        """
        try:
            import plotly.graph_objects as go
        except ImportError:
            return "Error: Plotly not available. Install with: pip install plotly"
        
        # Get structure data from original analyzer
        positions = analyzer.atoms.get_positions()
        symbols = analyzer.atoms.get_chemical_symbols()
        cell = analyzer.atoms.get_cell()
        
        # Get analysis results
        results = vector_analysis_results['vector_analysis_results']
        plane_data = vector_analysis_results['plane_analysis']
        terminal_data = vector_analysis_results['terminal_atom_analysis']
        
        # Get terminal halogen information
        halogen_symbol = terminal_data['halogen_symbol']
        low_plane_indices = terminal_data['low_plane_atom_indices']
        high_plane_indices = terminal_data['high_plane_atom_indices']
        
        # Get terminal halogen positions
        cl_low_1 = positions[low_plane_indices[0]]
        cl_low_2 = positions[low_plane_indices[1]]
        cl_high_1 = positions[high_plane_indices[0]]
        cl_high_2 = positions[high_plane_indices[1]]
        
        # Get z-axis and plane centers
        z_axis = cell[2]
        z_axis_normalized = z_axis / np.linalg.norm(z_axis)
        plane_center_low = np.array(plane_data['low_plane']['center_position'])
        plane_center_high = np.array(plane_data['high_plane']['center_position'])
        normal_vector_low = np.array(plane_data['low_plane']['normal_vector'])
        normal_vector_high = np.array(plane_data['high_plane']['normal_vector'])
        
        # Create Plotly figure
        fig = go.Figure()
        
        # Color mapping for atoms
        colors = {'Cl': 'green', 'Br': 'brown', 'I': 'purple', 'F': 'cyan', 'Pb': 'gray', 'N': 'blue', 'C': 'black', 'H': 'lightgray', 'other': 'silver'}
        unique_symbols = list(set(symbols))
        
        # Group atoms by type for better legend handling
        for symbol in unique_symbols:
            mask = np.array(symbols) == symbol
            atom_positions = positions[mask]
            color = colors.get(symbol, colors['other'])
            
            # Adjust size based on atom type
            size = 12 if symbol in ['Pb'] else 8 if symbol in ['Br', 'Cl', 'I', 'F'] else 4
            
            fig.add_trace(go.Scatter3d(
                x=atom_positions[:, 0],
                y=atom_positions[:, 1],
                z=atom_positions[:, 2],
                mode='markers',
                marker=dict(size=size, color=color, opacity=0.8),
                name=f'{symbol} atoms',
                hovertemplate=f'{symbol}<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
            ))
        
        # Highlight the terminal halogen atoms used for planes
        fig.add_trace(go.Scatter3d(
            x=[cl_low_1[0]], y=[cl_low_1[1]], z=[cl_low_1[2]],
            mode='markers', marker=dict(size=18, color='red', symbol='circle'),
            name=f'Terminal {halogen_symbol} {low_plane_indices[0]} (low plane)',
            hovertemplate=f'Terminal {halogen_symbol} {low_plane_indices[0]} (low plane)<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[cl_low_2[0]], y=[cl_low_2[1]], z=[cl_low_2[2]],
            mode='markers', marker=dict(size=18, color='darkred', symbol='circle'),
            name=f'Terminal {halogen_symbol} {low_plane_indices[1]} (low plane)',
            hovertemplate=f'Terminal {halogen_symbol} {low_plane_indices[1]} (low plane)<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[cl_high_1[0]], y=[cl_high_1[1]], z=[cl_high_1[2]],
            mode='markers', marker=dict(size=18, color='blue', symbol='circle'),
            name=f'Terminal {halogen_symbol} {high_plane_indices[0]} (high plane)',
            hovertemplate=f'Terminal {halogen_symbol} {high_plane_indices[0]} (high plane)<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[cl_high_2[0]], y=[cl_high_2[1]], z=[cl_high_2[2]],
            mode='markers', marker=dict(size=18, color='darkblue', symbol='circle'),
            name=f'Terminal {halogen_symbol} {high_plane_indices[1]} (high plane)',
            hovertemplate=f'Terminal {halogen_symbol} {high_plane_indices[1]} (high plane)<br>X: %{{x:.2f}} Å<br>Y: %{{y:.2f}} Å<br>Z: %{{z:.2f}} Å<extra></extra>'
        ))
        
        # Draw vectors between terminal halogen atoms
        fig.add_trace(go.Scatter3d(
            x=[cl_low_1[0], cl_low_2[0]], y=[cl_low_1[1], cl_low_2[1]], z=[cl_low_1[2], cl_low_2[2]],
            mode='lines', line=dict(color='red', width=10), name='Low plane vector',
            hovertemplate='Low plane vector<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[cl_high_1[0], cl_high_2[0]], y=[cl_high_1[1], cl_high_2[1]], z=[cl_high_1[2], cl_high_2[2]],
            mode='lines', line=dict(color='blue', width=10), name='High plane vector',
            hovertemplate='High plane vector<extra></extra>'
        ))
        
        # Create plane surfaces
        plane_size = np.max(np.max(positions, axis=0) - np.min(positions, axis=0)) * 0.6
        u = np.linspace(-plane_size/2, plane_size/2, 15)
        v = np.linspace(-plane_size/2, plane_size/2, 15)
        U, V = np.meshgrid(u, v)
        
        # Low plane
        v1_low = cl_low_2 - cl_low_1
        v1_low = v1_low / np.linalg.norm(v1_low)
        v2_low = np.cross(normal_vector_low, v1_low)
        v2_low = v2_low / np.linalg.norm(v2_low)
        
        plane_points_low = plane_center_low[:, np.newaxis, np.newaxis] + U[np.newaxis, :, :] * v1_low[:, np.newaxis, np.newaxis] + V[np.newaxis, :, :] * v2_low[:, np.newaxis, np.newaxis]
        
        fig.add_trace(go.Surface(
            x=plane_points_low[0], y=plane_points_low[1], z=plane_points_low[2],
            opacity=0.4, colorscale=[[0, 'yellow'], [1, 'orange']], showscale=False,
            name='Low Plane (Terminal atoms)', hovertemplate='Low Plane<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
        ))
        
        # High plane
        v1_high = cl_high_2 - cl_high_1
        v1_high = v1_high / np.linalg.norm(v1_high)
        v2_high = np.cross(normal_vector_high, v1_high)
        v2_high = v2_high / np.linalg.norm(v2_high)
        
        plane_points_high = plane_center_high[:, np.newaxis, np.newaxis] + U[np.newaxis, :, :] * v1_high[:, np.newaxis, np.newaxis] + V[np.newaxis, :, :] * v2_high[:, np.newaxis, np.newaxis]
        
        fig.add_trace(go.Surface(
            x=plane_points_high[0], y=plane_points_high[1], z=plane_points_high[2],
            opacity=0.4, colorscale=[[0, 'lightblue'], [1, 'blue']], showscale=False,
            name='High Plane (Terminal atoms)', hovertemplate='High Plane<br>X: %{x:.2f} Å<br>Y: %{y:.2f} Å<br>Z: %{z:.2f} Å<extra></extra>'
        ))
        
        # Draw normal vectors
        normal_scale = plane_size * 0.4
        
        # Low plane normal vector
        normal_end_low = plane_center_low + normal_vector_low * normal_scale
        fig.add_trace(go.Scatter3d(
            x=[plane_center_low[0], normal_end_low[0]],
            y=[plane_center_low[1], normal_end_low[1]],
            z=[plane_center_low[2], normal_end_low[2]],
            mode='lines+markers', line=dict(color='orange', width=8),
            marker=dict(size=[10, 15], color='orange', symbol=['circle', 'diamond']),
            name='Low Normal vector', hovertemplate='Low Normal vector<extra></extra>'
        ))
        
        # High plane normal vector
        normal_end_high = plane_center_high + normal_vector_high * normal_scale
        fig.add_trace(go.Scatter3d(
            x=[plane_center_high[0], normal_end_high[0]],
            y=[plane_center_high[1], normal_end_high[1]],
            z=[plane_center_high[2], normal_end_high[2]],
            mode='lines+markers', line=dict(color='purple', width=8),
            marker=dict(size=[10, 15], color='purple', symbol=['circle', 'diamond']),
            name='High Normal vector', hovertemplate='High Normal vector<extra></extra>'
        ))
        
        # Draw z-axis vector
        origin = np.mean(positions, axis=0)
        z_axis_end = origin + z_axis_normalized * normal_scale * 0.7
        
        fig.add_trace(go.Scatter3d(
            x=[origin[0], z_axis_end[0]], y=[origin[1], z_axis_end[1]], z=[origin[2], z_axis_end[2]],
            mode='lines+markers', line=dict(color='cyan', width=6),
            marker=dict(size=[8, 12], color='cyan', symbol=['circle', 'diamond']),
            name='Z-axis', hovertemplate='Z-axis vector<extra></extra>'
        ))
        
        # Update layout
        title_text = f'Vector Analysis (Terminal {halogen_symbol} Atoms) - Angle: {results["angle_between_planes_degrees"]:.2f}°, Distance: {results["distance_between_plane_centers_angstrom"]:.3f} Å'
        subtitle_text = f'Terminal atoms: {terminal_data["total_terminal_atoms_found"]} found - Low: {low_plane_indices}, High: {high_plane_indices}'
        
        fig.update_layout(
            title=f'{title_text}<br><sub>{subtitle_text}</sub>',
            scene=dict(
                xaxis_title='X (Å)', yaxis_title='Y (Å)', zaxis_title='Z (Å)',
                aspectmode='cube', camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=1400, height=900, showlegend=True
        )
        
        # Save HTML file
        fig.write_html(output_filename)
        return output_filename
        
    def get_vector_summary(self, vector_analysis_results):
        """
        Get the key vector analysis values without text formatting.
        
        Parameters:
            vector_analysis_results (dict): Results from analyze_salt_structure_vectors
            
        Returns:
            dict: Key vector analysis values
        """
        if 'error' in vector_analysis_results:
            return {'error': vector_analysis_results['error']}
        
        results = vector_analysis_results['vector_analysis_results']
        plane_data = vector_analysis_results['plane_analysis']
        
        return {
            'angle_between_planes_degrees': results['angle_between_planes_degrees'],
            'distance_between_plane_centers_angstrom': results['distance_between_plane_centers_angstrom'],
            'low_plane_vs_z_axis_degrees': plane_data['low_plane']['angle_vs_z_axis_degrees'],
            'high_plane_vs_z_axis_degrees': plane_data['high_plane']['angle_vs_z_axis_degrees']
        } 

    def get_penetration_depth(self, salt_structure, spacer_molecules, x_symbol='Cl', analyzer=None):
        """
        Calculate penetration depth for spacer molecules through the halogen planes.
        
        The main idea is to find the vectors of molecules N1 and N2 (terminal nitrogen atoms)
        and see where they intersect with the planes. We calculate the distance of:
        - N1 to lower plane intersection (mol_low_A)
        - Lower plane intersection to higher plane intersection  
        - Higher plane intersection to N2 (mol_high_A)
        
        We return the penetration depth values for all spacer molecules.
        
        Parameters:
            salt_structure (ase.Atoms): ASE Atoms object with salt structure
            spacer_molecules (dict): Dictionary of spacer molecules from analyzer
            x_symbol (str): Chemical symbol of the halogen atoms (default: 'Cl')
            analyzer: q2D_analyzer instance for accessing original structure and connectivity
            
        Returns:
            dict: Penetration depth analysis results
        """
        if salt_structure is None or len(salt_structure) == 0:
            return {
                'error': 'No salt structure available for penetration depth analysis'
            }
        
        if not spacer_molecules or len(spacer_molecules) < 1:
            return {
                'error': 'Need at least 1 spacer molecule for penetration depth analysis'
            }
        
        # Ensure we have the analyzer for accessing original coordinates
        if analyzer is None:
            return {
                'error': 'Analyzer instance required for accessing original coordinates'
            }
        
        # First, get the plane analysis by reusing existing method
        vector_analysis = self.analyze_salt_structure_vectors(salt_structure, x_symbol, analyzer=analyzer)
        
        if 'error' in vector_analysis:
            return {'error': f'Plane analysis failed: {vector_analysis["error"]}'}
        
        # Extract plane information
        plane_data = vector_analysis['plane_analysis']
        low_plane_normal = np.array(plane_data['low_plane']['normal_vector'])
        low_plane_center = np.array(plane_data['low_plane']['center_position'])
        high_plane_normal = np.array(plane_data['high_plane']['normal_vector'])
        high_plane_center = np.array(plane_data['high_plane']['center_position'])
        
        # Calculate plane equations: ax + by + cz = d
        # For a plane with normal vector (a,b,c) passing through point (x0,y0,z0): ax + by + cz = ax0 + by0 + cz0
        low_plane_d = np.dot(low_plane_normal, low_plane_center)
        high_plane_d = np.dot(high_plane_normal, high_plane_center)
        
        # Get original structure data
        original_positions = analyzer.atoms.get_positions()
        original_symbols = analyzer.atoms.get_chemical_symbols()
        
        penetration_results = {}
        
        # Process each spacer molecule
        for mol_id, mol_data in spacer_molecules.items():
            mol_atom_indices = mol_data['atom_indices']  # Global indices in original structure
            
            # Get coordinates and symbols from original structure using global indices
            mol_coordinates = original_positions[mol_atom_indices]
            mol_symbols = [original_symbols[i] for i in mol_atom_indices]
            
            # Find nitrogen atoms (N1 and N2) - find the pair with minimum XY distance (same molecule)
            nitrogen_indices = [i for i, symbol in enumerate(mol_symbols) if symbol == 'N']
            
            if len(nitrogen_indices) < 2:
                penetration_results[f'molecule_{mol_id}'] = {
                    'error': f'Molecule {mol_id} has less than 2 nitrogen atoms for N1-N2 vector'
                }
                continue
            
            n_coords = mol_coordinates[nitrogen_indices]
            
            # Find the nitrogen pair with minimum XY distance and assign N1/N2 based on plane positions
            min_xy_distance = float('inf')
            best_n1_coord = None
            best_n2_coord = None
            
            for i in range(len(n_coords)):
                for j in range(i+1, len(n_coords)):
                    coord1 = n_coords[i]
                    coord2 = n_coords[j]
                    
                    # Calculate XY distance (ignore Z for now)
                    xy_distance = np.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)
                    
                    if xy_distance < min_xy_distance:
                        min_xy_distance = xy_distance
                        
                        # Dynamically assign N1 and N2 based on position relative to planes
                        # Calculate signed distances to low plane for both atoms
                        coord1_to_low = np.dot(coord1 - low_plane_center, low_plane_normal)
                        coord2_to_low = np.dot(coord2 - low_plane_center, low_plane_normal)
                        
                        # N1 should be the one closer to (or below) the low plane
                        # N2 should be the one further from (or above) the low plane
                        if coord1_to_low < coord2_to_low:
                            best_n1_coord = coord1  # coord1 is closer to low plane
                            best_n2_coord = coord2  # coord2 is further from low plane
                        else:
                            best_n1_coord = coord2  # coord2 is closer to low plane
                            best_n2_coord = coord1  # coord1 is further from low plane
            
            if best_n1_coord is None or best_n2_coord is None:
                penetration_results[f'molecule_{mol_id}'] = {
                    'error': f'Molecule {mol_id} could not find valid N1-N2 pair'
                }
                continue
            
            n1_coord = best_n1_coord
            n2_coord = best_n2_coord
            
            # Calculate the N1-N2 vector using PBC-aware distance calculation
            # Use the analyzer's geometry calculator for PBC-aware vector calculation
            n1_n2_distance = analyzer.geometry_calc.calculate_distance(n1_coord, n2_coord)
            
            # Get the PBC-aware vector from N1 to N2 using the geometry calculator's PBC method
            if analyzer.geometry_calc.use_pbc and analyzer.geometry_calc.cell is not None:
                # Use PBC-aware vector calculation
                vector_a, vector_b, vector_c = analyzer.geometry_calc.cell[0], analyzer.geometry_calc.cell[1], analyzer.geometry_calc.cell[2]
                dx, dy, dz = analyzer.geometry_calc._calculate_distance_pbc_signo(
                    n1_coord[0], n1_coord[1], n1_coord[2],
                    n2_coord[0], n2_coord[1], n2_coord[2],
                    vector_a, vector_b, vector_c
                )
                mol_vector = np.array([dx, dy, dz])
            else:
                # Standard vector if no PBC
                mol_vector = n2_coord - n1_coord
            
            if n1_n2_distance < 1e-6:  # Very small vector
                penetration_results[f'molecule_{mol_id}'] = {
                    'error': f'Molecule {mol_id} has N1 and N2 too close together'
                }
                continue
            
            # Find intersections with planes using the original molecular vector (not normalized)
            # For a line from point P with direction vector V: P + t*V
            # Intersection with plane (normal N, passing through point C): (P + t*V) · N = C · N
            # Solving for t: t = (C · N - P · N) / (V · N)
            
            # Intersection with low plane
            denom_low = np.dot(mol_vector, low_plane_normal)
            if abs(denom_low) < 1e-6:  # Vector parallel to plane
                penetration_results[f'molecule_{mol_id}'] = {
                    'error': f'Molecule {mol_id} vector is parallel to low plane'
                }
                continue
            
            t_low = (low_plane_d - np.dot(n1_coord, low_plane_normal)) / denom_low
            
            # Intersection with high plane
            denom_high = np.dot(mol_vector, high_plane_normal)
            if abs(denom_high) < 1e-6:  # Vector parallel to plane
                penetration_results[f'molecule_{mol_id}'] = {
                    'error': f'Molecule {mol_id} vector is parallel to high plane'
                }
                continue
            
            t_high = (high_plane_d - np.dot(n1_coord, high_plane_normal)) / denom_high
            
            # Validate the molecular vector intersects both planes correctly
            # Calculate signed distances to planes for verification
            n1_to_low_plane_signed = np.dot(n1_coord - low_plane_center, low_plane_normal)
            n2_to_low_plane_signed = np.dot(n2_coord - low_plane_center, low_plane_normal)
            n1_to_high_plane_signed = np.dot(n1_coord - high_plane_center, high_plane_normal)
            n2_to_high_plane_signed = np.dot(n2_coord - high_plane_center, high_plane_normal)
            
            # Check that N1 and N2 are on opposite sides of at least one plane
            # This ensures the molecule actually spans across the plane structure
            spans_low_plane = (n1_to_low_plane_signed * n2_to_low_plane_signed) < 0
            spans_high_plane = (n1_to_high_plane_signed * n2_to_high_plane_signed) < 0
            
            if not (spans_low_plane or spans_high_plane):
                # The molecule doesn't span across either plane - it might be entirely on one side
                penetration_results[f'molecule_{mol_id}'] = {
                    'error': f'Molecule {mol_id} does not span across the planes. N1-low: {n1_to_low_plane_signed:.3f}, N2-low: {n2_to_low_plane_signed:.3f}, N1-high: {n1_to_high_plane_signed:.3f}, N2-high: {n2_to_high_plane_signed:.3f}'
                }
                continue
            
            # Ensure t values are valid and in correct order
            # Allow some tolerance for molecules that extend slightly beyond planes
            if t_low >= t_high:
                penetration_results[f'molecule_{mol_id}'] = {
                    'error': f'Molecule {mol_id} has invalid intersection order. t_low: {t_low:.3f}, t_high: {t_high:.3f}'
                }
                continue
            
            # Clamp t values to [0, 1] range if they're close but slightly outside
            t_low_clamped = max(0.0, min(1.0, t_low))
            t_high_clamped = max(0.0, min(1.0, t_high))
            
            # Use clamped values for distance calculation
            t_low = t_low_clamped
            t_high = t_high_clamped
            
            # Calculate penetration distances along the straight N1-N2 vector
            # The plane intersections are based on the straight line from N1 to N2
            # t represents the fraction along this straight vector
            
            # Segment 1: N1 to low plane intersection
            dist_n1_to_low = t_low * n1_n2_distance
            
            # Segment 2: Low plane intersection to high plane intersection  
            dist_low_to_high = (t_high - t_low) * n1_n2_distance
            
            # Segment 3: High plane intersection to N2
            dist_high_to_n2 = (1.0 - t_high) * n1_n2_distance
            
            # For verification, calculate intersection points
            intersection_low = n1_coord + t_low * mol_vector
            intersection_high = n1_coord + t_high * mol_vector
            
            # Total distance check (should equal n1_n2_distance)
            total_calculated = dist_n1_to_low + dist_low_to_high + dist_high_to_n2
            
            penetration_results[f'molecule_{mol_id}'] = {
                'formula': mol_data['formula'],
                'n1_position': n1_coord.tolist(),
                'n2_position': n2_coord.tolist(),
                'molecular_vector': mol_vector.tolist(),
                'n1_n2_straight_distance': float(n1_n2_distance),
                'low_plane_intersection': intersection_low.tolist(),
                'high_plane_intersection': intersection_high.tolist(),
                'penetration_segments': {
                    'n1_to_low_plane': float(dist_n1_to_low),
                    'low_plane_to_high_plane': float(dist_low_to_high), 
                    'high_plane_to_n2': float(dist_high_to_n2),
                    'total_calculated': float(total_calculated),
                    'n1_n2_straight_distance': float(n1_n2_distance),
                    'length_difference': float(abs(total_calculated - n1_n2_distance))
                },
                'penetration_parameters': {
                    't_low_plane': float(t_low),
                    't_high_plane': float(t_high)
                }
            }
        
        # Add overall analysis for molecules with successful calculations
        successful_molecules = [k for k, v in penetration_results.items() if 'error' not in v]
        
        if len(successful_molecules) >= 2:
            # Take the first two successful molecules for comparison
            mol_ids = successful_molecules[:2]
            mol1_data = penetration_results[mol_ids[0]]
            mol2_data = penetration_results[mol_ids[1]]
            
            # Calculate comparative penetration
            comparative_analysis = {
                'molecule_1': {
                    'id': mol_ids[0],
                    'segments': mol1_data['penetration_segments']
                },
                'molecule_2': {
                    'id': mol_ids[1],
                    'segments': mol2_data['penetration_segments']
                },
                'penetration_comparison': {
                    'n1_to_low_diff': float(abs(mol1_data['penetration_segments']['n1_to_low_plane'] - 
                                              mol2_data['penetration_segments']['n1_to_low_plane'])),
                    'low_to_high_diff': float(abs(mol1_data['penetration_segments']['low_plane_to_high_plane'] - 
                                                mol2_data['penetration_segments']['low_plane_to_high_plane'])),
                    'high_to_n2_diff': float(abs(mol1_data['penetration_segments']['high_plane_to_n2'] - 
                                               mol2_data['penetration_segments']['high_plane_to_n2']))
                }
            }
            penetration_results['comparative_analysis'] = comparative_analysis
        
        # Include the plane analysis for reference
        penetration_results['plane_reference'] = vector_analysis
        
        return penetration_results 