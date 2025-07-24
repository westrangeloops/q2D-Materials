"""
Connectivity analysis for octahedral structures.

This module handles connectivity analysis between octahedra,
including shared atom detection and network topology.
"""

import numpy as np
from .geometry import GeometryCalculator


class ConnectivityAnalyzer:
    """
    Handles connectivity analysis between octahedral structures.
    """
    
    def __init__(self, geometry_calculator=None):
        """
        Initialize connectivity analyzer.
        
        Parameters:
        geometry_calculator: GeometryCalculator instance
        """
        self.geometry_calc = geometry_calculator or GeometryCalculator()
    
    def analyze_octahedra_connectivity(self, octahedra_data, all_coords, all_symbols):
        """
        Analyze which atoms are shared between octahedra.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        all_coords: All atomic coordinates
        all_symbols: All atomic symbols
        
        Returns:
        dict: Connectivity analysis including shared atoms and terminal axial information
        """
        connectivity = {
            "shared_atoms": {},
            "octahedra_connections": {},
            "sharing_types": {},
            "network_statistics": {},
            "terminal_axial_atoms": {}
        }
        
        # Create a mapping of which octahedra each atom belongs to
        atom_to_octahedra = {}
        
        for oct_key, oct_data in octahedra_data.items():
            central_index = oct_data['central_atom']['global_index']
            ligand_indices = oct_data['ligand_atoms']['all_ligand_global_indices']
            
            # Track central atoms
            if central_index not in atom_to_octahedra:
                atom_to_octahedra[central_index] = []
            atom_to_octahedra[central_index].append((oct_key, 'central'))
            
            # Track ligand atoms
            for ligand_idx in ligand_indices:
                if ligand_idx not in atom_to_octahedra:
                    atom_to_octahedra[ligand_idx] = []
                atom_to_octahedra[ligand_idx].append((oct_key, 'ligand'))
        
        # Find shared atoms
        for atom_idx, octahedra_list in atom_to_octahedra.items():
            if len(octahedra_list) > 1:
                connectivity["shared_atoms"][int(atom_idx)] = [
                    {"octahedron": oct_key, "role": role} for oct_key, role in octahedra_list
                ]
        
        # Analyze connections between octahedra
        for oct_key in octahedra_data.keys():
            connectivity["octahedra_connections"][oct_key] = []
            
        for atom_idx, octahedra_list in atom_to_octahedra.items():
            if len(octahedra_list) > 1:
                # This atom is shared between multiple octahedra
                ligand_octahedra = [oct_key for oct_key, role in octahedra_list if role == 'ligand']
                
                # Add connections with enhanced information
                for i, oct1 in enumerate(ligand_octahedra):
                    for oct2 in ligand_octahedra[i+1:]:
                        # Determine connection type (axial or equatorial)
                        connection_type = self._determine_connection_type(oct1, oct2, atom_idx, octahedra_data)
                        
                        connection_info = {
                            "connected_octahedron": oct2,
                            "shared_atom_index": int(atom_idx),
                            "atom_symbol": str(all_symbols[atom_idx]),
                            "connection_type": connection_type
                        }
                        
                        # Avoid duplicates
                        if connection_info not in connectivity["octahedra_connections"][oct1]:
                            connectivity["octahedra_connections"][oct1].append(connection_info)
                        
                        reverse_connection = {
                            "connected_octahedron": oct1,
                            "shared_atom_index": int(atom_idx),
                            "atom_symbol": str(all_symbols[atom_idx]),
                            "connection_type": connection_type
                        }
                        
                        if reverse_connection not in connectivity["octahedra_connections"][oct2]:
                            connectivity["octahedra_connections"][oct2].append(reverse_connection)
        
        # Identify terminal axial atoms using PBC-aware calculations
        connectivity["terminal_axial_atoms"] = self._identify_terminal_axial_atoms(
            octahedra_data, connectivity["shared_atoms"]
        )
        
        # Analyze sharing types
        connectivity["sharing_types"] = self._analyze_sharing_types(
            connectivity["shared_atoms"], octahedra_data, all_coords, all_symbols
        )
        
        # Calculate network statistics
        connectivity["network_statistics"] = self._calculate_network_statistics(
            connectivity["octahedra_connections"], connectivity["shared_atoms"]
        )
        
        return connectivity
    
    def _determine_connection_type(self, oct1_key, oct2_key, shared_atom_idx, octahedra_data):
        """
        Determine if a connection is through axial or equatorial atoms.
        
        Parameters:
        oct1_key, oct2_key (str): Keys of connected octahedra
        shared_atom_idx (int): Index of shared atom
        octahedra_data (dict): Dictionary of octahedra data
        
        Returns:
        str: 'axial', 'equatorial', or 'unknown'
        """
        for oct_key in [oct1_key, oct2_key]:
            if oct_key in octahedra_data:
                oct_data = octahedra_data[oct_key]
                axial_indices = oct_data['ligand_atoms']['axial_global_indices']
                equatorial_indices = oct_data['ligand_atoms']['equatorial_global_indices']
                
                if shared_atom_idx in axial_indices:
                    return 'axial'
                elif shared_atom_idx in equatorial_indices:
                    return 'equatorial'
        
        return 'unknown'
    
    def _identify_terminal_axial_atoms(self, octahedra_data, shared_atoms):
        """
        Identify axial atoms that are not connected to other octahedral centers.
        Uses PBC-aware distance calculations through the geometry calculator.
        
        Parameters:
        octahedra_data (dict): Dictionary of octahedra data
        shared_atoms (dict): Shared atoms data from connectivity analysis
        
        Returns:
        dict: Terminal axial atoms data
        """
        terminal_axial = {}
        
        for oct_key, oct_data in octahedra_data.items():
            axial_indices = oct_data['ligand_atoms']['axial_global_indices']
            
            terminal_atoms = []
            connected_atoms = []
            
            for axial_idx in axial_indices:
                is_terminal = True
                
                # Check if this axial atom is shared with other octahedra
                if axial_idx in shared_atoms:
                    sharing_info = shared_atoms[axial_idx]
                    # Check if it's shared with other octahedra as ligand
                    other_octahedra = [
                        info['octahedron'] for info in sharing_info 
                        if info['octahedron'] != oct_key and info['role'] == 'ligand'
                    ]
                    
                    if other_octahedra:
                        is_terminal = False
                        connected_atoms.append({
                            'atom_index': int(axial_idx),
                            'connected_to_octahedra': other_octahedra
                        })
                
                if is_terminal:
                    terminal_atoms.append({
                        'atom_index': int(axial_idx),
                        'atom_type': 'terminal_axial'
                    })
            
            # Only include octahedra that have terminal or connected axial atoms
            if terminal_atoms or connected_atoms:
                terminal_axial[oct_key] = {
                    'terminal_axial_atoms': terminal_atoms,
                    'connected_axial_atoms': connected_atoms
                }
        
        return terminal_axial
    
    def _analyze_sharing_types(self, shared_atoms, octahedra_data, all_coords, all_symbols):
        """
        Analyze the types of sharing between octahedra.
        
        Returns:
        dict: Analysis of sharing types (corner, edge, face sharing)
        """
        sharing_types = {
            "corner_sharing": [],  # 1 shared atom
            "edge_sharing": [],    # 2 shared atoms
            "face_sharing": [],    # 3+ shared atoms
            "summary": {}
        }
        
        # Group shared atoms by octahedra pairs
        octahedra_pairs = {}
        for atom_idx, sharing_info in shared_atoms.items():
            if len(sharing_info) == 2:  # Only consider pairs
                oct1 = sharing_info[0]["octahedron"]
                oct2 = sharing_info[1]["octahedron"]
                
                # Create a sorted tuple for consistent pair identification
                pair = tuple(sorted([oct1, oct2]))
                
                if pair not in octahedra_pairs:
                    octahedra_pairs[pair] = []
                octahedra_pairs[pair].append(atom_idx)
        
        # Classify sharing types
        for pair, shared_atom_indices in octahedra_pairs.items():
            num_shared = len(shared_atom_indices)
            sharing_info = {
                "octahedron_1": pair[0],
                "octahedron_2": pair[1],
                "shared_atoms": shared_atom_indices,
                "num_shared_atoms": num_shared,
                "shared_atom_symbols": [all_symbols[idx] for idx in shared_atom_indices]
            }
            
            if num_shared == 1:
                sharing_types["corner_sharing"].append(sharing_info)
            elif num_shared == 2:
                sharing_types["edge_sharing"].append(sharing_info)
            elif num_shared >= 3:
                sharing_types["face_sharing"].append(sharing_info)
        
        # Summary statistics
        sharing_types["summary"] = {
            "total_corner_sharing_pairs": len(sharing_types["corner_sharing"]),
            "total_edge_sharing_pairs": len(sharing_types["edge_sharing"]),
            "total_face_sharing_pairs": len(sharing_types["face_sharing"]),
            "total_connected_pairs": len(octahedra_pairs)
        }
        
        return sharing_types
    
    def _calculate_network_statistics(self, octahedra_connections, shared_atoms):
        """
        Calculate network topology statistics.
        
        Returns:
        dict: Network statistics
        """
        # Count connections per octahedron
        connections_per_octahedron = {
            oct_key: len(connections) 
            for oct_key, connections in octahedra_connections.items()
        }
        
        # Calculate degree distribution
        degrees = list(connections_per_octahedron.values())
        
        statistics = {
            "total_octahedra": len(octahedra_connections),
            "total_shared_atoms": len(shared_atoms),
            "connections_per_octahedron": connections_per_octahedron,
            "average_connectivity": float(np.mean(degrees)) if degrees else 0.0,
            "max_connectivity": int(max(degrees)) if degrees else 0,
            "min_connectivity": int(min(degrees)) if degrees else 0,
            "connectivity_variance": float(np.var(degrees)) if degrees else 0.0
        }
        
        # Find most and least connected octahedra
        if degrees:
            max_connections = max(degrees)
            min_connections = min(degrees)
            
            statistics["most_connected_octahedra"] = [
                oct_key for oct_key, count in connections_per_octahedron.items() 
                if count == max_connections
            ]
            
            statistics["least_connected_octahedra"] = [
                oct_key for oct_key, count in connections_per_octahedron.items() 
                if count == min_connections
            ]
        
        return statistics
    
    def get_connectivity_summary(self, connectivity_analysis):
        """
        Generate a human-readable summary of connectivity analysis.
        
        Parameters:
        connectivity_analysis: Result from analyze_octahedra_connectivity
        
        Returns:
        str: Formatted summary
        """
        shared_atoms = connectivity_analysis.get('shared_atoms', {})
        sharing_types = connectivity_analysis.get('sharing_types', {})
        network_stats = connectivity_analysis.get('network_statistics', {})
        terminal_axial = connectivity_analysis.get('terminal_axial_atoms', {})
        
        summary_lines = []
        summary_lines.append("=== CONNECTIVITY ANALYSIS SUMMARY ===")
        
        # Overall statistics
        summary_lines.append(f"Total shared atoms: {len(shared_atoms)}")
        summary_lines.append(f"Total octahedra: {network_stats.get('total_octahedra', 0)}")
        summary_lines.append(f"Average connectivity: {network_stats.get('average_connectivity', 0):.2f}")
        
        # Terminal axial statistics
        total_terminal = sum(len(data['terminal_axial_atoms']) for data in terminal_axial.values())
        total_connected_axial = sum(len(data['connected_axial_atoms']) for data in terminal_axial.values())
        summary_lines.append(f"Terminal axial atoms: {total_terminal}")
        summary_lines.append(f"Connected axial atoms: {total_connected_axial}")
        
        # Sharing types
        sharing_summary = sharing_types.get('summary', {})
        summary_lines.append(f"\nSharing Types:")
        summary_lines.append(f"  Corner sharing pairs: {sharing_summary.get('total_corner_sharing_pairs', 0)}")
        summary_lines.append(f"  Edge sharing pairs: {sharing_summary.get('total_edge_sharing_pairs', 0)}")
        summary_lines.append(f"  Face sharing pairs: {sharing_summary.get('total_face_sharing_pairs', 0)}")
        
        # Most/least connected
        most_connected = network_stats.get('most_connected_octahedra', [])
        least_connected = network_stats.get('least_connected_octahedra', [])
        
        if most_connected:
            summary_lines.append(f"\nMost connected: {', '.join(most_connected)} ({network_stats.get('max_connectivity', 0)} connections)")
        if least_connected:
            summary_lines.append(f"Least connected: {', '.join(least_connected)} ({network_stats.get('min_connectivity', 0)} connections)")
        
        # Terminal axial details
        if terminal_axial:
            summary_lines.append(f"\nTerminal Axial Atoms:")
            for oct_key, data in terminal_axial.items():
                if data['terminal_axial_atoms']:
                    terminal_atoms = [str(atom['atom_index']) for atom in data['terminal_axial_atoms']]
                    summary_lines.append(f"  {oct_key}: {', '.join(terminal_atoms)}")
        
        return '\n'.join(summary_lines) 