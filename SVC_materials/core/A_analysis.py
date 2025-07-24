"""
A-site atom identification for perovskite structures.

Simplified module that identifies A-site atoms between octahedral layers
and provides basic statistics and atom assignments.
"""

import numpy as np
from collections import defaultdict


class ASiteIdentifier:
    """
    Simplified A-site atom identifier for perovskite structures.
    Just identifies atoms and assigns them to A1, A2, etc. groups.
    """
    
    def __init__(self, atoms=None, geometry_calculator=None, bond_tolerance=0.3):
        """
        Initialize A-site identifier.
        
        Parameters:
        atoms: ASE Atoms object
        geometry_calculator: GeometryCalculator instance (optional)
        bond_tolerance: Not used in simplified version
        """
        self.atoms = atoms
        self.geometry_calc = geometry_calculator
        self.a_site_analysis = {}
        
        # Common A-site elements
        self.common_a_site_elements = ['Cs', 'Rb', 'K', 'Na', 'Li']
        self.organic_elements = ['C', 'H', 'O', 'N']
        
    def identify_a_site_atoms(self, octahedra_data, layers_analysis, all_coords, all_symbols):
        """
        Identify A-site atoms and assign them to A1, A2, etc. groups.
        
        Parameters:
        octahedra_data: Dictionary of octahedra data
        layers_analysis: Layer analysis results
        all_coords: All atomic coordinates
        all_symbols: All atomic symbols
        
        Returns:
        dict: Simplified A-site analysis with atom assignments
        """
        # Step 1: Identify all atoms that are part of octahedra
        octahedral_atoms = self._get_octahedral_atoms(octahedra_data)
        
        # Step 2: Find non-octahedral atoms (A-site candidates)
        non_octahedral_atoms = self._find_non_octahedral_atoms(
            octahedral_atoms, all_coords, all_symbols
        )
        
        # Step 3: Assign atoms to A-site groups based on layers
        a_site_groups = self._assign_atoms_to_a_site_groups(
            non_octahedral_atoms, layers_analysis, octahedra_data
        )
        
        # Step 4: Calculate basic statistics
        statistics = self._calculate_basic_statistics(a_site_groups, all_coords, all_symbols)
        
        self.a_site_analysis = {
            "octahedral_atoms": list(octahedral_atoms),
            "non_octahedral_atoms": non_octahedral_atoms,
            "a_site_groups": a_site_groups,
            "statistics": statistics,
            "summary": self._create_simple_summary(a_site_groups, statistics)
        }
        
        return self.a_site_analysis
    
    def _get_octahedral_atoms(self, octahedra_data):
        """
        Get all atom indices that are part of octahedra.
        """
        octahedral_atoms = set()
        
        for oct_key, oct_data in octahedra_data.items():
            # Add central atom
            central_idx = oct_data['central_atom']['global_index']
            octahedral_atoms.add(central_idx)
            
            # Add ligand atoms
            ligand_indices = oct_data['ligand_atoms']['all_ligand_global_indices']
            octahedral_atoms.update(ligand_indices)
        
        return octahedral_atoms
    
    def _find_non_octahedral_atoms(self, octahedral_atoms, all_coords, all_symbols):
        """
        Find atoms that are not part of any octahedron.
        """
        non_octahedral = {}
        
        for i, (coord, symbol) in enumerate(zip(all_coords, all_symbols)):
            if i not in octahedral_atoms:
                non_octahedral[i] = {
                    'global_index': i,
                    'symbol': symbol,
                    'coordinates': {
                        'x': float(coord[0]),
                        'y': float(coord[1]),
                        'z': float(coord[2])
                    },
                    'z_position': float(coord[2]),
                    'is_organic': symbol in self.organic_elements,
                    'is_common_a_site': symbol in self.common_a_site_elements
                }
        
        return non_octahedral
    
    def _assign_atoms_to_a_site_groups(self, non_octahedral_atoms, layers_analysis, octahedra_data):
        """
        Assign non-octahedral atoms to A-site groups (A1, A2, etc.) based on layer positions.
        """
        if not layers_analysis or not non_octahedral_atoms:
            return {}
        
        # Get layer z-ranges
        layer_z_ranges = self._get_layer_z_ranges(layers_analysis, octahedra_data)
        
        # Sort layers by z-position
        sorted_layers = sorted(layer_z_ranges.items(), key=lambda x: x[1]['mean_z'])
        
        a_site_groups = {}
        group_id = 1
        
        # For each interlayer region, assign atoms to A-site groups
        for i in range(len(sorted_layers)):
            if i == 0:
                # Below first layer
                region_name = f"below_{sorted_layers[0][0]}"
            elif i == len(sorted_layers):
                # Above last layer  
                region_name = f"above_{sorted_layers[-1][0]}"
            else:
                # Between layers
                current_layer = sorted_layers[i-1]
                next_layer = sorted_layers[i]
                region_name = f"between_{current_layer[0]}_{next_layer[0]}"
            
            # Find atoms in this region
            region_atoms = []
            
            if i == 0:
                # Below first layer
                z_threshold = sorted_layers[0][1]['min_z']
                for atom_idx, atom_data in non_octahedral_atoms.items():
                    if atom_data['z_position'] < z_threshold:
                        region_atoms.append(atom_idx)
            elif i == len(sorted_layers):
                # Above last layer
                z_threshold = sorted_layers[-1][1]['max_z']
                for atom_idx, atom_data in non_octahedral_atoms.items():
                    if atom_data['z_position'] > z_threshold:
                        region_atoms.append(atom_idx)
            else:
                # Between layers
                z_min = sorted_layers[i-1][1]['max_z']
                z_max = sorted_layers[i][1]['min_z']
                for atom_idx, atom_data in non_octahedral_atoms.items():
                    if z_min <= atom_data['z_position'] <= z_max:
                        region_atoms.append(atom_idx)
            
            if region_atoms:
                a_site_key = f"A{group_id}"
                a_site_groups[a_site_key] = {
                    "group_id": group_id,
                    "group_name": a_site_key,
                    "region": region_name,
                    "atom_indices": sorted(region_atoms),
                    "atom_count": len(region_atoms),
                    "atom_symbols": [non_octahedral_atoms[idx]['symbol'] for idx in region_atoms],
                    "atom_coordinates": [non_octahedral_atoms[idx]['coordinates'] for idx in region_atoms]
                }
                group_id += 1
        
        return a_site_groups
    
    def _get_layer_z_ranges(self, layers_analysis, octahedra_data):
        """
        Get z-coordinate ranges for each octahedral layer.
        """
        layer_z_ranges = {}
        layers = layers_analysis.get('layers', {})
        
        for layer_key, layer_data in layers.items():
            octahedra_keys = layer_data['octahedron_keys']
            z_coords = []
            
            for oct_key in octahedra_keys:
                if oct_key in octahedra_data:
                    z_coord = octahedra_data[oct_key]['central_atom']['coordinates']['z']
                    z_coords.append(z_coord)
            
            if z_coords:
                layer_z_ranges[layer_key] = {
                    'min_z': min(z_coords),
                    'max_z': max(z_coords),
                    'mean_z': np.mean(z_coords)
                }
        
        return layer_z_ranges
    
    def _calculate_basic_statistics(self, a_site_groups, all_coords, all_symbols):
        """
        Calculate basic statistics for A-site groups.
        """
        stats = {
            "total_a_site_groups": len(a_site_groups),
            "total_a_site_atoms": sum(group['atom_count'] for group in a_site_groups.values()),
            "group_statistics": {},
            "element_distribution": {}
        }
        
        all_a_site_symbols = []
        
        for group_key, group_data in a_site_groups.items():
            symbols = group_data['atom_symbols']
            coords = group_data['atom_coordinates']
            
            all_a_site_symbols.extend(symbols)
            
            # Calculate centroid
            if coords:
                centroid = {
                    'x': float(np.mean([c['x'] for c in coords])),
                    'y': float(np.mean([c['y'] for c in coords])),
                    'z': float(np.mean([c['z'] for c in coords]))
                }
            else:
                centroid = {'x': 0.0, 'y': 0.0, 'z': 0.0}
            
            stats["group_statistics"][group_key] = {
                "atom_count": group_data['atom_count'],
                "unique_elements": list(set(symbols)),
                "element_counts": {elem: symbols.count(elem) for elem in set(symbols)},
                "centroid": centroid,
                "z_range": {
                    'min': float(min([c['z'] for c in coords])) if coords else 0.0,
                    'max': float(max([c['z'] for c in coords])) if coords else 0.0
                }
            }
        
        # Overall element distribution
        for symbol in all_a_site_symbols:
            if symbol not in stats["element_distribution"]:
                stats["element_distribution"][symbol] = 0
            stats["element_distribution"][symbol] += 1
        
        return stats
    
    def _create_simple_summary(self, a_site_groups, statistics):
        """
        Create a simple summary of A-site analysis.
        """
        return {
            "total_groups": len(a_site_groups),
            "total_atoms": statistics["total_a_site_atoms"],
            "group_names": list(a_site_groups.keys()),
            "elements_found": list(statistics["element_distribution"].keys()),
            "dominant_element": max(statistics["element_distribution"].items(), key=lambda x: x[1])[0] if statistics["element_distribution"] else None
        }
    
    def get_a_site_summary(self):
        """
        Get a formatted summary of A-site analysis.
        """
        if not self.a_site_analysis:
            return "No A-site analysis available. Run identify_a_site_atoms() first."
        
        summary = self.a_site_analysis.get("summary", {})
        groups = self.a_site_analysis.get("a_site_groups", {})
        
        output = []
        output.append("=== A-SITE ANALYSIS SUMMARY ===")
        output.append(f"Total A-site groups: {summary.get('total_groups', 0)}")
        output.append(f"Total A-site atoms: {summary.get('total_atoms', 0)}")
        
        if summary.get('dominant_element'):
            output.append(f"Dominant element: {summary['dominant_element']}")
        
        output.append(f"Elements found: {', '.join(summary.get('elements_found', []))}")
        
        # Group details
        for group_key in summary.get('group_names', []):
            if group_key in groups:
                group_data = groups[group_key]
                output.append(f"\n{group_key}:")
                output.append(f"  Atoms: {group_data['atom_count']}")
                output.append(f"  Elements: {', '.join(set(group_data['atom_symbols']))}")
                output.append(f"  Indices: {group_data['atom_indices']}")
        
        return "\n".join(output)