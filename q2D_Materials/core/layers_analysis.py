"""
Layers analysis for octahedral structures.

Simple layer identification: octahedra connected via equatorial atoms = same layer.
"""

from collections import defaultdict, deque
import numpy as np


class LayersAnalyzer:
    """
    Simple layer identification based on equatorial connectivity.
    """
    
    def __init__(self):
        """
        Initialize layers analyzer based on the ontology of connectivity analysis.
        """
        self.layers = {}
        self.layer_distortion_stats = {}
        
    def identify_layers(self, octahedra_data, connectivity_analysis=None):
        """
        Identify layers: octahedra connected via equatorial atoms = same layer.
        
        Returns:
        dict: Simple layer data with octahedron keys per layer
        """
        if not octahedra_data or not connectivity_analysis:
            return {"layers": {}, "statistics": {"total_layers": 0}}
        
        # Get shared atoms and find equatorial connections
        shared_atoms = connectivity_analysis.get('shared_atoms', {})
        equatorial_connections = defaultdict(set)
        
        # Build equatorial connection map
        for atom_idx, sharing_info in shared_atoms.items():
            if len(sharing_info) >= 2:
                octahedra_sharing = [info['octahedron'] for info in sharing_info if info['role'] == 'ligand']
                
                for i, oct1 in enumerate(octahedra_sharing):
                    for oct2 in octahedra_sharing[i+1:]:
                        if oct1 in octahedra_data and oct2 in octahedra_data:
                            # Check if atom is equatorial in either octahedron
                            if (atom_idx in octahedra_data[oct1]['ligand_atoms']['equatorial_global_indices'] or
                                atom_idx in octahedra_data[oct2]['ligand_atoms']['equatorial_global_indices']):
                                equatorial_connections[oct1].add(oct2)
                                equatorial_connections[oct2].add(oct1)
        
        # Find connected components (layers)
        all_octahedra = set(octahedra_data.keys())
        assigned = set()
        layers = {}
        layer_num = 1
        
        for oct_key in all_octahedra:
            if oct_key not in assigned:
                # Find all equatorially connected octahedra
                layer_octahedra = self._find_connected_group(oct_key, equatorial_connections, assigned)
                layers[f"layer_{layer_num}"] = {
                    "layer_number": layer_num,
                    "octahedron_keys": layer_octahedra,
                    "octahedra_count": len(layer_octahedra)
                }
                assigned.update(layer_octahedra)
                layer_num += 1
        
        self.layers = layers
        
        # Calculate distortion statistics for each layer
        self._calculate_layer_distortion_stats(octahedra_data)
        
        return {
            "layers": layers,
            "statistics": {"total_layers": len(layers)},
            "layer_distortion_stats": self.layer_distortion_stats
        }
    
    def _calculate_layer_distortion_stats(self, octahedra_data):
        """
        Calculate mean distortion statistics for each layer.
        
        Parameters:
            octahedra_data: Dictionary containing octahedron information with distortion parameters
        """
        self.layer_distortion_stats = {}
        
        for layer_key, layer_data in self.layers.items():
            octahedron_keys = layer_data['octahedron_keys']
            
            # Collect distortion parameters for all octahedra in this layer
            layer_distortions = {
                'zeta': [],
                'delta': [],
                'sigma': [],
                'theta_mean': [],
                'theta_min': [],
                'theta_max': [],
                'mean_bond_distance': [],
                'bond_distance_variance': [],
                'octahedral_volume': []
            }
            
            for oct_key in octahedron_keys:
                if oct_key in octahedra_data:
                    oct_data = octahedra_data[oct_key]
                    
                    # Extract distortion parameters
                    dist_params = oct_data.get('distortion_parameters', {})
                    bond_analysis = oct_data.get('bond_distance_analysis', {})
                    geometric = oct_data.get('geometric_properties', {})
                    
                    layer_distortions['zeta'].append(dist_params.get('zeta', 0))
                    layer_distortions['delta'].append(dist_params.get('delta', 0))
                    layer_distortions['sigma'].append(dist_params.get('sigma', 0))
                    layer_distortions['theta_mean'].append(dist_params.get('theta_mean', 0))
                    layer_distortions['theta_min'].append(dist_params.get('theta_min', 0))
                    layer_distortions['theta_max'].append(dist_params.get('theta_max', 0))
                    layer_distortions['mean_bond_distance'].append(bond_analysis.get('mean_bond_distance', 0))
                    layer_distortions['bond_distance_variance'].append(bond_analysis.get('bond_distance_variance', 0))
                    layer_distortions['octahedral_volume'].append(geometric.get('octahedral_volume', 0))
            
            # Calculate statistics for each parameter
            layer_stats = {}
            for param, values in layer_distortions.items():
                if values:  # Only calculate if we have data
                    values_array = np.array(values)
                    layer_stats[param] = {
                        'mean': float(np.mean(values_array)),
                        'std': float(np.std(values_array)),
                        'min': float(np.min(values_array)),
                        'max': float(np.max(values_array)),
                        'count': len(values_array)
                    }
                else:
                    layer_stats[param] = {
                        'mean': 0.0,
                        'std': 0.0,
                        'min': 0.0,
                        'max': 0.0,
                        'count': 0
                    }
            
            self.layer_distortion_stats[layer_key] = layer_stats
    
    def _find_connected_group(self, start_oct, connections, already_assigned):
        """
        Find all octahedra connected to start_oct via equatorial connections.
        """
        group = []
        visited = set()
        queue = deque([start_oct])
        
        while queue:
            current = queue.popleft()
            if current in visited or current in already_assigned:
                continue
                
            visited.add(current)
            group.append(current)
            
            # Add connected octahedra
            if current in connections:
                for connected in connections[current]:
                    if connected not in visited and connected not in already_assigned:
                        queue.append(connected)
        
        return group
    
    def get_layer_summary(self):
        """Enhanced layer summary with distortion statistics."""
        if not self.layers:
            return "No layers found."
        
        summary = []
        for layer_key, layer_data in self.layers.items():
            octahedra = ', '.join(layer_data['octahedron_keys'])
            summary.append(f"{layer_key}: {octahedra}")
            
            # Add distortion statistics if available
            if layer_key in self.layer_distortion_stats:
                stats = self.layer_distortion_stats[layer_key]
                summary.append(f"  Distortion Statistics:")
                summary.append(f"    Zeta: {stats['zeta']['mean']:.4f} ± {stats['zeta']['std']:.4f}")
                summary.append(f"    Delta: {stats['delta']['mean']:.6f} ± {stats['delta']['std']:.6f}")
                summary.append(f"    Sigma: {stats['sigma']['mean']:.2f}° ± {stats['sigma']['std']:.2f}°")
                summary.append(f"    Theta (mean): {stats['theta_mean']['mean']:.2f}° ± {stats['theta_mean']['std']:.2f}°")
                summary.append(f"    Bond distance: {stats['mean_bond_distance']['mean']:.3f} ± {stats['mean_bond_distance']['std']:.3f} Å")
                summary.append(f"    Volume: {stats['octahedral_volume']['mean']:.2f} ± {stats['octahedral_volume']['std']:.2f} Å³")
        
        return "\n".join(summary)
    
    def get_layer_distortion_stats(self, layer_key=None):
        """
        Get distortion statistics for a specific layer or all layers.
        
        Parameters:
            layer_key: Specific layer to get stats for, or None for all layers
            
        Returns:
            Dictionary of distortion statistics
        """
        if layer_key:
            return self.layer_distortion_stats.get(layer_key, {})
        return self.layer_distortion_stats
    
    def compare_layer_distortions(self):
        """
        Compare distortion parameters between layers.
        
        Returns:
            Dictionary comparing distortion parameters across layers
        """
        if len(self.layer_distortion_stats) < 2:
            return "Need at least 2 layers for comparison."
        
        comparison = {}
        parameters = ['zeta', 'delta', 'sigma', 'theta_mean', 'mean_bond_distance', 'octahedral_volume']
        
        for param in parameters:
            comparison[param] = {}
            layer_means = []
            layer_names = []
            
            for layer_key, stats in self.layer_distortion_stats.items():
                if param in stats and stats[param]['count'] > 0:
                    layer_means.append(stats[param]['mean'])
                    layer_names.append(layer_key)
            
            if len(layer_means) > 1:
                comparison[param] = {
                    'layer_means': dict(zip(layer_names, layer_means)),
                    'overall_mean': float(np.mean(layer_means)),
                    'overall_std': float(np.std(layer_means)),
                    'min_layer': layer_names[np.argmin(layer_means)],
                    'max_layer': layer_names[np.argmax(layer_means)],
                    'range': float(np.max(layer_means) - np.min(layer_means))
                }
        
        return comparison
    
    def get_octahedra_in_layer(self, layer_key):
        """Get octahedra in a specific layer."""
        return self.layers.get(layer_key, {}).get('octahedron_keys', [])
    
    def get_layer_by_octahedron(self, octahedron_key):
        """Find which layer contains a specific octahedron."""
        for layer_key, layer_data in self.layers.items():
            if octahedron_key in layer_data['octahedron_keys']:
                return layer_key
        return None
    
    def export_layer_analysis(self, filename=None):
        """Export layer analysis with distortion statistics to JSON format."""
        import json
        if filename is None:
            filename = "layer_analysis.json"
        
        export_data = {
            "layers": self.layers,
            "layer_distortion_stats": self.layer_distortion_stats
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        return export_data
    
    
