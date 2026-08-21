"""Layer-specific structural analysis for perovskite materials.

This module provides analysis functions for individual layers and layer pairs,
including intra-layer and inter-layer B-X-B angle calculations.

Functions
---------
get_intralayer_bxb
    Calculate B-X-B angles within a single layer
get_interlayer_bxb
    Calculate B-X-B angles between two layers
get_all_interlayer_bxb
    Calculate all inter-layer B-X-B angles
"""

from typing import Dict, Optional, List, Any
import numpy as np
from ..octahedral_processing.octahedral_detection import find_shared_atoms
from ..utils.geometry_helpers import (
    calculate_angle_between_vectors,
    get_all_x_atoms_from_octahedron,
    normalize_layer_id,
)
from ...utils.geometry.pbc_distances import find_nearest_image_positions


def get_intralayer_bxb(analyzer, layer_id: str) -> Dict[str, Any]:
    """Calculate B-Xequatorial-B angles within a single layer (intra-layer only).
    
    For a given layer, finds all B-X-B angle triplets where both B atoms
    belong to octahedra in the same layer, and the X atom is a shared
    EQUATORIAL atom between them (B-Xeq-B angles only, excluding axial).
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    layer_id : str
        Layer ID (e.g., '0', '1'). Must match a layer in the structure.
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'bxb_angles': numpy array of B-Xeq-B angles in degrees (equatorial X only)
        - 'bxb_mean': mean B-X-B angle
        - 'bxb_std': standard deviation of B-X-B angles
        - 'count': number of angles found
        - 'layer_id': the layer ID
        
    Raises
    ------
    ValueError
        If layer_id not found in structure
    """
    layer_id = str(normalize_layer_id(layer_id))
    
    # Get octahedra for this layer
    graph = analyzer.get_graph()
    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())
    all_octahedra = analyzer.get_octahedra()
    
    # Find octahedra in this layer
    layer_octahedra_ids = []
    layer_node_id = f'layer_{layer_id}'
    
    if layer_node_id not in graph:
        raise ValueError(f"Layer {layer_id} not found in structure")
    
    for neighbor in graph.neighbors(layer_node_id):
        edge_data = graph.get_edge_data(layer_node_id, neighbor)
        if edge_data and edge_data.get('edge_type') == 'contains':
            if neighbor.startswith('octahedron_'):
                try:
                    oct_idx = int(neighbor.replace('octahedron_', ''))
                    layer_octahedra_ids.append(oct_idx)
                except ValueError:
                    continue
    
    if not layer_octahedra_ids:
        return {
            'bxb_angles': np.array([]),
            'bxb_mean': None,
            'bxb_std': None,
            'count': 0,
            'layer_id': layer_id
        }
    
    # Extract octahedra data for this layer
    layer_octahedra = [all_octahedra[i] for i in layer_octahedra_ids if i < len(all_octahedra)]
    
    # Build neighbor indices and find shared atoms
    neighbor_indices = []
    for oct in layer_octahedra:
        all_neighbors = get_all_x_atoms_from_octahedron(oct)
        neighbor_indices.append(all_neighbors)
    
    shared_atoms = find_shared_atoms(neighbor_indices)
    
    # Build X atom properties mapping from graph
    x_atom_properties = {}
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'atom':
            vasp_idx = data.get('vasp_index')
            if vasp_idx is not None:
                x_atom_properties[vasp_idx] = {
                    'is_terminal': data.get('is_terminal', False),
                    'is_equatorial': data.get('is_equatorial', False),
                    'is_interlayer': data.get('is_interlayer', False),
                }
    
    # Get cached B atom positions for performance
    b_x_data = analyzer.get_b_x_atoms()
    b_idx_to_pos = {b_idx: b_x_data['b_positions'][i] for i, b_idx in enumerate(b_x_data['b_indices'])}
    
    # Calculate B-X-B angles for pairs in this layer
    bxb_angles = []
    
    for (oct_i, oct_j), shared_x_indices in shared_atoms.items():
        if oct_i >= len(layer_octahedra) or oct_j >= len(layer_octahedra):
            continue
        
        oct_i_data = layer_octahedra[oct_i]
        oct_j_data = layer_octahedra[oct_j]
        
        b_i_idx = oct_i_data.get("central_atom_index")
        b_j_idx = oct_j_data.get("central_atom_index")
        
        if b_i_idx is None or b_j_idx is None:
            continue
        
        # Calculate angle for each shared X-site
        for x_idx in shared_x_indices:
            if x_idx >= len(atom_positions):
                continue
            
            # Filter to only equatorial X atoms (B-Xeq-B angles, not B-Xaxial-B)
            x_properties = x_atom_properties.get(x_idx, {})
            if not x_properties.get('is_equatorial', False):
                continue
            
            x_pos = atom_positions[x_idx]
            
            # Find nearest periodic images of B_i and B_j
            b_i_base_pos = b_idx_to_pos.get(b_i_idx, atom_positions[b_i_idx])
            b_i_data = find_nearest_image_positions(
                reference_position=x_pos,
                candidate_positions=np.array([b_i_base_pos]),
                candidate_indices=np.array([b_i_idx]),
                cell=cell,
                n_neighbors=1,
                pbc=True,
            )
            _, b_i_nearest_pos, _, _ = b_i_data
            b_i_pos = b_i_nearest_pos[0]
            
            # Find nearest periodic image of B_j
            b_j_base_pos = b_idx_to_pos.get(b_j_idx, atom_positions[b_j_idx])
            b_j_data = find_nearest_image_positions(
                reference_position=x_pos,
                candidate_positions=np.array([b_j_base_pos]),
                candidate_indices=np.array([b_j_idx]),
                cell=cell,
                n_neighbors=1,
                pbc=True,
            )
            _, b_j_nearest_pos, _, _ = b_j_data
            b_j_pos = b_j_nearest_pos[0]
            
            # Calculate angle
            vec_xi = b_i_pos - x_pos
            vec_xj = b_j_pos - x_pos
            angle = calculate_angle_between_vectors(vec_xi, vec_xj, cell=cell, apply_pbc=False)
            bxb_angles.append(angle)
    
    bxb_angles_array = np.array(bxb_angles) if bxb_angles else np.array([])
    
    return {
        'bxb_angles': bxb_angles_array,
        'bxb_mean': float(np.mean(bxb_angles_array)) if len(bxb_angles_array) > 0 else None,
        'bxb_std': float(np.std(bxb_angles_array)) if len(bxb_angles_array) > 0 else None,
        'count': len(bxb_angles),
        'layer_id': layer_id
    }


def get_interlayer_bxb(analyzer, layer_id1: str, layer_id2: str) -> Dict[str, Any]:
    """Calculate B-Xaxial-B angles between two layers (inter-layer only).
    
    For two given layers, finds all B-X-B angle triplets where one B atom is from
    layer_id1, the other B atom is from layer_id2, and the X atom is a shared
    AXIAL/INTERLAYER atom between them (B-Xaxial-B angles only, excluding equatorial).
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
    layer_id1 : str
        First layer ID (e.g., '0')
    layer_id2 : str
        Second layer ID (e.g., '1')
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'bxb_angles': numpy array of B-Xaxial-B angles in degrees (interlayer X only)
        - 'bxb_mean': mean B-X-B angle
        - 'bxb_std': standard deviation of B-X-B angles
        - 'count': number of angles found
        - 'layer_pair': tuple of (layer_id1, layer_id2)
        
    Raises
    ------
    ValueError
        If either layer_id not found in structure
    """
    layer_id1 = str(normalize_layer_id(layer_id1))
    layer_id2 = str(normalize_layer_id(layer_id2))
    
    # Get octahedra for both layers
    graph = analyzer.get_graph()
    atom_positions = analyzer.cell.get_positions()
    cell = np.array(analyzer.cell.get_cell())
    all_octahedra = analyzer.get_octahedra()
    
    # Find octahedra in both layers
    def _get_layer_octahedra(layer_id):
        layer_node_id = f'layer_{layer_id}'
        if layer_node_id not in graph:
            raise ValueError(f"Layer {layer_id} not found in structure")
        
        oct_ids = []
        for neighbor in graph.neighbors(layer_node_id):
            edge_data = graph.get_edge_data(layer_node_id, neighbor)
            if edge_data and edge_data.get('edge_type') == 'contains':
                if neighbor.startswith('octahedron_'):
                    try:
                        oct_idx = int(neighbor.replace('octahedron_', ''))
                        oct_ids.append(oct_idx)
                    except ValueError:
                        continue
        return oct_ids
    
    layer1_oct_ids = _get_layer_octahedra(layer_id1)
    layer2_oct_ids = _get_layer_octahedra(layer_id2)
    
    if not layer1_oct_ids or not layer2_oct_ids:
        return {
            'bxb_angles': np.array([]),
            'bxb_mean': None,
            'bxb_std': None,
            'count': 0,
            'layer_pair': (layer_id1, layer_id2)
        }
    
    # Extract octahedra data for both layers
    layer1_octahedra = [all_octahedra[i] for i in layer1_oct_ids if i < len(all_octahedra)]
    layer2_octahedra = [all_octahedra[i] for i in layer2_oct_ids if i < len(all_octahedra)]
    
    # Build neighbor indices for both layers
    neighbor_indices_layer1 = []
    for oct in layer1_octahedra:
        all_neighbors = get_all_x_atoms_from_octahedron(oct)
        neighbor_indices_layer1.append(all_neighbors)
    
    neighbor_indices_layer2 = []
    for oct in layer2_octahedra:
        all_neighbors = get_all_x_atoms_from_octahedron(oct)
        neighbor_indices_layer2.append(all_neighbors)
    
    # Build list combining both layers for shared atom detection
    all_neighbor_indices = neighbor_indices_layer1 + neighbor_indices_layer2
    shared_atoms = find_shared_atoms(all_neighbor_indices)
    
    # Build X atom properties mapping from graph
    x_atom_properties = {}
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'atom':
            vasp_idx = data.get('vasp_index')
            if vasp_idx is not None:
                x_atom_properties[vasp_idx] = {
                    'is_terminal': data.get('is_terminal', False),
                    'is_equatorial': data.get('is_equatorial', False),
                    'is_interlayer': data.get('is_interlayer', False),
                }
    
    # Get cached B atom positions
    b_x_data = analyzer.get_b_x_atoms()
    b_idx_to_pos = {b_idx: b_x_data['b_positions'][i] for i, b_idx in enumerate(b_x_data['b_indices'])}
    
    # Calculate B-X-B angles for inter-layer pairs only
    bxb_angles = []
    
    for (oct_i, oct_j), shared_x_indices in shared_atoms.items():
        # Check if this is an inter-layer pair
        # oct_i and oct_j are indices in the combined list
        oct_i_in_layer1 = oct_i < len(layer1_octahedra)
        oct_j_in_layer1 = oct_j < len(layer1_octahedra)
        
        # Skip if both in same layer
        if oct_i_in_layer1 == oct_j_in_layer1:
            continue
        
        # Get actual octahedra
        if oct_i_in_layer1:
            oct_i_data = layer1_octahedra[oct_i]
            oct_j_data = layer2_octahedra[oct_j - len(layer1_octahedra)]
        else:
            oct_i_data = layer2_octahedra[oct_i - len(layer1_octahedra)]
            oct_j_data = layer1_octahedra[oct_j]
        
        b_i_idx = oct_i_data.get("central_atom_index")
        b_j_idx = oct_j_data.get("central_atom_index")
        
        if b_i_idx is None or b_j_idx is None:
            continue
        
        # Calculate angle for each shared X-site
        for x_idx in shared_x_indices:
            if x_idx >= len(atom_positions):
                continue
            
            # Filter to only interlayer/axial X atoms (B-Xaxial-B angles)
            x_properties = x_atom_properties.get(x_idx, {})
            if not x_properties.get('is_interlayer', False):
                continue
            
            x_pos = atom_positions[x_idx]
            
            # Find nearest periodic images
            b_i_base_pos = b_idx_to_pos.get(b_i_idx, atom_positions[b_i_idx])
            b_i_data = find_nearest_image_positions(
                reference_position=x_pos,
                candidate_positions=np.array([b_i_base_pos]),
                candidate_indices=np.array([b_i_idx]),
                cell=cell,
                n_neighbors=1,
                pbc=True,
            )
            _, b_i_nearest_pos, _, _ = b_i_data
            b_i_pos = b_i_nearest_pos[0]
            
            b_j_base_pos = b_idx_to_pos.get(b_j_idx, atom_positions[b_j_idx])
            b_j_data = find_nearest_image_positions(
                reference_position=x_pos,
                candidate_positions=np.array([b_j_base_pos]),
                candidate_indices=np.array([b_j_idx]),
                cell=cell,
                n_neighbors=1,
                pbc=True,
            )
            _, b_j_nearest_pos, _, _ = b_j_data
            b_j_pos = b_j_nearest_pos[0]
            
            # Calculate angle
            vec_xi = b_i_pos - x_pos
            vec_xj = b_j_pos - x_pos
            angle = calculate_angle_between_vectors(vec_xi, vec_xj, cell=cell, apply_pbc=False)
            bxb_angles.append(angle)
    
    bxb_angles_array = np.array(bxb_angles) if bxb_angles else np.array([])
    
    return {
        'bxb_angles': bxb_angles_array,
        'bxb_mean': float(np.mean(bxb_angles_array)) if len(bxb_angles_array) > 0 else None,
        'bxb_std': float(np.std(bxb_angles_array)) if len(bxb_angles_array) > 0 else None,
        'count': len(bxb_angles),
        'layer_pair': (layer_id1, layer_id2)
    }


def get_all_interlayer_bxb(analyzer) -> Dict[str, Any]:
    """Calculate all inter-layer B-Xaxial-B angles in the structure.
    
    Finds all adjacent layer pairs and calculates B-Xaxial-B angles between them
    (using only axial/interlayer X atoms that bridge between layers).
    
    Parameters
    ----------
    analyzer : q2D_analyzer
        Analyzer instance with analyzed structure
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'bxb_angles': numpy array of all inter-layer B-Xaxial-B angles
        - 'bxb_mean': mean B-X-B angle
        - 'bxb_std': standard deviation
        - 'count': total number of angles
        - 'pairs': list of layer pairs analyzed
        - 'pair_data': dict mapping layer_pair to individual angle data
    """
    graph = analyzer.get_graph()
    
    # Get all layer IDs
    layer_ids = []
    for node, data in graph.nodes(data=True):
        if data.get('node_type') == 'layer':
            layer_id = str(normalize_layer_id(node.replace('layer_', '')))
            layer_ids.append(layer_id)
    
    layer_ids = sorted(set(layer_ids), key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    if len(layer_ids) < 2:
        return {
            'bxb_angles': np.array([]),
            'bxb_mean': None,
            'bxb_std': None,
            'count': 0,
            'pairs': [],
            'pair_data': {}
        }
    
    # Calculate angles for adjacent layer pairs
    all_angles = []
    pair_data = {}
    pairs = []
    
    for i in range(len(layer_ids) - 1):
        layer1_id = layer_ids[i]
        layer2_id = layer_ids[i + 1]
        pairs.append((layer1_id, layer2_id))
        
        try:
            result = get_interlayer_bxb(analyzer, layer1_id, layer2_id)
            pair_data[(layer1_id, layer2_id)] = result
            if result['bxb_angles'] is not None and len(result['bxb_angles']) > 0:
                all_angles.extend(result['bxb_angles'])
        except ValueError:
            # Layer pair might not have interlayer connections
            continue
    
    all_angles_array = np.array(all_angles) if all_angles else np.array([])
    
    return {
        'bxb_angles': all_angles_array,
        'bxb_mean': float(np.mean(all_angles_array)) if len(all_angles_array) > 0 else None,
        'bxb_std': float(np.std(all_angles_array)) if len(all_angles_array) > 0 else None,
        'count': len(all_angles),
        'pairs': pairs,
        'pair_data': pair_data
    }
